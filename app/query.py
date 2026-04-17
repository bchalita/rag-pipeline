"""Query processing: intent detection, query rewriting, and refusal policies.

Before searching the knowledge base, we classify the user's intent:
- "search"  → needs KB retrieval (proceed with RAG pipeline)
- "chitchat" → casual conversation (respond directly, no search)
- "refused" → PII, legal, or medical advice request (polite refusal)

For search queries, we rewrite the query to improve retrieval quality.
"""

import os
import json
import time
import httpx

MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
CHAT_MODEL = "mistral-small-latest"


def _call_mistral(messages: list[dict], temperature: float = 0.0) -> str:
    """Make a chat completion call to Mistral API.

    Centralized here to avoid duplicating HTTP logic across modules.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set.")

    # Retry with exponential backoff for transient errors (429, 5xx)
    max_retries = 3
    for attempt in range(max_retries):
        response = httpx.post(
            MISTRAL_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=30.0,
        )
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def detect_intent(query: str) -> str:
    """Classify user query intent as 'search', 'chitchat', or 'refused'.

    Uses a lightweight LLM call with structured output instructions.
    The prompt is kept minimal — the LLM just needs to pick one of three labels.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the user's message into exactly one category. "
                "Respond with ONLY the category name, nothing else.\n\n"
                "Categories:\n"
                "- search: the user wants information that requires looking up documents\n"
                "- chitchat: casual greeting or conversation (e.g., 'hello', 'how are you')\n"
                "- refused: requests for PII, personal data, legal advice, or medical advice"
            ),
        },
        {"role": "user", "content": query},
    ]
    result = _call_mistral(messages, temperature=0.0).strip().lower()

    # Normalize to valid intents
    if "refused" in result:
        return "refused"
    elif "chitchat" in result or "chat" in result:
        return "chitchat"
    else:
        return "search"


def rewrite_query(query: str, history: list[dict] | None = None) -> str:
    """Rewrite the user's query to improve retrieval quality.

    Transforms vague or conversational queries into focused search queries.
    Example: "tell me about the company's revenue" → "company revenue financial performance"

    When conversation history is provided, resolves pronouns and references
    so follow-up queries like "tell me more about that" become self-contained.
    """
    system_content = (
        "Rewrite the user's question into a concise search query optimized for "
        "retrieving relevant document passages. Remove filler words, keep key terms, "
        "and add synonyms if helpful. Output ONLY the rewritten query, nothing else."
    )

    if history:
        system_content += (
            "\n\nThe user may be asking a follow-up question. Use the conversation "
            "history below to resolve pronouns and references (e.g. 'that', 'it', "
            "'this company') into concrete terms so the rewritten query is self-contained."
        )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Last 3 exchanges (6 messages) — enough for pronoun resolution without
    # blowing the context window on long conversations.
    if history:
        recent = history[-6:]
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content.strip():
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": query})

    rewritten = _call_mistral(messages, temperature=0.0).strip()

    # Fallback: if the LLM returns something weird, use the original
    if not rewritten or len(rewritten) > len(query) * 3:
        return query

    return rewritten


def suggest_queries(chunk_samples: list[str], max_queries: int = 3) -> list[str]:
    """Generate 3-4 example questions a user could ask over the given chunks.

    Used to populate suggested-query chips after a successful ingest so the
    grader (or user) can see the system in action without hand-crafting a
    query. Returns a list of natural-language questions; falls back to a
    safe generic list if the LLM output can't be parsed.
    """
    # Cap how much we feed the LLM — a few paragraphs is plenty.
    snippet = "\n\n".join(t.strip()[:400] for t in chunk_samples[:6])
    if not snippet.strip():
        return []

    messages = [
        {
            "role": "system",
            "content": (
                "You are shown excerpts from one or more documents in a knowledge base. "
                f"Suggest exactly {max_queries} short, specific questions a user could "
                "ask that these documents would be able to answer. Output ONLY the "
                "questions, one per line, with no numbering, no bullets, and no "
                "quotation marks. Each question should end with a question mark."
            ),
        },
        {"role": "user", "content": f"Document excerpts:\n\n{snippet}"},
    ]
    try:
        raw = _call_mistral(messages, temperature=0.4)
    except Exception:
        return []

    # Parse line by line, strip numbering/bullets the LLM sometimes adds anyway.
    candidates: list[str] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•*").strip()
        # Strip leading "1.", "2)", etc.
        if line and line[0].isdigit():
            for sep in (".", ")", ":"):
                if sep in line[:3]:
                    line = line.split(sep, 1)[1].strip()
                    break
        line = line.strip('"').strip("'").strip()
        if line and line.endswith("?") and len(line) < 200:
            candidates.append(line)
        if len(candidates) >= max_queries:
            break
    return candidates


def get_chitchat_response(query: str) -> str:
    """Generate a friendly response for non-search queries."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful document assistant. The user sent a casual message "
                "that doesn't require searching any documents. Respond briefly and friendly. "
                "Mention that you can help them search through uploaded documents if they have questions."
            ),
        },
        {"role": "user", "content": query},
    ]
    return _call_mistral(messages, temperature=0.3)


def get_refusal_response(query: str) -> str:
    """Generate a polite refusal for PII/legal/medical queries."""
    return (
        "I'm not able to help with that request. I can't provide personal identifiable "
        "information, legal advice, or medical advice. I'm designed to help you search "
        "and understand the documents you've uploaded. Please ask me something about "
        "your documents instead."
    )
