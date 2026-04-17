"""Answer generation with citations, answer shaping, and hallucination filtering.

Pipeline:
1. Build a prompt from retrieved chunks + user query
2. Shape the prompt based on detected query type (explain, list, compare)
3. Generate answer via Mistral LLM
4. Extract citations from the response
5. Run hallucination filter to verify claims against evidence
"""

import re

from app.models import Chunk, Citation
from app.query import _call_mistral

# Citation parsing is two-stage to tolerate the many shapes LLMs produce:
#   [Source 3], [source 3], (Source 3)                     — simple
#   [Source 3, p.31]                                       — with page reference
#   [Source 1, Source 2, Source 3]                         — multiple per bracket
#   [Source 2, p. 31]                                      — with spaces
# We first find any bracket/paren group, then extract every "Source N" token
# inside it. A naive single regex can't capture multiples per group.
_CITATION_GROUP_RE = re.compile(r"[\[\(]([^\]\)]+)[\]\)]")
_SOURCE_N_RE = re.compile(r"\bsources?\s+(\d+)", re.IGNORECASE)

# Relevance threshold — applied to RRF scores (typical range: 0.005–0.035)
# A chunk appearing in top-5 of both semantic + keyword lists scores ~0.03
# A chunk in only one list scores ~0.016. Threshold of 0.005 filters truly irrelevant results.
RELEVANCE_THRESHOLD = 0.005


def _format_context(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is labeled with its source for citation tracing.
    """
    context_parts = []
    for i, (chunk, score) in enumerate(chunks_with_scores):
        label = f"[Source {i+1}: {chunk.source_file}, p.{chunk.page_number}]"
        context_parts.append(f"{label}\n{chunk.text}")
    return "\n\n---\n\n".join(context_parts)


def _detect_query_type(query: str) -> str:
    """Detect query type to shape the answer format.

    Simple heuristic — avoids an extra LLM call:
    - "list" queries → bullet-point answer
    - "compare" queries → structured comparison
    - default → explanatory paragraph
    """
    query_lower = query.lower()

    list_keywords = ["list", "enumerate", "what are", "name the", "give me", "show me"]
    compare_keywords = ["compare", "difference", "versus", "vs", "contrast", "differ"]

    if any(kw in query_lower for kw in compare_keywords):
        return "compare"
    elif any(kw in query_lower for kw in list_keywords):
        return "list"
    else:
        return "explain"


_CITATION_STRIP_RE = re.compile(r"\[Source[^\]]*\]", re.IGNORECASE)


def _build_prompt(
    query: str,
    context: str,
    query_type: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """Build the prompt messages for the LLM based on query type.

    Answer shaping: different system prompts for different query types
    to produce appropriately structured outputs.

    When conversation history is provided, prior turns are inserted between
    the system prompt and the current user message so the LLM can resolve
    follow-up references. We strip [Source N] markers from prior assistant
    messages to keep the context clean.
    """
    format_instructions = {
        "explain": "Provide a clear, concise explanation based on the sources.",
        "list": "Structure your answer as a bulleted list. Each item should be concise.",
        "compare": (
            "Structure your answer as a comparison. Use a clear format like:\n"
            "- Point of comparison\n"
            "  - Option A: ...\n"
            "  - Option B: ..."
        ),
    }

    followup_note = ""
    if history:
        followup_note = (
            "\n\nIMPORTANT: You are continuing a conversation. Even for follow-up "
            "questions, you MUST cite [Source N] for every claim using the sources "
            "provided above. Do not rely on or repeat previous answers without citing."
        )

    system_prompt = (
        "You are a document Q&A assistant. Answer the user's question using ONLY the "
        "provided source documents. Follow these rules STRICTLY:\n\n"
        "1. EVERY factual statement must end with a [Source N] citation. No claim is "
        "allowed without a citation. If you cannot cite a source for a claim, omit the claim.\n"
        "2. Every paragraph or bullet must contain at least one [Source N] citation.\n"
        "3. If the sources do not contain enough information to answer, say so explicitly "
        "and stop. Do not fabricate an answer from general knowledge.\n"
        "4. Do NOT restate information that isn't in the sources, even if you believe it is correct.\n"
        f"5. {format_instructions.get(query_type, format_instructions['explain'])}"
        f"{followup_note}\n\n"
        f"--- SOURCES ---\n{context}\n--- END SOURCES ---"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Insert conversation history (last 3 exchanges, 6 messages max).
    if history:
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "assistant":
                # Strip citation markers from prior answers — they reference
                # a different set of source numbers and would confuse the LLM.
                content = _CITATION_STRIP_RE.sub("", content).strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": query})
    return messages


def _make_citation(chunk: Chunk, score: float) -> Citation:
    excerpt = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
    return Citation(
        source_file=chunk.source_file,
        page_number=chunk.page_number,
        text_excerpt=excerpt,
        relevance_score=round(score, 4),
    )


# Phrases that indicate the answer itself is a refusal / "no evidence" response.
# Running a hallucination check on these is nonsensical — they have no claims.
_REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "don't contain",
    "doesn't contain",
    "do not provide enough",
    "does not provide enough",
    "do not provide information",
    "does not provide information",
    "not provide enough",
    "no information about",
    "no information on",
    "insufficient evidence",
    "not enough information",
    "could not find",
    "couldn't find",
    "cannot provide",
    "cannot answer",
    "unable to provide",
    "unable to answer",
)


def _is_refusal(answer: str) -> bool:
    """Detect whether the answer itself is a 'no evidence' / refusal response."""
    return any(marker in answer.lower() for marker in _REFUSAL_MARKERS)


def _extract_citations(
    answer: str, chunks_with_scores: list[tuple[Chunk, float]]
) -> list[Citation]:
    """Extract citation objects from the generated answer.

    Regex-matches multiple citation formats — tolerates case, whitespace,
    and parenthesis vs bracket variants: [Source 3], (source 3), [ Source  3 ].
    Only chunks the LLM actually referenced become citations.

    If the answer is itself a refusal / "no evidence" response, we return an
    empty citation list: the LLM may have still attached source markers per
    the prompt rule, but citing nonexistent evidence for a refusal is misleading.
    """
    if _is_refusal(answer):
        return []

    cited_indices: list[int] = []
    seen: set[int] = set()

    for group_match in _CITATION_GROUP_RE.finditer(answer):
        inner = group_match.group(1)
        for num_match in _SOURCE_N_RE.finditer(inner):
            n = int(num_match.group(1))
            i = n - 1  # sources are 1-indexed in the prompt
            if 0 <= i < len(chunks_with_scores) and i not in seen:
                seen.add(i)
                cited_indices.append(i)

    return [
        _make_citation(chunks_with_scores[i][0], chunks_with_scores[i][1])
        for i in cited_indices
    ]


def check_hallucination(answer: str, context: str) -> str:
    """Post-hoc hallucination filter: verify answer claims against source evidence.

    Uses the LLM as a judge to identify unsupported claims.
    Returns the original answer if clean, or a flagged version if issues found.

    Short-circuits for honest refusal answers (no evidence claim to verify).
    """
    ans_lower = answer.lower()
    if any(marker in ans_lower for marker in _REFUSAL_MARKERS):
        return answer

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fact-checker. Compare the ANSWER against the SOURCES below. "
                "If every claim in the answer is supported by the sources, respond with "
                "exactly: VERIFIED\n"
                "If any claim is NOT supported, respond with exactly: "
                "FLAGGED: <brief description of the unsupported claim>\n\n"
                f"--- SOURCES ---\n{context}\n--- END SOURCES ---"
            ),
        },
        {"role": "user", "content": f"ANSWER: {answer}"},
    ]
    result = _call_mistral(messages, temperature=0.0).strip()

    if result.startswith("VERIFIED"):
        return answer
    elif result.startswith("FLAGGED"):
        flag_note = result.replace("FLAGGED:", "").strip()
        return f"{answer}\n\n*Note: Some claims in this answer may not be fully supported by the source documents. ({flag_note})*"
    else:
        # If the checker gives an unexpected response, return the answer as-is
        return answer


def generate_answer(
    query: str,
    chunks_with_scores: list[tuple[Chunk, float]],
    history: list[dict] | None = None,
) -> tuple[str, list[Citation]]:
    """Full generation pipeline: prompt → LLM → citations → hallucination check.

    Returns (answer_text, list_of_citations).
    If chunks don't meet the similarity threshold, returns an "insufficient evidence" message.
    Citation relevance scores are normalized so the top result is 1.0 (100%) —
    raw RRF scores are tiny (~0.016) which looks misleading as a percentage.
    """
    # Check if we have sufficient evidence
    if not chunks_with_scores:
        return "I don't have any documents to search through. Please upload some PDFs first.", []

    best_score = max(score for _, score in chunks_with_scores)
    if best_score < RELEVANCE_THRESHOLD:
        return (
            "I couldn't find sufficient evidence in the uploaded documents to answer "
            "this question. The available content doesn't seem to cover this topic.",
            [],
        )

    # Build context and detect query type
    context = _format_context(chunks_with_scores)
    query_type = _detect_query_type(query)

    # Generate answer
    messages = _build_prompt(query, context, query_type, history=history)
    answer = _call_mistral(messages, temperature=0.1)

    # Extract citations
    citations = _extract_citations(answer, chunks_with_scores)

    # Normalize relevance scores so the top citation is 1.0 (100%).
    # Raw RRF scores are tiny (0.005–0.033) which looks like low confidence
    # to anyone expecting a percentage — normalizing makes the ranking
    # intuitive without changing the actual ordering.
    if citations:
        max_score = max(c.relevance_score for c in citations)
        if max_score > 0:
            for c in citations:
                c.relevance_score = round(c.relevance_score / max_score, 4)

    # Hallucination filter
    answer = check_hallucination(answer, context)

    return answer, citations
