#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

# TODO A: write this before you read aip/rag.py::ANSWER_SYSTEM.
ANSWER_SYSTEM = f"""\
You are an insurance policy assistant. Your job is to answer questions using \
ONLY the provided numbered sources. Follow these rules:

1. **Answer only from the sources.** Do not use general knowledge. If a source \
says something directly, you can reference it. If it does not, say so.

2. **Cite by index.** Use [1], [2], or [1][2] to cite sources. Every factual \
claim must have a citation.

3. **Exact citations only.** Only cite source numbers that were actually provided. \
If you cite [7] but only 5 sources were given, that is a mechanical error.

4. **Exact refusal.** If the sources do not contain ANY part of the answer, say \
exactly: "I don't have enough information in the provided sources to answer that."

5. **Partial answers are required, not optional.** If the sources answer part of \
the question but not all of it, DO NOT refuse. Instead, state the part that is \
supported with its citation, then explicitly say which part is missing -- e.g. \
"Platinum includes international emergency cover [2], but the coverage limit is \
not stated in the provided sources." Only use the exact refusal string when the \
sources give you nothing at all to work with.

6. **When sources conflict,** state both positions clearly. Never silently pick \
one when the sources disagree.

7. **Be concise.** Give a short answer (1-3 sentences usually). Longer answers \
are fine only if the question needs it. Every output token costs latency and money.

{UNTRUSTED_SYSTEM_CLAUSE}
"""

# C4: a stricter refusal setting, appended to ANSWER_SYSTEM when strict=True.
# Rule 5 above (partial answers required) is what makes the system answer
# things like "premium for a 35-year-old in Bengaluru" with a half-relevant
# fact instead of declining outright. This clause dials that back: it asks
# for a *complete, unambiguous* answer before committing to anything, which
# should raise refusal recall at the cost of refusal precision.
STRICT_CLAUSE = """\
STRICT MODE: only answer if the sources give a complete and unambiguous \
answer to every part of the question. If there is any gap, ambiguity, or \
missing detail, use the exact refusal string instead of a partial answer -- \
even if you could state something true and relevant. When in doubt, refuse.
"""


# Phrases that mark a *partial* decline -- the model answered what it could
# and flagged the rest as unsupported, rather than issuing the exact REFUSAL
# string. Distinct from a full refusal: C4/C3 need to tell "answered nothing"
# apart from "answered part, flagged the gap" when scoring refusal recall.
_PARTIAL_DECLINE_MARKERS = (
    "not stated in the provided sources",
    "not provided in the sources",
    "sources do not state",
    "not available in the provided sources",
    "is not stated in",
)


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    partial_decline: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False
    repaired: bool = False


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:
    """Validate an answer for citation correctness and completeness.

    Checks:
      - every [n] is between 1 and n_sources
      - not truncated (finish_reason == "length" means cut-off)
      - a non-refusal answer contains at least one citation
    """
    from aip.guards import enforce_citations

    is_refusal = text.strip() == REFUSAL

    # Check citations
    valid, invalid = enforce_citations(text, n_sources)

    # Check truncation
    truncated = finish_reason == "length"

    # Count citations in text
    citations = re.findall(r'\[\d+\]', text)
    n_citations = len(set(int(m[1:-1]) for m in citations)) if citations else 0

    # A valid answer is:
    # - either a refusal, OR
    # - has valid citations AND at least one citation AND not truncated
    is_valid = is_refusal or (valid and not truncated and n_citations > 0)

    return {
        "valid": is_valid,
        "refused": is_refusal,
        "invalid_citations": invalid,
        "n_citations": n_citations,
        "truncated": truncated,
        "reason": "" if is_valid else (
            "truncated" if truncated else
            f"invalid citations {invalid}" if invalid else
            "no citations in non-refusal answer"
        )
    }


def _count_sources_in_context(context: str) -> int:
    """format_context can drop trailing hits once max_chars is exceeded, so the
    number of sources actually shown to the model can be less than len(hits).
    Citation validity must be checked against what was shown, not what was
    retrieved."""
    return len(re.findall(r'^\[\d+\] \(source:', context, re.MULTILINE))


def _build_user_prompt(question: str, context: str, n_sources: int) -> str:
    """Assemble the numbered sources and the question into one user turn."""
    return (
        f"Sources:\n{context}\n\n"
        f"({n_sources} sources total, numbered [1] through [{n_sources}])\n\n"
        f"Question: {question}"
    )


def _is_partial_decline(text: str) -> bool:
    """A partial decline names a gap in the sources without matching REFUSAL
    exactly -- e.g. "Platinum has an international benefit [2], but the limit
    is not stated in the provided sources." Distinct from a full refusal."""
    if text.strip() == REFUSAL:
        return False
    low = text.lower()
    return any(marker in low for marker in _PARTIAL_DECLINE_MARKERS)


def _generate_with_repair(question: str, context: str, n_sources: int, *,
                          tier: str, strict: bool) -> tuple[str, dict, bool]:
    """Shared core of both public entry points: generate, validate, and on
    failure retry once with a corrective message before giving up.

    B3 strategy: a citation-invalid answer is never returned as-is. The first
    retry gets one chance to self-correct with the exact failure reason; if
    that also fails validation, we fall back to REFUSAL rather than pass a
    bad citation downstream. Returns (text, validation, repaired) where
    `repaired` is True iff the first attempt needed the retry path.
    """
    system = ANSWER_SYSTEM + ("\n" + STRICT_CLAUSE if strict else "")

    def _call(prompt: str) -> tuple[str, str | None]:
        r = chat([{"role": "user", "content": prompt}], system=system,
                 tier=tier, max_tokens=1024, return_full=True)
        return r.get("text", ""), r.get("finish_reason")

    text, finish_reason = _call(_build_user_prompt(question, context, n_sources))
    validation = validate_answer(text, n_sources, finish_reason)
    if validation["valid"]:
        return text, validation, False

    corrective = (
        f"Your previous answer had a citation problem: {validation['reason']}.\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Rewrite the answer so that:\n"
        f"1. It only cites sources [1] through [{n_sources}]\n"
        f"2. Every factual claim carries a citation\n"
        f"3. It is not cut off\n"
        f"If the sources give you nothing to work with, reply exactly: {REFUSAL!r}"
    )
    text, finish_reason = _call(corrective)
    validation = validate_answer(text, n_sources, finish_reason)
    if validation["valid"]:
        return text, validation, True

    # Both attempts failed validation -- refuse rather than risk a bad citation.
    return REFUSAL, {"valid": True, "refused": True, "invalid_citations": [],
                     "n_citations": 0, "truncated": False}, True


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "MAIN",
                    strict: bool = False) -> Answer:
    """RAG pipeline: retrieve → generate → validate → repair if needed.

    `strict` is C4's second refusal setting -- same pipeline, stricter system
    instruction (see STRICT_CLAUSE), used to compare refusal recall/precision
    against the default at two operating points.

    Lab 5 fix (mode 6, distractor dilution): `final_k` was accepted here but
    never applied -- all `k` retrieved chunks reached the generator regardless
    of `final_k`. Retrieve wide (`k`) for recall, then narrow to `final_k`
    before generating, so the generator sees a focused shortlist instead of
    every candidate.
    """
    hits = retriever.search(question, k=k)[:final_k]
    context = format_context(hits, max_chars=8000)
    n_sources = _count_sources_in_context(context)

    text, validation, repaired = _generate_with_repair(
        question, context, n_sources, tier=tier, strict=strict)

    return Answer(
        question=question, text=text, hits=hits,
        refused=validation["refused"],
        partial_decline=_is_partial_decline(text),
        citations_valid=True,
        invalid_citations=validation["invalid_citations"],
        n_citations=validation["n_citations"],
        truncated=validation["truncated"],
        repaired=repaired,
    )


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN") -> Answer:
    """E2: same generator, gold context only (no retrieval).

    Used to measure generation ceiling: how good can answers be if retrieval
    was perfect? The difference between this and answer_question() quantifies
    retrieval loss. Same generation+repair path as answer_question() -- the
    only thing that differs is where the context comes from.
    """
    from aip.chunking import Chunk
    from aip.retrieval import Hit

    # Hit wraps a Chunk; text/doc_id are properties read off it, not fields
    # of Hit itself.
    hits = [Hit(chunk=Chunk(text=doc[:8000], doc_id=f"gold-{i}",
                            chunk_id=f"gold-{i}::0"), score=1.0, rank=i + 1)
            for i, doc in enumerate(gold_docs)]
    context = "\n\n".join(f"[{i}] {doc[:8000]}" for i, doc in enumerate(gold_docs, 1))
    n_sources = len(gold_docs)  # built manually above, so nothing is dropped

    text, validation, repaired = _generate_with_repair(
        question, context, n_sources, tier=tier, strict=False)

    return Answer(
        question=question, text=text, hits=hits,
        refused=validation["refused"],
        partial_decline=_is_partial_decline(text),
        citations_valid=True,
        invalid_citations=validation["invalid_citations"],
        n_citations=validation["n_citations"],
        truncated=validation["truncated"],
        repaired=repaired,
    )
