#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

Implements the T4 §5 diagnostic tree. Everything that can be decided by code
is decided by code; mode 2 needs your eyes and the script says so.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.evaluate import build_retriever, judge_correctness  # noqa: E402
from labs.lab4.rag import answer_with_gold_context  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}

_NUMBER = re.compile(r"\d[\d,]*")

# Part A2 -- human judgements, made by opening the chunks around each gold
# answer (ranks are for the real question against the Lab 4 index, recorded
# in each case's `gold_chunk_ranks`). They override the automated mode; the
# automated mode is kept alongside as `auto_mode` so every override is visible.
HUMAN_CHECKS: dict[str, tuple[int, str]] = {
    "Q04": (4, "A2: the chunk that holds the answer is pre-existing-conditions::m7 "
               "(rider reduces 36 months to 24 or 12, bought at inception or first "
               "renewal) and it ranks 21 -- in the top 30, outside the final 12. The "
               "rank-1 'gold chunk' m0 is a generic intro the lexical matcher picked; "
               "it holds none of the answer"),
    "Q23": (2, "A2: the answer straddles outpatient-and-wellness::m0 ('not covered "
               "under the base plan on any tier ... The optional OPD rider covers "
               "them:') and ::m1 (the table row 'Bronze | Not available'). The "
               "chunker split the sentence from its table: m1 ranks 1, m0 ranks 17, "
               "and the generator got only the Bronze half"),
    "Q29": (4, "A2: claims-timelines::m3 holds the whole answer ('Response to an "
               "Aurora query | 45 days') and is in context at rank 5. The answer gets "
               "45 days right but adds a 30-day deadline from motor-claims-timelines "
               "(ranks 4 and 7): the chunk is complete, off-domain chunks outrank it"),
    "Q32": (2, "A2: the co-payment row ('| Co-payment | 20% above age 60 | 10% above "
               "age 60 | Nil | Nil |', plans-overview::m2) is in context at rank 4, but "
               "the header naming the columns Bronze/Silver/Gold/Platinum is in ::m1, "
               "which is not in the top 30. A table split from its header -- the row "
               "cannot be read"),
}


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test: is the fact the gold answer relies on actually in the corpus?

    The original substring/word-overlap test filters out every token of length
    <=4, which silently drops every number ("36", "45", "24") -- exactly the
    kind of fact a mode-1 vs. mode-3/4 distinction hinges on. Two answers can
    share every generic word ("the", "plan", "covers") while disagreeing on
    the one number that matters, and the old test would call that a match.

    Fix: extract every number in the gold answer and require ALL of them to
    appear in the relevant documents' raw text (numbers are the cheapest,
    highest-precision signal that a specific fact is or isn't in the corpus).
    Only fall back to fuzzy word overlap when the gold answer has no numbers
    to check at all (e.g. a purely qualitative yes/no answer).

    Numbers are matched as whole numbers, not substrings: a substring test
    lets "36" pass on "360" and "12" on "2012", which would wrongly clear
    mode 1 on exactly the facts it exists to check.
    """
    text = " ".join(corpus.get(d, "") for d in relevant_docs)
    if not text:
        return False

    numbers = {n.replace(",", "") for n in _NUMBER.findall(gold_answer)}
    if numbers:
        present = {n.replace(",", "") for n in _NUMBER.findall(text)}
        if not numbers <= present:
            return False

    text_norm = text.lower()
    tokens = [t.strip(".,;()") for t in gold_answer.lower().split()
              if len(t) > 4 and not t[0].isdigit()]
    if not tokens:
        return True
    return sum(1 for t in tokens if t in text_norm) / len(tokens) > 0.5


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_final_k: bool | None = None,
             in_top_30: bool | None = None,
             dropped_by_reranker: bool | None = None,
             chunk_self_retrievable: bool | None = None,
             gold_rank: int | None = None,
             final_k: int = 12) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence).

    Branch order is load-bearing: it is what keeps the modes mutually
    exclusive. Do not re-order it.

    `in_final_k` and `in_top_30` must be measured at FACT granularity (is the
    specific chunk carrying the gold answer retrieved), not document
    granularity. A first version of this checked "is any chunk of the
    relevant DOCUMENT anywhere in the top 30" -- and on this corpus, where one
    document splits into 9-18 topically similar chunks, that check returned
    "yes, rank 1" for cases where the specific fact-bearing chunk never made
    it into the actual k=12 the generator saw. Document-level presence was a
    false positive for the thing that actually matters: did the FACT arrive.
    """
    # Mode 7 first: right answer, wrong citation. Not a retrieval failure at all.
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: is the answer even in the corpus?
    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "a fact the gold answer depends on is not present in the relevant documents"

    # Mode 6: does gold context fix it? THE INVERSION TRAP LIVES HERE.
    # gold_context_fixes_it == True means retrieval starved a capable
    # generator -- that is a RETRIEVAL failure, not generation. Only a
    # confirmed False here means the generator itself is the problem.
    if gold_context_fixes_it is False:
        return 6, "handed the gold documents directly, the answer is still wrong -- generation, not retrieval"
    if gold_context_fixes_it is None:
        return 2, "needs_human_check: gold-context re-generation was not run for this case"

    # From here on, gold_context_fixes_it is True: the generator is capable,
    # so per the T4 §5 tree this is a RETRIEVAL problem. An earlier version
    # sent "gold chunk was already in the final k" back to mode 6 here --
    # that is the inversion trap by another route: the same generator that
    # fails on the retrieved context succeeds on the gold documents, so what
    # differs is the context retrieval assembled, not the generator.
    rank = f"rank {gold_rank}" if gold_rank else "not ranked"
    if in_top_30:
        if dropped_by_reranker:
            return 5, f"gold fact was in the top 30 ({rank}) and dropped by the reranker before final_k"
        if in_final_k:
            # The fact-bearing chunk did arrive. The tree still says 4 (the
            # assembled context, not the generator, is what differs), but gold
            # DOCUMENTS fixing it while the gold CHUNK did not can also mean
            # the chunk holds only part of the answer -- a mode-2 candidate
            # that only a human can settle.
            return 4, (f"gold chunk in the final context ({rank} of {final_k}) yet wrong, "
                       "while gold documents alone fix it -- the retrieved context, not "
                       "the generator, is at fault. needs_human_check: confirm the "
                       "chunk holds the whole answer (else mode 2)")
        return 4, (f"gold fact in the top 30 ({rank}) but outside the final {final_k} "
                   "-- a ranking problem")

    # Fact not recoverable even at k=30.
    if chunk_self_retrievable:
        return 3, "gold chunk is unreachable for the real question even at k=30, but retrieves itself when queried by its own text -- an embedding/query-mismatch, not a broken chunk"
    return 2, "needs_human_check: gold chunk does not even retrieve itself at k=30 -- open the chunk boundaries and look"


def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "█" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal computation -- the automated half of the diagnostic tree.
# ---------------------------------------------------------------------------
def _relevant_chunks(corpus: dict[str, str], relevant_docs: list[str], size: int = 400):
    """Same chunking config as labs/lab4/evaluate.py::build_retriever, so the
    self-retrieval test operates on the exact chunks the live system indexed."""
    return [c for d in relevant_docs if d in corpus for c in markdown_chunks(corpus[d], d, size=size)]


def _chunk_self_retrievable(retriever, chunks, k: int = 5) -> bool:
    """"Is the gold chunk retrievable by its own text?" -- query with each
    candidate chunk's own text and check whether it retrieves itself. If it
    can't even find itself, the chunk is structurally broken (mode 2); if it
    can, the real question's wording is what failed to reach it (mode 3)."""
    for c in chunks:
        hits = retriever.search(c.text, k=k)
        if any(h.chunk.chunk_id == c.chunk_id for h in hits):
            return True
    return False


_WORD = re.compile(r"[a-z0-9]+")


def _identify_gold_chunks(gold_answer: str, chunks, top_n: int = 2):
    """Pin down WHICH specific chunk(s) among the relevant document's many
    chunks actually contain the gold answer's supporting text, by lexical
    overlap. Necessary because a relevant document splits into 9-18
    topically-similar chunks on this corpus -- checking "is any chunk of the
    right document present" is a false positive for "is the specific
    fact-bearing chunk present" (verified: for Q37 this correctly identifies
    the one chunk mentioning the international benefit out of 19 candidates,
    where a document-level check would have falsely passed on any of them).
    """
    gold_tokens = {t for t in _WORD.findall(gold_answer.lower()) if len(t) > 2}
    scored = sorted(chunks, key=lambda c: -len(gold_tokens & set(_WORD.findall(c.text.lower()))))
    return scored[:top_n]


def compute_signals(row: dict, q: dict, corpus: dict[str, str], retriever,
                    final_k: int = 12) -> dict:
    """Run the checks that CODE_GUIDE.md marks 'automated': the final-k /
    top-30 presence check, the mode-6 gold-context regeneration + re-judge,
    and the mode-3/2 self-retrieval probe. Reranker-drop is structurally
    impossible here -- the Lab 4 pipeline has no reranker stage.

    `in_final_k` / `in_top_30` track a SPECIFIC identified gold chunk (see
    `_identify_gold_chunks`), not document-level presence -- document-level
    presence is a false positive whenever a relevant document has multiple
    chunks and only one of them actually carries the needed fact.

    Caller is expected to hold an active Budget context -- both the
    generation call and the judge call inside here record against whichever
    Budget is active on aip.cost's stack, so this function itself takes no
    budget argument.
    """
    relevant = q["relevant_docs"]
    if not relevant:
        return {"gold_context_fixes_it": None, "in_final_k": None, "in_top_30": None,
                "dropped_by_reranker": False, "chunk_self_retrievable": None,
                "gold_rank": None}

    relevant_chunk_objs = _relevant_chunks(corpus, relevant)
    gold_chunks = _identify_gold_chunks(q["gold_answer"], relevant_chunk_objs)
    gold_chunk_ids = {c.chunk_id for c in gold_chunks}

    hits_30 = retriever.search(q["question"], k=30)
    gold_rank = next((i + 1 for i, h in enumerate(hits_30)
                      if h.chunk.chunk_id in gold_chunk_ids), None)
    in_final_k = gold_rank is not None and gold_rank <= final_k
    in_top_30 = gold_rank is not None

    gold_docs = [corpus[d] for d in relevant if d in corpus]
    gold_answer_obj = answer_with_gold_context(q["question"], gold_docs)
    gold_score = judge_correctness(q["question"], gold_answer_obj.text, q["gold_answer"])
    gold_context_fixes_it = (gold_score is not None) and (gold_score >= 2)

    chunk_self_retrievable = None
    if not in_top_30:
        # Probe only the identified gold chunks. Probing every chunk of the
        # relevant documents almost always finds SOME chunk that retrieves
        # itself, which would label every miss mode 3 and hide mode 2.
        chunk_self_retrievable = _chunk_self_retrievable(retriever, gold_chunks, k=30)

    return {
        "gold_context_fixes_it": gold_context_fixes_it,
        "in_final_k": in_final_k,
        "in_top_30": in_top_30,
        "dropped_by_reranker": False,  # no reranker in this pipeline (see labs/lab4/report.md)
        "chunk_self_retrievable": chunk_self_retrievable,
        "gold_rank": gold_rank,
        # Every candidate gold chunk's rank (None = outside the top 30): the
        # evidence a Part A2 human check starts from.
        "gold_chunk_ranks": {c.chunk_id: next((i + 1 for i, h in enumerate(hits_30)
                                               if h.chunk.chunk_id == c.chunk_id), None)
                             for c in gold_chunks},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    ap.add_argument("--final-k", type=int, default=12,
                    help="the final_k the pipeline under diagnosis actually used at generation time")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    retriever = build_retriever()

    out, tally = [], Counter()
    with Budget(limit_usd=1.00, label="lab5-diagnose") as budget:
        for r in failures:
            q = questions[r["id"]]
            signals = compute_signals(r, q, corpus, retriever, final_k=args.final_k)
            auto = classify(r, q, corpus, **{k: v for k, v in signals.items()
                                             if k != "gold_chunk_ranks"},
                            final_k=args.final_k)
            mode, evidence = HUMAN_CHECKS.get(r["id"], auto)
            tally[mode] += 1
            out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                        "mode_name": MODES[mode], "evidence": evidence,
                        "question": q["question"], "gold_answer": q["gold_answer"],
                        "answer": r["answer"][:300], "signals": signals,
                        "auto_mode": auto[0], "auto_evidence": auto[1],
                        "human_checked": r["id"] in HUMAN_CHECKS,
                        "needs_human_check": ("needs_human_check" in auto[1]
                                              and r["id"] not in HUMAN_CHECKS)})
            override = f"  [auto: {MODES[auto[0]]}]" if mode != auto[0] else ""
            print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}{override}")

    print("\n" + pareto(tally))
    print("\n" + budget.report())
    print("\nCases marked needs_human_check are Part A2. Open them.")

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
