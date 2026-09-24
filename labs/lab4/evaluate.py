#!/usr/bin/env python3
"""Lab 4 evaluation harness.

    python labs/lab4/evaluate.py --full --save reports/lab4.json
    python labs/lab4/evaluate.py --gold-context
    python labs/lab4/evaluate.py --strict         # C4: both refusal settings
    python labs/lab4/evaluate.py --calibrate      # writes the hand-label sheet
    python labs/lab4/evaluate.py --kappa
    python labs/lab4/evaluate.py --failures       # E3: the Lab 5 backlog
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import judge_agreement, llm_judge  # noqa: E402
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import answer_question, answer_with_gold_context  # noqa: E402

GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
LABEL_SHEET = ROOT / "labs/lab4/calibration_labels.jsonl"


def build_retriever():
    """Lab 3's winning configuration: markdown-aware chunks at 400 characters,
    exact dense retrieval (see reports/lab3_sweeps.json -- nDCG@10=0.8527,
    beating fixed/sliding/recursive and every other chunk size tested).

    Archived documents are dropped at index-build time rather than filtered
    per-query. Lab 3's D3 filtered `status=archived` at query time through
    Chroma's metadata index; this harness uses DenseRetriever, which has no
    metadata filter, so the equivalent here is simplest applied once, at
    build time, since the filter value (exclude archived) never changes
    between queries. D1 already showed exact and ANN search give identical
    quality at this corpus size, so nothing about D1's finding is affected.
    """
    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items() if "ARCHIVED" not in doc_id
              for c in markdown_chunks(text, doc_id, size=400)]
    return DenseRetriever(chunks, show_progress=False)


# ---------------------------------------------------------------------------
# D1 -- the two single-criterion judges
# ---------------------------------------------------------------------------
# Rewritten from aip/evals.py's starting templates. The changes that matter:
# faithfulness now treats a partial decline as supported (rather than
# penalising the part it didn't answer), and correctness now scores a correct
# refusal at full marks instead of zero -- the shipped templates are silent
# on both, which punishes exactly the behaviour Part A asks the system to do.

RUBRIC_FAITHFULNESS = """\
Decide whether every claim in ANSWER is backed by CONTEXT. Judge grounding
only -- not helpfulness, not whether the claim happens to be true in the real
world, only whether CONTEXT actually says it.

- A claim is unsupported if CONTEXT does not contain it, even if it is a true
  fact about insurance in general.
- If CONTEXT gives a qualified or partial statement (e.g. "usually", "in most
  cases", or a range) and ANSWER states it as absolute or picks a single
  value, that is a strengthening claim -- unsupported.
- A refusal is supported when CONTEXT genuinely lacks the answer, and
  unsupported when CONTEXT actually contains it.
- A PARTIAL answer -- one that answers what it can and explicitly flags the
  rest as not covered -- counts as supported, as long as the part it does
  answer is grounded. Do not dock it for the part it declined; declining is
  the correct move for that part.
- Citation index correctness (whether [3] is a valid source number) is
  checked separately in code -- ignore it here.

CONTEXT:
{context}

ANSWER:
{answer}

Return JSON: {{"score": 0 or 1, "unsupported_claims": [...], "reason": "one sentence"}}
"""

RUBRIC_CORRECTNESS = """\
Compare CANDIDATE to REFERENCE, both answering the same QUESTION.

Refusal cases -- check these before anything else:
- REFERENCE itself is a refusal / states there is no answer in the sources:
  CANDIDATE refusing scores 2 (that is the correct behaviour).
  CANDIDATE answering with confidence scores 0, no matter how plausible.
- REFERENCE contains a partial answer plus an acknowledged gap: CANDIDATE
  must both answer the covered part and name the gap to score 2. Doing only
  one of those scores 1. Inventing an answer for the gap scores 0.

Otherwise, score against what the QUESTION actually asked, not against every
detail REFERENCE happens to include:
- Score 2: substantively answers the question as asked, matching REFERENCE's
  content (wording can differ). Extra correct detail is not penalised.
  Omitted detail is not penalised if the question didn't ask for it.
- Score 1: gets the core answer right but leaves out something the question
  DID ask for, or adds something that contradicts REFERENCE.
- Score 0: wrong, or a confident answer where REFERENCE says to refuse.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Return JSON: {{"score": 0|1|2, "reason": "one sentence"}}
"""


def _score(verdict: dict) -> int | None:
    """Pull the numeric score out of a judge verdict -- or None if the judge
    failed to produce parseable JSON.

    CODE_GUIDE.md's trap #2, almost verbatim: `llm_judge` sets `parse_error`
    when the verdict didn't parse, most often because a reasoning judge burned
    its token budget on hidden thinking and the JSON got cut off mid-object.
    That is missing data, not a bad answer. Silently defaulting it to 0 is
    exactly what produced the reference's phantom faithfulness drop (0.667
    measured vs. 0.933 true) -- so this function refuses to do that.
    """
    if verdict.get("parse_error"):
        return None
    return int(verdict.get("score", 0))


def judge_faithfulness(answer_text: str, context: str) -> int | None:
    verdict = llm_judge(RUBRIC_FAITHFULNESS.format(
        context=context[:8000], answer=answer_text), tier="LARGE")
    return _score(verdict)


def judge_correctness(question: str, candidate: str, reference: str) -> int | None:
    verdict = llm_judge(RUBRIC_CORRECTNESS.format(
        question=question, reference=reference, candidate=candidate), tier="LARGE")
    return _score(verdict)


def _mean(values) -> float:
    """Average over the cases that actually got a score. A None here means
    the judge failed to parse, not that the answer scored zero -- mixing the
    two back in would reproduce the exact bias this lab is warning about."""
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else 0.0


# ---------------------------------------------------------------------------
def evaluate_all(questions, retriever, *, strict: bool = False,
                 judge: bool = True) -> list[dict]:
    """Answer and (optionally) judge every question, timing each end to end.

    `judge=False` skips both judge calls -- used by run_strictness(), which
    only needs refuse/don't-refuse and would otherwise pay for 90 judge calls
    to measure something judging has no bearing on.
    """
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        a = answer_question(q["question"], retriever, strict=strict)
        latency_ms = (time.perf_counter() - t0) * 1000
        ctx = format_context(a.hits)
        rows.append({
            "id": q["id"], "kind": q["kind"],
            "unanswerable": not q["relevant_docs"] or q["kind"] == "unanswerable",
            "question": q["question"], "answer": a.text, "refused": a.refused,
            "partial_decline": a.partial_decline,
            "citations_valid": a.citations_valid,
            "invalid_citations": a.invalid_citations,
            "n_citations": a.n_citations, "repaired": a.repaired,
            "truncated": a.truncated, "latency_ms": latency_ms,
            "faithfulness": judge_faithfulness(a.text, ctx) if judge else None,
            "correctness": (judge_correctness(q["question"], a.text, q["gold_answer"])
                            if judge else None),
            "retrieved": [h.doc_id for h in a.hits],
            "relevant": q["relevant_docs"], "gold_answer": q["gold_answer"],
        })
    return rows


def refusal_stats(rows: list[dict]) -> dict:
    """C3/C4: both directions, with raw counts, under two definitions of
    "declined".

    n=5 unanswerable questions means one flipped case moves precision by
    roughly 0.12 and recall by roughly 0.20 -- ratios alone hide that, raw
    counts don't, so both are returned.

    The second definition folds partial declines (see rag.py's
    _is_partial_decline) into "declined", because a system that answers 80%
    of an unanswerable question and flags the rest did partially refuse, and
    counting it as a clean non-refusal understates recall on questions like
    Q37/Q40 that expect exactly that behaviour.
    """
    una = [r for r in rows if r["unanswerable"]]
    out = {"n_unanswerable": len(una), "n_answerable": len(rows) - len(una)}
    for suffix, declined in (
        ("", lambda r: r["refused"]),
        ("_incl_partial", lambda r: r["refused"] or r.get("partial_decline")),
    ):
        declined_rows = [r for r in rows if declined(r)]
        caught = sum(1 for r in una if declined(r))
        out[f"recall{suffix}"] = caught / len(una) if una else 0.0
        out[f"precision{suffix}"] = caught / len(declined_rows) if declined_rows else 1.0
        out[f"caught{suffix}"] = caught
        out[f"n_declined{suffix}"] = len(declined_rows)
    return out


def print_report(rows: list[dict], budget: Budget | None = None) -> None:
    ans = [r for r in rows if not r["unanswerable"]]
    rs = refusal_stats(rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    n_parse_fail = sum(1 for r in rows if r["faithfulness"] is None)

    print(f"\nn = {len(rows)}  ({len(ans)} answerable, {rs['n_unanswerable']} unanswerable)")
    print(f"citation validity   {_mean(r['citations_valid'] for r in rows):.3f}   (target 1.000)")
    print(f"faithfulness        {_mean(r['faithfulness'] for r in rows):.3f}"
          f"   (scored {len(rows) - n_parse_fail}/{len(rows)}, {n_parse_fail} judge parse failures excluded)")
    corr = _mean(r["correctness"] for r in ans)
    print(f"correctness (0-2)   {corr:.3f}   normalised {corr / 2:.3f}")
    print(f"refusal recall      {rs['recall']:.3f}   ({rs['caught']}/{rs['n_unanswerable']} full refusals)")
    print(f"refusal precision   {rs['precision']:.3f}   ({rs['caught']}/{rs['n_declined']} declines)")
    print(f"  incl. partials    recall {rs['recall_incl_partial']:.3f}"
          f" ({rs['caught_incl_partial']}/{rs['n_unanswerable']})"
          f"   precision {rs['precision_incl_partial']:.3f}"
          f" ({rs['caught_incl_partial']}/{rs['n_declined_incl_partial']})")
    print(f"repair rate         {_mean(r['repaired'] for r in rows):.3f}"
          "   (fraction that needed the citation-repair retry)")
    print(f"p95 latency         {latencies[int(0.95 * (len(latencies) - 1))]:.0f} ms")
    if budget is not None:
        print(f"cost                ${budget.spent_usd:.4f} total"
              f"   (${budget.spent_usd / max(len(rows), 1):.4f}/query)")

    print("\nby question kind (mean correctness / 2):")
    for kind in sorted({r["kind"] for r in ans}):
        sub = [r for r in ans if r["kind"] == kind]
        print(f"  {kind:<16} {_mean(r['correctness'] for r in sub) / 2:.3f}   n={len(sub)}")


def run_full(save: str = "", strict: bool = False) -> None:
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    with Budget(limit_usd=1.00, label="lab4-full") as b:
        rows = evaluate_all(questions, retriever, strict=strict)
    print_report(rows, b)

    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")


def run_strictness() -> None:
    """C4: run both refusal settings so the recall/precision trade-off is
    something you measured, not something you asserted.

    Judging is skipped on purpose -- C4 is about where the refusal dial sits,
    and paying for 90 extra judge calls (45 questions x 2 rubrics x 2
    settings) would double the cost of this run without telling us anything
    the citation/refusal fields don't already say.
    """
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    settings = {}
    with Budget(limit_usd=2.00, label="lab4-strictness"):
        for label, strict in (("default", False), ("strict", True)):
            rows = evaluate_all(questions, retriever, strict=strict, judge=False)
            s = refusal_stats(rows)
            s["wrongly_refused"] = sum(1 for r in rows
                                       if r["refused"] and not r["unanswerable"])
            settings[label] = s

    print(f"\n{'setting':<10}{'recall':>9}{'precision':>11}{'caught':>9}"
          f"{'declines':>10}{'wrongly refused':>17}")
    print("-" * 66)
    for label, s in settings.items():
        caught_str = f"{s['caught']}/{s['n_unanswerable']}"
        print(f"{label:<10}{s['recall']:>9.3f}{s['precision']:>11.3f}{caught_str:>9}"
              f"{s['n_declined']:>10}{s['wrongly_refused']:>17}")
    n_una = settings["default"]["n_unanswerable"]
    print(f"\nn = {n_una} unanswerable questions, so one flipped case is worth "
          f"~{1 / n_una:.2f} of recall -- treat any gap under ~0.15 as noise.")


def run_gold_context() -> None:
    """E2: the decomposition. Ten minutes, highest signal-to-noise experiment
    in the lab."""
    questions = [q for q in load_questions() if q["relevant_docs"]]
    retriever = build_retriever()
    corpus = load_corpus()

    retrieved_scores, gold_scores = [], []
    with Budget(limit_usd=1.00, label="lab4-decomposition"):
        for q in questions:
            a = answer_question(q["question"], retriever)
            retrieved_scores.append(judge_correctness(q["question"], a.text, q["gold_answer"]))
            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus])
            gold_scores.append(judge_correctness(q["question"], g.text, q["gold_answer"]))

    A, B = _mean(gold_scores) / 2, _mean(retrieved_scores) / 2
    print(f"\nn = {len(questions)} answerable questions")
    print(f"correctness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever number is larger is where Lab 5's effort should go.")


# ---------------------------------------------------------------------------
# E3 -- tag every wrong answer with one of T4 §5's seven failure modes.
# Mode 5 (reranking) cannot occur here: this pipeline has no reranker.
# ---------------------------------------------------------------------------
FAILURE_MODES = {
    1: "missing content -- the answer is not in the corpus at all",
    2: "chunk boundary -- the answer straddles two chunks, neither has it whole",
    3: "embedding miss -- the right chunk exists but never ranks in the top 30",
    4: "ranking -- the right chunk is in the top 30 but misses the final top-k",
    6: "generation -- the right chunk WAS in context and the answer is still wrong",
    7: "presentation -- the content is right but the citation is wrong or missing",
}

_NUMBER = re.compile(r"\d[\d,]*")


def evidence_in_context(gold_answer: str, context: str) -> bool | None:
    """A cheap, deterministic proxy for "did the evidence actually arrive":
    check whether every number in the gold answer shows up somewhere in the
    context text.

    Doc-level retrieval success isn't enough to separate mode 2 from mode 6 --
    a question can retrieve the right *document* and still fail because the
    chunker cut the answer's row out of the table it lived in (right doc,
    wrong chunk). Matching the gold answer's own numbers against the context
    catches that distinction for free, without asking another model to judge
    it. Returns None when the gold answer has no numbers to check against.
    """
    numbers = {n.replace(",", "") for n in _NUMBER.findall(gold_answer)}
    if not numbers:
        return None
    flat_context = context.replace(",", "")
    return all(n in flat_context for n in numbers)


def classify_failure(row: dict, retriever) -> tuple[int, str]:
    """Assign exactly one failure mode to a wrong answer, mechanically where
    the data allows it. Order matters: presentation is checked first because
    it's provable from the row alone; only call it generation once we can
    show the evidence genuinely reached the model's context."""
    if not row["citations_valid"]:
        return 7, FAILURE_MODES[7]

    hits = retriever.search(row["question"], k=30)
    context = format_context(hits[:5])
    present = evidence_in_context(row["gold_answer"], context)
    doc_matched = bool(set(row["retrieved"]) & set(row["relevant"]))

    if present:
        return 6, FAILURE_MODES[6]              # evidence present, still wrong
    if present is None:                          # nothing numeric to check
        return (6, FAILURE_MODES[6]) if doc_matched else (4, FAILURE_MODES[4])

    rank = next((i + 1 for i, h in enumerate(hits) if h.doc_id in row["relevant"]), None)
    if doc_matched:
        return 2, FAILURE_MODES[2]               # right doc, wrong chunk
    if rank:
        return 4, FAILURE_MODES[4]               # ranked, but outside final-k
    return 3, FAILURE_MODES[3]                   # never ranked at all


def run_failures() -> None:
    """E3: tag every answer that scored below 2 with a failure mode. This
    tally is Lab 5's backlog -- bring it."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    wrong = sorted((r for r in rows if r["correctness"] is not None and r["correctness"] < 2),
                   key=lambda r: r["correctness"])
    retriever = build_retriever()

    tally: dict[int, int] = {}
    print(f"{len(wrong)} of {len(rows)} answers scored below 2\n")
    for r in wrong:
        mode, why = classify_failure(r, retriever)
        tally[mode] = tally.get(mode, 0) + 1
        r["failure_mode"] = mode
        print(f"{r['id']:<5} ({r['kind']:<13}) correctness={r['correctness']} "
              f"faithfulness={r['faithfulness']} -> MODE {mode}: {why}")
        print(f"  Q:    {r['question'][:94]}")
        print(f"  gold: {' '.join(r['gold_answer'].split())[:94]}")
        print(f"  got:  {' '.join(r['answer'].split())[:94]}\n")

    print("tally (Lab 5 backlog, largest first):")
    for mode, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  mode {mode}   n={n:<3} {FAILURE_MODES[mode]}")

    out = ROOT / "reports/lab4_failures.json"
    out.write_text(json.dumps(wrong, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


def make_calibration_sheet() -> None:
    """D2: writes 20 answers to hand-label BEFORE looking at the judge."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    sample = rows[:20]
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "question": r["question"], "answer": r["answer"],
        "gold_answer": r["gold_answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    print(f"wrote {LABEL_SHEET}")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")


def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    machine = {r["id"]: r for r in
               json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))}
    for field in ("faithfulness", "correctness"):
        # Drop pairs where either side is missing -- a machine parse failure
        # has no score to agree or disagree with, and pairing it with 0 would
        # deflate kappa for a reason that has nothing to do with the rubric.
        labeled = [r for r in human if r[f"human_{field}"] is not None]
        pairs = [(machine[r["id"]][field], r[f"human_{field}"]) for r in labeled
                 if machine[r["id"]][field] is not None]
        if not pairs:
            print(f"{field}: no usable labels yet")
            continue
        m_scores = [p[0] for p in pairs]
        h_scores = [p[1] for p in pairs]
        print(f"{field}: {judge_agreement(m_scores, h_scores)}")
        disagreements = [r["id"] for (m, h), r in zip(pairs, labeled) if m != h]
        if disagreements:
            print(f"  disagreements: {', '.join(disagreements)}")
    print("\nkappa < 0.4 -> fix the rubric, not the model. Read your disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-context", action="store_true")
    ap.add_argument("--strict", action="store_true", help="C4: both refusal settings")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--failures", action="store_true", help="E3: the Lab 5 backlog")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    if a.full:
        run_full(a.save)
    if a.gold_context:
        run_gold_context()
    if a.strict:
        run_strictness()
    if a.calibrate:
        make_calibration_sheet()
    if a.kappa:
        report_kappa()
    if a.failures:
        run_failures()
    if not any([a.full, a.gold_context, a.strict, a.calibrate, a.kappa, a.failures]):
        ap.print_help()
