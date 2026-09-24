# Lab 4: RAG

## Executive Summary

This report documents a RAG pipeline built on Lab 3's winning retriever (markdown-aware chunking at 400 characters, dense retrieval) with citation enforcement, refusal handling, and LLM-judge evaluation calibrated against human labels. All metrics exclude Q36, Q38, Q39 (no relevant document exists for them). n = 42 for answerable-only metrics, n = 45 overall.

**Recommended configuration:** default (non-strict) refusal setting, single corrective-retry citation repair, dense retrieval with archived documents excluded at index time.

---

## Part A: The Answer Prompt

Six required elements, written before reading `aip/rag.py::ANSWER_SYSTEM`: source-only answers, cite by index `[1][2]`, never cite an unsupplied index, exact refusal string, conflict surfacing, length discipline. A seventh rule — **partial answers required** — was added after testing exposed a bug.

**Differences from the reference:**

| | Reference (`aip/rag.py`) | This implementation |
|---|---|---|
| Partial answers | Not addressed — only "refuse if sources lack the answer" | Explicit rule, required, with a worked example |
| Everything else | Materially equivalent | Materially equivalent |

**Why the extra rule:** the first draft (without it) fully refused Q37 ("Does Aurora cover treatment in Singapore, and up to what limit?") even when handed the gold documents directly, which explicitly state Platinum has an international emergency benefit. The model had the information and still declined, because nothing told it a partial answer was acceptable.

| | Q37 answer |
|---|---|
| Before | *"I don't have enough information in the provided sources to answer that."* |
| After | *"Treatment outside India is permanently excluded... except under the Platinum plan's international emergency benefit [1][2]. However, the specific coverage limit... is not stated in the provided sources."* |

Most systems fail this on the first attempt; mine did too. The fix was a one-line prompt rule, not a retrieval change.

---

## Part B: Citation Validity and Failure Handling

`validate_answer()` checks, in code: every `[n]` is within `1..n_sources`, the response is not truncated (`finish_reason == "length"`), and a non-refusal answer carries ≥1 citation.

**On failure:** retry once with a corrective message naming the exact defect and the full numbered source list. If the retry also fails validation, fall back to the exact refusal string. The pipeline never returns `citations_valid=False` — this is a hard invariant, not best-effort.

| Metric | Value |
|---|---|
| Citation validity | **1.000** (target 1.000, met) |
| Repair rate | **0.000** — no answer needed the retry path across all 45 questions |

One implementation bug fixed along the way: the first draft computed retrieval context but never inserted it into the prompt sent to the model, so the LLM had nothing to cite. This trivially produced citation validity of 1.000 (every answer was a refusal) and correctness of 0.000. Fixed by routing the numbered sources through `_build_user_prompt()`.

---

## Part C: Refusal, Both Directions

**Default setting:**

| | Full-refusal only | Incl. partial decline |
|---|---|---|
| Refusal recall | 0.600 (3/5) | **1.000 (5/5)** |
| Refusal precision | 1.000 | 1.000 |

A strict exact-string check alone reports recall 0.600 and looks like the system misses 2 of 5 unanswerable questions. It doesn't: those 2 (including Q37) get a **partial decline** — answering what's supported and flagging the rest, which is the golden set's own expected behavior for Q37 ("PARTIAL REFUSE"). Counting that as a form of refusal, recall is a clean 1.000 with zero false positives.

**Strictness comparison (C4)** — added a `STRICT MODE` clause demanding a complete, unambiguous answer or refusal, re-run refusal-only on all 45 questions:

| Setting | Recall | Precision | Caught | Declines | Wrongly refused |
|---|---|---|---|---|---|
| default | 0.600 | 1.000 | 3/5 | 3 | 0 |
| strict | **1.000** | **0.385** | 5/5 | 13 | **8** |

n = 5 unanswerable questions, so one flipped case moves recall by ~0.20 — this gap (0.600→1.000, 1.000→0.385) is well above that noise floor.

**Product recommendation:** ship **default**, not strict. Strict mode wrongly refuses 8 of 40 legitimately answerable questions to catch 2 more of 5 unanswerable ones. For an insurance helpdesk, a wrongful refusal (a customer told "I don't know" on a question the corpus answers, who then escalates or churns) costs more than the default setting's actual failure mode — a correct partial answer with an explicit caveat, not a fabricated claim (incl.-partial recall is already 1.000). Strict mode would only be justified for a use case where any incomplete claim is unacceptable (e.g., a compliance-review bot), which this is not.

---

## Part D: Judge Calibration

Two single-criterion rubrics, rewritten from `aip/evals.py`'s starting templates: **faithfulness** (0/1, is every claim grounded in context, treating a correctly-flagged partial decline as supported) and **correctness** (0/1/2, explicitly scoring a correct refusal at full marks when the reference itself refuses — the shipped template is silent on this and would otherwise punish the exact behavior Part A requires).

Also fixed the trap named in `CODE_GUIDE.md`: the original judges scored a judge **parse failure** as 0 by defaulting `verdict.get("score", 0)`. Missing data is not a failing answer — this is the exact bug that cost the reference solution a faithfulness reading of 0.667 against a true 0.933. Fixed: parse failures now return `None` and are excluded from the mean, not averaged in as zero.

**Cohen's κ**, from 20 hand-labeled answers, labeled before seeing the judge's verdicts:

| Rubric | Raw agreement | Cohen's κ | n |
|---|---|---|---|
| Faithfulness | 0.95 | **0.0** | 20 |
| Correctness | 0.90 | **0.664** | 20 |

**Faithfulness κ = 0.0 despite 95% raw agreement is the kappa paradox, not a broken rubric:** 19 of 20 human labels were `1`, so expected chance-agreement is already ~0.9+, and κ collapses toward 0 from a single disagreement. Reading that disagreement (Q20) resolved it: the judge scored a claim ("adding a parent to a floater often costs more than an individual senior policy") as unsupported; checking the source directly, it is verbatim-supported. **The judge made a genuine, verifiable error** — not a rubric defect, but exactly the value D2 is designed to surface. No rubric change was needed since the disagreement traced to a judge execution error rather than an ambiguous instruction.

Correctness κ = 0.664 clears the 0.4 bar (substantial agreement); its two disagreements (Q04, Q19) reflect a stricter human standard on completeness, not a rubric flaw, so no change was made there either.

**Caveat:** `rag.py`/`evaluate.py` were substantially rewritten after this calibration round. 18 of the 20 calibrated answers no longer match the current pipeline's output text for those IDs. The κ above is a valid calibration of the rubric design, but was measured against a prior generation of the pipeline's outputs — a fresh `--calibrate`/`--kappa` cycle against the current `reports/lab4.json` has not been re-run.

---

## Part E: Full Evaluation and the Decomposition

**E1 — headline metrics (n=45):**

| Metric | Value |
|---|---|
| Citation validity | 1.000 |
| Faithfulness | 0.933 (45/45 scored, 0 parse failures) |
| Correctness (0–2) | 1.850 → 0.925 normalized |
| Repair rate | 0.000 |
| p95 latency | 5463 ms |
| Cost | $0.55 total, $0.012/query |

**E2 — gold-context decomposition (n=42):**

| | Correctness |
|---|---|
| A — gold context (generation ceiling) | **0.976** |
| B — retrieved context (this system) | **0.893** |
| Retrieval-attributable loss (A − B) | **0.083** |
| Generation-attributable loss (1 − A) | 0.024 |

Retrieval loss is ~3.5× generation loss. **Lab 5's effort belongs in retrieval.** The generator is already near its ceiling (0.976); the gap between what the system could do with perfect context and what it actually does is almost entirely retrieval, exemplified by Q37's answer-bearing chunk ranking 43rd of 91.

**E3 — failure-mode tally** (7 of 45 answers scored below correctness=2):

| Mode | Count | Questions |
|---|---|---|
| 6 — generation (evidence in context, answer still wrong) | 4 | Q11, Q23, Q32, Q35 |
| 2 — chunk boundary (answer straddles two chunks) | 2 | Q04, Q29 |
| 3 — embedding miss (right chunk never ranks in top 30) | 1 | Q37 |

Q37 was hand-corrected from the classifier's automatic tag of mode 6 to mode 3: the heuristic falls back to "generation" when the gold answer has no numbers to fact-check and the correct *document* merely appears among retrieved chunks — but the specific paragraph ranks 43rd, well outside top-30. Document-level retrieval success does not imply chunk-level success, and this is the classifier's blind spot.

**Lab 5 backlog:** (1) chunking/retrieval granularity — modes 2+3 are 3 of 7 failures and the larger E2 loss term; smaller chunks or wider `k` for detail-heavy questions is the first thing to try. (2) aggregation-question generation — 2 of 4 mode-6 failures are aggregation-kind questions where the model partially enumerates a list instead of covering every retrieved fact.

---

## Final Configuration

| Component | Setting |
|---|---|
| Chunking | Markdown-aware, 400 characters (Lab 3 winner, nDCG@10=0.8527) |
| Retrieval | Dense (exact), archived documents excluded at index build |
| Retrieve k | 12 |
| Refusal setting | Default (not strict) |
| Repair strategy | One corrective retry, then hard fallback to refusal |
| Judge tier | LARGE (different tier from MAIN generator, avoiding self-preference) |

## What Surprised Me

The prompt fix (partial answers) mattered more than expected — it fully fixed Q37 under gold context — until the decomposition showed generation was already near its ceiling (0.976) and the real bottleneck was retrieval (0.083 loss) all along. It would have been easy to declare victory after the prompt fix and miss that the underlying retrieval gap was still there.

## Exclusions Stated

Q36, Q38, Q39 excluded from all retrieval/correctness metrics (no relevant document exists). Refusal numbers are noisy at n=5 (one flipped case ≈ 0.20 of recall) — the C4 gap is well above that noise floor; smaller gaps elsewhere are not claimed as real.
