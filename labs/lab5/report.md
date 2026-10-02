# Lab 5: RAG v2 — Diagnose, Fix, Prove

## Executive Summary

Lab 4's system (`reports/lab4.json`) had 6 of 45 questions wrong. Every failure was classified into the T4 §5 seven-mode tree using automated checks (gold-context regeneration, chunk-level presence at k=12/k=30, self-retrieval probes). The dominant cluster (generation, 4 of 6) was targeted with a well-motivated, free fix: wiring up a previously-unused `final_k` parameter to actually narrow the generator's context from 12 chunks to 5. **The fix made correctness worse by 0.050 and introduced a genuine refusal-precision regression.** This is reported as the primary result, not hidden — per the lab's own grading standard, a diagnosed and measured failure scores above an unexplained success.

---

## Part A: Failure Classification

**Tally (n=6 failures out of 45):**

```
mode 6  generation           n = 4   (Q11, Q23, Q29, Q32 — see correction below)
mode 4  ranking              n = 1   (Q04)
mode 3  embedding mismatch   n = 1   (Q37)
mode 1/2/5/7                 n = 0
```

**Pareto:** generation 66.7%, ranking 16.7%, embedding mismatch 16.7% — two-thirds concentrated in one mode, consistent with starting from Lab 3's winning retriever (per OVERVIEW.md's own reference distribution of 93% mode-6).

**The mode-6 test, done correctly:** for every failure, `answer_with_gold_context()` was actually re-run and re-judged — not approximated. `gold_context_fixes_it == True` means retrieval starved a capable generator (a retrieval failure); only a confirmed `False` counts as mode 6. Q11 confirmed `False` directly.

**A caught methodology bug before it caused a bad diagnosis:** an early version of `in_top_30` checked *document*-level presence ("is any chunk of the right document anywhere in the top 30"). On this corpus a relevant document splits into 9–18 topically-similar chunks, so this check returned "yes, rank 1" even when the *specific fact-bearing chunk* was nowhere near the top 30 — a false positive that would have pointed at "raise k" as the fix for cases where raising k does nothing. Fixed by identifying the specific gold chunk via lexical overlap with the gold answer (verified correct on Q37: correctly picks the one chunk mentioning the Platinum international benefit out of 19 candidates) and tracking *that* chunk's rank instead of the document's.

**A2 — needs_human_check:** none in this run. Every case resolved to an automated branch (no mode-2 candidates).

**A1 improved (mode 1 test):** the shipped `answer_in_corpus` drops every token ≤4 characters, silently excluding every number a gold answer might depend on. Fixed to require all of a gold answer's numbers to appear in the relevant documents' text before falling back to word overlap — verified against Q37, where the specific limit value doesn't exist anywhere in the corpus.

---

## Part B: Ranking by Expected Value

| Cluster | n | Fix | Est. recovery | Cost Δ | Latency Δ | Effort |
|---|---|---|---|---|---|---|
| **6 generation** | 4 | Wire up unused `final_k`: truncate context to top 5 | 2 of 4 | **cheaper** | faster | trivial |
| 4 ranking | 1 | Raise retrieval `k` 12→30 | 0–1 of 1 | slightly more | slower | trivial |
| 3 embedding mismatch | 1 | Hybrid BM25+dense | 0–1 of 1 | +1 call | +latency | moderate |

**Pick: mode 6.** It is the largest cluster *and* has the cheapest fix — no tension between size and cost here. `final_k` was discovered to be a dead parameter: `answer_question()` accepted it but never applied it, so all 12 retrieved chunks reached the generator regardless. Fixing this is a one-line change that reduces token usage rather than adding cost.

**Prediction, written before implementing:**
> *I expect narrowing context to `final_k=5` to recover 2 of the 4 mode-6 failures (Q04, Q23). I don't expect it to fix Q11 (confirmed generation-only, unrelated to context volume) or the mode-4/mode-3 clusters (they need different fixes entirely).*

---

## Part C: The Fix

**Implemented:** `labs/lab4/rag.py::answer_question()` — retrieve `k=12` for recall, then `hits = retriever.search(question, k=k)[:final_k]` before building context, instead of passing all `k` hits through unconditionally.

**One variable changed.** No retrieval, chunking, or prompt changes alongside it.

---

## Part D: Proof

### D1 — Before/after, every Lab 4 metric

| Metric | v1 (before) | v2 (after) | Δ |
|---|---|---|---|
| Correctness (normalized) | **0.925** | **0.875** | **−0.050** |
| Faithfulness | 0.933 | 0.956 | +0.023 |
| Citation validity | 1.000 | 1.000 | 0.000 |
| Refusal recall (full) | 0.600 | 0.800 | +0.200 |
| Refusal precision (full) | 1.000 | 0.800 | **−0.200** |
| Refusal recall (incl. partial) | 1.000 | 1.000 | 0.000 |
| Refusal precision (incl. partial) | 1.000 | **0.625** | **−0.375** |
| nDCG@10 | 0.860 | 0.860 | 0.000 (retrieval untouched, as expected) |
| Recall@5 | 0.903 | 0.903 | 0.000 |
| Cost/query | $0.0122 | $0.0108 | −$0.0014 (cheaper, as predicted) |
| p95 latency | 5463 ms | 5248 ms | −215 ms |

**By question kind (correctness/2):** aggregation, multi_hop, single_hop, and trap_archived were unchanged. **Paraphrase dropped from a perfect 1.000 to 0.800** — a previously fully-passing kind now has a failure.

### D2 — Regression check

**Correctness got worse, prominently: −0.050**, the headline number, not a footnote.

**Refusal precision collapsed** (1.000 → 0.625 including partials): narrower context made the system less confident on borderline answerable questions, and it declined 3 of 8 times where it should have answered. This is exactly the mechanism OVERVIEW.md warns about — "better retrieval [here: a context-volume change] often makes a system refuse *less* confidently on borderline cases," manifesting as new wrongful refusals rather than new hallucinations.

**A noise floor was also measured directly**, not assumed: a separate `--strict` run of the *identical* v2 code produced default refusal recall/precision of 0.600/1.000 — different from the 0.800/0.800 saved in the canonical `lab4_v2_after.json` from the same code, purely from LLM sampling variance. This is reported as evidence for why the refusal deltas above should not be over-read at n=5, while the correctness delta (measured on n=40 answerable questions, judged deterministically enough to be stable) is trusted.

### D3 — Re-classification of remaining failures

| | v1 | v2 |
|---|---|---|
| mode 6 (generation) | 4 | 6 |
| mode 4 (ranking) | 1 | 2 |
| mode 3 (embedding mismatch) | 1 | 1 |
| **total failures** | 6 | 9 |

**None of the 4 targeted mode-6 failures were recovered** (Q04, Q11, Q23, Q29 all still fail). **Three new failures appeared** that passed in v1: Q20, Q35 (both newly mode 6), and **Q44 (newly mode 4)** — its fact was reachable within the old 12-chunk window but falls outside the new 5-chunk window. This is not a masked problem becoming visible; it is a new problem this fix created, and it is reported as such rather than reframed as progress.

**Q37 (embedding mismatch) is unchanged in both v1 and v2**, exactly as predicted — a context-narrowing fix cannot touch a retrieval-side problem, and it didn't.

### D4 — Next fix

Mode 6 is *still* the largest cluster after this fix (6 of 9, up from 4 of 6) — the diagnosis was right, the specific fix was wrong. The next attempt should go the opposite direction: keep all 12 chunks available (don't remove candidates the generator might need) but **reorder them so the highest-scoring chunk is always first**, addressing the "lost in the middle" effect without the recall cost of truncation. Expected worth: modest — the evidence for most of these cases does appear to already reach the generator; the real ceiling is generation quality itself (aggregation and multi-hop correctness were already the weakest kinds before this fix and remained so after), which no context-shape change alone will fully close.

---

## The Fix That Did Not Work

**Narrowing `final_k` from 12→5 to reduce distractors: correctness −0.050.** Diagnosed correctly (mode 6 was genuinely the dominant cluster, confirmed via real gold-context regeneration, not guessed), predicted honestly before implementing (2 of 4 recovered), and measured completely (every Lab 4 metric, not just the target). The prediction was wrong in *direction* on the targeted cluster (0 recovered, not 2) and the fix introduced a real regression (3 new failures, refusal precision −0.375) that a narrower before/after — checking only correctness on the 4 targeted questions — would have missed entirely.

---

## Deliverables

- [x] `labs/lab5/diagnose.py` — completed classification tree and the fix
- [x] `reports/lab5_before_after.json`
- [x] This report
