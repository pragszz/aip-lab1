# Lab 7: Ship It

**Part E (the two-page evaluation report, all seven sections) is [`EVALUATION_REPORT.md`](../../EVALUATION_REPORT.md)** at the repo root, the name the brief's deliverables list uses. This file is the lab write-up for Parts A–D: what was built, and the evidence for each requirement.

## Summary

| Requirement | Target | Result |
|---|---|---|
| `/ask`, `/health`, `/metrics` | working | ✅ plus `/ask/stream`, `/trace/{id}`, `/feedback` |
| p95 cached | ≤ 800 ms | ✅ 0.9 ms (exact hit, server); 745 ms (semantic hit) |
| p95 uncached | ≤ 6,000 ms | ✅ 5,755 ms replayed (golden set) · ❌ 16,028 ms live during a Gemini slowdown |
| Streaming TTFT | ≤ 1,500 ms | ❌ 3,834 ms p50 cold. The model reasons before emitting any text (B2) |
| Cost per query | ≤ $0.01 | ✅ $0.0023 |
| Regression gate fails on a breach | demonstrated | ✅ `final_k=1` → 3 gates fail, exit 1 (D3) |

Shipped pipeline ([`pipeline.py`](pipeline.py)): the Lab 3 retriever (markdown chunks @400, exact dense, archived docs excluded) feeds the Lab 4 generator (validate, then one repair retry, else refuse), with `k = final_k = 12`. That is not the Lab 5 change: Lab 5 measured `final_k=5` at −0.050 correctness, so the better-measured setting ships.

## Part A: the service

**A1.** `POST /ask` returns `{answer, refused, citations[{index, doc_id, excerpt}], sources, latency_ms, cost_usd, cached, trace_id}`, plus `cache_layer`, `matched_question` (so a semantic hit says which question it actually answered) and `guard_flags`. `cost_usd` is per request, from a per-thread meter around `aip.llm.raw_call`. `aip.cost.Budget` keeps one process-wide list, so two concurrent requests would each be billed for the other's calls.

**A2.** The pipeline is built once, in the FastAPI `lifespan` (index build: ~100 ms from cache). The Lab 6 guards sit on the request path:

| Path | Layers | What they do |
|---|---|---|
| `mode=rag` | 1, 2, 5 | Retrieved text delimited as data · retrieved chunks screened for injection (a flagged excerpt is withheld, not the whole answer) · output filter (prompt leak, unsupplied URLs/PII, repetition) |
| `mode=tools` | 1–5, **read-only** | The allowlist is `search_policy` + `compute_premium` only. `get_policy_details` is not exposed: Lab 6 N05 showed it returns *any* customer's record, and this service has no login to bind a policy number to. `issue_refund` needs human confirmation, and an HTTP request has none |

Live check: a refund request in tools mode returned `guard_flags: ["tool_denied:issue_refund"]`. No money moved.

**A3.** Tested with `curl`, each against a real fault rather than a mocked one:

| Case | How it was produced | Response |
|---|---|---|
| Malformed | `{"question":"hi","mode":"sql"}`; a non-JSON body | **422**, with both field errors listed |
| Provider outage | Service started with a dead `HTTPS_PROXY` → `APIConnectionError` | **503**, `Retry-After: 30`, `upstream: APIConnectionError` |
| Provider outage (offline) | `AIP_OFFLINE=1` + an uncached question → `CacheMiss` | **503**, `Retry-After: 30` |
| Budget exhausted | `AIP_BUDGET_USD=0.001` | **429** on the first and every later request |
| Provider hang (found live) | A `/ask` hung > 2 min while Gemini was timing out (4 retries × 60 s) | Now **503** at the 30 s `LAB7_DEADLINE_S` |

Outages are classified by **exception type name**, walking the cause chain, never by message text. `aip.llm`'s retry check matches "500" anywhere in a message, so a bug about "Rs 500" would have become a 503. No response body carries a stack trace; a genuine bug is a 500 with a trace id.

**A4.** [`ui.py`](ui.py): every citation is an expander showing the exact excerpt the model was given, alongside latency, time to first token, cost, cache layer and the trace link. A 👎 button appends the case to `reports/review_queue.jsonl` with its trace id (stretch 3). Smoke-tested with Streamlit `AppTest` in both streaming and blocking modes.

## Part B: caching, streaming, the latency budget

**B1. Two cache layers** ([`caching.py`](caching.py)), sitting in front of the pipeline:
- **Exact:** key = hash of (normalised question, `top_k`, `mode`, pipeline fingerprint). Change the model or config and you miss.
- **Semantic:** cosine similarity over question embeddings, plus an **entity guard**: two questions that name a different plan, product or number never share an answer.
- **Refusals and guard-modified answers are never cached.** A refusal caused by a broken index must not outlive the fix, and it must stay visible to the C4 alert.

**The threshold, measured** ([`semantic_sweep.py`](semantic_sweep.py), [`reports/lab7_semantic_sweep.json`](../../reports/lab7_semantic_sweep.json)). Setup: 40 new probe questions (21 paraphrases, 19 "twins" that change one slot) are looked up against the 45 cached golden answers. A hit is wrong when the Lab 4 correctness judge scores the served answer below 2 against the pipeline's own answer to the probe.

| Threshold | Hits / 40 | Wrong, no guard | Wrong, with guard |
|---|---|---|---|
| 0.85 | 26 | **10** | 4 |
| 0.90 | 17 | **4** | 1 |
| 0.93 | 13 | **2** | 0 |
| **0.94 (shipped)** | 11 | **0** | 0 |
| 0.95 (starter) | 8 | 0 | 0 |

**Wrong hits begin at cosine 0.931**: *"I had a claim last year on **Gold**…"* was served the **Silver** answer, which says the bonus drops 10 points; on Gold it does not. Others:
- Silver vs Gold caesarean, 0.931.
- 9-dioptre vs 6-dioptre LASIK, 0.902: served "excluded", and the truth is the opposite.

**Shipped at 0.94 with the guard.** That is the lowest threshold with zero wrong hits *even without* the guard, so the guard is defence in depth rather than the only thing holding it up. Compared with the reasoned 0.95, it gains 3 hits (28% vs 20% of probes). The sample is 40 probes, so the boundary is an estimate; the guard is the margin.

**B2. Streaming:** `POST /ask/stream`, server-sent events: `meta`, then `token`…, then `validation` (or `error`). Cold, n=12: **TTFT p50 3,834 ms, p95 5,592 ms. Total is only ~10–250 ms later.** `gemini-3.7-flash` spends most of its 429 completion tokens reasoning invisibly, so the first visible token arrives shortly before the last. Blocking `/ask` on the same kind of questions: p50 ~4.5 s. **Streaming saves almost nothing on this model.** The same prompts on the non-reasoning SMALL model gave TTFT p50 **834 ms** (n=10). See B4.

An outage before the first token still returns a real **503 / 429** status: the handler waits for the first event before committing to a 200. A stall mid-stream emits an `error` event.

**B3. Decision: stream, then send a validation event the UI acts on.** The UI shows text as "unverified" while it streams. The `validation` event then runs Lab 4's validator, `enforce_citations` and guard layer 5, and is either:
- `verified`, which shows the citation excerpts, or
- `replaced`, which swaps in the repaired or refused answer with a notice.

Why this one:
- Buffering throws away the only thing streaming offers.
- Holding back only the citations still leaves invalid prose on screen.
- This option keeps the latency win **and** guarantees no unvalidated answer stays visible.

The cost is a window in which the user may read text that is later replaced. It is small here: citation validity was 45/45 on the golden set.

**B4. The latency budget**, cold, from traces (45 golden-set requests; embedding from 27 live requests):

```
embed query      662 ms p50 / 1,943 p95   (semantic lookup embeds once; retrieval reuses it at 0 ms)
retrieve           2 ms p50 /   605 p95
rerank             —                       (the Lab 3 config ships without one)
screen (guard 2)   1 ms p50 /     1 p95
generate       2,803 ms p50 / 6,894 p95   <- 97% of p50
validate (+guard 5) 0.3 ms p50 / 0.6 p95
total          2,887 ms p50 / 6,899 p95
```

**Optimise generation first, by changing the model, not the prompt length.** The SMALL tier through the full gate ([`reports/lab7_gate_tier_small.json`](../../reports/lab7_gate_tier_small.json)): p95 **1,655 ms**, $0.0006/query (4× cheaper), refusal recall 5/5. But **correctness falls to 0.888 and the gate blocks it.** Its misses (Q04, Q29, Q32) are the scope-mixing failures that `EVALUATION_REPORT.md` §7 targets first. The guards and validation together cost under 2 ms.

## Part C: observability

**C1.** Every request is a span tree under `http.ask`. Its trace id `<run_id>.<span_id>` is in the response, and `GET /trace/{id}` ranks the stages by time.

*"Why did request `20261005-154222.8793da6fd147` take 17 s?"* From the trace alone: `stage.generate` took 15,980 ms (94%), all of it one `llm.call` that produced 335 tokens with **0 retries**. That is a slow provider (~21 tokens/s), not our code and not backoff.

Two gaps found and fixed:
- **Spans are written on exit**, so a hung request was invisible. An `http.start` event now marks it at entry.
- **"Uncached" latency excludes requests whose model calls all replayed from `aip.cache`.** They skip generation, and counting them would flatter the SLO.

**C2.** `GET /metrics` ([`traces.py`](traces.py), computed from today's trace files, so it survives restarts) reports:
- cost today, cost per query and per cold query
- response- and call-level cache hit rate, by layer
- p50/p95/p99 for all, cold, cached and stream-TTFT requests
- the per-stage breakdown
- error rate by HTTP status and by exception type
- tool calls executed and denied, by tool
- refusal rate and the alert

**C3.** [`dashboard.py`](dashboard.py) is built on the same `traces.py`, so the dashboard and `/metrics` cannot disagree. It shows:
- the KPI row
- the alert banner and runbook
- cached vs uncached latency
- p50/p95 per B4 stage and per span name
- **p95 per stage over time**, with a bucket slider
- cumulative cost
- cache-hit and error rate over time
- errors by type and tool calls
- a slowest-first request table that expands any request into its span tree

**C4. Alert: refusal rate doubling.** It fires when the refusal rate over the last 50 cold requests is at least 2× baseline and at least 10%. Baseline is the earlier traffic, or the golden-set rate (4/45 = 9%) until 30 requests exist. A broken index throws no errors, adds no latency and costs nothing extra; it just stops finding things, and the system declines. When it fires:
1. Check `/health` index size (231 chunks, 29 docs).
2. Open three refusal traces: are the right docs in `retrieved`?
3. Check guard-2 `n_flagged` for a withheld document.
4. If the index is fine, compare the questions with the golden set.
5. Roll back the last corpus change and rerun the gate.

**It fired during this lab** at 24% against 9%. The cause was our deliberately off-corpus benchmark questions, which the system correctly declined. That is its documented false positive, and step 4 is how you tell the two apart.

## Part D: the regression gate

**D1.** [`gate.py`](gate.py) runs the shipped pipeline over all 45 golden questions with Lab 4's judges and `refusal_stats`.

**Cost and latency are replayed**: each question is charged the cost and model latency recorded in the cache when the call was first made. Under `AIP_OFFLINE=1` every call is a cache hit, so wall time and spend would read ~0, and the cost and latency gates would pass whatever changed.

Thresholds sit about one standard error below what was measured; the rationale is in [`thresholds.yml`](thresholds.yml). Refusal recall is the weakest gate (n=5: one flip moves it 0.20).

| | correctness | faithfulness | citations | refusal R / P | hit@5 | cost | p95 |
|---|---|---|---|---|---|---|---|
| Gate | ≥ 0.91 | ≥ 0.87 | ≥ 0.98 | ≥ 0.60 / ≥ 0.75 | ≥ 0.95 | ≤ $0.003 | ≤ 6,000 |
| **Shipped** | 0.938 | 0.911 | 1.000 | 0.80 / 1.00 | 1.000 | $0.0023 | 5,755 |
| **D3: `final_k=1`** | **0.738** ✗ | 0.956 | 1.000 | 0.60 / **0.375** ✗ | 1.000 | $0.0014 | **8,322** ✗ |
| SMALL tier | **0.888** ✗ | 0.956 | 1.000 | 1.00 / 1.00 | 1.000 | $0.0006 | 1,655 |

**D2.** [`.github/workflows/eval.yml`](../../.github/workflows/eval.yml) runs with `AIP_OFFLINE=1` against the committed cache: no key, no cost, deterministic. The working cache was 147 MB, over GitHub's 100 MB per-file limit. [`ci_cache.py`](ci_cache.py) records every key the gate touches and exports just those rows, giving `.aip_cache/calls.sqlite3` at 7 MB. The rest of the cache stays local in `.aip_cache_full/`. Offline replay reproduced the online numbers exactly. **CI is green** on commit `6e8a3fa`: [run 37664711870](https://github.com/pragszz/aip-lab1/actions/runs/37664711870).

**D3. The break:** `LAB7_FINAL_K=1`. Offline it fails correctness, refusal precision and p95, **exit 1**. Two lessons:
- **Cost fell**, so a cost-only gate would have passed it.
- **The cost gate catches what quality gates cannot.** Swapping MAIN for LARGE would cost roughly 2.3× by list price (~$0.0053, an estimate, not a measured run) and fail the $0.003 ceiling. So will the scheduled 2027 price rise for `gemini-3.7-flash`, with no code change.

To reproduce the red build in CI, push a commit that changes `final_k: int = 12` to `1` in `PipelineConfig` (`pipeline.py`). The gate takes its defaults from `PipelineConfig`, so it measures what the service ships. The cache already holds that run's calls, so CI will fail on metrics, not on a cache miss.
