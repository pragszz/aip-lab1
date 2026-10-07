# Aurora Policy Assistant — Evaluation Report

*Lab 7 · measured 5 October 2026 · generator `gemini-3.7-flash`, judge `gemini-3.5-flash` · all figures reproducible offline from the committed cache (`AIP_OFFLINE=1 python labs/lab7/gate.py`)*

## 1 · What it does

The assistant answers customers' questions about Aurora's health insurance plans, using only Aurora's own 29 current policy documents. Every statement in an answer points to the passage it came from, and the customer can open that passage to check it. When the documents don't answer the question, the assistant says so instead of guessing. On our test set of 45 questions it answered 35 of 40 answerable questions fully correctly and 5 partly correctly, with none wholly wrong. It declined 4 of the 5 questions it should not answer, and its fifth reply partly declined. A typical answer takes about 3 seconds and costs about a quarter of a US cent. Repeat questions come back in under a hundredth of a second and cost nothing. The assistant explains what the published plan terms say; it does not decide whether a particular claim will be paid (see section 6).

## 2 · How well it works

45-question golden set (40 answerable, 5 that must be refused). The pipeline is the Lab 3 retriever (markdown chunks of 400 characters, exact dense search, archived documents excluded) feeding the Lab 4 generator, which validates each answer and retries once. It keeps 12 sources, not 5: Lab 5 measured that cutting to 5 lost 0.050 correctness. Lab 6 guard layers 1, 2 and 5 sit on the request path. Source: `reports/lab7_gate.json`.

| Metric | Value | 95% CI / SE | Gate |
|---|---|---|---|
| Correctness (judge, 0–2 rescaled) | **0.938** | SE 0.026 | ≥ 0.91 |
| Faithfulness (every claim backed by context) | 0.911 (41/45) | SE 0.042 | ≥ 0.87 |
| Citation validity (code-checked) | **1.000** (45/45) | CI 0.92–1.00 | ≥ 0.98 |
| Refusal recall, full refusals | 0.80 (4/5) | CI 0.38–0.96 | ≥ 0.60 |
| Refusal recall incl. partial declines | 1.00 (5/5) | — | — |
| Refusal precision | 1.00 (4/4) | — | ≥ 0.75 |
| Retrieval hit rate@5 (doc level) | 1.000 (40/40) | — | ≥ 0.95 |
| Cost per query (cold) | $0.0023 | — | ≤ $0.0030 |
| p95 latency (cold, replayed) | 5,756 ms | — | ≤ 6,000 ms |

By kind (correctness): paraphrase 1.00 (n=5) · multi-hop 0.95 (10) · single-hop 0.94 (18) · aggregation 0.88 (4) · archived-document trap 0.83 (3). Judge parse failures: 0. The judge was calibrated in Lab 4: correctness κ = 0.66. Faithfulness κ was uninformative (0.0 at 95% raw agreement, the kappa paradox).

**The gate, proven to fail (D3).** Setting `final_k=1` replays offline to correctness 0.738, refusal precision 0.375 and p95 8,322 ms: **3 gates fail, exit 1**. Cost *fell* to $0.0014, so a gate that watched only cost would have passed it. A real candidate also fails: switching generation to the cheaper SMALL model gives correctness 0.888, below the 0.91 floor (section 7).

## 3 · Where it fails (golden set, counts)

| Failure mode | Count | Example |
|---|---|---|
| **Scope mixing**: pulls in a product line the customer didn't ask about | 3/40 (Q04, Q29, Q32) | Q29 *"how long to respond to a query on my claim"*: answers 45 days, then adds the **motor** policy's 30 days. Q04 adds Senior Care's 24 months to the retail 36 |
| **Uncited derived numbers**: arithmetic the sources don't state | 2/45 (Q25, Q27) | Q27 asserts *"a total maximum discount of up to 45%"*, adding 30% + 15% itself. Q25's ₹60,000 is right but computed, not quoted |
| **Incomplete**: omits a fact the question needed | 2/40 (Q11, Q23) | Q23 says physiotherapy is "not mentioned" instead of "OPD is not covered on any base plan"; Q11 omits Platinum's air ambulance |
| **Partial instead of full refusal** | 1/5 (Q40) | Says the helpline number isn't stated, then points to "your policy schedule". Harmless, but not a refusal |
| Judge error (faithfulness 0, answer correct) | 1/45 (Q20) | The same judge error Lab 4 found by hand |

**Outside the golden set**, from live tests on the service:
- **Tools mode makes a false follow-up promise.** On a refund request the guard correctly denied `issue_refund`, and no money moved. The model then told the customer *"a representative will review… and follow up"*, and no tool does that (1/1 refund attempts; Lab 6 N03 is still open). **This is our worst remaining failure:** it is a commitment the company never made, given to a customer with a money problem.
- **The semantic cache gives wrong answers below cosine 0.94.** *"No-claim bonus on **Gold**"* was served the **Silver** answer at 0.931; a 9-dioptre LASIK question got the 6-dioptre answer ("excluded"; the truth is the opposite) at 0.902. Shipped at 0.94 plus an entity guard (section 5).
- **Poisoned content passes every guard.** Lab 6's N01 (an edited page saying "claims window now 7 days") is quoted with a real citation. No layer can catch true-looking text.

## 4 · What it costs

| | Cold (no cache) | At 30% response-cache hits |
|---|---|---|
| Per query | $0.0023 | $0.0016 |
| Per 1,000 queries | $2.32 | $1.62 |
| Per year at 10,000/day | **$8,473** | **$5,930** |

Generation is essentially all of it. The query embedding (~$0.000002) and our guards add nothing measurable. `mode=tools` costs $0.0036–$0.0044 per query (n=2). **Price risk:** `gemini-3.7-flash` doubles to $1.50/$7.50 per million tokens on 1 January 2027 (`aip/config.py`). That takes cold cost to ~$0.0046, about $17,000/year, and **the cost gate will go red that day with no code change.** That is the gate working, and it is a decision three months out.

## 5 · How fast it is

| Path | p50 | p95 | n |
|---|---|---|---|
| Cold, golden set (replayed, deterministic) | 2,906 ms | **5,756 ms** | 45 |
| Cold, live service today (Gemini degraded, timeouts seen) | 4,509 ms | 16,028 ms | 30 |
| Response-cache hit (server time; 20 exact + 1 semantic) | 0.4 ms | 0.9 ms | 21 |
| …of which the semantic hit (must embed the question) | 745 ms | — | 1 |
| Streaming, time to first token (cold) | 3,834 ms | 5,592 ms | 12 |

**Stage breakdown, cold** (traces of the 45-question run; embedding from 27 live requests):

```
embed query      662 ms p50 / 1,943 p95   (live; retrieval then reuses it at 0 ms)
retrieve           2 ms p50 /   605 p95
rerank             —  (none: Lab 3 config ships without a reranker)
screen (guard 2)   1 ms p50 /     1 p95
generate       2,803 ms p50 / 6,894 p95   <- 97% of p50
validate (+guard 5) 0.3 ms p50 /  0.6 p95
total          2,887 ms p50 / 6,899 p95
```

Targets: cached ≤ 800 ms ✅ · uncached p95 ≤ 6,000 ms ✅ replayed, ❌ live during today's provider slowdown · **TTFT ≤ 1,500 ms ❌**. Streaming barely helps this model: `gemini-3.7-flash` reasons silently first (429 completion tokens per answer against roughly 50–150 visible), so the first token arrives only ~100 ms before the last. **The slow traffic is diagnosed from traces alone.** The 17.0 s request `20261005-154222.8793da6fd147` spent 94% in one model call that produced 335 tokens with zero retries: a slow provider, not our code. This is why the service now has a 30 s deadline that returns 503 + `Retry-After`; before it, one request hung for more than 2 minutes.

**Optimise first: generation.** Swapping the model is the lever, not streaming: the SMALL model streams its first token at 834 ms p50 (n=10) and finishes the whole gate at p95 1,655 ms (section 7).

**Streaming and validation (B3).** We stream the text, then send a validation event. The UI marks text "unverified" until the event arrives, then either confirms it (and shows the citation excerpts) or **replaces** it with the checked answer. Buffering would throw away the only latency win streaming has. Holding back only the citations would still leave invalid prose on screen. The price is a short window in which a reader may see text that is later replaced. On this system that is rare: citation validity was 45/45.

## 6 · What it is not safe for

**It may run unsupervised for:** looking up what the published plan terms say. That means limits, waiting periods, exclusions, how to file or escalate, and where a customer can read the exact passage and the cost of being wrong is a follow-up question. This is the ground the golden set covers well: single-hop 0.94, paraphrase 1.00, citations 45/45.

**It must not be relied on, without a human, for:**
1. **Any coverage or claim decision ("will this be paid?").** In 3 of 40 answers it blended in terms from a product the customer doesn't hold (motor, Senior Care, corporate). It cannot tell which plan the customer has: there is no login, and we deliberately did not expose the policy-lookup tool, because Lab 6 showed it returns *any* customer's record. A blended answer cites real sources, so it looks right.
2. **Money figures it had to compute.** There is no calculator on the answer path. 2 of 45 answers stated numbers the sources do not contain (*"up to 45%"*). Premiums come only from `mode=tools`, which uses a deterministic premium tool, and even those are illustrative, not quotes.
3. **Refunds or anything that promises action.** Tools mode cannot pay refunds, by design, but it will tell the customer someone will follow up when nobody will (section 3).
4. **Trusting the documents themselves.** Anyone who can edit the corpus can change the answers, and every guard will pass it with a citation (Lab 6 N01).
5. **Judging how well it refuses.** Refusal recall rests on 5 questions. The 95% interval for 4/5 is 0.38–0.96, and the gate can only notice the loss of two refusals, not one.

## 7 · What we would do next (ranked by expected value)

1. **Scope answers to the customer's product, and forbid uncited arithmetic.** Tag chunks with a product line and filter on it, and add a prompt rule that every number must appear verbatim in a cited source. *Expected:* fixes the most common failure modes (3 scope-mixing + 2 derived numbers = 5 cases), worth up to +0.04 correctness (Q04, Q29, Q32) and +0.07 faithfulness (Q25, Q27, Q29), at no added cost or latency. Under a day of work. It also removes the two biggest blockers in section 6.
2. **Re-test the SMALL model after (1), and ship it if it passes the gate.** Today SMALL is 4× cheaper ($0.0006 vs $0.0023, about $6,400/year saved at 10k/day), 3.5× faster (p95 1,655 vs 5,756 ms, and it **meets the TTFT target**: 834 ms), and refuses 5/5. But it loses 0.050 correctness, so the gate blocks it. Its misses (Q04, Q29, Q32) are the same scope-mixing cases (1) targets. *Expected:* if (1) recovers them, it fixes every latency target and halves the 2027 price-rise risk.
3. **Grow the unanswerable set from 5 to 30 or more, starting from the 👎 review queue** (`reports/review_queue.jsonl`), plus an action-promise check in tools mode. *Expected:* turns refusal recall from a ±0.3 guess into a gate that can catch a single lost refusal. That is the metric the refusal-rate alert relies on to spot a broken index.

---
*Artefacts: `reports/lab7_gate.json` (shipped), `lab7_gate_break_final_k1.json` (D3), `lab7_gate_tier_small.json`, `lab7_semantic_sweep.json`. Streaming, live-latency and SMALL-vs-MAIN TTFT figures come from the live session's traces (`.aip_traces/`, local, not committed): unlike the gate numbers they cannot be replayed offline. Lab write-up for Parts A–D: [`labs/lab7/report.md`](labs/lab7/report.md). Alert: refusal rate ≥ 2× baseline over the last 50 cold requests (`labs/lab7/traces.py`). It fired today on our off-corpus benchmark questions, which is its known false-positive mode; runbook step 4 covers it.*
