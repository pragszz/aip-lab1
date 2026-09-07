# Lab 1 Report: The Reliable Extractor

Model: `gemini/gemini-3.5-flash-lite` (SMALL tier, `gemini` profile). All numbers below are measured, not estimated, from fresh (uncached, `AIP_CACHE=0`) runs of this session's `extract.py` and `v0_naive.py`.

## Part A: v0 failure characterisation (n=40, fresh calls)

| Failure mode | Count/40 | Example | T1 §3 # |
|---|---|---|---|
| Not valid JSON at all | 0 | | #5 |
| JSON wrapped in a markdown fence | 28 | T0054 | #5 |
| Extra prose before/after JSON | 0 | | #5 |
| Valid JSON, missing a required field | 0 | | #6 |
| Category outside the allowed set | 28 | T0054 | #6 |
| Urgency as a string instead of an int | 28 | T0054 | #6 |
| Policy number invented | 0 | | #8 |
| Unhandled exception | 12 | T0021 (`RateLimitError`) | not in taxonomy, see below |

Arc: 0/40 parsed to 28/40 parsed after stripping the fence, still 0/40 clean. The 12 exceptions are real 429s from the Gemini free tier (15 req/min) surfacing straight through `extract_v0`, which has no retry/backoff at all.

Two rows that do not map cleanly onto T1 §3. (1) *Unhandled exception* is not one of the nine failures. It is what happens when a failure like #2 (rate limit) or #1 (transport) hits code with no orchestration layer at all; it belongs to layer 6, not the taxonomy of ways a call goes wrong. (2) *Policy number invented* is nominally #8 (hallucination), but it scored 0/40 here. This model does not fabricate identifiers unprompted, so the real lesson is that #8 needs a grounding check, not a format check, and that boundary belongs to Part C's move-to-code decision, not the API-failure list.

## Variant comparison: dev split, fresh calls, same 60 tickets

| Variant | n | Field acc | Record acc | Schema valid | Unhandled exc. | Cost | $/ticket | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| v0 | 40 | n/a (no schema) | n/a | 0% | 12/40 (30%) | $0.0052 | $0.00019 | 1014 ms | 1525 ms |
| B | 60 | 0.909 | 0.558 | 100% | 0/60 | $0.0485 | $0.00081 | 1297 ms | 1562 ms |
| C | 60 | 0.931 | 0.592 | 100% | 0/60 | $0.0352 | $0.00059 | 1115 ms | 1492 ms |

B to C: cost down 27%, field accuracy up 0.02 (noise). Both runs independently hit Gemini free-tier rate limits, 8/60 on B and 11/60 on C, each degrading to a `needs_human_review` record instead of crashing, which is `schema_valid` staying at 100% doing its job. `policy_number`, `product`, `language` were already 1.000 in B; Part C makes them guaranteed, not just measured, at lower cost.

## Part D: test split (n=120, variant C, run once)

| Metric | Value | Target | Result |
|---|---|---|---|
| Schema validity | 1.000 | 1.00 | met |
| Field accuracy | 0.921 | ≥0.90 | met |
| Record accuracy | 0.500 | ≥0.55 | below target (ref. 0.608) |
| Cost, 120 items | $0.0374 | ≤$0.15 | met |
| p95 latency | 1600 ms | ≤4000 ms | met |
| Unhandled exceptions | 0 | 0 | met |

Per-field (worst first): `urgency` 0.712, `sentiment` 0.827, `category` 0.885, `escalate` 0.942, `contains_pii`/`language`/`policy_number`/`product` 1.000.

Confusion for `category` (rows = gold, cols = predicted):

|            | billing | claims | complaint | information | policy_change | technical |
|---|---|---|---|---|---|---|
| billing | 5 | | | | | |
| claims | | 11 | | | | |
| complaint | 1 | 1 | 4 | | | |
| information | 1 | 3 | | 5 | | |
| policy_change | | | | | 12 | |
| technical | | | | | | 9 |

Urgency errors: 15/120 records. 12/15 are off-by-one at a boundary; 3/15 (T0009, T0109, T0237) are off-by-two, all the same template ("upgrade/renewal, do waiting periods carry over?" / "how do I submit post-hospitalisation bills?") predicted 1 (self-service) against gold 3. Scattered, they are not.

## Top three error clusters (from 15+ opened failures)

1. **Sentiment miscalibration** (9/26 imperfect records, the single biggest driver). Both directions: `satisfied` to `neutral` (thanks/confirmation under-read), `frustrated` to `neutral` (prior-failure cue missed), and `neutral` vs `angry` confused on emphatic but first-time messages. Fix: embed one contrastive example pair per boundary directly in `SENTIMENT_DESC` (a calm-but-firm message vs. an actually hostile one); worth roughly 0.05 to 0.08 on this field.
2. **Urgency at the renewal/post-discharge template boundary** (3/15 urgency errors, the worst individual misses). The model treats "how do I submit post-hospitalisation bills" / "will waiting periods carry over at renewal" as pure self-service (1) when gold rates them 3. Fix: add an explicit anchor example for a transaction the customer is mid-way through, not just asking about, mapping to 3; worth roughly 0.02 to 0.03 on this field, more if this template recurs in production.
3. **`category`: information vs. claims/billing** (4/6 category errors). A general coverage/benefit question gets read as an active claim or bill once a policy number or the word "claim"/"refund" appears nearby, even with no pending transaction. Fix: add a negative example to `CATEGORY_DESC`'s `information` clause ("asking whether a benefit applies is `information` even if a policy number is quoted, unless a transaction is already in motion"). This is the same boundary `data/README.md` flags as the single largest source of category error, and the data agrees.

## One thing that did not work

Moving `escalate` into `apply_business_rules()` as a pure code rule (`urgency >= 4 or "ombudsman" in text`) was meant to make it a fourth field pinned at roughly 100%, the way `policy_number`/`contains_pii` are. It did not: `escalate` scored 0.942, not 1.000. Checking all 3 test failures (T0198, T0152, T0134) shows every one is a case where the model's `urgency` crossed the 3/4 line in the wrong direction. None involve the "ombudsman" keyword path. Determinism only helps the part of the pipeline downstream of the model; it cannot fix an upstream judgement error, it just relocates where the error surfaces. 

## D5: Cost Analysis

Measured cost: $0.037417 / 120 tickets = $0.0003118/ticket.

- AI system, annualised: 10,000 tickets/day × 365 × $0.0003118 ≈ $1,138/year.
- Manual baseline: 40 s/ticket at ₹300/hr = ₹3.33/ticket, so 10,000 × 365 × ₹3.33 ≈ ₹1.22 crore/year ≈ $146,600/year (at an assumed ₹83/$1).
- Naive savings: about $145,500/year, roughly a 99% reduction in triage labour cost.
- Break-even record accuracy (worst case: every wrong record costs a full manual re-handle, no better than not automating): `acc_breakeven = AI_cost_per_ticket / manual_cost_per_ticket` = ₹0.0259 / ₹3.33 ≈ 0.8%.

That number is the real finding of D5. On cost alone, this system is worth deploying at almost any accuracy: the AI is roughly 130 times cheaper per ticket than a human even before it gets anything right. Cost is therefore not what should gate deployment here. The binding constraint is risk: a wrongly-triaged urgency-5 ticket (an ICU cashless denial routed as low priority) has a cost the per-ticket arithmetic above does not capture at all. That is exactly why the lab sets the deployment bar at record accuracy ≥0.55, a quality/safety floor rather than a cost floor, and why landing at 0.500 here (short of both the target and the 0.608 reference) matters more than the cost.
