# Lab 2 Report: The Prompt Lab

Profile `gemini`. SMALL = `gemini-3.5-flash-lite` ($0.30/$2.50 per Mtok), MAIN =
`gemini-3.7-flash` ($0.75/$3.75). Grid run on `dev` (n=60), saved to
`reports/lab2_grid.json`. All numbers come from that file.

**One problem to know about first.** The free tier allows 15 calls/min on SMALL
and 5 calls/min on MAIN. Some variants were already cached and some were not, so
they hit that limit unequally: `zero_shot` 0% errors, `few_shot` 0%,
`few_shot_reasoned` 23.3%, `cascade` 40.0%, `zero_shot_main` **93.3%**.


## Part B — the grid

Cost is shown as $/ticket (tokens × price × calls per ticket). The raw `cost_usd`
column is misleading because it mostly reflects how much was cached.

| variant | record | field | err rate | compl tok/call | $/ticket | p95 ms |
|---|---|---|---|---|---|---|
| `zero_shot` | 0.5500 | 0.9104 | 0.000 | 110.5 | **0.000931** | (all cached) |
| `zero_shot_main` | 0.5000 | 0.8438 | 0.933 | 769.2 | 0.004549 | 5667 |
| `few_shot` | 0.5500 | 0.9250 | 0.000 | 91.7 | 0.001196 | 1454 |
| `few_shot_reasoned` | 0.5435 | 0.9212 | 0.233 | 229.7 | 0.001533 | 2422 |
| `cascade` | 0.5556 | 0.9306 | 0.400 | 119.6 | 0.002114 | 5798 |

### 1. Which knob mattered more — prompt or model?

**Neither improved accuracy. The model choice only changed cost and speed.**

The prompt knob (`zero_shot` → `few_shot`) raised field accuracy by 1.46 points and
record accuracy by exactly zero, at 1.28× the cost. The paired test on all 60
tickets gives b=6, c=6, p=1.0000. So even that 1.46 is noise.

The model knob did not improve accuracy either. On the 4 tickets MAIN finished, it
scored 0.8438 field against SMALL's 0.8750 on those same 4 (b=1, c=1, p=1.0000).
But it cost **4.9× more per call** ($0.004549 vs $0.000931) and took 5,667 ms at p95
against 1,454 ms for a real SMALL call.

We would generally expect the bigger model to be better and the prompt to be a
cheap tweak. Measured here, the model tier is a **cost lever that did not improve
quality**, and the prompt is cheap and also did nothing. Both do nothing for the
same reason: the labelling rules are already in the Pydantic field. 
The schema is doing the work, so there is nothing left for a bigger
model or for examples to add.

### 2. What did the reasoning field cost, and what did it buy?

`few_shot` → `few_shot_reasoned`. Same prompt. The only change is a `reasoning`
field added first in the schema.

| | few_shot | few_shot_reasoned | change |
|---|---|---|---|
| completion tokens/call | 91.7 | 229.7 | **+138 (2.51×)** |
| prompt tokens/call | 3157 | 3195 | +38 (basically unchanged) |
| $/ticket | 0.001196 | 0.001533 | **+28.2%** |
| field accuracy | 0.9250 | 0.9212 | **−0.38 points** |
| record accuracy | 0.5500 | 0.5435 | −0.65 points |

All the extra cost is on the output side. Prompt tokens barely moved. Completion
tokens went up 2.5×. 

**Accuracy points per rupee: −3.73 × 10⁻⁶.** The extra cost is ₹27.96 per 1,000
tickets, or **₹102,059 per year at 10,000 tickets/day** (at ₹83/$1). It bought
−0.38 field accuracy points. The return is negative.

The paired test on the 46 shared tickets gives b=6, c=7, p=1.0000, so the −0.38 is
noise too. Therefore, the reasoning field did not help and cost 28% more.

### 3. Any dominated configurations?

The harness says `zero_shot_main`, `few_shot`, `few_shot_reasoned`. But it compared
them against a `zero_shot` row showing $0.0000 and 0 ms, which is only true because
everything was cached. 

- `zero_shot_main` — worse record accuracy (0.5000 vs 0.5500), 4.9× cost, 5,667 ms.
  **Dominated on all three.**
- `few_shot` — same record accuracy (0.5500), 1.28× cost. **Dominated** (ties on
  quality, loses on cost).
- `few_shot_reasoned` — worse record accuracy, 1.65× cost, slower. **Dominated**,
  by `few_shot` too.
- `cascade` — only avoids the list because it has the highest raw record accuracy
  (0.5556). On the 36 shared tickets it gives **the same answer as `zero_shot` on
  every single one** (Part D) at 2.27× the cost. So it is dominated as well.

**All four alternatives are dominated by the plain zero-shot SMALL baseline.**

## Part C — the cascade

Take a second SMALL sample at temperature 0.7, and escalate to MAIN if the two disagree on
`category/urgency/sentiment/product/language`.

| | value |
|---|---|
| **escalation rate** | **5.6%** (2 of 36 tickets) — not zero, so the cache trap was avoided |
| **blended cost** | **$0.002114/ticket** = 2×SMALL + 0.056×MAIN |
| vs pure SMALL | **2.27× more expensive** ($0.000931) |
| vs pure MAIN | 0.46×, less than half ($0.004549) |
| **blended field accuracy** | 0.9306 (pure SMALL 0.9104, pure MAIN 0.8438) |
| **blended record accuracy** | 0.5556 (pure SMALL 0.5500, pure MAIN 0.5000) |

That blended accuracy looks like the best in the grid. It is not. On the 36 tickets
`cascade` and `zero_shot` both answered, they gave the same result on **every
ticket** (b=0, c=0, field accuracy 0.9306 both ways). The cascade escalated twice,
and both times MAIN returned the same outcome SMALL already had. It paid 2.27× for
the same result

### Does the trigger carry any signal?

| | agreed | disagreed | agreement rate |
|---|---|---|---|
| SMALL was **right** (20) | 19 | 1 | **95.0%** |
| SMALL was **wrong** (16) | 15 | 1 | **93.8%** |

Fisher's exact test on this table: **p = 1.00**.

The trigger fired on 6.2% of SMALL's errors (1 of 16) and on 5.0% of its correct
answers. It flags right answers about as often as wrong ones, so no relevant conclusion
can be drawn from it.

The reference solution found the same thing (94% vs 83%, catching 2 of 12). 
Sampling twice detects **variance**, but this model's problem is **bias**. It is not
unsure about the `complaint`/`billing` boundary. It is confidently wrong there every
time, so both samples agree and both are wrong.

## Part D — is any difference real?

### D1 · Confidence intervals (Wilson, 95%, record accuracy)

| variant | n | correct | record acc | 95% CI | width |
|---|---|---|---|---|---|
| `zero_shot` | 60 | 33 | 0.5500 | [0.425, 0.669] | 0.244 |
| `few_shot` | 60 | 33 | 0.5500 | [0.425, 0.669] | 0.244 |
| `few_shot_reasoned` | 46 | 25 | 0.5435 | [0.402, 0.678] | 0.277 |
| `cascade` | 36 | 20 | 0.5556 | [0.396, 0.705] | 0.309 |
| `zero_shot_main` | 4 | 2 | 0.5000 | [0.150, 0.850] | 0.700 |

Every interval is about 0.25 wide. The gap between best and worst is only 0.05.

### D2 · Paired tests (McNemar, exact binomial)

**(i) As run.** A failed ticket counts as wrong. Introduced the rate-limit problem.

| comparison | b | c | p | verdict |
|---|---|---|---|---|
| `zero_shot` vs `zero_shot_main` | 32 | 1 | 0.0000 | A better |
| `zero_shot` vs `few_shot` | 6 | 6 | 1.0000 | no difference |
| `zero_shot` vs `few_shot_reasoned` | 15 | 7 | 0.1338 | no difference |
| `zero_shot` vs `cascade` | 13 | 0 | 0.0002 | A better |

**(ii) Shared tickets only.** Only tickets both variants answered. This removes the
rate-limit problem and measures quality alone:

| comparison | n | b | c | p | verdict |
|---|---|---|---|---|---|
| `zero_shot` vs `zero_shot_main` | 4 | 1 | 1 | 1.0000 | no difference |
| `zero_shot` vs `few_shot` | 60 | 6 | 6 | 1.0000 | no difference |
| `zero_shot` vs `few_shot_reasoned` | 46 | 6 | 7 | 1.0000 | no difference |
| `zero_shot` vs `cascade` | 36 | 0 | 0 | 1.0000 | **same on every ticket** |

Both "significant" results in (i) disappear in (ii). `zero_shot_main` goes from
p=0.0000 to p=1.0000. `cascade` goes from p=0.0002 to b=0, c=0. Both were measuring
MAIN's 5 calls/min limit, not how good MAIN is. Reporting (i) on its own would have 
resulted in a false conclusion with a p-value attached to it.

### D3 · Conclusion

The comparison that matters is `zero_shot` vs `cascade`
**b = 0, c = 0, discordant = 0, p = 1.0000.**
Not one of the 36 shared tickets tells them apart.

On the shared-ticket tests, **every p-value is 1.0000**. Nothing in this grid beats
plain zero-shot on SMALL. So the choice comes down to cost, and the baseline wins by
1.28× (few-shot), 1.65× (reasoning), 2.27× (cascade) and 4.9× (MAIN).

## Negative results

1. **Few-shot did not help.** +1.46 field points, 0.00 record, p=1.0000, at 1.28×
   cost. The prompt already carries the labelling rules in its field descriptions,
   so six examples had nothing left to teach.
2. **The reasoning field did not help** and cost +28%, about ₹102k/year at scale.
   Accuracy per rupee is negative.
