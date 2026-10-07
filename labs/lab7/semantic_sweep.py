#!/usr/bin/env python3
"""Lab 7 B1 — find the cosine at which the semantic cache starts lying.

    python labs/lab7/semantic_sweep.py            # needs reports/lab7_gate.json

Setup. The cache holds the 45 golden questions with the answers the shipped
pipeline gave them (from the gate run). Each probe below is a NEW question:

    para   a paraphrase of one golden question  -> the cached answer SHOULD fit
    twin   same template, one slot changed      -> the cached answer should NOT

A probe looks up its nearest cached question by cosine. If that cosine clears
the threshold, the cache serves the cached answer. Whether that answer is RIGHT
is not assumed from the label: the Lab 4 correctness judge scores the served
answer against what the pipeline answers for the probe itself. Score 2 is a
safe hit; anything less is a wrong hit -- an answer to a question nobody asked.

Run with and without the entity guard (caching.entities), sweeping 0.80-0.99.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.cost import Budget  # noqa: E402
from aip.embed import embed_batch  # noqa: E402
from labs.lab4.evaluate import judge_correctness  # noqa: E402
from labs.lab7.caching import entities  # noqa: E402
from labs.lab7.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "reports/lab7_semantic_sweep.json"

# (base golden id, kind, probe)
PROBES: list[tuple[str, str, str]] = [
    ("Q01", "para", "After I'm discharged, how long do I get to file my reimbursement claim?"),
    ("Q01", "twin", "How many days does Aurora take to settle a reimbursement claim after I submit it?"),
    ("Q02", "para", "How much room rent does the Silver plan allow?"),
    ("Q02", "twin", "What is the room rent limit on the Bronze plan?"),
    ("Q03", "para", "Does Bronze cover maternity?"),
    ("Q03", "twin", "Is maternity covered on the Gold plan?"),
    ("Q05", "para", "What's the grace period if I pay my premium annually?"),
    ("Q05", "twin", "How long is the grace period for an instalment policy?"),
    ("Q07", "para", "How much no-claim bonus does the Gold plan give?"),
    ("Q07", "twin", "What is the no-claim bonus on Silver?"),
    ("Q08", "para", "Is IVF covered by Aurora?"),
    ("Q08", "twin", "Can I claim for knee replacement treatment?"),
    ("Q09", "para", "How much does Gold pay for cataract surgery?"),
    ("Q09", "twin", "What is the cataract sub-limit on Silver?"),
    ("Q11", "para", "What is the limit for ambulance charges?"),
    ("Q11", "twin", "Is air ambulance covered on Bronze?"),
    ("Q12", "para", "Does the policy pay for psychiatric treatment?"),
    ("Q12", "twin", "Is cosmetic surgery covered?"),
    ("Q13", "para", "How many days do I have to return the policy if I change my mind?"),
    ("Q13", "twin", "What is the grace period?"),
    ("Q14", "para", "How many points does the annual health check-up earn in the wellness programme?"),
    ("Q14", "twin", "How many wellness points do I get for completing the quit-tobacco programme?"),
    ("Q16", "para", "For an emergency cashless admission, how quickly must the hospital inform Aurora?"),
    ("Q16", "twin", "Within how many hours must a planned cashless admission be notified to Aurora?"),
    ("Q24", "para", "What happens to my Silver no-claim bonus if I made a claim last year?"),
    ("Q24", "twin", "I had a claim last year on Gold. What happens to my accumulated no-claim bonus?"),
    ("Q25", "para", "On Gold, if a caesarean bill is 1,60,000, how much do I pay myself?"),
    ("Q25", "twin", "How much would a caesarean cost me out of pocket on Silver if the hospital bills 1,60,000?"),
    ("Q26", "para", "If I buy the policy today, can I claim for a knee replacement right away?"),
    ("Q26", "twin", "Is cataract surgery covered immediately after I buy the policy?"),
    ("Q29", "para", "If Aurora asks me a question about my claim, how long do I have to reply?"),
    ("Q29", "twin", "How many days does Aurora have to respond to my grievance?"),
    ("Q30", "para", "How early should I inform Aurora about a planned cashless hospitalisation?"),
    ("Q30", "twin", "How soon after an emergency admission must cashless be notified?"),
    ("Q32", "para", "On which plans do I not have to pay any co-pay?"),
    ("Q32", "twin", "Which plans have no room-rent limit?"),
    ("Q45", "para", "Will Aurora pay for LASIK to correct 6 dioptres?"),
    ("Q45", "twin", "Is a 9 dioptre lasik covered?"),
    ("Q17", "para", "Is there a third party administrator handling Aurora claims?"),
    ("Q33", "para", "What are the pre and post hospitalisation periods for each plan?"),
]

THRESHOLDS = [round(0.80 + 0.01 * i, 2) for i in range(20)]


def main() -> None:
    gate = json.loads((ROOT / "reports/lab7_gate.json").read_text(encoding="utf-8"))
    cached = [{"id": r["id"], "question": r["question"], "answer": r["answer"],
               "refused": r["refused"]} for r in gate["rows"]]
    # Refusals are never cached by the service, so they cannot be served.
    cached = [c for c in cached if not c["refused"]]
    cvec = embed_batch([c["question"] for c in cached], input_type="query")
    pvec = embed_batch([p[2] for p in PROBES], input_type="query")
    sims = pvec @ cvec.T

    pipe = Pipeline()
    judged: dict[tuple[int, int], int | None] = {}
    probes = []
    with Budget(limit_usd=1.50, label="lab7-semantic-sweep") as budget:
        for i, (base, kind, q) in enumerate(PROBES):
            own = pipe.answer(q)
            # every cached question above the lowest threshold is a potential hit
            cands = [j for j in np.argsort(-sims[i]) if sims[i, j] >= THRESHOLDS[0]]
            for j in cands[:3]:
                judged[(i, j)] = judge_correctness(q, cached[j]["answer"], own.answer)
            probes.append({"base": base, "kind": kind, "question": q,
                           "own_answer": own.answer, "own_refused": own.refused,
                           "entities": sorted(entities(q)),
                           "candidates": [{"cached_id": cached[j]["id"],
                                           "cached_question": cached[j]["question"],
                                           "cosine": round(float(sims[i, j]), 4),
                                           "same_entities": entities(q)
                                           == entities(cached[j]["question"]),
                                           "judge": judged[(i, j)]}
                                          for j in cands[:3]]})
    print(budget.report())

    def served(p: dict, t: float, guard: bool) -> dict | None:
        for c in p["candidates"]:            # sorted by cosine, descending
            if c["cosine"] < t:
                return None
            if guard and not c["same_entities"]:
                continue
            return c
        return None

    sweep = []
    for guard in (False, True):
        for t in THRESHOLDS:
            hits = [(p, served(p, t, guard)) for p in probes]
            hits = [(p, c) for p, c in hits if c is not None]
            wrong = [(p, c) for p, c in hits if c["judge"] is not None and c["judge"] < 2]
            sweep.append({
                "entity_guard": guard, "threshold": t,
                "hit_rate": len(hits) / len(probes), "hits": len(hits),
                "wrong_hits": len(wrong),
                "wrong_hit_rate": len(wrong) / len(probes),
                "hit_precision": (1 - len(wrong) / len(hits)) if hits else 1.0,
                "para_hits": sum(p["kind"] == "para" for p, _ in hits),
                "twin_hits": sum(p["kind"] == "twin" for p, _ in hits),
                "wrong_examples": [{"asked": p["question"], "served_for":
                                    c["cached_question"], "cosine": c["cosine"]}
                                   for p, c in wrong][:6],
            })

    def safe_floor(guard: bool) -> float | None:
        """Lowest threshold at which no wrong hit occurs at it or anywhere above."""
        rows = [s for s in sweep if s["entity_guard"] == guard]
        ok = None
        for s in sorted(rows, key=lambda s: -s["threshold"]):
            if s["wrong_hits"]:
                break
            ok = s["threshold"]
        return ok

    worst = {g: max((c["cosine"] for p in probes for c in p["candidates"]
                     if c["judge"] is not None and c["judge"] < 2
                     and (not g or c["same_entities"])), default=None)
             for g in (False, True)}
    summary = {"n_probes": len(probes), "n_para": sum(p["kind"] == "para" for p in probes),
               "n_twin": sum(p["kind"] == "twin" for p in probes),
               "highest_wrong_cosine": {"no_guard": worst[False], "guard": worst[True]},
               "safe_threshold": {"no_guard": safe_floor(False), "guard": safe_floor(True)}}
    OUT.write_text(json.dumps({"summary": summary, "sweep": sweep, "probes": probes},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'guard':<6}{'thr':>6}{'hit%':>7}{'hits':>6}{'wrong':>7}{'para':>6}"
          f"{'twin':>6}{'precision':>11}")
    for s in sweep:
        print(f"{'yes' if s['entity_guard'] else 'no':<6}{s['threshold']:>6.2f}"
              f"{100 * s['hit_rate']:>6.0f}%{s['hits']:>6}{s['wrong_hits']:>7}"
              f"{s['para_hits']:>6}{s['twin_hits']:>6}{s['hit_precision']:>11.2f}")
    print(f"\n{json.dumps(summary, indent=2)}\nsaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
