#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py --config labs/lab7/thresholds.yml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / "reports/lab7_gate.json"


# Under AIP_OFFLINE=1 every call is a cache hit, so wall-clock latency is ~0 and
# real spend is $0 -- a cost and latency gate on those would pass whatever you
# changed. The cache stores what each call originally cost and how long it took,
# so the gate charges each question its RECORDED cost and model latency
# (pipeline.metering). Swap MAIN for LARGE and the replayed cost rises: that is
# what the cost gate is for.


def _se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def measure() -> dict[str, float]:
    """D1: run the golden set through the SHIPPED pipeline and return the
    metric dict. Keys match thresholds.yml.

    Quality metrics reuse Lab 4's harness (judges, refusal_stats) so the gate
    and the earlier reports measure the same thing. Run under AIP_OFFLINE=1 and
    every call replays from the committed cache: no key, no cost, deterministic.
    """
    from aip.evals import retrieval_metrics
    from labs.lab3.search import load_questions
    from labs.lab4.evaluate import judge_correctness, judge_faithfulness, refusal_stats
    from labs.lab7.pipeline import Pipeline, PipelineConfig, install_meter, metering

    install_meter()
    # LAB7_FINAL_K exists for D3: the deliberate break is `LAB7_FINAL_K=1`.
    # LAB7_TIER evaluates a candidate generator (B4: SMALL vs MAIN).
    # Defaults come from PipelineConfig, so the gate measures what the service
    # ships: a regression committed to the config itself is caught too.
    cfg = PipelineConfig(final_k=int(os.getenv("LAB7_FINAL_K", PipelineConfig.final_k)),
                         tier=os.getenv("LAB7_TIER", PipelineConfig.tier))
    pipe = Pipeline(cfg)
    questions = load_questions(include_unanswerable=True)

    rows = []
    for q in questions:
        t0 = time.perf_counter()
        with metering() as m:
            r = pipe.answer(q["question"])
        wall_ms = (time.perf_counter() - t0) * 1000
        # local compute as measured, plus model time as originally recorded
        latency_ms = wall_ms - m["wall_ms"] + m["recorded_ms"]
        unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
        rows.append({
            "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
            "question": q["question"], "answer": r.answer, "refused": r.refused,
            "partial_decline": r.partial_decline, "citations_valid": r.citations_valid,
            "repaired": r.repaired, "guard_flags": r.guard_flags,
            "retrieved": r.retrieved, "sources": r.sources, "relevant": q["relevant_docs"],
            "hit_at_5": (retrieval_metrics(r.retrieved, q["relevant_docs"], ks=(5,))
                         ["hit_rate@5"] if q["relevant_docs"] else None),
            "llm_calls": m["calls"], "cost_usd": m["recorded_cost_usd"],
            "latency_ms": latency_ms, "model_ms": m["recorded_ms"],
            "faithfulness": judge_faithfulness(r.answer, r.context),
            "correctness": (None if unanswerable else
                            judge_correctness(q["question"], r.answer, q["gold_answer"])),
            "gold_answer": q["gold_answer"],
        })

    ans = [r for r in rows if not r["unanswerable"]]
    corr = [r["correctness"] / 2 for r in ans if r["correctness"] is not None]
    faith = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    hits = [r["hit_at_5"] for r in ans if r["hit_at_5"] is not None]
    lat = sorted(r["latency_ms"] for r in rows)
    rs = refusal_stats(rows)

    metrics = {
        "correctness": statistics.fmean(corr),
        "faithfulness": statistics.fmean(faith),
        "citation_validity": statistics.fmean(r["citations_valid"] for r in rows),
        "refusal_recall": rs["recall"],
        "refusal_precision": rs["precision"],
        "hit_rate_at_5": statistics.fmean(hits),
        "cost_per_query_usd": statistics.fmean(r["cost_usd"] for r in rows),
        "p95_latency_ms": lat[int(0.95 * (len(lat) - 1))],
    }
    detail = {
        "config": {"k": cfg.k, "final_k": cfg.final_k, "tier": cfg.tier,
                   "layers": sorted(cfg.layers), "fingerprint": cfg.fingerprint()},
        "offline": os.getenv("AIP_OFFLINE", "0"),
        "n": len(rows), "n_answerable": len(ans), "n_unanswerable": rs["n_unanswerable"],
        "standard_errors": {
            "correctness": statistics.pstdev(corr) / math.sqrt(len(corr)),
            "faithfulness": _se(metrics["faithfulness"], len(faith)),
            "citation_validity": _se(metrics["citation_validity"], len(rows)),
            "refusal_recall": _se(rs["recall"], rs["n_unanswerable"]),
            "refusal_precision": _se(rs["precision"], max(rs["n_declined"], 1)),
            "hit_rate_at_5": _se(metrics["hit_rate_at_5"], len(hits)),
        },
        "refusal": rs,
        "judge_parse_failures": sum(r["faithfulness"] is None for r in rows)
                                + sum(r["correctness"] is None for r in ans),
        "latency_p50_ms": statistics.median(lat),
        "model_ms_p95": sorted(r["model_ms"] for r in rows)[int(0.95 * (len(rows) - 1))],
        "metrics": metrics,
        "rows": rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"n={len(rows)} ({len(ans)} answerable)  report -> {REPORT.relative_to(ROOT)}\n")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    args = ap.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    metrics = measure()

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
