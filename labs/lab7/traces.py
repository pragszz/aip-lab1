#!/usr/bin/env python3
"""Lab 7 Part C — reading aip.tracing's JSONL back.

One module, used by /metrics, /trace/{id} and the dashboard, so the three can
never disagree about what a number means.

A request is the subtree under one `http.ask` or `http.ask_stream` span. Its
trace id is "<run_id>.<span_id>": the run id names the file, the span id names
the root, and the parent ids give the tree. That is everything needed to answer
"why did request X take 9 seconds?" without a log search.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from aip.config import settings

REQUEST_SPANS = ("http.ask", "http.ask_stream")

# B4's stage names, mapped to the spans that measure them. A query embedding is
# an embed.batch with n=1 under a request that actually called the API (the
# semantic-cache lookup embeds first; retrieval then reuses it from aip.cache
# at 0 ms, which is not a second embedding). The corpus build is n=231 and is
# not on the request path.
STAGES = [
    ("embed query", lambda s: s["name"] == "embed.batch" and s.get("n") == 1
     and s.get("n_uncached") == 1),
    ("retrieve", lambda s: s["name"] == "stage.retrieve"),
    ("screen (guard L2)", lambda s: s["name"] == "stage.screen"),
    ("rerank", lambda s: s["name"].startswith("rerank.")),
    ("generate", lambda s: s["name"] == "stage.generate"),
    ("validate (+guard L5)", lambda s: s["name"] == "stage.validate"),
]


def _files(since: float | None = None, run_id: str | None = None) -> list[Path]:
    if run_id:
        p = settings.trace_dir / f"{run_id}.jsonl"
        return [p] if p.exists() else []
    files = sorted(settings.trace_dir.glob("*.jsonl"))
    if since is not None:
        files = [p for p in files if p.stat().st_mtime >= since]
    return files


def load(since: float | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
    """Every span written since `since` (epoch seconds), oldest first."""
    rows: list[dict[str, Any]] = []
    for p in _files(since, run_id):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:   # a line being written right now
                        continue
    if since is not None:
        rows = [r for r in rows if r.get("ts", 0) >= since]
    return sorted(rows, key=lambda r: r.get("ts", 0))


def start_of_today() -> float:
    t = time.localtime()
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def attach_requests(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag every span with the trace id of the request it belongs to (or None)."""
    by_id = {(s["run_id"], s["span_id"]): s for s in spans}
    for s in spans:
        node, hops = s, 0
        while node is not None and node["name"] not in REQUEST_SPANS and hops < 50:
            parent = node.get("parent_id")
            node = by_id.get((s["run_id"], parent)) if parent else None
            hops += 1
        s["_request"] = (f"{node['run_id']}.{node['span_id']}"
                         if node is not None and node["name"] in REQUEST_SPANS else None)
    return spans


def is_cold(r: dict[str, Any]) -> bool:
    """A cold request paid for at least one model call. A response-cache hit
    is not cold, and neither is a request whose every model call replayed from
    aip.cache: both skip generation, so counting them would flatter the SLO."""
    return not r.get("cached") and not r.get("model_cached")


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 1)


def latency_summary(xs: list[float]) -> dict[str, float]:
    return {"n": len(xs), "p50_ms": pct(xs, 50), "p95_ms": pct(xs, 95),
            "p99_ms": pct(xs, 99)}


def stage_breakdown(spans: list[dict[str, Any]], uncached_only: bool = True
                    ) -> dict[str, dict[str, float]]:
    """p50/p95 per B4 stage, over requests (default: cold requests only)."""
    roots = {f"{s['run_id']}.{s['span_id']}": s for s in spans
             if s["name"] in REQUEST_SPANS and s.get("status") == "ok"}
    keep = {tid for tid, r in roots.items()
            if not (uncached_only and not is_cold(r))}
    per_stage: dict[str, list[float]] = defaultdict(list)
    for s in spans:
        if s.get("_request") not in keep:
            continue
        for stage, match in STAGES:
            if match(s):
                per_stage[stage].append(float(s.get("duration_ms", 0)))
    out = {stage: {"n": len(per_stage[stage]), "p50_ms": pct(per_stage[stage], 50),
                   "p95_ms": pct(per_stage[stage], 95)} for stage, _ in STAGES}
    totals = [float(roots[t]["duration_ms"]) for t in keep]
    out["total"] = {"n": len(totals), "p50_ms": pct(totals, 50), "p95_ms": pct(totals, 95)}
    return out


def by_span_name(spans: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for s in spans:
        if s.get("duration_ms", 0) > 0:
            groups[s["name"]].append(float(s["duration_ms"]))
    return {k: {"n": len(v), "p50_ms": pct(v, 50), "p95_ms": pct(v, 95),
                "total_ms": round(sum(v), 1)}
            for k, v in sorted(groups.items(), key=lambda kv: -sum(kv[1]))}


# ---------------------------------------------------------------------------
# C4 -- the alert. Refusal rate doubling.
# ---------------------------------------------------------------------------
ALERT_WINDOW = 50          # most recent requests
ALERT_MIN_BASELINE = 30    # need this many earlier requests to have a baseline
ALERT_FLOOR = 0.10         # golden-set refusal rate is 4/45 = 0.09; never alert below 10%


def refusal_alert(roots: list[dict[str, Any]]) -> dict[str, Any]:
    """Fire when the refusal rate over the last ALERT_WINDOW answered requests
    is at least twice the rate over everything before them (and >= 10%).

    Cached responses are excluded: refusals are never cached, so including
    cache hits would dilute the recent rate and hide the very failure this
    watches for. Without enough history, fall back to the golden-set rate
    measured by the gate (reports/lab7_gate.json).
    """
    fresh = [r for r in roots if r.get("status") == "ok" and not r.get("cached")]
    recent, before = fresh[-ALERT_WINDOW:], fresh[:-ALERT_WINDOW]
    def rate(rs: list[dict[str, Any]]) -> float:
        return sum(bool(r.get("refused")) for r in rs) / len(rs) if rs else 0.0

    baseline, source = rate(before), "traffic"
    if len(before) < ALERT_MIN_BASELINE:
        baseline, source = _golden_refusal_rate(), "golden set"
    current = rate(recent)
    firing = len(recent) >= 10 and current >= max(2 * baseline, ALERT_FLOOR)
    return {
        "name": "refusal_rate_doubling",
        "firing": firing,
        "recent_refusal_rate": round(current, 3), "recent_n": len(recent),
        "baseline_refusal_rate": round(baseline, 3), "baseline_source": source,
        "condition": f"refusal rate over last {ALERT_WINDOW} uncached requests >= "
                     f"2x baseline and >= {ALERT_FLOOR:.0%}",
        "runbook": [
            "1. GET /health: is n_chunks still 231 and n_docs 29? A drop means the "
            "corpus or index build broke.",
            "2. Open /trace/<id> for three recent refusals: are the right docs in "
            "`retrieved`? If not, retrieval is broken, not generation.",
            "3. Check stage.screen n_flagged: a poisoned or reformatted doc can be "
            "withheld by guard layer 2 and starve every answer that needs it.",
            "4. If the index is fine, diff recent questions against the golden set: "
            "a genuinely new topic is a corpus gap, not an incident.",
            "5. Rebuild the index / roll back the last corpus change; rerun gate.py.",
        ],
    }


def _golden_refusal_rate() -> float:
    p = settings.cache_dir.parent / "reports/lab7_gate.json"
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        return sum(r["refused"] for r in rows) / len(rows)
    except Exception:  # noqa: BLE001 -- no gate run yet
        return 0.09


# ---------------------------------------------------------------------------
# C2 -- /metrics
# ---------------------------------------------------------------------------
def summarise(spans: list[dict[str, Any]]) -> dict[str, Any]:
    attach_requests(spans)
    roots = [s for s in spans if s["name"] in REQUEST_SPANS]
    ok = [r for r in roots if r.get("status") == "ok"]
    cached = [r for r in ok if r.get("cached")]
    uncached = [r for r in ok if is_cold(r)]
    warm = [r for r in ok if not r.get("cached") and r.get("model_cached")]
    errors = [r for r in roots if r.get("status") == "error"]
    cost = sum(float(r.get("cost_usd") or 0) for r in roots)
    llm = [s for s in spans if s["name"] == "llm.call"]
    tools = Counter(s.get("tool") for s in spans if s["name"] == "tool.call")
    denied = Counter(s.get("tool") for s in spans if s["name"] == "tool.denied")
    streams = [r for r in ok if r["name"] == "http.ask_stream" and r.get("ttft_ms")
               and is_cold(r)]
    return {
        "requests": len(roots),
        "cost_today_usd": round(cost, 6),
        "cost_per_query_usd": round(cost / len(roots), 6) if roots else 0.0,
        "cost_per_uncached_query_usd": (round(cost / len(uncached), 6) if uncached
                                        else 0.0),
        "cache": {
            "response_hit_rate": round(len(cached) / len(ok), 3) if ok else 0.0,
            "by_layer": dict(Counter(r.get("cache_layer") for r in cached)),
            "llm_call_hit_rate": (round(sum(bool(s.get("cached")) for s in llm)
                                        / len(llm), 3) if llm else 0.0),
        },
        "latency": {
            "all": latency_summary([r["duration_ms"] for r in ok]),
            "uncached": latency_summary([r["duration_ms"] for r in uncached]),
            "cached": latency_summary([r["duration_ms"] for r in cached]),
            "model_cache_only": latency_summary([r["duration_ms"] for r in warm]),
            "stream_ttft": latency_summary([r["ttft_ms"] for r in streams]),
        },
        "stages_uncached": stage_breakdown(spans),
        "refusal_rate": (round(sum(bool(r.get("refused")) for r in ok) / len(ok), 3)
                         if ok else 0.0),
        "errors": {
            "rate": round(len(errors) / len(roots), 3) if roots else 0.0,
            "by_status": dict(Counter(str(r.get("http_status", "?")) for r in errors)),
            "by_type": dict(Counter(r.get("error_type", "?") for r in errors)),
        },
        "tool_calls": {"executed": dict(tools), "denied": dict(denied),
                       "total": sum(tools.values())},
        "by_mode": dict(Counter(r.get("mode", "rag") for r in roots)),
        "alerts": [refusal_alert(roots)],
    }


# ---------------------------------------------------------------------------
# C1 -- one request, explained
# ---------------------------------------------------------------------------
def explain(trace_id: str) -> dict[str, Any] | None:
    run_id, _, span_id = trace_id.partition(".")
    spans = load(run_id=run_id)
    by_id = {s["span_id"]: s for s in spans}
    root = by_id.get(span_id)
    if root is None:
        return None
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in spans:
        if s.get("parent_id"):
            children[s["parent_id"]].append(s)

    def node(s: dict[str, Any], depth: int) -> list[dict[str, Any]]:
        attrs = {k: v for k, v in s.items()
                 if k not in ("run_id", "span_id", "parent_id", "name", "ts",
                              "duration_ms", "status") and not k.startswith("_")}
        out = [{"depth": depth, "name": s["name"], "duration_ms": s.get("duration_ms"),
                "start_offset_ms": round((s["ts"] - root["ts"]) * 1000, 1),
                "status": s.get("status"), **attrs}]
        for c in sorted(children.get(s["span_id"], []), key=lambda c: c["ts"]):
            out += node(c, depth + 1)
        return out

    tree = node(root, 0)
    total = float(root.get("duration_ms") or 0)
    top = sorted((t for t in tree if t["depth"] == 1),
                 key=lambda t: -(t["duration_ms"] or 0))
    llm = [t for t in tree if t["name"] == "llm.call" and not t.get("cached")]
    return {
        "trace_id": trace_id, "total_ms": total, "status": root.get("status"),
        "why": [f"{t['name']}: {t['duration_ms']:.0f} ms "
                f"({100 * (t['duration_ms'] or 0) / total:.0f}%)" for t in top[:4]]
               if total else [],
        "model_calls": len(llm),
        "completion_tokens": sum(t.get("completion_tokens") or 0 for t in llm),
        "retries": sum(1 for t in tree if t["name"] == "llm.retry"),
        "spans": tree,
    }


if __name__ == "__main__":
    s = load(since=start_of_today())
    print(json.dumps(summarise(s), indent=2))
