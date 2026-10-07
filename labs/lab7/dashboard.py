#!/usr/bin/env python3
"""Lab 7 — the observability dashboard, read from local traces.

    streamlit run labs/lab7/dashboard.py

`aip.tracing` writes one JSONL file per run to .aip_traces/. This page reads
them back. It is a teaching-scale stand-in for Langfuse / LangSmith / Phoenix;
the concept -- structured spans with a run id and a parent id -- is identical.

Every number here comes from labs/lab7/traces.py, the same code behind
GET /metrics, so the dashboard and the endpoint cannot disagree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402
from labs.lab7 import traces  # noqa: E402

st.set_page_config(page_title="Aurora Assistant — Ops", layout="wide")
st.title("Aurora Policy Assistant — operations")

runs = sorted(settings.trace_dir.glob("*.jsonl"), reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

# Default to the runs that contain service traffic (http.* spans).
service_runs = [p.stem for p in runs[:60]
                if any(f'"name": "{n}"' in p.read_text(encoding="utf-8")
                       for n in traces.REQUEST_SPANS)]
chosen = st.sidebar.multiselect("runs", [p.stem for p in runs],
                                default=service_runs[:5] or [runs[0].stem])
spans = [s for rid in chosen for s in traces.load(run_id=rid)]
if not spans:
    st.stop()
traces.attach_requests(spans)
summary = traces.summarise(spans)

df = pd.DataFrame(spans)
df["ts"] = pd.to_datetime(df["ts"], unit="s")
req = df[df["name"].isin(traces.REQUEST_SPANS)].copy()

# ---------------------------------------------------------------------------
c = st.columns(6)
c[0].metric("requests", summary["requests"])
c[1].metric("cost (selected runs)", f"${summary['cost_today_usd']:.4f}")
c[2].metric("cost / query", f"${summary['cost_per_query_usd']:.5f}")
c[3].metric("response cache hit rate", f"{summary['cache']['response_hit_rate']:.0%}")
c[4].metric("error rate", f"{summary['errors']['rate']:.0%}")
c[5].metric("refusal rate", f"{summary['refusal_rate']:.0%}")

# ---------------------------------------------------------------------------
# C4 -- the alert
# ---------------------------------------------------------------------------
alert = summary["alerts"][0]
st.subheader("Alert: refusal rate doubling")
msg = (f"recent {alert['recent_refusal_rate']:.0%} over {alert['recent_n']} uncached "
       f"requests vs baseline {alert['baseline_refusal_rate']:.0%} "
       f"({alert['baseline_source']}).  Condition: {alert['condition']}.")
(st.error if alert["firing"] else st.success)(
    ("FIRING — " if alert["firing"] else "OK — ") + msg)
with st.expander("What to do when it fires (runbook)"):
    st.markdown(
        "A broken or stale index throws no errors, adds no latency and costs no "
        "more. It quietly stops finding things, and a correctly built RAG system "
        "responds by **declining**. So the refusal rate is where a silent data "
        "failure becomes visible.\n\n" + "\n".join(f"- {s}" for s in alert["runbook"]))

# ---------------------------------------------------------------------------
st.subheader("Latency (end to end)")
lat = summary["latency"]
st.dataframe(pd.DataFrame(lat).T.rename(columns=str), width="stretch")
st.caption("Cached and uncached are reported separately: they are different systems, "
           "and a blended p95 lets a high hit rate hide a slow cold path.")

# ---------------------------------------------------------------------------
# C3 -- p50/p95 per span name, and per B4 stage
# ---------------------------------------------------------------------------
st.subheader("Latency by stage — uncached requests (B4)")
st.dataframe(pd.DataFrame(summary["stages_uncached"]).T, width="stretch")

st.subheader("Latency by span name")
st.dataframe(pd.DataFrame(traces.by_span_name(spans)).T, width="stretch")

st.subheader("Latency by stage over time")
stage_rows = []
for s in spans:
    if not s.get("_request"):
        continue
    for stage, match in traces.STAGES:
        if match(s):
            stage_rows.append({"ts": pd.to_datetime(s["ts"], unit="s"), "stage": stage,
                               "ms": s.get("duration_ms", 0)})
if stage_rows:
    sdf = pd.DataFrame(stage_rows)
    bucket = st.select_slider("bucket", ["1min", "5min", "15min", "1h"], value="5min")
    p95 = (sdf.set_index("ts").groupby("stage")["ms"].resample(bucket).quantile(0.95)
           .unstack(0))
    st.line_chart(p95)
    st.caption(f"p95 per stage per {bucket}")

# ---------------------------------------------------------------------------
st.subheader("Cost over time (cumulative)")
if len(req) and "cost_usd" in req:
    cum = req.sort_values("ts").assign(cum=lambda d: d["cost_usd"].fillna(0).cumsum())
    st.line_chart(cum.set_index("ts")["cum"])

st.subheader("Cache hit rate and error rate over time")
if len(req):
    r = req.set_index("ts").sort_index()
    r["cached_f"] = r.get("cached", pd.Series(False, index=r.index)).fillna(False).astype(float)
    r["error_f"] = (r["status"] == "error").astype(float)
    st.line_chart(r[["cached_f", "error_f"]].resample("5min").mean()
                  .rename(columns={"cached_f": "cache hit rate", "error_f": "error rate"}))

st.subheader("Errors by type")
e = summary["errors"]
col1, col2 = st.columns(2)
col1.write({"by HTTP status": e["by_status"]})
col2.write({"by exception type": e["by_type"]})
errs = df[df["status"] == "error"]
if len(errs):
    st.dataframe(errs[["ts", "name", "error"]], width="stretch")

st.subheader("Tool calls")
st.write(summary["tool_calls"])

# ---------------------------------------------------------------------------
# C1 -- why did request X take 9 seconds?
# ---------------------------------------------------------------------------
st.subheader("Requests — slowest first")
if len(req):
    cols = [c for c in ("ts", "trace_id", "duration_ms", "cached", "cache_layer",
                        "refused", "cost_usd", "http_status", "question") if c in req]
    table = req.sort_values("duration_ms", ascending=False)[cols]
    st.dataframe(table, width="stretch")
    tid = st.selectbox("explain a request", table["trace_id"].dropna().tolist())
    if tid:
        t = traces.explain(tid)
        if t:
            st.markdown(f"**{t['total_ms']:.0f} ms total** — "
                        + "; ".join(t["why"])
                        + f" — {t['model_calls']} uncached model calls, "
                          f"{t['completion_tokens']} completion tokens, "
                          f"{t['retries']} retries")
            tree = pd.DataFrame(t["spans"])
            tree["span"] = tree.apply(lambda r: "  " * r["depth"] + r["name"], axis=1)
            show = [c for c in ("span", "duration_ms", "start_offset_ms", "status",
                                "cached", "completion_tokens", "cost_usd") if c in tree]
            st.dataframe(tree[show], width="stretch")
            with st.expander("raw"):
                st.code(json.dumps(t, indent=2, default=str)[:20000])
