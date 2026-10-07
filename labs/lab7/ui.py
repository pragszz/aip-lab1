#!/usr/bin/env python3
"""Lab 7 — Streamlit front end.

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

The one non-negotiable UI requirement: **citations must be expandable to show
the source text.** Grounding the user cannot check is decoration.

Streaming (default) acts on the B3 validation event: text arrives marked
"unverified"; when the service's validation event arrives it is either marked
verified (and the citation expanders appear) or REPLACED with the validated
answer, with a notice saying so.
"""
from __future__ import annotations

import json
import time

import requests
import streamlit as st

API = st.sidebar.text_input("Service URL", "http://localhost:8000")
stream = st.sidebar.toggle("Stream the answer", value=True)
mode = st.sidebar.radio("Mode", ["rag", "tools"], horizontal=True,
                        help="tools: Lab 6 agent, read-only tools (search + premium)")

st.title("Aurora Policy Assistant")
st.caption("Answers come only from Aurora's policy documents. "
           "Every claim is cited. When the documents do not cover a question, "
           "the assistant says so instead of guessing.")

q = st.text_input("Ask a question",
                  placeholder="How long do I have to file a reimbursement claim?")


def show_error(status: int, body: str, headers: dict | None = None) -> None:
    if status == 503:
        st.error(f"The model provider is unavailable. Try again in "
                 f"{(headers or {}).get('Retry-After', 'a few')} seconds. (503)")
    elif status == 429:
        st.error("The service has reached its spending limit. (429)")
    elif status == 422:
        st.error(f"That request was not valid: {body[:300]} (422)")
    else:
        st.error(f"{status}: {body[:300]}")


def ask_blocking(question: str) -> dict | None:
    try:
        r = requests.post(f"{API}/ask", json={"question": question, "mode": mode},
                          timeout=90)
    except requests.RequestException as exc:
        st.error(f"service unreachable: {exc}")
        return None
    if r.status_code != 200:
        show_error(r.status_code, r.text, dict(r.headers))
        return None
    return r.json()


def ask_streaming(question: str) -> dict | None:
    """Read the SSE stream; render tokens as they arrive; act on `validation`."""
    box = st.empty()
    note = st.empty()
    text, final, t0, ttft = "", None, time.perf_counter(), None
    try:
        with requests.post(f"{API}/ask/stream", json={"question": question},
                           stream=True, timeout=90) as r:
            if r.status_code != 200:
                show_error(r.status_code, r.text, dict(r.headers))
                return None
            event = None
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("event:"):
                    event = raw.split(":", 1)[1].strip()
                    continue
                if not raw.startswith("data:"):
                    continue
                data = json.loads(raw.split(":", 1)[1])
                if event == "token":
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    text += data["text"]
                    box.markdown(text + " ▌")
                    note.caption("streaming — citations not yet verified")
                elif event == "validation":
                    final = data
                elif event == "error":
                    show_error(int(data.get("status", 500)), data.get("message", ""),
                               {"Retry-After": data.get("retry_after")})
                    return None
    except requests.RequestException as exc:
        st.error(f"service unreachable: {exc}")
        return None
    if final is None:
        st.error("The stream ended without a validation event; the answer above "
                 "is unverified and should not be relied on.")
        return None
    box.empty()
    note.empty()
    if final["status"] == "replaced":
        st.info("The streamed draft failed validation (citations or a safety "
                "check) and was replaced with the checked answer below.")
    final["client_ttft_ms"] = ttft
    return final


if st.button("Ask", type="primary") and q:
    if stream and mode == "rag":
        data = ask_streaming(q)
    else:
        with st.spinner("thinking"):
            data = ask_blocking(q)
    if data:
        st.session_state["last"] = {"q": q, **data}

data = st.session_state.get("last")
if data:
    if data.get("refused"):
        st.warning(data["answer"])
    else:
        st.markdown(data["answer"])
        if data.get("status") == "verified" or "status" not in data:
            st.caption("✓ every citation checked against the sources shown to the model")

    if data.get("cache_layer") == "semantic" and data.get("matched_question"):
        st.info(f"Served from the semantic cache: this is the answer to "
                f"“{data['matched_question']}”.")

    # A4: every citation opens to the exact excerpt the model was given.
    for c in data.get("citations", []):
        with st.expander(f"[{c['index']}] {c['doc_id']}"):
            st.text(c["excerpt"])
    if data.get("sources") and not data.get("citations") and not data.get("refused"):
        st.caption("sources consulted: " + ", ".join(data["sources"]))

    cols = st.columns(5)
    cols[0].metric("latency", f"{data.get('latency_ms', 0):.0f} ms")
    if data.get("client_ttft_ms"):
        cols[1].metric("first token", f"{data['client_ttft_ms']:.0f} ms")
    cols[2].metric("cost", f"${data.get('cost_usd', 0):.5f}")
    cols[3].metric("cached", data.get("cache_layer") or "no")
    cols[4].metric("citations", len(data.get("citations", [])))
    st.caption(f"trace: `{data.get('trace_id', '')}` — "
               f"{API}/trace/{data.get('trace_id', '')}")

    # Stretch 3: the cheapest real feedback loop there is.
    with st.form("feedback"):
        why = st.text_input("What was wrong? (optional)")
        if st.form_submit_button("👎 Flag this answer for review"):
            try:
                requests.post(f"{API}/feedback", timeout=10, json={
                    "trace_id": data.get("trace_id", ""), "question": data["q"],
                    "answer": data["answer"], "note": why})
                st.success("Added to the review queue (reports/review_queue.jsonl).")
            except requests.RequestException as exc:
                st.error(f"could not record feedback: {exc}")
