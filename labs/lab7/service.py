#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq

Endpoints
    POST /ask            answer + citations + sources + latency + cost + trace id
    POST /ask/stream     the same, as server-sent events (B2/B3)
    GET  /health         index size, models, cache stats, uptime
    GET  /metrics        cost, cache hit rate, p50/p95/p99, errors by type, tools
    GET  /trace/{id}     one request's span tree: "why did request X take 9 s?"
    POST /feedback       thumbs-down -> reports/review_queue.jsonl (stretch 3)

Environment
    LAB7_SEMANTIC_THRESHOLD  cosine for the semantic cache; 0 disables it.
                             Default from the measured sweep (see caching.py).
    LAB7_RETRY_AFTER_S       Retry-After on a 503 (default 30)
    LAB7_DEADLINE_S          per-request deadline; past it, 503 (default 30)
    AIP_BUDGET_USD           process spend ceiling; past it every call is a 429
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, cost, tracing  # noqa: E402
from aip.config import resolve_model, settings  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402
from aip.guards import redact_pii  # noqa: E402
from labs.lab7 import traces  # noqa: E402
from labs.lab7.caching import ResponseCache  # noqa: E402
from labs.lab7.pipeline import (  # noqa: E402
    Pipeline,
    PipelineConfig,
    install_meter,
    metering,
    validate_answer,
)

# Measured, not reasoned (labs/lab7/semantic_sweep.py, 40 probes): wrong hits
# reach cosine 0.931 without the entity guard ("no-claim bonus on Gold" served
# the Silver answer) and 0.921 with it. 0.94 is the lowest threshold with zero
# wrong hits EVEN WITHOUT the guard, so the guard is defence in depth, not the
# only thing holding it up. See reports/lab7_semantic_sweep.json.
SEMANTIC_THRESHOLD = float(os.getenv("LAB7_SEMANTIC_THRESHOLD", "0.94"))
RETRY_AFTER_S = int(os.getenv("LAB7_RETRY_AFTER_S", "30"))
DEADLINE_S = float(os.getenv("LAB7_DEADLINE_S", "30"))
REVIEW_QUEUE = ROOT / "reports/review_queue.jsonl"

_PIPELINE: Pipeline | None = None
_CACHE: ResponseCache | None = None
_STARTED = time.time()


def pipeline() -> Pipeline:
    """A2: built ONCE, at startup, and cached. Building it per request re-reads
    and re-hashes the corpus on every call; the guards are wired inside it."""
    global _PIPELINE, _CACHE
    if _PIPELINE is None:
        install_meter()
        _PIPELINE = Pipeline(PipelineConfig())
        _CACHE = ResponseCache(_PIPELINE.config.fingerprint(),
                               semantic_threshold=SEMANTIC_THRESHOLD)
    return _PIPELINE


def response_cache() -> ResponseCache:
    pipeline()
    assert _CACHE is not None
    return _CACHE


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pipeline()
    yield


app = FastAPI(title="Aurora Policy Assistant", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    # 12, not 5: Lab 5 measured final_k=5 at correctness 0.875 vs 0.925 for 12.
    top_k: int = Field(default=12, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    sources: list[str]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str
    cache_layer: str | None = None          # "exact" | "semantic" when cached
    matched_question: str | None = None     # semantic hits: what was really answered
    guard_flags: list[str] = []


# ---------------------------------------------------------------------------
# A3 -- error semantics
# ---------------------------------------------------------------------------
class UpstreamUnavailable(RuntimeError):
    """The model provider is down, throttling us, or unreachable."""


_OUTAGE_MARKERS = ("ratelimit", "timeout", "serviceunavailable", "apiconnection",
                   "internalserver", "overloaded", "badgateway", "connecterror",
                   "connectionerror", "cachemiss", "upstreamunavailable",
                   "deadlineexceeded")


def is_outage(exc: BaseException | None) -> bool:
    """Walk the cause chain looking for a provider-side failure, by type name.

    By NAME, not by message text: aip.llm's retry check matches "500" anywhere
    in the message, which would turn our own bug about "Rs 500" into a 503.
    CacheMiss counts: under AIP_OFFLINE=1 the cache IS the provider.
    """
    seen = 0
    while exc is not None and seen < 10:
        name = type(exc).__name__.lower()
        if any(m in name for m in _OUTAGE_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return False


def http_error(exc: BaseException, trace_id: str) -> HTTPException:
    if isinstance(exc, BudgetExceeded):
        return HTTPException(status_code=429, detail={
            "error": "budget_exhausted", "message": "Spend limit reached. Back off.",
            "trace_id": trace_id})
    if is_outage(exc):
        return HTTPException(
            status_code=503, headers={"Retry-After": str(RETRY_AFTER_S)},
            detail={"error": "upstream_unavailable",
                    "message": "The model provider is unavailable. Retry later.",
                    "upstream": type(exc).__name__, "trace_id": trace_id})
    # A genuine bug. The trace has the stack; the client gets an id, not internals.
    return HTTPException(status_code=500, detail={
        "error": "internal_error", "message": "Unexpected error. Quote the trace id.",
        "trace_id": trace_id})


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
    """Last line: nothing leaves as a bare 500 with a stack trace."""
    return JSONResponse(status_code=500, content={"detail": {
        "error": "internal_error", "message": "Unexpected error.",
        "type": type(exc).__name__}})


def _check_budget() -> None:
    """Refuse up front once the process ceiling is gone, instead of paying for
    a retrieval and then failing on the model call."""
    b = global_budget()
    if b.spent_usd >= b.limit_usd:
        raise BudgetExceeded(f"process budget ${b.limit_usd:.2f} exhausted "
                             f"(spent ${b.spent_usd:.4f})")


def _trace_id(span: dict[str, Any]) -> str:
    return f"{tracing.RUN_ID}.{span['span_id']}"


def _cacheable(payload: dict[str, Any]) -> bool:
    """Refusals and guard-modified answers are never cached. A refusal caused
    by a broken index must not outlive the fix, and must stay visible to the
    refusal-rate alert."""
    return not payload["refused"] and not payload["guard_flags"]


# ---------------------------------------------------------------------------
# A1 -- POST /ask
# ---------------------------------------------------------------------------
class DeadlineExceeded(TimeoutError):
    """The request ran past LAB7_DEADLINE_S, almost always inside a model call."""


def with_deadline(fn, *args, **kwargs):
    """Run fn on a worker thread and stop waiting at DEADLINE_S.

    aip.llm retries a timed-out call 4 times at AIP_TIMEOUT_S=60 each: a hung
    provider holds a request for up to four minutes (seen live: a /ask hung
    >120 s while Gemini was timing out). Lab 6 met the same bug in its agent
    loop. The worker inherits the caller's span stack and meter, so the trace
    tree and the per-request cost survive the thread hop. An abandoned call
    may still finish and be billed in the background; that is the price of
    giving the user an answer (a 503 + Retry-After) in bounded time.
    """
    from labs.lab7 import pipeline as pl

    parent_stack = list(tracing._span_stack())  # noqa: SLF001
    meter = getattr(pl._METER, "current", None)  # noqa: SLF001
    box: dict[str, Any] = {}

    def target():
        tracing._LOCAL.stack = list(parent_stack)  # noqa: SLF001
        pl._METER.current = meter  # noqa: SLF001
        try:
            box["out"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 -- re-raised by the caller
            box["exc"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(DEADLINE_S)
    if t.is_alive():
        raise DeadlineExceeded(f"no answer within {DEADLINE_S:.0f}s")
    if "exc" in box:
        raise box["exc"]
    return box["out"]


def _run_rag(req: AskRequest) -> dict[str, Any]:
    r = with_deadline(pipeline().answer, req.question, final_k=req.top_k)
    return {"answer": r.answer, "refused": r.refused, "citations": r.citations,
            "sources": r.sources, "guard_flags": r.guard_flags}


def _run_tools(req: AskRequest) -> tuple[dict[str, Any], float]:
    r = pipeline().answer_with_tools(req.question, max_seconds=DEADLINE_S)
    if r["stopped_because"] == "budget":
        raise BudgetExceeded("per-request tool budget exhausted")
    if r["stopped_because"] == "error":
        # run_agent reports instead of raising; recover the type from its text.
        if any(m in r["answer"].lower().replace(" ", "") for m in _OUTAGE_MARKERS):
            raise UpstreamUnavailable(r["answer"])
        raise RuntimeError(r["answer"])
    from labs.lab6.agent import REFUSAL as TOOL_REFUSAL
    from labs.lab7.pipeline import REFUSAL as RAG_REFUSAL

    answer = r["answer"] or RAG_REFUSAL
    refused = (answer.strip() in (TOOL_REFUSAL, RAG_REFUSAL)
               or r["stopped_because"] == "blocked_output")
    return ({"answer": answer, "refused": refused, "citations": [],
             "sources": r["retrieved_docs"],
             "guard_flags": r["output_flags"] + [
                 f"tool_denied:{t['tool']}" for t in r["tool_log"] if not t.get("ok")]},
            float(r["cost_usd"]))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """A1. cost_usd and trace_id are in the body: the trace id is how anyone
    finds this request's spans later (GET /trace/{id}); the cost is how the
    person paying sees what a question costs without opening a dashboard."""
    t0 = time.perf_counter()
    rc = response_cache()
    safe_q, pii = redact_pii(req.question)
    with tracing.trace("http.ask", question=safe_q[:120], mode=req.mode,
                       top_k=req.top_k, pii_redacted=sum(pii.values())) as span:
        trace_id = span["trace_id"] = _trace_id(span)
        # Spans are written on EXIT, so a hung request is invisible until it
        # ends. This start event makes an in-flight request findable.
        tracing.event("http.start", trace_id=trace_id)
        try:
            hit = rc.get(req.question, top_k=req.top_k, mode=req.mode)
            if hit:
                payload, cost_usd = hit.payload, 0.0
            else:
                _check_budget()
                with metering() as m:
                    if req.mode == "tools":
                        payload, cost_usd = _run_tools(req)
                    else:
                        payload = _run_rag(req)
                        cost_usd = m["cost_usd"]
                if _cacheable(payload):
                    rc.put(req.question, payload, top_k=req.top_k, mode=req.mode)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            span.update(cached=bool(hit), cache_layer=hit.layer if hit else None,
                        similarity=round(hit.similarity, 4) if hit else None,
                        refused=payload["refused"], n_citations=len(payload["citations"]),
                        guard_flags=payload["guard_flags"], cost_usd=round(cost_usd, 6),
                        model_cached=bool(not hit and m["calls"]
                                          and m["cached_calls"] == m["calls"]),
                        http_status=200)
        except Exception as exc:  # noqa: BLE001 -- classified, never leaked
            err = http_error(exc, trace_id)
            span.update(http_status=err.status_code, error_type=type(exc).__name__)
            raise err from exc
    return AskResponse(
        answer=payload["answer"], refused=payload["refused"],
        citations=[Citation(**c) for c in payload["citations"]],
        sources=payload["sources"], latency_ms=latency_ms, cost_usd=round(cost_usd, 6),
        cached=bool(hit), trace_id=trace_id,
        cache_layer=hit.layer if hit else None,
        matched_question=(hit.matched_question if hit and hit.layer == "semantic"
                          else None),
        guard_flags=payload["guard_flags"])


# ---------------------------------------------------------------------------
# B2/B3 -- POST /ask/stream
#
# B3 decision: STREAM, THEN SEND A VALIDATION EVENT THE UI ACTS ON.
#   token*      prose as it is generated. The UI shows it as "unverified".
#   validation  after the full answer: enforce_citations + Lab 4's validator +
#               Lab 6 layer 5. status is
#                 verified   citations checked; here are the excerpts
#                 replaced   validation failed or a guard changed the text: the
#                            UI must replace what it showed with `answer`
#   error       {status, retry_after} if the provider fails mid-stream
# Why this one: the alternatives each give up the thing that matters most here.
# Buffering throws away the whole TTFT gain (generation is ~98% of latency).
# Holding citations alone does not stop an invalid answer being read. The
# validation event keeps TTFT AND keeps the guarantee that no invalid answer is
# left on screen; the cost is UI work and a window in which a user may read
# text that is later replaced -- small on this system, because the gate
# measured citation validity at 1.000 on the golden set.
# ---------------------------------------------------------------------------
def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data)}


def _stream_llm(messages: list[dict[str, Any]], emit) -> dict[str, Any]:
    """Stream one generation. Same request dict as aip.llm.raw_call, so the
    result lands in the same cache entry and /ask replays it (and vice versa)."""
    model = resolve_model(pipeline().config.tier)
    request = {"model": model, "messages": messages,
               "temperature": settings.temperature, "max_tokens": 1024,
               "response_format": None, "tools": None, "tool_choice": None}
    key = cache.make_key("chat", request)
    t0 = time.perf_counter()
    hit = cache.get(key)
    if hit is not None:
        emit(hit["text"])
        cost.record(cost.Usage(model, hit["usage"]["prompt_tokens"],
                               hit["usage"]["completion_tokens"], 0.0, 0.0,
                               cached=True, calls=1))
        tracing.event("llm.call", model=model, cached=True, cost_usd=0.0)
        return {"text": hit["text"], "finish_reason": hit.get("finish_reason"),
                "ttft_ms": (time.perf_counter() - t0) * 1000, "cost_usd": 0.0,
                "cached": True}
    if settings.offline:
        raise cache.CacheMiss("AIP_OFFLINE=1 and this generation is not cached")

    from litellm import completion

    with tracing.trace("llm.call", model=model, cached=False, streaming=True) as span:
        resp = completion(model=model, messages=messages, stream=True,
                          temperature=settings.temperature, max_tokens=1024,
                          timeout=settings.timeout_s,
                          stream_options={"include_usage": True})
        parts, ttft, finish, usage = [], None, None, None
        for chunk in resp:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            finish = chunk.choices[0].finish_reason or finish
            if delta:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                parts.append(delta)
                emit(delta)
        latency_ms = (time.perf_counter() - t0) * 1000
        text = "".join(parts)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        usd = cost.price_of(model, pt, ct)
        span.update(prompt_tokens=pt, completion_tokens=ct, cost_usd=round(usd, 6),
                    latency_ms=round(latency_ms, 1), ttft_ms=round(ttft or latency_ms, 1),
                    finish_reason=finish)
        cache.put(key, "chat", request, {
            "text": text, "tool_calls": [], "finish_reason": finish,
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "cost_usd": usd,
                      "latency_ms": latency_ms, "cached": False}})
        cost.record(cost.Usage(model, pt, ct, usd, latency_ms, cached=False, calls=1,
                               priced=cost.is_priced(model)))
    return {"text": text, "finish_reason": finish, "ttft_ms": ttft or latency_ms,
            "cost_usd": usd, "cached": False}


def _stream_worker(req: AskRequest, out: queue.Queue) -> None:
    """Runs on ONE thread end to end, so aip.tracing's thread-local span stack
    stays intact (a generator resumed on pool threads would corrupt it)."""
    t0 = time.perf_counter()
    rc, pipe = response_cache(), pipeline()
    safe_q, _ = redact_pii(req.question)
    state = {"started": False}
    try:
        with metering() as m, tracing.trace("http.ask_stream", question=safe_q[:120],
                                            mode="rag", top_k=req.top_k) as span:
            trace_id = span["trace_id"] = _trace_id(span)

            def emit(text: str) -> None:
                if not state["started"]:
                    state["started"] = True
                    span["ttft_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    out.put(_sse("meta", {"trace_id": trace_id,
                                          "cached": bool(span.get("cached")),
                                          "ttft_ms": span["ttft_ms"]}))
                out.put(_sse("token", {"text": text}))

            try:
                hit = rc.get(req.question, top_k=req.top_k, mode="rag")
                if hit:
                    p = hit.payload
                    span["cached"] = True
                    emit(p["answer"])
                    span.update(cached=True, cache_layer=hit.layer, refused=p["refused"],
                                cost_usd=0.0, http_status=200)
                    out.put(_sse("validation", {
                        "status": "verified", **p, "cached": True,
                        "cache_layer": hit.layer, "trace_id": trace_id, "cost_usd": 0.0,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "matched_question": (hit.matched_question
                                             if hit.layer == "semantic" else None)}))
                    return
                _check_budget()
                with tracing.trace("pipeline.answer", streaming=True):
                    prep = pipe.prepare(req.question, final_k=req.top_k)
                    with tracing.trace("stage.generate", n_sources=prep["n_sources"],
                                       streaming=True):
                        g = _stream_llm(prep["messages"], emit)
                        streamed = g["text"]
                        v = validate_answer(streamed, prep["n_sources"],
                                            g["finish_reason"])
                        text, repaired = streamed, False
                        if not v["valid"]:
                            text, repaired = pipe.repair(req.question, prep)
                    r = pipe.finish(req.question, text, prep["hits"], prep["raw"],
                                    prep["n_sources"], repaired, prep["retrieved"],
                                    prep["flags"])
                status = "verified" if r.answer == streamed else "replaced"
                payload = {"answer": r.answer, "refused": r.refused,
                           "citations": r.citations, "sources": r.sources,
                           "guard_flags": r.guard_flags}
                if status == "verified" and _cacheable(payload):
                    rc.put(req.question, payload, top_k=req.top_k, mode="rag")
                span.update(cached=False, refused=r.refused, cost_usd=round(m["cost_usd"], 6),
                            model_cached=bool(g["cached"]),
                            validation=status, http_status=200,
                            n_citations=len(r.citations), guard_flags=r.guard_flags)
                out.put(_sse("validation", {
                    "status": status, **payload, "cached": False, "trace_id": trace_id,
                    "reason": None if v["valid"] else v.get("reason"),
                    "cost_usd": round(m["cost_usd"], 6), "ttft_ms": span.get("ttft_ms"),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}))
            except Exception as exc:  # noqa: BLE001
                err = http_error(exc, trace_id)
                span.update(http_status=err.status_code, error_type=type(exc).__name__)
                out.put(_sse("error", {"status": err.status_code, **err.detail,
                                       "retry_after": (err.headers or {}).get("Retry-After")}))
                raise
    except Exception:  # noqa: BLE001 -- already reported as an event
        pass
    finally:
        out.put(None)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    if req.mode != "rag":
        raise HTTPException(status_code=422, detail="streaming supports mode='rag' only")
    from sse_starlette.sse import EventSourceResponse

    q: queue.Queue = queue.Queue()
    threading.Thread(target=_stream_worker, args=(req, q), daemon=True).start()
    # Wait for the first event before committing to a 200. An outage or an
    # exhausted budget shows up before the first token, so it still gets its
    # real status code (503 + Retry-After / 429) instead of a 200 with an
    # error buried in the stream.
    try:
        first = await asyncio.to_thread(q.get, True, DEADLINE_S)
    except queue.Empty:
        return JSONResponse(status_code=503, headers={"Retry-After": str(RETRY_AFTER_S)},
                            content={"detail": {"error": "upstream_unavailable",
                                                "message": f"no first token within "
                                                           f"{DEADLINE_S:.0f}s"}})
    if first is None or first["event"] == "error":
        detail = json.loads(first["data"]) if first else {"status": 500}
        status = int(detail.pop("status", 500))
        retry = detail.pop("retry_after", None)
        return JSONResponse(status_code=status, content={"detail": detail},
                            headers={"Retry-After": str(retry)} if retry else None)

    async def events():
        yield first
        while True:
            try:
                item = await asyncio.to_thread(q.get, True, DEADLINE_S)
            except queue.Empty:
                yield _sse("error", {"status": 503, "error": "upstream_unavailable",
                                     "message": "stream stalled", "retry_after":
                                     str(RETRY_AFTER_S)})
                return
            if item is None:
                return
            yield item

    return EventSourceResponse(events())


# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    p, rc = pipeline(), response_cache()
    return {
        "status": "ok", "uptime_s": round(time.time() - _STARTED, 1),
        "index": {"n_chunks": p.n_chunks, "n_docs": p.n_docs, "build_ms": p.build_ms,
                  "chunking": "markdown@400", "retriever": "dense (exact)",
                  "archived_docs": "excluded at build"},
        "pipeline": {"k": p.config.k, "final_k": p.config.final_k,
                     "guard_layers_rag": sorted(p.config.layers),
                     "fingerprint": p.config.fingerprint()},
        "models": {"profile": settings.profile, "generate": resolve_model("MAIN"),
                   "embed": resolve_model("EMBED"), "judge": resolve_model("LARGE")},
        "offline": settings.offline,
        "cache": {"llm_calls": cache.stats(), "responses": rc.size(),
                  "semantic_threshold": rc.threshold, "entity_guard": rc.entity_guard,
                  "since_start": rc.stats},
    }


@app.get("/metrics")
def metrics() -> dict:
    """C2, computed from today's traces (survives restarts), plus this process's
    budget meter from aip.cost."""
    out = traces.summarise(traces.load(since=traces.start_of_today()))
    out["process_budget"] = {**global_budget().as_dict(),
                             "limit_usd": global_budget().limit_usd}
    out["semantic_cache"] = response_cache().stats
    return out


@app.get("/trace/{trace_id}")
def get_trace(trace_id: str) -> dict:
    """C1: the span tree for one request, with the stages ranked by time."""
    t = traces.explain(trace_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"no trace {trace_id}")
    return t


class Feedback(BaseModel):
    trace_id: str
    question: str = Field(max_length=1000)
    answer: str = Field(max_length=5000)
    note: str = Field(default="", max_length=1000)


@app.post("/feedback")
def feedback(fb: Feedback) -> dict:
    """Stretch 3: a thumbs-down becomes a review-queue row with its trace id,
    which is how a real golden set grows."""
    REVIEW_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_QUEUE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**fb.model_dump(), "ts": time.time()}) + "\n")
    tracing.event("feedback.thumbs_down", trace_id=fb.trace_id)
    return {"queued": True}
