#!/usr/bin/env python3
"""Lab 7 — the shipped pipeline: Labs 3-5 for answers, Lab 6 for guards.

Everything here is imported from earlier labs; this file only decides which
configuration ships and puts the guards on the request path.

    Lab 3  retriever    markdown chunks @400, exact dense, archived docs
                        dropped at build time (labs.lab4.evaluate.build_retriever)
    Lab 4  generator    ANSWER_SYSTEM + validate + one repair retry, refusing
                        rather than returning a bad citation (_generate_with_repair)
    Lab 5  final_k=12   Lab 5 cut the context to final_k=5 and measured
                        correctness 0.925 -> 0.875 and refusal precision
                        1.000 -> 0.800. We ship the setting that measured better.
    Lab 6  guards       layer 1  delimit_untrusted + data-not-instructions clause
                        layer 2  screen_retrieved on retrieved text only (D3 fix)
                        layer 5  filter_output before anything leaves
                        layers 3+4 on mode="tools": structured final answer and a
                        READ-ONLY allowlist (search + premium). get_policy_details
                        is not exposed: Lab 6 N05 showed it leaks any customer's
                        record, and this service has no authenticated session to
                        bind a policy number to. issue_refund needs a human, and
                        an HTTP request has none, so it is never reachable.

Build it once with `Pipeline()`. Building it re-embeds the corpus (cached, but
still ~1 s of hashing and numpy), so the service builds it at startup.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import tracing  # noqa: E402
from aip.config import settings  # noqa: E402
from aip.guards import delimit_untrusted, enforce_citations  # noqa: E402
from aip.retrieval import Hit, format_context  # noqa: E402
from labs.lab4.evaluate import build_retriever  # noqa: E402
from labs.lab4.rag import (  # noqa: E402
    ANSWER_SYSTEM,
    REFUSAL,
    _build_user_prompt,
    _count_sources_in_context,
    _generate_with_repair,
    _is_partial_decline,
    validate_answer,
)
from labs.lab6 import agent as lab6  # noqa: E402

# ---------------------------------------------------------------------------
# Per-request call meter. aip.cost.Budget keeps one process-wide list of
# active budgets, so two concurrent requests would each be charged for the
# other's calls. This counts per thread instead, by wrapping aip.llm.raw_call
# (which chat() and structured() look up at call time).
#
#   cost_usd       what this request actually spent (a cache hit is $0)
#   recorded_*     what the calls cost / took when they were first made. Under
#                  AIP_OFFLINE=1 every call is a cache hit, so this is what the
#                  gate charges -- otherwise the cost and latency gates would
#                  pass whatever you changed.
# ---------------------------------------------------------------------------
_METER = threading.local()


def install_meter() -> None:
    import aip.llm

    if getattr(aip.llm.raw_call, "_metered", False):
        return
    real = aip.llm.raw_call

    def metered(*args, **kwargs):
        t0 = time.perf_counter()
        out = real(*args, **kwargs)
        m = getattr(_METER, "current", None)
        if m is not None:
            usage = out.get("usage", {})
            m["calls"] += 1
            m["cached_calls"] += bool(usage.get("cached"))
            m["wall_ms"] += (time.perf_counter() - t0) * 1000
            m["recorded_ms"] += float(usage.get("latency_ms") or 0.0)
            m["recorded_cost_usd"] += float(usage.get("cost_usd") or 0.0)
            if not usage.get("cached"):
                m["cost_usd"] += float(usage.get("cost_usd") or 0.0)
        return out

    metered._metered = True  # type: ignore[attr-defined]
    aip.llm.raw_call = metered


@contextlib.contextmanager
def metering() -> Iterator[dict[str, float]]:
    m = {"calls": 0, "cached_calls": 0, "wall_ms": 0.0, "recorded_ms": 0.0,
         "recorded_cost_usd": 0.0, "cost_usd": 0.0}
    prev = getattr(_METER, "current", None)
    _METER.current = m
    try:
        yield m
    finally:
        _METER.current = prev


GUARD_LAYERS = frozenset({1, 2, 5})          # on the RAG path
TOOL_LAYERS = frozenset({1, 2, 3, 4, 5})     # on the tools path
MAX_CONTEXT_CHARS = 8000


@dataclass(frozen=True)
class PipelineConfig:
    k: int = 12
    final_k: int = 12
    tier: str = "MAIN"
    layers: frozenset[int] = GUARD_LAYERS

    def fingerprint(self) -> str:
        """Part of every response-cache key: change the config, miss the cache."""
        blob = json.dumps({"k": self.k, "final_k": self.final_k, "tier": self.tier,
                           "model": settings.models[self.tier],
                           "embed": settings.models["EMBED"],
                           "layers": sorted(self.layers)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class Result:
    question: str
    answer: str
    refused: bool
    partial_decline: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)   # ranked, de-duplicated doc ids
    n_sources: int = 0
    citations_valid: bool = True
    repaired: bool = False
    guard_flags: list[str] = field(default_factory=list)
    context: str = ""


def cited_indices(text: str) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})


def build_citations(text: str, hits: list[Hit], n_sources: int) -> list[dict[str, Any]]:
    """Every [n] in the answer, resolved to the excerpt the model was shown."""
    return [{"index": i, "doc_id": hits[i - 1].doc_id, "excerpt": hits[i - 1].text.strip()}
            for i in cited_indices(text) if 1 <= i <= n_sources]


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        t0 = time.perf_counter()
        with tracing.trace("pipeline.build") as span:
            self.retriever = build_retriever()
            span["n_chunks"] = len(self.retriever.chunks)
        self.build_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.n_chunks = len(self.retriever.chunks)
        self.n_docs = len({c.doc_id for c in self.retriever.chunks})

    # -- stages ---------------------------------------------------------------
    def retrieve(self, question: str, final_k: int | None = None
                 ) -> tuple[list[Hit], list[str]]:
        """Wide for recall (k), narrow for the generator (final_k)."""
        final_k = final_k or self.config.final_k
        with tracing.trace("stage.retrieve", k=self.config.k, final_k=final_k) as span:
            hits = self.retriever.search(question, k=max(self.config.k, final_k))
            ranked: list[str] = []
            for h in hits:
                if h.doc_id not in ranked:
                    ranked.append(h.doc_id)
            span["top_doc"] = ranked[0] if ranked else None
        return hits[:final_k], ranked

    def screen(self, hits: list[Hit]) -> tuple[list[Hit], list[str]]:
        """Lab 6 layer 2: withhold one flagged excerpt, keep the rest."""
        if 2 not in self.config.layers:
            return hits, []
        flags: list[str] = []
        with tracing.trace("stage.screen", n=len(hits)) as span:
            out = []
            for h in hits:
                signals = lab6.screen_retrieved(h.text)
                if signals:
                    flags.append(f"withheld:{h.doc_id}:{'+'.join(signals)}")
                    note = (f"[excerpt withheld: flagged by the injection screen "
                            f"({', '.join(signals)})]")
                    h = replace(h, chunk=replace(h.chunk, text=note))
                out.append(h)
            span["n_flagged"] = len(flags)
        return out, flags

    def build_context(self, hits: list[Hit]) -> tuple[str, str, int]:
        """Returns (raw context, the context the model sees, n_sources shown)."""
        raw = format_context(hits, max_chars=MAX_CONTEXT_CHARS)
        n_sources = _count_sources_in_context(raw)
        shown = delimit_untrusted(raw) if 1 in self.config.layers else raw
        return raw, shown, n_sources

    def finish(self, question: str, text: str, hits: list[Hit], raw_context: str,
               n_sources: int, repaired: bool, retrieved: list[str],
               flags: list[str]) -> Result:
        """Final validation and Lab 6 layer 5. Shared with the streaming path."""
        with tracing.trace("stage.validate") as span:
            if 5 in self.config.layers:
                text, out_flags = lab6.filter_output(text, raw_context)
                flags = flags + out_flags
            refused = text.strip() in (REFUSAL, lab6.REFUSAL)
            ok, invalid = enforce_citations(text, n_sources)
            citations_valid = refused or ok
            span.update(refused=refused, citations_valid=citations_valid,
                        n_flags=len(flags))
        return Result(
            question=question, answer=text, refused=refused,
            partial_decline=_is_partial_decline(text),
            citations=[] if refused else build_citations(text, hits, n_sources),
            sources=list(dict.fromkeys(h.doc_id for h in hits[:n_sources])),
            retrieved=retrieved,
            n_sources=n_sources, citations_valid=citations_valid, repaired=repaired,
            guard_flags=flags, context=raw_context,
        )

    # -- the whole RAG path ---------------------------------------------------
    def answer(self, question: str, final_k: int | None = None) -> Result:
        with tracing.trace("pipeline.answer", final_k=final_k or self.config.final_k):
            hits, retrieved = self.retrieve(question, final_k)
            hits, flags = self.screen(hits)
            raw, shown, n_sources = self.build_context(hits)
            with tracing.trace("stage.generate", n_sources=n_sources,
                               tier=self.config.tier):
                text, _validation, repaired = _generate_with_repair(
                    question, shown, n_sources, tier=self.config.tier, strict=False)
            return self.finish(question, text, hits, raw, n_sources, repaired,
                               retrieved, flags)

    # -- streaming support ----------------------------------------------------
    def prepare(self, question: str, final_k: int | None = None) -> dict[str, Any]:
        """Everything before generation, for the streaming endpoint."""
        hits, retrieved = self.retrieve(question, final_k)
        hits, flags = self.screen(hits)
        raw, shown, n_sources = self.build_context(hits)
        return {"hits": hits, "retrieved": retrieved, "flags": flags, "raw": raw,
                "n_sources": n_sources,
                "messages": [{"role": "system", "content": ANSWER_SYSTEM},
                             {"role": "user",
                              "content": _build_user_prompt(question, shown, n_sources)}]}

    def repair(self, question: str, prep: dict[str, Any]) -> tuple[str, bool]:
        """The Lab 4 repair path, for a streamed answer that failed validation."""
        shown = (delimit_untrusted(prep["raw"]) if 1 in self.config.layers
                 else prep["raw"])
        text, _v, repaired = _generate_with_repair(
            question, shown, prep["n_sources"], tier=self.config.tier, strict=False)
        return text, repaired

    # -- the tools path (Lab 6) -----------------------------------------------
    def answer_with_tools(self, question: str, *, budget_usd: float = 0.05,
                          max_seconds: float = 45.0) -> dict[str, Any]:
        guard = lab6.make_guard(set(TOOL_LAYERS), read_only=True,
                                confirm_fn=lambda *_: False)
        with tracing.trace("pipeline.tools") as span:
            r = lab6.run_agent(question, guard=guard, layers=TOOL_LAYERS,
                               tier=self.config.tier, budget_usd=budget_usd,
                               max_seconds=max_seconds)
            span.update(stopped_because=r["stopped_because"],
                        n_tool_calls=len(r["tool_log"]),
                        tools=[t["tool"] for t in r["tool_log"]])
        return r


__all__ = ["Pipeline", "PipelineConfig", "Result", "REFUSAL", "validate_answer",
           "build_citations", "install_meter", "metering"]
