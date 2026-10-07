#!/usr/bin/env python3
"""Lab 7 B1 — two response-cache layers, and why they have different risks.

    exact     key = hash(normalised question, top_k, mode, pipeline fingerprint)
              A hit IS the same question. It cannot be wrong.
    semantic  key = embedding of the question; hit when cosine >= threshold.
              A hit is a DIFFERENT question that looks similar. It can be wrong,
              and when it is, nothing errors: the answer is fast, cited, and
              about something the user did not ask.

Both sit in front of the pipeline, so a hit skips retrieval and generation.
They are separate from aip.cache, which caches individual model calls.

The semantic threshold is NOT the starter's 0.95. It is measured -- see
labs/lab7/semantic_sweep.py and reports/lab7_semantic_sweep.json -- and set by
LAB7_SEMANTIC_THRESHOLD. The entity guard is the second mitigation from T1
§4.1: two questions that name different plans, products or numbers never
share an answer, however close their embeddings are.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from aip.config import settings
from aip.embed import embed

_DB = settings.cache_dir / "lab7_responses.sqlite3"
_LOCK = threading.Lock()


def normalise(question: str) -> str:
    """Case, whitespace and trailing punctuation do not change the question."""
    q = re.sub(r"\s+", " ", question.strip().lower())
    return q.rstrip(" ?!.")


# Words that change the answer while barely moving the embedding. The sweep's
# wrong hits were almost all of this shape: same template, one slot changed.
_ENTITY_PATTERNS = [
    re.compile(r"\b(bronze|silver|gold|platinum)\b"),
    re.compile(r"\b(senior|group|corporate|top-?up|super top-?up|critical illness|"
               r"personal accident|hospital cash|travel|motor|life|maternity|"
               r"outpatient|opd|dental|ayush|cataract|domiciliary)\b"),
    re.compile(r"\b(cashless|reimbursement|annual|annually|instalment|monthly|individual|"
               r"floater|archived|2024)\b"),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\b"),
]


_CANONICAL = {"annually": "annual", "opd": "outpatient"}


def entities(question: str) -> frozenset[str]:
    q = normalise(question)
    found = (m.group(0).replace("-", "").replace(" ", "")
             for p in _ENTITY_PATTERNS for m in p.finditer(q))
    return frozenset(_CANONICAL.get(e, e) for e in found)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB, timeout=30)
    conn.execute("""CREATE TABLE IF NOT EXISTS responses (
                    key TEXT PRIMARY KEY, question TEXT NOT NULL, scope TEXT NOT NULL,
                    vector TEXT, payload TEXT NOT NULL, created_at REAL NOT NULL)""")
    return conn


@dataclass
class CacheHit:
    layer: str                 # "exact" | "semantic"
    payload: dict[str, Any]
    matched_question: str
    similarity: float = 1.0


class ResponseCache:
    """Exact layer always on; semantic layer on when threshold > 0."""

    def __init__(self, scope: str, *, semantic_threshold: float = 0.0,
                 entity_guard: bool = True):
        self.scope = scope                     # pipeline fingerprint
        self.threshold = semantic_threshold
        self.entity_guard = entity_guard
        self.stats = {"exact_hits": 0, "semantic_hits": 0, "semantic_blocked": 0,
                      "misses": 0}
        self._questions: list[str] = []
        self._vectors: list[np.ndarray] = []
        with _LOCK, _connect() as conn:
            for q, vec in conn.execute(
                    "SELECT question, vector FROM responses WHERE scope = ? "
                    "AND vector IS NOT NULL", (scope,)):
                self._questions.append(q)
                self._vectors.append(np.frombuffer(base64.b64decode(vec), np.float32))

    def key(self, question: str, **params: Any) -> str:
        blob = json.dumps({"q": normalise(question), "scope": self.scope, **params},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, question: str, **params: Any) -> CacheHit | None:
        with _LOCK, _connect() as conn:
            row = conn.execute("SELECT question, payload FROM responses WHERE key = ?",
                               (self.key(question, **params),)).fetchone()
        if row:
            self.stats["exact_hits"] += 1
            return CacheHit("exact", json.loads(row[1]), row[0])
        if self.threshold > 0 and self._vectors and params.get("mode", "rag") == "rag":
            hit = self._semantic(question, params)
            if hit:
                self.stats["semantic_hits"] += 1
                return hit
        self.stats["misses"] += 1
        return None

    def _semantic(self, question: str, params: dict[str, Any]) -> CacheHit | None:
        q = embed(question, input_type="query")
        sims = np.vstack(self._vectors) @ q
        mine = entities(question)
        for i in np.argsort(-sims):
            if sims[i] < self.threshold:
                return None
            if self.entity_guard and entities(self._questions[i]) != mine:
                self.stats["semantic_blocked"] += 1
                continue
            payload = self._load(self._questions[i], params)
            if payload is not None:
                return CacheHit("semantic", payload, self._questions[i], float(sims[i]))
        return None

    def _load(self, question: str, params: dict[str, Any]) -> dict[str, Any] | None:
        with _LOCK, _connect() as conn:
            row = conn.execute("SELECT payload FROM responses WHERE key = ?",
                               (self.key(question, **params),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, question: str, payload: dict[str, Any], **params: Any) -> None:
        vec_b64 = None
        if self.threshold > 0 and params.get("mode", "rag") == "rag":
            v = embed(question, input_type="query").astype(np.float32)
            vec_b64 = base64.b64encode(v.tobytes()).decode("ascii")
            self._questions.append(question)
            self._vectors.append(v)
        with _LOCK, _connect() as conn:
            conn.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?)",
                         (self.key(question, **params), question, self.scope, vec_b64,
                          json.dumps(payload), time.time()))

    def size(self) -> int:
        with _LOCK, _connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM responses WHERE scope = ?",
                                (self.scope,)).fetchone()[0]

    def clear(self) -> int:
        self._questions, self._vectors = [], []
        with _LOCK, _connect() as conn:
            return conn.execute("DELETE FROM responses WHERE scope = ?",
                                (self.scope,)).rowcount
