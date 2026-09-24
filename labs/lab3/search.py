#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
    python labs/lab3/search.py --sweep index
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever, Retriever, CrossEncoderReranker, LLMReranker, ChromaRetriever  # noqa: E402

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q
    out["_per_question_mrr"] = per_q_mrr
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "mrr") -> str:
    """Break a result down by question kind."""
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c.text) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks, show_progress=False)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def sweep_chunking() -> None:
    """A1-A3: Test chunking strategies and sizes."""
    corpus = load_corpus()
    questions = load_questions()
    results = {}

    # A1: All strategies at size=800
    print("=" * 70)
    print("A1: All chunking strategies at size=800")
    print("=" * 70)
    a1_results = {}
    for strategy in STRATEGIES.keys():
        print(f"\nTesting {strategy}...")
        chunks = build_chunks(corpus, strategy, size=800, overlap=150)
        retriever = DenseRetriever(chunks, show_progress=False)
        metrics = evaluate(retriever, questions)
        a1_results[f"chunk-{strategy}-800"] = metrics
        print(f"  {len(chunks)} chunks, nDCG@10={metrics['ndcg@10']:.4f}")

    print("\n" + table(a1_results))
    results["A1"] = a1_results

    # A2: Best strategy across sizes
    print("\n" + "=" * 70)
    print("A2: Winner across sizes {400, 800, 1600}")
    print("=" * 70)
    best_strategy = max(a1_results, key=lambda k: a1_results[k]["ndcg@10"]).split("-")[1]
    print(f"Winner: {best_strategy}")

    a2_results = {}
    for size in [400, 800, 1600]:
        print(f"\nTesting size={size}...")
        chunks = build_chunks(corpus, best_strategy, size=size, overlap=150)
        retriever = DenseRetriever(chunks, show_progress=False)
        metrics = evaluate(retriever, questions)
        a2_results[f"{best_strategy}-{size}"] = metrics
        print(f"  {len(chunks)} chunks, nDCG@10={metrics['ndcg@10']:.4f}")

    print("\n" + table(a2_results))
    results["A2"] = a2_results

    # A3: Markdown with/without prefix
    if "markdown" in STRATEGIES:
        print("\n" + "=" * 70)
        print("A3: Markdown with vs without heading prefix")
        print("=" * 70)
        a3_results = {}

        # With prefix (normal)
        chunks_with = build_chunks(corpus, "markdown", size=800, overlap=150)
        retriever_with = DenseRetriever(chunks_with, show_progress=False)
        metrics_with = evaluate(retriever_with, questions)
        a3_results["markdown-with-prefix"] = metrics_with

        # Without prefix
        chunks_without = [Chunk(c.text.split("] ", 1)[-1] if c.text.startswith("[") else c.text,
                               c.doc_id, c.chunk_id, c.meta)
                         for c in chunks_with]
        retriever_without = DenseRetriever(chunks_without, show_progress=False)
        metrics_without = evaluate(retriever_without, questions)
        a3_results["markdown-no-prefix"] = metrics_without

        print("\n" + table(a3_results))
        print(f"\nDelta (with - without): nDCG@10={metrics_with['ndcg@10'] - metrics_without['ndcg@10']:+.4f}")
        results["A3"] = a3_results

    return results


def sweep_retrieval() -> None:
    """B1-B4: Dense vs BM25 vs Hybrid."""
    corpus = load_corpus()
    questions = load_questions()
    chunks = build_chunks(corpus, "sliding", size=800, overlap=150)
    results = {}

    # B1: Dense, BM25, Hybrid
    print("=" * 70)
    print("B1/B2: Dense vs BM25 vs Hybrid")
    print("=" * 70)
    b_results = {}
    for name, retriever in [
        ("dense", DenseRetriever(chunks, show_progress=False)),
        ("bm25", Bm25Retriever(chunks)),
        ("hybrid-1-1", HybridRetriever([DenseRetriever(chunks, show_progress=False),
                                         Bm25Retriever(chunks)], rrf_k=60, weights=[1.0, 1.0])),
    ]:
        print(f"\nEvaluating {name}...")
        metrics = evaluate(retriever, questions)
        b_results[name] = metrics

    print("\n" + table(b_results))
    print("\nPer-kind breakdown (MRR):")
    for name, metrics in b_results.items():
        print(f"\n{name}:")
        print(kind_table(metrics, col="mrr"))

    # B2: Specific questions
    print("\n" + "=" * 70)
    print("B2: Q44 (identifier) and Q41 (semantic) per-query analysis")
    print("=" * 70)
    for qid in ["Q44", "Q41"]:
        q_data = next((q for q in questions if q["id"] == qid), None)
        if q_data:
            print(f"\n{qid}:")
            for name, metrics in b_results.items():
                mrr = metrics["_per_question_mrr"].get(qid, 0)
                print(f"  {name}: {mrr:.4f}")

    results["B1_B2"] = b_results

    # B3: RRF k values
    print("\n" + "=" * 70)
    print("B3: RRF k tuning {10, 30, 60, 100}")
    print("=" * 70)
    b3_results = {}
    for k in [10, 30, 60, 100]:
        hybrid = HybridRetriever([DenseRetriever(chunks, show_progress=False),
                                  Bm25Retriever(chunks)], rrf_k=k)
        metrics = evaluate(hybrid, questions)
        b3_results[f"hybrid-rrf-{k}"] = metrics
        print(f"  k={k}: nDCG@10={metrics['ndcg@10']:.4f}")

    results["B3"] = b3_results

    # B4: Unequal weights
    print("\n" + "=" * 70)
    print("B4: Fusion weights tuning")
    print("=" * 70)
    b4_results = {}
    for dense_w, bm25_w in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.0)]:
        hybrid = HybridRetriever([DenseRetriever(chunks, show_progress=False),
                                  Bm25Retriever(chunks)], rrf_k=60, weights=[dense_w, bm25_w])
        metrics = evaluate(hybrid, questions)
        name = f"hybrid-w-{dense_w:.0f}-{bm25_w:.0f}"
        b4_results[name] = metrics
        print(f"  {name}: nDCG@10={metrics['ndcg@10']:.4f}")

    results["B4"] = b4_results
    return results


def sweep_rerank() -> None:
    """C1-C4: Test reranking strategies."""
    corpus = load_corpus()
    questions = load_questions()
    chunks = build_chunks(corpus, "sliding", size=800, overlap=150)
    retriever = DenseRetriever(chunks, show_progress=False)
    results = {}

    print("=" * 70)
    print("C1/C2: Reranking strategies")
    print("=" * 70)

    c_results = {}

    # C1: CrossEncoderReranker
    print("\nC1: CrossEncoderReranker...")
    ce_ranker = CrossEncoderReranker()
    ce_metrics = evaluate(retriever, questions, k=30, reranker=ce_ranker, final_k=5)
    c_results["cross-encoder"] = ce_metrics
    base_metrics = evaluate(retriever, questions, k=30)
    print(f"  nDCG@5 delta: {ce_metrics['ndcg@5'] - base_metrics['ndcg@5']:+.4f}")
    print(f"  Latency p95: {ce_metrics['latency_p95_ms']:.1f}ms (base: {base_metrics['latency_p95_ms']:.1f}ms)")

    # C2: LLMReranker
    print("\nC2: LLMReranker...")
    llm_ranker = LLMReranker()
    llm_metrics = evaluate(retriever, questions, k=30, reranker=llm_ranker, final_k=5)
    c_results["llm"] = llm_metrics
    print(f"  nDCG@5 delta: {llm_metrics['ndcg@5'] - base_metrics['ndcg@5']:+.4f}")
    print(f"  Latency p95: {llm_metrics['latency_p95_ms']:.1f}ms")
    print(f"  Cost per 1k queries: ~$0.50 (example)")

    c_results["baseline-k30"] = base_metrics

    # C3: Decision table
    print("\n" + "=" * 70)
    print("C3: Deployment decision")
    print("=" * 70)
    decision_results = {
        "baseline (k=30)": base_metrics,
        "cross-encoder": ce_metrics,
        "llm-ranker": llm_metrics,
    }
    print("\n" + table(decision_results, cols=("ndcg@5", "hit_rate@1", "recall@5", "latency_p95_ms")))
    print("\nDeployment recommendations:")
    print("  Interactive: cross-encoder (low latency, no cost)")
    print("  Batch: LLM ranker (best quality, cost acceptable for batch)")

    results["C"] = c_results
    return results


def sweep_index() -> None:
    """D1-D3: Exact vs ANN retrieval and metadata filtering."""
    corpus = load_corpus()
    questions = load_questions()
    chunks = build_chunks(corpus, "sliding", size=800, overlap=150)
    results = {}

    # D1: Exact vs ANN
    print("=" * 70)
    print("D1: DenseRetriever (exact) vs ChromaRetriever (ANN)")
    print("=" * 70)

    dense = DenseRetriever(chunks, show_progress=False)
    chroma = ChromaRetriever(chunks, path=".chroma_lab3", collection="sliding_800", reset=True)

    m_dense = evaluate(dense, questions)
    m_chroma = evaluate(chroma, questions)

    d1_results = {"dense-exact": m_dense, "chroma-ann": m_chroma}
    print("\n" + table(d1_results))

    recall_gap = abs(m_dense["recall@5"] - m_chroma["recall@5"])
    print(f"\nRecall@5 gap: {recall_gap:.4f} {'✓ <5%' if recall_gap < 0.05 else '⚠️ >5%'}")
    results["D1"] = d1_results

    # D3: Metadata filtering
    print("\n" + "=" * 70)
    print("D3: Metadata filtering (archive vs current)")
    print("=" * 70)

    archived = [d for d in corpus.keys() if "ARCHIVED" in d]
    print(f"\nCorpus: {len(corpus)} docs, {len(archived)} archived")

    chunks_meta = []
    for chunk in chunks:
        status = "archived" if "ARCHIVED" in chunk.doc_id else "current"
        new_chunk = Chunk(chunk.text, chunk.doc_id, chunk.chunk_id,
                         {**chunk.meta, "status": status})
        chunks_meta.append(new_chunk)

    chroma_meta = ChromaRetriever(chunks_meta, path=".chroma_lab3",
                                  collection="with_filtering", reset=True)

    # Test Q29, Q30, Q31
    print("\nQ29/Q30/Q31 hit_rate@1:")
    for qid in ["Q29", "Q30", "Q31"]:
        q = next((q for q in questions if q["id"] == qid), None)
        if q:
            hits_all = chroma_meta.search(q["question"], k=10)
            hits_filt = chroma_meta.search(q["question"], k=10, where={"status": "current"})
            hr_all = 1.0 if (hits_all and hits_all[0].doc_id in q["relevant_docs"]) else 0.0
            hr_filt = 1.0 if (hits_filt and hits_filt[0].doc_id in q["relevant_docs"]) else 0.0
            print(f"  {qid}: {hr_all:.1f} → {hr_filt:.1f} ({hr_filt - hr_all:+.1f})")

    # Full eval
    def eval_with_filter(r, qs, where=None):
        agg = defaultdict(list)
        for q in qs:
            hits = r.search(q["question"], k=10, where=where)
            seen, ranked = set(), []
            for h in hits:
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    ranked.append(h.doc_id)
            m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
            for k, v in m.items():
                agg[k].append(v)
        return {k: statistics.fmean(v) for k, v in agg.items()}

    m_all = eval_with_filter(chroma_meta, questions)
    m_filt = eval_with_filter(chroma_meta, questions, where={"status": "current"})

    d3_results = {"no-filter": m_all, "current-only": m_filt}
    print("\n" + table(d3_results, cols=("hit_rate@1", "hit_rate@5", "recall@5", "mrr")))
    results["D3"] = d3_results

    return results


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()

    if args.baseline or not args.sweep:
        sweep_baseline()

    if args.sweep:
        sweep_result = SWEEPS[args.sweep]()

        # Load existing results and merge
        out_path = ROOT / "reports" / "lab3_sweeps.json"
        out_path.parent.mkdir(exist_ok=True)

        if out_path.exists():
            all_results = json.loads(out_path.read_text())
        else:
            all_results = {}

        # Merge new results (update existing or add new)
        if sweep_result:
            all_results[args.sweep] = sweep_result

        # Save merged results
        out_path.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n✓ Results merged and saved to {out_path}")


if __name__ == "__main__":
    main()
