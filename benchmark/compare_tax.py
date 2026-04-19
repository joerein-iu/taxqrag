"""
benchmark/compare_tax.py

Head-to-head benchmark: Classical vs Quantum for tax strategy selection.

Key metrics for the paper:
  combination_score  — synergy minus conflicts in selected set
  conflict_rate      — % of selections containing conflicting strategies
  synergy_captured   — % of available synergies captured
  relevance          — avg relevance score of selected docs
  diversity          — variety of strategy types selected
"""

import json
import numpy as np
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.quantum_tax import QuantumTaxRAG
from pipeline.classical import ClassicalRAG
from quantum.interactions import score_combination
from quantum.qubo_tax import greedy_tax_fallback
from retrieval.vector_store import VectorStore
from config import TOP_K_FINAL, MAX_TOKENS_CONTEXT


# Realistic tax queries covering different user situations
TEST_QUERIES = [
    # Self-employed trader scenarios
    "I trade MNQ futures full time through an S-Corp. How do I minimize taxes?",
    "What retirement strategies work best for an active futures trader with TTS?",
    "I have both W-2 income and S-Corp income. How do I optimize my tax position?",

    # Small business owner scenarios
    "I run a profitable S-Corp and want to maximize deductions this year",
    "How can I use my home for business tax purposes without triggering conflicts?",
    "What's the best combination of retirement accounts for a self-employed person?",

    # Strategy combination scenarios
    "Can I use the Augusta Rule and home office deduction at the same time?",
    "How do Solo 401k and S-Corp salary optimization work together?",
    "What's the interaction between Section 1256 treatment and S-Corp election?",

    # Complex multi-strategy scenarios
    "I want to maximize retirement savings, minimize SE tax, and capture business deductions",
    "How do I structure a year with a business loss to maximize tax efficiency?",
    "What strategies stack legally for a self-employed person in the 32% bracket?",
]


def run_tax_benchmark(num_runs: int = 2):
    print("=" * 70)
    print("QRAG TAX BENCHMARK: Classical vs Quantum")
    print(f"Queries: {len(TEST_QUERIES)} | Runs per query: {num_runs}")
    print("=" * 70)

    vs = VectorStore()
    quantum_rag = QuantumTaxRAG(use_real_hardware=False)
    classical_rag = ClassicalRAG()

    results = {"classical": [], "quantum": []}

    for query in TEST_QUERIES:
        print(f"\nQuery: {query[:65]}...")

        for pipeline_name in ["classical", "quantum"]:
            run_metrics = []

            for run in range(num_runs):
                if pipeline_name == "quantum":
                    result = quantum_rag.query(query, verbose=False)
                    docs_used = result["docs_used"]
                    combo = result["combination_analysis"]
                else:
                    # Classical: get candidates then greedy select
                    candidates = vs.search(query, k=50)
                    docs_used = greedy_tax_fallback(
                        candidates, k=TOP_K_FINAL, token_budget=MAX_TOKENS_CONTEXT
                    )
                    combo = score_combination(docs_used)
                    result = {"timings": {"reranking_ms": 0, "total_ms": 0}}

                run_metrics.append({
                    "combination_score": combo["combination_score"],
                    "total_synergy": combo["total_synergy"],
                    "total_conflict": combo["total_conflict"],
                    "has_conflict": len(combo["warnings"]) > 0,
                    "synergies_found": len(combo["synergies_found"]),
                    "avg_relevance": np.mean([d["score"] for d in docs_used]) if docs_used else 0,
                    "reranking_ms": result["timings"].get("reranking_ms", 0),
                })

            avg = {k: np.mean([r[k] for r in run_metrics]) for k in run_metrics[0]}
            avg["query"] = query
            avg["pipeline"] = pipeline_name
            results[pipeline_name].append(avg)

            print(f"  [{pipeline_name:10}] "
                  f"combo_score: {avg['combination_score']:+.3f} | "
                  f"synergies: {avg['synergies_found']:.1f} | "
                  f"conflicts: {avg['has_conflict']:.1%} | "
                  f"relevance: {avg['avg_relevance']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY — KEY METRICS FOR PAPER")
    print("=" * 70)

    metrics = [
        ("combination_score", "Combination Quality Score", "higher is better"),
        ("total_synergy", "Total Synergy Captured", "higher is better"),
        ("total_conflict", "Conflict Rate", "lower is better"),
        ("synergies_found", "Synergies Identified", "higher is better"),
        ("avg_relevance", "Avg Relevance Score", "higher is better"),
    ]

    paper_results = {}
    for metric, label, direction in metrics:
        c_avg = np.mean([r[metric] for r in results["classical"]])
        q_avg = np.mean([r[metric] for r in results["quantum"]])
        delta = ((q_avg - c_avg) / max(abs(c_avg), 0.001)) * 100
        arrow = "↑" if delta > 0 else "↓"
        print(f"{label:35} | Classical: {c_avg:7.3f} | Quantum: {q_avg:7.3f} | Δ {arrow}{abs(delta):.1f}%  ({direction})")
        paper_results[metric] = {"classical": float(c_avg), "quantum": float(q_avg), "delta_pct": float(delta)}

    # Conflict avoidance rate
    c_conflict = np.mean([r["has_conflict"] for r in results["classical"]])
    q_conflict = np.mean([r["has_conflict"] for r in results["quantum"]])
    print(f"\nConflict avoidance rate:")
    print(f"  Classical: {c_conflict:.1%} of queries selected conflicting strategies")
    print(f"  Quantum:   {q_conflict:.1%} of queries selected conflicting strategies")

    # Save
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "methodology": "Multi-constraint QUBO vs greedy classical selection",
        "num_queries": len(TEST_QUERIES),
        "num_runs_per_query": num_runs,
        "results": results,
        "summary": paper_results,
        "conflict_rate": {"classical": float(c_conflict), "quantum": float(q_conflict)}
    }

    os.makedirs("./data/benchmarks", exist_ok=True)
    fname = f"./data/benchmarks/tax_benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {fname}")
    return output


if __name__ == "__main__":
    run_tax_benchmark(num_runs=2)
