"""
benchmark/compare_tax_qaoa_aer.py

Table 1 — QAOA Simulator column.
Runs all 12 tax benchmark queries through QAOA on the local Aer simulator
with seed=43 (best-performing seed from the Section 4.3 hardware sweep).

Also runs the classical greedy baseline on the same queries for reference.
Aer is used here to conserve IBM Quantum free-tier minutes; real hardware
is reserved for targeted runs (Sections 4.3 and 5.x of the paper).
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_store import VectorStore
from quantum.sampler_ibm import solve_ibm_quantum
from quantum.qubo_tax import greedy_tax_fallback
from quantum.interactions import score_combination
from config import TOP_K_CANDIDATES, TOP_K_FINAL, MAX_TOKENS_CONTEXT


QAOA_SEED = 43

TEST_QUERIES = [
    "I trade MNQ futures full time through an S-Corp. How do I minimize taxes?",
    "What retirement strategies work best for an active futures trader with TTS?",
    "I have both W-2 income and S-Corp income. How do I optimize my tax position?",
    "I run a profitable S-Corp and want to maximize deductions this year",
    "How can I use my home for business tax purposes without triggering conflicts?",
    "What's the best combination of retirement accounts for a self-employed person?",
    "Can I use the Augusta Rule and home office deduction at the same time?",
    "How do Solo 401k and S-Corp salary optimization work together?",
    "What's the interaction between Section 1256 treatment and S-Corp election?",
    "I want to maximize retirement savings, minimize SE tax, and capture business deductions",
    "How do I structure a year with a business loss to maximize tax efficiency?",
    "What strategies stack legally for a self-employed person in the 32% bracket?",
]


def measure(selected):
    combo = score_combination(selected)
    return {
        "combination_score": combo["combination_score"],
        "total_synergy": combo["total_synergy"],
        "total_conflict": combo["total_conflict"],
        "max_audit_compound": combo["max_audit_compound"],
        "has_conflict": len(combo["warnings"]) > 0,
        "synergies_found": len(combo["synergies_found"]),
        "avg_relevance": float(np.mean([d["score"] for d in selected])) if selected else 0.0,
        "num_selected": len(selected),
    }


def main():
    print("=" * 78)
    print(f"QRAG TAX BENCHMARK — QAOA Simulator column (seed={QAOA_SEED})")
    print(f"Queries: {len(TEST_QUERIES)} | Pipeline: local Aer QAOA (reps=2)")
    print("=" * 78)

    vs = VectorStore()
    classical_rows = []
    qaoa_rows = []

    for qi, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n[{qi:2}/12] {query[:70]}")
        candidates = vs.search(query, k=TOP_K_CANDIDATES)

        # Classical greedy with conflict avoidance
        t0 = time.time()
        classical_docs = greedy_tax_fallback(
            candidates, k=TOP_K_FINAL, token_budget=MAX_TOKENS_CONTEXT
        )
        classical_ms = (time.time() - t0) * 1000
        c_m = measure(classical_docs)
        c_m.update({"query": query, "pipeline": "classical_greedy", "reranking_ms": classical_ms})
        classical_rows.append(c_m)

        # QAOA on Aer simulator
        t0 = time.time()
        out = solve_ibm_quantum(
            candidates,
            k=TOP_K_FINAL,
            token_budget=MAX_TOKENS_CONTEXT,
            use_real_hardware=False,
            shots=1024,
            qaoa_seed=QAOA_SEED,
            optimization_level=3,
        )
        qaoa_ms = (time.time() - t0) * 1000
        q_m = measure(out["selected_docs"])
        q_m.update({
            "query": query,
            "pipeline": "qaoa_aer",
            "reranking_ms": qaoa_ms,
            "energy": out["energy"],
            "method": out["method"],
        })
        qaoa_rows.append(q_m)

        print(f"   classical : combo={c_m['combination_score']:+.3f} "
              f"syn={c_m['synergies_found']} "
              f"conf={int(c_m['has_conflict'])} "
              f"rel={c_m['avg_relevance']:.3f}")
        print(f"   qaoa_aer  : combo={q_m['combination_score']:+.3f} "
              f"syn={q_m['synergies_found']} "
              f"conf={int(q_m['has_conflict'])} "
              f"rel={q_m['avg_relevance']:.3f} "
              f"E={q_m['energy']:.2f}")

    # ── Aggregate summary ────────────────────────────────────────────────
    def agg(rows, key):
        return float(np.mean([r[key] for r in rows]))

    metrics = [
        ("combination_score", "Combination Quality Score", "higher"),
        ("total_synergy", "Total Synergy Captured", "higher"),
        ("total_conflict", "Conflict Rate", "lower"),
        ("synergies_found", "Synergies Identified", "higher"),
        ("avg_relevance", "Avg Relevance Score", "higher"),
    ]

    print("\n" + "=" * 78)
    print(f"TABLE 1 — QAOA SIMULATOR COLUMN (seed={QAOA_SEED}, reps=2, optimization_level=3)")
    print("=" * 78)
    print(f"{'Metric':<36} {'Classical':>11} {'QAOA Aer':>11} {'Δ %':>9}  direction")
    for m, label, direction in metrics:
        c = agg(classical_rows, m)
        q = agg(qaoa_rows, m)
        delta = ((q - c) / max(abs(c), 1e-3)) * 100
        arrow = "↑" if delta > 0 else "↓"
        print(f"{label:<36} {c:>11.3f} {q:>11.3f} {arrow}{abs(delta):>7.1f}%  ({direction})")

    c_conflict = float(np.mean([r["has_conflict"] for r in classical_rows]))
    q_conflict = float(np.mean([r["has_conflict"] for r in qaoa_rows]))
    print(f"\nConflict avoidance rate:")
    print(f"  Classical : {c_conflict:.1%} of queries contained a conflicting pair")
    print(f"  QAOA Aer  : {q_conflict:.1%} of queries contained a conflicting pair")

    total_qaoa_time = sum(r["reranking_ms"] for r in qaoa_rows)
    total_classical_time = sum(r["reranking_ms"] for r in classical_rows)
    print(f"\nRuntime (sum across 12 queries):")
    print(f"  Classical : {total_classical_time:>8.0f} ms ({total_classical_time/12:.1f} ms/query)")
    print(f"  QAOA Aer  : {total_qaoa_time:>8.0f} ms ({total_qaoa_time/12:.1f} ms/query)")

    # Per-query table for the paper appendix
    print("\n" + "=" * 78)
    print("PER-QUERY TABLE (combo score, synergies, relevance)")
    print("=" * 78)
    print(f"{'#':>2} {'query':<48} {'cls combo':>9} {'qaoa combo':>10} {'qaoa syn':>8} {'qaoa rel':>8}")
    for i, (c, q) in enumerate(zip(classical_rows, qaoa_rows), start=1):
        short = q["query"][:48]
        print(f"{i:>2} {short:<48} "
              f"{c['combination_score']:>9.3f} "
              f"{q['combination_score']:>10.3f} "
              f"{q['synergies_found']:>8.1f} "
              f"{q['avg_relevance']:>8.3f}")

    # Persist
    os.makedirs("./data/benchmarks", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = f"./data/benchmarks/qaoa_aer_seed{QAOA_SEED}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": QAOA_SEED,
            "classical": classical_rows,
            "qaoa_aer": qaoa_rows,
            "timestamp": ts,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
