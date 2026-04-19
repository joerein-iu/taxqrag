"""
benchmark/qaoa_constraint_test.py

Section 4.2 — QUBO Tuning Ablation.

Hypothesis: the 66.7% conflict rate observed for QAOA Aer in Section 4.1
is driven by the cardinality constraint being too weak relative to the
synergy/conflict quadratic terms — when QAOA-at-random-params produces a
near-uniform bitstring, the decoder returns all 20 qubits=1 because the
count penalty isn't large enough to pull the solver away from that region.

Test: raise lambda_count from its default (1.0) to 5.0 and re-run Aer on
the 4 conflict-producing queries at identical seed/reps/opt-level.
All other QUBO weights unchanged. No modifications to qubo_tax.py or
interactions.py — lambda_count is a public parameter of build_tax_qubo.

Queries chosen: 5, 7, 9, 11 from the benchmark (all had conf=1 in the
seed=43 Aer run). Queries 5 and 7 produced combo=-2.900 on the Augusta
↔ HomeOffice hard-conflict pair.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_store import VectorStore
from quantum.qubo_tax import build_tax_qubo, decode_tax_solution
from quantum.interactions import score_combination
from quantum.sampler_ibm import qubo_to_ising, build_qaoa_circuit
from config import TOP_K_CANDIDATES, TOP_K_FINAL, MAX_TOKENS_CONTEXT


QAOA_SEED = 43
REPS = 2
OPT_LEVEL = 1
SHOTS = 1024
MAX_QUBITS = 20

TEST_QUERIES = [
    (5,  "How can I use my home for business tax purposes without triggering conflicts?"),
    (7,  "Can I use the Augusta Rule and home office deduction at the same time?"),
    (9,  "What's the interaction between Section 1256 treatment and S-Corp election?"),
    (11, "How do I structure a year with a business loss to maximize tax efficiency?"),
]


def aer_qaoa_solve(bqm, candidates_hw, qaoa_seed=QAOA_SEED, reps=REPS,
                   optimization_level=OPT_LEVEL, shots=SHOTS):
    """Inline Aer QAOA solve — keeps the ablation self-contained.

    Mirrors the behavior of quantum.sampler_ibm._solve_qaoa_simulator but
    with an explicit `optimization_level` passed to qiskit.transpile.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    num_qubits = len(candidates_hw)
    h, J, offset = qubo_to_ising(bqm)

    qaoa_circuit, _ham = build_qaoa_circuit(h, J, num_qubits, reps=reps)
    num_params = len(qaoa_circuit.parameters)
    rng = np.random.default_rng(qaoa_seed)
    params = rng.uniform(0, np.pi, num_params)
    bound = qaoa_circuit.assign_parameters(params)
    bound.measure_all()

    simulator = AerSimulator()
    compiled = transpile(
        bound,
        simulator,
        optimization_level=optimization_level,
        seed_transpiler=qaoa_seed,
    )
    result = simulator.run(compiled, shots=shots, seed_simulator=qaoa_seed).result()
    counts = result.get_counts()

    best_bitstring = max(counts, key=counts.get)
    sample = {i: int(bit) for i, bit in enumerate(reversed(best_bitstring))}
    energy = bqm.energy(sample)
    depth = compiled.depth()

    selected = decode_tax_solution(sample, candidates_hw)
    return {
        "sample": sample,
        "selected_docs": selected,
        "energy": energy,
        "circuit_depth": depth,
        "best_bitstring": best_bitstring,
    }


def run_one(query, candidates_hw, lambda_count):
    bqm = build_tax_qubo(
        candidates_hw,
        k=TOP_K_FINAL,
        token_budget=MAX_TOKENS_CONTEXT,
        lambda_count=lambda_count,
    )
    t0 = time.time()
    out = aer_qaoa_solve(bqm, candidates_hw)
    wall_ms = (time.time() - t0) * 1000
    selected = out["selected_docs"]
    combo = score_combination(selected)
    strat_ids = [
        d.get("metadata", {}).get("strategy_id") or d["id"]
        for d in selected
    ]
    return {
        "lambda_count": lambda_count,
        "num_selected": len(selected),
        "circuit_depth": out["circuit_depth"],
        "energy": out["energy"],
        "combination_score": combo["combination_score"],
        "total_synergy": combo["total_synergy"],
        "total_conflict": combo["total_conflict"],
        "max_audit_compound": combo["max_audit_compound"],
        "has_conflict": len(combo["warnings"]) > 0,
        "synergies_found": len(combo["synergies_found"]),
        "warnings_count": len(combo["warnings"]),
        "avg_relevance": float(np.mean([d["score"] for d in selected])) if selected else 0.0,
        "strategies": strat_ids,
        "synergies_text": combo["synergies_found"],
        "warnings_text": combo["warnings"],
        "wall_ms": wall_ms,
    }


def main():
    print("=" * 78)
    print(f"Section 4.2 — QUBO λ_count Ablation | seed={QAOA_SEED} reps={REPS} opt_level={OPT_LEVEL}")
    print(f"{len(TEST_QUERIES)} conflict-producing queries | Aer simulator")
    print("=" * 78)

    vs = VectorStore()
    results = []

    for idx, query in TEST_QUERIES:
        print(f"\n--- Q{idx}: {query[:68]}")
        candidates = vs.search(query, k=TOP_K_CANDIDATES)
        candidates_hw = sorted(candidates, key=lambda x: x["score"], reverse=True)[:MAX_QUBITS]

        row = {"query_index": idx, "query": query, "runs": {}}
        for lc in [1.0, 5.0]:
            r = run_one(query, candidates_hw, lambda_count=lc)
            row["runs"][str(lc)] = r
            tag = "conflict!" if r["has_conflict"] else "clean    "
            print(f"   λ_count={lc:>3}  n={r['num_selected']:>2}  "
                  f"combo={r['combination_score']:>+7.3f}  "
                  f"syn={r['total_synergy']:>5.2f}  "
                  f"con={r['total_conflict']:>4.2f}  "
                  f"aud={r['max_audit_compound']:>4.2f}  "
                  f"E={r['energy']:>9.2f}  [{tag}]")
            if r["strategies"]:
                print(f"            selected: {', '.join(r['strategies'])}")
        results.append(row)

    # ── Aggregate comparison ────────────────────────────────────────────
    def agg(results, lc, key):
        vals = [r["runs"][str(lc)][key] for r in results]
        return float(np.mean(vals))

    print("\n" + "=" * 78)
    print("AGGREGATE — λ_count=1.0 vs 5.0, mean across 4 conflict-producing queries")
    print("=" * 78)
    metrics = [
        ("num_selected", "Docs selected"),
        ("combination_score", "Combination score"),
        ("total_synergy", "Total synergy"),
        ("total_conflict", "Total conflict (penalty)"),
        ("synergies_found", "Synergies identified"),
        ("avg_relevance", "Avg relevance"),
        ("circuit_depth", "Circuit depth"),
    ]
    print(f"{'Metric':<32} {'λ=1.0':>10} {'λ=5.0':>10} {'Δ':>10}")
    for key, label in metrics:
        m1 = agg(results, 1.0, key)
        m5 = agg(results, 5.0, key)
        print(f"{label:<32} {m1:>10.3f} {m5:>10.3f} {m5-m1:>+10.3f}")

    def conflict_rate(results, lc):
        return float(np.mean([r["runs"][str(lc)]["has_conflict"] for r in results]))

    print(f"\nConflict rate:")
    print(f"  λ_count=1.0 : {conflict_rate(results, 1.0):.1%} of the 4 queries")
    print(f"  λ_count=5.0 : {conflict_rate(results, 5.0):.1%} of the 4 queries")

    # Persist
    os.makedirs("./data/benchmarks", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = f"./data/benchmarks/qaoa_lambda_count_ablation_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": QAOA_SEED,
            "reps": REPS,
            "optimization_level": OPT_LEVEL,
            "lambda_counts_tested": [1.0, 5.0],
            "queries": results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
