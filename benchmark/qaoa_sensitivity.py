"""
benchmark/qaoa_sensitivity.py

Section 4.3 — QAOA Parameter Sensitivity Analysis.

Runs the same QUBO problem on IBM real hardware across:
  - seed=42 at optimization_level=1 and 3  (depth comparison, held seed)
  - seeds 43, 44, 45 at optimization_level=1 (variance data)

Reports circuit depths, energies, combination scores, selected strategies,
and standard deviation across seeds at level=1.
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_store import VectorStore
from quantum.sampler_ibm import solve_ibm_quantum
from quantum.interactions import score_combination
from config import TOP_K_CANDIDATES, TOP_K_FINAL, MAX_TOKENS_CONTEXT


QUERY = (
    "I trade MNQ futures full time through an S-Corp. "
    "How do I minimize taxes?"
)

# (seed, optimization_level) tuples, in run order
SWEEP = [
    (42, 1),
    (42, 3),
    (43, 1),
    (44, 1),
    (45, 1),
]


def main():
    print(f"Query: {QUERY}")
    vs = VectorStore()
    candidates = vs.search(QUERY, k=TOP_K_CANDIDATES)
    print(f"Retrieved {len(candidates)} candidates.")
    print()

    results = []
    for seed, opt_level in SWEEP:
        print("=" * 70)
        print(f"Run: seed={seed} | optimization_level={opt_level}")
        print("=" * 70)
        t0 = time.time()
        out = solve_ibm_quantum(
            candidates,
            k=TOP_K_FINAL,
            token_budget=MAX_TOKENS_CONTEXT,
            use_real_hardware=True,
            shots=1024,
            qaoa_seed=seed,
            optimization_level=opt_level,
        )
        wall_ms = (time.time() - t0) * 1000

        selected = out["selected_docs"]
        combo = score_combination(selected)
        extras = out.get("hardware_extras", {}) or {}
        strat_ids = [
            d.get("metadata", {}).get("strategy_id") or d["id"]
            for d in selected
        ]

        row = {
            "seed": seed,
            "optimization_level": opt_level,
            "backend": extras.get("backend"),
            "job_id": extras.get("job_id"),
            "circuit_depth": extras.get("circuit_depth"),
            "energy": out["energy"],
            "combination_score": combo["combination_score"],
            "total_synergy": combo["total_synergy"],
            "total_conflict": combo["total_conflict"],
            "max_audit_compound": combo["max_audit_compound"],
            "num_selected": len(selected),
            "strategies": strat_ids,
            "synergies_found": combo["synergies_found"],
            "warnings": combo["warnings"],
            "method": out["method"],
            "wall_ms": wall_ms,
            "solve_time_ms": out["solve_time_ms"],
        }
        results.append(row)

        print(f"  depth={row['circuit_depth']} energy={row['energy']:.4f} "
              f"combo={row['combination_score']:.3f} job={row['job_id']}")
        print()

    # Aggregates — variance across seed-only runs (held optimization_level=1)
    level1_runs = [r for r in results if r["optimization_level"] == 1]
    depths_l1 = [r["circuit_depth"] for r in level1_runs if r["circuit_depth"] is not None]
    energies_l1 = [r["energy"] for r in level1_runs if r["energy"] is not None]
    scores_l1 = [r["combination_score"] for r in level1_runs]

    def stats(xs):
        if not xs:
            return None
        if len(xs) == 1:
            return {"mean": xs[0], "stdev": 0.0, "min": xs[0], "max": xs[0], "n": 1}
        return {
            "mean": statistics.mean(xs),
            "stdev": statistics.stdev(xs),
            "min": min(xs),
            "max": max(xs),
            "n": len(xs),
        }

    summary = {
        "level1_depth_stats": stats(depths_l1),
        "level1_energy_stats": stats(energies_l1),
        "level1_combo_score_stats": stats(scores_l1),
        "seed42_level1_vs_level3": {
            "depth": {
                "level_1": next((r["circuit_depth"] for r in results if r["seed"] == 42 and r["optimization_level"] == 1), None),
                "level_3": next((r["circuit_depth"] for r in results if r["seed"] == 42 and r["optimization_level"] == 3), None),
            },
            "energy": {
                "level_1": next((r["energy"] for r in results if r["seed"] == 42 and r["optimization_level"] == 1), None),
                "level_3": next((r["energy"] for r in results if r["seed"] == 42 and r["optimization_level"] == 3), None),
            },
            "combination_score": {
                "level_1": next((r["combination_score"] for r in results if r["seed"] == 42 and r["optimization_level"] == 1), None),
                "level_3": next((r["combination_score"] for r in results if r["seed"] == 42 and r["optimization_level"] == 3), None),
            },
        },
    }

    # Print summary table
    print("=" * 70)
    print("RESULTS — per run")
    print("=" * 70)
    print(f"{'seed':>5} {'lvl':>4} {'depth':>6} {'energy':>10} {'combo':>7} {'syn':>6} {'con':>6} {'aud':>5} {'job_id':>22}")
    for r in results:
        print(f"{r['seed']:>5} {r['optimization_level']:>4} "
              f"{r['circuit_depth']!s:>6} "
              f"{r['energy']:>10.4f} "
              f"{r['combination_score']:>7.3f} "
              f"{r['total_synergy']:>6.2f} "
              f"{r['total_conflict']:>6.2f} "
              f"{r['max_audit_compound']:>5.2f} "
              f"{r['job_id'] or '-':>22}")

    print()
    print("=" * 70)
    print("SEED=42 — level=1 vs level=3 (held QAOA params)")
    print("=" * 70)
    pair = summary["seed42_level1_vs_level3"]
    print(f"  depth   : level_1={pair['depth']['level_1']}  level_3={pair['depth']['level_3']}")
    print(f"  energy  : level_1={pair['energy']['level_1']}  level_3={pair['energy']['level_3']}")
    print(f"  combo   : level_1={pair['combination_score']['level_1']}  level_3={pair['combination_score']['level_3']}")

    print()
    print("=" * 70)
    print("SEED VARIANCE — level=1, seeds 42/43/44/45")
    print("=" * 70)
    for label, s in [
        ("depth", summary["level1_depth_stats"]),
        ("energy", summary["level1_energy_stats"]),
        ("combo_score", summary["level1_combo_score_stats"]),
    ]:
        if s is None:
            print(f"  {label}: no data")
        else:
            print(f"  {label}: n={s['n']} mean={s['mean']:.4f} stdev={s['stdev']:.4f} min={s['min']:.4f} max={s['max']:.4f}")

    os.makedirs("./data/benchmarks", exist_ok=True)
    fname = f"./data/benchmarks/qaoa_sensitivity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({"query": QUERY, "runs": results, "summary": summary}, f, indent=2, default=str)
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()
