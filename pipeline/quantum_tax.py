"""
pipeline/quantum_tax.py

Quantum-hybrid tax strategy RAG pipeline.
Uses multi-constraint QUBO with synergy/conflict interaction matrix.
"""

import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_store import VectorStore
from quantum.qubo_tax import build_tax_qubo, decode_tax_solution, greedy_tax_fallback
from quantum.interactions import score_combination
from generation.llm_tax import generate_tax
from dimod import SimulatedAnnealingSampler
from config import TOP_K_CANDIDATES, TOP_K_FINAL, MAX_TOKENS_CONTEXT, DWAVE_API_TOKEN


class QuantumTaxRAG:
    def __init__(self, use_real_hardware: bool = False):
        self.vs = VectorStore()
        self.use_real_hardware = use_real_hardware

    def query(self, question: str, verbose: bool = True) -> dict:
        total_start = time.time()

        # Step 1: Vector search
        t0 = time.time()
        candidates = self.vs.search(question, k=TOP_K_CANDIDATES)
        retrieval_time = (time.time() - t0) * 1000

        if verbose:
            print(f"\nRetrieved {len(candidates)} candidates in {retrieval_time:.0f}ms")

        # Step 2: Build multi-constraint QUBO
        t0 = time.time()
        bqm = build_tax_qubo(candidates, k=TOP_K_FINAL, token_budget=MAX_TOKENS_CONTEXT)

        # Solve
        method = None
        sample = None
        energy = None

        if self.use_real_hardware and DWAVE_API_TOKEN:
            try:
                from dwave.system import LeapHybridBQMSampler
                sampler = LeapHybridBQMSampler(token=DWAVE_API_TOKEN)
                response = sampler.sample(bqm, time_limit=3, label="QRAG-Tax")
                best = response.first
                sample = dict(best.sample)
                energy = best.energy
                method = "quantum_hybrid"
            except Exception as e:
                if verbose:
                    print(f"D-Wave failed: {e}, falling back to simulated annealing")

        if sample is None:
            sa = SimulatedAnnealingSampler()
            response = sa.sample(bqm, num_reads=50, num_sweeps=500)
            best = response.first
            sample = dict(best.sample)
            energy = best.energy
            method = "simulated_annealing"

        rerank_time = (time.time() - t0) * 1000

        # Decode solution
        selected = decode_tax_solution(sample, candidates)

        # Fallback if solver returned garbage
        if not selected or len(selected) == 0:
            selected = greedy_tax_fallback(candidates, k=TOP_K_FINAL)
            method = "greedy_fallback"

        if verbose:
            print(f"Quantum reranking: {method} | {rerank_time:.0f}ms")
            print(f"Selected {len(selected)} strategies:")
            for doc in selected:
                strat = doc.get("metadata", {}).get("strategy_id", doc["id"])
                print(f"  ✓ {strat} (score: {doc['score']:.3f})")

        # Analyze the combination
        combo = score_combination(selected, verbose=verbose)

        # Step 3: LLM generation
        t0 = time.time()
        result = generate_tax(question, selected, analyze_interactions=True)
        generation_time = (time.time() - t0) * 1000

        total_time = (time.time() - total_start) * 1000

        return {
            "answer": result["answer"],
            "pipeline": f"quantum_{method}",
            "docs_used": selected,
            "num_candidates": len(candidates),
            "quantum_energy": energy,
            "combination_analysis": combo,
            "timings": {
                "retrieval_ms": retrieval_time,
                "reranking_ms": rerank_time,
                "generation_ms": generation_time,
                "total_ms": total_time
            },
            "tokens": {
                "input": result["input_tokens"],
                "output": result["output_tokens"]
            }
        }


if __name__ == "__main__":
    rag = QuantumTaxRAG(use_real_hardware=False)

    test_queries = [
        "I run an S-Corp doing active futures trading on MNQ contracts. What strategies work best together?",
        "How can I minimize self-employment taxes while maximizing retirement contributions as a self-employed trader?",
        "I have a profitable S-Corp year and want to maximize deductions before year end. What strategies can I combine?",
        "What's the optimal strategy stack for a self-employed person with W-2 income and side business income?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)
        result = rag.query(query, verbose=True)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCombination score: {result['combination_analysis']['combination_score']:.3f}")
        if result['combination_analysis']['warnings']:
            for w in result['combination_analysis']['warnings']:
                print(w)
