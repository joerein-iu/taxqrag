"""
pipeline/quantum_tax_ibm.py

Quantum-hybrid tax RAG pipeline using IBM Quantum backend.
Drop-in replacement for pipeline/quantum_tax.py — identical
interface, different quantum backend.
"""

import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_store import VectorStore
from quantum.sampler_ibm import solve_ibm_quantum
from quantum.interactions import score_combination
from generation.llm_tax import generate_tax
from config import TOP_K_CANDIDATES, TOP_K_FINAL, MAX_TOKENS_CONTEXT


class IBMQuantumTaxRAG:
    def __init__(self, use_real_hardware: bool = True):
        """
        Args:
            use_real_hardware: True  = IBM Quantum real QPU (uses free tier minutes)
                               False = Local Aer QAOA simulator (free, unlimited)
        """
        self.vs = VectorStore()
        self.use_real_hardware = use_real_hardware

    def query(self, question: str, verbose: bool = True) -> dict:
        total_start = time.time()

        # Step 1: Classical vector search
        t0 = time.time()
        candidates = self.vs.search(question, k=TOP_K_CANDIDATES)
        retrieval_time = (time.time() - t0) * 1000

        if verbose:
            print(f"\nRetrieved {len(candidates)} candidates in {retrieval_time:.0f}ms")

        # Step 2: IBM Quantum QAOA reranking
        t0 = time.time()
        quantum_result = solve_ibm_quantum(
            candidates,
            k=TOP_K_FINAL,
            token_budget=MAX_TOKENS_CONTEXT,
            use_real_hardware=self.use_real_hardware
        )
        selected = quantum_result["selected_docs"]
        rerank_time = (time.time() - t0) * 1000

        if verbose:
            print(f"IBM QAOA: {quantum_result['method']} | {rerank_time:.0f}ms")
            print(f"Qubits used: {quantum_result['num_qubits_used']}")
            for doc in selected:
                strat = doc.get("metadata", {}).get("strategy_id", doc["id"])
                print(f"  ✓ {strat} (score: {doc['score']:.3f})")

        # Analyze combination
        combo = score_combination(selected, verbose=verbose)

        # Step 3: LLM generation
        t0 = time.time()
        result = generate_tax(question, selected)
        generation_time = (time.time() - t0) * 1000

        total_time = (time.time() - total_start) * 1000

        return {
            "answer": result["answer"],
            "pipeline": f"ibm_{quantum_result['method']}",
            "docs_used": selected,
            "num_candidates": len(candidates),
            "num_qubits": quantum_result["num_qubits_used"],
            "quantum_energy": quantum_result["energy"],
            "combination_analysis": combo,
            "timings": {
                "retrieval_ms": retrieval_time,
                "reranking_ms": rerank_time,
                "generation_ms": generation_time,
                "total_ms": total_time
            }
        }


if __name__ == "__main__":
    # Start with simulator — free and unlimited
    # Flip use_real_hardware=True when ready to use IBM QPU minutes
    rag = IBMQuantumTaxRAG(use_real_hardware=False)

    query = "I run an S-Corp doing active futures trading on MNQ contracts. What strategies work best together?"
    print(f"Query: {query}\n")
    result = rag.query(query, verbose=True)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nCombination score: {result['combination_analysis']['combination_score']:.3f}")
