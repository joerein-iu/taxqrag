"""
pipeline/classical.py

Classical baseline RAG pipeline — greedy selection with conflict avoidance.

Kept minimal because benchmark/compare_tax.py inlines the classical path
(vs.search → greedy_tax_fallback → score_combination). This module
exists so the benchmark's import resolves.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_store import VectorStore
from quantum.qubo_tax import greedy_tax_fallback
from quantum.interactions import score_combination
from config import TOP_K_CANDIDATES, TOP_K_FINAL, MAX_TOKENS_CONTEXT


class ClassicalRAG:
    def __init__(self):
        self.vs = VectorStore()

    def query(self, question: str, verbose: bool = False) -> dict:
        t0 = time.time()
        candidates = self.vs.search(question, k=TOP_K_CANDIDATES)
        retrieval_ms = (time.time() - t0) * 1000

        t0 = time.time()
        selected = greedy_tax_fallback(
            candidates, k=TOP_K_FINAL, token_budget=MAX_TOKENS_CONTEXT
        )
        rerank_ms = (time.time() - t0) * 1000

        combo = score_combination(selected, verbose=verbose)

        return {
            "pipeline": "classical_greedy",
            "docs_used": selected,
            "num_candidates": len(candidates),
            "combination_analysis": combo,
            "timings": {
                "retrieval_ms": retrieval_ms,
                "reranking_ms": rerank_ms,
                "total_ms": retrieval_ms + rerank_ms,
            },
        }
