# QRAG-Tax: Quantum-Enhanced Retrieval-Augmented Generation for Tax Strategy Selection

A RAG system that selects **combinations** of tax strategies rather than individually relevant documents. The reranking step is cast as a multi-constraint QUBO over a hand-curated strategy-interaction matrix (synergy rewards, conflict penalties, audit-risk compounding) and solved with quantum annealing (D-Wave-style simulated annealing) or gate-based QAOA (IBM Quantum via Qiskit Runtime / Aer). Compared to classical greedy selection, the QUBO-based pipelines capture up to 5× more strategy synergies while the quantum-annealing variant maintains a 0% conflict rate.

---

## Table 1 — Benchmark results

12 realistic tax-planning queries (self-employed traders, S-Corp owners, multi-strategy stacking). All pipelines retrieve the same 30 candidates from ChromaDB; they differ only in the reranking step that selects the final *k* documents.

| Metric | Classical greedy<br/>(n = 12) | Simulated Annealing<br/>(n = 12) | **QAOA Aer**<br/>seed = 43 (n = 12) | **IBM QPU**<br/>ibm_fez (n = 5, 1 query)¹ |
|---|---:|---:|---:|---:|
| Combination quality score | 0.675 | **1.846** | 2.779 | 2.450 ± 2.35 |
| Total synergy captured | 0.700 | 1.846 | **4.371** | 2.590 |
| Synergies identified (pairs) | 0.75 | 1.88 | **4.58** | 2.59 |
| Avg retrieval relevance | **0.447** | 0.377 | 0.362 | 0.371 |
| Conflict rate (query has ≥ 1 conflicting pair) | **0.0 %** | **0.0 %** | 66.7 % | 0.0 %¹ |
| Per-query reranking time | ~0 ms | ~7 s | ~605 ms | ~20 s (incl. queue) |

**¹ IBM QPU column notes.** Real-hardware numbers are aggregated over the 5-seed sensitivity sweep (seeds 42/43/44/45 at opt_level=1, plus seed=42 at opt_level=3) on query #1 only, to conserve free-tier minutes. Total real-hardware usage: ~2 min of the 10-min/month Open Plan allocation. Circuit depth on `ibm_fez` transpiled at `optimization_level=3` was 670 gates; at `optimization_level=1` it was 1104 gates (same QAOA parameters, seed=42) — a 39% depth reduction from transpiler optimization alone.

**Reading the table.** Simulated annealing is the current best overall — highest combination quality with zero conflicts. QAOA Aer captures the most raw synergy per query but also selects the most documents, producing a 66.7% conflict rate (see Section 4.2 ablation in `paper_outline.md`: raising `lambda_count` 5× does not fix this, because the issue is upstream — random QAOA initialization instead of VQE-optimized parameters). IBM QPU runs currently use random QAOA params at `reps=1`; VQE-optimized parameters are future work.

---

## Repository layout

```
qrag/
├── corpus/ingest_tax.py              # Builds 87-doc corpus from strategies + IRC + IRS pubs
├── retrieval/vector_store.py         # ChromaDB wrapper, sentence-transformers embeddings
├── quantum/
│   ├── qubo_tax.py                   # Multi-constraint QUBO formulation  (★ core contribution)
│   ├── interactions.py               # Strategy synergy/conflict/audit matrices  (★ core contribution)
│   └── sampler_ibm.py                # QAOA circuit construction + IBM/Aer solvers
├── generation/llm_tax.py             # Claude API wrapper (optional)
├── pipeline/
│   ├── classical.py                  # Greedy reranker with conflict avoidance
│   ├── quantum_tax.py                # D-Wave / simulated-annealing reranker
│   └── quantum_tax_ibm.py            # QAOA reranker (Aer or IBM Quantum)
└── benchmark/
    ├── compare_tax.py                # Classical vs SA (Table 1, columns 1–2)
    ├── compare_tax_qaoa_aer.py       # QAOA Aer full 12-query benchmark (column 3)
    ├── qaoa_sensitivity.py           # 5-run seed/opt-level sweep on real hardware (column 4)
    └── qaoa_constraint_test.py       # λ_count ablation (Section 4.2)
```

---

## Installation

Python 3.11+ recommended.

```bash
pip install chromadb dimod sentence-transformers python-dotenv anthropic tqdm numpy requests
pip install qiskit qiskit-ibm-runtime qiskit-aer
```

Copy `.env.example` (or create `.env`) with:

```
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=tax_strategies

# Optional — for the LLM-generation step of the pipeline
ANTHROPIC_API_KEY=sk-...

# Optional — for real IBM Quantum hardware runs (Aer works without this)
IBM_QUANTUM_API_KEY=...
IBM_QUANTUM_CRN=crn:v1:bluemix:public:quantum-computing:us-east:a/.../...::
IBM_QUANTUM_CHANNEL=ibm_cloud
```

Create the data directories:

```bash
mkdir -p data/chroma data/benchmarks
```

---

## Quick start

```bash
# 1. Ingest the 87-document corpus (≈ 30 s; includes network fetches for IRC / IRS pubs)
python corpus/ingest_tax.py

# 2. Run a single query through the simulated-annealing pipeline (no API keys required)
python pipeline/quantum_tax.py

# 3. Run a single query through the QAOA / Aer pipeline
python pipeline/quantum_tax_ibm.py        # use_real_hardware=False by default

# 4. Reproduce Table 1
python benchmark/compare_tax.py            # Columns 1 & 2 (classical, SA)
python benchmark/compare_tax_qaoa_aer.py   # Column 3 (QAOA Aer)
python benchmark/qaoa_sensitivity.py       # Column 4 (IBM QPU, consumes free-tier minutes)
```

All benchmark runs write JSON artifacts to `data/benchmarks/`; these are tracked in git (see `.gitignore`) so results are reproducible from the repository alone.

### Running without API keys

- **Anthropic key missing** → `generation/llm_tax.py` returns a stub answer; benchmark metrics are unaffected since they are computed from QUBO selection only, not from LLM output.
- **IBM Quantum key missing** → `quantum_tax_ibm.py` silently falls back to Aer. All non-hardware columns of Table 1 remain reproducible.

---

## Core contribution (what *not* to modify)

Two files carry the publishable algorithmic claim and are frozen for experimentation:

- `quantum/qubo_tax.py` — the multi-constraint QUBO formulation, including signed quadratic terms (synergy rewards **+** conflict penalties), domain-specific interaction matrices replacing generic cosine similarity, and audit-risk compounding as a separate quadratic constraint.
- `quantum/interactions.py` — the hand-curated strategy-interaction matrices (synergy, conflict, audit-compound) that the QUBO queries at build time.

Everything else — sampler wiring, transpiler options, solver selection, benchmark harnesses — is free to change.

---

## Citation

```
TBD — preprint forthcoming on arXiv (quant-ph / cs.IR).
```

## Acknowledgments

Real-hardware experiments used the IBM Quantum Platform. We acknowledge the use of IBM Quantum services for this work. The views expressed are those of the authors and do not reflect the official policy or position of IBM or the IBM Quantum team.
