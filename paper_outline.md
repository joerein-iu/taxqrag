# QRAG-Tax — Paper Outline

**Working title.** Multi-Constraint QUBO Reranking for Domain-Specific Retrieval-Augmented Generation: A Case Study in US Tax Strategy Selection on IBM Quantum Hardware.

**Venue targets (primary → fallback).**
1. QTML 2026 (Quantum Techniques in Machine Learning) — best fit, empirical quantum-ML with domain application and hardware ablation.
2. QuantumML @ ICML 2026 (workshop) — backup if QTML reviewer fit is narrow.

**Authorship, dates, length.** TBD. Target 8 pages + references, supplementary appendix for per-query tables and full interaction matrices.

---

## Abstract (draft)

Retrieval-augmented generation typically selects documents individually relevant to a query, but real-world advisory tasks — tax planning, treatment protocols, regulatory compliance — require selecting *combinations* of items that interact. We formulate document reranking as a multi-constraint QUBO whose quadratic terms are domain-specific interaction scores (synergy rewards, conflict penalties, audit-risk compounding) rather than generic cosine similarity. On 12 realistic US-tax-planning queries over an 87-document corpus, simulated annealing on this QUBO captures 163% more synergies than classical greedy selection at 0% conflict rate. We further evaluate the formulation on gate-based QAOA via IBM Quantum (ibm_fez, 156-qubit Heron R2), using up-to-20-qubit problems, and provide a transpilation-optimization ablation (39% circuit-depth reduction at fixed QAOA parameters) and a QAOA seed-variance study (n=4 at optimization_level=1). We document a negative ablation: raising the cardinality-constraint weight does not fix QAOA's high conflict rate at random parameter initialization, isolating the root cause as parameter optimization rather than QUBO weighting.

---

## 1. Introduction (~1 page)

- RAG today: individual-relevance reranking (dot product / cross-encoder / LLM rerank).
- Why it fails for advisory domains: tax-law strategies interact — Augusta Rule conflicts with Home Office deduction, S-Corp salary synergizes with Solo 401(k) employer match, §1256 treatment is disqualified by S-Corp trading.
- Contribution (3 bullets):
  1. **Signed quadratic QUBO** with domain-specific interaction matrices replacing semantic similarity.
  2. **End-to-end pipeline** benchmarked on four solver backends: classical greedy, D-Wave-style simulated annealing, QAOA Aer, IBM Quantum hardware (ibm_fez).
  3. **Ablation isolating the QAOA random-init pathology** from QUBO weighting — evidence that the reps=2 / random-params conflict-rate problem is upstream of the QUBO design.

---

## 2. Related Work (~½ page)

- QRAG / variational reranking (prior art using cosine-similarity quadratic terms).
- D-Wave QUBO formulations for combinatorial optimization in NLP.
- QAOA for ML-adjacent problems — Farhi et al., Zhou et al., recent IBM benchmarks.
- Domain-specific RAG: legal, medical, tax — none combine the interaction structure into the reranker.

---

## 3. Method (~1½ pages)

### 3.1 Pipeline
Retrieve → candidate set → QUBO construction → quantum/classical solve → decode → LLM generation.

### 3.2 Multi-Constraint QUBO (Section 3.2, `quantum/qubo_tax.py`)
Linear terms: `α · relevance + β · credibility + γ · recency`.
Quadratic terms (signed, this is the novelty):
- `−λ_syn · Σ synergy_ij x_i x_j` (reward)
- `+λ_con · Σ conflict_ij x_i x_j` (penalty)
- `+λ_aud · Σ (audit_ij − 1) x_i x_j` (compound-risk penalty)
- Token-budget constraint (penalty)
- Cardinality constraint `|S| = k` (penalty)

Default weights: `α=0.5, β=0.3, γ=0.2, λ_syn=1.5, λ_con=3.0, λ_aud=1.0, λ_tok=1.5, λ_count=1.0`.

### 3.3 Interaction Matrices (`quantum/interactions.py`)
Hand-curated ground-truth from tax law. 14 synergy entries (0.65–0.95), 7 conflict entries (0.80–1.00), 4 audit-compound entries (1.2×–2.0×). 10 primary strategies × 30+ secondary IRC / IRS-publication docs.

### 3.4 Solver Backends
- **Classical greedy with conflict avoidance** — `greedy_tax_fallback` (baseline).
- **Simulated annealing** — `dimod.SimulatedAnnealingSampler`, 50 reads × 500 sweeps.
- **QAOA Aer** — local simulator, reps=2, random-init θ,γ.
- **QAOA IBM Quantum** — `ibm_fez` (Heron R2, 156 qubits), reps=1, transpiler `optimization_level` ablated.

---

## 4. Experiments (~3 pages)

### 4.1 Main Comparison (Table 1)
12 tax queries × all 4 backends. **Key numbers to fill in:**

| Metric | Classical | SA | QAOA Aer (seed=43) | IBM QPU (n=5, 1q) |
|---|---:|---:|---:|---:|
| Combination score | 0.675 | **1.846** | 2.779 | 2.450 ± 2.35 |
| Total synergy | 0.700 | 1.846 | **4.371** | 2.590 |
| Synergies identified | 0.75 | 1.88 | **4.58** | 2.59 |
| Avg relevance | **0.447** | 0.377 | 0.362 | 0.371 |
| Conflict rate | **0 %** | **0 %** | 66.7 % | 0 % (n=5, 1 query) |

Headline claim (defensible): **SA captures 173% more combination quality than classical at zero conflict cost, with a 16% relevance trade-off.**
Honest caveat (must include): **QAOA Aer with random-init parameters is high-conflict** (66.7%) and selects ~20 docs per query rather than k=5 — see Section 4.2.

### 4.2 QUBO Tuning Ablation (`benchmark/qaoa_constraint_test.py`)
4 conflict queries (5, 7, 9, 11) × `λ_count ∈ {1.0, 5.0}` at seed=43, reps=2, opt_level=1.
**Key finding:** 3 of 4 queries produce bit-identical output — raising the cardinality weight 5× does **not** change the selected bitstring. Only Q7 flips, and it flips to the all-zeros degenerate solution (0 docs, 0 combo score).
Aggregate conflict rate: 100% → 75% (but entirely driven by one degenerate flip).
**Interpretation:** the 66.7% conflict rate in §4.1 is **not** a QUBO-weighting problem; it is a QAOA-parameter-initialization problem. This isolates the next research target (Section 5) from a weighting confound.

### 4.3 QAOA Parameter Sensitivity & Transpiler Ablation (`benchmark/qaoa_sensitivity.py`)
5 runs on `ibm_fez`, query #1 only (to conserve free-tier minutes).
Seed=42 @ opt_level=1: depth **1104**, energy **−44.39**, combo **1.200**.
Seed=42 @ opt_level=3: depth **670**, energy **−73.61**, combo **0.700**.
**Transpilation delta (held QAOA params):** 39.3% depth reduction, 66% lower Ising energy, but lower downstream combo score.
Seed variance (n=4, opt_level=1, seeds 42/43/44/45): depth stdev 161 (CV 16%), combo stdev 2.55 (CV 88%), energy stdev 35.0.
**Headline finding:** energy and combination-score are **not monotone** — the best-energy run (seed=42 lvl=3, E=−73.61) produced the worst downstream combination (0.70); the worst-energy run (seed=43, E=+24.63) produced the best combination (6.65). The QAOA objective function is not aligned with the downstream synergy-quality objective at `reps=1` with random init parameters.

---

## 5. Discussion & Future Work (~½ page)

- VQE-optimized QAOA parameters as the principled fix for Section 4.2's finding.
- Deeper QAOA layers (reps=3+) vs current NISQ-friendly reps=1–2.
- Larger candidate sets — current MAX_QUBITS=20 truncates 30 → 20; future hardware with higher-fidelity 2-qubit gates could support the full set.
- Cross-domain transfer: medical treatment protocols, legal statute combinations, financial-product packaging — same formulation, different interaction matrices.

---

## 6. Reproducibility (~¼ page, or Appendix)

- All benchmark JSON artifacts committed in `data/benchmarks/`.
- Seeds and transpiler options reported per-experiment.
- `sampler_ibm.py` now takes `qaoa_seed` and `optimization_level` as explicit parameters; `seed_transpiler` and `seed_simulator` passed through to Aer for full determinism at the solver level.
- Interaction matrices (`quantum/interactions.py`) are the only source of tax-domain knowledge; swapping them yields a different-domain QRAG without touching the QUBO formulation.

---

## Figures & Tables

- **Table 1** (main comparison, §4.1) — 4 columns × 5 metrics + runtime row.
- **Table 2** (λ_count ablation, §4.2) — per-query, both weightings.
- **Table 3** (QAOA sensitivity, §4.3) — 5 hardware runs × 7 columns (seed, lvl, depth, energy, combo, synergy, job_id).
- **Figure 1** — pipeline architecture diagram (retrieve → QUBO → solve → decode → generate).
- **Figure 2** — interaction-matrix heatmap (synergy minus conflict).
- **Figure 3** — bar chart of combination score per query, grouped by backend.
- **Figure 4** — circuit-depth vs `optimization_level` scatter (could extend beyond 1-vs-3 with additional hardware runs if free-tier budget permits).
- **Figure 5** — energy ↔ combo-score scatter across seeds, labeled with job IDs (supports the "not monotone" claim in §4.3).

---

## Status tracker

- [x] Corpus ingestion working (87 docs)
- [x] Classical greedy baseline
- [x] Simulated annealing benchmark (12 queries)
- [x] QAOA Aer benchmark (12 queries, seeded)
- [x] Real IBM hardware smoke test (ibm_fez, job d7i5h23jne2c7394sh60)
- [x] QAOA seed-variance sweep (5 runs, Section 4.3)
- [x] λ_count ablation (Section 4.2)
- [ ] Full-corpus ingestion with all IRS publications (stretch)
- [ ] VQE-optimized QAOA parameter runs (Section 5 future work — prototype before final draft)
- [ ] Diagram generation (Figures 1 & 2)
- [ ] Manuscript draft
- [ ] arXiv preprint upload
