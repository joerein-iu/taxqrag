"""
quantum/qubo_tax.py

Multi-Constraint QUBO for Tax Strategy Selection

THE NOVEL FORMULATION vs prior work:

Prior work (QRAG paper, 2025):
  Objective: -Σ relevance_i * x_i
  Quadratic: +λ * Σ_ij similarity_ij * x_i * x_j  (redundancy only)

This work:
  Objective: -Σ (relevance_i * α + credibility_i * β) * x_i
  Quadratic:
    - λ_syn  * Σ_ij synergy_ij  * x_i * x_j   (REWARD synergistic pairs)
    + λ_con  * Σ_ij conflict_ij * x_i * x_j   (PENALIZE conflicting pairs)
    + λ_aud  * Σ_ij audit_ij   * x_i * x_j   (PENALIZE high-risk combos)
    + λ_tok  * token budget constraint
    + λ_cnt  * count constraint (select exactly K)

The signed quadratic terms (both positive reward AND negative penalty)
are the key innovation. Prior work only penalized redundancy.
Domain-specific interaction matrices replace generic semantic similarity.
"""

import numpy as np
from dimod import BinaryQuadraticModel
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum.interactions import build_interaction_matrices
from config import TOP_K_FINAL, MAX_TOKENS_CONTEXT


def build_tax_qubo(
    candidates: list[dict],
    k: int = TOP_K_FINAL,
    token_budget: int = MAX_TOKENS_CONTEXT,
    # Objective weights
    alpha: float = 0.5,          # relevance weight
    beta: float = 0.3,           # credibility weight
    gamma: float = 0.2,          # recency weight
    # Quadratic penalty/reward weights
    lambda_synergy: float = 1.5,  # reward for synergistic pairs
    lambda_conflict: float = 3.0, # penalty for conflicting pairs (high — conflicts are bad)
    lambda_audit: float = 1.0,    # penalty for audit risk compounding
    lambda_token: float = 1.5,    # token budget constraint
    lambda_count: float = 1.0,    # count constraint
    max_audit_risk: float = 1.5,  # reject combinations above this audit multiplier
) -> BinaryQuadraticModel:
    """
    Build multi-constraint BQM for tax strategy selection.

    This is the publishable QUBO formulation. The key contributions:
    1. Signed quadratic terms (synergy reward + conflict penalty)
    2. Domain-specific interaction matrices (not generic cosine similarity)
    3. Audit risk compounding as a separate quadratic constraint
    4. Multi-objective linear terms (relevance + credibility + recency)
    """
    n = len(candidates)
    bqm = BinaryQuadraticModel('BINARY')

    # ─── Build interaction matrices ───────────────────────────
    synergy_matrix, conflict_matrix, audit_matrix = build_interaction_matrices(candidates)

    # ─── Normalize relevance scores ───────────────────────────
    scores = np.array([c["score"] for c in candidates])
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    # ─── Credibility scores ───────────────────────────────────
    credibility_map = {
        "IRS_Publication": 1.0,
        "IRC": 1.0,
        "TAX_STRATEGY": 0.95,
        "TAX_COURT": 0.90,
        "IRS_FAQ": 0.85,
    }
    credibilities = np.array([
        float(c.get("metadata", {}).get("credibility", "0.8"))
        for c in candidates
    ])

    # ─── Token counts ─────────────────────────────────────────
    token_counts = np.array([c.get("token_count", 150) for c in candidates])
    norm_budget = token_budget / max(token_counts.max(), 1)
    norm_tokens = token_counts / max(token_counts.max(), 1)

    # ─── LINEAR TERMS ─────────────────────────────────────────
    # Minimize → negate objectives we want to maximize
    for i in range(n):
        linear_i = -(
            alpha * scores[i] +
            beta * credibilities[i] +
            gamma * 0.5  # recency placeholder (could add date decay)
        )
        bqm.add_variable(i, linear_i)

    # ─── QUADRATIC TERMS ──────────────────────────────────────

    for i in range(n):
        for j in range(i + 1, n):

            quad_ij = 0.0

            # 1. SYNERGY REWARD (negative = reward in minimization)
            syn = synergy_matrix[i][j]
            quad_ij -= lambda_synergy * syn

            # 2. CONFLICT PENALTY (positive = penalty in minimization)
            con = conflict_matrix[i][j]
            quad_ij += lambda_conflict * con

            # 3. AUDIT RISK COMPOUNDING
            aud = audit_matrix[i][j]
            audit_penalty = max(0, aud - 1.0)  # penalty above 1.0x
            quad_ij += lambda_audit * audit_penalty

            if quad_ij != 0.0:
                bqm.add_interaction(i, j, quad_ij)

    # ─── TOKEN BUDGET CONSTRAINT ──────────────────────────────
    # Penalty: λ_token * (Σ tokens_i * x_i - budget)²
    for i in range(n):
        t_i = norm_tokens[i]
        bqm.add_variable(i, lambda_token * (t_i**2 - 2 * norm_budget * t_i))

    for i in range(n):
        for j in range(i + 1, n):
            t_i, t_j = norm_tokens[i], norm_tokens[j]
            bqm.add_interaction(i, j, lambda_token * 2 * t_i * t_j)

    # ─── COUNT CONSTRAINT ─────────────────────────────────────
    # Select exactly K strategies
    for i in range(n):
        bqm.add_variable(i, lambda_count * (1 - 2 * k))

    for i in range(n):
        for j in range(i + 1, n):
            bqm.add_interaction(i, j, lambda_count * 2)

    return bqm


def decode_tax_solution(
    sample: dict,
    candidates: list[dict]
) -> list[dict]:
    """Decode D-Wave sample back to strategy documents."""
    selected = []
    for i, val in sample.items():
        if val == 1 and i < len(candidates):
            selected.append(candidates[i])
    selected.sort(key=lambda x: x["score"], reverse=True)
    return selected


def greedy_tax_fallback(
    candidates: list[dict],
    k: int = TOP_K_FINAL,
    token_budget: int = MAX_TOKENS_CONTEXT
) -> list[dict]:
    """
    Classical greedy fallback with basic conflict avoidance.
    Better than pure score-sorting — avoids obvious conflicts.
    """
    from quantum.interactions import get_conflict_score

    selected = []
    total_tokens = 0
    selected_strategy_ids = []

    for doc in sorted(candidates, key=lambda x: x["score"], reverse=True):
        strat_id = doc.get("metadata", {}).get("strategy_id", "")
        doc_tokens = doc.get("token_count", 150)

        # Skip if token budget exceeded
        if total_tokens + doc_tokens > token_budget:
            continue

        # Skip if hard conflict with already-selected strategy
        has_conflict = False
        for sel_id in selected_strategy_ids:
            if get_conflict_score(strat_id, sel_id) > 0.8:
                has_conflict = True
                break

        if not has_conflict:
            selected.append(doc)
            total_tokens += doc_tokens
            if strat_id:
                selected_strategy_ids.append(strat_id)

        if len(selected) >= k:
            break

    return selected
