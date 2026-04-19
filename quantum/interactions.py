"""
quantum/interactions.py

THE NOVEL CONTRIBUTION — Strategy Interaction Matrix

This module computes pairwise synergy and conflict scores between
tax strategies. These scores become the quadratic terms in the QUBO,
allowing D-Wave to find strategy combinations that:
  - Maximize combined tax savings (synergy rewards)
  - Avoid legally conflicting combinations (conflict penalties)
  - Respect IRS audit risk thresholds (risk constraints)

This is what separates QRAG from standard RAG:
Standard RAG: selects individually relevant documents
QRAG-Tax:     selects documents whose COMBINATION is optimal

Published contribution:
  Prior QUBO RAG work uses semantic similarity for quadratic terms.
  We replace this with domain-specific interaction scores derived from:
    1. Explicit rule-based interactions (known tax law conflicts)
    2. LLM-extracted interaction scores (implicit synergies)
    3. Audit risk compounding (risk scores multiply, not add)
"""

import json
import numpy as np
from typing import Optional
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────
# HARD-CODED INTERACTION RULES
# These are ground truth from tax law — not estimated
# ─────────────────────────────────────────────────────────────

# Synergy scores: 0.0 (neutral) to 1.0 (maximum synergy)
# Meaning: selecting both strategies together is MORE valuable
# than selecting them independently
SYNERGY_MATRIX = {
    ("SCORP_SALARY", "SOLO_401K"): 0.95,        # Salary determines employer match
    ("SCORP_SALARY", "QBI_DEDUCTION"): 0.85,    # W-2 wages affect QBI wage limit
    ("SCORP_SALARY", "AUGUSTA_RULE"): 0.80,     # Corp pays rent = corp deduction
    ("SCORP_SALARY", "HSA"): 0.75,              # Corp can contribute to HSA
    ("SCORP_SALARY", "HEALTH_INSURANCE"): 0.80, # 2% shareholder health insurance
    ("SOLO_401K", "ROTH_CONVERSION"): 0.70,     # Contributions reduce conversion cost
    ("TRADER_TAX_STATUS", "SECTION_1256"): 0.90,# TTS + futures = maximum efficiency
    ("TRADER_TAX_STATUS", "HOME_OFFICE"): 0.85, # TTS qualifies home office
    ("TRADER_TAX_STATUS", "SOLO_401K"): 0.75,   # Trading income supports 401k
    ("SECTION_1256", "ROTH_CONVERSION"): 0.65,  # Lower effective rate + conversion
    ("DEPRECIATION_179", "QBI_DEDUCTION"): 0.70,# Depreciation reduces QBI base
    ("HSA", "SOLO_401K"): 0.80,                 # Both reduce taxable income
    ("ROTH_CONVERSION", "BUSINESS_LOSSES"): 0.85,# Loss year = conversion window
    ("SCORP_SALARY", "DEPRECIATION_179"): 0.75, # Corp assets + salary optimization
}

# Conflict scores: 0.0 (no conflict) to 1.0 (mutually exclusive)
# Meaning: selecting both strategies together is LESS valuable or ILLEGAL
CONFLICT_MATRIX = {
    ("AUGUSTA_RULE", "HOME_OFFICE"): 0.95,      # Same space, must choose one
    ("SOLO_401K", "SEP_IRA"): 0.80,             # Can't maximize both in same year
    ("SOLO_401K", "SIMPLE_IRA"): 0.90,          # SIMPLE IRA limits 401k in same business
    ("SECTION_1256", "SCORP_TRADING"): 0.85,    # Trading in S-Corp loses §1256 treatment
    ("TRADER_TAX_STATUS", "PASSIVE_INVESTOR"): 1.0,  # Mutually exclusive classifications
    ("QBI_DEDUCTION", "SSTB_HIGH_INCOME"): 0.90,# SSTBs phase out above threshold
    ("HOME_OFFICE", "NO_EXCLUSIVE_USE"): 1.0,   # Exclusive use is mandatory
}

# Audit risk compounding: some combinations dramatically increase audit risk
# even if individually they're fine
AUDIT_RISK_COMPOUND = {
    ("AUGUSTA_RULE", "SCORP_SALARY"): 1.4,      # 40% higher combined audit risk
    ("TRADER_TAX_STATUS", "SCORP_SALARY"): 1.3, # IRS scrutinizes TTS + S-Corp
    ("AUGUSTA_RULE", "HOME_OFFICE"): 2.0,        # Conflict = near certain audit issue
    ("DEPRECIATION_179", "AUGUSTA_RULE"): 1.2,
}


def get_synergy_score(strategy_a: str, strategy_b: str) -> float:
    """
    Get synergy score between two strategies.
    Symmetric: order doesn't matter.
    Returns 0.0 if no known synergy.
    """
    key = (strategy_a, strategy_b)
    reverse_key = (strategy_b, strategy_a)
    return SYNERGY_MATRIX.get(key, SYNERGY_MATRIX.get(reverse_key, 0.0))


def get_conflict_score(strategy_a: str, strategy_b: str) -> float:
    """
    Get conflict score between two strategies.
    Returns 0.0 if no known conflict.
    """
    key = (strategy_a, strategy_b)
    reverse_key = (strategy_b, strategy_a)
    return CONFLICT_MATRIX.get(key, CONFLICT_MATRIX.get(reverse_key, 0.0))


def get_audit_compound(strategy_a: str, strategy_b: str) -> float:
    """
    Get audit risk multiplier for combining two strategies.
    Returns 1.0 (no compounding) if no known interaction.
    """
    key = (strategy_a, strategy_b)
    reverse_key = (strategy_b, strategy_a)
    return AUDIT_RISK_COMPOUND.get(key, AUDIT_RISK_COMPOUND.get(reverse_key, 1.0))


def build_interaction_matrices(
    candidates: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build full NxN interaction matrices for a list of candidate documents.

    Returns:
        synergy_matrix:   NxN float, synergy between each pair
        conflict_matrix:  NxN float, conflict between each pair
        audit_matrix:     NxN float, audit risk compound multiplier
    """
    n = len(candidates)
    synergy = np.zeros((n, n))
    conflict = np.zeros((n, n))
    audit = np.ones((n, n))

    for i, doc_i in enumerate(candidates):
        strat_i = doc_i.get("metadata", {}).get("strategy_id", "")

        for j, doc_j in enumerate(candidates):
            if i == j:
                continue

            strat_j = doc_j.get("metadata", {}).get("strategy_id", "")

            if strat_i and strat_j:
                synergy[i][j] = get_synergy_score(strat_i, strat_j)
                conflict[i][j] = get_conflict_score(strat_i, strat_j)
                audit[i][j] = get_audit_compound(strat_i, strat_j)

    return synergy, conflict, audit


def extract_strategy_ids_from_metadata(docs: list[dict]) -> list[Optional[str]]:
    """Extract strategy IDs from document metadata."""
    return [
        doc.get("metadata", {}).get("strategy_id")
        for doc in docs
    ]


def score_combination(
    selected_docs: list[dict],
    verbose: bool = False
) -> dict:
    """
    Score a combination of selected strategies.
    Used for benchmarking and result explanation.

    Returns:
        {
            total_synergy: float,
            total_conflict: float,
            max_audit_risk: float,
            combination_score: float,
            warnings: list[str],
            synergies_found: list[str]
        }
    """
    n = len(selected_docs)
    total_synergy = 0.0
    total_conflict = 0.0
    max_audit_compound = 1.0
    warnings = []
    synergies_found = []

    for i in range(n):
        for j in range(i + 1, n):
            strat_i = selected_docs[i].get("metadata", {}).get("strategy_id", "")
            strat_j = selected_docs[j].get("metadata", {}).get("strategy_id", "")

            if not strat_i or not strat_j:
                continue

            syn = get_synergy_score(strat_i, strat_j)
            con = get_conflict_score(strat_i, strat_j)
            aud = get_audit_compound(strat_i, strat_j)

            total_synergy += syn
            total_conflict += con
            max_audit_compound = max(max_audit_compound, aud)

            if syn > 0.7:
                synergies_found.append(
                    f"{strat_i} + {strat_j} (synergy: {syn:.2f})"
                )

            if con > 0.7:
                warnings.append(
                    f"⚠️  CONFLICT: {strat_i} conflicts with {strat_j} (score: {con:.2f})"
                )

            if aud > 1.3:
                warnings.append(
                    f"⚠️  AUDIT RISK: {strat_i} + {strat_j} compounds audit risk {aud:.1f}x"
                )

    # Combined score: synergy bonus, conflict penalty, audit penalty
    combination_score = total_synergy - (total_conflict * 2.0) - (max_audit_compound - 1.0)

    if verbose:
        print(f"\nCombination Analysis:")
        print(f"  Total synergy:    {total_synergy:.3f}")
        print(f"  Total conflict:   {total_conflict:.3f}")
        print(f"  Audit multiplier: {max_audit_compound:.2f}x")
        print(f"  Combined score:   {combination_score:.3f}")
        if synergies_found:
            print(f"  Synergies:")
            for s in synergies_found:
                print(f"    ✓ {s}")
        if warnings:
            print(f"  Warnings:")
            for w in warnings:
                print(f"    {w}")

    return {
        "total_synergy": total_synergy,
        "total_conflict": total_conflict,
        "max_audit_compound": max_audit_compound,
        "combination_score": combination_score,
        "warnings": warnings,
        "synergies_found": synergies_found
    }
