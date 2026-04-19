"""
generation/llm_tax.py

Claude API wrapper for tax strategy RAG generation.
"""

import anthropic
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANTHROPIC_API_KEY, LLM_MODEL
from quantum.interactions import score_combination

# If ANTHROPIC_API_KEY is unset, instantiate without an explicit key so the SDK
# falls back to whatever credentials the runtime has. If neither is present,
# client construction still succeeds; calls will raise at request time and we
# catch that in generate_tax() to return a stub.
if ANTHROPIC_API_KEY:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    try:
        client = anthropic.Anthropic()
    except Exception:
        client = None

TAX_SYSTEM_PROMPT = """You are an expert tax strategist with deep knowledge of the US Tax Code, 
IRS publications, and advanced tax planning strategies. You analyze combinations of tax strategies 
and explain how they work together, their synergies, conflicts, and combined impact.

CRITICAL DISCLAIMER: This is for educational and research purposes only. Nothing here 
constitutes tax advice. Always consult a licensed CPA or tax attorney before implementing 
any tax strategy.

When answering:
1. Explain how the retrieved strategies work together as a COMBINATION
2. Identify synergies — where one strategy amplifies another
3. Flag any conflicts or ordering requirements
4. Reference IRC sections and IRS publications when relevant
5. Be specific about qualifications and limitations
6. Note audit risk considerations honestly"""


def generate_tax(query: str, context_docs: list[dict], analyze_interactions: bool = True) -> dict:
    """
    Generate tax strategy answer using context documents.
    Includes interaction analysis in the response.
    """
    # Analyze the combination quality
    combo_analysis = score_combination(context_docs) if analyze_interactions else {}

    # Build context
    context_parts = []
    for i, doc in enumerate(context_docs):
        source = doc.get("metadata", {}).get("source", "")
        strat_id = doc.get("metadata", {}).get("strategy_id", "")
        label = strat_id or source or f"Document {i+1}"
        context_parts.append(f"[{label}]\n{doc['text']}")

    context_str = "\n\n---\n\n".join(context_parts)

    # Add interaction context to prompt if available
    interaction_note = ""
    if combo_analysis.get("synergies_found"):
        syns = "\n".join(f"- {s}" for s in combo_analysis["synergies_found"])
        interaction_note = f"\n\nKnown strategy synergies in this combination:\n{syns}"

    if combo_analysis.get("warnings"):
        warns = "\n".join(combo_analysis["warnings"])
        interaction_note += f"\n\nWarnings for this combination:\n{warns}"

    user_message = f"""Tax Strategy Context Documents:
{context_str}
{interaction_note}

---

Question: {query}

Analyze these strategies as a combination and explain:
1. How they work together for this situation
2. The synergies between them
3. Any conflicts or important ordering
4. Estimated combined tax impact (directional, not specific dollar amounts)
5. Key implementation requirements"""

    try:
        if client is None:
            raise RuntimeError("Anthropic client unavailable (no API key in env)")
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1500,
            system=TAX_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        return {
            "answer": response.content[0].text,
            "docs_used": len(context_docs),
            "model": LLM_MODEL,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "combination_analysis": combo_analysis
        }
    except Exception as e:
        # Stub response when the LLM isn't reachable — preserves benchmark
        # flow. Metrics scored by compare_tax.py don't depend on answer text.
        labels = [
            d.get("metadata", {}).get("strategy_id") or d.get("id", "?")
            for d in context_docs
        ]
        stub = (
            "[LLM stub — no API key available]\n"
            f"Selected strategies: {', '.join(labels)}\n"
            f"Combination score: {combo_analysis.get('combination_score', 0):.3f}\n"
            f"LLM error: {type(e).__name__}: {e}"
        )
        return {
            "answer": stub,
            "docs_used": len(context_docs),
            "model": "stub",
            "input_tokens": 0,
            "output_tokens": 0,
            "combination_analysis": combo_analysis
        }
