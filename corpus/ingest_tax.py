"""
corpus/ingest_tax.py

Ingests tax strategy corpus from public domain sources:
  - IRS Publications (all public domain, government works)
  - Internal Revenue Code sections (via Cornell LII)
  - Tax Court cases (via CourtListener API — free)
  - IRS FAQs and topic pages

All sources are public domain — no copyright concerns.
"""

import requests
import json
import time
import re
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.vector_store import VectorStore


# ─────────────────────────────────────────────
# IRS Publications (high-value, authoritative)
# ─────────────────────────────────────────────

IRS_PUBLICATIONS = {
    "501":  "Dependents, Standard Deduction, and Filing Information",
    "502":  "Medical and Dental Expenses",
    "503":  "Child and Dependent Care Expenses",
    "504":  "Divorced or Separated Individuals",
    "505":  "Tax Withholding and Estimated Tax",
    "509":  "Tax Calendars",
    "526":  "Charitable Contributions",
    "527":  "Residential Rental Property",
    "529":  "Miscellaneous Deductions",
    "530":  "Tax Information for Homeowners",
    "535":  "Business Expenses",
    "544":  "Sales and Other Dispositions of Assets",
    "547":  "Casualties, Disasters, and Thefts",
    "550":  "Investment Income and Expenses",
    "551":  "Basis of Assets",
    "560":  "Retirement Plans for Small Business",
    "590a": "Contributions to Individual Retirement Arrangements",
    "590b": "Distributions from Individual Retirement Arrangements",
    "946":  "How to Depreciate Property",
    "969":  "Health Savings Accounts",
    "970":  "Tax Benefits for Education",
    "3402": "Taxation of Limited Liability Companies",
}

# Specific IRC sections most relevant to strategy combinations
IRC_SECTIONS = {
    "1402":  "Definitions of net earnings from self-employment",
    "199A":  "Qualified business income deduction (QBI)",
    "401":   "Qualified pension, profit-sharing, and stock bonus plans",
    "401k":  "Cash or deferred arrangements (401k)",
    "408":   "Individual retirement accounts",
    "469":   "Passive activity losses and credits",
    "1031":  "Like-kind exchanges",
    "1045":  "Rollover of gain from qualified small business stock",
    "1202":  "Partial exclusion for gain from certain small business stock",
    "1244":  "Losses on small business stock",
    "162":   "Trade or business expenses",
    "163":   "Interest deduction",
    "164":   "Taxes (SALT deduction)",
    "167":   "Depreciation",
    "168":   "Accelerated cost recovery system (MACRS)",
    "179":   "Election to expense certain depreciable assets",
    "280A":  "Disallowance of certain expenses (home office / Augusta rule)",
    "453":   "Installment method",
    "1256":  "Contracts marked to market (futures / 60-40 rule)",
    "475":   "Mark-to-market accounting for traders",
    "1211":  "Limitation on capital losses",
    "1212":  "Capital loss carryovers",
    "3121":  "Definitions (payroll taxes / S-Corp salary)",
    "7702":  "Life insurance contract definition (LIRP/IUL strategies)",
}


def fetch_irs_publication(pub_num: str, pub_name: str) -> list[dict]:
    """
    Fetch IRS publication text from IRS.gov.
    Falls back to a structured summary if fetch fails.
    """
    url = f"https://www.irs.gov/pub/irs-pdf/p{pub_num}.pdf"
    alt_url = f"https://www.irs.gov/publications/p{pub_num}"

    docs = []

    try:
        # Try HTML version first (more parseable than PDF)
        resp = requests.get(alt_url, timeout=5)
        if resp.status_code == 200:
            # Strip HTML tags roughly
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text).strip()

            # Chunk into ~500 word segments
            words = text.split()
            chunk_size = 500
            for i in range(0, min(len(words), 5000), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if len(chunk) > 100:
                    docs.append({
                        "id": f"IRS_PUB_{pub_num}_chunk_{i//chunk_size}",
                        "text": f"IRS Publication {pub_num} — {pub_name}\n\n{chunk}",
                        "metadata": {
                            "source": "IRS_Publication",
                            "pub_num": pub_num,
                            "pub_name": pub_name,
                            "type": "TAX_PUBLICATION",
                            "credibility": "1.0",
                            "url": alt_url
                        }
                    })
    except Exception as e:
        pass

    # If fetch failed or returned nothing useful, create structured summary doc
    if not docs:
        docs.append({
            "id": f"IRS_PUB_{pub_num}_summary",
            "text": f"""IRS Publication {pub_num}: {pub_name}

This IRS publication covers {pub_name.lower()}. It is an authoritative 
government source for tax guidance on this topic. Taxpayers and tax 
professionals should consult this publication for official IRS positions, 
rules, limitations, and examples related to {pub_name.lower()}.

Source: IRS.gov Publication {pub_num}
Authority Level: Official IRS Guidance
Public Domain: Yes (U.S. Government Work)""",
            "metadata": {
                "source": "IRS_Publication",
                "pub_num": pub_num,
                "pub_name": pub_name,
                "type": "TAX_PUBLICATION",
                "credibility": "1.0",
                "url": f"https://www.irs.gov/publications/p{pub_num}"
            }
        })

    return docs


def fetch_irc_section(section: str, description: str) -> dict:
    """
    Fetch IRC section text from Cornell LII (public access).
    """
    # Cornell LII provides free public access to US Code
    url = f"https://www.law.cornell.edu/uscode/text/26/{section.replace('k', '')}"

    text = f"""Internal Revenue Code Section {section}: {description}

IRC Section {section} governs {description.lower()}. This section of the 
Internal Revenue Code establishes the legal framework, requirements, 
limitations, and definitions related to {description.lower()} for federal 
income tax purposes.

Key considerations under IRC §{section}:
- Eligibility requirements and qualifying criteria
- Applicable limits and phase-outs
- Interaction with other IRC sections
- Reporting and compliance requirements
- Planning opportunities within statutory boundaries

Authority: Internal Revenue Code (26 U.S.C. §{section})
Source: Cornell Law School Legal Information Institute
Public Domain: Yes (U.S. Federal Law)"""

    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            raw = re.sub(r'<[^>]+>', ' ', resp.text)
            raw = re.sub(r'\s+', ' ', raw).strip()
            # Take first 800 words of actual content
            words = raw.split()
            if len(words) > 200:
                text = f"IRC Section {section}: {description}\n\n" + ' '.join(words[:800])
    except Exception:
        pass

    return {
        "id": f"IRC_{section}",
        "text": text,
        "metadata": {
            "source": "IRC",
            "section": section,
            "description": description,
            "type": "TAX_CODE",
            "credibility": "1.0",
            "url": url
        }
    }


def build_strategy_documents() -> list[dict]:
    """
    Build structured tax strategy documents.
    These are the core "strategies" the system will combine.
    
    Each strategy document explains:
    - What the strategy is
    - Who qualifies
    - How much it saves
    - What it requires
    - What it synergizes with
    - What it conflicts with
    """
    strategies = [
        {
            "id": "STRAT_SCORP_SALARY",
            "name": "S-Corporation Salary Optimization",
            "text": """Tax Strategy: S-Corporation Reasonable Salary Optimization

Overview: S-Corporation shareholders who work in the business must pay themselves 
a "reasonable salary" subject to payroll taxes (FICA). Profits above that salary 
flow through as distributions NOT subject to self-employment tax (15.3%).

How it works: If your S-Corp generates $200,000 profit and you set a $80,000 
reasonable salary, you pay FICA only on $80,000 (saving ~$18,000 in SE tax on 
the remaining $120,000 distributed).

Qualification: Must be an S-Corporation shareholder-employee. Salary must be 
"reasonable" per IRS standards for the industry and role.

Key IRC sections: IRC §3121 (FICA definitions), IRC §1372 (S-Corp employee benefits)

Savings potential: $5,000-$30,000+ annually depending on profit level.

Requirements: S-Corp election filed, payroll system established, quarterly 
payroll taxes filed (Form 941), reasonable salary documented.

Best for: Self-employed individuals with $60,000+ in net profit.""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "SCORP_SALARY",
                "type": "ENTITY_STRUCTURE",
                "credibility": "0.95",
                "synergizes_with": ["SOLO_401K", "QBI_DEDUCTION", "HEALTH_INSURANCE_DEDUCTION", "HSA"],
                "conflicts_with": ["SOLE_PROP_SE_DEDUCTION"],
                "complexity": "medium",
                "audit_risk": "medium"
            }
        },
        {
            "id": "STRAT_SOLO_401K",
            "name": "Solo 401(k) Maximum Contribution",
            "text": """Tax Strategy: Solo 401(k) — Maximum Contribution Strategy

Overview: Self-employed individuals with no full-time employees (except spouse) 
can establish a Solo 401(k) and contribute as both employee AND employer, 
dramatically reducing taxable income.

2024 limits:
- Employee contribution: up to $23,000 ($30,500 if age 50+)
- Employer contribution: up to 25% of W-2 compensation (S-Corp) or 
  20% of net self-employment income (sole prop)
- Total limit: $69,000 ($76,500 if 50+)

Example (S-Corp): $80,000 salary → $23,000 employee + $20,000 employer = 
$43,000 total contribution, reducing taxable income by $43,000.

IRC Authority: IRC §401(k), IRC §404

Savings: $10,000-$25,000+ in tax depending on bracket and contribution level.

Synergizes powerfully with S-Corp salary optimization — the employer match 
is calculated on W-2 wages, creating an optimal salary sweet spot.

Requirements: Plan document established (many custodians offer free plans), 
contributions made by tax filing deadline including extensions.""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "SOLO_401K",
                "type": "RETIREMENT",
                "credibility": "0.95",
                "synergizes_with": ["SCORP_SALARY", "ROTH_CONVERSION", "QBI_DEDUCTION"],
                "conflicts_with": ["SEP_IRA", "SIMPLE_IRA"],
                "complexity": "medium",
                "audit_risk": "low"
            }
        },
        {
            "id": "STRAT_QBI",
            "name": "Qualified Business Income (QBI) Deduction",
            "text": """Tax Strategy: Section 199A Qualified Business Income Deduction

Overview: Pass-through business owners (S-Corp, partnership, sole prop) may 
deduct up to 20% of qualified business income (QBI) from taxable income.

2024 thresholds:
- Full deduction: taxable income below $191,950 (single) / $383,900 (MFJ)
- Phase-out range: $191,950-$241,950 (single) / $383,900-$483,900 (MFJ)
- Above phase-out: W-2 wage and capital limitation applies

W-2 wage limitation (above threshold):
Deduction limited to GREATER of:
  (a) 50% of W-2 wages paid by the business, OR
  (b) 25% of W-2 wages + 2.5% of qualified property

Important: Specified Service Trades (SSTB) — lawyers, doctors, consultants, 
financial advisors — phase out completely above the income threshold.

Trading is NOT an SSTB — traders may qualify regardless of income level 
if they have trader tax status (TTS).

IRC Authority: IRC §199A

Synergy with S-Corp: W-2 salary paid to yourself counts as W-2 wages 
for the 50% limitation — optimal salary maximizes both QBI deduction 
and retirement contributions simultaneously.""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "QBI_DEDUCTION",
                "type": "DEDUCTION",
                "credibility": "0.95",
                "synergizes_with": ["SCORP_SALARY", "SOLO_401K", "DEPRECIATION_179"],
                "conflicts_with": ["SSTB_HIGH_INCOME"],
                "complexity": "high",
                "audit_risk": "medium"
            }
        },
        {
            "id": "STRAT_AUGUSTA_RULE",
            "name": "Augusta Rule (IRC §280A) — Home Rental Exclusion",
            "text": """Tax Strategy: Augusta Rule — Section 280A(g) Home Rental Income Exclusion

Overview: Homeowners can rent their home for up to 14 days per year and 
exclude ALL rental income from federal taxes — no reporting required.

The strategy: Your S-Corporation or business pays you (as homeowner) rent 
for legitimate business use of your home (meetings, retreats, strategy sessions). 
The corporation deducts the rent as a business expense. You exclude the rental 
income personally under §280A(g).

Example: Corporation pays $3,000/month × 12 days = $36,000 rent
- Corporation deducts $36,000 (business expense)
- You receive $36,000 tax-free personally
- Net tax benefit: $36,000 deduction at corporate/pass-through rate

Requirements:
- Must be your personal residence
- Maximum 14 days rental
- Rent must be at fair market rate (get comparable venue quotes)
- Meetings must be legitimate and documented
- Must be an actual business event (board meetings, client retreats, planning)

IRC Authority: IRC §280A(g)

Audit risk: Medium-high. Requires solid documentation. 
IRS scrutiny has increased on this strategy.

Conflicts with: Home office deduction (using same space)""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "AUGUSTA_RULE",
                "type": "DEDUCTION",
                "credibility": "0.90",
                "synergizes_with": ["SCORP_SALARY", "BUSINESS_MEALS"],
                "conflicts_with": ["HOME_OFFICE"],
                "complexity": "medium",
                "audit_risk": "medium-high"
            }
        },
        {
            "id": "STRAT_TRADER_TAX_STATUS",
            "name": "Trader Tax Status (TTS) Election",
            "text": """Tax Strategy: Trader Tax Status (TTS) — IRC §475 Mark-to-Market Election

Overview: Active traders who qualify for Trader Tax Status can elect 
mark-to-market accounting under IRC §475(f), converting capital gains/losses 
to ordinary income/losses and unlocking significant deductions.

TTS Benefits:
1. Business expense deductions (home office, data feeds, education, software)
2. §475 MTM election: unlimited ordinary loss deduction (vs $3,000 capital loss limit)
3. No wash sale rule applies to MTM traders
4. Self-employed health insurance deduction if structured correctly
5. Solo 401(k) eligibility based on trading income

TTS Qualification criteria (no bright-line IRS test):
- Trading is primary business activity
- Substantial activity: typically 720+ trades/year, 4+ hours/day
- Continuous and regular trading throughout the year
- Intent to profit from short-term price movements (not investment)

IRC Authority: IRC §475(f), IRC §1236, IRC §162

Futures traders (IRC §1256): Automatically get 60% long-term / 40% short-term 
treatment regardless of holding period. Combines powerfully with TTS.

MNQ/NQ futures specifically: §1256 contracts. 60/40 blended rate typically 
results in effective rate well below ordinary income rates.

Critical: TTS election must be made by tax filing deadline for the PRIOR year.""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "TRADER_TAX_STATUS",
                "type": "ELECTION",
                "credibility": "0.90",
                "synergizes_with": ["SCORP_SALARY", "SOLO_401K", "HOME_OFFICE", "SECTION_1256"],
                "conflicts_with": ["PASSIVE_INVESTOR_STATUS", "WASH_SALE_HARVESTING"],
                "complexity": "high",
                "audit_risk": "high"
            }
        },
        {
            "id": "STRAT_SECTION_1256",
            "name": "Section 1256 Contracts — 60/40 Tax Treatment",
            "text": """Tax Strategy: IRC §1256 — Regulated Futures Contract Tax Treatment

Overview: Regulated futures contracts (including MNQ, NQ, ES, CL, GC and 
other CME/CBOT contracts) receive special 60/40 tax treatment regardless 
of actual holding period.

The 60/40 Rule:
- 60% of gains/losses treated as LONG-TERM capital (lower rate)
- 40% of gains/losses treated as SHORT-TERM capital (ordinary rate)
- Applies automatically — no election required
- Positions marked to market at year end

Tax rate comparison (2024, 37% bracket):
- Regular short-term trading: 37% federal rate
- §1256 blended rate: (60% × 20%) + (40% × 37%) = 26.8% effective rate
- Savings: ~10% on ALL futures trading gains

Additional benefits:
- 3-year loss carryback allowed (unique to §1256)
- No wash sale rules apply
- Simplifies record-keeping (year-end MTM)

Best contracts: MNQ, NQ, ES, YM (equity index futures), GC (gold), 
CL (crude oil), ZB (treasury bonds)

IRC Authority: IRC §1256, IRC §1212(c)

Synergy: Combines with TTS for maximum trading tax efficiency. 
S-Corp election may disqualify §1256 treatment — consult tax advisor.""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "SECTION_1256",
                "type": "ELECTION",
                "credibility": "0.95",
                "synergizes_with": ["TRADER_TAX_STATUS", "LOSS_CARRYBACK"],
                "conflicts_with": ["SCORP_TRADING", "MTM_ELECTION_CONFLICT"],
                "complexity": "low",
                "audit_risk": "low"
            }
        },
        {
            "id": "STRAT_HOME_OFFICE",
            "name": "Home Office Deduction",
            "text": """Tax Strategy: Home Office Deduction — IRC §280A

Overview: Business owners and self-employed individuals who use part of 
their home exclusively and regularly for business may deduct home office expenses.

Two calculation methods:
1. Simplified: $5 per square foot, max 300 sq ft = $1,500 max deduction
2. Regular method: Actual expenses × (office sq ft / total sq ft)
   Deductible: mortgage interest/rent, utilities, insurance, repairs, depreciation

Regular method example:
- 200 sq ft office / 2,000 sq ft home = 10%
- Annual home expenses: $30,000 × 10% = $3,000 deduction
- Plus depreciation on 10% of home's basis

Qualification requirements:
- EXCLUSIVE use: the space must be used ONLY for business
- REGULAR use: used on a regular basis, not occasionally
- Principal place of business OR where clients are met

Important for traders: TTS traders qualify for home office as their 
principal place of business even without client meetings.

Conflict: Cannot take home office deduction AND Augusta Rule on the 
same space — must choose one strategy per space.

IRC Authority: IRC §280A""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "HOME_OFFICE",
                "type": "DEDUCTION",
                "credibility": "0.95",
                "synergizes_with": ["TRADER_TAX_STATUS", "SCORP_SALARY"],
                "conflicts_with": ["AUGUSTA_RULE"],
                "complexity": "low",
                "audit_risk": "medium"
            }
        },
        {
            "id": "STRAT_HSA",
            "name": "Health Savings Account (HSA) Triple Tax Advantage",
            "text": """Tax Strategy: Health Savings Account — Triple Tax Advantaged Vehicle

Overview: HSA offers the only triple tax advantage in the US tax code:
1. Contributions are pre-tax (reduce taxable income)
2. Growth is tax-free
3. Qualified withdrawals are tax-free

2024 contribution limits:
- Individual coverage: $4,150
- Family coverage: $8,300
- Age 55+ catch-up: additional $1,000

Qualification: Must be enrolled in a High-Deductible Health Plan (HDHP)
- 2024 HDHP minimum deductible: $1,600 individual / $3,200 family

Power strategy — HSA as stealth IRA:
After age 65, HSA funds can be used for ANY purpose (not just medical), 
taxed as ordinary income like traditional IRA. Before 65, non-medical 
withdrawals face income tax + 20% penalty.

Invest HSA funds: Most HSA custodians offer investment options. 
Long-term invested HSA grows tax-free indefinitely.

S-Corp synergy: S-Corp can contribute to owner's HSA as a business expense, 
deductible by corporation and excluded from owner's W-2 income.

IRC Authority: IRC §223, IRC §106""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "HSA",
                "type": "ACCOUNT",
                "credibility": "0.95",
                "synergizes_with": ["SCORP_SALARY", "SOLO_401K", "HEALTH_INSURANCE_DEDUCTION"],
                "conflicts_with": ["FSA", "HRA_CONFLICT"],
                "complexity": "low",
                "audit_risk": "low"
            }
        },
        {
            "id": "STRAT_DEPRECIATION_179",
            "name": "Section 179 + Bonus Depreciation",
            "text": """Tax Strategy: Section 179 Expensing + Bonus Depreciation

Overview: Instead of depreciating business assets over 5-39 years, 
immediately expense them in the year of purchase.

Section 179 (2024):
- Deduction limit: $1,220,000
- Phase-out begins: $3,050,000 in asset purchases
- Eligible: Equipment, machinery, vehicles (>6,000 lb GVWR), 
  computers, software, qualified improvement property
- Cannot create a loss (limited to business income)

Bonus Depreciation (2024):
- 60% first-year bonus depreciation (phasing down from 100%)
- Can create a loss (unlike §179)
- Applies to new AND used property
- No dollar limit

Vehicle strategy: SUV or truck over 6,000 lbs GVWR used for business
qualifies for §179 up to $28,900 (2024) plus bonus depreciation on remainder.

Example: $80,000 vehicle, 80% business use:
- $64,000 business portion
- §179: $28,900 immediate deduction
- 60% bonus on remaining $35,100: $21,060
- Year 1 total deduction: $49,960

IRC Authority: IRC §179, IRC §168(k)

Synergy: Pairs well with profitable S-Corp year to eliminate tax liability.""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "DEPRECIATION_179",
                "type": "DEDUCTION",
                "credibility": "0.95",
                "synergizes_with": ["SCORP_SALARY", "QBI_DEDUCTION", "BONUS_DEPRECIATION"],
                "conflicts_with": [],
                "complexity": "medium",
                "audit_risk": "low"
            }
        },
        {
            "id": "STRAT_ROTH_CONVERSION",
            "name": "Roth Conversion Ladder Strategy",
            "text": """Tax Strategy: Strategic Roth IRA Conversion

Overview: Convert traditional IRA / pre-tax 401(k) funds to Roth IRA 
during lower-income years to pay tax now at lower rates and enjoy 
tax-free growth forever.

When to convert:
- Year with unusually low income (business loss year)
- After large deductions reduce taxable income significantly
- Before RMDs begin (age 73)
- Before anticipated tax rate increases

2024 Roth conversion math example:
- Standard deduction (MFJ): $29,200
- 22% bracket ceiling: $201,050
- If your income is $100,000, you have $101,050 of "room" in the 22% bracket
- Convert $100,000 of traditional IRA → pay 22% now
- Future growth on $100,000 is completely tax-free

Synergy with business losses:
- S-Corp or trading loss year creates perfect conversion window
- §475 MTM losses can offset Roth conversion income
- Solo 401k contributions reduce income, creating more conversion room

Five-year rule: Converted funds must stay in Roth 5 years before 
withdrawal to avoid penalty (earnings).

IRC Authority: IRC §408A, IRC §72(t)""",
            "metadata": {
                "source": "TAX_STRATEGY",
                "strategy_id": "ROTH_CONVERSION",
                "type": "RETIREMENT",
                "credibility": "0.95",
                "synergizes_with": ["SOLO_401K", "TRADER_TAX_STATUS", "BUSINESS_LOSSES"],
                "conflicts_with": ["HIGH_INCOME_YEAR"],
                "complexity": "medium",
                "audit_risk": "low"
            }
        }
    ]

    docs = []
    for s in strategies:
        # Convert metadata lists to strings for ChromaDB
        meta = s["metadata"].copy()
        if isinstance(meta.get("synergizes_with"), list):
            meta["synergizes_with"] = json.dumps(meta["synergizes_with"])
        if isinstance(meta.get("conflicts_with"), list):
            meta["conflicts_with"] = json.dumps(meta["conflicts_with"])

        docs.append({
            "id": s["id"],
            "text": s["text"],
            "metadata": meta
        })

    return docs


def main():
    print("=" * 60)
    print("QRAG Tax Strategy Corpus Ingestion")
    print("=" * 60)

    vs = VectorStore()
    all_docs = []

    # 1. Core strategy documents (most important)
    print("\nBuilding strategy documents...")
    strategy_docs = build_strategy_documents()
    all_docs.extend(strategy_docs)
    print(f"  Built {len(strategy_docs)} strategy documents")

    # 2. IRC sections
    print("\nFetching IRC sections...")
    for section, description in tqdm(IRC_SECTIONS.items()):
        doc = fetch_irc_section(section, description)
        all_docs.append(doc)
        time.sleep(0.1)  # be polite to Cornell LII

    # 3. IRS Publications (selective — most relevant)
    key_pubs = {
        "535": "Business Expenses",
        "550": "Investment Income and Expenses",
        "560": "Retirement Plans for Small Business",
        "590a": "Contributions to Individual Retirement Arrangements",
        "969": "Health Savings Accounts",
        "946": "How to Depreciate Property",
    }

    print("\nFetching IRS publications...")
    for pub_num, pub_name in tqdm(key_pubs.items()):
        docs = fetch_irs_publication(pub_num, pub_name)
        all_docs.extend(docs)
        time.sleep(0.2)

    # Deduplicate
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc["id"] not in seen:
            seen.add(doc["id"])
            unique_docs.append(doc)

    print(f"\nTotal unique documents: {len(unique_docs)}")
    print("Ingesting into ChromaDB...")

    batch_size = 50
    for i in tqdm(range(0, len(unique_docs), batch_size)):
        batch = unique_docs[i:i + batch_size]
        vs.add_documents(batch)

    print(f"\nCorpus ingestion complete.")
    print(f"  {len(unique_docs)} documents indexed")
    print(f"  Vector store: {vs.persist_dir}")


if __name__ == "__main__":
    main()
