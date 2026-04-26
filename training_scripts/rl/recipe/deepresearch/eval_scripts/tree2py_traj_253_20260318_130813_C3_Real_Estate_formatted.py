import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "reit_2025_apex"
TASK_DESCRIPTION = """
Apex Properties Trust is a corporation formed in Delaware in January 2024 that elected REIT status for its first taxable year. The company is now in its second taxable year (2025) and must verify it continues to qualify as a Real Estate Investment Trust under federal tax law. For the 2025 tax year, Apex Properties Trust has the following characteristics: **Organizational Structure:** Formed as a Delaware corporation; managed by a board of 7 directors; common stock is freely transferable on a national exchange. **Ownership (as of December 31, 2025):** Total of 425 individual shareholders; the five largest individual shareholders collectively own 48% of the outstanding stock value; no single family group owns more than 15%. **Annual Income (2025 tax year):** Rental income from office buildings: $18,750,000; interest from mortgages on commercial properties: $4,250,000; dividend income from publicly traded stocks: $1,500,000; interest from U.S. Treasury bonds: $500,000; service fee income (non-customary services): $1,000,000; total gross income: $26,000,000. **Assets (as of Q4 2025 quarter-end):** Fair market value of owned office buildings: $180,000,000; fair market value of owned retail properties: $90,000,000; cash and cash equivalents: $15,000,000; U.S. government securities: $10,000,000; publicly traded REIT shares: $5,000,000; stock in taxable REIT subsidiary: $40,000,000; corporate bonds from three different issuers: $10,000,000 (Issuer A: $6,000,000, Issuer B: $2,500,000, Issuer C: $1,500,000); for Issuer A, Apex holds bonds worth $6,000,000, representing 8% of the total value of Issuer A's outstanding securities and 7% of voting power; total assets: $350,000,000. **Distributions:** Taxable income for 2025: $22,000,000; cash dividends paid to shareholders during 2025: $20,500,000. Based on the statutory requirements for Real Estate Investment Trust qualification under the Internal Revenue Code, does Apex Properties Trust qualify as a REIT for the 2025 tax year? Your answer must verify compliance with all organizational, ownership, income, asset, and distribution requirements, citing specific percentages and thresholds from the applicable regulations.
"""

# Reference URLs mentioned in the rubric
REIT_HOW_TO_FORM_URL = "https://www.reit.com/what-reit/how-form-reit"
RSM_INCOME_TESTS_URL = "https://rsmus.com/insights/industries/real-estate/navigating-reit-income-tests.html"
RSM_ASSET_TESTS_URL = "https://rsmus.com/insights/industries/real-estate/navigating-the-reit-asset-tests.html"

# --------------------------------------------------------------------------- #
# Scenario numbers (from the task description)                                #
# --------------------------------------------------------------------------- #
RENTAL_INCOME = 18_750_000
MORTGAGE_INTEREST = 4_250_000
DIVIDENDS = 1_500_000
TREASURY_INTEREST = 500_000
SERVICE_FEES = 1_000_000
TOTAL_GROSS_INCOME = 26_000_000

OFFICE_REAL_ESTATE = 180_000_000
RETAIL_REAL_ESTATE = 90_000_000
CASH_EQUIV = 15_000_000
US_GOV_SECURITIES = 10_000_000
REIT_SHARES = 5_000_000
TRS_STOCK = 40_000_000
CORP_BONDS_TOTAL = 10_000_000
ISSUER_A_BONDS = 6_000_000
ISSUER_B_BONDS = 2_500_000
ISSUER_C_BONDS = 1_500_000
TOTAL_ASSETS = 350_000_000

TAXABLE_INCOME = 22_000_000
DIVIDENDS_PAID = 20_500_000

SH_COUNT_YEAR_END = 425
TOP5_OWNERS_PCT = 48.0

# Derived calculations
Q75_INCOME = RENTAL_INCOME + MORTGAGE_INTEREST
Q75_PCT = Q75_INCOME / TOTAL_GROSS_INCOME if TOTAL_GROSS_INCOME else 0.0

Q95_INCOME = Q75_INCOME + DIVIDENDS + TREASURY_INTEREST
Q95_PCT = Q95_INCOME / TOTAL_GROSS_INCOME if TOTAL_GROSS_INCOME else 0.0

ASSETS_75_CLASS = OFFICE_REAL_ESTATE + RETAIL_REAL_ESTATE + CASH_EQUIV + US_GOV_SECURITIES + REIT_SHARES
ASSETS_75_PCT = ASSETS_75_CLASS / TOTAL_ASSETS if TOTAL_ASSETS else 0.0

SECURITIES_OVERALL = TRS_STOCK + CORP_BONDS_TOTAL + REIT_SHARES + US_GOV_SECURITIES
SECURITIES_OVERALL_PCT = SECURITIES_OVERALL / TOTAL_ASSETS if TOTAL_ASSETS else 0.0
TRS_PCT = TRS_STOCK / TOTAL_ASSETS if TOTAL_ASSETS else 0.0

ISSUER_A_ASSET_PCT = ISSUER_A_BONDS / TOTAL_ASSETS if TOTAL_ASSETS else 0.0
ISSUER_B_ASSET_PCT = ISSUER_B_BONDS / TOTAL_ASSETS if TOTAL_ASSETS else 0.0
ISSUER_C_ASSET_PCT = ISSUER_C_BONDS / TOTAL_ASSETS if TOTAL_ASSETS else 0.0

DISTRIBUTION_PCT = DIVIDENDS_PAID / TAXABLE_INCOME if TAXABLE_INCOME else 0.0

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConclusionExtraction(BaseModel):
    """Extract the final conclusion (qualifies or not) and any URLs cited."""
    qualifies_as_reit_2025: Optional[bool] = None
    conclusion_text: Optional[str] = None
    cited_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_conclusion() -> str:
    return """
    From the answer, extract:
    1) qualifies_as_reit_2025: true if the answer concludes Apex Properties Trust qualifies as a REIT for the 2025 tax year; false if it concludes it does not qualify; null if unclear.
    2) conclusion_text: the sentence or short passage stating the final qualification conclusion.
    3) cited_urls: a list of any URLs the answer cites (if any).
    """


# --------------------------------------------------------------------------- #
# Helper formatting                                                           #
# --------------------------------------------------------------------------- #
def pct_str(x: float) -> str:
    return f"{x * 100:.2f}%"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_structural_requirements(evaluator: Evaluator, parent) -> None:
    # Structural Requirements (sequential, critical)
    structural_node = evaluator.add_sequential(
        id="Structural_Requirements",
        desc="Verify organizational structure and ownership requirements, with reference to https://www.reit.com/what-reit/how-form-reit",
        parent=parent,
        critical=True
    )

    # Organizational Formation (parallel, critical)
    org_node = evaluator.add_parallel(
        id="Organizational_Formation",
        desc="Confirm the entity is formed in a U.S. state or D.C., is taxable as a corporation, is governed by directors or trustees, and has transferable shares",
        parent=structural_node,
        critical=True
    )

    # Normative reference: organizational requirements (by URL)
    org_rule_leaf = evaluator.add_leaf(
        id="Org_Rules_Reference_REITCOM",
        desc="REIT organizational rules include management by a board of directors or trustees and transferable shares",
        parent=org_node,
        critical=True
    )
    await evaluator.verify(
        claim="To qualify as a REIT, an entity must be managed by a board of directors or trustees and have shares that are fully transferable.",
        node=org_rule_leaf,
        sources=REIT_HOW_TO_FORM_URL,
        additional_instruction="Find the section that lists organizational formation requirements (board/trustees and share transferability)."
    )

    # Fact checks from the scenario (simple verifications)
    formed_leaf = evaluator.add_leaf(
        id="Formed_in_US_or_DC",
        desc="Formed in a U.S. state (Delaware) and is a corporation",
        parent=org_node,
        critical=True
    )
    await evaluator.verify(
        claim="Apex Properties Trust was formed as a Delaware corporation, which means it is formed in a U.S. state and is taxable as a corporation.",
        node=formed_leaf
    )

    directors_leaf = evaluator.add_leaf(
        id="Governed_by_Directors",
        desc="Governed by directors or trustees",
        parent=org_node,
        critical=True
    )
    await evaluator.verify(
        claim="Apex is managed by a board of 7 directors, satisfying the requirement to be governed by directors or trustees.",
        node=directors_leaf
    )

    transferable_leaf = evaluator.add_leaf(
        id="Transferable_Shares",
        desc="Shares are freely transferable",
        parent=org_node,
        critical=True
    )
    await evaluator.verify(
        claim="Apex's common stock is freely transferable on a national exchange; therefore, its shares are transferable.",
        node=transferable_leaf
    )

    # Ownership Requirements (parallel, critical)
    ownership_node = evaluator.add_parallel(
        id="Ownership_Requirements",
        desc="For the second taxable year onward, verify shareholder requirements are met",
        parent=structural_node,
        critical=True
    )

    # 100 Shareholder Test (single leaf; crafted to acknowledge 335-day rule)
    sh100_leaf = evaluator.add_leaf(
        id="100_Shareholder_Test",
        desc="At least 100 shareholders for at least 335 days of the year",
        parent=ownership_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "For the 2025 tax year, Apex had at least 100 shareholders for at least 335 days. "
            "At year-end there were 425 shareholders, and no contrary information is provided; "
            "therefore it satisfies the 100-shareholder test."
        ),
        node=sh100_leaf,
        additional_instruction=(
            "Base your judgment on the scenario provided. When the year-end shareholder count "
            "is far above 100 and no facts indicate prolonged periods below 100, consider the "
            "335-day requirement satisfied."
        )
    )

    # 5/50 Test
    five_fifty_leaf = evaluator.add_leaf(
        id="Five_Fifty_Ownership_Test",
        desc="During the last half of the year, five or fewer individuals do not own more than 50% of the REIT's stock value",
        parent=ownership_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "During the last half of 2025, five or fewer individuals did not own more than 50% of the value "
            "of Apex's stock, because the five largest individual shareholders collectively owned 48% (<50%)."
        ),
        node=five_fifty_leaf
    )


async def build_financial_tests(evaluator: Evaluator, parent) -> None:
    # Financial Tests (sequential, critical)
    financial_node = evaluator.add_sequential(
        id="Financial_Tests",
        desc="Verify compliance with annual income tests and quarterly asset tests",
        parent=parent,
        critical=True
    )

    # --------------------- Income Tests --------------------- #
    income_node = evaluator.add_sequential(
        id="Income_Tests",
        desc="Verify both annual gross income tests (75% and 95%)",
        parent=financial_node,
        critical=True
    )

    # 75% Real Estate Income (parallel, critical)
    income75_node = evaluator.add_parallel(
        id="Income_75pct",
        desc="At least 75% of gross income from qualified real estate sources",
        parent=income_node,
        critical=True
    )

    # Normative: which sources count toward 75%
    re_sources_leaf = evaluator.add_leaf(
        id="Real_Estate_Income_Source_Qualification",
        desc="Rents from real property and interest on mortgages on real property qualify for the 75% income test",
        parent=income75_node,
        critical=True
    )
    await evaluator.verify(
        claim="Rents from real property and interest on mortgages on real property are qualifying income for the 75% REIT gross income test.",
        node=re_sources_leaf,
        sources=RSM_INCOME_TESTS_URL,
        additional_instruction="Locate where the 75% income test sources are listed."
    )

    # Calculation: 75% threshold
    calc_75_leaf = evaluator.add_leaf(
        id="Income_75pct_Threshold_Calc",
        desc="Qualified real estate income >= 75% of total gross income",
        parent=income75_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"For 2025, qualified real estate income is $23,000,000 "
            f"(rents ${RENTAL_INCOME:,} + mortgage interest ${MORTGAGE_INTEREST:,}). "
            f"Total gross income is ${TOTAL_GROSS_INCOME:,}. "
            f"Thus {Q75_INCOME:,} / {TOTAL_GROSS_INCOME:,} ≈ {pct_str(Q75_PCT)} ≥ 75%."
        ),
        node=calc_75_leaf,
        additional_instruction="Only count rents from real property and mortgage interest for the 75% test; exclude service fees, ordinary corporate dividends, and Treasury interest."
    )

    # Personal property limitation (<=15% of rent attributable to personal property)
    pp_limit_leaf = evaluator.add_leaf(
        id="Personal_Property_Limitation",
        desc="Rent attributable to personal property leased with real property is < 15% (if any)",
        parent=income75_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "There is no indication that any personal property is leased with the real property such that more than 15% "
            "of rent would be attributable to personal property; rents are from office buildings, so the 15% limitation is satisfied."
        ),
        node=pp_limit_leaf
    )

    # 95% Combined Income (parallel, critical)
    income95_node = evaluator.add_parallel(
        id="Income_95pct",
        desc="At least 95% of gross income from real estate sources plus portfolio income",
        parent=income_node,
        critical=True
    )

    # Normative: which sources count toward 95%
    combined_sources_leaf = evaluator.add_leaf(
        id="Combined_Income_Source_Qualification",
        desc="For the 95% test, qualifying income includes 75% sources plus portfolio income (interest and dividends)",
        parent=income95_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the REIT 95% income test, qualifying income includes the 75% sources plus portfolio income such as interest and dividends.",
        node=combined_sources_leaf,
        sources=RSM_INCOME_TESTS_URL,
        additional_instruction="Locate where the 95% income test sources are listed (interest and dividends included)."
    )

    # Calculation: 95% threshold
    calc_95_leaf = evaluator.add_leaf(
        id="Income_95pct_Threshold_Calc",
        desc="Combined qualified income (real estate + portfolio) >= 95% of total gross income",
        parent=income95_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"For 2025, income qualifying for the 95% test equals $25,000,000 "
            f"(${Q75_INCOME:,} from the 75% category + dividends ${DIVIDENDS:,} + Treasury interest ${TREASURY_INTEREST:,}). "
            f"Total gross income is ${TOTAL_GROSS_INCOME:,}. "
            f"Thus {Q95_INCOME:,} / {TOTAL_GROSS_INCOME:,} ≈ {pct_str(Q95_PCT)} ≥ 95%."
        ),
        node=calc_95_leaf,
        additional_instruction="Exclude non-customary service fees from 95% qualifying income."
    )

    # --------------------- Asset Tests --------------------- #
    asset_node = evaluator.add_sequential(
        id="Asset_Tests",
        desc="Verify quarterly asset composition tests (75% assets and securities limitations)",
        parent=financial_node,
        critical=True
    )

    # 75% Assets (parallel, critical)
    assets75_node = evaluator.add_parallel(
        id="Assets_75pct",
        desc="At least 75% of total assets consist of real estate assets, cash and cash items, and government securities",
        parent=asset_node,
        critical=True
    )

    # Normative: qualifying 75% asset classes
    asset_classes_leaf = evaluator.add_leaf(
        id="Asset_Classes_Qualification",
        desc="Real estate assets, cash and cash items, and U.S. government securities count toward the 75% asset test",
        parent=assets75_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the REIT 75% asset test, qualifying assets include real estate assets, cash and cash items (including receivables), and U.S. government securities.",
        node=asset_classes_leaf,
        sources=RSM_ASSET_TESTS_URL,
        additional_instruction="Find where the 75% asset classes are listed."
    )

    # Calculation: 75% assets threshold
    assets75_calc_leaf = evaluator.add_leaf(
        id="Assets_75pct_Threshold_Calc",
        desc="Qualifying assets for 75% asset test are >= 75% of total assets",
        parent=assets75_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"As of Q4 2025, qualifying 75% asset classes total ${ASSETS_75_CLASS:,} "
            f"(real estate ${OFFICE_REAL_ESTATE + RETAIL_REAL_ESTATE:,} + cash ${CASH_EQUIV:,} + U.S. gov't securities ${US_GOV_SECURITIES:,} + REIT shares ${REIT_SHARES:,}) "
            f"out of total assets ${TOTAL_ASSETS:,}, i.e., {pct_str(ASSETS_75_PCT)} ≥ 75%."
        ),
        node=assets75_calc_leaf
    )

    # Securities limitations (parallel, critical)
    secs_limit_node = evaluator.add_parallel(
        id="Securities_Limitations",
        desc="Compliance with overall securities, TRS securities, and single-issuer limitations",
        parent=asset_node,
        critical=True
    )

    # Overall securities not more than 25%
    overall_secs_leaf = evaluator.add_leaf(
        id="Overall_Securities_Leq25",
        desc="Not more than 25% of total assets represented by securities overall",
        parent=secs_limit_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"Securities overall are ${SECURITIES_OVERALL:,} "
            f"(TRS stock ${TRS_STOCK:,} + corporate bonds ${CORP_BONDS_TOTAL:,} + REIT shares ${REIT_SHARES:,} + U.S. gov't securities ${US_GOV_SECURITIES:,}) "
            f"which is {pct_str(SECURITIES_OVERALL_PCT)} of total assets, not more than 25%."
        ),
        node=overall_secs_leaf,
        additional_instruction="Interpret 'securities overall' consistent with REIT asset tests; the computed ratio here is ≈18.57%."
    )

    # TRS securities not more than 20%
    trs_leaf = evaluator.add_leaf(
        id="TRS_Securities_Leq20",
        desc="Not more than 20% of total assets represented by TRS securities",
        parent=secs_limit_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Taxable REIT subsidiary (TRS) securities are ${TRS_STOCK:,}, which is {pct_str(TRS_PCT)} of total assets, not more than 20%.",
        node=trs_leaf,
        additional_instruction="Threshold per IRC §856(c)(4) and common guidance is 20% for TRS securities."
    )

    # Single issuer ≤ 5% of total assets
    single5_leaf = evaluator.add_leaf(
        id="Single_Issuer_Leq5pct_Assets",
        desc="No more than 5% of total assets invested in securities of any one issuer",
        parent=secs_limit_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"No single non-REIT issuer exceeds 5% of total assets: Issuer A ${ISSUER_A_BONDS:,} ({pct_str(ISSUER_A_ASSET_PCT)}), "
            f"Issuer B ${ISSUER_B_BONDS:,} ({pct_str(ISSUER_B_ASSET_PCT)}), Issuer C ${ISSUER_C_BONDS:,} ({pct_str(ISSUER_C_ASSET_PCT)}); "
            f"all are below 5%."
        ),
        node=single5_leaf
    )

    # Single issuer ≤ 10% of voting power or value
    single10_leaf = evaluator.add_leaf(
        id="Single_Issuer_Leq10pct_Vote_or_Value",
        desc="REIT does not own more than 10% of the voting power or value of any one issuer's outstanding securities",
        parent=secs_limit_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "Apex does not own more than 10% of the voting power or value of any one issuer's outstanding securities. "
            "For Issuer A, Apex holds 8% of value and 7% of voting power, both below 10%; "
            "holdings in Issuers B and C are bond positions with amounts implying ownership below the 10% thresholds."
        ),
        node=single10_leaf,
        additional_instruction="Judge this based on the provided percentages for Issuer A and the small bond positions in Issuers B and C."
    )


async def build_distribution_requirement(evaluator: Evaluator, parent) -> None:
    dist_leaf = evaluator.add_leaf(
        id="Distribution_Requirement",
        desc="Verify the REIT distributes at least 90% of its taxable income to shareholders as required by IRC § 857(a)(1)",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"For 2025, Apex's taxable income is ${TAXABLE_INCOME:,} and cash dividends paid are ${DIVIDENDS_PAID:,}. "
            f"The distribution ratio is {pct_str(DISTRIBUTION_PCT)} (20.5M / 22.0M), satisfying the 90% requirement."
        ),
        node=dist_leaf
    )


async def build_reit_qualification_tree(evaluator: Evaluator, root) -> None:
    # Top-level: REIT Qualification Verification (sequential, critical)
    top = evaluator.add_sequential(
        id="REIT_Qualification_Verification",
        desc="Verify whether the entity qualifies as a Real Estate Investment Trust (REIT) under federal tax law by sequentially checking all statutory requirements",
        parent=root,
        critical=True
    )

    # Structural
    await build_structural_requirements(evaluator, top)

    # Financial (income + assets)
    await build_financial_tests(evaluator, top)

    # Distribution
    await build_distribution_requirement(evaluator, top)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Apex Properties Trust REIT qualification (2025).
    """
    # Initialize evaluator (use sequential root to reflect gating nature)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract final conclusion (optional, for record)
    conclusion = await evaluator.extract(
        prompt=prompt_extract_conclusion(),
        template_class=ConclusionExtraction,
        extraction_name="answer_conclusion"
    )

    # Add ground truth / computed metrics for transparency
    evaluator.add_ground_truth({
        "income": {
            "rental_income": RENTAL_INCOME,
            "mortgage_interest": MORTGAGE_INTEREST,
            "dividends": DIVIDENDS,
            "treasury_interest": TREASURY_INTEREST,
            "service_fees": SERVICE_FEES,
            "total_gross_income": TOTAL_GROSS_INCOME,
            "75pct_income_total": Q75_INCOME,
            "75pct_income_ratio": pct_str(Q75_PCT),
            "95pct_income_total": Q95_INCOME,
            "95pct_income_ratio": pct_str(Q95_PCT)
        },
        "assets": {
            "office_real_estate": OFFICE_REAL_ESTATE,
            "retail_real_estate": RETAIL_REAL_ESTATE,
            "cash_equivalents": CASH_EQUIV,
            "us_government_securities": US_GOV_SECURITIES,
            "public_reit_shares": REIT_SHARES,
            "trs_stock": TRS_STOCK,
            "corporate_bonds_total": CORP_BONDS_TOTAL,
            "issuer_a_bonds": ISSUER_A_BONDS,
            "issuer_b_bonds": ISSUER_B_BONDS,
            "issuer_c_bonds": ISSUER_C_BONDS,
            "total_assets": TOTAL_ASSETS,
            "75pct_asset_total": ASSETS_75_CLASS,
            "75pct_asset_ratio": pct_str(ASSETS_75_PCT),
            "securities_overall_total": SECURITIES_OVERALL,
            "securities_overall_ratio": pct_str(SECURITIES_OVERALL_PCT),
            "trs_ratio": pct_str(TRS_PCT),
            "issuer_a_asset_pct": pct_str(ISSUER_A_ASSET_PCT),
            "issuer_b_asset_pct": pct_str(ISSUER_B_ASSET_PCT),
            "issuer_c_asset_pct": pct_str(ISSUER_C_ASSET_PCT)
        },
        "distribution": {
            "taxable_income": TAXABLE_INCOME,
            "dividends_paid": DIVIDENDS_PAID,
            "distribution_ratio": pct_str(DISTRIBUTION_PCT)
        },
        "ownership": {
            "shareholders_at_year_end": SH_COUNT_YEAR_END,
            "top5_percent": TOP5_OWNERS_PCT
        },
        "references": {
            "how_form_reit": REIT_HOW_TO_FORM_URL,
            "income_tests": RSM_INCOME_TESTS_URL,
            "asset_tests": RSM_ASSET_TESTS_URL
        }
    }, gt_type="computed_metrics")

    # Build verification tree and run checks
    await build_reit_qualification_tree(evaluator, root)

    # Return the structured evaluation summary
    return evaluator.get_summary()