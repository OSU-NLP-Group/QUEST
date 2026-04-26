import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "financial_market_reference_2024Q4"
TASK_DESCRIPTION = (
    "Compile a comprehensive financial market reference document for November-December 2024 that includes: "
    "(1) Current interest rate benchmarks for mortgages and federal policy; "
    "(2) Holiday trading hours for U.S. equity and bond markets on Black Friday 2024; "
    "(3) Regulatory distribution requirements and tax treatment for REITs and municipal bonds including credit rating thresholds; "
    "(4) Social Security program parameters including full retirement age, delayed retirement credits, and COLA adjustment for 2025; and "
    "(5) Standard dividend payment frequencies and federal income tax bracket structure. "
    "Each piece of information must be accurately stated with proper numerical values, percentages, times, or categorical descriptors as applicable."
)

# Optional ground-truth/expected references (normative expectations used by rubric)
EXPECTED_REFERENCE = {
    "Interest_Rates": {
        "Mortgage_30yr_Range": "Approximately 6.20%–6.40% for Nov 2024",
        "Fed_Funds_Range_Dec2024": "Target range 4.25%–4.5%",
    },
    "Trading_Hours_Black_Friday_2024": {
        "NYSE_Close": "1:00 PM ET",
        "Bond_Market_Close": "2:00 PM ET",
    },
    "Investment_Regulations": {
        "REIT_Distribution": "At least 90% of taxable income must be distributed",
        "Muni_Federal_Tax": "Interest generally exempt from federal income tax",
        "IG_Threshold_SP_Fitch": "BBB- or higher",
        "IG_Threshold_Moodys": "Baa3 or higher",
    },
    "Social_Security": {
        "FRA_1960_or_later": "Age 67",
        "Delayed_Retirement_Credit": "8% per year up to age 70",
        "COLA_2025": "Either 2.5% or 2.8% acceptable by rubric",
    },
    "Tax_and_Payments": {
        "Brackets_2024": [10, 12, 22, 24, 32, 35, 37],
        "Dividend_Frequency": "Most stocks quarterly; some REITs monthly",
        "Muni_Default_Rate": "Very low (acceptable: <0.1% or 'very low/rare')",
    }
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class InterestRates(BaseModel):
    mortgage_30yr_value_str: Optional[str] = None
    fed_funds_range_str: Optional[str] = None
    interest_rate_sources: List[str] = Field(default_factory=list)


class MarketTradingHours(BaseModel):
    nyse_black_friday_close: Optional[str] = None
    bond_market_black_friday_close: Optional[str] = None
    trading_hours_sources: List[str] = Field(default_factory=list)


class InvestmentRegulations(BaseModel):
    reit_distribution_requirement: Optional[str] = None
    muni_bond_federal_tax_status: Optional[str] = None
    investment_grade_sp_fitch: Optional[str] = None
    investment_grade_moodys: Optional[str] = None
    investment_reg_sources: List[str] = Field(default_factory=list)


class SocialSecurityParams(BaseModel):
    full_retirement_age: Optional[str] = None
    delayed_retirement_credits: Optional[str] = None
    cola_2025: Optional[str] = None
    ss_sources: List[str] = Field(default_factory=list)


class TaxAndPaymentStructures(BaseModel):
    federal_tax_brackets: List[str] = Field(default_factory=list)
    dividend_payment_frequency: Optional[str] = None
    muni_bond_default_rate_statement: Optional[str] = None
    tax_sources: List[str] = Field(default_factory=list)


class FinancialReferenceExtraction(BaseModel):
    interest_rates: InterestRates = Field(default_factory=InterestRates)
    trading_hours: MarketTradingHours = Field(default_factory=MarketTradingHours)
    investment_regulations: InvestmentRegulations = Field(default_factory=InvestmentRegulations)
    social_security: SocialSecurityParams = Field(default_factory=SocialSecurityParams)
    tax_and_payments: TaxAndPaymentStructures = Field(default_factory=TaxAndPaymentStructures)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_financial_reference() -> str:
    return """
    Extract the following structured information exactly as it appears in the answer. Use strings for numbers, percentages, times, and ranges; include units and descriptors as written by the answer. Also extract all explicit source URLs mentioned for each category.

    interest_rates:
      - mortgage_30yr_value_str: The stated national 30-year fixed mortgage rate (for Nov 2024), possibly a single value or a range (e.g., "6.3%" or "6.2–6.4%").
      - fed_funds_range_str: The stated Federal Reserve federal funds target rate range as of Dec 2024 (e.g., "4.25%-4.5%").
      - interest_rate_sources: All URLs the answer cites for these interest rate figures (Freddie Mac, Federal Reserve, etc.).

    trading_hours:
      - nyse_black_friday_close: The stated NYSE close time on Black Friday 2024 (e.g., "1:00 PM ET", "1 PM Eastern").
      - bond_market_black_friday_close: The stated U.S. bond market close time on Black Friday 2024 (e.g., "2:00 PM ET").
      - trading_hours_sources: All URLs the answer cites for holiday trading hours (NYSE, SIFMA, etc.).

    investment_regulations:
      - reit_distribution_requirement: The stated REIT distribution requirement (e.g., "must distribute at least 90% of taxable income").
      - muni_bond_federal_tax_status: The stated federal tax treatment of municipal bond interest (e.g., "generally exempt from federal income tax").
      - investment_grade_sp_fitch: The stated investment grade threshold on S&P and/or Fitch scales (e.g., "BBB- or higher").
      - investment_grade_moodys: The stated investment grade threshold on Moody's scale (e.g., "Baa3 or higher").
      - investment_reg_sources: All URLs cited for these regulations/definitions (IRS, rating agencies, etc.).

    social_security:
      - full_retirement_age: The stated FRA rule (e.g., "67 for those born in 1960 or later").
      - delayed_retirement_credits: The stated DRC rule (e.g., "8% per year up to age 70").
      - cola_2025: The stated 2025 COLA percentage (e.g., "2.5%" or "2.8%").
      - ss_sources: All URLs cited for Social Security info (e.g., ssa.gov).

    tax_and_payments:
      - federal_tax_brackets: The list (or enumerated text) of 2024 federal income tax bracket percentages included in the answer (e.g., ["10%", "12%", "22%", "24%", "32%", "35%", "37%"] or a single string with them).
      - dividend_payment_frequency: The stated standard dividend frequency pattern (e.g., "most pay quarterly; some REITs pay monthly").
      - muni_bond_default_rate_statement: The stated default-rate characterization for investment-grade municipal bonds (e.g., "very low" or "<0.1%").
      - tax_sources: All URLs cited for these tax/payment facts (IRS, financial data providers, etc.).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_valid_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_interest_rates(evaluator: Evaluator, parent_node, data: InterestRates) -> None:
    section = evaluator.add_parallel(
        id="Interest_Rate_Benchmarks",
        desc="Current interest rate information for key benchmarks as of November-December 2024",
        parent=parent_node,
        critical=True
    )

    # Mortgage rate (range expectation)
    node_mortgage = evaluator.add_leaf(
        id="Mortgage_Rate_30_Year",
        desc="States the 30-year fixed mortgage rate for November 2024 as approximately 6.2-6.4% range (acceptable: any value between 6.20% and 6.40%)",
        parent=section,
        critical=True
    )
    mortgage_claim = (
        "The 30-year fixed mortgage rate for November 2024 is approximately between 6.20% and 6.40%."
    )
    await evaluator.verify(
        claim=mortgage_claim,
        node=node_mortgage,
        sources=data.interest_rate_sources,  # If URLs are provided, verify against them; otherwise fallback to answer
        additional_instruction=(
            "Accept slight wording variations like '~6.3%' or '6.2–6.4%'. "
            "If verifying by URL(s), confirm the page supports a national 30-year fixed rate in that approximate range during November 2024. "
            "If no URLs, check whether the answer clearly states a value in 6.20%–6.40%."
        )
    )

    # Federal funds rate range (as-of Dec 2024)
    node_fed = evaluator.add_leaf(
        id="Federal_Funds_Rate",
        desc="States the Federal Reserve's federal funds target rate range as of December 2024 as 4.25%-4.5%",
        parent=section,
        critical=True
    )
    fed_claim = (
        "As of December 2024, the Federal Reserve's federal funds target range is 4.25% to 4.5%."
    )
    await evaluator.verify(
        claim=fed_claim,
        node=node_fed,
        sources=data.interest_rate_sources,
        additional_instruction=(
            "Verify the stated target range exactly or with minor formatting variations (e.g., '4.25%–4.50%'). "
            "If using URLs, confirm the range on the cited Federal Reserve or other authoritative source page."
        )
    )

    # Source reference presence (URLs)
    evaluator.add_custom_node(
        result=has_valid_urls(data.interest_rate_sources),
        id="Interest_Rate_Source_Reference",
        desc="Provides verifiable source URL for interest rate data (e.g., Freddie Mac, Federal Reserve)",
        parent=section,
        critical=True
    )


async def verify_trading_hours(evaluator: Evaluator, parent_node, data: MarketTradingHours) -> None:
    section = evaluator.add_parallel(
        id="Market_Trading_Hours",
        desc="Holiday trading schedule information for Black Friday 2024",
        parent=parent_node,
        critical=True
    )

    # NYSE closes 1:00 PM ET
    node_nyse = evaluator.add_leaf(
        id="NYSE_Black_Friday_Close",
        desc="States that NYSE closes at 1:00 PM ET (or 1 PM Eastern Time) on Black Friday",
        parent=section,
        critical=True
    )
    nyse_claim = "On Black Friday 2024, the NYSE closes at 1:00 PM Eastern Time."
    await evaluator.verify(
        claim=nyse_claim,
        node=node_nyse,
        sources=data.trading_hours_sources,
        additional_instruction=(
            "Accept phrasing like '1 PM ET' or '1:00 p.m. ET' or 'early close at 1 PM Eastern'. "
            "If URLs are provided (e.g., NYSE holiday calendar), confirm the early close time."
        )
    )

    # Bond market closes 2:00 PM ET
    node_bond = evaluator.add_leaf(
        id="Bond_Market_Black_Friday_Close",
        desc="States that U.S. bond markets close at 2:00 PM ET (or 2 PM Eastern Time) on Black Friday",
        parent=section,
        critical=True
    )
    bond_claim = "On Black Friday 2024, the U.S. bond markets close at 2:00 PM Eastern Time."
    await evaluator.verify(
        claim=bond_claim,
        node=node_bond,
        sources=data.trading_hours_sources,
        additional_instruction=(
            "Accept phrasing like '2 PM ET'. If URLs (e.g., SIFMA recommendations) are provided, verify the early close time there."
        )
    )

    # Source reference presence (URLs)
    evaluator.add_custom_node(
        result=has_valid_urls(data.trading_hours_sources),
        id="Trading_Hours_Source_Reference",
        desc="Provides verifiable source URL for trading hours information (e.g., NYSE, SIFMA)",
        parent=section,
        critical=True
    )


async def verify_investment_regulations(evaluator: Evaluator, parent_node, data: InvestmentRegulations) -> None:
    section = evaluator.add_parallel(
        id="Investment_Regulations_And_Tax_Treatment",
        desc="Regulatory requirements and tax treatment for REITs and municipal bonds",
        parent=parent_node,
        critical=True
    )

    # Build leaf nodes
    node_reit = evaluator.add_leaf(
        id="REIT_Distribution_Requirement",
        desc="States that REITs must distribute at least 90% of taxable income to shareholders",
        parent=section,
        critical=True
    )
    node_muni_tax = evaluator.add_leaf(
        id="Municipal_Bond_Federal_Tax_Status",
        desc="States that municipal bond interest is generally exempt from federal income tax",
        parent=section,
        critical=True
    )
    node_sp_fitch = evaluator.add_leaf(
        id="Investment_Grade_Rating_SP_Fitch",
        desc="States that investment grade threshold is BBB- or higher for S&P and Fitch rating scales",
        parent=section,
        critical=True
    )
    node_moodys = evaluator.add_leaf(
        id="Investment_Grade_Rating_Moodys",
        desc="States that investment grade threshold is Baa3 or higher for Moody's rating scale",
        parent=section,
        critical=True
    )

    # Prepare claims and batch verify
    claims = [
        (
            "Real Estate Investment Trusts (REITs) must distribute at least 90% of their taxable income to shareholders.",
            data.investment_reg_sources,
            node_reit,
            "Accept minor wording variations. If URLs are provided (e.g., IRS publications), verify the 90% distribution requirement."
        ),
        (
            "Municipal bond interest is generally exempt from federal income tax.",
            data.investment_reg_sources,
            node_muni_tax,
            "Accept standard phrasing like 'generally exempt from federal income tax'. If URLs are provided (e.g., IRS), verify the statement."
        ),
        (
            "On S&P and Fitch scales, investment grade is BBB- or higher.",
            data.investment_reg_sources,
            node_sp_fitch,
            "Accept minor variants (e.g., 'BBB minus'). If URLs are provided (e.g., S&P, Fitch documentation), verify the threshold."
        ),
        (
            "On Moody's scale, investment grade is Baa3 or higher.",
            data.investment_reg_sources,
            node_moodys,
            "Accept minor variants. If URLs are provided (e.g., Moody's documentation), verify the threshold."
        )
    ]
    await evaluator.batch_verify(claims)

    # Source reference presence (URLs)
    evaluator.add_custom_node(
        result=has_valid_urls(data.investment_reg_sources),
        id="Investment_Regulations_Source_Reference",
        desc="Provides verifiable source URL for investment regulations (e.g., IRS, rating agency documentation)",
        parent=section,
        critical=True
    )


async def verify_social_security(evaluator: Evaluator, parent_node, data: SocialSecurityParams) -> None:
    section = evaluator.add_parallel(
        id="Social_Security_Parameters",
        desc="Social Security program rules and adjustments for 2024-2025",
        parent=parent_node,
        critical=True
    )

    node_fra = evaluator.add_leaf(
        id="Full_Retirement_Age",
        desc="States that full retirement age is 67 for those born in 1960 or later",
        parent=section,
        critical=True
    )
    node_drc = evaluator.add_leaf(
        id="Delayed_Retirement_Credits",
        desc="States that benefits increase by 8% per year (or approximately 8 percent annually) for each year delayed past full retirement age up to age 70",
        parent=section,
        critical=True
    )
    node_cola = evaluator.add_leaf(
        id="COLA_2025",
        desc="States that the Social Security cost-of-living adjustment for 2025 is 2.5% (or 2.8%, as both were mentioned in different contexts)",
        parent=section,
        critical=True
    )

    claims = [
        (
            "For people born in 1960 or later, the Social Security full retirement age (FRA) is 67.",
            data.ss_sources,
            node_fra,
            "Accept equivalent phrasing such as 'FRA is 67 for those born in 1960 or after'. If URLs are provided (ssa.gov), verify the rule."
        ),
        (
            "Social Security benefits increase by about 8% per year for each year delayed beyond FRA up to age 70.",
            data.ss_sources,
            node_drc,
            "Accept 'approximately 8 percent annually' and equivalent language. If URLs are provided (ssa.gov), verify this DRC rule."
        ),
        (
            "The 2025 Social Security COLA is 2.5% or 2.8% (either acceptable).",
            data.ss_sources,
            node_cola,
            "If the answer states either 2.5% or 2.8%, treat as acceptable per rubric. If URLs are provided, verify."
        )
    ]
    await evaluator.batch_verify(claims)

    evaluator.add_custom_node(
        result=has_valid_urls(data.ss_sources),
        id="Social_Security_Source_Reference",
        desc="Provides verifiable source URL for Social Security information (e.g., SSA.gov)",
        parent=section,
        critical=True
    )


async def verify_tax_and_payments(evaluator: Evaluator, parent_node, data: TaxAndPaymentStructures) -> None:
    section = evaluator.add_parallel(
        id="Tax_And_Payment_Structures",
        desc="Federal tax brackets and typical investment payment frequencies",
        parent=parent_node,
        critical=True
    )

    node_brackets = evaluator.add_leaf(
        id="Federal_Tax_Brackets_2024",
        desc="States that there are seven federal income tax brackets for 2024: 10%, 12%, 22%, 24%, 32%, 35%, and 37%",
        parent=section,
        critical=True
    )
    node_divfreq = evaluator.add_leaf(
        id="Dividend_Payment_Frequency",
        desc="States that most dividend-paying stocks pay quarterly, with some REITs paying monthly",
        parent=section,
        critical=True
    )
    node_muni_default = evaluator.add_leaf(
        id="Municipal_Bond_Default_Rate",
        desc="States that investment grade municipal bonds have very low historical default rates (acceptable: any statement indicating rates below 0.1% or very low/rare defaults)",
        parent=section,
        critical=True
    )

    claims = [
        (
            "For tax year 2024, there are seven U.S. federal income tax brackets: 10%, 12%, 22%, 24%, 32%, 35%, and 37%.",
            data.tax_sources,
            node_brackets,
            "Accept minor formatting differences. If URLs (IRS) are provided, verify bracket percentages."
        ),
        (
            "Most dividend-paying stocks pay quarterly dividends, and some REITs pay monthly dividends.",
            data.tax_sources,
            node_divfreq,
            "Accept standard phrasing indicating 'most quarterly' and 'some REITs monthly'. If URLs are provided, verify."
        ),
        (
            "Investment grade municipal bonds have very low historical default rates, e.g., below 0.1% or otherwise characterized as very low/rare.",
            data.tax_sources,
            node_muni_default,
            "Accept any clear statement that IG muni defaults are very low (e.g., long-term default rate under 0.1%). If URLs are provided, verify."
        )
    ]
    await evaluator.batch_verify(claims)

    evaluator.add_custom_node(
        result=has_valid_urls(data.tax_sources),
        id="Tax_Structure_Source_Reference",
        desc="Provides verifiable source URL for tax and payment structure information (e.g., IRS, financial data providers)",
        parent=section,
        critical=True
    )


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
    Evaluate an answer for the financial market reference compilation (Nov–Dec 2024).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
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

    # Build a critical wrapper node since initialize() root is non-critical by design
    top = evaluator.add_parallel(
        id="Root_Financial_Market_Reference_Compilation",
        desc="Complete and accurate financial market reference document covering all required categories with proper sourcing",
        parent=root,
        critical=True
    )

    # Extract structured info
    extracted: FinancialReferenceExtraction = await evaluator.extract(
        prompt=prompt_extract_financial_reference(),
        template_class=FinancialReferenceExtraction,
        extraction_name="financial_reference_extraction"
    )

    # Add expected reference info (for transparency; not used directly for scoring)
    evaluator.add_ground_truth(EXPECTED_REFERENCE, gt_type="expected_reference_rubric")

    # Verify each section (all critical under the top-level critical node)
    await verify_interest_rates(evaluator, top, extracted.interest_rates)
    await verify_trading_hours(evaluator, top, extracted.trading_hours)
    await verify_investment_regulations(evaluator, top, extracted.investment_regulations)
    await verify_social_security(evaluator, top, extracted.social_security)
    await verify_tax_and_payments(evaluator, top, extracted.tax_and_payments)

    # Return the evaluation summary
    return evaluator.get_summary()