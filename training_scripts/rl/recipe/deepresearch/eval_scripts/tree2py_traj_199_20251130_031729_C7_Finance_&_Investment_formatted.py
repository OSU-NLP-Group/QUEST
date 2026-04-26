import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "largest_tech_financial_profile_nov2025"
TASK_DESCRIPTION = (
    "As of November 2025, identify the largest technology company by market capitalization, and provide a comprehensive "
    "financial profile including: (1) the stock exchange where it is listed, (2) its ticker symbol, (3) current market "
    "capitalization, (4) current share price, (5) price-to-earnings (P/E) ratio, (6) dividend yield or dividend payment status, "
    "(7) beta coefficient, (8) institutional ownership percentage, (9) debt-to-equity ratio, (10) 52-week high and low prices, "
    "(11) average daily trading volume, (12) earnings per share (EPS), (13) revenue growth rate, and (14) analyst consensus rating or price target."
)


# ----------------------------- Data Models ----------------------------- #
class CompanyFinancialProfile(BaseModel):
    # Company identification
    company_name: Optional[str] = None
    sector_name: Optional[str] = None
    sector_source_urls: List[str] = Field(default_factory=list)
    largest_by_market_cap_sources: List[str] = Field(default_factory=list)

    # Exchange & ticker
    exchange: Optional[str] = None
    exchange_sources: List[str] = Field(default_factory=list)
    ticker: Optional[str] = None
    ticker_sources: List[str] = Field(default_factory=list)

    # Financial metrics
    market_cap: Optional[str] = None
    market_cap_sources: List[str] = Field(default_factory=list)

    share_price: Optional[str] = None
    share_price_sources: List[str] = Field(default_factory=list)

    pe_ratio: Optional[str] = None
    pe_ratio_sources: List[str] = Field(default_factory=list)

    dividend_info: Optional[str] = None
    dividend_sources: List[str] = Field(default_factory=list)

    beta_coefficient: Optional[str] = None
    beta_sources: List[str] = Field(default_factory=list)

    institutional_ownership_percentage: Optional[str] = None
    institutional_ownership_sources: List[str] = Field(default_factory=list)

    debt_to_equity_ratio: Optional[str] = None
    debt_to_equity_sources: List[str] = Field(default_factory=list)

    fifty_two_week_high: Optional[str] = None
    fifty_two_week_low: Optional[str] = None
    fifty_two_week_sources: List[str] = Field(default_factory=list)

    average_daily_trading_volume: Optional[str] = None
    avg_volume_sources: List[str] = Field(default_factory=list)

    eps: Optional[str] = None
    eps_sources: List[str] = Field(default_factory=list)

    revenue_growth_rate: Optional[str] = None
    revenue_growth_sources: List[str] = Field(default_factory=list)

    analyst_consensus_or_price_target: Optional[str] = None
    analyst_sources: List[str] = Field(default_factory=list)

    # Global/fallback sources
    global_sources: List[str] = Field(default_factory=list)


# ----------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_company_financial_profile() -> str:
    return (
        "Extract from the answer the single identified largest technology company by market capitalization as of November 2025, "
        "and all requested financial/profile fields. Return strings for values and arrays of URLs for sources. "
        "If any value is missing, set it to null; if sources are missing for a specific field, set the corresponding sources list to an empty array. "
        "Additionally, capture any general 'sources' provided in the answer as 'global_sources'. "
        "Fields to extract:\n"
        "1) company_name: Name of the identified company.\n"
        "2) sector_name: Sector classification (e.g., 'Technology' or 'Information Technology').\n"
        "3) sector_source_urls: URLs that support the sector classification.\n"
        "4) largest_by_market_cap_sources: URLs that support the claim that this is the largest technology company by market cap as of Nov 2025.\n"
        "5) exchange: The stock exchange where the company is listed (e.g., NASDAQ or NYSE).\n"
        "6) exchange_sources: URLs that support the exchange listing.\n"
        "7) ticker: The company’s ticker symbol.\n"
        "8) ticker_sources: URLs that support the ticker symbol.\n"
        "9) market_cap: Market capitalization value (string; include units if present) as of or within November 2025.\n"
        "10) market_cap_sources: URLs that support the market cap value (Nov 2025 time context where available).\n"
        "11) share_price: A current/recent share price within November 2025 (string, keep formatting as in answer).\n"
        "12) share_price_sources: URLs that support the share price.\n"
        "13) pe_ratio: P/E ratio (string; trailing or forward; as stated in the answer).\n"
        "14) pe_ratio_sources: URLs that support the P/E ratio.\n"
        "15) dividend_info: Dividend yield OR payment status (e.g., 'does not pay a dividend').\n"
        "16) dividend_sources: URLs that support the dividend info.\n"
        "17) beta_coefficient: Beta coefficient value (string).\n"
        "18) beta_sources: URLs that support the beta.\n"
        "19) institutional_ownership_percentage: Institutional ownership percentage (string, e.g., '62%').\n"
        "20) institutional_ownership_sources: URLs that support the institutional ownership.\n"
        "21) debt_to_equity_ratio: Debt-to-equity ratio (string).\n"
        "22) debt_to_equity_sources: URLs that support the debt-to-equity value.\n"
        "23) fifty_two_week_high: 52-week high price (string).\n"
        "24) fifty_two_week_low: 52-week low price (string).\n"
        "25) fifty_two_week_sources: URLs that support the 52-week high/low values.\n"
        "26) average_daily_trading_volume: Average daily trading volume (string).\n"
        "27) avg_volume_sources: URLs that support the average daily trading volume.\n"
        "28) eps: Earnings per share (string; specify TTM or period if present in the answer).\n"
        "29) eps_sources: URLs that support the EPS value.\n"
        "30) revenue_growth_rate: Revenue growth rate (string; specify period if present, e.g., YoY, quarterly).\n"
        "31) revenue_growth_sources: URLs that support the revenue growth rate.\n"
        "32) analyst_consensus_or_price_target: Analyst consensus rating OR price target (string).\n"
        "33) analyst_sources: URLs that support the analyst consensus/price target.\n"
        "34) global_sources: All general source URLs mentioned anywhere in the answer.\n"
        "Follow the SPECIAL RULES FOR URL SOURCES EXTRACTION: only extract actual URLs present in the answer."
    )


# ----------------------------- Helpers ----------------------------- #
def safe_str(x: Optional[str]) -> str:
    return x.strip() if isinstance(x, str) else ""


def resolve_sources(primary: List[str], extracted: CompanyFinancialProfile) -> Optional[List[str]]:
    if primary and len(primary) > 0:
        return primary
    if extracted.global_sources and len(extracted.global_sources) > 0:
        return extracted.global_sources
    return None


# ----------------------------- Verification Subtrees ----------------------------- #
async def build_company_identification_and_eligibility(
    evaluator: Evaluator,
    parent_node,
    extracted: CompanyFinancialProfile,
) -> None:
    node = evaluator.add_parallel(
        id="company_identification_and_eligibility",
        desc="Correctly identifies a single eligible company that is the largest by market capitalization as of November 2025 under the stated constraints.",
        parent=parent_node,
        critical=True,
    )

    # Identifies single company (judge from the answer text itself)
    single_company_leaf = evaluator.add_leaf(
        id="identifies_single_company",
        desc="Names one specific publicly traded company unambiguously (not multiple candidates).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer identifies exactly one specific publicly traded company, not multiple candidates.",
        node=single_company_leaf,
        additional_instruction="Read the answer context. If multiple companies are proposed or no single company is selected, this should be judged incorrect. Minor mentions are fine; focus on the chosen subject company."
    )

    # Sector eligibility
    sector_leaf = evaluator.add_leaf(
        id="sector_eligibility",
        desc="The identified company is in the Technology or Information Technology sector (per a publicly recognized classification).",
        parent=node,
        critical=True,
    )
    company = safe_str(extracted.company_name)
    sector = safe_str(extracted.sector_name)
    await evaluator.verify(
        claim=f"The company {company} is categorized in the Technology or Information Technology sector per a recognized classification (e.g., GICS). The answer states the sector as '{sector}'.",
        node=sector_leaf,
        sources=resolve_sources(extracted.sector_source_urls, extracted),
        additional_instruction="Confirm the sector classification on the provided source(s). Allow minor wording variants such as 'Information Technology'. Reject if sources clearly place it in another sector."
    )

    # Largest by market cap as of Nov 2025
    largest_leaf = evaluator.add_leaf(
        id="largest_by_market_cap_as_of_nov_2025",
        desc="Supports that the identified eligible company is the largest by market capitalization as of November 2025 (with verifiable public evidence).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of November 2025, {company} is the largest technology company by market capitalization.",
        node=largest_leaf,
        sources=resolve_sources(extracted.largest_by_market_cap_sources, extracted),
        additional_instruction="The evidence should indicate 'largest by market cap' in the Technology/Information Technology sector around November 2025. Accept evidence within November 2025 and allow minor timing variance. Prefer reputable finance sources (e.g., Bloomberg, Yahoo Finance, S&P, etc.)."
    )


async def build_financial_profile_outputs(
    evaluator: Evaluator,
    parent_node,
    extracted: CompanyFinancialProfile,
) -> None:
    node = evaluator.add_parallel(
        id="financial_profile_outputs",
        desc="Provides all requested financial/profile fields for the identified company, with values that are publicly verifiable (and time-appropriate for November 2025 where specified).",
        parent=parent_node,
        critical=True,
    )
    company = safe_str(extracted.company_name)

    # Exchange listing
    provided_exchange = evaluator.add_custom_node(
        result=bool(safe_str(extracted.exchange)),
        id="exchange_listing_provided",
        desc="Exchange listing value is provided.",
        parent=node,
        critical=True,
    )
    exchange_leaf = evaluator.add_leaf(
        id="exchange_listing",
        desc="Provides the stock exchange where the company is listed, and it must be a major U.S. exchange (NYSE or NASDAQ), in a publicly verifiable manner.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{company} is listed on {safe_str(extracted.exchange)}.",
        node=exchange_leaf,
        sources=resolve_sources(extracted.exchange_sources, extracted),
        additional_instruction="Verify the stated exchange (e.g., NASDAQ or NYSE) from the provided source(s). Confirm it is a major U.S. exchange (NYSE or NASDAQ)."
    )

    # Ticker symbol
    provided_ticker = evaluator.add_custom_node(
        result=bool(safe_str(extracted.ticker)),
        id="ticker_symbol_provided",
        desc="Ticker symbol value is provided.",
        parent=node,
        critical=True,
    )
    ticker_leaf = evaluator.add_leaf(
        id="ticker_symbol",
        desc="Provides the company’s ticker symbol in a publicly verifiable manner.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ticker symbol for {company} is {safe_str(extracted.ticker)}.",
        node=ticker_leaf,
        sources=resolve_sources(extracted.ticker_sources, extracted),
        additional_instruction="Confirm the ticker on the referenced source(s). Allow for class variants if explicitly indicated, but prefer the primary common ticker."
    )

    # Current market cap
    provided_market_cap = evaluator.add_custom_node(
        result=bool(safe_str(extracted.market_cap)),
        id="current_market_cap_provided",
        desc="Market capitalization value is provided.",
        parent=node,
        critical=True,
    )
    market_cap_leaf = evaluator.add_leaf(
        id="current_market_cap",
        desc="Provides the market capitalization as of (or within) November 2025, publicly verifiable and with clear units.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of within November 2025, the market capitalization of {company} is {safe_str(extracted.market_cap)}.",
        node=market_cap_leaf,
        sources=resolve_sources(extracted.market_cap_sources, extracted),
        additional_instruction="Check the market capitalization on the provided source(s). Accept November 2025 values and allow minor rounding or formatting differences (e.g., trillions vs billions labels)."
    )

    # Current share price
    provided_share_price = evaluator.add_custom_node(
        result=bool(safe_str(extracted.share_price)),
        id="current_share_price_provided",
        desc="Share price value is provided.",
        parent=node,
        critical=True,
    )
    share_price_leaf = evaluator.add_leaf(
        id="current_share_price",
        desc="Provides a current/recent share price from within November 2025, publicly verifiable and with clear units.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Within November 2025, the share price of {company} is {safe_str(extracted.share_price)}.",
        node=share_price_leaf,
        sources=resolve_sources(extracted.share_price_sources, extracted),
        additional_instruction="Confirm the share price within November 2025 from the provided source(s). Accept minor rounding differences."
    )

    # P/E ratio
    provided_pe = evaluator.add_custom_node(
        result=bool(safe_str(extracted.pe_ratio)),
        id="pe_ratio_provided",
        desc="P/E ratio value is provided.",
        parent=node,
        critical=True,
    )
    pe_leaf = evaluator.add_leaf(
        id="pe_ratio",
        desc="Provides the P/E ratio (trailing or forward) in a publicly verifiable manner (or provides enough to calculate it from cited figures).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The P/E ratio of {company} is {safe_str(extracted.pe_ratio)}.",
        node=pe_leaf,
        sources=resolve_sources(extracted.pe_ratio_sources, extracted),
        additional_instruction="Confirm the P/E ratio (TTM or forward) from the provided source(s). Minor variance due to timing is acceptable."
    )

    # Dividend info
    provided_dividend = evaluator.add_custom_node(
        result=bool(safe_str(extracted.dividend_info)),
        id="dividend_info_provided",
        desc="Dividend info value is provided.",
        parent=node,
        critical=True,
    )
    dividend_leaf = evaluator.add_leaf(
        id="dividend_info",
        desc="Provides dividend yield OR clearly states dividend payment status (e.g., pays/does not pay), publicly verifiable.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Dividend information for {company}: {safe_str(extracted.dividend_info)}.",
        node=dividend_leaf,
        sources=resolve_sources(extracted.dividend_sources, extracted),
        additional_instruction="Verify whether {company} pays a dividend and the yield if applicable, from the provided source(s). Accept variants such as 'no dividend' or 'does not pay a dividend'."
    )

    # Beta coefficient
    provided_beta = evaluator.add_custom_node(
        result=bool(safe_str(extracted.beta_coefficient)),
        id="beta_coefficient_provided",
        desc="Beta coefficient value is provided.",
        parent=node,
        critical=True,
    )
    beta_leaf = evaluator.add_leaf(
        id="beta_coefficient",
        desc="Provides the beta coefficient, publicly verifiable.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The beta coefficient of {company} is {safe_str(extracted.beta_coefficient)}.",
        node=beta_leaf,
        sources=resolve_sources(extracted.beta_sources, extracted),
        additional_instruction="Confirm beta value on the provided source(s). Allow minor variation depending on calculation window."
    )

    # Institutional ownership percentage
    provided_inst = evaluator.add_custom_node(
        result=bool(safe_str(extracted.institutional_ownership_percentage)),
        id="institutional_ownership_percentage_provided",
        desc="Institutional ownership percentage value is provided.",
        parent=node,
        critical=True,
    )
    inst_leaf = evaluator.add_leaf(
        id="institutional_ownership_percentage",
        desc="Provides institutional ownership percentage, publicly verifiable.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The institutional ownership percentage of {company} is {safe_str(extracted.institutional_ownership_percentage)}.",
        node=inst_leaf,
        sources=resolve_sources(extracted.institutional_ownership_sources, extracted),
        additional_instruction="Confirm institutional ownership percentage from the provided source(s). Allow minor rounding (e.g., 61.8% vs 62%)."
    )

    # Debt-to-equity ratio
    provided_de = evaluator.add_custom_node(
        result=bool(safe_str(extracted.debt_to_equity_ratio)),
        id="debt_to_equity_ratio_provided",
        desc="Debt-to-equity ratio value is provided.",
        parent=node,
        critical=True,
    )
    de_leaf = evaluator.add_leaf(
        id="debt_to_equity_ratio",
        desc="Provides debt-to-equity ratio, publicly verifiable (or provides enough to calculate it from cited financial statement figures).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The debt-to-equity ratio of {company} is {safe_str(extracted.debt_to_equity_ratio)}.",
        node=de_leaf,
        sources=resolve_sources(extracted.debt_to_equity_sources, extracted),
        additional_instruction="Verify D/E ratio from the provided source(s). Accept minor differences due to period or calculation conventions."
    )

    # 52-week high and low
    provided_52w = evaluator.add_custom_node(
        result=bool(safe_str(extracted.fifty_two_week_high)) and bool(safe_str(extracted.fifty_two_week_low)),
        id="52_week_high_low_provided",
        desc="52-week high/low values are provided.",
        parent=node,
        critical=True,
    )
    fifty_two_leaf = evaluator.add_leaf(
        id="52_week_high_low",
        desc="Provides the 52-week high and 52-week low prices, publicly verifiable.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 52-week high for {company} is {safe_str(extracted.fifty_two_week_high)}, and the 52-week low is {safe_str(extracted.fifty_two_week_low)}.",
        node=fifty_two_leaf,
        sources=resolve_sources(extracted.fifty_two_week_sources, extracted),
        additional_instruction="Confirm both 52-week high and low values from the provided source(s). Allow minor rounding differences."
    )

    # Average daily trading volume
    provided_vol = evaluator.add_custom_node(
        result=bool(safe_str(extracted.average_daily_trading_volume)),
        id="average_daily_trading_volume_provided",
        desc="Average daily trading volume value is provided.",
        parent=node,
        critical=True,
    )
    vol_leaf = evaluator.add_leaf(
        id="average_daily_trading_volume",
        desc="Provides average daily trading volume, publicly verifiable.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The average daily trading volume of {company} is {safe_str(extracted.average_daily_trading_volume)}.",
        node=vol_leaf,
        sources=resolve_sources(extracted.avg_volume_sources, extracted),
        additional_instruction="Confirm average daily trading volume on the provided source(s). Accept common presentation formats (e.g., 'M' for million)."
    )

    # EPS
    provided_eps = evaluator.add_custom_node(
        result=bool(safe_str(extracted.eps)),
        id="earnings_per_share_eps_provided",
        desc="EPS value is provided.",
        parent=node,
        critical=True,
    )
    eps_leaf = evaluator.add_leaf(
        id="earnings_per_share_eps",
        desc="Provides EPS from recent financial reports, publicly verifiable.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The EPS for {company} is {safe_str(extracted.eps)}.",
        node=eps_leaf,
        sources=resolve_sources(extracted.eps_sources, extracted),
        additional_instruction="Confirm EPS from the provided source(s). Accept TTM or latest reported EPS as stated. Minor rounding differences are acceptable."
    )

    # Revenue growth rate
    provided_growth = evaluator.add_custom_node(
        result=bool(safe_str(extracted.revenue_growth_rate)),
        id="revenue_growth_rate_provided",
        desc="Revenue growth rate value is provided.",
        parent=node,
        critical=True,
    )
    growth_leaf = evaluator.add_leaf(
        id="revenue_growth_rate",
        desc="Provides a revenue growth rate (YoY or quarterly) that is publicly verifiable and clearly defines the period used.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The revenue growth rate for {company} is {safe_str(extracted.revenue_growth_rate)}.",
        node=growth_leaf,
        sources=resolve_sources(extracted.revenue_growth_sources, extracted),
        additional_instruction="Confirm the growth rate and period (YoY, quarterly, etc.) from the provided source(s). Minor rounding differences are acceptable."
    )

    # Analyst consensus or price target
    provided_analyst = evaluator.add_custom_node(
        result=bool(safe_str(extracted.analyst_consensus_or_price_target)),
        id="analyst_consensus_or_price_target_provided",
        desc="Analyst consensus or price target value is provided.",
        parent=node,
        critical=True,
    )
    analyst_leaf = evaluator.add_leaf(
        id="analyst_consensus_or_price_target",
        desc="Provides an analyst consensus rating OR an analyst price target, sourced from recognized/credible public sources.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Analyst consensus rating or price target for {company}: {safe_str(extracted.analyst_consensus_or_price_target)}.",
        node=analyst_leaf,
        sources=resolve_sources(extracted.analyst_sources, extracted),
        additional_instruction="Verify the analyst consensus or price target on recognized sources (e.g., major finance portals). Minor phrasing differences are acceptable if the substance matches."
    )


# ----------------------------- Main Evaluation Entry ----------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # First identify company, then verify financial profile
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured company and financial profile info
    extracted = await evaluator.extract(
        prompt=prompt_extract_company_financial_profile(),
        template_class=CompanyFinancialProfile,
        extraction_name="company_financial_profile",
    )

    # Build verification tree according to rubric
    await build_company_identification_and_eligibility(evaluator, root, extracted)
    await build_financial_profile_outputs(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()