import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "three_stock_ira_portfolio_2026"
TASK_DESCRIPTION = (
    "An investor wants to build a dividend-focused portfolio within a 2026 Roth IRA with a total contribution of "
    "$7,500. The portfolio must consist of three stocks, with funds allocated equally ($2,500 per stock). "
    "The investor has established detailed selection criteria for three stocks (A: Telecommunications, "
    "B: Asset Management Firm, C: BDC managed by B). Provide ticker, full name, and current dividend yield "
    "(as of March 2026) for each stock, plus: Stock A consecutive dividend increase years, "
    "Stock B total AUM (most recent quarter), and Stock C quarterly dividend amount per share. "
    "Verify the total investment complies with the 2026 Roth IRA contribution limit for individuals under 50."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StockBase(BaseModel):
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    structure: Optional[str] = None  # e.g., "BDC" for Stock C
    dividend_yield: Optional[str] = None  # as mentioned in the answer (string, e.g., "5.4%")

    # URL sources (explicitly cited in the answer)
    identity_sources: List[str] = Field(default_factory=list)            # listing, sector/industry/structure proof
    dividend_yield_sources: List[str] = Field(default_factory=list)      # yield proof
    general_sources: List[str] = Field(default_factory=list)             # any additional sources for this stock


class StockAExtraction(StockBase):
    consecutive_increase_years: Optional[str] = None
    dividend_history_sources: List[str] = Field(default_factory=list)    # proof of consecutive increases


class StockBExtraction(StockBase):
    platforms: List[str] = Field(default_factory=list)                   # names of investment platforms
    platform_sources: List[str] = Field(default_factory=list)            # proof of platforms
    aum: Optional[str] = None                                            # AUM string as stated
    aum_sources: List[str] = Field(default_factory=list)                 # proof of AUM


class StockCExtraction(StockBase):
    management_firm_name: Optional[str] = None                           # firm managing the BDC (should match Stock B)
    management_sources: List[str] = Field(default_factory=list)          # proof of management relationship
    dividend_frequency: Optional[str] = None                             # "quarterly", "monthly", etc.
    quarterly_dividend_amount: Optional[str] = None                      # amount per share (string)
    distribution_sources: List[str] = Field(default_factory=list)        # proof of dividend schedule/amount


class PortfolioExtraction(BaseModel):
    allocation_statement: Optional[str] = None                           # statement like "allocate $2,500 to each"
    allocations: List[str] = Field(default_factory=list)                 # amounts per stock if explicitly stated
    total_investment: Optional[str] = None                               # total amount mentioned (e.g., "$7,500")
    ira_limit_amount: Optional[str] = None                               # stated 2026 IRA limit (e.g., "$7,500")
    ira_sources: List[str] = Field(default_factory=list)                 # proof of 2026 IRA limit for <50


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_stock_a() -> str:
    return """
    Extract all details for Stock A (Telecommunications Company) as explicitly provided in the answer.

    Required fields:
    - ticker: The ticker symbol
    - company_name: The full company name
    - exchange: The stock exchange mentioned (e.g., NYSE)
    - sector: The sector name if provided (e.g., Communication Services)
    - industry: The industry name if provided (e.g., Telecommunications, Wireless Telecommunications Services)
    - dividend_yield: The current dividend yield stated (as of March 2026), as a string including '%' if given
    - consecutive_increase_years: The specific number of consecutive years the company has increased its dividend, as provided in the answer

    URL sources:
    - identity_sources: All URLs cited that support listing on the exchange and the sector/industry classification
    - dividend_yield_sources: All URLs cited that support the current dividend yield
    - dividend_history_sources: All URLs cited that support the consecutive dividend increase years
    - general_sources: Any other URLs associated with Stock A provided in the answer

    If any field is not explicitly present in the answer, set it to null (for strings) or an empty list (for URLs).
    """


def prompt_extract_stock_b() -> str:
    return """
    Extract all details for Stock B (Asset Management Firm) as explicitly provided in the answer.

    Required fields:
    - ticker: The ticker symbol
    - company_name: The full company name
    - exchange: The stock exchange mentioned (e.g., NYSE)
    - sector: Sector name if provided
    - industry: Industry if provided (e.g., Asset Management, Investment Management)
    - structure: If a special structure is given (usually null for Stock B)
    - dividend_yield: The current dividend yield stated (as of March 2026), as a string including '%' if given
    - platforms: The list of distinct investment platforms operated by the firm (e.g., Credit, Real Assets, GP Strategic Capital)
    - aum: The total Assets Under Management (most recent quarter) as stated in the answer

    URL sources:
    - identity_sources: All URLs cited that support listing on the exchange and classification as asset management
    - dividend_yield_sources: All URLs cited that support the current dividend yield
    - platform_sources: All URLs cited that support the investment platforms information
    - aum_sources: All URLs cited that support the AUM figure
    - general_sources: Any other URLs associated with Stock B provided in the answer

    If any field is not explicitly present in the answer, set it to null (for strings) or an empty list (for URLs/lists).
    """


def prompt_extract_stock_c() -> str:
    return """
    Extract all details for Stock C (Business Development Company managed by Stock B's firm) as explicitly provided in the answer.

    Required fields:
    - ticker: The ticker symbol
    - company_name: The full company name
    - exchange: The stock exchange mentioned (e.g., NYSE)
    - sector: Sector if provided
    - industry: Industry if provided
    - structure: The structure (e.g., BDC / Business Development Company)
    - dividend_yield: The current dividend yield stated (as of March 2026), as a string including '%' if given
    - management_firm_name: The name of the firm that manages this BDC (should match Stock B's firm name)
    - dividend_frequency: The dividend distribution frequency (e.g., quarterly)
    - quarterly_dividend_amount: The specific quarterly dividend amount per share stated in the answer

    URL sources:
    - identity_sources: URLs supporting exchange listing and BDC structure
    - dividend_yield_sources: URLs supporting the current dividend yield
    - management_sources: URLs supporting that Stock C is managed by the same firm as Stock B
    - distribution_sources: URLs supporting dividend schedule and/or quarterly dividend amount
    - general_sources: Any other URLs associated with Stock C provided in the answer

    If any field is not explicitly present in the answer, set it to null (for strings) or an empty list (for URLs).
    """


def prompt_extract_portfolio() -> str:
    return """
    Extract portfolio-level details explicitly provided in the answer.

    Required fields:
    - allocation_statement: A textual statement indicating allocation (e.g., "$2,500 per stock", or "equally across three stocks")
    - allocations: A list of allocation amounts per stock if the answer lists them individually (strings like "$2,500")
    - total_investment: The total portfolio investment amount (e.g., "$7,500"), if stated
    - ira_limit_amount: The 2026 Roth IRA contribution limit for individuals under age 50, as stated in the answer (e.g., "$7,500")

    URL sources:
    - ira_sources: Any URLs cited to support the 2026 Roth IRA contribution limit information

    If any field is not explicitly present in the answer, set it to null (for strings) or an empty list (for URLs/lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_nonempty(*vals: Optional[str]) -> str:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return "the company"


def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


def _list_to_english(items: List[str]) -> str:
    items = [x for x in items if x and x.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------- #
# Verification logic per stock                                                #
# --------------------------------------------------------------------------- #
async def verify_stock_a(evaluator: Evaluator, parent, a: StockAExtraction):
    # Parent node for Stock A
    stock_a_node = evaluator.add_parallel(
        id="stock_a",
        desc="Identification and verification of a qualifying telecommunications stock",
        parent=parent,
        critical=False
    )

    # 1) Identity & Exchange (critical)
    identity_node = evaluator.add_parallel(
        id="stock_a_identity_exchange",
        desc="Basic identification information and exchange listing for Stock A",
        parent=stock_a_node,
        critical=True
    )

    # 1.1) Ticker & Name provided (existence check)
    evaluator.add_custom_node(
        result=bool(a.ticker and a.ticker.strip()) and bool(a.company_name and a.company_name.strip()),
        id="stock_a_ticker_and_name",
        desc="Ticker symbol and full company name provided for Stock A",
        parent=identity_node,
        critical=True
    )

    # 1.2) NYSE & Telecom classification (URL verification)
    node_nyse_telecom = evaluator.add_leaf(
        id="stock_a_nyse_and_telecom",
        desc="Stock A is listed on NYSE and classified in telecommunications industry",
        parent=identity_node,
        critical=True
    )
    company_ref = _first_nonempty(a.company_name, a.ticker)
    claim_nyse_telecom = (
        f"{company_ref} is listed on the New York Stock Exchange (NYSE) and is classified in the "
        f"telecommunications industry (communication services/telecom is acceptable)."
    )
    await evaluator.verify(
        claim=claim_nyse_telecom,
        node=node_nyse_telecom,
        sources=_combine_sources(a.identity_sources, a.general_sources, a.dividend_history_sources, a.dividend_yield_sources),
        additional_instruction=(
            "Allow reasonable synonyms and classifications (e.g., 'Communication Services' sector or "
            "'Telecommunications Services' industry counts as telecommunications). Verify NYSE listing explicitly."
        )
    )

    # 2) Dividend History (critical)
    hist_node = evaluator.add_parallel(
        id="stock_a_dividend_history",
        desc="Historical dividend increase record for Stock A",
        parent=stock_a_node,
        critical=True
    )

    # 2.1) At least 20 consecutive years (URL verification)
    node_20yrs = evaluator.add_leaf(
        id="stock_a_20yr_streak",
        desc="Stock A has increased dividends for at least 20 consecutive years",
        parent=hist_node,
        critical=True
    )
    claim_20yrs = f"{company_ref} has increased its dividend for at least 20 consecutive years."
    await evaluator.verify(
        claim=claim_20yrs,
        node=node_20yrs,
        sources=_combine_sources(a.dividend_history_sources, a.identity_sources, a.general_sources),
        additional_instruction="Look for 'dividend increase streak' or 'consecutive years of dividend increases' of 20+ years."
    )

    # 2.2) Specific number of consecutive years provided (existence check)
    evaluator.add_custom_node(
        result=bool(a.consecutive_increase_years and a.consecutive_increase_years.strip()),
        id="stock_a_year_count_reported",
        desc="Specific number of consecutive years provided for Stock A",
        parent=hist_node,
        critical=True
    )

    # 3) Current Dividend Yield (critical)
    yield_node = evaluator.add_parallel(
        id="stock_a_dividend_yield",
        desc="Current dividend yield meeting minimum threshold for Stock A",
        parent=stock_a_node,
        critical=True
    )

    # 3.1) Yield threshold >= 5% (URL verification)
    node_yield_thresh = evaluator.add_leaf(
        id="stock_a_yield_threshold",
        desc="Stock A dividend yield is at least 5%",
        parent=yield_node,
        critical=True
    )
    claim_yield_a = f"As of March 2026, the dividend yield of {company_ref} is at least 5%."
    await evaluator.verify(
        claim=claim_yield_a,
        node=node_yield_thresh,
        sources=_combine_sources(a.dividend_yield_sources, a.identity_sources, a.general_sources),
        additional_instruction=(
            "Use reputable sources (IR, major financial sites). Allow small rounding differences; focus on 'at least 5%'."
        )
    )

    # 3.2) Specific yield value provided (existence check)
    evaluator.add_custom_node(
        result=bool(a.dividend_yield and a.dividend_yield.strip()),
        id="stock_a_yield_value_reported",
        desc="Specific yield percentage provided for Stock A (as of March 2026)",
        parent=yield_node,
        critical=True
    )


async def verify_stock_b(evaluator: Evaluator, parent, b: StockBExtraction):
    # Parent node for Stock B
    stock_b_node = evaluator.add_parallel(
        id="stock_b",
        desc="Identification and verification of a qualifying asset management company",
        parent=parent,
        critical=False
    )

    # 1) Identity & Exchange (critical)
    identity_node = evaluator.add_parallel(
        id="stock_b_identity_exchange",
        desc="Basic identification information and exchange listing for Stock B",
        parent=stock_b_node,
        critical=True
    )

    # 1.1) Ticker & Name provided (existence)
    evaluator.add_custom_node(
        result=bool(b.ticker and b.ticker.strip()) and bool(b.company_name and b.company_name.strip()),
        id="stock_b_ticker_and_name",
        desc="Ticker symbol and full company name provided for Stock B",
        parent=identity_node,
        critical=True
    )

    # 1.2) NYSE & Asset Management classification (URL verification)
    node_nyse_asset = evaluator.add_leaf(
        id="stock_b_nyse_and_asset_mgmt",
        desc="Stock B is listed on NYSE and classified as asset management firm",
        parent=identity_node,
        critical=True
    )
    company_ref = _first_nonempty(b.company_name, b.ticker)
    claim_nyse_asset = (
        f"{company_ref} is listed on the New York Stock Exchange (NYSE) and is an asset management company."
    )
    await evaluator.verify(
        claim=claim_nyse_asset,
        node=node_nyse_asset,
        sources=_combine_sources(b.identity_sources, b.general_sources, b.platform_sources, b.aum_sources),
        additional_instruction="Confirm both NYSE listing and classification as an asset/ investment management firm."
    )

    # 2) Investment Platforms (critical)
    platforms_node = evaluator.add_parallel(
        id="stock_b_investment_platforms",
        desc="Verification of multiple investment platforms operated by Stock B",
        parent=stock_b_node,
        critical=True
    )

    # 2.1) At least 2 distinct platforms (URL verification)
    node_platform_count = evaluator.add_leaf(
        id="stock_b_platform_count",
        desc="Stock B operates at least 2 distinct investment platforms",
        parent=platforms_node,
        critical=True
    )
    list_platforms = _list_to_english(b.platforms)
    claim_platform_count = (
        f"{company_ref} operates at least two distinct investment platforms"
        + (f", including {list_platforms}." if list_platforms else ".")
    )
    await evaluator.verify(
        claim=claim_platform_count,
        node=node_platform_count,
        sources=_combine_sources(b.platform_sources, b.identity_sources, b.general_sources),
        additional_instruction=(
            "Examples of platforms include Credit, Real Assets, GP Strategic Capital, Insurance Solutions, etc. "
            "Confirm there are 2 or more distinct platforms."
        )
    )

    # 2.2) Names of specific platforms identified (URL verification)
    node_platform_names = evaluator.add_leaf(
        id="stock_b_platform_names",
        desc="Names of specific platforms identified (e.g., Credit, Real Assets, GP Strategic Capital)",
        parent=platforms_node,
        critical=True
    )
    claim_platform_names = (
        f"The investment platforms of {company_ref} include: {list_platforms}."
        if list_platforms else f"The investment platforms of {company_ref} are explicitly identified."
    )
    await evaluator.verify(
        claim=claim_platform_names,
        node=node_platform_names,
        sources=_combine_sources(b.platform_sources, b.identity_sources, b.general_sources),
        additional_instruction="Verify that the named platform categories appear on the company's official materials."
    )

    # 3) Assets Under Management (critical)
    aum_node = evaluator.add_parallel(
        id="stock_b_aum",
        desc="Total AUM meeting minimum threshold for Stock B",
        parent=stock_b_node,
        critical=True
    )

    # 3.1) AUM exceeds $200B (URL verification)
    node_aum_thresh = evaluator.add_leaf(
        id="stock_b_aum_threshold",
        desc="Stock B has total AUM exceeding $200 billion",
        parent=aum_node,
        critical=True
    )
    claim_aum = f"As of the most recent quarter, {company_ref}'s total Assets Under Management (AUM) exceed $200 billion."
    await evaluator.verify(
        claim=claim_aum,
        node=node_aum_thresh,
        sources=_combine_sources(b.aum_sources, b.identity_sources, b.general_sources),
        additional_instruction="Prefer the latest quarter investor materials or fact sheet. Threshold strictly > $200B."
    )

    # 3.2) Specific AUM amount provided (existence)
    evaluator.add_custom_node(
        result=bool(b.aum and b.aum.strip()),
        id="stock_b_aum_value_reported",
        desc="Specific AUM amount provided for Stock B",
        parent=aum_node,
        critical=True
    )

    # 4) Dividend Yield (critical)
    yield_node = evaluator.add_parallel(
        id="stock_b_dividend_yield",
        desc="Current dividend yield meeting minimum threshold for Stock B",
        parent=stock_b_node,
        critical=True
    )

    # 4.1) Yield threshold >= 9% (URL verification)
    node_yield_thresh = evaluator.add_leaf(
        id="stock_b_yield_threshold",
        desc="Stock B dividend yield is at least 9%",
        parent=yield_node,
        critical=True
    )
    claim_yield_b = f"As of March 2026, the dividend yield of {company_ref} is at least 9%."
    await evaluator.verify(
        claim=claim_yield_b,
        node=node_yield_thresh,
        sources=_combine_sources(b.dividend_yield_sources, b.identity_sources, b.general_sources),
        additional_instruction="Use reputable sources. Allow minor rounding; focus on 'at least 9%'."
    )

    # 4.2) Specific yield value provided (existence)
    evaluator.add_custom_node(
        result=bool(b.dividend_yield and b.dividend_yield.strip()),
        id="stock_b_yield_value_reported",
        desc="Specific yield percentage provided for Stock B (as of March 2026)",
        parent=yield_node,
        critical=True
    )


async def verify_stock_c(evaluator: Evaluator, parent, c: StockCExtraction, stock_b: StockBExtraction):
    # Parent node for Stock C
    stock_c_node = evaluator.add_parallel(
        id="stock_c",
        desc="Identification and verification of a qualifying BDC managed by Stock B's firm",
        parent=parent,
        critical=False
    )

    # 1) Identity & Exchange (critical)
    identity_node = evaluator.add_parallel(
        id="stock_c_identity_exchange",
        desc="Basic identification information and exchange listing for Stock C",
        parent=stock_c_node,
        critical=True
    )

    # 1.1) Ticker & Name provided (existence)
    evaluator.add_custom_node(
        result=bool(c.ticker and c.ticker.strip()) and bool(c.company_name and c.company_name.strip()),
        id="stock_c_ticker_and_name",
        desc="Ticker symbol and full company name provided for Stock C",
        parent=identity_node,
        critical=True
    )

    # 1.2) NYSE & BDC structure (URL verification)
    node_nyse_bdc = evaluator.add_leaf(
        id="stock_c_nyse_and_bdc",
        desc="Stock C is listed on NYSE and structured as a Business Development Company",
        parent=identity_node,
        critical=True
    )
    company_ref = _first_nonempty(c.company_name, c.ticker)
    claim_nyse_bdc = f"{company_ref} is listed on the NYSE and structured as a Business Development Company (BDC)."
    await evaluator.verify(
        claim=claim_nyse_bdc,
        node=node_nyse_bdc,
        sources=_combine_sources(c.identity_sources, c.general_sources, c.distribution_sources),
        additional_instruction="Confirm both NYSE listing and BDC structure from credible sources."
    )

    # 2) Management relationship (critical)
    node_mgmt = evaluator.add_leaf(
        id="stock_c_management_relationship",
        desc="Verification that Stock C is managed by the same firm as Stock B",
        parent=stock_c_node,
        critical=True
    )
    b_name = _first_nonempty(stock_b.company_name, stock_b.ticker)
    mgmt_name = _first_nonempty(c.management_firm_name, stock_b.company_name, stock_b.ticker)
    claim_mgmt = f"{company_ref} is managed by {mgmt_name}, the same asset management firm as Stock B ({b_name})."
    await evaluator.verify(
        claim=claim_mgmt,
        node=node_mgmt,
        sources=_combine_sources(c.management_sources, c.identity_sources, c.general_sources),
        additional_instruction=(
            "Verify that the BDC is externally managed by the same firm as Stock B. "
            "Treat minor naming variations (e.g., with/without 'Inc.') as matching."
        )
    )

    # 3) Dividend Yield (critical)
    yield_node = evaluator.add_parallel(
        id="stock_c_dividend_yield",
        desc="Current dividend yield meeting minimum threshold for Stock C",
        parent=stock_c_node,
        critical=True
    )

    # 3.1) Yield threshold >= 12% (URL verification)
    node_yield_thresh = evaluator.add_leaf(
        id="stock_c_yield_threshold",
        desc="Stock C dividend yield is at least 12%",
        parent=yield_node,
        critical=True
    )
    claim_yield_c = f"As of March 2026, the dividend yield of {company_ref} is at least 12%."
    await evaluator.verify(
        claim=claim_yield_c,
        node=node_yield_thresh,
        sources=_combine_sources(c.dividend_yield_sources, c.identity_sources, c.general_sources),
        additional_instruction="Use reputable sources. Allow minor rounding; focus on 'at least 12%'."
    )

    # 3.2) Specific yield value provided (existence)
    evaluator.add_custom_node(
        result=bool(c.dividend_yield and c.dividend_yield.strip()),
        id="stock_c_yield_value_reported",
        desc="Specific yield percentage provided for Stock C (as of March 2026)",
        parent=yield_node,
        critical=True
    )

    # 4) Quarterly Distribution (critical)
    dist_node = evaluator.add_parallel(
        id="stock_c_quarterly_distribution",
        desc="Verification of quarterly dividend payment schedule and amount",
        parent=stock_c_node,
        critical=True
    )

    # 4.1) Quarterly schedule (URL verification)
    node_quarterly_sched = evaluator.add_leaf(
        id="stock_c_quarterly_schedule",
        desc="Stock C distributes dividends on a quarterly basis",
        parent=dist_node,
        critical=True
    )
    claim_quarterly = f"{company_ref} pays dividends on a quarterly schedule."
    await evaluator.verify(
        claim=claim_quarterly,
        node=node_quarterly_sched,
        sources=_combine_sources(c.distribution_sources, c.identity_sources, c.general_sources),
        additional_instruction="Confirm distribution frequency explicitly as quarterly."
    )

    # 4.2) Specific quarterly amount provided (existence)
    evaluator.add_custom_node(
        result=bool(c.quarterly_dividend_amount and c.quarterly_dividend_amount.strip()),
        id="stock_c_quarterly_amount",
        desc="Specific quarterly dividend amount per share provided for Stock C",
        parent=dist_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Portfolio-level verification                                                #
# --------------------------------------------------------------------------- #
async def verify_portfolio_requirements(evaluator: Evaluator, parent, p: PortfolioExtraction):
    portfolio_node = evaluator.add_parallel(
        id="portfolio_requirements",
        desc="Verification that portfolio allocation meets IRA contribution limits and equal distribution requirements",
        parent=parent,
        critical=False
    )

    # 1) IRA contribution limit compliance (critical)
    node_ira_compliance = evaluator.add_leaf(
        id="ira_contribution_limit_compliance",
        desc="Total investment amount does not exceed 2026 Roth IRA contribution limit",
        parent=portfolio_node,
        critical=True
    )
    # Build claim using stated limit if provided; otherwise directly state $7,500
    limit_text = _first_nonempty(p.ira_limit_amount, "$7,500")
    claim_ira = (
        f"The 2026 Roth IRA contribution limit for individuals under age 50 is {limit_text}, "
        f"so a total contribution of $7,500 is compliant."
    )
    await evaluator.verify(
        claim=claim_ira,
        node=node_ira_compliance,
        sources=_combine_sources(p.ira_sources),
        additional_instruction=(
            "Prefer authoritative sources (e.g., IRS) for the 2026 Roth IRA limit for individuals under 50. "
            "Verify that $7,500 does not exceed the stated limit."
        )
    )

    # 2) Equal allocation requirement (critical)
    node_equal_alloc = evaluator.add_leaf(
        id="equal_allocation_requirement",
        desc="Investment amount is equally distributed across all three stocks ($2,500 per stock)",
        parent=portfolio_node,
        critical=True
    )
    claim_equal = (
        "The portfolio allocates $2,500 to each of the three stocks, i.e., an equal distribution."
    )
    await evaluator.verify(
        claim=claim_equal,
        node=node_equal_alloc,
        sources=None,  # Check based on the answer text itself
        additional_instruction=(
            "Use the answer content to verify. If the answer states equal allocation across three stocks with total $7,500, "
            "it implies $2,500 per stock."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    # Initialize evaluator with a parallel root aggregation (non-critical)
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

    # Extract all entities concurrently
    task_a = evaluator.extract(
        prompt=prompt_extract_stock_a(),
        template_class=StockAExtraction,
        extraction_name="stock_a_extraction"
    )
    task_b = evaluator.extract(
        prompt=prompt_extract_stock_b(),
        template_class=StockBExtraction,
        extraction_name="stock_b_extraction"
    )
    task_c = evaluator.extract(
        prompt=prompt_extract_stock_c(),
        template_class=StockCExtraction,
        extraction_name="stock_c_extraction"
    )
    task_p = evaluator.extract(
        prompt=prompt_extract_portfolio(),
        template_class=PortfolioExtraction,
        extraction_name="portfolio_extraction"
    )

    stock_a, stock_b, stock_c, portfolio = await asyncio.gather(task_a, task_b, task_c, task_p)

    # Build the main rubric tree children
    # Parent node matching top-level rubric entry (non-critical parallel)
    portfolio_construction_node = evaluator.add_parallel(
        id="three_stock_ira_portfolio_construction",
        desc="Complete portfolio of three dividend-paying stocks meeting specified criteria for a 2026 Roth IRA",
        parent=root,
        critical=False
    )

    # Verify Stock A, B, C
    await verify_stock_a(evaluator, portfolio_construction_node, stock_a)
    await verify_stock_b(evaluator, portfolio_construction_node, stock_b)
    await verify_stock_c(evaluator, portfolio_construction_node, stock_c, stock_b)

    # Verify portfolio-level requirements
    await verify_portfolio_requirements(evaluator, portfolio_construction_node, portfolio)

    # Return structured evaluation summary
    return evaluator.get_summary()