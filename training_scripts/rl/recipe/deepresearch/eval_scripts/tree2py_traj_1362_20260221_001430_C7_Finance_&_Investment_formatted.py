import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dividend_aristocrats_portfolio_2026"
TASK_DESCRIPTION = (
    "I am building a dividend-focused investment portfolio and want to identify S&P 500 Dividend Aristocrat companies with strong fundamentals. "
    "Please find exactly 4 companies that meet ALL of the following criteria:\n\n"
    "- Must be a constituent of the S&P 500 index as of February 2026\n"
    "- Must be a Dividend Aristocrat (25+ consecutive years of dividend increases)\n"
    "- Must have a market capitalization of at least $100 billion as of February 2026\n"
    "- Must have a current dividend yield between 2.0% and 4.0%\n"
    "- Must be primarily listed on either the New York Stock Exchange (NYSE) or the NASDAQ Stock Market\n"
    "- Must pay dividends on a quarterly basis (four times per year)\n"
    "- The 4 companies must collectively represent at least 3 different business sectors\n\n"
    "For each of the 4 companies, provide the following information:\n"
    "1. Company name and stock ticker symbol\n"
    "2. Primary exchange listing (NYSE or NASDAQ)\n"
    "3. Business sector\n"
    "4. Current stock price per share\n"
    "5. Total annual dividend amount per share\n"
    "6. Current dividend yield (as a percentage)\n"
    "7. Next scheduled dividend payment date\n"
    "8. Company headquarters location (city and state)\n"
    "9. 52-week high stock price\n"
    "10. 52-week low stock price\n"
    "11. Institutional ownership percentage\n"
    "12. A direct reference URL to verify the company's information"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CompanyEntry(BaseModel):
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    exchange: Optional[str] = None  # Prefer "NYSE" or "NASDAQ" if provided
    sector: Optional[str] = None
    current_price: Optional[str] = None
    annual_dividend_per_share: Optional[str] = None
    dividend_yield_percent: Optional[str] = None
    next_dividend_date: Optional[str] = None
    headquarters: Optional[str] = None  # city and state
    week_52_high: Optional[str] = None
    week_52_low: Optional[str] = None
    institutional_ownership_percent: Optional[str] = None
    reference_url: Optional[str] = None

    # Helpful fields to support constraints verification
    market_cap: Optional[str] = None  # e.g., "$250B", "USD 250,000,000,000"
    dividend_frequency: Optional[str] = None  # e.g., "Quarterly", "4 per year"


class PortfolioExtraction(BaseModel):
    companies: List[CompanyEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_portfolio() -> str:
    return (
        "Extract all companies mentioned in the answer that are proposed for the S&P 500 Dividend Aristocrat portfolio. "
        "Return an array 'companies' with each element containing the following fields as strings (use null if missing):\n"
        "- company_name\n"
        "- ticker\n"
        "- exchange (prefer 'NYSE' or 'NASDAQ' exactly if present; otherwise return the provided exchange text)\n"
        "- sector\n"
        "- current_price\n"
        "- annual_dividend_per_share\n"
        "- dividend_yield_percent (e.g., '2.8%' or '2.8')\n"
        "- next_dividend_date\n"
        "- headquarters (city and state)\n"
        "- week_52_high\n"
        "- week_52_low\n"
        "- institutional_ownership_percent\n"
        "- reference_url (a direct verification URL explicitly shown in the answer)\n"
        "- market_cap (if available in the answer; otherwise null)\n"
        "- dividend_frequency (e.g., 'Quarterly', '4 per year'; otherwise null)\n\n"
        "GENERAL RULES:\n"
        "1) Extract only what is explicitly present in the answer; do not invent.\n"
        "2) If multiple URLs are shown for a company, choose one primary direct URL that best verifies the company's information.\n"
        "3) Preserve percentages and currency as strings; do not convert to numbers.\n"
        "4) If any field is not provided, set it to null.\n"
        "5) Extract all companies the answer lists; do not limit to 4 here (the evaluator will select the first 4)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(idx: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"][idx] if idx < 6 else f"#{idx + 1}"


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _company_display(company: CompanyEntry) -> str:
    name = company.company_name or "Unknown Company"
    ticker = company.ticker or "Unknown Ticker"
    return f"{name} ({ticker})"


# --------------------------------------------------------------------------- #
# Per-company verification                                                    #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    root_parent,
    company: CompanyEntry,
    index: int,
) -> None:
    """
    Build verification subtree for one company and run checks.
    Follows the rubric nodes and uses the provided reference URL for source-grounded verifications.
    """
    company_id = f"Company_{index + 1}"
    company_title = f"{_ordinal(index)} company meeting all specified criteria"

    # Create parent node for this company (parallel aggregation, non-critical at parent level)
    company_node = evaluator.add_parallel(
        id=company_id,
        desc=company_title,
        parent=root_parent,
        critical=False,
    )

    # ------------- Presence checks (critical) ----------------
    # Reference URL presence (critical, also becomes an auto-prerequisite for other leaf verifications)
    ref_present = evaluator.add_custom_node(
        result=_non_empty(company.reference_url),
        id=f"{company_id}_Reference_URL",
        desc=f"A direct reference URL to verify information is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # Ticker symbol provided
    evaluator.add_custom_node(
        result=_non_empty(company.ticker),
        id=f"{company_id}_Ticker_Symbol",
        desc=f"Stock ticker symbol is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # Current stock price provided
    evaluator.add_custom_node(
        result=_non_empty(company.current_price),
        id=f"{company_id}_Current_Price",
        desc=f"Current stock price per share is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # Annual dividend per share provided
    evaluator.add_custom_node(
        result=_non_empty(company.annual_dividend_per_share),
        id=f"{company_id}_Annual_Dividend",
        desc=f"Total annual dividend amount per share is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # Next dividend payment date provided
    evaluator.add_custom_node(
        result=_non_empty(company.next_dividend_date),
        id=f"{company_id}_Next_Payment_Date",
        desc=f"Next scheduled dividend payment date is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # Headquarters provided
    evaluator.add_custom_node(
        result=_non_empty(company.headquarters),
        id=f"{company_id}_Headquarters",
        desc=f"Company headquarters location (city and state) is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # 52-week high provided
    evaluator.add_custom_node(
        result=_non_empty(company.week_52_high),
        id=f"{company_id}_52Week_High",
        desc=f"52-week high stock price is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # 52-week low provided
    evaluator.add_custom_node(
        result=_non_empty(company.week_52_low),
        id=f"{company_id}_52Week_Low",
        desc=f"52-week low stock price is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # Institutional ownership percentage provided
    evaluator.add_custom_node(
        result=_non_empty(company.institutional_ownership_percent),
        id=f"{company_id}_Institutional_Ownership",
        desc=f"Institutional ownership percentage is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # Sector provided
    evaluator.add_custom_node(
        result=_non_empty(company.sector),
        id=f"{company_id}_Sector",
        desc=f"Business sector is provided for {_company_display(company)}",
        parent=company_node,
        critical=True,
    )

    # ------------- Constraint verifications (critical, with sources) ----------------
    ref_url = company.reference_url or None

    # S&P 500 membership as of February 2026
    sp500_node = evaluator.add_leaf(
        id=f"{company_id}_SP500_Member",
        desc=f"{_company_display(company)} is a constituent of the S&P 500 index as of February 2026",
        parent=company_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_company_display(company)} is included in the S&P 500 index (membership current as of February 2026).",
        node=sp500_node,
        sources=ref_url,
        additional_instruction=(
            "Use the referenced page to confirm S&P 500 membership. Accept official index pages, credible listings, "
            "or authoritative profiles that explicitly indicate S&P 500 membership. If the URL is irrelevant, outdated, "
            "or does not support membership, mark as not supported."
        ),
    )

    # Dividend Aristocrat status (25+ consecutive years of increases)
    aristocrat_node = evaluator.add_leaf(
        id=f"{company_id}_Dividend_Aristocrat",
        desc=f"{_company_display(company)} has increased dividends for at least 25 consecutive years (Dividend Aristocrat)",
        parent=company_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_company_display(company)} is a Dividend Aristocrat with at least 25 consecutive years of dividend increases.",
        node=aristocrat_node,
        sources=ref_url,
        additional_instruction=(
            "Confirm explicit inclusion in the 'S&P 500 Dividend Aristocrats' or a statement that the company has increased dividends "
            "for 25+ consecutive years. Prefer authoritative sources (S&P Dow Jones Indices, official investor relations, or reputable finance sites)."
        ),
    )

    # Market capitalization >= $100B as of Feb 2026
    marketcap_node = evaluator.add_leaf(
        id=f"{company_id}_Market_Cap",
        desc=f"{_company_display(company)} has a market cap of at least $100B as of February 2026",
        parent=company_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_company_display(company)} has market capitalization of at least $100 billion (as of February 2026).",
        node=marketcap_node,
        sources=ref_url,
        additional_instruction=(
            "Check the market capitalization value shown or implied on the referenced page. Allow reasonable rounding and currency formatting. "
            "If the page shows a current market cap below $100B or lacks market cap info, mark as not supported."
        ),
    )

    # Dividend yield between 2.0% and 4.0% (current)
    yield_node = evaluator.add_leaf(
        id=f"{company_id}_Dividend_Yield",
        desc=f"{_company_display(company)} has a current dividend yield between 2.0% and 4.0%",
        parent=company_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The current dividend yield for {_company_display(company)} is between 2.0% and 4.0%.",
        node=yield_node,
        sources=ref_url,
        additional_instruction=(
            "Use the referenced page to locate the current dividend yield. Accept small rounding differences (e.g., 1.99% approximates 2.0%? be conservative). "
            "If the yield is outside the range or cannot be found, mark as not supported."
        ),
    )

    # Primary exchange listing is NYSE or NASDAQ
    exchange_node = evaluator.add_leaf(
        id=f"{company_id}_Exchange",
        desc=f"{_company_display(company)} is primarily listed on NYSE or NASDAQ",
        parent=company_node,
        critical=True,
    )
    exchange_txt = company.exchange or "Unknown exchange"
    await evaluator.verify(
        claim=f"The primary listing for {_company_display(company)} is on {exchange_txt}, and it is either NYSE or NASDAQ.",
        node=exchange_node,
        sources=ref_url,
        additional_instruction=(
            "Confirm the main U.S. exchange for the company's primary listing (NYSE or NASDAQ). "
            "If the page indicates another primary exchange, or does not specify NYSE/NASDAQ, mark as not supported."
        ),
    )

    # Pays dividends quarterly (four times per year)
    quarterly_node = evaluator.add_leaf(
        id=f"{company_id}_Quarterly_Dividends",
        desc=f"{_company_display(company)} pays dividends quarterly (four times per year)",
        parent=company_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_company_display(company)} pays dividends quarterly (four times per year).",
        node=quarterly_node,
        sources=ref_url,
        additional_instruction=(
            "Look for explicit mention of 'quarterly dividends', 'four times per year', or a consistent schedule indicating four payments per year on the page. "
            "If frequency is unclear or inconsistent, mark as not supported."
        ),
    )


# --------------------------------------------------------------------------- #
# Sector diversity verification                                               #
# --------------------------------------------------------------------------- #
def add_sector_diversity_check(evaluator: Evaluator, root_parent, companies: List[CompanyEntry]) -> None:
    sectors = [c.sector.strip() for c in companies if _non_empty(c.sector)]
    distinct = set(sectors)
    result = len(distinct) >= 3

    evaluator.add_custom_node(
        result=result,
        id="Sector_Diversity",
        desc="The 4 companies collectively represent at least 3 different business sectors",
        parent=root_parent,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Dividend Aristocrat portfolio task and return a summary dict.
    """
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
        default_model=model,
    )

    # 1) Extract portfolio companies from the answer text
    extraction = await evaluator.extract(
        prompt=prompt_extract_portfolio(),
        template_class=PortfolioExtraction,
        extraction_name="portfolio_extraction",
    )

    # 2) Select exactly the first 4 companies; pad with empty entries if fewer
    companies = list(extraction.companies[:4])
    while len(companies) < 4:
        companies.append(CompanyEntry())

    # 3) Add sector diversity critical check at the root level
    add_sector_diversity_check(evaluator, root, companies)

    # 4) Build and verify each company subtree
    tasks = []
    for idx, comp in enumerate(companies):
        tasks.append(verify_company(evaluator, root, comp, idx))
    await asyncio.gather(*tasks)

    # 5) Return the final structured summary
    return evaluator.get_summary()