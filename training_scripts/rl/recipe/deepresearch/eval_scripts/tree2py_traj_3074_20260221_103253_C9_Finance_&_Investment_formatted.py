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
TASK_ID = "dividend_aristocrats_2026"
TASK_DESCRIPTION = """
Identify four S&P 500 Dividend Aristocrat companies that meet all specified dividend, financial, and operational criteria for investment consideration in 2026.

You must ensure each of the four companies satisfies:
1) Dividend Aristocrat status (25+ consecutive years of increases),
2) Dividend yield ≥ 3.0%,
3) Dividend payout ratio < 75%,
4) Quarterly dividend payment frequency,
5) A dividend increase announced/implemented during 2026,
6) TTM P/E ratio between 10 and 25,
7) Market capitalization ≥ $10B (USD),
8) Consensus analyst rating "Moderate Buy" or better,
9) Q1 2026 earnings scheduled between Apr 1 and May 31, 2026,
10) Listed on NYSE or NASDAQ,
11) Collectively represent at least three different S&P 500 sectors.

For each company, the answer should include:
- Company name
- Stock ticker symbol
- S&P 500 sector classification
- Current dividend yield (%)
- Dividend payout ratio (%)
- Number of consecutive years of dividend increases
- Date of 2026 dividend increase announcement
- Trailing twelve-month P/E ratio
- Market capitalization (in billions USD)
- Analyst consensus rating
- Q1 2026 earnings report date
- Stock exchange (NYSE or NASDAQ)
- A reference URL (IR page or major finance site) confirming the above information
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CompanyItem(BaseModel):
    # Core identification
    name: Optional[str] = None
    ticker: Optional[str] = None
    sector: Optional[str] = None
    exchange: Optional[str] = None  # Expect "NYSE" or "NASDAQ" or synonymous variant
    
    # Dividend Aristocrat / years
    consecutive_years_increase: Optional[str] = None  # Keep as string for flexibility
    aristocrat_url: Optional[str] = None

    # Dividend metrics
    dividend_yield_percent: Optional[str] = None
    dividend_yield_url: Optional[str] = None
    payout_ratio_percent: Optional[str] = None
    payout_ratio_url: Optional[str] = None
    payment_frequency: Optional[str] = None  # e.g., "quarterly"
    payment_frequency_url: Optional[str] = None
    dividend_increase_2026_date: Optional[str] = None
    dividend_increase_2026_url: Optional[str] = None

    # Financial health
    pe_ratio_ttm: Optional[str] = None
    pe_ratio_url: Optional[str] = None
    market_cap_billion_usd: Optional[str] = None
    market_cap_url: Optional[str] = None

    # Analyst rating
    analyst_rating: Optional[str] = None
    analyst_rating_url: Optional[str] = None

    # Earnings schedule
    earnings_q1_2026_date: Optional[str] = None
    earnings_schedule_url: Optional[str] = None

    # Identification / membership URL
    identification_url: Optional[str] = None

    # Extra URLs (fallbacks)
    extra_urls: List[str] = Field(default_factory=list)


class CompaniesExtraction(BaseModel):
    companies: List[CompanyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return """
    Extract up to four companies described in the answer that are presented as S&P 500 Dividend Aristocrats satisfying the required investment criteria for 2026. For each company, extract the following fields exactly as they appear in the answer:

    Required fields per company:
    - name: Company name
    - ticker: Stock ticker symbol
    - sector: S&P 500 sector classification (e.g., Industrials, Consumer Staples, Utilities, Financials, Energy, Healthcare, etc.)
    - exchange: The stock exchange ("NYSE" or "NASDAQ")
    - consecutive_years_increase: Number of consecutive years of dividend increases
    - aristocrat_url: URL confirming Dividend Aristocrat status or 25+ years of increases
    - dividend_yield_percent: Current dividend yield percentage (string)
    - dividend_yield_url: URL showing the current dividend yield
    - payout_ratio_percent: Dividend payout ratio percentage (string)
    - payout_ratio_url: URL showing the payout ratio
    - payment_frequency: Dividend payment frequency (e.g., "quarterly")
    - payment_frequency_url: URL showing dividend payment schedule/frequency
    - dividend_increase_2026_date: Date in 2026 when a dividend increase was announced or implemented
    - dividend_increase_2026_url: URL announcing or documenting the 2026 dividend increase
    - pe_ratio_ttm: Trailing twelve-month P/E ratio (string)
    - pe_ratio_url: URL showing the TTM P/E ratio
    - market_cap_billion_usd: Market capitalization in billions of USD (string)
    - market_cap_url: URL showing market capitalization
    - analyst_rating: Consensus analyst rating (e.g., "Moderate Buy", "Strong Buy")
    - analyst_rating_url: URL showing the analyst consensus rating
    - earnings_q1_2026_date: Q1 2026 earnings report date
    - earnings_schedule_url: URL confirming the earnings schedule/date
    - identification_url: URL used for company identification or S&P 500 membership (IR page or major finance site)
    - extra_urls: Any additional URLs cited in the answer that pertain to the company (array of strings)

    Rules:
    - Extract only what is explicitly in the answer. If a field is not provided, set it to null.
    - For URL fields, extract the actual URL string (including protocol). Do NOT invent URLs.
    - If more than four companies are provided, include only the first four as they appear in the answer.
    - If fewer than four companies are provided, include only those that appear.

    Return a JSON object with one field:
    - companies: an array of company objects with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(company: CompanyItem, primary: Optional[str], include_ident: bool = True) -> List[str]:
    """Build a source list combining a primary URL, identification URL, and extra_urls."""
    urls: List[str] = []
    for u in [primary]:
        if u and isinstance(u, str) and u.strip():
            urls.append(u.strip())
    if include_ident and company.identification_url and company.identification_url.strip():
        urls.append(company.identification_url.strip())
    # Add extra URLs
    for u in company.extra_urls:
        if u and isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


# --------------------------------------------------------------------------- #
# Verification per company                                                    #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    parent_node,
    company: CompanyItem,
    index: int,
) -> None:
    # ---------------- Identification (Critical / Parallel) ---------------- #
    ident_node = evaluator.add_parallel(
        id=f"Company_{index}_Identification",
        desc="Provide company name, ticker symbol, and S&P 500 sector classification",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(company.name) and _nonempty(company.ticker),
        id=f"Company_{index}_Name_Ticker",
        desc="Company name and stock ticker symbol are provided",
        parent=ident_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(company.sector),
        id=f"Company_{index}_Sector",
        desc="S&P 500 sector classification is provided",
        parent=ident_node,
        critical=True,
    )

    # Exchange verification via URL
    exch_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Exchange",
        desc="Company is listed on NYSE or NASDAQ",
        parent=ident_node,
        critical=True,
    )
    exch_claim = f"The company '{company.name or ''}' (ticker '{company.ticker or ''}') is listed on {company.exchange or 'NYSE/NASDAQ'}."
    await evaluator.verify(
        claim=exch_claim,
        node=exch_leaf,
        sources=_collect_sources(company, company.identification_url, include_ident=True),
        additional_instruction="Confirm listing exchange. Accept synonymous naming (e.g., 'New York Stock Exchange' or 'Nasdaq'). The page should clearly indicate the listing.",
    )

    # Identification / S&P 500 membership URL support
    ident_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Identification_URL",
        desc="URL reference for company identification and S&P 500 membership",
        parent=ident_node,
        critical=True,
    )
    ident_url_claim = "This page is an authoritative source (IR or major finance site) for the company's identification and ideally confirms S&P 500 membership."
    await evaluator.verify(
        claim=ident_url_claim,
        node=ident_url_leaf,
        sources=_collect_sources(company, company.identification_url, include_ident=False),
        additional_instruction="Prefer the company's investor relations page or a major finance site (e.g., S&P Global, NASDAQ, NYSE, SEC, Yahoo Finance, Bloomberg, Morningstar). If explicit S&P 500 membership is not stated but the page is clearly authoritative for identification, consider it acceptable.",
    )

    # -------- Dividend Aristocrat Status (Critical / Sequential) ---------- #
    arist_node = evaluator.add_sequential(
        id=f"Company_{index}_Dividend_Aristocrat_Status",
        desc="Verify the company is a Dividend Aristocrat with 25+ consecutive years of dividend increases",
        parent=parent_node,
        critical=True,
    )

    arist_verify_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Dividend_Aristocrat_Verification",
        desc="Company has increased dividends for at least 25 consecutive years",
        parent=arist_node,
        critical=True,
    )
    arist_claim = "The company has increased its dividend for at least 25 consecutive years (Dividend Aristocrat qualification)."
    await evaluator.verify(
        claim=arist_claim,
        node=arist_verify_leaf,
        sources=_collect_sources(company, company.aristocrat_url, include_ident=True),
        additional_instruction="Look for explicit mention of 'Dividend Aristocrat' or documentation of 25+ consecutive years of increases. Accept credible IR pages and respected finance sites.",
    )

    evaluator.add_custom_node(
        result=_nonempty(company.consecutive_years_increase),
        id=f"Company_{index}_Consecutive_Years_Count",
        desc="The specific number of consecutive years of dividend increases is provided",
        parent=arist_node,
        critical=True,
    )

    arist_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Dividend_Aristocrat_URL",
        desc="URL reference confirming Dividend Aristocrat status",
        parent=arist_node,
        critical=True,
    )
    arist_url_claim = "This page confirms Dividend Aristocrat status or clearly documents 25+ consecutive dividend increases."
    await evaluator.verify(
        claim=arist_url_claim,
        node=arist_url_leaf,
        sources=_collect_sources(company, company.aristocrat_url, include_ident=False),
        additional_instruction="Verify that the page supports Aristocrat status or the ≥25 consecutive years claim.",
    )

    # --------------- Dividend Metrics (Critical / Parallel) --------------- #
    divm_node = evaluator.add_parallel(
        id=f"Company_{index}_Dividend_Metrics",
        desc="Verify dividend yield, payout ratio, payment frequency, and recent increase",
        parent=parent_node,
        critical=True,
    )

    # Dividend Yield sub-node
    dy_node = evaluator.add_parallel(
        id=f"Company_{index}_Dividend_Yield",
        desc="Current dividend yield is at least 3.0%",
        parent=divm_node,
        critical=True,
    )
    dy_val_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Dividend_Yield_Value",
        desc="Dividend yield percentage is provided and meets 3.0% minimum",
        parent=dy_node,
        critical=True,
    )
    dy_claim = (
        f"The company's current dividend yield is at least 3.0%."
        + (f" The reported yield is {company.dividend_yield_percent}." if _nonempty(company.dividend_yield_percent) else "")
    )
    await evaluator.verify(
        claim=dy_claim,
        node=dy_val_leaf,
        sources=_collect_sources(company, company.dividend_yield_url, include_ident=True),
        additional_instruction="Confirm the page shows the dividend yield and that it is ≥ 3.0%. Allow minor rounding differences.",
    )
    dy_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Dividend_Yield_URL",
        desc="URL reference for dividend yield data",
        parent=dy_node,
        critical=True,
    )
    dy_url_claim = "This page displays the company's current dividend yield."
    await evaluator.verify(
        claim=dy_url_claim,
        node=dy_url_leaf,
        sources=_collect_sources(company, company.dividend_yield_url, include_ident=False),
        additional_instruction="Ensure the page contains dividend yield information for the specified company/ticker.",
    )

    # Payout Ratio sub-node
    pr_node = evaluator.add_parallel(
        id=f"Company_{index}_Payout_Ratio",
        desc="Dividend payout ratio is below 75%",
        parent=divm_node,
        critical=True,
    )
    pr_val_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Payout_Ratio_Value",
        desc="Payout ratio percentage is provided and below 75%",
        parent=pr_node,
        critical=True,
    )
    pr_claim = (
        f"The company's dividend payout ratio is below 75%."
        + (f" The reported payout ratio is {company.payout_ratio_percent}." if _nonempty(company.payout_ratio_percent) else "")
    )
    await evaluator.verify(
        claim=pr_claim,
        node=pr_val_leaf,
        sources=_collect_sources(company, company.payout_ratio_url, include_ident=True),
        additional_instruction="Confirm payout ratio is < 75%. Allow minor rounding differences and standard payout ratio definitions.",
    )
    pr_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Payout_Ratio_URL",
        desc="URL reference for payout ratio data",
        parent=pr_node,
        critical=True,
    )
    pr_url_claim = "This page displays the company's dividend payout ratio."
    await evaluator.verify(
        claim=pr_url_claim,
        node=pr_url_leaf,
        sources=_collect_sources(company, company.payout_ratio_url, include_ident=False),
        additional_instruction="Ensure the page contains payout ratio information for the specified company/ticker.",
    )

    # Payment Frequency sub-node
    pf_node = evaluator.add_parallel(
        id=f"Company_{index}_Payment_Frequency",
        desc="Company pays quarterly dividends",
        parent=divm_node,
        critical=True,
    )
    pf_ver_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Payment_Frequency_Verification",
        desc="Dividend payment frequency is quarterly",
        parent=pf_node,
        critical=True,
    )
    pf_claim = "The company pays dividends quarterly (four payments per year)."
    await evaluator.verify(
        claim=pf_claim,
        node=pf_ver_leaf,
        sources=_collect_sources(company, company.payment_frequency_url, include_ident=True),
        additional_instruction="Look for 'quarterly dividend' or payment schedule showing four distributions per year.",
    )
    pf_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Payment_Frequency_URL",
        desc="URL reference for dividend payment schedule",
        parent=pf_node,
        critical=True,
    )
    pf_url_claim = "This page shows the company's dividend payment schedule or frequency."
    await evaluator.verify(
        claim=pf_url_claim,
        node=pf_url_leaf,
        sources=_collect_sources(company, company.payment_frequency_url, include_ident=False),
        additional_instruction="Ensure the page documents the frequency/schedule of dividend payments.",
    )

    # 2026 Dividend Increase sub-node
    di_node = evaluator.add_parallel(
        id=f"Company_{index}_2026_Dividend_Increase",
        desc="Company announced or implemented a dividend increase in 2026",
        parent=divm_node,
        critical=True,
    )
    di_ver_leaf = evaluator.add_leaf(
        id=f"Company_{index}_2026_Increase_Verification",
        desc="Dividend increase in 2026 is confirmed with announcement date",
        parent=di_node,
        critical=True,
    )
    di_claim = (
        "In 2026, the company announced or implemented a dividend increase."
        + (f" The announcement date provided is {company.dividend_increase_2026_date}." if _nonempty(company.dividend_increase_2026_date) else "")
    )
    await evaluator.verify(
        claim=di_claim,
        node=di_ver_leaf,
        sources=_collect_sources(company, company.dividend_increase_2026_url, include_ident=True),
        additional_instruction="Confirm a dividend increase occurred in calendar year 2026. Investor relations press releases or credible finance/news sites are acceptable.",
    )
    di_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_2026_Increase_URL",
        desc="URL reference for 2026 dividend increase announcement",
        parent=di_node,
        critical=True,
    )
    di_url_claim = "This page announces or documents a dividend increase in 2026 for the company."
    await evaluator.verify(
        claim=di_url_claim,
        node=di_url_leaf,
        sources=_collect_sources(company, company.dividend_increase_2026_url, include_ident=False),
        additional_instruction="Verify that this page is specifically about a 2026 dividend increase announcement or implementation.",
    )

    # ------------ Financial Health (Critical / Parallel) ------------------ #
    fin_node = evaluator.add_parallel(
        id=f"Company_{index}_Financial_Health",
        desc="Verify P/E ratio and market capitalization meet requirements",
        parent=parent_node,
        critical=True,
    )

    # PE Ratio sub-node
    pe_node = evaluator.add_parallel(
        id=f"Company_{index}_PE_Ratio",
        desc="Trailing twelve-month P/E ratio is between 10 and 25",
        parent=fin_node,
        critical=True,
    )
    pe_val_leaf = evaluator.add_leaf(
        id=f"Company_{index}_PE_Ratio_Value",
        desc="P/E ratio is provided and within 10-25 range",
        parent=pe_node,
        critical=True,
    )
    pe_claim = (
        "The company's trailing twelve-month (TTM) P/E ratio is between 10 and 25."
        + (f" The reported P/E is {company.pe_ratio_ttm}." if _nonempty(company.pe_ratio_ttm) else "")
    )
    await evaluator.verify(
        claim=pe_claim,
        node=pe_val_leaf,
        sources=_collect_sources(company, company.pe_ratio_url, include_ident=True),
        additional_instruction="Confirm the TTM P/E ratio falls within [10, 25]. Allow minor rounding; ensure TTM context.",
    )
    pe_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_PE_Ratio_URL",
        desc="URL reference for P/E ratio data",
        parent=pe_node,
        critical=True,
    )
    pe_url_claim = "This page displays the company's trailing twelve-month P/E ratio."
    await evaluator.verify(
        claim=pe_url_claim,
        node=pe_url_leaf,
        sources=_collect_sources(company, company.pe_ratio_url, include_ident=False),
        additional_instruction="Ensure the page contains TTM P/E information for the specified company/ticker.",
    )

    # Market Cap sub-node
    mc_node = evaluator.add_parallel(
        id=f"Company_{index}_Market_Cap",
        desc="Market capitalization is at least $10 billion",
        parent=fin_node,
        critical=True,
    )
    mc_val_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Market_Cap_Value",
        desc="Market cap value is provided and meets $10B minimum",
        parent=mc_node,
        critical=True,
    )
    mc_claim = (
        "The company's market capitalization is at least $10 billion (USD)."
        + (f" The reported market cap is {company.market_cap_billion_usd} billion USD." if _nonempty(company.market_cap_billion_usd) else "")
    )
    await evaluator.verify(
        claim=mc_claim,
        node=mc_val_leaf,
        sources=_collect_sources(company, company.market_cap_url, include_ident=True),
        additional_instruction="Confirm market cap ≥ $10B. If the page reports in billions, ensure the value is ≥ 10. Allow minor rounding.",
    )
    mc_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Market_Cap_URL",
        desc="URL reference for market capitalization data",
        parent=mc_node,
        critical=True,
    )
    mc_url_claim = "This page displays the company's market capitalization."
    await evaluator.verify(
        claim=mc_url_claim,
        node=mc_url_leaf,
        sources=_collect_sources(company, company.market_cap_url, include_ident=False),
        additional_instruction="Ensure the page contains market cap information for the specified company/ticker.",
    )

    # ----------- Analyst Rating (Critical / Sequential) ------------------- #
    rating_node = evaluator.add_sequential(
        id=f"Company_{index}_Analyst_Rating",
        desc="Verify analyst consensus rating is Moderate Buy or better",
        parent=parent_node,
        critical=True,
    )
    rating_ver_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Rating_Verification",
        desc="Consensus analyst rating is Moderate Buy, Strong Buy, or equivalent positive rating",
        parent=rating_node,
        critical=True,
    )
    rating_claim = (
        "The company's consensus analyst rating is 'Moderate Buy' or better."
        + (f" The reported rating is {company.analyst_rating}." if _nonempty(company.analyst_rating) else "")
    )
    await evaluator.verify(
        claim=rating_claim,
        node=rating_ver_leaf,
        sources=_collect_sources(company, company.analyst_rating_url, include_ident=True),
        additional_instruction="Confirm rating wording indicates Moderate Buy or Strong Buy (or equivalent).",
    )
    rating_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Rating_URL",
        desc="URL reference for analyst consensus rating",
        parent=rating_node,
        critical=True,
    )
    rating_url_claim = "This page reports the company's analyst consensus rating."
    await evaluator.verify(
        claim=rating_url_claim,
        node=rating_url_leaf,
        sources=_collect_sources(company, company.analyst_rating_url, include_ident=False),
        additional_instruction="Ensure the page includes the consensus analyst rating for the specified company/ticker.",
    )

    # --------- Earnings Schedule (Critical / Sequential) ------------------ #
    earn_node = evaluator.add_sequential(
        id=f"Company_{index}_Earnings_Schedule",
        desc="Verify Q1 2026 earnings report is scheduled between April 1 and May 31, 2026",
        parent=parent_node,
        critical=True,
    )
    earn_date_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Earnings_Date",
        desc="Q1 2026 earnings date is provided and falls within April 1 - May 31, 2026",
        parent=earn_node,
        critical=True,
    )
    earn_claim = (
        "The company's Q1 2026 earnings report is scheduled between 2026-04-01 and 2026-05-31."
        + (f" The provided date is {company.earnings_q1_2026_date}." if _nonempty(company.earnings_q1_2026_date) else "")
    )
    await evaluator.verify(
        claim=earn_claim,
        node=earn_date_leaf,
        sources=_collect_sources(company, company.earnings_schedule_url, include_ident=True),
        additional_instruction="Confirm Q1 2026 earnings date falls within Apr 1–May 31, 2026. Earnings calendar pages or IR event pages are acceptable.",
    )
    earn_url_leaf = evaluator.add_leaf(
        id=f"Company_{index}_Earnings_URL",
        desc="URL reference for earnings calendar or schedule",
        parent=earn_node,
        critical=True,
    )
    earn_url_claim = "This page shows the company's earnings calendar or event schedule, including Q1 2026."
    await evaluator.verify(
        claim=earn_url_claim,
        node=earn_url_leaf,
        sources=_collect_sources(company, company.earnings_schedule_url, include_ident=False),
        additional_instruction="Ensure the page documents the relevant earnings schedule/date.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Dividend Aristocrats 2026 investment criteria task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across companies & sector diversification
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

    # Extract companies
    extraction = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=CompaniesExtraction,
        extraction_name="companies_extraction",
    )

    # Normalize to exactly 4 entries (padding with empty placeholders if needed)
    companies: List[CompanyItem] = list(extraction.companies[:4])
    while len(companies) < 4:
        companies.append(CompanyItem())

    # Build verification subtrees for each company
    for i, company in enumerate(companies, start=1):
        company_node = evaluator.add_sequential(
            id=f"Company_{i}",
            desc=[
                "First qualifying company meeting all criteria",
                "Second qualifying company meeting all criteria",
                "Third qualifying company meeting all criteria",
                "Fourth qualifying company meeting all criteria",
            ][i - 1],
            parent=root,
            critical=False,  # Allow partial credit per company; inner sections enforce critical criteria
        )
        await verify_company(evaluator, company_node, company, i)

    # Sector diversification check (Critical / Sequential with one leaf)
    sectors = [c.sector.strip() for c in companies if _nonempty(c.sector)]
    distinct_sector_count = len(set(sectors))
    sector_node = evaluator.add_sequential(
        id="Sector_Diversification",
        desc="Verify the four companies represent at least three different S&P 500 sectors",
        parent=root,
        critical=True,
    )
    evaluator.add_custom_node(
        result=distinct_sector_count >= 3,
        id="Sector_Count_Verification",
        desc="At least three distinct S&P 500 sectors are represented among the four companies",
        parent=sector_node,
        critical=True,
    )

    # Return the evaluation summary
    return evaluator.get_summary()