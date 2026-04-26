import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dividend_aristocrats_yield_3pct_march_2026"
TASK_DESCRIPTION = (
    "Identify four different companies that are currently S&P 500 Dividend Aristocrats "
    "(companies that have increased their dividends for at least 25 consecutive years) and have a current "
    "dividend yield of at least 3.0%. For each of the four companies, provide comprehensive information: "
    "basic identification (name, ticker, S&P 500 sector, and a confirming URL), Dividend Aristocrat status "
    "(current S&P 500 membership, >=25 consecutive years of dividend increases, and a confirming URL), current "
    "dividend info (yield >=3.0%, annualized dividend/share, next ex-dividend date, and a confirming URL), key "
    "financial metrics (market cap, TTM P/E, institutional ownership, with a confirming URL), and recent corporate "
    "activity (most recent quarterly earnings date and official investor relations URL). All information must be "
    "current as of March 2026 and verifiable via provided reference URLs."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CompanyBasic(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    sector: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)


class CompanyAristocrat(BaseModel):
    sp500_urls: List[str] = Field(default_factory=list)
    aristocrat_urls: List[str] = Field(default_factory=list)


class CompanyDividend(BaseModel):
    dividend_yield: Optional[str] = None        # Keep as string; could be "3.2%" or similar
    annual_dividend: Optional[str] = None       # Keep as string; e.g., "$5.00", "5.00", etc.
    ex_dividend_date: Optional[str] = None      # Keep as string; e.g., "2026-03-14", "Mar 14, 2026"
    dividend_urls: List[str] = Field(default_factory=list)


class CompanyFinancial(BaseModel):
    market_cap: Optional[str] = None
    pe_ratio_ttm: Optional[str] = None
    institutional_ownership: Optional[str] = None
    financial_urls: List[str] = Field(default_factory=list)


class CompanyRecent(BaseModel):
    latest_earnings_date: Optional[str] = None
    investor_relations_url: Optional[str] = None
    earnings_urls: List[str] = Field(default_factory=list)


class CompanyItem(BaseModel):
    basic: Optional[CompanyBasic] = None
    aristocrat: Optional[CompanyAristocrat] = None
    dividend: Optional[CompanyDividend] = None
    financial: Optional[CompanyFinancial] = None
    recent: Optional[CompanyRecent] = None


class CompaniesExtraction(BaseModel):
    companies: List[CompanyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return """
You will extract up to FOUR distinct companies described in the answer as current S&P 500 Dividend Aristocrats
(with ≥25 consecutive years of dividend increases) and having a current dividend yield of at least 3.0%.
Extract the first four if more than four are mentioned. If fewer than four are present, still return however many exist.

For each company, return an object with the following nested structure. IMPORTANT: 
- Extract ONLY what is explicitly present in the answer.
- Keep numbers and dates as strings exactly as they appear in the answer.
- For any missing field, return null (for a string) or an empty array (for lists).
- For URLs, extract only explicit URLs present in the answer (including markdown links). Do NOT invent URLs.
- If a URL is missing a protocol, prepend "http://".

Schema to output (JSON):

{
  "companies": [
    {
      "basic": {
        "name": string|null,
        "ticker": string|null,
        "sector": string|null,                           // S&P 500 (GICS) sector as stated in the answer
        "identification_urls": [string, ...]            // URL(s) confirming identification details (name/ticker/sector)
      },
      "aristocrat": {
        "sp500_urls": [string, ...],                    // URL(s) supporting current S&P 500 membership
        "aristocrat_urls": [string, ...]                // URL(s) supporting Dividend Aristocrat (>=25 years raises)
      },
      "dividend": {
        "dividend_yield": string|null,                  // e.g., "3.2%"
        "annual_dividend": string|null,                 // e.g., "$5.00" or "5.00"
        "ex_dividend_date": string|null,                // e.g., "2026-03-14" or "Mar 14, 2026"
        "dividend_urls": [string, ...]                  // URL(s) used to confirm yield, annual dividend, ex-div date
      },
      "financial": {
        "market_cap": string|null,                      // e.g., "$100B", "$99.5B"
        "pe_ratio_ttm": string|null,                    // e.g., "15.2", "15.2x"
        "institutional_ownership": string|null,         // e.g., "65%", "65.1%"
        "financial_urls": [string, ...]                 // URL(s) that display these metrics
      },
      "recent": {
        "latest_earnings_date": string|null,            // most recent quarterly earnings report date as quoted
        "investor_relations_url": string|null,          // official IR homepage URL, if given
        "earnings_urls": [string, ...]                  // URL(s) used to confirm the latest quarterly earnings date
      }
    }
  ]
}

Notes:
- Do not add or infer values. If not in the answer, leave as null or [].
- Preserve the exact textual formatting for numbers/dates as presented.
- Return strictly valid JSON following the schema above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(v: Optional[str], default: str = "(missing)") -> str:
    return v.strip() if isinstance(v, str) and v.strip() else default


def _gather_urls(*url_lists: Optional[List[str]]) -> List[str]:
    """Flatten, clean and de-duplicate multiple URL lists while preserving order."""
    seen = set()
    result: List[str] = []
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    result.append(u2)
    return result


# --------------------------------------------------------------------------- #
# Verification logic per company                                              #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    parent_node,
    company: CompanyItem,
    idx: int,
) -> None:
    """
    Build the verification subtree for a single company and run all checks per rubric.
    """
    tag = f"company_{idx + 1}"

    # Create top-level node for this company (parallel; non-critical as per rubric)
    company_node = evaluator.add_parallel(
        id=tag,
        desc=[
            "First qualifying company: S&P 500 Dividend Aristocrat with yield ≥3.0%",
            "Second qualifying company: S&P 500 Dividend Aristocrat with yield ≥3.0%",
            "Third qualifying company: S&P 500 Dividend Aristocrat with yield ≥3.0%",
            "Fourth qualifying company: S&P 500 Dividend Aristocrat with yield ≥3.0%",
        ][idx],
        parent=parent_node,
        critical=False,
    )

    # Normalize all nested structures
    basic = company.basic or CompanyBasic()
    arist = company.aristocrat or CompanyAristocrat()
    div = company.dividend or CompanyDividend()
    fin = company.financial or CompanyFinancial()
    rec = company.recent or CompanyRecent()

    # --------------------------- Identification --------------------------- #
    ident_node = evaluator.add_parallel(
        id=f"{tag}_identification",
        desc="Basic company identification information",
        parent=company_node,
        critical=True,  # rubric: critical
    )

    # Gate: require at least one identification URL
    ident_url_gate = evaluator.add_custom_node(
        result=bool(basic.identification_urls),
        id=f"{tag}_identification_url",
        desc="Provide reference URL confirming company identification details",
        parent=ident_node,
        critical=True,
    )

    # Create identification detail leaves (critical)
    name_leaf = evaluator.add_leaf(
        id=f"{tag}_name",
        desc="Provide the official company name",
        parent=ident_node,
        critical=True,
    )
    ticker_leaf = evaluator.add_leaf(
        id=f"{tag}_ticker",
        desc="Provide the stock ticker symbol",
        parent=ident_node,
        critical=True,
    )
    sector_leaf = evaluator.add_leaf(
        id=f"{tag}_sector",
        desc="Provide the S&P 500 sector classification",
        parent=ident_node,
        critical=True,
    )

    # Prepare claims for identification
    name_claim = f"The official company name is '{_safe(basic.name)}'."
    ticker_context = _safe(basic.name, "the company")
    ticker_claim = f"The stock ticker symbol for {ticker_context} is '{_safe(basic.ticker)}'."
    sector_claim = f"The S&P 500 (GICS) sector classification for {ticker_context} is '{_safe(basic.sector)}'."

    await evaluator.batch_verify(
        [
            (
                name_claim,
                basic.identification_urls,
                name_leaf,
                "Match the official company name on the provided identification page(s). Allow minor variations such as "
                "Inc. vs Incorporated, Co. vs Company, punctuation, ampersand vs 'and', and casing differences."
            ),
            (
                ticker_claim,
                basic.identification_urls,
                ticker_leaf,
                "Verify the stock ticker symbol as shown on the provided page(s). Accept variants like 'NYSE: TICKER', "
                "'Nasdaq: TICKER', or similar exchange notation."
            ),
            (
                sector_claim,
                basic.identification_urls,
                sector_leaf,
                "Verify the S&P 500 (GICS) sector classification for the company. Allow reasonable synonyms or formatting "
                "differences (e.g., 'Information Technology' vs 'Technology') but ensure the sector meaning matches."
            ),
        ]
    )

    # --------------------- Dividend Aristocrat Status --------------------- #
    arist_node = evaluator.add_parallel(
        id=f"{tag}_dividend_aristocrat_status",
        desc="Verification of Dividend Aristocrat qualification",
        parent=company_node,
        critical=True,  # rubric: critical
    )

    # Gate: at least one URL supporting Aristocrat/SP500
    arist_url_gate = evaluator.add_custom_node(
        result=bool(_gather_urls(arist.aristocrat_urls, arist.sp500_urls)),
        id=f"{tag}_aristocrat_url",
        desc="Provide reference URL confirming Dividend Aristocrat status",
        parent=arist_node,
        critical=True,
    )

    sp500_leaf = evaluator.add_leaf(
        id=f"{tag}_sp500_membership",
        desc="Verify current membership in S&P 500 Index",
        parent=arist_node,
        critical=True,
    )
    consec_leaf = evaluator.add_leaf(
        id=f"{tag}_consecutive_years",
        desc="Verify dividend increases for ≥25 consecutive years",
        parent=arist_node,
        critical=True,
    )

    sp500_claim = f"{_safe(basic.name, 'This company')} is currently a constituent of the S&P 500 Index."
    consec_claim = f"{_safe(basic.name, 'This company')} has increased its dividend for at least 25 consecutive years."

    arist_sources = _gather_urls(arist.sp500_urls, arist.aristocrat_urls)

    await evaluator.batch_verify(
        [
            (
                sp500_claim,
                arist_sources,
                sp500_leaf,
                "The provided page(s) should state that the company is a current S&P 500 constituent (as of March 2026). "
                "Accept explicit membership listings or authoritative index/constituent references."
            ),
            (
                consec_claim,
                arist.aristocrat_urls,
                consec_leaf,
                "The page(s) should explicitly confirm Dividend Aristocrat status and/or state ≥25 consecutive years of "
                "dividend increases. Accept authoritative lists of 'S&P 500 Dividend Aristocrats' as evidence."
            ),
        ]
    )

    # -------------------------- Dividend Details -------------------------- #
    dividend_node = evaluator.add_parallel(
        id=f"{tag}_dividend_details",
        desc="Current dividend information",
        parent=company_node,
        critical=True,  # rubric: critical
    )

    # Gate: require at least one dividend info URL
    dividend_url_gate = evaluator.add_custom_node(
        result=bool(div.dividend_urls),
        id=f"{tag}_dividend_url",
        desc="Provide reference URL for dividend information",
        parent=dividend_node,
        critical=True,
    )

    yield_leaf = evaluator.add_leaf(
        id=f"{tag}_dividend_yield",
        desc="Provide current dividend yield (must be ≥3.0%)",
        parent=dividend_node,
        critical=True,
    )
    annual_div_leaf = evaluator.add_leaf(
        id=f"{tag}_annual_dividend",
        desc="Provide annualized dividend per share amount",
        parent=dividend_node,
        critical=True,
    )
    exdiv_leaf = evaluator.add_leaf(
        id=f"{tag}_ex_dividend_date",
        desc="Provide next ex-dividend date",
        parent=dividend_node,
        critical=True,
    )

    # Claims for dividend details
    subject = f"{_safe(basic.name)} ({_safe(basic.ticker)})" if basic.ticker or basic.name else "the company"
    if div.dividend_yield and div.dividend_yield.strip():
        yield_claim = f"The current dividend yield for {subject} is '{_safe(div.dividend_yield)}' and it is at least 3.0%."
    else:
        yield_claim = f"The current dividend yield for {subject} shown on the provided page(s) is at least 3.0%."

    annual_div_claim = f"The annualized dividend per share for {subject} is '{_safe(div.annual_dividend)}'."
    exdiv_claim = f"The next ex-dividend date for {subject} is '{_safe(div.ex_dividend_date)}'."

    await evaluator.batch_verify(
        [
            (
                yield_claim,
                div.dividend_urls,
                yield_leaf,
                "Confirm that the page(s) display a 'Dividend Yield' of at least 3.0%. If multiple yield figures appear, "
                "prefer the standard/current dividend yield. Allow rounding differences."
            ),
            (
                annual_div_claim,
                div.dividend_urls,
                annual_div_leaf,
                "Confirm the annualized (forward) dividend per share as shown on the page(s). Allow minor formatting or "
                "currency symbol variations."
            ),
            (
                exdiv_claim,
                div.dividend_urls,
                exdiv_leaf,
                "Confirm the next ex-dividend date. If multiple ex-dividend dates are listed, choose the next upcoming one "
                "as of March 2026."
            ),
        ]
    )

    # ------------------------- Financial Metrics -------------------------- #
    fin_node = evaluator.add_parallel(
        id=f"{tag}_financial_metrics",
        desc="Key financial metrics",
        parent=company_node,
        critical=False,  # rubric: non-critical
    )

    fin_url_gate = evaluator.add_custom_node(
        result=bool(fin.financial_urls),
        id=f"{tag}_financial_url",
        desc="Provide reference URL for financial metrics",
        parent=fin_node,
        critical=False,
    )

    mcap_leaf = evaluator.add_leaf(
        id=f"{tag}_market_cap",
        desc="Provide current market capitalization",
        parent=fin_node,
        critical=False,
    )
    pe_leaf = evaluator.add_leaf(
        id=f"{tag}_pe_ratio",
        desc="Provide trailing twelve-month P/E ratio",
        parent=fin_node,
        critical=False,
    )
    inst_own_leaf = evaluator.add_leaf(
        id=f"{tag}_institutional_ownership",
        desc="Provide institutional ownership percentage",
        parent=fin_node,
        critical=False,
    )

    mcap_claim = f"The current market capitalization for {subject} is '{_safe(fin.market_cap)}'."
    pe_claim = f"The trailing twelve-month (TTM) P/E ratio for {subject} is '{_safe(fin.pe_ratio_ttm)}'."
    inst_own_claim = f"The institutional ownership percentage for {subject} is '{_safe(fin.institutional_ownership)}'."

    await evaluator.batch_verify(
        [
            (
                mcap_claim,
                fin.financial_urls,
                mcap_leaf,
                "Verify the market cap as shown on the page(s). Allow equivalent formatting (e.g., $100B vs $100,000,000,000) "
                "and minor rounding differences."
            ),
            (
                pe_claim,
                fin.financial_urls,
                pe_leaf,
                "Verify the TTM P/E ratio as shown. Accept reasonable formatting differences (e.g., '15.2' vs '15.2x'). "
                "Ensure it is clearly identified as TTM or equivalent."
            ),
            (
                inst_own_claim,
                fin.financial_urls,
                inst_own_leaf,
                "Verify the institutional ownership percentage as shown on the page(s). Allow rounding differences."
            ),
        ]
    )

    # ----------------------- Recent Corporate Activity -------------------- #
    recent_node = evaluator.add_parallel(
        id=f"{tag}_recent_activity",
        desc="Recent corporate activity information",
        parent=company_node,
        critical=False,  # rubric: non-critical
    )

    # Latest earnings date
    latest_earnings_leaf = evaluator.add_leaf(
        id=f"{tag}_latest_earnings_date",
        desc="Provide date of most recent quarterly earnings report",
        parent=recent_node,
        critical=False,
    )

    earnings_sources = _gather_urls(rec.earnings_urls, [rec.investor_relations_url] if rec.investor_relations_url else [])
    earnings_claim = f"The most recent quarterly earnings report date for {subject} is '{_safe(rec.latest_earnings_date)}'."

    await evaluator.verify(
        claim=earnings_claim,
        node=latest_earnings_leaf,
        sources=earnings_sources,
        additional_instruction="Verify the most recent quarter's earnings report date on the provided page(s). Prefer "
                               "official IR earnings press releases or earnings calendar entries."
    )

    # Investor Relations URL (verify if provided; otherwise fail this non-critical leaf)
    if rec.investor_relations_url and rec.investor_relations_url.strip():
        ir_leaf = evaluator.add_leaf(
            id=f"{tag}_investor_relations_url",
            desc="Provide official investor relations website URL",
            parent=recent_node,
            critical=False,
        )
        ir_claim = f"This URL is the official Investor Relations website for {_safe(basic.name, 'the company')}."
        await evaluator.verify(
            claim=ir_claim,
            node=ir_leaf,
            sources=rec.investor_relations_url,
            additional_instruction="Confirm that the page clearly represents the company's official Investor Relations site "
                                   "(e.g., header/footer text, company branding, 'Investor Relations' labeling)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{tag}_investor_relations_url",
            desc="Provide official investor relations website URL",
            parent=recent_node,
            critical=False,
        )


# --------------------------------------------------------------------------- #
# Optional: root-level uniqueness check (non-critical)                        #
# --------------------------------------------------------------------------- #
def add_uniqueness_check(evaluator: Evaluator, parent_node, companies: List[CompanyItem]) -> None:
    """Check that the companies are distinct by ticker (case-insensitive) when provided."""
    tickers = []
    for c in companies:
        t = (c.basic.ticker.strip().lower() if c and c.basic and c.basic.ticker else None)
        if t:
            tickers.append(t)
    unique = len(set(tickers)) == len(tickers) if tickers else False
    evaluator.add_custom_node(
        result=unique,
        id="unique_companies_by_ticker",
        desc="All provided companies are distinct by ticker (ignoring case) when tickers are present",
        parent=parent_node,
        critical=False,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point for evaluating an answer to the Dividend Aristocrats (≥3.0% yield) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()

    # Note: root set to non-critical to allow non-critical child nodes per framework constraint
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

    # Extract companies
    extracted = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=CompaniesExtraction,
        extraction_name="companies_extraction",
    )

    # Keep exactly 4 entries (pad if fewer)
    companies = list(extracted.companies[:4])
    while len(companies) < 4:
        companies.append(CompanyItem())

    # Build and verify each company subtree
    verify_tasks = []
    for i in range(4):
        verify_tasks.append(verify_company(evaluator, root, companies[i], i))
    await asyncio.gather(*verify_tasks)

    # Optional uniqueness check
    add_uniqueness_check(evaluator, root, companies)

    # Return the evaluation summary
    return evaluator.get_summary()