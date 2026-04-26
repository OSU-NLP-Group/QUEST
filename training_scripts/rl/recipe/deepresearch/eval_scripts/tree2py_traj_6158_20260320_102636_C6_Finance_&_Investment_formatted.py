import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "recent_ipos_2024_2025_with_requirements"
TASK_DESCRIPTION = """
I am conducting research on recently public companies for a potential investment portfolio. I need to identify three distinct companies that meet the following criteria:

1. IPO Timing: The company completed its initial public offering (IPO) between January 1, 2024, and December 31, 2025.
2. Exchange Listing: The company is currently listed on either the New York Stock Exchange (NYSE) or NASDAQ.
3. Market Capitalization: The company's current market capitalization is between $1 billion and $10 billion.
4. Share Price: The company's current stock price is at least $1.00 per share (meeting NASDAQ's minimum bid price requirement for continued listing).
5. Public Float: The market value of the company's publicly held shares meets the minimum requirement for its exchange ($15 million for NYSE American or $18 million for NASDAQ).
6. Institutional Ownership: At least one institutional investor has filed a Schedule 13D or Schedule 13G with the SEC, disclosing beneficial ownership of 5% or more of the company's outstanding voting shares.
7. SEC Compliance: The company has filed its required Form S-1 for the IPO, and if more than 45 days have passed since the end of the first complete fiscal quarter after the IPO, the company has filed at least one Form 10-Q quarterly report.

For each of the three companies, please provide:
- Company name and stock ticker symbol
- IPO date and listing exchange
- Current market capitalization and stock price
- Name of at least one institutional investor holding 5% or more
- Official reference URLs from SEC EDGAR for: (a) Form S-1 filing, (b) Schedule 13D/13G filing showing institutional ownership, and (c) recent periodic filings (10-Q or 10-K)
- Any additional reference URL confirming current market data
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CompanyItem(BaseModel):
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    ipo_date: Optional[str] = None  # Keep as text to be robust (e.g., "May 2, 2024")
    exchange: Optional[str] = None  # e.g., "NASDAQ", "NYSE"
    market_cap: Optional[str] = None  # e.g., "$3.2B"
    stock_price: Optional[str] = None  # e.g., "$12.34"
    public_float_market_value: Optional[str] = None  # e.g., "$200M" or textual description

    institutional_investor_name: Optional[str] = None
    institutional_ownership_percent: Optional[str] = None  # e.g., "7.2%"

    s1_urls: List[str] = Field(default_factory=list)  # SEC S-1 or F-1
    ownership_urls: List[str] = Field(default_factory=list)  # SEC 13D/13G URLs
    periodic_filing_urls: List[str] = Field(default_factory=list)  # 10-Q or 10-K
    ipo_doc_urls: List[str] = Field(default_factory=list)  # confirms IPO date + exchange listing
    financial_data_urls: List[str] = Field(default_factory=list)  # market data (price, cap, float)


class IPOCompaniesExtraction(BaseModel):
    companies: List[CompanyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return """
Extract up to three distinct companies from the answer that the author claims meet these constraints:
- IPO date between Jan 1, 2024 and Dec 31, 2025 (inclusive).
- Currently listed on NYSE or NASDAQ.
- Current market capitalization between $1B and $10B.
- Current stock price at least $1.00.
- Public float market value meets or exceeds the exchange minimum (≥$15M for NYSE American, ≥$18M for NASDAQ).
- At least one institutional investor has filed a Schedule 13D or 13G disclosing ≥5% beneficial ownership.
- SEC filings compliance: Form S-1 (or F-1) filed; if applicable based on time since IPO, at least one 10-Q filed.

For each company, extract the following fields exactly as written in the answer (use strings; do not transform units):
- company_name
- ticker (uppercase if given)
- ipo_date (any date format appearing in the answer)
- exchange (e.g., "NASDAQ", "NYSE")
- market_cap (string, e.g., "$3.2B")
- stock_price (string, e.g., "$12.34")
- public_float_market_value (string, if available, e.g., "$200M")
- institutional_investor_name (e.g., "BlackRock, Inc.")
- institutional_ownership_percent (e.g., "7.2%")

Also extract the following URL lists (include only explicit URLs shown in the answer; if none are provided, return an empty list):
- s1_urls: SEC EDGAR links to Form S-1 or F-1 or S-1/A (registration statements)
- ownership_urls: SEC EDGAR links to Schedule 13D, 13D/A, 13G, or 13G/A filings for ≥5% ownership
- periodic_filing_urls: SEC EDGAR links to recent Form 10-Q or 10-K (company filings page or direct filing)
- ipo_doc_urls: official URLs confirming IPO date and exchange (EDGAR, listing exchange site, or company IR)
- financial_data_urls: market data sources for current market cap and price (e.g., Nasdaq.com, NYSE.com, Yahoo Finance, Bloomberg, company IR)

Rules:
- Return up to the first 3 companies explicitly discussed. If more are present, include only the first three.
- If any field is missing from the answer, set it to null (or an empty list for URL fields).
- Do not invent data or URLs.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_len(urls: Optional[List[str]]) -> int:
    return len(urls) if urls else 0


def _norm_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _threshold_for_float(exchange: Optional[str]) -> str:
    """
    Returns the applicable minimum public float market value threshold as a string.
    We use $18M for NASDAQ, $15M for NYSE American; if unknown or NYSE main board, default to $15M for this task.
    """
    ex = (_norm_str(exchange) or "").upper()
    if "NAS" in ex:  # NASDAQ, NASDAQGS, NASDAQGM etc.
        return "$18 million"
    return "$15 million"


def _today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    parent_node,
    company: CompanyItem,
    idx: int,
) -> None:
    """
    Build the full verification subtree for a single company.
    """

    # Company container (parallel; allow partial scoring within company)
    company_node = evaluator.add_parallel(
        id=f"Company_{idx+1}",
        desc=f"Company #{idx+1} meets all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # ------------------- Company Identification ------------------- #
    ident_node = evaluator.add_parallel(
        id=f"company_{idx}_identification",
        desc="Provide basic company identification information",
        parent=company_node,
        critical=False,
    )

    evaluator.add_custom_node(
        result=bool(_norm_str(company.company_name)),
        id=f"company_{idx}_name_provided",
        desc="The full company name is provided",
        parent=ident_node,
        critical=False,
    )

    evaluator.add_custom_node(
        result=bool(_norm_str(company.ticker)),
        id=f"company_{idx}_ticker_provided",
        desc="The stock ticker symbol is provided",
        parent=ident_node,
        critical=False,
    )

    # ------------------- IPO and Listing Verification -------------- #
    ipo_node = evaluator.add_parallel(
        id=f"company_{idx}_ipo_and_listing",
        desc="Verify the company's IPO occurred within the specified timeframe and on a qualifying exchange",
        parent=company_node,
        critical=False,  # Keep non-critical to allow optional URL leaf; critical checks live as leaves
    )

    # IPO date valid (2024-01-01 to 2025-12-31)
    ipo_date_leaf = evaluator.add_leaf(
        id=f"company_{idx}_ipo_date_valid",
        desc="The company's IPO date falls between January 1, 2024 and December 31, 2025 (inclusive)",
        parent=ipo_node,
        critical=True,
    )
    ipo_date_claim = (
        f"The company's IPO date is '{_norm_str(company.ipo_date)}', and it falls between January 1, 2024 and December 31, 2025 (inclusive). "
        "If the IPO date is missing or outside the range, mark this as Incorrect."
    )
    await evaluator.verify(
        claim=ipo_date_claim,
        node=ipo_date_leaf,
        sources=company.ipo_doc_urls,  # ground on sources if provided
        additional_instruction="Use any provided IPO/exchange/SEC sources to confirm. If no source is provided and the date cannot be confirmed, mark Incorrect.",
    )

    # Listed on qualifying exchange (NYSE or NASDAQ) - current status
    list_leaf = evaluator.add_leaf(
        id=f"company_{idx}_listed_on_exchange",
        desc="The company is currently listed on either the New York Stock Exchange (NYSE) or NASDAQ",
        parent=ipo_node,
        critical=True,
    )
    exchange_text = _norm_str(company.exchange) or "NYSE or NASDAQ"
    list_claim = (
        f"The ticker '{_norm_str(company.ticker)}' is currently listed on the {exchange_text}."
    )
    await evaluator.verify(
        claim=list_claim,
        node=list_leaf,
        sources=(company.ipo_doc_urls + company.financial_data_urls),
        additional_instruction="Confirm CURRENT listing on NYSE or NASDAQ from reliable sources (exchange website, Nasdaq/NYSE listing page, or similar).",
    )

    # Form S-1 (or F-1) registration filed
    s1_leaf = evaluator.add_leaf(
        id=f"company_{idx}_form_s1_filed",
        desc="The company filed SEC Form S-1 (or Form F-1 for foreign private issuers) registration statement as part of its IPO process",
        parent=ipo_node,
        critical=True,
    )
    s1_claim = "There exists an SEC registration statement for this company on Form S-1 or Form F-1 (including any amendments like S-1/A or F-1/A)."
    await evaluator.verify(
        claim=s1_claim,
        node=s1_leaf,
        sources=company.s1_urls,
        additional_instruction="Verify via the provided SEC EDGAR links. If no registration statement URL is provided, mark as Not Supported.",
    )

    # IPO documentation URL confirms IPO date & listing (non-critical)
    ipo_doc_leaf = evaluator.add_leaf(
        id=f"company_{idx}_ipo_doc_url",
        desc="Provide official URL source confirming IPO date and exchange listing (from SEC EDGAR, exchange website, or company investor relations)",
        parent=ipo_node,
        critical=False,
    )
    ipo_doc_claim = (
        "At least one provided source explicitly confirms the company's IPO date and the exchange on which it is listed."
    )
    await evaluator.verify(
        claim=ipo_doc_claim,
        node=ipo_doc_leaf,
        sources=company.ipo_doc_urls,
        additional_instruction="Fail if no URL is provided. Accept SEC EDGAR filings, official exchange listings, or company IR announcements.",
    )

    # ------------------- Financial Requirements -------------------- #
    fin_node = evaluator.add_parallel(
        id=f"company_{idx}_financial_requirements",
        desc="Verify the company meets market capitalization, share price, and public float requirements",
        parent=company_node,
        critical=False,
    )

    # Market cap between $1B and $10B
    mc_leaf = evaluator.add_leaf(
        id=f"company_{idx}_market_cap_range",
        desc="Current market capitalization is between $1 billion and $10 billion",
        parent=fin_node,
        critical=True,
    )
    mc_claim = "The company's current market capitalization falls between $1 billion and $10 billion (inclusive)."
    await evaluator.verify(
        claim=mc_claim,
        node=mc_leaf,
        sources=company.financial_data_urls,
        additional_instruction="Use numeric values on the referenced market data page(s); allow standard rounding and $B/$M units.",
    )

    # Share price at least $1.00
    price_leaf = evaluator.add_leaf(
        id=f"company_{idx}_min_share_price",
        desc="Current stock price is at least $1.00 per share",
        parent=fin_node,
        critical=True,
    )
    price_claim = "The company's current stock price is at least $1.00 per share."
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=company.financial_data_urls,
        additional_instruction="Use the live/most recent price on reputable market data pages (e.g., Nasdaq.com, NYSE.com, Yahoo Finance).",
    )

    # Public float requirement satisfied
    float_leaf = evaluator.add_leaf(
        id=f"company_{idx}_public_float_req",
        desc="Market value of publicly held shares meets the exchange's minimum requirement (at least $15 million for NYSE American or $18 million for NASDAQ)",
        parent=fin_node,
        critical=True,
    )
    threshold_text = _threshold_for_float(company.exchange)
    float_claim = (
        f"The company's public float market value meets or exceeds the minimum requirement for its exchange (i.e., {threshold_text})."
    )
    await evaluator.verify(
        claim=float_claim,
        node=float_leaf,
        sources=(company.financial_data_urls + company.s1_urls + company.periodic_filing_urls),
        additional_instruction=(
            "Prefer explicit 'public float' figures. If unavailable, infer conservatively from reliable disclosures (e.g., shares held by non‑affiliates × recent price) "
            "as stated in SEC filings or reputable sources. If there is insufficient evidence, mark as Not Supported."
        ),
    )

    # Financial data URL present and relevant (non-critical)
    fin_url_leaf = evaluator.add_leaf(
        id=f"company_{idx}_financial_data_url",
        desc="Provide official URL source for current market capitalization, share price, and public float data",
        parent=fin_node,
        critical=False,
    )
    fin_url_claim = "At least one provided URL is a reliable source for current market capitalization and stock price for this company."
    await evaluator.verify(
        claim=fin_url_claim,
        node=fin_url_leaf,
        sources=company.financial_data_urls,
        additional_instruction="Fail if no URL is provided. Accept exchange sites (NASDAQ/NYSE), company IR, or reputable finance portals (Yahoo Finance, Bloomberg, etc.).",
    )

    # ------------------- Ownership and Compliance ------------------ #
    own_node = evaluator.add_parallel(
        id=f"company_{idx}_ownership_compliance",
        desc="Verify institutional ownership disclosure and SEC reporting compliance",
        parent=company_node,
        critical=False,
    )

    # Institutional Ownership Disclosure
    own_disc_node = evaluator.add_parallel(
        id=f"company_{idx}_institutional_disclosure",
        desc="At least one institutional investor has disclosed ownership of 5% or more of outstanding voting shares",
        parent=own_node,
        critical=False,
    )

    sched_leaf = evaluator.add_leaf(
        id=f"company_{idx}_schedule_13x_filed",
        desc="A Schedule 13D or Schedule 13G has been filed with the SEC documenting institutional ownership of 5% or more",
        parent=own_disc_node,
        critical=True,
    )
    sched_claim = "At least one SEC Schedule 13D or 13G filing discloses beneficial ownership of 5% or more for this company."
    await evaluator.verify(
        claim=sched_claim,
        node=sched_leaf,
        sources=company.ownership_urls,
        additional_instruction="Verify using the provided SEC EDGAR 13D/13G URLs (including amendments). Fail if no SEC 13D/13G link is provided.",
    )

    holder_leaf = evaluator.add_leaf(
        id=f"company_{idx}_institutional_holder_identified",
        desc="The institutional investor's name and ownership percentage are clearly stated",
        parent=own_disc_node,
        critical=True,
    )
    holder_claim = (
        f"The institutional investor '{_norm_str(company.institutional_investor_name)}' holds at least "
        f"{_norm_str(company.institutional_ownership_percent)} (≥5%) of the company's outstanding voting shares, as shown in the Schedule 13D/13G."
    )
    await evaluator.verify(
        claim=holder_claim,
        node=holder_leaf,
        sources=company.ownership_urls,
        additional_instruction="Allow 'more than 5%' wording. If the name or percent cannot be confirmed in the filing, mark as Incorrect.",
    )

    own_url_leaf = evaluator.add_leaf(
        id=f"company_{idx}_ownership_filing_url",
        desc="Provide SEC EDGAR URL for the Schedule 13D or 13G filing",
        parent=own_disc_node,
        critical=False,
    )
    own_url_claim = "At least one provided URL is an SEC EDGAR link to a Schedule 13D or Schedule 13G filing for this company."
    await evaluator.verify(
        claim=own_url_claim,
        node=own_url_leaf,
        sources=company.ownership_urls,
        additional_instruction="Fail if no URL is provided.",
    )

    # SEC Quarterly Reporting
    sec_q_node = evaluator.add_parallel(
        id=f"company_{idx}_sec_quarterly_reporting",
        desc="Company has filed required Form 10-Q if applicable based on time elapsed since IPO",
        parent=own_node,
        critical=False,
    )

    q_required_leaf = evaluator.add_leaf(
        id=f"company_{idx}_form_10q_if_required",
        desc="If more than 45 days have passed since the end of the first complete fiscal quarter after IPO, at least one Form 10-Q has been filed",
        parent=sec_q_node,
        critical=True,
    )
    q_required_claim = (
        f"Given the IPO date '{_norm_str(company.ipo_date)}' and today's date {_today_str()}, "
        "the company has filed at least one Form 10-Q if required (i.e., more than 45 days have passed since the end of the first complete fiscal quarter after IPO)."
    )
    await evaluator.verify(
        claim=q_required_claim,
        node=q_required_leaf,
        sources=company.periodic_filing_urls,
        additional_instruction=(
            "Use the SEC EDGAR company filings page or direct 10-Q links to confirm. If the time requirement is not met yet, this can still be marked Correct."
        ),
    )

    recent_current_leaf = evaluator.add_leaf(
        id=f"company_{idx}_recent_filings_current",
        desc="The most recently required periodic filing (10-Q or 10-K) has been filed and is up to date",
        parent=sec_q_node,
        critical=True,
    )
    recent_current_claim = "The company's most recent required periodic filing (10-Q or 10-K) appears current and up to date."
    await evaluator.verify(
        claim=recent_current_claim,
        node=recent_current_leaf,
        sources=company.periodic_filing_urls,
        additional_instruction="Confirm with SEC EDGAR links. Consider standard SEC deadlines; if overdue, mark Incorrect.",
    )

    sec_urls_leaf = evaluator.add_leaf(
        id=f"company_{idx}_sec_filings_url",
        desc="Provide SEC EDGAR URL showing the company's recent periodic filings",
        parent=sec_q_node,
        critical=False,
    )
    sec_urls_claim = "At least one provided URL is an SEC EDGAR page that shows the company's recent periodic filings (10-Q/10-K)."
    await evaluator.verify(
        claim=sec_urls_claim,
        node=sec_urls_leaf,
        sources=company.periodic_filing_urls,
        additional_instruction="Fail if no URL is provided.",
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
        default_model=model,
    )

    # Extract companies
    extraction = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=IPOCompaniesExtraction,
        extraction_name="recent_ipos_extraction",
    )

    # Keep exactly 3 slots (pad if fewer)
    companies: List[CompanyItem] = list(extraction.companies[:3])
    while len(companies) < 3:
        companies.append(CompanyItem())

    # Add distinctness check (critical gating at root)
    seen_tickers = []
    seen_names = []
    for c in companies:
        if _norm_str(c.ticker):
            seen_tickers.append(_norm_str(c.ticker).upper())
        if _norm_str(c.company_name):
            seen_names.append(_norm_str(c.company_name).lower())

    unique_by_ticker = len([t for t in seen_tickers if t]) == len(set([t for t in seen_tickers if t]))
    unique_by_name = len([n for n in seen_names if n]) == len(set([n for n in seen_names if n]))
    distinct_ok = unique_by_ticker and unique_by_name and len(companies) == 3

    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinct_companies",
        desc="Three provided companies are distinct by ticker/name",
        parent=root,
        critical=True,
    )

    # Build the rubric root node
    rubric_root = evaluator.add_parallel(
        id="Find_Three_Companies",
        desc="Identify three distinct companies that completed their IPO between January 2024 and December 2025 on NYSE or NASDAQ, meeting specified market capitalization, institutional ownership, and SEC compliance requirements",
        parent=root,
        critical=False,  # Adjusted to allow partial credit and avoid child critical consistency constraints
    )

    # Verify each company subtree
    for i, comp in enumerate(companies):
        await verify_company(evaluator, rubric_root, comp, i)

    # Return structured summary
    return evaluator.get_summary()