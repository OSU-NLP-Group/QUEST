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
TASK_ID = "dividend_aristocrats_portfolio"
TASK_DESCRIPTION = """
You are conducting investment research for a retirement portfolio focused on dividend growth stocks. Your objective is to identify four S&P 500 Dividend Aristocrat companies for diversified sector exposure, selecting one company from each of the following four sectors: Information Technology, Health Care, Financials, and Consumer Staples.

For each of the four companies you select, provide the following comprehensive investment analysis:

1. Dividend Aristocrat Verification: Confirm the company is an official S&P 500 Dividend Aristocrat (has increased dividends for 25+ consecutive years and is a current S&P 500 member)
2. Sector Classification: Verify the company belongs to the specified sector according to GICS (Global Industry Classification Standard)
3. Market Capitalization: Confirm the company's current market capitalization meets or exceeds the S&P 500 minimum threshold of $22.7 billion
4. Current Dividend Yield: Provide the company's current annual dividend yield as a percentage
5. Lowest-Cost ETF Access: Identify which ETF among the major providers (BlackRock/iShares, Vanguard, or State Street/SPDR) that holds this company has the lowest expense ratio, and state that expense ratio
6. Institutional Ownership: Identify at least one major institutional investor that holds this company according to recent Form 13F filings
7. Additional Index Memberships: Identify at least one other S&P index (beyond the S&P 500) that includes this company
8. Aristocrat Achievement Year: Determine the year when the company achieved Dividend Aristocrat status (the 25th consecutive year of dividend increases)
9. S&P 500 Index Weight: Provide the company's current approximate weight or percentage allocation in the S&P 500 index
10. Primary Exchange Listing: Identify the stock exchange where the company's shares are primarily listed
11. Dividend Payment Frequency: Specify how often the company pays dividends (e.g., quarterly, monthly, annually)
12. Supporting References: Provide reference URLs to verify key information, particularly Dividend Aristocrat status and current financial metrics

Your analysis should enable an investor to make informed decisions about portfolio construction with proper sector diversification across established dividend-growth companies.
"""

SNP500_MIN_MARKET_CAP_USD_BN = 22.7

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFInfo(BaseModel):
    provider: Optional[str] = None  # One of: iShares/BlackRock, Vanguard, SPDR/State Street
    name: Optional[str] = None
    expense_ratio: Optional[str] = None  # Keep string to allow formats like "0.03%" or "0.03"
    url: Optional[str] = None           # ETF product page
    holdings_url: Optional[str] = None  # ETF holdings page (optional)


class CompanyAnalysis(BaseModel):
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    sector: Optional[str] = None

    # Key metrics and sources
    market_cap: Optional[str] = None
    market_cap_source_url: Optional[str] = None

    dividend_yield_pct: Optional[str] = None
    dividend_yield_source_url: Optional[str] = None

    # Dividend Aristocrat
    aristocrat_status_url: Optional[str] = None
    aristocrat_achievement_year: Optional[str] = None

    # Sector source
    sector_source_url: Optional[str] = None

    # ETF info
    etfs: List[ETFInfo] = Field(default_factory=list)
    etf_lowest_provider: Optional[str] = None
    etf_lowest_name: Optional[str] = None
    etf_lowest_expense_ratio: Optional[str] = None
    etf_lowest_url: Optional[str] = None
    etf_lowest_holdings_url: Optional[str] = None

    # Institutional ownership
    institutional_investor: Optional[str] = None
    institutional_url: Optional[str] = None

    # Additional S&P index membership
    additional_index_name: Optional[str] = None
    additional_index_url: Optional[str] = None

    # S&P 500 weight
    sp500_weight_pct: Optional[str] = None
    sp500_weight_source_url: Optional[str] = None

    # Exchange listing
    primary_exchange: Optional[str] = None
    exchange_source_url: Optional[str] = None

    # Dividend frequency
    dividend_frequency: Optional[str] = None
    dividend_frequency_source_url: Optional[str] = None

    # General references
    reference_urls: List[str] = Field(default_factory=list)


class PortfolioExtraction(BaseModel):
    company_technology: Optional[CompanyAnalysis] = None
    company_healthcare: Optional[CompanyAnalysis] = None
    company_financials: Optional[CompanyAnalysis] = None
    company_consumer_staples: Optional[CompanyAnalysis] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_portfolio() -> str:
    return """
    Extract the complete portfolio analysis for four S&P 500 Dividend Aristocrat companies, one from each sector:
    - Information Technology
    - Health Care
    - Financials
    - Consumer Staples

    For each selected company, extract the following fields exactly as presented in the answer text. If any field is missing, return null (or an empty list for list fields). Use strings for numbers to accommodate formats like "0.03%" or "22.7B".

    Per company fields:
    - company_name
    - ticker
    - sector
    - market_cap
    - market_cap_source_url
    - dividend_yield_pct
    - dividend_yield_source_url
    - aristocrat_status_url  (a URL that verifies Dividend Aristocrat status)
    - aristocrat_achievement_year (the year the 25th consecutive increase was achieved)
    - sector_source_url
    - etfs: an array where each item has:
        * provider (BlackRock/iShares, Vanguard, or SPDR/State Street)
        * name
        * expense_ratio
        * url
        * holdings_url (if available)
    - etf_lowest_provider
    - etf_lowest_name
    - etf_lowest_expense_ratio
    - etf_lowest_url
    - etf_lowest_holdings_url (if available)
    - institutional_investor (e.g., BlackRock, Vanguard, State Street, Fidelity, etc.)
    - institutional_url (a URL to a recent Form 13F listing or a reputable holdings summary page)
    - additional_index_name (e.g., S&P 100, sector-specific S&P indices)
    - additional_index_url
    - sp500_weight_pct (approximate current % weight in S&P 500)
    - sp500_weight_source_url
    - primary_exchange (e.g., NYSE, NASDAQ)
    - exchange_source_url
    - dividend_frequency (e.g., quarterly, monthly, annual)
    - dividend_frequency_source_url
    - reference_urls (array of supporting URLs, including those for Aristocrat status and financial metrics)

    Organize the output with four top-level objects:
    - company_technology
    - company_healthcare
    - company_financials
    - company_consumer_staples
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: List[Optional[str]]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip()]

def _company_label(c: CompanyAnalysis) -> str:
    name = (c.company_name or "").strip()
    ticker = (c.ticker or "").strip()
    return f"{name} ({ticker})" if name and ticker else (name or ticker or "the company")

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    parent_node,
    company: CompanyAnalysis,
    expected_sector: str,
    node_id_prefix: str,
    company_desc: str,
) -> None:
    """
    Build verification nodes and perform checks for one company.
    All child leaves are marked critical as per the rubric; failure in any will zero the company node score.
    """
    # Create company node (non-critical parallel aggregator)
    company_node = evaluator.add_parallel(
        id=node_id_prefix,
        desc=company_desc,
        parent=parent_node,
        critical=False
    )

    # Reference URLs existence check (critical custom node, acts as a gate for subsequent verifications)
    refs_ok = bool(company.reference_urls) and len(company.reference_urls) > 0
    evaluator.add_custom_node(
        result=refs_ok,
        id=f"{node_id_prefix}_reference_urls",
        desc="Supporting reference URLs are provided for key information including Dividend Aristocrat verification and financial data",
        parent=company_node,
        critical=True
    )

    # 1) Dividend Aristocrat Verification
    aristocrat_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_dividend_aristocrat_status",
        desc="Company is confirmed as an S&P 500 Dividend Aristocrat (25+ consecutive years of dividend increases and current S&P 500 member)",
        parent=company_node,
        critical=True
    )
    claim_aristocrat = (
        f"{_company_label(company)} is an official S&P 500 Dividend Aristocrat "
        f"(has increased dividends for 25+ consecutive years and is currently an S&P 500 constituent)."
    )
    await evaluator.verify(
        claim=claim_aristocrat,
        node=aristocrat_leaf,
        sources=company.aristocrat_status_url,
        additional_instruction="Use official S&P sources or reputable finance sites listing S&P 500 Dividend Aristocrats to confirm both 25+ years of dividend increases and current S&P 500 membership."
    )

    # 2) Sector Classification (GICS)
    sector_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_sector_classification",
        desc=f"Company is classified in the {expected_sector} sector according to GICS classification",
        parent=company_node,
        critical=True
    )
    claim_sector = f"{_company_label(company)} belongs to the {expected_sector} sector under GICS."
    sector_sources = company.sector_source_url or (company.reference_urls[0] if company.reference_urls else None)
    await evaluator.verify(
        claim=claim_sector,
        node=sector_leaf,
        sources=sector_sources,
        additional_instruction="Verify the company's sector per GICS classification. Allow common synonyms and consider reputable sources (S&P indices pages, company IR classification, or finance portals)."
    )

    # 3) Market Capitalization threshold
    mcap_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_market_capitalization",
        desc=f"Company has current market capitalization of at least ${SNP500_MIN_MARKET_CAP_USD_BN} billion",
        parent=company_node,
        critical=True
    )
    claim_mcap = f"The current market capitalization of {_company_label(company)} meets or exceeds ${SNP500_MIN_MARKET_CAP_USD_BN} billion."
    await evaluator.verify(
        claim=claim_mcap,
        node=mcap_leaf,
        sources=company.market_cap_source_url,
        additional_instruction=f"Use the provided financial data source URL to confirm market cap is ≥ ${SNP500_MIN_MARKET_CAP_USD_BN}B. Allow reasonable fluctuations and rounding."
    )

    # 4) Current Dividend Yield (%)
    dy_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_dividend_yield",
        desc="Current dividend yield is provided with accurate percentage",
        parent=company_node,
        critical=True
    )
    dy_val = (company.dividend_yield_pct or "").strip()
    claim_dy = f"The current annual dividend yield of {_company_label(company)} is approximately {dy_val}%."
    await evaluator.verify(
        claim=claim_dy,
        node=dy_leaf,
        sources=company.dividend_yield_source_url,
        additional_instruction="Verify the stated dividend yield from the provided source. Allow minor rounding differences (e.g., 2.49% ~ 2.5%)."
    )

    # 5) Lowest-Cost ETF Access (among major providers)
    etf_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_lowest_expense_etf",
        desc="The ETF with the lowest expense ratio among major providers (BlackRock/iShares, Vanguard, State Street/SPDR) that holds this company is identified with expense ratio value",
        parent=company_node,
        critical=True
    )
    # Build claim summarizing winner and competitors
    winner_name = (company.etf_lowest_name or "").strip()
    winner_provider = (company.etf_lowest_provider or "").strip()
    winner_ratio = (company.etf_lowest_expense_ratio or "").strip()
    company_label = _company_label(company)

    # Collect sources: winner product page, holdings page (if any), and competitor ETF pages (if extracted)
    comp_urls = []
    for etf in company.etfs:
        comp_urls.extend(_clean_urls([etf.url, etf.holdings_url]))
    etf_sources = _clean_urls([
        company.etf_lowest_url,
        company.etf_lowest_holdings_url
    ] + comp_urls)

    claim_etf = (
        f"Among iShares (BlackRock), Vanguard, and SPDR (State Street), the ETF with the lowest expense ratio that holds {company_label} "
        f"is {winner_name} from {winner_provider} with an expense ratio of {winner_ratio}."
    )
    await evaluator.verify(
        claim=claim_etf,
        node=etf_leaf,
        sources=etf_sources if etf_sources else None,
        additional_instruction="Confirm the winner ETF's expense ratio from its product page and that it holds the company (via holdings page). If competitor ETFs are provided, check that the winner's ratio is not higher."
    )

    # 6) Institutional Ownership (13F)
    inst_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_institutional_holders",
        desc="At least one major institutional investor holding this company per Form 13F filings is identified",
        parent=company_node,
        critical=True
    )
    inst_name = (company.institutional_investor or "").strip()
    claim_inst = f"A major institutional investor, {inst_name}, holds {_company_label(company)} per recent Form 13F filings."
    await evaluator.verify(
        claim=claim_inst,
        node=inst_leaf,
        sources=company.institutional_url,
        additional_instruction="Use the provided 13F or holdings summary source to verify that the named institution holds the company."
    )

    # 7) Additional S&P Index Membership
    add_idx_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_additional_indices",
        desc="At least one additional S&P index membership beyond S&P 500 (e.g., S&P 100, sector-specific index) is identified",
        parent=company_node,
        critical=True
    )
    idx_name = (company.additional_index_name or "").strip()
    claim_idx = f"{_company_label(company)} is included in the S&P {idx_name} index beyond the S&P 500."
    await evaluator.verify(
        claim=claim_idx,
        node=add_idx_leaf,
        sources=company.additional_index_url,
        additional_instruction="Verify membership in another S&P index (e.g., S&P 100 or a sector-specific S&P index) using the provided source."
    )

    # 8) Aristocrat Achievement Year
    ach_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_aristocrat_achievement",
        desc="The year when company achieved Dividend Aristocrat status (25th consecutive year of dividend increases) is provided",
        parent=company_node,
        critical=True
    )
    ach_year = (company.aristocrat_achievement_year or "").strip()
    claim_ach = f"{_company_label(company)} achieved Dividend Aristocrat status in {ach_year}, marking its 25th consecutive year of dividend increases."
    await evaluator.verify(
        claim=claim_ach,
        node=ach_leaf,
        sources=company.aristocrat_status_url,
        additional_instruction="Use reliable sources (S&P Dividend Aristocrats documentation or company dividend history) to confirm the 25th consecutive increase year."
    )

    # 9) S&P 500 Index Weight
    w_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_sp500_weight",
        desc="Company's current weight or approximate weight in the S&P 500 index is provided",
        parent=company_node,
        critical=True
    )
    w_pct = (company.sp500_weight_pct or "").strip()
    claim_w = f"The current approximate S&P 500 index weight of {_company_label(company)} is about {w_pct}%."
    await evaluator.verify(
        claim=claim_w,
        node=w_leaf,
        sources=company.sp500_weight_source_url,
        additional_instruction="Verify approximate index weight from a reliable source (S&P index data, ETF fact sheets, or reputable finance sites). Allow approximate values and rounding."
    )

    # 10) Primary Exchange Listing
    exch_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_exchange_listing",
        desc="The stock exchange where the company is primarily listed (NYSE, NASDAQ, etc.) is identified",
        parent=company_node,
        critical=True
    )
    exch = (company.primary_exchange or "").strip()
    claim_exch = f"The primary exchange listing for {_company_label(company)} is {exch}."
    await evaluator.verify(
        claim=claim_exch,
        node=exch_leaf,
        sources=company.exchange_source_url,
        additional_instruction="Verify primary listing exchange via official exchange pages, company IR, or reputable finance sites."
    )

    # 11) Dividend Payment Frequency
    freq_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_dividend_frequency",
        desc="The company's dividend payment frequency (quarterly, monthly, annual, etc.) is correctly identified",
        parent=company_node,
        critical=True
    )
    freq = (company.dividend_frequency or "").strip()
    claim_freq = f"{_company_label(company)} pays dividends {freq}."
    await evaluator.verify(
        claim=claim_freq,
        node=freq_leaf,
        sources=company.dividend_frequency_source_url,
        additional_instruction="Verify dividend frequency via company IR dividend page or reputable finance portals."
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
    Evaluate the provided answer for the Dividend Aristocrats diversified sector portfolio task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across the four companies (diversified sectors)
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

    # Extract structured portfolio data from the answer
    portfolio = await evaluator.extract(
        prompt=prompt_extract_portfolio(),
        template_class=PortfolioExtraction,
        extraction_name="portfolio_extraction"
    )

    # Add minimal ground truth-like info (threshold) for transparency
    evaluator.add_custom_info(
        info={"snp500_min_market_cap_billion_usd": SNP500_MIN_MARKET_CAP_USD_BN},
        info_type="constants",
        info_name="snp500_thresholds"
    )

    # Build and verify each company node
    # Note: Root is non-critical to allow partial credit across companies.
    # Each company node's children are critical (as per rubric), so failure in one requirement zeros that company node.
    companies_plan = [
        (
            "company_1_technology",
            "First company: S&P 500 Dividend Aristocrat from Information Technology sector meeting all specified investment criteria",
            portfolio.company_technology,
            "Information Technology",
        ),
        (
            "company_2_healthcare",
            "Second company: S&P 500 Dividend Aristocrat from Health Care sector meeting all specified investment criteria",
            portfolio.company_healthcare,
            "Health Care",
        ),
        (
            "company_3_financials",
            "Third company: S&P 500 Dividend Aristocrat from Financials sector meeting all specified investment criteria",
            portfolio.company_financials,
            "Financials",
        ),
        (
            "company_4_consumer_staples",
            "Fourth company: S&P 500 Dividend Aristocrat from Consumer Staples sector meeting all specified investment criteria",
            portfolio.company_consumer_staples,
            "Consumer Staples",
        ),
    ]

    # Verify each company independently
    for node_id, node_desc, comp, sector in companies_plan:
        # If the extraction returned None, create an empty placeholder to proceed with checks (they will likely fail)
        comp_obj = comp or CompanyAnalysis()
        await verify_company(
            evaluator=evaluator,
            parent_node=root,
            company=comp_obj,
            expected_sector=sector,
            node_id_prefix=node_id,
            company_desc=node_desc
        )

    # Return the summary including verification tree
    return evaluator.get_summary()