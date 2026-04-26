import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "healthcare_dividend_aristocrats_screen_2026"
TASK_DESCRIPTION = """
Identify 2 publicly traded U.S. healthcare companies (pharmaceutical or healthcare services sectors) that meet ALL of the following investment screening criteria as of March 2026:

1. Dividend Aristocrat Status: The company must have increased its dividend payout for at least 25 consecutive years and be a current member of the S&P 500 Index.

2. Market Capitalization: The company must be classified as large-cap, with a market capitalization between $10 billion and $200 billion.

3. Dividend Performance:
   - The company must have a current dividend yield of at least 2.0%
   - The company must have declared and paid a quarterly dividend during Q1 2026 (January through March 2026)

4. Institutional Ownership: At least 70% of the company's outstanding shares must be held by institutional investors.

5. Credit Rating: If the company has issued corporate bonds, those bonds must carry an investment-grade rating of at least BBB- (S&P or Fitch) or Baa3 (Moody's).

For each company, provide:
- Company name and stock ticker symbol
- Verification that all criteria are met
- Supporting URL references for dividend information
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CompanyItem(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[str] = None
    dividend_yield: Optional[str] = None
    institutional_ownership: Optional[str] = None
    dividend_aristocrat_status: Optional[str] = None
    q1_2026_dividend_note: Optional[str] = None
    bond_ratings: Optional[str] = None
    # URL sources
    source_urls: List[str] = Field(default_factory=list, description="General supporting URLs (profile, index membership, market cap, ownership, ratings, etc.)")
    dividend_urls: List[str] = Field(default_factory=list, description="Dividend-related URLs (dividend history, yield, press releases, etc.)")


class CompaniesExtraction(BaseModel):
    companies: List[CompanyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return """
    From the provided answer, extract up to TWO (2) companies that the answer claims satisfy the given screening criteria.
    Keep only the first two companies mentioned if more are present. If fewer than two are present, return whatever is provided.

    For each company, extract the following fields (use strings for all values when applicable):
    - name: The full company name (e.g., "Johnson & Johnson")
    - ticker: The stock ticker symbol (e.g., "JNJ")
    - sector: The sector or description of operations (e.g., "Pharmaceuticals", "Healthcare services", "Managed care")
    - market_cap: The market capitalization string as stated (e.g., "$150B", "$47,123,456,789", or similar)
    - dividend_yield: The current dividend yield (e.g., "2.3%")
    - institutional_ownership: The proportion held by institutions (e.g., "74%" or "0.74")
    - dividend_aristocrat_status: Any explicit statement indicating >= 25 years of consecutive dividend increases and/or "Dividend Aristocrat"
    - q1_2026_dividend_note: Any statement that the company declared and PAID a dividend during Q1 2026 (Jan-Mar 2026)
    - bond_ratings: Any ratings mentioned (e.g., "A-/A3", "BBB+", "Baa1", or issuer ratings)
    - source_urls: All general supporting URLs mentioned in the answer (company profile, S&P 500 membership list, market cap sources, ownership pages, ratings pages, etc.)
    - dividend_urls: Dividend-related URLs mentioned (dividend history pages, press releases, dividend yield pages, etc.)

    Rules:
    - Do not fabricate any information. If a field is not present in the answer, set it to null (or [] for URL lists).
    - Only extract URLs explicitly present in the answer (including markdown links).
    - Return a JSON object with a "companies" array of company objects in the original order of appearance.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _label_company(c: CompanyItem, idx: int) -> str:
    if c and c.name and c.ticker:
        return f"{c.name} ({c.ticker})"
    if c and c.name:
        return c.name
    if c and c.ticker:
        return c.ticker
    return f"Company #{idx + 1}"


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification logic per company                                              #
# --------------------------------------------------------------------------- #
async def verify_company(evaluator: Evaluator, parent_node, company: CompanyItem, idx: int) -> None:
    """
    Build the verification subtree and run checks for a single company.
    """
    cid = f"company_{idx + 1}"

    company_node = evaluator.add_parallel(
        id=f"{cid}_evaluation",
        desc=f"Complete evaluation of {cid} against all investment criteria",
        parent=parent_node,
        critical=False
    )

    # Prepare URL bundles
    general_sources = _dedup_urls(company.source_urls or [])
    dividend_sources = _dedup_urls(company.dividend_urls or [])
    all_sources = _dedup_urls(dividend_sources + general_sources)

    label = _label_company(company, idx)

    # --------------------- Basic Profile (critical) --------------------- #
    basic_node = evaluator.add_parallel(
        id=f"{cid}_basic_profile",
        desc=f"Verification of {cid}'s sector classification and index membership",
        parent=company_node,
        critical=True
    )

    # Name provided (existence) - custom node (critical)
    name_ok = bool(company and company.name and company.ticker)
    evaluator.add_custom_node(
        result=name_ok,
        id=f"{cid}_name_provided",
        desc="Company name and ticker symbol are clearly identified",
        parent=basic_node,
        critical=True
    )

    # Healthcare sector (critical)
    sector_leaf = evaluator.add_leaf(
        id=f"{cid}_healthcare_sector",
        desc="Company operates in the healthcare sector (pharmaceutical or healthcare services)",
        parent=basic_node,
        critical=True
    )
    sector_claim = (
        f"{label} operates in the healthcare sector specifically as either a pharmaceutical/biopharmaceutical company "
        f"or a healthcare services/provider/managed care entity."
    )
    await evaluator.verify(
        claim=sector_claim,
        node=sector_leaf,
        sources=general_sources,
        additional_instruction=(
            "Accept if credible sources indicate the company is a pharmaceutical/biopharmaceutical firm or a "
            "healthcare services/provider/managed care organization. Names for sector classifications may vary "
            "slightly (e.g., 'Health Care' sector, 'Managed Health Care', 'Health Care Providers & Services'). "
            "Do not accept if the company is only medical devices/equipment without services or pharmaceuticals."
        )
    )

    # S&P 500 membership (critical)
    sp500_leaf = evaluator.add_leaf(
        id=f"{cid}_sp500_member",
        desc="Company is a current member of the S&P 500 Index",
        parent=basic_node,
        critical=True
    )
    sp500_claim = f"{label} is a current constituent of the S&P 500 Index (as of March 2026)."
    await evaluator.verify(
        claim=sp500_claim,
        node=sp500_leaf,
        sources=general_sources,
        additional_instruction=(
            "Verify membership using credible lists such as S&P Dow Jones Indices, official S&P 500 lists, "
            "or up-to-date sources like Wikipedia's 'List of S&P 500 companies' that indicate current membership "
            "as of March 2026."
        )
    )

    # --------------------- Dividend Record (critical) ------------------- #
    dividend_node = evaluator.add_parallel(
        id=f"{cid}_dividend_record",
        desc=f"Verification of {cid}'s dividend history and current performance",
        parent=company_node,
        critical=True
    )

    # Dividend URL existence (critical custom)
    evaluator.add_custom_node(
        result=(len(dividend_sources) > 0),
        id=f"{cid}_dividend_url",
        desc="URL reference provided for dividend information",
        parent=dividend_node,
        critical=True
    )

    # Dividend Aristocrat status (critical)
    aristocrat_leaf = evaluator.add_leaf(
        id=f"{cid}_dividend_aristocrat",
        desc="Company has increased dividends for at least 25 consecutive years",
        parent=dividend_node,
        critical=True
    )
    aristocrat_claim = (
        f"{label} has increased its dividend for at least 25 consecutive years and thus qualifies as a Dividend Aristocrat "
        f"(and is an S&P 500 constituent)."
    )
    await evaluator.verify(
        claim=aristocrat_claim,
        node=aristocrat_leaf,
        sources=all_sources,
        additional_instruction=(
            "Look for authoritative confirmation (e.g., S&P Dow Jones Indices Dividend Aristocrats list, "
            "ProShares NOBL materials, company investor relations pages explicitly stating 25+ consecutive annual increases). "
            "Minor wording variations are acceptable as long as the 25+ year streak is clear."
        )
    )

    # Q1 2026 dividend declared AND paid (critical)
    q1_leaf = evaluator.add_leaf(
        id=f"{cid}_q1_2026_dividend",
        desc="Company declared and paid a quarterly dividend in Q1 2026 (January-March 2026)",
        parent=dividend_node,
        critical=True
    )
    q1_claim = (
        f"{label} declared and paid a quarterly cash dividend during Q1 2026 (between January 1, 2026 and March 31, 2026)."
    )
    await evaluator.verify(
        claim=q1_claim,
        node=q1_leaf,
        sources=dividend_sources,
        additional_instruction=(
            "Accept if the dividend history or press releases show a pay date in Jan, Feb, or Mar 2026. "
            "If both declaration and pay dates are shown, ensure the pay date falls within Q1 2026."
        )
    )

    # Dividend yield >= 2.0% (critical)
    yield_leaf = evaluator.add_leaf(
        id=f"{cid}_dividend_yield",
        desc="Current dividend yield is at least 2.0%",
        parent=dividend_node,
        critical=True
    )
    yield_claim = f"{label} has a current dividend yield of at least 2.0 percent."
    await evaluator.verify(
        claim=yield_claim,
        node=yield_leaf,
        sources=all_sources,
        additional_instruction=(
            "Use dividend yield shown on credible sources (IR dividend page, Yahoo/Google Finance, Nasdaq, etc.). "
            "Allow small rounding differences (e.g., 1.99% rounds to 2.0% is NOT acceptable; 2.01% is acceptable). "
            "Treat 'current' as approximately contemporaneous with the cited page."
        )
    )

    # --------------------- Financial Standing (critical) ---------------- #
    financial_node = evaluator.add_parallel(
        id=f"{cid}_financial_standing",
        desc=f"Verification of {cid}'s market capitalization, ownership structure, and credit rating",
        parent=company_node,
        critical=True
    )

    # Market cap between $10B and $200B (critical)
    mcap_leaf = evaluator.add_leaf(
        id=f"{cid}_market_cap",
        desc="Market capitalization is between $10 billion and $200 billion (large-cap range)",
        parent=financial_node,
        critical=True
    )
    mcap_claim = f"{label}'s market capitalization lies between $10 billion and $200 billion USD."
    await evaluator.verify(
        claim=mcap_claim,
        node=mcap_leaf,
        sources=general_sources,
        additional_instruction=(
            "Use market cap shown on credible finance sources (e.g., IR, Yahoo/Google Finance, Bloomberg snapshots, Nasdaq). "
            "Units may be shown in billions; approximate within reason as long as it clearly falls within the $10B–$200B range."
        )
    )

    # Institutional ownership >= 70% (critical)
    inst_own_leaf = evaluator.add_leaf(
        id=f"{cid}_institutional_ownership",
        desc="Institutional ownership is at least 70% of shares outstanding",
        parent=financial_node,
        critical=True
    )
    inst_claim = f"At least 70 percent of {label}'s outstanding shares are held by institutional investors."
    await evaluator.verify(
        claim=inst_claim,
        node=inst_own_leaf,
        sources=general_sources,
        additional_instruction=(
            "Use credible sources such as Nasdaq 'Institutional Holdings', Yahoo Finance 'Holders', or IR disclosures. "
            "If the source shows a percentage held by institutions, verify it is >= 70%."
        )
    )

    # Investment-grade bond rating if bonds exist (critical)
    rating_leaf = evaluator.add_leaf(
        id=f"{cid}_bond_rating",
        desc="If company has issued corporate bonds, they carry an investment-grade rating of at least BBB- (S&P/Fitch) or Baa3 (Moody's)",
        parent=financial_node,
        critical=True
    )
    rating_claim = (
        f"If {label} has issued corporate bonds, those bonds (or the issuer/senior unsecured rating) are investment-grade "
        f"with a rating of at least BBB- (S&P/Fitch) or Baa3 (Moody's)."
    )
    await evaluator.verify(
        claim=rating_claim,
        node=rating_leaf,
        sources=general_sources,
        additional_instruction=(
            "Accept evidence of ratings from S&P, Moody's, Fitch, or credible aggregators (e.g., company filings or IR pages citing ratings). "
            "Issuer credit rating or senior unsecured rating qualifies. "
            "If the company has no outstanding bonds and a credible source states this, the condition is vacuously satisfied."
        )
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the healthcare Dividend Aristocrats investment screening task.
    """
    # Initialize the evaluator
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

    # Add a top-level task node (non-critical to allow partial credit if only one valid company)
    task_node = evaluator.add_parallel(
        id="investment_screening_task",
        desc="Identify 2 U.S. healthcare companies that meet all specified investment criteria",
        parent=root,
        critical=False
    )

    # Extract up to 2 companies from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=CompaniesExtraction,
        extraction_name="companies_extraction"
    )
    companies: List[CompanyItem] = list(extraction.companies or [])

    # Ensure exactly 2 entries (pad with empty if needed)
    while len(companies) < 2:
        companies.append(CompanyItem())
    companies = companies[:2]

    # Verify each company
    for i in range(2):
        await verify_company(evaluator, task_node, companies[i], i)

    # Return structured summary
    return evaluator.get_summary()