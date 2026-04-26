import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "div_aristocrats_2026"
TASK_DESCRIPTION = (
    "I am building a conservative dividend income portfolio and want to identify three companies from the S&P 500 "
    "Dividend Aristocrats Index that offer strong credit quality and sustainable dividend payments. Please identify "
    "three S&P 500 Dividend Aristocrat companies that meet ALL of the following criteria:\n\n"
    "Company Selection Requirements:\n"
    "- Each company must be officially listed in the 2026 S&P 500 Dividend Aristocrats Index\n"
    "- Each company must have at least 25 consecutive years of annual dividend increases\n"
    "- The three companies must come from three different sectors:\n"
    "  - One from the Consumer Staples sector\n"
    "  - One from the Industrials sector\n"
    "  - One from the Healthcare sector\n\n"
    "Credit Quality Requirement:\n"
    "- Each company must have an investment grade credit rating (BBB-/Baa3 or higher) from at least one major rating agency (S&P Global Ratings, Moody's, or Fitch Ratings)\n\n"
    "Dividend Sustainability Requirement:\n"
    "- Each company's dividend payout ratio must fall within the sustainable range of 40-75%\n\n"
    "Information to Provide for Each Company:\n"
    "1. Company name and stock ticker symbol\n"
    "2. GICS sector classification\n"
    "3. Current credit rating from at least one major rating agency\n"
    "4. Current annual dividend yield (as a percentage)\n"
    "5. Current dividend payout ratio (as a percentage)\n"
    "6. Reference URLs supporting each piece of information (from official company sources, S&P Dividend Aristocrats lists, credit rating agencies, or reputable financial data providers)"
)

REQUIRED_SECTORS = ["Consumer Staples", "Industrials", "Healthcare"]
SUSTAINABLE_PAYOUT_MIN = 40.0
SUSTAINABLE_PAYOUT_MAX = 75.0


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CompanySources(BaseModel):
    aristocrat_urls: List[str] = Field(default_factory=list)
    sp500_urls: List[str] = Field(default_factory=list)
    sector_urls: List[str] = Field(default_factory=list)
    rating_urls: List[str] = Field(default_factory=list)
    yield_urls: List[str] = Field(default_factory=list)
    payout_urls: List[str] = Field(default_factory=list)
    streak_urls: List[str] = Field(default_factory=list)


class CompanyRecord(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    sector: Optional[str] = None
    rating_agency: Optional[str] = None  # e.g., "S&P", "Moody's", "Fitch"
    rating_value: Optional[str] = None   # e.g., "A-", "BBB+", "Baa1", etc.
    dividend_yield: Optional[str] = None # e.g., "2.5%", "2.5", "2.5 percent"
    payout_ratio: Optional[str] = None   # e.g., "55%", "0.55", "55 percent"
    sources: CompanySources = Field(default_factory=CompanySources)


class CompaniesExtraction(BaseModel):
    companies: List[CompanyRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return (
        "Extract all companies mentioned in the answer that are proposed as S&P 500 Dividend Aristocrats. For each company, "
        "return the following fields strictly as they appear in the answer (do not invent missing data):\n"
        "- name: Company name\n"
        "- ticker: Stock ticker symbol\n"
        "- sector: GICS sector (e.g., Consumer Staples, Industrials, Health Care / Healthcare)\n"
        "- rating_agency: One of S&P, Moody's, Fitch (if provided)\n"
        "- rating_value: The current credit rating string from that agency (e.g., A-, BBB+, Baa1)\n"
        "- dividend_yield: The current annual dividend yield as presented (percentage or text)\n"
        "- payout_ratio: The current dividend payout ratio as presented (percentage or text)\n"
        "- sources: URL lists for each information category, strictly from URLs explicitly present in the answer text:\n"
        "  • aristocrat_urls: URLs proving inclusion in the S&P 500 Dividend Aristocrats (prefer official S&P pages or reputable providers)\n"
        "  • sp500_urls: URLs confirming S&P 500 membership\n"
        "  • sector_urls: URLs confirming GICS sector classification\n"
        "  • rating_urls: URLs confirming the stated rating\n"
        "  • yield_urls: URLs confirming the stated dividend yield\n"
        "  • payout_urls: URLs confirming the payout ratio\n"
        "  • streak_urls: URLs confirming 25+ consecutive years of dividend increases\n\n"
        "Rules:\n"
        "1) Only extract URLs explicitly present in the answer (including markdown links). Do not infer or invent URLs.\n"
        "2) If a specific field or URL list is missing, set it to null (for single field) or an empty array (for URL lists).\n"
        "3) Return a JSON object with 'companies' as an array of such company objects, preserving the order they appear in the answer.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_sector(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s_lower = s.strip().lower()
    # GICS official names include "Consumer Staples", "Industrials", "Health Care"
    # Accept common synonyms like "Healthcare" and "Consumer Defensives"
    if "consumer" in s_lower and ("staple" in s_lower or "defensive" in s_lower):
        return "Consumer Staples"
    if "industrial" in s_lower:
        return "Industrials"
    if "health care" in s_lower or "healthcare" in s_lower:
        return "Healthcare"
    # If exact match in required set (case-insensitive)
    for req in REQUIRED_SECTORS:
        if s_lower == req.lower():
            return req
    # Unknown or other sectors; return original cleaned capitalization
    return s.strip()


def sector_matches(candidate: Optional[str], required: str) -> bool:
    if not candidate:
        return False
    normalized = normalize_sector(candidate)
    req_norm = normalize_sector(required)
    return normalized is not None and req_norm is not None and normalized.lower() == req_norm.lower()


def parse_percentage(text: Optional[str]) -> Optional[float]:
    """
    Parse a textual percentage to a numeric value in [0, 100].
    Examples accepted: "55%", "55 percent", "0.55", "~55%", "45-50%" -> returns first number (45)
    """
    if not text:
        return None
    s = text.strip().lower()
    # Extract the first decimal or integer number
    match = re.search(r"(\d+(\.\d+)?)", s)
    if not match:
        return None
    val = float(match.group(1))
    # If appears to be a fraction (0 < val <= 1) and there's no explicit '%' or 'percent', convert to percent
    if ('%' not in s and 'percent' not in s) and 0 < val <= 1.0:
        val = val * 100.0
    # Clip to [0, 100] as a sanity check
    if val < 0:
        return None
    if val > 1000:  # clearly wrong
        return None
    return min(val, 100.0)


def combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for url in lst:
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


def pick_companies_by_sectors(all_companies: List[CompanyRecord]) -> Dict[str, CompanyRecord]:
    """
    Select the first matching company for each required sector.
    If not found, return an empty placeholder to allow the evaluation to proceed and fail appropriately.
    """
    selected: Dict[str, CompanyRecord] = {}
    for required in REQUIRED_SECTORS:
        for rec in all_companies:
            if sector_matches(rec.sector, required) and required not in selected:
                selected[required] = rec
                break
        if required not in selected:
            selected[required] = CompanyRecord(sector=required)  # placeholder
    return selected


# --------------------------------------------------------------------------- #
# Verification functions for each company                                     #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    parent_root,
    company: CompanyRecord,
    company_idx: int,
    expected_sector: str
) -> None:
    """
    Build and verify the full sub-tree for a single company, following the rubric.
    company_idx: 1-based index (1, 2, 3)
    expected_sector: one of Consumer Staples, Industrials, Healthcare
    """

    # Parent sequential node for this company
    company_node = evaluator.add_sequential(
        id=f"company_{company_idx}",
        desc=(
            "First qualifying Dividend Aristocrat company identification and verification"
            if company_idx == 1 else
            "Second qualifying Dividend Aristocrat company identification and verification"
            if company_idx == 2 else
            "Third qualifying Dividend Aristocrat company identification and verification"
        ),
        parent=parent_root,
        critical=False
    )

    name = company.name or "Unknown Company"
    ticker = company.ticker or "Unknown Ticker"

    # 1) Identification (critical, parallel)
    ident_node = evaluator.add_parallel(
        id=f"company_{company_idx}_identification",
        desc="Correct identification of a company that is an official S&P 500 Dividend Aristocrat for 2026",
        parent=company_node,
        critical=True
    )

    # 1.a) Reference source existence (critical) – gate other checks
    aristocrat_refs_exist = len(company.sources.aristocrat_urls) > 0
    ref_src_node = evaluator.add_custom_node(
        result=aristocrat_refs_exist,
        id=f"company_{company_idx}_reference_source",
        desc="Valid reference URL provided confirming Dividend Aristocrat status from official S&P source or reputable financial data provider",
        parent=ident_node,
        critical=True
    )

    # 1.b) Aristocrat status (critical, verify with URLs)
    aristocrat_node = evaluator.add_leaf(
        id=f"company_{company_idx}_aristocrat_status",
        desc="Company is officially listed in the 2026 S&P 500 Dividend Aristocrats Index",
        parent=ident_node,
        critical=True
    )
    aristocrat_claim = f"{name} is officially listed in the 2026 S&P 500 Dividend Aristocrats Index."
    await evaluator.verify(
        claim=aristocrat_claim,
        node=aristocrat_node,
        sources=company.sources.aristocrat_urls,
        additional_instruction=(
            "Verify the page explicitly supports that the company is an S&P 500 Dividend Aristocrat. "
            "Prefer S&P Dow Jones Indices pages; reputable sources (e.g., ProShares S&P 500 Dividend Aristocrats ETF holdings, "
            "major financial data providers) are acceptable. The year 2026 applies to this list."
        ),
        extra_prerequisites=[ref_src_node]
    )

    # 1.c) S&P 500 membership (critical)
    sp500_node = evaluator.add_leaf(
        id=f"company_{company_idx}_sp500_membership",
        desc="Company is a current member of the S&P 500 Index",
        parent=ident_node,
        critical=True
    )
    sp500_sources = combine_sources(company.sources.sp500_urls, company.sources.aristocrat_urls)
    sp500_claim = f"{name} is a current constituent of the S&P 500 Index."
    await evaluator.verify(
        claim=sp500_claim,
        node=sp500_node,
        sources=sp500_sources,
        additional_instruction=(
            "Confirm the company's current S&P 500 membership using S&P indices pages, company investor relations, "
            "or reputable financial data providers."
        ),
        extra_prerequisites=[ref_src_node]
    )

    # 1.d) Dividend streak (critical)
    streak_node = evaluator.add_leaf(
        id=f"company_{company_idx}_dividend_streak",
        desc="Company has 25 or more consecutive years of annual dividend increases",
        parent=ident_node,
        critical=True
    )
    streak_sources = combine_sources(company.sources.streak_urls, company.sources.aristocrat_urls)
    streak_claim = f"{name} has at least 25 consecutive years of annual dividend increases."
    await evaluator.verify(
        claim=streak_claim,
        node=streak_node,
        sources=streak_sources,
        additional_instruction=(
            "Membership in the Dividend Aristocrats implies ≥25 consecutive years of dividend increases. "
            "Confirm explicitly if possible from S&P, company dividend history, or reputable sources."
        ),
        extra_prerequisites=[ref_src_node]
    )

    # 2) Sector verification (critical, parallel)
    sector_node = evaluator.add_parallel(
        id=f"company_{company_idx}_sector_verification",
        desc=f"Verification that the company belongs to the {expected_sector} sector according to GICS classification",
        parent=company_node,
        critical=True
    )
    # 2.a) sector_reference existence (critical) – gate classification check
    sector_refs_exist = len(company.sources.sector_urls) > 0
    sector_ref_node = evaluator.add_custom_node(
        result=sector_refs_exist,
        id=f"company_{company_idx}_sector_reference",
        desc="Valid reference URL confirming sector classification from official company filing or financial data source",
        parent=sector_node,
        critical=True
    )

    # 2.b) sector classification claim (critical)
    sector_class_node = evaluator.add_leaf(
        id=f"company_{company_idx}_sector_classification",
        desc=f"Company is classified under GICS {expected_sector} sector",
        parent=sector_node,
        critical=True
    )
    sector_claim = f"{name} is classified under the GICS {expected_sector} sector."
    await evaluator.verify(
        claim=sector_claim,
        node=sector_class_node,
        sources=company.sources.sector_urls,
        additional_instruction=(
            "Use company filings, S&P/MSCI GICS sources, or reputable financial data providers to confirm GICS sector. "
            "Treat 'Health Care' and 'Healthcare' equivalently."
        ),
        extra_prerequisites=[sector_ref_node]
    )

    # 3) Credit rating (critical, parallel)
    rating_node = evaluator.add_parallel(
        id=f"company_{company_idx}_credit_rating",
        desc="Verification of investment grade credit rating from major rating agencies",
        parent=company_node,
        critical=True
    )
    # 3.a) rating_source existence (critical)
    rating_refs_exist = len(company.sources.rating_urls) > 0
    rating_ref_node = evaluator.add_custom_node(
        result=rating_refs_exist,
        id=f"company_{company_idx}_rating_source",
        desc="Valid reference URL from credit rating agency website, company investor relations, or SEC filing confirming the rating",
        parent=rating_node,
        critical=True
    )

    # 3.b) rating threshold check (critical)
    rating_threshold_node = evaluator.add_leaf(
        id=f"company_{company_idx}_rating_threshold",
        desc="Company has investment grade rating (BBB-/Baa3 or higher) from at least one major rating agency (S&P, Moody's, or Fitch)",
        parent=rating_node,
        critical=True
    )
    agency = (company.rating_agency or "a recognized credit rating agency").strip()
    rating_val = (company.rating_value or "an investment grade rating").strip()
    rating_claim = (
        f"The current credit rating for {name} from {agency} is {rating_val}, "
        f"which is investment grade (BBB-/Baa3 or higher)."
    )
    await evaluator.verify(
        claim=rating_claim,
        node=rating_threshold_node,
        sources=company.sources.rating_urls,
        additional_instruction=(
            "Confirm the rating on the provided page. Investment grade thresholds: "
            "S&P/Fitch: BBB- or higher (BBB-, BBB, BBB+, A-, ...); Moody's: Baa3 or higher (Baa3, Baa2, Baa1, A3, ...). "
            "Minor naming variants are acceptable."
        ),
        extra_prerequisites=[rating_ref_node]
    )

    # 4) Dividend metrics (non-critical, parallel)
    div_node = evaluator.add_parallel(
        id=f"company_{company_idx}_dividend_metrics",
        desc="Dividend sustainability metrics including yield and payout ratio",
        parent=company_node,
        critical=False
    )

    # 4.a) Dividend yield (non-critical, parallel)
    yield_main = evaluator.add_parallel(
        id=f"company_{company_idx}_dividend_yield",
        desc="Current annual dividend yield is provided",
        parent=div_node,
        critical=False
    )

    # 4.a.i) Yield reference existence (non-critical)
    yield_refs_exist = len(company.sources.yield_urls) > 0
    yield_ref_node = evaluator.add_custom_node(
        result=yield_refs_exist,
        id=f"company_{company_idx}_yield_reference",
        desc="Valid reference URL from company investor relations or financial data provider confirming the yield",
        parent=yield_main,
        critical=False
    )

    # 4.a.ii) Yield value verification (non-critical)
    yield_value_node = evaluator.add_leaf(
        id=f"company_{company_idx}_yield_value",
        desc="Dividend yield value is stated as a percentage",
        parent=yield_main,
        critical=False
    )
    yield_val_text = (company.dividend_yield or "").strip()
    yield_claim = (
        f"The current annual dividend yield for {name} ({ticker}) is {yield_val_text}."
        if yield_val_text else
        f"The current annual dividend yield for {name} ({ticker}) is provided."
    )
    await evaluator.verify(
        claim=yield_claim,
        node=yield_value_node,
        sources=company.sources.yield_urls,
        additional_instruction=(
            "Verify the yield shown on the page. Accept small discrepancies due to real-time changes or rounding."
        ),
        extra_prerequisites=[yield_ref_node] if yield_refs_exist else None
    )

    # 4.b) Payout ratio (critical, parallel)
    payout_main = evaluator.add_parallel(
        id=f"company_{company_idx}_payout_ratio",
        desc="Dividend payout ratio falls within sustainable range (40-75%)",
        parent=div_node,
        critical=True
    )

    # 4.b.i) Payout value range check (critical, custom)
    payout_pct = parse_percentage(company.payout_ratio)
    payout_in_range = payout_pct is not None and SUSTAINABLE_PAYOUT_MIN <= payout_pct <= SUSTAINABLE_PAYOUT_MAX
    payout_value_node = evaluator.add_custom_node(
        result=payout_in_range,
        id=f"company_{company_idx}_payout_value",
        desc="Payout ratio value is provided and falls between 40% and 75%",
        parent=payout_main,
        critical=True
    )

    # 4.b.ii) Payout ratio verification against sources (critical)
    payout_ref_verify_node = evaluator.add_leaf(
        id=f"company_{company_idx}_payout_reference",
        desc="Valid reference URL from financial statements or data provider confirming the payout ratio calculation",
        parent=payout_main,
        critical=True
    )
    payout_val_text = (company.payout_ratio or "").strip()
    payout_claim = (
        f"The dividend payout ratio for {name} ({ticker}) is {payout_val_text}."
        if payout_val_text else
        f"The dividend payout ratio for {name} ({ticker}) is within a sustainable range."
    )
    await evaluator.verify(
        claim=payout_claim,
        node=payout_ref_verify_node,
        sources=company.sources.payout_urls,
        additional_instruction=(
            "Confirm the payout ratio (TTM or FY) on the page. Minor rounding differences are acceptable. "
            "If multiple payout ratio definitions exist, use the commonly reported payout ratio (Dividends/Net Income)."
        ),
        extra_prerequisites=[payout_value_node]  # Only verify reference if provided value is in range
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
    Evaluate an answer for selecting three S&P 500 Dividend Aristocrats in 2026 that satisfy
    sector diversification, investment-grade rating, and dividend sustainability requirements.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Companies evaluated independently
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=CompaniesExtraction,
        extraction_name="companies_extraction"
    )

    # Select companies by required sectors
    selected_map = pick_companies_by_sectors(extracted.companies)
    evaluator.add_custom_info(
        info={
            "selected_companies": {
                "Consumer Staples": {
                    "name": selected_map["Consumer Staples"].name,
                    "ticker": selected_map["Consumer Staples"].ticker
                },
                "Industrials": {
                    "name": selected_map["Industrials"].name,
                    "ticker": selected_map["Industrials"].ticker
                },
                "Healthcare": {
                    "name": selected_map["Healthcare"].name,
                    "ticker": selected_map["Healthcare"].ticker
                }
            },
            "payout_ratio_range": [SUSTAINABLE_PAYOUT_MIN, SUSTAINABLE_PAYOUT_MAX],
            "required_sectors": REQUIRED_SECTORS
        },
        info_type="selection_summary"
    )

    # Build verification subtrees for each required sector
    sector_order = ["Consumer Staples", "Industrials", "Healthcare"]
    for idx, sector in enumerate(sector_order, start=1):
        await verify_company(
            evaluator=evaluator,
            parent_root=root,
            company=selected_map[sector],
            company_idx=idx,
            expected_sector=sector
        )

    # Return summary
    return evaluator.get_summary()