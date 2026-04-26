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
TASK_ID = "sp500_pharma_dividend_2024"
TASK_DESCRIPTION = (
    "Find 3 pharmaceutical companies that are current constituents of the S&P 500 index and meet ALL of the "
    "following criteria:\n\n"
    "1. Sector Classification: The company must be classified in the Healthcare sector according to the Global "
    "Industry Classification Standard (GICS), specifically within the pharmaceutical industry.\n"
    "2. Dividend Increase in 2024: The company must have announced at least one quarterly dividend increase during "
    "calendar year 2024. Provide the specific details including the previous dividend amount, the new increased "
    "amount, and the announcement date.\n"
    "3. Minimum Dividend Yield: The company must have a current dividend yield of at least 3.0%.\n"
    "4. Minimum Institutional Ownership: The company must have institutional ownership of at least 60% of its "
    "outstanding shares.\n\n"
    "For each of the 3 companies identified, provide: Company name; Stock ticker symbol; Details of the 2024 dividend "
    "increase (previous quarterly amount, new quarterly amount, announcement date); Current dividend yield (as a "
    "percentage); Institutional ownership percentage; Reference URLs that support each of the above data points."
)

MIN_DIVIDEND_YIELD = 3.0  # percent
MIN_INSTITUTIONAL_OWNERSHIP = 60.0  # percent
DIVIDEND_INCREASE_YEAR = 2024


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DividendIncrease(BaseModel):
    previous_amount: Optional[str] = None
    new_amount: Optional[str] = None
    announcement_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CompanyItem(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None

    # Eligibility evidence and classification
    gics_sector: Optional[str] = None
    gics_industry: Optional[str] = None
    sp500_sources: List[str] = Field(default_factory=list)
    gics_sources: List[str] = Field(default_factory=list)

    # Dividend increase (2024)
    dividend_increase: Optional[DividendIncrease] = None

    # Current dividend yield
    dividend_yield_percent: Optional[str] = None
    dividend_yield_sources: List[str] = Field(default_factory=list)

    # Institutional ownership
    institutional_ownership_percent: Optional[str] = None
    ownership_sources: List[str] = Field(default_factory=list)


class CompaniesExtraction(BaseModel):
    companies: List[CompanyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return f"""
Extract all companies the answer claims satisfy the task. For each company, extract the following fields strictly
from the answer text (do not invent values):

- name: Company name exactly as written.
- ticker: Stock ticker symbol if provided.
- gics_sector: The GICS sector label as written (e.g., "Health Care" or "Healthcare").
- gics_industry: The GICS industry or sub-industry label as written (e.g., "Pharmaceuticals").
- sp500_sources: A list of URL(s) explicitly cited that support the claim the company is a current S&P 500 constituent.
- gics_sources: A list of URL(s) explicitly cited that support the GICS classification (sector/industry).
- dividend_increase: An object with:
    - previous_amount: The prior quarterly dividend amount (string, include currency symbol if present).
    - new_amount: The new quarterly dividend amount (string).
    - announcement_date: The announcement date for the increase (string; use YYYY-MM-DD if clearly provided, otherwise copy as written).
    - sources: A list of URL(s) explicitly cited that support the dividend increase details. The increase must be announced in calendar year {DIVIDEND_INCREASE_YEAR}.
- dividend_yield_percent: The current dividend yield as presented (e.g., "3.2%").
- dividend_yield_sources: A list of URL(s) explicitly cited that show the current dividend yield.
- institutional_ownership_percent: The institutional ownership percentage as presented (e.g., "62%").
- ownership_sources: A list of URL(s) explicitly cited that show institutional ownership.

Rules:
1) Only extract URLs that are explicitly present in the answer (including markdown links). Do not infer or create URLs.
2) If a field is missing in the answer, set it to null (for strings) or [] (for lists).
3) If the answer mentions more than 3 companies, extract them all; the evaluator will check only the first 3.
4) Preserve text exactly as written for amounts, percentages, and dates; do not normalize or compute.

Return a JSON object with a single field:
- companies: an array of company objects as specified above, in the same order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return [u for u in (lst or []) if isinstance(u, str) and len(u.strip()) > 0]


def _company_display(c: CompanyItem, idx: int) -> str:
    base = c.name or f"Company #{idx + 1}"
    if c.ticker:
        return f"{base} ({c.ticker})"
    return base


def _first_k_companies(extracted: CompaniesExtraction, k: int = 3) -> List[CompanyItem]:
    items = extracted.companies[:k] if extracted and extracted.companies else []
    # Pad to k with empty placeholders if needed
    while len(items) < k:
        items.append(CompanyItem())
    return items


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    parent_node,
    company: CompanyItem,
    index: int,
) -> None:
    """
    Build the verification subtree for one company (index is 0-based).
    Structure (all children inside company node are critical to pass the company):
      - Eligibility (critical):
          * Sources provided for S&P 500 membership (critical, custom)
          * S&P 500 membership supported by sources (critical, URL verify)
          * Sources provided for GICS classification (critical, custom)
          * GICS: Health Care sector and Pharmaceuticals industry supported by sources (critical, URL verify)
      - Dividend criteria (critical):
          * Sources provided for 2024 dividend increase plus all details present (critical, custom)
          * 2024 dividend increase details supported by sources (critical, URL verify)
          * Sources provided for current dividend yield value (critical, custom)
          * Dividend yield >= 3.0% supported by sources (critical, URL verify)
      - Institutional ownership (critical):
          * Sources provided for institutional ownership (critical, custom)
          * Institutional ownership >= 60% supported by sources (critical, URL verify)
    """
    display = _company_display(company, index)

    # Company container (non-critical at root level; critical checks live below)
    company_node = evaluator.add_parallel(
        id=f"company_{index+1}",
        desc=f"{['First','Second','Third'][index]} pharmaceutical company meets all eligibility and financial criteria",
        parent=parent_node,
        critical=False,
    )

    # Optional non-critical info presence node for logging
    evaluator.add_custom_node(
        result=bool(company.name) and bool(company.ticker),
        id=f"company_{index+1}_basic_info_present",
        desc=f"{display}: basic info (name and ticker) present in the answer",
        parent=company_node,
        critical=False,
    )

    # ---------------------- Eligibility group ---------------------- #
    elig_node = evaluator.add_parallel(
        id=f"company_{index+1}_eligibility",
        desc=f"{display}: Eligibility – S&P 500 membership and GICS Health Care sector (Pharmaceuticals)",
        parent=company_node,
        critical=True,
    )

    sp500_sources = _safe_list(company.sp500_sources)
    gics_sources = _safe_list(company.gics_sources)

    # Sources provided checks (critical)
    sp500_src_node = evaluator.add_custom_node(
        result=len(sp500_sources) > 0,
        id=f"company_{index+1}_sp500_sources_provided",
        desc=f"{display}: S&P 500 membership sources provided",
        parent=elig_node,
        critical=True,
    )

    gics_src_node = evaluator.add_custom_node(
        result=len(gics_sources) > 0,
        id=f"company_{index+1}_gics_sources_provided",
        desc=f"{display}: GICS classification sources provided",
        parent=elig_node,
        critical=True,
    )

    # S&P 500 membership supported
    sp500_supported_leaf = evaluator.add_leaf(
        id=f"company_{index+1}_sp500_supported",
        desc=f"{display}: is a current constituent of the S&P 500 (supported by cited sources)",
        parent=elig_node,
        critical=True,
    )
    sp500_claim = f"{display} is a current constituent of the S&P 500 index."
    sp500_addins = (
        "Verify the provided webpages explicitly indicate the company is an S&P 500 constituent. "
        "Accept official S&P Dow Jones Indices lists, S&P Global, index provider factsheets, "
        "or reliable listings (e.g., company IR pages or reputable financial sites). "
        "If the page is outdated, rely on what the page explicitly states."
    )

    # GICS Health Care (Healthcare) + Pharmaceuticals classification supported
    gics_supported_leaf = evaluator.add_leaf(
        id=f"company_{index+1}_gics_healthcare_pharma",
        desc=f"{display}: classified under GICS Health Care sector and within Pharmaceuticals industry",
        parent=elig_node,
        critical=True,
    )
    sector_text = company.gics_sector or "Health Care"
    industry_text = company.gics_industry or "Pharmaceuticals"
    gics_claim = (
        f"{display} is classified under the GICS Health Care sector and within the Pharmaceuticals "
        f"industry (or Pharmaceuticals sub-industry). Stated sector: '{sector_text}'. Stated industry: '{industry_text}'."
    )
    gics_addins = (
        "Verify that the sources explicitly show GICS classification placing the company in the Health Care sector "
        "and specifically in the Pharmaceuticals industry (or pharmaceuticals sub-industry). "
        "Treat 'Health Care' and 'Healthcare' as equivalent spellings. "
        "Do NOT accept 'Biotechnology' alone as meeting the 'Pharmaceuticals' requirement."
    )

    # ---------------------- Dividend group ------------------------- #
    div_node = evaluator.add_parallel(
        id=f"company_{index+1}_dividend",
        desc=f"{display}: Dividend criteria – 2024 increase details and yield threshold",
        parent=company_node,
        critical=True,
    )

    di = company.dividend_increase or DividendIncrease()
    di_sources = _safe_list(di.sources)

    # Existence and sources provided for the dividend increase details
    di_details_present = bool(di.previous_amount) and bool(di.new_amount) and bool(di.announcement_date)
    di_src_provided_node = evaluator.add_custom_node(
        result=di_details_present and (len(di_sources) > 0),
        id=f"company_{index+1}_dividend_increase_sources_provided",
        desc=f"{display}: 2024 dividend increase details present and sources provided",
        parent=div_node,
        critical=True,
    )

    di_supported_leaf = evaluator.add_leaf(
        id=f"company_{index+1}_dividend_increase_2024_supported",
        desc=f"{display}: announced a quarterly dividend increase during {DIVIDEND_INCREASE_YEAR} with stated details",
        parent=div_node,
        critical=True,
    )
    di_claim = (
        f"On {di.announcement_date}, {display} announced a quarterly dividend increase from "
        f"{di.previous_amount} to {di.new_amount}, and this announcement occurred during calendar year {DIVIDEND_INCREASE_YEAR}."
    )
    di_addins = (
        f"Check the press release or investor relations page to confirm an announced quarterly dividend increase in "
        f"{DIVIDEND_INCREASE_YEAR}, with the stated previous and new amounts and the announcement date. "
        "Minor formatting differences (e.g., $0.41 vs 41 cents) are acceptable if the amounts clearly match."
    )

    # Dividend yield existence and sources
    dy_sources = _safe_list(company.dividend_yield_sources)
    dy_present = bool(company.dividend_yield_percent)
    dy_src_provided_node = evaluator.add_custom_node(
        result=dy_present and (len(dy_sources) > 0),
        id=f"company_{index+1}_dividend_yield_sources_provided",
        desc=f"{display}: current dividend yield value present and sources provided",
        parent=div_node,
        critical=True,
    )

    dy_supported_leaf = evaluator.add_leaf(
        id=f"company_{index+1}_dividend_yield_meets_min",
        desc=f"{display}: current dividend yield is at least {MIN_DIVIDEND_YIELD:.1f}%",
        parent=div_node,
        critical=True,
    )
    dy_val = company.dividend_yield_percent or ""
    dy_claim = (
        f"The current dividend yield for {display} is at least {MIN_DIVIDEND_YIELD:.1f}% "
        f"(the cited source shows a value such as '{dy_val}')."
    )
    dy_addins = (
        f"Confirm the page shows a current dividend yield >= {MIN_DIVIDEND_YIELD:.1f}%. "
        "Allow minor rounding differences (e.g., 3.0% vs 3.01%). "
        "Use the clearly labeled 'dividend yield' value; if multiple yields are shown, a standard/current figure suffices."
    )

    # ---------------------- Ownership group ------------------------ #
    own_node = evaluator.add_parallel(
        id=f"company_{index+1}_ownership",
        desc=f"{display}: Institutional ownership meets minimum threshold",
        parent=company_node,
        critical=True,
    )

    own_sources = _safe_list(company.ownership_sources)
    own_present = bool(company.institutional_ownership_percent)
    own_src_provided_node = evaluator.add_custom_node(
        result=own_present and (len(own_sources) > 0),
        id=f"company_{index+1}_ownership_sources_provided",
        desc=f"{display}: institutional ownership value present and sources provided",
        parent=own_node,
        critical=True,
    )

    own_supported_leaf = evaluator.add_leaf(
        id=f"company_{index+1}_ownership_meets_min",
        desc=f"{display}: institutional ownership is at least {MIN_INSTITUTIONAL_OWNERSHIP:.0f}%",
        parent=own_node,
        critical=True,
    )
    own_val = company.institutional_ownership_percent or ""
    own_claim = (
        f"The institutional ownership for {display} is at least {MIN_INSTITUTIONAL_OWNERSHIP:.0f}% "
        f"(the cited source shows a value such as '{own_val}')."
    )
    own_addins = (
        f"Confirm the page shows institutional ownership >= {MIN_INSTITUTIONAL_OWNERSHIP:.0f}%. "
        "Accept equivalent phrasings like 'percent held by institutions' or 'institutional investors'. "
        "Allow minor rounding differences."
    )

    # ---------------------- Execute verifications ------------------ #
    # Note: Custom nodes above already recorded pass/fail for prerequisites.
    # Using batch_verify to parallelize evidence-backed checks.
    claims_and_sources = [
        (sp500_claim, sp500_sources, sp500_supported_leaf, sp500_addins),
        (gics_claim, gics_sources, gics_supported_leaf, gics_addins),
        (di_claim, di_sources, di_supported_leaf, di_addins),
        (dy_claim, dy_sources, dy_supported_leaf, dy_addins),
        (own_claim, own_sources, own_supported_leaf, own_addins),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the S&P 500 Pharma Dividend 2024 task.
    """
    # Initialize evaluator (root is non-critical by design; we'll build critical sub-nodes)
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

    # Extract all companies from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=CompaniesExtraction,
        extraction_name="companies_extraction",
    )

    # Keep only the first 3 for verification (as per final reminder policy)
    selected = _first_k_companies(extracted, k=3)

    # Record constraint info as ground truth meta
    evaluator.add_ground_truth({
        "requirements": {
            "sp500_constituent": "current constituent",
            "gics_sector": "Health Care (Healthcare)",
            "gics_industry": "Pharmaceuticals",
            "dividend_increase_year": DIVIDEND_INCREASE_YEAR,
            "min_dividend_yield_percent": MIN_DIVIDEND_YIELD,
            "min_institutional_ownership_percent": MIN_INSTITUTIONAL_OWNERSHIP,
            "num_companies_required": 3
        }
    })

    # Top-level aggregation node (non-critical to allow soft scoring across companies,
    # with critical sub-requirements inside each company)
    top = evaluator.add_parallel(
        id="overall",
        desc="Evaluate whether 3 pharmaceutical companies in the S&P 500 are identified, each meeting all specified criteria",
        parent=root,
        critical=False,
    )

    # Company count check (critical): allow 'at least 3' to avoid penalizing answers that list >3;
    # evaluator only checks the first 3.
    total_identified = sum(1 for c in (extracted.companies if extracted and extracted.companies else []) if c.name)
    evaluator.add_custom_node(
        result=total_identified >= 3,
        id="company_count",
        desc="At least 3 companies are provided in the answer (the evaluator will assess the first 3)",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_info({"extracted_company_count": total_identified}, info_type="stats", info_name="extraction_stats")

    # Build and verify each selected company subtree
    for i, comp in enumerate(selected):
        await verify_company(evaluator, top, comp, i)

    # Return evaluation summary
    return evaluator.get_summary()