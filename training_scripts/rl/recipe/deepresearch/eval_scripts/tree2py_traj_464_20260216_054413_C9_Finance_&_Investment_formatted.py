import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_us_public_pensions_top3_fy24_25"
TASK_DESCRIPTION = (
    "Identify the three largest US public pension funds by total assets as of fiscal year 2024 or fiscal year 2025. "
    "For each of the three funds, provide comprehensive documentation including: "
    "(1) the fund's official name, "
    "(2) total assets under management with the specific dollar amount and the fiscal year-end date, "
    "(3) a detailed asset allocation breakdown showing percentage allocations to at least four major asset classes including public equities and alternative investments, "
    "(4) verification of whether the fund's public equity allocation is above or below the US public pension average of 43.6%, "
    "(5) verification of whether the fund's alternative investment allocation is above or below the US average of 25.8%, "
    "(6) a source URL documenting the fund's total assets, and "
    "(7) a source URL documenting the fund's asset allocation data. "
    "Each fund must have publicly accessible documentation from official sources dated within fiscal year 2024 or 2025."
)

US_PUBLIC_PENSION_EQUITY_AVG = 43.6
US_PUBLIC_PENSION_ALT_AVG = 25.8


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AllocationItem(BaseModel):
    class_name: Optional[str] = None
    percentage: Optional[str] = None


class FundRecord(BaseModel):
    official_name: Optional[str] = None

    # Assets documentation
    assets_amount: Optional[str] = None
    assets_fy_date: Optional[str] = None  # textual date string (e.g., "June 30, 2024" or "FY2024")
    assets_source_url: Optional[str] = None

    # Allocation documentation
    allocation_items: List[AllocationItem] = Field(default_factory=list)
    public_equity_pct: Optional[str] = None
    alternatives_total_pct: Optional[str] = None
    fixed_income_pct: Optional[str] = None
    real_assets_pct: Optional[str] = None
    private_equity_pct: Optional[str] = None
    allocation_fy_date: Optional[str] = None
    allocation_source_url: Optional[str] = None

    # Optional extra references
    fund_homepage_url: Optional[str] = None
    ranking_source_urls: List[str] = Field(default_factory=list)


class FundsExtraction(BaseModel):
    funds: List[FundRecord] = Field(default_factory=list)
    ranking_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_funds() -> str:
    return """
Extract information about the three largest US public pension funds as presented in the answer.

For each fund mentioned, extract the following fields exactly as they appear in the answer text. If a field is not present, return null for that field (or an empty list for list fields):

- official_name: The official name of the fund.
- assets_amount: The total assets dollar amount (e.g., "$470 billion" or "$473.5B").
- assets_fy_date: The fiscal year-end date associated with the assets figure (e.g., "June 30, 2024", "FY2024", "as of 6/30/2024").
- assets_source_url: The URL used to support/verify the total assets figure.
- allocation_items: Array of asset allocation components. Each item has:
    - class_name: The name of the asset class (e.g., "Public Equity", "Global Equity", "Fixed Income", "Private Equity", "Real Assets", "Hedge Funds", "Real Estate", "Infrastructure", "Cash", "Treasuries").
    - percentage: The percentage allocation as presented (e.g., "44%", "44.2%", "approximately 44%"). Include the percent sign if present.
- public_equity_pct: The allocation percentage for public equities (or an equivalent label such as "Global Equity", "Public Equity", "Domestic Equity" when it refers to publicly traded equities).
- alternatives_total_pct: The total allocation percentage for alternative investments (if explicitly provided). If not explicitly provided, set to null.
- fixed_income_pct: The allocation percentage for fixed income/bonds (if explicitly provided; otherwise null).
- real_assets_pct: The allocation percentage for real assets (if explicitly provided; otherwise null).
- private_equity_pct: The allocation percentage for private equity (if explicitly provided; otherwise null).
- allocation_fy_date: The date or fiscal year label for the allocation data (e.g., "FY2024", "as of June 30, 2024").
- allocation_source_url: The URL used to support/verify the allocation data.
- fund_homepage_url: The fund's official homepage URL if provided in the answer.
- ranking_source_urls: A list of URLs specifically used to support rankings or statements that the fund is one of the largest US public pension funds (or to list top funds).

Also extract a top-level list:
- ranking_sources: A list of URLs in the answer that provide an overall ranking or listing of the largest US public pension funds (e.g., media or industry rankings). If none are provided, return an empty list.

Important:
- Do not infer or fabricate any fields. Only extract what is explicitly given in the answer text.
- Keep percentages and dollar figures as strings exactly as shown in the answer.
- Keep dates as strings exactly as shown in the answer.
- If more than three funds are presented, include all in the 'funds' array; downstream evaluation will only consider the first three.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _extract_first_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(20\d{2})", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # Sometimes FY2024/FY 2025 patterns
    m2 = re.search(r"FY\s*?(20\d{2})", text, flags=re.IGNORECASE)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None


def _percent_to_float(percent_text: Optional[str]) -> Optional[float]:
    if not percent_text:
        return None
    # Extract first float-like number from text
    m = re.search(r"(-?\d+(?:\.\d+)?)", percent_text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _is_alt_class(name: Optional[str]) -> bool:
    if not name:
        return False
    s = name.strip().lower()
    alt_keywords = [
        "alternative", "alternatives",
        "private equity", "venture capital", "private credit", "private debt",
        "hedge fund", "hedge funds",
        "real assets", "real asset", "real estate", "infrastructure",
        "commodities", "timber", "natural resources", "opportunistic", "absolute return"
    ]
    return any(k in s for k in alt_keywords)


def _is_equity_public_like(name: Optional[str]) -> bool:
    if not name:
        return False
    s = name.strip().lower()
    # Avoid private equity; require public/global/domestic equity
    if "private equity" in s:
        return False
    equity_keywords = ["public equity", "global equity", "domestic equity", "public equities", "equity"]
    return any(k in s for k in equity_keywords)


def _is_fixed_income_like(name: Optional[str]) -> bool:
    if not name:
        return False
    s = name.strip().lower()
    fi_keywords = ["fixed income", "bonds", "credit", "treasur", "investment grade", "core bonds"]
    return any(k in s for k in fi_keywords)


def _is_real_assets_like(name: Optional[str]) -> bool:
    if not name:
        return False
    s = name.strip().lower()
    ra_keywords = ["real assets", "real asset", "real estate", "infrastructure", "commodit"]
    return any(k in s for k in ra_keywords)


def _find_class_pct_from_items(items: List[AllocationItem], predicate) -> Optional[str]:
    for it in items:
        if predicate(it.class_name) and it.percentage:
            return it.percentage
    return None


def _compute_alt_total_from_items(items: List[AllocationItem]) -> Optional[float]:
    alt_values: List[float] = []
    for it in items:
        if _is_alt_class(it.class_name) and it.percentage:
            val = _percent_to_float(it.percentage)
            if val is not None:
                alt_values.append(val)
    if len(alt_values) == 0:
        return None
    total = sum(alt_values)
    # Guard against unreasonable totals
    if total <= 0 or total > 1000:
        return None
    return total


def _collect_sources(*urls: Optional[str], extra: Optional[List[str]] = None) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if u and isinstance(u, str):
            su = u.strip()
            if su and su not in seen:
                seen.add(su)
                result.append(su)
    if extra:
        for u in extra:
            if u and isinstance(u, str):
                su = u.strip()
                if su and su not in seen:
                    seen.add(su)
                    result.append(su)
    return result


def _normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return re.sub(r"\s+", " ", name).strip().lower()


# --------------------------------------------------------------------------- #
# Verification logic per fund                                                 #
# --------------------------------------------------------------------------- #
async def verify_single_fund(
    evaluator: Evaluator,
    parent_node,
    fund: FundRecord,
    fund_idx_zero_based: int,
    global_ranking_sources: List[str]
) -> Dict[str, Any]:
    idx = fund_idx_zero_based + 1
    # Fund container (non-critical to allow partial credit)
    fund_node = evaluator.add_parallel(
        id=f"Fund_{idx}_Largest",
        desc=f"{['First', 'Second', 'Third'][fund_idx_zero_based]} largest US public pension fund completely documented",
        parent=parent_node,
        critical=False
    )

    # ---------------- Identification (critical) ---------------- #
    ident_node = evaluator.add_parallel(
        id=f"Fund_{idx}_Identification",
        desc=f"Fund {idx} properly identified and named",
        parent=fund_node,
        critical=True
    )

    # Official name provided (existence)
    evaluator.add_custom_node(
        result=bool(fund.official_name and fund.official_name.strip()),
        id=f"Fund_{idx}_Official_Name",
        desc=f"Fund {idx} official name provided",
        parent=ident_node,
        critical=True
    )

    # Is US public pension fund (verify with sources)
    is_us_leaf = evaluator.add_leaf(
        id=f"Fund_{idx}_Is_US_Fund",
        desc=f"Fund {idx} is confirmed as a US public pension fund",
        parent=ident_node,
        critical=True
    )
    us_sources = _collect_sources(
        fund.fund_homepage_url, fund.assets_source_url, fund.allocation_source_url
    )
    name_for_claim = fund.official_name or f"Fund {idx}"
    await evaluator.verify(
        claim=f"'{name_for_claim}' is a US public pension fund (a state or local government retirement system in the United States).",
        node=is_us_leaf,
        sources=us_sources,
        additional_instruction="Verify that the organization is a US public sector pension/retirement system (e.g., state employees, teachers, or public employees). Official sites or authoritative documentation should indicate this clearly."
    )

    # Ranking verification (verify with ranking sources)
    ranking_leaf = evaluator.add_leaf(
        id=f"Fund_{idx}_Ranking_Verification",
        desc=f"Fund {idx} is verified as one of the three largest US public pension funds",
        parent=ident_node,
        critical=True
    )
    ranking_sources = _collect_sources(extra=(fund.ranking_source_urls or []) + (global_ranking_sources or []))
    await evaluator.verify(
        claim=f"The fund '{name_for_claim}' is one of the three largest US public pension funds by total assets.",
        node=ranking_leaf,
        sources=ranking_sources,
        additional_instruction="Look for a ranking or list of largest US public pension funds (e.g., industry rankings or authoritative reports) and verify that this fund appears within the top three."
    )

    # ---------------- Asset documentation (critical) ------------ #
    assets_node = evaluator.add_parallel(
        id=f"Fund_{idx}_Asset_Documentation",
        desc=f"Fund {idx} total assets properly documented",
        parent=fund_node,
        critical=True
    )

    # Asset amount verification
    asset_amount_leaf = evaluator.add_leaf(
        id=f"Fund_{idx}_Asset_Amount",
        desc=f"Fund {idx} total assets dollar amount provided",
        parent=assets_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total assets of '{name_for_claim}' are {fund.assets_amount}.",
        node=asset_amount_leaf,
        sources=fund.assets_source_url or None,
        additional_instruction="Verify that the page shows the total assets (or net position held in trust) figure matching the provided dollar amount for the public pension fund."
    )

    # Asset date verification (FY2024 or FY2025)
    asset_date_leaf = evaluator.add_leaf(
        id=f"Fund_{idx}_Asset_Date",
        desc=f"Fund {idx} asset data date is from FY2024 or FY2025",
        parent=assets_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The asset data for '{name_for_claim}' is from fiscal year 2024 or fiscal year 2025 (date string in the answer: '{fund.assets_fy_date}').",
        node=asset_date_leaf,
        sources=fund.assets_source_url or None,
        additional_instruction="Confirm the timeframe on the source page. Accept 'FY2024' or 'FY2025' or a date that clearly falls within fiscal year 2024 or 2025 (e.g., year-end June 30, 2024 or 2025)."
    )

    # Asset URL exists (existence check)
    evaluator.add_custom_node(
        result=bool(fund.assets_source_url and fund.assets_source_url.strip()),
        id=f"Fund_{idx}_Asset_URL",
        desc=f"URL reference for Fund {idx} asset size verification",
        parent=assets_node,
        critical=True
    )

    # ---------------- Allocation documentation (non-critical) --- #
    # Set non-critical because some children are non-critical; framework requires critical parent to have only critical children
    alloc_node = evaluator.add_parallel(
        id=f"Fund_{idx}_Allocation_Data",
        desc=f"Fund {idx} asset allocation breakdown documented",
        parent=fund_node,
        critical=False
    )

    # Min four asset classes present in the answer
    has_four_classes = len([it for it in fund.allocation_items if it.class_name and it.percentage]) >= 4
    evaluator.add_custom_node(
        result=has_four_classes,
        id=f"Fund_{idx}_Min_Four_Asset_Classes",
        desc=f"Fund {idx} provides at least four major asset classes in allocation breakdown",
        parent=alloc_node,
        critical=True
    )

    # Public equity percentage value verification (grounded by allocation URL)
    pe_leaf = evaluator.add_leaf(
        id=f"Fund_{idx}_Public_Equity_Pct",
        desc=f"Fund {idx} public equity allocation percentage provided",
        parent=alloc_node,
        critical=True
    )
    # Use extracted public_equity_pct if missing try to recover from items
    pub_eq_pct_text = fund.public_equity_pct or _find_class_pct_from_items(fund.allocation_items, _is_equity_public_like) or ""
    await evaluator.verify(
        claim=f"The public equity allocation for '{name_for_claim}' is {pub_eq_pct_text}.",
        node=pe_leaf,
        sources=fund.allocation_source_url or None,
        additional_instruction="Verify from the allocation source that the public equity (or equivalent category such as 'Global Equity' referring to publicly-traded equities) matches the stated percentage. Allow minor rounding differences."
    )

    # Alternatives total provided or calculable (existence / calculability gate)
    alternatives_available = False
    if fund.alternatives_total_pct and fund.alternatives_total_pct.strip():
        alternatives_available = True
    else:
        # Check if at least one alternative sub-class with a percentage exists (so it's at least calculable in principle)
        alternatives_available = _compute_alt_total_from_items(fund.allocation_items) is not None

    alt_total_node = evaluator.add_custom_node(
        result=alternatives_available,
        id=f"Fund_{idx}_Alternatives_Total",
        desc=f"Fund {idx} total alternative investments percentage provided or calculable",
        parent=alloc_node,
        critical=True
    )

    # Fixed income percentage provided (non-critical)
    has_fi = bool(fund.fixed_income_pct and fund.fixed_income_pct.strip()) or (
        _find_class_pct_from_items(fund.allocation_items, _is_fixed_income_like) is not None
    )
    evaluator.add_custom_node(
        result=has_fi,
        id=f"Fund_{idx}_Fixed_Income_Pct",
        desc=f"Fund {idx} fixed income allocation percentage provided",
        parent=alloc_node,
        critical=False
    )

    # Real assets percentage provided (non-critical)
    has_ra = bool(fund.real_assets_pct and fund.real_assets_pct.strip()) or (
        _find_class_pct_from_items(fund.allocation_items, _is_real_assets_like) is not None
    )
    evaluator.add_custom_node(
        result=has_ra,
        id=f"Fund_{idx}_Real_Assets_Pct",
        desc=f"Fund {idx} real assets allocation percentage provided",
        parent=alloc_node,
        critical=False
    )

    # Private equity percentage provided (non-critical)
    has_pe = bool(fund.private_equity_pct and fund.private_equity_pct.strip()) or (
        _find_class_pct_from_items(fund.allocation_items, lambda n: (n or "").strip().lower().find("private equity") >= 0) is not None
    )
    evaluator.add_custom_node(
        result=has_pe,
        id=f"Fund_{idx}_Private_Equity_Pct",
        desc=f"Fund {idx} private equity allocation percentage provided",
        parent=alloc_node,
        critical=False
    )

    # Allocation date verification (FY2024 or FY2025)
    alloc_date_leaf = evaluator.add_leaf(
        id=f"Fund_{idx}_Allocation_Date",
        desc=f"Fund {idx} allocation data is from FY2024 or FY2025",
        parent=alloc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The allocation data for '{name_for_claim}' is from fiscal year 2024 or 2025 (date string in the answer: '{fund.allocation_fy_date}').",
        node=alloc_date_leaf,
        sources=fund.allocation_source_url or None,
        additional_instruction="Confirm on the allocation source that the data corresponds to FY2024 or FY2025 (e.g., as-of June 30, 2024/2025 or explicitly labeled FY2024/FY2025)."
    )

    # Allocation URL exists (existence)
    evaluator.add_custom_node(
        result=bool(fund.allocation_source_url and fund.allocation_source_url.strip()),
        id=f"Fund_{idx}_Allocation_URL",
        desc=f"URL reference for Fund {idx} allocation data",
        parent=alloc_node,
        critical=True
    )

    # ---------------- Comparative analysis (critical) ----------- #
    comp_node = evaluator.add_parallel(
        id=f"Fund_{idx}_Comparative_Analysis",
        desc=f"Fund {idx} allocations compared to US averages",
        parent=fund_node,
        critical=True
    )

    # Equity vs average (use allocation URL; depend on public equity pct verification to skip if missing/failed)
    eq_vs_avg_leaf = evaluator.add_leaf(
        id=f"Fund_{idx}_Equity_vs_Average",
        desc=f"Fund {idx} public equity allocation compared to 43.6% US average",
        parent=comp_node,
        critical=True
    )

    # Build direction string based on extracted numeric if available
    pub_eq_num = _percent_to_float(pub_eq_pct_text)
    if pub_eq_num is not None:
        direction_eq = "above" if pub_eq_num > US_PUBLIC_PENSION_EQUITY_AVG else "below or equal to"
        claim_eq = f"The public equity allocation for '{name_for_claim}' is {direction_eq} {US_PUBLIC_PENSION_EQUITY_AVG}%."
    else:
        # Fall back to a directly stated comparison claim that LLM can verify by reading page numbers
        claim_eq = (
            f"The public equity allocation for '{name_for_claim}', as shown on the allocation page, is either above or below {US_PUBLIC_PENSION_EQUITY_AVG}% "
            f"depending on the actual percentage on the page."
        )

    await evaluator.verify(
        claim=claim_eq,
        node=eq_vs_avg_leaf,
        sources=fund.allocation_source_url or None,
        additional_instruction="Read the public equity percentage from the page and compare it to 43.6%. If it is strictly greater, it is 'above'; if equal or lower, it is 'below or equal to'. Allow for small rounding differences."
    )

    # Alternatives vs average (depend on alternatives_total availability gate)
    alt_vs_avg_leaf = evaluator.add_leaf(
        id=f"Fund_{idx}_Alt_vs_Average",
        desc=f"Fund {idx} alternative allocation compared to 25.8% US average",
        parent=comp_node,
        critical=True
    )

    # Use alternatives_total_pct if present, otherwise compute from items
    alt_pct_num: Optional[float] = None
    if fund.alternatives_total_pct and fund.alternatives_total_pct.strip():
        alt_pct_num = _percent_to_float(fund.alternatives_total_pct)
    if alt_pct_num is None:
        alt_pct_num = _compute_alt_total_from_items(fund.allocation_items)

    if alt_pct_num is not None:
        direction_alt = "above" if alt_pct_num > US_PUBLIC_PENSION_ALT_AVG else "below or equal to"
        claim_alt = f"The total alternative investments allocation for '{name_for_claim}' is {direction_alt} {US_PUBLIC_PENSION_ALT_AVG}%."
    else:
        claim_alt = (
            f"The total alternative investments allocation for '{name_for_claim}', as can be determined from the allocation page, "
            f"is above or below {US_PUBLIC_PENSION_ALT_AVG}% depending on the actual figures."
        )

    await evaluator.verify(
        claim=claim_alt,
        node=alt_vs_avg_leaf,
        sources=fund.allocation_source_url or None,
        additional_instruction="Determine the total alternatives allocation from the page (either explicitly provided or by summing alternative categories such as private equity, real assets, hedge funds, etc.) and compare to 25.8%. If strictly greater, 'above'; otherwise 'below or equal to'."
    )

    return {
        "nodes": {
            "is_us_leaf": is_us_leaf,
            "ranking_leaf": ranking_leaf,
            "asset_amount_leaf": asset_amount_leaf,
            "asset_date_leaf": asset_date_leaf,
            "alloc_public_equity_leaf": pe_leaf,
            "alloc_alt_total_node": alt_total_node,
            "eq_vs_avg_leaf": eq_vs_avg_leaf,
            "alt_vs_avg_leaf": alt_vs_avg_leaf,
        }
    }


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
    # Initialize evaluator (root node defaults to non-critical parallel)
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

    # Extract structured funds info
    extraction: FundsExtraction = await evaluator.extract(
        prompt=prompt_extract_funds(),
        template_class=FundsExtraction,
        extraction_name="funds_extraction"
    )

    # Build nodes for the three funds (pad if fewer)
    funds: List[FundRecord] = list(extraction.funds) if extraction and extraction.funds else []
    while len(funds) < 3:
        funds.append(FundRecord())

    # Create three top-level fund nodes parented to root
    verification_handles: List[Dict[str, Any]] = []

    # For ranking aggregation check, collect names and ranking sources
    names_for_aggregate: List[str] = []
    for i in range(3):
        fund = funds[i]
        if fund and fund.official_name:
            names_for_aggregate.append(fund.official_name)

    global_ranking_sources = extraction.ranking_sources or []

    # Verify each fund
    fund_tasks = []
    for i in range(3):
        fund_tasks.append(
            verify_single_fund(evaluator, root, funds[i], i, global_ranking_sources)
        )
    verification_handles = await asyncio.gather(*fund_tasks)

    # ---------------- Aggregate cross-fund requirements (critical) -------- #
    agg_node = evaluator.add_parallel(
        id="Aggregate_Requirements",
        desc="Cross-fund verification requirements",
        parent=root,
        critical=True
    )

    # All three distinct
    distinct_ok = False
    if len(names_for_aggregate) == 3 and all(n and n.strip() for n in names_for_aggregate):
        norm_names = [_normalize_name(n) for n in names_for_aggregate]
        if None not in norm_names:
            distinct_ok = len(set(norm_names)) == 3

    evaluator.add_custom_node(
        result=distinct_ok,
        id="All_Three_Distinct",
        desc="All three funds are distinct institutions",
        parent=agg_node,
        critical=True
    )

    # All properly ranked (verify against ranking sources)
    ranked_leaf = evaluator.add_leaf(
        id="All_Properly_Ranked",
        desc="The three funds are verified as the three largest US public pension funds",
        parent=agg_node,
        critical=True
    )
    # Build claim listing the three names (order not enforced in claim text)
    if len(names_for_aggregate) == 3:
        claim_rank = f"The three largest US public pension funds by total assets are: {names_for_aggregate[0]}, {names_for_aggregate[1]}, and {names_for_aggregate[2]}."
    else:
        claim_rank = "These three funds are the three largest US public pension funds by total assets."

    # Collect ranking sources: global + per-fund ranking sources
    per_fund_ranking_sources: List[str] = []
    for i in range(3):
        frs = funds[i].ranking_source_urls if funds[i] and funds[i].ranking_source_urls else []
        per_fund_ranking_sources.extend(frs)
    ranking_sources_all = _collect_sources(extra=(global_ranking_sources + per_fund_ranking_sources))

    await evaluator.verify(
        claim=claim_rank,
        node=ranked_leaf,
        sources=ranking_sources_all,
        additional_instruction="Verify against a reputable ranking or list of largest US public pension funds that the three named funds are indeed the top three by assets. Allow minor naming variants (e.g., abbreviations vs. full names)."
    )

    # All have recent data (FY2024 or FY2025 for assets and allocation)
    all_recent = True
    for i in range(3):
        f = funds[i]
        asset_year = _extract_first_year(f.assets_fy_date)
        alloc_year = _extract_first_year(f.allocation_fy_date)
        if asset_year not in (2024, 2025) or alloc_year not in (2024, 2025):
            all_recent = False
            break

    evaluator.add_custom_node(
        result=all_recent,
        id="All_Have_Recent_Data",
        desc="All three funds have data from FY2024 or FY2025",
        parent=agg_node,
        critical=True
    )

    # Return evaluation summary
    return evaluator.get_summary()