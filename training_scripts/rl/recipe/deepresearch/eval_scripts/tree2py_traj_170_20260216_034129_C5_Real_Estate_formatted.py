import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "denver_mf_investment_locations_2026"
TASK_DESCRIPTION = (
    "As a real estate investment analyst for a multifamily apartment development firm, you are conducting market "
    "research to identify optimal acquisition or development locations in the Denver metropolitan area. Your firm has "
    "established specific investment criteria based on market analysis and economic trends through early 2026.\n\n"
    "Identify 4 distinct neighborhoods or submarkets in the Denver metro area that meet ALL of the following "
    "investment criteria:\n\n"
    "Investment Criteria:\n"
    "1. Walkability Requirement: The neighborhood must have a Walk Score of at least 60.\n"
    "2. Price Point Requirement: The median home price must be between $350,000 and $650,000 based on 2025-2026 data.\n"
    "3. Market Performance Requirement: YoY price appreciation of 5% or less (including negative/depreciation).\n"
    "4. Location Requirement: Within 30 miles of Denver International Airport OR within 10 miles of downtown Denver "
    "(16th Street Mall).\n"
    "5. Transit/Accessibility Requirement (nice-to-have): Preference for RTD light rail access, walking distance to "
    "downtown, or other public transit connectivity.\n\n"
    "Required info for each neighborhood: name, Walk Score, median home price (with source and date), YoY price change %, "
    "distance from DIA or downtown, transit access details, and reference URL(s)."
)

ALLOWED_PRICE_YEARS = {"2025", "2026"}


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class Neighborhood(BaseModel):
    name: Optional[str] = None
    walk_score: Optional[str] = None
    median_home_price: Optional[str] = None
    price_data_date: Optional[str] = None
    yoy_price_change: Optional[str] = None
    distance_target: Optional[str] = None  # "airport" or "downtown"
    distance_miles: Optional[str] = None
    transit_access: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NeighborhoodsExtraction(BaseModel):
    neighborhoods: List[Neighborhood] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_neighborhoods() -> str:
    return """
Extract up to 4 neighborhoods or submarkets in the Denver metro area exactly as presented in the answer. For each neighborhood, return the following fields:

- name: Neighborhood or submarket name.
- walk_score: The stated Walk Score number if provided (e.g., "72", "Walk Score 72", "approx. 70"); extract as presented.
- median_home_price: The stated median home price figure as written (e.g., "$475,000", "median list price $520k", "typical value $410,000").
- price_data_date: The date or year associated with the price figure as written (e.g., "2026", "as of Jan 2026", "Q4 2025").
- yoy_price_change: The year-over-year price change percentage as written (e.g., "-1.8%", "3%", "0.5% YoY").
- distance_target: Which anchor the distance is measured to if given: use exactly "airport" for Denver International Airport or "downtown" for downtown Denver (16th Street Mall area). If both are given, pick one mentioned in the answer (prefer 'downtown' if clearly stated as the basis for compliance).
- distance_miles: The stated distance in miles to the chosen target, as written (e.g., "8.4 mi", "~9 miles", "7 mi"). If kilometers are provided, keep the original string (do not convert).
- transit_access: Transit/access description if provided (e.g., "near RTD L Line", "walking distance to Union Station", "multiple RTD bus routes").
- sources: An array of all URLs explicitly cited for this neighborhood’s facts (walk score, prices, YoY, distance, transit, etc.). Include all URLs mentioned for this neighborhood. Extract actual URLs only.

General rules:
- Do not infer or invent any information or URLs; only extract what appears in the answer.
- Keep numbers and symbols exactly as written (e.g., keep "$", "%", "k").
- If a field is missing for a neighborhood, set it to null (or empty array for sources).
- Return neighborhoods in the order they appear in the answer; if more than 4 are provided, return only the first 4.
"""


# -----------------------------------------------------------------------------
# Helper parsing utilities
# -----------------------------------------------------------------------------
def _first_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def parse_walk_score(text: Optional[str]) -> Optional[float]:
    # Extract first number; Walk Score is typically 0-100
    val = _first_number(text)
    if val is None:
        return None
    # If it's an odd case like "Walk Score 70/100", first number is fine
    return val


def parse_price(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = text.strip().lower().replace(",", "")
    # Handle $ and k/m suffix heuristics
    # Examples: "$475000", "$520k", "$0.4m", "520k", "410000"
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    num = float(m.group(0))
    if "m" in s:
        num *= 1_000_000
    elif "k" in s:
        num *= 1_000
    return num


def parse_percent(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = text.strip().lower().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    return float(m.group(0))


def parse_miles(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = text.strip().lower().replace(",", "")
    # Prefer values clearly in miles; if "km" detect and convert
    km = "km" in s
    val = _first_number(s)
    if val is None:
        return None
    if km:
        # Convert km to miles
        val *= 0.621371
    return val


def extract_years(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return re.findall(r"\b(20\d{2})\b", text)


def normalize_target(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    if "air" in t or "dia" in t or "airport" in t:
        return "airport"
    if "down" in t or "16th" in t or "mall" in t or "union station" in t:
        return "downtown"
    if t in {"airport", "downtown"}:
        return t
    return None


def urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# -----------------------------------------------------------------------------
# Verification for a single neighborhood
# -----------------------------------------------------------------------------
async def verify_neighborhood(evaluator: Evaluator, parent, nb: Neighborhood, idx: int) -> None:
    neigh_node = evaluator.add_parallel(
        id=f"neighborhood_{idx+1}",
        desc=f"Neighborhood {idx + 1}: Qualifying neighborhood meeting investment criteria",
        parent=parent,
        critical=False
    )

    # URL Reference (critical): at least one URL provided for verification
    url_ref_node = evaluator.add_custom_node(
        result=urls_present(nb.sources),
        id=f"nh_{idx+1}_url_reference",
        desc="Provide reference URL(s) for verification of neighborhood data",
        parent=neigh_node,
        critical=True
    )

    # Walkability (critical)
    walk_parent = evaluator.add_parallel(
        id=f"nh_{idx+1}_walkability",
        desc="Neighborhood has a Walk Score of 60 or higher",
        parent=neigh_node,
        critical=True
    )

    # Walk score threshold (custom)
    ws_val = parse_walk_score(nb.walk_score)
    evaluator.add_custom_node(
        result=(ws_val is not None and ws_val >= 60),
        id=f"nh_{idx+1}_walk_score_threshold",
        desc=f"Walk Score value {nb.walk_score or 'N/A'} is >= 60",
        parent=walk_parent,
        critical=True
    )

    # Walk score supported by sources
    walk_leaf = evaluator.add_leaf(
        id=f"nh_{idx+1}_walk_score_supported",
        desc=f"Walk Score for {nb.name or 'the neighborhood'} is supported by cited sources",
        parent=walk_parent,
        critical=True
    )
    walk_claim = f"The neighborhood {nb.name or 'the neighborhood'} has a Walk Score of {nb.walk_score}."
    await evaluator.verify(
        claim=walk_claim,
        node=walk_leaf,
        sources=nb.sources,
        additional_instruction=(
            "Accept if the page clearly shows a Walk Score or an equivalent walkability metric for the neighborhood "
            "or for a representative central address within it. Allow minor rounding differences."
        )
    )

    # Price Range (critical): split into supported-by-URL, numeric range, and year 2025/2026 support
    price_parent = evaluator.add_parallel(
        id=f"nh_{idx+1}_price_range",
        desc="Median home price is between $350,000 and $650,000 based on 2025-2026 data",
        parent=neigh_node,
        critical=True
    )

    # Price supported by URL
    price_supported_leaf = evaluator.add_leaf(
        id=f"nh_{idx+1}_price_supported",
        desc=f"Median home price for {nb.name or 'the neighborhood'} is supported by cited sources",
        parent=price_parent,
        critical=True
    )
    price_claim = (
        f"The median home price for {nb.name or 'the neighborhood'} is {nb.median_home_price}."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_supported_leaf,
        sources=nb.sources,
        additional_instruction=(
            "Confirm that the page reports the stated price (or a clearly equivalent metric like 'median list price' "
            "or 'typical home value'). Allow minor rounding (e.g., $520k vs $520,000)."
        )
    )

    # Price in required numeric range (custom)
    price_val = parse_price(nb.median_home_price)
    evaluator.add_custom_node(
        result=(price_val is not None and 350_000 <= price_val <= 650_000),
        id=f"nh_{idx+1}_price_value_in_range",
        desc=f"Median home price {nb.median_home_price or 'N/A'} is within $350,000–$650,000",
        parent=price_parent,
        critical=True
    )

    # Price date 2025 or 2026 supported by URL
    price_year_leaf = evaluator.add_leaf(
        id=f"nh_{idx+1}_price_date_2025_2026",
        desc="Price figure is based on 2025 or 2026 data as supported by sources",
        parent=price_parent,
        critical=True
    )
    # If the extracted date contains a year, include it in the claim; otherwise, phrase generally
    extracted_years = extract_years(nb.price_data_date)
    if extracted_years:
        yr_str = ", ".join(extracted_years)
        price_date_claim = (
            f"The stated price figure for {nb.name or 'the neighborhood'} is explicitly tied to {yr_str} data (within 2025–2026)."
        )
    else:
        price_date_claim = (
            f"The cited source(s) indicate that the price figure for {nb.name or 'the neighborhood'} is based on 2025 or 2026 data."
        )
    await evaluator.verify(
        claim=price_date_claim,
        node=price_year_leaf,
        sources=nb.sources,
        additional_instruction=(
            "Accept if the page shows the price metric is reported for 2025 or 2026 (e.g., page timestamp, chart date, or textual date near the price)."
        )
    )

    # Market Performance (critical): YoY <= 5% (including negative)
    market_parent = evaluator.add_parallel(
        id=f"nh_{idx+1}_market_performance",
        desc="Year-over-year price appreciation is 5% or less (including negative values)",
        parent=neigh_node,
        critical=True
    )

    # YoY supported by URL
    yoy_supported_leaf = evaluator.add_leaf(
        id=f"nh_{idx+1}_yoy_supported",
        desc=f"YoY price change for {nb.name or 'the neighborhood'} is supported by cited sources",
        parent=market_parent,
        critical=True
    )
    yoy_claim = (
        f"The year-over-year price change for {nb.name or 'the neighborhood'} is {nb.yoy_price_change}."
    )
    await evaluator.verify(
        claim=yoy_claim,
        node=yoy_supported_leaf,
        sources=nb.sources,
        additional_instruction=(
            "Confirm the YoY price change percentage (or clearly equivalent metric). Accept minor rounding or "
            "formatting differences. Negative values are allowed."
        )
    )

    # YoY threshold check (custom)
    yoy_val = parse_percent(nb.yoy_price_change)
    evaluator.add_custom_node(
        result=(yoy_val is not None and yoy_val <= 5.0),
        id=f"nh_{idx+1}_yoy_threshold",
        desc=f"YoY price change {nb.yoy_price_change or 'N/A'} is ≤ 5%",
        parent=market_parent,
        critical=True
    )

    # Location Proximity (critical): within threshold depending on target
    loc_parent = evaluator.add_parallel(
        id=f"nh_{idx+1}_location_proximity",
        desc="Located within 30 miles of Denver International Airport OR within 10 miles of downtown Denver",
        parent=neigh_node,
        critical=True
    )

    # Location supported by URL
    loc_supported_leaf = evaluator.add_leaf(
        id=f"nh_{idx+1}_location_supported",
        desc=f"Distance to {'chosen anchor' if not nb.distance_target else nb.distance_target} for {nb.name or 'the neighborhood'} is supported by sources",
        parent=loc_parent,
        critical=True
    )
    target_norm = normalize_target(nb.distance_target) or "downtown"
    if target_norm == "airport":
        anchor_text = "Denver International Airport"
    else:
        anchor_text = "downtown Denver (16th Street Mall area)"
    loc_claim = (
        f"The distance from {anchor_text} to {nb.name or 'the neighborhood'} is approximately {nb.distance_miles}."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_supported_leaf,
        sources=nb.sources,
        additional_instruction=(
            "Accept Google Maps or similar route/straight-line links or pages explicitly stating distance. "
            "Allow small rounding differences."
        )
    )

    # Location threshold (custom)
    dist_val = parse_miles(nb.distance_miles)
    if target_norm == "airport":
        threshold = 30.0
    else:
        threshold = 10.0
    evaluator.add_custom_node(
        result=(dist_val is not None and dist_val <= threshold),
        id=f"nh_{idx+1}_location_threshold",
        desc=f"Distance {nb.distance_miles or 'N/A'} to {anchor_text} is within {threshold} miles",
        parent=loc_parent,
        critical=True
    )

    # Transit Access (non-critical nice-to-have)
    transit_parent = evaluator.add_parallel(
        id=f"nh_{idx+1}_transit_access",
        desc="Has access to RTD light rail or other documented transit connectivity (nice-to-have)",
        parent=neigh_node,
        critical=False
    )

    # Transit details provided (custom, non-critical)
    evaluator.add_custom_node(
        result=bool(nb.transit_access and nb.transit_access.strip()),
        id=f"nh_{idx+1}_transit_provided",
        desc="Transit/access details are provided",
        parent=transit_parent,
        critical=False
    )

    # Transit supported by URL (non-critical)
    transit_supported_leaf = evaluator.add_leaf(
        id=f"nh_{idx+1}_transit_supported",
        desc=f"Transit/access details for {nb.name or 'the neighborhood'} are supported by sources",
        parent=transit_parent,
        critical=False
    )
    transit_claim = (
        f"The neighborhood {nb.name or 'the neighborhood'} has the following transit access: {nb.transit_access}."
    )
    await evaluator.verify(
        claim=transit_claim,
        node=transit_supported_leaf,
        sources=nb.sources,
        additional_instruction=(
            "Accept if sources indicate RTD light rail station presence, proximity to downtown by foot, "
            "or other public transit routes serving the neighborhood."
        )
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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

    # Record criteria for reference in summary
    evaluator.add_custom_info(
        info={
            "walk_score_min": 60,
            "price_range_usd": [350000, 650000],
            "price_years_allowed": sorted(list(ALLOWED_PRICE_YEARS)),
            "yoy_max_percent": 5.0,
            "distance_threshold_miles": {"airport": 30.0, "downtown": 10.0},
            "transit": "nice-to-have"
        },
        info_type="investment_criteria",
        info_name="investment_criteria"
    )

    # Extract neighborhoods from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_neighborhoods(),
        template_class=NeighborhoodsExtraction,
        extraction_name="neighborhoods_extraction"
    )

    # Normalize to exactly 4 neighborhoods (pad if fewer)
    neighborhoods: List[Neighborhood] = list(extracted.neighborhoods[:4])
    while len(neighborhoods) < 4:
        neighborhoods.append(Neighborhood())

    # Build high-level task node (non-critical to allow partial credit across neighborhoods)
    task_node = evaluator.add_parallel(
        id="investment_analysis_task",
        desc="Identify 4 Denver metro neighborhoods suitable for multifamily investment based on criteria",
        parent=root,
        critical=False
    )

    # Verify each neighborhood
    for i in range(4):
        await verify_neighborhood(evaluator, task_node, neighborhoods[i], i)

    return evaluator.get_summary()