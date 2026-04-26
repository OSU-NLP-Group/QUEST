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
TASK_ID = "kl_office_market_q4_2024"
TASK_DESCRIPTION = """
Provide a comprehensive analysis of the Kuala Lumpur commercial office real estate market as of Q4 2024 (or the most recent available quarter). Your analysis must include the following specific information:

1. Market Overview: Total office supply in square feet, overall occupancy rate (%), and private Purpose-Built Office (PBO) occupancy rate (%)

2. Rental Rates: Average rental rates in RM per square foot (psf) for: (a) Prime/Golden Triangle area, (b) City Fringe area, and (c) Grade A offices

3. Largest Recent Completion: For the largest office building completed in 2023-2024 by Net Leasable Area (NLA), provide: building name, NLA in square feet, completion year, and green building certification (type and level)

4. 2025 Pipeline: Total number of office buildings scheduled for completion in 2025, total NLA of the 2025 pipeline in square feet, and identify the largest building in the 2025 pipeline by NLA

5. Market Trends: Quantify the occupancy rate premium (in percentage or percentage range) that green-certified office buildings command over non-certified buildings, and describe the primary market trend affecting office demand

6. Source Documentation: Provide reference URLs for your data sources

All data must be sourced from credible commercial real estate market reports, industry publications, or official building documentation.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class KLOfficeAnalysisExtraction(BaseModel):
    # 1) Market Overview
    total_office_supply_sqft: Optional[str] = None
    overall_occupancy_pct: Optional[str] = None
    private_pbo_occupancy_pct: Optional[str] = None

    # 2) Rental Rates
    prime_gt_rent_psf: Optional[str] = None
    city_fringe_rent_psf: Optional[str] = None
    grade_a_rent_psf: Optional[str] = None

    # 3) Largest Recent Completion (2023-2024)
    largest_completion_building_name: Optional[str] = None
    largest_completion_nla_sqft: Optional[str] = None
    largest_completion_year: Optional[str] = None
    largest_completion_green_cert_type: Optional[str] = None
    largest_completion_green_cert_level: Optional[str] = None

    # 4) 2025 Pipeline
    pipeline_2025_num_buildings: Optional[str] = None
    pipeline_2025_total_nla_sqft: Optional[str] = None
    pipeline_2025_largest_building_name: Optional[str] = None
    pipeline_2025_largest_building_nla_sqft: Optional[str] = None

    # 5) Market Trends
    green_occupancy_premium: Optional[str] = None  # percentage or percentage range as string
    primary_demand_trend: Optional[str] = None

    # 6) Sources (section-specific + global)
    sources_overview: List[str] = Field(default_factory=list)
    sources_rental: List[str] = Field(default_factory=list)
    sources_largest_completion: List[str] = Field(default_factory=list)
    sources_pipeline_2025: List[str] = Field(default_factory=list)
    sources_trends: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_kl_office_analysis() -> str:
    return """
You must extract structured data exactly as stated in the answer. Do not invent, average, or recalculate values. Return strings for all values (even numbers or ranges). Include percent signs or "psf" only if shown in the answer.

Extract the following fields (set to null if missing). For any URL lists, include only explicit URLs present in the answer (plain or markdown), normalized to full form (prepend http:// if missing):

1) MARKET OVERVIEW (as of Q4 2024, or the most recent available quarter close to that period):
- total_office_supply_sqft: total office supply for Kuala Lumpur in square feet (string as presented)
- overall_occupancy_pct: overall occupancy rate percentage for Kuala Lumpur office market (string, may include %)
- private_pbo_occupancy_pct: private Purpose-Built Office (PBO) occupancy rate percentage (string, may include %)

2) RENTAL RATES (RM per square foot, psf; ranges allowed as strings):
- prime_gt_rent_psf: rental for Prime/Golden Triangle (e.g., KLCC/Golden Triangle); keep exact format (e.g., "RM 7.00–10.00 psf")
- city_fringe_rent_psf: rental for City Fringe area; keep exact format
- grade_a_rent_psf: rental for Grade A offices; keep exact format

3) LARGEST RECENT COMPLETION (2023–2024, by NLA):
- largest_completion_building_name
- largest_completion_nla_sqft: NLA in square feet (string, keep units/format from answer)
- largest_completion_year: "2023" or "2024" (string)
- largest_completion_green_cert_type: e.g., "LEED", "GBI", "GreenRE", "BCA Green Mark", etc. (string)
- largest_completion_green_cert_level: e.g., "Platinum", "Gold", "Silver", "Certified", etc. (string)

4) 2025 PIPELINE:
- pipeline_2025_num_buildings: number of office buildings scheduled for 2025 completion (string)
- pipeline_2025_total_nla_sqft: total NLA for 2025 pipeline in square feet (string)
- pipeline_2025_largest_building_name
- pipeline_2025_largest_building_nla_sqft: NLA of the largest 2025 building (string)

5) MARKET TRENDS:
- green_occupancy_premium: occupancy rate premium (percentage or percentage range) for green-certified vs non-certified buildings (string, keep exactly as stated, e.g., "3–6%")
- primary_demand_trend: a brief phrase the answer cites as the primary market trend affecting office demand

6) SOURCES:
- sources_overview: URLs the answer cites that support the market overview values
- sources_rental: URLs that support the rental rates
- sources_largest_completion: URLs that support the largest completion details
- sources_pipeline_2025: URLs that support the 2025 pipeline data
- sources_trends: URLs that support the trends/premium statements
- reference_urls: any general reference URLs listed in the answer (e.g., a references section)

General rules:
- Extract only what appears in the answer. Keep original formatting (including RM, psf, %, ranges, commas).
- Do not perform any calculation; do not convert sqm to sqft. If the answer shows sqm, still extract exactly as shown.
- For URLs: include only explicit URLs found in the answer text; for markdown links, extract the URL target. If a URL is missing protocol, prepend "http://".
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def _combine_sources(*args: List[str]) -> List[str]:
    combined: List[str] = []
    for arr in args:
        combined.extend(arr or [])
    return _clean_urls(combined)


def _has_value(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _has_urls(urls: Optional[List[str]]) -> bool:
    return len(_clean_urls(urls)) > 0


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_field_with_sources(
    evaluator: Evaluator,
    parent_node,
    *,
    field_value: Optional[str],
    field_sources: List[str],
    leaf_id: str,
    leaf_desc: str,
    claim_text: str,
    add_ins: str,
    critical: bool = True,
) -> None:
    """
    Build a small sequential chain:
      1) value_present (critical)
      2) sources_provided (critical)
      3) verification leaf (critical by default)
    """
    seq = evaluator.add_sequential(
        id=f"{leaf_id}_seq",
        desc=f"Gated verification for {leaf_desc}",
        parent=parent_node,
        critical=critical,
    )

    evaluator.add_custom_node(
        result=_has_value(field_value),
        id=f"{leaf_id}_value_present",
        desc=f"Value for {leaf_desc} is provided in the answer",
        parent=seq,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_urls(field_sources),
        id=f"{leaf_id}_sources_provided",
        desc=f"At least one supporting URL is provided for {leaf_desc}",
        parent=seq,
        critical=True,
    )

    verify_leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=seq,
        critical=True,
    )

    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=field_sources,
        additional_instruction=add_ins,
    )


async def verify_market_overview(evaluator: Evaluator, parent, ex: KLOfficeAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Market_Overview_Data",
        desc="Provides total office supply, overall occupancy rate, and private PBO occupancy rate for Kuala Lumpur as of Q4 2024",
        parent=parent,
        critical=True,
    )

    sources = _combine_sources(ex.sources_overview, ex.reference_urls)

    common_addins = (
        "Verify that the value pertains to the Kuala Lumpur office market (city or the commonly reported KL office market scope; "
        "avoid values for Malaysia as a whole unless explicitly matched). Timeframe should be Q4 2024 or, if not available, "
        "the nearest recent quarter (e.g., Q3 2024 or Q1 2025) clearly indicated by the source. "
        "Allow minor rounding differences and formatting variants."
    )

    # Total Office Supply (sqft). Accept sqm evidence if the figure is equivalent after unit conversion.
    await _verify_field_with_sources(
        evaluator,
        node,
        field_value=ex.total_office_supply_sqft,
        field_sources=sources,
        leaf_id="Total_Office_Supply",
        leaf_desc="States total office supply in Kuala Lumpur in square feet as of Q4 2024",
        claim_text=f"The total office supply for Kuala Lumpur is {ex.total_office_supply_sqft} in square feet as of Q4 2024 or the most recent available quarter.",
        add_ins=common_addins + " If the source reports area in square meters, treat as supported if converting (1 sqm = 10.7639 sqft) "
                                 "would reasonably match the stated sqft within ±5% tolerance.",
    )

    # Overall Occupancy Rate
    await _verify_field_with_sources(
        evaluator,
        node,
        field_value=ex.overall_occupancy_pct,
        field_sources=sources,
        leaf_id="Overall_Occupancy_Rate",
        leaf_desc="States overall occupancy rate for Kuala Lumpur office market as a percentage",
        claim_text=f"The overall office occupancy rate in Kuala Lumpur is {ex.overall_occupancy_pct} as of Q4 2024 or the most recent available quarter.",
        add_ins=common_addins + " Accept small rounding (±1%) and percent format variants.",
    )

    # Private PBO Occupancy Rate
    await _verify_field_with_sources(
        evaluator,
        node,
        field_value=ex.private_pbo_occupancy_pct,
        field_sources=sources,
        leaf_id="Private_PBO_Occupancy_Rate",
        leaf_desc="States private Purpose-Built Office (PBO) occupancy rate as a percentage",
        claim_text=f"The private Purpose-Built Office (PBO) occupancy rate in Kuala Lumpur is {ex.private_pbo_occupancy_pct} "
                   f"as of Q4 2024 or the most recent available quarter.",
        add_ins=common_addins + " Confirm that the figure specifically refers to Private PBO (private sector purpose-built offices), "
                                 "not the total PBO including government or other categories, unless the answer and source clearly align.",
    )


async def verify_rental_market(evaluator: Evaluator, parent, ex: KLOfficeAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Rental_Market_Data",
        desc="Provides rental rates in RM psf for Prime/Golden Triangle, City Fringe, and Grade A segments",
        parent=parent,
        critical=True,
    )

    sources = _combine_sources(ex.sources_rental, ex.reference_urls)

    base_ins = (
        "Verify the average/asking rental rate or range in RM per square foot (psf) for the specified segment. "
        "Allow formatting differences (e.g., 'RM 7.00–10.00 psf', 'RM7-10 psf'). "
        "Timeframe should be Q4 2024 or the most recent nearby quarter and clearly related to Kuala Lumpur. "
    )

    # Prime / Golden Triangle (KLCC/Golden Triangle synonyms acceptable)
    await _verify_field_with_sources(
        evaluator,
        node,
        field_value=ex.prime_gt_rent_psf,
        field_sources=sources,
        leaf_id="Prime_Golden_Triangle_Rental",
        leaf_desc="States rental rate range for Prime/Golden Triangle area in RM per square foot",
        claim_text=f"The average rental rate in the Prime/Golden Triangle area of Kuala Lumpur is {ex.prime_gt_rent_psf} (RM psf).",
        add_ins=base_ins + "Treat 'Prime/Golden Triangle' as synonymous with KLCC/Golden Triangle core. "
                           "Accept KL city centre/prime CBD if it is clearly equivalent.",
    )

    # City Fringe
    await _verify_field_with_sources(
        evaluator,
        node,
        field_value=ex.city_fringe_rent_psf,
        field_sources=sources,
        leaf_id="City_Fringe_Rental",
        leaf_desc="States rental rate range for City Fringe area in RM per square foot",
        claim_text=f"The average rental rate in the City Fringe area of Kuala Lumpur is {ex.city_fringe_rent_psf} (RM psf).",
        add_ins=base_ins + "Segment synonyms such as 'Fringe City' or 'City Fringe' are acceptable if contextually equivalent.",
    )

    # Grade A
    await _verify_field_with_sources(
        evaluator,
        node,
        field_value=ex.grade_a_rent_psf,
        field_sources=sources,
        leaf_id="Grade_A_Rental",
        leaf_desc="States rental rate range for Grade A offices in RM per square foot",
        claim_text=f"The average rental rate for Grade A office space in Kuala Lumpur is {ex.grade_a_rent_psf} (RM psf).",
        add_ins=base_ins + "Ensure the grade mentioned is Grade A for office buildings in Kuala Lumpur.",
    )


async def verify_largest_recent_completion(evaluator: Evaluator, parent, ex: KLOfficeAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Largest_Recent_Completion_Data",
        desc="Provides complete information about the largest office building completed in 2023-2024 by NLA",
        parent=parent,
        critical=True,
    )

    sources = _combine_sources(ex.sources_largest_completion, ex.reference_urls)

    # Add gate for presence and sources at section level
    gate = evaluator.add_sequential(
        id="Largest_Recent_Completion_gate",
        desc="Gate checks for Largest Recent Completion section",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_value(ex.largest_completion_building_name),
        id="Largest_Recent_Completion_building_present",
        desc="Largest completion building name is provided",
        parent=gate,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_urls(sources),
        id="Largest_Recent_Completion_sources_provided",
        desc="At least one supporting URL is provided for Largest Recent Completion",
        parent=gate,
        critical=True,
    )

    # Building Name (verify claim includes "largest by NLA in 2023–2024")
    leaf_building_name = evaluator.add_leaf(
        id="Building_Name",
        desc="States the name of the largest office building completed in 2023-2024 by NLA",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The largest office building by NLA completed in 2023–2024 in Kuala Lumpur is {ex.largest_completion_building_name}."
        ),
        node=leaf_building_name,
        sources=sources,
        additional_instruction=(
            "Confirm that the cited sources indicate this building is the largest (by net leasable area) among office completions "
            "during 2023–2024 in Kuala Lumpur or the commonly defined KL office market area."
        ),
        extra_prerequisites=[gate],
    )

    # Building NLA
    leaf_nla = evaluator.add_leaf(
        id="Building_NLA",
        desc="States the Net Leasable Area (NLA) of the largest building in square feet",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The net leasable area (NLA) of {ex.largest_completion_building_name} is {ex.largest_completion_nla_sqft} in square feet.",
        node=leaf_nla,
        sources=sources,
        additional_instruction=(
            "If the source reports area in square meters, treat as supported if converting (1 sqm = 10.7639 sqft) would reasonably match "
            "the stated sqft value within ±5% tolerance. Prefer explicitly stated NLA over GFA if both are present."
        ),
        extra_prerequisites=[gate],
    )

    # Completion Year
    leaf_year = evaluator.add_leaf(
        id="Completion_Year",
        desc="States the year of completion (2023 or 2024)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The building {ex.largest_completion_building_name} was completed in {ex.largest_completion_year}.",
        node=leaf_year,
        sources=sources,
        additional_instruction=(
            "The completion year should be explicitly 2023 or 2024 in the cited source. "
            "If the source indicates practical completion/CCC/TOP in that year, consider it as completion."
        ),
        extra_prerequisites=[gate],
    )

    # Green Certification (type + level)
    leaf_cert = evaluator.add_leaf(
        id="Green_Certification",
        desc="States the green building certification type (e.g., LEED, GreenRE) and level (e.g., Platinum, Gold)",
        parent=node,
        critical=True,
    )
    cert_combo = None
    if _has_value(ex.largest_completion_green_cert_type) and _has_value(ex.largest_completion_green_cert_level):
        cert_combo = f"{ex.largest_completion_green_cert_type} at {ex.largest_completion_green_cert_level} level"
    else:
        # Fall back to whatever was provided; still verify using what's present
        cert_combo = f"{ex.largest_completion_green_cert_type or ''} {ex.largest_completion_green_cert_level or ''}".strip()

    await evaluator.verify(
        claim=f"The building {ex.largest_completion_building_name} holds a green building certification: {cert_combo}.",
        node=leaf_cert,
        sources=sources,
        additional_instruction=(
            "Verify the certification scheme (e.g., LEED, GBI, GreenRE, Green Mark) and the level (e.g., Platinum, Gold). "
            "Minor wording variants are acceptable as long as the certification type and level are clearly supported."
        ),
        extra_prerequisites=[gate],
    )


async def verify_pipeline_2025(evaluator: Evaluator, parent, ex: KLOfficeAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Pipeline_2025_Data",
        desc="Provides information about office buildings scheduled for completion in 2025",
        parent=parent,
        critical=True,
    )

    sources = _combine_sources(ex.sources_pipeline_2025, ex.reference_urls)

    gate = evaluator.add_sequential(
        id="Pipeline_2025_gate",
        desc="Gate checks for 2025 pipeline section",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_urls(sources),
        id="Pipeline_2025_sources_provided",
        desc="At least one supporting URL is provided for the 2025 pipeline",
        parent=gate,
        critical=True,
    )

    common_ins = (
        "Verify that the information pertains specifically to the 2025 completion pipeline for Kuala Lumpur (or the commonly "
        "reported KL office market/Klang Valley if the answer relies on that scope). Minor rounding and formatting differences are acceptable."
    )

    # Number of buildings
    leaf_num = evaluator.add_leaf(
        id="Number_of_Buildings_2025",
        desc="States the number of office buildings scheduled for completion in 2025",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The number of office buildings scheduled for completion in 2025 is {ex.pipeline_2025_num_buildings}.",
        node=leaf_num,
        sources=sources,
        additional_instruction=common_ins,
        extra_prerequisites=[gate],
    )

    # Total pipeline NLA
    leaf_total_nla = evaluator.add_leaf(
        id="Total_Pipeline_NLA_2025",
        desc="States the total NLA of 2025 office pipeline in square feet",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The total net leasable area (NLA) scheduled for 2025 completion is {ex.pipeline_2025_total_nla_sqft} square feet.",
        node=leaf_total_nla,
        sources=sources,
        additional_instruction=common_ins + " If sources report sqm, accept with conversion equivalence (1 sqm = 10.7639 sqft) within ±5%.",
        extra_prerequisites=[gate],
    )

    # Largest building in 2025 pipeline
    leaf_largest_2025 = evaluator.add_leaf(
        id="Largest_Building_2025",
        desc="Identifies the largest building in the 2025 pipeline by NLA with its name and size",
        parent=node,
        critical=True,
    )
    claim_str = (
        f"The largest office building in the 2025 Kuala Lumpur pipeline is {ex.pipeline_2025_largest_building_name} "
        f"with {ex.pipeline_2025_largest_building_nla_sqft} square feet of NLA."
    )
    await evaluator.verify(
        claim=claim_str,
        node=leaf_largest_2025,
        sources=sources,
        additional_instruction=common_ins,
        extra_prerequisites=[gate],
    )


async def verify_market_trends(evaluator: Evaluator, parent, ex: KLOfficeAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Market_Trend_Analysis",
        desc="Provides quantitative and qualitative analysis of key market trends",
        parent=parent,
        critical=True,
    )

    sources = _combine_sources(ex.sources_trends, ex.reference_urls)

    gate = evaluator.add_sequential(
        id="Trends_gate",
        desc="Gate checks for trend analysis section",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_urls(sources),
        id="Trends_sources_provided",
        desc="At least one supporting URL is provided for market trends",
        parent=gate,
        critical=True,
    )

    # Green-certified occupancy premium
    leaf_premium = evaluator.add_leaf(
        id="Green_Building_Occupancy_Premium",
        desc="Quantifies the occupancy rate premium for green-certified buildings over non-certified buildings as a percentage or percentage range",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Green-certified office buildings command an occupancy rate premium of {ex.green_occupancy_premium} compared to non-certified buildings in Kuala Lumpur.",
        node=leaf_premium,
        sources=sources,
        additional_instruction=(
            "Confirm that the cited material quantifies an occupancy premium (difference in occupancy rates) for green-certified buildings "
            "vs non-certified. Accept a single percentage or a range (e.g., '3–6%'). Minor wording differences allowed."
        ),
        extra_prerequisites=[gate],
    )

    # Primary demand trend
    leaf_trend = evaluator.add_leaf(
        id="Primary_Demand_Trend",
        desc="Describes the primary market trend affecting office demand in Kuala Lumpur",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The primary market trend affecting office demand in Kuala Lumpur is: {ex.primary_demand_trend}.",
        node=leaf_trend,
        sources=sources,
        additional_instruction=(
            "Verify that the described theme is highlighted by the source as a key/primary trend affecting office demand "
            "(e.g., flight-to-quality/green, ESG-driven upgrades, hybrid work impacts, new supply overhang, etc.). "
            "Paraphrase matches are acceptable if the essence aligns."
        ),
        extra_prerequisites=[gate],
    )


async def verify_sources_block(evaluator: Evaluator, parent, ex: KLOfficeAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Source_Documentation",
        desc="Provides reference URLs for data sources",
        parent=parent,
        critical=True,  # Mark critical to comply with critical parent constraint and task requirement.
    )

    evaluator.add_custom_node(
        result=_has_urls(ex.reference_urls) or any([
            _has_urls(ex.sources_overview),
            _has_urls(ex.sources_rental),
            _has_urls(ex.sources_largest_completion),
            _has_urls(ex.sources_pipeline_2025),
            _has_urls(ex.sources_trends),
        ]),
        id="Reference_URLs_Provided",
        desc="Includes at least one credible reference URL for commercial real estate market data",
        parent=node,
        critical=True,
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
    # 1) Initialize evaluator
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

    # 2) Top-level critical node representing the whole rubric
    analysis_node = evaluator.add_parallel(
        id="KL_Office_Market_Analysis_Complete",
        desc="Complete and accurate analysis of Kuala Lumpur commercial office real estate market as of Q4 2024 with all required data components",
        parent=root,
        critical=True,
    )

    # 3) Extraction
    extracted: KLOfficeAnalysisExtraction = await evaluator.extract(
        prompt=prompt_extract_kl_office_analysis(),
        template_class=KLOfficeAnalysisExtraction,
        extraction_name="kl_office_analysis",
    )

    # 4) Build and run verification subtrees
    await verify_market_overview(evaluator, analysis_node, extracted)
    await verify_rental_market(evaluator, analysis_node, extracted)
    await verify_largest_recent_completion(evaluator, analysis_node, extracted)
    await verify_pipeline_2025(evaluator, analysis_node, extracted)
    await verify_market_trends(evaluator, analysis_node, extracted)
    await verify_sources_block(evaluator, analysis_node, extracted)

    # 5) Return evaluation summary
    return evaluator.get_summary()