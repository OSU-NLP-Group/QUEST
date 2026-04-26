import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cvs_fw_site_requirements"
TASK_DESCRIPTION = (
    "A commercial real estate advisory firm has been engaged to prepare a comprehensive site requirements analysis for "
    "CVS Pharmacy. CVS is considering expansion in Fort Worth, Texas, specifically within a 3-mile radius of the GM "
    "Financial headquarters located at 801 Cherry Street, Fort Worth, TX 76102. Based on CVS's published new location "
    "criteria, provide a complete analysis that includes: (1) All mandatory physical site specifications (lot size, site "
    "type, building features); (2) All required location and accessibility features (intersection type, traffic access); "
    "(3) All visibility and signage requirements; (4) Parking requirements from both CVS standards and Fort Worth municipal "
    "regulations; (5) Market demographic requirements (minimum trade area population); (6) Verification that the Fort Worth "
    "market area meets the population density threshold. For each requirement category, cite the specific source documentation "
    "(URLs) where these criteria are published."
)

GM_HQ_ADDRESS = "801 Cherry Street, Fort Worth, TX 76102"
THREE_MILE_RADIUS = "3-mile radius"
CVS_MIN_PARKING = 60
FW_RETAIL_PARKING_RATIO_DENOM = 250  # 1 space per 250 sq ft
PROTOTYPE_FOOTPRINT_TEXT = "95' x 160' (approximately 14,600 square feet)"
DEFAULT_BUILDING_AREA_SF = 14600
MIN_TRADE_AREA_POP = 18000


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ScopeExtraction(BaseModel):
    mentions_fort_worth: Optional[bool] = None
    mentions_gm_hq_address: Optional[bool] = None
    mentions_three_mile_radius: Optional[bool] = None
    gm_hq_address_text: Optional[str] = None


class PhysicalSpecsExtraction(BaseModel):
    freestanding_statement: Optional[str] = None
    lot_size_range_text: Optional[str] = None
    drive_thru_statement: Optional[str] = None
    prototype_building_footprint_text: Optional[str] = None
    zoning_permits_pharmacy_statement: Optional[str] = None
    citations_urls: List[str] = Field(default_factory=list)


class LocationAccessExtraction(BaseModel):
    high_traffic_intersection_statement: Optional[str] = None
    easy_access_signal_statement: Optional[str] = None
    citations_urls: List[str] = Field(default_factory=list)


class VisibilitySignageExtraction(BaseModel):
    high_visibility_pylon_statement: Optional[str] = None
    citations_urls: List[str] = Field(default_factory=list)


class ParkingExtraction(BaseModel):
    cvs_parking_minimum_text: Optional[str] = None
    fw_parking_ratio_text: Optional[str] = None
    building_area_sqft_text: Optional[str] = None
    fw_minimum_spaces_calculated_text: Optional[str] = None
    cvs_citation_urls: List[str] = Field(default_factory=list)
    fw_citation_urls: List[str] = Field(default_factory=list)


class DemographicsExtraction(BaseModel):
    min_trade_area_population_requirement_text: Optional[str] = None
    trade_area_population_estimate_text: Optional[str] = None
    trade_area_population_pass_fail_statement: Optional[str] = None
    requirement_citation_urls: List[str] = Field(default_factory=list)
    population_data_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_scope() -> str:
    return (
        "Extract whether the answer explicitly states the following items:\n"
        "1. The analysis pertains to Fort Worth, Texas.\n"
        "2. The GM Financial headquarters at 801 Cherry Street, Fort Worth, TX 76102 is used as the reference point.\n"
        "3. The considered area is within a 3-mile radius of the reference point.\n"
        "Return:\n"
        "- mentions_fort_worth: boolean\n"
        "- mentions_gm_hq_address: boolean\n"
        "- mentions_three_mile_radius: boolean\n"
        "- gm_hq_address_text: if the address is mentioned, extract the exact address text as it appears; otherwise null."
    )


def prompt_extract_physical() -> str:
    return (
        "Extract the physical site requirement statements and citations provided in the answer. Specifically:\n"
        "- freestanding_statement: text stating the site must be freestanding (not in a shopping center/mall).\n"
        "- lot_size_range_text: text stating the lot size must be between 1.5 and 2 acres.\n"
        "- drive_thru_statement: text stating drive-thru pharmacy capability is required.\n"
        "- prototype_building_footprint_text: text indicating the prototype building footprint (e.g., 95' x 160', ~14,600 sf).\n"
        "- zoning_permits_pharmacy_statement: text stating commercial zoning/entitlements that permit retail pharmacy.\n"
        "- citations_urls: all URLs cited in the answer that support physical site specifications. Extract actual URLs only.\n"
        "If any item is not present, set it to null. If no citations are provided for this category, return an empty array."
    )


def prompt_extract_location_access() -> str:
    return (
        "Extract the location and accessibility requirement statements and citations from the answer. Specifically:\n"
        "- high_traffic_intersection_statement: text stating the site must be at or near a high-traffic intersection.\n"
        "- easy_access_signal_statement: text stating the site must have easy access, ideally with a traffic signal.\n"
        "- citations_urls: all URLs cited that support location/access requirements.\n"
        "Set missing items to null. If no citations, return an empty array."
    )


def prompt_extract_visibility() -> str:
    return (
        "Extract the visibility/signage requirement statements and citations from the answer. Specifically:\n"
        "- high_visibility_pylon_statement: text stating high visibility with pylon sign capability.\n"
        "- citations_urls: all URLs cited that support visibility/signage requirements.\n"
        "Set missing items to null. If no citations, return an empty array."
    )


def prompt_extract_parking() -> str:
    return (
        "Extract the parking requirements and calculation details from the answer, along with category-specific citations.\n"
        "Return:\n"
        "- cvs_parking_minimum_text: text stating CVS parking minimum (e.g., at least 60 cars).\n"
        "- fw_parking_ratio_text: text stating Fort Worth retail parking ratio (e.g., one space per 250 sq ft).\n"
        "- building_area_sqft_text: text stating the building area used for the calculation (e.g., 14,600 sq ft).\n"
        "- fw_minimum_spaces_calculated_text: text with the computed minimum spaces per the FW ratio (e.g., 59 spaces).\n"
        "- cvs_citation_urls: URLs for CVS parking standards cited.\n"
        "- fw_citation_urls: URLs for Fort Worth parking regulations cited.\n"
        "Set missing texts to null. If no citations, return empty arrays."
    )


def prompt_extract_demographics() -> str:
    return (
        "Extract the demographic requirement and verification items from the answer. Return:\n"
        "- min_trade_area_population_requirement_text: text stating the minimum trade area population requirement (e.g., 18,000).\n"
        "- trade_area_population_estimate_text: text with the estimated/measured population for the defined 3-mile radius.\n"
        "- trade_area_population_pass_fail_statement: text stating whether the trade area meets or fails the 18,000 threshold.\n"
        "- requirement_citation_urls: URLs that cite the demographic requirement/policy.\n"
        "- population_data_urls: URLs for the data source used to verify the 3-mile radius population.\n"
        "Set missing texts to null. If no citations, return empty arrays."
    )


# --------------------------------------------------------------------------- #
# Helper parsing functions                                                    #
# --------------------------------------------------------------------------- #
def _to_int_safe(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    s = "".join(ch for ch in text if ch.isdigit())
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _parse_sqft(text: Optional[str]) -> Optional[int]:
    """
    Parse a plausible square footage from text. Prefer large numbers (e.g., 14600).
    """
    if not text:
        return None
    nums: List[int] = []
    current = ""
    for ch in text:
        if ch.isdigit():
            current += ch
        else:
            if current:
                try:
                    nums.append(int(current))
                except Exception:
                    pass
                current = ""
    if current:
        try:
            nums.append(int(current))
        except Exception:
            pass
    if not nums:
        return None
    # Prefer values > 1000 as sf; pick the max
    large = [n for n in nums if n >= 1000]
    if large:
        return max(large)
    # Fallback if only small numbers (like 95 and 160), try product if exactly two numbers
    if len(nums) == 2:
        try:
            prod = nums[0] * nums[1]
            return prod
        except Exception:
            return None
    return None


def _parse_ratio_denom(text: Optional[str]) -> Optional[int]:
    """
    Extract denominator from ratio text like '1 space per 250 square feet'.
    """
    if not text:
        return None
    # Prefer explicit '250' if present
    if "250" in text.replace(",", ""):
        return 250
    val = _to_int_safe(text)
    return val


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


# --------------------------------------------------------------------------- #
# Verification building functions                                             #
# --------------------------------------------------------------------------- #
async def add_citations_nodes(
    evaluator: Evaluator,
    parent_node,
    physical: PhysicalSpecsExtraction,
    location: LocationAccessExtraction,
    visibility: VisibilitySignageExtraction,
    parking: ParkingExtraction,
    demo: DemographicsExtraction,
) -> Dict[str, Any]:
    """
    Build 'Source_Citations_By_Category' node and its critical children.
    Return a dict of the created nodes keyed by category for dependency wiring.
    """
    citations_root = evaluator.add_parallel(
        id="Source_Citations_By_Category",
        desc="Provides published source documentation URLs for each requirement category, as requested.",
        parent=parent_node,
        critical=True,
    )

    physical_ok = bool(physical.citations_urls)
    location_ok = bool(location.citations_urls)
    visibility_ok = bool(visibility.citations_urls)
    parking_ok = bool(parking.cvs_citation_urls) and bool(parking.fw_citation_urls)
    demo_ok = bool(demo.requirement_citation_urls) and bool(demo.population_data_urls)

    node_physical = evaluator.add_custom_node(
        result=physical_ok,
        id="Citations_Physical_Specifications",
        desc="Includes at least one URL citation for the physical site specifications criteria.",
        parent=citations_root,
        critical=True,
    )
    node_location = evaluator.add_custom_node(
        result=location_ok,
        id="Citations_Location_Accessibility",
        desc="Includes at least one URL citation for the location and accessibility criteria.",
        parent=citations_root,
        critical=True,
    )
    node_visibility = evaluator.add_custom_node(
        result=visibility_ok,
        id="Citations_Visibility_Signage",
        desc="Includes at least one URL citation for the visibility and signage criteria.",
        parent=citations_root,
        critical=True,
    )
    node_parking = evaluator.add_custom_node(
        result=parking_ok,
        id="Citations_Parking",
        desc="Includes URL citation(s) for both CVS parking standard(s) and Fort Worth municipal parking regulation(s).",
        parent=citations_root,
        critical=True,
    )
    node_demo = evaluator.add_custom_node(
        result=demo_ok,
        id="Citations_Market_Demographics",
        desc="Includes URL citation(s) for the demographic requirement(s) and for the data source used to verify the trade-area population.",
        parent=citations_root,
        critical=True,
    )

    return {
        "physical": node_physical,
        "location": node_location,
        "visibility": node_visibility,
        "parking": node_parking,
        "demographics": node_demo,
    }


async def verify_scope_and_geography(evaluator: Evaluator, parent_node, scope: ScopeExtraction) -> None:
    scope_node = evaluator.add_parallel(
        id="Scope_and_Geography",
        desc="Analysis is correctly scoped to the requested geography.",
        parent=parent_node,
        critical=True,
    )

    # Fort Worth context
    leaf_fw = evaluator.add_leaf(
        id="Fort_Worth_TX_Context",
        desc="States that the analysis pertains to Fort Worth, Texas.",
        parent=scope_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The analysis pertains to Fort Worth, Texas.",
        node=leaf_fw,
        additional_instruction="Verify the answer text explicitly frames the analysis around Fort Worth, Texas.",
    )

    # GM HQ reference point
    leaf_gm = evaluator.add_leaf(
        id="GM_Financial_HQ_Reference_Point",
        desc=f"Uses GM Financial HQ at {GM_HQ_ADDRESS} as the reference point for the trade area.",
        parent=scope_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The analysis uses GM Financial HQ at {GM_HQ_ADDRESS} as the reference point for the trade area.",
        node=leaf_gm,
        additional_instruction="Allow minor formatting variants of the address; confirm the HQ and address are used as the focal point.",
    )

    # 3-mile radius scope
    leaf_radius = evaluator.add_leaf(
        id="Three_Mile_Radius_Scope",
        desc="States that the considered area is within a 3-mile radius of the reference point.",
        parent=scope_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The considered area is defined as within a 3-mile radius of the reference point.",
        node=leaf_radius,
        additional_instruction="Allow phrasing variants like 'within three miles' or '3 mi radius'.",
    )


async def verify_physical_specs(
    evaluator: Evaluator,
    parent_node,
    physical: PhysicalSpecsExtraction,
    prereq_node,
) -> None:
    physical_node = evaluator.add_parallel(
        id="Physical_Site_Specifications",
        desc="Physical site requirements are stated per the provided constraints.",
        parent=parent_node,
        critical=True,
    )

    # Freestanding
    leaf_free = evaluator.add_leaf(
        id="Freestanding_Site",
        desc="States the site must be freestanding (not within a shopping center or mall).",
        parent=physical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="CVS new location criteria require the site to be freestanding (not within a shopping center or mall).",
        node=leaf_free,
        sources=physical.citations_urls,
        additional_instruction="Confirm that the cited CVS real estate criteria (or equivalent official documentation) specify a freestanding site.",
        extra_prerequisites=[prereq_node],
    )

    # Lot size range
    leaf_lot = evaluator.add_leaf(
        id="Lot_Size_Range",
        desc="States the lot size must be between 1.5 and 2 acres.",
        parent=physical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="CVS new location criteria require the lot size to be between 1.5 and 2.0 acres.",
        node=leaf_lot,
        sources=physical.citations_urls,
        additional_instruction="Accept minor variants like 'approximately' or ranges that include 1.5–2 acres.",
        extra_prerequisites=[prereq_node],
    )

    # Drive-thru capability
    leaf_dt = evaluator.add_leaf(
        id="Drive_Thru_Capability",
        desc="States the site must have drive-thru pharmacy capability.",
        parent=physical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="CVS new location criteria require the site to support a pharmacy drive-thru.",
        node=leaf_dt,
        sources=physical.citations_urls,
        additional_instruction="Confirm the requirement for drive-thru capability or drive-through lane in the official criteria.",
        extra_prerequisites=[prereq_node],
    )

    # Prototype footprint
    leaf_proto = evaluator.add_leaf(
        id="Prototype_Building_Footprint",
        desc="States the prototype building footprint is 95' x 160' (approximately 14,600 square feet).",
        parent=physical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="CVS prototype building footprint is roughly 95 feet by 160 feet (approximately 14,600 square feet).",
        node=leaf_proto,
        sources=physical.citations_urls,
        additional_instruction="Minor rounding is acceptable (e.g., ~14,600 sf).",
        extra_prerequisites=[prereq_node],
    )

    # Zoning allows retail pharmacy
    leaf_zone = evaluator.add_leaf(
        id="Zoning_Allows_Retail_Pharmacy",
        desc="States the site must have commercial zoning (or equivalent entitlement) that permits retail pharmacy use.",
        parent=physical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The site must have commercial zoning or equivalent entitlements permitting retail pharmacy use.",
        node=leaf_zone,
        sources=physical.citations_urls,
        additional_instruction="Confirm the zoning/entitlement requirement as part of CVS site criteria.",
        extra_prerequisites=[prereq_node],
    )


async def verify_location_access(
    evaluator: Evaluator,
    parent_node,
    location: LocationAccessExtraction,
    prereq_node,
) -> None:
    loc_node = evaluator.add_parallel(
        id="Location_and_Accessibility",
        desc="Location/access requirements are stated per the provided constraints.",
        parent=parent_node,
        critical=True,
    )

    # High traffic intersection
    leaf_int = evaluator.add_leaf(
        id="High_Traffic_Intersection",
        desc="States the site must be located at or near a high-traffic intersection.",
        parent=loc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="CVS site criteria prefer locations at or near high-traffic intersections.",
        node=leaf_int,
        sources=location.citations_urls,
        additional_instruction="Verify the cited documentation indicates intersection prominence/traffic levels.",
        extra_prerequisites=[prereq_node],
    )

    # Easy access, preferably signalized
    leaf_sig = evaluator.add_leaf(
        id="Easy_Access_Traffic_Signal_Preference",
        desc="States the site must have easy access, ideally with a traffic signal.",
        parent=loc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="CVS site criteria require easy vehicular access, ideally with a signalized access point.",
        node=leaf_sig,
        sources=location.citations_urls,
        additional_instruction="Confirm preference for signalized access or equivalent phrasing.",
        extra_prerequisites=[prereq_node],
    )


async def verify_visibility_signage(
    evaluator: Evaluator,
    parent_node,
    visibility: VisibilitySignageExtraction,
    prereq_node,
) -> None:
    vis_node = evaluator.add_parallel(
        id="Visibility_and_Signage",
        desc="Visibility/signage requirements are stated per the provided constraints.",
        parent=parent_node,
        critical=True,
    )

    leaf_vis = evaluator.add_leaf(
        id="High_Visibility_with_Pylon_Sign_Capability",
        desc="States the site must have high visibility with pylon sign capability.",
        parent=vis_node,
        critical=True,
    )
    await evaluator.verify(
        claim="CVS site criteria require high visibility with capability for a pylon sign.",
        node=leaf_vis,
        sources=visibility.citations_urls,
        additional_instruction="Confirm visibility/signage requirements; allow synonyms like monument/pylon where applicable.",
        extra_prerequisites=[prereq_node],
    )


async def verify_parking_requirements(
    evaluator: Evaluator,
    parent_node,
    parking: ParkingExtraction,
    physical: PhysicalSpecsExtraction,
    prereq_node,
) -> None:
    park_node = evaluator.add_parallel(
        id="Parking_Requirements",
        desc="Parking requirements include both CVS standard and Fort Worth regulation, with an applied calculation.",
        parent=parent_node,
        critical=True,
    )

    # CVS minimum parking
    leaf_cvs = evaluator.add_leaf(
        id="CVS_Parking_Minimum",
        desc="States that parking must accommodate at least 60 cars (CVS standard per constraints).",
        parent=park_node,
        critical=True,
    )
    await evaluator.verify(
        claim="CVS standard requires that parking accommodate at least 60 cars.",
        node=leaf_cvs,
        sources=parking.cvs_citation_urls,
        additional_instruction="Verify the minimum parking capacity requirement in official CVS criteria.",
        extra_prerequisites=[prereq_node],
    )

    # Fort Worth retail parking ratio
    leaf_fw_ratio = evaluator.add_leaf(
        id="Fort_Worth_Parking_Ratio",
        desc="States Fort Worth’s retail parking requirement is one space per 250 square feet of building area (per constraints).",
        parent=park_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Fort Worth retail parking requirement is one parking space per 250 square feet of building area.",
        node=leaf_fw_ratio,
        sources=parking.fw_citation_urls,
        additional_instruction="Verify the ratio (1 per 250 sf) in Fort Worth municipal code/standards.",
        extra_prerequisites=[prereq_node],
    )

    # Minimum spaces calculation (custom math check)
    # Determine building area: prefer parking.building_area_sqft_text; fallback to physical.prototype text or default.
    area_from_parking = _parse_sqft(parking.building_area_sqft_text)
    area_from_physical = _parse_sqft(physical.prototype_building_footprint_text)
    building_area = area_from_parking or area_from_physical or DEFAULT_BUILDING_AREA_SF

    denom = _parse_ratio_denom(parking.fw_parking_ratio_text) or FW_RETAIL_PARKING_RATIO_DENOM
    expected_min_spaces = _ceil_div(building_area, denom)

    reported_min_spaces = _to_int_safe(parking.fw_minimum_spaces_calculated_text)

    calc_ok = reported_min_spaces is not None and reported_min_spaces >= expected_min_spaces

    evaluator.add_custom_node(
        result=calc_ok,
        id="Fort_Worth_Minimum_Spaces_Calculation",
        desc="Applies the Fort Worth ratio to the stated 14,600 sq ft building area and reports the resulting minimum required number of spaces.",
        parent=park_node,
        critical=True,
    )


async def verify_demographics(
    evaluator: Evaluator,
    parent_node,
    demo: DemographicsExtraction,
    prereq_node,
) -> None:
    demo_node = evaluator.add_parallel(
        id="Market_Demographics_and_Threshold_Verification",
        desc="States demographic threshold(s) and verifies whether the defined trade area meets them.",
        parent=parent_node,
        critical=True,
    )

    # Minimum trade area population requirement
    leaf_req = evaluator.add_leaf(
        id="Minimum_Trade_Area_Population_Requirement",
        desc="States the minimum trade area population requirement is 18,000 people (per constraints).",
        parent=demo_node,
        critical=True,
    )
    await evaluator.verify(
        claim="CVS site criteria require a minimum trade area population of 18,000 people.",
        node=leaf_req,
        sources=demo.requirement_citation_urls,
        additional_instruction="Confirm the stated minimum population threshold in official CVS criteria.",
        extra_prerequisites=[prereq_node],
    )

    # Population estimate / measure and pass/fail conclusion
    leaf_est = evaluator.add_leaf(
        id="Trade_Area_Population_Estimate_or_Measure",
        desc="Provides an estimate/measurement for population in the defined trade area (3-mile radius) and a clear pass/fail conclusion vs 18,000.",
        parent=demo_node,
        critical=True,
    )

    # Build a claim that the 3-mile radius population around the GM HQ meets/exceeds 18,000
    claim_text = (
        f"The population within a {THREE_MILE_RADIUS} of {GM_HQ_ADDRESS} is at least {MIN_TRADE_AREA_POP} people."
    )
    await evaluator.verify(
        claim=claim_text,
        node=leaf_est,
        sources=demo.population_data_urls,
        additional_instruction=(
            "Use the cited demographic data source(s) to verify the 3-mile radius population for the given address meets or exceeds 18,000. "
            "Allow minor rounding differences and different data vintages."
        ),
        extra_prerequisites=[prereq_node],
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
    Evaluate an answer for the CVS Fort Worth site requirements analysis task.
    """
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

    # Extract all category information in parallel
    scope_extraction_task = evaluator.extract(
        prompt=prompt_extract_scope(),
        template_class=ScopeExtraction,
        extraction_name="scope_geography",
    )
    physical_extraction_task = evaluator.extract(
        prompt=prompt_extract_physical(),
        template_class=PhysicalSpecsExtraction,
        extraction_name="physical_specs",
    )
    location_extraction_task = evaluator.extract(
        prompt=prompt_extract_location_access(),
        template_class=LocationAccessExtraction,
        extraction_name="location_access",
    )
    visibility_extraction_task = evaluator.extract(
        prompt=prompt_extract_visibility(),
        template_class=VisibilitySignageExtraction,
        extraction_name="visibility_signage",
    )
    parking_extraction_task = evaluator.extract(
        prompt=prompt_extract_parking(),
        template_class=ParkingExtraction,
        extraction_name="parking_requirements",
    )
    demo_extraction_task = evaluator.extract(
        prompt=prompt_extract_demographics(),
        template_class=DemographicsExtraction,
        extraction_name="market_demographics",
    )

    (
        scope_extraction,
        physical_extraction,
        location_extraction,
        visibility_extraction,
        parking_extraction,
        demo_extraction,
    ) = await asyncio.gather(
        scope_extraction_task,
        physical_extraction_task,
        location_extraction_task,
        visibility_extraction_task,
        parking_extraction_task,
        demo_extraction_task,
    )

    # Create critical analysis root node under evaluator.root
    cvs_root = evaluator.add_parallel(
        id="CVS_Site_Requirements_Analysis",
        desc="Meets the question’s required CVS site-criteria analysis for the specified Fort Worth submarket and includes verifiable, cited requirements.",
        parent=root,
        critical=True,
    )

    # Build citations nodes first and capture them for prerequisites
    citations_nodes = await add_citations_nodes(
        evaluator,
        cvs_root,
        physical_extraction,
        location_extraction,
        visibility_extraction,
        parking_extraction,
        demo_extraction,
    )

    # Build and verify each category
    await verify_scope_and_geography(evaluator, cvs_root, scope_extraction)
    await verify_physical_specs(evaluator, cvs_root, physical_extraction, citations_nodes["physical"])
    await verify_location_access(evaluator, cvs_root, location_extraction, citations_nodes["location"])
    await verify_visibility_signage(evaluator, cvs_root, visibility_extraction, citations_nodes["visibility"])
    await verify_parking_requirements(evaluator, cvs_root, parking_extraction, physical_extraction, citations_nodes["parking"])
    await verify_demographics(evaluator, cvs_root, demo_extraction, citations_nodes["demographics"])

    # Return structured evaluation summary
    return evaluator.get_summary()