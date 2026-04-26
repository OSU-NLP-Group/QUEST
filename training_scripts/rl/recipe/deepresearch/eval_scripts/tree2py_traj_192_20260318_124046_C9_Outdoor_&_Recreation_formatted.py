import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_accessible_outdoor_facilities"
TASK_DESCRIPTION = (
    "A disability advocacy organization is developing a comprehensive accessibility guide for outdoor recreation in California. "
    "They need to document 4 distinct types of accessible facilities that showcase different outdoor activities available to wheelchair users and people with mobility disabilities. "
    "Identify and provide detailed specifications for the following 4 facilities: "
    "(1) An accessible paved trail in a California National Park that is wheelchair-accessible with documented specifications including length and surface type; "
    "(2) An accessible camping facility in California with documented accessible campsites, accessible restrooms, and accessible amenities; "
    "(3) An adaptive skiing program location in California that offers adaptive ski lessons with specialized equipment for individuals with disabilities; "
    "(4) An accessible day-use or picnic area in a California State Park with accessible tables and paths. "
    "For each facility, provide the specific name and location, detailed specifications of accessible features, and verification through reference URLs from official sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TrailFacility(BaseModel):
    name: Optional[str] = None
    park: Optional[str] = None  # e.g., Yosemite National Park
    location: Optional[str] = None  # more specific area if provided
    accessible_designation_text: Optional[str] = None  # words like "wheelchair-accessible", "ADA accessible"
    surface_type: Optional[str] = None  # asphalt, concrete, boardwalk, firm/stable, etc.
    width: Optional[str] = None  # e.g., "48 inches", "4 ft"
    length: Optional[str] = None  # e.g., "1 mile", "0.8 mi"
    sources: List[str] = Field(default_factory=list)


class CampingFacility(BaseModel):
    name: Optional[str] = None
    park_or_area: Optional[str] = None  # park/campground system
    location: Optional[str] = None  # city/region if present
    accessible_campsites_info: Optional[str] = None  # e.g., "2 ADA sites"
    accessible_restrooms_info: Optional[str] = None
    accessible_picnic_tables_info: Optional[str] = None
    accessible_routes_info: Optional[str] = None  # paved/level/accessible routes
    accessible_showers_info: Optional[str] = None  # may be None; only required if showers exist
    sources: List[str] = Field(default_factory=list)


class SkiProgramFacility(BaseModel):
    resort_or_location: Optional[str] = None  # e.g., "Palisades Tahoe", "Mammoth Mountain"
    region_or_city: Optional[str] = None
    program_name: Optional[str] = None  # e.g., "Achieve Tahoe", "Disabled Sports Eastern Sierra"
    equipment_types: List[str] = Field(default_factory=list)  # mono-ski, bi-ski, outriggers, sit-ski, etc.
    equipment_provided: Optional[str] = None  # text indicating provided by program
    lessons_info: Optional[str] = None  # adaptive lessons/instruction mention
    access_info: Optional[str] = None  # reservations/contact/how to book
    sources: List[str] = Field(default_factory=list)


class PicnicFacility(BaseModel):
    park_name: Optional[str] = None  # California State Park name
    picnic_area_name: Optional[str] = None  # site/area name if provided
    location: Optional[str] = None  # city/region if present
    accessible_tables_info: Optional[str] = None
    accessible_paths_info: Optional[str] = None
    accessible_parking_info: Optional[str] = None
    additional_amenities_info: Optional[str] = None  # e.g., accessible restrooms/fire rings
    sources: List[str] = Field(default_factory=list)


class AccessibilityFacilitiesExtraction(BaseModel):
    trail: Optional[TrailFacility] = None
    camping: Optional[CampingFacility] = None
    skiing: Optional[SkiProgramFacility] = None
    picnic: Optional[PicnicFacility] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract structured information for FOUR facilities described in the answer. Return a single JSON object with keys:
    - trail: An accessible paved trail in a California National Park
      Fields:
        • name: Specific trail name
        • park: The National Park name (must be in California)
        • location: Specific area/location if provided
        • accessible_designation_text: Exact phrase(s) indicating wheelchair/ADA accessible if present
        • surface_type: Surface type as stated (e.g., paved, asphalt, concrete, boardwalk, firm/stable)
        • width: Documented width (e.g., "36 inches", "4 ft"), if given
        • length: Specific length as stated (e.g., "1 mile", "0.8 mi")
        • sources: All URL(s) cited for this trail in the answer (official/authoritative if available)
    - camping: An accessible camping facility in California
      Fields:
        • name
        • park_or_area
        • location
        • accessible_campsites_info: Exact text indicating accessible sites exist or quantity
        • accessible_restrooms_info
        • accessible_picnic_tables_info
        • accessible_routes_info: Text indicating level surfaces/paved/accessible routes
        • accessible_showers_info: If showers are provided and accessible; otherwise null
        • sources: All URL(s) cited for this camping facility
    - skiing: An adaptive skiing program location in California
      Fields:
        • resort_or_location
        • region_or_city
        • program_name
        • equipment_types: List of specific adaptive equipment types mentioned (e.g., mono-ski, bi-ski, sit-ski, outriggers)
        • equipment_provided: Text indicating equipment is provided by the program, if stated
        • lessons_info: Text indicating adaptive lessons/instruction are offered
        • access_info: Text indicating reservation/contact/how to access the program
        • sources: All URL(s) cited for this adaptive skiing program
    - picnic: An accessible day-use or picnic area in a California State Park
      Fields:
        • park_name
        • picnic_area_name
        • location
        • accessible_tables_info
        • accessible_paths_info
        • accessible_parking_info
        • additional_amenities_info
        • sources: All URL(s) cited for this picnic area
    IMPORTANT:
    1) Extract ONLY what is explicitly present in the answer text. Do not invent values.
    2) For any field not mentioned, return null (or empty list for arrays).
    3) For sources: extract the exact URLs mentioned (including ones inside markdown links). If no URL is provided for a facility, return an empty list for that facility's sources.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def _sources_list(xs: Optional[List[str]]) -> List[str]:
    if not xs:
        return []
    # Deduplicate while preserving order; simple normalization
    seen = set()
    out: List[str] = []
    for u in xs:
        if not isinstance(u, str):
            continue
        url = u.strip()
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _fmt_list(items: List[str]) -> str:
    return ", ".join(items) if items else ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_trail_verification(evaluator: Evaluator, parent_node, trail: Optional[TrailFacility]) -> None:
    node = evaluator.add_parallel(
        id="accessible_trail_facility",
        desc="Verify the accessible paved trail in a California National Park meets all requirements",
        parent=parent_node,
        critical=False  # Keep parent non-critical to comply with framework constraint rules
    )

    t_name = trail.name if trail else ""
    t_park = trail.park if trail else ""
    t_loc = trail.location if trail else ""
    t_design = trail.accessible_designation_text if trail else ""
    t_surface = trail.surface_type if trail else ""
    t_width = trail.width if trail else ""
    t_length = trail.length if trail else ""
    t_sources = _sources_list(trail.sources if trail else [])

    # 1) Location identification (critical)
    n_loc = evaluator.add_leaf(
        id="trail_location_identification",
        desc="The facility is correctly identified as a specific accessible trail within a California National Park",
        parent=node,
        critical=True
    )
    claim_loc = (
        f"The trail '{t_name}' is located within '{t_park}', which is a National Park in California."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=n_loc,
        sources=t_sources,
        additional_instruction="Pass if the provided source(s) clearly indicate the trail is within the named National Park and that park is in California."
    )

    # 2) Accessibility designation (critical)
    n_acc = evaluator.add_leaf(
        id="trail_accessibility_designation",
        desc="The trail is officially documented and designated as wheelchair-accessible or ADA-accessible",
        parent=node,
        critical=True
    )
    claim_acc = (
        f"The provided source(s) document that the trail '{t_name}' is wheelchair-accessible or ADA-accessible."
    )
    await evaluator.verify(
        claim=claim_acc,
        node=n_acc,
        sources=t_sources,
        additional_instruction="Look for explicit phrases such as 'wheelchair-accessible', 'ADA accessible', 'accessible trail', or official accessibility designation."
    )

    # 3) Surface specifications (parent non-critical due to mixed criticalities)
    surf_parent = evaluator.add_parallel(
        id="trail_surface_specifications",
        desc="Verify trail surface and width specifications meet accessibility standards",
        parent=node,
        critical=False
    )
    # 3.1) Surface type (critical)
    n_surface_type = evaluator.add_leaf(
        id="surface_type",
        desc="Trail surface is documented as paved or firm and stable material suitable for wheelchairs",
        parent=surf_parent,
        critical=True
    )
    claim_surface_type = (
        f"The trail '{t_name}' has a surface described as '{t_surface}', which is paved or otherwise firm and stable for wheelchair use."
    )
    await evaluator.verify(
        claim=claim_surface_type,
        node=n_surface_type,
        sources=t_sources,
        additional_instruction="Consider 'asphalt', 'concrete', 'boardwalk', and 'paved' as firm/stable surfaces. Compacted, firm, and stable surfaces can also qualify if explicitly stated."
    )

    # 3.2) Surface width (non-critical)
    n_surface_width = evaluator.add_leaf(
        id="surface_width",
        desc="Trail width meets the minimum requirement of 36 inches clear tread width (or documented accessible width standard)",
        parent=surf_parent,
        critical=False
    )
    claim_surface_width = (
        f"The trail '{t_name}' provides at least 36 inches of clear tread width OR the page explicitly states the trail meets ADA accessibility width standards. "
        f"Documented width: '{t_width}'."
    )
    await evaluator.verify(
        claim=claim_surface_width,
        node=n_surface_width,
        sources=t_sources,
        additional_instruction="Pass if either a numeric width >= 36 inches is shown or the source explicitly states the trail meets ADA width standards even if an exact number isn't provided."
    )

    # 4) Length specification (critical)
    n_length = evaluator.add_leaf(
        id="trail_length_specification",
        desc="Trail length is provided with specific measurement (in miles or kilometers)",
        parent=node,
        critical=True
    )
    claim_length = f"The length of the trail '{t_name}' is documented as '{t_length}'."
    await evaluator.verify(
        claim=claim_length,
        node=n_length,
        sources=t_sources,
        additional_instruction="Verify that the source explicitly states a distance (e.g., 0.8 miles, 1.3 km). Reasonable rounding or abbreviated formats are acceptable."
    )

    # 5) Reference documentation (critical) - existence of at least one source URL
    evaluator.add_custom_node(
        result=len(t_sources) > 0,
        id="trail_reference_documentation",
        desc="At least one official/authoritative reference URL is provided for the trail",
        parent=node,
        critical=True
    )


async def build_camping_verification(evaluator: Evaluator, parent_node, camping: Optional[CampingFacility]) -> None:
    node = evaluator.add_parallel(
        id="accessible_camping_facility",
        desc="Verify the accessible camping facility in California meets all requirements",
        parent=parent_node,
        critical=False
    )

    c_name = camping.name if camping else ""
    c_area = camping.park_or_area if camping else ""
    c_loc = camping.location if camping else ""
    c_sites = camping.accessible_campsites_info if camping else ""
    c_rest = camping.accessible_restrooms_info if camping else ""
    c_tables = camping.accessible_picnic_tables_info if camping else ""
    c_routes = camping.accessible_routes_info if camping else ""
    c_showers = camping.accessible_showers_info if camping else ""
    c_sources = _sources_list(camping.sources if camping else [])

    # 1) Location identification (critical)
    n_loc = evaluator.add_leaf(
        id="camping_location_identification",
        desc="The facility is correctly identified as a specific camping location in California",
        parent=node,
        critical=True
    )
    claim_loc = f"'{c_name}' is a specific public campground or camping area located in California (park/system: '{c_area}', location: '{c_loc}')."
    await evaluator.verify(
        claim=claim_loc,
        node=n_loc,
        sources=c_sources,
        additional_instruction="Pass if the source indicates this is a campground/camping facility in California (can be within a National/State Park or other public campground)."
    )

    # 2) Accessible campsites (critical)
    n_sites = evaluator.add_leaf(
        id="accessible_campsites_availability",
        desc="The facility provides designated accessible campsites (quantity or availability documented)",
        parent=node,
        critical=True
    )
    claim_sites = f"The campground '{c_name}' provides designated accessible campsites. Details: '{c_sites}'."
    await evaluator.verify(
        claim=claim_sites,
        node=n_sites,
        sources=c_sources,
        additional_instruction="Look for mentions like 'ADA site(s)', 'wheelchair-accessible campsite(s)', or equivalent."
    )

    # 3) Accessible amenities (parent non-critical due to mixed child criticalities in rubric)
    amenities_parent = evaluator.add_parallel(
        id="camping_accessible_amenities",
        desc="Verify required accessible amenities are available at the camping facility",
        parent=node,
        critical=False
    )

    # 3.1) Accessible restrooms (critical)
    n_rest = evaluator.add_leaf(
        id="accessible_restrooms",
        desc="Accessible restroom facilities are documented as available at or near the campsites",
        parent=amenities_parent,
        critical=True
    )
    claim_rest = f"Accessible restroom facilities are available at or near the campsites at '{c_name}'. Details: '{c_rest}'."
    await evaluator.verify(
        claim=claim_rest,
        node=n_rest,
        sources=c_sources,
        additional_instruction="Look for 'accessible restroom(s)', 'ADA restroom(s)', or family/unisex accessible restrooms near campsites."
    )

    # 3.2) Accessible picnic tables (non-critical)
    n_tables = evaluator.add_leaf(
        id="accessible_picnic_tables",
        desc="Accessible picnic tables (at appropriate height for wheelchair users) are documented as available",
        parent=amenities_parent,
        critical=False
    )
    claim_tables = f"Accessible picnic tables are documented as available at '{c_name}'. Details: '{c_tables}'."
    await evaluator.verify(
        claim=claim_tables,
        node=n_tables,
        sources=c_sources,
        additional_instruction="Pass if the source indicates accessible picnic tables or ADA-height tables are provided."
    )

    # 3.3) Level surfaces or paved paths (non-critical)
    n_routes = evaluator.add_leaf(
        id="level_surfaces_or_paved_paths",
        desc="Level surfaces, paved paths, or accessible routes to facilities are documented",
        parent=amenities_parent,
        critical=False
    )
    claim_routes = f"Level surfaces, paved paths, or accessible routes are documented for '{c_name}'. Details: '{c_routes}'."
    await evaluator.verify(
        claim=claim_routes,
        node=n_routes,
        sources=c_sources,
        additional_instruction="Pass if the page indicates accessible routes/paths (e.g., paved/level/ADA routes) connecting campsites to facilities."
    )

    # 4) Accessible showers (non-critical; only required if showers exist)
    # We implement as a custom conditional check: pass if not applicable or verified by sources.
    if _nonempty(c_showers):
        n_showers = evaluator.add_leaf(
            id="camping_shower_facilities",
            desc="Accessible shower facilities are documented (if showers are provided at the facility)",
            parent=node,
            critical=False
        )
        claim_showers = f"Accessible shower facilities are documented at '{c_name}'. Details: '{c_showers}'."
        await evaluator.verify(
            claim=claim_showers,
            node=n_showers,
            sources=c_sources,
            additional_instruction="Pass if showers exist at this campground and at least one is accessible; if no showers at the facility, this check is not required."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="camping_shower_facilities",
            desc="Accessible shower facilities check not applicable or not required (no showers documented at facility)",
            parent=node,
            critical=False
        )

    # 5) Reference documentation (critical)
    evaluator.add_custom_node(
        result=len(c_sources) > 0,
        id="camping_reference_documentation",
        desc="At least one reference URL is provided from an official/authoritative source for the camping facility",
        parent=node,
        critical=True
    )


async def build_skiing_verification(evaluator: Evaluator, parent_node, ski: Optional[SkiProgramFacility]) -> None:
    node = evaluator.add_parallel(
        id="adaptive_skiing_program",
        desc="Verify the adaptive skiing program location in California meets all requirements",
        parent=parent_node,
        critical=False
    )

    s_loc = ski.resort_or_location if ski else ""
    s_city = ski.region_or_city if ski else ""
    s_prog = ski.program_name if ski else ""
    s_equips = ski.equipment_types if ski else []
    s_equips_text = _fmt_list(s_equips)
    s_equips_provided = ski.equipment_provided if ski else ""
    s_lessons = ski.lessons_info if ski else ""
    s_access = ski.access_info if ski else ""
    s_sources = _sources_list(ski.sources if ski else [])

    # 1) Skiing location identification (critical)
    n_loc = evaluator.add_leaf(
        id="skiing_location_identification",
        desc="The facility is correctly identified as a specific ski resort or location in California that offers adaptive skiing programs",
        parent=node,
        critical=True
    )
    claim_loc = f"'{s_loc}' is a ski resort/location in California (region/city: '{s_city}')."
    await evaluator.verify(
        claim=claim_loc,
        node=n_loc,
        sources=s_sources,
        additional_instruction="Pass if the page clearly indicates the resort/location is in California."
    )

    # 2) Adaptive program existence (critical)
    n_prog = evaluator.add_leaf(
        id="adaptive_program_existence",
        desc="An adaptive skiing program or adaptive sports organization is documented as operating at this location",
        parent=node,
        critical=True
    )
    claim_prog = f"An adaptive skiing program (e.g., '{s_prog}' or equivalent) operates at '{s_loc}'."
    await evaluator.verify(
        claim=claim_prog,
        node=n_prog,
        sources=s_sources,
        additional_instruction="Look for a dedicated adaptive program page or clear mention of adaptive skiing/snow sports program at this resort/location."
    )

    # 3) Adaptive equipment availability (parent non-critical due to mixed child criticalities)
    equip_parent = evaluator.add_parallel(
        id="adaptive_equipment_availability",
        desc="Verify that specialized adaptive skiing equipment is available",
        parent=node,
        critical=False
    )

    # 3.1) Equipment types documented (critical)
    n_equip_types = evaluator.add_leaf(
        id="equipment_types_documented",
        desc="Specific types of adaptive equipment are documented (e.g., mono-skis, bi-skis, outriggers, sit-skis, or other adaptive devices)",
        parent=equip_parent,
        critical=True
    )
    claim_equip_types = (
        f"The adaptive program at '{s_loc}' provides specialized equipment such as: {s_equips_text}."
    )
    await evaluator.verify(
        claim=claim_equip_types,
        node=n_equip_types,
        sources=s_sources,
        additional_instruction="Pass if the page lists at least one specific adaptive ski device (e.g., mono-ski, bi-ski, sit-ski) or outriggers."
    )

    # 3.2) Equipment provided by program (non-critical)
    n_equip_prov = evaluator.add_leaf(
        id="equipment_provided_by_program",
        desc="Documentation indicates that equipment is provided as part of the program (not requiring participants to bring their own adaptive equipment)",
        parent=equip_parent,
        critical=False
    )
    claim_equip_prov = (
        f"The adaptive skiing program at '{s_loc}' provides adaptive equipment for participants as part of the program. Details: '{s_equips_provided}'."
    )
    await evaluator.verify(
        claim=claim_equip_prov,
        node=n_equip_prov,
        sources=s_sources,
        additional_instruction="Pass if the page indicates equipment is provided/rented by the program; fail if it clearly requires participants to bring their own without program provision."
    )

    # 4) Adaptive lessons or instruction (critical)
    n_lessons = evaluator.add_leaf(
        id="adaptive_lessons_or_instruction",
        desc="Adaptive ski lessons or instruction services are documented as available",
        parent=node,
        critical=True
    )
    claim_lessons = f"Adaptive ski lessons/instruction are offered at '{s_loc}'. Details: '{s_lessons}'."
    await evaluator.verify(
        claim=claim_lessons,
        node=n_lessons,
        sources=s_sources,
        additional_instruction="Pass if the page mentions adaptive lessons, instruction, coaching, or trained instructors for adaptive skiing."
    )

    # 5) Program access information (non-critical)
    n_access = evaluator.add_leaf(
        id="program_access_information",
        desc="Information about how to access the program is provided (e.g., reservation requirements, contact information, or program details)",
        parent=node,
        critical=False
    )
    claim_access = f"The page for '{s_loc}' provides information on how to access the program (reservations, booking, or contact). Details: '{s_access}'."
    await evaluator.verify(
        claim=claim_access,
        node=n_access,
        sources=s_sources,
        additional_instruction="Pass if the page includes reservation/booking flow, phone/email contact, or scheduling details for the adaptive program."
    )

    # 6) Reference documentation (critical)
    evaluator.add_custom_node(
        result=len(s_sources) > 0,
        id="skiing_reference_documentation",
        desc="At least one reference URL is provided from an official/authoritative source verifying the adaptive skiing program",
        parent=node,
        critical=True
    )


async def build_picnic_verification(evaluator: Evaluator, parent_node, picnic: Optional[PicnicFacility]) -> None:
    node = evaluator.add_parallel(
        id="accessible_picnic_area",
        desc="Verify the accessible picnic area in a California State Park meets all requirements",
        parent=parent_node,
        critical=False
    )

    p_park = picnic.park_name if picnic else ""
    p_area = picnic.picnic_area_name if picnic else ""
    p_loc = picnic.location if picnic else ""
    p_tables = picnic.accessible_tables_info if picnic else ""
    p_paths = picnic.accessible_paths_info if picnic else ""
    p_parking = picnic.accessible_parking_info if picnic else ""
    p_add = picnic.additional_amenities_info if picnic else ""
    p_sources = _sources_list(picnic.sources if picnic else [])

    # 1) Location identification (critical)
    n_loc = evaluator.add_leaf(
        id="picnic_location_identification",
        desc="The facility is correctly identified as a specific accessible picnic area within a California State Park",
        parent=node,
        critical=True
    )
    claim_loc = f"The picnic/day-use area '{p_area}' is within '{p_park}', which is a California State Park (location: '{p_loc}')."
    await evaluator.verify(
        claim=claim_loc,
        node=n_loc,
        sources=p_sources,
        additional_instruction="Pass if the page indicates the site is a picnic/day-use area in a California State Park."
    )

    # 2) Accessibility designation (critical)
    n_des = evaluator.add_leaf(
        id="picnic_accessibility_designation",
        desc="The picnic area is documented as having accessible features or meeting accessibility standards",
        parent=node,
        critical=True
    )
    claim_des = f"The picnic/day-use area in '{p_park}' is documented as accessible or includes accessible features."
    await evaluator.verify(
        claim=claim_des,
        node=n_des,
        sources=p_sources,
        additional_instruction="Look for explicit mentions of accessibility or accessible features in the picnic/day-use area."
    )

    # 3) Accessible features (parent non-critical due to mixed child criticalities)
    feat_parent = evaluator.add_parallel(
        id="picnic_accessible_features",
        desc="Verify required accessible features are present at the picnic area",
        parent=node,
        critical=False
    )

    # 3.1) Accessible picnic tables (critical)
    n_tables = evaluator.add_leaf(
        id="accessible_picnic_tables",
        desc="Accessible picnic tables (at appropriate height for wheelchair users) are documented",
        parent=feat_parent,
        critical=True
    )
    claim_tables = f"Accessible picnic tables are documented at the picnic/day-use area in '{p_park}'. Details: '{p_tables}'."
    await evaluator.verify(
        claim=claim_tables,
        node=n_tables,
        sources=p_sources,
        additional_instruction="Pass if the page mentions accessible picnic tables or ADA tables."
    )

    # 3.2) Accessible paths or routes (critical)
    n_paths = evaluator.add_leaf(
        id="accessible_paths_or_routes",
        desc="Accessible paths, routes, or level surfaces to picnic sites are documented",
        parent=feat_parent,
        critical=True
    )
    claim_paths = f"Accessible paths/routes or level surfaces to the picnic/day-use area are documented at '{p_park}'. Details: '{p_paths}'."
    await evaluator.verify(
        claim=claim_paths,
        node=n_paths,
        sources=p_sources,
        additional_instruction="Pass if the page describes accessible routes/paths/sidewalks or level surfaces connecting parking to picnic sites."
    )

    # 3.3) Accessible parking (non-critical)
    n_parking = evaluator.add_leaf(
        id="accessible_parking",
        desc="Accessible parking is documented as available at or near the picnic area",
        parent=feat_parent,
        critical=False
    )
    claim_parking = f"Accessible parking is documented at or near the picnic/day-use area in '{p_park}'. Details: '{p_parking}'."
    await evaluator.verify(
        claim=claim_parking,
        node=n_parking,
        sources=p_sources,
        additional_instruction="Pass if the page mentions ADA/accessible parking spaces near the picnic/day-use area."
    )

    # 4) Additional accessible amenities (non-critical)
    if _nonempty(p_add):
        n_add = evaluator.add_leaf(
            id="picnic_additional_amenities",
            desc="Additional accessible amenities are documented (e.g., accessible restrooms, accessible fire rings, or other features)",
            parent=node,
            critical=False
        )
        claim_add = f"Additional accessible amenities are documented for the picnic/day-use area in '{p_park}'. Details: '{p_add}'."
        await evaluator.verify(
            claim=claim_add,
            node=n_add,
            sources=p_sources,
            additional_instruction="Pass if the page documents other accessible amenities (e.g., accessible restrooms at the area, accessible fire rings, etc.)."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="picnic_additional_amenities",
            desc="Additional accessible amenities check not applicable or not provided in answer",
            parent=node,
            critical=False
        )

    # 5) Reference documentation (critical)
    evaluator.add_custom_node(
        result=len(p_sources) > 0,
        id="picnic_reference_documentation",
        desc="At least one reference URL is provided from an official/authoritative source verifying the picnic area features",
        parent=node,
        critical=True
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
    # Initialize evaluator (root is always non-critical by framework design)
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=AccessibilityFacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Build verifications for each of the four facility categories
    await build_trail_verification(evaluator, root, extraction.trail)
    await build_camping_verification(evaluator, root, extraction.camping)
    await build_skiing_verification(evaluator, root, extraction.skiing)
    await build_picnic_verification(evaluator, root, extraction.picnic)

    # Return standardized summary
    return evaluator.get_summary()