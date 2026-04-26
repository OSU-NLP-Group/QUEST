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
TASK_ID = "nc_accessible_reunion_2026"
TASK_DESCRIPTION = (
    "Identify a North Carolina state park recreation area that offers comprehensive accessible facilities "
    "suitable for hosting a large family reunion in summer 2026, where several family members use wheelchairs. "
    "The facility must provide all of the following accessible features: (1) Accessible camping sites with multiple "
    "sites available, level surfaces, and proper accessibility features; (2) Accessible picnic facilities with proper "
    "clear ground space for wheelchairs; (3) At least one accessible trail that family members can walk together; "
    "(4) Designated accessible parking spaces; (5) Accessible routes to the lake or water features; "
    "(6) Accessible restrooms throughout the facility; (7) An accessible swimming area or beach where all family "
    "members can enjoy the water; (8) Group facilities such as group camping or large picnic shelters to accommodate "
    "the extended family; (9) Convenient location of accessible parking relative to main facilities. Provide the name "
    "of the state park recreation area and its location (county)."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class RecreationFacilityExtraction(BaseModel):
    facility_name: Optional[str] = None
    county: Optional[str] = None

    # General/official facility sources (homepage, accessibility page, brochure, map)
    facility_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)

    # Seasonality or operations info (e.g., summer availability of group-use facilities)
    seasonality_urls: List[str] = Field(default_factory=list)

    # Feature-specific sources
    camping_urls: List[str] = Field(default_factory=list)
    picnic_urls: List[str] = Field(default_factory=list)
    trail_urls: List[str] = Field(default_factory=list)
    parking_urls: List[str] = Field(default_factory=list)
    water_access_urls: List[str] = Field(default_factory=list)
    restrooms_urls: List[str] = Field(default_factory=list)
    swimming_urls: List[str] = Field(default_factory=list)
    group_facilities_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility() -> str:
    return """
    From the provided answer, extract the single North Carolina state park recreation area being recommended for a large, wheelchair-inclusive family reunion in summer 2026.

    You must extract the following fields strictly from the answer (do NOT invent anything):
    - facility_name: The name of the North Carolina state park recreation area.
    - county: The county location provided in the answer (e.g., "Wake County"). If multiple are mentioned, pick the one explicitly tied to the recommended facility.

    Also extract all URLs explicitly cited in the answer for each category below. Only include URLs actually present in the answer text. Do not infer or fabricate URLs.

    - facility_urls: Official/general facility pages, accessibility pages, park brochure or official map pages for the facility.
    - location_urls: Pages that directly support the county where the facility is located (can reuse facility URLs if county is stated there).
    - seasonality_urls: Pages that indicate seasonal/operational status relevant to summer availability of facilities (e.g., swim beach season, group-use operating seasons).
    - camping_urls: Pages that describe accessible camping (e.g., campsite list, accessibility notes).
    - picnic_urls: Pages that describe accessible picnic facilities/tables.
    - trail_urls: Pages that describe at least one accessible/universal access trail (width, surface, grade, etc.).
    - parking_urls: Pages that describe designated accessible parking (and ideally location/route info).
    - water_access_urls: Pages that describe accessible routes to water features (lake, pier, boardwalk, etc.).
    - restrooms_urls: Pages that state accessible restrooms (ideally throughout main areas).
    - swimming_urls: Pages that describe accessible swimming areas or beaches (e.g., beach wheelchairs, accessible paths to swim areas).
    - group_facilities_urls: Pages that describe group-use facilities (group camping, large picnic shelters) suitable for extended family gatherings.

    IMPORTANT:
    - Return null for any missing text fields.
    - For every URL list field, return an empty array if the answer provides none.
    - Do not modify or invent URLs; extract exactly as written (convert markdown links to their actual URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def nonempty(urls: Optional[List[str]]) -> bool:
    return bool(urls and any(isinstance(u, str) and u.strip() for u in urls))


def merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            v = (u or "").strip()
            if v and v not in seen:
                seen.add(v)
                merged.append(v)
    return merged


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_facility_identification(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    node = evaluator.add_parallel(
        id="facility_identification",
        desc="Answer identifies a North Carolina state park recreation area and provides its county location.",
        parent=parent,
        critical=True
    )

    # Existence checks (critical)
    name_exists = evaluator.add_custom_node(
        result=bool(ext.facility_name and ext.facility_name.strip()),
        id="facility_name_provided",
        desc="Facility name is provided",
        parent=node,
        critical=True
    )

    county_exists = evaluator.add_custom_node(
        result=bool(ext.county and ext.county.strip()),
        id="county_provided",
        desc="County location is provided",
        parent=node,
        critical=True
    )

    srcs_exist = evaluator.add_custom_node(
        result=nonempty(ext.facility_urls) or nonempty(ext.location_urls),
        id="facility_id_sources_provided",
        desc="At least one facility/location source URL is provided",
        parent=node,
        critical=True
    )

    # Verifications
    urls = merge_urls(ext.facility_urls, ext.location_urls)

    leaf_nc_system = evaluator.add_leaf(
        id="facility_is_nc_state_park",
        desc="Facility is a recreation area within North Carolina's state park system",
        parent=node,
        critical=True
    )
    claim_nc = (
        f"The facility named '{ext.facility_name}' is a recreation area within the North Carolina state park system "
        f"(managed by the North Carolina Division of Parks and Recreation) and is located in North Carolina."
    )
    await evaluator.verify(
        claim=claim_nc,
        node=leaf_nc_system,
        sources=urls,
        additional_instruction="Accept pages on ncparks.gov or other official sources that clearly indicate the facility belongs to North Carolina State Parks."
    )

    leaf_county = evaluator.add_leaf(
        id="facility_county_supported",
        desc="Facility's county location is supported by sources",
        parent=node,
        critical=True
    )
    claim_county = f"The facility '{ext.facility_name}' is located in {ext.county} County, North Carolina."
    await evaluator.verify(
        claim=claim_county,
        node=leaf_county,
        sources=urls,
        additional_instruction="Verify that the cited page(s) explicitly state or clearly show the facility's county."
    )


async def verify_summer_suitability(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    node = evaluator.add_parallel(
        id="summer_2026_suitability",
        desc="Answer addresses suitability for hosting a large family reunion in summer 2026 (e.g., indicates the relevant group-use facilities are available/open/operational during summer).",
        parent=parent,
        critical=True
    )

    # Source gating
    srcs_exist = evaluator.add_custom_node(
        result=nonempty(ext.group_facilities_urls) or nonempty(ext.seasonality_urls) or nonempty(ext.facility_urls),
        id="summer_sources_provided",
        desc="Seasonality/operations or group-facility source URL(s) are provided",
        parent=node,
        critical=True
    )

    urls = merge_urls(ext.group_facilities_urls, ext.seasonality_urls, ext.facility_urls)

    leaf_open = evaluator.add_leaf(
        id="summer_group_facilities_open",
        desc="Group-use facilities are available/open during summer months",
        parent=node,
        critical=True
    )
    claim_open = (
        f"At '{ext.facility_name}', the group-use facilities (group camping and/or large picnic shelters) are "
        f"available/open during the summer months (June–August). It is reasonable to expect the same availability in summer 2026."
    )
    await evaluator.verify(
        claim=claim_open,
        node=leaf_open,
        sources=urls,
        additional_instruction="Confirm summer availability or operations for group camping or large group picnic shelters. An explicit season window including the summer months or a statement of year-round availability is sufficient."
    )


async def verify_accessible_camping(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    node = evaluator.add_parallel(
        id="accessible_camping_sites",
        desc="Facility provides accessible camping sites with multiple sites available (at least two accessible sites), including level/accessible surfaces and required accessibility features.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(ext.camping_urls),
        id="camping_sources_provided",
        desc="Accessible camping source URL(s) are provided",
        parent=node,
        critical=True
    )

    urls = merge_urls(ext.camping_urls)

    # Multiple accessible sites
    leaf_multi = evaluator.add_leaf(
        id="camping_multiple_accessible_sites",
        desc="At least two accessible campsites are available",
        parent=node,
        critical=True
    )
    claim_multi = (
        f"'{ext.facility_name}' provides multiple ADA-accessible campsites (at least two accessible sites)."
    )
    await evaluator.verify(
        claim=claim_multi,
        node=leaf_multi,
        sources=urls,
        additional_instruction="Look for campsite inventories, accessibility notes, or symbols that indicate two or more designated accessible campsites."
    )

    # Features and surfaces
    leaf_features = evaluator.add_leaf(
        id="camping_accessible_features_and_surface",
        desc="Accessible campsites have level, firm/stable surfaces and required features",
        parent=node,
        critical=True
    )
    claim_features = (
        f"The designated accessible campsites at '{ext.facility_name}' have level, firm/stable surfaces and include required accessibility features "
        f"(e.g., accessible route to facilities, accessible picnic table or fire ring, and other ADA features)."
    )
    await evaluator.verify(
        claim=claim_features,
        node=leaf_features,
        sources=urls,
        additional_instruction="Accept explicit mentions of 'accessible' campsite features and ADA-compliant surfaces; numeric slope/grade is not required if ADA compliance is stated."
    )


async def verify_accessible_picnic(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    node = evaluator.add_parallel(
        id="accessible_picnic_facilities",
        desc="Facility provides accessible picnic facilities with clear ground space for wheelchair users.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(ext.picnic_urls) or nonempty(ext.facility_urls),
        id="picnic_sources_provided",
        desc="Accessible picnic source URL(s) are provided",
        parent=node,
        critical=True
    )

    urls = merge_urls(ext.picnic_urls, ext.facility_urls)

    leaf_picnic = evaluator.add_leaf(
        id="picnic_clear_space",
        desc="Accessible picnic facilities provide clear space for wheelchairs",
        parent=node,
        critical=True
    )
    claim_picnic = (
        f"'{ext.facility_name}' provides accessible picnic facilities with appropriate wheelchair clear ground space "
        f"(around 36×48 inches or ADA-standard 30×48 inches), for example via accessible picnic tables with wheelchair clearance."
    )
    await evaluator.verify(
        claim=claim_picnic,
        node=leaf_picnic,
        sources=urls,
        additional_instruction="Accept explicit mentions of accessible picnic tables or wheelchair spaces as sufficient evidence, even if exact inches are not specified, provided ADA compliance is stated or clearly implied."
    )


async def verify_accessible_trail(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    node = evaluator.add_parallel(
        id="accessible_trail",
        desc="Facility has at least one accessible trail with a minimum clear tread width of about 36 inches.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(ext.trail_urls),
        id="trail_sources_provided",
        desc="Accessible trail source URL(s) are provided",
        parent=node,
        critical=True
    )

    urls = merge_urls(ext.trail_urls)

    leaf_trail = evaluator.add_leaf(
        id="trail_min_width",
        desc="At least one accessible/universal access trail is available (≈36 in minimum width)",
        parent=node,
        critical=True
    )
    claim_trail = (
        f"'{ext.facility_name}' has at least one accessible/universal access trail suitable for family members walking together, "
        f"with an approximate minimum clear tread width of 36 inches (or otherwise ADA-accessible with firm/stable surface and gentle grades)."
    )
    await evaluator.verify(
        claim=claim_trail,
        node=leaf_trail,
        sources=urls,
        additional_instruction="Accept explicit 'accessible' or 'universal access' trail designations as sufficient evidence even if exact width is not stated, provided ADA accessibility is indicated."
    )


async def verify_accessible_parking(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    # Dimensions
    node_dim = evaluator.add_parallel(
        id="accessible_parking_dimensions",
        desc="Facility provides designated accessible parking spaces that meet ADA width requirements.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(ext.parking_urls),
        id="parking_sources_provided",
        desc="Accessible parking source URL(s) are provided",
        parent=node_dim,
        critical=True
    )

    urls = merge_urls(ext.parking_urls)

    leaf_width = evaluator.add_leaf(
        id="parking_width_requirement",
        desc="Accessible parking meets ADA width requirements (≥ 96 in for car space or provides van-accessible width/aisle as applicable)",
        parent=node_dim,
        critical=True
    )
    claim_width = (
        f"The designated accessible parking at '{ext.facility_name}' meets ADA width requirements "
        f"(for example, a minimum 96-inch parking space width and appropriate access aisle, or 'van accessible' spaces meeting ADA width standards)."
    )
    await evaluator.verify(
        claim=claim_width,
        node=leaf_width,
        sources=urls,
        additional_instruction="Accept explicit ADA-compliant or 'van accessible' parking statements as sufficient evidence even if exact inch values are not printed."
    )

    # Location and Route
    node_loc = evaluator.add_parallel(
        id="accessible_parking_location_and_route",
        desc="Accessible parking is conveniently located relative to main facilities and connected by an accessible route.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(ext.parking_urls) or nonempty(ext.facility_urls),
        id="parking_loc_route_sources_provided",
        desc="Accessible parking location/route source URL(s) are provided",
        parent=node_loc,
        critical=True
    )

    urls_loc = merge_urls(ext.parking_urls, ext.facility_urls)

    leaf_convenient = evaluator.add_leaf(
        id="parking_convenient_location",
        desc="Accessible parking located conveniently near main facilities",
        parent=node_loc,
        critical=True
    )
    claim_convenient = (
        f"Accessible parking at '{ext.facility_name}' is located conveniently/close to main facilities "
        f"(e.g., beach, picnic areas, restrooms, shelters)."
    )
    await evaluator.verify(
        claim=claim_convenient,
        node=leaf_convenient,
        sources=urls_loc,
        additional_instruction="Look for map legends, amenity descriptions, or accessibility notes that indicate accessible parking near the primary activity areas."
    )

    leaf_route = evaluator.add_leaf(
        id="parking_accessible_route",
        desc="Accessible parking lies on an accessible route to the main facilities",
        parent=node_loc,
        critical=True
    )
    claim_route = (
        f"From accessible parking at '{ext.facility_name}', there is an accessible (step-free, ADA-compliant) route to the main facilities."
    )
    await evaluator.verify(
        claim=claim_route,
        node=leaf_route,
        sources=urls_loc,
        additional_instruction="Accept explicit mentions of 'accessible route', paved/boardwalk paths, or step-free connections to the main facilities."
    )


async def verify_accessible_routes_to_water(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    node = evaluator.add_parallel(
        id="accessible_routes_to_water",
        desc="Facility provides accessible routes to the lake or other water features.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(ext.water_access_urls) or nonempty(ext.swimming_urls) or nonempty(ext.facility_urls),
        id="water_route_sources_provided",
        desc="Water access route source URL(s) are provided",
        parent=node,
        critical=True
    )

    urls = merge_urls(ext.water_access_urls, ext.swimming_urls, ext.facility_urls)

    leaf_water = evaluator.add_leaf(
        id="accessible_route_to_water_leaf",
        desc="Accessible route to lake/water feature exists",
        parent=node,
        critical=True
    )
    claim_water = (
        f"'{ext.facility_name}' provides an accessible route (e.g., paved path, boardwalk) to the lake or water features."
    )
    await evaluator.verify(
        claim=claim_water,
        node=leaf_water,
        sources=urls,
        additional_instruction="Look for accessible paths to the beach, pier, or waterfront areas; explicit 'accessible route' phrasing or equivalent is sufficient."
    )


async def verify_accessible_restrooms(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    node = evaluator.add_parallel(
        id="accessible_restrooms_throughout",
        desc="Facility has accessible restrooms throughout the facility that comply with ADA/ABA standards.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(ext.restrooms_urls) or nonempty(ext.facility_urls),
        id="restrooms_sources_provided",
        desc="Accessible restroom source URL(s) are provided",
        parent=node,
        critical=True
    )

    urls = merge_urls(ext.restrooms_urls, ext.facility_urls)

    leaf_rr = evaluator.add_leaf(
        id="restrooms_accessible_throughout_leaf",
        desc="Accessible restrooms available throughout main facilities",
        parent=node,
        critical=True
    )
    claim_rr = (
        f"Accessible restrooms are available throughout the main areas of '{ext.facility_name}', compliant with ADA/ABA standards."
    )
    await evaluator.verify(
        claim=claim_rr,
        node=leaf_rr,
        sources=urls,
        additional_instruction="Accept explicit mentions like 'accessible restrooms available' in multiple key areas (e.g., beach/picnic/trailheads). Exact counts not required."
    )


async def verify_accessible_swimming_area(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    node = evaluator.add_parallel(
        id="accessible_swimming_area",
        desc="Facility provides an accessible swimming area or beach.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(ext.swimming_urls),
        id="swimming_sources_provided",
        desc="Accessible swimming/beach source URL(s) are provided",
        parent=node,
        critical=True
    )

    urls = merge_urls(ext.swimming_urls)

    leaf_swim = evaluator.add_leaf(
        id="swimming_accessible_leaf",
        desc="Accessible swimming area/beach exists",
        parent=node,
        critical=True
    )
    claim_swim = (
        f"'{ext.facility_name}' provides an accessible swimming area or accessible beach that wheelchair users and family members can enjoy together."
    )
    await evaluator.verify(
        claim=claim_swim,
        node=leaf_swim,
        sources=urls,
        additional_instruction="Accept explicit mentions of accessible beach access, beach wheelchairs, or accessible routes to the designated swimming area."
    )


async def verify_group_facilities(evaluator: Evaluator, parent, ext: RecreationFacilityExtraction):
    node = evaluator.add_parallel(
        id="group_facilities",
        desc="Facility offers group facilities to accommodate extended family gatherings (group camping and/or large picnic shelters).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(ext.group_facilities_urls),
        id="group_sources_provided",
        desc="Group-facility source URL(s) are provided",
        parent=node,
        critical=True
    )

    urls = merge_urls(ext.group_facilities_urls)

    leaf_group = evaluator.add_leaf(
        id="group_facilities_exist_leaf",
        desc="Group camping and/or large picnic shelters available",
        parent=node,
        critical=True
    )
    claim_group = (
        f"'{ext.facility_name}' offers group facilities suitable for extended family gatherings, such as group camping and/or large picnic shelters."
    )
    await evaluator.verify(
        claim=claim_group,
        node=leaf_group,
        sources=urls,
        additional_instruction="Look for 'group camp', 'group campsite', 'group shelter', 'large picnic shelter' or similar terms indicating group-use capacity."
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
    Evaluate an answer for the NC accessible family-reunion facility task using the Mind2Web2 evaluation framework.
    """
    # Initialize evaluator (root is non-critical container)
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

    # Extract structured info from the answer
    extracted: RecreationFacilityExtraction = await evaluator.extract(
        prompt=prompt_extract_facility(),
        template_class=RecreationFacilityExtraction,
        extraction_name="facility_extraction"
    )

    # Build the critical evaluation subtree (as per rubric root)
    main = evaluator.add_parallel(
        id="comprehensive_accessible_recreation_facility",
        desc="Evaluate whether the identified North Carolina state park recreation area meets all stated accessibility and group-use constraints for a large family reunion in summer 2026.",
        parent=root,
        critical=True
    )

    # Construct and verify each rubric component
    await verify_facility_identification(evaluator, main, extracted)
    await verify_summer_suitability(evaluator, main, extracted)
    await verify_accessible_camping(evaluator, main, extracted)
    await verify_accessible_picnic(evaluator, main, extracted)
    await verify_accessible_trail(evaluator, main, extracted)
    await verify_accessible_parking(evaluator, main, extracted)
    await verify_accessible_routes_to_water(evaluator, main, extracted)
    await verify_accessible_restrooms(evaluator, main, extracted)
    await verify_accessible_swimming_area(evaluator, main, extracted)
    await verify_group_facilities(evaluator, main, extracted)

    # Return evaluation summary
    return evaluator.get_summary()