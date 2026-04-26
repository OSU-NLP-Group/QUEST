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
TASK_ID = "broadway_tech_ops_requirements"
TASK_DESCRIPTION = (
    "Identify a performing arts theater located in Manhattan's Theater District, New York City, that qualifies as a Broadway theater and meets the following comprehensive technical and operational requirements:\n"
    "1. Has a seating capacity of 500 or more seats (Broadway classification requirement)\n"
    "2. Provides ADA-compliant wheelchair-accessible seating spaces in the required ratio for its capacity\n"
    "3. Has a minimum of 3 emergency exits to meet fire safety code requirements for its capacity\n"
    "4. Provides adequate restroom facilities meeting assembly venue code requirements\n"
    "5. Has an HVAC system meeting ASHRAE Standard 62.1-2019 (minimum 5 CFM per person plus 0.06 CFM per square foot)\n"
    "6. Has a loading dock at the standard 48-inch (4-foot) height\n"
    "7. Has dressing rooms equipped with washstands providing hot and cold running water\n"
    "8. Has a stage apron (extension in front of the proscenium)\n"
    "9. Has an orchestra pit with adjustable height capability\n"
    "10. Has an adjustable proscenium opening\n"
    "11. Requires proof of general liability insurance (minimum $1,000,000) from renters/performers\n"
    "12. Provides direct backstage access from the loading dock to the stage area\n"
    "13. Provides internet connectivity (minimum 10 Mbps per 100 people recommended)\n"
    "14. Maintains climate control with temperature between 70-76°F and relative humidity not exceeding 65%\n\n"
    "Provide the name of the theater, its street address, seating capacity, and reference URLs documenting how it meets each of these requirements."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RequirementSources(BaseModel):
    performing_arts_theater: List[str] = Field(default_factory=list)
    theater_district_location: List[str] = Field(default_factory=list)
    capacity: List[str] = Field(default_factory=list)
    ada: List[str] = Field(default_factory=list)
    emergency_exits: List[str] = Field(default_factory=list)
    restroom: List[str] = Field(default_factory=list)
    hvac: List[str] = Field(default_factory=list)
    loading_dock: List[str] = Field(default_factory=list)
    dressing_rooms: List[str] = Field(default_factory=list)
    stage_apron: List[str] = Field(default_factory=list)
    orchestra_pit: List[str] = Field(default_factory=list)
    adjustable_proscenium: List[str] = Field(default_factory=list)
    insurance: List[str] = Field(default_factory=list)
    backstage_access: List[str] = Field(default_factory=list)
    internet: List[str] = Field(default_factory=list)
    climate_control: List[str] = Field(default_factory=list)


class TheaterExtraction(BaseModel):
    theater_name: Optional[str] = None
    street_address: Optional[str] = None
    seating_capacity: Optional[str] = None
    general_sources: List[str] = Field(default_factory=list)
    requirement_sources: RequirementSources = Field(default_factory=RequirementSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theater_info() -> str:
    return """
Extract the theater identification and all cited reference URLs from the answer.

Return a JSON object with:
- theater_name: The theater's name, exactly as in the answer (string or null).
- street_address: The theater's street address as stated (string or null).
- seating_capacity: The seating capacity value as stated (string or null; keep original formatting).
- general_sources: An array of all URLs that generally describe the theater (home page, Wikipedia, booking pages, official specs, etc.) if present.
- requirement_sources: A JSON object mapping each requirement to the list of URLs explicitly cited in the answer for that requirement. Use the following keys:
  - performing_arts_theater
  - theater_district_location
  - capacity
  - ada
  - emergency_exits
  - restroom
  - hvac
  - loading_dock
  - dressing_rooms
  - stage_apron
  - orchestra_pit
  - adjustable_proscenium
  - insurance
  - backstage_access
  - internet
  - climate_control

Rules:
- Extract only URLs explicitly present in the answer text (including plain URLs or markdown links).
- Do not invent or infer any URL.
- If a field/key has no URLs mentioned, return an empty list for it.
- If a string field is not present in the answer, return null for that field.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(
    extracted: TheaterExtraction,
    requirement_keys: List[str],
    include_general: bool = True
) -> List[str]:
    """Merge requirement-specific sources with general sources, deduplicated."""
    urls: List[str] = []
    if include_general and extracted.general_sources:
        urls.extend(extracted.general_sources)

    rs = extracted.requirement_sources
    for key in requirement_keys:
        arr: List[str] = getattr(rs, key, []) if hasattr(rs, key) else []
        if arr:
            urls.extend(arr)

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def no_url_fallback_instruction(extra: str = "") -> str:
    base = (
        "Judge support strictly based on the provided URL(s). "
        "If no relevant URL is provided for this specific claim, you must mark it as not supported."
    )
    if extra:
        return f"{base} {extra}"
    return base


async def add_and_verify_requirement(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    critical: bool,
    extra_prereqs: Optional[List[Any]] = None,
    additional_instruction: Optional[str] = None,
) -> None:
    """Add a leaf for a single requirement and verify it."""
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    add_ins = additional_instruction or ""
    if not sources:
        add_ins = no_url_fallback_instruction(add_ins)

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction=add_ins,
        extra_prerequisites=extra_prereqs,
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
    Evaluate an answer for the Broadway technical/operational requirements task.
    """
    # Initialize evaluator (root is non-critical to allow critical and non-critical children)
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

    # Extract structured info
    extracted: TheaterExtraction = await evaluator.extract(
        prompt=prompt_extract_theater_info(),
        template_class=TheaterExtraction,
        extraction_name="theater_extraction",
    )

    # Build the rubric tree according to the provided JSON

    # 1) Venue Identification Information (critical group)
    id_group = evaluator.add_parallel(
        id="Venue_Identification_Information",
        desc="Solution provides required identifying information and references.",
        parent=root,
        critical=True,
    )

    # Existence checks as custom nodes (critical)
    theater_name_str = (extracted.theater_name or "").strip()
    street_addr_str = (extracted.street_address or "").strip()
    seating_capacity_str = (extracted.seating_capacity or "").strip()

    # Calculate overall presence of any references
    all_req_keys = [
        "performing_arts_theater", "theater_district_location", "capacity", "ada", "emergency_exits",
        "restroom", "hvac", "loading_dock", "dressing_rooms", "stage_apron", "orchestra_pit",
        "adjustable_proscenium", "insurance", "backstage_access", "internet", "climate_control"
    ]
    all_urls_union = merge_sources(extracted, all_req_keys, include_general=True)

    name_node = evaluator.add_custom_node(
        result=bool(theater_name_str),
        id="Theater_Name_Provided",
        desc="Provides the theater name.",
        parent=id_group,
        critical=True,
    )
    address_node = evaluator.add_custom_node(
        result=bool(street_addr_str),
        id="Street_Address_Provided",
        desc="Provides the theater street address.",
        parent=id_group,
        critical=True,
    )
    capacity_node = evaluator.add_custom_node(
        result=bool(seating_capacity_str),
        id="Seating_Capacity_Provided",
        desc="Provides the theater seating capacity value.",
        parent=id_group,
        critical=True,
    )
    refs_node = evaluator.add_custom_node(
        result=bool(all_urls_union),
        id="Reference_URLs_Provided",
        desc="Provides reference URL(s) that document the claims used to satisfy the constraints/requirements.",
        parent=id_group,
        critical=True,
    )

    # 2) Venue Type and Location (critical group)
    type_loc_group = evaluator.add_parallel(
        id="Venue_Type_And_Location",
        desc="Venue matches the required type and location from the question.",
        parent=root,
        critical=True,
    )

    tn = theater_name_str if theater_name_str else "the theater"

    # 2.1 Is Performing Arts Theater
    sources_is_theater = merge_sources(extracted, ["performing_arts_theater"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=type_loc_group,
        node_id="Is_Performing_Arts_Theater",
        desc="The identified venue is a performing arts theater (not a non-theater venue).",
        claim=f"{tn} is a performing arts theater venue designed and used for live performances.",
        sources=sources_is_theater,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Confirm that the webpages explicitly indicate it's a theater used for performing arts (live performances), not merely an event hall or non-theater facility.",
    )

    # 2.2 Located In Manhattan Theater District
    sources_location = merge_sources(extracted, ["theater_district_location"], include_general=True)
    location_claim = (
        f"{tn} is located in Manhattan's Theater District in New York City."
        + (f" The stated address is {street_addr_str}." if street_addr_str else "")
    )
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=type_loc_group,
        node_id="Located_In_Manhattan_Theater_District",
        desc="The venue is located in Manhattan's Theater District, New York City.",
        claim=location_claim,
        sources=sources_location,
        critical=True,
        extra_prereqs=[name_node, address_node, refs_node],
        additional_instruction="Look for explicit references to Theater District/Times Square/Broadway district in Manhattan, NYC, or authoritative address context indicating location within the Theater District.",
    )

    # 3) Broadway capacity classification (>= 500 seats)
    sources_capacity = merge_sources(extracted, ["capacity"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Broadway_Capacity_Classification",
        desc="Venue seating capacity is at least 500 seats (Broadway classification requirement as stated).",
        claim=f"{tn} has a seating capacity of at least 500 seats (meeting the Broadway classification threshold).",
        sources=sources_capacity,
        critical=True,
        extra_prereqs=[name_node, capacity_node, refs_node],
        additional_instruction="Confirm that the cited source(s) show the seating capacity and that it is ≥ 500.",
    )

    # 4) ADA Wheelchair Accessibility
    sources_ada = merge_sources(extracted, ["ada"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="ADA_Wheelchair_Accessibility",
        desc="Venue provides ADA-compliant wheelchair-accessible seating spaces in the required ratio for its capacity.",
        claim=f"{tn} provides ADA-compliant wheelchair-accessible seating spaces in the required ratio for its capacity.",
        sources=sources_ada,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Look for explicit ADA seating compliance statements or specifications indicating wheelchair seating and compliance with ratios required by ADA/NYC code.",
    )

    # 5) Emergency Exit Compliance
    sources_exits = merge_sources(extracted, ["emergency_exits"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Emergency_Exit_Compliance",
        desc="Venue has at least 3 emergency exits to meet fire safety code requirements for its capacity.",
        claim=f"{tn} has at least three emergency exits appropriate for its capacity per code requirements.",
        sources=sources_exits,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Verify explicit documentation or specifications regarding the count/availability of emergency exits meeting code (≥ 3).",
    )

    # 6) Restroom Facilities
    sources_restroom = merge_sources(extracted, ["restroom"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Restroom_Facilities",
        desc="Venue provides adequate restroom facilities meeting assembly venue code requirements.",
        claim=f"{tn} provides restroom facilities that meet assembly venue code requirements.",
        sources=sources_restroom,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Check documents for statements about restroom adequacy/capacity or compliance with assembly space codes.",
    )

    # 7) HVAC Ventilation Standards (ASHRAE 62.1-2019)
    sources_hvac = merge_sources(extracted, ["hvac"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="HVAC_Ventilation_Standards",
        desc="HVAC meets ASHRAE 62.1-2019 ventilation requirement (minimum 5 CFM/person plus 0.06 CFM/sq ft).",
        claim=f"The HVAC in {tn} meets ASHRAE 62.1-2019 ventilation requirements (≥ 5 CFM per person plus 0.06 CFM per sq ft).",
        sources=sources_hvac,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Look for explicit statements about ASHRAE 62.1-2019 compliance or equivalent ventilation performance metrics.",
    )

    # 8) Loading Dock Specs (48-inch height)
    sources_loading = merge_sources(extracted, ["loading_dock"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Loading_Dock_Specifications",
        desc="Venue has a loading dock at standard 48-inch (4-foot) height.",
        claim=f"{tn} has a loading dock with a standard height of approximately 48 inches (4 feet).",
        sources=sources_loading,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Confirm the presence and approximate height of the loading dock at ~48 inches.",
    )

    # 9) Dressing Room Amenities
    sources_dressing = merge_sources(extracted, ["dressing_rooms"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Dressing_Room_Amenities",
        desc="Dressing rooms have washstands with hot and cold running water.",
        claim=f"{tn} has dressing rooms equipped with washstands providing hot and cold running water.",
        sources=sources_dressing,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Look for dressing room amenity descriptions including washstands/sinks with hot and cold water.",
    )

    # 10) Stage Apron
    sources_apron = merge_sources(extracted, ["stage_apron"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Stage_Apron",
        desc="Venue has a stage apron (extension in front of the proscenium).",
        claim=f"{tn} has a stage apron (an extension in front of the proscenium).",
        sources=sources_apron,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Verify mention of a stage apron or forestage extending in front of the proscenium.",
    )

    # 11) Orchestra Pit (adjustable height)
    sources_pit = merge_sources(extracted, ["orchestra_pit"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Orchestra_Pit",
        desc="Venue has an orchestra pit with adjustable height capability.",
        claim=f"{tn} has an orchestra pit with adjustable height capability.",
        sources=sources_pit,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Confirm presence of an orchestra pit and that its height is adjustable (e.g., lift mechanism).",
    )

    # 12) Adjustable Proscenium Opening
    sources_prosc = merge_sources(extracted, ["adjustable_proscenium"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Adjustable_Proscenium",
        desc="Venue has an adjustable proscenium opening.",
        claim=f"{tn} has an adjustable proscenium opening.",
        sources=sources_prosc,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Look for specifications indicating that width/height of the proscenium can be adjusted.",
    )

    # 13) Insurance Requirements
    sources_ins = merge_sources(extracted, ["insurance"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Insurance_Requirements",
        desc="Venue requires proof of general liability insurance from renters/performers (minimum $1,000,000).",
        claim=f"{tn} requires renters/performers to provide proof of general liability insurance with at least $1,000,000 coverage.",
        sources=sources_ins,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Verify rental/booking policy language specifying general liability insurance requirements and minimum coverage of $1,000,000.",
    )

    # 14) Backstage Access
    sources_backstage = merge_sources(extracted, ["backstage_access"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Backstage_Access",
        desc="Venue provides direct backstage access from the loading dock to the stage area.",
        claim=f"{tn} provides direct backstage access from the loading dock to the stage area.",
        sources=sources_backstage,
        critical=True,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Confirm descriptions or diagrams noting direct access from loading dock to stage/backstage.",
    )

    # 15) Internet Connectivity (non-critical)
    sources_net = merge_sources(extracted, ["internet"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Internet_Connectivity",
        desc="Venue provides internet connectivity meeting the recommended minimum (10 Mbps per 100 people).",
        claim=f"{tn} provides internet connectivity meeting at least 10 Mbps per 100 people (recommended minimum).",
        sources=sources_net,
        critical=False,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Check for stated bandwidth, Wi-Fi specifications, or equivalent capacity statements meeting or exceeding the recommendation.",
    )

    # 16) Climate Control (non-critical)
    sources_climate = merge_sources(extracted, ["climate_control"], include_general=True)
    await add_and_verify_requirement(
        evaluator=evaluator,
        parent=root,
        node_id="Climate_Control",
        desc="Venue maintains climate control: 70–76°F and relative humidity ≤ 65%.",
        claim=f"{tn} maintains climate control with temperatures between 70–76°F and relative humidity not exceeding 65%.",
        sources=sources_climate,
        critical=False,
        extra_prereqs=[name_node, refs_node],
        additional_instruction="Look for environmental controls policy/standards indicating temperature and humidity ranges as specified.",
    )

    # Return the final evaluation summary
    return evaluator.get_summary()