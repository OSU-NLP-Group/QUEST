import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_gulf_coast_hotel_compliance"
TASK_DESCRIPTION = """
Identify a beachfront hotel development project located on the Texas Gulf Coast that is either currently under construction or was completed within the last 3 years (2023-2026). The hotel must meet the following requirements:

1. Guest Room Size: Guest rooms must be at least 325 square feet in size
2. Parking Accessibility: The parking facilities must comply with ADA requirements, providing at least one accessible parking space for every 25 regular parking spaces
3. Pool Safety Features: If the hotel includes a commercial swimming pool, it must have VGBA-compliant anti-entrapment drain covers, proper depth markers and safety signage, and safety equipment such as life rings, reaching poles, or first aid kits
4. Fire Safety Systems: The hotel must be equipped with both a fire alarm system and an automatic sprinkler system
5. Coastal Construction Compliance: The project must comply with Texas coastal construction regulations, including the Texas Open Beaches Act, and must use materials designed to withstand high winds and heavy rain
6. Total Room Count: Provide the total number of guest rooms in the hotel

For your answer, provide:
- The name of the hotel
- Its specific location on the Texas Gulf Coast
- Documentation or references confirming it meets each of the above requirements
- The total number of guest rooms
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelExtraction(BaseModel):
    # Identification
    hotel_name: Optional[str] = None
    location: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)
    beachfront_statement: Optional[str] = None
    beachfront_sources: List[str] = Field(default_factory=list)
    status_statement: Optional[str] = None  # e.g., "under construction (2025 completion)"
    status_sources: List[str] = Field(default_factory=list)

    # Requirements evidence
    guest_room_size_statement: Optional[str] = None  # e.g., "All guestrooms start at 340 sqft"
    guest_room_size_sources: List[str] = Field(default_factory=list)

    parking_ada_statement: Optional[str] = None  # e.g., "ADA parking ratio compliant (1 per 25)"
    parking_ada_sources: List[str] = Field(default_factory=list)

    has_commercial_pool: Optional[bool] = None  # True/False if clearly stated; null if unclear
    pool_presence_statement: Optional[str] = None
    pool_presence_sources: List[str] = Field(default_factory=list)

    vgba_statement: Optional[str] = None
    vgba_sources: List[str] = Field(default_factory=list)

    depth_signage_statement: Optional[str] = None
    depth_signage_sources: List[str] = Field(default_factory=list)

    safety_equipment_statement: Optional[str] = None  # Should mention life rings, reaching poles, first aid kits
    safety_equipment_sources: List[str] = Field(default_factory=list)

    fire_alarm_statement: Optional[str] = None
    fire_alarm_sources: List[str] = Field(default_factory=list)

    sprinkler_statement: Optional[str] = None
    sprinkler_sources: List[str] = Field(default_factory=list)

    open_beaches_compliance_statement: Optional[str] = None
    open_beaches_compliance_sources: List[str] = Field(default_factory=list)

    materials_resilience_statement: Optional[str] = None
    materials_resilience_sources: List[str] = Field(default_factory=list)

    setback_statement: Optional[str] = None
    setback_sources: List[str] = Field(default_factory=list)

    total_room_count: Optional[str] = None
    total_room_count_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
Extract the following fields from the provided answer EXACTLY as stated. Do not invent information. For each item, also extract all cited source URLs that directly support the item. If an item is not present, set it to null, and return an empty list for its sources. Return valid, complete URLs only.

Required fields (JSON keys):
- hotel_name: string | null
- location: string | null
- location_sources: string[]    // URLs supporting the specific Texas Gulf Coast location
- beachfront_statement: string | null
- beachfront_sources: string[]  // URLs supporting that the project is beachfront (directly on or adjacent to beach/shoreline)
- status_statement: string | null   // e.g., "under construction as of 2025", "opened 2024"
- status_sources: string[]          // URLs supporting status within 2023–2026

- guest_room_size_statement: string | null  // evidence that guest rooms are >= 325 sq ft
- guest_room_size_sources: string[]

- parking_ada_statement: string | null      // evidence that parking provides >= 1 accessible space per 25 regular spaces and complies with ADA/TAS
- parking_ada_sources: string[]

- has_commercial_pool: boolean | null       // true if a hotel commercial pool exists; false if explicitly states no pool; null if unclear
- pool_presence_statement: string | null
- pool_presence_sources: string[]

- vgba_statement: string | null             // VGBA-compliant anti-entrapment drain covers for the pool; if no pool, may be N/A statement
- vgba_sources: string[]                    // If no pool, use sources proving there is no pool

- depth_signage_statement: string | null    // proper depth markers and safety signage; if no pool, may be N/A statement
- depth_signage_sources: string[]           // If no pool, use sources proving there is no pool

- safety_equipment_statement: string | null // evidence that safety equipment includes life rings, reaching poles, AND first aid kits; if no pool, may be N/A statement
- safety_equipment_sources: string[]        // If no pool, use sources proving there is no pool

- fire_alarm_statement: string | null       // hotel has a fire alarm system and is regularly tested/inspected
- fire_alarm_sources: string[]

- sprinkler_statement: string | null        // hotel has an automatic sprinkler system installed
- sprinkler_sources: string[]

- open_beaches_compliance_statement: string | null   // evidence of Texas Open Beaches Act / coastal regulations compliance
- open_beaches_compliance_sources: string[]

- materials_resilience_statement: string | null      // materials/design withstand high winds/heavy rain/debris (coastal resilience)
- materials_resilience_sources: string[]

- setback_statement: string | null                  // required setbacks from dunes and water
- setback_sources: string[]

- total_room_count: string | null                   // total number of guest rooms; extract the numeral exactly as stated
- total_room_count_sources: string[]                // URLs supporting total room count

RULES:
- Only extract URLs explicitly present in the answer (plain text or markdown). Do not invent URLs.
- If a URL lacks protocol, prepend http://
- For has_commercial_pool, return true/false only if the answer clearly states it. Otherwise, return null.
- Do not normalize wording. Keep statements verbatim when possible, but concise.
""".strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _clean_sources(sources: Optional[List[str]]) -> List[str]:
    if not sources:
        return []
    return [s.strip() for s in sources if isinstance(s, str) and s.strip()]


def _add_sources_presence_gate(
    evaluator: Evaluator,
    parent,
    base_id: str,
    what: str,
    sources: List[str],
    critical: bool = True,
):
    """Add a critical custom node to gate verification when sources are missing."""
    return evaluator.add_custom_node(
        result=len(_clean_sources(sources)) > 0,
        id=f"{base_id}_sources_present",
        desc=f"Evidence URLs provided for {what}",
        parent=parent,
        critical=critical,
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_identification_subtree(evaluator: Evaluator, parent, data: HotelExtraction):
    """
    Build 'identify_project' subtree:
      - hotel_name_provided
      - texas_gulf_coast_location (with sources)
      - beachfront_verification (with sources)
      - status_timeframe_2023_2026 (with sources)
    """
    node = evaluator.add_parallel(
        id="identify_project",
        desc="Identify a single qualifying beachfront hotel development project on the Texas Gulf Coast within the allowed timeframe.",
        parent=parent,
        critical=True,
    )

    # hotel_name_provided (existence)
    evaluator.add_custom_node(
        result=bool(data.hotel_name and str(data.hotel_name).strip()),
        id="hotel_name_provided",
        desc="Provide the name of the hotel development project.",
        parent=node,
        critical=True,
    )

    # texas_gulf_coast_location
    loc_sources = _clean_sources(data.location_sources)
    _add_sources_presence_gate(
        evaluator, node, "texas_gulf_coast_location", "Texas Gulf Coast location", loc_sources, critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="texas_gulf_coast_location",
        desc="Provide a specific location and a reference supporting that it is on the Texas Gulf Coast.",
        parent=node,
        critical=True,
    )
    loc_name = data.location or "the stated location"
    hotel_name = data.hotel_name or "the hotel"
    loc_claim = (
        f"The hotel '{hotel_name}' is located in '{loc_name}', and that location is on the Texas Gulf Coast."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=loc_sources,
        additional_instruction=(
            "Confirm the location is in Texas and on the Gulf of Mexico coastline/barrier islands; "
            "acceptable examples include Galveston, Corpus Christi, Port Aransas, South Padre Island, "
            "Rockport, Surfside Beach, etc., and coastal counties along the Gulf. "
            "If the sources do not clearly place it on the Texas Gulf Coast, mark as not supported."
        ),
    )

    # beachfront_verification
    beachfront_sources = _clean_sources(data.beachfront_sources)
    _add_sources_presence_gate(
        evaluator, node, "beachfront_verification", "beachfront status", beachfront_sources, critical=True
    )
    beachfront_leaf = evaluator.add_leaf(
        id="beachfront_verification",
        desc="Provide a reference showing the project is beachfront (directly on/adjacent to the beach/shoreline).",
        parent=node,
        critical=True,
    )
    beach_claim = (
        f"The project '{hotel_name}' is beachfront (directly on or adjacent to the beach/shoreline)."
    )
    await evaluator.verify(
        claim=beach_claim,
        node=beachfront_leaf,
        sources=beachfront_sources,
        additional_instruction=(
            "The page should show that the property fronts the beach/shoreline or is immediately adjacent with direct beach access. "
            "If only 'near the beach' without direct adjacency, mark not supported."
        ),
    )

    # status_timeframe_2023_2026
    status_sources = _clean_sources(data.status_sources)
    _add_sources_presence_gate(
        evaluator, node, "status_timeframe_2023_2026", "status/timeframe (2023–2026)", status_sources, critical=True
    )
    status_leaf = evaluator.add_leaf(
        id="status_timeframe_2023_2026",
        desc="Provide a reference showing the project is under construction or completed within 2023-2026.",
        parent=node,
        critical=True,
    )
    status_claim = (
        "The project is either currently under construction or was completed within the years 2023 to 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=status_sources,
        additional_instruction=(
            "Accept phrasing such as 'under construction as of 2023/2024/2025/2026', "
            "'opened in 2024', 'completion expected 2025', or similar. "
            "If the timeline falls outside 2023–2026 or is not clear, mark not supported."
        ),
    )


async def build_requirements_subtree(evaluator: Evaluator, parent, data: HotelExtraction):
    """
    Build 'verify_requirements' subtree, including all requirement checks and total room count.
    """
    req = evaluator.add_parallel(
        id="verify_requirements",
        desc="Verify, with referenced evidence, that the identified project meets each stated design/regulatory requirement and reports total room count.",
        parent=parent,
        critical=True,
    )

    hotel_name = data.hotel_name or "the hotel"

    # guest_room_size
    gr_sources = _clean_sources(data.guest_room_size_sources)
    _add_sources_presence_gate(evaluator, req, "guest_room_size", "guest room size", gr_sources, critical=True)
    gr_leaf = evaluator.add_leaf(
        id="guest_room_size",
        desc="Provide referenced evidence that guest rooms are at least 325 square feet.",
        parent=req,
        critical=True,
    )
    gr_claim = f"Guest rooms at '{hotel_name}' are at least 325 square feet in size."
    await evaluator.verify(
        claim=gr_claim,
        node=gr_leaf,
        sources=gr_sources,
        additional_instruction=(
            "Pass if sources show typical/standard room sizes are 325 sqft or larger (e.g., 'starting at 330 sqft'). "
            "If only suites are >=325 but standard rooms are smaller, mark not supported."
        ),
    )

    # parking_ada_ratio
    ada_sources = _clean_sources(data.parking_ada_sources)
    _add_sources_presence_gate(evaluator, req, "parking_ada_ratio", "ADA parking ratio/compliance", ada_sources, critical=True)
    ada_leaf = evaluator.add_leaf(
        id="parking_ada_ratio",
        desc="Provide referenced evidence that parking complies with ADA accessibility requirements, including at least one accessible parking space per 25 regular parking spaces.",
        parent=req,
        critical=True,
    )
    ada_claim = (
        f"The parking at '{hotel_name}' complies with ADA/TAS accessibility requirements by providing at least one "
        "accessible parking space for every 25 regular parking spaces."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=ada_sources,
        additional_instruction=(
            "Accept explicit statements of ADA or Texas Accessibility Standards compliance that confirm the minimum ratio "
            "(1 accessible space per 25 regular spaces) or show counts meeting/exceeding that ratio. "
            "If evidence is only general ADA guidance without project-specific compliance, mark not supported."
        ),
    )

    # pool_safety_conditional (parallel, all critical)
    pool_node = evaluator.add_parallel(
        id="pool_safety_conditional",
        desc="If the hotel includes a commercial swimming pool, provide referenced evidence it meets all specified pool safety features; if no commercial pool exists, explicitly state that and provide supporting evidence.",
        parent=req,
        critical=True,
    )

    # pool_presence_and_type
    pool_presence_sources = _clean_sources(data.pool_presence_sources)
    _add_sources_presence_gate(
        evaluator, pool_node, "pool_presence_and_type", "pool presence/type", pool_presence_sources, critical=True
    )
    pool_presence_leaf = evaluator.add_leaf(
        id="pool_presence_and_type",
        desc="State whether a commercial swimming pool is included, with a reference.",
        parent=pool_node,
        critical=True,
    )
    if data.has_commercial_pool is True:
        presence_claim = f"'{hotel_name}' includes a commercial swimming pool."
    elif data.has_commercial_pool is False:
        presence_claim = f"'{hotel_name}' does not include a commercial swimming pool."
    else:
        # Unknown; verify that sources accurately state whatever the answer claims about pool presence/type
        stated = data.pool_presence_statement or "the stated pool presence/type for the hotel"
        presence_claim = f"The sources support {stated} at '{hotel_name}'."
    await evaluator.verify(
        claim=presence_claim,
        node=pool_presence_leaf,
        sources=pool_presence_sources,
        additional_instruction=(
            "Confirm whether a hotel pool that would be considered a commercial/public accommodation pool exists. "
            "If the page clearly shows or states a hotel pool, treat it as a commercial pool."
        ),
    )

    # Helper to choose applicable sources for conditional items
    def _conditional_sources(has_pool: Optional[bool], item_sources: List[str], fallback_no_pool_sources: List[str]) -> List[str]:
        if has_pool is True:
            return _clean_sources(item_sources)
        else:
            # If no pool (or unclear but claimed N/A), rely on evidence that no commercial pool exists
            return _clean_sources(fallback_no_pool_sources if fallback_no_pool_sources else item_sources)

    # VGBA drains
    vgba_app_sources = _conditional_sources(data.has_commercial_pool, data.vgba_sources, pool_presence_sources)
    _add_sources_presence_gate(
        evaluator, pool_node, "vgba_drain_covers_if_applicable", "VGBA-compliant drain covers or N/A justification", vgba_app_sources, critical=True
    )
    vgba_leaf = evaluator.add_leaf(
        id="vgba_drain_covers_if_applicable",
        desc="If a commercial pool exists: provide referenced evidence of VGBA-compliant anti-entrapment drain covers; otherwise mark N/A with supporting evidence.",
        parent=pool_node,
        critical=True,
    )
    if data.has_commercial_pool is True:
        vgba_claim = f"The hotel's commercial pool at '{hotel_name}' uses VGBA-compliant anti-entrapment drain covers."
    else:
        vgba_claim = f"'{hotel_name}' has no commercial swimming pool; therefore VGBA drain cover requirements are not applicable."
    await evaluator.verify(
        claim=vgba_claim,
        node=vgba_leaf,
        sources=vgba_app_sources,
        additional_instruction=(
            "If a commercial pool exists, the source must explicitly indicate VGBA/VGB compliance or compliant anti-entrapment drain covers. "
            "If no commercial pool exists, confirm that and treat the requirement as N/A."
        ),
    )

    # Depth markers and safety signage
    signage_app_sources = _conditional_sources(data.has_commercial_pool, data.depth_signage_sources, pool_presence_sources)
    _add_sources_presence_gate(
        evaluator, pool_node, "depth_markers_signage_if_applicable", "depth markers and safety signage or N/A justification", signage_app_sources, critical=True
    )
    signage_leaf = evaluator.add_leaf(
        id="depth_markers_signage_if_applicable",
        desc="If a commercial pool exists: provide referenced evidence of proper depth markers and safety signage; otherwise mark N/A with supporting evidence.",
        parent=pool_node,
        critical=True,
    )
    if data.has_commercial_pool is True:
        signage_claim = f"The hotel's commercial pool at '{hotel_name}' has proper depth markers and required safety signage."
    else:
        signage_claim = f"'{hotel_name}' has no commercial swimming pool; therefore pool depth markers/signage are not applicable."
    await evaluator.verify(
        claim=signage_claim,
        node=signage_leaf,
        sources=signage_app_sources,
        additional_instruction=(
            "If a commercial pool exists, verify the presence of visible/required depth markers and safety signage. "
            "If no commercial pool exists, confirm that and treat this as N/A."
        ),
    )

    # Safety equipment
    safety_app_sources = _conditional_sources(data.has_commercial_pool, data.safety_equipment_sources, pool_presence_sources)
    _add_sources_presence_gate(
        evaluator, pool_node, "safety_equipment_if_applicable", "pool safety equipment or N/A justification", safety_app_sources, critical=True
    )
    safety_leaf = evaluator.add_leaf(
        id="safety_equipment_if_applicable",
        desc="If a commercial pool exists: provide referenced evidence that required safety equipment includes life rings, reaching poles, AND first aid kits; otherwise mark N/A with supporting evidence.",
        parent=pool_node,
        critical=True,
    )
    if data.has_commercial_pool is True:
        safety_claim = (
            f"The hotel's commercial pool at '{hotel_name}' provides required safety equipment including life rings, reaching poles, and first aid kits."
        )
    else:
        safety_claim = f"'{hotel_name}' has no commercial swimming pool; therefore pool safety equipment requirements are not applicable."
    await evaluator.verify(
        claim=safety_claim,
        node=safety_leaf,
        sources=safety_app_sources,
        additional_instruction=(
            "If a commercial pool exists, confirm that all three are present: life rings, reaching poles, and first aid kits. "
            "If no commercial pool exists, confirm that and treat this as N/A."
        ),
    )

    # fire_safety_systems (parallel, critical)
    fire_node = evaluator.add_parallel(
        id="fire_safety_systems",
        desc="Provide referenced evidence the hotel has both a fire alarm system and an automatic sprinkler system.",
        parent=req,
        critical=True,
    )

    # fire_alarm_installed_and_tested
    fire_alarm_sources = _clean_sources(data.fire_alarm_sources)
    _add_sources_presence_gate(
        evaluator, fire_node, "fire_alarm_installed_and_tested", "fire alarm installation/testing", fire_alarm_sources, critical=True
    )
    fire_alarm_leaf = evaluator.add_leaf(
        id="fire_alarm_installed_and_tested",
        desc="Provide referenced evidence that a fire alarm system is installed and regularly tested.",
        parent=fire_node,
        critical=True,
    )
    fire_alarm_claim = (
        f"'{hotel_name}' has a fire alarm system installed and it is tested/inspected regularly (e.g., by required inspections)."
    )
    await evaluator.verify(
        claim=fire_alarm_claim,
        node=fire_alarm_leaf,
        sources=fire_alarm_sources,
        additional_instruction=(
            "Look for references to installed fire alarm systems and testing/inspection schedules, certificates, or AHJ approvals."
        ),
    )

    # automatic_sprinkler_system
    sprinkler_sources = _clean_sources(data.sprinkler_sources)
    _add_sources_presence_gate(
        evaluator, fire_node, "automatic_sprinkler_system", "automatic sprinkler system", sprinkler_sources, critical=True
    )
    sprinkler_leaf = evaluator.add_leaf(
        id="automatic_sprinkler_system",
        desc="Provide referenced evidence that an automatic sprinkler system is installed.",
        parent=fire_node,
        critical=True,
    )
    sprinkler_claim = f"'{hotel_name}' has an automatic sprinkler system installed."
    await evaluator.verify(
        claim=sprinkler_claim,
        node=sprinkler_leaf,
        sources=sprinkler_sources,
        additional_instruction="Confirm a building-wide sprinkler system is installed (NFPA-13 or equivalent).",
    )

    # coastal_construction_compliance (parallel, critical)
    coast_node = evaluator.add_parallel(
        id="coastal_construction_compliance",
        desc="Provide referenced evidence of Texas coastal construction compliance, including Open Beaches Act compliance, resilient materials, and required setbacks.",
        parent=req,
        critical=True,
    )

    # open_beaches_act_compliance
    oba_sources = _clean_sources(data.open_beaches_compliance_sources)
    _add_sources_presence_gate(
        evaluator, coast_node, "open_beaches_act_compliance", "Texas Open Beaches Act / coastal regulations compliance", oba_sources, critical=True
    )
    oba_leaf = evaluator.add_leaf(
        id="open_beaches_act_compliance",
        desc="Provide referenced evidence the project complies with Texas coastal construction regulations including the Texas Open Beaches Act.",
        parent=coast_node,
        critical=True,
    )
    oba_claim = (
        f"The project at '{hotel_name}' complies with Texas coastal construction regulations, including the Texas Open Beaches Act."
    )
    await evaluator.verify(
        claim=oba_claim,
        node=oba_leaf,
        sources=oba_sources,
        additional_instruction=(
            "Accept explicit references to Open Beaches Act compliance, approvals/permits from Texas GLO or local coastal programs, "
            "or coastal construction certificates that imply OBA compliance."
        ),
    )

    # materials_resilience
    mat_sources = _clean_sources(data.materials_resilience_sources)
    _add_sources_presence_gate(
        evaluator, coast_node, "materials_resilience", "coastal-resilient materials/design", mat_sources, critical=True
    )
    mat_leaf = evaluator.add_leaf(
        id="materials_resilience",
        desc="Provide referenced evidence that materials/design are intended to withstand high winds, heavy rain, and potential debris impacts.",
        parent=coast_node,
        critical=True,
    )
    mat_claim = (
        f"The project's materials and/or design for '{hotel_name}' are intended to withstand high winds, heavy rain, and debris impacts (coastal resilience)."
    )
    await evaluator.verify(
        claim=mat_claim,
        node=mat_leaf,
        sources=mat_sources,
        additional_instruction=(
            "Look for hurricane/wind-rated assemblies, impact-resistant glazing, corrosion-resistant materials, "
            "IBHS/ASCE/IBC coastal wind design references, or equivalent statements."
        ),
    )

    # setback_from_dunes_and_water
    setback_sources = _clean_sources(data.setback_sources)
    _add_sources_presence_gate(
        evaluator, coast_node, "setback_from_dunes_and_water", "setbacks from dunes/water", setback_sources, critical=True
    )
    setback_leaf = evaluator.add_leaf(
        id="setback_from_dunes_and_water",
        desc="Provide referenced evidence that the project satisfies required setbacks from dunes and water per Texas coastal regulations.",
        parent=coast_node,
        critical=True,
    )
    setback_claim = (
        f"The project for '{hotel_name}' satisfies required coastal setbacks from dunes and the waterline per applicable Texas regulations."
    )
    await evaluator.verify(
        claim=setback_claim,
        node=setback_leaf,
        sources=setback_sources,
        additional_instruction=(
            "Accept site plans, permits, or regulatory approvals explicitly noting dune/beach access lines, "
            "Erosion Response Plans, or setback distances that meet requirements."
        ),
    )

    # total_room_count
    trc_sources = _clean_sources(data.total_room_count_sources)
    _add_sources_presence_gate(
        evaluator, req, "total_room_count", "total room count", trc_sources, critical=True
    )
    trc_leaf = evaluator.add_leaf(
        id="total_room_count",
        desc="Provide the total number of guest rooms in the hotel, with a reference.",
        parent=req,
        critical=True,
    )
    room_count_txt = data.total_room_count or "the stated total number"
    trc_claim = f"The hotel has a total of {room_count_txt} guest rooms."
    await evaluator.verify(
        claim=trc_claim,
        node=trc_leaf,
        sources=trc_sources,
        additional_instruction=(
            "Confirm the total guest room count (keys). Accept equivalent phrasing like 'keys' or 'guestrooms.' "
            "If only a range/approximate number is given and the answer states a specific total, mark not supported unless they match."
        ),
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
    Evaluate an answer for the Texas Gulf Coast beachfront hotel compliance task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Identify first, then requirements
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction",
    )

    # Build verification tree
    await build_identification_subtree(evaluator, root, extracted)
    await build_requirements_subtree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()