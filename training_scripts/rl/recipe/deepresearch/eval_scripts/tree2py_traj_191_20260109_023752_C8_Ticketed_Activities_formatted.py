import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "austin_concert_venue_2026"
TASK_DESCRIPTION = (
    "Identify a suitable concert venue in Austin, Texas for a touring music artist planning a live performance event in 2026. "
    "The venue must meet ALL of the following requirements:\n\n"
    "1. Located within Austin, Texas city limits\n"
    "2. Total capacity (seating plus standing) between 15,000 and 20,000 people\n"
    "3. Wheelchair-accessible seating equal to at least 1% of total capacity with adjacent companion seats (ADA compliant)\n"
    "4. Adequate restroom facilities meeting building code requirements for the occupancy load\n"
    "5. Professional-grade sound system suitable for live concerts\n"
    "6. Performance stage with minimum dimensions of 40 feet wide by 30 feet deep\n"
    "7. Loading dock or equipment access with door dimensions of at least 10 feet high by 12 feet wide\n"
    "8. Backstage facilities including dressing rooms for performers\n"
    "9. On-site or immediately adjacent parking capacity for at least 2,000 vehicles\n"
    "10. Compliance with NFPA 101 Life Safety Code for emergency exits and egress\n"
    "11. Adequate HVAC or climate control systems for the capacity\n"
    "12. Designated VIP seating or hospitality areas\n"
    "13. Support for major ticketing platforms (Ticketmaster, AXS, SeatGeek, or equivalent)\n"
    "14. Suitable for hosting concerts and live music events (not exclusively sports-only)\n"
    "15. Security infrastructure capable of supporting minimum recommended staffing ratios (approximately 150 security personnel for 15,000+ capacity events)\n\n"
    "Provide the name of ONE venue that meets all these requirements, along with supporting documentation (URLs) that verify each requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementEvidence(BaseModel):
    """Statement and source URLs that support a specific requirement."""
    claim: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VenueSelectionExtraction(BaseModel):
    """Extraction for the single selected venue and evidence for each requirement."""
    venue_name: Optional[str] = None

    location: RequirementEvidence = Field(default_factory=RequirementEvidence)
    capacity: RequirementEvidence = Field(default_factory=RequirementEvidence)
    wheelchair_access: RequirementEvidence = Field(default_factory=RequirementEvidence)
    restrooms: RequirementEvidence = Field(default_factory=RequirementEvidence)
    sound_system: RequirementEvidence = Field(default_factory=RequirementEvidence)
    stage: RequirementEvidence = Field(default_factory=RequirementEvidence)
    loading_dock: RequirementEvidence = Field(default_factory=RequirementEvidence)
    backstage: RequirementEvidence = Field(default_factory=RequirementEvidence)
    parking: RequirementEvidence = Field(default_factory=RequirementEvidence)
    emergency_exit: RequirementEvidence = Field(default_factory=RequirementEvidence)
    hvac: RequirementEvidence = Field(default_factory=RequirementEvidence)
    vip: RequirementEvidence = Field(default_factory=RequirementEvidence)
    ticketing: RequirementEvidence = Field(default_factory=RequirementEvidence)
    venue_type: RequirementEvidence = Field(default_factory=RequirementEvidence)
    security: RequirementEvidence = Field(default_factory=RequirementEvidence)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_selection() -> str:
    return (
        "Extract the single venue recommended in the answer and the specific evidence provided for each requirement.\n"
        "Return a JSON object with the following fields:\n"
        "1) venue_name: The name of the ONE venue the answer selects. If multiple venues are mentioned, extract the primary recommended venue (or the first clearly recommended one).\n"
        "2) For each requirement below, extract:\n"
        "   - claim: The statement (or concise paraphrase) in the answer asserting the requirement is met for the selected venue.\n"
        "   - sources: All URLs cited in the answer that support this requirement (use only explicit URLs present in the answer, including markdown links). If none are cited, return an empty list.\n"
        "Requirements and corresponding JSON keys:\n"
        "- location: Venue is located within Austin, Texas city limits.\n"
        "- capacity: Total capacity (seating plus standing) between 15,000 and 20,000 people.\n"
        "- wheelchair_access: Wheelchair-accessible seating equals at least 1% of total capacity with adjacent companion seats (ADA compliant).\n"
        "- restrooms: Adequate restroom facilities meeting building code requirements for the occupancy load.\n"
        "- sound_system: Professional-grade sound system suitable for live concerts.\n"
        "- stage: Performance stage with minimum dimensions of 40 feet wide by 30 feet deep.\n"
        "- loading_dock: Loading dock or equipment access with door dimensions of at least 10 feet high by 12 feet wide.\n"
        "- backstage: Backstage facilities including dressing rooms for performers.\n"
        "- parking: On-site or immediately adjacent parking capacity for at least 2,000 vehicles.\n"
        "- emergency_exit: Compliance with NFPA 101 Life Safety Code for emergency exits and egress.\n"
        "- hvac: Adequate HVAC or climate control systems for the capacity.\n"
        "- vip: Designated VIP seating or hospitality areas.\n"
        "- ticketing: Support for major ticketing platforms (Ticketmaster, AXS, SeatGeek, or equivalent).\n"
        "- venue_type: Suitable for hosting concerts and live music events (not exclusively sports-only).\n"
        "- security: Security infrastructure capable of supporting minimum recommended staffing ratios (approximately 150 security personnel for 15,000+ capacity events).\n"
        "If a requirement is not discussed in the answer, set 'claim' to null and 'sources' to an empty list for that requirement. "
        "Extract only URLs explicitly present in the answer text. Do not invent URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def default_claim_for_requirement(venue_name: Optional[str], req_key: str) -> str:
    vn = venue_name or "the venue"
    mapping = {
        "location": f"The venue '{vn}' is located within Austin, Texas city limits.",
        "capacity": f"The venue '{vn}' has a total capacity between 15,000 and 20,000 people.",
        "wheelchair_access": f"The venue '{vn}' provides wheelchair-accessible seating equal to at least 1% of total capacity with adjacent companion seats and is ADA compliant.",
        "restrooms": f"The venue '{vn}' has adequate restroom facilities that meet building code requirements for the occupancy load.",
        "sound_system": f"The venue '{vn}' has a professional-grade sound system suitable for live concert performances.",
        "stage": f"The performance stage at '{vn}' meets minimum dimensions of at least 40 feet wide by 30 feet deep.",
        "loading_dock": f"The venue '{vn}' has loading dock/equipment access with doors at least 10 feet high and 12 feet wide.",
        "backstage": f"The venue '{vn}' provides backstage facilities including dressing rooms for performers.",
        "parking": f"The venue '{vn}' has on-site or immediately adjacent parking capacity for at least 2,000 vehicles.",
        "emergency_exit": f"The venue '{vn}' complies with NFPA 101 Life Safety Code requirements for emergency exits and egress.",
        "hvac": f"The venue '{vn}' has adequate HVAC/climate control systems appropriate for its capacity.",
        "vip": f"The venue '{vn}' provides designated VIP seating or hospitality areas.",
        "ticketing": f"The venue '{vn}' supports major ticketing platforms such as Ticketmaster, AXS, SeatGeek, or equivalent.",
        "venue_type": f"The venue '{vn}' is suitable for hosting concerts and live music events (not exclusively sports-only).",
        "security": f"The venue '{vn}' has security infrastructure capable of supporting approximately 150 security personnel for 15,000+ capacity events."
    }
    return mapping.get(req_key, f"The venue '{vn}' meets the requirement '{req_key}'.")  # fallback


async def verify_requirement(
    evaluator: Evaluator,
    parent_node,
    venue_extraction: VenueSelectionExtraction,
    req_key: str,
    req_node_id: str,
    leaf_id: str,
    req_desc: str,
    add_ins: str,
) -> None:
    """
    Build a sequential critical requirement node with:
      - A critical existence check for sources
      - A critical leaf verification using the provided sources
    """
    # Create requirement node under the main critical selection node
    req_node = evaluator.add_sequential(
        id=req_node_id,
        desc=req_desc,
        parent=parent_node,
        critical=True
    )

    # Get evidence object
    evidence: RequirementEvidence = getattr(venue_extraction, req_key, RequirementEvidence())

    # Existence check: must have at least one source URL
    sources_exist = bool(evidence.sources)
    evaluator.add_custom_node(
        result=sources_exist,
        id=f"{req_node_id}_Sources_Provided",
        desc=f"Sources are provided in the answer for {req_desc}",
        parent=req_node,
        critical=True
    )

    # Leaf verification: claim must be supported by the cited URL(s)
    verify_leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=f"Provide URL reference documenting: {req_desc}",
        parent=req_node,
        critical=True
    )

    # Compose claim (prefer extracted claim, fallback to default)
    claim_text = evidence.claim or default_claim_for_requirement(venue_extraction.venue_name, req_key)

    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=evidence.sources,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Main verification orchestration                                             #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: VenueSelectionExtraction
) -> None:
    """
    Build the verification tree for the concert venue selection task.
    """
    # Create a critical top-level node under root to reflect the rubric's critical root
    main_node = evaluator.add_parallel(
        id="Concert_Venue_Selection",
        desc="Identify a suitable concert venue in Austin, Texas that meets all specified requirements for a 15,000-20,000 capacity live music event",
        parent=evaluator.root,
        critical=True
    )

    # Ensure a single venue name is provided (critical to the overall selection)
    venue_name_exists = extraction.venue_name is not None and extraction.venue_name.strip() != ""
    evaluator.add_custom_node(
        result=venue_name_exists,
        id="Venue_Name_Provided",
        desc="The answer provides the name of ONE specific venue",
        parent=main_node,
        critical=True
    )

    # Requirement configuration mapping
    reqs = [
        {
            "req_key": "location",
            "req_node_id": "Location_Austin_TX",
            "leaf_id": "Location_Reference",
            "desc": "The venue must be physically located within Austin, Texas city limits",
            "add_ins": "Verify that the page explicitly indicates the venue's address within the Austin, Texas city limits (not merely the metro area). Accept official venue pages, event pages, or authoritative directories showing an Austin, TX address."
        },
        {
            "req_key": "capacity",
            "req_node_id": "Capacity_Range_15000_20000",
            "leaf_id": "Capacity_Reference",
            "desc": "The venue's total capacity (seating plus standing) must be between 15,000 and 20,000 people",
            "add_ins": "Check the stated total capacity. If the page shows a single value (e.g., 18,000) ensure it lies within 15,000–20,000. If a range is given, ensure it includes values within that band."
        },
        {
            "req_key": "wheelchair_access",
            "req_node_id": "Wheelchair_Accessible_Seating",
            "leaf_id": "Wheelchair_Access_Reference",
            "desc": "The venue provides wheelchair-accessible seating spaces equal to at least 1% of total capacity with adjacent companion seats, meeting ADA requirements",
            "add_ins": "Look for ADA seating policies, counts or ratios, and companion seating. Evidence should indicate ADA compliance and adequate accessible seating capacity consistent with at least 1% of total capacity."
        },
        {
            "req_key": "restrooms",
            "req_node_id": "Restroom_Facilities",
            "leaf_id": "Restroom_Reference",
            "desc": "The venue has adequate restroom facilities meeting building code requirements for the stated occupancy load",
            "add_ins": "Look for building code compliance statements or specifications suggesting adequate restroom fixtures for large occupancy loads."
        },
        {
            "req_key": "sound_system",
            "req_node_id": "Professional_Sound_System",
            "leaf_id": "Sound_System_Reference",
            "desc": "The venue has a professional-grade sound system suitable for live concert performances",
            "add_ins": "Check for mentions of concert-quality audio systems, professional PA, acoustic design, or production specifications indicating a suitable sound system for live music."
        },
        {
            "req_key": "stage",
            "req_node_id": "Stage_Dimensions",
            "leaf_id": "Stage_Reference",
            "desc": "The performance stage has minimum dimensions of 40 feet wide by 30 feet deep",
            "add_ins": "Look for stage dimension specifications; dimensions meeting or exceeding 40 ft width and 30 ft depth satisfy this requirement."
        },
        {
            "req_key": "loading_dock",
            "req_node_id": "Loading_Dock_Access",
            "leaf_id": "Loading_Dock_Reference",
            "desc": "The venue has a loading dock or equipment access with door dimensions of at least 10 feet high by 12 feet wide",
            "add_ins": "Look for loading dock or freight access details; doors with dimensions >=10 ft high and >=12 ft wide meet the requirement."
        },
        {
            "req_key": "backstage",
            "req_node_id": "Backstage_Facilities",
            "leaf_id": "Backstage_Reference",
            "desc": "The venue provides backstage facilities including dressing rooms for performers",
            "add_ins": "Check for mentions of backstage areas, green rooms, dressing rooms, or artist facilities."
        },
        {
            "req_key": "parking",
            "req_node_id": "Parking_Capacity",
            "leaf_id": "Parking_Reference",
            "desc": "The venue has on-site parking or immediate adjacent parking capacity for at least 2,000 vehicles",
            "add_ins": "Look for parking capacity on site or in immediately adjacent facilities; total capacity across adjacent lots/garages reaching >=2,000 vehicles satisfies the requirement."
        },
        {
            "req_key": "emergency_exit",
            "req_node_id": "Emergency_Exit_Compliance",
            "leaf_id": "Emergency_Exit_Reference",
            "desc": "The venue complies with NFPA 101 Life Safety Code requirements for emergency exits and egress",
            "add_ins": "Look for explicit compliance statements with NFPA 101 or equivalent life safety codes adopted by local jurisdiction, indicating compliant egress and emergency exits."
        },
        {
            "req_key": "hvac",
            "req_node_id": "HVAC_Climate_Control",
            "leaf_id": "HVAC_Reference",
            "desc": "The venue has adequate HVAC or climate control systems appropriate for the capacity",
            "add_ins": "Check for HVAC/climate control specifications suitable for large audiences; references to engineered HVAC systems or capacity-specific climate control are acceptable."
        },
        {
            "req_key": "vip",
            "req_node_id": "VIP_Areas",
            "leaf_id": "VIP_Reference",
            "desc": "The venue provides designated VIP seating or hospitality areas",
            "add_ins": "Look for VIP seating, suites, hospitality lounges, or similar premium areas explicitly designated for VIPs."
        },
        {
            "req_key": "ticketing",
            "req_node_id": "Ticketing_Platform_Integration",
            "leaf_id": "Ticketing_Reference",
            "desc": "The venue supports major ticketing platforms such as Ticketmaster, AXS, SeatGeek, or equivalent professional platforms",
            "add_ins": "Evidence can include official venue pages linking to Ticketmaster/AXS/SeatGeek events or partner listings indicating integration/support."
        },
        {
            "req_key": "venue_type",
            "req_node_id": "Concert_Venue_Type",
            "leaf_id": "Venue_Type_Reference",
            "desc": "The venue is suitable for hosting concerts and live music events (not exclusively a sports-only facility)",
            "add_ins": "Look for past or scheduled concerts, live music programming, or venue descriptions positioning it as suitable for live music."
        },
        {
            "req_key": "security",
            "req_node_id": "Security_Infrastructure",
            "leaf_id": "Security_Reference",
            "desc": "The venue has security infrastructure capable of supporting minimum recommended staffing ratios (approximately 150 security personnel for 15,000+ capacity events)",
            "add_ins": "Look for security operations descriptions, staffing plans, or capabilities indicating support for large-event security staffing on the order of ~150 personnel for 15k+ capacity."
        },
    ]

    # Create nodes and verifications for each requirement
    for r in reqs:
        await verify_requirement(
            evaluator=evaluator,
            parent_node=main_node,
            venue_extraction=extraction,
            req_key=r["req_key"],
            req_node_id=r["req_node_id"],
            leaf_id=r["leaf_id"],
            req_desc=r["desc"],
            add_ins=r["add_ins"]
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
    Evaluate the agent's answer for the Austin concert venue selection task.
    Returns the evaluator's structured summary with the verification tree and final score.
    """
    # Initialize evaluator with a parallel root (we'll add a critical main node under it)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venue_selection(),
        template_class=VenueSelectionExtraction,
        extraction_name="venue_selection_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return standard summary
    return evaluator.get_summary()