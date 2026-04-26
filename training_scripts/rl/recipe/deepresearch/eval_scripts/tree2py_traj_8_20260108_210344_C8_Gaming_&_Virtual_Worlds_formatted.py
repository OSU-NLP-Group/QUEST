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
TASK_ID = "us_esports_venue"
TASK_DESCRIPTION = (
    "Identify a major esports venue in the United States that meets ALL of the following comprehensive requirements "
    "for hosting a professional esports tournament:\n\n"
    "Capacity & Space Requirements:\n"
    "- Provides seating capacity for at least 1,000 spectators\n"
    "- Has a dedicated stage area suitable for esports competition\n\n"
    "Location Requirement:\n"
    "- Located in a major metropolitan area within the United States\n\n"
    "Accessibility & Safety Requirements:\n"
    "- Provides wheelchair-accessible seating meeting ADA requirements\n"
    "- Has adequate emergency exits and fire safety compliance\n"
    "- Has emergency medical facilities or medical support capabilities\n"
    "- Has adequate restroom facilities based on venue capacity\n\n"
    "Technical Infrastructure Requirements:\n"
    "- Has high-speed internet connectivity sufficient for multiple gaming stations and streaming\n"
    "- Has production/broadcast control room facilities\n"
    "- Has adequate lighting systems for production purposes\n\n"
    "Amenity Requirements:\n"
    "- Has VIP lounge or premium seating areas\n"
    "- Has food service facilities (food court or concessions)\n"
    "- Has player lounge or green room facilities for competitors\n\n"
    "Operational Infrastructure Requirements:\n"
    "- Has loading dock for equipment load-in and setup\n"
    "- Has adequate parking capacity for tournament attendees\n"
    "- Has backup power or generator systems\n"
    "- Has HVAC/climate control systems\n\n"
    "Provide the full name of the venue, its location (city and state), and reference URLs that verify each of the "
    "major requirement categories."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfoExtraction(BaseModel):
    """Core venue identification from the answer."""
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None


class VenueRefsExtraction(BaseModel):
    """Reference URLs grouped by major requirement categories from the answer."""
    capacity_and_space: List[str] = Field(default_factory=list)
    location: List[str] = Field(default_factory=list)
    accessibility_and_safety: List[str] = Field(default_factory=list)
    technical_infrastructure: List[str] = Field(default_factory=list)
    amenities: List[str] = Field(default_factory=list)
    operational_infrastructure: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return (
        "From the answer, extract the single primary venue being recommended for hosting the professional esports "
        "tournament. If multiple venues are mentioned, choose the first explicitly recommended venue.\n"
        "Return the following fields:\n"
        "1. venue_name: The full official name of the venue.\n"
        "2. city: The city where the venue is located.\n"
        "3. state: The US state where the venue is located (use the two-letter abbreviation or full state name present in the answer).\n"
        "If any field is missing, set it to null."
    )


def prompt_extract_refs_by_category() -> str:
    return (
        "Extract the reference URLs explicitly provided in the answer, grouped into the following categories. "
        "These URLs should directly support the stated requirements for the selected venue.\n"
        "Return arrays of URLs for each category:\n"
        "- capacity_and_space: URLs supporting seating capacity and the presence of a dedicated stage.\n"
        "- location: URLs confirming the venue's location (city/state) and indicating it is in a major US metropolitan area.\n"
        "- accessibility_and_safety: URLs supporting ADA wheelchair-accessible seating, emergency exits & fire safety, medical support, and restroom facilities.\n"
        "- technical_infrastructure: URLs supporting high-speed internet connectivity, production/broadcast control room, and production lighting.\n"
        "- amenities: URLs supporting VIP/premium seating areas, food service (concessions/food court), and player lounge/green room.\n"
        "- operational_infrastructure: URLs supporting loading dock, parking capacity, backup power/generator, and HVAC/climate control.\n\n"
        "Rules:\n"
        "- Extract only URLs explicitly present in the answer (plain URLs or markdown links).\n"
        "- Do not invent URLs. If none are provided for a category, return an empty list for that category.\n"
        "- Include full URLs. If the protocol is missing, prepend http://.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(x: Optional[str]) -> str:
    return x or ""

def _city_state_str(city: Optional[str], state: Optional[str]) -> str:
    city_s = _safe_str(city).strip()
    state_s = _safe_str(state).strip()
    if city_s and state_s:
        return f"{city_s}, {state_s}"
    return city_s or state_s or ""


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def add_reference_category_nodes(
    evaluator: Evaluator,
    parent_node,
    refs: VenueRefsExtraction,
) -> None:
    """
    Build 'Reference_URLs_By_Category' node with existence checks for each category.
    All children are critical existence checks to gate downstream verifications.
    """
    refs_root = evaluator.add_parallel(
        id="Reference_URLs_By_Category",
        desc="Provides reference URLs that verify each major requirement category from the question.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(refs.capacity_and_space),
        id="Refs_Capacity_And_Space",
        desc="Includes reference URL(s) supporting the Capacity & Space requirements (e.g., seating capacity and stage).",
        parent=refs_root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(refs.location),
        id="Refs_Location",
        desc="Includes reference URL(s) supporting the Location requirement (US major metro).",
        parent=refs_root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(refs.accessibility_and_safety),
        id="Refs_Accessibility_And_Safety",
        desc="Includes reference URL(s) supporting Accessibility & Safety requirements (ADA seating, exits/fire safety, medical, restrooms).",
        parent=refs_root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(refs.technical_infrastructure),
        id="Refs_Technical_Infrastructure",
        desc="Includes reference URL(s) supporting Technical Infrastructure requirements (internet, broadcast/control room, lighting).",
        parent=refs_root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(refs.amenities),
        id="Refs_Amenities",
        desc="Includes reference URL(s) supporting Amenity requirements (VIP/premium areas, food service, player lounge/green room).",
        parent=refs_root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(refs.operational_infrastructure),
        id="Refs_Operational_Infrastructure",
        desc="Includes reference URL(s) supporting Operational Infrastructure requirements (loading dock, parking, backup power, HVAC).",
        parent=refs_root,
        critical=True,
    )


async def add_basic_venue_nodes(
    evaluator: Evaluator,
    parent_node,
    venue: VenueInfoExtraction,
) -> None:
    """
    Build basic identification nodes: Venue_Name and Venue_Location_City_State.
    Both are critical leaf verifications against the answer text.
    """
    # Venue Name (critical leaf)
    name_leaf = evaluator.add_leaf(
        id="Venue_Name",
        desc="Provides the full name of the venue.",
        parent=parent_node,
        critical=True,
    )
    name_claim = (
        f"The answer provides the venue full name as '{_safe_str(venue.venue_name)}'. "
        f"Treat this as correct if the answer clearly names the venue and the extracted string matches."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction=(
            "Confirm the answer explicitly mentions this venue name. "
            "Allow minor variations in letter casing, abbreviations, or punctuation."
        ),
    )

    # Venue Location City/State (critical leaf)
    loc_leaf = evaluator.add_leaf(
        id="Venue_Location_City_State",
        desc="Provides the venue location as city and state (within the United States).",
        parent=parent_node,
        critical=True,
    )
    city_state = _city_state_str(venue.city, venue.state)
    loc_claim = (
        f"The answer provides the venue location as '{city_state}', which is in the United States."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        additional_instruction=(
            "Pass if the answer gives both a city and a US state for the venue. "
            "Minor formatting variations are acceptable."
        ),
    )


async def add_constraint_compliance_nodes(
    evaluator: Evaluator,
    parent_node,
    venue: VenueInfoExtraction,
    refs: VenueRefsExtraction,
) -> None:
    """
    Build 'Constraint_Compliance' node and all requirement checks as critical leaves,
    verified by the provided category URLs (verify_by_urls).
    """
    venue_label = _safe_str(venue.venue_name) or "the venue"
    city_state = _city_state_str(venue.city, venue.state)

    comp_root = evaluator.add_parallel(
        id="Constraint_Compliance",
        desc="Venue satisfies all stated venue requirements (capacity, infrastructure, safety, amenities, operations).",
        parent=parent_node,
        critical=True,
    )

    # Capacity & Space
    seat_leaf = evaluator.add_leaf(
        id="Seating_Capacity",
        desc="Venue provides seating capacity for at least 1,000 spectators.",
        parent=comp_root,
        critical=True,
    )
    seat_claim = (
        f"{venue_label} has seating capacity of at least 1,000 spectators."
    )
    await evaluator.verify(
        claim=seat_claim,
        node=seat_leaf,
        sources=refs.capacity_and_space,
        additional_instruction=(
            "Verify on the cited pages that the seating capacity is >= 1,000. "
            "Look for phrases like 'capacity', 'seats', 'attendance'. "
            "Numbers such as 1,000+ or 2,000 are acceptable."
        ),
    )

    stage_leaf = evaluator.add_leaf(
        id="Competition_Stage",
        desc="Venue has a dedicated stage area suitable for esports competition.",
        parent=comp_root,
        critical=True,
    )
    stage_claim = (
        f"{venue_label} has a dedicated stage suitable for esports competition."
    )
    await evaluator.verify(
        claim=stage_claim,
        node=stage_leaf,
        sources=refs.capacity_and_space,
        additional_instruction=(
            "Confirm that the venue has a 'stage', 'competition stage', or "
            "'performance stage' space appropriate for hosted events."
        ),
    )

    # Location - Major US metro
    metro_leaf = evaluator.add_leaf(
        id="US_Major_Metro_Location",
        desc="Venue is located in a major metropolitan area within the United States.",
        parent=comp_root,
        critical=True,
    )
    metro_claim = (
        f"{venue_label} is located in {city_state}, which is a major metropolitan area in the United States."
    )
    await evaluator.verify(
        claim=metro_claim,
        node=metro_leaf,
        sources=refs.location,
        additional_instruction=(
            "Use the provided sources to confirm the city/state and that it's widely recognized as a major metro. "
            "Well-known large cities (e.g., Los Angeles, New York, Chicago, Dallas, Houston, Las Vegas) count. "
            "If the city is part of a large metro area by common understanding, consider it acceptable."
        ),
    )

    # Accessibility & Safety
    ada_leaf = evaluator.add_leaf(
        id="ADA_Wheelchair_Seating",
        desc="Venue provides wheelchair-accessible seating meeting ADA requirements.",
        parent=comp_root,
        critical=True,
    )
    ada_claim = (
        f"{venue_label} provides wheelchair-accessible seating compliant with ADA requirements."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=refs.accessibility_and_safety,
        additional_instruction=(
            "Look for 'ADA compliant', 'accessible seating', 'wheelchair seating', 'ADA accessibility'."
        ),
    )

    fire_leaf = evaluator.add_leaf(
        id="Emergency_Exits_Fire_Safety",
        desc="Venue has adequate emergency exits and fire safety compliance.",
        parent=comp_root,
        critical=True,
    )
    fire_claim = (
        f"{venue_label} has adequate emergency exits and is compliant with fire safety requirements."
    )
    await evaluator.verify(
        claim=fire_claim,
        node=fire_leaf,
        sources=refs.accessibility_and_safety,
        additional_instruction=(
            "Look for mentions of 'emergency exits', 'fire code compliance', 'sprinkler systems', "
            "'evacuation plans'. Passing if the page indicates compliance or suitable facilities."
        ),
    )

    medical_leaf = evaluator.add_leaf(
        id="Medical_Support",
        desc="Venue has emergency medical facilities or medical support capabilities.",
        parent=comp_root,
        critical=True,
    )
    medical_claim = (
        f"{venue_label} has emergency medical facilities or onsite medical support (e.g., first aid room, medics)."
    )
    await evaluator.verify(
        claim=medical_claim,
        node=medical_leaf,
        sources=refs.accessibility_and_safety,
        additional_instruction=(
            "Look for 'first aid', 'medical staff', 'AED devices', 'onsite medics', or 'medical support'."
        ),
    )

    restroom_leaf = evaluator.add_leaf(
        id="Restroom_Facilities",
        desc="Venue has adequate restroom facilities based on venue capacity.",
        parent=comp_root,
        critical=True,
    )
    restroom_claim = (
        f"{venue_label} has adequate restroom facilities appropriate for its capacity."
    )
    await evaluator.verify(
        claim=restroom_claim,
        node=restroom_leaf,
        sources=refs.accessibility_and_safety,
        additional_instruction=(
            "Confirm the presence of ample restrooms or facilities for large audiences; "
            "phrases like 'ample restrooms' or capacity-scale facilities suffice."
        ),
    )

    # Technical Infrastructure
    internet_leaf = evaluator.add_leaf(
        id="High_Speed_Internet",
        desc="Venue has high-speed internet connectivity sufficient for multiple gaming stations and streaming.",
        parent=comp_root,
        critical=True,
    )
    internet_claim = (
        f"{venue_label} has high-speed internet sufficient for multiple gaming stations and live streaming "
        "(e.g., fiber, gigabit, dedicated bandwidth)."
    )
    await evaluator.verify(
        claim=internet_claim,
        node=internet_leaf,
        sources=refs.technical_infrastructure,
        additional_instruction=(
            "Look for terms like 'fiber', 'gigabit', 'dedicated internet', 'high-speed bandwidth', 'LAN'."
        ),
    )

    control_leaf = evaluator.add_leaf(
        id="Broadcast_Control_Room",
        desc="Venue has production/broadcast control room facilities.",
        parent=comp_root,
        critical=True,
    )
    control_claim = (
        f"{venue_label} has production or broadcast control room facilities (e.g., control room, production booth)."
    )
    await evaluator.verify(
        claim=control_claim,
        node=control_leaf,
        sources=refs.technical_infrastructure,
        additional_instruction=(
            "Confirm mentions of 'control room', 'broadcast booth', 'production room', or similar facilities."
        ),
    )

    lighting_leaf = evaluator.add_leaf(
        id="Production_Lighting",
        desc="Venue has adequate lighting systems for production purposes.",
        parent=comp_root,
        critical=True,
    )
    lighting_claim = (
        f"{venue_label} has adequate production lighting systems suitable for broadcast or stage events."
    )
    await evaluator.verify(
        claim=lighting_claim,
        node=lighting_leaf,
        sources=refs.technical_infrastructure,
        additional_instruction=(
            "Look for 'stage lighting', 'production lighting', 'LED rigs', 'professional lighting systems'."
        ),
    )

    # Amenities
    vip_leaf = evaluator.add_leaf(
        id="VIP_Premium_Areas",
        desc="Venue has VIP lounge or premium seating areas.",
        parent=comp_root,
        critical=True,
    )
    vip_claim = (
        f"{venue_label} has VIP lounges or premium seating areas."
    )
    await evaluator.verify(
        claim=vip_claim,
        node=vip_leaf,
        sources=refs.amenities,
        additional_instruction=(
            "Look for 'VIP lounge', 'premium seating', 'club level', 'suite', or similar premium offerings."
        ),
    )

    food_leaf = evaluator.add_leaf(
        id="Food_Service",
        desc="Venue has food service facilities (food court or concessions).",
        parent=comp_root,
        critical=True,
    )
    food_claim = (
        f"{venue_label} provides food service facilities such as concessions or a food court."
    )
    await evaluator.verify(
        claim=food_claim,
        node=food_leaf,
        sources=refs.amenities,
        additional_instruction=(
            "Look for 'concessions', 'food court', 'restaurants', or onsite food service listings."
        ),
    )

    player_leaf = evaluator.add_leaf(
        id="Player_Lounge_Green_Room",
        desc="Venue has player lounge or green room facilities for competitors.",
        parent=comp_root,
        critical=True,
    )
    player_claim = (
        f"{venue_label} has a player lounge or green room facilities for competitors."
    )
    await evaluator.verify(
        claim=player_claim,
        node=player_leaf,
        sources=refs.amenities,
        additional_instruction=(
            "Confirm dedicated competitor areas such as 'player lounge', 'green room', 'backstage', "
            "or 'team rooms'."
        ),
    )

    # Operational Infrastructure
    dock_leaf = evaluator.add_leaf(
        id="Loading_Dock",
        desc="Venue has loading dock for equipment load-in and setup.",
        parent=comp_root,
        critical=True,
    )
    dock_claim = (
        f"{venue_label} has a loading dock suitable for equipment load-in and setup."
    )
    await evaluator.verify(
        claim=dock_claim,
        node=dock_leaf,
        sources=refs.operational_infrastructure,
        additional_instruction=(
            "Look for 'loading dock', 'freight access', 'load-in', 'back-of-house access'."
        ),
    )

    parking_leaf = evaluator.add_leaf(
        id="Parking_Capacity",
        desc="Venue has adequate parking capacity for tournament attendees.",
        parent=comp_root,
        critical=True,
    )
    parking_claim = (
        f"{venue_label} has adequate parking capacity for attendees (on-site lots or nearby garages)."
    )
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=refs.operational_infrastructure,
        additional_instruction=(
            "Confirm availability of large parking lots or garages and suitability for event-scale crowds."
        ),
    )

    backup_leaf = evaluator.add_leaf(
        id="Backup_Power",
        desc="Venue has backup power or generator systems.",
        parent=comp_root,
        critical=True,
    )
    backup_claim = (
        f"{venue_label} has backup power or generator systems."
    )
    await evaluator.verify(
        claim=backup_claim,
        node=backup_leaf,
        sources=refs.operational_infrastructure,
        additional_instruction=(
            "Look for 'backup generator', 'UPS', 'redundant power', or similar mentions."
        ),
    )

    hvac_leaf = evaluator.add_leaf(
        id="HVAC_Climate_Control",
        desc="Venue has HVAC/climate control systems.",
        parent=comp_root,
        critical=True,
    )
    hvac_claim = (
        f"{venue_label} has HVAC or climate control systems suitable for large events."
    )
    await evaluator.verify(
        claim=hvac_claim,
        node=hvac_leaf,
        sources=refs.operational_infrastructure,
        additional_instruction=(
            "Look for 'HVAC', 'air conditioning', 'climate control', or equivalent facilities."
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
    Evaluate an answer for the US esports venue requirements task.
    """
    # Initialize evaluator
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

    # Extract venue info and category references
    venue_info, refs_info = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_venue_info(),
            template_class=VenueInfoExtraction,
            extraction_name="venue_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_refs_by_category(),
            template_class=VenueRefsExtraction,
            extraction_name="category_refs",
        ),
    )

    # Build main critical node (Venue_Response)
    venue_response = evaluator.add_parallel(
        id="Venue_Response",
        desc="Answer identifies a US esports venue and provides required details, meeting all stated constraints and including verification references.",
        parent=root,
        critical=True,
    )

    # Add basic venue identification nodes (critical leaves)
    await add_basic_venue_nodes(evaluator, venue_response, venue_info)

    # Add reference existence nodes by category (critical gating)
    await add_reference_category_nodes(evaluator, venue_response, refs_info)

    # Add constraint compliance checks (all critical leaves)
    await add_constraint_compliance_nodes(evaluator, venue_response, venue_info, refs_info)

    # Record summary info
    evaluator.add_custom_info(
        {
            "venue": {
                "name": venue_info.venue_name,
                "city": venue_info.city,
                "state": venue_info.state,
            },
            "refs": {
                "capacity_and_space": refs_info.capacity_and_space,
                "location": refs_info.location,
                "accessibility_and_safety": refs_info.accessibility_and_safety,
                "technical_infrastructure": refs_info.technical_infrastructure,
                "amenities": refs_info.amenities,
                "operational_infrastructure": refs_info.operational_infrastructure,
            },
        },
        info_type="extracted_summary",
        info_name="extracted_venue_and_refs",
    )

    # Return structured result using the evaluator's summary
    return evaluator.get_summary()