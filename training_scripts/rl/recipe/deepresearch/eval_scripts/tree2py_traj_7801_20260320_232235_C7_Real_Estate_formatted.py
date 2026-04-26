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
TASK_ID = "phoenix_office_space_requirements"
TASK_DESCRIPTION = (
    "I am looking for available office space in the greater Phoenix, Arizona metropolitan area that meets specific "
    "requirements for my company. The space must be between 15,000 and 50,000 square feet within a building that has "
    "LEED certification at Silver level or higher, OR Energy Star certification. The building must provide on-site "
    "parking with a ratio of at least 3 parking spaces per 1,000 square feet of office space, and the parking "
    "facilities must include ADA-compliant accessible parking spaces (at least 96 inches wide with a 60-inch access "
    "aisle). If the building has 3 or more stories, it must have at least one elevator. The building must include at "
    "least one conference room capable of accommodating 10-15 people, and all restroom facilities must be "
    "ADA-compliant with doors at least 32 inches wide. Required common areas include a lobby or reception area and a "
    "break room or kitchen facility. The building must have a modern HVAC system, accessible entrances with ramps or "
    "level access for wheelchair users, and at least two emergency exits. The space must be currently available for "
    "lease. Please identify one such office space in Phoenix, provide the complete physical address, and a direct URL "
    "to the property listing."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class OfficeSpaceExtraction(BaseModel):
    # Identification and address
    property_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    # Sources
    listing_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)

    # Requirements (free-text as extracted from answer)
    size_sqft: Optional[str] = None                         # e.g., "20,000 SF", "15,000 - 25,000 SF"
    leed_cert_level: Optional[str] = None                   # e.g., "LEED Silver", "LEED Gold"
    energy_star_certified: Optional[str] = None             # e.g., "Energy Star certified", "ENERGY STAR"
    parking_ratio: Optional[str] = None                     # e.g., "4/1,000", "3.5 per 1,000"
    ada_accessible_parking_details: Optional[str] = None    # try to capture widths/aisle text if present

    building_stories: Optional[str] = None                  # e.g., "3", "Two-story"
    elevator_present: Optional[str] = None                  # e.g., "Elevator", "2 elevators"

    conference_room_capacity: Optional[str] = None          # e.g., "12-person"
    conference_room_sqft: Optional[str] = None              # e.g., "300 SF"

    ada_restroom_doors_width: Optional[str] = None          # e.g., "32 inches"
    ada_restroom_accessible_stall_width: Optional[str] = None  # e.g., "60 inches"

    lobby_or_reception_present: Optional[str] = None        # e.g., "Reception", "Lobby"
    break_room_or_kitchen_present: Optional[str] = None     # e.g., "Break room", "Kitchen"

    hvac_modern: Optional[str] = None                       # e.g., "modern HVAC", "new HVAC"
    ventilation_cfm_per_person: Optional[str] = None        # e.g., "8 CFM/person", "ASHRAE 62.1 compliant"

    accessible_entrances_detail: Optional[str] = None       # e.g., "ADA ramp", "level entry"
    emergency_exits_count_or_statement: Optional[str] = None  # e.g., "two exits", "multiple exits"

    currently_available: Optional[str] = None               # e.g., "Available", "Now leasing"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_office_space() -> str:
    return """
Extract from the answer details about a single identified office space. Do NOT invent anything; return null when not stated. 
Return a JSON with the following fields (strings are preferred; keep numbers as they appear in text, including units):
- property_name: The building or property name, if provided.
- address: Street number and street name (e.g., "123 Main St").
- city: City name.
- state: State abbreviation or name (e.g., "AZ").
- zip_code: ZIP code.
- listing_url: The direct URL to the property listing page (NOT a general search results page or a site homepage). If multiple URLs appear, pick the most direct listing page.
- additional_urls: Any other URLs cited in the answer relevant to this property (exclude duplicates of listing_url).

Requirements-related fields (free text as present in the answer):
- size_sqft
- leed_cert_level
- energy_star_certified
- parking_ratio
- ada_accessible_parking_details
- building_stories
- elevator_present
- conference_room_capacity
- conference_room_sqft
- ada_restroom_doors_width
- ada_restroom_accessible_stall_width
- lobby_or_reception_present
- break_room_or_kitchen_present
- hvac_modern
- ventilation_cfm_per_person
- accessible_entrances_detail
- emergency_exits_count_or_statement
- currently_available

Rules:
- Extract values exactly as written in the answer. If the answer provides a capacity like "12-person", put that string.
- If a URL appears without protocol, prepend "http://".
- If any field is not present, return null or an empty list accordingly.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def collect_sources(extracted: OfficeSpaceExtraction) -> List[str]:
    urls: List[str] = []
    if extracted and extracted.listing_url:
        urls.append(extracted.listing_url)
    if extracted and extracted.additional_urls:
        for u in extracted.additional_urls:
            if u and (u not in urls):
                urls.append(u)
    return urls


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def verify_office_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: OfficeSpaceExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run URL-grounded checks.
    Note: The rubric designates one non-critical item ("Conference_Room_Size_Typical_Range").
    The parent node here is non-critical to satisfy framework constraints (critical parents cannot have non-critical children).
    All mandatory checks are set as critical leaves.
    """
    office_node = evaluator.add_parallel(
        id="Office_Space_Requirements",
        desc="Evaluate whether the identified office space and the provided output satisfy all stated requirements from the question and constraints.",
        parent=parent_node,
        critical=False  # Parent must be non-critical because one child is non-critical
    )

    sources = collect_sources(extracted)

    # 0) Direct listing URL must be provided (critical) - first to gate others via extra prerequisites
    if extracted and extracted.listing_url:
        direct_url_node = evaluator.add_leaf(
            id="Direct_Property_Listing_URL_Provided",
            desc="The answer provides a direct URL to the property listing.",
            parent=office_node,
            critical=True
        )
        await evaluator.verify(
            claim="This URL is a direct property listing page for a specific property or suite (not a generic search results page or a site homepage).",
            node=direct_url_node,
            sources=extracted.listing_url,
            additional_instruction="Verify the page is a specific property/suite listing page with property details, not a homepage or search results."
        )
    else:
        direct_url_node = evaluator.add_custom_node(
            result=False,
            id="Direct_Property_Listing_URL_Provided",
            desc="The answer provides a direct URL to the property listing.",
            parent=office_node,
            critical=True
        )

    # 1) Complete physical address provided (critical)
    complete_addr_ok = bool(extracted and extracted.address and extracted.city and extracted.state and extracted.zip_code)
    if complete_addr_ok:
        addr_leaf = evaluator.add_leaf(
            id="Complete_Physical_Address_Provided",
            desc="The answer provides the complete physical address of the identified office space/building.",
            parent=office_node,
            critical=True
        )
        pretty_addr = f"{extracted.address}, {extracted.city}, {extracted.state} {extracted.zip_code}"
        await evaluator.verify(
            claim=f"The answer provides the complete physical address for the property: {pretty_addr}. A complete address includes street number and name, city, state, and ZIP code.",
            node=addr_leaf,
            additional_instruction="Judge based on the provided answer text only. All components must be present."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Complete_Physical_Address_Provided",
            desc="The answer provides the complete physical address of the identified office space/building.",
            parent=office_node,
            critical=True
        )

    # Helper for URL-grounded checks (most leaves)
    async def url_check(node_id: str, desc: str, claim: str, critical: bool = True, add_ins: str = ""):
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=office_node,
            critical=critical
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=sources,
            additional_instruction=add_ins,
            extra_prerequisites=[direct_url_node]
        )

    # 2) Location in greater Phoenix metro (critical)
    await url_check(
        "Location_Phoenix_Metro",
        "The identified office space is located in the greater Phoenix, Arizona metropolitan area.",
        "This property is located in the greater Phoenix, Arizona metropolitan area (acceptable cities include Phoenix, Scottsdale, Tempe, Mesa, Chandler, Gilbert, Glendale, Peoria, Surprise, Avondale, Goodyear, Buckeye, and Paradise Valley).",
        critical=True,
        add_ins="Verify using the address/location shown on the listing page."
    )

    # 3) Space size range 15,000–50,000 SF (critical)
    await url_check(
        "Space_Size_Range",
        "The available office space size is between 15,000 and 50,000 square feet (inclusive).",
        "The available office space being offered in this listing is between 15,000 and 50,000 square feet inclusive. "
        "If multiple suites are listed, at least one single, contiguous suite falls within this range.",
        critical=True,
        add_ins="Accept formats like 'sf' or 'sq ft'. If only a divisible/max range is shown (e.g., up to 60,000 with divisibility), require that a specific option within 15–50k is clearly offered."
    )

    # 4) Energy certification (critical): LEED Silver+ OR Energy Star
    await url_check(
        "Energy_Certification",
        "The building has LEED certification at Silver level or higher OR has Energy Star certification.",
        "The building has either LEED Silver (or Gold/Platinum) certification, or an ENERGY STAR certification.",
        critical=True,
        add_ins="Look for explicit mentions like 'LEED Silver/Gold/Platinum' or 'ENERGY STAR certified'. If absent or ambiguous, judge not supported."
    )

    # 5) Parking ratio >= 3 per 1,000 SF (critical)
    await url_check(
        "Parking_Ratio_Met",
        "On-site parking is provided at a ratio of at least 3 parking spaces per 1,000 square feet of office space.",
        "On-site parking ratio is at least 3.0 spaces per 1,000 square feet (>= 3/1,000).",
        critical=True,
        add_ins="Accept formats such as '3/1,000', '3 per 1,000', or higher numbers like '4/1,000'."
    )

    # 6) ADA accessible parking with required widths (critical)
    await url_check(
        "ADA_Accessible_Parking",
        "ADA-compliant accessible parking is present, with accessible spaces at least 96 inches wide and a 60-inch access aisle.",
        "The property offers ADA-compliant accessible parking spaces that are at least 96 inches wide with an adjacent access aisle at least 60 inches wide.",
        critical=True,
        add_ins="Require explicit support for the dimensions. If only 'ADA parking' is mentioned without the 96\" space and 60\" aisle widths, judge not supported."
    )

    # 7) Elevator if 3+ stories (critical)
    await url_check(
        "Elevator_If_3Plus_Stories",
        "If the building has 3 or more stories, it has at least one elevator (otherwise this requirement is not applicable).",
        "If the building has three or more stories, it has at least one elevator; if the building has fewer than three stories, this requirement is satisfied by default.",
        critical=True,
        add_ins="Use the listing to determine stories. If stories are >=3, elevator must be explicitly present. If stories <3, pass. If stories are unknown and no elevator is shown, judge not supported."
    )

    # 8) Conference room 10–15 people (critical)
    await url_check(
        "Conference_Room_10_15",
        "At least one conference room is available that can accommodate 10–15 people.",
        "The property includes at least one conference/meeting room that can accommodate between 10 and 15 people.",
        critical=True,
        add_ins="Accept synonyms like 'boardroom'. Seating count must be explicitly within 10–15; if capacity is not specified, judge not supported."
    )

    # 9) Conference room size typical range 250–350 SF (non-critical)
    await url_check(
        "Conference_Room_Size_Typical_Range",
        "If conference room square footage is provided, it is consistent with the stated typical range (250–350 square feet) for a 10–15 person room.",
        "If the conference room square footage is explicitly provided, it falls between 250 and 350 square feet. If no conference room square footage is provided, consider this not applicable and treat as supported.",
        critical=False,
        add_ins="If an explicit square footage is stated and it's outside 250–350 SF, judge not supported. If no square footage is stated, mark supported (N/A)."
    )

    # 10) ADA-compliant restrooms: 32-inch doors AND 60-inch accessible stall (critical)
    await url_check(
        "ADA_Restroom_Compliance_Door_And_Stall",
        "Restrooms are ADA-compliant with restroom doors at least 32 inches wide AND accessible stalls at least 60 inches wide.",
        "Restroom facilities are ADA-compliant, including restroom doors of at least 32 inches clear width and at least one accessible stall that is at least 60 inches wide.",
        critical=True,
        add_ins="Require explicit support for both the 32-inch door width and the 60-inch accessible stall width. If only 'ADA restroom' is stated without dimensions, judge not supported."
    )

    # 11) Lobby or reception (critical)
    await url_check(
        "Lobby_Or_Reception",
        "A lobby or reception/common entry area is present.",
        "The property includes a lobby or reception/common entry area.",
        critical=True,
        add_ins="Look for explicit mention of lobby, reception, or common entry area."
    )

    # 12) Break room or kitchen (critical)
    await url_check(
        "Break_Room_Or_Kitchen",
        "A break room or kitchen facility is available.",
        "The property includes a break room or a kitchen facility for occupants.",
        critical=True,
        add_ins="Accept 'break room', 'kitchen', 'pantry', or similar occupant kitchen amenity if clearly stated."
    )

    # 13) Modern HVAC and ventilation standard (5–10 CFM/person) (critical)
    await url_check(
        "HVAC_Modern_And_Ventilation_Standard",
        "A modern HVAC system is present AND the provided information indicates it meets the stated ventilation standard (5–10 CFM per person minimum).",
        "The building has a modern HVAC system and the information indicates a ventilation rate of at least 5 CFM per person (5–10 CFM/person standard).",
        critical=True,
        add_ins="Accept if the page explicitly mentions >=5 CFM/person or ASHRAE 62.1 compliance with typical office ventilation rates. If only 'modern HVAC' is stated without ventilation rate or standard, judge not supported."
    )

    # 14) Accessible entrances (critical)
    await url_check(
        "Accessible_Entrances",
        "Accessible entrances are provided via ramps or level access suitable for wheelchair users.",
        "Entrances are accessible to wheelchair users via ramps or level access.",
        critical=True,
        add_ins="Look for ADA-accessible entry, ramp access, or level threshold. If not explicitly supported, judge not supported."
    )

    # 15) Emergency exits at least two (critical)
    await url_check(
        "Emergency_Exits_Min_Two",
        "At least two emergency exits are present.",
        "The building provides at least two emergency exits (two means of egress).",
        critical=True,
        add_ins="Look for 'two exits', 'two means of egress', or explicit equivalent. If not stated, judge not supported."
    )

    # 16) Currently available for lease (critical)
    await url_check(
        "Currently_Available_For_Lease",
        "The space is currently available for lease at the time of the answer (as indicated by the provided source).",
        "The listing indicates the space is currently available for lease (e.g., 'Available', 'Now Leasing', 'For Lease', active availability).",
        critical=True,
        add_ins="Use the listing's status/availability indicators. If it shows 'Leased', 'Off-market', or no current availability, judge not supported."
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
    Evaluate an answer for the Phoenix office space requirements task.
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

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_office_space(),
        template_class=OfficeSpaceExtraction,
        extraction_name="office_space_extraction",
    )

    # 2) Add helpful custom info
    evaluator.add_custom_info(
        info={
            "property_name": extracted.property_name,
            "address": extracted.address,
            "city": extracted.city,
            "state": extracted.state,
            "zip_code": extracted.zip_code,
            "listing_url": extracted.listing_url,
            "additional_urls": extracted.additional_urls,
            "size_sqft": extracted.size_sqft,
            "leed_cert_level": extracted.leed_cert_level,
            "energy_star_certified": extracted.energy_star_certified,
            "parking_ratio": extracted.parking_ratio,
            "conference_room_capacity": extracted.conference_room_capacity,
        },
        info_type="extraction_summary",
        info_name="extracted_overview"
    )

    # 3) Build verification tree and run checks
    await verify_office_requirements(evaluator, root, extracted)

    # 4) Return structured result
    return evaluator.get_summary()