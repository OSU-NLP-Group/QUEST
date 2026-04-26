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
TASK_ID = "port_canaveral_hotel_selection"
TASK_DESCRIPTION = """
A family is planning a 7-day cruise departing from Port Canaveral, Florida in April 2026. They are driving to the port and need to find a hotel for the night before their cruise that meets the following requirements: 
(1) The hotel must offer a park-and-cruise package that includes one night's accommodation and parking for at least 7 days during their cruise; 
(2) The hotel must provide complimentary shuttle service to Port Canaveral cruise terminal; 
(3) The hotel must accept dogs with a weight limit of at least 70 pounds (their dog weighs 68 pounds); 
(4) The hotel must have ADA-compliant accessible rooms with roll-in showers available, as one family member uses a wheelchair; 
(5) The hotel must be located within 5 miles of Port Canaveral cruise terminal; 
(6) Ideally, the hotel should offer covered parking facilities (non-critical); 
(7) Ideally, the hotel should provide complimentary shuttle service to Orlando International Airport or a nearby airport (non-critical). 
Identify a hotel that meets all the critical requirements listed above, and provide its name, distance from Port Canaveral, and a reference URL confirming these features.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    # Core provided info
    name: Optional[str] = None
    distance_to_port_text: Optional[str] = None  # e.g., "3.2 miles", "about 4 miles"
    distance_to_port_miles: Optional[str] = None  # numeric string if present (e.g., "3.2"), else null
    official_site_url: Optional[str] = None

    # Feature-specific sources (URLs explicitly cited in the answer)
    park_and_cruise_urls: List[str] = Field(default_factory=list)
    cruise_terminal_shuttle_urls: List[str] = Field(default_factory=list)
    pet_policy_urls: List[str] = Field(default_factory=list)
    ada_roll_in_shower_urls: List[str] = Field(default_factory=list)
    proximity_urls: List[str] = Field(default_factory=list)
    covered_parking_urls: List[str] = Field(default_factory=list)
    airport_shuttle_urls: List[str] = Field(default_factory=list)

    # Any additional or general references provided
    general_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
Extract exactly one primary hotel candidate recommended in the answer for the family's pre-cruise stay near Port Canaveral. 
If multiple hotels are mentioned, select the first recommended hotel. 
Extract the following fields:

1) name: The hotel's official name as written in the answer.
2) distance_to_port_text: The stated distance from Port Canaveral (keep the original text, e.g., "3.2 miles", "about 4 miles"). If not provided, set to null.
3) distance_to_port_miles: If the distance is provided, extract only the numeric miles value (e.g., "3.2" for "3.2 miles"). If not provided or unclear, set to null.
4) official_site_url: If an official hotel website URL is cited, extract it; otherwise null.

For the URLs below, extract only the actual URLs explicitly present in the answer text (plain or markdown). 
Categorize them into the following arrays. If none for a category, return an empty array:
- park_and_cruise_urls: URLs that specifically mention a "park and cruise", "stay and cruise", or "snooze and cruise" package including parking during the cruise.
- cruise_terminal_shuttle_urls: URLs that specifically mention shuttle service to Port Canaveral cruise terminals (note if complimentary/free).
- pet_policy_urls: URLs describing the hotel's pet policy (including any weight limits).
- ada_roll_in_shower_urls: URLs that specifically mention ADA-compliant accessible rooms with roll-in showers.
- proximity_urls: URLs that mention or allow confirming the distance or proximity to Port Canaveral (e.g., location/distance pages or maps).
- covered_parking_urls: URLs that mention covered parking, garage parking, or roofed parking (if any).
- airport_shuttle_urls: URLs that mention airport shuttle service to Orlando International Airport (MCO) or nearby airports (e.g., SFB, MLB), ideally complimentary (if any).
- general_reference_urls: Any other reference URLs in the answer that support the hotel's features or details but do not neatly fit the above categories.

Rules:
- Do NOT invent URLs. Only extract URLs explicitly present in the answer.
- Normalize markdown links to just their URL.
- If the answer provides a single combined reference link for all features, include it in general_reference_urls (and also in a specific category if it clearly supports that category).
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _union_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str) and u.strip() and u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _choose_sources(preferred: List[str], fallback: List[str]) -> List[str] | None:
    if preferred and len(preferred) > 0:
        return preferred
    if fallback and len(fallback) > 0:
        return fallback
    return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_hotel_selection(evaluator: Evaluator, parent_node, hotel: HotelInfo) -> None:
    """
    Build the verification tree under a single 'Hotel_Selection' node and run verifications.
    """
    # Container node (non-critical to allow non-critical leaves inside)
    hotel_node = evaluator.add_parallel(
        id="Hotel_Selection",
        desc="Identifies a hotel near Port Canaveral that meets all specified requirements for a family's pre-cruise accommodation needs and provides the required information",
        parent=parent_node,
        critical=False
    )

    # Collect all URLs (used for existence check and fallback)
    all_urls = _union_urls(
        hotel.park_and_cruise_urls,
        hotel.cruise_terminal_shuttle_urls,
        hotel.pet_policy_urls,
        hotel.ada_roll_in_shower_urls,
        hotel.proximity_urls,
        hotel.covered_parking_urls,
        hotel.airport_shuttle_urls,
        hotel.general_reference_urls,
        [hotel.official_site_url] if hotel.official_site_url else []
    )

    # Basic presence checks (critical)
    name_exists_node = evaluator.add_custom_node(
        result=bool(hotel.name and hotel.name.strip()),
        id="Hotel_Name_Provided",
        desc="The answer provides the name of the hotel",
        parent=hotel_node,
        critical=True
    )

    distance_exists_node = evaluator.add_custom_node(
        result=bool(hotel.distance_to_port_text and hotel.distance_to_port_text.strip()),
        id="Distance_Information_Provided",
        desc="The answer provides the distance from Port Canaveral to the hotel",
        parent=hotel_node,
        critical=True
    )

    ref_urls_exists_node = evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="Reference_URL_Provided",
        desc="The answer provides reference URL(s) confirming the hotel's features",
        parent=hotel_node,
        critical=True
    )

    # Prepare a readable hotel name for claims
    hotel_name = hotel.name if (hotel.name and hotel.name.strip()) else "the recommended hotel"

    # Feature verifications (critical ones first)
    # 1) Park & Cruise package: includes 1 night and parking for at least 7 days (duration of cruise)
    park_node = evaluator.add_leaf(
        id="Park_And_Cruise_Package",
        desc="Hotel offers a park-and-cruise package that includes one night accommodation and parking for the duration of a cruise (at least 7 days)",
        parent=hotel_node,
        critical=True
    )
    park_sources = _choose_sources(hotel.park_and_cruise_urls, all_urls)
    park_claim = f"The hotel '{hotel_name}' offers a 'park and cruise' (or 'stay and cruise' / 'snooze and cruise') package that includes 1 night's accommodation AND parking for at least 7 days (i.e., for the full duration of a typical 7-day cruise)."
    await evaluator.verify(
        claim=park_claim,
        node=park_node,
        sources=park_sources,
        additional_instruction="Treat synonymous package names (e.g., Park & Cruise, Stay & Cruise, Snooze & Cruise) as equivalent. Parking language like 'parking for the duration of your cruise' or 'up to 7 nights' satisfies the ≥7 days requirement. The page must clearly indicate parking during the cruise, not just overnight.",
        extra_prerequisites=[ref_urls_exists_node]
    )

    # 2) Complimentary shuttle to Port Canaveral cruise terminal
    port_shuttle_node = evaluator.add_leaf(
        id="Cruise_Terminal_Shuttle",
        desc="Hotel provides complimentary shuttle service to Port Canaveral cruise terminal",
        parent=hotel_node,
        critical=True
    )
    port_shuttle_sources = _choose_sources(hotel.cruise_terminal_shuttle_urls, all_urls)
    port_shuttle_claim = f"The hotel '{hotel_name}' provides complimentary (free) shuttle service to Port Canaveral cruise terminals."
    await evaluator.verify(
        claim=port_shuttle_claim,
        node=port_shuttle_node,
        sources=port_shuttle_sources,
        additional_instruction="Confirm the shuttle specifically serves Port Canaveral cruise terminals and is complimentary/free. Mentions of 'paid shuttle' do NOT satisfy this requirement.",
        extra_prerequisites=[ref_urls_exists_node]
    )

    # 3) Pet policy allows at least 70 lb
    pet_node = evaluator.add_leaf(
        id="Pet_Policy_Compliance",
        desc="Hotel accepts dogs with a weight limit of at least 70 pounds",
        parent=hotel_node,
        critical=True
    )
    pet_sources = _choose_sources(hotel.pet_policy_urls, all_urls)
    pet_claim = f"The pet policy of '{hotel_name}' allows dogs weighing at least 70 pounds (a 68-pound dog is acceptable)."
    await evaluator.verify(
        claim=pet_claim,
        node=pet_node,
        sources=pet_sources,
        additional_instruction="Acceptable evidence includes an explicit weight limit ≥ 70 lb, or statements like 'no weight limit.' If policy states a lower limit (e.g., 50 lb), it fails.",
        extra_prerequisites=[ref_urls_exists_node]
    )

    # 4) ADA rooms with roll-in showers
    ada_node = evaluator.add_leaf(
        id="ADA_Accessible_Rooms",
        desc="Hotel has ADA-compliant accessible rooms with roll-in showers available for guests with mobility disabilities",
        parent=hotel_node,
        critical=True
    )
    ada_sources = _choose_sources(hotel.ada_roll_in_shower_urls, all_urls)
    ada_claim = f"The hotel '{hotel_name}' offers ADA-compliant accessible rooms that include roll-in showers (available to book)."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_node,
        sources=ada_sources,
        additional_instruction="The source must explicitly reference 'roll-in shower' in accessible/ADA rooms. Generic 'accessible bathroom' without 'roll-in shower' is insufficient.",
        extra_prerequisites=[ref_urls_exists_node]
    )

    # 5) Within 5 miles of Port Canaveral cruise terminal
    proximity_node = evaluator.add_leaf(
        id="Proximity_To_Port",
        desc="Hotel is located within 5 miles of Port Canaveral cruise terminal",
        parent=hotel_node,
        critical=True
    )
    proximity_sources = _choose_sources(hotel.proximity_urls, all_urls)
    proximity_claim = f"The hotel '{hotel_name}' is within 5 miles of the Port Canaveral cruise terminal."
    await evaluator.verify(
        claim=proximity_claim,
        node=proximity_node,
        sources=proximity_sources,
        additional_instruction="Accept approximations that clearly indicate ≤ 5 miles to Port Canaveral terminals. If only driving time is given, convert reasonably (e.g., ~10 min on local roads can imply ~3–4 miles). If the page shows > 5 miles, it fails.",
        extra_prerequisites=[ref_urls_exists_node]
    )

    # Non-critical preferences
    covered_node = evaluator.add_leaf(
        id="Covered_Parking_Availability",
        desc="Hotel offers covered parking facilities for guest vehicles",
        parent=hotel_node,
        critical=False
    )
    covered_sources = _choose_sources(hotel.covered_parking_urls, all_urls)
    covered_claim = f"The hotel '{hotel_name}' offers covered parking (e.g., garage, roofed or under-cover parking)."
    await evaluator.verify(
        claim=covered_claim,
        node=covered_node,
        sources=covered_sources,
        additional_instruction="Look for terms like 'covered parking,' 'garage,' 'under cover.' Open surface lots do NOT satisfy.",
        extra_prerequisites=[ref_urls_exists_node]
    )

    airport_node = evaluator.add_leaf(
        id="Airport_Shuttle_Service",
        desc="Hotel provides complimentary shuttle service to Orlando International Airport or nearby regional airport",
        parent=hotel_node,
        critical=False
    )
    airport_sources = _choose_sources(hotel.airport_shuttle_urls, all_urls)
    airport_claim = f"The hotel '{hotel_name}' provides complimentary shuttle service to Orlando International Airport (MCO) or a nearby regional airport (e.g., Orlando Sanford SFB or Melbourne MLB)."
    await evaluator.verify(
        claim=airport_claim,
        node=airport_node,
        sources=airport_sources,
        additional_instruction="Confirm that the service is complimentary/free and explicitly airport-related (MCO, SFB, MLB or similar). If paid-only, do not pass.",
        extra_prerequisites=[ref_urls_exists_node]
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
    Evaluate an answer for the Port Canaveral pre-cruise hotel selection task.
    """
    # Initialize evaluator (root as non-critical to allow non-critical leaves within)
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

    # Extract one hotel candidate and associated URLs from the answer
    hotel_info: HotelInfo = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelInfo,
        extraction_name="hotel_extraction"
    )

    # Build verification tree and run checks
    await verify_hotel_selection(evaluator, root, hotel_info)

    # Return structured summary
    return evaluator.get_summary()