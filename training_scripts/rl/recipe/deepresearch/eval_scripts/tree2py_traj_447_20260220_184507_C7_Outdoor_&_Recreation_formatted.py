import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "southeast_airport_hub_amenities"
TASK_DESCRIPTION = (
    "Identify a major commercial airport in the Southeastern United States that serves as a hub for a major U.S. airline "
    "and provides all of the following passenger amenities and services: free WiFi throughout the terminal, charging stations "
    "in gate areas, hand sanitizer stations, dedicated mother's rooms, pet relief areas, currency exchange or ATM services, "
    "an interfaith chapel, an airport volunteer or passenger assistance program, accessibility services for passengers with disabilities, "
    "rental car facilities, TSA PreCheck services, airline lounges, post-security dining options, and post-security shopping options. "
    "Provide the airport's three-letter IATA code and official name."
)

SOUTHEASTERN_STATES = [
    "North Carolina", "South Carolina", "Georgia", "Florida", "Alabama", "Tennessee",
    "Kentucky", "Virginia", "West Virginia", "Mississippi", "Louisiana", "Arkansas"
]

MAJOR_US_AIRLINES = [
    "American Airlines", "Delta Air Lines", "United Airlines", "Southwest Airlines",
    "Alaska Airlines", "JetBlue", "Spirit Airlines", "Frontier Airlines"
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AmenitySources(BaseModel):
    wifi: List[str] = Field(default_factory=list)
    charging: List[str] = Field(default_factory=list)
    sanitizer: List[str] = Field(default_factory=list)
    mothers_rooms: List[str] = Field(default_factory=list)
    pet_relief: List[str] = Field(default_factory=list)
    currency_exchange_or_atm: List[str] = Field(default_factory=list)
    chapel: List[str] = Field(default_factory=list)
    volunteer_program: List[str] = Field(default_factory=list)
    accessibility_services: List[str] = Field(default_factory=list)
    rental_car: List[str] = Field(default_factory=list)
    tsa_precheck: List[str] = Field(default_factory=list)
    airline_lounges: List[str] = Field(default_factory=list)
    dining_post_security: List[str] = Field(default_factory=list)
    shopping_post_security: List[str] = Field(default_factory=list)


class AirportExtraction(BaseModel):
    iata_code: Optional[str] = None
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    hub_airlines: List[str] = Field(default_factory=list)

    # Source groups
    general_sources: List[str] = Field(default_factory=list)  # General airport info / location pages
    hub_sources: List[str] = Field(default_factory=list)      # Pages supporting hub/focus city claim

    amenities: AmenitySources = Field(default_factory=AmenitySources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airport() -> str:
    return """
    Extract the single airport identified in the answer and all cited URLs supporting the claims. Return a JSON object with these fields:

    Required identification fields:
    - iata_code: The three-letter IATA airport code provided in the answer (string). If missing, null.
    - official_name: The official airport name as stated in the answer (string). If missing, null.
    - city: The airport city (string) if given; else null.
    - state: The airport state (string) if given; else null.
    - hub_airlines: Array of airline names mentioned as hub/focus city (use exact names from the answer; empty array if none).

    Source fields:
    - general_sources: Array of URLs cited in the answer that support general facts like airport location, identity, or overview pages.
    - hub_sources: Array of URLs cited for the hub/focus city claim.

    Amenity sources (extract URLs ONLY if explicitly present in the answer; do not invent):
    - amenities.wifi: URLs supporting free WiFi throughout terminal(s)
    - amenities.charging: URLs supporting charging stations or gate-area power outlets
    - amenities.sanitizer: URLs supporting hand sanitizer stations
    - amenities.mothers_rooms: URLs supporting dedicated mother's/nursing rooms or lactation pods
    - amenities.pet_relief: URLs supporting pet/service animal relief areas
    - amenities.currency_exchange_or_atm: URLs supporting currency exchange services OR ATMs (either qualifies)
    - amenities.chapel: URLs supporting an interfaith chapel or worship space
    - amenities.volunteer_program: URLs supporting an airport volunteer or ambassador/passenger assistance program
    - amenities.accessibility_services: URLs supporting accessibility services for passengers with disabilities
    - amenities.rental_car: URLs supporting rental car facilities (on-site or via rental car center/shuttles)
    - amenities.tsa_precheck: URLs supporting TSA PreCheck screening lanes OR enrollment center availability
    - amenities.airline_lounges: URLs supporting airline lounges (e.g., Delta Sky Club, Admirals Club, United Club, Centurion, Escape Lounge)
    - amenities.dining_post_security: URLs showing dining options specifically in post-security/concourse/gate areas
    - amenities.shopping_post_security: URLs showing shopping/retail options specifically in post-security/concourse/gate areas

    IMPORTANT:
    - Extract only URLs explicitly present in the answer (plain URLs or in markdown links).
    - If the answer gives a source description without URL, return an empty array for that field.
    - Do not create or infer any URLs.
    - If an amenity is mentioned without a URL, simply return an empty array for that amenity.

    Keep strings as provided, do not normalize or reformat.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def truthy_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())

def combined_sources(primary: List[str], *fallbacks: List[List[str]]) -> List[str]:
    if primary:
        return primary
    out: List[str] = []
    for fb in fallbacks:
        out.extend(fb or [])
    return out

def major_airline_list_str() -> str:
    return ", ".join(MAJOR_US_AIRLINES)

def southeastern_states_str() -> str:
    return ", ".join(SOUTHEASTERN_STATES)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_airport_verifications(evaluator: Evaluator, parent_node, info: AirportExtraction) -> None:
    """
    Build leaf nodes per rubric and run verifications.
    """

    airport_node = evaluator.add_parallel(
        id="Airport_Identification",
        desc="Verify that the identified airport meets all specified regional, operational, and amenity requirements, and that required identifying information is provided",
        parent=parent_node,
        critical=False
    )

    # 1) IATA Code Provided (Critical, existence check)
    evaluator.add_custom_node(
        result=(truthy_str(info.iata_code) and len(info.iata_code.strip()) == 3),
        id="IATA_Code_Provided",
        desc="The answer must provide the three-letter IATA code of the identified airport",
        parent=airport_node,
        critical=True
    )

    # 2) Official Name Provided (Critical, existence check)
    evaluator.add_custom_node(
        result=truthy_str(info.official_name),
        id="Official_Name_Provided",
        desc="The answer must provide the official name of the identified airport",
        parent=airport_node,
        critical=True
    )

    # 3) Regional Location (Critical, URL-backed)
    loc_node = evaluator.add_leaf(
        id="Regional_Location",
        desc="The airport must be located in the Southeastern United States (states including: North Carolina, South Carolina, Georgia, Florida, Alabama, Tennessee, Kentucky, Virginia, West Virginia, Mississippi, Louisiana, Arkansas)",
        parent=airport_node,
        critical=True
    )
    # Compose claim for location
    city = info.city or ""
    state = info.state or ""
    name = info.official_name or ""
    code = info.iata_code or ""

    loc_claim = (
        f"The airport '{name}' (IATA {code}) is located in {city}, {state}. "
        f"The state '{state}' is within the Southeastern United States list: {southeastern_states_str()}."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=info.general_sources,
        additional_instruction=(
            "First, use the provided URLs to confirm the airport's city/state location. "
            "Second, independently confirm that the stated state is within the enumerated Southeastern list above. "
            "Treat as PASS only if both conditions hold. You may use your general knowledge for the regional classification step."
        )
    )

    # 4) Major Airline Hub (Critical, URL-backed)
    hub_node = evaluator.add_leaf(
        id="Major_Airline_Hub",
        desc="The airport must serve as a hub or focus city for at least one major U.S. airline",
        parent=airport_node,
        critical=True
    )
    airlines_str = ", ".join(info.hub_airlines) if info.hub_airlines else "unknown airline"
    hub_claim = (
        f"The airport '{name}' (IATA {code}) serves as a hub or focus city for at least one major U.S. airline "
        f"(e.g., {major_airline_list_str()}). The answer lists: {airlines_str}."
    )
    await evaluator.verify(
        claim=hub_claim,
        node=hub_node,
        sources=combined_sources(info.hub_sources, [info.general_sources]),
        additional_instruction=(
            "From the URL(s), verify that the airport is a 'hub', 'focus city', or equivalent operational base for at least one major U.S. airline "
            f"(consider major airlines among: {major_airline_list_str()}). Synonyms like 'focus city' or 'operating base' are acceptable."
        )
    )

    # 5) WiFi Service (Non-critical)
    wifi_node = evaluator.add_leaf(
        id="WiFi_Service",
        desc="The airport must provide free WiFi service throughout the terminal(s)",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The airport '{name}' provides free WiFi throughout its terminal(s).",
        node=wifi_node,
        sources=combined_sources(info.amenities.wifi, [info.general_sources]),
        additional_instruction="Accept phrases like 'free Wi-Fi', 'complimentary WiFi', 'terminal-wide Wi-Fi'."
    )

    # 6) Charging Stations (Non-critical)
    charging_node = evaluator.add_leaf(
        id="Charging_Stations",
        desc="The airport must have charging stations or power outlets available in gate areas",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Charging stations or power outlets are available in gate areas at '{name}'.",
        node=charging_node,
        sources=combined_sources(info.amenities.charging, [info.general_sources]),
        additional_instruction="Look for 'charging stations', 'USB ports', 'power outlets' near gates/concourses."
    )

    # 7) Sanitizer Stations (Non-critical)
    sanitizer_node = evaluator.add_leaf(
        id="Sanitizer_Stations",
        desc="The airport must have hand sanitizer stations available",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Hand sanitizer stations are available at '{name}'.",
        node=sanitizer_node,
        sources=combined_sources(info.amenities.sanitizer, [info.general_sources]),
        additional_instruction="Accept 'hand sanitizer dispensers' or similar wording across terminals."
    )

    # 8) Mothers Rooms (Non-critical)
    mothers_node = evaluator.add_leaf(
        id="Mothers_Rooms",
        desc="The airport must have dedicated mother's rooms or nursing rooms",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"'{name}' has dedicated mother's/nursing rooms (including lactation pods) available.",
        node=mothers_node,
        sources=combined_sources(info.amenities.mothers_rooms, [info.general_sources]),
        additional_instruction="Synonyms include 'nursing room', 'mother's room', 'lactation room', 'Mamava pod'."
    )

    # 9) Pet Relief Areas (Non-critical)
    pet_node = evaluator.add_leaf(
        id="Pet_Relief_Areas",
        desc="The airport must have pet relief areas for service animals and traveling pets",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"'{name}' provides pet/service animal relief areas.",
        node=pet_node,
        sources=combined_sources(info.amenities.pet_relief, [info.general_sources]),
        additional_instruction="Look for 'pet relief area', 'animal relief area', including post-security or outdoor locations."
    )

    # 10) Currency Exchange or ATM (Non-critical)
    currency_node = evaluator.add_leaf(
        id="Currency_Exchange",
        desc="The airport must have currency exchange services or ATMs",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"'{name}' offers either currency exchange services or ATMs on premises (either qualifies).",
        node=currency_node,
        sources=combined_sources(info.amenities.currency_exchange_or_atm, [info.general_sources]),
        additional_instruction="PASS if either currency exchange service OR ATMs are available anywhere at the airport."
    )

    # 11) Chapel Service (Non-critical)
    chapel_node = evaluator.add_leaf(
        id="Chapel_Service",
        desc="The airport must have a chapel or interfaith worship space",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"'{name}' has an interfaith chapel or worship/prayer room.",
        node=chapel_node,
        sources=combined_sources(info.amenities.chapel, [info.general_sources]),
        additional_instruction="Accept 'chapel', 'interfaith room', 'meditation room', or equivalent worship space."
    )

    # 12) Volunteer Program (Non-critical)
    volunteer_node = evaluator.add_leaf(
        id="Volunteer_Program",
        desc="The airport must have an airport volunteer program or passenger assistance service",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"'{name}' operates an airport volunteer/ambassador or passenger assistance program.",
        node=volunteer_node,
        sources=combined_sources(info.amenities.volunteer_program, [info.general_sources]),
        additional_instruction="Synonyms include 'Airport Ambassador', 'Volunteer Program', 'Passenger Assistance', 'information volunteers'."
    )

    # 13) Accessibility Services (Non-critical)
    access_node = evaluator.add_leaf(
        id="Accessibility_Services",
        desc="The airport must provide accessibility services for passengers with disabilities",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"'{name}' provides accessibility services for passengers with disabilities.",
        node=access_node,
        sources=combined_sources(info.amenities.accessibility_services, [info.general_sources]),
        additional_instruction="Look for 'ADA services', 'wheelchair assistance', 'TTY/TDD', 'visual paging', accessibility accommodations."
    )

    # 14) Rental Car Facilities (Non-critical)
    rental_node = evaluator.add_leaf(
        id="Rental_Car_Facilities",
        desc="The airport must have rental car facilities available (on-site or via consolidated rental car center)",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Rental car facilities (on-site or via rental car center/shuttles) are available at '{name}'.",
        node=rental_node,
        sources=combined_sources(info.amenities.rental_car, [info.general_sources]),
        additional_instruction="Accept 'Rental Car Center', 'car rentals', 'rental car agencies', with any location model (on-site/off-site via shuttle)."
    )

    # 15) TSA PreCheck (Non-critical)
    tsa_node = evaluator.add_leaf(
        id="TSA_PreCheck",
        desc="The airport must have TSA PreCheck processing or enrollment services available",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"TSA PreCheck screening lanes or enrollment services are available at '{name}'.",
        node=tsa_node,
        sources=combined_sources(info.amenities.tsa_precheck, [info.general_sources]),
        additional_instruction="PASS if either PreCheck screening lanes OR an on-site enrollment center are available."
    )

    # 16) Airline Lounges (Non-critical)
    lounges_node = evaluator.add_leaf(
        id="Airline_Lounges",
        desc="The airport must have airline lounges available to passengers",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"'{name}' has airline lounges (e.g., Delta Sky Club, Admirals Club, United Club, Centurion, Escape Lounge).",
        node=lounges_node,
        sources=combined_sources(info.amenities.airline_lounges, [info.general_sources]),
        additional_instruction="Look for named airline lounges or third-party lounges accessible to passengers."
    )

    # 17) Dining Options (Non-critical)
    dining_node = evaluator.add_leaf(
        id="Dining_Options",
        desc="The airport must have dining options available in the post-security area",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Post-security dining options are available at '{name}'.",
        node=dining_node,
        sources=combined_sources(info.amenities.dining_post_security, [info.general_sources]),
        additional_instruction="Accept concourse/gate-area food listings as post-security; verify that options exist beyond security checkpoints."
    )

    # 18) Shopping Options (Non-critical)
    shopping_node = evaluator.add_leaf(
        id="Shopping_Options",
        desc="The airport must have shopping options available in the post-security area",
        parent=airport_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Post-security shopping/retail options are available at '{name}'.",
        node=shopping_node,
        sources=combined_sources(info.amenities.shopping_post_security, [info.general_sources]),
        additional_instruction="Accept concourse/gate-area retail listings as post-security; verify that shops exist after security."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Southeastern airport hub amenities task.
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

    # Extract structured airport info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_airport(),
        template_class=AirportExtraction,
        extraction_name="airport_extraction",
    )

    # Record helpful reference info
    evaluator.add_custom_info(
        info={"southeastern_states": SOUTHEASTERN_STATES, "major_us_airlines": MAJOR_US_AIRLINES},
        info_type="reference_sets",
        info_name="reference_sets"
    )

    # Build verification tree and run checks
    await build_airport_verifications(evaluator, root, extracted_info)

    return evaluator.get_summary()