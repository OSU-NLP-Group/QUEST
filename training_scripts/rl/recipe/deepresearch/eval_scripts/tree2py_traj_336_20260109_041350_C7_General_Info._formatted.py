import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "airport_facilities_verification"
TASK_DESCRIPTION = (
    "Identify a major United States commercial airport that serves at least 10 million passengers annually. "
    "Provide the airport's three-letter IATA code and verify the availability of the following passenger facilities "
    "and services at this airport: (1) Animal/pet relief areas for traveling pets and service animals; "
    "(2) Mother's nursing stations or lactation rooms for breastfeeding parents; (3) Airport lounges (airline lounges "
    "or credit card lounges); (4) Free WiFi service available throughout terminal areas; "
    "(5) Electronic device charging stations for passenger use; (6) ATM locations within the terminal facilities; "
    "(7) Kids play areas or children's activity zones; (8) Multiple parking options (at least two types such as economy, "
    "garage, valet, or long-term parking); (9) Dining establishments or restaurants; "
    "(10) Retail shopping stores or duty-free shops; (11) Ground transportation options (such as rental car facilities, "
    "airport shuttles, or public transit connections); (12) TSA security screening checkpoints; "
    "(13) Accessible or adult assisted care restroom facilities. For each facility, provide evidence of its availability "
    "at your chosen airport through reference to the airport's official website or other reliable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EligibilityInfo(BaseModel):
    airport_name: Optional[str] = None
    iata_code: Optional[str] = None
    airport_official_url: Optional[str] = None
    # Evidence for US location and commercial passenger service
    us_service_sources: List[str] = Field(default_factory=list)
    # Annual passenger volume statement and sources
    passenger_volume_statement: Optional[str] = None
    passenger_volume_sources: List[str] = Field(default_factory=list)


class FacilitySources(BaseModel):
    pet_relief_urls: List[str] = Field(default_factory=list)
    nursing_rooms_urls: List[str] = Field(default_factory=list)
    lounges_urls: List[str] = Field(default_factory=list)
    free_wifi_urls: List[str] = Field(default_factory=list)
    charging_stations_urls: List[str] = Field(default_factory=list)
    atm_locations_urls: List[str] = Field(default_factory=list)
    kids_play_areas_urls: List[str] = Field(default_factory=list)
    parking_options_urls: List[str] = Field(default_factory=list)
    dining_options_urls: List[str] = Field(default_factory=list)
    shopping_options_urls: List[str] = Field(default_factory=list)
    ground_transportation_urls: List[str] = Field(default_factory=list)
    tsa_security_urls: List[str] = Field(default_factory=list)
    assisted_care_restrooms_urls: List[str] = Field(default_factory=list)


class AirportExtraction(BaseModel):
    eligibility: Optional[EligibilityInfo] = None
    facilities: Optional[FacilitySources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airport_info() -> str:
    return """
    You will extract structured information about ONE chosen major U.S. commercial airport from the answer.
    If the answer mentions multiple airports, select the FIRST one and ignore others.

    Extract the following fields:

    eligibility:
      - airport_name: The primary airport name chosen in the answer (e.g., "Hartsfield-Jackson Atlanta International Airport").
      - iata_code: The three-letter IATA code for the chosen airport (e.g., "ATL"). If missing, return null.
      - airport_official_url: The official airport website URL, if explicitly provided in the answer. Otherwise null.
      - us_service_sources: A list of URL(s) cited in the answer that demonstrate the airport is in the United States AND has regular commercial passenger service. If none are cited, return an empty list.
      - passenger_volume_statement: The statement in the answer regarding annual passenger volume (e.g., "served 50 million passengers in 2023"). If not included, return null.
      - passenger_volume_sources: A list of URL(s) cited in the answer that support the annual passenger volume statement (e.g., official stats, FAA data, airport facts pages). If none are cited, return an empty list.

    facilities:
      For EACH facility below, extract ONLY the URL(s) explicitly present in the answer that support the availability of that facility AT THE CHOSEN AIRPORT. Use an empty list if none are cited.
      - pet_relief_urls: Animal/pet relief areas.
      - nursing_rooms_urls: Mother's nursing stations or lactation rooms.
      - lounges_urls: Airport lounges (airline or credit card).
      - free_wifi_urls: Free WiFi in terminal areas.
      - charging_stations_urls: Electronic device charging stations.
      - atm_locations_urls: ATMs in terminal facilities.
      - kids_play_areas_urls: Kids play areas / children’s zones.
      - parking_options_urls: Parking options (ensure at least two distinct types are referenced).
      - dining_options_urls: Dining establishments / restaurants.
      - shopping_options_urls: Retail / duty-free shopping.
      - ground_transportation_urls: Rental cars, shuttles, and/or public transit connections.
      - tsa_security_urls: TSA security screening checkpoints.
      - assisted_care_restrooms_urls: Accessible or adult assisted care restroom facilities.

    IMPORTANT:
    - Extract only URLs explicitly present in the answer text (including markdown links). Do NOT invent or infer URLs.
    - If the answer provides references like "according to the official website" without actual URL(s), return an empty list for that field.
    - If a URL is missing the protocol (http/https), prepend "http://".
    - If anything is missing, return null or empty list accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _display_airport_name(elig: Optional[EligibilityInfo]) -> str:
    if elig and elig.airport_name:
        return elig.airport_name
    return "the chosen airport"


def _display_iata(elig: Optional[EligibilityInfo]) -> str:
    if elig and elig.iata_code:
        return elig.iata_code
    return "N/A"


def _make_additional_instruction_for_sources(
    base_instruction: str,
    sources: Optional[List[str]]
) -> str:
    """
    Build a strong verification instruction emphasizing reliance on provided sources
    and marking unsupported if none are present.
    """
    if sources and len(sources) > 0:
        return (
            base_instruction
            + " Use ONLY the cited webpage(s) to judge support. Do not rely on your own knowledge."
            + " Allow minor naming variations but require explicit or clearly implied confirmation."
        )
    else:
        return (
            "No citations (URLs) are provided in the answer for this verification."
            " According to the rubric, the claim must be supported by official or reliable sources."
            " Therefore, mark this claim as NOT SUPPORTED."
        )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_airport_eligibility(
    evaluator: Evaluator,
    parent_node,
    extraction: AirportExtraction
) -> None:
    """
    Build and verify the Airport Identification and Eligibility subtree.
    """
    elig_node = evaluator.add_parallel(
        id="Airport_Identification_and_Eligibility",
        desc="Airport is a qualifying major US commercial airport with regular passenger service and meets the passenger-volume threshold; airport is clearly identified.",
        parent=parent_node,
        critical=True
    )

    elig: EligibilityInfo = extraction.eligibility or EligibilityInfo()

    # IATA Code Provided (custom, critical)
    iata_ok = bool(elig.iata_code) and bool(re.fullmatch(r"[A-Za-z]{3}", elig.iata_code.strip()))
    evaluator.add_custom_node(
        result=iata_ok,
        id="IATA_Code_Provided",
        desc="A valid three-letter IATA code for the chosen airport is provided.",
        parent=elig_node,
        critical=True
    )

    # Build descriptive names for claims
    airport_name = _display_airport_name(elig)
    iata = _display_iata(elig)

    # Leaf: US Commercial airport with regular passenger service
    us_service_leaf = evaluator.add_leaf(
        id="Airport_in_United_States_with_Regular_Passenger_Service",
        desc="The identified airport is a US commercial airport with regular passenger service (supported by a reliable/official source).",
        parent=elig_node,
        critical=True
    )
    us_service_claim = (
        f"{airport_name} (IATA {iata}) is located in the United States and operates as a commercial airport with regular passenger service."
    )
    us_service_add_ins = _make_additional_instruction_for_sources(
        base_instruction=(
            "Confirm that the page(s) clearly indicate the airport is in the United States and "
            "has regular commercial passenger service (scheduled passenger flights)."
        ),
        sources=elig.us_service_sources
    )

    # Leaf: Annual passengers at least 10 million
    pax_leaf = evaluator.add_leaf(
        id="Annual_Passengers_At_Least_10_Million",
        desc="Evidence shows the airport serves at least 10 million passengers annually (supported by a reliable/official source).",
        parent=elig_node,
        critical=True
    )
    pax_claim = (
        f"{airport_name} (IATA {iata}) serves at least 10 million passengers annually."
    )
    pax_base_instruction = (
        "Verify from the cited page(s) that annual passenger traffic is >= 10,000,000. "
        "Phrasings like '10+ million', 'over 10 million', or explicit counts >= 10,000,000 should be accepted. "
        "Use the most recent or clearly stated annual figure; if ambiguous or below 10 million, mark as not supported."
    )
    if elig.passenger_volume_statement:
        pax_base_instruction += f" The answer's extracted statement was: '{elig.passenger_volume_statement}'."

    pax_add_ins = _make_additional_instruction_for_sources(
        base_instruction=pax_base_instruction,
        sources=elig.passenger_volume_sources
    )

    # Batch verify the two evidence-based eligibility checks
    await evaluator.batch_verify([
        (us_service_claim, elig.us_service_sources, us_service_leaf, us_service_add_ins),
        (pax_claim, elig.passenger_volume_sources, pax_leaf, pax_add_ins),
    ])


async def verify_facilities(
    evaluator: Evaluator,
    parent_node,
    extraction: AirportExtraction
) -> None:
    """
    Build and verify the Required Facilities and Services subtree.
    """
    fac_node = evaluator.add_parallel(
        id="Required_Facilities_and_Services_Verified_With_Evidence",
        desc="Each required facility/service is stated as available at the chosen airport and is supported by a citation to the airport's official website or another reliable source.",
        parent=parent_node,
        critical=True
    )

    elig: EligibilityInfo = extraction.eligibility or EligibilityInfo()
    fac: FacilitySources = extraction.facilities or FacilitySources()

    airport_name = _display_airport_name(elig)
    iata = _display_iata(elig)

    # Prepare leaf nodes for each facility
    leaves: Dict[str, Any] = {}

    def add_leaf_node(node_id: str, description: str) -> Any:
        node = evaluator.add_leaf(
            id=node_id,
            desc=description,
            parent=fac_node,
            critical=True
        )
        leaves[node_id] = node
        return node

    add_leaf_node("Pet_Relief_Areas", "Animal/pet relief areas are available (with official/reliable citation).")
    add_leaf_node("Nursing_Rooms", "Mother's nursing stations or lactation rooms are available (with official/reliable citation).")
    add_leaf_node("Lounges", "Airport lounges (airline and/or credit card lounges) are available (with official/reliable citation).")
    add_leaf_node("Free_WiFi", "Free WiFi is available throughout terminal areas (with official/reliable citation).")
    add_leaf_node("Charging_Stations", "Electronic device charging stations are available for passengers (with official/reliable citation).")
    add_leaf_node("ATM_Locations", "ATMs are available within terminal facilities (with official/reliable citation).")
    add_leaf_node("Kids_Play_Areas", "Kids play areas or children's activity zones are available (with official/reliable citation).")
    add_leaf_node("Multiple_Parking_Options", "At least two types of parking options are offered (e.g., economy, garage, valet, long-term) (with official/reliable citation).")
    add_leaf_node("Dining_Options", "Dining establishments/restaurants are available (with official/reliable citation).")
    add_leaf_node("Shopping_Options", "Retail shopping stores and/or duty-free shops are available (with official/reliable citation).")
    add_leaf_node("Ground_Transportation", "Ground transportation options are available (e.g., rental cars, shuttles, public transit) (with official/reliable citation).")
    add_leaf_node("TSA_Security", "TSA security screening checkpoints are present (with official/reliable citation).")
    add_leaf_node("Accessible_or_Adult_Assisted_Care_Restrooms", "Accessible or adult assisted care restroom facilities are available (with official/reliable citation).")

    # Build claims and sources for batch verification
    ops: List[Tuple[str, List[str] | None, Any, str]] = []

    # Pet relief areas
    claim_pet = f"Pet/animal relief areas are available at {airport_name} (IATA {iata})."
    ins_pet = _make_additional_instruction_for_sources(
        base_instruction="Look for 'pet relief areas', 'animal relief areas', or similar terms on the cited page(s). Confirm they are at this airport.",
        sources=fac.pet_relief_urls
    )
    ops.append((claim_pet, fac.pet_relief_urls, leaves["Pet_Relief_Areas"], ins_pet))

    # Nursing rooms
    claim_nurse = f"Lactation rooms or nursing stations are available at {airport_name} (IATA {iata})."
    ins_nurse = _make_additional_instruction_for_sources(
        base_instruction="Terms may include 'lactation room', 'mother's room', 'nursing station', or 'Mamava'. Confirm availability at this airport.",
        sources=fac.nursing_rooms_urls
    )
    ops.append((claim_nurse, fac.nursing_rooms_urls, leaves["Nursing_Rooms"], ins_nurse))

    # Lounges
    claim_lounge = f"Airport lounges (airline and/or credit card lounges) are available at {airport_name} (IATA {iata})."
    ins_lounge = _make_additional_instruction_for_sources(
        base_instruction="Confirm presence of airline lounges or credit card lounges (e.g., Amex Centurion, Priority Pass) at this airport.",
        sources=fac.lounges_urls
    )
    ops.append((claim_lounge, fac.lounges_urls, leaves["Lounges"], ins_lounge))

    # Free WiFi
    claim_wifi = f"Free WiFi service is available throughout terminal areas at {airport_name} (IATA {iata})."
    ins_wifi = _make_additional_instruction_for_sources(
        base_instruction="Confirm the WiFi is complimentary (free) and available across terminal areas, per the cited page(s).",
        sources=fac.free_wifi_urls
    )
    ops.append((claim_wifi, fac.free_wifi_urls, leaves["Free_WiFi"], ins_wifi))

    # Charging stations
    claim_charge = f"Electronic device charging stations are available for passengers at {airport_name} (IATA {iata})."
    ins_charge = _make_additional_instruction_for_sources(
        base_instruction="Look for 'charging stations', 'power outlets', or similar amenities being available to passengers.",
        sources=fac.charging_stations_urls
    )
    ops.append((claim_charge, fac.charging_stations_urls, leaves["Charging_Stations"], ins_charge))

    # ATMs
    claim_atm = f"ATMs are available within terminal facilities at {airport_name} (IATA {iata})."
    ins_atm = _make_additional_instruction_for_sources(
        base_instruction="Confirm presence of ATM locations within terminals or accessible passenger areas.",
        sources=fac.atm_locations_urls
    )
    ops.append((claim_atm, fac.atm_locations_urls, leaves["ATM_Locations"], ins_atm))

    # Kids play areas
    claim_kids = f"Kids play areas or children's activity zones are available at {airport_name} (IATA {iata})."
    ins_kids = _make_additional_instruction_for_sources(
        base_instruction="Look for 'children's play area', 'kids zone', or similar amenities being present.",
        sources=fac.kids_play_areas_urls
    )
    ops.append((claim_kids, fac.kids_play_areas_urls, leaves["Kids_Play_Areas"], ins_kids))

    # Multiple parking options
    claim_parking = f"At least two distinct types of parking options are offered at {airport_name} (IATA {iata})."
    ins_parking = _make_additional_instruction_for_sources(
        base_instruction="Confirm that at least two distinct parking types (e.g., economy, garage, valet, long-term) are offered.",
        sources=fac.parking_options_urls
    )
    ops.append((claim_parking, fac.parking_options_urls, leaves["Multiple_Parking_Options"], ins_parking))

    # Dining options
    claim_dining = f"Dining establishments or restaurants are available at {airport_name} (IATA {iata})."
    ins_dining = _make_additional_instruction_for_sources(
        base_instruction="Confirm presence of dining/restaurant options listed for this airport.",
        sources=fac.dining_options_urls
    )
    ops.append((claim_dining, fac.dining_options_urls, leaves["Dining_Options"], ins_dining))

    # Shopping options
    claim_shop = f"Retail shopping stores and/or duty-free shops are available at {airport_name} (IATA {iata})."
    ins_shop = _make_additional_instruction_for_sources(
        base_instruction="Confirm presence of retail shops and/or duty-free at this airport.",
        sources=fac.shopping_options_urls
    )
    ops.append((claim_shop, fac.shopping_options_urls, leaves["Shopping_Options"], ins_shop))

    # Ground transportation
    claim_gt = f"Ground transportation options (rental cars, shuttles, or public transit) are available at {airport_name} (IATA {iata})."
    ins_gt = _make_additional_instruction_for_sources(
        base_instruction="Confirm availability of rental car facilities, airport shuttles, and/or public transit connections.",
        sources=fac.ground_transportation_urls
    )
    ops.append((claim_gt, fac.ground_transportation_urls, leaves["Ground_Transportation"], ins_gt))

    # TSA Security
    claim_tsa = f"TSA security screening checkpoints are present at {airport_name} (IATA {iata})."
    ins_tsa = _make_additional_instruction_for_sources(
        base_instruction="Confirm presence of TSA security screening checkpoints for passenger processing at this airport.",
        sources=fac.tsa_security_urls
    )
    ops.append((claim_tsa, fac.tsa_security_urls, leaves["TSA_Security"], ins_tsa))

    # Accessible/Adult Assisted Care Restrooms
    claim_assist = f"Accessible or adult assisted care restroom facilities are available at {airport_name} (IATA {iata})."
    ins_assist = _make_additional_instruction_for_sources(
        base_instruction="Confirm availability of accessible restrooms and/or adult assisted care facilities (e.g., adult changing stations).",
        sources=fac.assisted_care_restrooms_urls
    )
    ops.append((claim_assist, fac.assisted_care_restrooms_urls, leaves["Accessible_or_Adult_Assisted_Care_Restrooms"], ins_assist))

    # Execute all facility verifications in parallel to avoid cross-sibling precondition skipping
    await evaluator.batch_verify(ops)


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
    Evaluate an answer for the airport facilities verification task.
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

    # Create the main critical node under the root
    main_node = evaluator.add_parallel(
        id="Airport_Facilities_Verification",
        desc="Identify one qualifying major US commercial airport and verify (with reliable citations) the availability of each listed passenger facility/service.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_airport_info(),
        template_class=AirportExtraction,
        extraction_name="airport_extraction"
    )

    # Add custom info helpful for inspecting evaluation context
    elig_info = extraction.eligibility.dict() if extraction.eligibility else {}
    evaluator.add_custom_info(
        info={"airport_name": elig_info.get("airport_name"),
              "iata_code": elig_info.get("iata_code"),
              "official_url": elig_info.get("airport_official_url")},
        info_type="extracted_airport_summary"
    )

    # Build and verify subtrees
    await verify_airport_eligibility(evaluator, main_node, extraction)
    await verify_facilities(evaluator, main_node, extraction)

    # Return unified summary
    return evaluator.get_summary()