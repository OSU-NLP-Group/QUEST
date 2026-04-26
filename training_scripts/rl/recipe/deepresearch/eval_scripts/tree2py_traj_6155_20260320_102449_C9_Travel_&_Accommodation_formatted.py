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
TASK_ID = "oc_trip_plan"
TASK_DESCRIPTION = """You are planning a 3-day trip to Orange County, California and need to arrange accommodations and activities. Provide the following information:

1. Pet-Friendly Hotel: Identify one hotel located in the city of Orange, California that meets these requirements:
   - Allows at least two pets per room
   - Does not prohibit leaving pets unattended in the room
   - Provide: hotel name, complete address, pet fee structure, and a link to the official website or pet policy page

2. Train Station: Identify one Amtrak Pacific Surfliner train station located in Orange County, California. Provide: the official station name, complete physical address, and a link to the official Pacific Surfliner station page.

3. Orange County Zoo Visit: Provide the following details about the Orange County Zoo:
   - Complete physical address
   - Admission price per person (for visitors over age 3)
   - Operating hours for both weekdays and weekends
   - Link to the official OC Zoo information page

4. Additional Parks: Identify two other Orange County parks or wilderness areas (not including Irvine Regional Park where the OC Zoo is located). For each park, provide:
   - Official park name
   - Location information (address or general area description)
   - Daily parking fee
   - Link to official park information page
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    hotel_name: Optional[str] = None
    hotel_address: Optional[str] = None
    hotel_official_url: Optional[str] = None
    pet_policy_url: Optional[str] = None
    pet_fee_structure: Optional[str] = None


class StationInfo(BaseModel):
    station_name: Optional[str] = None
    station_address: Optional[str] = None
    station_url: Optional[str] = None


class ZooInfo(BaseModel):
    zoo_address: Optional[str] = None
    admission_price: Optional[str] = None
    hours_weekday: Optional[str] = None
    hours_weekend: Optional[str] = None
    zoo_url: Optional[str] = None


class ParkInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # Address or general area description
    parking_fee: Optional[str] = None
    url: Optional[str] = None


class ParksExtraction(BaseModel):
    parks: List[ParkInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel() -> str:
    return """
    Extract exactly one pet-friendly hotel mentioned in the answer that is located in the City of Orange, California (not just Orange County).
    For that single hotel, extract the following fields:
    - hotel_name: The hotel's official name as written in the answer.
    - hotel_address: The complete physical address as provided in the answer (street, city, state, ZIP if available).
    - hotel_official_url: A direct URL to the hotel's official website or booking page mentioned in the answer.
    - pet_policy_url: A URL to the hotel's official pet policy page (brand site or the hotel's own official page) if mentioned in the answer.
    - pet_fee_structure: The pet fee structure as stated in the answer (e.g., amount, per pet/per night/per stay, deposit, etc.).
    Only extract URLs explicitly present in the answer text. If multiple hotels are mentioned, choose the first one that fits; otherwise, pick the first hotel mentioned even if incomplete.
    If any field is not available in the answer, set it to null.
    """


def prompt_extract_station() -> str:
    return """
    Extract exactly one Amtrak Pacific Surfliner station mentioned in the answer that is located in Orange County, California.
    For that station, extract:
    - station_name: The official station name as written in the answer.
    - station_address: The complete physical address as provided in the answer.
    - station_url: The URL to the official Pacific Surfliner station page as provided in the answer.
    Only extract URLs explicitly present in the answer text. If multiple stations are mentioned, choose the first one. If some fields are not available, set them to null.
    """


def prompt_extract_zoo() -> str:
    return """
    Extract the details for the Orange County Zoo (OC Zoo) as provided in the answer.
    Extract:
    - zoo_address: The complete physical address provided in the answer.
    - admission_price: The per-person admission price for visitors over age 3 as stated in the answer. If the answer lists separate prices (e.g., children 3–12 and adults 13+), include the combined statement that appears in the answer.
    - hours_weekday: The operating hours for weekdays as given in the answer (e.g., Mon–Fri 10am–4:30pm).
    - hours_weekend: The operating hours for weekends as given in the answer (e.g., Sat–Sun 10am–4:30pm).
    - zoo_url: The URL to the official OC Zoo information page provided in the answer.
    Extract URLs only if explicitly in the answer. If a field is missing, set it to null.
    """


def prompt_extract_parks() -> str:
    return """
    Extract up to two additional Orange County parks or wilderness areas (not including Irvine Regional Park).
    For each of the first two parks mentioned, extract:
    - name: The official park name as stated in the answer.
    - location: The location info as provided in the answer (full address or general area description).
    - parking_fee: The daily parking fee as stated in the answer (e.g., $5 per vehicle, free, no parking fee).
    - url: The URL to the official park information page provided in the answer.
    Only include parks that are in Orange County. If more than two are listed, take the first two. If fewer than two are listed, return only those available.
    Extract URLs only if explicitly present in the answer. Use null for any missing field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _url_list(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip() != ""]


# --------------------------------------------------------------------------- #
# Verification routines                                                       #
# --------------------------------------------------------------------------- #
async def verify_hotel(evaluator: Evaluator, parent_node, hotel: HotelInfo) -> None:
    # Parent group: Hotel selection (critical section)
    hotel_node = evaluator.add_parallel(
        id="Hotel_Selection",
        desc="Pet-friendly hotel meeting all accommodation requirements",
        parent=parent_node,
        critical=True
    )

    # Documentation URLs (critical siblings, gate others)
    docs_node = evaluator.add_parallel(
        id="Hotel_Documentation",
        desc="Required documentation and URL references for hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel.hotel_official_url and hotel.hotel_official_url.strip()),
        id="Hotel_Official_URL",
        desc="Direct link to hotel's official website or booking page provided",
        parent=docs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel.pet_policy_url and hotel.pet_policy_url.strip()),
        id="Pet_Policy_URL",
        desc="URL reference provided for pet policy verification",
        parent=docs_node,
        critical=True
    )

    # Basic info (critical)
    basic_node = evaluator.add_parallel(
        id="Hotel_Basic_Info",
        desc="Hotel identification and location information",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel.hotel_name and hotel.hotel_name.strip()),
        id="Hotel_Name",
        desc="Hotel name is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel.hotel_address and hotel.hotel_address.strip()),
        id="Hotel_Complete_Address",
        desc="Complete physical address of the hotel is provided",
        parent=basic_node,
        critical=True
    )
    # City of Orange verification (use official/policy URLs)
    hotel_city_leaf = evaluator.add_leaf(
        id="Hotel_Location_City",
        desc="Hotel is located in the city of Orange, California",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's property address is in the City of Orange, California (e.g., address shows 'Orange, CA').",
        node=hotel_city_leaf,
        sources=_url_list(hotel.hotel_official_url, hotel.pet_policy_url),
        additional_instruction="Pass if the page shows a full address containing 'Orange, CA' or clearly states the hotel is in the City of Orange. Minor formatting differences are acceptable."
    )

    # Pet policy details (critical)
    policy_node = evaluator.add_parallel(
        id="Pet_Policy_Details",
        desc="Hotel pet policy meets all specified requirements",
        parent=hotel_node,
        critical=True
    )
    # Two pets allowed
    two_pets_leaf = evaluator.add_leaf(
        id="Two_Pets_Allowed",
        desc="Hotel allows at least two pets per room",
        parent=policy_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel's pet policy allows at least two pets per room (e.g., 'up to 2 pets', '2 pets maximum', or more).",
        node=two_pets_leaf,
        sources=_url_list(hotel.pet_policy_url, hotel.hotel_official_url),
        additional_instruction="Look for explicit language like '2 pets allowed', 'two pets per room', or policies that allow 2 or more pets. Synonyms like 'dogs' or 'cats' count as pets."
    )
    # No prohibition on unattended pets
    unattended_leaf = evaluator.add_leaf(
        id="No_Unattended_Restriction",
        desc="Hotel does not prohibit leaving pets unattended in rooms",
        parent=policy_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's pet policy does not prohibit leaving pets unattended in guest rooms (it either allows unattended when crated or does not include a prohibition).",
        node=unattended_leaf,
        sources=_url_list(hotel.pet_policy_url, hotel.hotel_official_url),
        additional_instruction="Fail only if the page clearly states pets may NOT be left unattended. If it explicitly allows leaving pets unattended (e.g., if crated), pass. If the policy is silent on this, treat it as 'no prohibition' and pass."
    )
    # Pet fee structure provided (as answer-provided check)
    evaluator.add_custom_node(
        result=bool(hotel.pet_fee_structure and hotel.pet_fee_structure.strip()),
        id="Pet_Fee_Structure",
        desc="Pet fee structure is clearly documented",
        parent=policy_node,
        critical=True
    )


async def verify_station(evaluator: Evaluator, parent_node, station: StationInfo) -> None:
    station_node = evaluator.add_parallel(
        id="Train_Station_Selection",
        desc="Amtrak Pacific Surfliner train station information",
        parent=parent_node,
        critical=True
    )

    # Documentation (critical URL presence)
    station_docs = evaluator.add_parallel(
        id="Station_Documentation",
        desc="Required documentation for train station",
        parent=station_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(station.station_url and station.station_url.strip()),
        id="Station_URL",
        desc="URL reference to official Pacific Surfliner station page",
        parent=station_docs,
        critical=True
    )

    # Basic info (critical)
    station_basic = evaluator.add_parallel(
        id="Station_Basic_Info",
        desc="Train station identification information",
        parent=station_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(station.station_name and station.station_name.strip()),
        id="Station_Name",
        desc="Official name of the Pacific Surfliner station provided",
        parent=station_basic,
        critical=True
    )
    station_type_leaf = evaluator.add_leaf(
        id="Station_Type",
        desc="Station is confirmed as an Amtrak Pacific Surfliner station",
        parent=station_basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"This is an official Pacific Surfliner station page for '{station.station_name or ''}'.",
        node=station_type_leaf,
        sources=station.station_url,
        additional_instruction="The page should be on PacificSurfliner.com or explicitly mention 'Pacific Surfliner' and the station."
    )

    # Location (critical)
    station_loc = evaluator.add_parallel(
        id="Station_Location",
        desc="Train station location information",
        parent=station_node,
        critical=True
    )
    in_oc_leaf = evaluator.add_leaf(
        id="Station_In_Orange_County",
        desc="Station is located within Orange County, California",
        parent=station_loc,
        critical=True
    )
    await evaluator.verify(
        claim="This station is located in Orange County, California.",
        node=in_oc_leaf,
        sources=station.station_url,
        additional_instruction="Pass if the page mentions Orange County or lists a city that is in Orange County (e.g., Anaheim, Fullerton, Irvine, Santa Ana, Orange, Tustin, Buena Park, San Clemente, San Juan Capistrano, Laguna Niguel/Mission Viejo)."
    )
    evaluator.add_custom_node(
        result=bool(station.station_address and station.station_address.strip()),
        id="Station_Complete_Address",
        desc="Complete physical address of the station provided",
        parent=station_loc,
        critical=True
    )


async def verify_zoo(evaluator: Evaluator, parent_node, zoo: ZooInfo) -> None:
    zoo_node = evaluator.add_parallel(
        id="OC_Zoo_Visit",
        desc="Orange County Zoo visit details with complete information",
        parent=parent_node,
        critical=True
    )

    # Documentation (critical URL presence)
    zoo_docs = evaluator.add_parallel(
        id="Zoo_Documentation",
        desc="Required documentation for OC Zoo",
        parent=zoo_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(zoo.zoo_url and zoo.zoo_url.strip()),
        id="Zoo_URL",
        desc="URL reference to official OC Zoo information page",
        parent=zoo_docs,
        critical=True
    )

    # Location info
    zoo_loc = evaluator.add_parallel(
        id="Zoo_Location_Info",
        desc="Orange County Zoo location information",
        parent=zoo_node,
        critical=True
    )
    zoo_addr_leaf = evaluator.add_leaf(
        id="Zoo_Complete_Address",
        desc="Complete physical address of the OC Zoo provided",
        parent=zoo_loc,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page shows the OC Zoo's address as: {zoo.zoo_address or ''}.",
        node=zoo_addr_leaf,
        sources=zoo.zoo_url,
        additional_instruction="Pass if the page lists an address matching the provided one (minor formatting and abbreviation differences are acceptable). Fail if the address is missing in the answer or clearly mismatched."
    )

    # Pricing
    zoo_price = evaluator.add_parallel(
        id="Zoo_Pricing_Info",
        desc="Orange County Zoo pricing information",
        parent=zoo_node,
        critical=True
    )
    price_leaf = evaluator.add_leaf(
        id="Admission_Price",
        desc="Per-person admission price for visitors over age 3 provided",
        parent=zoo_price,
        critical=True
    )
    await evaluator.verify(
        claim=f"The admission price for visitors age 3 and up matches: {zoo.admission_price or ''}.",
        node=price_leaf,
        sources=zoo.zoo_url,
        additional_instruction="If the page lists separate prices for children (3–12) and adults (13+), it's acceptable as long as it aligns with the provided statement. Minor format differences are OK."
    )

    # Schedule
    zoo_sched = evaluator.add_parallel(
        id="Zoo_Schedule_Info",
        desc="Orange County Zoo operating schedule",
        parent=zoo_node,
        critical=True
    )
    hours_leaf = evaluator.add_leaf(
        id="Operating_Hours",
        desc="Operating hours for both weekdays and weekends provided",
        parent=zoo_sched,
        critical=True
    )
    combined_hours = f"Weekday hours: {zoo.hours_weekday or ''}; Weekend hours: {zoo.hours_weekend or ''}."
    await evaluator.verify(
        claim=f"The OC Zoo's operating hours match the following: {combined_hours}",
        node=hours_leaf,
        sources=zoo.zoo_url,
        additional_instruction="Pass if the page shows hours that match for weekdays and weekends as provided, or if it states the same hours daily covering both. Minor formatting differences and notes about holidays are acceptable."
    )


async def verify_park(evaluator: Evaluator, parent_node, park: ParkInfo, index: int) -> None:
    park_node = evaluator.add_parallel(
        id=f"Park_Visit_{index}",
        desc=f"{'First' if index == 1 else 'Second'} additional Orange County park or wilderness area details",
        parent=parent_node,
        critical=False  # Non-critical at root level; allows partial credit
    )

    # Documentation URL (critical inside park group)
    park_docs = evaluator.add_parallel(
        id=f"Park_{index}_Documentation",
        desc=f"Required documentation for {'first' if index == 1 else 'second'} park",
        parent=park_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(park.url and park.url.strip()),
        id=f"Park_{index}_URL",
        desc="URL reference to official park information page",
        parent=park_docs,
        critical=True
    )

    # Basic info (critical inside park group)
    park_basic = evaluator.add_parallel(
        id=f"Park_{index}_Basic_Info",
        desc=f"{'First' if index == 1 else 'Second'} park identification information",
        parent=park_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(park.name and park.name.strip()),
        id=f"Park_{index}_Name",
        desc=f"Official name of the {'first' if index == 1 else 'second'} park provided",
        parent=park_basic,
        critical=True
    )
    in_oc_leaf = evaluator.add_leaf(
        id=f"Park_{index}_Orange_County_Location",
        desc=f"{'First' if index == 1 else 'Second'} park is located in Orange County",
        parent=park_basic,
        critical=True
    )
    await evaluator.verify(
        claim="This park is located in Orange County, California.",
        node=in_oc_leaf,
        sources=park.url,
        additional_instruction="Pass if the official page indicates it is an OC Parks site or mentions 'Orange County'. A city clearly within Orange County is also acceptable."
    )
    not_irvine_leaf = evaluator.add_leaf(
        id=f"Park_{index}_Not_Irvine_Regional",
        desc=f"{'First' if index == 1 else 'Second'} park is not Irvine Regional Park",
        parent=park_basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The park name '{(park.name or '').strip()}' is not 'Irvine Regional Park' (case-insensitive).",
        node=not_irvine_leaf,
        additional_instruction="Treat minor punctuation or capitalization differences as equivalent; this should fail only if the park is Irvine Regional Park."
    )

    # Location details (critical inside park group)
    park_loc = evaluator.add_parallel(
        id=f"Park_{index}_Location_Details",
        desc=f"{'First' if index == 1 else 'Second'} park location details",
        parent=park_node,
        critical=True
    )
    park_loc_leaf = evaluator.add_leaf(
        id=f"Park_{index}_Address_Or_Area",
        desc="Park location information (address or general area description) provided",
        parent=park_loc,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page indicates the park location consistent with: {park.location or ''}.",
        node=park_loc_leaf,
        sources=park.url,
        additional_instruction="Pass if the page shows an address or area description that matches the provided location text (minor differences in formatting or wording are acceptable)."
    )

    # Pricing (critical inside park group)
    park_price = evaluator.add_parallel(
        id=f"Park_{index}_Pricing_Info",
        desc=f"{'First' if index == 1 else 'Second'} park pricing information",
        parent=park_node,
        critical=True
    )
    park_fee_leaf = evaluator.add_leaf(
        id=f"Park_{index}_Parking_Fee",
        desc="Daily parking fee information provided",
        parent=park_price,
        critical=True
    )
    await evaluator.verify(
        claim=f"The daily parking fee matches: {park.parking_fee or ''}.",
        node=park_fee_leaf,
        sources=park.url,
        additional_instruction="Pass if the page confirms the stated daily parking fee (e.g., per vehicle/day). If the page states 'free' or 'no parking fee' and the provided text is equivalent, pass."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Orange County travel plan task.
    """
    # Initialize evaluator (root: PARALLEL, non-critical to allow partial credit across sections)
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

    # Extract information concurrently
    hotel_task = evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )
    station_task = evaluator.extract(
        prompt=prompt_extract_station(),
        template_class=StationInfo,
        extraction_name="station_info"
    )
    zoo_task = evaluator.extract(
        prompt=prompt_extract_zoo(),
        template_class=ZooInfo,
        extraction_name="zoo_info"
    )
    parks_task = evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_info"
    )

    hotel, station, zoo, parks_extraction = await asyncio.gather(
        hotel_task, station_task, zoo_task, parks_task
    )

    # Build and run verifications
    await verify_hotel(evaluator, root, hotel)
    await verify_station(evaluator, root, station)
    await verify_zoo(evaluator, root, zoo)

    parks: List[ParkInfo] = parks_extraction.parks[:2] if parks_extraction and parks_extraction.parks else []
    while len(parks) < 2:
        parks.append(ParkInfo())  # pad if fewer than 2 provided

    await verify_park(evaluator, root, parks[0], index=1)
    await verify_park(evaluator, root, parks[1], index=2)

    # Return structured result
    return evaluator.get_summary()