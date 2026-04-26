import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "genz_trip_2026"
TASK_DESCRIPTION = (
    "A Gen Z traveler born in 2005 is planning a trip in 2026 for a group of 5 people. They want to fly direct from "
    "Bangor International Airport (BGR) in Maine to a Florida beach destination using Allegiant Air, which offers "
    "year-round service. After spending time at the beach, the group will drive to Pigeon Forge, Tennessee to visit "
    "Dollywood and needs to stay at one of Dollywood's official on-property resorts in a single room that can "
    "accommodate all 5 people. Additionally, they are considering a Caribbean alternative that would involve flying "
    "from Nashville International Airport (BNA) instead, but only to a destination with a U.S. State Department Level 1 "
    "travel advisory (Exercise Normal Precautions) and direct flight service available in 2026. Answer the following: "
    "(1) Which Florida beach destination (city/airport) should they fly to on Allegiant Air from Bangor? "
    "(2) Which of Dollywood's two official resorts (DreamMore Resort or HeartSong Lodge) has room types that can accommodate 5 guests? "
    "(3) What is the specific name of a room type at that resort that officially sleeps 5 or more people? "
    "(4) Identify one Caribbean destination accessible via direct flights from Nashville (BNA) in 2026 that has a Level 1 U.S. travel advisory, and name the airline operating that route. "
    "Provide URL references from official sources to support each answer."
)

CURRENT_TRIP_YEAR = 2026
BIRTH_YEAR = 2005


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FloridaExtraction(BaseModel):
    destination_city: Optional[str] = None
    destination_airport_code: Optional[str] = None  # e.g., SFB, PGD, PIE, VPS, etc.
    destination_airport_name: Optional[str] = None  # e.g., Orlando Sanford International Airport
    year_round_mentioned: Optional[str] = None      # text mention like "year-round", "seasonal"
    direct_nonstop_mentioned: Optional[str] = None  # text mention like "nonstop", "direct"
    allegiant_route_urls: List[str] = Field(default_factory=list)  # Airline or airport official pages
    beach_urls: List[str] = Field(default_factory=list)            # City/region official page confirming beach/coastal


class ResortExtraction(BaseModel):
    resort_name: Optional[str] = None               # DreamMore Resort or HeartSong Lodge
    resort_city: Optional[str] = None               # should be Pigeon Forge, Tennessee
    room_type_name: Optional[str] = None            # official room type name
    room_capacity_text: Optional[str] = None        # text showing sleeps 5+, e.g., "Sleeps up to 6"
    bed_configuration: Optional[str] = None         # e.g., "2 queen beds + sleeper sofa"
    resort_urls: List[str] = Field(default_factory=list)  # official resort overview/landing pages
    room_urls: List[str] = Field(default_factory=list)    # official room detail pages confirming capacity


class CaribbeanExtraction(BaseModel):
    destination_name: Optional[str] = None          # e.g., Grand Cayman, Aruba, etc.
    country_name: Optional[str] = None              # Cayman Islands, Aruba, etc.
    airline: Optional[str] = None                   # e.g., Southwest, American, JetBlue
    advisory_level_text: Optional[str] = None       # e.g., "Level 1", "Exercise Normal Precautions"
    flight_route_urls: List[str] = Field(default_factory=list)  # official airline/BNA route pages
    advisory_urls: List[str] = Field(default_factory=list)      # U.S. State Dept advisory page URL(s)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_florida() -> str:
    return """
    Extract the Florida beach destination and official sources from the answer.

    Required fields:
    - destination_city: The Florida city name mentioned (e.g., Punta Gorda, Destin/Fort Walton Beach, Clearwater).
    - destination_airport_code: The airport code (e.g., PGD, VPS, PIE, SFB).
    - destination_airport_name: The full official airport name if provided.
    - year_round_mentioned: The exact text in the answer that indicates whether service is year-round or seasonal.
    - direct_nonstop_mentioned: The exact text in the answer indicating "direct" or "nonstop".
    - allegiant_route_urls: List all official URLs cited that confirm Allegiant (or airport) service from Bangor (BGR) to the destination. Prefer Allegiant Air or airport official pages. Include all URLs the answer provides for this route.
    - beach_urls: List any official city/region/tourism URLs cited that confirm the destination is a beach/coastal location in Florida.

    If any field is not present in the answer, set it to null or an empty list as appropriate.
    Only include URLs explicitly shown in the answer.
    """


def prompt_extract_resort() -> str:
    return """
    Extract the Dollywood resort selection and the specific room type details from the answer.

    Required fields:
    - resort_name: The selected resort name (must be one of DreamMore Resort or HeartSong Lodge).
    - resort_city: The city of the resort (should be Pigeon Forge, Tennessee).
    - room_type_name: The official room type name that sleeps 5+.
    - room_capacity_text: The exact capacity text from the answer (e.g., "Sleeps up to 5", "Sleeps up to 6").
    - bed_configuration: The bed configuration described (e.g., "2 queens + sleeper sofa").
    - resort_urls: Official Dollywood resort page URLs cited in the answer.
    - room_urls: Official room detail page URLs cited in the answer that confirm the capacity and bed configuration.

    If any field is missing, set it to null or an empty list as appropriate.
    Only include URLs explicitly shown in the answer.
    """


def prompt_extract_caribbean() -> str:
    return """
    Extract the Caribbean alternative details and official sources from the answer.

    Required fields:
    - destination_name: The Caribbean destination name (island/city).
    - country_name: The country or territory name for the destination.
    - airline: The airline operating the direct route from BNA.
    - advisory_level_text: The advisory level text (e.g., "Level 1: Exercise Normal Precautions").
    - flight_route_urls: Official airline or BNA airport route/schedule page URLs cited in the answer that confirm direct service in 2026.
    - advisory_urls: Official U.S. State Department travel advisory page URLs cited for the destination/country.

    If any field is missing, set it to null or an empty list as appropriate.
    Only include URLs explicitly shown in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _fallback_sources(primary: List[str], secondary: List[str]) -> List[str]:
    """Return primary if non-empty, else secondary (could be empty)."""
    return primary if primary else secondary


# --------------------------------------------------------------------------- #
# Florida destination verification                                            #
# --------------------------------------------------------------------------- #
async def verify_florida_destination(evaluator: Evaluator, parent_node, florida: FloridaExtraction) -> None:
    # Parent node: critical (essential requirement)
    fl_node = evaluator.add_parallel(
        id="florida_destination",
        desc="Identify the Florida destination accessible via Allegiant Air direct flight from Bangor",
        parent=parent_node,
        critical=True
    )

    # Existence: destination and at least one route URL
    dest_exists = bool((florida.destination_city or florida.destination_airport_code or florida.destination_airport_name))
    urls_exist = bool(florida.allegiant_route_urls)
    evaluator.add_custom_node(
        result=(dest_exists and urls_exist),
        id="fl_dest_and_sources_provided",
        desc="Florida destination and official route source URLs are provided",
        parent=fl_node,
        critical=True
    )

    # Reference URL existence as a separate critical node (as required by rubric)
    evaluator.add_custom_node(
        result=urls_exist,
        id="reference_url_florida",
        desc="Provide valid URL reference confirming the Allegiant Air route from Bangor",
        parent=fl_node,
        critical=True
    )

    # Airport verification: Allegiant serves the destination from BGR with year-round service
    airport_claim_parts = []
    if florida.destination_airport_code:
        airport_claim_parts.append(f"airport code {florida.destination_airport_code}")
    if florida.destination_airport_name:
        airport_claim_parts.append(f"{florida.destination_airport_name}")
    airport_desc = ", ".join(airport_claim_parts) if airport_claim_parts else "the destination airport"

    airport_verify_leaf = evaluator.add_leaf(
        id="airport_verification",
        desc="The destination airport must be served by Allegiant Air with year-round direct service from BGR",
        parent=fl_node,
        critical=True
    )
    airport_claim = (
        f"Allegiant Air offers year-round service from Bangor International Airport (BGR) to {airport_desc}."
    )
    await evaluator.verify(
        claim=airport_claim,
        node=airport_verify_leaf,
        sources=florida.allegiant_route_urls,
        additional_instruction=(
            "Use the official airline or airport page(s) to confirm the route exists and is operated year-round in 2026. "
            "If the page indicates seasonal service only, the claim is not supported."
        )
    )

    # Flight directness: must be nonstop/direct with no connections
    direct_leaf = evaluator.add_leaf(
        id="flight_directness",
        desc="Service must be non-stop/direct with no connections",
        parent=fl_node,
        critical=True
    )
    direct_claim = (
        f"The Allegiant Air service from Bangor (BGR) to {airport_desc} is nonstop/direct with no connections."
    )
    await evaluator.verify(
        claim=direct_claim,
        node=direct_leaf,
        sources=florida.allegiant_route_urls,
        additional_instruction=(
            "Confirm the service is described as 'nonstop' or 'direct' on the official source. "
            "If it requires a connection or stop with plane change, treat as not supported."
        )
    )

    # Beach location: must be a beach/coastal location in Florida
    beach_leaf = evaluator.add_leaf(
        id="beach_location",
        desc="The destination must be a beach/coastal location in Florida",
        parent=fl_node,
        critical=True
    )
    city_or_airport = florida.destination_city or florida.destination_airport_name or "the destination"
    beach_sources = _fallback_sources(florida.beach_urls, florida.allegiant_route_urls)
    beach_claim = f"{city_or_airport} is a beach or coastal location in Florida."
    await evaluator.verify(
        claim=beach_claim,
        node=beach_leaf,
        sources=beach_sources,
        additional_instruction=(
            "Verify the location is coastal or commonly recognized as a Florida beach destination (Atlantic or Gulf coast). "
            "Check for explicit statements indicating beach/coast."
        )
    )


# --------------------------------------------------------------------------- #
# Dollywood resort verification                                               #
# --------------------------------------------------------------------------- #
async def verify_dollywood_resort(evaluator: Evaluator, parent_node, resort: ResortExtraction) -> None:
    # Parent node: critical
    resort_node = evaluator.add_parallel(
        id="dollywood_resort",
        desc="Identify which of the two Dollywood resorts (DreamMore or HeartSong) is the answer",
        parent=parent_node,
        critical=True
    )

    # Existence: resort name and at least one relevant URL
    name_exists = bool(resort.resort_name)
    resort_urls_exist = bool(resort.resort_urls or resort.room_urls)
    evaluator.add_custom_node(
        result=(name_exists and resort_urls_exist),
        id="resort_name_and_sources_provided",
        desc="Resort name is provided with official source URLs",
        parent=resort_node,
        critical=True
    )

    # Resort identification: must be Dollywood DreamMore or HeartSong
    resort_ident_leaf = evaluator.add_leaf(
        id="resort_identification",
        desc="The resort must be one of the two official Dollywood-owned properties: DreamMore Resort or HeartSong Lodge",
        parent=resort_node,
        critical=True
    )
    ident_claim = (
        f"The selected resort '{resort.resort_name}' is an official Dollywood on-property resort: "
        "either 'Dollywood's DreamMore Resort and Spa' or 'Dollywood's HeartSong Lodge & Resort'."
    )
    await evaluator.verify(
        claim=ident_claim,
        node=resort_ident_leaf,
        sources=resort.resort_urls,
        additional_instruction=(
            "Use the official Dollywood resort site to verify the property is one of the two owned resorts on Dollywood grounds."
        )
    )

    # Location: must be Pigeon Forge, Tennessee
    location_leaf = evaluator.add_leaf(
        id="location_pigeon_forge",
        desc="The resort must be located in Pigeon Forge, Tennessee",
        parent=resort_node,
        critical=True
    )
    loc_claim = (
        f"The resort '{resort.resort_name}' is located in Pigeon Forge, Tennessee."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=location_leaf,
        sources=resort.resort_urls,
        additional_instruction="Confirm the resort address/location as Pigeon Forge, TN on the official site."
    )

    # Reference URL confirming room types that accommodate 5
    ref_resort_leaf = evaluator.add_leaf(
        id="reference_url_resort",
        desc="Provide valid URL reference confirming the resort has room types accommodating 5 guests",
        parent=resort_node,
        critical=True
    )
    ref_resort_claim = (
        f"The official resort site for '{resort.resort_name}' shows at least one room type that accommodates 5 guests."
    )
    await evaluator.verify(
        claim=ref_resort_claim,
        node=ref_resort_leaf,
        sources=resort.room_urls or resort.resort_urls,
        additional_instruction="Look for occupancy/capacity details on official room pages indicating sleeps 5 or more."
    )


# --------------------------------------------------------------------------- #
# Room type specification verification                                        #
# --------------------------------------------------------------------------- #
async def verify_room_type_specification(evaluator: Evaluator, parent_node, resort: ResortExtraction) -> None:
    # Parent node: critical in rubric. To satisfy framework constraints (critical parent cannot have non-critical children),
    # we mark all children here as critical, including bed_configuration.
    room_node = evaluator.add_parallel(
        id="room_type_specification",
        desc="Identify the specific room type(s) at the selected resort that accommodate 5 people",
        parent=parent_node,
        critical=True
    )

    # Existence: room type and at least one official room page URL
    room_exists = bool(resort.room_type_name)
    room_url_exists = bool(resort.room_urls)
    evaluator.add_custom_node(
        result=(room_exists and room_url_exists),
        id="room_type_provided",
        desc="A specific room type and official room page URL(s) are provided",
        parent=room_node,
        critical=True
    )

    # Room name is official
    room_name_leaf = evaluator.add_leaf(
        id="room_name",
        desc="Provide the official name/designation of a room type that sleeps 5 or more",
        parent=room_node,
        critical=True
    )
    room_name_claim = (
        f"'{resort.room_type_name}' is an official room type listed at {resort.resort_name}."
    )
    await evaluator.verify(
        claim=room_name_claim,
        node=room_name_leaf,
        sources=resort.room_urls,
        additional_instruction="Verify the room type name as shown on the official resort room detail page(s)."
    )

    # Capacity verification: sleeps at least 5
    capacity_leaf = evaluator.add_leaf(
        id="capacity_verification",
        desc="Confirm the room capacity is at least 5 guests as stated by the resort",
        parent=room_node,
        critical=True
    )
    capacity_claim = (
        f"The room type '{resort.room_type_name}' at {resort.resort_name} officially sleeps at least 5 guests."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=resort.room_urls,
        additional_instruction="Confirm occupancy/capacity on the official page indicates 5 or more guests can sleep in a single room."
    )

    # Bed configuration supports 5-person occupancy (marked critical to satisfy framework constraints)
    bed_leaf = evaluator.add_leaf(
        id="bed_configuration",
        desc="Describe the bed configuration that enables 5-person occupancy",
        parent=room_node,
        critical=True
    )
    bed_desc = resort.bed_configuration or "the room's bed configuration"
    bed_claim = (
        f"The bed configuration for '{resort.room_type_name}' supports sleeping 5 people; the configuration described is '{bed_desc}'."
    )
    await evaluator.verify(
        claim=bed_claim,
        node=bed_leaf,
        sources=resort.room_urls,
        additional_instruction=(
            "Confirm the official description mentions enough beds/bedding (e.g., 2 queens + sleeper sofa) to sleep 5 or more."
        )
    )

    # Reference URL confirming specific room capacity
    ref_room_leaf = evaluator.add_leaf(
        id="reference_url_room_type",
        desc="Provide valid URL reference confirming the specific room type capacity",
        parent=room_node,
        critical=True
    )
    ref_room_claim = (
        f"The official room page(s) for '{resort.room_type_name}' confirm the capacity (5 or more)."
    )
    await evaluator.verify(
        claim=ref_room_claim,
        node=ref_room_leaf,
        sources=resort.room_urls,
        additional_instruction="Verify capacity text on the official room page(s)."
    )


# --------------------------------------------------------------------------- #
# Gen Z traveler age verification                                             #
# --------------------------------------------------------------------------- #
async def verify_gen_z_age(evaluator: Evaluator, parent_node) -> None:
    genz_node = evaluator.add_parallel(
        id="gen_z_traveler_age",
        desc="Confirm the primary traveler born in 2005 falls within Gen Z range",
        parent=parent_node,
        critical=True
    )

    # Birth year in Gen Z range 1997-2012 inclusive
    birth_leaf = evaluator.add_leaf(
        id="birth_year_range",
        desc="Birth year 2005 must be between 1997 and 2012 inclusive",
        parent=genz_node,
        critical=True
    )
    birth_claim = "People born in 2005 are part of Generation Z (defined roughly as 1997–2012 inclusive)."
    await evaluator.verify(
        claim=birth_claim,
        node=birth_leaf,
        additional_instruction=(
            "This is a simple definitional check. Accept the commonly used Gen Z range 1997–2012 (inclusive)."
        )
    )

    # Age in 2026 between 14 and 29
    age_leaf = evaluator.add_leaf(
        id="age_calculation_2026",
        desc="The calculated age in 2026 must be between 14 and 29 years old",
        parent=genz_node,
        critical=True
    )
    age_in_2026 = CURRENT_TRIP_YEAR - BIRTH_YEAR  # 21
    age_claim = (
        f"A person born in {BIRTH_YEAR} will be {age_in_2026} years old in {CURRENT_TRIP_YEAR}, "
        "which lies between 14 and 29 inclusive."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        additional_instruction="This is a straightforward math check; 2026 − 2005 = 21, and 21 is between 14 and 29."
    )


# --------------------------------------------------------------------------- #
# Caribbean alternative verification                                          #
# --------------------------------------------------------------------------- #
async def verify_caribbean_alternative(evaluator: Evaluator, parent_node, carib: CaribbeanExtraction) -> None:
    carib_node = evaluator.add_parallel(
        id="caribbean_alternative",
        desc="Identify a Caribbean destination alternative from Nashville (BNA) with direct flights and Level 1 travel advisory",
        parent=parent_node,
        critical=False
    )

    # Existence: destination, airline, and both flight/advisory URLs provided
    data_provided = bool(carib.destination_name and carib.airline and carib.flight_route_urls and carib.advisory_urls)
    evaluator.add_custom_node(
        result=data_provided,
        id="caribbean_data_provided",
        desc="Caribbean destination, airline, and official flight/advisory sources are provided",
        parent=carib_node,
        critical=False
    )

    # Reference URL presence node
    evaluator.add_custom_node(
        result=bool(carib.flight_route_urls and carib.advisory_urls),
        id="reference_url_caribbean",
        desc="Provide valid URL references for flight routes and travel advisory",
        parent=carib_node,
        critical=False
    )

    # Destination is Caribbean
    dest_leaf = evaluator.add_leaf(
        id="caribbean_destination",
        desc="A specific Caribbean island/destination accessible from BNA",
        parent=carib_node,
        critical=False
    )
    dest_claim = f"{carib.destination_name} is a Caribbean destination."
    union_sources = (carib.flight_route_urls or []) + (carib.advisory_urls or [])
    await evaluator.verify(
        claim=dest_claim,
        node=dest_leaf,
        sources=union_sources,
        additional_instruction="Confirm the destination is geographically part of the Caribbean region."
    )

    # Direct flight from BNA in 2026
    direct_leaf = evaluator.add_leaf(
        id="bna_direct_flight",
        desc="The destination must have direct flight service from Nashville (BNA) in 2026",
        parent=carib_node,
        critical=False
    )
    direct_claim = f"There is direct (nonstop) flight service from Nashville International Airport (BNA) to {carib.destination_name} in {CURRENT_TRIP_YEAR}."
    await evaluator.verify(
        claim=direct_claim,
        node=direct_leaf,
        sources=carib.flight_route_urls,
        additional_instruction="Use official airline or airport route pages to confirm nonstop service is operated in 2026 (seasonal or year-round both acceptable)."
    )

    # Airline operating the route
    airline_leaf = evaluator.add_leaf(
        id="carrier_information",
        desc="Identify which airline(s) operate the direct route from BNA",
        parent=carib_node,
        critical=False
    )
    airline_claim = f"The direct BNA–{carib.destination_name} route is operated by {carib.airline}."
    await evaluator.verify(
        claim=airline_claim,
        node=airline_leaf,
        sources=carib.flight_route_urls,
        additional_instruction="Confirm the carrier name on the official route page(s)."
    )

    # Level 1 travel advisory
    level_leaf = evaluator.add_leaf(
        id="level_1_advisory",
        desc="The destination must have a U.S. State Department Level 1 travel advisory (Exercise Normal Precautions)",
        parent=carib_node,
        critical=False
    )
    level_claim = (
        f"The U.S. State Department Travel Advisory level for {carib.country_name or carib.destination_name} "
        "is Level 1: Exercise Normal Precautions."
    )
    await evaluator.verify(
        claim=level_claim,
        node=level_leaf,
        sources=carib.advisory_urls,
        additional_instruction="Confirm the Advisory Level is explicitly 'Level 1: Exercise Normal Precautions' on the official travel.state.gov page."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Build and execute the verification tree for the Gen Z group trip planning task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates independent sub-requirements
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

    # Extract structured info in parallel
    florida_task = evaluator.extract(
        prompt=prompt_extract_florida(),
        template_class=FloridaExtraction,
        extraction_name="florida_extraction"
    )
    resort_task = evaluator.extract(
        prompt=prompt_extract_resort(),
        template_class=ResortExtraction,
        extraction_name="resort_extraction"
    )
    caribbean_task = evaluator.extract(
        prompt=prompt_extract_caribbean(),
        template_class=CaribbeanExtraction,
        extraction_name="caribbean_extraction"
    )

    florida, resort, carib = await asyncio.gather(florida_task, resort_task, caribbean_task)

    # Build verification subtrees
    await verify_florida_destination(evaluator, root, florida)
    await verify_dollywood_resort(evaluator, root, resort)
    await verify_room_type_specification(evaluator, root, resort)
    await verify_gen_z_age(evaluator, root)
    await verify_caribbean_alternative(evaluator, root, carib)

    # Return summary
    return evaluator.get_summary()