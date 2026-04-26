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
TASK_ID = "wkc_2026_service_dog_planning"
TASK_DESCRIPTION = (
    "I plan to attend the 2026 Westminster Kennel Club Dog Show in New York City with my service dog and need "
    "comprehensive information for trip planning. Please provide: (1) The exact dates and venues where the 2026 Westminster Dog Show will take place, "
    "(2) The ADA requirements and policies for bringing a service dog to the event venues, including control requirements and access rights, "
    "(3) Hotel accommodation policies for service dogs under ADA regulations, specifically whether hotels can charge pet fees or restrict access, "
    "(4) Information about 24-hour emergency veterinary services available in New York City, and "
    "(5) Standard safety and vaccination requirements for dog parks in New York City, "
    "in case I need exercise facilities for my service dog. For each category, please provide specific details and reference URLs to support your information."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventDetails(BaseModel):
    dates: List[str] = Field(default_factory=list)
    daytime_breed_venue: Optional[str] = None
    evening_venue: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ADARequirements(BaseModel):
    control_requirement: Optional[str] = None
    venue_access_rights: Optional[str] = None
    documentation_policy: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HotelPolicies(BaseModel):
    hotel_obligation: Optional[str] = None
    no_pet_fees: Optional[str] = None
    access_restrictions: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class EmergencyVetServices(BaseModel):
    availability_statement: Optional[str] = None
    example_providers: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class DogParkStandards(BaseModel):
    fencing_requirement: Optional[str] = None
    double_gate_entry: Optional[str] = None
    vaccination_requirements: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class TripPlanningExtraction(BaseModel):
    event: Optional[EventDetails] = None
    ada: Optional[ADARequirements] = None
    hotel: Optional[HotelPolicies] = None
    emergency: Optional[EmergencyVetServices] = None
    dog_park: Optional[DogParkStandards] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_planning() -> str:
    return """
    Extract structured information from the answer across the following five categories. Return exactly what the answer states (do not invent). 
    Also extract the reference URLs explicitly mentioned for each category.

    1) 2026 Westminster Dog Show – dates and venues:
       - dates: List all specific date statements mentioned (e.g., "May 11, 2026", "May 12, 2026", etc.). If a range is stated, include each date string as provided in the answer or include the range string if individual dates are not listed.
       - daytime_breed_venue: The venue name for daytime breed judging.
       - evening_venue: The venue name for evening competitions (e.g., Group and Best in Show).
       - urls: All supporting reference URLs cited for dates/venues.

    2) ADA service-dog requirements for event venues:
       - control_requirement: The ADA requirement for harness/leash/tether, including the exception if these interfere with the animal’s work or cannot be used.
       - venue_access_rights: Statement that service animals must be permitted in areas where the public/customers are normally allowed.
       - documentation_policy: Statement that venues/hotels cannot require documentation or certification for service animals.
       - urls: All supporting reference URLs cited for ADA requirements.

    3) Hotel policies under ADA:
       - hotel_obligation: Statement that hotels must accommodate service dogs under ADA regulations.
       - no_pet_fees: Statement that hotels cannot charge pet fees or deposits for service animals.
       - access_restrictions: Statement that service dogs must be allowed in guest rooms and common areas open to guests (no improper restrictions).
       - urls: All supporting reference URLs cited for hotel ADA policies.

    4) NYC 24-hour emergency veterinary services:
       - availability_statement: Statement confirming 24-hour emergency veterinary services are available in NYC.
       - example_providers: List of any example facility/service names mentioned (e.g., "BluePearl Midtown", "AMC NYC").
       - urls: All supporting reference URLs cited for emergency veterinary services.

    5) NYC dog park standards:
       - fencing_requirement: Standard dog-park fencing requirement (e.g., complete perimeter fencing or barriers preventing exit).
       - double_gate_entry: The standard double-gate entry system statement.
       - vaccination_requirements: A list of typical vaccination requirements mentioned (e.g., "rabies", "DHPP", "bordetella").
       - urls: All supporting reference URLs cited for dog park safety/vaccination requirements.

    Rules:
    - Extract only what appears in the answer text; if something is missing, return null or an empty list as appropriate.
    - For URLs, return valid URLs explicitly present in the answer (plain or markdown links). Do not infer or fabricate.
    - Keep strings exactly as stated in the answer; do not rephrase.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _join_items(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_event_details(evaluator: Evaluator, root_node, data: TripPlanningExtraction) -> None:
    node = evaluator.add_parallel(
        id="event_details",
        desc="Exact dates and venues for the 2026 Westminster Dog Show, with supporting sources.",
        parent=root_node,
        critical=False  # Allow partial credit across categories at root
    )

    event = data.event or EventDetails()

    # Leaf: show_dates (verify dates statement against sources)
    show_dates_leaf = evaluator.add_leaf(
        id="show_dates",
        desc="Provide the three specific dates when the 2026 Westminster Dog Show will take place.",
        parent=node,
        critical=True
    )
    dates_str = _join_items(event.dates)
    show_dates_claim = (
        f"The 2026 Westminster Kennel Club Dog Show will take place on the following date statements: {dates_str}."
        if dates_str else
        "The answer provides the specific date statements for the 2026 Westminster Kennel Club Dog Show."
    )
    await evaluator.verify(
        claim=show_dates_claim,
        node=show_dates_leaf,
        sources=event.urls,
        additional_instruction="Verify that the cited sources explicitly show the 2026 event dates; allow normal date-format variations. If three distinct dates are claimed, ensure all are present or implied."
    )

    # Leaf: primary_venue_breed_judging
    breed_venue_leaf = evaluator.add_leaf(
        id="primary_venue_breed_judging",
        desc="Identify the venue where daytime breed judging will occur.",
        parent=node,
        critical=True
    )
    breed_claim = (
        f"Daytime breed judging for the 2026 Westminster Kennel Club Dog Show will occur at {event.daytime_breed_venue}."
        if event.daytime_breed_venue else
        "The answer identifies the venue for daytime breed judging at the 2026 Westminster Kennel Club Dog Show."
    )
    await evaluator.verify(
        claim=breed_claim,
        node=breed_venue_leaf,
        sources=event.urls,
        additional_instruction="Check the official event schedule or announcements to confirm the daytime breed judging venue."
    )

    # Leaf: evening_venue
    evening_venue_leaf = evaluator.add_leaf(
        id="evening_venue",
        desc="Identify the venue where evening competitions will take place.",
        parent=node,
        critical=True
    )
    evening_claim = (
        f"Evening competitions (e.g., Group and Best in Show) for the 2026 Westminster Kennel Club Dog Show will take place at {event.evening_venue}."
        if event.evening_venue else
        "The answer identifies the venue for evening competitions at the 2026 Westminster Kennel Club Dog Show."
    )
    await evaluator.verify(
        claim=evening_claim,
        node=evening_venue_leaf,
        sources=event.urls,
        additional_instruction="Confirm the evening session venue (Group and Best in Show) via the cited sources."
    )

    # Leaf: event_reference_url (existence check)
    evaluator.add_custom_node(
        result=(len(event.urls) > 0),
        id="event_reference_url",
        desc="Provide at least one supporting reference URL for the 2026 event dates/venues.",
        parent=node,
        critical=True
    )


async def verify_service_dog_requirements(evaluator: Evaluator, root_node, data: TripPlanningExtraction) -> None:
    node = evaluator.add_parallel(
        id="service_dog_requirements",
        desc="ADA requirements and policies for bringing a service dog to the event venues, with supporting sources.",
        parent=root_node,
        critical=False
    )
    ada = data.ada or ADARequirements()

    # ADA control requirement
    control_leaf = evaluator.add_leaf(
        id="ada_control_requirement",
        desc="State the ADA control requirement for service animals (harness/leash/tether with applicable exception).",
        parent=node,
        critical=True
    )
    control_claim = (
        f"The ADA service-animal control requirement is: \"{ada.control_requirement}\"."
        if ada.control_requirement else
        "The answer states the ADA service-animal control requirement regarding harness/leash/tether and the exception."
    )
    await evaluator.verify(
        claim=control_claim,
        node=control_leaf,
        sources=ada.urls,
        additional_instruction=(
            "Confirm that ADA Title II/III require service animals to be harnessed, leashed, or tethered unless these devices "
            "interfere with the service animal’s work or the person’s disability prevents using them; in such cases, control must be maintained through voice, signal, or other effective means."
        )
    )

    # Venue access rights
    access_leaf = evaluator.add_leaf(
        id="venue_access_rights",
        desc="State that service animals must be permitted in all areas where the public/customers are normally allowed.",
        parent=node,
        critical=True
    )
    access_claim = (
        f"The ADA venue access rule stated is: \"{ada.venue_access_rights}\"."
        if ada.venue_access_rights else
        "The answer states that service animals must be permitted in areas open to the public/customers."
    )
    await evaluator.verify(
        claim=access_claim,
        node=access_leaf,
        sources=ada.urls,
        additional_instruction="Verify ADA language requiring service animals be permitted in areas where the public/customers are allowed."
    )

    # Documentation policy
    doc_leaf = evaluator.add_leaf(
        id="documentation_policy",
        desc="State that venues and hotels cannot require documentation or certification for service animals.",
        parent=node,
        critical=True
    )
    doc_claim = (
        f"The ADA documentation policy stated is: \"{ada.documentation_policy}\"."
        if ada.documentation_policy else
        "The answer states that staff cannot require documentation/certification for service animals."
    )
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=ada.urls,
        additional_instruction="Confirm ADA guidance that entities cannot require documentation/certification for a service animal; only limited questions are permitted."
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=(len(ada.urls) > 0),
        id="ada_reference_url",
        desc="Provide at least one supporting reference URL for the stated ADA service animal requirements.",
        parent=node,
        critical=True
    )


async def verify_hotel_policies(evaluator: Evaluator, root_node, data: TripPlanningExtraction) -> None:
    node = evaluator.add_parallel(
        id="hotel_service_dog_policies",
        desc="Hotel accommodation policies for service dogs under ADA (fees, access), with supporting sources.",
        parent=root_node,
        critical=False
    )
    hotel = data.hotel or HotelPolicies()

    # ADA hotel obligation
    obligation_leaf = evaluator.add_leaf(
        id="ada_hotel_obligation",
        desc="Confirm hotels must accommodate service dogs under ADA regulations.",
        parent=node,
        critical=True
    )
    obligation_claim = (
        f"The ADA hotel obligation stated is: \"{hotel.hotel_obligation}\"."
        if hotel.hotel_obligation else
        "The answer states that hotels must accommodate service dogs under ADA Title III."
    )
    await evaluator.verify(
        claim=obligation_claim,
        node=obligation_leaf,
        sources=hotel.urls,
        additional_instruction="Verify that hotels (public accommodations) must allow service animals under ADA Title III."
    )

    # No pet fees
    no_fees_leaf = evaluator.add_leaf(
        id="no_pet_fees",
        desc="Confirm hotels cannot charge additional pet fees or deposits for service animals.",
        parent=node,
        critical=True
    )
    no_fees_claim = (
        f"The no-pet-fee policy stated is: \"{hotel.no_pet_fees}\"."
        if hotel.no_pet_fees else
        "The answer states that hotels cannot charge pet fees or deposits for service animals."
    )
    await evaluator.verify(
        claim=no_fees_claim,
        node=no_fees_leaf,
        sources=hotel.urls,
        additional_instruction="Confirm ADA guidance that surcharges/pet fees cannot be imposed on service animals."
    )

    # Access restrictions
    access_rest_leaf = evaluator.add_leaf(
        id="access_restrictions",
        desc="Confirm service dogs must be allowed in guest rooms and common areas where guests are normally permitted.",
        parent=node,
        critical=True
    )
    access_rest_claim = (
        f"The access policy stated is: \"{hotel.access_restrictions}\"."
        if hotel.access_restrictions else
        "The answer states that service dogs must be allowed in guest rooms and common areas where guests are permitted."
    )
    await evaluator.verify(
        claim=access_rest_claim,
        node=access_rest_leaf,
        sources=hotel.urls,
        additional_instruction="Confirm service animals must be allowed in areas open to guests (guest rooms and common areas)."
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=(len(hotel.urls) > 0),
        id="hotel_policy_reference_url",
        desc="Provide at least one supporting reference URL for hotel ADA service-animal policies.",
        parent=node,
        critical=True
    )


async def verify_emergency_vet(evaluator: Evaluator, root_node, data: TripPlanningExtraction) -> None:
    node = evaluator.add_parallel(
        id="emergency_veterinary_services",
        desc="Information about 24-hour emergency veterinary services available in NYC, with supporting sources.",
        parent=root_node,
        critical=False
    )
    emergency = data.emergency or EmergencyVetServices()

    # Availability statement
    availability_leaf = evaluator.add_leaf(
        id="nyc_24_hour_emergency_availability",
        desc="State that 24-hour emergency veterinary services are available in New York City.",
        parent=node,
        critical=True
    )
    availability_claim = (
        f"The answer states: \"{emergency.availability_statement}\"."
        if emergency.availability_statement else
        "The answer states that 24-hour emergency veterinary services are available in New York City."
    )
    await evaluator.verify(
        claim=availability_claim,
        node=availability_leaf,
        sources=emergency.urls,
        additional_instruction="Verify that the cited sources indicate 24/7 emergency veterinary services exist in NYC."
    )

    # Optional example provider
    example_leaf = evaluator.add_leaf(
        id="example_emergency_provider_optional",
        desc="Optionally provide at least one NYC 24-hour emergency veterinary facility/service by name.",
        parent=node,
        critical=False
    )
    provider_name = emergency.example_providers[0] if emergency.example_providers else ""
    provider_claim = (
        f"An example 24-hour emergency veterinary facility in NYC is {provider_name}."
        if provider_name else
        "The answer provides at least one named example of a 24-hour emergency veterinary facility in NYC."
    )
    await evaluator.verify(
        claim=provider_claim,
        node=example_leaf,
        sources=emergency.urls,
        additional_instruction="Check whether any of the cited URLs identify a specific NYC 24-hour emergency veterinary provider by name."
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=(len(emergency.urls) > 0),
        id="emergency_vet_reference_url",
        desc="Provide at least one supporting reference URL for the NYC 24-hour emergency veterinary service information.",
        parent=node,
        critical=True
    )


async def verify_dog_park_standards(evaluator: Evaluator, root_node, data: TripPlanningExtraction) -> None:
    node = evaluator.add_parallel(
        id="dog_park_standards",
        desc="Standard safety and vaccination requirements for NYC dog parks, with supporting sources.",
        parent=root_node,
        critical=False
    )
    park = data.dog_park or DogParkStandards()

    # Fencing requirement
    fencing_leaf = evaluator.add_leaf(
        id="fencing_requirement",
        desc="State the standard dog-park fencing requirement (complete perimeter fencing or natural barriers preventing exit).",
        parent=node,
        critical=True
    )
    fencing_claim = (
        f"The dog-park fencing standard stated is: \"{park.fencing_requirement}\"."
        if park.fencing_requirement else
        "The answer states that dog parks must have complete perimeter fencing or effective barriers preventing exit."
    )
    await evaluator.verify(
        claim=fencing_claim,
        node=fencing_leaf,
        sources=park.urls,
        additional_instruction="Verify that the cited sources indicate perimeter fencing or effective barriers are standard requirements for dog parks."
    )

    # Double-gate entry
    double_gate_leaf = evaluator.add_leaf(
        id="double_gate_entry",
        desc="State the standard double-gate entry system safety feature for dog parks.",
        parent=node,
        critical=True
    )
    double_gate_claim = (
        f"The double-gate entry standard stated is: \"{park.double_gate_entry}\"."
        if park.double_gate_entry else
        "The answer states that dog parks typically use a double-gate entry system for safety."
    )
    await evaluator.verify(
        claim=double_gate_claim,
        node=double_gate_leaf,
        sources=park.urls,
        additional_instruction="Verify that the cited sources mention double-gate entry systems as standard dog-park safety design."
    )

    # Vaccination requirements
    vaccination_leaf = evaluator.add_leaf(
        id="vaccination_requirements",
        desc="List typical vaccination requirements mentioned in constraints (rabies, DHPP, bordetella).",
        parent=node,
        critical=True
    )
    vaccines_str = _join_items(park.vaccination_requirements)
    vaccination_claim = (
        f"Typical dog-park vaccination requirements include: {vaccines_str}."
        if vaccines_str else
        "The answer lists typical dog-park vaccination requirements (e.g., rabies, DHPP, bordetella)."
    )
    await evaluator.verify(
        claim=vaccination_claim,
        node=vaccination_leaf,
        sources=park.urls,
        additional_instruction="Confirm that the cited sources mention typical dog-park vaccination expectations such as rabies, core DHPP, and bordetella."
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=(len(park.urls) > 0),
        id="dog_park_reference_url",
        desc="Provide at least one supporting reference URL for dog park safety/vaccination requirements.",
        parent=node,
        critical=True
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
    Evaluate an answer for the WKC 2026 service dog trip-planning task.
    """
    # Initialize evaluator (root parallel, non-critical for partial scoring across categories)
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_trip_planning(),
        template_class=TripPlanningExtraction,
        extraction_name="trip_planning_extraction"
    )

    # Build verification tree according to rubric
    await verify_event_details(evaluator, root, extraction)
    await verify_service_dog_requirements(evaluator, root, extraction)
    await verify_hotel_policies(evaluator, root, extraction)
    await verify_emergency_vet(evaluator, root, extraction)
    await verify_dog_park_standards(evaluator, root, extraction)

    # Return the evaluation summary
    return evaluator.get_summary()