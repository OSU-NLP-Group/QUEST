import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "denver_to_grand_cayman_weekend_2026"
TASK_DESCRIPTION = (
    "You are planning a weekend getaway from Denver, Colorado to Grand Cayman, Cayman Islands for February 2026. "
    "You need to arrange the following:\n\n"
    "1. Flight: Identify a direct (nonstop) flight option from Denver International Airport (DEN) to Grand Cayman "
    "(Owen Roberts International Airport, GCM) that operates on Saturdays. Provide the operating airline and include "
    "the flight schedule details if available.\n\n"
    "2. Accommodation: Identify two different 4-star hotels that are located on Seven Mile Beach in Grand Cayman and "
    "are beachfront properties. Provide the specific names of both hotels.\n\n"
    "3. Baggage: Determine the standard checked baggage weight limit (in pounds) for international flights to ensure "
    "your luggage complies with airline requirements.\n\n"
    "4. Entry Requirements: Identify the passport and visa requirements for U.S. citizens traveling to Grand Cayman "
    "for tourism purposes, including passport validity requirements.\n\n"
    "5. Transportation: Provide the approximate distance or travel time from Owen Roberts International Airport to the "
    "Seven Mile Beach hotel area.\n\n"
    "For each component of your answer, include reference URLs from reliable sources (airline websites, official tourism "
    "sites, hotel booking platforms, or government travel information) that support your findings."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FlightExtraction(BaseModel):
    airline: Optional[str] = None
    route_type: Optional[str] = None  # e.g., "direct", "nonstop"
    saturday_operation: Optional[str] = None  # mentions Saturday service if present
    departure_time: Optional[str] = None  # any format as provided
    arrival_time: Optional[str] = None
    duration: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Hotel(BaseModel):
    name: Optional[str] = None
    star_rating: Optional[str] = None  # e.g., "4-star", "★★★★"
    location_note: Optional[str] = None  # text mentioning Seven Mile Beach
    beachfront_note: Optional[str] = None  # text mentioning beachfront
    sources: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[Hotel] = Field(default_factory=list)


class BaggageExtraction(BaseModel):
    weight_limit_lbs: Optional[str] = None  # e.g., "50 lb", "50 pounds", "23 kg (50 lb)"
    dimension_limit_in: Optional[str] = None  # e.g., "62 linear inches"
    sources: List[str] = Field(default_factory=list)


class EntryExtraction(BaseModel):
    passport_required: Optional[str] = None  # e.g., "valid passport required"
    passport_validity: Optional[str] = None  # e.g., "valid for duration of stay"
    visa_required_for_us: Optional[str] = None  # e.g., "no visa required for 90 days"
    visa_duration_limit: Optional[str] = None  # e.g., "up to 90 days"
    entry_urls: List[str] = Field(default_factory=list)
    visa_urls: List[str] = Field(default_factory=list)


class TransferExtraction(BaseModel):
    distance_or_time: Optional[str] = None  # e.g., "10-15 minutes", "~10 km"
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flight() -> str:
    return (
        "From the answer, extract the details about a direct (nonstop) flight from Denver (DEN) to Grand Cayman (GCM) "
        "that operates on Saturdays.\n"
        "Return the following fields:\n"
        "- airline: the operating airline name explicitly mentioned.\n"
        "- route_type: whether the answer states 'direct' or 'nonstop' for the DEN–GCM flight.\n"
        "- saturday_operation: any text that indicates the flight operates on Saturdays (e.g., 'Saturday service').\n"
        "- departure_time: the stated departure time from DEN, if provided.\n"
        "- arrival_time: the stated arrival time at GCM, if provided.\n"
        "- duration: the stated flight duration, if provided.\n"
        "- sources: all URLs cited that support the flight details (airline schedule/route pages, etc.).\n"
        "If any field is not provided in the answer, return null for that field. Extract only URLs explicitly present in the answer."
    )


def prompt_extract_hotels() -> str:
    return (
        "From the answer, extract up to two hotel options that match: 4-star, located on Seven Mile Beach in Grand Cayman, "
        "and beachfront. If more than two hotels are provided, extract the first two. For each hotel, return:\n"
        "- name: the hotel's name as stated.\n"
        "- star_rating: the rating text (e.g., '4-star' or '★★★★').\n"
        "- location_note: text indicating the hotel is on Seven Mile Beach.\n"
        "- beachfront_note: text indicating the hotel is beachfront.\n"
        "- sources: URLs cited for the hotel (official site, booking platform, tourism site).\n"
        "If any field is missing for a hotel, set it to null. Extract only URLs explicitly present in the answer."
    )


def prompt_extract_baggage() -> str:
    return (
        "From the answer, extract the standard checked baggage policy for international flights as stated. Return:\n"
        "- weight_limit_lbs: the weight limit in pounds text (e.g., '50 pounds', '23 kg (50 lb)').\n"
        "- dimension_limit_in: the maximum linear dimensions text if provided (e.g., '62 linear inches').\n"
        "- sources: URLs cited that support the baggage policy.\n"
        "If any field is missing, return null for that field. Extract only URLs explicitly present in the answer."
    )


def prompt_extract_entry() -> str:
    return (
        "From the answer, extract entry requirements for U.S. citizens traveling to Grand Cayman for tourism. Return:\n"
        "- passport_required: text indicating a valid passport is required.\n"
        "- passport_validity: text about passport validity (e.g., 'valid for duration of stay').\n"
        "- visa_required_for_us: text indicating visa requirements (e.g., 'no visa required' for up to 90 days).\n"
        "- visa_duration_limit: if a duration is mentioned (e.g., 'up to 90 days'), return it.\n"
        "- entry_urls: URLs cited for passport requirements (government/tourism sites).\n"
        "- visa_urls: URLs cited for visa requirements (government/tourism sites).\n"
        "If any field is missing, return null for that field. Extract only URLs explicitly present in the answer."
    )


def prompt_extract_transfer() -> str:
    return (
        "From the answer, extract the approximate transfer distance or travel time from Owen Roberts International Airport "
        "(GCM) to the Seven Mile Beach hotel area. Return:\n"
        "- distance_or_time: the stated time or distance (e.g., '10-15 minutes', '~10 km').\n"
        "- sources: URLs cited that support this transfer information.\n"
        "If the answer does not provide this, return null and an empty sources list. Extract only URLs explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_flight(
    evaluator: Evaluator,
    parent_node,
    flight: FlightExtraction,
) -> None:
    """
    Build and verify the flight subtree under parent_node.
    """
    flight_node = evaluator.add_sequential(
        id="Direct_Flight_Selection",
        desc="Identify and verify a direct Saturday flight from Denver to Grand Cayman",
        parent=parent_node,
        critical=False,
    )

    # Flight Identification (parallel)
    flight_ident_node = evaluator.add_parallel(
        id="Flight_Identification",
        desc="Verify the existence and details of a direct Saturday flight",
        parent=flight_node,
        critical=False,
    )

    # URL Reference existence (critical within identification)
    urls_present = bool(flight.sources)
    url_ref_node = evaluator.add_custom_node(
        result=urls_present,
        id="URL_Reference_Flight",
        desc="Valid URL reference provided for flight information",
        parent=flight_ident_node,
        critical=True,
    )

    # Route Verification (parallel, critical)
    route_ver_node = evaluator.add_parallel(
        id="Route_Verification",
        desc="Confirm flight operates direct Denver to Grand Cayman on Saturdays",
        parent=flight_ident_node,
        critical=True,
    )

    # Airline identified (critical existence)
    airline_present = flight.airline is not None and flight.airline.strip() != ""
    airline_ident_node = evaluator.add_custom_node(
        result=airline_present,
        id="Airline_Identified",
        desc="Operating airline name is provided",
        parent=route_ver_node,
        critical=True,
    )

    # Direct Route Exists (critical verification)
    direct_route_leaf = evaluator.add_leaf(
        id="Direct_Route_Exists",
        desc="Flight is direct (nonstop) from Denver to Grand Cayman",
        parent=route_ver_node,
        critical=True,
    )
    claim_direct = (
        "There is a direct (nonstop) flight option between Denver International Airport (DEN) "
        "and Owen Roberts International Airport (GCM)."
    )
    await evaluator.verify(
        claim=claim_direct,
        node=direct_route_leaf,
        sources=flight.sources,
        additional_instruction=(
            "Verify the route is nonstop (no connections) from DEN to GCM. "
            "Accept seasonal Saturday service if evidence shows such operation."
        ),
    )

    # Saturday Operation (critical verification)
    saturday_leaf = evaluator.add_leaf(
        id="Saturday_Operation",
        desc="Flight operates on Saturdays",
        parent=route_ver_node,
        critical=True,
    )
    claim_sat = "This DEN–GCM nonstop flight operates on Saturdays."
    await evaluator.verify(
        claim=claim_sat,
        node=saturday_leaf,
        sources=flight.sources,
        additional_instruction=(
            "Check the schedule/calendar on the provided sources to confirm Saturday operation "
            "around February 2026. Minor seasonal variations are acceptable if Saturday service is indicated."
        ),
    )

    # Schedule Information (non-critical parallel)
    sched_node = evaluator.add_parallel(
        id="Schedule_Information",
        desc="Additional schedule details for the flight",
        parent=flight_node,
        critical=False,
    )

    # Departure time (non-critical)
    dep_leaf = evaluator.add_leaf(
        id="Departure_Time",
        desc="Departure time from Denver is provided",
        parent=sched_node,
        critical=False,
    )
    dep_claim = f"The flight's departure time from DEN is '{flight.departure_time}'." if flight.departure_time else \
        "The answer provides a specific departure time from DEN for this DEN–GCM flight."
    await evaluator.verify(
        claim=dep_claim,
        node=dep_leaf,
        sources=flight.sources,
        additional_instruction=(
            "Verify the provided departure time (if any) aligns with the schedule shown on the source pages. "
            "If the answer did not provide a specific time, mark this as incorrect."
        ),
    )

    # Arrival time (non-critical)
    arr_leaf = evaluator.add_leaf(
        id="Arrival_Time",
        desc="Arrival time in Grand Cayman is provided",
        parent=sched_node,
        critical=False,
    )
    arr_claim = f"The flight's arrival time at GCM is '{flight.arrival_time}'." if flight.arrival_time else \
        "The answer provides a specific arrival time at GCM for this DEN–GCM flight."
    await evaluator.verify(
        claim=arr_claim,
        node=arr_leaf,
        sources=flight.sources,
        additional_instruction=(
            "Verify the provided arrival time (if any) aligns with the schedule shown on the source pages. "
            "If the answer did not provide a specific time, mark this as incorrect."
        ),
    )

    # Flight duration (non-critical)
    dur_leaf = evaluator.add_leaf(
        id="Flight_Duration",
        desc="Flight duration is provided",
        parent=sched_node,
        critical=False,
    )
    dur_claim = f"The flight duration is '{flight.duration}'." if flight.duration else \
        "The answer provides a specific flight duration for the DEN–GCM nonstop flight."
    await evaluator.verify(
        claim=dur_claim,
        node=dur_leaf,
        sources=flight.sources,
        additional_instruction=(
            "Verify the provided duration (if any) aligns with the schedule or route timing shown on the source pages. "
            "If the answer did not provide a specific duration, mark this as incorrect."
        ),
    )


async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: Hotel,
    idx: int,
) -> None:
    """
    Build and verify a hotel subtree under parent_node for the given hotel.
    """
    base_id_prefix = f"Hotel_{idx+1}"

    hotel_node = evaluator.add_sequential(
        id="First_Hotel_Option" if idx == 0 else "Second_Hotel_Option",
        desc=f"Identify {'first' if idx == 0 else 'second'} 4-star beachfront hotel on Seven Mile Beach",
        parent=parent_node,
        critical=False,
    )

    qual_node = evaluator.add_parallel(
        id="Hotel_Qualifications",
        desc="Verify hotel meets all required criteria",
        parent=hotel_node,
        critical=False,
    )

    # Basic Information
    basic_node = evaluator.add_parallel(
        id="Basic_Information",
        desc="Verify hotel name and star rating",
        parent=qual_node,
        critical=True,
    )

    # Hotel name provided (critical existence)
    name_present = hotel.name is not None and hotel.name.strip() != ""
    name_leaf = evaluator.add_custom_node(
        result=name_present,
        id="Hotel_Name_Provided",
        desc="Specific hotel name is provided",
        parent=basic_node,
        critical=True,
    )

    # Four-star rating (critical verification)
    rating_leaf = evaluator.add_leaf(
        id="Four_Star_Rating",
        desc="Hotel has a 4-star rating",
        parent=basic_node,
        critical=True,
    )
    if name_present:
        claim_rating = f"The hotel '{hotel.name}' is a 4-star property."
    else:
        claim_rating = "This hotel is a 4-star property."
    await evaluator.verify(
        claim=claim_rating,
        node=rating_leaf,
        sources=hotel.sources,
        additional_instruction=(
            "Confirm that the hotel's rating is 4 stars as shown on the provided source (official site or reputable booking platform). "
            "Allow minor format variations like '★★★★'."
        ),
    )

    # Location Verification
    loc_node = evaluator.add_parallel(
        id="Location_Verification",
        desc="Verify hotel location meets requirements",
        parent=qual_node,
        critical=True,
    )

    # Seven Mile Beach (critical)
    smb_leaf = evaluator.add_leaf(
        id="Seven_Mile_Beach_Location",
        desc="Hotel is located on Seven Mile Beach",
        parent=loc_node,
        critical=True,
    )
    smb_claim = f"The hotel '{hotel.name}' is located on Seven Mile Beach." if name_present else \
        "The hotel is located on Seven Mile Beach."
    await evaluator.verify(
        claim=smb_claim,
        node=smb_leaf,
        sources=hotel.sources,
        additional_instruction=(
            "Verify that the hotel's location is explicitly stated as on Seven Mile Beach on the source page."
        ),
    )

    # Beachfront property (critical)
    bf_leaf = evaluator.add_leaf(
        id="Beachfront_Property",
        desc="Hotel is a beachfront property",
        parent=loc_node,
        critical=True,
    )
    bf_claim = f"The hotel '{hotel.name}' is a beachfront property." if name_present else \
        "The hotel is a beachfront property."
    await evaluator.verify(
        claim=bf_claim,
        node=bf_leaf,
        sources=hotel.sources,
        additional_instruction=(
            "Verify that the hotel is beachfront (directly on the beach) per the provided source page."
        ),
    )

    # URL Reference existence (critical in qualifications)
    hotel_urls_present = bool(hotel.sources)
    url_ref_hotel = evaluator.add_custom_node(
        result=hotel_urls_present,
        id="URL_Reference_Hotel1" if idx == 0 else "URL_Reference_Hotel2",
        desc="Valid URL reference provided for hotel information",
        parent=qual_node,
        critical=True,
    )


async def verify_baggage(
    evaluator: Evaluator,
    parent_node,
    baggage: BaggageExtraction,
) -> None:
    """
    Build and verify baggage requirements subtree.
    """
    bag_node = evaluator.add_parallel(
        id="Baggage_Requirements",
        desc="Identify baggage weight and size restrictions for the flight",
        parent=parent_node,
        critical=False,
    )

    weight_node = evaluator.add_parallel(
        id="Weight_Limit_Identification",
        desc="Standard checked baggage weight limit is identified",
        parent=bag_node,
        critical=False,
    )

    # URL reference existence (critical within weight identification)
    bag_urls_present = bool(baggage.sources)
    bag_url_leaf = evaluator.add_custom_node(
        result=bag_urls_present,
        id="URL_Reference_Baggage",
        desc="Valid URL reference provided for baggage policy",
        parent=weight_node,
        critical=True,
    )

    # Standard weight limit (critical verification)
    wt_leaf = evaluator.add_leaf(
        id="Standard_Weight_Limit",
        desc="Weight limit of 50 pounds (23 kg) or less for standard checked baggage",
        parent=weight_node,
        critical=True,
    )
    claim_wt = (
        "The standard checked baggage weight limit for international economy tickets is 50 pounds (23 kg). "
        "Some airlines may set limits at or below 50 lb for standard checked bags."
    )
    await evaluator.verify(
        claim=claim_wt,
        node=wt_leaf,
        sources=baggage.sources,
        additional_instruction=(
            "Verify the standard checked baggage weight limit; most airlines set 50 lb (23 kg) for standard checked bags. "
            "Accept equivalent phrasing like 'up to 50 pounds' or '23 kg'."
        ),
    )

    # Size restrictions (non-critical)
    size_node = evaluator.add_parallel(
        id="Size_Restrictions",
        desc="Baggage size restrictions are provided",
        parent=bag_node,
        critical=False,
    )

    dim_leaf = evaluator.add_leaf(
        id="Linear_Dimension_Limit",
        desc="Maximum linear dimensions (62 inches total) are mentioned",
        parent=size_node,
        critical=False,
    )
    claim_dim = (
        "The maximum linear dimensions for a standard checked bag are approximately 62 inches (length + width + height)."
    )
    await evaluator.verify(
        claim=claim_dim,
        node=dim_leaf,
        sources=baggage.sources,
        additional_instruction=(
            "Confirm that the policy mentions 62 linear inches (or an equivalent standard) for checked bag dimensions."
        ),
    )


async def verify_entry_requirements(
    evaluator: Evaluator,
    parent_node,
    entry: EntryExtraction,
) -> None:
    """
    Build and verify entry requirements subtree.
    """
    entry_node = evaluator.add_parallel(
        id="Entry_Requirements",
        desc="Identify passport and visa requirements for US citizens traveling to Grand Cayman",
        parent=parent_node,
        critical=False,
    )

    # Passport Requirements
    passport_node = evaluator.add_parallel(
        id="Passport_Requirements",
        desc="Verify passport requirements for entry",
        parent=entry_node,
        critical=True,
    )

    entry_urls_present = bool(entry.entry_urls)
    url_entry_leaf = evaluator.add_custom_node(
        result=entry_urls_present,
        id="URL_Reference_Entry",
        desc="Valid URL reference provided for entry requirements",
        parent=passport_node,
        critical=True,
    )

    pass_req_leaf = evaluator.add_leaf(
        id="Valid_Passport_Required",
        desc="Valid passport is required for entry",
        parent=passport_node,
        critical=True,
    )
    claim_pass_req = "U.S. citizens are required to present a valid passport to enter the Cayman Islands."
    await evaluator.verify(
        claim=claim_pass_req,
        node=pass_req_leaf,
        sources=entry.entry_urls,
        additional_instruction=(
            "Use official government or tourism sources to confirm that a valid passport is required for entry."
        ),
    )

    pass_valid_leaf = evaluator.add_leaf(
        id="Passport_Validity_Period",
        desc="Passport must be valid for duration of stay",
        parent=passport_node,
        critical=True,
    )
    claim_valid = "For tourism entry into the Cayman Islands, a U.S. citizen's passport must be valid for the duration of stay."
    await evaluator.verify(
        claim=claim_valid,
        node=pass_valid_leaf,
        sources=entry.entry_urls,
        additional_instruction=(
            "Confirm passport validity requirements; accept phrasing like 'valid for duration of stay' if present on official sources."
        ),
    )

    # Visa Requirements
    visa_node = evaluator.add_parallel(
        id="Visa_Requirements",
        desc="Verify visa requirements for US citizens",
        parent=entry_node,
        critical=True,
    )

    visa_urls_present = bool(entry.visa_urls)
    url_visa_leaf = evaluator.add_custom_node(
        result=visa_urls_present,
        id="URL_Reference_Visa",
        desc="Valid URL reference provided for visa requirements",
        parent=visa_node,
        critical=True,
    )

    visa_req_leaf = evaluator.add_leaf(
        id="US_Citizens_Visa_Status",
        desc="US citizens do not require a visa for Grand Cayman (tourism, up to 90 days)",
        parent=visa_node,
        critical=True,
    )
    visa_claim = (
        "U.S. citizens do not require a visa for tourism visits to the Cayman Islands for stays up to approximately 90 days."
    )
    await evaluator.verify(
        claim=visa_claim,
        node=visa_req_leaf,
        sources=entry.visa_urls,
        additional_instruction=(
            "Confirm on official sources (e.g., Cayman Islands Government immigration or U.S. State Department) "
            "that U.S. tourists can enter visa-free for short stays (around 90 days)."
        ),
    )


async def verify_transfer(
    evaluator: Evaluator,
    parent_node,
    transfer: TransferExtraction,
) -> None:
    """
    Build and verify airport transfer information subtree.
    """
    transfer_node = evaluator.add_parallel(
        id="Airport_Transfer_Information",
        desc="Provide information about airport to hotel area transfer",
        parent=parent_node,
        critical=False,
    )

    dist_node = evaluator.add_parallel(
        id="Transfer_Distance",
        desc="Distance from Owen Roberts Airport to Seven Mile Beach is provided",
        parent=transfer_node,
        critical=False,
    )

    # URL reference existence (critical under transfer distance)
    transfer_urls_present = bool(transfer.sources)
    url_transfer_leaf = evaluator.add_custom_node(
        result=transfer_urls_present,
        id="URL_Reference_Transfer",
        desc="Valid URL reference provided for transfer information",
        parent=dist_node,
        critical=True,
    )

    # Distance/Time specification (non-critical)
    dist_leaf = evaluator.add_leaf(
        id="Distance_Specification",
        desc="Distance is approximately 10-15 minutes or 10 km",
        parent=dist_node,
        critical=False,
    )
    claim_dist = (
        "The transfer from Owen Roberts International Airport (GCM) to the Seven Mile Beach hotel area is approximately "
        "10–15 minutes by car or roughly around 10 km."
    )
    await evaluator.verify(
        claim=claim_dist,
        node=dist_leaf,
        sources=transfer.sources,
        additional_instruction=(
            "Verify that typical travel time or distance from GCM to Seven Mile Beach is in the ~10–15 minutes or ~10 km range. "
            "Allow reasonable approximations from tourism/transport pages."
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
    Evaluate the provided answer for the Denver to Grand Cayman weekend travel plan.
    """
    # Initialize evaluator (root should be non-critical to allow partial credit across components)
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

    # Extract structured data from the answer
    flight_extraction = await evaluator.extract(
        prompt=prompt_extract_flight(),
        template_class=FlightExtraction,
        extraction_name="flight_info",
    )
    hotels_extraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_info",
    )
    baggage_extraction = await evaluator.extract(
        prompt=prompt_extract_baggage(),
        template_class=BaggageExtraction,
        extraction_name="baggage_info",
    )
    entry_extraction = await evaluator.extract(
        prompt=prompt_extract_entry(),
        template_class=EntryExtraction,
        extraction_name="entry_requirements",
    )
    transfer_extraction = await evaluator.extract(
        prompt=prompt_extract_transfer(),
        template_class=TransferExtraction,
        extraction_name="transfer_info",
    )

    # Build and verify flight subtree
    await verify_flight(evaluator, root, flight_extraction)

    # Verify hotels: take first two; pad if fewer
    hotels: List[Hotel] = hotels_extraction.hotels[:2]
    while len(hotels) < 2:
        hotels.append(Hotel())
    await verify_hotel(evaluator, root, hotels[0], idx=0)
    await verify_hotel(evaluator, root, hotels[1], idx=1)

    # Verify baggage
    await verify_baggage(evaluator, root, baggage_extraction)

    # Verify entry requirements
    await verify_entry_requirements(evaluator, root, entry_extraction)

    # Verify transfer information
    await verify_transfer(evaluator, root, transfer_extraction)

    # Return evaluation summary
    return evaluator.get_summary()