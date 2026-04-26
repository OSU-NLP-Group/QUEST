import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "atl_trip_planning"
TASK_DESCRIPTION = """You are planning a 5-day business trip to Atlanta and will be flying in and out of Hartsfield-Jackson Atlanta International Airport (ATL). You need to gather comprehensive information for your trip logistics.

Please provide the following information:

Hotel Accommodations:
Identify three different hotels that meet ALL of the following criteria:
- Located within 5 miles of Hartsfield-Jackson Atlanta International Airport
- Offer complimentary (free) airport shuttle service
- Shuttle service runs at least every 30 minutes, or provides on-demand service
For each hotel, provide: hotel name, confirmation of shuttle service, shuttle frequency/schedule, and a reference URL.

Parking Information:
- What is the total cost to park for 5 consecutive days in the on-airport Economy parking lot at ATL? (Provide the daily rate and total cost calculation)
- Provide a reference URL for the parking rates.

Rental Car Center:
- What is the complete address of the ATL Rental Car Center?
- What transportation method is available from the airport terminals to the Rental Car Center, and what is the cost?
- Provide a reference URL for this information.

TSA PreCheck:
- Confirm whether TSA PreCheck lanes are available at the Domestic Terminal South checkpoint.
- What are the operating hours for TSA PreCheck at this location?
- Provide a reference URL.

Airport Lounges:
- Confirm whether there is a Delta Sky Club in Concourse A.
- In which specific concourse is the United Club located at ATL?
- Name and location (concourse) of at least one lounge that accepts Priority Pass membership.
- Provide a reference URL for lounge information.

Baggage Claim:
- Where is the international baggage claim located? (Specify the terminal and/or concourse)
- Provide a reference URL.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    shuttle_service_confirmation: Optional[str] = None
    shuttle_frequency: Optional[str] = None
    reference_url: Optional[str] = None


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


class ParkingInfo(BaseModel):
    economy_daily_rate: Optional[str] = None
    total_cost_5_days: Optional[str] = None
    reference_url: Optional[str] = None


class RentalCarInfo(BaseModel):
    address: Optional[str] = None
    transport_method: Optional[str] = None
    transport_cost: Optional[str] = None
    reference_url: Optional[str] = None


class TSAPrecheckInfo(BaseModel):
    availability_confirmation: Optional[str] = None
    hours: Optional[str] = None
    reference_url: Optional[str] = None


class LoungesInfo(BaseModel):
    delta_sky_club_concourse_a: Optional[str] = None
    united_club_concourse: Optional[str] = None
    priority_pass_lounge_name: Optional[str] = None
    priority_pass_lounge_concourse: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class BaggageInfo(BaseModel):
    international_baggage_claim_location: Optional[str] = None
    reference_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
Extract up to three hotel entries that the answer claims meet ALL of these criteria:
- within 5 miles of Hartsfield-Jackson Atlanta International Airport (ATL),
- offer complimentary (free) airport shuttle service,
- shuttle runs at least every 30 minutes OR is on-demand.

For each hotel, extract:
- name: the hotel name exactly as stated,
- shuttle_service_confirmation: the exact text confirming complimentary airport shuttle service,
- shuttle_frequency: the described shuttle frequency or schedule (e.g., "every 20–30 minutes", "on demand", "24/7 on request"),
- reference_url: a URL provided in the answer for this hotel; must be an explicit URL present in the answer (plain or markdown link).

Return a JSON with a 'hotels' array of up to 3 items. If fewer than 3 are present, return whatever is available. If a field is missing, set it to null.
Apply the SPECIAL RULES FOR URL SOURCES EXTRACTION and URL EXTRACTION strictly.
"""


def prompt_extract_parking() -> str:
    return """
Extract the ATL on-airport Economy parking pricing details cited in the answer.
Return:
- economy_daily_rate: the daily rate string exactly as stated (e.g., "$14/day"),
- total_cost_5_days: the total 5-day cost as stated in the answer (e.g., "$70"),
- reference_url: the URL provided for the parking rates (must be an explicit URL present in the answer).

If any field is missing, set it to null. Do not invent values. Apply URL extraction rules strictly.
"""


def prompt_extract_rental_car() -> str:
    return """
Extract information about the ATL Rental Car Center from the answer:
- address: the complete street address as provided (include city/state/ZIP if present),
- transport_method: the method from the terminals to the Rental Car Center (e.g., "ATL SkyTrain"),
- transport_cost: the stated cost (e.g., "free"),
- reference_url: a URL cited for this information (must be explicitly present in the answer).

If a field is missing, return null for that field. Do not infer values. Apply URL extraction rules strictly.
"""


def prompt_extract_tsa_precheck() -> str:
    return """
Extract the TSA PreCheck details for the Domestic Terminal South checkpoint:
- availability_confirmation: the statement confirming whether TSA PreCheck lanes are available (e.g., "Yes, TSA PreCheck is available at Domestic Terminal South"),
- hours: the operating hours as stated in the answer (include qualifiers like "daily", "varies", or time ranges),
- reference_url: a URL cited for this information (must be explicitly present in the answer).

If a field is missing, set it to null. Do not invent values. Apply URL extraction rules strictly.
"""


def prompt_extract_lounges() -> str:
    return """
Extract the following lounge information cited in the answer:
- delta_sky_club_concourse_a: a confirmation statement indicating whether a Delta Sky Club exists in Concourse A (e.g., "Yes, Delta Sky Club in Concourse A"),
- united_club_concourse: the concourse where the United Club at ATL is located (e.g., "Concourse D"),
- priority_pass_lounge_name: the name of at least one lounge that accepts Priority Pass,
- priority_pass_lounge_concourse: the concourse where that Priority Pass lounge is located,
- reference_urls: an array of one or more URLs provided in the answer for lounge information (extract all lounge-related URLs).

If items are missing, set them to null (or empty array for URLs). Do not invent values. Apply URL extraction rules strictly.
"""


def prompt_extract_baggage() -> str:
    return """
Extract the international baggage claim location at ATL as stated in the answer:
- international_baggage_claim_location: a concise description of where the international baggage claim is located (terminal and/or concourse, e.g., "International Terminal, Concourse F"),
- reference_url: a URL cited for this information (must be explicitly present in the answer).

If any field is missing, set it to null. Apply URL extraction rules strictly.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal_name(index: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third"}
    return mapping.get(index, f"#{index + 1}")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    hotel_index: int,
) -> None:
    """
    Build and verify the hotel subtree for one hotel entry.
    """
    hotel_label = ordinal_name(hotel_index)
    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{hotel_index+1}",
        desc=f"{hotel_label} hotel option meeting all specified criteria",
        parent=parent_node,
        critical=True  # Critical within the hotels group
    )

    # Reference URL existence (gate all other checks)
    ref_url_ok = bool(hotel.reference_url and str(hotel.reference_url).strip())
    hotel_ref_node = evaluator.add_custom_node(
        result=ref_url_ok,
        id=f"Hotel_{hotel_index+1}_Reference_URL",
        desc=f"Valid reference URL for the hotel information",
        parent=hotel_node,
        critical=True
    )

    # Hotel name and within 5 miles
    name_loc_leaf = evaluator.add_leaf(
        id=f"Hotel_{hotel_index+1}_Name_And_Location",
        desc="Hotel name and verification it is within 5 miles of ATL",
        parent=hotel_node,
        critical=True
    )
    name_val = hotel.name or ""
    claim_name_loc = (
        f"This webpage is for the hotel named '{name_val}', and it indicates the property is within about 5 miles of "
        f"Hartsfield-Jackson Atlanta International Airport (ATL)."
    )
    await evaluator.verify(
        claim=claim_name_loc,
        node=name_loc_leaf,
        sources=hotel.reference_url if ref_url_ok else None,
        additional_instruction=(
            "Accept reasonable equivalents such as: the page shows the correct hotel name; and the location is 'near the airport', "
            "gives a distance ≤ 5 miles (or ≤ ~8 km), or explicitly states 'within 5 miles'. Mention like 'minutes from the airport' "
            "or 'adjacent to ATL' can be treated as within 5 miles if clearly implied."
        )
    )

    # Complimentary shuttle confirmation
    shuttle_leaf = evaluator.add_leaf(
        id=f"Hotel_{hotel_index+1}_Shuttle_Service",
        desc="Confirmation of complimentary airport shuttle service",
        parent=hotel_node,
        critical=True
    )
    claim_shuttle = (
        "This hotel offers complimentary (free) airport shuttle service to or from ATL."
    )
    await evaluator.verify(
        claim=claim_shuttle,
        node=shuttle_leaf,
        sources=hotel.reference_url if ref_url_ok else None,
        additional_instruction=(
            "Look for phrases like 'complimentary airport shuttle', 'free shuttle', 'free airport transportation'. "
            "If the page indicates a 'shuttle fee' or only 'local area shuttle' without airport service, this should fail."
        )
    )

    # Shuttle frequency/on-demand
    freq_leaf = evaluator.add_leaf(
        id=f"Hotel_{hotel_index+1}_Shuttle_Frequency",
        desc="Shuttle runs at least every 30 minutes or has on-demand service",
        parent=hotel_node,
        critical=True
    )
    claim_freq = (
        "The hotel's airport shuttle runs at least every 30 minutes, or it is on-demand/on request."
    )
    await evaluator.verify(
        claim=claim_freq,
        node=freq_leaf,
        sources=hotel.reference_url if ref_url_ok else None,
        additional_instruction=(
            "Pass if phrasing indicates 'every 30 minutes' or more frequent (e.g., every 15–20 minutes), "
            "or if shuttle is 'on demand', 'on request', 'call for pickup', or similar. "
            "If the page only states a frequency slower than every 30 minutes (e.g., every hour) and not on-demand, fail."
        )
    )


async def verify_parking(
    evaluator: Evaluator,
    parent_node,
    parking: ParkingInfo
) -> None:
    # Reference URL gate
    ref_ok = bool(parking.reference_url and str(parking.reference_url).strip())
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="Parking_Cost_5_Days_Reference",
        desc="Valid reference URL for parking rates",
        parent=parent_node,
        critical=True
    )

    # Parking cost leaf (daily + total for 5 days)
    parking_leaf = evaluator.add_leaf(
        id="Parking_Cost_5_Days",
        desc="Cost to park for 5 consecutive days in on-airport Economy parking",
        parent=parent_node,
        critical=True
    )
    rate = parking.economy_daily_rate or ""
    total = parking.total_cost_5_days or ""
    claim_parking = (
        f"According to the ATL parking rates page, the on-airport Economy parking daily rate is '{rate}', "
        f"and the total for 5 consecutive days is '{total}'."
    )
    await evaluator.verify(
        claim=claim_parking,
        node=parking_leaf,
        sources=parking.reference_url if ref_ok else None,
        additional_instruction=(
            "Verify that the page supports the Economy parking daily rate. The 5-day total may be a straightforward calculation "
            "based on the daily rate; accept a correct computation even if not explicitly stated verbatim on the page."
        )
    )


async def verify_rental_car(
    evaluator: Evaluator,
    parent_node,
    rcc: RentalCarInfo
) -> None:
    # Reference URL gate
    ref_ok = bool(rcc.reference_url and str(rcc.reference_url).strip())
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="Rental_Car_Reference",
        desc="Valid reference URL for rental car center information",
        parent=parent_node,
        critical=True
    )

    # Address
    addr_leaf = evaluator.add_leaf(
        id="Rental_Car_Center_Address",
        desc="Complete address of the ATL Rental Car Center",
        parent=parent_node,
        critical=True
    )
    addr = rcc.address or ""
    claim_addr = f"The complete address of the ATL Rental Car Center is '{addr}'."
    await evaluator.verify(
        claim=claim_addr,
        node=addr_leaf,
        sources=rcc.reference_url if ref_ok else None,
        additional_instruction="Match the full address as given on the referenced page; allow minor formatting differences."
    )

    # Transport method and cost
    transport_leaf = evaluator.add_leaf(
        id="Rental_Car_Center_Transportation",
        desc="Method and cost of transportation from terminals to Rental Car Center",
        parent=parent_node,
        critical=True
    )
    method = rcc.transport_method or ""
    cost = rcc.transport_cost or ""
    claim_transport = (
        f"The transportation from ATL terminals to the Rental Car Center is via '{method}', and the cost is '{cost}'."
    )
    await evaluator.verify(
        claim=claim_transport,
        node=transport_leaf,
        sources=rcc.reference_url if ref_ok else None,
        additional_instruction="At ATL this is commonly the ATL SkyTrain; confirm both the method and that it is free (or the stated cost)."
    )


async def verify_tsa(
    evaluator: Evaluator,
    parent_node,
    tsa: TSAPrecheckInfo
) -> None:
    # Reference URL gate
    ref_ok = bool(tsa.reference_url and str(tsa.reference_url).strip())
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="TSA_PreCheck_Reference",
        desc="Valid reference URL for TSA PreCheck information",
        parent=parent_node,
        critical=True
    )

    # Availability confirmation
    avail_leaf = evaluator.add_leaf(
        id="TSA_PreCheck_Domestic_South",
        desc="Confirmation that TSA PreCheck is available at Domestic Terminal South",
        parent=parent_node,
        critical=True
    )
    claim_avail = (
        "TSA PreCheck lanes are available at the Domestic Terminal South security checkpoint at ATL."
    )
    await evaluator.verify(
        claim=claim_avail,
        node=avail_leaf,
        sources=tsa.reference_url if ref_ok else None,
        additional_instruction="The page should clearly indicate TSA PreCheck availability at the Domestic Terminal South checkpoint."
    )

    # Hours
    hours_leaf = evaluator.add_leaf(
        id="TSA_PreCheck_Hours",
        desc="Operating hours for TSA PreCheck at Domestic Terminal South",
        parent=parent_node,
        critical=True
    )
    hours = tsa.hours or ""
    claim_hours = f"The operating hours for TSA PreCheck at the Domestic Terminal South checkpoint are '{hours}'."
    await evaluator.verify(
        claim=claim_hours,
        node=hours_leaf,
        sources=tsa.reference_url if ref_ok else None,
        additional_instruction=(
            "Match the hours as presented (including ranges or notes like 'varies'). Allow weekday/weekend variations where applicable."
        )
    )


async def verify_lounges(
    evaluator: Evaluator,
    parent_node,
    lounges: LoungesInfo
) -> None:
    # Reference URLs gate (require at least one lounge URL)
    refs = lounges.reference_urls or []
    ref_ok = bool(refs and any(str(u).strip() for u in refs))
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="Lounge_Reference",
        desc="Valid reference URL for airport lounge information",
        parent=parent_node,
        critical=True
    )

    # Delta Sky Club in Concourse A
    delta_leaf = evaluator.add_leaf(
        id="Delta_Sky_Club_Concourse_A",
        desc="Confirmation that a Delta Sky Club exists in Concourse A",
        parent=parent_node,
        critical=True
    )
    claim_delta = "There is a Delta Sky Club located in Concourse A at ATL."
    await evaluator.verify(
        claim=claim_delta,
        node=delta_leaf,
        sources=refs if ref_ok else None,
        additional_instruction="Confirm a Delta Sky Club presence specifically in Concourse A (not other concourses)."
    )

    # United Club location
    united_leaf = evaluator.add_leaf(
        id="United_Club_Location",
        desc="Specific concourse location of the United Club at ATL",
        parent=parent_node,
        critical=True
    )
    united_loc = lounges.united_club_concourse or ""
    claim_united = f"The United Club at ATL is located in Concourse {united_loc}."
    await evaluator.verify(
        claim=claim_united,
        node=united_leaf,
        sources=refs if ref_ok else None,
        additional_instruction="Verify the concourse for the United Club at ATL (e.g., Concourse D)."
    )

    # Priority Pass lounge (name + concourse)
    pp_leaf = evaluator.add_leaf(
        id="Priority_Pass_Lounge",
        desc="Name and location of at least one Priority Pass lounge at ATL",
        parent=parent_node,
        critical=True
    )
    pp_name = lounges.priority_pass_lounge_name or ""
    pp_conc = lounges.priority_pass_lounge_concourse or ""
    claim_pp = f"There is a Priority Pass lounge named '{pp_name}' located in Concourse {pp_conc} at ATL."
    await evaluator.verify(
        claim=claim_pp,
        node=pp_leaf,
        sources=refs if ref_ok else None,
        additional_instruction=(
            "Confirm that the named lounge accepts Priority Pass and that the concourse matches."
        )
    )


async def verify_baggage(
    evaluator: Evaluator,
    parent_node,
    baggage: BaggageInfo
) -> None:
    # Reference URL gate
    ref_ok = bool(baggage.reference_url and str(baggage.reference_url).strip())
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="Baggage_Claim_Reference",
        desc="Valid reference URL for baggage claim information",
        parent=parent_node,
        critical=True
    )

    # International baggage claim location
    intl_leaf = evaluator.add_leaf(
        id="International_Baggage_Claim",
        desc="Location of international baggage claim (terminal and concourse)",
        parent=parent_node,
        critical=True
    )
    loc = baggage.international_baggage_claim_location or ""
    claim_intl = f"The international baggage claim at ATL is located at '{loc}'."
    await evaluator.verify(
        claim=claim_intl,
        node=intl_leaf,
        sources=baggage.reference_url if ref_ok else None,
        additional_instruction=(
            "Confirm the location, typically referencing the Maynard H. Jackson Jr. International Terminal (Concourse F), "
            "or an equivalent precise description as provided on the source page."
        )
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
    Evaluate an answer for the ATL trip planning task and return a structured summary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent categories; allow partial credit overall
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

    # Parallelize extractions
    hotels_task = evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction"
    )
    parking_task = evaluator.extract(
        prompt=prompt_extract_parking(),
        template_class=ParkingInfo,
        extraction_name="parking_info"
    )
    rental_task = evaluator.extract(
        prompt=prompt_extract_rental_car(),
        template_class=RentalCarInfo,
        extraction_name="rental_car_info"
    )
    tsa_task = evaluator.extract(
        prompt=prompt_extract_tsa_precheck(),
        template_class=TSAPrecheckInfo,
        extraction_name="tsa_precheck_info"
    )
    lounges_task = evaluator.extract(
        prompt=prompt_extract_lounges(),
        template_class=LoungesInfo,
        extraction_name="lounges_info"
    )
    baggage_task = evaluator.extract(
        prompt=prompt_extract_baggage(),
        template_class=BaggageInfo,
        extraction_name="baggage_info"
    )

    hotels_ex, parking_ex, rental_ex, tsa_ex, lounges_ex, baggage_ex = await asyncio.gather(
        hotels_task, parking_task, rental_task, tsa_task, lounges_task, baggage_task
    )

    # Top-level aggregator for trip planning requirements (set non-critical to avoid cross-sibling gating and allow partial credit)
    trip_node = evaluator.add_parallel(
        id="Trip_Planning_Requirements",
        desc="Verify all trip planning requirements are satisfied for a 5-day stay near Atlanta Airport",
        parent=root,
        critical=False  # Adjusted to allow partial scoring across independent categories
    )

    # Group: Hotels
    hotels_group = evaluator.add_parallel(
        id="Hotels_Group",
        desc="Three hotel options near ATL with complimentary shuttle and required frequency",
        parent=trip_node,
        critical=False
    )
    # Prepare up to 3 hotels (pad if fewer)
    hotels_list = list(hotels_ex.hotels)[:3]
    while len(hotels_list) < 3:
        hotels_list.append(HotelItem())

    for idx, hotel in enumerate(hotels_list[:3]):
        await verify_hotel(evaluator, hotels_group, hotel, idx)

    # Group: Parking
    parking_group = evaluator.add_parallel(
        id="Parking_Group",
        desc="ATL Economy parking daily rate and 5-day total with reference",
        parent=trip_node,
        critical=False
    )
    await verify_parking(evaluator, parking_group, parking_ex)

    # Group: Rental Car Center
    rental_group = evaluator.add_parallel(
        id="Rental_Car_Group",
        desc="ATL Rental Car Center address and transport details with reference",
        parent=trip_node,
        critical=False
    )
    await verify_rental_car(evaluator, rental_group, rental_ex)

    # Group: TSA PreCheck
    tsa_group = evaluator.add_parallel(
        id="TSA_PreCheck_Group",
        desc="TSA PreCheck availability and hours at Domestic Terminal South with reference",
        parent=trip_node,
        critical=False
    )
    await verify_tsa(evaluator, tsa_group, tsa_ex)

    # Group: Lounges
    lounges_group = evaluator.add_parallel(
        id="Lounges_Group",
        desc="Delta Sky Club in Concourse A, United Club concourse, and a Priority Pass lounge with reference",
        parent=trip_node,
        critical=False
    )
    await verify_lounges(evaluator, lounges_group, lounges_ex)

    # Group: Baggage Claim
    baggage_group = evaluator.add_parallel(
        id="Baggage_Claim_Group",
        desc="International baggage claim location with reference",
        parent=trip_node,
        critical=False
    )
    await verify_baggage(evaluator, baggage_group, baggage_ex)

    # Return final structured evaluation summary
    return evaluator.get_summary()