import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tampa_cruise_hotel"
TASK_DESCRIPTION = """
Recommend a hotel in Tampa, Florida that meets all of the following requirements for a traveler flying from Bangor, Maine on Allegiant Air to join a Carnival cruise departing from Port Tampa Bay:

Location Requirements:
- Must be located in Tampa, Florida
- Must be within 15 miles of Port Tampa Bay Terminal 3 (located at 815 Channelside Drive, Tampa, FL 33602)
- Should offer or have access to ground transportation to the cruise terminal

Policy Requirements:
- Check-in time must be at or before 3:00 PM
- Checkout time must be at or after 11:00 AM
- Minimum age requirement must be 18 years or lower
- Preferably allows cancellation with at least 24 hours notice

Amenities:
- Preferably offers parking facilities
- Preferably provides shuttle service to the cruise terminal or airport
- Preferably offers luggage storage

Required Information:
- Provide the full hotel name
- Provide the complete address (street address, city, state, ZIP code)
- Provide a valid contact phone number
- Provide a reference URL from an official source (hotel website, booking platform, or hotel chain website)

Timing Logistics:
- The hotel must be reachable from either Tampa International Airport (TPA) or St. Pete-Clearwater International Airport (PIE) within 45 minutes
- Port Tampa Bay cruise terminal must be reachable from the hotel within 30 minutes to allow arrival by 11:30 AM

Provide your recommendation with complete details and supporting documentation.
"""

PORT_TERMINAL_ADDR = "Port Tampa Bay Terminal 3, 815 Channelside Drive, Tampa, FL 33602"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelExtraction(BaseModel):
    # Core identity and contact
    name: Optional[str] = None
    address_line1: Optional[str] = None  # Street address line (e.g., "123 Main St")
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    phone: Optional[str] = None

    # Policies
    check_in_time: Optional[str] = None
    check_out_time: Optional[str] = None
    minimum_age_requirement: Optional[str] = None
    cancellation_policy_text: Optional[str] = None

    # Amenities mentions (verbatim or summarized; keep as strings)
    parking_mentioned: Optional[str] = None
    shuttle_service_mentioned: Optional[str] = None
    luggage_storage_mentioned: Optional[str] = None

    # URLs for grounding different claims
    reference_urls: List[str] = Field(default_factory=list)                # main reference link(s) included in the answer
    location_support_urls: List[str] = Field(default_factory=list)         # address/city/near-port support
    policy_support_urls: List[str] = Field(default_factory=list)           # check-in/out, min age, cancellation
    amenities_support_urls: List[str] = Field(default_factory=list)        # parking, shuttle, luggage storage
    ground_transport_support_urls: List[str] = Field(default_factory=list) # explicit port/airport shuttle or transportation info

    # Travel-time/distance sources (ideally map routes)
    tpa_travel_urls: List[str] = Field(default_factory=list)               # TPA -> hotel
    pie_travel_urls: List[str] = Field(default_factory=list)               # PIE -> hotel
    hotel_to_port_travel_urls: List[str] = Field(default_factory=list)     # hotel -> Port Tampa Bay Terminal 3

    # Optional: distance to port sources if provided
    port_distance_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel() -> str:
    return f"""
    Extract details for the primary hotel recommended in the answer (if multiple hotels are mentioned, take the first or clearly primary recommendation). Extract EXACTLY what appears in the answer, without adding or inventing any data.

    Return the following fields:
    - name: Full official hotel name
    - address_line1: Street address line (e.g., "123 Main St" or "123 Main St, Suite 200")
    - city: City name
    - state: Two-letter state abbreviation (e.g., "FL") if available; otherwise the state name string as shown
    - zip_code: 5-digit ZIP (or ZIP+4) if shown
    - phone: A contact phone number as shown (any format is fine)
    - check_in_time: The stated check-in time, as written (e.g., "3:00 PM" or "3 PM")
    - check_out_time: The stated checkout time, as written
    - minimum_age_requirement: The stated minimum check-in age (e.g., "18", "21", "18+")
    - cancellation_policy_text: Verbatim or concise summary of the cancellation policy if provided

    Amenity mentions (use short phrases or single words as shown in the answer; null if absent):
    - parking_mentioned
    - shuttle_service_mentioned
    - luggage_storage_mentioned

    Provide URL lists used in the answer for each category (extract only actual URLs present in the answer; do not invent):
    - reference_urls: Main official or major booking page(s) cited for the hotel
    - location_support_urls: Pages/links supporting the address or that the hotel is in Tampa; can include the reference URLs or other pages used in the answer
    - policy_support_urls: Pages/links that show check-in/out times, minimum age, cancellation policy
    - amenities_support_urls: Pages/links that show parking, shuttle, or luggage storage information
    - ground_transport_support_urls: Links that mention transportation to the cruise terminal or airport (hotel shuttle, arranged rides, etc.)
    - tpa_travel_urls: Google/Apple/Bing Maps links or equivalent pages used to show travel time from Tampa International Airport (TPA) to the hotel
    - pie_travel_urls: Similar travel links used for St. Pete–Clearwater International Airport (PIE) to the hotel
    - hotel_to_port_travel_urls: Travel links used to show time from the hotel to "{PORT_TERMINAL_ADDR}"
    - port_distance_urls: Links that explicitly state the hotel's distance to Port Tampa Bay or Terminal 3, if cited

    Rules:
    - Extract only what is explicitly present in the answer. If an item is not present, set it to null or an empty list as appropriate.
    - For time fields, keep the formatting as shown (do not convert formats).
    - For URLs, include only valid URLs. Include markdown link targets if used.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            u = u.strip()
            if not u:
                continue
            if not (u.startswith("http://") or u.startswith("https://")):
                # Keep only plausible http(s) URLs
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _safe_sources(urls: List[str]) -> Optional[List[str]]:
    return urls if urls and len(urls) > 0 else None


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _phone_is_valid(phone: Optional[str]) -> bool:
    if not phone:
        return False
    digits = _digits_only(phone)
    return len(digits) >= 10


def _zip_is_valid(zip_code: Optional[str]) -> bool:
    if not zip_code:
        return False
    return re.search(r"\b\d{5}(?:-\d{4})?\b", zip_code.strip()) is not None


def _address_is_complete(addr: Optional[str], city: Optional[str], state: Optional[str], zip_code: Optional[str]) -> bool:
    if not addr or not city or not state or not zip_code:
        return False
    # At least has a number in street line
    if not re.search(r"\d", addr):
        return False
    if "tampa" not in city.lower():
        return False
    if "fl" not in state.lower():
        return False
    if not _zip_is_valid(zip_code):
        return False
    return True


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_location_musts(evaluator: Evaluator, parent, ex: HotelExtraction) -> None:
    location_node = evaluator.add_parallel(
        id="location_musts",
        desc="Hotel location meets proximity requirements (critical musts)",
        parent=parent,
        critical=True
    )

    # In Tampa, Florida
    in_tampa_node = evaluator.add_leaf(
        id="in_tampa_florida",
        desc="Hotel is located in Tampa, Florida",
        parent=location_node,
        critical=True
    )
    urls_loc = _combine_urls(ex.location_support_urls, ex.reference_urls)
    hotel_name = ex.name or "the recommended hotel"
    claim_in_tampa = f"The hotel '{hotel_name}' is located in Tampa, Florida (city: Tampa, state: FL)."
    await evaluator.verify(
        claim=claim_in_tampa,
        node=in_tampa_node,
        sources=_safe_sources(urls_loc),
        additional_instruction="Confirm the hotel's city is Tampa, Florida (accept 'Tampa, FL' formatting). Use the cited page(s) only."
    )

    # Within 15 miles of Port Tampa Bay Terminal 3
    proximity_node = evaluator.add_leaf(
        id="proximity_to_port",
        desc=f"Hotel is within 15 miles of Port Tampa Bay (at {PORT_TERMINAL_ADDR})",
        parent=location_node,
        critical=True
    )
    urls_prox = _combine_urls(ex.hotel_to_port_travel_urls, ex.port_distance_urls, ex.location_support_urls, ex.reference_urls)
    claim_proximity = (
        f"The hotel '{hotel_name}' is within 15 miles of Port Tampa Bay Terminal 3 "
        f"(address: {PORT_TERMINAL_ADDR})."
    )
    await evaluator.verify(
        claim=claim_proximity,
        node=proximity_node,
        sources=_safe_sources(urls_prox),
        additional_instruction="Prefer explicit mileage <= 15 on a cited page or a map route showing distance. "
                               "If only a map time is shown, consider typical urban speeds: <= 30 minutes normally "
                               "indicates roughly <= 15 miles in Tampa."
    )


async def add_policy_musts(evaluator: Evaluator, parent, ex: HotelExtraction) -> None:
    policy_node = evaluator.add_parallel(
        id="policy_musts",
        desc="Hotel policies accommodate required schedule (critical musts)",
        parent=parent,
        critical=True
    )

    # Check-in at or before 3:00 PM
    checkin_node = evaluator.add_leaf(
        id="check_in_time",
        desc="Hotel check-in time is at or before 3:00 PM to accommodate standard arrival time",
        parent=policy_node,
        critical=True
    )
    urls_pol = _combine_urls(ex.policy_support_urls, ex.reference_urls)
    ci = ex.check_in_time or "unknown"
    claim_ci = f"The hotel's stated check-in time is '{ci}', which is at or before 3:00 PM."
    await evaluator.verify(
        claim=claim_ci,
        node=checkin_node,
        sources=_safe_sources(urls_pol),
        additional_instruction="Interpret times leniently (e.g., '3 PM', '3:00PM', '15:00'). "
                               "If the page lists an earlier time (e.g., 2 PM), it also satisfies."
    )

    # Checkout at or after 11:00 AM
    checkout_node = evaluator.add_leaf(
        id="check_out_time",
        desc="Hotel checkout time is at or after 11:00 AM to allow morning departure to cruise terminal",
        parent=policy_node,
        critical=True
    )
    co = ex.check_out_time or "unknown"
    claim_co = f"The hotel's stated checkout time is '{co}', which is at or after 11:00 AM."
    await evaluator.verify(
        claim=claim_co,
        node=checkout_node,
        sources=_safe_sources(urls_pol),
        additional_instruction="Interpret times leniently (e.g., '11 AM', '11:00AM', '11.00am', '11:30 AM'). "
                               "If the page lists a later time (e.g., noon), it also satisfies."
    )

    # Minimum age requirement <= 18
    minage_node = evaluator.add_leaf(
        id="minimum_age_requirement",
        desc="Hotel minimum age requirement is 18 years or lower",
        parent=policy_node,
        critical=True
    )
    ma = ex.minimum_age_requirement or "unknown"
    claim_ma = f"The hotel's stated minimum check-in age is '{ma}', which is 18 or lower."
    await evaluator.verify(
        claim=claim_ma,
        node=minage_node,
        sources=_safe_sources(urls_pol),
        additional_instruction="Confirm the minimum check-in age on the cited page(s). "
                               "Phrases like '18+' or 'must be 18' satisfy. If it states 21, it fails."
    )


async def add_booking_info_musts(evaluator: Evaluator, parent, ex: HotelExtraction) -> None:
    booking_node = evaluator.add_parallel(
        id="booking_contact_information_musts",
        desc="Complete and verifiable hotel booking information is provided (critical musts)",
        parent=parent,
        critical=True
    )

    # Presence checks (about the answer content) - use custom nodes
    evaluator.add_custom_node(
        result=bool(ex.name and ex.name.strip()),
        id="hotel_name",
        desc="Full official name of the hotel is provided",
        parent=booking_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_address_is_complete(ex.address_line1, ex.city, ex.state, ex.zip_code),
        id="complete_address",
        desc="Complete street address including city, state, and ZIP code is provided",
        parent=booking_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_phone_is_valid(ex.phone),
        id="contact_phone",
        desc="Valid contact phone number for the hotel is provided",
        parent=booking_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ex.reference_urls and len(ex.reference_urls) > 0),
        id="reference_url",
        desc="A reference URL from an official hotel website, major booking platform, or hotel chain website that confirms the hotel's existence and details",
        parent=booking_node,
        critical=True
    )


async def add_timing_musts(evaluator: Evaluator, parent, ex: HotelExtraction) -> None:
    timing_node = evaluator.add_parallel(
        id="timing_logistics_musts",
        desc="Hotel location and schedule support the required timeline (critical musts)",
        parent=parent,
        critical=True
    )

    # Airport to hotel within 45 minutes (TPA OR PIE)
    # Perform two standalone verifications and OR the results, then add a custom node.
    hotel_name = ex.name or "the recommended hotel"

    tpa_claim = (
        f"The typical driving time from Tampa International Airport (TPA) to '{hotel_name}' is 45 minutes or less."
    )
    pie_claim = (
        f"The typical driving time from St. Pete–Clearwater International Airport (PIE) to '{hotel_name}' is 45 minutes or less."
    )

    tpa_urls = _combine_urls(ex.tpa_travel_urls)
    pie_urls = _combine_urls(ex.pie_travel_urls)

    tpa_ok = await evaluator.verify(
        claim=tpa_claim,
        node=None,
        sources=_safe_sources(tpa_urls),
        additional_instruction="Use the cited map or official page. Consider normal traffic estimates on the page."
    )
    pie_ok = await evaluator.verify(
        claim=pie_claim,
        node=None,
        sources=_safe_sources(pie_urls),
        additional_instruction="Use the cited map or official page. Consider normal traffic estimates on the page."
    )

    evaluator.add_custom_info(
        info={
            "tpa_within_45": bool(tpa_ok),
            "pie_within_45": bool(pie_ok),
            "tpa_urls": tpa_urls,
            "pie_urls": pie_urls
        },
        info_type="airport_reachability_checks",
        info_name="airport_reachability"
    )

    evaluator.add_custom_node(
        result=bool(tpa_ok or pie_ok),
        id="airport_to_hotel_timing",
        desc="Hotel is reachable from TPA or PIE within 45 minutes via available ground transportation",
        parent=timing_node,
        critical=True
    )

    # Hotel to port within 30 minutes
    hotel_to_port_node = evaluator.add_leaf(
        id="hotel_to_port_timing",
        desc="Port Tampa Bay cruise terminal is reachable from hotel within 30 minutes to allow arrival by 11:30 AM for standard noon boarding",
        parent=timing_node,
        critical=True
    )
    urls_htp = _combine_urls(ex.hotel_to_port_travel_urls, ex.port_distance_urls)
    claim_htp = (
        f"The typical driving time from '{hotel_name}' to {PORT_TERMINAL_ADDR} is 30 minutes or less."
    )
    await evaluator.verify(
        claim=claim_htp,
        node=hotel_to_port_node,
        sources=_safe_sources(urls_htp),
        additional_instruction="Use cited map routes or official timing references. Consider normal traffic."
    )


async def add_optional_preferences(evaluator: Evaluator, parent, ex: HotelExtraction) -> None:
    optional_node = evaluator.add_parallel(
        id="optional_preferences",
        desc="Optional preferences and nice-to-have amenities/policies (non-critical)",
        parent=parent,
        critical=False
    )

    # Ground transportation to cruise terminal (optional)
    gt_node = evaluator.add_leaf(
        id="ground_transportation_available",
        desc="Hotel offers or has access to ground transportation to the cruise terminal",
        parent=optional_node,
        critical=False
    )
    urls_gt = _combine_urls(ex.ground_transport_support_urls, ex.amenities_support_urls, ex.reference_urls)
    hotel_name = ex.name or "the hotel"
    claim_gt = (
        f"{hotel_name} offers or facilitates ground transportation to Port Tampa Bay cruise terminal "
        f"(e.g., shuttle, taxi/rideshare arrangement, or concierge-arranged service)."
    )
    await evaluator.verify(
        claim=claim_gt,
        node=gt_node,
        sources=_safe_sources(urls_gt),
        additional_instruction="Accept explicit port shuttle or clear statements that the hotel can arrange rides/taxis to the cruise terminal."
    )

    # Amenities (optional)
    # Parking
    parking_node = evaluator.add_leaf(
        id="parking_facilities",
        desc="Hotel offers on-site or nearby parking facilities",
        parent=optional_node,
        critical=False
    )
    urls_am = _combine_urls(ex.amenities_support_urls, ex.reference_urls)
    claim_parking = f"{hotel_name} offers parking on-site or has nearby parking available."
    await evaluator.verify(
        claim=claim_parking,
        node=parking_node,
        sources=_safe_sources(urls_am),
        additional_instruction="Look for 'parking', 'on-site parking', 'valet', or nearby parking references."
    )

    # Shuttle
    shuttle_node = evaluator.add_leaf(
        id="shuttle_service",
        desc="Hotel provides shuttle service to Port Tampa Bay cruise terminal or Tampa International Airport",
        parent=optional_node,
        critical=False
    )
    claim_shuttle = f"{hotel_name} provides shuttle service to either Port Tampa Bay or Tampa International Airport (TPA)."
    await evaluator.verify(
        claim=claim_shuttle,
        node=shuttle_node,
        sources=_safe_sources(urls_am),
        additional_instruction="Accept airport shuttle. Cruise-port shuttle counts if explicitly mentioned."
    )

    # Luggage storage
    luggage_node = evaluator.add_leaf(
        id="luggage_storage",
        desc="Hotel offers luggage storage for early arrivals or late departures",
        parent=optional_node,
        critical=False
    )
    claim_luggage = f"{hotel_name} offers luggage storage or bag hold service."
    await evaluator.verify(
        claim=claim_luggage,
        node=luggage_node,
        sources=_safe_sources(urls_am),
        additional_instruction="Look for 'luggage storage', 'bag storage', 'baggage hold', or 'hold bags'."
    )

    # Cancellation policy preference (optional)
    cancel_pref_node = evaluator.add_leaf(
        id="cancellation_policy",
        desc="Hotel cancellation policy allows cancellation with at least 24 hours notice",
        parent=optional_node,
        critical=False
    )
    urls_pol = _combine_urls(ex.policy_support_urls, ex.reference_urls)
    ctext = ex.cancellation_policy_text or ""
    claim_cancel = (
        "The hotel's cancellation policy allows cancellation with at least 24 hours' notice "
        "(e.g., free cancellation up to 24 hours before arrival, or a flexible policy that is ≥24 hours)."
    )
    await evaluator.verify(
        claim=claim_cancel,
        node=cancel_pref_node,
        sources=_safe_sources(urls_pol),
        additional_instruction="If multiple rate plans exist, a standard flexible rate allowing ≥24h notice satisfies this preference. "
                               "Non-refundable or special rates do not negate the presence of a flexible policy."
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
    Evaluate an answer for the Tampa cruise hotel recommendation task.
    """
    # Initialize evaluator (root is non-critical to allow mixing critical and non-critical groups safely)
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

    # Extract structured hotel info from the answer
    extracted: HotelExtraction = await evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction"
    )

    # Build verification tree
    # Group: essential/mandatory requirements (critical)
    musts = evaluator.add_parallel(
        id="hotel_must_requirements",
        desc="All essential requirements are satisfied (location, policies, booking info, timing)",
        parent=root,
        critical=True
    )

    # Location musts
    await add_location_musts(evaluator, musts, extracted)

    # Policy musts
    await add_policy_musts(evaluator, musts, extracted)

    # Booking/contact information musts
    await add_booking_info_musts(evaluator, musts, extracted)

    # Timing musts
    await add_timing_musts(evaluator, musts, extracted)

    # Optional preferences and amenities (non-critical, partial credit)
    await add_optional_preferences(evaluator, root, extracted)

    # Return final summary
    return evaluator.get_summary()