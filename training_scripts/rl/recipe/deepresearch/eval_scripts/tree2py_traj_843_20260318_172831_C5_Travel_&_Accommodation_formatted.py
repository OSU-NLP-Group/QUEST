import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "us_to_luanda_itinerary"
TASK_DESCRIPTION = (
    "Plan a complete travel itinerary for a US citizen traveling from Columbia Metropolitan Airport (CAE) in South Carolina to Luanda, Angola. "
    "Your itinerary must include: (1) A multi-leg flight route that connects through Miami International Airport (MIA), with a minimum connection time of 3 hours at Miami to accommodate customs and security procedures for international connections; "
    "(2) At least one flight segment operated by a Star Alliance member airline; "
    "(3) Complete flight details for each segment including airline name, flight number or route confirmation, and departure and arrival airports with IATA codes; "
    "(4) A hotel accommodation in Luanda with a rating of 4 stars or higher that can be booked through major online travel platforms (Booking.com, Expedia, or Agoda); "
    "(5) Confirmation of the visa requirements for US citizens traveling to Angola for tourism purposes, including the maximum visa-free stay duration and passport validity requirements. "
    "Provide specific details for each component with supporting reference URLs from your research."
)


# =========================
# Data Models for Extraction
# =========================

class FlightSegment(BaseModel):
    airline: Optional[str] = None
    flight_number: Optional[str] = None
    departure_airport_name: Optional[str] = None
    departure_iata: Optional[str] = None
    arrival_airport_name: Optional[str] = None
    arrival_iata: Optional[str] = None
    confirmation_urls: List[str] = Field(default_factory=list)


class ItineraryExtraction(BaseModel):
    segment1: Optional[FlightSegment] = None
    segment2: Optional[FlightSegment] = None
    segment3: Optional[FlightSegment] = None

    # Connection time at Miami in minutes (computed from the answer if times are present)
    mia_connection_time_minutes: Optional[int] = None

    # Evidence that at least one airline is a Star Alliance member
    star_alliance_evidence_urls: List[str] = Field(default_factory=list)

    # Hotel info
    hotel_name: Optional[str] = None
    hotel_star_rating_text: Optional[str] = None  # e.g., "4-star", "5 stars"
    hotel_star_rating_value: Optional[float] = None  # numeric if parseable (e.g., 4.0, 4.5, 5.0)
    hotel_booking_platform: Optional[str] = None  # e.g., "Booking.com", "Expedia", "Agoda"
    hotel_booking_urls: List[str] = Field(default_factory=list)

    # Travel documentation sources and extracted interpretations
    documentation_urls: List[str] = Field(default_factory=list)
    visa_free_max_days_text: Optional[str] = None  # e.g., "30 days"
    passport_validity_months: Optional[str] = None  # e.g., "6 months", "at least six months"


# =========================
# Extraction Prompt
# =========================

def prompt_extract_itinerary() -> str:
    return """
Extract the full itinerary details exactly as presented in the answer, following these rules strictly:
- Do not invent or infer any information that is not present in the answer.
- For all URL fields, extract only actual URLs explicitly present in the answer (including markdown links).
- Normalize IATA codes to uppercase if possible.

Required JSON fields:

1) segment1: The first flight segment (CAE -> MIA must be intended here if provided).
   - airline: Airline operating the segment (if provided)
   - flight_number: Flight number (if provided)
   - departure_airport_name: Departure airport full name (if provided)
   - departure_iata: Departure airport IATA code (if provided)
   - arrival_airport_name: Arrival airport full name (if provided)
   - arrival_iata: Arrival airport IATA code (if provided)
   - confirmation_urls: All URLs in the answer that directly support or show this segment's flight/route/schedule.

2) segment2: The second flight segment (must depart MIA if provided).
   - airline
   - flight_number
   - departure_airport_name
   - departure_iata
   - arrival_airport_name
   - arrival_iata
   - confirmation_urls

3) segment3: The third flight segment (should arrive in Luanda; accept LAD or AIAAN per the answer).
   - airline
   - flight_number
   - departure_airport_name
   - departure_iata
   - arrival_airport_name
   - arrival_iata
   - confirmation_urls

4) mia_connection_time_minutes:
   - If the answer provides the arrival time for segment1 into MIA and the departure time for segment2 out of MIA, or states a layover like "3h 20m", compute the layover in total minutes (e.g., 200). Otherwise, return null.

5) star_alliance_evidence_urls:
   - Any URLs in the answer that substantiate Star Alliance membership for any airline in the segments (e.g., staralliance.com, airline official page, Wikipedia membership page).

6) Hotel information:
   - hotel_name: Specific hotel name in Luanda (if provided).
   - hotel_star_rating_text: The star rating text exactly as shown in the answer (e.g., "4-star", "5 stars"). If not present, null.
   - hotel_star_rating_value: Numeric value for star rating if clearly available (e.g., 4.0, 4.5, 5.0). If not present, null.
   - hotel_booking_platform: Name of platform if explicitly mentioned (Booking.com, Expedia, or Agoda); else null.
   - hotel_booking_urls: All booking platform URLs provided for this hotel (only Booking.com, Expedia, or Agoda links if present; include others as well if the answer cites them, but do not invent).

7) Travel documentation:
   - documentation_urls: URLs from the answer that state visa/entry/passport validity requirements (prioritize official sources such as travel.state.gov, embassy/consulate, or official Angolan gov sites).
   - visa_free_max_days_text: The maximum visa-free stay duration for U.S. citizens as quoted in the answer (e.g., "30 days"). If not explicitly stated, null.
   - passport_validity_months: The passport validity requirement as quoted in the answer (e.g., "6 months"). If not explicitly stated, null.

Return a single JSON object matching the schema. For any missing fields, return null (or empty array for URL lists).
"""


# =========================
# Helper utilities
# =========================

def domain_in(url: str, allowed_domains: List[str]) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    for d in allowed_domains:
        if d in netloc:
            return True
    return False


def any_hotel_url_on_allowed_platform(hotel_urls: List[str]) -> bool:
    allowed = ["booking.com", "expedia.com", "agoda.com"]
    return any(domain_in(u, allowed) for u in hotel_urls)


def collect_all_segment_urls(segments: List[Optional[FlightSegment]]) -> List[str]:
    urls: List[str] = []
    for s in segments:
        if s and s.confirmation_urls:
            urls.extend(s.confirmation_urls)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def build_route_claim(seg: Optional[FlightSegment], expected_dep_iata: Optional[str] = None,
                      expected_arr_iata: Optional[str] = None) -> str:
    dep_iata = (seg.departure_iata or "").upper() if seg else ""
    arr_iata = (seg.arrival_iata or "").upper() if seg else ""
    airline = seg.airline or ""
    flight_no = seg.flight_number or ""
    parts = []
    if expected_dep_iata and expected_arr_iata:
        parts.append(f"a flight or route from {expected_dep_iata} to {expected_arr_iata}")
    else:
        parts.append(f"a flight or route from {dep_iata} to {arr_iata}")
    if airline:
        parts.append(f"operated by {airline}")
    if flight_no:
        parts.append(f"with flight number {flight_no}")
    return "This page confirms " + " ".join(parts) + "."


# =========================
# Verification Subroutines
# =========================

async def verify_flight_segment_1(evaluator: Evaluator, parent, seg: Optional[FlightSegment]) -> None:
    node = evaluator.add_parallel(
        id="flight_segment_1",
        desc="First flight segment from Columbia Metropolitan Airport (CAE) to Miami International Airport (MIA)",
        parent=parent,
        critical=False
    )

    # departure_airport: must be CAE
    dep_ok = bool(seg and seg.departure_iata and seg.departure_iata.strip().upper() == "CAE")
    evaluator.add_custom_node(
        result=dep_ok,
        id="departure_airport",
        desc="Departure airport is Columbia Metropolitan Airport with IATA code CAE",
        parent=node,
        critical=True
    )

    # arrival_airport_segment_1: must be MIA
    arr_ok = bool(seg and seg.arrival_iata and seg.arrival_iata.strip().upper() == "MIA")
    evaluator.add_custom_node(
        result=arr_ok,
        id="arrival_airport_segment_1",
        desc="Arrival airport for first segment is Miami International Airport with IATA code MIA",
        parent=node,
        critical=True
    )

    # airline_segment_1: airline name AND (flight number OR at least one supporting URL)
    airline_ok = bool(
        seg and seg.airline and seg.airline.strip() and ((seg.flight_number and seg.flight_number.strip()) or (seg.confirmation_urls and len(seg.confirmation_urls) > 0))
    )
    evaluator.add_custom_node(
        result=airline_ok,
        id="airline_segment_1",
        desc="Airline name and flight number or route confirmation provided for first segment",
        parent=node,
        critical=True
    )

    # reference_url_segment_1: verify with provided URLs (route CAE->MIA)
    if seg and seg.confirmation_urls:
        leaf = evaluator.add_leaf(
            id="reference_url_segment_1",
            desc="URL reference provided supporting the flight information for segment 1",
            parent=node,
            critical=True
        )
        claim = build_route_claim(seg, expected_dep_iata="CAE", expected_arr_iata="MIA")
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=seg.confirmation_urls,
            additional_instruction="Accept airline websites, flight aggregators, and booking/schedule pages that clearly show the CAE→MIA flight or route. Minor formatting differences are acceptable."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="reference_url_segment_1",
            desc="URL reference provided supporting the flight information for segment 1",
            parent=node,
            critical=True
        )


async def verify_flight_segment_2(evaluator: Evaluator, parent, seg: Optional[FlightSegment], conn_minutes: Optional[int]) -> None:
    node = evaluator.add_parallel(
        id="flight_segment_2",
        desc="Second flight segment from Miami International Airport (MIA) to an international connecting hub",
        parent=parent,
        critical=False
    )

    # departure_airport_segment_2: must depart MIA
    dep_ok = bool(seg and seg.departure_iata and seg.departure_iata.strip().upper() == "MIA")
    evaluator.add_custom_node(
        result=dep_ok,
        id="departure_airport_segment_2",
        desc="Departure airport for second segment is Miami International Airport with IATA code MIA",
        parent=node,
        critical=True
    )

    # connection_time: at least 3 hours (180 minutes)
    conn_ok = bool(conn_minutes is not None and conn_minutes >= 180)
    evaluator.add_custom_node(
        result=conn_ok,
        id="connection_time",
        desc="Connection time at Miami International Airport is at least 3 hours to allow for customs and security",
        parent=node,
        critical=True
    )

    # airline_segment_2: airline name AND (flight number OR at least one supporting URL)
    airline_ok = bool(
        seg and seg.airline and seg.airline.strip() and ((seg.flight_number and seg.flight_number.strip()) or (seg.confirmation_urls and len(seg.confirmation_urls) > 0))
    )
    evaluator.add_custom_node(
        result=airline_ok,
        id="airline_segment_2",
        desc="Airline name and flight number or route confirmation provided for second segment",
        parent=node,
        critical=True
    )

    # reference_url_segment_2: verify route from MIA to wherever segment 2 arrives
    if seg and seg.confirmation_urls:
        leaf = evaluator.add_leaf(
            id="reference_url_segment_2",
            desc="URL reference provided supporting the flight information for segment 2",
            parent=node,
            critical=True
        )
        # If no expected arrival code known, just use actual extracted
        claim = build_route_claim(seg, expected_dep_iata="MIA", expected_arr_iata=(seg.arrival_iata or "").upper() or None)
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=seg.confirmation_urls,
            additional_instruction="The page should show a flight or route departing from MIA. Airline schedule/search results or booking pages are acceptable."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="reference_url_segment_2",
            desc="URL reference provided supporting the flight information for segment 2",
            parent=node,
            critical=True
        )


async def verify_flight_segment_3(evaluator: Evaluator, parent, seg: Optional[FlightSegment]) -> None:
    node = evaluator.add_parallel(
        id="flight_segment_3",
        desc="Third flight segment from international hub to Luanda, Angola",
        parent=parent,
        critical=False
    )

    # final_destination: must be LAD or AIAAN per rubric
    arr_code = (seg.arrival_iata or "").upper() if seg and seg.arrival_iata else ""
    final_ok = arr_code in {"LAD", "AIAAN"}
    evaluator.add_custom_node(
        result=final_ok,
        id="final_destination",
        desc="Final arrival airport is in Luanda, Angola (either LAD or AIAAN airport code)",
        parent=node,
        critical=True
    )

    # airline_segment_3: airline name AND (flight number OR at least one supporting URL)
    airline_ok = bool(
        seg and seg.airline and seg.airline.strip() and ((seg.flight_number and seg.flight_number.strip()) or (seg.confirmation_urls and len(seg.confirmation_urls) > 0))
    )
    evaluator.add_custom_node(
        result=airline_ok,
        id="airline_segment_3",
        desc="Airline name and flight number or route confirmation provided for third segment",
        parent=node,
        critical=True
    )

    # reference_url_segment_3: verify that this segment arrives in Luanda (LAD or AIAAN)
    if seg and seg.confirmation_urls:
        leaf = evaluator.add_leaf(
            id="reference_url_segment_3",
            desc="URL reference provided supporting the flight information for segment 3",
            parent=node,
            critical=True
        )
        # Prefer explicit target to Luanda
        target_code = "LAD" if arr_code == "LAD" else (arr_code if arr_code else "LAD")
        claim = build_route_claim(seg, expected_dep_iata=(seg.departure_iata or "").upper() or None, expected_arr_iata=target_code)
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=seg.confirmation_urls,
            additional_instruction="The page should clearly show the arrival airport in Luanda (LAD or AIAAN). Accept airline or booking pages."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="reference_url_segment_3",
            desc="URL reference provided supporting the flight information for segment 3",
            parent=node,
            critical=True
        )


async def verify_star_alliance_requirement(
    evaluator: Evaluator,
    parent,
    segments: List[Optional[FlightSegment]],
    star_urls: List[str],
):
    # Prepare airline list from the three segments
    airlines = []
    for seg in segments:
        if seg and seg.airline:
            airlines.append(seg.airline)
    airlines_text = ", ".join([a for a in airlines if a]) if airlines else "no airline provided"

    # Combine all possible helpful URLs (membership pages + segment confirmations)
    sources = list(star_urls or [])
    sources.extend(collect_all_segment_urls(segments))

    leaf = evaluator.add_leaf(
        id="star_alliance_requirement",
        desc="At least one of the three flight segments must be operated by a Star Alliance member airline",
        parent=parent,
        critical=True
    )

    claim = (
        f"At least one of these airlines is a Star Alliance member: {airlines_text}. "
        f"If any one of them is a Star Alliance member, consider the claim supported."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction="Prefer staralliance.com membership lists or official airline/Wikipedia membership pages. Focus on 'operated by' not merely 'marketed by' when linking a segment to a member."
    )


async def verify_accommodation(evaluator: Evaluator, parent, it: ItineraryExtraction):
    node = evaluator.add_parallel(
        id="accommodation",
        desc="Hotel accommodation in Luanda, Angola",
        parent=parent,
        critical=False
    )

    # hotel_name provided
    hotel_name_ok = bool(it.hotel_name and it.hotel_name.strip())
    evaluator.add_custom_node(
        result=hotel_name_ok,
        id="hotel_name",
        desc="Specific hotel name in Luanda is provided",
        parent=node,
        critical=True
    )

    # booking_platform: must be bookable at Booking.com, Expedia or Agoda
    platform_ok = any_hotel_url_on_allowed_platform(it.hotel_booking_urls or [])
    evaluator.add_custom_node(
        result=platform_ok,
        id="booking_platform",
        desc="Hotel is bookable on at least one major platform (Booking.com, Expedia, or Agoda)",
        parent=node,
        critical=True
    )

    # reference_url_hotel: verify booking page shows hotel details in Luanda
    if it.hotel_booking_urls:
        leaf_ref = evaluator.add_leaf(
            id="reference_url_hotel",
            desc="URL reference provided from a hotel booking platform showing the hotel details",
            parent=node,
            critical=True
        )
        hotel_for_claim = it.hotel_name or "the specified hotel"
        await evaluator.verify(
            claim=f"This page is a booking/listing page for {hotel_for_claim} located in Luanda, Angola.",
            node=leaf_ref,
            sources=it.hotel_booking_urls,
            additional_instruction="Accept Booking.com, Expedia, or Agoda pages that display the property details and location (Luanda)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="reference_url_hotel",
            desc="URL reference provided from a hotel booking platform showing the hotel details",
            parent=node,
            critical=True
        )

    # hotel_rating: verify at least 4-star on booking page
    leaf_rating = evaluator.add_leaf(
        id="hotel_rating",
        desc="Hotel has a rating of 4 stars or higher",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This hotel page shows an official star rating of at least 4 stars (4★, 4.5★, or 5★). Guest review scores are not the same as star ratings.",
        node=leaf_rating,
        sources=it.hotel_booking_urls if it.hotel_booking_urls else None,
        additional_instruction="Focus on the hotel's star category. If both star rating and review score appear, judge by the star rating only."
    )


async def verify_travel_documentation(evaluator: Evaluator, parent, it: ItineraryExtraction):
    node = evaluator.add_parallel(
        id="travel_documentation",
        desc="Travel documentation requirements for US citizens traveling to Angola",
        parent=parent,
        critical=False
    )

    # reference_url_documentation: must be from an official source confirming visa requirements
    if it.documentation_urls:
        leaf_ref = evaluator.add_leaf(
            id="reference_url_documentation",
            desc="URL reference provided from an official source (e.g., travel.state.gov or embassy website) confirming visa requirements",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="This page is an official government or embassy/consulate source that states entry/visa requirements for U.S. citizens traveling to Angola.",
            node=leaf_ref,
            sources=it.documentation_urls,
            additional_instruction="Prefer .gov domains (e.g., travel.state.gov), embassy/consular official pages, or official Angolan government pages."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="reference_url_documentation",
            desc="URL reference provided from an official source (e.g., travel.state.gov or embassy website) confirming visa requirements",
            parent=node,
            critical=True
        )

    # visa_requirement: 30-day visa-free for US citizens (as claimed in the answer) – verify with provided docs
    leaf_visa = evaluator.add_leaf(
        id="visa_requirement",
        desc="Confirmation that US citizens can visit Angola visa-free for tourism up to 30 days",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="U.S. citizens can visit Angola visa-free for tourism for up to 30 days.",
        node=leaf_visa,
        sources=it.documentation_urls if it.documentation_urls else None,
        additional_instruction="Only consider the claim supported if the page explicitly states visa-free entry for U.S. citizens up to 30 days for tourism."
    )

    # passport_validity: at least 6 months beyond intended stay
    leaf_passport = evaluator.add_leaf(
        id="passport_validity",
        desc="Passport must be valid for at least 6 months beyond the intended stay",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A U.S. passport must be valid for at least 6 months beyond the intended stay when traveling to Angola for tourism.",
        node=leaf_passport,
        sources=it.documentation_urls if it.documentation_urls else None,
        additional_instruction="Look for explicit mention of 'valid for at least 6 months' or equivalent wording."
    )


# =========================
# Main Evaluation
# =========================

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

    # Extraction
    itinerary: ItineraryExtraction = await evaluator.extract(
        prompt=prompt_extract_itinerary(),
        template_class=ItineraryExtraction,
        extraction_name="itinerary_extraction"
    )

    # Build root description node (already root initialized)

    # Flight segments verifications
    await verify_flight_segment_1(evaluator, root, itinerary.segment1)
    await verify_flight_segment_2(evaluator, root, itinerary.segment2, itinerary.mia_connection_time_minutes)
    await verify_flight_segment_3(evaluator, root, itinerary.segment3)

    # Star Alliance requirement (critical at root level)
    await verify_star_alliance_requirement(
        evaluator,
        root,
        [itinerary.segment1, itinerary.segment2, itinerary.segment3],
        itinerary.star_alliance_evidence_urls or []
    )

    # Accommodation verification
    await verify_accommodation(evaluator, root, itinerary)

    # Travel documentation verification
    await verify_travel_documentation(evaluator, root, itinerary)

    return evaluator.get_summary()