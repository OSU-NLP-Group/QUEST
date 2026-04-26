import asyncio
import logging
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aruba_conference_2026"
TASK_DESCRIPTION = """
A professional association is planning a 3-day conference in Aruba for 100 attendees scheduled for June 2026. As the conference coordinator, you need to develop a comprehensive venue and travel plan that satisfies the following requirements:

Venue Requirements:
- The hotel must be located in Aruba
- Minimum 80 total guest rooms to support group booking needs
- At least 2,500 square feet of meeting space (based on industry standard of 25 sq ft per person)
- A primary ballroom or conference room that can accommodate the full group of 100 attendees
- Minimum 4 ADA-accessible guest rooms (meeting ADA requirements for hotels in this size range)
- All meeting and conference spaces must be ADA accessible
- On-site catering services for conference meals and breaks
- Accepts group reservations of 10 or more rooms
- Attrition policy that allows up to 20% reduction from reserved room block
- Accepts advance group bookings 90-120 days prior to the event

Travel Logistics:
- Attendees will be traveling from three major US regions: New York/New Jersey area (JFK or Newark airports), Boston area (Logan airport), and Miami/Fort Lauderdale area
- Identify available flight routes (direct or one-stop) from at least two of these three departure regions to Aruba
- For each identified route, specify the operating airline

Deliverables:
Provide a complete conference plan that includes:
1. The name and location of a specific hotel in Aruba that meets all venue requirements
2. Documentation (URL references) confirming the hotel's capacity, meeting space, accessibility features, and group booking policies
3. Identified flight options with airline names and route confirmation for at least two of the three departure regions
4. Verification that the overall plan meets ADA accessibility standards

Your proposal must be grounded in real, currently available information from verifiable sources (hotel websites, airline route maps, travel booking sites, etc.). All claims about hotel capacity, meeting space, accessibility features, policies, and flight availability must be supported by URL references to reliable sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConferenceBasicsExtraction(BaseModel):
    duration_days: Optional[str] = None  # e.g., "3 days"
    scheduled_month_year: Optional[str] = None  # e.g., "June 2026"
    attendee_count: Optional[str] = None  # e.g., "100"
    explicit_ada_statement: Optional[str] = None  # any explicit text affirming ADA compliance


class HotelExtraction(BaseModel):
    # Identity/location
    name: Optional[str] = None
    location_text: Optional[str] = None  # e.g., "Noord, Aruba", "Palm Beach, Aruba"
    urls_location: List[str] = Field(default_factory=list)

    # Capacity
    total_guest_rooms: Optional[str] = None  # e.g., "320 rooms"
    meeting_space_sqft: Optional[str] = None  # e.g., "10,000 sq ft"
    primary_room_capacity: Optional[str] = None  # e.g., "Ballroom max 150 theatre"

    # Accessibility
    ada_accessible_guest_rooms: Optional[str] = None  # e.g., "6 accessible rooms"
    meeting_spaces_ada_accessible: Optional[str] = None  # e.g., "All meeting rooms are ADA accessible"

    # Services/policies
    onsite_catering_available: Optional[str] = None
    group_reservations_10plus: Optional[str] = None
    attrition_policy_20pct: Optional[str] = None
    advance_group_booking_window_days: Optional[str] = None  # e.g., "Bookable 90-120 days in advance"

    # URL evidence (per-requirement)
    urls_room_count: List[str] = Field(default_factory=list)
    urls_meeting_space: List[str] = Field(default_factory=list)
    urls_primary_room_capacity: List[str] = Field(default_factory=list)
    urls_ada_guest_rooms: List[str] = Field(default_factory=list)
    urls_ada_meeting_spaces: List[str] = Field(default_factory=list)
    urls_onsite_catering: List[str] = Field(default_factory=list)
    urls_group_reservations_10plus: List[str] = Field(default_factory=list)
    urls_attrition_20pct: List[str] = Field(default_factory=list)
    urls_advance_booking_window: List[str] = Field(default_factory=list)


class FlightRouteExtraction(BaseModel):
    departure_airport: Optional[str] = None  # Prefer IATA (e.g., JFK, EWR, BOS, MIA, FLL)
    departure_region: Optional[str] = None  # e.g., "New York", "Newark", "Boston", "Miami", "Fort Lauderdale"
    destination_airport: Optional[str] = None  # Prefer IATA; expect "AUA"
    connectivity: Optional[str] = None  # "direct", "nonstop", "one stop", etc.
    airline: Optional[str] = None  # e.g., "JetBlue", "United"
    url_proofs: List[str] = Field(default_factory=list)  # URLs confirming route exists


class FlightLogisticsExtraction(BaseModel):
    routes: List[FlightRouteExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_conference_basics() -> str:
    return """
    Extract the conference basics explicitly stated in the answer.

    Return a JSON object with:
    - duration_days: the explicitly stated duration (e.g., "3 days" or "three-day")
    - scheduled_month_year: the explicitly stated time window (e.g., "June 2026")
    - attendee_count: the explicitly stated attendee count (e.g., "100")
    - explicit_ada_statement: if the answer explicitly states that the overall plan meets ADA accessibility requirements (guest rooms + meeting spaces), return that phrase or a short "yes" indicator. If not present, return null.

    Only extract what is explicitly stated in the answer text.
    """


def prompt_extract_hotel_info() -> str:
    return """
    Extract details about the proposed hotel venue in Aruba and all supporting URLs cited in the answer.

    Return a JSON object with these fields (use strings for values when applicable; if not stated, set to null; for URL lists, return [] if none):
    - name
    - location_text
    - urls_location: all URLs that demonstrate the hotel is located in Aruba
    - total_guest_rooms
    - meeting_space_sqft
    - primary_room_capacity
    - ada_accessible_guest_rooms
    - meeting_spaces_ada_accessible
    - onsite_catering_available
    - group_reservations_10plus
    - attrition_policy_20pct
    - advance_group_booking_window_days

    - urls_room_count: URLs supporting total_guest_rooms
    - urls_meeting_space: URLs supporting meeting_space_sqft (total or sufficient meeting space that clearly equals/exceeds 2,500 sq ft)
    - urls_primary_room_capacity: URLs showing the primary ballroom/conference room can handle 100 attendees
    - urls_ada_guest_rooms: URLs supporting at least 4 ADA-accessible guest rooms
    - urls_ada_meeting_spaces: URLs stating meeting/conference spaces are ADA accessible
    - urls_onsite_catering: URLs supporting on-site catering/banquet services
    - urls_group_reservations_10plus: URLs showing acceptance of group reservations of 10+ rooms (or an equivalent group policy)
    - urls_attrition_20pct: URLs supporting an attrition policy that allows up to 20% reduction from the room block
    - urls_advance_booking_window: URLs supporting that advance group bookings are accepted 90–120 days prior to the event

    Important:
    - Extract only URLs that are explicitly present in the answer (plain links or markdown).
    - Do not invent any values or URLs; if not present, use null for values and [] for URL lists.
    """


def prompt_extract_flight_logistics() -> str:
    return """
    Extract the flight logistics listed in the answer. We only need to capture flight route options mentioned (at least two if provided).

    Return a JSON object:
    {
      "routes": [
        {
          "departure_airport": "IATA code if present, else airport/city name (e.g., JFK, EWR, BOS, MIA, FLL)",
          "departure_region": "free text region as claimed, e.g., New York/Newark/Boston/Miami/Fort Lauderdale",
          "destination_airport": "IATA code if present; Aruba should be AUA",
          "connectivity": "nonstop/direct or one stop",
          "airline": "operating airline as cited",
          "url_proofs": ["..."]  // one or more URLs confirming this route option
        },
        ...
      ]
    }

    Rules:
    - Extract only what is explicitly present in the answer.
    - For each route, include all URLs cited to confirm route existence or airline operation.
    - If any field is not present in the answer for a route, set it to null (or [] for url_proofs).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_non_empty(urls: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for u in urls or []:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _all_hotel_urls(h: HotelExtraction) -> List[str]:
    urls: List[str] = []
    urls.extend(h.urls_location or [])
    urls.extend(h.urls_room_count or [])
    urls.extend(h.urls_meeting_space or [])
    urls.extend(h.urls_primary_room_capacity or [])
    urls.extend(h.urls_ada_guest_rooms or [])
    urls.extend(h.urls_ada_meeting_spaces or [])
    urls.extend(h.urls_onsite_catering or [])
    urls.extend(h.urls_group_reservations_10plus or [])
    urls.extend(h.urls_attrition_20pct or [])
    urls.extend(h.urls_advance_booking_window or [])
    return _unique_non_empty(urls)


def _normalize_region(airport_code_or_name: Optional[str], free_text_region: Optional[str]) -> Optional[str]:
    s = (airport_code_or_name or "") + " " + (free_text_region or "")
    s_up = s.upper()

    # NYC/NJ region
    if any(tag in s_up for tag in ["JFK", "EWR", "NEW YORK", "NYC", "NEWARK"]):
        return "NYC/NJ"

    # Boston
    if any(tag in s_up for tag in ["BOS", "BOSTON", "LOGAN"]):
        return "BOS"

    # Miami/Fort Lauderdale
    if any(tag in s_up for tag in ["MIA", "MIAMI", "FLL", "FORT LAUDERDALE", "FT. LAUDERDALE"]):
        return "MIA/FLL"

    return None


def _distinct_allowed_regions(routes: List[FlightRouteExtraction]) -> Set[str]:
    regs: Set[str] = set()
    for r in routes or []:
        norm = _normalize_region(r.departure_airport, r.departure_region)
        if norm in {"NYC/NJ", "BOS", "MIA/FLL"}:
            regs.add(norm)
    return regs


def _claims_hotel_name(h: HotelExtraction) -> str:
    return h.name or "the hotel"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_conference_basics_checks(evaluator: Evaluator, parent_node, basics: ConferenceBasicsExtraction) -> None:
    basics_node = evaluator.add_parallel(
        id="Conference_Basics",
        desc="Plan matches stated conference basics (duration, timing, attendee count).",
        parent=parent_node,
        critical=True,
    )

    # Duration: 3 days
    duration = evaluator.add_leaf(
        id="Duration_3_Days",
        desc="Proposal specifies the conference is 3 days long.",
        parent=basics_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The proposal specifies the conference is 3 days long (a three-day event).",
        node=duration,
        additional_instruction="Verify this from the provided answer text only."
    )

    # Scheduled: June 2026
    scheduled = evaluator.add_leaf(
        id="Scheduled_June_2026",
        desc="Proposal specifies the conference is scheduled for June 2026.",
        parent=basics_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The proposal explicitly states the conference is scheduled for June 2026.",
        node=scheduled,
        additional_instruction="Verify this from the provided answer text only."
    )

    # Attendee count: 100
    attendees = evaluator.add_leaf(
        id="Attendee_Count_100",
        desc="Proposal specifies planning for 100 attendees.",
        parent=basics_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The proposal explicitly states the conference is planned for 100 attendees.",
        node=attendees,
        additional_instruction="Verify this from the provided answer text only."
    )


async def build_hotel_venue_plan_checks(evaluator: Evaluator, parent_node, h: HotelExtraction) -> None:
    hotel_node = evaluator.add_parallel(
        id="Hotel_Venue_Plan",
        desc="Identifies a specific Aruba hotel that meets all venue requirements and provides required documentation URLs.",
        parent=parent_node,
        critical=True,
    )

    # 1) Hotel Identity and Location
    id_loc = evaluator.add_parallel(
        id="Hotel_Identity_and_Location",
        desc="Hotel is identified and located in Aruba.",
        parent=hotel_node,
        critical=True,
    )

    name_exists = evaluator.add_custom_node(
        result=bool(h.name and h.name.strip()),
        id="Hotel_Name_Provided",
        desc="Provides a specific hotel's name.",
        parent=id_loc,
        critical=True,
    )

    located_leaf = evaluator.add_leaf(
        id="Hotel_Located_in_Aruba",
        desc="Hotel is located in Aruba.",
        parent=id_loc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_claims_hotel_name(h)} is located in Aruba.",
        node=located_leaf,
        sources=_unique_non_empty((h.urls_location or []) or _all_hotel_urls(h)),
        additional_instruction="Confirm via the provided URL(s) that the hotel's address/location is in Aruba."
    )

    # 2) Venue Capacity Requirements
    cap = evaluator.add_parallel(
        id="Venue_Capacity_Requirements",
        desc="Hotel satisfies required room and meeting-space capacities.",
        parent=hotel_node,
        critical=True,
    )

    rooms_leaf = evaluator.add_leaf(
        id="Minimum_80_Guest_Rooms",
        desc="Hotel has at least 80 total guest rooms.",
        parent=cap,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_claims_hotel_name(h)} has at least 80 total guest rooms.",
        node=rooms_leaf,
        sources=_unique_non_empty(h.urls_room_count or []),
        additional_instruction="From the provided webpage(s), confirm the hotel's total room count is 80 or more; synonyms like 'guest rooms'/'rooms' acceptable."
    )

    meeting_space_leaf = evaluator.add_leaf(
        id="Minimum_2500_SqFt_Meeting_Space",
        desc="Hotel provides at least 2,500 square feet of meeting space.",
        parent=cap,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_claims_hotel_name(h)} offers at least 2,500 square feet of meeting space (total meeting/event space).",
        node=meeting_space_leaf,
        sources=_unique_non_empty(h.urls_meeting_space or []),
        additional_instruction="Verify total meeting space equals or exceeds 2,500 sq ft; allow reasonable phrasing variations (e.g., 'total function space')."
    )

    primary_room_leaf = evaluator.add_leaf(
        id="Primary_Room_Accommodates_100",
        desc="Hotel has a primary ballroom/conference room that can accommodate 100 attendees.",
        parent=cap,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The primary ballroom or main conference room at {_claims_hotel_name(h)} can accommodate at least 100 attendees (in any common setup like theater/banquet/classroom).",
        node=primary_room_leaf,
        sources=_unique_non_empty(h.urls_primary_room_capacity or []),
        additional_instruction="Look for maximum capacity charts or room details showing at least 100 people in any setup."
    )

    # 3) Accessibility Requirements
    acc = evaluator.add_parallel(
        id="Accessibility_Requirements",
        desc="Hotel satisfies ADA accessibility requirements.",
        parent=hotel_node,
        critical=True,
    )

    ada_rooms_leaf = evaluator.add_leaf(
        id="Minimum_4_ADA_Accessible_Guest_Rooms",
        desc="Hotel provides at least 4 ADA-accessible guest rooms.",
        parent=acc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_claims_hotel_name(h)} provides at least 4 ADA-accessible (accessible) guest rooms.",
        node=ada_rooms_leaf,
        sources=_unique_non_empty(h.urls_ada_guest_rooms or []),
        additional_instruction="Confirm at least 4 accessible rooms; synonyms like 'accessible rooms', 'ADA-compliant rooms' are acceptable."
    )

    ada_meeting_leaf = evaluator.add_leaf(
        id="All_Meeting_Spaces_ADA_Accessible",
        desc="All meeting and conference spaces are ADA accessible.",
        parent=acc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"All meeting and conference spaces at {_claims_hotel_name(h)} are ADA accessible.",
        node=ada_meeting_leaf,
        sources=_unique_non_empty(h.urls_ada_meeting_spaces or []),
        additional_instruction="Verify policy/statement that meeting and event spaces are wheelchair/ADA accessible."
    )

    # 4) Services and Policies Requirements
    svc = evaluator.add_parallel(
        id="Services_and_Policies_Requirements",
        desc="Hotel provides required services and group policies.",
        parent=hotel_node,
        critical=True,
    )

    catering_leaf = evaluator.add_leaf(
        id="Onsite_Catering_Provided",
        desc="Hotel provides on-site catering services for conference meals/breaks.",
        parent=svc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_claims_hotel_name(h)} provides on-site catering or banquet services for meetings and events.",
        node=catering_leaf,
        sources=_unique_non_empty(h.urls_onsite_catering or []),
        additional_instruction="Look for event/catering/banquet services mentioned on the provided URL(s)."
    )

    groups_10_leaf = evaluator.add_leaf(
        id="Accepts_Group_Reservations_10plus_Rooms",
        desc="Hotel accepts group reservations of 10 or more rooms.",
        parent=svc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_claims_hotel_name(h)} accepts group reservations of 10 or more rooms.",
        node=groups_10_leaf,
        sources=_unique_non_empty(h.urls_group_reservations_10plus or []),
        additional_instruction="Confirm group booking policy indicates acceptance of 10+ rooms (or equivalent)."
    )

    attrition_leaf = evaluator.add_leaf(
        id="Attrition_Allows_Up_To_20pct_Reduction",
        desc="Hotel attrition policy allows up to 20% reduction from reserved room block.",
        parent=svc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The group attrition policy at {_claims_hotel_name(h)} allows up to a 20% reduction from the reserved room block.",
        node=attrition_leaf,
        sources=_unique_non_empty(h.urls_attrition_20pct or []),
        additional_instruction="Look for written attrition terms; 20% tolerance or 'up to 20%' acceptable."
    )

    advance_leaf = evaluator.add_leaf(
        id="Accepts_Advance_Group_Bookings_90_120_Days",
        desc="Hotel accepts advance group bookings 90–120 days prior to the event.",
        parent=svc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_claims_hotel_name(h)} accepts advance group bookings approximately 90–120 days before the event.",
        node=advance_leaf,
        sources=_unique_non_empty(h.urls_advance_booking_window or []),
        additional_instruction="Confirm that the hotel's policy or sales material allows booking groups 90–120 days in advance (reasonable wording allowed)."
    )

    # 5) Hotel Documentation URLs (existence checks for URL evidence per requirement)
    doc_urls = evaluator.add_parallel(
        id="Hotel_Documentation_URLs",
        desc="Provides URL references supporting the required hotel claims (capacity, meeting space, accessibility, and group booking policies/services).",
        parent=hotel_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_unique_non_empty(h.urls_room_count or [])) > 0,
        id="URL_Supports_Room_Count",
        desc="Provides a URL that supports the stated total guest-room count.",
        parent=doc_urls,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_unique_non_empty(h.urls_meeting_space or [])) > 0,
        id="URL_Supports_Total_Meeting_Space",
        desc="Provides a URL that supports the stated total meeting-space square footage.",
        parent=doc_urls,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_unique_non_empty(h.urls_primary_room_capacity or [])) > 0,
        id="URL_Supports_Primary_Room_Capacity_100",
        desc="Provides a URL that supports the primary room’s capacity to accommodate 100 attendees.",
        parent=doc_urls,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_unique_non_empty(h.urls_ada_guest_rooms or [])) > 0,
        id="URL_Supports_ADA_Accessible_Guest_Rooms",
        desc="Provides a URL supporting the claim about ADA-accessible guest rooms (at least 4).",
        parent=doc_urls,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_unique_non_empty(h.urls_ada_meeting_spaces or [])) > 0,
        id="URL_Supports_ADA_Accessible_Meeting_Spaces",
        desc="Provides a URL supporting that meeting/conference spaces are ADA accessible.",
        parent=doc_urls,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_unique_non_empty(h.urls_onsite_catering or [])) > 0,
        id="URL_Supports_Onsite_Catering",
        desc="Provides a URL supporting on-site catering/banquet services.",
        parent=doc_urls,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_unique_non_empty(h.urls_group_reservations_10plus or [])) > 0,
        id="URL_Supports_Group_Reservations_10plus",
        desc="Provides a URL supporting group reservations of 10+ rooms (or equivalent group booking policy).",
        parent=doc_urls,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_unique_non_empty(h.urls_attrition_20pct or [])) > 0,
        id="URL_Supports_Attrition_20pct",
        desc="Provides a URL supporting the attrition policy terms (up to 20% reduction).",
        parent=doc_urls,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_unique_non_empty(h.urls_advance_booking_window or [])) > 0,
        id="URL_Supports_Advance_Booking_90_120_Days",
        desc="Provides a URL supporting advance group booking window (90–120 days).",
        parent=doc_urls,
        critical=True,
    )


async def _build_one_route_checks(
    evaluator: Evaluator,
    parent_node,
    route: Optional[FlightRouteExtraction],
    index_label: str,
) -> None:
    """
    Build checks for one route option (Route_Option_1 or Route_Option_2).
    index_label should be "1" or "2" (used in node IDs).
    """
    node = evaluator.add_parallel(
        id=f"Route_Option_{index_label}",
        desc=f"{'First' if index_label=='1' else 'Second'} flight route option with airline and URL proof.",
        parent=parent_node,
        critical=True,
    )

    # URL Proof existence (critical)
    url_proof_node = evaluator.add_custom_node(
        result=bool(route and route.url_proofs and len(_unique_non_empty(route.url_proofs)) > 0),
        id=f"Route{index_label}_URL_Proof",
        desc=f"Route {index_label} provides a URL confirming the route exists.",
        parent=node,
        critical=True,
    )

    # Departure region allowed (critical)
    allowed = False
    if route:
        norm = _normalize_region(route.departure_airport, route.departure_region)
        allowed = norm in {"NYC/NJ", "BOS", "MIA/FLL"}

    evaluator.add_custom_node(
        result=allowed,
        id=f"Route{index_label}_Departure_Region_Is_Allowed",
        desc=f"Route {index_label} departs from one of: JFK/EWR, BOS (Logan), MIA/FLL-area airport.",
        parent=node,
        critical=True,
    )

    # Is direct or one-stop to Aruba (critical, URL-verified)
    direct_or_one_stop = evaluator.add_leaf(
        id=f"Route{index_label}_Is_Direct_Or_OneStop_To_Aruba",
        desc=f"Route {index_label} is direct or one-stop to Aruba.",
        parent=node,
        critical=True,
    )

    dep_air = (route.departure_airport or route.departure_region or "the departure airport") if route else "the departure airport"
    claim_route = f"There is a published flight option from {dep_air} to Aruba (AUA) that is either nonstop/direct or has no more than one stop."
    await evaluator.verify(
        claim=claim_route,
        node=direct_or_one_stop,
        sources=_unique_non_empty(route.url_proofs if route else []),
        additional_instruction="Use the provided URL(s), such as airline route maps or booking pages, to confirm the route is nonstop or one-stop (≤1 connection).",
        extra_prerequisites=[url_proof_node]  # If URL proof fails, this verification will be skipped.
    )

    # Operating airline specified (critical - presence in answer)
    airline_specified = evaluator.add_custom_node(
        result=bool(route and (route.airline or "").strip()),
        id=f"Route{index_label}_Operating_Airline_Specified",
        desc=f"Route {index_label} specifies the operating airline.",
        parent=node,
        critical=True,
    )


async def build_flight_logistics_checks(evaluator: Evaluator, parent_node, flights: FlightLogisticsExtraction) -> None:
    flight_node = evaluator.add_parallel(
        id="Flight_Logistics_Plan",
        desc="Identifies flight routes (direct or one-stop) from at least two of the three specified departure regions, with operating airline and URL confirmation for each route.",
        parent=parent_node,
        critical=True,
    )

    # Two distinct regions covered (custom check)
    regs = _distinct_allowed_regions(flights.routes or [])
    evaluator.add_custom_node(
        result=len(regs) >= 2,
        id="Two_Distinct_Regions_Covered",
        desc="Includes routes from at least two distinct regions among: (1) NYC/NJ (JFK or Newark), (2) Boston (Logan), (3) Miami/Fort Lauderdale area.",
        parent=flight_node,
        critical=True,
    )

    # Route option 1 and 2 (use first two routes if present; pad with None to force failure when missing)
    r1 = flights.routes[0] if len(flights.routes) >= 1 else None
    r2 = flights.routes[1] if len(flights.routes) >= 2 else None

    await _build_one_route_checks(evaluator, flight_node, r1, "1")
    await _build_one_route_checks(evaluator, flight_node, r2, "2")


async def build_ada_overall_checks(evaluator: Evaluator, parent_node, basics: ConferenceBasicsExtraction) -> None:
    ada_node = evaluator.add_parallel(
        id="ADA_Overall_Verification",
        desc="Proposal explicitly verifies that the overall plan meets ADA accessibility standards (in light of the stated hotel accessibility requirements).",
        parent=parent_node,
        critical=True,
    )

    ada_leaf = evaluator.add_leaf(
        id="Explicit_ADA_Compliance_Verification_Statement",
        desc="Includes an explicit statement that the plan meets ADA accessibility requirements (guest rooms + meeting spaces).",
        parent=ada_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The proposal includes an explicit statement that the overall plan meets ADA accessibility requirements for both guest rooms and meeting/conference spaces.",
        node=ada_leaf,
        additional_instruction="Judge this only from the answer text."
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
    Evaluate a single answer for the Aruba conference planning task and return a structured result dictionary.
    """
    # Initialize evaluator (framework root is a wrapper; task tree root will be under it)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall aggregation of major sections in parallel
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

    # Extract structured information in parallel
    basics_task = evaluator.extract(
        prompt=prompt_extract_conference_basics(),
        template_class=ConferenceBasicsExtraction,
        extraction_name="conference_basics",
    )
    hotel_task = evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelExtraction,
        extraction_name="hotel_venue_info",
    )
    flights_task = evaluator.extract(
        prompt=prompt_extract_flight_logistics(),
        template_class=FlightLogisticsExtraction,
        extraction_name="flight_logistics",
    )

    basics, hotel, flights = await asyncio.gather(basics_task, hotel_task, flights_task)

    # Build the rubric tree under a critical top-level node
    top = evaluator.add_parallel(
        id="Conference_Planning_Proposal",
        desc="Complete venue + travel plan for a 3-day Aruba conference for 100 attendees in June 2026, meeting all stated constraints and providing required URL support.",
        parent=root,
        critical=True,
    )

    # Subsections
    await build_conference_basics_checks(evaluator, top, basics)
    await build_hotel_venue_plan_checks(evaluator, top, hotel)
    await build_flight_logistics_checks(evaluator, top, flights)
    await build_ada_overall_checks(evaluator, top, basics)

    # Return structured summary
    return evaluator.get_summary()