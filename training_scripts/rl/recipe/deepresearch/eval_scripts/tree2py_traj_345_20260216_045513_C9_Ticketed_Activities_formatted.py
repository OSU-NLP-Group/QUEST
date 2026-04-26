import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bruno_romantic_tour_apr2026_venues"
TASK_DESCRIPTION = (
    'Bruno Mars is embarking on "The Romantic Tour" in 2026, with multiple stadium shows scheduled across '
    "North America starting in April 2026. For concert planning and accessibility research purposes, identify "
    "four stadium venues hosting Bruno Mars' The Romantic Tour specifically in April 2026, and provide comprehensive "
    "information for each venue including: (1) Venue Identification: The official stadium name, city/state location, "
    "and specific concert date(s) in April 2026; (2) Capacity Information: The stadium's seating capacity; "
    "(3) Accessibility Features: ADA-compliant wheelchair-accessible seating availability and accessible parking near "
    "the entrance; (4) Ticket Information: Current ticket availability status or pricing information; "
    "(5) Transportation Access: On-site parking facilities availability; (6) Venue Policies: Clear bag policy or bag "
    "size restrictions, and outside food/beverage policy. For each piece of information provided, include a reference "
    "URL from an official venue website, tour website, or reliable ticketing platform that confirms the stated "
    "information. The four venues should represent different cities/states from Bruno Mars' April 2026 tour schedule."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueIdentification(BaseModel):
    stadium_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    concert_dates_april_2026: List[str] = Field(default_factory=list)
    stadium_name_urls: List[str] = Field(default_factory=list)
    city_state_urls: List[str] = Field(default_factory=list)
    concert_date_urls: List[str] = Field(default_factory=list)


class VenueCapacity(BaseModel):
    seating_capacity: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)


class VenueAccessibility(BaseModel):
    wheelchair_seating_info: Optional[str] = None
    wheelchair_urls: List[str] = Field(default_factory=list)
    accessible_parking_info: Optional[str] = None
    accessible_parking_urls: List[str] = Field(default_factory=list)


class VenueTickets(BaseModel):
    ticket_info: Optional[str] = None
    ticket_urls: List[str] = Field(default_factory=list)


class VenueTransportation(BaseModel):
    onsite_parking_info: Optional[str] = None
    parking_urls: List[str] = Field(default_factory=list)


class VenuePolicies(BaseModel):
    bag_policy: Optional[str] = None
    bag_policy_urls: List[str] = Field(default_factory=list)
    outside_food_beverage_policy: Optional[str] = None
    outside_food_beverage_urls: List[str] = Field(default_factory=list)


class VenueItem(BaseModel):
    identification: Optional[VenueIdentification] = None
    capacity: Optional[VenueCapacity] = None
    accessibility: Optional[VenueAccessibility] = None
    tickets: Optional[VenueTickets] = None
    transportation: Optional[VenueTransportation] = None
    policies: Optional[VenuePolicies] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to four (4) stadium venues in April 2026 for Bruno Mars' "The Romantic Tour" as presented in the answer.
    IMPORTANT:
    - Only include shows that occur in April 2026.
    - If the answer lists more than four venues, keep only the first four in the original order.
    - If fewer than four are provided, include those and do NOT invent missing ones.
    - For each required piece of information, also extract the specific reference URL(s) mentioned in the answer (URLs only).

    For each venue, extract these fields:

    identification:
      - stadium_name: Official stadium name (string)
      - city: City (string)
      - state: State (string or abbreviation)
      - concert_dates_april_2026: Array of one or more date strings in April 2026 (e.g., "April 12, 2026")
      - stadium_name_urls: Array of URL strings supporting the official stadium name
      - city_state_urls: Array of URL strings supporting the city/state
      - concert_date_urls: Array of URL strings confirming the April 2026 date(s)

    capacity:
      - seating_capacity: Stadium seating capacity as stated in the answer (string; ranges or approx allowed)
      - capacity_urls: Array of URL strings supporting the capacity

    accessibility:
      - wheelchair_seating_info: Text confirming ADA-compliant wheelchair-accessible seating availability
      - wheelchair_urls: Array of URL strings supporting wheelchair seating availability
      - accessible_parking_info: Text confirming accessible parking near the entrance
      - accessible_parking_urls: Array of URL strings supporting accessible parking near the entrance

    tickets:
      - ticket_info: Current ticket availability status and/or pricing information (string; can be brief)
      - ticket_urls: Array of URL strings supporting ticket availability and/or pricing for this venue/date

    transportation:
      - onsite_parking_info: Text confirming on-site parking availability
      - parking_urls: Array of URL strings supporting on-site parking availability

    policies:
      - bag_policy: The clear bag policy or bag size restrictions (string)
      - bag_policy_urls: Array of URL strings supporting the bag policy
      - outside_food_beverage_policy: The outside food/beverage policy (string)
      - outside_food_beverage_urls: Array of URL strings supporting the outside food/beverage policy

    Return a JSON object:
    {
      "venues": [VenueItem, VenueItem, ...]  // up to 4 items
    }

    SPECIAL RULES:
    - Extract only URLs explicitly present in the answer.
    - Accept URLs in plain form or markdown; output actual URL strings.
    - If a field’s value is missing, set it to null; if its URLs are missing, use [].
    - Do NOT create or infer any data not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
STADIUM_NAME_HINTS = [
    "stadium", "field", "park", "dome", "coliseum", "bowl", "superdome", "speedway", "ballpark"
]

ALLOWED_SOURCE_KEYWORDS = [
    # Official tour / artist
    "brunomars.com",
    # Major ticketing platforms
    "ticketmaster", "livenation", "axs.com", "seatgeek", "stubhub", "vividseats", "tickpick", "tixr",
    # Common official stadium/venue domains often contain these substrings:
    "stadium", "field", "park", "coliseum", "dome", "bowl",
    # Pro teams/league sites sometimes host venue pages:
    "nfl.com", "mlb.com", "mls.com",
    # Universities / .edu may host official stadium pages:
    ".edu",
]


def infer_is_stadium(name: Optional[str]) -> bool:
    if not name:
        return False
    lname = name.lower()
    return any(hint in lname for hint in STADIUM_NAME_HINTS)


def normalize_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def is_allowed_source(url: str) -> bool:
    d = normalize_domain(url)
    if not d:
        return False
    for kw in ALLOWED_SOURCE_KEYWORDS:
        if kw in d:
            return True
    # Heuristic: Many official venues use custom domains ending with .com that include the venue name.
    # As a permissive fallback, accept domains that contain a stadium-like hint.
    for hint in STADIUM_NAME_HINTS:
        if hint in d:
            return True
    return False


def gather_all_urls(venues: List[VenueItem]) -> List[str]:
    urls: List[str] = []
    for v in venues:
        if v.identification:
            urls += v.identification.stadium_name_urls
            urls += v.identification.city_state_urls
            urls += v.identification.concert_date_urls
        if v.capacity:
            urls += v.capacity.capacity_urls
        if v.accessibility:
            urls += v.accessibility.wheelchair_urls
            urls += v.accessibility.accessible_parking_urls
        if v.tickets:
            urls += v.tickets.ticket_urls
        if v.transportation:
            urls += v.transportation.parking_urls
        if v.policies:
            urls += v.policies.bag_policy_urls
            urls += v.policies.outside_food_beverage_urls
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and (u not in seen):
            seen.add(u)
            deduped.append(u)
    return deduped


def city_state_pair(v: VenueItem) -> Optional[Tuple[str, str]]:
    if v.identification and v.identification.city and v.identification.state:
        return (v.identification.city.strip(), v.identification.state.strip())
    return None


def fmt_dates(dates: List[str]) -> str:
    if not dates:
        return ""
    return "; ".join(dates)


# --------------------------------------------------------------------------- #
# Venue verification subroutine                                               #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    venue_index: int
) -> None:
    # Create Venue node (non-critical; partial credit allowed per venue)
    venue_node = evaluator.add_parallel(
        id=f"Venue_{venue_index + 1}",
        desc=f"Venue #{venue_index + 1} (one of the four April 2026 stadium shows).",
        parent=parent_node,
        critical=False
    )

    ident = venue.identification or VenueIdentification()
    cap = venue.capacity or VenueCapacity()
    acc = venue.accessibility or VenueAccessibility()
    tix = venue.tickets or VenueTickets()
    trans = venue.transportation or VenueTransportation()
    pol = venue.policies or VenuePolicies()

    # ------------------- Identification (critical) ------------------- #
    ident_node = evaluator.add_parallel(
        id=f"V{venue_index + 1}_Identification",
        desc="Venue identification and April 2026 show date(s), each supported by reference URL(s).",
        parent=venue_node,
        critical=True
    )

    # Stadium Name - Reference URL presence (critical)
    v_name_ref = evaluator.add_custom_node(
        result=bool(ident.stadium_name_urls),
        id=f"V{venue_index + 1}_Official_Stadium_Name_Reference_URL",
        desc="Provides reference URL(s) confirming the official stadium name.",
        parent=ident_node,
        critical=True
    )
    # Stadium Name - Value verification (critical)
    v_name_leaf = evaluator.add_leaf(
        id=f"V{venue_index + 1}_Official_Stadium_Name",
        desc="Provides the official stadium name.",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official stadium name is '{ident.stadium_name}'.",
        node=v_name_leaf,
        sources=ident.stadium_name_urls,
        additional_instruction="Verify the page shows the venue's official stadium name (minor variations in sponsor naming acceptable).",
        extra_prerequisites=[v_name_ref]
    )

    # City/State - Reference URL presence (critical)
    v_city_ref = evaluator.add_custom_node(
        result=bool(ident.city_state_urls),
        id=f"V{venue_index + 1}_City_State_Reference_URL",
        desc="Provides reference URL(s) confirming the city and state location.",
        parent=ident_node,
        critical=True
    )
    # City/State - Value verification (critical)
    v_city_leaf = evaluator.add_leaf(
        id=f"V{venue_index + 1}_City_State",
        desc="Provides the city and state location.",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stadium is located in {ident.city}, {ident.state}.",
        node=v_city_leaf,
        sources=ident.city_state_urls,
        additional_instruction="Verify the city and state for this stadium (minor formatting variations acceptable).",
        extra_prerequisites=[v_city_ref]
    )

    # Dates in April 2026 - Reference URL presence (critical)
    v_date_ref = evaluator.add_custom_node(
        result=bool(ident.concert_date_urls),
        id=f"V{venue_index + 1}_Concert_Dates_Reference_URL",
        desc="Provides reference URL(s) confirming the April 2026 concert date(s).",
        parent=ident_node,
        critical=True
    )
    # Dates in April 2026 - Value verification (critical)
    v_date_leaf = evaluator.add_leaf(
        id=f"V{venue_index + 1}_Concert_Dates_April_2026",
        desc="Provides specific concert date(s) in April 2026.",
        parent=ident_node,
        critical=True
    )
    dates_str = fmt_dates(ident.concert_dates_april_2026)
    await evaluator.verify(
        claim=f"There is at least one Bruno Mars concert at this stadium in April 2026 on these date(s): {dates_str}.",
        node=v_date_leaf,
        sources=ident.concert_date_urls,
        additional_instruction="Confirm at least one of the listed dates is an April 2026 Bruno Mars show at this venue. Accept minor naming variations for the tour.",
        extra_prerequisites=[v_date_ref]
    )

    # ------------------- Capacity (critical) ------------------- #
    cap_node = evaluator.add_parallel(
        id=f"V{venue_index + 1}_Capacity",
        desc="Stadium seating capacity, supported by reference URL(s).",
        parent=venue_node,
        critical=True
    )

    cap_ref = evaluator.add_custom_node(
        result=bool(cap.capacity_urls),
        id=f"V{venue_index + 1}_Capacity_Reference_URL",
        desc="Provides reference URL(s) confirming the seating capacity.",
        parent=cap_node,
        critical=True
    )
    cap_val = evaluator.add_leaf(
        id=f"V{venue_index + 1}_Seating_Capacity_Value",
        desc="Provides the stadium seating capacity.",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The seating capacity of {ident.stadium_name} is '{cap.seating_capacity}'.",
        node=cap_val,
        sources=cap.capacity_urls,
        additional_instruction="Verify the stated seating capacity. Accept approximate or range values if commonly reported.",
        extra_prerequisites=[cap_ref]
    )

    # ------------------- Accessibility (critical) ------------------- #
    acc_node = evaluator.add_parallel(
        id=f"V{venue_index + 1}_Accessibility",
        desc="Accessibility features (ADA), each supported by reference URL(s).",
        parent=venue_node,
        critical=True
    )

    # Wheelchair seating
    acc_w_ref = evaluator.add_custom_node(
        result=bool(acc.wheelchair_urls),
        id=f"V{venue_index + 1}_Wheelchair_Accessible_Seating_Reference_URL",
        desc="Provides reference URL(s) confirming wheelchair-accessible seating availability.",
        parent=acc_node,
        critical=True
    )
    acc_w_val = evaluator.add_leaf(
        id=f"V{venue_index + 1}_Wheelchair_Accessible_Seating",
        desc="Confirms ADA-compliant wheelchair-accessible seating availability.",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The stadium provides ADA-compliant wheelchair-accessible seating.",
        node=acc_w_val,
        sources=acc.wheelchair_urls,
        additional_instruction="Verify mention of ADA-compliant wheelchair-accessible seating (may be on accessibility or ticketing pages).",
        extra_prerequisites=[acc_w_ref]
    )

    # Accessible parking near entrance
    acc_p_ref = evaluator.add_custom_node(
        result=bool(acc.accessible_parking_urls),
        id=f"V{venue_index + 1}_Accessible_Parking_Reference_URL",
        desc="Provides reference URL(s) confirming accessible parking near the entrance.",
        parent=acc_node,
        critical=True
    )
    acc_p_val = evaluator.add_leaf(
        id=f"V{venue_index + 1}_Accessible_Parking_Near_Entrance",
        desc="Confirms accessible parking near the entrance.",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Accessible parking near or close to an entrance is available at the stadium.",
        node=acc_p_val,
        sources=acc.accessible_parking_urls,
        additional_instruction="Verify accessible/ADA parking availability and that it is near or convenient to an entrance.",
        extra_prerequisites=[acc_p_ref]
    )

    # ------------------- Tickets (critical) ------------------- #
    tix_node = evaluator.add_parallel(
        id=f"V{venue_index + 1}_Tickets",
        desc="Ticket availability status or pricing, supported by reference URL(s).",
        parent=venue_node,
        critical=True
    )

    tix_ref = evaluator.add_custom_node(
        result=bool(tix.ticket_urls),
        id=f"V{venue_index + 1}_Ticket_Reference_URL",
        desc="Provides reference URL(s) confirming the ticket availability and/or pricing information.",
        parent=tix_node,
        critical=True
    )
    tix_val = evaluator.add_leaf(
        id=f"V{venue_index + 1}_Ticket_Availability_Or_Pricing",
        desc="Provides current ticket availability status or pricing information.",
        parent=tix_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Ticket availability or pricing information applies to Bruno Mars at {ident.stadium_name}: {tix.ticket_info}.",
        node=tix_val,
        sources=tix.ticket_urls,
        additional_instruction="Verify the referenced page shows current availability or pricing for Bruno Mars at this venue (date within April 2026). Minor variations acceptable.",
        extra_prerequisites=[tix_ref]
    )

    # ------------------- Transportation / Parking (critical) ------------------- #
    trans_node = evaluator.add_parallel(
        id=f"V{venue_index + 1}_Transportation_Parking",
        desc="Transportation access / on-site parking facilities availability, supported by reference URL(s).",
        parent=venue_node,
        critical=True
    )

    trans_ref = evaluator.add_custom_node(
        result=bool(trans.parking_urls),
        id=f"V{venue_index + 1}_Parking_Reference_URL",
        desc="Provides reference URL(s) confirming on-site parking availability.",
        parent=trans_node,
        critical=True
    )
    trans_val = evaluator.add_leaf(
        id=f"V{venue_index + 1}_Onsite_Parking_Availability",
        desc="Confirms on-site parking facilities availability.",
        parent=trans_node,
        critical=True
    )
    await evaluator.verify(
        claim="On-site parking is available at the stadium.",
        node=trans_val,
        sources=trans.parking_urls,
        additional_instruction="Verify that on-site parking is available (may be event-day or general parking info).",
        extra_prerequisites=[trans_ref]
    )

    # ------------------- Policies (critical) ------------------- #
    pol_node = evaluator.add_parallel(
        id=f"V{venue_index + 1}_Policies",
        desc="Venue policies (bags and outside food/beverage), each supported by reference URL(s).",
        parent=venue_node,
        critical=True
    )

    # Bag policy
    pol_b_ref = evaluator.add_custom_node(
        result=bool(pol.bag_policy_urls),
        id=f"V{venue_index + 1}_Bag_Policy_Reference_URL",
        desc="Provides reference URL(s) confirming the bag policy or bag size restrictions.",
        parent=pol_node,
        critical=True
    )
    pol_b_val = evaluator.add_leaf(
        id=f"V{venue_index + 1}_Clear_Bag_Or_Bag_Size_Policy",
        desc="States the clear bag policy or bag size restrictions.",
        parent=pol_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Bag policy for the stadium: {pol.bag_policy}.",
        node=pol_b_val,
        sources=pol.bag_policy_urls,
        additional_instruction="Verify a clear bag policy or bag size restriction is described. Minor wording differences acceptable.",
        extra_prerequisites=[pol_b_ref]
    )

    # Outside food/beverage
    pol_f_ref = evaluator.add_custom_node(
        result=bool(pol.outside_food_beverage_urls),
        id=f"V{venue_index + 1}_Outside_Food_Beverage_Reference_URL",
        desc="Provides reference URL(s) confirming the outside food/beverage policy.",
        parent=pol_node,
        critical=True
    )
    pol_f_val = evaluator.add_leaf(
        id=f"V{venue_index + 1}_Outside_Food_Beverage_Policy",
        desc="States the outside food/beverage policy.",
        parent=pol_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Outside food and beverage policy: {pol.outside_food_beverage_policy}.",
        node=pol_f_val,
        sources=pol.outside_food_beverage_urls,
        additional_instruction="Verify whether outside food and/or beverages are permitted or restricted at the venue.",
        extra_prerequisites=[pol_f_ref]
    )


# --------------------------------------------------------------------------- #
# Global constraints verification                                             #
# --------------------------------------------------------------------------- #
def compute_exactly_four(venues_extracted: VenuesExtraction) -> bool:
    # We instruct the extractor to return at most 4 items. If it returns exactly 4, pass.
    return venues_extracted is not None and venues_extracted.venues is not None and len(venues_extracted.venues) == 4


def compute_all_stadiums(venues_list: List[VenueItem]) -> bool:
    if not venues_list or len(venues_list) == 0:
        return False
    for v in venues_list:
        if not v.identification or not infer_is_stadium(v.identification.stadium_name):
            return False
    return True


def compute_all_distinct_city_state(venues_list: List[VenueItem]) -> bool:
    if not venues_list or len(venues_list) < 2:
        return False
    pairs = []
    for v in venues_list:
        cs = city_state_pair(v)
        if not cs:
            return False
        pairs.append((cs[0].lower(), cs[1].lower()))
    return len(set(pairs)) == len(pairs)


def compute_allowed_sources(venues_list: List[VenueItem]) -> bool:
    urls = gather_all_urls(venues_list)
    if not urls:
        return False
    # Consider allowed if each URL matches allowed sources (permissive heuristics as defined).
    return all(is_allowed_source(u) for u in urls)


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
    # Initialize evaluator with a parallel root strategy
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

    # Extract structured venues info from the answer
    venues_extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Create main task node
    task_node = evaluator.add_parallel(
        id="Bruno_Mars_April_2026_Tour_Venues",
        desc="Identify exactly four stadium venues hosting Bruno Mars' The Romantic Tour in April 2026 and provide required venue details with supporting reference URLs.",
        parent=root,
        critical=False
    )

    # Global Constraints (critical)
    global_node = evaluator.add_parallel(
        id="Global_Constraints",
        desc="Global requirements that apply across the full set of four venues.",
        parent=task_node,
        critical=True
    )

    # Determine which venues to operate on (use at most first 4, pad for structure only)
    original_venues = (venues_extracted.venues or [])[:4]
    venues_for_tree = list(original_venues)  # copy
    while len(venues_for_tree) < 4:
        venues_for_tree.append(VenueItem())

    # Exactly Four Venues
    evaluator.add_custom_node(
        result=compute_exactly_four(venues_extracted),
        id="Exactly_Four_Venues",
        desc="Provides exactly four venues (no more, no fewer).",
        parent=global_node,
        critical=True
    )

    # All Venues Are Stadiums
    evaluator.add_custom_node(
        result=compute_all_stadiums(original_venues) if original_venues else False,
        id="All_Venues_Are_Stadiums",
        desc="All four venues are stadium venues (not arenas or smaller venues).",
        parent=global_node,
        critical=True
    )

    # All Venues Different City/State
    evaluator.add_custom_node(
        result=compute_all_distinct_city_state(original_venues) if original_venues else False,
        id="All_Venues_Different_City_State",
        desc="The four venues are in different city/state pairs (no duplicates).",
        parent=global_node,
        critical=True
    )

    # References From Allowed Sources
    evaluator.add_custom_node(
        result=compute_allowed_sources(original_venues) if original_venues else False,
        id="References_From_Allowed_Sources",
        desc="All reference URLs come from an official venue site, official tour site, or reliable ticketing platform.",
        parent=global_node,
        critical=True
    )

    # Per-venue verification (Venue_1 ... Venue_4)
    for idx in range(4):
        await verify_single_venue(
            evaluator=evaluator,
            parent_node=task_node,
            venue=venues_for_tree[idx],
            venue_index=idx
        )

    return evaluator.get_summary()