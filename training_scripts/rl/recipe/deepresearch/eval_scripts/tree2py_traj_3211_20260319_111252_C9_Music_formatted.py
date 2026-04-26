import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "stapleton_summer_2026_concerts"
TASK_DESCRIPTION = (
    "I'm a Chris Stapleton fan planning to attend multiple concerts during summer 2026. "
    "Please help me find three concerts from his All-American Road Show 2026 tour that meet the following requirements:\n\n"
    "1. Each concert must be scheduled between May 1 and August 31, 2026\n"
    "2. Each concert must be in a different U.S. state\n"
    "3. At least two of the three concerts must be at large stadium venues with a seating capacity of 60,000 or more\n"
    "4. Tickets or VIP packages must be available for purchase\n\n"
    "For each concert, please provide:\n"
    "- Venue name, city, and state\n"
    "- Concert date\n"
    "- Venue seating capacity\n"
    "- Any special guest performers listed for that show\n"
    "- A link to purchase tickets or VIP packages\n"
    "- A reference link confirming the venue's seating capacity\n\n"
    "All information should be verifiable through Chris Stapleton's official tour website (chrisstapleton.com/tour) or major ticketing platforms."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConcertItem(BaseModel):
    # Core event details
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # full name or 2-letter
    concert_date: Optional[str] = None  # keep as string for flexibility

    # Tour & artist
    tour_name: Optional[str] = None  # e.g., "All-American Road Show 2026"
    tour_reference_url: Optional[str] = None  # should be from chrisstapleton.com/tour

    # Tickets
    ticket_status: Optional[str] = None  # e.g., "On Sale", "Available", "Sold Out", etc.
    purchase_url: Optional[str] = None
    ticketing_platform: Optional[str] = None  # e.g., Ticketmaster, SeatGeek, AXS

    # Guests
    special_guests: List[str] = Field(default_factory=list)

    # Capacity
    venue_capacity: Optional[str] = None  # string to allow ranges or approx
    capacity_reference_url: Optional[str] = None

    # Date confirmation source (could be same as tour or ticket page)
    date_reference_url: Optional[str] = None


class ConcertsExtraction(BaseModel):
    concerts: List[ConcertItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concerts() -> str:
    return """
    Extract up to five (5) concerts from the answer that are claimed to be part of Chris Stapleton's All-American Road Show 2026 tour.
    For each concert, extract the following fields exactly as they appear in the answer:

    - venue_name: The venue/stadium/arena name
    - city: City of the event
    - state: U.S. state (full name or 2-letter abbreviation)
    - concert_date: The concert date as stated in the answer (keep the original format)
    - tour_name: The tour name as written, if provided (e.g., "All-American Road Show 2026")
    - tour_reference_url: A URL from chrisstapleton.com/tour that references this specific event; if the answer cites it, include it; otherwise null
    - ticket_status: The ticket availability status text from the answer (e.g., "On Sale", "Available", "VIP packages available", "Sold Out"); if not specified, set to null
    - purchase_url: A direct URL for purchasing tickets or VIP packages for this concert (e.g., Ticketmaster, SeatGeek, AXS, or official site)
    - ticketing_platform: The name of the platform for the purchase_url (e.g., Ticketmaster, SeatGeek, AXS, Venue site); if unclear or not given, null
    - special_guests: A list of special guests (strings) listed for this concert; if none are mentioned, return an empty list
    - venue_capacity: The seating capacity (string) for the venue as reported in the answer (allow ranges or approximations); if not present, null
    - capacity_reference_url: A URL that the answer uses to support the capacity number (e.g., venue official page, Wikipedia, etc.); if not present, null
    - date_reference_url: A URL (could be the tour page or ticketing page) that supports the specific date; include it if present in the answer; otherwise null

    Rules:
    - Do NOT invent or infer any information that is not explicitly present in the answer.
    - Always extract full URLs as provided in the answer; allow markdown links by extracting the actual link target.
    - If any field is missing in the answer for a concert, set it to null (or empty list for special_guests).
    - Keep all text fields exactly as in the answer (do not normalize or reformat).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def coalesce_sources(*urls: Optional[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for u in urls:
        if non_empty(u):
            u2 = str(u).strip()
            if u2 not in seen:
                ordered.append(u2)
                seen.add(u2)
    return ordered


def parse_capacity_to_int(capacity_text: Optional[str]) -> int:
    """
    Try to extract a reasonable capacity integer from a textual capacity string.
    Strategy: find all integer-like tokens (with commas allowed), take the maximum.
    Returns 0 if parsing fails.
    """
    if not non_empty(capacity_text):
        return 0
    text = capacity_text.lower()
    # remove words like 'approx', '+', etc., but we'll just find numbers
    nums = re.findall(r"\b\d{1,3}(?:,\d{3})*\b", text)
    if not nums:
        return 0
    try:
        max_num = max(int(n.replace(",", "")) for n in nums)
        return max_num
    except Exception:
        return 0


def normalize_state_for_uniqueness(state: Optional[str]) -> Optional[str]:
    if not non_empty(state):
        return None
    return state.strip().upper()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_concert(
    evaluator: Evaluator,
    parent_node,
    concert: ConcertItem,
    idx_one_based: int,
):
    """
    Build verification subtree for one concert.
    Adjustments: To satisfy framework constraint (critical parents cannot have non-critical children),
    we set group parents as non-critical while marking essential children as critical.
    """
    # Create concert node container (parallel, non-critical)
    concert_node = evaluator.add_parallel(
        id=f"concert_{idx_one_based}",
        desc=f"Concert #{idx_one_based} meets the specified requirements",
        parent=parent_node,
        critical=False
    )

    # ---------------------- Artist & Tour Info (parent non-critical) ----------------------
    artist_tour_node = evaluator.add_parallel(
        id=f"c{idx_one_based}_artist_tour_info",
        desc="Artist and tour information",
        parent=concert_node,
        critical=False  # parent non-critical; criticality pushed to children
    )

    # c{i}_artist_verification (critical)
    node_artist_ver = evaluator.add_leaf(
        id=f"c{idx_one_based}_artist_verification",
        desc="Concert is from Chris Stapleton's All-American Road Show tour",
        parent=artist_tour_node,
        critical=True
    )
    claim_artist_ver = (
        f"This page confirms that the event featuring Chris Stapleton is part of his All-American Road Show tour"
        f" (2026 or a continuing 'All-American Road Show' branding) and matches the listed concert details"
        f" such as venue '{concert.venue_name}', city '{concert.city}', state '{concert.state}', or date '{concert.concert_date}'."
    )
    await evaluator.verify(
        claim=claim_artist_ver,
        node=node_artist_ver,
        sources=coalesce_sources(concert.tour_reference_url, concert.purchase_url),
        additional_instruction="Allow reasonable variations in naming (e.g., 'All-American Road Show' vs 'All American Road Show'). A match on the official tour site is preferred."
    )

    # c{i}_headliner_status (critical)
    node_headliner = evaluator.add_leaf(
        id=f"c{idx_one_based}_headliner_status",
        desc="Chris Stapleton is the headlining artist",
        parent=artist_tour_node,
        critical=True
    )
    claim_headliner = (
        f"The event page indicates that Chris Stapleton is the headlining artist for this concert"
        f" at '{concert.venue_name}' in {concert.city}, {concert.state}."
    )
    await evaluator.verify(
        claim=claim_headliner,
        node=node_headliner,
        sources=coalesce_sources(concert.tour_reference_url, concert.purchase_url),
        additional_instruction="Headlining means he is the main billed artist for the show."
    )

    # c{i}_special_guests (non-critical)
    node_guests = evaluator.add_leaf(
        id=f"c{idx_one_based}_special_guests",
        desc="Special guest performers are identified if applicable",
        parent=artist_tour_node,
        critical=False
    )
    guests_text = ", ".join(concert.special_guests) if concert.special_guests else "none"
    claim_guests = (
        f"The page's listing of special guests for this concert is: {guests_text}. "
        f"If no special guests are listed on the page, then reporting 'none' is correct."
    )
    await evaluator.verify(
        claim=claim_guests,
        node=node_guests,
        sources=coalesce_sources(concert.tour_reference_url, concert.purchase_url),
        additional_instruction="Accept equivalently named guest artist entries. If the page does not list guests, 'none' is acceptable."
    )

    # c{i}_tour_reference_url (critical)
    node_tour_url = evaluator.add_leaf(
        id=f"c{idx_one_based}_tour_reference_url",
        desc="Reference URL from Chris Stapleton's official tour page",
        parent=artist_tour_node,
        critical=True
    )
    claim_tour_url = (
        f"This URL is an official tour page on Chris Stapleton's website (chrisstapleton.com/tour) for the concert"
        f" at '{concert.venue_name}' in {concert.city}, {concert.state} on '{concert.concert_date}'."
    )
    await evaluator.verify(
        claim=claim_tour_url,
        node=node_tour_url,
        sources=concert.tour_reference_url,
        additional_instruction="The domain should be chrisstapleton.com/tour and the page should clearly correspond to this specific event."
    )

    # ---------------------- Venue Details (parent non-critical) ---------------------------
    venue_node = evaluator.add_parallel(
        id=f"c{idx_one_based}_venue_details",
        desc="Complete and accurate venue information",
        parent=concert_node,
        critical=False
    )

    # Existence checks (non-critical)
    evaluator.add_custom_node(
        result=non_empty(concert.venue_name),
        id=f"c{idx_one_based}_venue_name",
        desc="Venue name is provided",
        parent=venue_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty(concert.city),
        id=f"c{idx_one_based}_venue_city",
        desc="City name is provided",
        parent=venue_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty(concert.state),
        id=f"c{idx_one_based}_venue_state",
        desc="U.S. state is provided",
        parent=venue_node,
        critical=False
    )

    # c{i}_us_location (critical)
    node_us_loc = evaluator.add_leaf(
        id=f"c{idx_one_based}_us_location",
        desc="Venue is located in the United States",
        parent=venue_node,
        critical=True
    )
    claim_us_loc = (
        f"The event location '{concert.city}, {concert.state}' is in the United States, "
        f"and the page indicates the event is in the U.S."
    )
    await evaluator.verify(
        claim=claim_us_loc,
        node=node_us_loc,
        sources=coalesce_sources(concert.tour_reference_url, concert.purchase_url),
        additional_instruction="Rely on the event listing location; U.S. territories should be treated cautiously; focus on U.S. states."
    )

    # c{i}_venue_capacity (critical existence)
    evaluator.add_custom_node(
        result=non_empty(concert.venue_capacity),
        id=f"c{idx_one_based}_venue_capacity",
        desc="Venue seating capacity is provided",
        parent=venue_node,
        critical=True
    )

    # c{i}_capacity_reference_url (critical, verify by URL)
    node_cap_ref = evaluator.add_leaf(
        id=f"c{idx_one_based}_capacity_reference_url",
        desc="Reference URL confirming venue capacity",
        parent=venue_node,
        critical=True
    )
    claim_capacity = (
        f"The venue '{concert.venue_name}' has a typical seating capacity of {concert.venue_capacity} (approximately)."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=node_cap_ref,
        sources=concert.capacity_reference_url,
        additional_instruction="The page should explicitly state the venue's capacity or provide a commonly cited capacity. "
                               "Allow minor variations or ranges; 'seating capacity' vs 'capacity' is acceptable."
    )

    # ---------------------- Date Requirements (parent non-critical) -----------------------
    date_node = evaluator.add_parallel(
        id=f"c{idx_one_based}_date_requirements",
        desc="Concert date meets specified requirements",
        parent=concert_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=non_empty(concert.concert_date),
        id=f"c{idx_one_based}_specific_date",
        desc="Specific concert date is provided",
        parent=date_node,
        critical=False
    )

    node_date_in_range = evaluator.add_leaf(
        id=f"c{idx_one_based}_date_in_range",
        desc="Concert date is between May 1 and August 31, 2026",
        parent=date_node,
        critical=True
    )
    claim_date_range = (
        f"The concert date '{concert.concert_date}' falls between May 1, 2026 and August 31, 2026 inclusive."
    )
    await evaluator.verify(
        claim=claim_date_range,
        node=node_date_in_range,
        additional_instruction="Interpret common date formats (e.g., 'June 14, 2026', '6/14/26'). Focus only on the date range check."
    )

    node_date_ref = evaluator.add_leaf(
        id=f"c{idx_one_based}_date_reference_url",
        desc="Reference URL confirming concert date",
        parent=date_node,
        critical=True
    )
    claim_date_supported = (
        f"This page confirms that the concert at '{concert.venue_name}' in {concert.city}, {concert.state} "
        f"is scheduled on '{concert.concert_date}'."
    )
    await evaluator.verify(
        claim=claim_date_supported,
        node=node_date_ref,
        sources=coalesce_sources(concert.date_reference_url, concert.tour_reference_url, concert.purchase_url),
        additional_instruction="Allow minor formatting differences in the date; the event entry should clearly match the same show."
    )

    # ---------------------- Ticket Information (parent non-critical) ----------------------
    ticket_node = evaluator.add_parallel(
        id=f"c{idx_one_based}_ticket_information",
        desc="Ticket purchasing information is available",
        parent=concert_node,
        critical=False
    )

    # c{i}_ticket_status (non-critical) - availability
    node_ticket_status = evaluator.add_leaf(
        id=f"c{idx_one_based}_ticket_status",
        desc="Ticket or VIP package availability status",
        parent=ticket_node,
        critical=False
    )
    claim_ticket_status = (
        "Tickets or VIP packages are available for purchase for this event (e.g., 'On Sale', 'Get Tickets', or similar). "
        "If the page indicates 'Sold Out' everywhere with no purchase option, then this claim is false."
    )
    await evaluator.verify(
        claim=claim_ticket_status,
        node=node_ticket_status,
        sources=concert.purchase_url,
        additional_instruction="Look for active 'Buy Tickets', 'Get Tickets', 'VIP', or similar purchase flows. "
                               "Resale availability counts as 'available'."
    )

    # c{i}_ticket_purchase_url (critical) - existence
    evaluator.add_custom_node(
        result=non_empty(concert.purchase_url),
        id=f"c{idx_one_based}_ticket_purchase_url",
        desc="URL for purchasing tickets or VIP packages is provided",
        parent=ticket_node,
        critical=True
    )

    # c{i}_ticketing_platform (non-critical) - identification/existence
    evaluator.add_custom_node(
        result=non_empty(concert.ticketing_platform),
        id=f"c{idx_one_based}_ticketing_platform",
        desc="Ticketing platform is identified (Ticketmaster, SeatGeek, official site, etc.)",
        parent=ticket_node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Diversity and special requirements                                          #
# --------------------------------------------------------------------------- #
def compute_different_states(concerts: List[ConcertItem]) -> bool:
    states = [normalize_state_for_uniqueness(c.state) for c in concerts]
    if any(s is None or s == "" for s in states):
        return False
    return len(set(states)) == 3


def compute_at_least_two_stadiums(concerts: List[ConcertItem]) -> bool:
    capacities = [parse_capacity_to_int(c.venue_capacity) for c in concerts]
    count_60k = sum(1 for cap in capacities if cap >= 60000)
    return count_60k >= 2


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an agent's answer for the Chris Stapleton summer 2026 concerts task.
    """
    # Initialize evaluator (root parallel)
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

    # Extract structured concerts
    extracted = await evaluator.extract(
        prompt=prompt_extract_concerts(),
        template_class=ConcertsExtraction,
        extraction_name="concerts_extraction",
    )

    # Take the first 3 concerts; pad if fewer
    concerts: List[ConcertItem] = list(extracted.concerts[:3])
    while len(concerts) < 3:
        concerts.append(ConcertItem())

    # Build per-concert verification subtrees
    for i in range(3):
        await verify_concert(evaluator, root, concerts[i], i + 1)

    # Diversity and stadium requirements (critical as a group)
    diversity_node = evaluator.add_parallel(
        id="diversity_requirements",
        desc="Concerts meet diversity and special requirements",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=compute_different_states(concerts),
        id="different_states",
        desc="Each of the three concerts is in a different U.S. state",
        parent=diversity_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=compute_at_least_two_stadiums(concerts),
        id="stadium_venues",
        desc="At least two of the three concerts are at stadium venues with capacity of 60,000 or more",
        parent=diversity_node,
        critical=True
    )

    # Return evaluation summary
    return evaluator.get_summary()