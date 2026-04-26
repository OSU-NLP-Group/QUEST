import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "venues_2026_ticketed"
TASK_DESCRIPTION = """
Identify 4 venues in the United States that are hosting ticketed entertainment events in 2026, meeting the following requirements:

1. Venue 1: Located in California with a seating or standing capacity under 500 people
2. Venue 2: Located in California with a seating or standing capacity over 1,000 people
3. Venue 3: Located outside of California with a seating or standing capacity under 500 people
4. Venue 4: Located in New York State with an event taking place specifically in March 2026

For each venue, provide:
- The venue's complete street address, city, and state
- The venue's seating or standing capacity
- The name of the specific ticketed entertainment event being hosted
- The exact date of the event in 2026
- Evidence of ticketed admission (ticket prices or confirmation that tickets are being sold)
- A reference URL that supports this information

All events must be ticketed (paid admission required) and must include live performances, screenings, or special entertainment presentations scheduled for 2026.
"""
YEAR = 2026

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    venue_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None
    event_name: Optional[str] = None
    event_date: Optional[str] = None
    ticket_info: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venue1: Optional[VenueItem] = None  # CA, capacity < 500
    venue2: Optional[VenueItem] = None  # CA, capacity > 1000
    venue3: Optional[VenueItem] = None  # non-CA, capacity < 500
    venue4: Optional[VenueItem] = None  # NY, event in March 2026


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return f"""
Extract structured information for exactly four venues that match the specified constraints. The fields must be extracted exactly as they appear in the answer text. Do not invent or infer missing information.

For each required venue, extract the following fields:
- venue_name: The name of the venue (if stated in the answer)
- street_address: The street address (e.g., "123 Main St")
- city: The city
- state: The state (use standard 2-letter postal abbreviations like CA, NY if the answer uses them; otherwise extract the full state name as shown)
- capacity: The seating or standing capacity as stated (keep any qualifiers like "approx.", "up to", "max")
- event_name: The name of the ticketed entertainment event
- event_date: The exact event date in 2026 as stated in the answer (e.g., "March 14, 2026"; keep the formatting from the answer)
- ticket_info: Any quoted ticket-related text indicating paid admission or prices (e.g., "Tickets from $29", "Buy Tickets", "General Admission $20-40")
- reference_urls: A list of 1-5 explicit URLs provided in the answer that support the venue/event/capacity/ticket details. Only include valid URLs explicitly present in the answer text.

Return a JSON object with the following top-level fields, each containing the above subfields:
- venue1: California venue with capacity under 500
- venue2: California venue with capacity over 1,000
- venue3: Outside California with capacity under 500
- venue4: New York State venue with an event in March 2026

If any field is missing in the answer, set it to null (or [] for reference_urls). Never fabricate information or URLs.
All events must be ticketed (paid admission required) and must include live performances, screenings, or special entertainment presentations scheduled for {YEAR}.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def compose_address(item: VenueItem) -> str:
    parts = [p for p in [item.street_address, item.city, item.state] if p and str(p).strip()]
    return ", ".join(parts)


def normalize_state_name(state: Optional[str]) -> str:
    return (state or "").strip()


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    root,
    group_id: str,
    group_desc: str,
    leaf_prefix: str,
    item: VenueItem,
    *,
    require_state_in: Optional[List[str]] = None,
    require_state_not_in: Optional[List[str]] = None,
    capacity_condition: str = "any",  # one of: "lt500", "gt1000", "any"
    require_month: Optional[str] = None,
    event_year: int = YEAR,
) -> None:
    # Parent node for this venue
    parent_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=root,
        critical=False
    )

    sources = item.reference_urls if item and item.reference_urls else []

    # 1) Location leaf
    location_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Location",
        desc="Provide the venue's complete address including street address, city, and state",
        parent=parent_node,
        critical=True
    )

    addr = compose_address(item or VenueItem())
    state = normalize_state_name(item.state if item else None)
    venue_name = item.venue_name if item and item.venue_name else "the venue"

    # Build location claim with constraints baked in
    constraint_clause = ""
    if require_state_in:
        # Examples: CA, California; NY, New York
        target_states_text = " or ".join(require_state_in)
        constraint_clause = f" The state is required to be {target_states_text}."
    elif require_state_not_in:
        banned_states_text = " or ".join(require_state_not_in)
        constraint_clause = f" The state must NOT be {banned_states_text}."

    location_claim = (
        f"The page shows that {venue_name} is located at '{addr}'. "
        f"The extracted state is '{state}'.{constraint_clause}"
    )

    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the webpage explicitly presents the complete address including street address, city, and state. "
            "Allow common abbreviations (e.g., 'CA' for California, 'NY' for New York). "
            "For 'outside California', confirm the state is not California (either 'CA' or 'California')."
        ),
    )

    # 2) Capacity leaf
    capacity_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Capacity",
        desc="Verify the venue's seating or standing capacity requirement",
        parent=parent_node,
        critical=True
    )

    cap_str = (item.capacity or "").strip()
    if capacity_condition == "lt500":
        capacity_claim = (
            f"The webpage indicates the venue capacity is '{cap_str}', which is under 500 people."
        )
        capacity_extra = (
            "Confirm the capacity shown implies fewer than 500 attendees (e.g., 'max 450', 'up to 300', "
            "'approximately 400'). If a range is given, use the upper bound. If multiple configurations are provided, "
            "use the configuration that represents the main/event capacity. Reject if the page indicates 500 or more."
        )
    elif capacity_condition == "gt1000":
        capacity_claim = (
            f"The webpage indicates the venue capacity is '{cap_str}', which is over 1,000 people."
        )
        capacity_extra = (
            "Confirm the capacity shown is greater than 1,000 attendees (e.g., '1,100', 'over 1,000', 'capacity 2,000'). "
            "If a range is given, use the upper bound. Reject if the page indicates 1,000 or fewer."
        )
    else:
        capacity_claim = (
            f"The webpage indicates the venue capacity is '{cap_str}'."
        )
        capacity_extra = (
            "Verify that the webpage explicitly provides a seating or standing capacity value for the venue. "
            "Match the extracted value where possible; allow approximate/phrased capacities."
        )

    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=sources,
        additional_instruction=capacity_extra,
    )

    # 3) Event leaf
    event_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Event",
        desc="Identify the specific ticketed entertainment event with exact date in 2026",
        parent=parent_node,
        critical=True
    )

    event_name = (item.event_name or "").strip()
    event_date = (item.event_date or "").strip()

    if require_month:
        event_claim = (
            f"The webpage shows that {venue_name} will host the event '{event_name}' on '{event_date}', "
            f"which occurs in {require_month} {event_year}."
        )
        event_extra = (
            f"Confirm the page explicitly shows the event '{event_name}' scheduled in {require_month} {event_year}. "
            "Allow common month abbreviations (e.g., 'Mar'), and typical date formats. "
            "The event must be in 2026, specifically in the required month."
        )
    else:
        event_claim = (
            f"The webpage shows that {venue_name} will host the event '{event_name}' on '{event_date}', "
            f"which is in {event_year}."
        )
        event_extra = (
            f"Confirm the page explicitly shows the named event scheduled in {event_year}. "
            "Allow common date formats and abbreviations."
        )

    await evaluator.verify(
        claim=event_claim,
        node=event_leaf,
        sources=sources,
        additional_instruction=(
            event_extra
            + " Also ensure the event is an entertainment presentation (live performance, screening, or special presentation), not a regular operating hour or unticketed open-house."
        ),
    )

    # 4) Ticket information leaf
    ticket_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Ticket_Info",
        desc="Provide evidence of ticketed admission (price or ticket sales information)",
        parent=parent_node,
        critical=True
    )

    ticket_snippet = (item.ticket_info or "").strip()
    ticket_claim = (
        "The event requires paid admission (tickets are being sold), evidenced by explicit ticket prices or a 'Buy Tickets' / 'Purchase' link or button."
        + (f" For example, the page shows: '{ticket_snippet}'." if ticket_snippet else "")
    )

    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_leaf,
        sources=sources,
        additional_instruction=(
            "Look for explicit signals of paid ticketing: prices (e.g., '$25', 'from $19'), links/buttons such as 'Buy Tickets', "
            "'Ticketmaster', 'Etix', 'Eventbrite' with pricing, or language like 'paid admission'. "
            "Reject if the event is free, donation-only, RSVP-only without price, or 'no admission charge'."
        ),
    )

    # 5) Reference URL leaf (existence/validity check)
    # Requirement focuses on providing at least one supporting URL; we treat it as an existence/format check.
    reference_ok = bool(sources) and any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in sources)
    evaluator.add_custom_node(
        result=reference_ok,
        id=f"{leaf_prefix}_Reference",
        desc="Provide reference URL supporting the venue information",
        parent=parent_node,
        critical=True
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
    model: str = "o4-mini",
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root should be non-sequential; venues are independent
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

    # NOTE: Set root critical to False to allow non-critical children per framework constraints
    root.critical = False

    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Store constraints for transparency
    evaluator.add_custom_info(
        info={
            "year": YEAR,
            "constraints": {
                "venue1": {"state": "CA", "capacity": "<500"},
                "venue2": {"state": "CA", "capacity": ">1000"},
                "venue3": {"state": "not CA", "capacity": "<500"},
                "venue4": {"state": "NY", "month": "March", "capacity": "provided"}
            }
        },
        info_type="constraints",
        info_name="task_constraints"
    )

    # Build four venue verification subtrees
    v1 = extracted.venue1 or VenueItem()
    await verify_single_venue(
        evaluator,
        root,
        group_id="Venue_1_California_Under_500",
        group_desc="Identify a venue in California with capacity under 500 people hosting a ticketed entertainment event in 2026",
        leaf_prefix="Venue_1",
        item=v1,
        require_state_in=["CA", "California"],
        capacity_condition="lt500",
        event_year=YEAR
    )

    v2 = extracted.venue2 or VenueItem()
    await verify_single_venue(
        evaluator,
        root,
        group_id="Venue_2_California_Over_1000",
        group_desc="Identify a venue in California with capacity over 1,000 people hosting a ticketed entertainment event in 2026",
        leaf_prefix="Venue_2",
        item=v2,
        require_state_in=["CA", "California"],
        capacity_condition="gt1000",
        event_year=YEAR
    )

    v3 = extracted.venue3 or VenueItem()
    await verify_single_venue(
        evaluator,
        root,
        group_id="Venue_3_Outside_California_Under_500",
        group_desc="Identify a venue outside California with capacity under 500 people hosting a ticketed entertainment event in 2026",
        leaf_prefix="Venue_3",
        item=v3,
        require_state_not_in=["CA", "California"],
        capacity_condition="lt500",
        event_year=YEAR
    )

    v4 = extracted.venue4 or VenueItem()
    await verify_single_venue(
        evaluator,
        root,
        group_id="Venue_4_New_York_March_2026",
        group_desc="Identify a venue in New York hosting a ticketed entertainment event specifically in March 2026",
        leaf_prefix="Venue_4",
        item=v4,
        require_state_in=["NY", "New York"],
        capacity_condition="any",
        require_month="March",
        event_year=YEAR
    )

    return evaluator.get_summary()