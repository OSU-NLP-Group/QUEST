import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dfw_march_2026_events"
TASK_DESCRIPTION = """
Identify four distinct live entertainment events (concerts, comedy shows, or theatrical performances) taking place in the Dallas-Fort Worth metropolitan area during March 2026. For each event, provide the following information:

1. Event Name
2. Performer/Artist
3. Venue Name
4. Venue Address
5. Event Date (specific date in March 2026)
6. Start Time
7. Ticket Purchase Link (direct URL)
8. Parking Information:
   - Opening time for parking lots (relative to event start time)
   - Parking cost or accepted payment methods
   - Specific parking lot names or locations
9. Accessibility: Information about wheelchair-accessible seating or parking
10. Venue Capacity Category: small (<3,000), medium (3,000–10,000), large (10,000–20,000), stadium/arena (>20,000)

Ensure all information is verifiable through official venue websites, ticketing platforms, or entertainment news sources.
"""

DFW_CITIES = [
    "Dallas", "Fort Worth", "Arlington", "Irving", "Plano", "Garland",
    "Grand Prairie", "Richardson", "Frisco", "Denton", "Mesquite",
    "McKinney", "Carrollton", "Lewisville", "Allen", "Flower Mound",
    "Grapevine", "Hurst", "Euless", "Bedford", "North Richland Hills",
    "Mansfield", "Cedar Hill", "Desoto", "Rowlett", "Wylie"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    event_name: Optional[str] = None
    performer: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    event_date: Optional[str] = None
    start_time: Optional[str] = None
    ticket_link: Optional[str] = None

    # Parking details
    parking_opening_time: Optional[str] = None
    parking_cost: Optional[str] = None
    parking_location: Optional[str] = None

    # Accessibility and capacity
    accessibility_info: Optional[str] = None
    capacity_category: Optional[str] = None

    # Sources that the answer claims support this event's details
    source_urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    events: List[EventInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to four distinct live entertainment events (concerts, comedy shows, or theatrical performances) in the Dallas-Fort Worth metropolitan area scheduled during March 2026 from the answer text.

    For each event, return an object with the following fields:
    - event_name: The official event title
    - performer: The performing artist, comedian, or group
    - venue_name: The official venue name
    - venue_address: The full street address of the venue (include city and state)
    - event_date: The specific date (e.g., 'March 12, 2026' or 'Mar 12, 2026')
    - start_time: The scheduled start time (e.g., '7:30 PM')
    - ticket_link: A direct URL to a page where tickets can be purchased for this event
    - parking_opening_time: When parking lots open relative to the event start time (e.g., '2 hours before showtime')
    - parking_cost: Parking cost or accepted payment methods (e.g., '$20 cash/card', 'Credit card only')
    - parking_location: Named lots/garages or specific parking locations (e.g., 'Lot A', 'West Garage')
    - accessibility_info: Information about wheelchair-accessible seating or parking (e.g., 'Accessible seating available', 'ADA parking in Lot B')
    - capacity_category: One of ['small', 'medium', 'large', 'stadium/arena'] as defined: small (<3,000), medium (3,000–10,000), large (10,000–20,000), stadium/arena (>20,000)
    - source_urls: A list of URLs cited in the answer that support this event's details (official venue pages, ticketing platforms like Ticketmaster/AXS/Etix/Live Nation, or credible entertainment news sites). Include all relevant URLs explicitly present in the answer.

    Rules:
    - Extract only information explicitly present in the answer text.
    - If the answer lists more than four events, include only the first four.
    - If any field is missing for an event, set it to null (for strings) or an empty list for source_urls.
    - Ensure ticket_link is the direct page for purchasing tickets (if provided); otherwise set it to null.

    Return a JSON object with a single key 'events' that is an array of event objects as described above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def build_event_urls(event: EventInfo) -> List[str]:
    urls: List[str] = []
    if event.ticket_link and isinstance(event.ticket_link, str) and event.ticket_link.strip():
        urls.append(event.ticket_link.strip())
    for u in event.source_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


# --------------------------------------------------------------------------- #
# Verification function for a single event                                    #
# --------------------------------------------------------------------------- #
async def verify_event(
    evaluator: Evaluator,
    parent_node,
    event: EventInfo,
    event_index: int,
) -> None:
    idx = event_index + 1
    event_node = evaluator.add_parallel(
        id=f"Event_{idx}",
        desc=f"{['First','Second','Third','Fourth'][event_index]} live entertainment event in Dallas-Fort Worth in March 2026",
        parent=parent_node,
        critical=False
    )

    # Precondition: sources exist (critical gating)
    all_urls = build_event_urls(event)
    evaluator.add_custom_node(
        result=bool(all_urls),
        id=f"Event_{idx}_Sources_Exist",
        desc=f"At least one supporting source URL is provided for Event #{idx}",
        parent=event_node,
        critical=True
    )

    # Geographic location verification
    geo_node = evaluator.add_leaf(
        id=f"Event_{idx}_Geographic_Location",
        desc=f"Verify that the event takes place within the Dallas-Fort Worth metropolitan area (including Dallas, Fort Worth, and Arlington)",
        parent=event_node,
        critical=True
    )
    geo_claim = (
        f"The venue '{event.venue_name or ''}' at address '{event.venue_address or ''}' "
        f"is located within the Dallas-Fort Worth metropolitan area."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=all_urls,
        additional_instruction=(
            "Confirm the venue/address belongs to DFW (e.g., Dallas, Fort Worth, Arlington, Irving, Plano, "
            "Garland, Grand Prairie, Richardson, Frisco, Denton, Mesquite, McKinney, etc.). "
            "Use address/city on the official venue or ticket page."
        ),
    )

    # Type validation
    type_node = evaluator.add_leaf(
        id=f"Event_{idx}_Type_Validation",
        desc="Verify that the event is a live entertainment performance requiring tickets (concert, comedy show, or theatrical performance)",
        parent=event_node,
        critical=True
    )
    type_claim = (
        "This event is a live entertainment performance (concert, comedy show, or theatrical performance) "
        "and requires tickets to attend."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=all_urls,
        additional_instruction="Look for show/performance descriptors and an option to purchase tickets on official or ticketing pages."
    )

    # Event name
    name_node = evaluator.add_leaf(
        id=f"Event_{idx}_Event_Name",
        desc="The official name or title of the event",
        parent=event_node,
        critical=True
    )
    name_claim = f"The event's official title is '{event.event_name or ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=all_urls,
        additional_instruction="Verify the event title exactly or with minor formatting variations on the referenced pages."
    )

    # Performer
    perf_node = evaluator.add_leaf(
        id=f"Event_{idx}_Performer",
        desc="The artist, comedian, or performing group for the event",
        parent=event_node,
        critical=True
    )
    perf_claim = f"The performer/artist for this event is '{event.performer or ''}'."
    await evaluator.verify(
        claim=perf_claim,
        node=perf_node,
        sources=all_urls,
        additional_instruction="Confirm the performer name on the event listing or ticketing page. Allow minor formatting variants."
    )

    # Venue name
    venue_node = evaluator.add_leaf(
        id=f"Event_{idx}_Venue_Name",
        desc="The official name of the venue hosting the event",
        parent=event_node,
        critical=True
    )
    venue_claim = f"The venue hosting the event is '{event.venue_name or ''}'."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_node,
        sources=all_urls,
        additional_instruction="Verify the venue name as shown on official venue pages or ticketing pages."
    )

    # Venue address
    addr_node = evaluator.add_leaf(
        id=f"Event_{idx}_Venue_Address",
        desc="The complete street address of the venue",
        parent=event_node,
        critical=True
    )
    addr_claim = f"The venue address is '{event.venue_address or ''}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=all_urls,
        additional_instruction="Match the full street address (including city and state) on official venue or ticketing pages."
    )

    # Event date
    date_node = evaluator.add_leaf(
        id=f"Event_{idx}_Date",
        desc="The specific date in March 2026 when the event occurs",
        parent=event_node,
        critical=True
    )
    date_claim = f"The event date is '{event.event_date or ''}', and it is in March 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=all_urls,
        additional_instruction="Confirm the event date on the official or ticketing page; accept formats like 'Mar 12, 2026' or 'March 12, 2026'."
    )

    # Start time
    time_node = evaluator.add_leaf(
        id=f"Event_{idx}_Start_Time",
        desc="The scheduled start time for the event",
        parent=event_node,
        critical=True
    )
    time_claim = f"The event starts at '{event.start_time or ''}' (local time)."
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=all_urls,
        additional_instruction="Confirm the listed start time; allow minor formatting variants (e.g., '7:30 PM' vs '7:30 p.m.')."
    )

    # Ticket purchase link
    ticket_node = evaluator.add_leaf(
        id=f"Event_{idx}_Ticket_Link",
        desc="A verified URL where tickets can be purchased for the event",
        parent=event_node,
        critical=True
    )
    ticket_claim = (
        f"This URL is a page where tickets can be purchased for the event "
        f"'{event.event_name or ''}' at '{event.venue_name or ''}' on '{event.event_date or ''}'."
    )
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_node,
        sources=event.ticket_link if event.ticket_link else None,
        additional_instruction="Check for a purchase button or seat selection on the page. Accept official ticketing platforms (Ticketmaster, AXS, Etix, Live Nation) or the venue's direct ticketing."
    )

    # Parking details (parallel critical group)
    parking_group = evaluator.add_parallel(
        id=f"Event_{idx}_Parking_Details",
        desc="Parking information for the event's venue",
        parent=event_node,
        critical=True
    )

    # Parking opening time
    p_open_node = evaluator.add_leaf(
        id=f"Event_{idx}_Parking_Opening_Time",
        desc="When parking lots open relative to the event's start time",
        parent=parking_group,
        critical=True
    )
    p_open_claim = f"Parking lots open '{event.parking_opening_time or ''}' relative to the event start time."
    await evaluator.verify(
        claim=p_open_claim,
        node=p_open_node,
        sources=all_urls,
        additional_instruction="Confirm from the venue's parking or event info page. Accept relative descriptors like '2 hours before showtime'."
    )

    # Parking cost
    p_cost_node = evaluator.add_leaf(
        id=f"Event_{idx}_Parking_Cost",
        desc="Parking cost or accepted payment methods for the venue",
        parent=parking_group,
        critical=True
    )
    p_cost_claim = f"Parking cost or payment methods: '{event.parking_cost or ''}'."
    await evaluator.verify(
        claim=p_cost_claim,
        node=p_cost_node,
        sources=all_urls,
        additional_instruction="Look for pricing or accepted payment methods (cash/card) on the venue's parking page."
    )

    # Parking location
    p_loc_node = evaluator.add_leaf(
        id=f"Event_{idx}_Parking_Location",
        desc="Specific parking lot names or locations for the venue",
        parent=parking_group,
        critical=True
    )
    p_loc_claim = f"Parking locations/lots include: '{event.parking_location or ''}'."
    await evaluator.verify(
        claim=p_loc_claim,
        node=p_loc_node,
        sources=all_urls,
        additional_instruction="Confirm named lots/garages (e.g., 'Lot A', 'West Garage') from venue maps or parking info pages."
    )

    # Accessibility
    access_node = evaluator.add_leaf(
        id=f"Event_{idx}_Accessibility",
        desc="Wheelchair-accessible seating or parking availability for the venue",
        parent=event_node,
        critical=True
    )
    access_claim = f"Wheelchair-accessible seating or parking is available as described: '{event.accessibility_info or ''}'."
    await evaluator.verify(
        claim=access_claim,
        node=access_node,
        sources=all_urls,
        additional_instruction="Check the venue's accessibility/ADA page or event info; confirm accessible seating or ADA parking availability."
    )

    # Venue capacity category (non-critical)
    capacity_node = evaluator.add_leaf(
        id=f"Event_{idx}_Venue_Capacity_Category",
        desc="The venue size category (small <3,000; medium 3,000–10,000; large 10,000–20,000; stadium/arena >20,000)",
        parent=event_node,
        critical=False
    )
    cap_claim = (
        f"The venue capacity category is '{event.capacity_category or ''}', "
        f"consistent with its typical capacity."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_node,
        sources=all_urls,
        additional_instruction="Use venue specs or reliable sources to judge capacity category; accept approximate capacity that clearly falls in the stated range."
    )

    # Source verification leaf
    src_ver_node = evaluator.add_leaf(
        id=f"Event_{idx}_Source_Verification",
        desc="Information is traceable to official venue websites, ticketing platforms, or verified entertainment news sources",
        parent=event_node,
        critical=True
    )
    src_ver_claim = (
        "The provided sources include official venue websites, ticketing platforms (e.g., Ticketmaster, AXS, Etix, Live Nation), "
        "or verified entertainment news sites, and they support the event details (title, performer, venue, date, and time)."
    )
    await evaluator.verify(
        claim=src_ver_claim,
        node=src_ver_node,
        sources=all_urls,
        additional_instruction="Assess domains and content to ensure sources are official or recognized platforms/news, and that event details are supported."
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
    Evaluate an answer for the Dallas-Fort Worth March 2026 events task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Events are independent
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

    # NOTE: Keep root non-critical to allow partial credit across events, and avoid
    # the framework constraint that critical parents must have all children critical.
    root.critical = False

    # Extract up to 4 events
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Normalize to exactly 4 events (pad with empty entries if fewer)
    events: List[EventInfo] = list(extracted.events[:4])
    while len(events) < 4:
        events.append(EventInfo())

    # Build subtrees and perform verifications for each event
    for i, ev in enumerate(events):
        await verify_event(evaluator, root, ev, i)

    # Return structured summary
    return evaluator.get_summary()