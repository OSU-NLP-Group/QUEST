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
TASK_ID = "ca_feb_2026_sporting_venues"
TASK_DESCRIPTION = (
    "In February 2026, California hosted multiple major national and international sporting events. "
    "Identify the three primary venues in California that hosted these significant sporting events during February 2026, "
    "and for each venue provide: (1) The exact venue name, (2) The specific city and region (Northern California or Southern California) "
    "where it is located, (3) The venue's seating capacity, (4) The specific sporting event(s) hosted at that venue in February 2026, "
    "(5) The exact date(s) of the event(s), (6) For championship-level events, identify the participating teams or key details about the event format. "
    "The venues must meet the following criteria: Located in California, hosted a nationally or internationally significant sporting event "
    "(such as professional league championships, all-star games, or Olympic events), the event occurred during the month of February 2026, and the "
    "venue served as a primary host location (not a practice facility or secondary support venue). Organize your answer by venue, clearly distinguishing "
    "between venues in Northern California and Southern California."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    name: Optional[str] = None
    dates: List[str] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    format_details: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None  # Expected values if present: "Northern California" or "Southern California"
    capacity: Optional[str] = None  # Keep as string for robustness (e.g., "68,500 (expandable to 75,000)")
    primary_host: Optional[bool] = None
    events: List[EventInfo] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract every venue the answer claims qualifies for this task (do not limit to three during extraction; include all venues the answer lists as qualifying).
    For each venue, extract the following fields exactly as stated in the answer:
    - name: The exact venue name (string).
    - city: The specific city (string), e.g., "Santa Clara, California" or "Inglewood, California".
    - region: If the answer explicitly states the California region, extract exactly "Northern California" or "Southern California". If not explicitly stated, return null.
    - capacity: The seating capacity as written (string; do not normalize; e.g., "68,500 (expandable to 75,000)").
    - primary_host: Return true if the answer clearly indicates this venue served as a primary host location for the event(s) (e.g., main game site), false if it clearly indicates otherwise, and null if not specified.
    - source_urls: All URLs the answer cites for this venue or its event(s). Include both general venue sources and event-specific sources.
    - events: An array where each element includes:
        - name: The specific event name (e.g., "Super Bowl LX", "NBA All-Star Game", "Rising Stars", etc.).
        - dates: A list of exact date strings as presented (e.g., "February 8, 2026", "Feb. 15, 2026", or ranges like "February 13–15, 2026").
        - participants: A list of participating teams or key named entities (if the answer lists them). Otherwise, an empty list.
        - format_details: Any key event-format details mentioned (e.g., "three-team tournament: USA Stars, USA Stripes, World"), or null if not provided.
        - source_urls: All URLs the answer associates specifically with this event at this venue.
    IMPORTANT:
    - Only extract what the answer explicitly provides. Do not infer or add information not present in the answer.
    - For URLs, extract the actual URLs (including from markdown links); if the answer cites no URL for a field, leave the corresponding list empty.
    - Maintain the order of venues as presented in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def select_event_by_keywords(venue: Optional[VenueInfo], keywords: List[str]) -> Optional[EventInfo]:
    if not venue:
        return None
    for ev in venue.events:
        ev_name = _norm(ev.name)
        if any(kw.lower() in ev_name for kw in keywords if kw):
            return ev
    return venue.events[0] if venue.events else None


def select_venue(
    venues: List[VenueInfo],
    expected_name: str,
    event_keywords: List[str]
) -> Optional[VenueInfo]:
    # 1) Try exact/approximate name match
    for v in venues:
        if _norm(v.name) == _norm(expected_name):
            return v
    # accept minor punctuation or apostrophe variants
    en = _norm(expected_name).replace("’", "'").replace("'", "")
    for v in venues:
        vn = _norm(v.name).replace("’", "'").replace("'", "")
        if vn == en:
            return v
    # 2) Try event keyword match
    for v in venues:
        ev = select_event_by_keywords(v, event_keywords)
        if ev is not None:
            return v
    return None


def collect_sources(venue: Optional[VenueInfo], event: Optional[EventInfo]) -> List[str]:
    urls: List[str] = []
    if venue and venue.source_urls:
        urls.extend([u for u in venue.source_urls if u])
    if event and event.source_urls:
        urls.extend([u for u in event.source_urls if u])
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def list_to_english(items: List[str]) -> str:
    items = [s for s in items if s]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------- #
# Venue-specific verification builders                                        #
# --------------------------------------------------------------------------- #
async def verify_super_bowl_venue(
    evaluator: Evaluator,
    parent_node,
    extracted: VenuesExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Venue_1_Super_Bowl_LX",
        desc="One qualifying venue corresponding to the Super Bowl LX host, with required details.",
        parent=parent_node,
        critical=False
    )

    expected_name = "Levi's Stadium"
    v = select_venue(extracted.venues, expected_name=expected_name, event_keywords=["Super Bowl", "Superbowl", "SB LX", "SB 60"])
    ev = select_event_by_keywords(v, ["Super Bowl", "Superbowl", "SB LX", "SB 60"])
    sources = collect_sources(v, ev)

    # V1_Venue_Name (simple match against expected)
    v1_name = evaluator.add_leaf(
        id="V1_Venue_Name",
        desc="Venue name is Levi's Stadium.",
        parent=node,
        critical=True
    )
    actual_name = v.name if v and v.name else ""
    await evaluator.verify(
        claim=f"The identified venue name for the Super Bowl LX host matches '{expected_name}'. The provided name is '{actual_name}'.",
        node=v1_name,
        additional_instruction="Judge based on the answer text. Allow minor punctuation differences or apostrophes."
    )

    # V1_City_State (simple match against expected)
    v1_city = evaluator.add_leaf(
        id="V1_City_State",
        desc="City/state given as Santa Clara, California.",
        parent=node,
        critical=True
    )
    actual_city_state = v.city if v and v.city else ""
    await evaluator.verify(
        claim=f"The answer specifies the city/state for {expected_name} as 'Santa Clara, California'. The provided value is '{actual_city_state}'.",
        node=v1_city,
        additional_instruction="Evaluate purely from the answer text; accept 'Santa Clara, CA' or equivalent."
    )

    # V1_Region (simple match against expected)
    v1_region = evaluator.add_leaf(
        id="V1_Region",
        desc="Region identified as Northern California (San Francisco Bay Area acceptable).",
        parent=node,
        critical=True
    )
    actual_region = v.region if v and v.region else ""
    await evaluator.verify(
        claim=f"The answer identifies the region for {expected_name} as 'Northern California' (Bay Area acceptable). The provided value is '{actual_region}'.",
        node=v1_region,
        additional_instruction="Judge from the answer text. Accept 'San Francisco Bay Area' as indicating Northern California."
    )

    # V1_Capacity (verify by URLs using provided capacity text)
    v1_capacity = evaluator.add_leaf(
        id="V1_Capacity",
        desc="Seating capacity is provided and matches the constraint (68,500 base; expandable to 75,000 acceptable).",
        parent=node,
        critical=True
    )
    cap_txt = v.capacity if v and v.capacity else ""
    await evaluator.verify(
        claim=f"Levi's Stadium has the seating capacity described as: {cap_txt}.",
        node=v1_capacity,
        sources=sources,
        additional_instruction=(
            "Verify that the stated capacity aligns with authoritative sources. "
            "Levi's Stadium base capacity is around 68,500; 'expandable to ~75,000' is acceptable. "
            "Minor formatting differences (commas, wording) are okay."
        )
    )

    # V1_Event_Name (verify by URLs)
    v1_event_name = evaluator.add_leaf(
        id="V1_Event_Name",
        desc="Event identified as Super Bowl LX (Super Bowl 60 acceptable).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Levi's Stadium in California hosted Super Bowl LX (also known as Super Bowl 60) in February 2026.",
        node=v1_event_name,
        sources=sources,
        additional_instruction="Treat 'Super Bowl LX' and 'Super Bowl 60' as equivalent."
    )

    # V1_Event_Date (verify by URLs; expected Feb 8, 2026)
    v1_event_date = evaluator.add_leaf(
        id="V1_Event_Date",
        desc="Event date provided as February 8, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Super Bowl LX took place on February 8, 2026.",
        node=v1_event_date,
        sources=sources,
        additional_instruction="Confirm the exact game date on the cited sources."
    )

    # V1_Primary_Host_Location (verify by URLs)
    v1_primary = evaluator.add_leaf(
        id="V1_Primary_Host_Location",
        desc="Makes clear the venue served as a primary host location for the event (not a practice facility or secondary support venue).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Levi's Stadium served as the primary host venue (game site) for Super Bowl LX.",
        node=v1_primary,
        sources=sources,
        additional_instruction="The page should indicate the game was played at Levi's Stadium (i.e., primary host, not auxiliary)."
    )

    # V1_Championship_Participants (simple match vs expected names from the answer)
    v1_participants = evaluator.add_leaf(
        id="V1_Championship_Participants",
        desc="Participating teams are identified (Seattle Seahawks and New England Patriots).",
        parent=node,
        critical=True
    )
    teams_answer_list = ev.participants if ev and ev.participants else []
    teams_answer = list_to_english(teams_answer_list)
    await evaluator.verify(
        claim=(
            "The identified participating teams for Super Bowl LX in the answer match 'Seattle Seahawks' and 'New England Patriots' "
            f"(order-insensitive). The answer lists: {teams_answer}."
        ),
        node=v1_participants,
        additional_instruction=(
            "Judge purely from the answer text. Allow minor naming variants (e.g., 'NE Patriots'). "
            "This is a match check; ignore cited URLs for this specific check."
        )
    )


async def verify_allstar_main_game_venue(
    evaluator: Evaluator,
    parent_node,
    extracted: VenuesExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Venue_2_NBA_AllStar_Main_Game",
        desc="One qualifying venue corresponding to the NBA All-Star main game host, with required details.",
        parent=parent_node,
        critical=False
    )

    expected_name = "Intuit Dome"
    v = select_venue(extracted.venues, expected_name=expected_name, event_keywords=["All-Star Game", "NBA All-Star"])
    ev = select_event_by_keywords(v, ["All-Star Game", "NBA All-Star"])
    sources = collect_sources(v, ev)

    # V2_Venue_Name
    v2_name = evaluator.add_leaf(
        id="V2_Venue_Name",
        desc="Venue name is Intuit Dome.",
        parent=node,
        critical=True
    )
    actual_name = v.name if v and v.name else ""
    await evaluator.verify(
        claim=f"The identified venue name for the NBA All-Star main game matches 'Intuit Dome'. The provided name is '{actual_name}'.",
        node=v2_name,
        additional_instruction="Judge based on the answer text; allow minor casing differences."
    )

    # V2_City_State
    v2_city = evaluator.add_leaf(
        id="V2_City_State",
        desc="City/state given as Inglewood, California.",
        parent=node,
        critical=True
    )
    actual_city_state = v.city if v and v.city else ""
    await evaluator.verify(
        claim=f"The answer specifies the city/state for {expected_name} as 'Inglewood, California'. The provided value is '{actual_city_state}'.",
        node=v2_city,
        additional_instruction="Judge from the answer text; 'Inglewood, CA' acceptable."
    )

    # V2_Region
    v2_region = evaluator.add_leaf(
        id="V2_Region",
        desc="Region identified as Southern California (Los Angeles area acceptable).",
        parent=node,
        critical=True
    )
    actual_region = v.region if v and v.region else ""
    await evaluator.verify(
        claim=f"The answer identifies the region for {expected_name} as 'Southern California' (Los Angeles area acceptable). The provided value is '{actual_region}'.",
        node=v2_region,
        additional_instruction="Judge from the answer text."
    )

    # V2_Capacity (by URLs using provided capacity)
    v2_capacity = evaluator.add_leaf(
        id="V2_Capacity",
        desc="Seating capacity is provided and matches the constraint (18,000).",
        parent=node,
        critical=True
    )
    cap_txt = v.capacity if v and v.capacity else ""
    await evaluator.verify(
        claim=f"Intuit Dome has the seating capacity described as: {cap_txt}.",
        node=v2_capacity,
        sources=sources,
        additional_instruction="Verify capacity on cited sources. Intuit Dome capacity is about 18,000; minor formatting differences acceptable."
    )

    # V2_Event_Name (by URLs)
    v2_event_name = evaluator.add_leaf(
        id="V2_Event_Name",
        desc="Event identified as the NBA All-Star Game (75th edition acceptable).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The NBA All-Star Game (75th edition acceptable phrasing) was hosted at Intuit Dome in February 2026.",
        node=v2_event_name,
        sources=sources,
        additional_instruction="Confirm that the main All-Star Game was held at Intuit Dome."
    )

    # V2_Event_Date (by URLs; expected Feb 15, 2026)
    v2_event_date = evaluator.add_leaf(
        id="V2_Event_Date",
        desc="Main game date provided as February 15, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The NBA All-Star main game took place on February 15, 2026.",
        node=v2_event_date,
        sources=sources,
        additional_instruction="Confirm the exact game date on the cited sources."
    )

    # V2_Primary_Host_Location (by URLs)
    v2_primary = evaluator.add_leaf(
        id="V2_Primary_Host_Location",
        desc="Makes clear the venue served as a primary host location for the event (not a practice facility or secondary support venue).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Intuit Dome served as a primary host venue for the NBA All-Star main game.",
        node=v2_primary,
        sources=sources,
        additional_instruction="The page should indicate the main game venue was Intuit Dome (primary host)."
    )

    # V2_Key_Format_Details (by URLs using answer's details)
    v2_format = evaluator.add_leaf(
        id="V2_Key_Format_Details",
        desc="Provides key event-format details as applicable (e.g., three-team tournament format: USA Stars, USA Stripes, World).",
        parent=node,
        critical=True
    )
    fmt = ev.format_details if ev and ev.format_details else ""
    await evaluator.verify(
        claim=f"The key event-format details for the 2026 NBA All-Star Game include: {fmt}.",
        node=v2_format,
        sources=sources,
        additional_instruction=(
            "Verify that the described format details align with the cited sources. "
            "For example, a three-team tournament (USA Stars, USA Stripes, World) would be acceptable if supported."
        )
    )


async def verify_allstar_weekend_other_venue(
    evaluator: Evaluator,
    parent_node,
    extracted: VenuesExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Venue_3_NBA_AllStar_Weekend_Other_Events",
        desc="One additional qualifying venue in California hosting significant sporting event(s) in February 2026, with required details.",
        parent=parent_node,
        critical=False
    )

    expected_name = "Kia Forum"
    v = select_venue(extracted.venues, expected_name=expected_name, event_keywords=["Rising Stars", "All-Star Saturday", "All-Star Weekend", "Skills", "3-Point", "Slam Dunk"])
    ev = select_event_by_keywords(v, ["Rising Stars", "All-Star Saturday", "All-Star Weekend", "Skills", "3-Point", "Slam Dunk"])
    sources = collect_sources(v, ev)

    # V3_Venue_Name
    v3_name = evaluator.add_leaf(
        id="V3_Venue_Name",
        desc="Venue name is Kia Forum.",
        parent=node,
        critical=True
    )
    actual_name = v.name if v and v.name else ""
    await evaluator.verify(
        claim=f"The identified venue name for NBA All-Star Weekend auxiliary events matches 'Kia Forum'. The provided name is '{actual_name}'.",
        node=v3_name,
        additional_instruction="Judge based on the answer text; allow minor casing differences."
    )

    # V3_City_State
    v3_city = evaluator.add_leaf(
        id="V3_City_State",
        desc="City/state given as Inglewood, California.",
        parent=node,
        critical=True
    )
    actual_city_state = v.city if v and v.city else ""
    await evaluator.verify(
        claim=f"The answer specifies the city/state for {expected_name} as 'Inglewood, California'. The provided value is '{actual_city_state}'.",
        node=v3_city,
        additional_instruction="Judge from the answer text; 'Inglewood, CA' acceptable."
    )

    # V3_Region
    v3_region = evaluator.add_leaf(
        id="V3_Region",
        desc="Region identified as Southern California (Los Angeles area acceptable).",
        parent=node,
        critical=True
    )
    actual_region = v.region if v and v.region else ""
    await evaluator.verify(
        claim=f"The answer identifies the region for {expected_name} as 'Southern California' (Los Angeles area acceptable). The provided value is '{actual_region}'.",
        node=v3_region,
        additional_instruction="Judge from the answer text."
    )

    # V3_Capacity (by URLs)
    v3_capacity = evaluator.add_leaf(
        id="V3_Capacity",
        desc="Seating capacity is provided as a numeric value.",
        parent=node,
        critical=True
    )
    cap_txt = v.capacity if v and v.capacity else ""
    await evaluator.verify(
        claim=f"The Kia Forum has the seating capacity described as: {cap_txt}.",
        node=v3_capacity,
        sources=sources,
        additional_instruction="Confirm the stated capacity using cited sources; minor formatting differences acceptable."
    )

    # V3_Event_Name (by URLs, generic wording)
    v3_event_name = evaluator.add_leaf(
        id="V3_Event_Name",
        desc="Event(s) hosted at this venue are identified as NBA All-Star Weekend events (or equivalent clear description).",
        parent=node,
        critical=True
    )
    ev_names = []
    if v and v.events:
        for e in v.events:
            if e.name:
                ev_names.append(e.name)
    joined_names = list_to_english(ev_names) if ev_names else "NBA All-Star Weekend events"
    await evaluator.verify(
        claim=f"Kia Forum hosted NBA All-Star Weekend events (e.g., {joined_names}) during February 2026.",
        node=v3_event_name,
        sources=sources,
        additional_instruction="Confirm that the cited pages show Kia Forum hosting NBA All-Star Weekend events."
    )

    # V3_Event_Exact_Dates_Provided (existence check from answer extraction)
    v3_dates_provided = evaluator.add_custom_node(
        result=bool(ev and ev.dates and any(d.strip() for d in ev.dates)),
        id="V3_Event_Exact_Dates_Provided",
        desc="Exact date(s) of the event(s) at this venue are explicitly listed.",
        parent=node,
        critical=True
    )

    # V3_Event_Dates_In_Feb_2026 (by URLs)
    v3_dates_in_feb = evaluator.add_leaf(
        id="V3_Event_Dates_In_Feb_2026",
        desc="The listed event date(s) occur in February 2026 (February 13–15, 2026 acceptable per constraints).",
        parent=node,
        critical=True
    )
    dates_txt = ", ".join(ev.dates) if ev and ev.dates else ""
    await evaluator.verify(
        claim=f"The events at Kia Forum occurred on these date(s): {dates_txt}, and these dates are within February 2026.",
        node=v3_dates_in_feb,
        sources=sources,
        additional_instruction="Verify that each listed date falls in February 2026; ranges wholly within Feb 2026 are acceptable."
    )

    # V3_Primary_Host_Location (by URLs)
    v3_primary = evaluator.add_leaf(
        id="V3_Primary_Host_Location",
        desc="Makes clear the venue served as a primary host location for the event(s) (not a practice facility or secondary support venue).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Kia Forum served as a primary host venue for its NBA All-Star Weekend events (e.g., where the events took place).",
        node=v3_primary,
        sources=sources,
        additional_instruction="Confirm that the page indicates events occurred at Kia Forum (primary host), not just practices or media days."
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
    Evaluate an answer for the California February 2026 sporting venues task.
    """
    # Initialize evaluator (root is a non-critical parallel aggregator to allow soft/critical mix under it)
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

    # Extract all venues referenced in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Add ground truth info (for transparency; not used as oracle)
    evaluator.add_ground_truth({
        "expected_core_venues": [
            "Levi's Stadium (Super Bowl LX)",
            "Intuit Dome (NBA All-Star Game)",
            "Kia Forum (NBA All-Star Weekend events)"
        ],
        "expected_key_dates": {
            "Super Bowl LX": "February 8, 2026",
            "NBA All-Star Game": "February 15, 2026",
            "NBA All-Star Weekend (aux events)": "February 13–15, 2026 (typical)"
        }
    }, gt_type="expected_targets")

    # Create a main node to mirror the rubric tree
    main_node = evaluator.add_parallel(
        id="California_February_2026_Sporting_Venues",
        desc="Identify three California venues that hosted nationally/internationally significant sporting events in February 2026 and provide the required venue/event details.",
        parent=root,
        critical=False
    )

    # Global critical checks
    # 1) Response identifies exactly three venues
    exactly_three_leaf = evaluator.add_custom_node(
        result=(len(extracted.venues) == 3),
        id="Response_Provides_Exactly_Three_Venues",
        desc="Response identifies exactly three venues.",
        parent=main_node,
        critical=True
    )

    # 2) Response organized by region (Northern vs Southern California)
    organized_leaf = evaluator.add_leaf(
        id="Response_Organized_By_Region",
        desc="Response clearly distinguishes/group-separates venues in Northern California vs Southern California.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly organizes the venues by region, distinguishing Northern California and Southern California (e.g., via headings or grouped sections).",
        node=organized_leaf,
        additional_instruction="Judge from the answer text only; look for explicit grouping or clear per-venue region labeling that makes the separation obvious."
    )

    # Venue-specific verifications
    await verify_super_bowl_venue(evaluator, main_node, extracted)
    await verify_allstar_main_game_venue(evaluator, main_node, extracted)
    await verify_allstar_weekend_other_venue(evaluator, main_node, extracted)

    # Return structured summary
    return evaluator.get_summary()