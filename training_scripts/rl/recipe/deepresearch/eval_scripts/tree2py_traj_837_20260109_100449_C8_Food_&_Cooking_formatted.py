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
TASK_ID = "farm_dinner_events"
TASK_DESCRIPTION = """
Identify 4 farm dinner events that meet ALL of the following criteria:

Location: The event must be located in California, Colorado, or New York State.

Multi-Course Menu: The event must offer a multi-course meal with at least 5 courses.

Beverage Pairing: The event must include wine or beverage pairing as part of the experience.

Farm Sourcing: The event description must explicitly mention local farm sourcing, farm-to-table ingredients, or being held at a farm location.

For each event, provide the event name, location, and a reference URL that verifies these requirements.
"""

ELIGIBLE_STATES = {
    "California": ["california", r"\bca\b"],
    "Colorado": ["colorado", r"\bco\b"],
    "New York": ["new york", r"\bny\b"],
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    events: List[EventInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract the farm dinner events mentioned in the answer. We need up to four events, but you should extract all events listed in the answer (up to 8) so we can filter later.

    For each event, extract:
    1. name: The event name as written in the answer (string). If there's no clear event name, set to null.
    2. location: The event location as written in the answer (string). This may be a city + state (e.g., "Sonoma, CA" or "Brooklyn, NY") or a venue/farm + city/state. If missing, set to null.
    3. reference_urls: A list of URL(s) that the answer cites for this specific event and that can be used to verify details (e.g., the event or organizer's page, ticketing page, official farm page, etc.). Extract only explicit URLs present in the answer (including markdown links). If none are provided, return an empty list.

    Return a JSON object:
    {
      "events": [
        {"name": ..., "location": ..., "reference_urls": [...]},
        ...
      ]
    }

    Rules:
    - Do not invent data. Only extract what is present in the answer.
    - Normalize URLs to include scheme (http/https). Ignore malformed URLs.
    - If multiple URLs are cited for an event, include all of them in reference_urls.
    - If the answer provides more than four events, include all; we will take the first four later.
    - If fewer than four events are present, just extract what's available.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def sanitize_urls(urls: List[str]) -> List[str]:
    cleaned = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            # Prepend http:// if missing scheme
            u = "http://" + u
        cleaned.append(u)
    return cleaned


def infer_eligible_state_from_location(location: Optional[str]) -> Optional[str]:
    if not location:
        return None
    loc = location.lower()
    # Direct string checks
    for state, patterns in ELIGIBLE_STATES.items():
        for p in patterns:
            # If pattern looks like a regex for abbreviations, use regex; otherwise substring match
            if p.startswith(r"\b"):
                if re.search(p, loc):
                    return state
            else:
                if p in loc:
                    return state
    return None


def pad_to_four(events: List[EventInfo]) -> List[EventInfo]:
    """Ensure exactly 4 items by padding with empty placeholders."""
    padded = list(events[:4])
    while len(padded) < 4:
        padded.append(EventInfo())
    return padded


# --------------------------------------------------------------------------- #
# Verification function for one event                                         #
# --------------------------------------------------------------------------- #
async def verify_event(
    evaluator: Evaluator,
    parent_node,
    event: EventInfo,
    index: int,
) -> None:
    """
    Build the verification subtree for a single event, with critical checks for all required criteria.
    """
    event_idx = index + 1
    event_node = evaluator.add_parallel(
        id=f"Event_{event_idx}",
        desc=f"{['First','Second','Third','Fourth'][index]} farm dinner event meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # Normalize URLs for safety
    urls = sanitize_urls(event.reference_urls)

    # 1) Event name provided (critical)
    evaluator.add_custom_node(
        result=bool(event.name and event.name.strip()),
        id=f"event_{index}_Event_Name_Provided",
        desc="Provides the event name",
        parent=event_node,
        critical=True
    )

    # 2) Reference URL provided (critical)
    evaluator.add_custom_node(
        result=bool(urls),
        id=f"event_{index}_Reference_URL_Provided",
        desc="Provides a reference URL for the event (intended to verify the listed requirements)",
        parent=event_node,
        critical=True
    )

    # 3) Location provided and eligible (split into two critical leaves under a critical aggregator)
    location_gate = evaluator.add_parallel(
        id=f"event_{index}_Location_Provided_And_Eligible",
        desc="Provides the event location and the location is in California, Colorado, or New York State",
        parent=event_node,
        critical=True
    )

    # 3.a) Location provided (critical)
    evaluator.add_custom_node(
        result=bool(event.location and event.location.strip()),
        id=f"event_{index}_Location_Provided",
        desc="Location is provided for the event",
        parent=location_gate,
        critical=True
    )

    # 3.b) Location eligible and supported by source (critical)
    loc_leaf = evaluator.add_leaf(
        id=f"event_{index}_Location_Eligible_Verified",
        desc="Event location is within California, Colorado, or New York State and supported by the reference URL(s)",
        parent=location_gate,
        critical=True
    )
    inferred_state = infer_eligible_state_from_location(event.location)
    # Build a robust claim. If we inferred a state from the provided location, use it;
    # otherwise, use a general eligible-states claim.
    if inferred_state:
        loc_claim = (
            f"The event page indicates the event takes place in {inferred_state}."
            f" The provided location is '{event.location}'."
        )
    else:
        loc_claim = (
            f"The event page indicates the event takes place in one of these states: California (CA), Colorado (CO), or New York (NY)."
            f" The provided location text is '{event.location}'."
        )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm the event is in California, Colorado, or New York State. "
            "Accept city-level mentions paired with standard state names or abbreviations (CA, CO, NY). "
            "Examples of acceptable mentions include 'Napa, CA', 'Brooklyn, NY', 'Denver, CO'. "
            "If the page shows a different state or lacks location info, mark as not supported."
        )
    )

    # 4) Multi-course menu with at least 5 courses (critical)
    multi_leaf = evaluator.add_leaf(
        id=f"event_{index}_Multi_Course_Menu_5Plus",
        desc="Event offers a multi-course meal with at least 5 courses",
        parent=event_node,
        critical=True
    )
    multi_claim = (
        "The event offers a multi-course meal with at least five courses (5+). "
        "Accept explicit phrases like 'five-course', '6-course', 'seven-course', 'tasting menu with 5 courses'."
    )
    await evaluator.verify(
        claim=multi_claim,
        node=multi_leaf,
        sources=urls,
        additional_instruction=(
            "Search the event page(s) for an explicit course count indicating at least five courses. "
            "Phrases like 'five-course dinner', 'six-course tasting', or similar are acceptable. "
            "Generic 'multi-course' without an explicit count of 5 or more should NOT pass."
        )
    )

    # 5) Beverage pairing included (critical)
    pairing_leaf = evaluator.add_leaf(
        id=f"event_{index}_Beverage_Pairing_Included",
        desc="Event includes wine or beverage pairing as part of the experience",
        parent=event_node,
        critical=True
    )
    pairing_claim = (
        "The event includes wine or beverage pairing as part of the experience. "
        "It can be included in the ticket or offered as an on-site pairing option clearly tied to the menu."
    )
    await evaluator.verify(
        claim=pairing_claim,
        node=pairing_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit mentions of 'wine pairing', 'drink pairing', 'beverage pairing', or similar. "
            "If the page only mentions general availability of drinks (e.g., cash bar) without pairing tied to courses, do not pass."
        )
    )

    # 6) Farm sourcing or farm location (critical)
    farm_leaf = evaluator.add_leaf(
        id=f"event_{index}_Farm_Sourcing_Or_Farm_Location",
        desc="Event description explicitly mentions local farm sourcing, farm-to-table ingredients, OR that it is held at a farm location",
        parent=event_node,
        critical=True
    )
    farm_claim = (
        "The event description explicitly references local farm sourcing, farm-to-table ingredients, or states that the dinner is held at a farm location."
    )
    await evaluator.verify(
        claim=farm_claim,
        node=farm_leaf,
        sources=urls,
        additional_instruction=(
            "Accept clear phrases such as 'held at the farm', 'on-farm dinner', 'farm-to-table menu', "
            "'ingredients sourced from local farms', 'grown onsite', or similar. "
            "If such references are absent, mark as not supported."
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
    Evaluate an answer for the farm dinner events task and return a structured result dictionary.
    """
    # Initialize evaluator with parallel aggregation for root
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

    # Extract events from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="farm_dinner_events_extraction",
    )

    # Prepare exactly 4 events (filter/pad as needed)
    events = pad_to_four(extracted.events)

    # Build verification tree for the four events
    events_root = evaluator.add_parallel(
        id="Farm_Dinner_Events",
        desc="Identify 4 farm dinner events meeting all specified criteria and provide required fields for each.",
        parent=root,
        critical=False
    )

    tasks = []
    for i in range(4):
        tasks.append(verify_event(evaluator, events_root, events[i], i))
    # Execute verifications sequentially to respect prerequisite gating
    for t in tasks:
        await t

    # Return evaluation summary
    return evaluator.get_summary()