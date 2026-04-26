import asyncio
import logging
from typing import Any, List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "q1_2026_us_gaming_conventions"
TASK_DESCRIPTION = (
    "Identify five gaming conventions or expo events that take place in the United States during the first quarter "
    "of 2026 (January 1 through March 31, 2026). For each convention, provide the following information: "
    "(1) The official convention name, (2) The complete event dates (including start date and end date), "
    "(3) The venue name, (4) The city and state where it takes place, and (5) A URL link to the official convention "
    "website or registration/ticketing page."
)

Q1_2026_RANGE_TEXT = "January 1, 2026 to March 31, 2026 (inclusive)"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConventionItem(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None   # Keep as free-form string to maximize compatibility
    end_date: Optional[str] = None
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    official_url: Optional[str] = None


class ConventionsExtraction(BaseModel):
    conventions: List[ConventionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conventions() -> str:
    return """
    Extract all gaming conventions or gaming-focused expo events explicitly mentioned in the answer.
    For each event, extract the following fields exactly as written in the answer (do not infer anything not present):
    - name: the official convention name
    - start_date: the event start date (any reasonable format if present)
    - end_date: the event end date (any reasonable format if present)
    - venue: the venue name (if present)
    - city: the city where the event takes place (if present)
    - state: the U.S. state where the event takes place (if present)
    - official_url: a URL to the official website or registration/ticketing page for that specific event
    Special notes:
    - Return every event mentioned in the answer, even if some fields are missing.
    - If a field is not present in the answer, set it to null.
    - For URLs, extract only actual URLs that are present in the answer text (plain links or markdown links).
    - Do not fabricate or infer URLs.
    - Preserve the order in which the events appear in the answer.
    Return a JSON object with a single top-level field "conventions" that is a list of event objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _ensure_five_slots(conventions: List[ConventionItem]) -> List[ConventionItem]:
    selected = conventions[:5]
    while len(selected) < 5:
        selected.append(ConventionItem())
    return selected


def _compute_distinct_first_five(conventions: List[ConventionItem]) -> bool:
    selected = conventions[:5]
    if len(selected) < 5:
        return False
    keys = set()
    for ev in selected:
        key = (_normalize_str(ev.name), _normalize_str(ev.official_url))
        keys.add(key)
    return len(keys) == 5


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    # Initialize evaluator and root
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_conventions(),
        template_class=ConventionsExtraction,
        extraction_name="conventions_extraction"
    )

    conventions_all: List[ConventionItem] = extracted.conventions or []
    selected_five: List[ConventionItem] = _ensure_five_slots(conventions_all)

    # Record custom info for debugging/traceability
    evaluator.add_custom_info(
        {
            "total_extracted": len(conventions_all),
            "selected_five_preview": [
                {
                    "name": ev.name,
                    "start_date": ev.start_date,
                    "end_date": ev.end_date,
                    "venue": ev.venue,
                    "city": ev.city,
                    "state": ev.state,
                    "official_url": ev.official_url
                }
                for ev in selected_five
            ]
        },
        info_type="extraction_summary",
        info_name="extraction_overview"
    )

    # ------------------------------ Critical Checks ------------------------------ #
    # 1) Exactly five distinct items identified (using first five if more provided)
    five_distinct = (len(conventions_all) >= 5) and _compute_distinct_first_five(conventions_all)
    evaluator.add_custom_node(
        result=five_distinct,
        id="Five_Conventions_Identified",
        desc="Exactly five distinct gaming conventions or gaming-focused expo events are identified (using the first five if more are provided).",
        parent=root,
        critical=True
    )

    # 2) Q1 2026 timeframe (Critical parent; each event leaf must pass)
    timeframe_parent = evaluator.add_parallel(
        id="Q1_2026_Timeframe",
        desc="All five conventions take place within the first quarter of 2026 (January 1 through March 31, 2026)",
        parent=root,
        critical=True
    )

    # 3) US location (Critical parent; each event leaf must pass)
    usloc_parent = evaluator.add_parallel(
        id="US_Locations",
        desc="All five conventions take place in the United States",
        parent=root,
        critical=True
    )

    # ---------------------------- Non-Critical Presence --------------------------- #
    # A) Official URLs provided
    urls_parent = evaluator.add_parallel(
        id="Official_URLs_Provided",
        desc="A URL link to an official convention website or registration/ticketing page is provided for each of the five conventions",
        parent=root,
        critical=False
    )

    # B) Official names provided
    names_parent = evaluator.add_parallel(
        id="Official_Names_Provided",
        desc="Official convention name is provided for each of the five conventions",
        parent=root,
        critical=False
    )

    # C) Complete dates provided
    dates_parent = evaluator.add_parallel(
        id="Complete_Dates_Provided",
        desc="Complete event dates (including both start date and end date) are provided for each of the five conventions",
        parent=root,
        critical=False
    )

    # D) Venue and location details
    vls_parent = evaluator.add_parallel(
        id="Venue_And_Location_Details",
        desc="Venue name, city, and state are provided for each of the five conventions",
        parent=root,
        critical=False
    )

    # -------------------------- Build per-event leaves/nodes ---------------------- #
    # Create URL presence nodes first so we can use them as prerequisites (gating)
    url_presence_nodes: List[VerificationNode] = []
    for idx, ev in enumerate(selected_five, start=1):
        url_ok = bool(ev.official_url and ev.official_url.strip())
        node = evaluator.add_custom_node(
            result=url_ok,
            id=f"event_{idx}_url_provided",
            desc=f"Event #{idx}: Official URL is provided",
            parent=urls_parent,
            critical=False
        )
        url_presence_nodes.append(node)

    # Names presence
    for idx, ev in enumerate(selected_five, start=1):
        name_ok = bool(ev.name and ev.name.strip())
        evaluator.add_custom_node(
            result=name_ok,
            id=f"event_{idx}_name_provided",
            desc=f"Event #{idx}: Official convention name provided",
            parent=names_parent,
            critical=False
        )

    # Complete dates presence (both start and end)
    for idx, ev in enumerate(selected_five, start=1):
        dates_ok = bool(ev.start_date and ev.end_date and ev.start_date.strip() and ev.end_date.strip())
        evaluator.add_custom_node(
            result=dates_ok,
            id=f"event_{idx}_complete_dates_provided",
            desc=f"Event #{idx}: Both start date and end date are provided",
            parent=dates_parent,
            critical=False
        )

    # Venue + City + State presence
    for idx, ev in enumerate(selected_five, start=1):
        vls_ok = bool(ev.venue and ev.venue.strip() and ev.city and ev.city.strip() and ev.state and ev.state.strip())
        evaluator.add_custom_node(
            result=vls_ok,
            id=f"event_{idx}_venue_city_state_provided",
            desc=f"Event #{idx}: Venue, city, and state are all provided",
            parent=vls_parent,
            critical=False
        )

    # ----------------------------- Source-grounded checks ------------------------- #
    # For timeframe and US-location, verify against official URLs when available.
    # If a URL is missing for an event, the corresponding verification leaf will be skipped via extra preconditions.

    # Timeframe verifications (Critical leaves under critical parent)
    for idx, ev in enumerate(selected_five, start=1):
        tf_leaf = evaluator.add_leaf(
            id=f"event_{idx}_timeframe_q1_2026",
            desc=f"Event #{idx}: Occurs within Q1 2026",
            parent=timeframe_parent,
            critical=True
        )
        # Build claim
        ev_name = ev.name or "this event"
        timeframe_claim = (
            f"The event '{ev_name}' takes place entirely within the period January 1, 2026 to March 31, 2026 (inclusive)."
        )

        add_ins = (
            "Verify that the 2026 edition of the event occurs within Q1 2026. "
            "If multiple years are shown, focus on 2026. "
            "Accept if both the start and the end date fall on or between January 1, 2026 and March 31, 2026. "
            "If the webpage does not clearly show 2026 dates or shows dates outside this range, mark as not supported."
        )

        await evaluator.verify(
            claim=timeframe_claim,
            node=tf_leaf,
            sources=ev.official_url if ev.official_url else None,
            additional_instruction=add_ins,
            extra_prerequisites=[url_presence_nodes[idx - 1]]
        )

    # US location verifications (Critical leaves under critical parent)
    for idx, ev in enumerate(selected_five, start=1):
        loc_leaf = evaluator.add_leaf(
            id=f"event_{idx}_us_location",
            desc=f"Event #{idx}: Takes place in the United States",
            parent=usloc_parent,
            critical=True
        )
        ev_name = ev.name or "this event"
        us_claim = f"The event '{ev_name}' takes place in the United States (U.S.)."
        add_ins_loc = (
            "Check the venue/location information on the page. "
            "City and state pairs (e.g., 'Boston, MA' or 'Seattle, Washington') indicate the event is in the U.S.; "
            "the page might also explicitly mention 'USA' or 'United States'. "
            "If the location is outside the U.S., or unclear, mark as not supported."
        )

        await evaluator.verify(
            claim=us_claim,
            node=loc_leaf,
            sources=ev.official_url if ev.official_url else None,
            additional_instruction=add_ins_loc,
            extra_prerequisites=[url_presence_nodes[idx - 1]]
        )

    # Return the evaluation summary
    return evaluator.get_summary()