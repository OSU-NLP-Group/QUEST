import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_career_fairs_2026_jan_feb"
TASK_DESCRIPTION = (
    "I am a college student preparing for my spring 2026 semester career search. "
    "I want to identify five in-person career fairs at universities in the United States that are scheduled for January or February 2026. "
    "For each career fair, please provide: (1) The university hosting the fair, (2) The specific date(s) of the fair, "
    "(3) The venue/location where the fair will be held, (4) The time range (start and end times), "
    "(5) The registration method or URL where students can register or find more information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CareerFair(BaseModel):
    """Structured data for one career fair entry extracted from the agent's answer."""
    host_university: Optional[str] = None
    event_name: Optional[str] = None
    dates: Optional[str] = None  # e.g., "Jan 29, 2026", "February 12–13, 2026", "Jan 31–Feb 1, 2026"
    venue: Optional[str] = None  # Physical building/room/hall name
    time_range: Optional[str] = None  # e.g., "10:00 AM – 3:00 PM"
    in_person: Optional[bool] = None  # True if explicitly stated as in-person in the answer; else null
    open_to_students: Optional[bool] = None  # True if explicitly stated open to students; else null
    registration_url: Optional[str] = None  # Primary registration/RSVP link if provided
    support_urls: List[str] = Field(default_factory=list)  # Any URLs mentioned for this fair (event page, Handshake, etc.)


class CareerFairsExtraction(BaseModel):
    """Model encapsulating all career fairs extracted from the answer."""
    fairs: List[CareerFair] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_fairs() -> str:
    return """
    Extract up to FIVE distinct career fair entries described in the answer, preserving their original order of appearance.
    For EACH fair, extract the following fields exactly as written in the answer text (do not normalize):
    - host_university: The hosting university's name (e.g., "University of X", "X College").
    - event_name: The official career fair title/name (if provided).
    - dates: The specific date(s) (e.g., "Jan 29, 2026", "February 12–13, 2026", "Jan 31–Feb 1, 2026").
    - venue: The physical venue/location name (building, hall, center, room, etc.). If missing, return null.
    - time_range: The start and end times as a range (e.g., "10:00 AM – 3:00 PM"). If missing, return null.
    - in_person: true/false if the answer explicitly states the fair is "in-person". If not explicit, return null.
    - open_to_students: true/false if the answer explicitly states students can attend (e.g., "open to students", "for undergraduates/graduate students"). If not explicit, return null.
    - registration_url: The main registration or RSVP URL if provided in the answer. If not provided, return null.
    - support_urls: ALL URLs mentioned for this particular fair (including Handshake/event pages, details pages, and the registration_url if applicable). Return an array. If no URLs are provided, return an empty array.

    IMPORTANT:
    - Only extract information explicitly present in the answer.
    - If the answer contains more than five fairs, extract ONLY the first five.
    - Treat URLs in plain text or markdown links. Extract the actual URL string.
    - Do NOT invent any data. Use null for any missing fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text_for_key(s: Optional[str]) -> str:
    if not s:
        return ""
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _fair_identity_key(f: CareerFair) -> str:
    """
    Build a normalized identity key to detect duplicates.
    We use a combination of host_university + dates + venue + registration_url for robustness.
    """
    parts = [
        _normalize_text_for_key(f.host_university),
        _normalize_text_for_key(f.dates),
        _normalize_text_for_key(f.venue),
        _normalize_text_for_key(f.registration_url),
    ]
    # If everything is empty, still return an empty string
    return "|".join(parts).strip("|")


def _fair_is_nonempty(f: CareerFair) -> bool:
    """
    Minimal non-empty criterion for distinct fairs: must have a host university and at least one of
    {dates, venue, time_range, registration_url, support_urls}.
    """
    return bool(
        (f.host_university and f.host_university.strip())
        and (
            (f.dates and f.dates.strip())
            or (f.venue and f.venue.strip())
            or (f.time_range and f.time_range.strip())
            or (f.registration_url and f.registration_url.strip())
            or (f.support_urls and len(f.support_urls) > 0)
        )
    )


def _collect_sources(f: CareerFair) -> List[str]:
    """Collect and deduplicate all URLs we can use as sources for verification."""
    urls: List[str] = []
    if f.registration_url and isinstance(f.registration_url, str) and f.registration_url.strip():
        urls.append(f.registration_url.strip())
    for u in f.support_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification for a single fair                                              #
# --------------------------------------------------------------------------- #
async def verify_fair(
    evaluator: Evaluator,
    parent_node,
    fair: CareerFair,
    fair_index: int,
) -> None:
    """
    Build verification sub-tree for a single fair and run checks.
    All child checks are critical under the fair node (which itself is non-critical to allow partial credit across fairs).
    """
    fair_num = fair_index + 1
    fair_node = evaluator.add_parallel(
        id=f"fair_{fair_num}",
        desc=f"Career fair #{fair_num} meets all constraints and includes all required details",
        parent=parent_node,
        critical=False,
    )

    sources = _collect_sources(fair)
    uni = fair.host_university or ""
    dates = fair.dates or ""
    venue = fair.venue or ""
    time_range = fair.time_range or ""

    # 1) Hosting university is a US university
    host_node = evaluator.add_leaf(
        id=f"fair_{fair_num}_host_university_us",
        desc="Identifies the hosting university and it is a university located in the United States",
        parent=fair_node,
        critical=True,
    )
    host_claim = (
        f"The event is hosted by {uni}, and {uni} is a university located in the United States."
        if uni else
        "The event page confirms a US university is the hosting institution."
    )
    await evaluator.verify(
        claim=host_claim,
        node=host_node,
        sources=sources,
        additional_instruction=(
            "Verify that the page indicates the event is organized by the stated institution and that the institution is a U.S. university. "
            "Strong signals include an official .edu domain or explicit location within the United States. "
            "Minor naming variations are fine (e.g., 'University of X' vs 'X University')."
        ),
    )

    # 2) Date(s) in January or February 2026
    date_node = evaluator.add_leaf(
        id=f"fair_{fair_num}_date_jan_feb_2026",
        desc="Provides specific date(s) for the fair, and the date(s) fall in January or February 2026",
        parent=fair_node,
        critical=True,
    )
    date_claim = (
        f"The fair is scheduled on {dates}, and the date(s) are in January or February 2026."
        if dates else
        "The event page confirms the fair occurs in January or February 2026."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources,
        additional_instruction=(
            "Check the event date on the provided page(s). Accept 'Jan'/'January' and 'Feb'/'February' formats. "
            "For multi-day ranges, confirm that the dates fall within Jan/Feb 2026 (events spanning Jan 31–Feb 1 are acceptable)."
        ),
    )

    # 3) Explicitly in-person
    in_person_node = evaluator.add_leaf(
        id=f"fair_{fair_num}_in_person",
        desc="Fair is explicitly in-person (not virtual/online-only)",
        parent=fair_node,
        critical=True,
    )
    in_person_claim = "This career fair is an in-person event (not virtual/online-only)."
    await evaluator.verify(
        claim=in_person_claim,
        node=in_person_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit or strongly implied physical attendance (e.g., 'in-person', 'on campus', a building/room, address). "
            "If the page states 'virtual only', the claim is not supported. Hybrid with an in-person component counts as in-person."
        ),
    )

    # 4) Open to students
    open_students_node = evaluator.add_leaf(
        id=f"fair_{fair_num}_open_to_students",
        desc="Fair is open to students (not restricted to employers only)",
        parent=fair_node,
        critical=True,
    )
    open_claim = "Students are invited and allowed to attend this career fair."
    await evaluator.verify(
        claim=open_claim,
        node=open_students_node,
        sources=sources,
        additional_instruction=(
            "Confirm that students (undergraduate or graduate) are invited or eligible to attend. "
            "Phrases such as 'for students', 'open to all majors', 'student career fair' support the claim."
        ),
    )

    # 5) Venue/location provided
    venue_node = evaluator.add_leaf(
        id=f"fair_{fair_num}_venue_location",
        desc="Provides a specific physical venue/location name for where the fair will be held",
        parent=fair_node,
        critical=True,
    )
    venue_claim = (
        f"The fair will be held at '{venue}'."
        if venue else
        "The event page provides a specific physical venue/location name for the fair."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_node,
        sources=sources,
        additional_instruction=(
            "Check for a named physical location (building/hall/center/room). "
            "Generic phrases without a venue (e.g., 'on campus' only) are insufficient; the page should identify a specific place."
        ),
    )

    # 6) Time range (start and end times)
    time_node = evaluator.add_leaf(
        id=f"fair_{fair_num}_time_range",
        desc="Provides start and end time(s) for the fair (time range documented)",
        parent=fair_node,
        critical=True,
    )
    time_claim = (
        f"The fair runs during '{time_range}' (a clear start–end time range)."
        if time_range else
        "The event page documents a start and end time range for the fair."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=sources,
        additional_instruction=(
            "Verify that a time window is provided (e.g., '10:00 AM–3:00 PM'). "
            "If multi-day, times may differ per day, but each day should include a start–end time range."
        ),
    )

    # 7) Registration/RSVP method or URL
    registration_node = evaluator.add_leaf(
        id=f"fair_{fair_num}_registration_method",
        desc="Provides a documented registration/RSVP method or URL for students to register or find more information",
        parent=fair_node,
        critical=True,
    )
    if fair.registration_url and fair.registration_url.strip():
        reg_claim = (
            f"A documented registration/RSVP method is provided and students can register via the page {fair.registration_url.strip()}."
        )
    else:
        reg_claim = (
            "A documented registration/RSVP method is provided on the event page(s) for students to register or find more information."
        )

    await evaluator.verify(
        claim=reg_claim,
        node=registration_node,
        sources=sources,
        additional_instruction=(
            "Look for an explicit registration mechanism: a button/link labeled 'Register', 'RSVP', a Handshake event page, or a form. "
            "If a specific URL is provided in the answer, verify that it leads to a page that enables registration or provides clear registration instructions."
        ),
    )


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
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for identifying five in-person US university career fairs in Jan/Feb 2026.
    Returns the evaluation summary dictionary generated by the evaluator.
    """
    # Initialize evaluator (root set to PARALLEL; set critical=False to allow partial scoring across fairs)
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

    # Extract fairs from the answer
    extracted: CareerFairsExtraction = await evaluator.extract(
        prompt=prompt_extract_fairs(),
        template_class=CareerFairsExtraction,
        extraction_name="career_fairs_extraction",
    )

    # Keep only the first 5 fairs to align with task requirement; pad with empty entries if fewer
    fairs: List[CareerFair] = list(extracted.fairs[:5])
    while len(fairs) < 5:
        fairs.append(CareerFair())

    # Distinctness check: exactly five distinct, non-empty fairs (based on identity key)
    keys: List[str] = [_fair_identity_key(f) for f in fairs if _fair_is_nonempty(f)]
    unique_keys = set(keys)
    distinct_nonempty_count = len(unique_keys)

    has_exactly_five_distinct = (distinct_nonempty_count == 5)

    # Record distinctness statistics for transparency
    evaluator.add_custom_info(
        info={
            "total_extracted_first5": len([f for f in fairs if _fair_is_nonempty(f)]),
            "distinct_nonempty_count_first5": distinct_nonempty_count,
            "identity_keys_first5": keys,
        },
        info_type="distinctness_stats",
        info_name="five_distinct_fairs_check",
    )

    # Add distinctness check node (critical under root)
    evaluator.add_custom_node(
        result=has_exactly_five_distinct,
        id="five_distinct_fairs_present",
        desc="Response includes exactly five distinct career fairs (no duplicates), each clearly separable as its own event entry",
        parent=root,
        critical=True,
    )

    # Build verification subtrees for each fair (non-critical under root for partial credit)
    for i in range(5):
        await verify_fair(evaluator, root, fairs[i], i)

    # Return final summary
    return evaluator.get_summary()