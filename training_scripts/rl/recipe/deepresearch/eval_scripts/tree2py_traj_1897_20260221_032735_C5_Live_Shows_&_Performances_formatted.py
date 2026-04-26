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
TASK_ID = "keybank_state_theatre_2026_performer"
TASK_DESCRIPTION = (
    "Who is the Cleveland-bred stand-up comedian scheduled to perform at KeyBank State Theatre (part of Playhouse Square) "
    "in Cleveland, Ohio on Friday, February 13, 2026 at 7:00pm? The performer must meet the following criteria: "
    "The venue (KeyBank State Theatre) has a seating capacity of 3,200, making it a large theater venue (capacity exceeds the 1,000-seat threshold for large theater classification); "
    "The performer has a television career that includes hosting TV shows; The performer starred in 'The Drew Carey Show'; "
    "The performer has been hosting 'The Price Is Right' since 2007. Provide the performer's name."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ScheduleInfo(BaseModel):
    date_text: Optional[str] = None
    time_text: Optional[str] = None
    venue_name: Optional[str] = None
    schedule_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    location_text: Optional[str] = None
    part_of_playhouse_square: Optional[str] = None
    capacity_text: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)


class BackgroundInfo(BaseModel):
    origin_text: Optional[str] = None
    stand_up_comedian_text: Optional[str] = None
    hosted_tv_shows: List[str] = Field(default_factory=list)
    starred_in_drew_carey_show_text: Optional[str] = None
    price_is_right_host_since_2007_text: Optional[str] = None
    background_urls: List[str] = Field(default_factory=list)


class PerformerExtraction(BaseModel):
    performer_name: Optional[str] = None
    schedule: Optional[ScheduleInfo] = None
    venue: Optional[VenueInfo] = None
    background: Optional[BackgroundInfo] = None
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_performer() -> str:
    return """
Extract the following structured information exactly as presented in the answer:

- performer_name: The specific individual's name that the answer claims is performing.
- schedule:
  - date_text: The stated performance date (e.g., "Friday, February 13, 2026").
  - time_text: The stated performance start time (e.g., "7:00pm", "7 PM").
  - venue_name: The stated venue name (e.g., "KeyBank State Theatre").
  - schedule_urls: All URLs explicitly cited that support the event schedule (date/time/venue). Include event pages, ticketing pages, or official listings.
- venue:
  - location_text: The stated venue location (e.g., "Cleveland, Ohio" or "Cleveland, OH").
  - part_of_playhouse_square: The stated claim indicating the venue is part of Playhouse Square (return the textual snippet if present, else null).
  - capacity_text: The stated seating capacity for KeyBank State Theatre (e.g., "3,200", "3200", or "3,200 seats").
  - venue_urls: All URLs explicitly cited that support venue details (location, being part of Playhouse Square, capacity).
- background:
  - origin_text: The stated origin of the performer (e.g., "Cleveland", "Cleveland, Ohio").
  - stand_up_comedian_text: The textual claim indicating the performer is a stand-up comedian (return the snippet if present).
  - hosted_tv_shows: A list of the names of any TV shows the performer has hosted mentioned in the answer (e.g., "The Price Is Right").
  - starred_in_drew_carey_show_text: The textual claim indicating the performer starred in "The Drew Carey Show" (return the snippet if present).
  - price_is_right_host_since_2007_text: The textual claim indicating the performer has been hosting "The Price Is Right" since 2007 (return the snippet if present).
  - background_urls: All URLs explicitly cited that support these background/TV career claims (Wikipedia pages, official bios, etc.).
- all_urls: A list of ALL URLs present anywhere in the answer (including those already listed above). Do not deduplicate—include every URL you see.

Important:
- Do not invent or infer information not explicitly present in the answer.
- For all URL fields, include only valid URLs explicitly present in the answer text (plain links or markdown links).
- If any field is missing from the answer, return null for that field or an empty list for list fields.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: Optional[List[str]]) -> List[str]:
    """Merge and deduplicate URL lists, preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    merged.append(u)
    return merged


def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    try:
        return int(digits) if digits else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification tree construction & checks                                     #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, data: PerformerExtraction) -> None:
    # Top-level critical node
    performer_node = evaluator.add_parallel(
        id="performer_identification",
        desc="Identify the performer and ensure all stated constraints are satisfied",
        parent=root,
        critical=True,
    )

    # 1) Performer name provided (critical)
    name_present = bool(data.performer_name and data.performer_name.strip())
    evaluator.add_custom_node(
        result=name_present,
        id="performer_name_provided",
        desc="The response provides the performer's name (a specific individual)",
        parent=performer_node,
        critical=True
    )

    # Prepare reusable variables
    performer_name = (data.performer_name or "").strip()

    # Extract source groups (with fallbacks)
    schedule_urls = merge_sources(
        data.schedule.schedule_urls if data and data.schedule else [],
        data.all_urls
    )
    venue_urls = merge_sources(
        data.venue.venue_urls if data and data.venue else [],
        data.all_urls
    )
    background_urls = merge_sources(
        data.background.background_urls if data and data.background else [],
        data.all_urls
    )

    # 2) Performance Schedule & Venue (critical, parallel)
    sched_node = evaluator.add_parallel(
        id="performance_schedule_and_venue",
        desc="Verify the scheduled show details and the specified venue",
        parent=performer_node,
        critical=True
    )

    # 2.1 Date
    date_leaf = evaluator.add_leaf(
        id="performance_date",
        desc="The performer has a scheduled show on Friday, February 13, 2026",
        parent=sched_node,
        critical=True
    )
    date_claim = (
        f"{performer_name} has a scheduled show on Friday, February 13, 2026 at KeyBank State Theatre in Cleveland, Ohio."
        if performer_name else
        "There is a scheduled show on Friday, February 13, 2026 at KeyBank State Theatre in Cleveland, Ohio."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=schedule_urls,
        additional_instruction="Verify the event date on the cited event/ticket/listing page(s). Allow abbreviated formats like 'Fri, Feb 13, 2026'."
    )

    # 2.2 Time
    time_leaf = evaluator.add_leaf(
        id="show_start_time",
        desc="The performance begins at 7:00pm",
        parent=sched_node,
        critical=True
    )
    time_claim = (
        f"The performance by {performer_name} begins at 7:00pm."
        if performer_name else
        "The performance begins at 7:00pm."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=schedule_urls,
        additional_instruction="Verify the event start time. Treat '7:00pm', '7 PM', or '7 p.m.' as equivalent."
    )

    # 2.3 Venue name
    venue_name_leaf = evaluator.add_leaf(
        id="specific_venue_name",
        desc="The performance takes place at KeyBank State Theatre",
        parent=sched_node,
        critical=True
    )
    venue_name_claim = (
        f"The performance by {performer_name} takes place at KeyBank State Theatre."
        if performer_name else
        "The performance takes place at KeyBank State Theatre."
    )
    await evaluator.verify(
        claim=venue_name_claim,
        node=venue_name_leaf,
        sources=schedule_urls,
        additional_instruction="Verify the venue name on the event page(s). Allow minor naming style differences (e.g., 'Theatre' vs 'theatre' casing)."
    )

    # 3) Venue Characteristics (critical, parallel)
    venue_node = evaluator.add_parallel(
        id="venue_characteristics",
        desc="Verify the venue details for the performance",
        parent=performer_node,
        critical=True
    )

    # 3.1 Location Cleveland, Ohio
    location_leaf = evaluator.add_leaf(
        id="venue_location_cleveland_ohio",
        desc="The venue (and thus the performance) is located in Cleveland, Ohio",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="KeyBank State Theatre is located in Cleveland, Ohio.",
        node=location_leaf,
        sources=venue_urls,
        additional_instruction="Verify the venue's location is Cleveland, Ohio (allow 'Cleveland, OH')."
    )

    # 3.2 Part of Playhouse Square
    playhouse_leaf = evaluator.add_leaf(
        id="venue_part_of_playhouse_square",
        desc="The venue is part of the Playhouse Square theater district",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="KeyBank State Theatre is part of Playhouse Square.",
        node=playhouse_leaf,
        sources=venue_urls,
        additional_instruction="Verify that KeyBank State Theatre is one of the Playhouse Square venues."
    )

    # 3.3 Capacity 3,200
    capacity_leaf = evaluator.add_leaf(
        id="venue_capacity_3200",
        desc="The venue has a seating capacity of 3,200 seats",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="KeyBank State Theatre has a seating capacity of 3,200.",
        node=capacity_leaf,
        sources=venue_urls,
        additional_instruction="Verify the seating capacity equals approximately 3,200. Allow minor formatting differences like '3200' vs '3,200'."
    )

    # 3.4 Classified as large (>1,000)
    # Ensure this runs after capacity verification so auto preconditions can skip if capacity fails
    large_leaf = evaluator.add_leaf(
        id="venue_classified_large_over_1000",
        desc="Based on the stated standard, the venue is classified as a large theater venue because its capacity exceeds 1,000 seats",
        parent=venue_node,
        critical=True
    )
    # Construct a simple logical claim using capacity if available
    cap_num = parse_int_from_text(data.venue.capacity_text if data and data.venue else None) or 3200
    large_claim = f"A venue with {cap_num} seats exceeds the 1,000-seat threshold and is therefore classified as a large theater venue."
    await evaluator.verify(
        claim=large_claim,
        node=large_leaf,
        additional_instruction="This is a simple logical check: if capacity > 1,000, classify as 'large'."
    )

    # 4) Performer Background & TV Career (critical, parallel)
    background_node = evaluator.add_parallel(
        id="performer_background_and_tv_career",
        desc="Verify the performer's origin and television-career constraints",
        parent=performer_node,
        critical=True
    )

    # 4.1 Cleveland origin
    origin_leaf = evaluator.add_leaf(
        id="cleveland_origin",
        desc="The performer is Cleveland-bred (originally from Cleveland)",
        parent=background_node,
        critical=True
    )
    origin_claim = (
        f"{performer_name} is originally from Cleveland, Ohio."
        if performer_name else
        "The performer is originally from Cleveland, Ohio."
    )
    await evaluator.verify(
        claim=origin_claim,
        node=origin_leaf,
        sources=background_urls,
        additional_instruction="Verify the performer's origin is Cleveland (allow 'Cleveland, OH' or 'Cleveland, Ohio')."
    )

    # 4.2 Stand-up comedian
    standup_leaf = evaluator.add_leaf(
        id="stand_up_comedian",
        desc="The performer is a stand-up comedian",
        parent=background_node,
        critical=True
    )
    standup_claim = (
        f"{performer_name} is a stand-up comedian."
        if performer_name else
        "The performer is a stand-up comedian."
    )
    await evaluator.verify(
        claim=standup_claim,
        node=standup_leaf,
        sources=background_urls,
        additional_instruction="Verify the person's profession includes stand-up comedy."
    )

    # 4.3 Hosted TV shows (general)
    hosted_leaf = evaluator.add_leaf(
        id="hosted_tv_shows",
        desc="The performer has hosted television shows",
        parent=background_node,
        critical=True
    )
    hosted_claim = (
        f"{performer_name} has served as the host of one or more television shows."
        if performer_name else
        "The performer has served as the host of one or more television shows."
    )
    await evaluator.verify(
        claim=hosted_claim,
        node=hosted_leaf,
        sources=background_urls,
        additional_instruction="Check biography or reliable sources indicating TV hosting roles."
    )

    # 4.4 Starred in The Drew Carey Show
    drew_show_leaf = evaluator.add_leaf(
        id="starred_in_drew_carey_show",
        desc="The performer starred in 'The Drew Carey Show'",
        parent=background_node,
        critical=True
    )
    drew_claim = (
        f"{performer_name} starred in 'The Drew Carey Show'."
        if performer_name else
        "The performer starred in 'The Drew Carey Show'."
    )
    await evaluator.verify(
        claim=drew_claim,
        node=drew_show_leaf,
        sources=background_urls,
        additional_instruction="Verify that the performer is a principal star of 'The Drew Carey Show'."
    )

    # 4.5 Hosting The Price Is Right since 2007
    pir_leaf = evaluator.add_leaf(
        id="hosted_price_is_right_since_2007",
        desc="The performer has been hosting 'The Price Is Right' since 2007",
        parent=background_node,
        critical=True
    )
    pir_claim = (
        f"{performer_name} has been the host of 'The Price Is Right' since 2007."
        if performer_name else
        "The performer has been the host of 'The Price Is Right' since 2007."
    )
    await evaluator.verify(
        claim=pir_claim,
        node=pir_leaf,
        sources=background_urls,
        additional_instruction="Verify the year 2007 as the start of hosting 'The Price Is Right'."
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
    Evaluate an answer for the KeyBank State Theatre (Feb 13, 2026) performer identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Single top-level criterion branch
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_performer(),
        template_class=PerformerExtraction,
        extraction_name="performer_extraction"
    )

    # (Optional) Record expected constraints as ground truth context
    evaluator.add_ground_truth({
        "expected_date": "Friday, February 13, 2026",
        "expected_time": "7:00pm",
        "expected_venue": "KeyBank State Theatre (Playhouse Square), Cleveland, Ohio",
        "expected_capacity": "3,200",
        "large_venue_threshold": "> 1,000 seats",
        "background_requirements": [
            "Cleveland-bred",
            "Stand-up comedian",
            "Hosted TV shows",
            "Starred in 'The Drew Carey Show'",
            "Hosting 'The Price Is Right' since 2007"
        ]
    })

    # Build and run verifications
    await build_verification_tree(evaluator, root, extracted)

    return evaluator.get_summary()