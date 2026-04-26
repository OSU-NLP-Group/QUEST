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
TASK_ID = "ca_music_festivals_2026_q2"
TASK_DESCRIPTION = """Identify three music festivals taking place in California between April 1, 2026, and June 30, 2026, that meet all of the following requirements:

1. The festival must be located in California
2. The festival must occur entirely within the date range of April 1 - June 30, 2026
3. The festival must be a multi-day event spanning at least 2 consecutive days
4. The festival must take place at an outdoor venue
5. The festival must feature at least 3 stages
6. The festival must include artists from at least 2 different music genres
7. The festival must be an established annual event (not a first-time festival)
8. The festival must accommodate at least 10,000 attendees
9. The festival dates must be officially announced and confirmed
10. At least one headlining artist must be publicly announced for the festival
11. Tickets must be available for purchase or officially announced
12. The total festival duration must be between 2-4 days
13. The festival must be accessible to the general public (not private or invite-only)
14. The festival's age policy must be clearly stated

For each festival, provide:
- The festival name
- The exact dates
- The venue/location
- At least one confirmed headlining artist
- A reference URL to the official festival website or reliable source
"""

DATE_RANGE_START = "April 1, 2026"
DATE_RANGE_END = "June 30, 2026"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Festival(BaseModel):
    name: Optional[str] = None
    venue: Optional[str] = None
    location: Optional[str] = None  # City/region text as given in the answer
    dates_text: Optional[str] = None  # Exact dates as written in the answer
    start_date: Optional[str] = None  # If the answer supplies parsable start date
    end_date: Optional[str] = None    # If the answer supplies parsable end date
    headliners: List[str] = Field(default_factory=list)

    # Optional descriptive fields (strings to maximize compatibility)
    outdoor: Optional[str] = None
    stages_count_text: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    annual: Optional[str] = None
    capacity_text: Optional[str] = None
    tickets_status: Optional[str] = None
    public_access: Optional[str] = None
    age_policy: Optional[str] = None

    # Sources explicitly provided in the answer
    reference_urls: List[str] = Field(default_factory=list)


class FestivalsExtraction(BaseModel):
    festivals: List[Festival] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals() -> str:
    return """
Extract up to three (3) music festivals from the answer that the user claims meet the specified criteria.
For each festival, extract the following fields if present in the answer text:

- name: The festival name as written
- venue: The specific venue name (e.g., "Empire Polo Club", "Golden Gate Park") if stated
- location: The city/area + state text as stated (e.g., "Indio, CA", "San Francisco, California")
- dates_text: The exact dates as written in the answer (e.g., "April 12–14, 2026")
- start_date: If the answer states a clear start date, copy it verbatim (e.g., "April 12, 2026")
- end_date: If the answer states a clear end date, copy it verbatim (e.g., "April 14, 2026")
- headliners: A list of at least one named headlining artist if provided
- outdoor: Any wording indicating outdoor setting (e.g., "outdoor", "open-air", "festival grounds", etc.), or null
- stages_count_text: Any wording indicating number of stages (e.g., "6 stages", "over 3 stages"), or null
- genres: A list of genres if explicitly mentioned (e.g., ["rock", "hip hop"]), or an empty list if not stated
- annual: Any wording that indicates it is an annual/recurring event (e.g., "annual", "since 2010", "Xth edition"), or null
- capacity_text: Any wording about attendance/capacity (e.g., "125,000 attendees", "capacity 10,000+"), or null
- tickets_status: Wording that tickets are on sale/announced ("tickets available", "pre-sale", "on sale soon"), or null
- public_access: Any wording indicating general public access ("General Admission", "open to public"), or null
- age_policy: Any wording that states age policy ("All Ages", "18+", "21+"), or null
- reference_urls: All explicit URLs associated with this festival in the answer (must be actual URLs shown or linked). Include the official festival site if present and any reliable source links given.

Important rules:
- Do not invent or infer any URLs or details that are not stated in the answer.
- Only include fields that are explicitly present; otherwise set the field to null or empty list as appropriate.
- If the answer provides more than three festivals, only extract the first three in the order they appear.
- If fewer than three festivals are present, return whatever is present; the missing ones will be handled later.
"""


# --------------------------------------------------------------------------- #
# Helper functions for claims/instructions                                    #
# --------------------------------------------------------------------------- #
def _fmt_name_for_claim(name: Optional[str], index: int) -> str:
    if name and name.strip():
        return name.strip()
    return f"festival #{index + 1}"


def _date_phrase(f: Festival) -> str:
    if f.start_date and f.end_date:
        return f"from {f.start_date} to {f.end_date}"
    if f.dates_text:
        return f.dates_text
    return "the listed dates"


def _join_sources(urls: List[str]) -> List[str]:
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification for a single festival                                          #
# --------------------------------------------------------------------------- #
async def verify_festival(
    evaluator: Evaluator,
    parent_node,
    fest: Festival,
    index: int
) -> None:
    idx = index + 1
    fest_node = evaluator.add_parallel(
        id=f"festival_{idx}",
        desc=f"{['First','Second','Third'][index]} qualifying festival identification and verification",
        parent=parent_node,
        critical=False  # Partial credit allowed per festival
    )

    # Existence / presence checks (critical)
    has_name = fest.name is not None and fest.name.strip() != ""
    evaluator.add_custom_node(
        result=has_name,
        id=f"festival_{idx}_name",
        desc="Festival name is provided",
        parent=fest_node,
        critical=True
    )

    has_venue = (fest.venue is not None and fest.venue.strip() != "") or (fest.location is not None and fest.location.strip() != "")
    evaluator.add_custom_node(
        result=has_venue,
        id=f"festival_{idx}_venue_provided",
        desc="Specific venue/location name is stated in the answer",
        parent=fest_node,
        critical=True
    )

    has_dates = (fest.dates_text is not None and fest.dates_text.strip() != "") or (fest.start_date and fest.end_date)
    evaluator.add_custom_node(
        result=has_dates,
        id=f"festival_{idx}_dates_provided",
        desc="Exact dates are stated in the answer",
        parent=fest_node,
        critical=True
    )

    has_headliner_name = bool(fest.headliners)
    evaluator.add_custom_node(
        result=has_headliner_name,
        id=f"festival_{idx}_headliner_provided",
        desc="At least one specific headliner name is stated in the answer",
        parent=fest_node,
        critical=True
    )

    has_ref = bool(fest.reference_urls)
    evaluator.add_custom_node(
        result=has_ref,
        id=f"festival_{idx}_reference",
        desc="Valid reference URL provided for festival information",
        parent=fest_node,
        critical=True
    )

    # Build verification leaves (each critical; evidence‑grounded)
    # All claims will use the same sources list from the answer
    sources = _join_sources(fest.reference_urls)
    fest_name = _fmt_name_for_claim(fest.name, index)
    date_phrase = _date_phrase(fest)

    # 1) Location in California
    node_loc = evaluator.add_leaf(
        id=f"festival_{idx}_california_location",
        desc="Festival takes place in California",
        parent=fest_node,
        critical=True
    )

    claim_loc = f"The festival '{fest_name}' takes place in California."
    add_ins_loc = "Check that the official site or a reliable source indicates the festival location is within the U.S. state of California (e.g., a city in CA, or 'California' explicitly)."

    # 2) Dates within the specified range
    node_range = evaluator.add_leaf(
        id=f"festival_{idx}_dates_within_range",
        desc="Festival occurs between April 1, 2026 and June 30, 2026",
        parent=fest_node,
        critical=True
    )

    claim_range = f"The dates of '{fest_name}' ({date_phrase}) fall entirely between April 1, 2026 and June 30, 2026 (inclusive)."
    add_ins_range = "Verify that the dates shown on the official site or a reliable source fall fully within 2026-04-01 to 2026-06-30."

    # 3) Multiday (>= 2 consecutive days)
    node_multiday = evaluator.add_leaf(
        id=f"festival_{idx}_multiday",
        desc="Festival spans at least 2 consecutive days",
        parent=fest_node,
        critical=True
    )

    claim_multiday = f"'{fest_name}' spans at least two consecutive days."
    add_ins_multiday = "Use the listed start and end dates (or explicit schedule) on the source to confirm it covers two or more consecutive days."

    # 4) Duration is 2–4 days
    node_duration = evaluator.add_leaf(
        id=f"festival_{idx}_duration",
        desc="Total festival duration is between 2-4 days",
        parent=fest_node,
        critical=True
    )

    claim_duration = f"'{fest_name}' lasts between two and four days inclusive."
    add_ins_duration = "Based on the start and end dates or official schedule, confirm that the total duration is 2, 3, or 4 days."

    # 5) Outdoor venue
    node_outdoor = evaluator.add_leaf(
        id=f"festival_{idx}_outdoor_venue",
        desc="Festival takes place at an outdoor venue",
        parent=fest_node,
        critical=True
    )

    venue_part = f" ({fest.venue})" if fest.venue else ""
    claim_outdoor = f"'{fest_name}' is held at an outdoor venue{venue_part}."
    add_ins_outdoor = "Look for explicit cues like 'outdoor', 'open-air', festival grounds, parks, polo club, fairgrounds, etc., on the source page."

    # 6) At least 3 stages
    node_stages = evaluator.add_leaf(
        id=f"festival_{idx}_multiple_stages",
        desc="Festival features at least 3 stages",
        parent=fest_node,
        critical=True
    )

    claim_stages = f"'{fest_name}' features at least three stages."
    add_ins_stages = "Confirm the site (e.g., map, lineup, FAQs) mentions three or more stages."

    # 7) Genre diversity (>= 2 genres)
    node_genres = evaluator.add_leaf(
        id=f"festival_{idx}_genre_diversity",
        desc="Festival includes artists from at least 2 different music genres",
        parent=fest_node,
        critical=True
    )

    if fest.genres and len(fest.genres) >= 2:
        genre_hint = f"such as {', '.join(fest.genres[:2])}"
    else:
        genre_hint = "across different styles"
    claim_genres = f"'{fest_name}' includes artists from at least two different music genres, {genre_hint}."
    add_ins_genres = "Seek explicit genre labels or clearly diverse headliners/lineups across genres on the source."

    # 8) Established annual event
    node_annual = evaluator.add_leaf(
        id=f"festival_{idx}_established",
        desc="Festival is an established annual event (not a first-time event)",
        parent=fest_node,
        critical=True
    )

    claim_annual = f"'{fest_name}' is an established annual/recurring festival with previous editions (not a first-time event)."
    add_ins_annual = "Look for evidence like 'annual', 'since <year>', edition numbers, or prior-year pages/posts on the official site or reliable source."

    # 9) Capacity >= 10,000
    node_capacity = evaluator.add_leaf(
        id=f"festival_{idx}_capacity",
        desc="Festival accommodates at least 10,000 attendees",
        parent=fest_node,
        critical=True
    )

    claim_capacity = f"'{fest_name}' accommodates at least 10,000 attendees."
    add_ins_capacity = "Confirm via official information or reliable reporting about capacity or typical attendance (>= 10,000)."

    # 10) Dates officially announced and confirmed
    node_confirmed = evaluator.add_leaf(
        id=f"festival_{idx}_confirmed_dates",
        desc="Festival dates are officially announced and confirmed",
        parent=fest_node,
        critical=True
    )

    claim_confirmed = f"The dates for '{fest_name}' are officially announced and confirmed."
    add_ins_confirmed = "Treat dates listed on the official festival website as official confirmation; otherwise use a reliable source confirming dates."

    # 11) Headliner publicly announced
    node_headliner_ann = evaluator.add_leaf(
        id=f"festival_{idx}_headliner_announced",
        desc="At least one headlining artist is publicly announced",
        parent=fest_node,
        critical=True
    )

    headliner_example = f" (for example, '{fest.headliners[0]}')" if fest.headliners else ""
    claim_headliner_ann = f"At least one headlining artist has been publicly announced for '{fest_name}'{headliner_example}."
    add_ins_headliner_ann = "Look for an official lineup or announcement naming headliners on the source link(s)."

    # 12) Tickets available or announced
    node_tickets = evaluator.add_leaf(
        id=f"festival_{idx}_tickets_available",
        desc="Tickets are available for purchase or have been announced",
        parent=fest_node,
        critical=True
    )

    claim_tickets = f"Tickets for '{fest_name}' are available for purchase or have been officially announced."
    add_ins_tickets = "Check for 'tickets', 'on sale', 'pre-sale', 'passes', or equivalent on the official site or reliable source."

    # 13) Public access (not private/invite-only)
    node_public = evaluator.add_leaf(
        id=f"festival_{idx}_public_access",
        desc="Festival is accessible to the general public (not private/invite-only)",
        parent=fest_node,
        critical=True
    )

    claim_public = f"'{fest_name}' is open to the general public (not private or invite-only)."
    add_ins_public = "Evidence includes 'General Admission', public ticket sales, or similar language."

    # 14) Age policy clearly stated
    node_age = evaluator.add_leaf(
        id=f"festival_{idx}_age_policy",
        desc="Festival age policy is clearly stated",
        parent=fest_node,
        critical=True
    )

    if fest.age_policy and fest.age_policy.strip():
        age_hint = f" ('{fest.age_policy}')"
    else:
        age_hint = ""
    claim_age = f"The age policy for '{fest_name}' is clearly stated{age_hint}."
    add_ins_age = "Look for 'All Ages', '18+', '21+', or other explicit age policy on the official site or reliable source."

    # Perform parallel verifications for factual leaves
    claims_and_sources = [
        (claim_loc, sources, node_loc, add_ins_loc),
        (claim_range, sources, node_range, add_ins_range),
        (claim_multiday, sources, node_multiday, add_ins_multiday),
        (claim_duration, sources, node_duration, add_ins_duration),
        (claim_outdoor, sources, node_outdoor, add_ins_outdoor),
        (claim_stages, sources, node_stages, add_ins_stages),
        (claim_genres, sources, node_genres, add_ins_genres),
        (claim_annual, sources, node_annual, add_ins_annual),
        (claim_capacity, sources, node_capacity, add_ins_capacity),
        (claim_confirmed, sources, node_confirmed, add_ins_confirmed),
        (claim_headliner_ann, sources, node_headliner_ann, add_ins_headliner_ann),
        (claim_tickets, sources, node_tickets, add_ins_tickets),
        (claim_public, sources, node_public, add_ins_public),
        (claim_age, sources, node_age, add_ins_age),
    ]
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Extract up to three festivals from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_festivals(),
        template_class=FestivalsExtraction,
        extraction_name="festivals_extraction"
    )

    # Enforce exactly 3 entries for evaluation (pad with empty)
    festivals: List[Festival] = (extracted.festivals or [])[:3]
    while len(festivals) < 3:
        festivals.append(Festival())

    # Add context info
    evaluator.add_custom_info(
        {"expected_date_range": {"start": DATE_RANGE_START, "end": DATE_RANGE_END}},
        info_type="context",
        info_name="evaluation_parameters"
    )

    # Build verification tree for three festivals (parallel across festivals)
    for i in range(3):
        await verify_festival(evaluator, root, festivals[i], i)

    return evaluator.get_summary()