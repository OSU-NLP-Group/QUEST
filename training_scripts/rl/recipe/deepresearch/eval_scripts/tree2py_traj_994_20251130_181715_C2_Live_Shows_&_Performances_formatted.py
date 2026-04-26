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
TASK_ID = "phantom_q1_2026_largest_venue"
TASK_DESCRIPTION = (
    "The Phantom of the Opera is embarking on a North American tour from 2025-2026. "
    "Identify the venue with the largest seating capacity that will host The Phantom of the Opera during the first quarter of 2026 (January 1 - March 31, 2026). "
    "Provide the official venue name, the city and state where it is located, and its exact seating capacity. "
    "Include a reference URL to the official tour schedule."
)

Q1_2026_START = "2026-01-01"
Q1_2026_END = "2026-03-31"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SelectedVenue(BaseModel):
    """The venue the answer claims is the largest-capacity Q1 2026 Phantom stop."""
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None  # keep as string for robustness (e.g., "3,000")
    tour_schedule_url: Optional[str] = None  # official tour schedule URL cited in the answer
    capacity_source_urls: List[str] = Field(default_factory=list)  # URLs cited for seating capacity
    date_text: Optional[str] = None  # free-form date description from the answer, if any
    start_date: Optional[str] = None  # if the answer provides machine-readable date
    end_date: Optional[str] = None


class CandidateVenue(BaseModel):
    """Any venue the answer lists as hosting Phantom in Q1 2026 (used for the 'largest' check)."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None
    tour_schedule_url: Optional[str] = None  # schedule URL for this specific venue, if provided
    dates_text: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)


class VenueExtraction(BaseModel):
    """Full extraction from the answer."""
    selected_venue: Optional[SelectedVenue] = None
    candidate_venues_q1_2026: List[CandidateVenue] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract the venue information the answer presents for the largest seating capacity among The Phantom of the Opera's North American tour stops during Q1 2026 (January 1 – March 31, 2026).

    You must extract:
    1) selected_venue:
       - official_name: The official venue name chosen as the largest-capacity Q1 2026 stop.
       - city: City of the venue.
       - state: State of the venue (use the full state name if provided; otherwise use the abbreviation as-is).
       - seating_capacity: The exact numerical seating capacity stated in the answer (keep any commas or formatting as-is).
       - tour_schedule_url: A URL that the answer claims is the official tour schedule confirming the venue and its dates.
       - capacity_source_urls: An array of URLs cited in the answer that support the venue's seating capacity (e.g., official venue page, Wikipedia, credible databases). If none are provided, return an empty array.
       - date_text: Any free-form date text mentioned for this venue in the answer (e.g., "March 10–15, 2026"), or null if not given.
       - start_date: The start date of performance at the venue if explicitly stated (prefer ISO 8601 like YYYY-MM-DD if present), otherwise null.
       - end_date: The end date if explicitly stated, otherwise null.

    2) candidate_venues_q1_2026:
       Extract all venues the answer lists as hosting Phantom in Q1 2026 (Jan 1–Mar 31, 2026), including potential comparisons for capacity. For each:
       - name
       - city
       - state
       - seating_capacity (as stated in the answer; keep formatting)
       - tour_schedule_url (the cited schedule URL for that venue, if given)
       - capacity_source_urls (array of capacity-supporting URLs, if given)
       - dates_text (any date mention for that venue)
       - start_date (if clearly stated; else null)
       - end_date (if clearly stated; else null)

    Special rules for URL extraction:
    - Extract only URLs explicitly present in the answer (including markdown links). If not provided, return null for single URL fields and [] for arrays.
    - Include full URLs; if a URL lacks protocol, prepend "http://".

    If the answer does not include some fields, return null (or [] for lists) for those fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_capacity_to_int(capacity_text: Optional[str]) -> Optional[int]:
    """
    Attempt to parse a seating capacity from a free-form string.
    Returns an integer if a clear number is found; otherwise None.
    Examples:
      "3,000" -> 3000
      "approximately 12,500 seats" -> 12500
      "10k" or "10,000+" -> 10000 (best-effort)
      "1500-2000" -> picks the larger if both numeric found; otherwise first numeric.
    """
    if not capacity_text:
        return None

    # Normalize common suffixes like "k" meaning thousand
    text = capacity_text.lower().strip()

    # Find all groups of digits separated by non-digits (e.g., handle "1,500", "1500", "10k")
    # Remove commas before regex to handle thousand separators
    cleaned = re.sub(r"[^\dk]", "", text.replace(",", ""))

    # Handle "10k" -> 10000
    if cleaned.endswith("k"):
        try:
            base = int(cleaned[:-1])
            return base * 1000
        except Exception:
            pass

    # Find numbers (could be ranges)
    nums = re.findall(r"\d+", text.replace(",", ""))
    if not nums:
        return None

    try:
        if len(nums) == 1:
            return int(nums[0])
        else:
            # If multiple numbers (e.g., "1500-2000"), select the max
            return max(int(n) for n in nums if n.isdigit())
    except Exception:
        return None


def list_q1_candidates_string(candidates: List[CandidateVenue]) -> str:
    """Create a human-readable summary of candidates and their capacities for the 'largest' check."""
    parts = []
    for c in candidates:
        cap = c.seating_capacity or "unknown"
        parts.append(f"{(c.name or 'unknown venue')} ({(c.city or 'unknown city')}, {(c.state or 'unknown state')}): capacity={cap}")
    return "; ".join(parts) if parts else "none listed"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_selection_validation(
    evaluator: Evaluator,
    parent_node,
    extraction: VenueExtraction,
) -> None:
    """
    Build and verify the 'Venue_Selection_Validation' parallel critical node:
    - On_Official_Tour_Schedule
    - Q1_2026_Date_Overlap
    - Largest_Capacity_Among_Qualifying
    """
    sel = extraction.selected_venue or SelectedVenue()
    venue_name = sel.official_name or ""
    city = sel.city or ""
    state = sel.state or ""
    schedule_url = sel.tour_schedule_url

    candidates = extraction.candidate_venues_q1_2026 or []
    candidates_summary = list_q1_candidates_string(candidates)
    selected_capacity_text = sel.seating_capacity or ""
    selected_capacity_int = parse_capacity_to_int(selected_capacity_text)

    # Create parallel critical node
    selection_node = evaluator.add_parallel(
        id="Venue_Selection_Validation",
        desc="The identified venue meets all three selection criteria: official tour participation, Q1 2026 date overlap, and largest capacity among qualifying venues",
        parent=parent_node,
        critical=True
    )

    # 1) On_Official_Tour_Schedule (leaf, critical)
    on_official_leaf = evaluator.add_leaf(
        id="On_Official_Tour_Schedule",
        desc="The identified venue is confirmed to be an official stop on The Phantom of the Opera's 2025-2026 North American tour",
        parent=selection_node,
        critical=True,
    )
    claim_official = (
        f"The official tour schedule page lists {venue_name} in {city}, {state} as a stop on "
        f"The Phantom of the Opera's 2025–2026 North American tour."
    )
    await evaluator.verify(
        claim=claim_official,
        node=on_official_leaf,
        sources=schedule_url,
        additional_instruction=(
            "Verify that the provided page is indeed an official tour schedule and that it specifically lists the venue "
            f"'{venue_name}' in '{city}, {state}' as a tour stop. Allow reasonable name variants."
        ),
    )

    # 2) Q1_2026_Date_Overlap (leaf, critical)
    q1_overlap_leaf = evaluator.add_leaf(
        id="Q1_2026_Date_Overlap",
        desc="The venue's scheduled performance dates fall within or overlap with the first quarter of 2026 (January 1 - March 31, 2026)",
        parent=selection_node,
        critical=True,
    )
    claim_q1 = (
        f"The scheduled performance date(s) for {venue_name} at {city}, {state} occur between January 1 and March 31, 2026 (inclusive), "
        "or overlap that window."
    )
    await evaluator.verify(
        claim=claim_q1,
        node=q1_overlap_leaf,
        sources=schedule_url,
        additional_instruction=(
            "Confirm that the tour schedule shows date(s) in Q1 2026. If the range overlaps Q1 2026 at least one day, consider it a valid overlap."
        ),
    )

    # 3) Largest_Capacity_Among_Qualifying (leaf, critical)
    largest_leaf = evaluator.add_leaf(
        id="Largest_Capacity_Among_Qualifying",
        desc="Among all venues meeting the tour and date criteria, the identified venue has the largest seating capacity",
        parent=selection_node,
        critical=True,
    )
    # Build a claim based on the answer-provided candidates
    # We do not enforce external evidence here; the judge checks the logic against the answer content.
    claim_largest = (
        f"Among the Q1 2026 venues listed in the answer, {venue_name} has the largest seating capacity "
        f"({selected_capacity_text}). Candidate venues considered: {candidates_summary}. "
        "No other candidate listed has a strictly greater seating capacity."
    )
    add_ins_largest = (
        "Use strictly numeric comparisons of capacities mentioned in the answer. "
        "Treat '3,000' equivalent to '3000'. If any candidate has missing or ambiguous capacity, treat the 'largest' claim as not verified. "
        "Ignore venues not stated as Q1 2026 in the answer."
    )
    await evaluator.verify(
        claim=claim_largest,
        node=largest_leaf,
        sources=None,  # Logical check against the answer content
        additional_instruction=add_ins_largest,
    )

    # Record helpful custom info
    evaluator.add_custom_info(
        info={
            "selected_capacity_text": selected_capacity_text,
            "selected_capacity_int": selected_capacity_int,
            "candidates_count": len(candidates),
            "candidates_summary": candidates_summary,
            "q1_window": {"start": Q1_2026_START, "end": Q1_2026_END},
        },
        info_type="selection_validation_insights"
    )


async def verify_complete_information(
    evaluator: Evaluator,
    parent_node,
    extraction: VenueExtraction,
) -> None:
    """
    Build and verify the 'Complete_Venue_Information' parallel critical node:
    - Tour_Schedule_Reference
    - Official_Venue_Name
    - City_Correct
    - State_Correct
    - Seating_Capacity_Number
    """
    sel = extraction.selected_venue or SelectedVenue()
    venue_name = sel.official_name or ""
    city = sel.city or ""
    state = sel.state or ""
    capacity_text = sel.seating_capacity or ""
    schedule_url = sel.tour_schedule_url
    capacity_urls = sel.capacity_source_urls or []

    # Create parallel critical node
    info_node = evaluator.add_parallel(
        id="Complete_Venue_Information",
        desc="All required venue details and documentation are accurately provided",
        parent=parent_node,
        critical=True
    )

    # 1) Tour_Schedule_Reference (leaf, critical)
    tour_ref_leaf = evaluator.add_leaf(
        id="Tour_Schedule_Reference",
        desc="Valid URL reference to the official Phantom of the Opera tour schedule confirming the venue and dates",
        parent=info_node,
        critical=True,
    )
    claim_tour_ref = (
        "This webpage is the official tour schedule for The Phantom of the Opera (North American tour 2025–2026), "
        "and it includes venue/date listings."
    )
    await evaluator.verify(
        claim=claim_tour_ref,
        node=tour_ref_leaf,
        sources=schedule_url,
        additional_instruction=(
            "Confirm the page is an official schedule (e.g., official show website or official producers). "
            "If the page is a third-party aggregator or unrelated, mark as not supported."
        ),
    )

    # 2) Official_Venue_Name (leaf, critical)
    venue_name_leaf = evaluator.add_leaf(
        id="Official_Venue_Name",
        desc="The official name of the venue is correctly provided",
        parent=info_node,
        critical=True,
    )
    claim_venue_name = f"The tour schedule lists the venue as '{venue_name}' (allowing minor name variants)."
    await evaluator.verify(
        claim=claim_venue_name,
        node=venue_name_leaf,
        sources=schedule_url,
        additional_instruction=(
            "Verify that the schedule page lists the venue with a name equivalent to the provided value "
            "(allowing minor punctuation, abbreviation, or branding variants)."
        ),
    )

    # 3) City_Correct (leaf, critical)
    city_leaf = evaluator.add_leaf(
        id="City_Correct",
        desc="The city where the venue is located is correctly provided",
        parent=info_node,
        critical=True,
    )
    claim_city = f"The venue is located in the city '{city}'."
    await evaluator.verify(
        claim=claim_city,
        node=city_leaf,
        sources=schedule_url,
        additional_instruction="Confirm the city for the listed venue matches the provided city (allow reasonable abbreviations).",
    )

    # 4) State_Correct (leaf, critical)
    state_leaf = evaluator.add_leaf(
        id="State_Correct",
        desc="The state where the venue is located is correctly provided",
        parent=info_node,
        critical=True,
    )
    claim_state = f"The venue is located in the state '{state}'."
    await evaluator.verify(
        claim=claim_state,
        node=state_leaf,
        sources=schedule_url,
        additional_instruction=(
            "Confirm the state associated with the venue matches the provided state. "
            "Allow standard postal abbreviations and full names to be equivalent (e.g., 'CA' == 'California')."
        ),
    )

    # 5) Seating_Capacity_Number (leaf, critical)
    capacity_leaf = evaluator.add_leaf(
        id="Seating_Capacity_Number",
        desc="The exact numerical seating capacity is correctly stated",
        parent=info_node,
        critical=True,
    )
    claim_capacity = f"The seating capacity of '{venue_name}' is {capacity_text}."
    # Use capacity source URLs if available; otherwise still attempt with schedule_url as fallback (may fail, acceptable).
    sources_for_capacity: List[str] = capacity_urls[:] if capacity_urls else []
    if schedule_url:
        sources_for_capacity.append(schedule_url)

    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=sources_for_capacity if sources_for_capacity else None,
        additional_instruction=(
            "Verify the venue's seating capacity using the provided capacity-supporting URL(s). "
            "Treat '3,000' equivalent to '3000'. If sources are missing or contradictory, mark as not supported."
        ),
    )

    # Custom info record
    evaluator.add_custom_info(
        info={
            "schedule_url": schedule_url,
            "capacity_urls": capacity_urls,
            "provided_capacity_text": capacity_text,
            "parsed_capacity_int": parse_capacity_to_int(capacity_text),
        },
        info_type="complete_info_insights"
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
    Evaluate an answer for the Phantom Q1 2026 largest venue task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root sequential (though it has one main child)
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

    # Extraction
    extraction: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build main critical sequential node (maps to Identify_Largest_Q1_2026_Venue)
    identify_node = evaluator.add_sequential(
        id="Identify_Largest_Q1_2026_Venue",
        desc="Correctly identify and provide complete information about the venue with the largest seating capacity hosting The Phantom of the Opera during Q1 2026 (January-March 2026)",
        parent=root,
        critical=True,
    )

    # First block: selection validation
    await verify_selection_validation(evaluator, identify_node, extraction)

    # Second block: complete venue information
    await verify_complete_information(evaluator, identify_node, extraction)

    # Return summary
    return evaluator.get_summary()