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
TASK_ID = "matt_rife_2026_consecutive_pair"
TASK_DESCRIPTION = """
From Matt Rife's 2026 Stay Golden World Tour schedule, identify the pair of consecutive tour dates where both venues have a concert capacity of at least 19,000 seats, the venues are located in different US states, both venues are in the Eastern Time Zone, and the second venue has a larger capacity than the first venue. Provide the dates, venue names, cities, states, and concert capacities for both performances.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Performance(BaseModel):
    date: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None
    # URLs explicitly cited in the answer for different verification aspects
    schedule_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    timezone_urls: List[str] = Field(default_factory=list)
    # Any explicit label or numbering used to indicate order (e.g., "first", "second", "1.", "2.")
    order_label: Optional[str] = None


class PairExtraction(BaseModel):
    performances: List[Performance] = Field(default_factory=list)
    # Whether the answer text clearly indicates which is the first vs second performance
    ordering_indicated: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pair() -> str:
    return """
    Extract all performances mentioned in the answer that relate to Matt Rife's 2026 Stay Golden World Tour.
    You must return them in the exact order they appear in the answer.

    For each performance, extract:
    - date: The performance date string as written (e.g., "March 3, 2026" or "2026-03-03")
    - venue_name: The venue (arena/stadium) name
    - city: The city
    - state: The US state or equivalent jurisdiction (e.g., "NY", "New York", "Washington, DC")
    - capacity: The concert capacity number as provided in the answer (keep as a raw string; do not convert)
    - schedule_urls: All URLs the answer cites that specifically support this date+venue being on Matt Rife's 2026 Stay Golden World Tour schedule
    - capacity_urls: All URLs the answer cites that specifically support the venue's concert seating capacity
    - timezone_urls: All URLs the answer cites that specifically support the city/state (or venue location) being in the Eastern Time Zone
    - order_label: If the answer explicitly labels or numbers the performances (e.g., "First", "Second", "1)", "2)", "Performance #1"), include that label; otherwise, set null.

    Also extract:
    - ordering_indicated: Set to true if the answer clearly indicates which performance is the "first" vs the "second", either via explicit labels or clear numbering/ordering markers; otherwise, false.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.
    - Keep all fields exactly as the answer presents them (do not normalize formats).
    - If a field is missing, set it to null. If a URL set is missing, return an empty array for that field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _normalize_state(s: Optional[str]) -> str:
    if not s:
        return ""
    t = s.strip().lower()
    t = t.replace(".", "").replace(",", "")
    t = re.sub(r"\s+", "", t)
    # Map common variants
    mapping = {
        "washingtondc": "dc",
        "districtofcolumbia": "dc",
        "newyork": "ny",
        "massachusetts": "ma",
        "michigan": "mi",
        "florida": "fl",
        "georgia": "ga",
        "northcarolina": "nc",
        "southcarolina": "sc",
        "pennsylvania": "pa",
        "ohio": "oh",
        "virginia": "va",
        "westvirginia": "wv",
        "newjersey": "nj",
        "connecticut": "ct",
        "rhodeisland": "ri",
        "delaware": "de",
        "maryland": "md",
        "maine": "me",
        "vermont": "vt",
        "newhampshire": "nh",
        "tennessee": "tn",
        "kentucky": "ky",
        "indiana": "in",
        "illinois": "il",
        "alabama": "al",
        "mississippi": "ms",
        "louisiana": "la",
        "arkansas": "ar",
        "texas": "tx",
        "dc": "dc",
        "ny": "ny",
        "nj": "nj",
        "pa": "pa",
        "ma": "ma",
        "ct": "ct",
        "ri": "ri",
        "de": "de",
        "md": "md",
        "va": "va",
        "nc": "nc",
        "sc": "sc",
        "ga": "ga",
        "fl": "fl",
        "mi": "mi",
        "oh": "oh",
        "tn": "tn",
        "ky": "ky",
        "wv": "wv",
        "me": "me",
        "vt": "vt",
        "nh": "nh",
        "in": "in",
        "il": "il",
        "al": "al",
        "ms": "ms",
        "la": "la",
        "ar": "ar",
        "tx": "tx",
    }
    return mapping.get(t, t)


def _ordering_clearly_indicated(extracted: PairExtraction) -> bool:
    if extracted.ordering_indicated is True:
        return True
    labels = []
    for p in extracted.performances[:2]:
        if _nonempty(p.order_label):
            labels.append(p.order_label.strip().lower())
    # Heuristics: if there are two labels and they indicate first/second or 1/2
    if len(labels) >= 2:
        first_like = any(any(k in labels[0] for k in ["first", "1", "one"]) for _ in [0])
        second_like = any(any(k in labels[1] for k in ["second", "2", "two"]) for _ in [0])
        if first_like and second_like:
            return True
    return False


def _build_schedule_claim(perf: Performance) -> str:
    return (
        f"The Matt Rife 2026 Stay Golden World Tour schedule lists a performance on {perf.date} "
        f"at {perf.venue_name} in {perf.city}, {perf.state}."
    )


def _build_capacity_min_claim(perf: Performance) -> str:
    return (
        f"The concert seating capacity at {perf.venue_name} in {perf.city}, {perf.state} is at least 19,000 seats."
    )


def _build_timezone_claim(perf: Performance) -> str:
    return (
        f"{perf.city}, {perf.state} (or the location of {perf.venue_name}) is in the Eastern Time Zone."
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    parent_root,
    extracted: PairExtraction,
) -> None:
    # Create critical top-level node (parallel aggregation)
    criteria_node = evaluator.add_parallel(
        id="Consecutive_Tour_Dates_Meeting_Criteria",
        desc="Identify a pair of consecutive tour dates meeting all constraints; verify all required details for both performances.",
        parent=parent_root,
        critical=True,
    )

    # Exactly two performances provided
    evaluator.add_custom_node(
        result=len(extracted.performances) == 2,
        id="Exactly_Two_Performances_Provided",
        desc="Response provides exactly two performances (no more, no less).",
        parent=criteria_node,
        critical=True,
    )

    # Performance order clearly indicated
    evaluator.add_custom_node(
        result=_ordering_clearly_indicated(extracted),
        id="Performance_Order_Clearly_Indicated",
        desc="Response clearly indicates which is the first vs the second performance.",
        parent=criteria_node,
        critical=True,
    )

    # Prepare first and second performances (use placeholders if missing)
    p1 = extracted.performances[0] if len(extracted.performances) > 0 else Performance()
    p2 = extracted.performances[1] if len(extracted.performances) > 1 else Performance()

    # First performance details (parallel, all critical)
    first_details = evaluator.add_parallel(
        id="First_Performance_Details",
        desc="All required fields for the first performance are provided.",
        parent=criteria_node,
        critical=True,
    )
    evaluator.add_custom_node(_nonempty(p1.date), "First_Date_Provided", "First performance date is provided.", first_details, True)
    evaluator.add_custom_node(_nonempty(p1.venue_name), "First_Venue_Name_Provided", "First performance venue name is provided.", first_details, True)
    evaluator.add_custom_node(_nonempty(p1.city), "First_City_Provided", "First performance city is provided.", first_details, True)
    evaluator.add_custom_node(_nonempty(p1.state), "First_State_Or_Jurisdiction_Provided", "First performance state or jurisdiction is provided.", first_details, True)
    evaluator.add_custom_node(_nonempty(p1.capacity), "First_Concert_Capacity_Provided", "First performance concert capacity is provided.", first_details, True)

    # Second performance details (parallel, all critical)
    second_details = evaluator.add_parallel(
        id="Second_Performance_Details",
        desc="All required fields for the second performance are provided.",
        parent=criteria_node,
        critical=True,
    )
    evaluator.add_custom_node(_nonempty(p2.date), "Second_Date_Provided", "Second performance date is provided.", second_details, True)
    evaluator.add_custom_node(_nonempty(p2.venue_name), "Second_Venue_Name_Provided", "Second performance venue name is provided.", second_details, True)
    evaluator.add_custom_node(_nonempty(p2.city), "Second_City_Provided", "Second performance city is provided.", second_details, True)
    evaluator.add_custom_node(_nonempty(p2.state), "Second_State_Or_Jurisdiction_Provided", "Second performance state or jurisdiction is provided.", second_details, True)
    evaluator.add_custom_node(_nonempty(p2.capacity), "Second_Concert_Capacity_Provided", "Second performance concert capacity is provided.", second_details, True)

    # Tour schedule verification (parallel; verify each performance via cited schedule URLs)
    schedule_verification = evaluator.add_parallel(
        id="Tour_Schedule_Verification",
        desc="Both performances (date+venue) are verified on Matt Rife's 2026 Stay Golden World Tour schedule.",
        parent=criteria_node,
        critical=True,
    )

    # First schedule verification (sequential: require URLs, then verify)
    first_sched_seq = evaluator.add_sequential(
        id="First_Schedule_Verification",
        desc="First performance schedule verification",
        parent=schedule_verification,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(p1.schedule_urls) > 0,
        id="First_Schedule_URLs_Provided",
        desc="First performance schedule URLs are provided.",
        parent=first_sched_seq,
        critical=True,
    )
    first_sched_leaf = evaluator.add_leaf(
        id="First_Schedule_On_Tour",
        desc="First performance is listed on the tour schedule.",
        parent=first_sched_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_schedule_claim(p1),
        node=first_sched_leaf,
        sources=p1.schedule_urls,
        additional_instruction=(
            "Confirm the event appears on Matt Rife's 2026 Stay Golden World Tour schedule or an official/primary listing "
            "that explicitly ties the date and venue to the 2026 tour. Match date, city/state, and venue."
        ),
    )

    # Second schedule verification (sequential: require URLs, then verify)
    second_sched_seq = evaluator.add_sequential(
        id="Second_Schedule_Verification",
        desc="Second performance schedule verification",
        parent=schedule_verification,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(p2.schedule_urls) > 0,
        id="Second_Schedule_URLs_Provided",
        desc="Second performance schedule URLs are provided.",
        parent=second_sched_seq,
        critical=True,
    )
    second_sched_leaf = evaluator.add_leaf(
        id="Second_Schedule_On_Tour",
        desc="Second performance is listed on the tour schedule.",
        parent=second_sched_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_schedule_claim(p2),
        node=second_sched_leaf,
        sources=p2.schedule_urls,
        additional_instruction=(
            "Confirm the event appears on Matt Rife's 2026 Stay Golden World Tour schedule or an official/primary listing "
            "that explicitly ties the date and venue to the 2026 tour. Match date, city/state, and venue."
        ),
    )

    # Consecutive dates (simple verification, relies on provided dates)
    consecutive_leaf = evaluator.add_leaf(
        id="Consecutive_Dates",
        desc="The two tour dates are consecutive calendar days (second is the next day after the first).",
        parent=criteria_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The second performance date ({p2.date}) occurs exactly one calendar day after the first performance date ({p1.date}).",
        node=consecutive_leaf,
        additional_instruction=(
            "Treat month/day/year formats and common textual date formats flexibly. "
            "Account for varying capitalization and punctuation. This is purely a date arithmetic check."
        ),
    )

    # Both venues meet minimum capacity (parallel; verify each via capacity URLs)
    capacity_min_node = evaluator.add_parallel(
        id="Both_Venues_Meet_Minimum_Capacity",
        desc="Both venues have a concert capacity of at least 19,000 seats.",
        parent=criteria_node,
        critical=True,
    )

    first_capacity_seq = evaluator.add_sequential(
        id="First_Capacity_Verification",
        desc="Minimum capacity verification for the first performance",
        parent=capacity_min_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(p1.capacity_urls) > 0,
        id="First_Capacity_URLs_Provided",
        desc="First performance capacity URLs are provided.",
        parent=first_capacity_seq,
        critical=True,
    )
    first_cap_min_leaf = evaluator.add_leaf(
        id="First_Capacity_AtLeast_19000",
        desc="First venue capacity is at least 19,000 seats",
        parent=first_capacity_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_capacity_min_claim(p1),
        node=first_cap_min_leaf,
        sources=p1.capacity_urls,
        additional_instruction=(
            "Look for venue seating capacity specifically for concerts/events. If a range or multiple configurations "
            "are listed, accept as 'at least 19,000' when any configuration meets or exceeds 19,000."
        ),
    )

    second_capacity_seq = evaluator.add_sequential(
        id="Second_Capacity_Verification",
        desc="Minimum capacity verification for the second performance",
        parent=capacity_min_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(p2.capacity_urls) > 0,
        id="Second_Capacity_URLs_Provided",
        desc="Second performance capacity URLs are provided.",
        parent=second_capacity_seq,
        critical=True,
    )
    second_cap_min_leaf = evaluator.add_leaf(
        id="Second_Capacity_AtLeast_19000",
        desc="Second venue capacity is at least 19,000 seats",
        parent=second_capacity_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_capacity_min_claim(p2),
        node=second_cap_min_leaf,
        sources=p2.capacity_urls,
        additional_instruction=(
            "Look for venue seating capacity specifically for concerts/events. If a range or multiple configurations "
            "are listed, accept as 'at least 19,000' when any configuration meets or exceeds 19,000."
        ),
    )

    # Capacity ordering (simple verification; depends on capacity minima verification)
    cap_order_leaf = evaluator.add_leaf(
        id="Capacity_Ordering",
        desc="The second venue's concert capacity is larger than the first venue's concert capacity.",
        parent=criteria_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The second venue's capacity ('{p2.capacity}') is larger than the first venue's capacity ('{p1.capacity}'). "
            "Treat numeric strings flexibly (commas, plus signs, or 'k' suffix). Compare the intended seat counts."
        ),
        node=cap_order_leaf,
        additional_instruction=(
            "Focus on interpreting the two capacity strings as seat counts; minor formatting differences "
            "like commas, '+' signs, or 'k' suffixes should be handled. This is a straightforward numeric comparison."
        ),
        extra_prerequisites=[first_cap_min_leaf, second_cap_min_leaf],
    )

    # Different states or jurisdictions (custom logic)
    diff_states_leaf = evaluator.add_custom_node(
        result=(_nonempty(p1.state) and _nonempty(p2.state) and _normalize_state(p1.state) != _normalize_state(p2.state)),
        id="Different_States_Or_Jurisdictions",
        desc="The two venues are located in different US states or distinct jurisdictions.",
        parent=criteria_node,
        critical=True,
    )

    # Eastern Time Zone verification (parallel; verify each via timezone URLs)
    eastern_node = evaluator.add_parallel(
        id="Eastern_Time_Zone",
        desc="Both venues are located in the Eastern Time Zone.",
        parent=criteria_node,
        critical=True,
    )

    first_tz_seq = evaluator.add_sequential(
        id="First_Eastern_TZ_Verification",
        desc="Eastern Time Zone verification for the first performance",
        parent=eastern_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(p1.timezone_urls) > 0,
        id="First_TimeZone_URLs_Provided",
        desc="First performance timezone URLs are provided.",
        parent=first_tz_seq,
        critical=True,
    )
    first_tz_leaf = evaluator.add_leaf(
        id="First_Eastern_Time_Zone_Verified",
        desc="First venue/location is in the Eastern Time Zone.",
        parent=first_tz_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_timezone_claim(p1),
        node=first_tz_leaf,
        sources=p1.timezone_urls,
        additional_instruction=(
            "Use reliable sources (e.g., city/venue pages or authoritative references) to confirm the location "
            "is in the Eastern Time Zone (ET)."
        ),
    )

    second_tz_seq = evaluator.add_sequential(
        id="Second_Eastern_TZ_Verification",
        desc="Eastern Time Zone verification for the second performance",
        parent=eastern_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(p2.timezone_urls) > 0,
        id="Second_TimeZone_URLs_Provided",
        desc="Second performance timezone URLs are provided.",
        parent=second_tz_seq,
        critical=True,
    )
    second_tz_leaf = evaluator.add_leaf(
        id="Second_Eastern_Time_Zone_Verified",
        desc="Second venue/location is in the Eastern Time Zone.",
        parent=second_tz_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_timezone_claim(p2),
        node=second_tz_leaf,
        sources=p2.timezone_urls,
        additional_instruction=(
            "Use reliable sources (e.g., city/venue pages or authoritative references) to confirm the location "
            "is in the Eastern Time Zone (ET)."
        ),
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
    Evaluate an answer for the Matt Rife 2026 consecutive tour dates criteria.
    """
    # Initialize evaluator (root node is non-critical by design)
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

    # Extract structured information from the answer
    extracted_pair = await evaluator.extract(
        prompt=prompt_extract_pair(),
        template_class=PairExtraction,
        extraction_name="pair_extraction",
    )

    # If the answer lists more than two performances, we will still verify using the first two,
    # but the "Exactly_Two_Performances_Provided" node will correctly fail (per rubric).
    if len(extracted_pair.performances) > 2:
        extracted_pair.performances = extracted_pair.performances[:2]

    # Build the verification tree and run verifications
    await build_verification_tree(evaluator, root, extracted_pair)

    # Return standard summary
    return evaluator.get_summary()