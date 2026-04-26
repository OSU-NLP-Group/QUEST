import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dwts2026_first_three_venues"
TASK_DESCRIPTION = (
    "For the Dancing with the Stars Live 2026 tour, identify the first three venues in chronological "
    "order of performance. For each venue, provide: (1) the performance date, (2) the city and state location, "
    "(3) the complete venue name, and (4) the seating capacity. Additionally, confirm the tour's start date, "
    "opening location (city and state), and provide context about the tour's total number of shows and when it concludes."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Stop(BaseModel):
    date: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue_name: Optional[str] = None
    capacity: Optional[str] = None
    # URLs explicitly mentioned in the answer that support this stop's info
    sources: List[str] = Field(default_factory=list)


class FirstThreeStopsExtraction(BaseModel):
    # First three stops as presented in the answer and claimed to be chronological
    stops: List[Stop] = Field(default_factory=list)
    # General schedule or official DWTS tour URLs referenced in the answer
    schedule_sources: List[str] = Field(default_factory=list)


class TourContextExtraction(BaseModel):
    # Overall tour context fields explicitly mentioned in the answer
    start_date: Optional[str] = None
    opening_city: Optional[str] = None
    opening_state: Optional[str] = None
    total_shows: Optional[str] = None  # keep as string to allow "about 60" etc.
    tour_end: Optional[str] = None     # date or timeframe (e.g., "late April 2026")
    # URLs used for overall context claims
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_first_three_stops() -> str:
    return """
    Extract the first three Dancing with the Stars (DWTS) Live 2026 tour stops as they are presented in the answer.
    The extracted list must contain at most three items and represent the first three stops in chronological order as claimed by the answer.

    For each stop, extract:
    - date: The performance date string exactly as written in the answer (e.g., "Jan 10, 2026" or "January 10, 2026").
    - city: The city name for that stop.
    - state: The state (use the abbreviation if the answer uses it, otherwise the full state name).
    - venue_name: The full venue name as stated (e.g., "XYZ Theater", "ABC Arena").
    - capacity: The seating capacity stated for the venue (if provided). Keep it as a string, do not convert to a number.
    - sources: A list of all URLs that the answer explicitly cites for this stop. Only include actual URLs present in the answer. If none are given, use an empty list.

    Also extract:
    - schedule_sources: A list of URL(s) explicitly cited in the answer that provide the overall tour schedule (e.g., official DWTS site schedule page, official tour announcements). If none are given, return an empty list.

    Return a JSON object with fields:
    {
      "stops": [ {stop1}, {stop2}, {stop3} ],
      "schedule_sources": [...]
    }

    Only include URLs that are explicitly present in the answer text. Do not fabricate URLs.
    If some required fields are missing for a stop, set them to null, and set sources to an empty list.
    """


def prompt_extract_tour_context() -> str:
    return """
    Extract the overall tour context for DWTS Live 2026 as stated in the answer.
    Specifically extract:
    - start_date: The tour's opening/start date (string as in the answer).
    - opening_city: The city of the opening stop.
    - opening_state: The state of the opening stop.
    - total_shows: The total number of shows for the tour (string as in the answer, e.g., "62" or "about 60").
    - tour_end: When the tour concludes (may be an exact date or a timeframe string, e.g., "April 28, 2026" or "late April 2026").
    - sources: All URLs explicitly cited in the answer for these overall tour context claims. Only include actual URLs from the answer.

    Return a JSON object with exactly these fields. If any field is not present in the answer, set it to null (or empty list for sources).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        if lst:
            merged.extend(lst)
    return _dedupe_urls(merged)


def _numbers_in_text(text: str) -> List[int]:
    if not text:
        return []
    nums = re.findall(r"(\d{1,3}(?:,\d{3})+|\d+)", text)
    out: List[int] = []
    for n in nums:
        try:
            out.append(int(n.replace(",", "")))
        except Exception:
            continue
    return out


def _capacities_within_range(stops: List[Stop], lower: int = 2600, upper: int = 8500) -> bool:
    """
    Non-critical consistency check: for each stop that has a numeric capacity value(s), ensure
    all stated numeric figures fall within the approximate range [lower, upper].
    If a stop's capacity is missing or non-numeric, ignore that stop in this check.
    """
    for s in stops:
        if not s.capacity:
            continue
        nums = _numbers_in_text(s.capacity)
        if not nums:
            # ignore non-numeric capacity descriptions (e.g., "mid-size theater")
            continue
        # If any numeric figure falls outside the range, consider it inconsistent
        for val in nums:
            if val < lower or val > upper:
                return False
    return True


def _stop_label(idx: int) -> str:
    # idx is 0-based
    return ["first", "second", "third"][idx] if idx < 3 else f"#{idx+1}"


def _compose_stop_date_claim(s: Stop, idx: int) -> Optional[str]:
    if not s.date:
        return None
    # Try to anchor using available context
    if s.venue_name and s.city and s.state:
        return f"On {s.date}, DWTS Live 2026 performs at {s.venue_name} in {s.city}, {s.state}."
    if s.city and s.state:
        return f"On {s.date}, DWTS Live 2026 performs in {s.city}, {s.state}."
    return f"The DWTS Live 2026 tour has a performance on {s.date}."


def _compose_stop_city_state_claim(s: Stop, idx: int) -> Optional[str]:
    if not s.city or not s.state:
        return None
    if s.date:
        return f"The DWTS Live 2026 performance on {s.date} takes place in {s.city}, {s.state}."
    if s.venue_name:
        return f"The DWTS Live 2026 performance at {s.venue_name} takes place in {s.city}, {s.state}."
    return f"The DWTS Live 2026 tour stop is in {s.city}, {s.state}."


def _compose_stop_venue_name_claim(s: Stop, idx: int) -> Optional[str]:
    if not s.venue_name:
        return None
    if s.date and s.city and s.state:
        return f"The venue for the DWTS Live 2026 performance on {s.date} in {s.city}, {s.state} is '{s.venue_name}'."
    if s.city and s.state:
        return f"The DWTS Live 2026 stop in {s.city}, {s.state} is at '{s.venue_name}'."
    return f"The venue name for the DWTS Live 2026 stop is '{s.venue_name}'."


def _compose_stop_capacity_claim(s: Stop, idx: int) -> Optional[str]:
    if not s.capacity or not s.venue_name:
        return None
    return f"The seating capacity of '{s.venue_name}' is approximately {s.capacity}."


def _compose_correctness_claim(stops: List[Stop]) -> Optional[str]:
    if len(stops) < 3:
        return None
    s1, s2, s3 = stops[0], stops[1], stops[2]
    # Build a concise, explicit list claim for schedule verification
    def fmt_stop(s: Stop) -> str:
        parts = []
        if s.date:
            parts.append(s.date)
        if s.venue_name:
            parts.append(s.venue_name)
        loc = ", ".join([p for p in [s.city or None, s.state or None] if p])
        if loc:
            parts.append(loc)
        return " — ".join(parts) if parts else "—"
    return (
        "According to the official or authoritative DWTS Live 2026 tour schedule, "
        "the first three stops in chronological order are:\n"
        f"1) {fmt_stop(s1)}\n"
        f"2) {fmt_stop(s2)}\n"
        f"3) {fmt_stop(s3)}\n"
        "Verify that the schedule shows exactly these as the first three dates in order."
    )


def _compose_chronological_order_claim(stops: List[Stop]) -> Optional[str]:
    if len(stops) < 3:
        return None
    d1 = stops[0].date or ""
    d2 = stops[1].date or ""
    d3 = stops[2].date or ""
    return f"These three dates are in ascending chronological order: {d1}, then {d2}, then {d3}."


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def _verify_stop(
    evaluator: Evaluator,
    parent_node,
    stop: Stop,
    idx: int,
    schedule_sources: List[str],
):
    """
    Build the subtree for a single stop with four leaf verifications:
    - Date
    - City & State
    - Venue Name
    - Capacity
    """
    stop_num = idx + 1
    stop_node = evaluator.add_parallel(
        id=f"Stop_{stop_num}",
        desc=f"Complete information for the {stop_num}rd tour performance venue" if stop_num == 3 else
             (f"Complete information for the {stop_num}nd tour performance venue" if stop_num == 2
              else "Complete information for the 1st tour performance venue"),
        parent=parent_node,
        critical=False,
    )

    # Prepare sources for this stop (union of stop-specific and schedule sources)
    stop_sources = _merge_sources(stop.sources, schedule_sources)

    # 1) Date
    date_claim = _compose_stop_date_claim(stop, idx)
    if date_claim and stop_sources:
        leaf = evaluator.add_leaf(
            id=f"Stop_{stop_num}_Date",
            desc=f"Provide the performance date for the { _stop_label(idx) } venue",
            parent=stop_node,
            critical=True,
        )
        await evaluator.verify(
            claim=date_claim,
            node=leaf,
            sources=stop_sources,
            additional_instruction=(
                "Verify that the specified date corresponds to a DWTS Live 2026 tour performance for the indicated stop. "
                "Use schedule pages or authoritative venue/ticketing pages. Allow minor formatting differences."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Stop_{stop_num}_Date",
            desc=f"Provide the performance date for the { _stop_label(idx) } venue",
            parent=stop_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # 2) City & State
    cs_claim = _compose_stop_city_state_claim(stop, idx)
    if cs_claim and stop_sources:
        leaf = evaluator.add_leaf(
            id=f"Stop_{stop_num}_City_State",
            desc=f"Provide the city and state for the { _stop_label(idx) } venue",
            parent=stop_node,
            critical=True,
        )
        await evaluator.verify(
            claim=cs_claim,
            node=leaf,
            sources=stop_sources,
            additional_instruction=(
                "Verify the city and state for the specified DWTS Live 2026 stop (e.g., on the given date). "
                "Allow common variations (state abbreviation vs full name)."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Stop_{stop_num}_City_State",
            desc=f"Provide the city and state for the { _stop_label(idx) } venue",
            parent=stop_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # 3) Venue Name
    vn_claim = _compose_stop_venue_name_claim(stop, idx)
    if vn_claim and stop_sources:
        leaf = evaluator.add_leaf(
            id=f"Stop_{stop_num}_Venue_Name",
            desc=f"Provide the complete venue name for the { _stop_label(idx) } venue",
            parent=stop_node,
            critical=True,
        )
        await evaluator.verify(
            claim=vn_claim,
            node=leaf,
            sources=stop_sources,
            additional_instruction=(
                "Check that the venue name for the specified DWTS Live 2026 stop matches the cited sources. "
                "Allow minor common variants (e.g., Theatre vs Theater, Centre vs Center) if clearly the same venue."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Stop_{stop_num}_Venue_Name",
            desc=f"Provide the complete venue name for the { _stop_label(idx) } venue",
            parent=stop_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # 4) Capacity
    cap_claim = _compose_stop_capacity_claim(stop, idx)
    if cap_claim and stop_sources:
        leaf = evaluator.add_leaf(
            id=f"Stop_{stop_num}_Capacity",
            desc=f"Provide the seating capacity for the { _stop_label(idx) } venue",
            parent=stop_node,
            critical=True,
        )
        await evaluator.verify(
            claim=cap_claim,
            node=leaf,
            sources=stop_sources,
            additional_instruction=(
                "Verify the seating capacity for the venue from reliable sources (venue official site, reputable ticketing, "
                "Wikipedia with citations). Accept approximate or range values if consistent."
            ),
        )
    else:
        evaluator.add_leaf(
            id=f"Stop_{stop_num}_Capacity",
            desc=f"Provide the seating capacity for the { _stop_label(idx) } venue",
            parent=stop_node,
            critical=True,
            score=0.0,
            status="failed",
        )


async def _verify_first_three_correctness(
    evaluator: Evaluator,
    parent_node,
    stops: List[Stop],
    schedule_sources: List[str],
):
    claim = _compose_correctness_claim(stops)
    if claim and schedule_sources:
        node = evaluator.add_leaf(
            id="First_Three_Stop_Correctness",
            desc="The selected venues correspond to tour stops #1–#3 per the schedule constraints (including the specified opening, second, and third stops/venues)",
            parent=parent_node,
            critical=True,
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=schedule_sources,
            additional_instruction=(
                "Use the official or authoritative schedule to confirm that the first three tour dates "
                "match the three stops listed (dates, cities, venues) and are indeed the first three in order."
            ),
        )
    else:
        evaluator.add_leaf(
            id="First_Three_Stop_Correctness",
            desc="The selected venues correspond to tour stops #1–#3 per the schedule constraints (including the specified opening, second, and third stops/venues)",
            parent=parent_node,
            critical=True,
            score=0.0,
            status="failed",
        )


async def _verify_chronological_order(
    evaluator: Evaluator,
    parent_node,
    stops: List[Stop],
):
    claim = _compose_chronological_order_claim(stops)
    if claim:
        node = evaluator.add_leaf(
            id="Chronological_Order_Presentation",
            desc="The three venues are presented in chronological order of performance date (stop 1, then stop 2, then stop 3)",
            parent=parent_node,
            critical=True,
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            additional_instruction=(
                "Pure logical check: determine if the three given dates are in ascending chronological order. "
                "Accept standard date format variations."
            ),
        )
    else:
        evaluator.add_leaf(
            id="Chronological_Order_Presentation",
            desc="The three venues are presented in chronological order of performance date (stop 1, then stop 2, then stop 3)",
            parent=parent_node,
            critical=True,
            score=0.0,
            status="failed",
        )


async def _verify_tour_scale_context(
    evaluator: Evaluator,
    parent_node,
    ctx: TourContextExtraction,
    schedule_sources: List[str],
):
    # Parent node for tour scale context
    context_node = evaluator.add_parallel(
        id="Tour_Scale_Context",
        desc="Provide context about the tour's total number of shows and when it concludes",
        parent=parent_node,
        critical=False,  # adjusted to allow a non-critical child below
    )

    # Combined sources for context verifications
    ctx_sources = _merge_sources(ctx.sources, schedule_sources)

    # Total number of shows
    if ctx.total_shows and ctx_sources:
        node = evaluator.add_leaf(
            id="Total_Tour_Shows",
            desc="Provide the tour's total number of shows",
            parent=context_node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The DWTS Live 2026 tour has a total of {ctx.total_shows} shows.",
            node=node,
            sources=ctx_sources,
            additional_instruction=(
                "Verify the total number of tour performances as stated. Use the official schedule or authoritative announcements. "
                "If the answer states an approximate number, ensure sources reasonably support it."
            ),
        )
    else:
        evaluator.add_leaf(
            id="Total_Tour_Shows",
            desc="Provide the tour's total number of shows",
            parent=context_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # Tour end timeframe/date
    if ctx.tour_end and ctx_sources:
        node = evaluator.add_leaf(
            id="Tour_End_Timeline",
            desc="Provide when the tour concludes (conclusion timeframe)",
            parent=context_node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The DWTS Live 2026 tour concludes on {ctx.tour_end}.",
            node=node,
            sources=ctx_sources,
            additional_instruction=(
                "Verify the final show date or stated end timeframe of the tour using the cited schedule or authoritative sources. "
                "Allow minor phrasing differences (e.g., 'late April 2026' vs exact date) if consistent."
            ),
        )
    else:
        evaluator.add_leaf(
            id="Tour_End_Timeline",
            desc="Provide when the tour concludes (conclusion timeframe)",
            parent=context_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # Capacity range consistency (non-critical logical check without external verification)
    cap_ok = _capacities_within_range(
        # We will try to fetch the first three stops from the extraction added to custom info later.
        # For isolation, caller should pass the stops list through parent scope. Here, we will defer;
        # To keep function self-contained, we use a placeholder False and let caller add this node.
        # However, for clarity and completeness, we will not add it here and instead add it outside
        # after stops are available. To maintain rubric hierarchy, we add it here using a custom node,
        # with a default True that caller can override via an explicit custom info entry. To avoid confusion,
        # we implement it outside. (Implementation below in main flow.)
        []
    )

    # We won't add the capacity consistency node here; it will be added in the main flow
    # after we have access to the extracted stops. This comment explains the design choice.


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict:
    """
    Evaluate an answer for the DWTS Live 2026 first three venues task.
    """
    # Initialize evaluator
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

    # Optional note: We adjust some critical flags to satisfy framework constraints
    evaluator.add_custom_info(
        info={
            "note": "Adjusted some critical flags to satisfy framework constraints: "
                    "Root and 'Tour_Scale_Context' set to non-critical to allow mixed critical children. "
                    "'Stops_1_to_3' kept non-critical to allow partial credit per stop, "
                    "while individual field leaves remain critical."
        },
        info_type="implementation_notes",
    )

    # Extract structured data
    stops_extraction, ctx_extraction = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_first_three_stops(),
            template_class=FirstThreeStopsExtraction,
            extraction_name="first_three_stops",
        ),
        evaluator.extract(
            prompt=prompt_extract_tour_context(),
            template_class=TourContextExtraction,
            extraction_name="tour_context",
        ),
    )

    # Build top-level rubric node to group everything (matches rubric name)
    dwts_node = evaluator.add_parallel(
        id="DWTS_2026_Tour_First_Three_Venues",
        desc="Identify the first three DWTS Live 2026 tour venues (chronological) with required per-venue details and overall tour scale context",
        parent=root,
        critical=False,  # adjusted to allow mixed children criticality underneath
    )

    # Stops 1–3 subtree
    stops_main = evaluator.add_parallel(
        id="Stops_1_to_3",
        desc="Provide complete information for the first three tour stops",
        parent=dwts_node,
        critical=False,  # allow partial credit across stops
    )

    # Normalize stops to exactly 3 (pad with empty if fewer)
    stops: List[Stop] = list(stops_extraction.stops[:3])
    while len(stops) < 3:
        stops.append(Stop())

    schedule_sources = _dedupe_urls(stops_extraction.schedule_sources)

    # Verify each stop
    for i in range(3):
        await _verify_stop(
            evaluator=evaluator,
            parent_node=stops_main,
            stop=stops[i],
            idx=i,
            schedule_sources=schedule_sources,
        )

    # First three correctness leaf
    await _verify_first_three_correctness(
        evaluator=evaluator,
        parent_node=dwts_node,
        stops=stops,
        schedule_sources=schedule_sources,
    )

    # Chronological order presentation leaf
    await _verify_chronological_order(
        evaluator=evaluator,
        parent_node=dwts_node,
        stops=stops,
    )

    # Tour scale context subtree (total shows + end timeline)
    await _verify_tour_scale_context(
        evaluator=evaluator,
        parent_node=dwts_node,
        ctx=ctx_extraction,
        schedule_sources=schedule_sources,
    )

    # Add Capacity_Range_Consistency under Tour_Scale_Context (non-critical, logical check)
    # We need to locate the parent "Tour_Scale_Context" node that we created above.
    tour_scale_node = evaluator.find_node("Tour_Scale_Context")
    if tour_scale_node is None:
        # Fallback: add it under dwts_node if something unexpected happened
        tour_scale_node = evaluator.add_parallel(
            id="Tour_Scale_Context",
            desc="Provide context about the tour's total number of shows and when it concludes",
            parent=dwts_node,
            critical=False,
        )

    capacity_consistent = _capacities_within_range(stops, lower=2600, upper=8500)
    evaluator.add_custom_node(
        result=capacity_consistent,
        id="Capacity_Range_Consistency",
        desc="Reported venue capacities are consistent with the stated approximate tour-wide capacity range (~2,600 to 8,500)",
        parent=tour_scale_node,
        critical=False,  # non-critical per rubric
    )

    # Return evaluation summary
    return evaluator.get_summary()