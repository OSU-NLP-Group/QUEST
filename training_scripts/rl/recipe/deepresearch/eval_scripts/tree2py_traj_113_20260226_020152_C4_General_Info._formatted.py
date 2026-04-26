import asyncio
import logging
import re
from datetime import date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gillette_concerts_2026"
TASK_DESCRIPTION = """Gillette Stadium in Foxborough, Massachusetts is hosting two major music tours in late summer 2026: BTS World Tour 'Arirang' in August and Bruno Mars 'The Romantic Tour' in September. For someone planning to attend both concert series, provide the following information:

1. The exact dates when BTS World Tour 'Arirang' will perform at Gillette Stadium
2. The start time for the BTS concerts
3. The exact dates when Bruno Mars 'The Romantic Tour' will perform at Gillette Stadium
4. The start time for the Bruno Mars concerts
5. The city and state where Gillette Stadium is located
6. The approximate concert seating capacity of Gillette Stadium
7. (Bonus) Calculate the time interval between the first BTS concert date and the first Bruno Mars concert date
"""

# Ground-truth expectations embedded in rubric descriptions
EXPECTED_BTS_DATES_TEXT = "August 5-6, 2026"
EXPECTED_BTS_START_TIME = "8:00 PM"
EXPECTED_BRUNO_DATES_TEXT = "September 5-6, 2026"
EXPECTED_BRUNO_START_TIME = "7:00 PM"
EXPECTED_STADIUM_CITY = "Foxborough"
EXPECTED_STADIUM_STATE = "Massachusetts"
EXPECTED_STADIUM_CAPACITY_APPROX = "approximately 65,878"
EXPECTED_TIME_GAP_DAYS = 31  # between Aug 5 and Sep 5, 2026


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TourDates(BaseModel):
    exact_dates_text: Optional[str] = None  # e.g., "August 5-6, 2026"
    start_time: Optional[str] = None        # e.g., "8:00 PM"
    source_urls: List[str] = Field(default_factory=list)


class StadiumInfo(BaseModel):
    city: Optional[str] = None              # e.g., "Foxborough"
    state: Optional[str] = None             # e.g., "Massachusetts"
    concert_capacity: Optional[str] = None  # e.g., "approximately 65,878"
    source_urls: List[str] = Field(default_factory=list)


class ConcertPlanExtraction(BaseModel):
    bts: Optional[TourDates] = None
    bruno: Optional[TourDates] = None
    stadium: Optional[StadiumInfo] = None
    # Bonus field: if the answer explicitly states the time gap, capture it; if not, null
    time_gap_text: Optional[str] = None     # e.g., "31 days", "1 month", "about one month"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concert_plan() -> str:
    return """
    Extract structured information as it appears in the answer, focusing on Gillette Stadium events in late summer 2026.

    Return a JSON object with the following fields:

    - bts: {
        exact_dates_text: the exact dates text (as written) for BTS World Tour 'Arirang' at Gillette Stadium, e.g., "August 5-6, 2026"; if absent, null
        start_time: the stated start time for BTS concerts at Gillette Stadium, e.g., "8:00 PM"; if absent, null
        source_urls: array of URLs (explicitly present in the answer) that support BTS dates and/or start time at Gillette Stadium; if none, return []
      }

    - bruno: {
        exact_dates_text: the exact dates text (as written) for Bruno Mars 'The Romantic Tour' at Gillette Stadium, e.g., "September 5-6, 2026"; if absent, null
        start_time: the stated start time for Bruno Mars concerts at Gillette Stadium, e.g., "7:00 PM"; if absent, null
        source_urls: array of URLs (explicitly present in the answer) that support Bruno Mars dates and/or start time at Gillette Stadium; if none, return []
      }

    - stadium: {
        city: the city name for Gillette Stadium (as written), e.g., "Foxborough"; if absent, null
        state: the state name for Gillette Stadium (as written), e.g., "Massachusetts"; if absent, null
        concert_capacity: the approximate concert seating capacity of Gillette Stadium (as written), e.g., "approximately 65,878"; if absent, null
        source_urls: array of URLs (explicitly present in the answer) that support city/state/capacity; if none, return []
      }

    - time_gap_text: if the answer explicitly states the time interval between the first BTS concert date and the first Bruno Mars concert date (e.g., "31 days", "1 month"), capture it verbatim; else null.

    IMPORTANT:
    - Extract ONLY what is explicitly present in the answer; do not invent.
    - For URLs, include only actual URLs present in the answer (including markdown links); if the answer mentions a site without URL, ignore it.
    - Keep original formatting for dates/time/capacity text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _month_to_number(m: str) -> Optional[int]:
    m = m.strip().lower()
    return _MONTHS.get(m)


def parse_first_date(date_text: Optional[str]) -> Optional[date]:
    """
    Parse the first date from strings like:
    - "August 5-6, 2026"
    - "Aug 5 & 6, 2026"
    - "5-6 August 2026"
    Returns a date object for the first day if parsable; else None.
    """
    if not date_text or not isinstance(date_text, str):
        return None

    s = date_text.strip()

    # Pattern A: Month first
    pat_a = re.compile(
        r"(?i)\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b[^\d]{0,3}(\d{1,2})(?:\s*[-–—/&,\s]*(\d{1,2}))?(?:[^\d]{0,3})(\d{4})"
    )
    m = pat_a.search(s)
    if m:
        month_name, d1, _, y = m.groups()
        month_num = _month_to_number(month_name or "")
        if month_num:
            try:
                return date(int(y), month_num, int(d1))
            except Exception:
                return None

    # Pattern B: Day first
    pat_b = re.compile(
        r"(?i)\b(\d{1,2})(?:\s*[-–—/&,\s]*(\d{1,2}))?\s*(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b[^\d]{0,3}(\d{4})"
    )
    m2 = pat_b.search(s)
    if m2:
        d1, _, month_name, y = m2.groups()
        month_num = _month_to_number(month_name or "")
        if month_num:
            try:
                return date(int(y), month_num, int(d1))
            except Exception:
                return None

    return None


def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_bts_section(
    evaluator: Evaluator,
    parent_node,
    extracted: ConcertPlanExtraction,
) -> None:
    bts = extracted.bts or TourDates()

    # BTS Dates group
    bts_dates_group = evaluator.add_parallel(
        id="bts_dates_main",
        desc="BTS dates verification group",
        parent=parent_node,
        critical=False
    )
    # existence node
    bts_dates_exists = evaluator.add_custom_node(
        result=bool(bts.exact_dates_text) and len(_non_empty_urls(bts.source_urls)) > 0,
        id="bts_dates_exists",
        desc="BTS dates and sources are provided in the answer",
        parent=bts_dates_group,
        critical=True
    )
    # equality to expected (answer-text check)
    bts_dates_expected_leaf = evaluator.add_leaf(
        id="BTS_Concert_Dates",
        desc="Correctly identifies the dates for BTS World Tour 'Arirang' concerts at Gillette Stadium (August 5-6, 2026)",
        parent=bts_dates_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"BTS World Tour 'Arirang' at Gillette Stadium is scheduled on {EXPECTED_BTS_DATES_TEXT}.",
        node=bts_dates_expected_leaf,
        additional_instruction="Judge solely against the answer text; accept minor format differences (e.g., 'Aug' vs 'August', separators like '-', '&', ',')."
    )
    # source support for the dates provided in the answer
    bts_dates_source_leaf = evaluator.add_leaf(
        id="BTS_Concert_Dates_Source",
        desc="BTS dates at Gillette Stadium are supported by cited sources",
        parent=bts_dates_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"BTS World Tour 'Arirang' will perform at Gillette Stadium on {bts.exact_dates_text or ''}.",
        node=bts_dates_source_leaf,
        sources=_non_empty_urls(bts.source_urls),
        additional_instruction="Confirm the event page or official source mentions two nights at Gillette Stadium in Foxborough, Massachusetts on the stated dates; allow minor text variations."
    )

    # BTS Start Time group
    bts_time_group = evaluator.add_parallel(
        id="bts_time_main",
        desc="BTS start time verification group",
        parent=parent_node,
        critical=False
    )
    bts_time_exists = evaluator.add_custom_node(
        result=bool(bts.start_time) and len(_non_empty_urls(bts.source_urls)) > 0,
        id="bts_time_exists",
        desc="BTS start time and sources are provided in the answer",
        parent=bts_time_group,
        critical=True
    )
    bts_time_expected_leaf = evaluator.add_leaf(
        id="BTS_Start_Time",
        desc="Correctly provides the start time for BTS concerts (8:00 PM)",
        parent=bts_time_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The BTS concerts at Gillette Stadium start at {EXPECTED_BTS_START_TIME}.",
        node=bts_time_expected_leaf,
        additional_instruction="Judge against the answer text; allow reasonable variants (e.g., '8 PM')."
    )
    bts_time_source_leaf = evaluator.add_leaf(
        id="BTS_Start_Time_Source",
        desc="BTS start time at Gillette Stadium is supported by cited sources",
        parent=bts_time_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The BTS concerts at Gillette Stadium start at {bts.start_time or ''}.",
        node=bts_time_source_leaf,
        sources=_non_empty_urls(bts.source_urls),
        additional_instruction="Confirm the start time on the provided event/source pages; minor format differences acceptable."
    )


async def verify_bruno_section(
    evaluator: Evaluator,
    parent_node,
    extracted: ConcertPlanExtraction,
) -> None:
    bruno = extracted.bruno or TourDates()

    # Bruno Mars Dates group
    bruno_dates_group = evaluator.add_parallel(
        id="bruno_dates_main",
        desc="Bruno Mars dates verification group",
        parent=parent_node,
        critical=False
    )
    bruno_dates_exists = evaluator.add_custom_node(
        result=bool(bruno.exact_dates_text) and len(_non_empty_urls(bruno.source_urls)) > 0,
        id="bruno_dates_exists",
        desc="Bruno Mars dates and sources are provided in the answer",
        parent=bruno_dates_group,
        critical=True
    )
    bruno_dates_expected_leaf = evaluator.add_leaf(
        id="Bruno_Mars_Concert_Dates",
        desc="Correctly identifies the dates for Bruno Mars 'The Romantic Tour' concerts at Gillette Stadium (September 5-6, 2026)",
        parent=bruno_dates_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Bruno Mars 'The Romantic Tour' at Gillette Stadium is scheduled on {EXPECTED_BRUNO_DATES_TEXT}.",
        node=bruno_dates_expected_leaf,
        additional_instruction="Judge solely against the answer text; accept minor format differences."
    )
    bruno_dates_source_leaf = evaluator.add_leaf(
        id="Bruno_Mars_Concert_Dates_Source",
        desc="Bruno Mars dates at Gillette Stadium are supported by cited sources",
        parent=bruno_dates_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Bruno Mars 'The Romantic Tour' will perform at Gillette Stadium on {bruno.exact_dates_text or ''}.",
        node=bruno_dates_source_leaf,
        sources=_non_empty_urls(bruno.source_urls),
        additional_instruction="Confirm the event page or official source lists the stated dates at Gillette Stadium."
    )

    # Bruno Mars Start Time group
    bruno_time_group = evaluator.add_parallel(
        id="bruno_time_main",
        desc="Bruno Mars start time verification group",
        parent=parent_node,
        critical=False
    )
    bruno_time_exists = evaluator.add_custom_node(
        result=bool(bruno.start_time) and len(_non_empty_urls(bruno.source_urls)) > 0,
        id="bruno_time_exists",
        desc="Bruno Mars start time and sources are provided in the answer",
        parent=bruno_time_group,
        critical=True
    )
    bruno_time_expected_leaf = evaluator.add_leaf(
        id="Bruno_Mars_Start_Time",
        desc="Correctly provides the start time for Bruno Mars concerts (7:00 PM)",
        parent=bruno_time_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Bruno Mars concerts at Gillette Stadium start at {EXPECTED_BRUNO_START_TIME}.",
        node=bruno_time_expected_leaf,
        additional_instruction="Judge against the answer text; allow reasonable variants (e.g., '7 PM')."
    )
    bruno_time_source_leaf = evaluator.add_leaf(
        id="Bruno_Mars_Start_Time_Source",
        desc="Bruno Mars start time at Gillette Stadium is supported by cited sources",
        parent=bruno_time_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Bruno Mars concerts at Gillette Stadium start at {bruno.start_time or ''}.",
        node=bruno_time_source_leaf,
        sources=_non_empty_urls(bruno.source_urls),
        additional_instruction="Confirm start time on cited event/source pages; minor format differences acceptable."
    )


async def verify_stadium_location_and_capacity(
    evaluator: Evaluator,
    parent_node,
    extracted: ConcertPlanExtraction,
) -> None:
    stadium = extracted.stadium or StadiumInfo()
    stadium_sources = _non_empty_urls(stadium.source_urls)

    # Stadium Location group (city/state)
    stad_loc_group = evaluator.add_parallel(
        id="stad_location_main",
        desc="Stadium city/state verification group",
        parent=parent_node,
        critical=False
    )
    stad_loc_exists = evaluator.add_custom_node(
        result=bool(stadium.city) and bool(stadium.state) and len(stadium_sources) > 0,
        id="stad_location_exists",
        desc="Stadium city/state and sources are provided in the answer",
        parent=stad_loc_group,
        critical=True
    )
    stad_city_expected_leaf = evaluator.add_leaf(
        id="Stadium_City",
        desc="Correctly identifies the city where Gillette Stadium is located (Foxborough)",
        parent=stad_loc_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Gillette Stadium is located in {EXPECTED_STADIUM_CITY}.",
        node=stad_city_expected_leaf,
        additional_instruction="Judge against the answer text; allow minor variations or qualifiers."
    )
    stad_city_source_leaf = evaluator.add_leaf(
        id="Stadium_City_Source",
        desc="Stadium city is supported by cited sources",
        parent=stad_loc_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Gillette Stadium is located in the city of {stadium.city or ''}.",
        node=stad_city_source_leaf,
        sources=stadium_sources,
        additional_instruction="Confirm the city on official or authoritative sources; minor text differences acceptable."
    )
    stad_state_expected_leaf = evaluator.add_leaf(
        id="Stadium_State",
        desc="Correctly identifies the state where Gillette Stadium is located (Massachusetts)",
        parent=stad_loc_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Gillette Stadium is located in {EXPECTED_STADIUM_STATE}.",
        node=stad_state_expected_leaf,
        additional_instruction="Judge against the answer text; allow minor variations."
    )
    stad_state_source_leaf = evaluator.add_leaf(
        id="Stadium_State_Source",
        desc="Stadium state is supported by cited sources",
        parent=stad_loc_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Gillette Stadium is located in the state of {stadium.state or ''}.",
        node=stad_state_source_leaf,
        sources=stadium_sources,
        additional_instruction="Confirm the state on official or authoritative sources; minor text differences acceptable."
    )

    # Stadium Capacity group
    stad_cap_group = evaluator.add_parallel(
        id="stad_capacity_main",
        desc="Stadium concert capacity verification group",
        parent=parent_node,
        critical=False
    )
    stad_cap_exists = evaluator.add_custom_node(
        result=bool(stadium.concert_capacity) and len(stadium_sources) > 0,
        id="stad_capacity_exists",
        desc="Stadium concert capacity and sources are provided in the answer",
        parent=stad_cap_group,
        critical=True
    )
    stad_capacity_expected_leaf = evaluator.add_leaf(
        id="Stadium_Concert_Capacity",
        desc="Correctly provides the approximate concert capacity of Gillette Stadium (approximately 65,878)",
        parent=stad_cap_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The concert seating capacity of Gillette Stadium is {EXPECTED_STADIUM_CAPACITY_APPROX}.",
        node=stad_capacity_expected_leaf,
        additional_instruction="Judge against the answer text; accept reasonable approximations (e.g., '~65k', 'around 66k') as equivalent to approximately 65,878."
    )
    stad_capacity_source_leaf = evaluator.add_leaf(
        id="Stadium_Concert_Capacity_Source",
        desc="Stadium concert capacity is supported by cited sources",
        parent=stad_cap_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Gillette Stadium's concert seating capacity is {stadium.concert_capacity or ''}.",
        node=stad_capacity_source_leaf,
        sources=stadium_sources,
        additional_instruction="Confirm the approximate concert capacity on authoritative sources; accept minor rounding differences."
    )


async def verify_time_gap_bonus(
    evaluator: Evaluator,
    parent_node,
    extracted: ConcertPlanExtraction,
) -> None:
    # Bonus group for time gap
    time_gap_group = evaluator.add_parallel(
        id="time_gap_main",
        desc="Time interval (bonus) verification group",
        parent=parent_node,
        critical=False
    )

    # Existence check: the answer actually provides a time gap statement
    time_gap_exists = evaluator.add_custom_node(
        result=bool(extracted.time_gap_text),
        id="time_gap_provided",
        desc="Answer provides a stated time interval between first BTS and first Bruno Mars dates",
        parent=time_gap_group,
        critical=True  # gate verification; if not provided, skip
    )

    # Compute first dates from extracted texts for additional instruction
    bts_first = parse_first_date(extracted.bts.exact_dates_text) if extracted.bts else None
    bruno_first = parse_first_date(extracted.bruno.exact_dates_text) if extracted.bruno else None
    bts_first_str = bts_first.isoformat() if bts_first else "unknown"
    bruno_first_str = bruno_first.isoformat() if bruno_first else "unknown"

    # Bonus verification leaf: correctness of the reported time gap
    time_gap_leaf = evaluator.add_leaf(
        id="Time_Gap_Between_Tours",
        desc="Correctly calculates the time interval between the first BTS concert and the first Bruno Mars concert (exactly 1 month or 31 days)",
        parent=time_gap_group,
        critical=False  # non-critical bonus
    )
    await evaluator.verify(
        claim="The time interval between the first BTS concert date and the first Bruno Mars concert date is 31 days (about one month).",
        node=time_gap_leaf,
        additional_instruction=(
            f"Use the dates explicitly provided in the answer to judge. "
            f"For assistance, the extracted first dates are: BTS={bts_first_str}, Bruno={bruno_first_str}. "
            f"Accept '31 days' or '1 month' (or clear equivalents) as correct. "
            f"If the answer's stated interval differs, mark incorrect."
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
    Evaluate an answer for the Gillette Stadium concerts in late summer 2026.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level aggregation across criteria
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

    # Add the rubric root node as a parallel child to mirror structure
    major_info_node = evaluator.add_parallel(
        id="Major_Concerts_Information",
        desc="Provides complete and accurate information about major concerts at Gillette Stadium during August-September 2026",
        parent=root,
        critical=False
    )

    # Extract structured info
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_concert_plan(),
        template_class=ConcertPlanExtraction,
        extraction_name="concert_plan_extraction"
    )

    # Record ground truth expectations
    evaluator.add_ground_truth({
        "expected_bts_dates": EXPECTED_BTS_DATES_TEXT,
        "expected_bts_start_time": EXPECTED_BTS_START_TIME,
        "expected_bruno_dates": EXPECTED_BRUNO_DATES_TEXT,
        "expected_bruno_start_time": EXPECTED_BRUNO_START_TIME,
        "expected_stadium_city": EXPECTED_STADIUM_CITY,
        "expected_stadium_state": EXPECTED_STADIUM_STATE,
        "expected_stadium_capacity_approx": EXPECTED_STADIUM_CAPACITY_APPROX,
        "expected_time_gap_days": EXPECTED_TIME_GAP_DAYS
    }, gt_type="ground_truth")

    # Build verification subtrees
    await verify_bts_section(evaluator, major_info_node, extracted_info)
    await verify_bruno_section(evaluator, major_info_node, extracted_info)
    await verify_stadium_location_and_capacity(evaluator, major_info_node, extracted_info)
    await verify_time_gap_bonus(evaluator, major_info_node, extracted_info)

    # Return summary
    return evaluator.get_summary()