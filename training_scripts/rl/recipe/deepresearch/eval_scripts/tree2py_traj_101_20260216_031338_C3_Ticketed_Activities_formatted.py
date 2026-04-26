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
TASK_ID = "jlo_lv_residency_2026"
TASK_DESCRIPTION = """
Identify Jennifer Lopez's Las Vegas residency scheduled for 2026. Provide the following information: (1) The official name of the residency, (2) The venue where it takes place, including the full venue name and location (city and state), (3) The seating capacity of the venue, and (4) Confirmation of the month when performances are scheduled in 2026. For each piece of information, include reference URLs from official ticketing platforms or venue websites to support your answer.
"""

EXPECTED_RESIDENCY_NAME = "Up All Night Live in Las Vegas"
EXPECTED_VENUE_NAME = "The Colosseum at Caesars Palace"
EXPECTED_CITY = "Las Vegas"
EXPECTED_STATE = "Nevada"
CAPACITY_MIN = 4100
CAPACITY_MAX = 4300
EXPECTED_MONTH_2026 = "March 2026"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResidencyExtraction(BaseModel):
    performer_name: Optional[str] = None
    residency_name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    seating_capacity: Optional[str] = None
    performance_month_2026: Optional[str] = None

    name_source_urls: List[str] = Field(default_factory=list)
    venue_source_urls: List[str] = Field(default_factory=list)
    capacity_source_urls: List[str] = Field(default_factory=list)
    month_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_residency_info() -> str:
    return """
    Extract the residency information for Jennifer Lopez's Las Vegas shows, as presented in the answer.

    Return a JSON object with the following fields (use null for any missing field):

    1. performer_name: The performer of the residency (e.g., "Jennifer Lopez").
    2. residency_name: The official residency name as stated in the answer (e.g., "Up All Night Live in Las Vegas").
    3. venue_name: The full venue name (e.g., "The Colosseum at Caesars Palace").
    4. venue_city: The city (e.g., "Las Vegas").
    5. venue_state: The state (e.g., "Nevada").
    6. seating_capacity: The seating capacity of the venue as stated in the answer (e.g., "4,300" or "approximately 4,200").
    7. performance_month_2026: The month(s) specifically mentioned for 2026 performances (e.g., "March 2026").

    Also extract official-source URLs that support each piece:
    - name_source_urls: URLs from official ticketing platforms (e.g., Ticketmaster, Live Nation, AXS) or official venue/organizer websites that show the residency name and performer.
    - venue_source_urls: URLs from official ticketing platforms or official venue websites that show the venue and location.
    - capacity_source_urls: URLs from official venue websites or official documentation that show the seating capacity.
    - month_source_urls: URLs from official ticketing platforms or official venue/organizer sites that show the 2026 performance month.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer text (plain URLs or markdown links). Do not invent or infer URLs.
    - Only include official sources (e.g., ticketmaster.com, livenation.com, axs.com, caesars.com, official venue domain). Do not include third-party blogs or unofficial aggregators.
    - Always include full URLs (with http:// or https://). If protocol is missing, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _equal_ignoring_case_and_spaces(a: Optional[str], b: Optional[str]) -> bool:
    a_norm = re.sub(r"\s+", " ", _norm(a))
    b_norm = re.sub(r"\s+", " ", _norm(b))
    return a_norm == b_norm and a_norm != ""


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    res = []
    for u in urls:
        if not u:
            continue
        uu = u.strip()
        if uu not in seen:
            seen.add(uu)
            res.append(uu)
    return res


def _extract_ints(text: str) -> List[int]:
    # Capture numbers like "4,300", "4300", "4.3k" is handled partially below
    ints = []
    for m in re.finditer(r"\b(\d{1,3}(?:[,\s]\d{3})+|\d+)\b", text):
        num_str = m.group(1)
        num_str = num_str.replace(",", "").replace(" ", "")
        try:
            ints.append(int(num_str))
        except Exception:
            pass

    # Handle patterns like "4k", "4.3k"
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*k\b", text.lower()):
        val = float(m.group(1)) * 1000
        ints.append(int(round(val)))
    return ints


def capacity_in_range(cap_text: Optional[str], low: int = CAPACITY_MIN, high: int = CAPACITY_MAX) -> bool:
    if not cap_text:
        return False
    nums = _extract_ints(cap_text)
    if not nums:
        return False
    # If any number lies within the range, accept
    for n in nums:
        if low <= n <= high:
            return True
    # If a range like "4100–4300" appears, accept
    if re.search(rf"{low}\D+{high}", cap_text.replace(",", "")):
        return True
    return False


def contains_march_2026(text: Optional[str]) -> bool:
    if not text:
        return False
    t = _norm(text)
    has_march = ("march" in t) or re.search(r"\bmar\b", t) is not None
    has_2026 = "2026" in t
    return has_march and has_2026


def official_source_instruction(context: str) -> str:
    return (
        f"Evaluate only with official evidence for: {context}. "
        "Accept domains such as ticketmaster.com, livenation.com, axs.com, caesars.com (including official venue pages), "
        "or other official organizer/venue sites. If the provided URL is missing, inaccessible, or clearly unofficial/review/blog/aggregator, "
        "you must conclude 'not supported'. Prefer explicit statements or event listings on the page. "
        "Allow minor formatting variations (e.g., punctuation/casing differences) when matching names."
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_core_constraints(
    evaluator: Evaluator,
    parent_node,
    data: ResidencyExtraction,
) -> List[tuple]:
    node = evaluator.add_parallel(
        id="ResidencyCoreConstraints",
        desc="Residency matches the core constraints (performer and 2026 scheduling).",
        parent=parent_node,
        critical=True,
    )

    # PerformerIsJenniferLopez (leaf)
    performer_leaf = evaluator.add_leaf(
        id="PerformerIsJenniferLopez",
        desc="The residency is explicitly identified as being performed by Jennifer Lopez.",
        parent=node,
        critical=True,
    )
    performer_claim = "The residency pages show that the performer is Jennifer Lopez."
    performer_sources = _unique_urls((data.name_source_urls or []) + (data.venue_source_urls or []) + (data.month_source_urls or []))
    performer_instruction = official_source_instruction("performer identity for Jennifer Lopez's residency")
    # Add to batch verification list
    tasks = [(performer_claim, performer_sources if performer_sources else None, performer_leaf, performer_instruction)]

    # ScheduledFor2026 (leaf)
    scheduled_leaf = evaluator.add_leaf(
        id="ScheduledFor2026",
        desc="The residency is explicitly indicated to have performances scheduled in the year 2026.",
        parent=node,
        critical=True,
    )
    scheduled_claim = "There are performances scheduled in 2026 for this residency."
    scheduled_sources = _unique_urls(data.month_source_urls or [])
    scheduled_instruction = official_source_instruction("presence of 2026 performances for the residency")
    tasks.append((scheduled_claim, scheduled_sources if scheduled_sources else None, scheduled_leaf, scheduled_instruction))

    return tasks


async def build_official_name_group(
    evaluator: Evaluator,
    parent_node,
    data: ResidencyExtraction,
) -> List[tuple]:
    node = evaluator.add_parallel(
        id="ResidencyOfficialName",
        desc="Provides the official name of the residency (must match constraint) with an official-source URL.",
        parent=parent_node,
        critical=True,
    )

    # ResidencyNameMatchesConstraint (custom check)
    name_match_result = _equal_ignoring_case_and_spaces(data.residency_name, EXPECTED_RESIDENCY_NAME)
    evaluator.add_custom_node(
        result=name_match_result,
        id="ResidencyNameMatchesConstraint",
        desc=f"States the official name exactly as: '{EXPECTED_RESIDENCY_NAME}'.",
        parent=node,
        critical=True,
    )

    # NameSupportedByOfficialSourceURL (leaf)
    name_support_leaf = evaluator.add_leaf(
        id="NameSupportedByOfficialSourceURL",
        desc="Includes a reference URL from an official ticketing platform or official venue/organizer site supporting the stated residency name.",
        parent=node,
        critical=True,
    )
    name_claim = f"The residency name shown on the official source is '{EXPECTED_RESIDENCY_NAME}'."
    name_sources = _unique_urls(data.name_source_urls or [])
    name_instruction = official_source_instruction("the official residency name")
    return [(name_claim, name_sources if name_sources else None, name_support_leaf, name_instruction)]


async def build_venue_name_location_group(
    evaluator: Evaluator,
    parent_node,
    data: ResidencyExtraction,
) -> List[tuple]:
    node = evaluator.add_parallel(
        id="VenueNameLocation",
        desc="Provides the venue and its location (must match constraints) with an official-source URL.",
        parent=parent_node,
        critical=True,
    )

    # VenueMatchesConstraint (custom)
    venue_match = _equal_ignoring_case_and_spaces(data.venue_name, EXPECTED_VENUE_NAME)
    evaluator.add_custom_node(
        result=venue_match,
        id="VenueMatchesConstraint",
        desc=f"States the venue as: {EXPECTED_VENUE_NAME}.",
        parent=node,
        critical=True,
    )

    # VenueLocationMatchesConstraint (custom)
    city_match = _equal_ignoring_case_and_spaces(data.venue_city, EXPECTED_CITY)
    state_match = _equal_ignoring_case_and_spaces(data.venue_state, EXPECTED_STATE)
    evaluator.add_custom_node(
        result=(city_match and state_match),
        id="VenueLocationMatchesConstraint",
        desc=f"States the venue location as {EXPECTED_CITY}, {EXPECTED_STATE} (city and state).",
        parent=node,
        critical=True,
    )

    # VenueSupportedByOfficialSourceURL (leaf)
    venue_support_leaf = evaluator.add_leaf(
        id="VenueSupportedByOfficialSourceURL",
        desc="Includes a reference URL from an official ticketing platform or official venue website supporting the venue name and/or location.",
        parent=node,
        critical=True,
    )
    venue_claim = f"The residency takes place at {EXPECTED_VENUE_NAME} in {EXPECTED_CITY}, {EXPECTED_STATE}."
    venue_sources = _unique_urls(data.venue_source_urls or [])
    venue_instruction = official_source_instruction("the venue name and location for the residency")
    return [(venue_claim, venue_sources if venue_sources else None, venue_support_leaf, venue_instruction)]


async def build_capacity_group(
    evaluator: Evaluator,
    parent_node,
    data: ResidencyExtraction,
) -> List[tuple]:
    node = evaluator.add_parallel(
        id="VenueSeatingCapacity",
        desc="Provides the venue seating capacity (must match constraint range) with an official-source URL.",
        parent=parent_node,
        critical=True,
    )

    # CapacityMatchesConstraintRange (custom)
    cap_ok = capacity_in_range(data.seating_capacity, CAPACITY_MIN, CAPACITY_MAX)
    evaluator.add_custom_node(
        result=cap_ok,
        id="CapacityMatchesConstraintRange",
        desc=f"States a seating capacity that is approximately within {CAPACITY_MIN:,}–{CAPACITY_MAX:,} seats.",
        parent=node,
        critical=True,
    )

    # CapacitySupportedByOfficialSourceURL (leaf)
    cap_leaf = evaluator.add_leaf(
        id="CapacitySupportedByOfficialSourceURL",
        desc="Includes a reference URL from an official venue website (or official venue documentation) supporting the capacity claim.",
        parent=node,
        critical=True,
    )
    cap_claim = (
        f"The seating capacity of {EXPECTED_VENUE_NAME} is approximately between {CAPACITY_MIN:,} and {CAPACITY_MAX:,} seats."
    )
    cap_sources = _unique_urls(data.capacity_source_urls or [])
    cap_instruction = official_source_instruction("the official venue seating capacity")
    return [(cap_claim, cap_sources if cap_sources else None, cap_leaf, cap_instruction)]


async def build_performance_month_group(
    evaluator: Evaluator,
    parent_node,
    data: ResidencyExtraction,
) -> List[tuple]:
    node = evaluator.add_parallel(
        id="PerformanceMonth2026",
        desc="Confirms the 2026 performance month (must match constraint) with an official-source URL.",
        parent=parent_node,
        critical=True,
    )

    # PerformanceMonthMatchesConstraint (custom)
    month_ok = contains_march_2026(data.performance_month_2026)
    evaluator.add_custom_node(
        result=month_ok,
        id="PerformanceMonthMatchesConstraint",
        desc=f"Explicitly states that 2026 performances are scheduled in {EXPECTED_MONTH_2026}.",
        parent=node,
        critical=True,
    )

    # MonthSupportedByOfficialSourceURL (leaf)
    month_leaf = evaluator.add_leaf(
        id="MonthSupportedByOfficialSourceURL",
        desc="Includes a reference URL from an official ticketing platform or official venue/organizer site supporting the stated 2026 performance month.",
        parent=node,
        critical=True,
    )
    month_claim = f"Performances for the residency are scheduled in {EXPECTED_MONTH_2026}."
    month_sources = _unique_urls(data.month_source_urls or [])
    month_instruction = official_source_instruction("the residency performance month in 2026")
    return [(month_claim, month_sources if month_sources else None, month_leaf, month_instruction)]


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Jennifer Lopez 2026 Las Vegas residency task.
    """
    # Initialize evaluator (root is non-critical parallel)
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

    # Extract structured residency info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_residency_info(),
        template_class=ResidencyExtraction,
        extraction_name="residency_info",
    )

    # Add Ground Truth info for transparency
    evaluator.add_ground_truth({
        "expected_residency_name": EXPECTED_RESIDENCY_NAME,
        "expected_venue_name": EXPECTED_VENUE_NAME,
        "expected_location": f"{EXPECTED_CITY}, {EXPECTED_STATE}",
        "expected_capacity_range": f"{CAPACITY_MIN}-{CAPACITY_MAX}",
        "expected_month_2026": EXPECTED_MONTH_2026,
    })

    # Build top-level critical node
    top = evaluator.add_parallel(
        id="ResidencyInformationComplete",
        desc="Answer identifies Jennifer Lopez's 2026 Las Vegas residency and provides required details with official-source URLs.",
        parent=root,
        critical=True,
    )

    # Build subtrees and collect verification tasks
    verify_tasks: List[tuple] = []
    verify_tasks += await build_core_constraints(evaluator, top, extracted)
    verify_tasks += await build_official_name_group(evaluator, top, extracted)
    verify_tasks += await build_venue_name_location_group(evaluator, top, extracted)
    verify_tasks += await build_capacity_group(evaluator, top, extracted)
    verify_tasks += await build_performance_month_group(evaluator, top, extracted)

    # Run all verifications in parallel where possible
    if verify_tasks:
        await evaluator.batch_verify(verify_tasks)

    # Return structured summary with verification tree and score
    return evaluator.get_summary()