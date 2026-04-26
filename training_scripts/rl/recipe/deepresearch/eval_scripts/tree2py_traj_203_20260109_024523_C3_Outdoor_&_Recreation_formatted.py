import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "co_bear_canister_park"
TASK_DESCRIPTION = (
    "Identify the national park in Colorado where wilderness backcountry camping requires hikers to use "
    "bear-resistant food storage canisters from April 1 through October 31 in areas below treeline. For this park, "
    "provide the following information: (1) The complete official name of the national park, "
    "(2) Confirmation of the bear canister requirement dates (April 1 - October 31) and the specific areas where it applies (below treeline), "
    "(3) The wilderness permit fee amount for trips from May 1 through October 31, "
    "(4) The date and time when reservations open for May through October wilderness permits, "
    "(5) The online platform used for making these reservations, and "
    "(6) The minimum number of days in advance that reservations must be made before the first camping date. "
    "Provide reference URLs from official park or government sources to support each piece of information."
)

# Expected values derived from the rubric requirements
EXPECTED_BEAR_START = "April 1"
EXPECTED_BEAR_END = "October 31"
EXPECTED_BEAR_SCOPE = "below treeline"
EXPECTED_FEE_AMOUNT = "$36"
EXPECTED_FEE_SEASON = "May 1 through October 31"
EXPECTED_RESERVATION_OPEN_DATE = "March 1"
EXPECTED_RESERVATION_OPEN_TIME = "8:00 a.m. Mountain Standard Time"
EXPECTED_RESERVATION_PLATFORM = "Recreation.gov"
EXPECTED_MIN_DAYS_ADVANCE = "3 days"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkTaskExtraction(BaseModel):
    # Identification
    park_name: Optional[str] = None
    park_urls: List[str] = Field(default_factory=list)  # Official/government URLs supporting park identity/location

    # Bear canister policy
    bear_required_statement: Optional[str] = None
    bear_required_sources: List[str] = Field(default_factory=list)

    bear_dates_start: Optional[str] = None  # e.g., "April 1"
    bear_dates_end: Optional[str] = None    # e.g., "October 31"
    bear_dates_sources: List[str] = Field(default_factory=list)

    bear_scope: Optional[str] = None        # e.g., "below treeline"
    bear_scope_sources: List[str] = Field(default_factory=list)

    # Wilderness permit fee
    fee_amount: Optional[str] = None        # e.g., "$36"
    fee_season_window: Optional[str] = None # e.g., "May 1 through October 31"
    fee_sources: List[str] = Field(default_factory=list)

    # Reservation details
    reservations_open_date: Optional[str] = None                     # e.g., "March 1"
    reservations_open_date_sources: List[str] = Field(default_factory=list)

    reservations_open_time: Optional[str] = None                     # e.g., "8:00 a.m. Mountain Standard Time"
    reservations_open_time_sources: List[str] = Field(default_factory=list)

    reservation_platform: Optional[str] = None                       # e.g., "Recreation.gov"
    reservation_platform_sources: List[str] = Field(default_factory=list)

    advance_booking_minimum_days: Optional[str] = None               # e.g., "3 days"
    advance_booking_minimum_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_task() -> str:
    return """
    From the provided answer, extract the following structured information about a SINGLE U.S. National Park in Colorado:

    1) park_name: The complete official name of the national park (exact wording as in the answer).
    2) park_urls: A list of official/government URLs used in the answer to support the park’s identity/location (e.g., nps.gov, recreation.gov, .gov sites).

    Bear-resistant canister policy:
    3) bear_required_statement: The statement indicating bear-resistant food storage canisters are required for wilderness/backcountry camping (as phrased in the answer).
    4) bear_required_sources: URLs (from official/government sources) cited in the answer supporting the requirement.
    5) bear_dates_start: The start date for the requirement (e.g., "April 1") if present in the answer; else null.
    6) bear_dates_end: The end date (e.g., "October 31") if present; else null.
    7) bear_dates_sources: URLs from official/government sources supporting the dates.
    8) bear_scope: The specific area scope (e.g., "below treeline") if present.
    9) bear_scope_sources: URLs from official/government sources supporting the scope.

    Wilderness permit fee (May 1 through October 31):
    10) fee_amount: The fee amount (e.g., "$36") as stated in the answer.
    11) fee_season_window: The season window description (e.g., "May 1 through October 31") if mentioned.
    12) fee_sources: URLs from official/government sources supporting the fee.

    Reservation details for May–October wilderness permits:
    13) reservations_open_date: The calendar date when reservations open (e.g., "March 1").
    14) reservations_open_date_sources: URLs from official/government sources supporting the opening date.
    15) reservations_open_time: The opening time and time zone (e.g., "8:00 a.m. Mountain Standard Time").
    16) reservations_open_time_sources: URLs supporting the opening time.
    17) reservation_platform: The platform used for reservations (e.g., "Recreation.gov").
    18) reservation_platform_sources: URLs supporting that reservations are made through this platform.
    19) advance_booking_minimum_days: The minimum number of days in advance that reservations must be made (e.g., "3 days").
    20) advance_booking_minimum_sources: URLs supporting the advance-booking requirement.

    IMPORTANT:
    - Extract EXACTLY what is present in the answer. Do NOT invent, normalize, or infer missing details.
    - For any field not present, return null (or empty list for sources).
    - Extract full URLs when possible, including protocol.
    - Focus on official/government sources (e.g., nps.gov, recreation.gov, .gov). If the answer includes non-official sources, include them too.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_official_url(url: str) -> bool:
    """Check whether a URL appears to be an official park/government URL."""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        netloc = url.lower()
    return (
        "nps.gov" in netloc
        or "recreation.gov" in netloc
        or netloc.endswith(".gov")
        or "doi.gov" in netloc
        or "blm.gov" in netloc
        or "fs.usda.gov" in netloc
    )


def add_sources_presence_and_official_checks(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    desc_prefix: str,
    urls: List[str],
) -> List:
    """
    Add two custom critical leaf nodes:
    - sources_present: at least one URL provided
    - sources_official: at least one official/government URL provided
    Return the created nodes to be used as prerequisites for verification leaves.
    """
    sources_present = evaluator.add_custom_node(
        result=bool(urls and len(urls) > 0),
        id=f"{base_id}_sources_present",
        desc=f"{desc_prefix}: At least one source URL is provided",
        parent=parent_node,
        critical=True
    )
    sources_official = evaluator.add_custom_node(
        result=bool(urls and any(is_official_url(u) for u in urls)),
        id=f"{base_id}_sources_official",
        desc=f"{desc_prefix}: At least one source is an official/government URL",
        parent=parent_node,
        critical=True
    )
    return [sources_present, sources_official]


def add_field_presence_check(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    desc_prefix: str,
    value: Optional[str],
) -> Any:
    """Add a critical custom leaf node checking that a text field is present and non-empty."""
    return evaluator.add_custom_node(
        result=bool(value and str(value).strip()),
        id=f"{base_id}_value_present",
        desc=f"{desc_prefix}: Value is present in the answer",
        parent=parent_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_identify_park_nodes(
    evaluator: Evaluator,
    root_node,
    info: ParkTaskExtraction
) -> None:
    # Parent node for identification (critical, parallel)
    identify_node = evaluator.add_parallel(
        id="identify_park",
        desc="Identify the correct park that matches the constraints (Colorado national park with the specified bear-canister rule)",
        parent=root_node,
        critical=True
    )

    # official_park_name: Needs value present + official sources
    prereqs_name_sources = add_sources_presence_and_official_checks(
        evaluator, identify_node, "official_park_name", "Official park name", info.park_urls
    )
    prereq_name_value = add_field_presence_check(
        evaluator, identify_node, "official_park_name", "Official park name", info.park_name
    )
    official_name_leaf = evaluator.add_leaf(
        id="official_park_name",
        desc="Provide the complete official name of the national park and at least one supporting official/government URL",
        parent=identify_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the park is '{info.park_name or ''}'.",
        node=official_name_leaf,
        sources=info.park_urls,
        additional_instruction="Verify that the cited official pages explicitly show the park’s complete official name (not abbreviations). Allow minor punctuation differences.",
        extra_prerequisites=[prereqs_name_sources[0], prereqs_name_sources[1], prereq_name_value]
    )

    # is_colorado_national_park: Must be US National Park in Colorado, supported by official/government URL
    prereqs_location_sources = add_sources_presence_and_official_checks(
        evaluator, identify_node, "is_colorado_national_park", "Colorado National Park proof", info.park_urls
    )
    is_co_np_leaf = evaluator.add_leaf(
        id="is_colorado_national_park",
        desc="Verify the identified park is a U.S. National Park located in Colorado, with at least one supporting official/government URL",
        parent=identify_node,
        critical=True
    )
    await evaluator.verify(
        claim="This protected area is a U.S. National Park located in the state of Colorado.",
        node=is_co_np_leaf,
        sources=info.park_urls,
        additional_instruction="Confirm both: (1) It is part of the National Park System, and (2) It is located in Colorado.",
        extra_prerequisites=prereqs_location_sources
    )


async def build_bear_nodes(
    evaluator: Evaluator,
    root_node,
    info: ParkTaskExtraction
) -> None:
    bear_node = evaluator.add_parallel(
        id="bear_canister_requirement",
        desc="Verify the bear-resistant canister requirement details for wilderness backcountry camping",
        parent=root_node,
        critical=True
    )

    # bear_canister_required
    prereqs_req_sources = add_sources_presence_and_official_checks(
        evaluator, bear_node, "bear_canister_required", "Bear canister requirement", info.bear_required_sources
    )
    bear_req_leaf = evaluator.add_leaf(
        id="bear_canister_required",
        desc="Verify the park requires bear-resistant food storage canisters for wilderness/backcountry camping, with a supporting official/government URL",
        parent=bear_node,
        critical=True
    )
    await evaluator.verify(
        claim="Bear-resistant food storage canisters are required for wilderness/backcountry camping in this park.",
        node=bear_req_leaf,
        sources=info.bear_required_sources,
        additional_instruction="Treat season- and area-specific language as a requirement when the policy says 'required' for those periods/areas.",
        extra_prerequisites=prereqs_req_sources
    )

    # bear_canister_dates
    prereqs_dates_sources = add_sources_presence_and_official_checks(
        evaluator, bear_node, "bear_canister_dates", "Bear canister dates", info.bear_dates_sources
    )
    bear_dates_leaf = evaluator.add_leaf(
        id="bear_canister_dates",
        desc="Verify the requirement applies from April 1 through October 31, with a supporting official/government URL",
        parent=bear_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The canister requirement applies from {EXPECTED_BEAR_START} through {EXPECTED_BEAR_END}.",
        node=bear_dates_leaf,
        sources=info.bear_dates_sources,
        additional_instruction="Confirm the exact date window April 1–October 31.",
        extra_prerequisites=prereqs_dates_sources
    )

    # bear_canister_scope_below_treeline
    prereqs_scope_sources = add_sources_presence_and_official_checks(
        evaluator, bear_node, "bear_canister_scope_below_treeline", "Bear canister scope", info.bear_scope_sources
    )
    bear_scope_leaf = evaluator.add_leaf(
        id="bear_canister_scope_below_treeline",
        desc="Verify the requirement applies specifically to areas below treeline, with a supporting official/government URL",
        parent=bear_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The canister requirement applies specifically to areas {EXPECTED_BEAR_SCOPE}.",
        node=bear_scope_leaf,
        sources=info.bear_scope_sources,
        additional_instruction="Confirm the rule’s geographic scope explicitly mentions 'below treeline'.",
        extra_prerequisites=prereqs_scope_sources
    )


async def build_fee_node(
    evaluator: Evaluator,
    root_node,
    info: ParkTaskExtraction
) -> None:
    fee_node = evaluator.add_parallel(
        id="wilderness_permit_fee",
        desc="Verify the wilderness permit fee for the specified season window",
        parent=root_node,
        critical=True
    )

    prereqs_fee_sources = add_sources_presence_and_official_checks(
        evaluator, fee_node, "fee_amount", "Wilderness permit fee", info.fee_sources
    )
    fee_leaf = evaluator.add_leaf(
        id="fee_amount",
        desc="Confirm the wilderness permit fee is $36 per trip for trips/camping dates from May 1 through October 31, with a supporting official/government URL",
        parent=fee_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The wilderness permit fee is {EXPECTED_FEE_AMOUNT} per trip for trips dated {EXPECTED_FEE_SEASON}.",
        node=fee_leaf,
        sources=info.fee_sources,
        additional_instruction="Confirm both the fee amount ($36) and that it applies to the May 1–October 31 season window.",
        extra_prerequisites=prereqs_fee_sources
    )


async def build_reservation_nodes(
    evaluator: Evaluator,
    root_node,
    info: ParkTaskExtraction
) -> None:
    res_node = evaluator.add_parallel(
        id="reservation_details",
        desc="Verify reservation opening details, reservation platform, and advance-booking requirement for May–October wilderness permits",
        parent=root_node,
        critical=True
    )

    # reservations_open_date
    prereqs_date_sources = add_sources_presence_and_official_checks(
        evaluator, res_node, "reservations_open_date", "Reservations open date", info.reservations_open_date_sources
    )
    res_open_date_leaf = evaluator.add_leaf(
        id="reservations_open_date",
        desc="Verify reservations for May through October wilderness permits open on March 1, with a supporting official/government URL",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Reservations for May through October wilderness permits open on {EXPECTED_RESERVATION_OPEN_DATE}.",
        node=res_open_date_leaf,
        sources=info.reservations_open_date_sources,
        additional_instruction="Focus on reservations open date for the May–October season.",
        extra_prerequisites=prereqs_date_sources
    )

    # reservations_open_time
    prereqs_time_sources = add_sources_presence_and_official_checks(
        evaluator, res_node, "reservations_open_time", "Reservations open time", info.reservations_open_time_sources
    )
    res_open_time_leaf = evaluator.add_leaf(
        id="reservations_open_time",
        desc="Verify reservation opening time is 8:00 a.m. Mountain Standard Time, with a supporting official/government URL",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Reservation opening time is {EXPECTED_RESERVATION_OPEN_TIME}.",
        node=res_open_time_leaf,
        sources=info.reservations_open_time_sources,
        additional_instruction="Many official pages use 'Mountain Time' and may specify MST/MDT depending on date; treat '8:00 a.m. Mountain Time' as equivalent to '8:00 a.m. MST/MDT' if the park’s page clearly indicates 8:00 a.m.",
        extra_prerequisites=prereqs_time_sources
    )

    # reservation_platform
    prereqs_platform_sources = add_sources_presence_and_official_checks(
        evaluator, res_node, "reservation_platform", "Reservation platform", info.reservation_platform_sources
    )
    res_platform_leaf = evaluator.add_leaf(
        id="reservation_platform",
        desc="Verify reservations for the May–October season are made online only through Recreation.gov, with a supporting official/government URL",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Reservations for the May–October season are made online only through {EXPECTED_RESERVATION_PLATFORM}.",
        node=res_platform_leaf,
        sources=info.reservation_platform_sources,
        additional_instruction="Confirm the official policy specifies Recreation.gov as the platform and 'online only'.",
        extra_prerequisites=prereqs_platform_sources
    )

    # advance_booking_minimum
    prereqs_min_days_sources = add_sources_presence_and_official_checks(
        evaluator, res_node, "advance_booking_minimum", "Advance booking minimum", info.advance_booking_minimum_sources
    )
    res_min_days_leaf = evaluator.add_leaf(
        id="advance_booking_minimum",
        desc="Verify reservations must be made at least 3 days before the first camping date, with a supporting official/government URL",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Reservations must be made at least {EXPECTED_MIN_DAYS_ADVANCE} before the first camping date.",
        node=res_min_days_leaf,
        sources=info.advance_booking_minimum_sources,
        additional_instruction="Confirm the minimum advance booking requirement explicitly states 'at least 3 days'.",
        extra_prerequisites=prereqs_min_days_sources
    )


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
    """
    Evaluate an answer for the Colorado bear-canister park identification task and verify all requested details.
    """
    # Initialize evaluator with a sequential root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    # Make root critical to enforce mandatory criteria
    root.critical = True

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_park_task(),
        template_class=ParkTaskExtraction,
        extraction_name="park_task_extraction"
    )

    # Add expected constraints as custom info for transparency
    evaluator.add_custom_info(
        {
            "expected_bear_dates": f"{EXPECTED_BEAR_START} - {EXPECTED_BEAR_END}",
            "expected_bear_scope": EXPECTED_BEAR_SCOPE,
            "expected_fee": f"{EXPECTED_FEE_AMOUNT} per trip ({EXPECTED_FEE_SEASON})",
            "expected_reservation_open_date": EXPECTED_RESERVATION_OPEN_DATE,
            "expected_reservation_open_time": EXPECTED_RESERVATION_OPEN_TIME,
            "expected_reservation_platform": EXPECTED_RESERVATION_PLATFORM,
            "expected_min_days_advance": EXPECTED_MIN_DAYS_ADVANCE
        },
        info_type="expected_values",
        info_name="expected_constraints"
    )

    # Build and verify all subtrees according to rubric
    await build_identify_park_nodes(evaluator, root, extracted_info)
    await build_bear_nodes(evaluator, root, extracted_info)
    await build_fee_node(evaluator, root, extracted_info)
    await build_reservation_nodes(evaluator, root, extracted_info)

    # Return final structured summary
    return evaluator.get_summary()