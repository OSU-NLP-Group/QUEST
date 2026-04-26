import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "penland_metals_concentration"
TASK_DESCRIPTION = (
    "I am interested in attending an 8-week concentration workshop in Metals at Penland School of Craft in North Carolina. "
    "Please identify a specific 8-week metals concentration workshop offered by Penland School and provide the following information: "
    "1. What are the start and end dates for this 8-week concentration session? "
    "2. What is the specific arrival time on the first day and departure time on the last day? "
    "3. What is the minimum age requirement for workshop participants? "
    "4. Confirm whether students are required to take only one workshop at a time during 8-week concentration sessions. "
    "5. What is the non-refundable deposit amount required for non-scholarship students? "
    "6. Based on the session timing you identified, what is the balance payment deadline for this workshop?"
)

# Expected policy constants (used for claims)
EXPECTED_ARRIVAL = "4:30 PM Sunday"
EXPECTED_DEPARTURE = "10:00 AM Friday"
EXPECTED_MIN_AGE = "18"
EXPECTED_DEPOSIT = "$300"

SEASON_DEADLINE_MAP = {
    "spring": "January 15",
    "summer": "April 15",
    "fall": "August 15",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WorkshopSelection(BaseModel):
    workshop_title: Optional[str] = None
    workshop_url: Optional[str] = None
    school_name: Optional[str] = None
    workshop_format: Optional[str] = None  # e.g., "8-week concentration"
    medium: Optional[str] = None  # e.g., "metals", "iron", "blacksmithing"


class SessionTiming(BaseModel):
    session_start_date: Optional[str] = None
    session_end_date: Optional[str] = None
    arrival_time: Optional[str] = None
    departure_time: Optional[str] = None
    season: Optional[str] = None  # e.g., "Spring", "Fall", "Summer"
    timing_urls: List[str] = Field(default_factory=list)


class RegistrationPolicies(BaseModel):
    minimum_age: Optional[str] = None
    exclusive_enrollment: Optional[str] = None  # e.g., "one workshop per session"
    requirements_urls: List[str] = Field(default_factory=list)


class Financials(BaseModel):
    deposit_amount: Optional[str] = None
    deposit_urls: List[str] = Field(default_factory=list)
    balance_deadline: Optional[str] = None
    deadline_urls: List[str] = Field(default_factory=list)


class PenlandMetalsExtraction(BaseModel):
    workshop: Optional[WorkshopSelection] = None
    timing: Optional[SessionTiming] = None
    registration: Optional[RegistrationPolicies] = None
    financials: Optional[Financials] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_penland_metals() -> str:
    return """
Extract information about a specific 8-week concentration workshop in metals at Penland School of Craft from the answer.

Return a JSON object with the following structure and fields (use null for any missing field):

{
  "workshop": {
    "workshop_title": string | null,
    "workshop_url": string | null,
    "school_name": string | null,
    "workshop_format": string | null,   // e.g., "8-week concentration", "eight-week concentration"
    "medium": string | null             // e.g., "metals", "iron", "blacksmithing", "jewelry/metals"
  },
  "timing": {
    "session_start_date": string | null,   // e.g., "March 3, 2026"
    "session_end_date": string | null,     // e.g., "April 28, 2026"
    "arrival_time": string | null,         // e.g., "4:30 PM Sunday"
    "departure_time": string | null,       // e.g., "10:00 AM Friday"
    "season": string | null,               // e.g., "Spring", "Summer", "Fall"
    "timing_urls": string[]                // All URLs cited for timing/dates/arrival-departure details
  },
  "registration": {
    "minimum_age": string | null,          // e.g., "18"
    "exclusive_enrollment": string | null, // e.g., "students take only one workshop at a time"
    "requirements_urls": string[]          // URLs cited for age/enrollment policy
  },
  "financials": {
    "deposit_amount": string | null,       // e.g., "$300"
    "deposit_urls": string[],              // URLs cited for deposit policy
    "balance_deadline": string | null,     // e.g., "January 15"
    "deadline_urls": string[]              // URLs cited for payment deadlines
  }
}

Important extraction rules:
- Extract only what is explicitly present in the answer. Do not invent values.
- For each URL list field, include all URLs that the answer cites for that specific topic.
- URLs can appear as plain links or markdown links; extract the actual URLs.
- If a URL is missing a protocol, prepend http://
- If multiple metals workshops are mentioned, extract the first one that clearly matches an 8-week concentration in metals.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _gather_sources(*args: List[Optional[List[str]]], single: Optional[str] = None) -> List[str]:
    """Flatten and deduplicate URL lists; optionally include a single URL."""
    urls: List[str] = []
    if single:
        urls.append(single)
    for lst in args:
        if lst:
            urls.extend([u for u in lst if isinstance(u, str) and u.strip()])
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _parse_month_from_date(date_str: Optional[str]) -> Optional[int]:
    """Attempt to parse month number from a free-form date string."""
    if not date_str:
        return None
    months = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }
    s = date_str.strip().lower()
    # Try month names
    for name, num in months.items():
        if re.search(rf"\b{name}\b", s):
            return num
    # Try numeric dates like 03/15/2026 or 3/15/26
    m = re.search(r"\b(\d{1,2})[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b", s)
    if m:
        try:
            mnum = int(m.group(1))
            if 1 <= mnum <= 12:
                return mnum
        except Exception:
            pass
    return None


def _derive_season(timing: SessionTiming, workshop: Optional[WorkshopSelection]) -> Optional[str]:
    """Derive season (spring/summer/fall) from extracted season label or start date month."""
    # Prefer explicit season label
    if timing and timing.season:
        lab = timing.season.strip().lower()
        if any(k in lab for k in ["spring", "summer", "fall"]):
            if "spring" in lab:
                return "spring"
            if "summer" in lab:
                return "summer"
            if "fall" in lab:
                return "fall"
    # Attempt from start date month
    month = _parse_month_from_date(timing.session_start_date if timing else None)
    if month:
        if month in (3, 4, 5, 1, 2):  # Penland spring concentrations often start Mar–May; map Jan/Feb toward spring deadline cycle
            return "spring"
        if month in (6, 7, 8):
            return "summer"
        if month in (9, 10, 11, 12):
            return "fall"
    # Try workshop_format hints
    if workshop and workshop.workshop_format:
        wf = workshop.workshop_format.lower()
        if "spring" in wf:
            return "spring"
        if "summer" in wf:
            return "summer"
        if "fall" in wf or "autumn" in wf:
            return "fall"
    return None


def _expected_deadline_for_season(season: Optional[str]) -> Optional[str]:
    if not season:
        return None
    return SEASON_DEADLINE_MAP.get(season.lower())


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_workshop_identification_nodes(
    evaluator: Evaluator,
    parent_node,
    data: PenlandMetalsExtraction
) -> None:
    node = evaluator.add_parallel(
        id="workshop_identification",
        desc="Correctly identify an 8-week concentration workshop at Penland School in metals",
        parent=parent_node,
        critical=True
    )

    workshop = data.workshop or WorkshopSelection()
    w_url = workshop.workshop_url

    # Leaf: correct_school
    school_leaf = evaluator.add_leaf(
        id="correct_school",
        desc="The workshop is offered at Penland School of Craft (not John C. Campbell Folk School or another institution)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage belongs to Penland School of Craft and describes a Penland School of Craft workshop.",
        node=school_leaf,
        sources=w_url,
        additional_instruction="Use the page content and site branding to confirm it is Penland School of Craft (penland.org), not any other institution."
    )

    # Leaf: correct_format
    format_leaf = evaluator.add_leaf(
        id="correct_format",
        desc="The workshop is an 8-week concentration workshop (not a 1-week, 2-week, or other duration)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This page describes an 8-week Concentration workshop (eight-week concentration) at Penland.",
        node=format_leaf,
        sources=w_url,
        additional_instruction="Allow variants like '8 week', 'eight-week', or 'concentration'. Ensure it is an 8-week Concentration format, not a 1- or 2-week session."
    )

    # Leaf: correct_medium
    medium_leaf = evaluator.add_leaf(
        id="correct_medium",
        desc="The workshop focuses on metals as the craft medium",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This workshop is in the Metals area (e.g., metals, iron, blacksmithing, or jewelry/metals).",
        node=medium_leaf,
        sources=w_url,
        additional_instruction="Confirm that the medium is metals. Synonyms like iron/blacksmithing/jewelry metals count as metals."
    )


async def build_workshop_details_nodes(
    evaluator: Evaluator,
    parent_node,
    data: PenlandMetalsExtraction
) -> None:
    node = evaluator.add_parallel(
        id="workshop_details_verification",
        desc="Verify session timing, schedule structure, and registration requirements for the identified workshop",
        parent=parent_node,
        critical=True
    )

    workshop = data.workshop or WorkshopSelection()
    timing = data.timing or SessionTiming()
    registration = data.registration or RegistrationPolicies()

    # Session timing (parallel)
    timing_node = evaluator.add_parallel(
        id="session_timing",
        desc="Verify the session timing and schedule structure",
        parent=node,
        critical=True
    )

    combined_timing_sources = _gather_sources(timing.timing_urls, single=workshop.workshop_url)

    # Leaf: session_dates
    dates_leaf = evaluator.add_leaf(
        id="session_dates",
        desc="Provide the start and end dates for the 8-week concentration session",
        parent=timing_node,
        critical=True
    )
    start = timing.session_start_date or ""
    end = timing.session_end_date or ""
    await evaluator.verify(
        claim=f"The concentration session runs from {start} to {end}.",
        node=dates_leaf,
        sources=combined_timing_sources,
        additional_instruction="Verify that both the start and end dates exactly match those shown on the cited page(s)."
    )

    # Leaf: arrival_departure_times
    arr_dep_leaf = evaluator.add_leaf(
        id="arrival_departure_times",
        desc="Confirm the specific arrival time on the first day (4:30 PM Sunday) and departure time on the last day (10:00 AM Friday)",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Arrival time is {EXPECTED_ARRIVAL} and departure time is {EXPECTED_DEPARTURE}.",
        node=arr_dep_leaf,
        sources=combined_timing_sources,
        additional_instruction="Confirm check-in/arrival is Sunday at 4:30 PM and departure is Friday at 10:00 AM as stated in Penland concentration schedule details."
    )

    # Leaf: timing_url_reference (existence of at least one URL)
    timing_url_ref = evaluator.add_custom_node(
        result=bool(timing.timing_urls and len(timing.timing_urls) > 0),
        id="timing_url_reference",
        desc="Provide a URL reference that confirms the session timing details",
        parent=timing_node,
        critical=True
    )

    # Registration policies (parallel)
    reg_node = evaluator.add_parallel(
        id="registration_policies",
        desc="Verify registration requirements and policies",
        parent=node,
        critical=True
    )
    combined_reg_sources = _gather_sources(registration.requirements_urls, single=workshop.workshop_url)

    # Leaf: age_requirement
    age_leaf = evaluator.add_leaf(
        id="age_requirement",
        desc="Confirm that the minimum age requirement is 18 years old",
        parent=reg_node,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum age requirement for Penland concentration students is 18 years old.",
        node=age_leaf,
        sources=combined_reg_sources,
        additional_instruction="Look for admissions or registration policies stating adult programs require age 18+."
    )

    # Leaf: exclusive_enrollment
    exclusive_leaf = evaluator.add_leaf(
        id="exclusive_enrollment",
        desc="Confirm that students take only one workshop at a time during concentration sessions",
        parent=reg_node,
        critical=True
    )
    await evaluator.verify(
        claim="During 8-week concentration sessions, each student enrolls in only one workshop at a time.",
        node=exclusive_leaf,
        sources=combined_reg_sources,
        additional_instruction="Confirm concentration registration policy indicates one workshop per student per session."
    )

    # Leaf: requirements_url_reference (existence)
    req_url_ref = evaluator.add_custom_node(
        result=bool(registration.requirements_urls and len(registration.requirements_urls) > 0),
        id="requirements_url_reference",
        desc="Provide a URL reference that confirms the registration requirements",
        parent=reg_node,
        critical=True
    )


async def build_financial_nodes(
    evaluator: Evaluator,
    parent_node,
    data: PenlandMetalsExtraction
) -> None:
    node = evaluator.add_sequential(
        id="financial_requirements",
        desc="Verify deposit amount and payment deadline for the identified workshop",
        parent=parent_node,
        critical=True
    )

    workshop = data.workshop or WorkshopSelection()
    financials = data.financials or Financials()
    timing = data.timing or SessionTiming()

    # Deposit verification (parallel)
    deposit_node = evaluator.add_parallel(
        id="deposit_verification",
        desc="Verify the non-refundable deposit amount required for non-scholarship students",
        parent=node,
        critical=True
    )
    combined_deposit_sources = _gather_sources(financials.deposit_urls, single=workshop.workshop_url)

    # Leaf: deposit_amount
    deposit_leaf = evaluator.add_leaf(
        id="deposit_amount",
        desc="The deposit amount is correctly stated as $300 for non-scholarship students",
        parent=deposit_node,
        critical=True
    )
    await evaluator.verify(
        claim="The non-refundable deposit required for non-scholarship students is $300.",
        node=deposit_leaf,
        sources=combined_deposit_sources,
        additional_instruction="Confirm deposit policy and amount ($300) from Penland's registration/tuition information."
    )

    # Leaf: deposit_url_reference (existence)
    deposit_url_ref = evaluator.add_custom_node(
        result=bool(financials.deposit_urls and len(financials.deposit_urls) > 0),
        id="deposit_url_reference",
        desc="Provide a URL reference that confirms the deposit amount",
        parent=deposit_node,
        critical=True
    )

    # Payment deadline (parallel)
    deadline_node = evaluator.add_parallel(
        id="payment_deadline",
        desc="Identify the correct balance payment deadline based on the workshop's season (Jan 15 for spring, Apr 15 for summer, Aug 15 for fall)",
        parent=node,
        critical=True
    )

    season = _derive_season(timing, workshop)
    expected_deadline = _expected_deadline_for_season(season)
    # Record inferred season/deadline for transparency
    evaluator.add_custom_info(
        info={"derived_season": season, "expected_deadline": expected_deadline},
        info_type="derived_info",
        info_name="season_deadline_inference"
    )

    # Leaf: correct_deadline
    correct_deadline_leaf = evaluator.add_leaf(
        id="correct_deadline",
        desc="The payment deadline matches the season of the identified workshop",
        parent=deadline_node,
        critical=True
    )
    # Build a robust claim
    if expected_deadline and season:
        claim_deadline = f"Penland's policy states that for {season} concentrations, the balance payment deadline is {expected_deadline}."
    else:
        # Fallback: use provided balance_deadline text if any (less ideal)
        fallback_deadline = financials.balance_deadline or "the correct balance deadline per Penland policy"
        claim_deadline = f"The balance payment deadline for this concentration session matches Penland's published schedule: {fallback_deadline}."

    combined_deadline_sources = _gather_sources(
        financials.deadline_urls,
        financials.deposit_urls,
        (timing.timing_urls if timing else []),
        single=workshop.workshop_url
    )

    await evaluator.verify(
        claim=claim_deadline,
        node=correct_deadline_leaf,
        sources=combined_deadline_sources,
        additional_instruction="Verify on Penland policy/tuition/registration pages that the balance payment deadline aligns with the identified season (Spring: Jan 15, Summer: Apr 15, Fall: Aug 15)."
    )

    # Leaf: deadline_url_reference (existence)
    deadline_url_ref = evaluator.add_custom_node(
        result=bool(financials.deadline_urls and len(financials.deadline_urls) > 0),
        id="deadline_url_reference",
        desc="Provide a URL reference that confirms the payment deadline schedule",
        parent=deadline_node,
        critical=True
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    # Initialize evaluator with a critical sequential root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Successfully identify a Penland School 8-week concentration workshop in metals and verify all registration, timing, and financial requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )
    # Mark root as critical (children of critical must be critical)
    root.critical = True

    # Extract structured information from the answer
    extracted: PenlandMetalsExtraction = await evaluator.extract(
        prompt=prompt_extract_penland_metals(),
        template_class=PenlandMetalsExtraction,
        extraction_name="penland_metals_extraction"
    )

    # Add useful expectation info
    evaluator.add_ground_truth({
        "expected_arrival": EXPECTED_ARRIVAL,
        "expected_departure": EXPECTED_DEPARTURE,
        "expected_min_age": EXPECTED_MIN_AGE,
        "expected_deposit": EXPECTED_DEPOSIT,
        "season_deadlines": SEASON_DEADLINE_MAP
    }, gt_type="policy_expectations")

    # Build verification tree according to rubric
    await build_workshop_identification_nodes(evaluator, root, extracted)
    await build_workshop_details_nodes(evaluator, root, extracted)
    await build_financial_nodes(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()