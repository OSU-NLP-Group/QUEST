import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "recreation_planning_2026"
TASK_DESCRIPTION = (
    "A couple living in Maryland, both aged 64 and U.S. citizens, is planning multiple outdoor recreation trips in 2026. "
    "They want to visit national parks several times throughout the year and need an annual federal recreation pass. "
    "They are also planning a camping trip to a Florida state park with an arrival date of July 15, 2026, and need to know when they can make their reservation. "
    "Additionally, they will visit two different Delaware ocean state parks on separate days during the Delaware state park fee season, using their Maryland-registered vehicle.\n\n"
    "Determine the following:\n"
    "1. What is the most cost-effective annual federal recreation pass option for this couple, and what is its cost?\n"
    "2. What is the earliest date and time when they can make their camping reservation for the July 15, 2026 arrival at a Florida state park?\n"
    "3. What are the total entrance fees they will pay for their two visits to Delaware ocean state parks?\n\n"
    "For each answer, provide supporting reference URLs from official sources."
)

# Pre-computed expected values based on task description
EXPECTED_FL_ARRIVAL = datetime(2026, 7, 15)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FederalPassExtraction(BaseModel):
    pass_type: Optional[str] = None
    pass_cost: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class FloridaReservationExtraction(BaseModel):
    arrival_date: Optional[str] = None
    earliest_reservation_date: Optional[str] = None
    earliest_reservation_time: Optional[str] = None
    timezone: Optional[str] = None
    non_resident_booking_window_months: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class DelawareFeesExtraction(BaseModel):
    per_visit_fee: Optional[str] = None
    total_for_two_visits: Optional[str] = None
    fee_season_text: Optional[str] = None
    vehicle_registration_state: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class RecreationExtraction(BaseModel):
    federal: Optional[FederalPassExtraction] = None
    florida: Optional[FloridaReservationExtraction] = None
    delaware: Optional[DelawareFeesExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract from the answer the information for the three subtasks. Return a single JSON object with keys: federal, florida, delaware.

    1) federal:
       - pass_type: The name of the recommended federal recreation pass (e.g., "Senior Annual Pass", "Senior Lifetime Pass", "America the Beautiful Annual Pass"). Use the exact wording from the answer if possible.
       - pass_cost: The cost that the answer states for the recommended pass (e.g., "$20", "$80"). If multiple numbers are mentioned, choose the one tied to the recommended pass.
       - source_urls: All official URLs provided for eligibility and pricing (e.g., NPS/USGS official pass pages). Return an array of URLs. If none are provided, return an empty array.

    2) florida:
       - arrival_date: The arrival date for the Florida state park camping (as given in the answer, if explicitly repeated).
       - earliest_reservation_date: The earliest date the answer claims one can book for that arrival.
       - earliest_reservation_time: The time of day when reservations open (e.g., "8:00 a.m.").
       - timezone: The timezone mentioned for the opening time (e.g., "Eastern Time", "ET").
       - non_resident_booking_window_months: The advance booking window in months for non-Florida residents as stated in the answer (e.g., "10 months").
       - source_urls: All official Florida State Parks or ReserveAmerica URLs cited to support booking window/time. Return all URLs mentioned. If none are provided, return an empty array.

    3) delaware:
       - per_visit_fee: The per-visit entrance fee that applies to an out-of-state vehicle at Delaware ocean parks (as stated in the answer, e.g., "$10").
       - total_for_two_visits: The total amount for two separate visits (as stated in the answer).
       - fee_season_text: Any statement in the answer regarding the fee season (e.g., "March 1 - November 30").
       - vehicle_registration_state: The vehicle registration state mentioned for the visits (as stated in the answer).
       - source_urls: All official Delaware State Parks URLs provided to support fees/season. Return all URLs mentioned. If none are provided, return an empty array.

    IMPORTANT:
    - Extract only what is explicitly in the answer. Do not infer new URLs or values.
    - For URLs, extract the full URL (http/https). If the answer uses markdown links, extract the actual link targets.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def subtract_months(dt: datetime, months: int) -> datetime:
    y = dt.year
    m = dt.month - months
    # adjust year and month
    while m <= 0:
        m += 12
        y -= 1
    d = min(dt.day, [31,
                     29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return datetime(y, m, d)


def format_month_day_year(dt: datetime) -> str:
    return dt.strftime("%B %-d, %Y") if hasattr(dt, "strftime") else ""


def format_month_day_year_portable(dt: datetime) -> str:
    # Windows-compatible day formatting without leading zero:
    return dt.strftime("%B %d, %Y").replace(" 0", " ")


def pick_expected_senior_pass_cost_text(pass_type: Optional[str], stated_cost: Optional[str]) -> str:
    """
    Choose a cost assertion claim snippet based on pass_type and/or stated_cost.
    """
    pt = (pass_type or "").lower()
    cost = (stated_cost or "").lower()
    if "lifetime" in pt:
        return "The Senior Lifetime Pass costs $80."
    if "annual" in pt:
        return "The Senior Annual Pass costs $20."
    # Infer from cost text if possible
    if "20" in cost:
        return "The Senior Annual Pass costs $20."
    if "80" in cost:
        return "The Senior Lifetime Pass costs $80."
    # Fallback generic statement that covers both official prices
    return "Official pricing for the Senior Pass is $20 for the Annual Senior Pass and $80 for the Lifetime Senior Pass."


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_federal_subtree(evaluator: Evaluator, parent_node, federal: Optional[FederalPassExtraction]) -> None:
    # Top-level critical node for federal pass selection
    fed_node = evaluator.add_parallel(
        id="Federal_Pass_Selection",
        desc="Correctly identify and justify the most cost-effective annual federal recreation pass option for the couple",
        parent=parent_node,
        critical=True
    )

    # Sequential analysis under federal
    analysis_node = evaluator.add_sequential(
        id="Pass_Analysis",
        desc="Determine eligibility, identify the pass type, and specify the cost",
        parent=fed_node,
        critical=True
    )

    # Eligibility analysis (parallel checks)
    elig_node = evaluator.add_parallel(
        id="Eligibility_Analysis",
        desc="Determine pass eligibility based on the couple's profile (age 64, Maryland residents)",
        parent=analysis_node,
        critical=True
    )

    fed_sources = federal.source_urls if federal else []

    # Age requirement (62+) and couple is 64
    age_leaf = evaluator.add_leaf(
        id="Age_Requirement_Check",
        desc="Verify that at age 64, the couple meets the Senior Pass age requirement (62+)",
        parent=elig_node,
        critical=True
    )
    age_claim = (
        "The Senior Pass requires the pass holder to be age 62 or older, and since each member of the couple is 64, "
        "they meet the Senior Pass age requirement."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=fed_sources,
        additional_instruction="Verify against official NPS/USGS Senior Pass pages (e.g., usgs.gov/store, nps.gov). If no official source is provided, treat as not supported."
    )

    # Residency requirement: U.S. citizen or permanent resident
    residency_leaf = evaluator.add_leaf(
        id="Residency_Requirement_Check",
        desc="Verify that as U.S. residents, the couple meets the Senior Pass residency requirement",
        parent=elig_node,
        critical=True
    )
    residency_claim = (
        "The Senior Pass requires the pass holder to be a U.S. citizen or permanent resident; the couple are U.S. citizens, "
        "so they meet this requirement."
    )
    await evaluator.verify(
        claim=residency_claim,
        node=residency_leaf,
        sources=fed_sources,
        additional_instruction="Use an official Senior Pass page to confirm the residency/citizenship requirement."
    )

    # Pass identification (Senior Pass - Annual or Lifetime acceptable)
    pass_id_leaf = evaluator.add_leaf(
        id="Pass_Identification",
        desc="Identify the specific pass type that is most cost-effective (Senior Pass - either Lifetime or Annual)",
        parent=analysis_node,
        critical=True
    )
    pass_type_text = federal.pass_type if federal and federal.pass_type else ""
    pass_id_claim = f"The recommended pass type in the answer ('{pass_type_text}') is a Senior Pass (either the Annual Senior Pass or the Lifetime Senior Pass)."
    await evaluator.verify(
        claim=pass_id_claim,
        node=pass_id_leaf,
        additional_instruction="Judge based on the answer text: does the named pass clearly refer to the Senior Pass (annual or lifetime)? Allow reasonable naming variants. No URL needed here."
    )

    # Cost specification - verify price matches official
    cost_leaf = evaluator.add_leaf(
        id="Cost_Specification",
        desc="Provide the accurate cost ($80 for Lifetime or $20 for Annual)",
        parent=analysis_node,
        critical=True
    )
    cost_claim = pick_expected_senior_pass_cost_text(
        pass_type=federal.pass_type if federal else None,
        stated_cost=federal.pass_cost if federal else None
    )
    await evaluator.verify(
        claim=cost_claim,
        node=cost_leaf,
        sources=fed_sources,
        additional_instruction="Confirm the Senior Pass price(s) on an official Senior Pass page. Annual Senior is $20; Lifetime Senior is $80."
    )

    # Reference URL confirms pricing and eligibility
    ref_leaf = evaluator.add_leaf(
        id="Federal_Reference_URL",
        desc="Provide valid official source URL confirming Senior Pass pricing and eligibility",
        parent=fed_node,
        critical=True
    )
    ref_claim = (
        "The provided official source URL(s) confirm Senior Pass eligibility (62+ and U.S. citizen or permanent resident) "
        "and the official pricing ($20 for the Annual Senior Pass and $80 for the Lifetime Senior Pass)."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=fed_sources,
        additional_instruction="Only accept if at least one provided URL is an official NPS/USGS page that explicitly states both eligibility and pricing."
    )


async def build_florida_subtree(evaluator: Evaluator, parent_node, florida: Optional[FloridaReservationExtraction]) -> None:
    fl_node = evaluator.add_parallel(
        id="Florida_Camping_Reservation",
        desc="Determine the earliest date and time to make the Florida state park camping reservation",
        parent=parent_node,
        critical=True
    )

    # Sequential analysis: residency -> window -> date calc -> time
    fl_analysis = evaluator.add_sequential(
        id="Reservation_Timing_Analysis",
        desc="Analyze residency status, apply booking rules, and calculate the earliest reservation date",
        parent=fl_node,
        critical=True
    )

    fl_sources = florida.source_urls if florida else []

    # Residency determination
    fl_resid_leaf = evaluator.add_leaf(
        id="Residency_Determination",
        desc="Correctly identify the couple as non-Florida residents (Maryland residents)",
        parent=fl_analysis,
        critical=True
    )
    resid_claim = "Because they live in Maryland, they are non-Florida residents for Florida State Parks reservations."
    await evaluator.verify(
        claim=resid_claim,
        node=fl_resid_leaf,
        additional_instruction="This is a straightforward logical determination based on the task description."
    )

    # Booking window (non-resident = 10 months)
    fl_window_leaf = evaluator.add_leaf(
        id="Booking_Window_Application",
        desc="Apply the correct advance booking window (10 months for non-Florida residents)",
        parent=fl_analysis,
        critical=True
    )
    window_claim = "Florida State Parks allow non-Florida residents to make camping reservations up to 10 months in advance."
    await evaluator.verify(
        claim=window_claim,
        node=fl_window_leaf,
        sources=fl_sources,
        additional_instruction="Only accept if an official Florida State Parks or its official reservation platform page states the non-resident window is 10 months."
    )

    # Date calculation: arrival 2026-07-15 => earliest reservation date 10 months prior: 2025-09-15
    computed_earliest = subtract_months(EXPECTED_FL_ARRIVAL, 10)
    expected_earliest_str = format_month_day_year_portable(computed_earliest)
    fl_date_leaf = evaluator.add_leaf(
        id="Date_Calculation",
        desc="Calculate the earliest reservation date correctly (10 months before July 15, 2026)",
        parent=fl_analysis,
        critical=True
    )
    date_claim = f"Given a 10-month non-resident booking window, the earliest reservation date for an arrival on July 15, 2026 is {expected_earliest_str}."
    await evaluator.verify(
        claim=date_claim,
        node=fl_date_leaf,
        additional_instruction="Check the arithmetic: subtracting 10 months from July 15, 2026 yields September 15, 2025."
    )

    # Time information (opening time)
    # NOTE: The underlying verification tree requires children of a critical parent to be critical as well.
    # We therefore mark this as critical to satisfy framework constraints.
    fl_time_leaf = evaluator.add_leaf(
        id="Time_Information",
        desc="Include the time when reservations become available (8:00 a.m. Eastern Time)",
        parent=fl_analysis,
        critical=True
    )
    time_claim = "Florida State Parks campsite reservations open at 8:00 a.m. Eastern Time."
    await evaluator.verify(
        claim=time_claim,
        node=fl_time_leaf,
        sources=fl_sources,
        additional_instruction="Confirm opening time on an official Florida State Parks reservations policy page."
    )

    # Reference URL for Florida booking window/time
    fl_ref_leaf = evaluator.add_leaf(
        id="Florida_Reference_URL",
        desc="Provide valid Florida State Parks URL confirming the 10-month non-resident booking window",
        parent=fl_node,
        critical=True
    )
    fl_ref_claim = (
        "The provided official Florida State Parks URL(s) confirm that non-residents can book camping 10 months in advance "
        "and that reservations open at 8:00 a.m. Eastern Time."
    )
    await evaluator.verify(
        claim=fl_ref_claim,
        node=fl_ref_leaf,
        sources=fl_sources,
        additional_instruction="Only accept if at least one provided URL is an official Florida State Parks or official reservation platform page explicitly stating these rules."
    )


async def build_delaware_subtree(evaluator: Evaluator, parent_node, delaware: Optional[DelawareFeesExtraction]) -> None:
    de_node = evaluator.add_parallel(
        id="Delaware_Park_Fees",
        desc="Calculate total entrance fees for two Delaware ocean state park visits",
        parent=parent_node,
        critical=True
    )

    de_analysis = evaluator.add_sequential(
        id="Fee_Calculation_Analysis",
        desc="Determine vehicle registration status, apply fee structure, and calculate total fees",
        parent=de_node,
        critical=True
    )

    de_sources = delaware.source_urls if delaware else []

    # Registration status: Maryland-registered => out-of-state in Delaware
    de_reg_leaf = evaluator.add_leaf(
        id="Registration_Status",
        desc="Identify vehicle registration status (out-of-state, registered in Maryland)",
        parent=de_analysis,
        critical=True
    )
    reg_state = (delaware.vehicle_registration_state if delaware and delaware.vehicle_registration_state else "Maryland")
    reg_claim = f"A vehicle registered in {reg_state} is out-of-state in Delaware for entrance fee purposes."
    await evaluator.verify(
        claim=reg_claim,
        node=de_reg_leaf,
        additional_instruction="Simple logical determination based on different states."
    )

    # Fee structure: $10 for out-of-state at ocean parks (during fee season)
    de_fee_leaf = evaluator.add_leaf(
        id="Fee_Structure_Application",
        desc="Apply correct fee for out-of-state vehicles at Delaware ocean parks ($10 per visit)",
        parent=de_analysis,
        critical=True
    )
    fee_claim = "During the entrance fee season, Delaware ocean parks charge $10 per day for vehicles registered out of state."
    await evaluator.verify(
        claim=fee_claim,
        node=de_fee_leaf,
        sources=de_sources,
        additional_instruction="Confirm on an official Delaware State Parks page (destateparks.com) that ocean parks charge $10/day for out-of-state vehicles."
    )

    # Total calculation: two visits
    de_total_leaf = evaluator.add_leaf(
        id="Total_Calculation",
        desc="Calculate total fees correctly ($20 for two visits)",
        parent=de_analysis,
        critical=True
    )
    total_claim = "Two separate visits at $10 per visit result in a total of $20."
    await evaluator.verify(
        claim=total_claim,
        node=de_total_leaf,
        additional_instruction="Simple arithmetic check."
    )

    # Season awareness (fee season March 1 - November 30)
    # NOTE: Marked critical to satisfy framework constraint (children of critical parent must be critical).
    de_season_leaf = evaluator.add_leaf(
        id="Season_Awareness",
        desc="Acknowledge that fees apply during the specified fee season (March 1 - November 30)",
        parent=de_analysis,
        critical=True
    )
    season_claim = "Delaware State Parks' entrance fee season runs from March 1 through November 30."
    await evaluator.verify(
        claim=season_claim,
        node=de_season_leaf,
        sources=de_sources,
        additional_instruction="Confirm the fee season dates on an official Delaware State Parks page."
    )

    # Reference URL confirms $10 out-of-state ocean fee
    de_ref_leaf = evaluator.add_leaf(
        id="Delaware_Reference_URL",
        desc="Provide valid Delaware State Parks URL confirming the $10 fee for out-of-state vehicles at ocean parks",
        parent=de_node,
        critical=True
    )
    de_ref_claim = "The provided official Delaware State Parks URL(s) confirm the $10 per day ocean park fee for out-of-state vehicles."
    await evaluator.verify(
        claim=de_ref_claim,
        node=de_ref_leaf,
        sources=de_sources,
        additional_instruction="Only accept if at least one provided URL is an official Delaware State Parks page explicitly stating $10/day for out-of-state vehicles at ocean parks."
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
        default_model=model
    )

    # Create a critical wrapper node to reflect the rubric's top-level critical requirement
    main_node = evaluator.add_parallel(
        id="Recreation_Planning_Task",
        desc="Complete evaluation of the couple's optimal recreation access strategy",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=RecreationExtraction,
        extraction_name="structured_extraction"
    )

    # Add ground-truth expectations used for logical checks
    earliest_res_dt = subtract_months(EXPECTED_FL_ARRIVAL, 10)
    evaluator.add_ground_truth({
        "expected_florida_earliest_reservation_date": format_month_day_year_portable(earliest_res_dt),
        "expected_florida_open_time": "8:00 a.m. Eastern Time",
        "expected_delaware_ocean_out_of_state_fee": "$10 per day",
        "expected_delaware_two_visits_total": "$20"
    }, gt_type="expected_values")

    # Build and verify each sub-tree
    await build_federal_subtree(evaluator, main_node, extraction.federal or FederalPassExtraction())
    await build_florida_subtree(evaluator, main_node, extraction.florida or FloridaReservationExtraction())
    await build_delaware_subtree(evaluator, main_node, extraction.delaware or DelawareFeesExtraction())

    # Return final structured evaluation summary
    return evaluator.get_summary()