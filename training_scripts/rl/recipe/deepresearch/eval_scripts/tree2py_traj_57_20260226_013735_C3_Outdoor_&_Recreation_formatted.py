import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wa_backcountry_permit_planning_2026"
TASK_DESCRIPTION = (
    "A wilderness backpacker is planning a 3-night backcountry camping trip in Washington state for a group of 4 adults "
    "during June 15-18, 2026. They are evaluating whether to apply for permits at Mount Rainier National Park or Olympic "
    "National Park. Given the current date is February 26, 2026, determine which wilderness area's permit system to "
    "prioritize based on permit availability and reservation windows, specify the exact dates/times (and timezone) to "
    "submit application/reservation, calculate the total cost of required permits/fees for 4 people for 3 nights, and "
    "identify any additional mandatory requirements for the selected wilderness area."
)

CURRENT_DATE = "February 26, 2026"
TRIP_START = "June 15, 2026"
TRIP_END = "June 18, 2026"
NUM_ADULTS = 4
NUM_NIGHTS = 3

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AreaSelectionExtraction(BaseModel):
    selected_area: Optional[str] = None  # "Mount Rainier National Park" or "Olympic National Park"
    permit_system_type: Optional[str] = None  # "early access lottery", "regular lottery", or "standard reservation"
    system_sources: List[str] = Field(default_factory=list)

    window_start_datetime: Optional[str] = None  # include date and exact time and timezone, e.g., "March 1, 2026 8:00 AM PT"
    window_end_policy: Optional[str] = None  # "date" or "rolling"
    window_end_datetime: Optional[str] = None  # required if window_end_policy == "date"

    sources_window_start: List[str] = Field(default_factory=list)
    sources_window_end: List[str] = Field(default_factory=list)


class FeeExtraction(BaseModel):
    reservation_fee_amount: Optional[str] = None  # e.g., "$6"
    per_person_nightly_rate: Optional[str] = None  # e.g., "$8 per person per night"
    per_person_nightly_total: Optional[str] = None  # optional: agent's calculated subtotal for nightly fees
    total_cost: Optional[str] = None  # agent's final total for group and nights

    sources_reservation_fee: List[str] = Field(default_factory=list)
    sources_nightly_fee: List[str] = Field(default_factory=list)


class AdditionalRequirementsExtraction(BaseModel):
    equipment_requirement_statement: Optional[str] = None  # e.g., "bear canisters required", "snowshoes required"
    equipment_sources: List[str] = Field(default_factory=list)
    permit_allocation_info: Optional[str] = None  # e.g., "% of permits via lottery vs walk-in"
    allocation_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_area_selection() -> str:
    return (
        "From the answer, extract the selected wilderness area (either 'Mount Rainier National Park' or 'Olympic National Park'), "
        "the applicable permit system type (use one of exactly these strings: 'early access lottery', 'regular lottery', or 'standard reservation'), "
        "and the application/reservation window details (start and end).\n"
        "Requirements:\n"
        "- selected_area: the chosen park's name exactly as stated in the answer text.\n"
        "- permit_system_type: one of the exact strings listed above, matching what the answer states for the selected area.\n"
        "- window_start_datetime: the exact opening datetime including timezone abbreviation as stated (e.g., 'March 1, 2026 8:00 AM PT'). If only a date is provided, include that date; if time/timezone is missing, still extract what is provided.\n"
        "- window_end_policy: use 'date' if a specific closing date is given; otherwise use 'rolling' if reservations continue with ongoing/rolling availability.\n"
        "- window_end_datetime: if window_end_policy is 'date', provide the exact closing date/time (including timezone if given). If 'rolling', set this to null.\n"
        "- system_sources: all URLs in the answer that support the permit system type for the selected area.\n"
        "- sources_window_start: all URLs in the answer that support the start datetime.\n"
        "- sources_window_end: all URLs in the answer that support the end date or rolling availability.\n"
        "Return a single JSON object with these fields. Only include URLs explicitly present in the answer text. If any field is missing, return null or an empty array as appropriate."
    )


def prompt_extract_fee_info() -> str:
    return (
        "From the answer, extract the permit/fee costs and sources for the selected wilderness area.\n"
        "Fields to extract:\n"
        "- reservation_fee_amount: the one-time reservation/application fee amount (e.g., '$6').\n"
        "- per_person_nightly_rate: the per-person-per-night camping fee rate (e.g., '$8 per person per night').\n"
        "- per_person_nightly_total: if the answer provides a computed subtotal for nightly fees for the group (4 adults × 3 nights), extract it verbatim; otherwise return null.\n"
        "- total_cost: if the answer provides a final total cost for all permits/fees for the entire group and trip, extract it verbatim; otherwise return null.\n"
        "- sources_reservation_fee: URLs cited in the answer that support the reservation/app fee amount.\n"
        "- sources_nightly_fee: URLs cited that support the per-person-per-night rate.\n"
        "Return a single JSON object with these fields. If a field is not present in the answer, return null or an empty array."
    )


def prompt_extract_additional_requirements() -> str:
    return (
        "From the answer, extract any additional mandatory requirements and supporting sources for the selected wilderness area.\n"
        "Fields:\n"
        "- equipment_requirement_statement: any stated mandatory gear or seasonal equipment (e.g., 'bear canister required', 'snowshoes required', 'ice axe required', etc.). If nothing is stated, return null.\n"
        "- equipment_sources: URLs supporting the equipment requirement.\n"
        "- permit_allocation_info: any information given about what percentage of permits are available through the identified system versus walk-in or other mechanisms; if not provided, return null.\n"
        "- allocation_sources: URLs supporting the allocation info.\n"
        "Return a single JSON object with these fields. Only include URLs explicitly present in the answer text."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_money_to_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Extract the first float-like number
    m = re.search(r"([0-9]+(?:\.[0-9]{1,2})?)", text.replace(",", ""))
    try:
        return float(m.group(1)) if m else None
    except Exception:
        return None


def format_money(amount: Optional[float]) -> str:
    if amount is None:
        return "unknown"
    return f"${amount:.2f}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_area_selection_verifications(
    evaluator: Evaluator,
    parent_node,
    area: AreaSelectionExtraction,
) -> None:
    # Wilderness Area Identification (Critical, Sequential)
    area_id_node = evaluator.add_sequential(
        id="WildernessAreaIdentification",
        desc="Correctly identify which wilderness area's permit system should be used based on the specified trip dates and reservation windows",
        parent=parent_node,
        critical=True,
    )

    # Permit System Type (Critical, Sequential)
    system_node = evaluator.add_sequential(
        id="PermitSystemType",
        desc="Correctly identify the type of permit system (early access lottery, regular lottery, or standard reservation) applicable to the selected wilderness area",
        parent=area_id_node,
        critical=True,
    )

    # Leaf: Verify the stated permit system type for the selected area against sources
    system_type_leaf = evaluator.add_leaf(
        id="SystemTypeValue",
        desc="Selected area's permit system type matches the stated system type",
        parent=system_node,
        critical=True,
    )
    system_claim = (
        f"For {area.selected_area}, the applicable wilderness/backcountry permit system type is "
        f"\"{area.permit_system_type}\"."
    )
    await evaluator.verify(
        claim=system_claim,
        node=system_type_leaf,
        sources=area.system_sources if area.system_sources else None,
        additional_instruction=(
            "Confirm the permit system type for the selected area's backpacking/wilderness permits for the 2026 season. "
            "Accept reasonable naming variants (e.g., 'early access' vs 'early-access lottery')."
        ),
    )

    # Application Window (Critical, Sequential)
    window_node = evaluator.add_sequential(
        id="ApplicationWindow",
        desc="Provide complete information about when and how to submit the application or reservation",
        parent=system_node,
        critical=True,
    )

    # DateTimeSpecification (Critical, Parallel)
    dt_spec_node = evaluator.add_parallel(
        id="DateTimeSpecification",
        desc="Specify the exact dates and times when applications or reservations can be submitted",
        parent=window_node,
        critical=True,
    )

    # WindowStartDate (Critical, Sequential)
    start_node = evaluator.add_sequential(
        id="WindowStartDate",
        desc="Correctly specify the exact start date and time (including timezone) when the permit application or reservation window opens",
        parent=dt_spec_node,
        critical=True,
    )

    # Leaf: Start date/time verification
    start_leaf = evaluator.add_leaf(
        id="WindowStartDateValue",
        desc="Window start date/time (with timezone) is correct per sources",
        parent=start_node,
        critical=True,
    )
    start_claim = (
        f"The application/reservation window for {area.selected_area} opens on {area.window_start_datetime} "
        f"(timezone included if stated)."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=area.sources_window_start if area.sources_window_start else None,
        additional_instruction=(
            "Verify the exact opening date and time, including the timezone if given, for the stated permit system. "
            "Match what the official source indicates."
        ),
    )

    # Leaf: Start date reference existence (custom check)
    evaluator.add_custom_node(
        result=bool(area.sources_window_start),
        id="StartDateReference",
        desc="Provide URL reference supporting the start date and time",
        parent=start_node,
        critical=True,
    )

    # WindowEndDate (Critical, Sequential)
    end_node = evaluator.add_sequential(
        id="WindowEndDate",
        desc="Correctly specify the end date of the application window or indicate if it operates on rolling availability",
        parent=dt_spec_node,
        critical=True,
    )

    # Leaf: End date or rolling verification
    end_leaf = evaluator.add_leaf(
        id="WindowEndDateValue",
        desc="Window end date or rolling availability mechanism is correct per sources",
        parent=end_node,
        critical=True,
    )
    if (area.window_end_policy or "").lower() == "rolling":
        end_claim = (
            "The application/reservation operates on rolling availability rather than a fixed end date."
        )
    else:
        end_claim = (
            f"The application/reservation window for {area.selected_area} closes on {area.window_end_datetime}."
        )
    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        sources=area.sources_window_end if area.sources_window_end else None,
        additional_instruction=(
            "If rolling availability is stated, confirm the source indicates ongoing/rolling openings. "
            "If a fixed closing date/time is stated, confirm that exact date/time matches the source."
        ),
    )

    # Leaf: End date reference existence (custom check)
    evaluator.add_custom_node(
        result=bool(area.sources_window_end),
        id="EndDateReference",
        desc="Provide URL reference supporting the end date or rolling availability mechanism",
        parent=end_node,
        critical=True,
    )


async def build_cost_verifications(
    evaluator: Evaluator,
    parent_node,
    fee: FeeExtraction,
) -> None:
    # CostCalculation (Critical, Sequential)
    cost_node = evaluator.add_sequential(
        id="CostCalculation",
        desc="Calculate the total cost of all required permits and fees for the trip",
        parent=parent_node,
        critical=True,
    )

    # PermitFeeBreakdown (Critical, Sequential)
    breakdown_node = evaluator.add_sequential(
        id="PermitFeeBreakdown",
        desc="Provide correct itemized breakdown of all permit fees and calculate total",
        parent=cost_node,
        critical=True,
    )

    # FeeComponentsAndTotal (Critical, Sequential)
    components_total_node = evaluator.add_sequential(
        id="FeeComponentsAndTotal",
        desc="Identify individual fee components and sum them to total cost",
        parent=breakdown_node,
        critical=True,
    )

    # IndividualFeeComponents (Critical, Parallel)
    components_node = evaluator.add_parallel(
        id="IndividualFeeComponents",
        desc="Identify and correctly calculate all individual fee components",
        parent=components_total_node,
        critical=True,
    )

    # ReservationFee (Critical, Sequential)
    reservation_node = evaluator.add_sequential(
        id="ReservationFee",
        desc="Correctly identify and state the one-time reservation or application fee amount",
        parent=components_node,
        critical=True,
    )

    # Leaf: Reservation fee amount verification
    reservation_leaf = evaluator.add_leaf(
        id="ReservationFeeValue",
        desc="Reservation/application fee amount is correct per sources",
        parent=reservation_node,
        critical=True,
    )
    reservation_claim = f"The reservation/application fee amount is {fee.reservation_fee_amount}."
    await evaluator.verify(
        claim=reservation_claim,
        node=reservation_leaf,
        sources=fee.sources_reservation_fee if fee.sources_reservation_fee else None,
        additional_instruction="Verify the exact dollar amount of the reservation/application fee from official sources.",
    )

    # Leaf: Reservation fee reference existence (custom)
    evaluator.add_custom_node(
        result=bool(fee.sources_reservation_fee),
        id="ReservationFeeReference",
        desc="Provide URL reference supporting the reservation fee amount",
        parent=reservation_node,
        critical=True,
    )

    # PerPersonNightlyFees (Critical, Sequential)
    nightly_node = evaluator.add_sequential(
        id="PerPersonNightlyFees",
        desc="Correctly calculate per-person camping fees: number of adults × number of nights × per-person-per-night rate",
        parent=components_node,
        critical=True,
    )

    # Leaf: Nightly fee rate verification
    nightly_rate_leaf = evaluator.add_leaf(
        id="NightlyFeeRateValue",
        desc="Per-person-per-night fee rate is correct per sources",
        parent=nightly_node,
        critical=True,
    )
    nightly_rate_claim = f"The per-person-per-night camping fee rate is {fee.per_person_nightly_rate}."
    await evaluator.verify(
        claim=nightly_rate_claim,
        node=nightly_rate_leaf,
        sources=fee.sources_nightly_fee if fee.sources_nightly_fee else None,
        additional_instruction="Verify the per-person-per-night wilderness camping fee rate as stated.",
    )

    # Leaf: Nightly fee reference existence (custom)
    evaluator.add_custom_node(
        result=bool(fee.sources_nightly_fee),
        id="NightlyFeeReference",
        desc="Provide URL reference supporting the per-person-per-night fee rate",
        parent=nightly_node,
        critical=True,
    )

    # Leaf: Per-person nightly fees total calculation verification
    nightly_total_leaf = evaluator.add_leaf(
        id="PerPersonNightlyFeesCalculated",
        desc="Computed total nightly fees for 4 adults × 3 nights is correct given the stated rate",
        parent=nightly_node,
        critical=True,
    )
    # Prepare arithmetic details for verification prompt
    rate_val = parse_money_to_float(fee.per_person_nightly_rate)
    expected_nightly_total = (rate_val or 0.0) * NUM_ADULTS * NUM_NIGHTS
    provided_nightly_total = fee.per_person_nightly_total or format_money(expected_nightly_total)
    nightly_total_claim = (
        f"Given {NUM_ADULTS} adults over {NUM_NIGHTS} nights at a per-person-per-night rate of "
        f"{fee.per_person_nightly_rate}, the total per-person nightly fees equals {provided_nightly_total}. "
        f"Confirm the arithmetic using the supported rate."
    )
    await evaluator.verify(
        claim=nightly_total_claim,
        node=nightly_total_leaf,
        sources=fee.sources_nightly_fee if fee.sources_nightly_fee else None,
        additional_instruction="Perform the multiplication using the rate; small rounding differences are acceptable.",
    )

    # TotalCost (Critical leaf): Verify final total equals reservation fee + nightly fees for group
    total_leaf = evaluator.add_leaf(
        id="TotalCost",
        desc="Provide accurate total cost by summing reservation fee and per-person nightly fees",
        parent=components_total_node,
        critical=True,
    )
    reservation_val = parse_money_to_float(fee.reservation_fee_amount)
    total_cost_claim = (
        f"For {NUM_ADULTS} adults and {NUM_NIGHTS} nights: nightly fees subtotal is "
        f"{provided_nightly_total}; adding reservation/app fee {fee.reservation_fee_amount} yields a final total of "
        f"{fee.total_cost}. Confirm this total using the official fee sources."
    )
    combined_sources = list(set((fee.sources_reservation_fee or []) + (fee.sources_nightly_fee or [])))
    await evaluator.verify(
        claim=total_cost_claim,
        node=total_leaf,
        sources=combined_sources if combined_sources else None,
        additional_instruction=(
            "Verify that total = (per-person nightly rate × number of adults × number of nights) + reservation fee. "
            "Accept reasonable rounding."
        ),
    )


async def build_additional_requirements_verifications(
    evaluator: Evaluator,
    parent_node,
    addl: AdditionalRequirementsExtraction,
) -> None:
    # AdditionalRequirements (Non-Critical, Parallel)
    addl_node = evaluator.add_parallel(
        id="AdditionalRequirements",
        desc="Identify any additional mandatory requirements such as equipment regulations or permit allocation information",
        parent=parent_node,
        critical=False,
    )

    equipment_leaf = evaluator.add_leaf(
        id="EquipmentRequirement",
        desc="State whether snowshoes, skis, or other equipment are required based on the area's regulations for the specified trip dates",
        parent=addl_node,
        critical=False,
    )
    equipment_claim = (
        f"Equipment requirement: {addl.equipment_requirement_statement or 'No specific mandatory equipment stated'}."
    )
    await evaluator.verify(
        claim=equipment_claim,
        node=equipment_leaf,
        sources=addl.equipment_sources if addl.equipment_sources else None,
        additional_instruction=(
            "Verify whether the stated mandatory equipment requirement is correct for the selected area and season. "
            "If no statement is provided, confirm that the sources do not indicate a mandatory requirement relevant to mid-June trips."
        ),
    )

    allocation_leaf = evaluator.add_leaf(
        id="PermitAllocationInfo",
        desc="Provide information about what percentage of permits are available through the identified system versus walk-in or other mechanisms",
        parent=addl_node,
        critical=False,
    )
    allocation_claim = (
        f"Permit allocation information: {addl.permit_allocation_info or 'No allocation percentage provided'}."
    )
    await evaluator.verify(
        claim=allocation_claim,
        node=allocation_leaf,
        sources=addl.allocation_sources if addl.allocation_sources else None,
        additional_instruction=(
            "Verify the stated allocation (e.g., % via lottery/reservations vs walk-in). "
            "If none is provided, check sources for any explicit allocation data."
        ),
    )


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the WA backcountry permit planning task.
    """
    # Initialize evaluator (Root node: sequential but NON-CRITICAL to allow non-critical branches later)
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

    # Record trip parameters as custom info
    evaluator.add_custom_info(
        info={
            "current_date": CURRENT_DATE,
            "trip_start": TRIP_START,
            "trip_end": TRIP_END,
            "num_adults": NUM_ADULTS,
            "num_nights": NUM_NIGHTS,
        },
        info_type="trip_parameters",
    )

    # Extract structured information from the answer
    area_extraction = await evaluator.extract(
        prompt=prompt_extract_area_selection(),
        template_class=AreaSelectionExtraction,
        extraction_name="area_selection",
    )

    fee_extraction = await evaluator.extract(
        prompt=prompt_extract_fee_info(),
        template_class=FeeExtraction,
        extraction_name="fee_info",
    )

    addl_extraction = await evaluator.extract(
        prompt=prompt_extract_additional_requirements(),
        template_class=AdditionalRequirementsExtraction,
        extraction_name="additional_requirements",
    )

    # Build verification tree
    await build_area_selection_verifications(evaluator, root, area_extraction)
    await build_cost_verifications(evaluator, root, fee_extraction)
    await build_additional_requirements_verifications(evaluator, root, addl_extraction)

    # Return structured summary
    return evaluator.get_summary()