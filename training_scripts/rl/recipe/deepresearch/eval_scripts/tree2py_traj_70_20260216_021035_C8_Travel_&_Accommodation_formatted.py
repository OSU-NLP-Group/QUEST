import asyncio
import logging
import calendar
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "yellowstone_ada_pet_2026"
TASK_DESCRIPTION = (
    "A family of non-US residents consisting of 3 adults and 1 teenager (age 16) is planning to visit Yellowstone "
    "National Park for 4 consecutive nights in July 2026. The family is traveling with one dog and requires "
    "wheelchair-accessible accommodations. They want to book a Western Cabin at Canyon Lodge inside Yellowstone "
    "National Park that is both pet-friendly and ADA-compliant. Based on the 2026 policies and fee structures, provide "
    "the following information: (1) Confirm that Canyon Lodge offers Western Cabins that are both pet-friendly and "
    "wheelchair-accessible, and list the required ADA accessibility features that must be present (roll-in shower, "
    "grab bars, accessible sink, doorway clear width); (2) Calculate the total pet fee for the 4-night stay in July "
    "2026, using the correct fee structure that will be in effect at that time; (3) State the cost of purchasing an "
    "America the Beautiful Annual Pass for a non-US resident; (4) Calculate the total nonresident surcharge fees for "
    "park entrance that will apply to this family (3 adults and 1 teenager aged 16); (5) Determine the earliest date "
    "when this family can make a reservation for a stay in July 2026, based on Yellowstone National Park Lodges' "
    "advance reservation booking window that opens 13 months ahead on the 5th of each month; (6) State the advance "
    "deposit amount required to secure the reservation. Provide specific dollar amounts, dates, and feature "
    "descriptions, with supporting reference URLs."
)

# Ground truth references for clarity in summary (not used for scoring directly)
GROUND_TRUTH = {
    "ada_required_features": [
        "roll-in shower",
        "grab bars",
        "accessible sink",
        "doorway clear width (at least 32 inches)"
    ],
    "pet_fee_rate_per_pet_per_night": "$40",
    "pet_fee_effective_date": "April 24, 2026",
    "pet_fee_total_for_4_nights_1_pet": "$160",
    "annual_pass_nonresident_price": "$250",
    "annual_pass_effective_date": "January 1, 2026",
    "nonresident_surcharge_per_person_16+": "$100",
    "family_eligible_count": "4",
    "nonresident_surcharge_total": "$400",
    "booking_window_rule": "Reservations open 13 months in advance on the 5th of each month",
    "deposit_policy": "Advance deposit equals the first night's rate at each location",
    "max_pets_per_cabin": "2"
}


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class LodgingExtraction(BaseModel):
    accommodation_type: Optional[str] = None
    pet_friendly_status: Optional[str] = None
    accessibility_status: Optional[str] = None
    ada_features: List[str] = Field(default_factory=list)
    max_pets: Optional[str] = None
    lodging_sources: List[str] = Field(default_factory=list)
    accessibility_sources: List[str] = Field(default_factory=list)


class PetFeeInfo(BaseModel):
    rate_per_pet_per_night: Optional[str] = None
    effective_date: Optional[str] = None
    nights_used_in_calculation: Optional[str] = None
    total_fee_stated: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AnnualPassInfo(BaseModel):
    price_nonresident: Optional[str] = None
    effective_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SurchargeInfo(BaseModel):
    per_person_amount_16_plus: Optional[str] = None
    eligible_count_applied: Optional[str] = None
    total_surcharge_stated: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BookingWindowInfo(BaseModel):
    rule_description: Optional[str] = None
    earliest_booking_date_for_july_2026: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DepositInfo(BaseModel):
    deposit_policy: Optional[str] = None
    payment_methods: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_lodging_info() -> str:
    return (
        "From the answer, extract details about Canyon Lodge Western Cabins:\n"
        "1. accommodation_type: The accommodation explicitly identified (e.g., 'Western Cabin at Canyon Lodge').\n"
        "2. pet_friendly_status: Whether these cabins are pet-friendly (use the exact phrasing from the answer; if not stated, null).\n"
        "3. accessibility_status: Whether these cabins are wheelchair-accessible or ADA-compliant (exact phrasing; if not stated, null).\n"
        "4. ada_features: List all ADA features mentioned for the cabin (normalize terms: 'roll-in shower', 'grab bars', "
        "'accessible sink', 'doorway clear width'). Include any variants present in the answer as best-match normalized terms.\n"
        "5. max_pets: The maximum number of pets allowed per cabin (as a string; if not stated, null).\n"
        "6. lodging_sources: All URLs in the answer that support lodging availability, pet-friendly status, or pet policies.\n"
        "7. accessibility_sources: All URLs in the answer that support accessibility/ADA features.\n"
        "Return null for any field not present in the answer. Extract only URLs explicitly present."
    )


def prompt_extract_pet_fee_info() -> str:
    return (
        "From the answer, extract the pet fee details applicable to July 2026:\n"
        "1. rate_per_pet_per_night: The stated fee per pet per night (e.g., '$40').\n"
        "2. effective_date: The date this fee becomes effective (e.g., 'April 24, 2026').\n"
        "3. nights_used_in_calculation: The number of nights used in calculating the total (e.g., '4').\n"
        "4. total_fee_stated: The final total pet fee stated for the trip (e.g., '$160').\n"
        "5. sources: All URLs supporting the pet fee rate and policy.\n"
        "Return null for any field not present in the answer."
    )


def prompt_extract_annual_pass_info() -> str:
    return (
        "From the answer, extract the America the Beautiful Annual Pass pricing for non-US residents:\n"
        "1. price_nonresident: The stated price for non-US residents (e.g., '$250').\n"
        "2. effective_date: The stated effective date (e.g., 'January 1, 2026').\n"
        "3. sources: All URLs supporting this pricing.\n"
        "Return null for any field not present in the answer."
    )


def prompt_extract_surcharge_info() -> str:
    return (
        "From the answer, extract the nonresident surcharge details for park entrance:\n"
        "1. per_person_amount_16_plus: The surcharge per person aged 16+ (e.g., '$100').\n"
        "2. eligible_count_applied: The number of people counted as 16+ in the calculation (e.g., '4').\n"
        "3. total_surcharge_stated: The final total surcharge stated (e.g., '$400').\n"
        "4. sources: All URLs supporting the surcharge policy and amounts.\n"
        "Return null for any field not present in the answer."
    )


def prompt_extract_booking_window_info() -> str:
    return (
        "From the answer, extract the reservation booking window details for Yellowstone National Park Lodges:\n"
        "1. rule_description: The booking window rule (e.g., 'Reservations open 13 months in advance on the 5th of each month').\n"
        "2. earliest_booking_date_for_july_2026: The specific earliest reservation date for a stay in July 2026 (e.g., 'June 5, 2025').\n"
        "3. sources: All URLs supporting the booking window information.\n"
        "Return null for any field not present in the answer."
    )


def prompt_extract_deposit_info() -> str:
    return (
        "From the answer, extract the advance deposit policy details:\n"
        "1. deposit_policy: The stated policy for deposit (e.g., 'Advance deposit equals the first night's rate at each location').\n"
        "2. payment_methods: The accepted payment methods for advance deposits (list strings; e.g., 'Visa', 'MasterCard', 'AMEX', 'Discover').\n"
        "3. sources: All URLs supporting the deposit policy and payment methods.\n"
        "Return null for any field not present in the answer."
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


def compute_earliest_booking_date(month: int, year: int) -> str:
    # Lodges open 13 months in advance on the 5th of month
    # July 2026 -> June 5, 2025
    open_month = month - 1
    open_year = year - 1
    month_name = calendar.month_name[open_month]
    return f"{month_name} 5, {open_year}"


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def build_lodging_identification(
    evaluator: Evaluator,
    parent: Any,
    lodging: LodgingExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Lodging_Identification",
        desc="Confirms that Canyon Lodge Western Cabins are identified as both pet-friendly and wheelchair-accessible",
        parent=parent,
        critical=True
    )

    # Reference URL existence (critical sibling prerequisite)
    lodging_sources_exist = bool(lodging.lodging_sources)
    evaluator.add_custom_node(
        result=lodging_sources_exist,
        id="Reference_URL_Lodging",
        desc="Provides at least one valid reference URL for Canyon Lodge information",
        parent=node,
        critical=True
    )

    # Western Cabins wheelchair-accessible
    leaf_accessible = evaluator.add_leaf(
        id="Western_Cabins_Confirmed",
        desc="States that Canyon Lodge has Western Cabins that are wheelchair-accessible",
        parent=node,
        critical=True
    )
    claim_accessible = (
        "Canyon Lodge inside Yellowstone National Park offers Western Cabins and these include wheelchair-accessible "
        "or ADA-compliant options."
    )
    await evaluator.verify(
        claim=claim_accessible,
        node=leaf_accessible,
        sources=combine_sources(lodging.lodging_sources, lodging.accessibility_sources),
        additional_instruction=(
            "Verify using official Yellowstone National Park Lodges pages that Western Cabins exist at Canyon Lodge "
            "and that ADA/accessible cabins are available. Accept reasonable synonyms like 'wheelchair accessible', "
            "'mobility accessible', or 'ADA-compliant'."
        ),
    )

    # Pet-friendly status
    leaf_pet = evaluator.add_leaf(
        id="Pet_Friendly_Status",
        desc="Confirms that Western Cabins at Canyon Lodge are pet-friendly",
        parent=node,
        critical=True
    )
    claim_pet = "Western Cabins at Canyon Lodge are pet-friendly (dogs allowed in designated pet-friendly accommodations)."
    await evaluator.verify(
        claim=claim_pet,
        node=leaf_pet,
        sources=lodging.lodging_sources,
        additional_instruction=(
            "Use the official lodges pet policy pages. Allow variations such as 'pet-friendly rooms/cabins' and "
            "designated pet accommodations."
        ),
    )

    # Max pets policy (make critical to satisfy framework constraint)
    leaf_max_pets = evaluator.add_leaf(
        id="Maximum_Pets_Policy",
        desc="States the maximum of 2 pets allowed per cabin",
        parent=node,
        critical=True
    )
    claim_max_pets = "The maximum number of pets allowed per cabin or room in pet-friendly lodging is 2."
    await evaluator.verify(
        claim=claim_max_pets,
        node=leaf_max_pets,
        sources=lodging.lodging_sources,
        additional_instruction=(
            "Accept synonyms including 'per room', 'per unit', or 'per accommodation'. Verify from Yellowstone Lodges pet policy."
        ),
    )


async def build_accessibility_features(
    evaluator: Evaluator,
    parent: Any,
    lodging: LodgingExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Accessibility_Features",
        desc="Lists all required ADA accessibility features that must be present in the cabin",
        parent=parent,
        critical=True
    )

    # Reference URL existence (critical sibling prerequisite)
    acc_sources_exist = bool(lodging.accessibility_sources)
    evaluator.add_custom_node(
        result=acc_sources_exist,
        id="Reference_URL_Accessibility",
        desc="Provides at least one valid reference URL for accessibility features information",
        parent=node,
        critical=True
    )

    # Roll-in shower
    roll_leaf = evaluator.add_leaf(
        id="Roll_In_Shower",
        desc="Mentions roll-in shower as an accessibility feature",
        parent=node,
        critical=True
    )
    claim_roll = "Accessible/ADA cabins include a roll-in shower."
    await evaluator.verify(
        claim=claim_roll,
        node=roll_leaf,
        sources=lodging.accessibility_sources,
        additional_instruction="Verify from official accessibility details for Canyon Lodge/Western Cabins.",
    )

    # Grab bars
    bars_leaf = evaluator.add_leaf(
        id="Grab_Bars",
        desc="Mentions grab bars as an accessibility feature",
        parent=node,
        critical=True
    )
    claim_bars = "Accessible/ADA cabins include grab bars in the bathroom."
    await evaluator.verify(
        claim=claim_bars,
        node=bars_leaf,
        sources=lodging.accessibility_sources,
        additional_instruction="Verify from official accessibility details for Canyon Lodge/Western Cabins.",
    )

    # Accessible sink
    sink_leaf = evaluator.add_leaf(
        id="Accessible_Sink",
        desc="Mentions accessible sink as an accessibility feature",
        parent=node,
        critical=True
    )
    claim_sink = "Accessible/ADA cabins include an accessible sink (e.g., knee clearance)."
    await evaluator.verify(
        claim=claim_sink,
        node=sink_leaf,
        sources=lodging.accessibility_sources,
        additional_instruction="Verify from official accessibility details for Canyon Lodge/Western Cabins.",
    )

    # Doorway width
    door_leaf = evaluator.add_leaf(
        id="Doorway_Width",
        desc="Mentions doorway clear width requirement (at least 32 inches) as an accessibility feature",
        parent=node,
        critical=True
    )
    claim_door = "Accessible/ADA cabins have doorways with at least 32 inches of clear opening width."
    await evaluator.verify(
        claim=claim_door,
        node=door_leaf,
        sources=lodging.accessibility_sources,
        additional_instruction="Verify from official accessibility details; accept reasonable phrasing indicating ≥32 inches.",
    )


async def build_pet_fee_calculation(
    evaluator: Evaluator,
    parent: Any,
    pet_fee: PetFeeInfo
) -> None:
    node = evaluator.add_sequential(
        id="Pet_Fee_Calculation",
        desc="Calculates the total pet fee for the 4-night stay in July 2026 using the correct fee structure",
        parent=parent,
        critical=True
    )

    # Reference URL existence (critical sibling prerequisite within this section)
    evaluator.add_custom_node(
        result=bool(pet_fee.sources),
        id="Reference_URL_Pet_Fee",
        desc="Provides at least one valid reference URL for pet fee information",
        parent=node,
        critical=True
    )

    # Correct fee rate applied
    rate_leaf = evaluator.add_leaf(
        id="Correct_Fee_Rate_Applied",
        desc="Uses the $40 per pet per night rate that is effective April 24, 2026 (applicable to July 2026 stay)",
        parent=node,
        critical=True
    )
    claim_rate = "The pet fee is $40 per pet per night, effective April 24, 2026."
    await evaluator.verify(
        claim=claim_rate,
        node=rate_leaf,
        sources=pet_fee.sources,
        additional_instruction="Confirm from Yellowstone Lodges pet policy; July 2026 is after this effective date.",
    )

    # Correct duration applied (verify against the answer context)
    duration_leaf = evaluator.add_leaf(
        id="Correct_Duration_Applied",
        desc="Applies the fee calculation to 4 nights",
        parent=node,
        critical=True
    )
    claim_duration = "The fee calculation uses 4 nights for the stay."
    await evaluator.verify(
        claim=claim_duration,
        node=duration_leaf,
        sources=None,
        additional_instruction="Check the answer's arithmetic and stated duration; it should use 4 nights.",
    )

    # Total pet fee correct (arithmetic check)
    total_leaf = evaluator.add_leaf(
        id="Total_Pet_Fee_Correct",
        desc="Calculates the total pet fee as $160 ($40 × 4 nights)",
        parent=node,
        critical=True
    )
    claim_total = "The total pet fee is $160 for one pet over 4 nights at $40 per pet per night."
    await evaluator.verify(
        claim=claim_total,
        node=total_leaf,
        sources=None,
        additional_instruction="Simple arithmetic verification based on the provided rate and duration.",
    )


async def build_annual_pass_cost(
    evaluator: Evaluator,
    parent: Any,
    apass: AnnualPassInfo
) -> None:
    node = evaluator.add_parallel(
        id="Annual_Pass_Cost",
        desc="States the cost of the America the Beautiful Annual Pass for non-US residents",
        parent=parent,
        critical=True
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(apass.sources),
        id="Reference_URL_Pass",
        desc="Provides at least one valid reference URL for annual pass pricing",
        parent=node,
        critical=True
    )

    price_leaf = evaluator.add_leaf(
        id="Nonresident_Pass_Price",
        desc="States the annual pass cost as $250 for non-US residents",
        parent=node,
        critical=True
    )
    claim_price = "For non-US residents, the America the Beautiful Annual Pass costs $250."
    await evaluator.verify(
        claim=claim_price,
        node=price_leaf,
        sources=apass.sources,
        additional_instruction="Verify using official DOI/NPS or authorized sales pages reflecting 2026 pricing.",
    )

    effective_leaf = evaluator.add_leaf(
        id="Effective_Date_Noted",
        desc="Notes that this pricing is effective January 1, 2026",
        parent=node,
        critical=True
    )
    claim_effective = "This $250 pricing is effective January 1, 2026."
    await evaluator.verify(
        claim=claim_effective,
        node=effective_leaf,
        sources=apass.sources,
        additional_instruction="Confirm the effective date from official sources.",
    )


async def build_nonresident_surcharge(
    evaluator: Evaluator,
    parent: Any,
    surcharge: SurchargeInfo
) -> None:
    node = evaluator.add_sequential(
        id="Nonresident_Surcharge_Calculation",
        desc="Calculates the total nonresident surcharge for park entrance for the family",
        parent=parent,
        critical=True
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(surcharge.sources),
        id="Reference_URL_Surcharge",
        desc="Provides at least one valid reference URL for nonresident surcharge information",
        parent=node,
        critical=True
    )

    per_person_leaf = evaluator.add_leaf(
        id="Per_Person_Surcharge",
        desc="States the per-person surcharge as $100 for visitors aged 16+",
        parent=node,
        critical=True
    )
    claim_per_person = "A nonresident surcharge of $100 per person applies to visitors aged 16 and over."
    await evaluator.verify(
        claim=claim_per_person,
        node=per_person_leaf,
        sources=surcharge.sources,
        additional_instruction="Verify from official policy pages specifying surcharge for non-residents aged 16+.",
    )

    # Family composition applied (answer-based check)
    family_leaf = evaluator.add_leaf(
        id="Family_Composition_Applied",
        desc="Identifies that 4 people in the family are aged 16+ (3 adults and 1 teenager age 16)",
        parent=node,
        critical=True
    )
    claim_family = "The surcharge calculation correctly applies to 4 people aged 16 or older (3 adults and one 16-year-old teenager)."
    await evaluator.verify(
        claim=claim_family,
        node=family_leaf,
        sources=None,
        additional_instruction="Verify using the task description and answer context.",
    )

    total_leaf = evaluator.add_leaf(
        id="Total_Surcharge_Calculated",
        desc="Calculates the total surcharge as $400 ($100 × 4 people)",
        parent=node,
        critical=True
    )
    claim_total = "The total nonresident surcharge is $400 ($100 × 4 people)."
    await evaluator.verify(
        claim=claim_total,
        node=total_leaf,
        sources=None,
        additional_instruction="Simple arithmetic verification.",
    )


async def build_reservation_booking_date(
    evaluator: Evaluator,
    parent: Any,
    booking: BookingWindowInfo
) -> None:
    node = evaluator.add_sequential(
        id="Reservation_Booking_Date",
        desc="Determines the earliest date when reservations can be made for July 2026",
        parent=parent,
        critical=True
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(booking.sources),
        id="Reference_URL_Booking",
        desc="Provides at least one valid reference URL for reservation booking window information",
        parent=node,
        critical=True
    )

    window_leaf = evaluator.add_leaf(
        id="Booking_Window_Explained",
        desc="Explains that reservations open 13 months in advance on the 5th of each month",
        parent=node,
        critical=True
    )
    claim_window = "Yellowstone National Park Lodges reservations open 13 months in advance on the 5th of each month."
    await evaluator.verify(
        claim=claim_window,
        node=window_leaf,
        sources=booking.sources,
        additional_instruction="Verify from official Yellowstone Lodges reservation policy pages.",
    )

    date_leaf = evaluator.add_leaf(
        id="Specific_Date_Calculated",
        desc="States that reservations for July 2026 open on June 5, 2025",
        parent=node,
        critical=True
    )
    claim_date = "Reservations for July 2026 open on June 5, 2025."
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=None,
        additional_instruction="Logical calculation based on the booking window rule.",
    )


async def build_deposit_requirement(
    evaluator: Evaluator,
    parent: Any,
    deposit: DepositInfo
) -> None:
    node = evaluator.add_parallel(
        id="Deposit_Requirement",
        desc="States the advance deposit requirement for securing the reservation",
        parent=parent,
        critical=True
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(deposit.sources),
        id="Reference_URL_Deposit",
        desc="Provides at least one valid reference URL for deposit policy information",
        parent=node,
        critical=True
    )

    deposit_leaf = evaluator.add_leaf(
        id="Deposit_Amount_Policy",
        desc="States that the deposit equals the first night's rate at each location",
        parent=node,
        critical=True
    )
    claim_deposit = "An advance deposit equal to the first night's rate at each location is required to secure the reservation."
    await evaluator.verify(
        claim=claim_deposit,
        node=deposit_leaf,
        sources=deposit.sources,
        additional_instruction="Verify against Yellowstone Lodges policy pages.",
    )

    payment_leaf = evaluator.add_leaf(
        id="Payment_Methods_Noted",
        desc="Notes the accepted payment methods for advance deposits",
        parent=node,
        critical=True
    )
    if deposit.payment_methods:
        methods_list = ", ".join(deposit.payment_methods)
        claim_pay = f"Accepted payment methods for advance deposits include {methods_list}."
    else:
        claim_pay = "Accepted payment methods for advance deposits include major credit cards."
    await evaluator.verify(
        claim=claim_pay,
        node=payment_leaf,
        sources=deposit.sources,
        additional_instruction="Verify accepted methods on official lodges policy pages; reasonable wording allowed.",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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

    # Concurrent extraction
    lodging_task = evaluator.extract(
        prompt=prompt_extract_lodging_info(),
        template_class=LodgingExtraction,
        extraction_name="lodging_info"
    )
    pet_fee_task = evaluator.extract(
        prompt=prompt_extract_pet_fee_info(),
        template_class=PetFeeInfo,
        extraction_name="pet_fee_info"
    )
    annual_pass_task = evaluator.extract(
        prompt=prompt_extract_annual_pass_info(),
        template_class=AnnualPassInfo,
        extraction_name="annual_pass_info"
    )
    surcharge_task = evaluator.extract(
        prompt=prompt_extract_surcharge_info(),
        template_class=SurchargeInfo,
        extraction_name="nonresident_surcharge_info"
    )
    booking_task = evaluator.extract(
        prompt=prompt_extract_booking_window_info(),
        template_class=BookingWindowInfo,
        extraction_name="booking_window_info"
    )
    deposit_task = evaluator.extract(
        prompt=prompt_extract_deposit_info(),
        template_class=DepositInfo,
        extraction_name="deposit_info"
    )

    (
        lodging_info,
        pet_fee_info,
        annual_pass_info,
        surcharge_info,
        booking_info,
        deposit_info
    ) = await asyncio.gather(
        lodging_task, pet_fee_task, annual_pass_task, surcharge_task, booking_task, deposit_task
    )

    # Add ground truth summary info
    evaluator.add_ground_truth(
        {
            "ada_required_features": GROUND_TRUTH["ada_required_features"],
            "pet_fee_rate_per_pet_per_night": GROUND_TRUTH["pet_fee_rate_per_pet_per_night"],
            "pet_fee_effective_date": GROUND_TRUTH["pet_fee_effective_date"],
            "pet_fee_total_for_4_nights_1_pet": GROUND_TRUTH["pet_fee_total_for_4_nights_1_pet"],
            "annual_pass_nonresident_price": GROUND_TRUTH["annual_pass_nonresident_price"],
            "annual_pass_effective_date": GROUND_TRUTH["annual_pass_effective_date"],
            "nonresident_surcharge_per_person_16+": GROUND_TRUTH["nonresident_surcharge_per_person_16+"],
            "family_eligible_count": GROUND_TRUTH["family_eligible_count"],
            "nonresident_surcharge_total": GROUND_TRUTH["nonresident_surcharge_total"],
            "booking_window_rule": GROUND_TRUTH["booking_window_rule"],
            "deposit_policy": GROUND_TRUTH["deposit_policy"],
            "max_pets_per_cabin": GROUND_TRUTH["max_pets_per_cabin"],
            "computed_earliest_booking_date_for_july_2026": compute_earliest_booking_date(7, 2026)
        },
        gt_type="ground_truth"
    )

    # Build verification subtrees
    await build_lodging_identification(evaluator, root, lodging_info)
    await build_accessibility_features(evaluator, root, lodging_info)
    await build_pet_fee_calculation(evaluator, root, pet_fee_info)
    await build_annual_pass_cost(evaluator, root, annual_pass_info)
    await build_nonresident_surcharge(evaluator, root, surcharge_info)
    await build_reservation_booking_date(evaluator, root, booking_info)
    await build_deposit_requirement(evaluator, root, deposit_info)

    # Return evaluation summary
    return evaluator.get_summary()