import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grand_canyon_trip_planning_2025"
TASK_DESCRIPTION = (
    "A U.S. family of 4 (parents ages 43 and 38, children ages 16 and 12) is planning to travel from San Diego, California to Grand Canyon National Park South Rim. "
    "They will fly on a budget airline and rent a car at their destination airport. "
    "They plan to visit exactly three different national parks this year, each charging a $35 vehicle entrance fee.\n\n"
    "Answer the following:\n\n"
    "1. Between Phoenix Sky Harbor Airport (PHX) and Las Vegas Airport (LAS) - both of which have budget airline service from San Diego - which airport is closer to Grand Canyon South Rim by driving distance?\n\n"
    "2. What is the approximate driving distance in miles from that closer airport to Grand Canyon South Rim?\n\n"
    "3. What is the standard entrance fee for a private vehicle at Grand Canyon National Park (valid for 7 days)?\n\n"
    "4. Calculate the total cost if the family pays the individual entrance fee at all three national parks they plan to visit.\n\n"
    "5. Would purchasing one America the Beautiful Annual Pass (for U.S. residents) be more cost-effective than paying the individual entrance fees? If so, how much money would they save?\n\n"
    "6. Who currently serves as the U.S. Transportation Secretary (as of January 2025)?"
)

# Ground truth and expected values (per rubric constraints)
EXPECTED_CLOSER_AIRPORT = "Phoenix Sky Harbor (PHX)"
EXPECTED_DISTANCE_MILES = 230  # Approximate expected driving distance from PHX to South Rim
EXPECTED_DISTANCE_TOLERANCE = 20  # Acceptable ± range around 230
VEHICLE_FEE_USD = 35
FEE_VALIDITY_DAYS = 7
NUMBER_OF_PARKS = 3
TOTAL_INDIVIDUAL_COST = NUMBER_OF_PARKS * VEHICLE_FEE_USD  # 105
ANNUAL_PASS_PRICE_USD = 80
EXPECTED_SAVINGS_USD = TOTAL_INDIVIDUAL_COST - ANNUAL_PASS_PRICE_USD  # 25
EXPECTED_TRANSPORTATION_SECRETARY = "Sean Duffy"  # As specified by rubric constraints

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TripPlanningExtraction(BaseModel):
    # Q1: Closer airport
    closer_airport: Optional[str] = None
    closer_airport_sources: List[str] = Field(default_factory=list)

    # Q2: Approximate driving distance from the closer airport to South Rim
    approx_distance_miles: Optional[str] = None
    distance_sources: List[str] = Field(default_factory=list)

    # Q3: Grand Canyon vehicle entrance fee and validity
    vehicle_entrance_fee: Optional[str] = None
    vehicle_fee_validity: Optional[str] = None
    fee_sources: List[str] = Field(default_factory=list)

    # Q4: Total cost for three parks if paying individual fees
    total_cost_three_parks: Optional[str] = None

    # Q5: Annual pass decision and savings
    annual_pass_decision: Optional[str] = None  # e.g., "Yes, more cost-effective" or "No"
    annual_pass_price: Optional[str] = None
    annual_pass_savings: Optional[str] = None
    annual_pass_sources: List[str] = Field(default_factory=list)

    # Q6: Transportation Secretary (as of Jan 2025)
    transportation_secretary_name: Optional[str] = None
    sec_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_planning() -> str:
    return (
        "Extract the requested answers from the provided response text for the Grand Canyon trip planning task. "
        "Return a JSON object with the following fields (return null if not mentioned):\n"
        "1) closer_airport: The airport the answer identifies as closer to Grand Canyon South Rim by driving distance (PHX or LAS). Accept any reasonable naming variant.\n"
        "   closer_airport_sources: List all URLs cited that support this determination (Google Maps or any source). If none, return an empty list.\n"
        "2) approx_distance_miles: The approximate driving distance in miles from the closer airport to Grand Canyon South Rim (string exactly as stated in the answer).\n"
        "   distance_sources: List any URLs cited for this distance (e.g., Google Maps). If none, return an empty list.\n"
        "3) vehicle_entrance_fee: The standard private vehicle entrance fee for Grand Canyon National Park (string as stated, e.g., \"$35\").\n"
        "   vehicle_fee_validity: The stated validity period (e.g., \"7 days\").\n"
        "   fee_sources: List URLs cited for fee information (e.g., NPS pages). If none, empty list.\n"
        "4) total_cost_three_parks: The total cost computed if paying individual fees at three parks (string as stated, e.g., \"$105\").\n"
        "5) annual_pass_decision: The answer's conclusion whether one America the Beautiful Annual Pass is more cost-effective than paying individual fees (e.g., \"Yes\", \"No\", or textual phrase).\n"
        "   annual_pass_price: The price stated for the annual pass (string, e.g., \"$80\").\n"
        "   annual_pass_savings: The stated savings if any (string, e.g., \"$25\").\n"
        "   annual_pass_sources: List URLs cited for the annual pass info. If none, empty list.\n"
        "6) transportation_secretary_name: The person the answer identifies as the U.S. Transportation Secretary (as of January 2025).\n"
        "   sec_sources: List URLs cited for this identification. If none, empty list.\n"
        "Important: Extract exactly what is written in the answer. Do not invent information. If any field is not mentioned, return null or empty list accordingly."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def extract_first_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_q1_closest_airport(evaluator: Evaluator, parent_node, data: TripPlanningExtraction) -> None:
    q1_node = evaluator.add_parallel(
        id="Q1_ClosestAirport",
        desc="Determine which airport (PHX vs LAS) is closer to Grand Canyon South Rim by driving distance.",
        parent=parent_node,
        critical=True
    )

    # Existence check
    evaluator.add_custom_node(
        result=bool(data.closer_airport and data.closer_airport.strip()),
        id="Q1_CloserAirport_Provided",
        desc="Closer airport is identified in the answer.",
        parent=q1_node,
        critical=True
    )

    # Leaf: Closer airport matches Phoenix (PHX)
    leaf = evaluator.add_leaf(
        id="CloserAirportIdentification",
        desc="Identify Phoenix Sky Harbor (PHX) as closer than Las Vegas (LAS) by driving distance (per constraints).",
        parent=q1_node,
        critical=True
    )

    claim = (
        f"The provided closer airport '{data.closer_airport or ''}' refers to Phoenix Sky Harbor (PHX). "
        f"Treat 'Phoenix', 'PHX', 'Sky Harbor', and 'Phoenix Sky Harbor International Airport' as equivalent."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="Focus on name equivalence only; ignore case and minor variants. Do not infer new information."
    )


async def build_q2_distance(evaluator: Evaluator, parent_node, data: TripPlanningExtraction) -> None:
    q2_node = evaluator.add_parallel(
        id="Q2_DrivingDistanceFromCloserAirport",
        desc="State the approximate driving distance in miles from the closer airport to Grand Canyon South Rim.",
        parent=parent_node,
        critical=True
    )

    # Existence check
    evaluator.add_custom_node(
        result=bool(data.approx_distance_miles and data.approx_distance_miles.strip()),
        id="Q2_Distance_Provided",
        desc="Approximate driving distance (miles) is provided.",
        parent=q2_node,
        critical=True
    )

    # Leaf: Approximate distance about 230 miles
    leaf = evaluator.add_leaf(
        id="ApproxDistanceMiles",
        desc="Provide an approximate mileage distance consistent with the constraints (about 230 miles from PHX to the South Rim).",
        parent=q2_node,
        critical=True
    )

    miles_val = extract_first_number(data.approx_distance_miles)
    lower = EXPECTED_DISTANCE_MILES - EXPECTED_DISTANCE_TOLERANCE
    upper = EXPECTED_DISTANCE_MILES + EXPECTED_DISTANCE_TOLERANCE

    if miles_val is not None:
        claim = (
            f"The stated driving distance of {miles_val:.0f} miles from Phoenix Sky Harbor (PHX) to Grand Canyon South Rim "
            f"is approximately {EXPECTED_DISTANCE_MILES} miles and acceptable if it lies between {lower} and {upper} miles."
        )
    else:
        claim = (
            f"The stated driving distance '{data.approx_distance_miles or ''}' is approximately {EXPECTED_DISTANCE_MILES} miles "
            f"from PHX to Grand Canyon South Rim (acceptable range {lower}–{upper} miles)."
        )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.distance_sources if data.distance_sources else None,
        additional_instruction="Accept reasonable approximations and rounding. If within the stated range, mark correct."
    )


async def build_q3_fee(evaluator: Evaluator, parent_node, data: TripPlanningExtraction) -> None:
    q3_node = evaluator.add_parallel(
        id="Q3_GrandCanyonVehicleEntranceFee",
        desc="State the standard entrance fee for a private vehicle at Grand Canyon National Park (valid for 7 days).",
        parent=parent_node,
        critical=True
    )

    # Existence check
    evaluator.add_custom_node(
        result=bool(data.vehicle_entrance_fee and data.vehicle_entrance_fee.strip()),
        id="Q3_Fee_Provided",
        desc="Vehicle entrance fee is provided.",
        parent=q3_node,
        critical=True
    )

    # Leaf: Fee and validity
    leaf = evaluator.add_leaf(
        id="FeeAndValidity",
        desc="State the standard private vehicle entrance fee is $35 and that it is valid for 7 consecutive days (per constraints).",
        parent=q3_node,
        critical=True
    )

    claim = (
        f"The standard private vehicle entrance fee at Grand Canyon National Park is ${VEHICLE_FEE_USD} and it is valid for {FEE_VALIDITY_DAYS} consecutive days."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.fee_sources if data.fee_sources else None,
        additional_instruction="Check consistency with the answer and any cited NPS sources; allow '$35' and '7 days' phrasing variants."
    )


async def build_q4_total(evaluator: Evaluator, parent_node, data: TripPlanningExtraction) -> None:
    q4_node = evaluator.add_parallel(
        id="Q4_TotalCostThreeParksIndividualFees",
        desc="Calculate total cost if paying individual entrance fees at all three parks (each $35).",
        parent=parent_node,
        critical=True
    )

    # Existence check
    evaluator.add_custom_node(
        result=bool(data.total_cost_three_parks and data.total_cost_three_parks.strip()),
        id="Q4_Total_Provided",
        desc="Total cost for three parks is provided.",
        parent=q4_node,
        critical=True
    )

    # Leaf: Total computation 3 × 35 = 105
    leaf = evaluator.add_leaf(
        id="TotalCostComputation",
        desc="Compute 3 × $35 = $105 (per constraints).",
        parent=q4_node,
        critical=True
    )

    claim = f"The total cost for three parks at ${VEHICLE_FEE_USD} each is ${TOTAL_INDIVIDUAL_COST}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="This is a simple multiplication: 3 × 35 = 105. Verify the answer's stated total aligns."
    )


async def build_q5_pass(evaluator: Evaluator, parent_node, data: TripPlanningExtraction) -> None:
    q5_node = evaluator.add_sequential(
        id="Q5_AnnualPassCostEffectiveness",
        desc="Decide whether one America the Beautiful Annual Pass is more cost-effective than paying the three individual fees, and compute savings if so.",
        parent=parent_node,
        critical=True
    )

    # Existence check for decision
    evaluator.add_custom_node(
        result=bool(data.annual_pass_decision and data.annual_pass_decision.strip()),
        id="Q5_Decision_Provided",
        desc="Annual pass cost-effectiveness decision is provided.",
        parent=q5_node,
        critical=True
    )

    # Leaf 1: More cost-effective decision
    decision_leaf = evaluator.add_leaf(
        id="MoreCostEffectiveDecision",
        desc="Correctly conclude the pass is more cost-effective than $105 in individual fees (per constraints).",
        parent=q5_node,
        critical=True
    )

    decision_claim = (
        f"Paying ${ANNUAL_PASS_PRICE_USD} for one Annual Pass is cheaper than paying ${TOTAL_INDIVIDUAL_COST} in individual fees; "
        f"therefore the pass is more cost-effective. Confirm that the answer concludes accordingly."
    )
    await evaluator.verify(
        claim=decision_claim,
        node=decision_leaf,
        additional_instruction="Judge correct only if the answer explicitly indicates the pass is more cost-effective."
    )

    # Leaf 2: Savings amount
    savings_leaf = evaluator.add_leaf(
        id="SavingsAmount",
        desc="Compute savings as $105 − $80 = $25 (per constraints).",
        parent=q5_node,
        critical=True
    )

    savings_claim = f"The savings from purchasing the Annual Pass is ${EXPECTED_SAVINGS_USD} (computed as ${TOTAL_INDIVIDUAL_COST} − ${ANNUAL_PASS_PRICE_USD})."
    await evaluator.verify(
        claim=savings_claim,
        node=savings_leaf,
        additional_instruction="Verify the answer's stated savings equals $25; allow symbols or words like '25 dollars'."
    )


async def build_q6_secretary(evaluator: Evaluator, parent_node, data: TripPlanningExtraction) -> None:
    q6_node = evaluator.add_parallel(
        id="Q6_TransportationSecretaryJan2025",
        desc="Identify who serves as the U.S. Transportation Secretary as of January 2025.",
        parent=parent_node,
        critical=True
    )

    # Existence check
    evaluator.add_custom_node(
        result=bool(data.transportation_secretary_name and data.transportation_secretary_name.strip()),
        id="Q6_Secretary_Provided",
        desc="Transportation Secretary name is provided.",
        parent=q6_node,
        critical=True
    )

    # Leaf: Secretary name matches expected (per constraints)
    leaf = evaluator.add_leaf(
        id="SecretaryName",
        desc="Identify Sean Duffy as the U.S. Transportation Secretary (as of January 2025) per constraints.",
        parent=q6_node,
        critical=True
    )

    claim = (
        f"The provided name '{data.transportation_secretary_name or ''}' refers to '{EXPECTED_TRANSPORTATION_SECRETARY}'. "
        f"Consider minor naming variants and ignore case."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.sec_sources if data.sec_sources else None,
        additional_instruction="Focus on whether the answer names Sean Duffy; treat minor variations (e.g., middle name) as equivalent."
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
    # Initialize evaluator with a parallel root
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

    # Add the task-level critical node that aggregates all sub-questions
    task_node = evaluator.add_parallel(
        id="GrandCanyonTripPlanning",
        desc="Answer all six requested sub-questions (closest airport, distance, entrance fee, total fees for 3 parks, annual pass comparison+savings, Transportation Secretary as of Jan 2025).",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_trip_planning(),
        template_class=TripPlanningExtraction,
        extraction_name="trip_planning_extraction",
    )

    # Add ground truth information per rubric constraints (for transparency)
    evaluator.add_ground_truth({
        "expected_closer_airport": EXPECTED_CLOSER_AIRPORT,
        "expected_distance_miles_approx": EXPECTED_DISTANCE_MILES,
        "accepted_distance_range": [EXPECTED_DISTANCE_MILES - EXPECTED_DISTANCE_TOLERANCE, EXPECTED_DISTANCE_MILES + EXPECTED_DISTANCE_TOLERANCE],
        "vehicle_fee_usd": VEHICLE_FEE_USD,
        "fee_validity_days": FEE_VALIDITY_DAYS,
        "number_of_parks": NUMBER_OF_PARKS,
        "total_individual_cost_usd": TOTAL_INDIVIDUAL_COST,
        "annual_pass_price_usd": ANNUAL_PASS_PRICE_USD,
        "expected_savings_usd": EXPECTED_SAVINGS_USD,
        "expected_transportation_secretary_jan_2025": EXPECTED_TRANSPORTATION_SECRETARY,
    }, gt_type="expected_values")

    # Build and verify each sub-question subtree
    await build_q1_closest_airport(evaluator, task_node, extraction)
    await build_q2_distance(evaluator, task_node, extraction)
    await build_q3_fee(evaluator, task_node, extraction)
    await build_q4_total(evaluator, task_node, extraction)
    await build_q5_pass(evaluator, task_node, extraction)
    await build_q6_secretary(evaluator, task_node, extraction)

    # Return structured result summary
    return evaluator.get_summary()