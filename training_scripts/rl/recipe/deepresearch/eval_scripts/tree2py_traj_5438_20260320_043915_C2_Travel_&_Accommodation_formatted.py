import asyncio
import logging
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nashville_grenada_trip_planning"
TASK_DESCRIPTION = """
You are planning a trip from Nashville, Tennessee to Grenada in the Caribbean. Your travel dates are as follows: departure from Nashville on June 15, 2026, and return to Nashville on June 25, 2026. You are a US citizen, and your passport expires on November 1, 2026.

You plan to fly with Sun Country Airlines for at least one segment of your journey. You will be checking one bag that weighs 48 pounds and has dimensions of 28 inches (length) × 20 inches (width) × 12 inches (height).

You will drive your personal vehicle to Nashville International Airport (BNA) and park in the Terminal Garage for the entire duration of your trip.

Please provide the following information:

1. Does your passport meet Grenada's entry requirements for US citizens regarding validity duration? State clearly whether it meets the requirements and explain why or why not, including the specific validity requirement and how many months your passport will be valid from your arrival date.

2. Does your checked bag meet Sun Country Airlines' requirements for checked baggage? Address both the weight limit and the size limit separately, stating the specific limits and whether your bag complies with each.

3. What is the total cost to park your vehicle in BNA's Terminal Garage for the duration of your trip? Provide the daily parking rate and show your calculation for the total cost.

For all answers, provide reference URLs from official or authoritative sources that support your information.
"""

# Scenario facts for logical checks
ARRIVAL_DATE = date(2026, 6, 15)
RETURN_DATE = date(2026, 6, 25)
PASSPORT_EXPIRY = date(2026, 11, 1)

BAG_WEIGHT_LB = 48
BAG_DIMS = (28, 20, 12)  # inches
LINEAR_INCHES = sum(BAG_DIMS)  # 60

PARKING_DAYS = (RETURN_DATE - ARRIVAL_DATE).days  # 10
EXPECTED_TERMINAL_GARAGE_DAILY_RATE = 33  # USD (as per rubric description)
EXPECTED_PARKING_TOTAL = PARKING_DAYS * EXPECTED_TERMINAL_GARAGE_DAILY_RATE  # 330

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PassportExtraction(BaseModel):
    meets_requirement: Optional[str] = None  # e.g., "meets", "does not meet", "yes", "no"
    requirement_description: Optional[str] = None  # e.g., "valid for at least 6 months from date of entry"
    months_valid_from_arrival: Optional[str] = None  # e.g., "about 4.5 months"
    cited_urls: List[str] = Field(default_factory=list)


class BaggageExtraction(BaseModel):
    weight_limit_stated: Optional[str] = None  # e.g., "50 lb (23 kg)"
    size_limit_stated: Optional[str] = None  # e.g., "62 linear inches"
    weight_compliance_statement: Optional[str] = None  # e.g., "complies / does not comply"
    size_compliance_statement: Optional[str] = None  # e.g., "complies / does not comply"
    cited_urls: List[str] = Field(default_factory=list)


class ParkingExtraction(BaseModel):
    daily_rate_stated: Optional[str] = None  # e.g., "$33 per day"
    total_cost_stated: Optional[str] = None  # e.g., "$330"
    days_used_in_calculation: Optional[str] = None  # e.g., "10 days"
    cited_urls: List[str] = Field(default_factory=list)


class TravelExtraction(BaseModel):
    passport: Optional[PassportExtraction] = None
    baggage: Optional[BaggageExtraction] = None
    parking: Optional[ParkingExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel() -> str:
    return """
    Extract the user's final answer content into structured fields for three sections: passport, baggage, and parking.
    Only extract information that is explicitly present in the answer text itself. Do not infer or compute new facts.

    For passport:
    - meets_requirement: The answer's explicit conclusion on whether the passport meets Grenada's validity requirement (use a short phrase like "meets", "does not meet", "yes", "no", or "unclear" if not clearly stated).
    - requirement_description: The specific validity requirement as stated in the answer (e.g., "valid for at least 6 months from date of entry").
    - months_valid_from_arrival: The validity duration from arrival as stated/calculated in the answer (e.g., "about 4.5 months"); if not provided, set to null.
    - cited_urls: All URLs the answer provides to support passport validity info.

    For baggage (Sun Country Airlines checked baggage):
    - weight_limit_stated: The weight limit text as stated in the answer (e.g., "50 lb (23 kg)"); if not stated, set to null.
    - size_limit_stated: The size/linear-dimensions limit text as stated (e.g., "62 linear inches"); if not stated, set to null.
    - weight_compliance_statement: The answer's explicit conclusion for weight compliance (e.g., "complies", "does not comply", or a sentence including that conclusion); if missing, set to null.
    - size_compliance_statement: The answer's explicit conclusion for size compliance; if missing, set to null.
    - cited_urls: All URLs supporting the baggage limits/compliance.

    For parking (BNA Terminal Garage):
    - daily_rate_stated: The daily parking rate the answer states (e.g., "$33 per day"); if missing, set to null.
    - total_cost_stated: The total parking cost the answer states; if missing, set to null.
    - days_used_in_calculation: The number of days used in the calculation as stated (e.g., "10 days"); if missing, set to null.
    - cited_urls: All URLs supporting the BNA Terminal Garage rate.

    Notes:
    - Return exactly what appears in the answer (as strings). Do not normalize currency symbols or units.
    - For cited_urls, include every URL string (plain or markdown link). If none provided, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper calculations                                                         #
# --------------------------------------------------------------------------- #
def compute_months_from_arrival(arrival: date, expiry: date) -> float:
    days = (expiry - arrival).days
    # Approximate calendar months
    months = days / 30.44 if days >= 0 else 0.0
    return round(months, 1)


def pretty_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_url_reference_leaf_or_fail(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    urls: List[str],
    claim: str,
    additional_instruction: str = "None",
    critical: bool = True
):
    """
    Create a URL-backed verification leaf if URLs exist; otherwise, force a fail
    for this node to enforce the requirement of providing references.
    """
    if urls and len(urls) > 0:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=additional_instruction
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=f"{desc} (failed: no reference URL provided in the answer)",
            parent=parent,
            critical=critical
        )


# --------------------------------------------------------------------------- #
# Subtree verifications                                                       #
# --------------------------------------------------------------------------- #
async def verify_passport_section(evaluator: Evaluator, parent_node, extraction: TravelExtraction):
    """
    Build and verify the passport validity subtree.
    """
    node_passport = evaluator.add_parallel(
        id="Passport_Validity_Question",
        desc="Evaluate the passport validity assessment and supporting documentation",
        parent=parent_node,
        critical=True
    )

    passport_info = extraction.passport or PassportExtraction()

    # Leaf: Passport_Validity_Assessment (logic + whether answer's conclusion aligns)
    leaf_assess = evaluator.add_leaf(
        id="Passport_Validity_Assessment",
        desc=("Correctly determine whether the passport meets Grenada's 6-month validity requirement from the arrival date "
              f"of {pretty_date(ARRIVAL_DATE)}, given the passport expiration date of {pretty_date(PASSPORT_EXPIRY)}, "
              "and provide clear explanation including the specific requirement and actual validity period"),
        parent=node_passport,
        critical=True
    )

    approx_months = compute_months_from_arrival(ARRIVAL_DATE, PASSPORT_EXPIRY)  # ~4.6 months
    # We assert that the correct outcome is "does not meet" for a strict 6-month-from-arrival rule
    claim_passport = (
        f"From the arrival date {pretty_date(ARRIVAL_DATE)} to the passport expiration {pretty_date(PASSPORT_EXPIRY)} "
        f"is about {approx_months} months, which is less than 6 months. Therefore, under a 'valid for at least 6 months "
        f"from date of entry' rule, the correct conclusion is that the passport does not meet the requirement. "
        f"The answer's stated conclusion matches this (it should clearly say 'does not meet' or equivalent)."
    )
    await evaluator.verify(
        claim=claim_passport,
        node=leaf_assess,
        additional_instruction=(
            "Judge only whether the answer's conclusion aligns with the correct determination for the given dates "
            "(less than 6 months). Also consider whether the answer includes a clear explanation mentioning the "
            "specific '6 months from arrival' requirement and the actual validity duration from arrival. "
            "Minor phrasing differences are fine; focus on correctness and clarity."
        )
    )

    # Leaf: Passport_URL_Reference (must provide supporting URL and it must say 6 months from entry/arrival)
    await add_url_reference_leaf_or_fail(
        evaluator=evaluator,
        parent=node_passport,
        node_id="Passport_URL_Reference",
        desc="Provide reference URLs from official or authoritative sources supporting the passport validity requirement information",
        urls=passport_info.cited_urls,
        claim=("Grenada requires a passport that is valid for at least six months from the date of entry "
               "(or arrival). The provided source(s) explicitly support this."),
        additional_instruction=(
            "Verify that at least one provided source explicitly states a 6‑month (six months) passport validity "
            "requirement, ideally phrased as from date of entry/arrival. Ignore general travel blogs if an official "
            "or clearly authoritative source is also provided."
        ),
        critical=True
    )


async def verify_baggage_section(evaluator: Evaluator, parent_node, extraction: TravelExtraction):
    """
    Build and verify the baggage compliance subtree.
    """
    node_baggage = evaluator.add_parallel(
        id="Baggage_Compliance_Question",
        desc="Evaluate the baggage compliance assessment and supporting documentation",
        parent=parent_node,
        critical=True
    )

    baggage_info = extraction.baggage or BaggageExtraction()

    # Subnode: Baggage_Compliance_Verification
    node_baggage_verify = evaluator.add_parallel(
        id="Baggage_Compliance_Verification",
        desc="Verify checked baggage compliance with Sun Country Airlines requirements for the specified bag (48 pounds, 60 linear inches)",
        parent=node_baggage,
        critical=True
    )

    # Leaf: Weight_Compliance
    leaf_weight = evaluator.add_leaf(
        id="Weight_Compliance",
        desc="Correctly determine whether the 48-pound checked bag meets Sun Country Airlines' 50-pound weight limit and state the specific limit",
        parent=node_baggage_verify,
        critical=True
    )
    claim_weight = (
        "According to Sun Country Airlines' checked baggage policy, the standard weight limit per checked bag is "
        "50 lb (23 kg). The answer explicitly states this 50 lb limit and concludes that a 48 lb bag complies. "
        "Using the provided source(s), this conclusion is correct."
    )
    await evaluator.verify(
        claim=claim_weight,
        node=leaf_weight,
        sources=baggage_info.cited_urls,
        additional_instruction=(
            "Check on the source page that the standard checked bag weight limit is 50 lb (23 kg). "
            "Then verify that the answer actually states that limit and that the 48 lb bag is within the limit. "
            "Allow minor formatting differences (e.g., 'lbs', 'lb')."
        )
    )

    # Leaf: Size_Compliance
    leaf_size = evaluator.add_leaf(
        id="Size_Compliance",
        desc="Correctly determine whether the 60 linear inches checked bag meets Sun Country Airlines' 62 linear inches size limit and state the specific limit",
        parent=node_baggage_verify,
        critical=True
    )
    claim_size = (
        "According to Sun Country Airlines' checked baggage policy, the maximum size for a checked bag is "
        "62 linear inches (length + width + height). The item described (28\" + 20\" + 12\" = 60 linear inches) "
        "is within this limit. The answer explicitly states the 62-inch limit and that the 60-inch bag complies; "
        "this is correct and supported by the provided source(s)."
    )
    await evaluator.verify(
        claim=claim_size,
        node=leaf_size,
        sources=baggage_info.cited_urls,
        additional_instruction=(
            "Confirm on the source page that the checked bag size limit is 62 linear inches (L+W+H). "
            "Then check the answer states 62 inches and notes that 60 linear inches complies. "
            "Minor wording differences are acceptable; focus on factual correctness."
        )
    )

    # Leaf: Baggage_URL_Reference (policy support for limits)
    await add_url_reference_leaf_or_fail(
        evaluator=evaluator,
        parent=node_baggage,
        node_id="Baggage_URL_Reference",
        desc="Provide reference URLs from official or authoritative sources supporting the baggage requirements information",
        urls=baggage_info.cited_urls,
        claim=("Sun Country Airlines' standard checked baggage allowance sets a maximum weight of 50 lb (23 kg) "
               "per bag and a maximum size of 62 linear inches (length + width + height). The provided source(s) "
               "explicitly support these limits."),
        additional_instruction="Prefer the official Sun Country policy page if available.",
        critical=True
    )


async def verify_parking_section(evaluator: Evaluator, parent_node, extraction: TravelExtraction):
    """
    Build and verify the parking cost subtree.
    """
    node_parking = evaluator.add_parallel(
        id="Parking_Cost_Question",
        desc="Evaluate the parking cost calculation and supporting documentation",
        parent=parent_node,
        critical=True
    )

    parking_info = extraction.parking or ParkingExtraction()

    # Leaf: Parking_Cost_Calculation
    leaf_calc = evaluator.add_leaf(
        id="Parking_Cost_Calculation",
        desc=f"Calculate the total parking cost for {PARKING_DAYS} days at BNA Terminal Garage, providing the correct daily rate of $33 per day and showing the calculation",
        parent=node_parking,
        critical=True
    )
    claim_calc = (
        f"The trip spans {PARKING_DAYS} days (from {pretty_date(ARRIVAL_DATE)} to {pretty_date(RETURN_DATE)}). "
        f"The answer explicitly uses a daily rate of ${EXPECTED_TERMINAL_GARAGE_DAILY_RATE} and shows the calculation "
        f"({PARKING_DAYS} × ${EXPECTED_TERMINAL_GARAGE_DAILY_RATE} = ${EXPECTED_PARKING_TOTAL}) to arrive at a total of "
        f"${EXPECTED_PARKING_TOTAL} for BNA Terminal Garage parking."
    )
    await evaluator.verify(
        claim=claim_calc,
        node=leaf_calc,
        additional_instruction=(
            "Check that the answer: (1) uses the correct duration of 10 days for parking based on the provided dates; "
            "(2) explicitly includes the daily rate '$33 per day'; and (3) shows or states the multiplication resulting in $330. "
            "Minor formatting variations (e.g., '$33/day') are okay, but all three elements must be present."
        )
    )

    # Leaf: Parking_URL_Reference (rate support)
    await add_url_reference_leaf_or_fail(
        evaluator=evaluator,
        parent=node_parking,
        node_id="Parking_URL_Reference",
        desc="Provide reference URLs from official or authoritative sources supporting the parking rate information",
        urls=parking_info.cited_urls,
        claim="The BNA Terminal Garage daily parking rate is $33 per day. The provided source(s) explicitly support this.",
        additional_instruction="Prefer an official Nashville International Airport (BNA) parking page if available.",
        critical=True
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
    Evaluate an answer for the Nashville → Grenada trip planning verification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level rubric is parallel across sections
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_travel(),
        template_class=TravelExtraction,
        extraction_name="travel_extraction"
    )

    # Add a top-level critical node to mirror rubric root
    travel_node = evaluator.add_parallel(
        id="Travel_Planning_Verification",
        desc="Verify all travel planning requirements for a trip from Nashville to Grenada",
        parent=root,
        critical=True
    )

    # Add ground truth/context info for transparency
    evaluator.add_ground_truth({
        "arrival_date": pretty_date(ARRIVAL_DATE),
        "return_date": pretty_date(RETURN_DATE),
        "passport_expiry": pretty_date(PASSPORT_EXPIRY),
        "months_valid_from_arrival_approx": compute_months_from_arrival(ARRIVAL_DATE, PASSPORT_EXPIRY),
        "bag_weight_lb": BAG_WEIGHT_LB,
        "bag_dimensions_in": {"L": BAG_DIMS[0], "W": BAG_DIMS[1], "H": BAG_DIMS[2]},
        "bag_linear_inches": LINEAR_INCHES,
        "parking_days": PARKING_DAYS,
        "expected_terminal_garage_daily_rate_usd": EXPECTED_TERMINAL_GARAGE_DAILY_RATE,
        "expected_parking_total_usd": EXPECTED_PARKING_TOTAL
    })

    # Build verification subtrees
    await verify_passport_section(evaluator, travel_node, extraction)
    await verify_baggage_section(evaluator, travel_node, extraction)
    await verify_parking_section(evaluator, travel_node, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()