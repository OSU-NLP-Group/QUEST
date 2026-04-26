import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "intl_trip_2026_planning_eval"
TASK_DESCRIPTION = (
    "A U.S. resident family of four (2 adults and 2 children under 16) is planning a 14-day international trip in 2026. "
    "They will drive to a major U.S. airport and park there for the entire duration of their trip. Upon returning, they plan to visit national parks. "
    "Based on 2026 travel industry data, provide the following information:\n\n"
    "1. Identify which major U.S. airport offers the cheapest long-term parking rate per day, and state this daily rate.\n"
    "2. Calculate the total parking cost for their 14-day trip at this airport.\n"
    "3. Determine the total cost of purchasing 2026 America the Beautiful Annual Passes for this family (2 adults and 2 children under 16).\n"
    "4. They plan to bring one carry-on bag per person with standard dimensions of 22″ × 14″ × 9″ and weighing 20 pounds. Verify whether this carry-on bag meets the standard 2026 airline requirements for international flights.\n"
    "5. State the generally recommended advance booking timeframe (in months) for international flights based on 2026 travel booking guidelines.\n\n"
    "For each answer, provide supporting reference URLs from reliable sources."
)

TRIP_PARKING_DAYS = 14
EXPECTED_PASS_PRICE_USD = 80.0
EXPECTED_PASSES_NEEDED = 2
EXPECTED_TOTAL_PASS_COST = 160.0  # 2 * 80
EXPECTED_BOOKING_TIMEFRAME_CLAIM = "3-5 months"  # Target wording for verification


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkingExtraction(BaseModel):
    """Extracted info for cheapest long-term airport parking."""
    airport_name: Optional[str] = None
    daily_rate: Optional[str] = None           # e.g., "$12/day", "$12", "12 USD"
    total_cost_14_days: Optional[str] = None   # e.g., "$168", "USD 168"
    urls: List[str] = Field(default_factory=list)  # Sources supporting rate/ranking


class PassExtraction(BaseModel):
    """Extracted info for America the Beautiful Annual Passes."""
    pass_price: Optional[str] = None          # e.g., "$80"
    passes_needed: Optional[str] = None       # e.g., "2", "two"
    total_pass_cost: Optional[str] = None     # e.g., "$160"
    urls: List[str] = Field(default_factory=list)  # Sources supporting pricing/policy


class BaggageExtraction(BaseModel):
    """Extracted sources for carry-on baggage size/weight policy."""
    urls: List[str] = Field(default_factory=list)


class BookingExtraction(BaseModel):
    """Extracted sources and statement for recommended booking window."""
    timeframe: Optional[str] = None          # e.g., "3-5 months"
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_parking() -> str:
    return """
    Extract the airport parking details the answer claims for 2026:
    - airport_name: The major U.S. airport identified as having the cheapest long-term parking daily rate.
    - daily_rate: The stated long-term parking daily rate at that airport (keep the currency symbol/format as in the answer).
    - total_cost_14_days: The total parking cost for a 14-day stay, as explicitly stated in the answer (if provided).
    - urls: All reference URLs that support the airport choice and/or the daily parking rate.

    If any field is missing in the answer, set it to null (or [] for urls).
    """


def prompt_extract_pass() -> str:
    return """
    Extract the America the Beautiful Annual Pass details the answer claims for 2026:
    - pass_price: The stated per-pass price for a U.S. resident adult (keep the currency symbol/format).
    - passes_needed: The number of adult passes the answer claims are needed for a family of 2 adults and 2 children under 16.
    - total_pass_cost: The total dollar cost for the passes (as stated/calculated in the answer).
    - urls: All reference URLs that support the pass pricing and/or eligibility policy.

    If any field is missing in the answer, set it to null (or [] for urls).
    """


def prompt_extract_baggage() -> str:
    return """
    Extract the carry-on baggage policy sources the answer cites for 2026 international flights:
    - urls: All reference URLs that mention or substantiate standard international carry-on size and/or weight limits.

    If there are none, return an empty array for urls.
    """


def prompt_extract_booking() -> str:
    return """
    Extract the recommended booking window statement and sources for international flights:
    - timeframe: The recommended months-in-advance window the answer states (e.g., "3-5 months").
    - urls: All reference URLs that support the recommendation.

    If no timeframe is stated, set it to null. If no URLs are present, return an empty array for urls.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_currency_to_float(text: Optional[str]) -> Optional[float]:
    """Parse a currency-like string to float. Returns None if not parseable."""
    if not text:
        return None
    # Remove commas, keep digits and dot; try to find first numeric segment
    cleaned = text.replace(",", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)", cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_int_from_str(text: Optional[str]) -> Optional[int]:
    """Parse an integer from a string, allowing spelled-out small numbers."""
    if not text:
        return None
    m = re.search(r"(\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Fallback: spelled numbers
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10
    }
    low = text.strip().lower()
    return word_map.get(low)


def approx_equal(a: Optional[float], b: Optional[float], tol: float = 0.01) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_parking_section(evaluator: Evaluator, root_node, parking: ParkingExtraction) -> None:
    # Parent node: cheapest long-term parking airport (critical)
    parent = evaluator.add_parallel(
        id="cheapest_long_term_parking_airport",
        desc="Identification of the major U.S. airport offering the cheapest long-term parking rate per day",
        parent=root_node,
        critical=True
    )

    # Leaf: airport_name (critical)
    node_airport_name = evaluator.add_leaf(
        id="parking_airport_name",
        desc="The airport name is correctly identified as the one with the lowest long-term parking rate among major U.S. airports",
        parent=parent,
        critical=True
    )
    claim_airport = (
        f"According to the provided sources, the major U.S. airport with the lowest (cheapest) long-term parking daily rate in 2026 is '{parking.airport_name}'. "
        "The source should explicitly rank or state it as the cheapest (or tie for cheapest) among major U.S. airports."
    )
    await evaluator.verify(
        claim=claim_airport,
        node=node_airport_name,
        sources=parking.urls if parking and parking.urls else None,
        additional_instruction="Prefer rankings or comparative summaries for 2026 data. Allow minor wording variants like 'least expensive' or 'lowest daily rate'."
    )

    # Leaf: daily_rate (critical)
    node_daily_rate = evaluator.add_leaf(
        id="parking_daily_rate",
        desc="The daily parking rate is correctly stated",
        parent=parent,
        critical=True
    )
    claim_rate = (
        f"The long-term parking daily rate at {parking.airport_name or 'the identified airport'} is {parking.daily_rate} per day (2026 data)."
    )
    await evaluator.verify(
        claim=claim_rate,
        node=node_daily_rate,
        sources=parking.urls if parking and parking.urls else None,
        additional_instruction="Confirm that the quoted daily rate matches the source for the identified airport's long-term/on-airport daily parking product. "
                               "If multiple lots exist, accept the cheapest official on-airport long-term daily rate."
    )

    # Leaf: reference_url (critical)
    node_parking_ref = evaluator.add_leaf(
        id="parking_reference_url",
        desc="A valid reference URL supporting the parking rate information is provided",
        parent=parent,
        critical=True
    )
    claim_parking_ref = (
        f"This page provides long-term parking daily rate information for {parking.airport_name or 'the identified airport'}."
    )
    await evaluator.verify(
        claim=claim_parking_ref,
        node=node_parking_ref,
        sources=parking.urls if parking and parking.urls else None,
        additional_instruction="The page should discuss official airport parking rates or a reliable aggregator comparing airports' long-term daily rates."
    )

    # Parking cost calculation (critical)
    calc_parent = evaluator.add_parallel(
        id="parking_cost_calculation",
        desc="Calculation of total parking cost for 14-day trip",
        parent=root_node,
        critical=True
    )

    # total_cost leaf (use custom deterministic check)
    extracted_rate = parse_currency_to_float(parking.daily_rate if parking else None)
    extracted_total = parse_currency_to_float(parking.total_cost_14_days if parking else None)
    expected_total = extracted_rate * TRIP_PARKING_DAYS if extracted_rate is not None else None
    total_ok = approx_equal(extracted_total, expected_total)

    evaluator.add_custom_node(
        result=bool(total_ok),
        id="parking_total_cost",
        desc="The total 14-day parking cost is correctly calculated by multiplying the daily rate by 14 days",
        parent=calc_parent,
        critical=True
    )

    # reference_url leaf (critical) – at least one URL supporting the rate used
    node_parking_calc_ref = evaluator.add_leaf(
        id="parking_cost_reference_url",
        desc="A valid reference URL supporting the parking rate information is provided",
        parent=calc_parent,
        critical=True
    )
    claim_parking_calc_ref = (
        f"This page provides the long-term parking daily rate for {parking.airport_name or 'the identified airport'}, which can be used to calculate a 14-day total."
    )
    await evaluator.verify(
        claim=claim_parking_calc_ref,
        node=node_parking_calc_ref,
        sources=parking.urls if parking and parking.urls else None,
        additional_instruction="The page must include the daily rate information that serves as the basis for the 14-day calculation."
    )


async def verify_pass_section(evaluator: Evaluator, root_node, passes: PassExtraction) -> None:
    # Parent node: national park pass cost (critical)
    parent = evaluator.add_parallel(
        id="national_park_pass_cost",
        desc="Determination of total cost for America the Beautiful Annual Passes for the family",
        parent=root_node,
        critical=True
    )

    # us_resident_pass_cost leaf – verify $80
    node_pass_price = evaluator.add_leaf(
        id="us_resident_pass_cost",
        desc="The 2026 U.S. resident annual pass cost of $80 per pass is correctly stated",
        parent=parent,
        critical=True
    )
    claim_pass_price = "The America the Beautiful Annual Pass price for 2026 is $80 (per pass)."
    await evaluator.verify(
        claim=claim_pass_price,
        node=node_pass_price,
        sources=passes.urls if passes and passes.urls else None,
        additional_instruction="Use official NPS/USGS or authoritative government sources. Minor wording differences are acceptable."
    )

    # number_of_passes_needed leaf – check equals 2
    parsed_needed = parse_int_from_str(passes.passes_needed if passes else None)
    evaluator.add_custom_node(
        result=(parsed_needed == EXPECTED_PASSES_NEEDED),
        id="number_of_passes_needed",
        desc="Correctly identifies that 2 adult passes are needed (children under 16 are covered by adult passes)",
        parent=parent,
        critical=True
    )

    # total_pass_cost leaf – check equals 160
    extracted_price = parse_currency_to_float(passes.pass_price if passes else None)
    extracted_total = parse_currency_to_float(passes.total_pass_cost if passes else None)

    # Use fallbacks if missing to still check coherence against expected rubric
    price_for_check = extracted_price if extracted_price is not None else EXPECTED_PASS_PRICE_USD
    needed_for_check = parsed_needed if parsed_needed is not None else EXPECTED_PASSES_NEEDED
    expected_total = price_for_check * needed_for_check if price_for_check is not None and needed_for_check is not None else None
    total_ok = approx_equal(extracted_total, expected_total) and approx_equal(expected_total, EXPECTED_TOTAL_PASS_COST)

    evaluator.add_custom_node(
        result=bool(total_ok),
        id="total_pass_cost",
        desc="The total cost is correctly calculated as 2 × $80 = $160",
        parent=parent,
        critical=True
    )

    # reference_url leaf – verify pricing/policy page
    node_pass_ref = evaluator.add_leaf(
        id="pass_reference_url",
        desc="A valid reference URL supporting the national park pass pricing is provided",
        parent=parent,
        critical=True
    )
    claim_pass_ref = "This page states the America the Beautiful Annual Pass price (and/or confirms children under 16 are admitted free)."
    await evaluator.verify(
        claim=claim_pass_ref,
        node=node_pass_ref,
        sources=passes.urls if passes and passes.urls else None,
        additional_instruction="Prefer official NPS/USGS sites. Either explicit $80 price or children-under-16 policy is acceptable to support the answer."
    )


async def verify_baggage_section(evaluator: Evaluator, root_node, baggage: BaggageExtraction) -> None:
    # Parent node: carry-on baggage compliance (critical)
    parent = evaluator.add_parallel(
        id="carry_on_baggage_compliance",
        desc="Verification that the specified carry-on bag meets 2026 airline requirements",
        parent=root_node,
        critical=True
    )

    # dimensions_compliance leaf
    node_dim = evaluator.add_leaf(
        id="dimensions_compliance",
        desc="Correctly verifies that 22″ × 14″ × 9″ dimensions meet or are within the standard 2026 carry-on size limits",
        parent=parent,
        critical=True
    )
    claim_dim = (
        "A carry-on bag of 22 x 14 x 9 inches (approximately 56 x 36 x 23 cm) is within standard airline carry-on size limits "
        "for international flights in 2026 (allowing reasonable airline-by-airline variation)."
    )
    await evaluator.verify(
        claim=claim_dim,
        node=node_dim,
        sources=baggage.urls if baggage and baggage.urls else None,
        additional_instruction="Accept authoritative airline policy pages or reputable aggregators summarizing international carry-on size limits."
    )

    # weight_compliance leaf
    node_weight = evaluator.add_leaf(
        id="weight_compliance",
        desc="Correctly verifies that 20 pounds is within the 15-22 pound international carry-on weight range",
        parent=parent,
        critical=True
    )
    claim_weight = (
        "A 20-pound (about 9 kg) carry-on is within the typical international airline carry-on weight allowance range (about 15–22 lb / 7–10 kg) in 2026."
    )
    await evaluator.verify(
        claim=claim_weight,
        node=node_weight,
        sources=baggage.urls if baggage and baggage.urls else None,
        additional_instruction="Accept authoritative airline policy pages or reputable aggregators summarizing weight limits. Allow small rounding differences."
    )

    # reference_url leaf
    node_bag_ref = evaluator.add_leaf(
        id="baggage_reference_url",
        desc="A valid reference URL supporting the baggage policy information is provided",
        parent=parent,
        critical=True
    )
    claim_bag_ref = "This page provides airline carry-on size and/or weight policy details relevant to international flights."
    await evaluator.verify(
        claim=claim_bag_ref,
        node=node_bag_ref,
        sources=baggage.urls if baggage and baggage.urls else None,
        additional_instruction="The page should contain specific size and/or weight allowances for carry-on baggage."
    )


async def verify_booking_section(evaluator: Evaluator, root_node, booking: BookingExtraction) -> None:
    # Parent node: international flight booking window (critical)
    parent = evaluator.add_parallel(
        id="international_flight_booking_window",
        desc="Statement of recommended advance booking timeframe for international flights",
        parent=root_node,
        critical=True
    )

    # booking_timeframe leaf – verify "3-5 months" with sources
    node_booking = evaluator.add_leaf(
        id="booking_timeframe",
        desc="The recommended booking window of 3-5 months in advance for international flights is correctly stated",
        parent=parent,
        critical=True
    )
    claim_booking = "The recommended advance booking window for international flights is 3–5 months."
    await evaluator.verify(
        claim=claim_booking,
        node=node_booking,
        sources=booking.urls if booking and booking.urls else None,
        additional_instruction="Use reputable travel industry sources or airlines. Allow phrasing variants like 'around 3 to 5 months'."
    )

    # reference_url leaf
    node_booking_ref = evaluator.add_leaf(
        id="booking_reference_url",
        desc="A valid reference URL supporting the booking timeframe recommendation is provided",
        parent=parent,
        critical=True
    )
    claim_booking_ref = "This page provides a recommended advance booking timeframe for international flights."
    await evaluator.verify(
        claim=claim_booking_ref,
        node=node_booking_ref,
        sources=booking.urls if booking and booking.urls else None,
        additional_instruction="Page should contain explicit booking lead-time guidance (in months) for international flights."
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
    Entry point to evaluate an answer for the 2026 international trip planning task.
    """
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

    # Perform extractions (in parallel)
    parking_task = evaluator.extract(
        prompt=prompt_extract_parking(),
        template_class=ParkingExtraction,
        extraction_name="parking_extraction"
    )
    pass_task = evaluator.extract(
        prompt=prompt_extract_pass(),
        template_class=PassExtraction,
        extraction_name="pass_extraction"
    )
    baggage_task = evaluator.extract(
        prompt=prompt_extract_baggage(),
        template_class=BaggageExtraction,
        extraction_name="baggage_extraction"
    )
    booking_task = evaluator.extract(
        prompt=prompt_extract_booking(),
        template_class=BookingExtraction,
        extraction_name="booking_extraction"
    )

    parking, passes, baggage, booking = await asyncio.gather(
        parking_task, pass_task, baggage_task, booking_task
    )

    # Build verification tree per rubric
    await verify_parking_section(evaluator, root, parking)
    await verify_pass_section(evaluator, root, passes)
    await verify_baggage_section(evaluator, root, baggage)
    await verify_booking_section(evaluator, root, booking)

    # Return structured evaluation summary
    return evaluator.get_summary()