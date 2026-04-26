import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# ---------------------------------------------------------------------------
# Task-specific constants
# ---------------------------------------------------------------------------
TASK_ID = "pwm_bcn_pls_itinerary_aa_grace_bay_march_2026"
TASK_DESCRIPTION = """You are planning a vacation from Portland, Maine to Turks and Caicos Islands in March 2026, with a stopover in Barcelona, Spain. You must depart Portland on a Saturday in March 2026 and spend at least one night in Barcelona before continuing to Turks and Caicos. All flights must be operated by American Airlines. You will check one bag that measures 28 inches (length) + 18 inches (width) + 16 inches (height) and weighs 48 pounds.

For accommodation in Turks and Caicos, you must stay at Grace Bay Club in their adults-only section (Hotel Building), as you are traveling with your 17-year-old daughter.

Provide a complete travel itinerary that includes:
1. Your outbound flight route from Portland (PWM) to Barcelona (BCN), including any necessary connection cities and airlines
2. Your flight from Barcelona (BCN) to Providenciales (PLS), Turks and Caicos
3. Confirmation that your checked baggage complies with American Airlines international baggage policy (include the maximum dimensions and weight limits)
4. The name of the specific section of Grace Bay Club where you will stay and confirm it meets the age requirement
5. Reference URLs for your flight routes and accommodation information
"""

# Fallback URLs (used only if the answer does not provide relevant sources)
AA_BAG_POLICY_FALLBACK_URLS = [
    "https://www.aa.com/i18n/travel-info/baggage/checked-baggage.jsp",
    "https://www.aa.com/i18n/travel-info/baggage/checked-bags.jsp",
]
GRACE_BAY_FALLBACK_URLS = [
    "https://www.gracebayresorts.com/grace-bay-club/the-hotel/",
    "https://www.gracebayresorts.com/grace-bay-club/",
]

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class FlightSegment(BaseModel):
    from_airport: Optional[str] = None  # IATA like PWM
    to_airport: Optional[str] = None    # IATA like CLT/PHL/JFK/etc.
    operating_carrier: Optional[str] = None  # e.g., American Airlines
    flight_number: Optional[str] = None
    connection_city: Optional[str] = None


class FlightItineraryExtraction(BaseModel):
    departure_airport: Optional[str] = None   # Should be PWM for outbound; BCN for onward
    departure_date_text: Optional[str] = None # Any human-readable date text
    arrival_airport: Optional[str] = None     # Should be BCN for outbound; PLS for onward
    segments: List[FlightSegment] = Field(default_factory=list)
    route_urls: List[str] = Field(default_factory=list)


class StopoverExtraction(BaseModel):
    has_at_least_one_night: Optional[bool] = None
    nights_in_barcelona: Optional[int] = None
    arrival_date_text_bcn: Optional[str] = None
    onward_departure_date_text_bcn: Optional[str] = None


class BaggageExtraction(BaseModel):
    length_in: Optional[str] = None    # Expect "28" or "28 in"
    width_in: Optional[str] = None     # "18"
    height_in: Optional[str] = None    # "16"
    weight_lbs: Optional[str] = None   # "48"
    calculated_linear_inches: Optional[str] = None  # "62" if stated
    stated_aa_limit_linear_inches: Optional[str] = None  # "62"
    stated_aa_limit_weight_lbs: Optional[str] = None     # "50"
    compliance_conclusion: Optional[str] = None  # e.g., "compliant", "within limits"
    baggage_policy_urls: List[str] = Field(default_factory=list)


class AccommodationExtraction(BaseModel):
    hotel_name: Optional[str] = None                   # Expect "Grace Bay Club"
    section_name: Optional[str] = None                 # Expect "Hotel Building" or "The Hotel (adults-only)"
    adults_only_min_age: Optional[str] = None          # Expect "16"
    confirm_17yo_meets_min_age: Optional[bool] = None  # True if explicitly confirmed
    accommodation_urls: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction Prompts
# ---------------------------------------------------------------------------

def prompt_extract_outbound() -> str:
    return """
    Extract the outbound flight itinerary from the answer for the trip from Portland, Maine to Barcelona, Spain.
    Required fields:
    - departure_airport: The stated origin airport IATA code (e.g., PWM).
    - departure_date_text: The stated outbound departure date text (e.g., "Saturday, March 14, 2026").
    - arrival_airport: The final arrival airport IATA code for the outbound (should be BCN).
    - segments: List of segments, each with:
        - from_airport (IATA)
        - to_airport (IATA)
        - operating_carrier (the actual operating airline, not just the marketing carrier; include "American Airlines" if indicated)
        - flight_number (if mentioned)
        - connection_city (if mentioned; otherwise null)
    - route_urls: All URLs cited in the answer that support this outbound flight routing.

    Return null for any field not explicitly present. Do not invent information.
    """


def prompt_extract_onward() -> str:
    return """
    Extract the onward flight itinerary from the answer for the trip from Barcelona (BCN) to Providenciales (PLS), Turks and Caicos.
    Required fields:
    - departure_airport: The stated origin airport IATA code for onward (should be BCN).
    - departure_date_text: The stated onward departure date text (if present; else null).
    - arrival_airport: The final arrival airport IATA code for the onward (should be PLS).
    - segments: List of segments, each with:
        - from_airport (IATA)
        - to_airport (IATA)
        - operating_carrier (actual operating airline)
        - flight_number (if mentioned)
        - connection_city (if mentioned; else null)
    - route_urls: All URLs cited in the answer that support this BCN->PLS routing.

    Return null for any field not explicitly present. Do not invent information.
    """


def prompt_extract_stopover() -> str:
    return """
    Extract the Barcelona stopover details from the answer.
    Required fields:
    - has_at_least_one_night: true if the answer clearly includes at least one hotel night in Barcelona; false otherwise.
    - nights_in_barcelona: integer count of nights, if explicitly stated (e.g., 1, 2); else null.
    - arrival_date_text_bcn: arrival date text into BCN if stated; else null.
    - onward_departure_date_text_bcn: departure date text from BCN for the onward leg, if stated; else null.

    Only use what is in the answer text. Do not infer beyond the answer.
    """


def prompt_extract_baggage() -> str:
    return """
    Extract the checked baggage details and American Airlines policy limits as stated in the answer.
    Required fields:
    - length_in: the stated bag length in inches (e.g., "28")
    - width_in: the stated bag width in inches (e.g., "18")
    - height_in: the stated bag height in inches (e.g., "16")
    - weight_lbs: the stated bag weight in pounds (e.g., "48")
    - calculated_linear_inches: the stated sum (if the answer provides it), e.g., "62"; else null
    - stated_aa_limit_linear_inches: the AA maximum linear dimension stated in the answer (e.g., "62"); else null
    - stated_aa_limit_weight_lbs: the AA maximum weight stated in the answer (e.g., "50"); else null
    - compliance_conclusion: the answer's explicit conclusion regarding compliance (e.g., "complies", "within limits", "allowed") if present; else null
    - baggage_policy_urls: any URLs cited by the answer to support AA baggage policy; else empty list

    Use only information that appears in the answer.
    """


def prompt_extract_accommodation() -> str:
    return """
    Extract the accommodation details from the answer for Turks and Caicos.
    Required fields:
    - hotel_name: the name of the property (should be "Grace Bay Club" if correct)
    - section_name: the specific section where the stay is planned (e.g., "Hotel Building", "The Hotel (adults-only)")
    - adults_only_min_age: the stated minimum age for the adults-only Hotel Building (e.g., "16") if the answer states it; else null
    - confirm_17yo_meets_min_age: true if the answer explicitly confirms a 17-year-old meets the minimum age; else false or null
    - accommodation_urls: all URLs cited by the answer to support the accommodation and section details

    Use only information from the answer. Do not invent data.
    """


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_int_safe(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        digits = "".join(ch for ch in s if (ch.isdigit()))
        if digits == "":
            return None
        return int(digits)
    except Exception:
        return None


def _bool_from_text(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    low = s.strip().lower()
    if any(k in low for k in ["yes", "true", "complies", "within", "meets", "allowed"]):
        return True
    if any(k in low for k in ["no", "false", "exceeds", "overweight", "oversize", "not allowed"]):
        return False
    return None


def _non_empty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------

async def verify_outbound(evaluator: Evaluator, parent, outbound: FlightItineraryExtraction) -> None:
    node = evaluator.add_parallel(
        id="Outbound_PWM_to_BCN",
        desc="Outbound flight routing from Portland (PWM) to Barcelona (BCN) meeting timing and airline constraints, including required flight-route URL(s).",
        parent=parent,
        critical=True,
    )

    # Critical existence of route URLs first (to gate URL-based verifications)
    evaluator.add_custom_node(
        result=_non_empty_urls(outbound.route_urls),
        id="Outbound_Route_Reference_URLs",
        desc="Provides at least one reference URL supporting the outbound flight route information.",
        parent=node,
        critical=True,
    )

    # Critical: Segments and operating carriers are listed (answer content check via extraction presence)
    has_segments_and_operators = bool(outbound.segments) and all(
        (seg.operating_carrier is not None and str(seg.operating_carrier).strip() != "")
        for seg in outbound.segments
    )
    evaluator.add_custom_node(
        result=has_segments_and_operators,
        id="Outbound_Segments_And_Operating_Carriers_Listed",
        desc="Lists the flight segment(s) including any connection city/cities and identifies the operating carrier for each segment.",
        parent=node,
        critical=True,
    )

    # PWM as departure (URL-verified)
    dep_pwm_leaf = evaluator.add_leaf(
        id="Outbound_Departure_Airport_PWM",
        desc="Outbound itinerary departs from Portland, Maine (PWM).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The outbound flight route departs from Portland International Jetport (IATA: PWM).",
        node=dep_pwm_leaf,
        sources=outbound.route_urls,
        additional_instruction="Look for route details that show PWM as the origin for the outbound journey.",
    )

    # Saturday in March 2026 (answer content check)
    sat_leaf = evaluator.add_leaf(
        id="Outbound_Departure_Date_Constraint",
        desc="Outbound departure occurs on a Saturday in March 2026.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The outbound departure occurs on a Saturday in March 2026.",
        node=sat_leaf,
        additional_instruction="Accept if the answer explicitly states a Saturday date in March 2026, or clearly says 'Saturday in March 2026' for the outbound departure.",
    )

    # BCN as arrival (URL-verified)
    arr_bcn_leaf = evaluator.add_leaf(
        id="Outbound_Arrival_Airport_BCN",
        desc="Outbound itinerary arrives in Barcelona (BCN).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The outbound flight route arrives in Barcelona–El Prat Airport (IATA: BCN).",
        node=arr_bcn_leaf,
        sources=outbound.route_urls,
        additional_instruction="The routing should show BCN as the final destination of the outbound leg.",
    )

    # All outbound segments AA-operated (URL-verified)
    all_aa_leaf = evaluator.add_leaf(
        id="Outbound_All_Segments_AA_Operated",
        desc="All outbound flight segments are operated by American Airlines.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Every outbound flight segment is operated by American Airlines (no segment operated by a different airline).",
        node=all_aa_leaf,
        sources=outbound.route_urls,
        additional_instruction="Codeshares are acceptable only if the operating carrier is American Airlines. Look for 'Operated by American Airlines' (or equivalent) per segment.",
    )


async def verify_stopover(evaluator: Evaluator, parent, stop: StopoverExtraction) -> None:
    node = evaluator.add_parallel(
        id="Barcelona_Stopover",
        desc="Barcelona stopover satisfies minimum duration requirement.",
        parent=parent,
        critical=True,
    )

    at_least_one_night_leaf = evaluator.add_leaf(
        id="At_Least_One_Night_in_Barcelona",
        desc="Itinerary includes at least one night in Barcelona between arrival and departure onward.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The itinerary includes at least one night in Barcelona (arrival day to at least the next day) before continuing onward.",
        node=at_least_one_night_leaf,
        additional_instruction="Judge based on the answer text only. Accept if the answer clearly mentions 1 or more hotel night(s) in Barcelona.",
    )


async def verify_onward(evaluator: Evaluator, parent, onward: FlightItineraryExtraction) -> None:
    node = evaluator.add_parallel(
        id="BCN_to_PLS",
        desc="Onward flight routing from Barcelona (BCN) to Providenciales (PLS) meeting airline constraints, including required flight-route URL(s).",
        parent=parent,
        critical=True,
    )

    # Critical existence of route URLs first (to gate URL-based verifications)
    evaluator.add_custom_node(
        result=_non_empty_urls(onward.route_urls),
        id="Onward_Route_Reference_URLs",
        desc="Provides at least one reference URL supporting the BCN->PLS flight route information.",
        parent=node,
        critical=True,
    )

    # Critical: Segments and operating carriers are listed (extraction presence)
    has_segments_and_operators = bool(onward.segments) and all(
        (seg.operating_carrier is not None and str(seg.operating_carrier).strip() != "")
        for seg in onward.segments
    )
    evaluator.add_custom_node(
        result=has_segments_and_operators,
        id="Onward_Segments_And_Operating_Carriers_Listed",
        desc="Lists the flight segment(s) including any connection city/cities and identifies the operating carrier for each segment.",
        parent=node,
        critical=True,
    )

    # BCN as departure (URL-verified)
    dep_bcn_leaf = evaluator.add_leaf(
        id="Onward_Departure_Airport_BCN",
        desc="Onward itinerary departs from Barcelona (BCN).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The onward flight route departs from Barcelona–El Prat Airport (IATA: BCN).",
        node=dep_bcn_leaf,
        sources=onward.route_urls,
        additional_instruction="Look for BCN as the origin for the onward routing.",
    )

    # PLS as arrival (URL-verified)
    arr_pls_leaf = evaluator.add_leaf(
        id="Onward_Arrival_Airport_PLS",
        desc="Onward itinerary arrives at Providenciales (PLS), Turks and Caicos.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The onward flight route arrives at Providenciales International Airport (IATA: PLS).",
        node=arr_pls_leaf,
        sources=onward.route_urls,
        additional_instruction="The final destination for the onward leg should show PLS.",
    )

    # All onward segments AA-operated (URL-verified)
    all_aa_leaf = evaluator.add_leaf(
        id="Onward_All_Segments_AA_Operated",
        desc="All onward flight segments are operated by American Airlines.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Every onward flight segment is operated by American Airlines (no segment operated by a different airline).",
        node=all_aa_leaf,
        sources=onward.route_urls,
        additional_instruction="Codeshares are acceptable only if the operating carrier is American Airlines. Look for 'Operated by American Airlines' per segment.",
    )


async def verify_baggage(evaluator: Evaluator, parent, bag: BaggageExtraction) -> None:
    node = evaluator.add_parallel(
        id="Baggage_Compliance",
        desc="Checked bag is evaluated against the stated American Airlines international checked baggage limits; limits are stated; compliance is concluded.",
        parent=parent,
        critical=True,
    )

    # Bag dimensions and weight stated (answer content via extraction presence)
    dims_weight_stated = all([
        bag.length_in is not None and str(bag.length_in).strip() != "",
        bag.width_in is not None and str(bag.width_in).strip() != "",
        bag.height_in is not None and str(bag.height_in).strip() != "",
        bag.weight_lbs is not None and str(bag.weight_lbs).strip() != "",
    ])
    evaluator.add_custom_node(
        result=dims_weight_stated,
        id="Bag_Dimensions_And_Weight_Stated",
        desc="States the checked bag dimensions (28 inches + 18 inches + 16 inches; may include 62 linear inches) and weight (48 lbs).",
        parent=node,
        critical=True,
    )

    # AA policy limits stated (answer content via extraction presence)
    limits_stated = all([
        bag.stated_aa_limit_linear_inches is not None and str(bag.stated_aa_limit_linear_inches).strip() != "",
        bag.stated_aa_limit_weight_lbs is not None and str(bag.stated_aa_limit_weight_lbs).strip() != "",
    ])
    evaluator.add_custom_node(
        result=limits_stated,
        id="AA_International_Checked_Bag_Limits_Stated",
        desc="States the applicable American Airlines international checked-baggage maximum dimension and weight limits (62 linear inches / 158 cm; 50 lbs / 23 kg for economy class, as specified in constraints).",
        parent=node,
        critical=True,
    )

    # Compliance conclusion correct (compute and check alignment with answer)
    L = _parse_int_safe(bag.length_in)
    W = _parse_int_safe(bag.width_in)
    H = _parse_int_safe(bag.height_in)
    weight = _parse_int_safe(bag.weight_lbs)
    stated_lin = _parse_int_safe(bag.stated_aa_limit_linear_inches)
    stated_wt = _parse_int_safe(bag.stated_aa_limit_weight_lbs)

    computed_linear_sum = L + W + H if (L is not None and W is not None and H is not None) else None
    # Assume AA standard limits if stated values not parsed, but note: this check is gated by limits stated leaf.
    lin_limit = stated_lin if stated_lin is not None else 62
    wt_limit = stated_wt if stated_wt is not None else 50

    calc_ok = (
        (computed_linear_sum is not None and computed_linear_sum <= lin_limit) and
        (weight is not None and weight <= wt_limit)
    )
    answer_conclusion_ok = _bool_from_text(bag.compliance_conclusion)
    compliance_correct = bool(calc_ok and (answer_conclusion_ok is None or answer_conclusion_ok is True))

    evaluator.add_custom_node(
        result=compliance_correct,
        id="Baggage_Compliance_Conclusion_Correct",
        desc="Correctly concludes whether the bag complies by comparing the stated bag size/weight to the stated AA limits.",
        parent=node,
        critical=True,
    )


async def verify_accommodation(evaluator: Evaluator, parent, accom: AccommodationExtraction) -> None:
    node = evaluator.add_parallel(
        id="Accommodation",
        desc="Accommodation meets the specified hotel/section and age-eligibility requirements and includes required accommodation reference URL(s).",
        parent=parent,
        critical=True,
    )

    # Critical: Accommodation reference URLs provided (to meet requirement)
    evaluator.add_custom_node(
        result=_non_empty_urls(accom.accommodation_urls),
        id="Accommodation_Reference_URLs",
        desc="Provides at least one reference URL supporting the accommodation/section information.",
        parent=node,
        critical=True,
    )

    # Grace Bay Club selected (answer content)
    is_grace_bay = accom.hotel_name is not None and ("grace bay club" in accom.hotel_name.strip().lower())
    evaluator.add_custom_node(
        result=is_grace_bay,
        id="Grace_Bay_Club_Selected",
        desc="Accommodation is at Grace Bay Club.",
        parent=node,
        critical=True,
    )

    # Adults-only Hotel Building specified (answer content)
    section_ok = accom.section_name is not None and (
        ("hotel" in accom.section_name.strip().lower()) or ("hotel building" in accom.section_name.strip().lower())
    )
    evaluator.add_custom_node(
        result=section_ok,
        id="Adults_Only_Hotel_Building_Section_Specified",
        desc="Specifies the stay is in Grace Bay Club’s adults-only section (Hotel Building).",
        parent=node,
        critical=True,
    )

    # Minimum age stated as 16 (answer content)
    min_age_val = _parse_int_safe(accom.adults_only_min_age)
    min_age_stated_correct = (min_age_val == 16)
    evaluator.add_custom_node(
        result=min_age_stated_correct,
        id="Adults_Only_Section_Minimum_Age_Stated",
        desc="States the minimum age requirement for the adults-only Hotel Building section (16 years, per constraints).",
        parent=node,
        critical=True,
    )

    # 17-year-old meets minimum age (answer content and logic)
    seventeen_meets = True if (min_age_val is not None and 17 >= min_age_val) else False
    # If the answer explicitly confirms, that also helps; but require logic to be true.
    if accom.confirm_17yo_meets_min_age is not None:
        seventeen_meets = bool(accom.confirm_17yo_meets_min_age and seventeen_meets)

    evaluator.add_custom_node(
        result=seventeen_meets,
        id="17_Year_Old_Meets_Minimum_Age",
        desc="Confirms the 17-year-old traveler meets the stated minimum age requirement for the adults-only section.",
        parent=node,
        critical=True,
    )


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
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
        default_model=model,
    )

    # Create the top-level critical node that mirrors the rubric root
    complete_node = evaluator.add_parallel(
        id="Complete_Travel_Itinerary",
        desc="Evaluate the required itinerary: PWM -> BCN (stopover) -> PLS, all flights AA-operated, baggage compliance confirmation (with stated limits), Grace Bay Club adults-only Hotel Building eligibility (min age), and required reference URLs for flights and accommodation.",
        parent=root,
        critical=True,
    )

    # Run extractions in parallel
    outbound_extraction_task = evaluator.extract(
        prompt=prompt_extract_outbound(),
        template_class=FlightItineraryExtraction,
        extraction_name="outbound_itinerary",
    )
    onward_extraction_task = evaluator.extract(
        prompt=prompt_extract_onward(),
        template_class=FlightItineraryExtraction,
        extraction_name="onward_itinerary",
    )
    stopover_extraction_task = evaluator.extract(
        prompt=prompt_extract_stopover(),
        template_class=StopoverExtraction,
        extraction_name="barcelona_stopover",
    )
    baggage_extraction_task = evaluator.extract(
        prompt=prompt_extract_baggage(),
        template_class=BaggageExtraction,
        extraction_name="baggage_info",
    )
    accommodation_extraction_task = evaluator.extract(
        prompt=prompt_extract_accommodation(),
        template_class=AccommodationExtraction,
        extraction_name="accommodation_info",
    )

    outbound, onward, stopover, bag, accom = await asyncio.gather(
        outbound_extraction_task,
        onward_extraction_task,
        stopover_extraction_task,
        baggage_extraction_task,
        accommodation_extraction_task,
    )

    # Add ground-truth-ish constraints summary for transparency
    evaluator.add_custom_info(
        info={
            "required_departure_airport": "PWM",
            "required_arrival_airports": {"outbound_final": "BCN", "onward_final": "PLS"},
            "outbound_departure_timing_constraint": "Saturday in March 2026",
            "all_segments_operating_carrier": "American Airlines",
            "bag_spec": {"L": 28, "W": 18, "H": 16, "Weight_lbs": 48, "Linear_sum": 62},
            "aa_checked_bag_limits_expected": {"linear_inches": 62, "weight_lbs": 50},
            "accommodation_property": "Grace Bay Club",
            "accommodation_section": "Hotel (adults-only) building",
            "adults_only_min_age_expected": 16,
            "traveler_age": 17,
        },
        info_type="constraints",
        info_name="task_constraints_summary"
    )

    # Verify each rubric subtree
    await verify_outbound(evaluator, complete_node, outbound)
    await verify_stopover(evaluator, complete_node, stopover)
    await verify_onward(evaluator, complete_node, onward)
    await verify_baggage(evaluator, complete_node, bag)
    await verify_accommodation(evaluator, complete_node, accom)

    return evaluator.get_summary()