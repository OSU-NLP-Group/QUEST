import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "yellowstone_mlk_2026"
TASK_DESCRIPTION = (
    "A family of four U.S. residents is planning a winter trip to Yellowstone National Park over MLK weekend 2026 "
    "(Friday, January 17 through Monday, January 20). They are flying from Bangor, Maine and need comprehensive travel planning assistance.\n\n"
    "Please provide the following information with supporting reference URLs:\n\n"
    "1. Winter Lodging: Which hotel inside Yellowstone National Park should they book that is accessible by regular vehicle (car/van) "
    "without requiring snowcoach or snowmobile transportation during winter? The hotel must be open during their visit dates.\n\n"
    "2. Flight Routing: Plan their flight route from Bangor to the Yellowstone region:\n"
    "   - Which budget airline operates from Bangor International Airport (BGR)?\n"
    "   - Which major hub city should they connect through to reach Montana?\n"
    "   - Which Montana airport is closest to Yellowstone's north entrance?\n\n"
    "3. Passenger Rights: If their outbound flight is canceled due to winter weather or significantly delayed:\n"
    "   - Are they entitled to a refund even though the cancellation is weather-related?\n"
    "   - How many hours of delay qualifies as a \"significant delay\" for domestic flights under current DOT rules?\n"
    "   - Are refunds provided automatically or must they be requested?\n\n"
    "4. Entrance Fees: Calculate their park entrance costs:\n"
    "   - Which specific day of their visit (include date and day of week) qualifies as a fee-free entrance day?\n"
    "   - What is the standard per-vehicle entrance fee for Yellowstone?\n"
    "   - How many days does a standard entrance pass cover?\n\n"
    "For each answer component, provide at least one reference URL from official sources (park service, airline, government agency, etc.) to verify the information."
)

# Ground-truth expectations used for "matches constraint" checks
EXPECTED_LODGING_NAME = "Mammoth Hot Springs Hotel"
EXPECTED_BUDGET_AIRLINE = "Allegiant Air"
EXPECTED_HUB_CITY = "Minneapolis–St. Paul (MSP)"
EXPECTED_CLOSEST_MT_AIRPORT = "Bozeman Yellowstone International Airport (BZN)"
EXPECTED_FEE_FREE_DATE = "January 19, 2026"
EXPECTED_FEE_FREE_WEEKDAY = "Monday"
EXPECTED_DELAY_THRESHOLD_DESC = "3 or more hours"
EXPECTED_PASS_VALIDITY_DESC = "7 consecutive days"

# Data models for extraction
class LodgingExtraction(BaseModel):
    hotel_name: Optional[str] = None
    open_dates_text: Optional[str] = None
    accessibility_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FlightRoutingExtraction(BaseModel):
    budget_airline_name: Optional[str] = None
    budget_airline_sources: List[str] = Field(default_factory=list)
    hub_city: Optional[str] = None
    hub_sources: List[str] = Field(default_factory=list)
    closest_airport_name: Optional[str] = None
    closest_airport_distance_metric: Optional[str] = None
    airport_sources: List[str] = Field(default_factory=list)


class PassengerRightsExtraction(BaseModel):
    refund_entitlement_statement: Optional[str] = None
    refund_sources: List[str] = Field(default_factory=list)
    delay_threshold_statement: Optional[str] = None
    delay_sources: List[str] = Field(default_factory=list)
    refund_process_statement: Optional[str] = None
    refund_process_sources: List[str] = Field(default_factory=list)


class EntranceFeesExtraction(BaseModel):
    fee_free_day_date_text: Optional[str] = None
    fee_free_day_weekday_text: Optional[str] = None
    fee_free_sources: List[str] = Field(default_factory=list)
    per_vehicle_fee_amount_text: Optional[str] = None
    fee_amount_sources: List[str] = Field(default_factory=list)
    pass_validity_days_text: Optional[str] = None
    validity_sources: List[str] = Field(default_factory=list)


# Extraction prompts
def prompt_extract_lodging() -> str:
    return """
    Extract the in-park winter lodging recommended by the answer that is stated to be accessible by regular vehicle (car/van) in winter and open during the Jan 17–20, 2026 visit window.
    Return fields:
    - hotel_name: the hotel named inside Yellowstone National Park
    - open_dates_text: the answer's statement about winter operating/open status covering Jan 17–20, 2026
    - accessibility_text: the answer's statement that it is accessible by regular vehicle and does not require snowcoach/snowmobile
    - sources: ALL official reference URLs the answer cites to verify winter operations and accessibility (e.g., NPS or Yellowstone National Park Lodges pages, park roads pages)
    Only extract URLs that appear in the answer (plain or markdown). If none are provided, return an empty list.
    """


def prompt_extract_flight_routing() -> str:
    return """
    Extract flight routing details mentioned in the answer.
    Return fields:
    - budget_airline_name: the budget airline operating from BGR (Bangor International Airport) named in the answer
    - budget_airline_sources: official reference URLs cited in the answer confirming this airline serves BGR (e.g., Allegiant or BGR official pages)
    - hub_city: the major hub city/airport recommended for connecting to reach Montana (e.g., Minneapolis–St. Paul (MSP))
    - hub_sources: official reference URLs cited in the answer supporting MSP as a major hub (airport or airline official sources preferred)
    - closest_airport_name: the Montana airport named as closest to Yellowstone’s North Entrance (e.g., Bozeman Yellowstone International Airport (BZN))
    - closest_airport_distance_metric: the exact distance/time metric text the answer provides (e.g., "~80 miles", "90 minutes")
    - airport_sources: reference URLs cited in the answer (official or authoritative) supporting the claimed proximity to the North Entrance
    Only extract URLs explicitly present in the answer. If any list is missing, return an empty list.
    """


def prompt_extract_passenger_rights() -> str:
    return """
    Extract DOT passenger rights statements mentioned in the answer.
    Return fields:
    - refund_entitlement_statement: the answer's statement whether passengers are entitled to a refund when a flight is canceled due to weather (conditions such as declining rebooking)
    - refund_sources: official DOT URLs cited to support the cancellation-refund rule
    - delay_threshold_statement: the answer's statement of the domestic "significant delay" threshold in hours
    - delay_sources: official DOT URLs cited to support the significant delay threshold
    - refund_process_statement: the answer's statement whether refunds must be automatic or must be requested under current DOT rules
    - refund_process_sources: official DOT URLs cited to support the refund process rule
    Only extract URLs present in the answer. If a URL list is missing, return an empty list.
    """


def prompt_extract_entrance_fees() -> str:
    return """
    Extract entrance fee information mentioned in the answer.
    Return fields:
    - fee_free_day_date_text: the specific calendar date identified as a fee-free day within Jan 17–20, 2026 (e.g., "January 19, 2026")
    - fee_free_day_weekday_text: the weekday named (e.g., "Monday")
    - fee_free_sources: official NPS URLs cited listing the fee-free day
    - per_vehicle_fee_amount_text: the standard per-vehicle entrance fee amount text for Yellowstone (e.g., "$35")
    - fee_amount_sources: official NPS URLs cited listing Yellowstone entrance fees
    - pass_validity_days_text: the validity duration of a standard entrance pass (e.g., "7 consecutive days")
    - validity_sources: official NPS URLs cited confirming the pass validity period
    Only extract URLs explicitly present in the answer. If a URL list is missing, return an empty list.
    """


# Verification subtrees
async def verify_winter_lodging(evaluator: Evaluator, root) -> None:
    lodging_node = evaluator.add_parallel(
        id="winter_lodging",
        desc="Winter lodging inside Yellowstone meeting access and open-date requirements.",
        parent=root,
        critical=True
    )

    # Extract lodging info
    lodging_info = await evaluator.extract(
        prompt=prompt_extract_lodging(),
        template_class=LodgingExtraction,
        extraction_name="lodging_info"
    )

    # Leaf: identifies Mammoth and states it is open during Jan 17–20, 2026
    leaf_choice = evaluator.add_leaf(
        id="lodging_choice_correct_per_constraints",
        desc="Identifies Mammoth Hot Springs Hotel and states it is open during Jan 17–20, 2026.",
        parent=lodging_node,
        critical=True
    )
    claim_choice = (
        f"The answer recommends Mammoth Hot Springs Hotel inside Yellowstone and explicitly states it is open during "
        f"January 17–20, 2026."
    )
    await evaluator.verify(
        claim=claim_choice,
        node=leaf_choice,
        additional_instruction="Check the answer text for both: (1) naming Mammoth Hot Springs Hotel, and (2) a clear statement that it is open during Jan 17–20, 2026."
    )

    # Leaf: official reference URL verifies winter operating dates AND regular vehicle accessibility
    leaf_refs = evaluator.add_leaf(
        id="lodging_official_reference_url",
        desc="Provides official reference URL(s) verifying winter operating dates and regular-vehicle accessibility.",
        parent=lodging_node,
        critical=True
    )
    claim_refs = (
        "Mammoth Hot Springs Hotel is open during the January 17–20, 2026 weekend and is accessible by regular vehicle "
        "(does not require snowcoach or snowmobile) in winter."
    )
    await evaluator.verify(
        claim=claim_refs,
        node=leaf_refs,
        sources=lodging_info.sources,
        additional_instruction="Prefer official sources like NPS or Yellowstone National Park Lodges. The page(s) must clearly support winter operation and car access to Mammoth."
    )


async def verify_flight_routing(evaluator: Evaluator, root) -> None:
    routing_node = evaluator.add_parallel(
        id="flight_routing",
        desc="Flight-route components from Bangor (BGR) to the Yellowstone region.",
        parent=root,
        critical=True
    )

    flight_info = await evaluator.extract(
        prompt=prompt_extract_flight_routing(),
        template_class=FlightRoutingExtraction,
        extraction_name="flight_routing_info"
    )

    # Budget airline from BGR
    budget_node = evaluator.add_parallel(
        id="budget_airline_from_bgr",
        desc="Budget airline operating from BGR.",
        parent=routing_node,
        critical=True
    )
    leaf_budget_match = evaluator.add_leaf(
        id="budget_airline_matches_constraint",
        desc="Identifies Allegiant Air as the budget airline operating from BGR.",
        parent=budget_node,
        critical=True
    )
    claim_budget_match = "The budget airline operating from Bangor International Airport (BGR) is Allegiant Air."
    await evaluator.verify(
        claim=claim_budget_match,
        node=leaf_budget_match,
        additional_instruction="Verify this against the answer text only; allow minor variants like 'Allegiant' vs 'Allegiant Air'."
    )

    leaf_budget_ref = evaluator.add_leaf(
        id="budget_airline_official_reference_url",
        desc="Provides official reference URL confirming Allegiant serves BGR.",
        parent=budget_node,
        critical=True
    )
    claim_budget_ref = "Allegiant Air operates from Bangor International Airport (BGR)."
    await evaluator.verify(
        claim=claim_budget_ref,
        node=leaf_budget_ref,
        sources=flight_info.budget_airline_sources,
        additional_instruction="Prefer official sources such as allegiantair.com or the official BGR airport site listing Allegiant service."
    )

    # Connection hub city
    hub_node = evaluator.add_parallel(
        id="connection_hub_city",
        desc="Major hub city for connecting to reach Montana.",
        parent=routing_node,
        critical=True
    )
    leaf_hub_match = evaluator.add_leaf(
        id="hub_matches_constraint",
        desc="Identifies Minneapolis–St. Paul (MSP) as the hub.",
        parent=hub_node,
        critical=True
    )
    claim_hub_match = "The major hub city/airport to connect through to reach Montana is Minneapolis–St. Paul (MSP)."
    await evaluator.verify(
        claim=claim_hub_match,
        node=leaf_hub_match,
        additional_instruction="Verify the answer text states MSP as the connection hub."
    )

    leaf_hub_ref = evaluator.add_leaf(
        id="hub_official_reference_url",
        desc="Provides official reference URL supporting MSP as a major hub.",
        parent=hub_node,
        critical=True
    )
    claim_hub_ref = "Minneapolis–St. Paul International Airport (MSP) is a major hub airport (e.g., for Sun Country or other carriers)."
    await evaluator.verify(
        claim=claim_hub_ref,
        node=leaf_hub_ref,
        sources=flight_info.hub_sources,
        additional_instruction="Prefer official airport or airline sources indicating MSP's hub status or prominence as a connecting airport."
    )

    # Closest Montana airport to North Entrance
    closest_node = evaluator.add_parallel(
        id="closest_montana_airport_to_north_entrance",
        desc="Closest Montana airport to Yellowstone’s North Entrance.",
        parent=routing_node,
        critical=True
    )

    leaf_airport_answered = evaluator.add_leaf(
        id="airport_answered_as_montana_airport",
        desc="Names a Montana airport and explicitly states a distance/time metric for 'closest to the North Entrance.'",
        parent=closest_node,
        critical=True
    )
    # Use simple verification to check answer content includes a Montana airport name and a metric
    airport_name = flight_info.closest_airport_name or ""
    metric_text = flight_info.closest_airport_distance_metric or ""
    claim_airport_answered = (
        f"The answer names a Montana airport ('{airport_name}') and explicitly provides a distance/time metric "
        f"('{metric_text}') describing closeness to Yellowstone’s North Entrance (Gardiner)."
    )
    await evaluator.verify(
        claim=claim_airport_answered,
        node=leaf_airport_answered,
        additional_instruction="Check the answer text for both a Montana airport name and a quantitative metric (miles/minutes/hours)."
    )

    leaf_airport_ref = evaluator.add_leaf(
        id="airport_proximity_supported_by_reference",
        desc="Provides reference URL(s) supporting the claimed proximity/route/distance relation to the North Entrance.",
        parent=closest_node,
        critical=True
    )
    claim_airport_ref = (
        f"{airport_name if airport_name else 'The named airport'} is the closest Montana airport to Yellowstone’s North Entrance (Gardiner), "
        f"with the distance/time metric consistent with the cited reference(s)."
    )
    await evaluator.verify(
        claim=claim_airport_ref,
        node=leaf_airport_ref,
        sources=flight_info.airport_sources,
        additional_instruction="Prefer official or authoritative sources (airport, DOT, state travel) that support proximity to Yellowstone’s North Entrance."
    )


async def verify_passenger_rights(evaluator: Evaluator, root) -> None:
    rights_node = evaluator.add_parallel(
        id="passenger_rights",
        desc="DOT passenger refund rights for cancellations and significant delays.",
        parent=root,
        critical=True
    )

    rights_info = await evaluator.extract(
        prompt=prompt_extract_passenger_rights(),
        template_class=PassengerRightsExtraction,
        extraction_name="passenger_rights_info"
    )

    # Weather cancellation refund
    weather_node = evaluator.add_parallel(
        id="weather_cancellation_refund",
        desc="Refund entitlement when a flight is canceled due to winter weather.",
        parent=rights_node,
        critical=True
    )
    leaf_refund_entitled = evaluator.add_leaf(
        id="refund_entitlement_per_constraints",
        desc="States passengers are entitled to a refund even if weather-related, when declining rebooking/alternatives.",
        parent=weather_node,
        critical=True
    )
    claim_refund_entitled = (
        "Passengers are entitled to a refund for a canceled flight even if the cancellation is weather-related, "
        "provided they decline rebooking or alternative travel."
    )
    await evaluator.verify(
        claim=claim_refund_entitled,
        node=leaf_refund_entitled,
        additional_instruction="Check the answer text for a clear statement of this refund entitlement condition."
    )

    leaf_refund_ref = evaluator.add_leaf(
        id="refund_entitlement_official_reference_url",
        desc="Provides official DOT reference URL supporting the cancellation-refund rule.",
        parent=weather_node,
        critical=True
    )
    claim_refund_ref = (
        "Under current DOT rules, a refund must be provided when a flight is canceled and the passenger does not accept rebooking, "
        "regardless of the reason (including weather)."
    )
    await evaluator.verify(
        claim=claim_refund_ref,
        node=leaf_refund_ref,
        sources=rights_info.refund_sources,
        additional_instruction="Use official DOT pages describing refund rights for cancellations."
    )

    # Significant delay threshold (domestic)
    delay_node = evaluator.add_parallel(
        id="significant_delay_threshold_domestic",
        desc="Domestic 'significant delay' threshold under current DOT rules.",
        parent=rights_node,
        critical=True
    )
    leaf_delay_match = evaluator.add_leaf(
        id="delay_threshold_matches_constraint",
        desc="States that a significant delay is 3 or more hours for domestic flights.",
        parent=delay_node,
        critical=True
    )
    claim_delay_match = "A 'significant delay' for domestic flights is defined as 3 or more hours under current DOT rules."
    await evaluator.verify(
        claim=claim_delay_match,
        node=leaf_delay_match,
        additional_instruction="Verify the answer text states a 3+ hour threshold for domestic flights."
    )

    leaf_delay_ref = evaluator.add_leaf(
        id="delay_threshold_official_reference_url",
        desc="Provides official DOT reference URL defining the significant-delay threshold.",
        parent=delay_node,
        critical=True
    )
    claim_delay_ref = "DOT defines 'significant delay' for domestic flights as 3 or more hours."
    await evaluator.verify(
        claim=claim_delay_ref,
        node=leaf_delay_ref,
        sources=rights_info.delay_sources,
        additional_instruction="Use official DOT rule summaries or final rule pages that explicitly state the 3+ hour threshold."
    )

    # Automatic vs requested refunds
    auto_node = evaluator.add_parallel(
        id="automatic_vs_requested_refunds",
        desc="Whether refunds are automatic or must be requested.",
        parent=rights_node,
        critical=True
    )
    leaf_auto_match = evaluator.add_leaf(
        id="refund_process_matches_constraint",
        desc="States refunds must be issued automatically under the referenced DOT rule.",
        parent=auto_node,
        critical=True
    )
    claim_auto_match = "Airlines must issue refunds automatically under the current DOT refund rules; passengers should not have to request them."
    await evaluator.verify(
        claim=claim_auto_match,
        node=leaf_auto_match,
        additional_instruction="Verify the answer text states refunds are automatic (proactively issued by airlines)."
    )

    leaf_auto_ref = evaluator.add_leaf(
        id="refund_process_official_reference_url",
        desc="Provides official DOT reference URL supporting the automatic-refund requirement.",
        parent=auto_node,
        critical=True
    )
    claim_auto_ref = "DOT requires airlines to automatically provide refunds when eligible (e.g., cancellations or significant delays) without requiring passengers to request them."
    await evaluator.verify(
        claim=claim_auto_ref,
        node=leaf_auto_ref,
        sources=rights_info.refund_process_sources,
        additional_instruction="Use official DOT rule pages that specify automatic refund issuance."
    )


async def verify_entrance_fees(evaluator: Evaluator, root) -> None:
    fees_node = evaluator.add_parallel(
        id="entrance_fees",
        desc="Requested Yellowstone entrance-fee elements.",
        parent=root,
        critical=True
    )

    fees_info = await evaluator.extract(
        prompt=prompt_extract_entrance_fees(),
        template_class=EntranceFeesExtraction,
        extraction_name="entrance_fees_info"
    )

    # Fee-free day
    fee_free_node = evaluator.add_parallel(
        id="fee_free_day",
        desc="Fee-free entrance day during the visit window.",
        parent=fees_node,
        critical=True
    )
    leaf_fee_free_match = evaluator.add_leaf(
        id="fee_free_day_matches_constraint",
        desc="Identifies Monday, January 19, 2026 as the fee-free entrance day within Jan 17–20, 2026.",
        parent=fee_free_node,
        critical=True
    )
    claim_fee_free_match = "The fee-free entrance day identified in the answer is Monday, January 19, 2026."
    await evaluator.verify(
        claim=claim_fee_free_match,
        node=leaf_fee_free_match,
        additional_instruction="Check the answer text for both the correct date and weekday."
    )

    leaf_fee_free_ref = evaluator.add_leaf(
        id="fee_free_day_official_reference_url",
        desc="Provides official NPS reference URL listing the fee-free day.",
        parent=fee_free_node,
        critical=True
    )
    claim_fee_free_ref = "Monday, January 19, 2026 (MLK Day) is a National Park Service fee-free day applicable to Yellowstone."
    await evaluator.verify(
        claim=claim_fee_free_ref,
        node=leaf_fee_free_ref,
        sources=fees_info.fee_free_sources,
        additional_instruction="Use official NPS pages (fee-free days schedules) that include MLK Day 2026."
    )

    # Standard per-vehicle fee
    vehicle_fee_node = evaluator.add_parallel(
        id="standard_per_vehicle_fee",
        desc="Standard per-vehicle entrance fee for Yellowstone.",
        parent=fees_node,
        critical=True
    )
    leaf_vehicle_fee_stated = evaluator.add_leaf(
        id="per_vehicle_fee_stated",
        desc="States the standard per-vehicle entrance fee amount for Yellowstone.",
        parent=vehicle_fee_node,
        critical=True
    )
    fee_amount_text = fees_info.per_vehicle_fee_amount_text or ""
    claim_vehicle_fee_stated = f"The answer states the standard per-vehicle entrance fee for Yellowstone as '{fee_amount_text}'."
    await evaluator.verify(
        claim=claim_vehicle_fee_stated,
        node=leaf_vehicle_fee_stated,
        additional_instruction="Confirm the answer includes a specific amount (e.g., $35)."
    )

    leaf_vehicle_fee_ref = evaluator.add_leaf(
        id="per_vehicle_fee_official_reference_url",
        desc="Provides official NPS reference URL for Yellowstone entrance fees.",
        parent=vehicle_fee_node,
        critical=True
    )
    claim_vehicle_fee_ref = f"The standard per-vehicle entrance fee for Yellowstone National Park is {fee_amount_text}."
    await evaluator.verify(
        claim=claim_vehicle_fee_ref,
        node=leaf_vehicle_fee_ref,
        sources=fees_info.fee_amount_sources,
        additional_instruction="Use official Yellowstone/NPS fee pages showing the private vehicle fee."
    )

    # Pass validity days
    validity_node = evaluator.add_parallel(
        id="pass_validity_days",
        desc="Standard entrance pass validity period.",
        parent=fees_node,
        critical=True
    )
    leaf_validity_match = evaluator.add_leaf(
        id="validity_matches_constraint",
        desc="States the standard entrance pass is valid for 7 consecutive days.",
        parent=validity_node,
        critical=True
    )
    claim_validity_match = "The standard Yellowstone entrance pass is valid for 7 consecutive days."
    await evaluator.verify(
        claim=claim_validity_match,
        node=leaf_validity_match,
        additional_instruction="Verify the answer text states the 7 consecutive days validity."
    )

    leaf_validity_ref = evaluator.add_leaf(
        id="validity_official_reference_url",
        desc="Provides official NPS reference URL confirming the pass validity period.",
        parent=validity_node,
        critical=True
    )
    claim_validity_ref = "A standard Yellowstone entrance pass covers 7 consecutive days."
    await evaluator.verify(
        claim=claim_validity_ref,
        node=leaf_validity_ref,
        sources=fees_info.validity_sources,
        additional_instruction="Use official Yellowstone/NPS pages that state the 7-day validity."
    )


# Main evaluation entry point
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
        task_description="Evaluate completeness and correctness of the requested winter Yellowstone MLK weekend 2026 plan, including at least one official reference URL for each requested sub-answer.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Add ground-truth expectations for transparency
    evaluator.add_ground_truth({
        "expected_values": {
            "lodging_name": EXPECTED_LODGING_NAME,
            "budget_airline_bgr": EXPECTED_BUDGET_AIRLINE,
            "hub_city": EXPECTED_HUB_CITY,
            "closest_mt_airport": EXPECTED_CLOSEST_MT_AIRPORT,
            "fee_free_day_date": EXPECTED_FEE_FREE_DATE,
            "fee_free_day_weekday": EXPECTED_FEE_FREE_WEEKDAY,
            "delay_threshold_domestic": EXPECTED_DELAY_THRESHOLD_DESC,
            "pass_validity": EXPECTED_PASS_VALIDITY_DESC
        }
    })

    # Build verification subtrees
    await verify_winter_lodging(evaluator, root)
    await verify_flight_routing(evaluator, root)
    await verify_passenger_rights(evaluator, root)
    await verify_entrance_fees(evaluator, root)

    # Return structured summary
    return evaluator.get_summary()