import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travel_nashville_southampton_aug2026"
TASK_DESCRIPTION = (
    "A traveler needs to journey from Nashville, Tennessee to Southampton, England to board the Celebrity Apex "
    "cruise departing on August 29, 2026 at 4:00 PM for a 14-night Norwegian Fjords voyage. They plan to fly "
    "JetBlue from Nashville and need to arrange connecting flights to reach Southampton. They will check two bags, "
    "each weighing 45 lbs with dimensions of 28 inches x 20 inches x 12 inches. Identify a viable flight itinerary "
    "that satisfies the following requirements: (1) The first flight segment must be a JetBlue flight departing from "
    "Nashville (BNA), (2) All connecting flights must provide adequate layover time (60-90 minutes for domestic "
    "connections, 90+ minutes for international connections), (3) The checked baggage must comply with international "
    "flight restrictions (maximum 50 lbs per bag and 62 inches total dimensions), (4) The arrival in Southampton must "
    "allow at least 2-3 hours before the 4:00 PM cruise departure, and (5) The booking must be completed with "
    "sufficient time to pay the full cruise balance at least 45 days before departure (by July 15, 2026). Provide "
    "the complete flight routing with airline(s), departure/arrival cities, and confirmation that all timing and "
    "baggage requirements are met."
)

CRUISE_DEPARTURE_LOCAL_DATE = "2026-08-29"
CRUISE_DEPARTURE_LOCAL_TIME = "16:00"  # 4:00 PM local time in Southampton
CRUISE_DEPARTURE_LOCAL_CITY = "Southampton"
CRUISE_DEPARTURE_LOCAL_PORT = "Celebrity Apex"  # For context
PAYMENT_DEADLINE_DATE = "2026-07-15"

# Given baggage from the task
BAG_WEIGHT_LBS = 45
BAG_DIMENSIONS_IN = (28, 20, 12)  # L x W x H
INTERNATIONAL_MAX_WEIGHT_LBS = 50
INTERNATIONAL_MAX_LINEAR_IN = 62


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class ItinerarySegment(BaseModel):
    airline: Optional[str] = None
    flight_number: Optional[str] = None
    depart_airport_code: Optional[str] = None
    depart_city: Optional[str] = None
    depart_datetime_local: Optional[str] = None  # e.g., "2026-08-28 07:15 AM CDT" or similar
    arrival_airport_code: Optional[str] = None
    arrival_city: Optional[str] = None
    arrival_datetime_local: Optional[str] = None  # e.g., "2026-08-28 10:15 AM EDT"
    source_urls: List[str] = Field(default_factory=list)  # URLs specifically supporting this segment


class ConnectionInfo(BaseModel):
    # Connection between segment i and i+1
    from_airport_code: Optional[str] = None
    to_airport_code: Optional[str] = None
    layover_minutes: Optional[int] = None  # Prefer integer minutes if answer provides it
    connection_type: Optional[str] = None  # "domestic" or "international" (case-insensitive)


class TravelPlanExtraction(BaseModel):
    segments: List[ItinerarySegment] = Field(default_factory=list)
    connections: List[ConnectionInfo] = Field(default_factory=list)
    final_arrival_airport_code: Optional[str] = None      # Should be "SOU" if truly reaching Southampton by flight
    final_arrival_datetime_local: Optional[str] = None    # Local time at final arrival airport
    supporting_urls: List[str] = Field(default_factory=list)  # Any general URLs cited in the answer
    payment_full_balance_date: Optional[str] = None       # Stated date by which full cruise balance will be paid
    notes: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_plan() -> str:
    return """
    Extract the proposed flight itinerary and supporting details from the answer. Return a structured JSON for:

    1) segments: List all flight segments in order as they appear in the proposed routing. For each segment include:
       - airline (e.g., "JetBlue", "British Airways", "KLM", etc.)
       - flight_number (e.g., "B6 1234", "BA 115")
       - depart_airport_code (IATA, e.g., "BNA", "JFK")
       - depart_city (e.g., "Nashville, TN")
       - depart_datetime_local (as written in the answer, local to the departure airport)
       - arrival_airport_code (IATA)
       - arrival_city
       - arrival_datetime_local (as written in the answer, local to the arrival airport)
       - source_urls (all URLs cited that directly support this segment, e.g., airline, Google Flights, OTA, etc.)

    2) connections: For each connection (i.e., between segment i and i+1), include:
       - from_airport_code (the arrival airport code of segment i)
       - to_airport_code (the departure airport code of segment i+1)
       - layover_minutes (integer minutes if the answer provides or you can reasonably infer; else null)
       - connection_type (domestic if both airports are in the same country; otherwise international; if unsure, guess based on context)

    3) final_arrival_airport_code: The airport code of the final flight segment's arrival (e.g., "SOU" for Southampton).
    4) final_arrival_datetime_local: The final arrival local date and time at that airport, as written in the answer.
    5) supporting_urls: Any additional or global URLs the answer cites for the itinerary (avoid duplicates already in segment.source_urls).
    6) payment_full_balance_date: The specific date by which the traveler will pay the full cruise balance, if stated (e.g., "2026-07-01" or "by July 15, 2026"); return null if not explicitly stated.

    IMPORTANT:
    - Extract EXACTLY what the answer states; do not invent data.
    - Include as many URL sources as the answer provides; valid forms include airline booking pages, Google Flights, OTA pages, airport schedules, etc.
    - The layover_minutes should be an integer number of minutes if at all possible (e.g., 85). If not available, set to null.
    - Use IATA codes as provided. If city names are present without codes, also try to infer IATA code only if the answer explicitly mentions it somewhere.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_segment(plan: TravelPlanExtraction) -> Optional[ItinerarySegment]:
    return plan.segments[0] if plan.segments else None


def _last_segment(plan: TravelPlanExtraction) -> Optional[ItinerarySegment]:
    return plan.segments[-1] if plan.segments else None


def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _segment_urls_or_global(seg: Optional[ItinerarySegment], plan: TravelPlanExtraction) -> List[str]:
    seg_urls = _non_empty_urls(seg.source_urls if seg else [])
    if seg_urls:
        return seg_urls
    return _non_empty_urls(plan.supporting_urls)


def _is_sou_destination(plan: TravelPlanExtraction) -> bool:
    # Prefer explicit field; otherwise fallback to last segment arrival code
    if (plan.final_arrival_airport_code or "").upper() == "SOU":
        return True
    last = _last_segment(plan)
    if last and (last.arrival_airport_code or "").upper() == "SOU":
        return True
    return False


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_flight_route_validity(evaluator: Evaluator, parent_node, plan: TravelPlanExtraction) -> None:
    """
    Verifies:
      - At least one segment exists
      - First segment is JetBlue out of BNA
      - Sources provided for flight(s)
      - Final destination is Southampton (SOU) by air
      - Key segments are supported by cited URLs
    """
    node = evaluator.add_parallel(
        id="Flight_Route_Validity",
        desc="The proposed flight route uses only airlines and routes that actually exist and operate on the required dates; first leg must be JetBlue from BNA and final arrival must be SOU (Southampton).",
        parent=parent_node,
        critical=True
    )

    # 1) Existence of itinerary segments
    exists = evaluator.add_custom_node(
        result=len(plan.segments) >= 1,
        id="route_segments_provided",
        desc="At least one flight segment is provided in the itinerary.",
        parent=node,
        critical=True
    )

    # 2) First segment JetBlue from BNA (simple-verify from the answer text)
    first = _first_segment(plan)
    first_desc = (f"airline '{first.airline}', flight '{first.flight_number}', "
                  f"departure '{first.depart_airport_code}' to '{first.arrival_airport_code}'") if first else "N/A"
    first_leg_leaf = evaluator.add_leaf(
        id="first_leg_jetblue_from_bna",
        desc="The first flight segment is operated by JetBlue and departs from Nashville BNA.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first flight segment is a JetBlue (JetBlue Airways / B6) flight departing from BNA (Nashville). The extracted first segment is: {first_desc}.",
        node=first_leg_leaf,
        additional_instruction="Allow 'JetBlue', 'JetBlue Airways', or 'B6' as equivalent names. Minor formatting or spacing differences are acceptable."
    )

    # 3) At least one source URL is provided for flights
    some_urls = _non_empty_urls(plan.supporting_urls)
    if not some_urls and plan.segments:
        for seg in plan.segments:
            some_urls.extend(_non_empty_urls(seg.source_urls))
    sources_present = evaluator.add_custom_node(
        result=len(some_urls) > 0,
        id="flight_sources_present",
        desc="At least one source URL supporting the flight itinerary is provided.",
        parent=node,
        critical=True
    )

    # 4) Final destination is Southampton (SOU)
    final_is_sou = evaluator.add_custom_node(
        result=_is_sou_destination(plan),
        id="final_destination_is_SOU",
        desc="The final arrival airport of the flight itinerary is Southampton (SOU).",
        parent=node,
        critical=True
    )

    # 5) Support key segments by cited URLs (first leg and final leg)
    # 5a) First segment supported by URL(s)
    first_seg_url_leaf = evaluator.add_leaf(
        id="first_segment_supported_by_sources",
        desc="The first JetBlue segment (BNA -> ...) is supported by cited source URL(s).",
        parent=node,
        critical=True
    )
    first_urls = _segment_urls_or_global(first, plan)
    first_seg_claim = "A JetBlue-operated flight from BNA on the specified date exists as shown by the cited source."
    if first and first.arrival_airport_code:
        first_seg_claim = f"A JetBlue-operated flight from BNA to {first.arrival_airport_code} on the specified date exists as shown by the cited source."
    await evaluator.verify(
        claim=first_seg_claim,
        node=first_seg_url_leaf,
        sources=first_urls,
        additional_instruction="Use the airline booking page, Google Flights, or OTA links to confirm the flight exists for around the proposed date. Allow codeshare equivalence where reasonable."
    )

    # 5b) Final segment to SOU supported by URL(s)
    last = _last_segment(plan)
    last_seg_url_leaf = evaluator.add_leaf(
        id="final_segment_supported_by_sources",
        desc="The final flight segment arriving at SOU is supported by cited source URL(s).",
        parent=node,
        critical=True
    )
    last_urls = _segment_urls_or_global(last, plan)
    final_claim = "A flight arriving at SOU (Southampton) for the itinerary exists as shown by the cited source."
    if last and last.depart_airport_code and last.airline:
        final_claim = (
            f"A flight operated by {last.airline} from {last.depart_airport_code} to SOU (Southampton) around the specified date exists as shown by the cited source."
        )
    await evaluator.verify(
        claim=final_claim,
        node=last_seg_url_leaf,
        sources=last_urls,
        additional_instruction="Confirm at least one cited URL clearly shows a flight service into SOU for the relevant date window."
    )


async def verify_baggage_compliance(evaluator: Evaluator, parent_node) -> None:
    """
    Verifies baggage meets international limits:
      - Weight <= 50 lbs per bag
      - Linear dimensions (L+W+H) <= 62 inches per bag
    Uses task-provided values (2 bags, each 45 lbs, 28x20x12 in).
    """
    node = evaluator.add_parallel(
        id="Baggage_Compliance",
        desc="All checked baggage meets the 50 lbs and 62 linear inches international limits.",
        parent=parent_node,
        critical=True
    )

    total_linear_in = sum(BAG_DIMENSIONS_IN)
    weight_ok = BAG_WEIGHT_LBS <= INTERNATIONAL_MAX_WEIGHT_LBS
    dims_ok = total_linear_in <= INTERNATIONAL_MAX_LINEAR_IN

    weight_leaf = evaluator.add_custom_node(
        result=weight_ok,
        id="bag_weight_within_limit",
        desc=f"Each bag weighs {BAG_WEIGHT_LBS} lbs (<= {INTERNATIONAL_MAX_WEIGHT_LBS} lbs limit).",
        parent=node,
        critical=True
    )

    dims_leaf = evaluator.add_custom_node(
        result=dims_ok,
        id="bag_dimensions_within_limit",
        desc=f"Each bag's linear size is {total_linear_in} inches (<= {INTERNATIONAL_MAX_LINEAR_IN} inches limit).",
        parent=node,
        critical=True
    )


async def verify_connection_time_adequacy(evaluator: Evaluator, parent_node, plan: TravelPlanExtraction) -> None:
    """
    For each connection:
      - Domestic: layover must be between 60 and 90 minutes (inclusive)
      - International: layover must be >= 90 minutes
    If layover_minutes not provided, fallback to LLM check on textual claim.
    """
    node = evaluator.add_parallel(
        id="Connection_Time_Adequacy",
        desc="Each connection provides adequate layover time (60-90 mins domestic; 90+ mins international).",
        parent=parent_node,
        critical=True
    )

    # Ensure there is at least one connection if multiple segments exist
    has_connections = evaluator.add_custom_node(
        result=(len(plan.segments) <= 1) or (len(plan.connections) >= max(0, len(plan.segments) - 1)),
        id="connections_listed",
        desc="Connections are listed between flight segments (if applicable).",
        parent=node,
        critical=True
    )

    # Create a leaf for each connection adequacy
    for idx, conn in enumerate(plan.connections):
        # Try numeric rule if minutes provided; else ask LLM to judge based on the extracted info
        ct = (conn.connection_type or "").strip().lower()
        minutes = conn.layover_minutes

        # Numeric evaluation path when available
        if isinstance(minutes, int):
            adequate = False
            if "domestic" in ct:
                adequate = 60 <= minutes <= 90
            elif "international" in ct:
                adequate = minutes >= 90

            evaluator.add_custom_node(
                result=adequate,
                id=f"conn_{idx}_adequate_numeric",
                desc=f"Connection {idx + 1} ({conn.from_airport_code} -> {conn.to_airport_code}) layover {minutes} min satisfies {('domestic 60–90' if 'domestic' in ct else 'international 90+') if ct else 'stated'} requirement.",
                parent=node,
                critical=True
            )
        else:
            # Fallback LLM verification when minutes not explicit
            leaf = evaluator.add_leaf(
                id=f"conn_{idx}_adequate_textual",
                desc=f"Connection {idx + 1} ({conn.from_airport_code} -> {conn.to_airport_code}) provides adequate layover time per requirement.",
                parent=node,
                critical=True
            )
            layover_str = "not specified" if conn.layover_minutes is None else str(conn.layover_minutes)
            rule_text = "between 60 and 90 minutes (inclusive) for domestic connections; at least 90 minutes for international connections"
            await evaluator.verify(
                claim=(
                    f"The layover between flights at {conn.from_airport_code} for the next departure from {conn.to_airport_code} "
                    f"is adequate according to this rule: {rule_text}. The extracted layover is {layover_str} minutes and the connection type is '{conn.connection_type}'."
                ),
                node=leaf,
                additional_instruction="Judge adequacy logically from the provided layover and type in the answer text. If the layover appears to be in the correct range per the rule, mark as correct."
            )


async def verify_cruise_boarding_timing(evaluator: Evaluator, parent_node, plan: TravelPlanExtraction) -> None:
    """
    Verifies that final arrival in Southampton allows at least 2 hours before the 4:00 PM (16:00) local cruise departure
    on August 29, 2026. Arrival on an earlier date also satisfies the constraint.
    """
    node = evaluator.add_parallel(
        id="Cruise_Boarding_Timing",
        desc="Arrival in Southampton allows at least 2–3 hours before the 4:00 PM cruise departure on Aug 29, 2026.",
        parent=parent_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="arrival_buffer_sufficient",
        desc="Final arrival timing provides at least a 2-hour buffer before 4:00 PM local cruise departure on 2026-08-29, or arrives on an earlier date.",
        parent=node,
        critical=True
    )

    arrival_text = plan.final_arrival_datetime_local or (_last_segment(plan).arrival_datetime_local if _last_segment(plan) else "unspecified")
    await evaluator.verify(
        claim=(
            f"The final flight arrives in Southampton (SOU) no later than 14:00 local time on {CRUISE_DEPARTURE_LOCAL_DATE} "
            f"(or on an earlier date), providing at least a 2-hour buffer before the 4:00 PM (16:00) cruise departure. "
            f"The extracted final arrival time is: {arrival_text}."
        ),
        node=leaf,
        additional_instruction="If the arrival happens on August 29, 2026, ensure it is at or before 2:00 PM local time. Any arrival earlier than that date also satisfies the requirement."
    )


async def verify_payment_deadline(evaluator: Evaluator, parent_node, plan: TravelPlanExtraction) -> None:
    """
    Verifies the booking/payment timeline ensures the full cruise balance is paid by July 15, 2026 (>=45 days before departure).
    """
    node = evaluator.add_parallel(
        id="Payment_Deadline_Met",
        desc="The booking/payment plan ensures the full cruise balance is paid by July 15, 2026.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: A clear statement (or date) is present to pay by or before the deadline
    presence_leaf = evaluator.add_leaf(
        id="payment_deadline_statement_present",
        desc="Answer explicitly states a plan/date to pay the full balance by the required deadline.",
        parent=node,
        critical=True
    )
    stated_date = plan.payment_full_balance_date or "unspecified"
    await evaluator.verify(
        claim=(
            f"The answer explicitly ensures that the full cruise balance will be paid by {PAYMENT_DEADLINE_DATE} "
            f"(or earlier). The extracted stated payment date or plan is: {stated_date}."
        ),
        node=presence_leaf,
        additional_instruction="If the answer states paying 'by July 15, 2026', 'before July 15, 2026', or provides a specific earlier date, mark as correct. If absent, mark as incorrect."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Nashville (BNA) to Southampton (SOU) travel plan with constraints.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: parallel aggregation across main criteria
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

    # Record ground-truth constraint info for transparency
    evaluator.add_ground_truth({
        "cruise_ship": CRUISE_DEPARTURE_LOCAL_PORT,
        "departure_city": CRUISE_DEPARTURE_LOCAL_CITY,
        "departure_date_local": CRUISE_DEPARTURE_LOCAL_DATE,
        "departure_time_local": CRUISE_DEPARTURE_LOCAL_TIME,
        "payment_deadline_date": PAYMENT_DEADLINE_DATE,
        "baggage_limits": {
            "max_weight_lbs_per_bag": INTERNATIONAL_MAX_WEIGHT_LBS,
            "max_linear_inches_per_bag": INTERNATIONAL_MAX_LINEAR_IN
        },
        "task_baggage": {
            "bags": 2,
            "each_weight_lbs": BAG_WEIGHT_LBS,
            "each_dimensions_in": list(BAG_DIMENSIONS_IN),
            "each_linear_total_in": sum(BAG_DIMENSIONS_IN)
        }
    }, gt_type="constraints")

    # Extract structured travel plan info from the answer
    plan: TravelPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_travel_plan(),
        template_class=TravelPlanExtraction,
        extraction_name="travel_plan"
    )

    # Build verification tree according to rubric (all 5 criteria under the root)
    await verify_flight_route_validity(evaluator, root, plan)
    await verify_baggage_compliance(evaluator, root)
    await verify_connection_time_adequacy(evaluator, root, plan)
    await verify_cruise_boarding_timing(evaluator, root, plan)
    await verify_payment_deadline(evaluator, root, plan)

    # Return the full evaluation summary (score + tree + extractions)
    return evaluator.get_summary()