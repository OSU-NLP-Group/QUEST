import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "clt_auh_roundtrip_parking_2026"
TASK_DESCRIPTION = (
    "A Charlotte resident needs to travel to Abu Dhabi, UAE for business. They plan to depart Charlotte on "
    "Wednesday, May 7, 2026, taking advantage of the new Etihad Airways nonstop service from Charlotte Douglas "
    "International Airport to Abu Dhabi. They will return to Charlotte on Wednesday, May 21, 2026, also using the "
    "Etihad Airways nonstop service. The traveler will drive their personal vehicle to Charlotte Douglas International "
    "Airport and needs to park for the entire duration of the trip.\n\n"
    "Provide a complete travel plan that includes:\n"
    "1. Outbound flight details (airline, flight day of week, departure airport, destination, aircraft type)\n"
    "2. Return flight details (airline, flight day of week, departure airport, destination, aircraft type)\n"
    "3. The most cost-effective parking option at Charlotte Douglas Airport for the 14-day trip duration (May 7 to "
    "May 21, 2026)\n"
    "4. Total parking cost for the selected option\n"
    "5. Reference URLs for Etihad Airways flight information and Charlotte airport parking rates"
)

TRIP_DAYS = 14  # May 7 -> May 21, 2026


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FlightInfo(BaseModel):
    airline: Optional[str] = None
    day_of_week: Optional[str] = None
    departure_airport_name: Optional[str] = None
    departure_airport_code: Optional[str] = None
    destination_airport_name: Optional[str] = None
    destination_airport_code: Optional[str] = None
    aircraft_type: Optional[str] = None
    nonstop: Optional[str] = None  # Accept 'nonstop', 'non-stop', 'direct', 'yes', etc.


class ParkingInfo(BaseModel):
    option_name: Optional[str] = None
    daily_rate: Optional[str] = None  # e.g., "$12/day", "$10 per day"
    duration_days: Optional[str] = None  # e.g., "14 days", "14"
    total_cost: Optional[str] = None  # e.g., "$168", "USD 168"


class TravelPlanExtraction(BaseModel):
    outbound: Optional[FlightInfo] = None
    return_flight: Optional[FlightInfo] = None
    parking: Optional[ParkingInfo] = None
    etihad_reference_urls: List[str] = Field(default_factory=list)
    clt_parking_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_plan() -> str:
    return """
    Extract the structured travel plan fields exactly as they appear in the provided answer.

    Required structure:
    - outbound:
        - airline
        - day_of_week  (e.g., "Wednesday" or "Wed")
        - departure_airport_name  (e.g., "Charlotte Douglas International Airport")
        - departure_airport_code  (e.g., "CLT")
        - destination_airport_name (e.g., "Abu Dhabi International Airport")
        - destination_airport_code (e.g., "AUH")
        - aircraft_type (e.g., "Boeing 787-9 Dreamliner" or "787-9")
        - nonstop (the answer's wording indicating a nonstop/direct flight; extract the exact phrase if present, else null)

    - return_flight:
        - airline
        - day_of_week
        - departure_airport_name
        - departure_airport_code
        - destination_airport_name
        - destination_airport_code
        - aircraft_type
        - nonstop

    - parking:
        - option_name (exact name of the selected CLT on-airport parking product)
        - daily_rate (as stated, including symbols or units if present, e.g., "$12/day")
        - duration_days (as stated in the answer, e.g., "14 days" or "14")
        - total_cost (as a single amount string, e.g., "$168")

    - etihad_reference_urls: a list of all URLs provided in the answer that support Etihad CLT↔AUH service info
    - clt_parking_reference_urls: a list of all URLs provided in the answer that support official CLT parking rates

    Rules:
    - Extract only what is explicitly stated in the answer. Do not invent or infer values.
    - For URLs, only include actual URLs that appear in the answer (plain or markdown links). If none are provided, return an empty list.
    - If a field is missing in the answer, set it to null (or an empty list for the URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
_money_regex = re.compile(r'(?i)\$?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)')


def parse_first_money_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = _money_regex.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except Exception:
        return None


def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'(\d+)', text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def normalize_yeslike(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    t = text.strip().lower()
    yes_tokens = {"yes", "y", "true", "nonstop", "non-stop", "direct"}
    no_tokens = {"no", "n", "false", "stop", "connection", "connected"}
    if t in yes_tokens:
        return True
    if t in no_tokens:
        return False
    # heuristic for phrases
    if "nonstop" in t or "non-stop" in t or "direct" in t:
        return True
    return None


def looks_like_wednesday(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return t in {"wednesday", "wed"}


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_reference_urls(evaluator: Evaluator, root_node, ex: TravelPlanExtraction) -> None:
    refs_node = evaluator.add_parallel(
        id="reference_urls",
        desc="Reference URLs are provided for required sources",
        parent=root_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(ex.etihad_reference_urls) > 0),
        id="etihad_reference_url",
        desc="At least one reference URL is provided for Etihad Airways CLT–AUH flight information",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(ex.clt_parking_reference_urls) > 0),
        id="clt_parking_reference_url",
        desc="At least one reference URL is provided for Charlotte Douglas Airport parking rates",
        parent=refs_node,
        critical=True
    )


async def verify_outbound_flight(evaluator: Evaluator, root_node, ex: TravelPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="outbound_flight",
        desc="Outbound flight details for travel from Charlotte to Abu Dhabi",
        parent=root_node,
        critical=True
    )

    outbound = ex.outbound or FlightInfo()
    etihad_sources = ex.etihad_reference_urls

    # 1) Airline
    n_airline = evaluator.add_leaf(
        id="outbound_airline",
        desc="Outbound airline is Etihad Airways",
        parent=node,
        critical=True
    )
    claim_airline = (
        "The answer specifies Etihad Airways as the outbound airline from Charlotte (CLT) to Abu Dhabi (AUH), "
        "and the provided source(s) confirm that Etihad operates the CLT–AUH route."
    )

    # 2) Departure airport
    n_dep = evaluator.add_leaf(
        id="outbound_departure_airport",
        desc="Outbound departure airport is Charlotte Douglas International Airport (CLT)",
        parent=node,
        critical=True
    )
    claim_dep = (
        "The answer identifies the outbound departure airport as Charlotte Douglas International Airport (CLT), "
        "and the source(s) confirm the route originates in Charlotte (CLT)."
    )

    # 3) Destination airport
    n_dest = evaluator.add_leaf(
        id="outbound_destination_airport",
        desc="Outbound destination airport is Abu Dhabi International Airport (AUH)",
        parent=node,
        critical=True
    )
    claim_dest = (
        "The answer identifies the outbound destination airport as Abu Dhabi International Airport (AUH), "
        "and the source(s) confirm the route arrives at Abu Dhabi (AUH)."
    )

    # 4) Day of week = Wednesday (from the answer)
    n_dow = evaluator.add_leaf(
        id="outbound_day_of_week",
        desc="Outbound flight day of week is Wednesday",
        parent=node,
        critical=True
    )
    # Use simple verify; accept 'Wednesday' or 'Wed'
    claim_dow = (
        f"The outbound flight day of week in the answer corresponds to Wednesday "
        f"(accept 'Wednesday' or 'Wed'). The extracted value is '{outbound.day_of_week or ''}'."
    )

    # 5) Aircraft type 787-9
    n_aircraft = evaluator.add_leaf(
        id="outbound_aircraft_type",
        desc="Outbound aircraft type is Boeing 787-9 Dreamliner",
        parent=node,
        critical=True
    )
    claim_aircraft = (
        "The answer lists the outbound aircraft type as Boeing 787-9 Dreamliner (or '787-9'), and the provided "
        "source(s) confirm that the Etihad CLT–AUH service uses a Boeing 787-9 aircraft."
    )

    # 6) Nonstop
    n_nonstop = evaluator.add_leaf(
        id="outbound_nonstop",
        desc="Outbound itinerary is nonstop (no connections)",
        parent=node,
        critical=True
    )
    claim_nonstop = (
        "The answer states the outbound itinerary is nonstop (no connections), and the provided source(s) confirm that "
        "the CLT–AUH service is nonstop (accept 'nonstop', 'non-stop', or 'direct')."
    )

    # Batch verify (mix of URL-backed and simple checks)
    await evaluator.batch_verify([
        (claim_airline, etihad_sources, n_airline, "Treat 'Etihad' and 'Etihad Airways' as equivalent."),
        (claim_dep, etihad_sources, n_dep, "Confirm that CLT is Charlotte Douglas International Airport."),
        (claim_dest, etihad_sources, n_dest, "Confirm that AUH is Abu Dhabi International Airport."),
        (claim_dow, None, n_dow, "Return Correct only if the extracted value indicates Wednesday (or 'Wed')."),
        (claim_aircraft, etihad_sources, n_aircraft, "Accept synonyms: 'B787-9', '787-9', 'Boeing 787-9 Dreamliner'."),
        (claim_nonstop, etihad_sources, n_nonstop, "Accept 'nonstop', 'non-stop', or 'direct' as equivalent.")
    ])


async def verify_return_flight(evaluator: Evaluator, root_node, ex: TravelPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="return_flight",
        desc="Return flight details for travel from Abu Dhabi to Charlotte",
        parent=root_node,
        critical=True
    )

    rf = ex.return_flight or FlightInfo()
    etihad_sources = ex.etihad_reference_urls

    # 1) Airline
    n_airline = evaluator.add_leaf(
        id="return_airline",
        desc="Return airline is Etihad Airways",
        parent=node,
        critical=True
    )
    claim_airline = (
        "The answer specifies Etihad Airways as the return airline from Abu Dhabi (AUH) to Charlotte (CLT), "
        "and the provided source(s) confirm that Etihad operates the AUH–CLT route."
    )

    # 2) Departure airport (AUH)
    n_dep = evaluator.add_leaf(
        id="return_departure_airport",
        desc="Return departure airport is Abu Dhabi International Airport (AUH)",
        parent=node,
        critical=True
    )
    claim_dep = (
        "The answer identifies the return departure airport as Abu Dhabi International Airport (AUH), "
        "and the source(s) confirm the route departs from Abu Dhabi (AUH)."
    )

    # 3) Destination airport (CLT)
    n_dest = evaluator.add_leaf(
        id="return_destination_airport",
        desc="Return destination airport is Charlotte Douglas International Airport (CLT)",
        parent=node,
        critical=True
    )
    claim_dest = (
        "The answer identifies the return destination airport as Charlotte Douglas International Airport (CLT), "
        "and the source(s) confirm the route arrives in Charlotte (CLT)."
    )

    # 4) Day of week = Wednesday
    n_dow = evaluator.add_leaf(
        id="return_day_of_week",
        desc="Return flight day of week is Wednesday",
        parent=node,
        critical=True
    )
    claim_dow = (
        f"The return flight day of week in the answer corresponds to Wednesday (accept 'Wednesday' or 'Wed'). "
        f"The extracted value is '{rf.day_of_week or ''}'."
    )

    # 5) Aircraft type 787-9
    n_aircraft = evaluator.add_leaf(
        id="return_aircraft_type",
        desc="Return aircraft type is Boeing 787-9 Dreamliner",
        parent=node,
        critical=True
    )
    claim_aircraft = (
        "The answer lists the return aircraft type as Boeing 787-9 Dreamliner (or '787-9'), and the provided "
        "source(s) confirm that the Etihad AUH–CLT service uses a Boeing 787-9 aircraft."
    )

    # 6) Nonstop
    n_nonstop = evaluator.add_leaf(
        id="return_nonstop",
        desc="Return itinerary is nonstop (no connections)",
        parent=node,
        critical=True
    )
    claim_nonstop = (
        "The answer states the return itinerary is nonstop (no connections), and the provided source(s) confirm that "
        "the AUH–CLT service is nonstop (accept 'nonstop', 'non-stop', or 'direct')."
    )

    await evaluator.batch_verify([
        (claim_airline, etihad_sources, n_airline, "Treat 'Etihad' and 'Etihad Airways' as equivalent."),
        (claim_dep, etihad_sources, n_dep, "Confirm that AUH is Abu Dhabi International Airport."),
        (claim_dest, etihad_sources, n_dest, "Confirm that CLT is Charlotte Douglas International Airport."),
        (claim_dow, None, n_dow, "Return Correct only if the extracted value indicates Wednesday (or 'Wed')."),
        (claim_aircraft, etihad_sources, n_aircraft, "Accept synonyms: 'B787-9', '787-9', 'Boeing 787-9 Dreamliner'."),
        (claim_nonstop, etihad_sources, n_nonstop, "Accept 'nonstop', 'non-stop', or 'direct' as equivalent.")
    ])


async def verify_parking_plan(evaluator: Evaluator, root_node, ex: TravelPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="parking",
        desc="Parking plan at Charlotte Douglas Airport for the full trip duration",
        parent=root_node,
        critical=True
    )

    pk = ex.parking or ParkingInfo()
    clt_sources = ex.clt_parking_reference_urls

    # 1) Parking duration is 14 days
    duration_parsed = parse_first_int(pk.duration_days)
    evaluator.add_custom_node(
        result=(duration_parsed == TRIP_DAYS),
        id="parking_duration",
        desc="Parking duration is 14 days (May 7 to May 21, 2026)",
        parent=node,
        critical=True
    )

    # 2) Specific named option selected
    evaluator.add_custom_node(
        result=(pk.option_name is not None and pk.option_name.strip() != ""),
        id="parking_option_named",
        desc="A single specific CLT parking facility/option is selected and named",
        parent=node,
        critical=True
    )

    # 3) Option is most cost-effective (lowest daily rate among on-airport options)
    n_cheapest = evaluator.add_leaf(
        id="parking_option_is_most_cost_effective",
        desc="Selected parking option is the lowest daily-rate option among the provided CLT parking rates",
        parent=node,
        critical=True
    )
    claim_cheapest = (
        f"The selected on-airport parking option '{pk.option_name or ''}' at CLT is the lowest daily-rate option "
        f"on the official Charlotte airport parking rates page, making it the most cost-effective for a 14-day stay. "
        f"If multiple options share the same lowest daily rate, consider this claim correct if the selected option is "
        f"one of those tied lowest-rate options. Ignore private/off-airport lots."
    )

    # 4) Daily rate provided
    rate_num = parse_first_money_value(pk.daily_rate)
    evaluator.add_custom_node(
        result=(pk.daily_rate is not None and pk.daily_rate.strip() != ""),
        id="parking_daily_rate_provided",
        desc="Daily parking rate for the selected facility is stated",
        parent=node,
        critical=True
    )

    # 5) Daily rate correct vs CLT page
    n_rate_correct = evaluator.add_leaf(
        id="parking_daily_rate_correct",
        desc="Stated daily parking rate matches the provided March 1, 2026 CLT parking rate for the selected facility",
        parent=node,
        critical=True
    )
    claim_rate_correct = (
        f"The daily parking rate for the selected CLT facility '{pk.option_name or ''}' is stated as "
        f"'{pk.daily_rate or ''}' in the answer, and the official CLT parking rates page shows the same daily rate "
        f"(as of March 1, 2026). Accept reasonable formatting like '$12 per day' vs '$12/day'."
    )

    # Execute the two URL-backed verifications
    await evaluator.batch_verify([
        (claim_cheapest, clt_sources, n_cheapest,
         "Check official on-airport parking options only (e.g., Long Term, Daily Deck, Express, Valet). Identify the lowest posted daily rate."),
        (claim_rate_correct, clt_sources, n_rate_correct,
         "Verify exact daily dollar rate for the named facility. Minor formatting differences are acceptable.")
    ])

    # Record parsed helpers
    evaluator.add_custom_info(
        info={
            "parsed_duration_days": duration_parsed,
            "parsed_daily_rate": rate_num
        },
        info_type="parsed_parking_info",
        info_name="parsed_parking_info"
    )


async def verify_parking_cost(evaluator: Evaluator, root_node, ex: TravelPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="parking_cost",
        desc="Total parking cost is provided and correctly computed",
        parent=root_node,
        critical=True
    )

    pk = ex.parking or ParkingInfo()
    rate_num = parse_first_money_value(pk.daily_rate)
    total_num = parse_first_money_value(pk.total_cost)

    # 1) Total cost stated as numeric
    evaluator.add_custom_node(
        result=(total_num is not None),
        id="total_parking_cost_stated",
        desc="Total parking cost is stated as a numeric amount",
        parent=node,
        critical=True
    )

    # 2) Total cost equals (daily rate × 14)
    correct_calc = False
    if (rate_num is not None) and (total_num is not None):
        expected = rate_num * TRIP_DAYS
        # Allow small rounding tolerance (cents)
        correct_calc = abs(total_num - expected) <= 0.02

    evaluator.add_custom_node(
        result=correct_calc,
        id="total_parking_cost_correct",
        desc="Total parking cost equals (stated daily rate × 14 days)",
        parent=node,
        critical=True
    )

    evaluator.add_custom_info(
        info={
            "parsed_total_cost": total_num,
            "computed_expected_total": (rate_num * TRIP_DAYS) if (rate_num is not None) else None
        },
        info_type="parking_cost_calc",
        info_name="parking_cost_calc"
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate a complete round-trip travel plan from CLT to AUH with parking and references.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is non-critical in framework; children groups marked critical.
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

    # Extract structured information from the answer
    extracted: TravelPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_travel_plan(),
        template_class=TravelPlanExtraction,
        extraction_name="travel_plan_extraction",
    )

    # Add some task constants for transparency
    evaluator.add_custom_info(
        info={"trip_days_expected": TRIP_DAYS, "outbound_target_weekday": "Wednesday", "return_target_weekday": "Wednesday"},
        info_type="task_expectations",
        info_name="task_expectations"
    )

    # Build and verify each rubric section
    await verify_reference_urls(evaluator, root, extracted)
    await verify_outbound_flight(evaluator, root, extracted)
    await verify_return_flight(evaluator, root, extracted)
    await verify_parking_plan(evaluator, root, extracted)
    await verify_parking_cost(evaluator, root, extracted)

    # Return standard summary
    return evaluator.get_summary()