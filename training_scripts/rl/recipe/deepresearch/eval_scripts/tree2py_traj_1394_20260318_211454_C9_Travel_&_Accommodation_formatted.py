import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "frontier_4city_lounge_shuttle"
TASK_DESCRIPTION = """
A corporate travel coordinator is planning a multi-city business trip departing from Atlanta, Georgia (ATL). Identify 4 different US cities that meet ALL of the following requirements:

1. The city must be accessible via direct Frontier Airlines flights from Atlanta (ATL)
2. The destination airport must offer either an American Express Centurion Lounge OR Priority Pass lounge access (including Priority Pass dining options)
3. At least one hotel within 5 miles of the destination airport must provide complimentary shuttle service to/from the airport
4. Each of the 4 cities must be located in a different US state
5. For each city, provide: (a) the airport code, (b) the specific lounge name or dining facility available, (c) the name of at least one hotel with shuttle service and its approximate distance from the airport

Ensure that all information is verifiable through publicly available sources and provide reference URLs for each claim.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CityItem(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    airport_code: Optional[str] = None

    # Direct Frontier flights from ATL references
    flight_sources: List[str] = Field(default_factory=list)

    # Lounge information
    lounge_name: Optional[str] = None
    lounge_type: Optional[str] = None  # e.g., "Centurion Lounge", "Priority Pass lounge", "Priority Pass dining"
    lounge_sources: List[str] = Field(default_factory=list)

    # Hotel information
    hotel_name: Optional[str] = None
    hotel_distance: Optional[str] = None  # keep as string for robustness (e.g., "3.2 miles", "~4 mi")
    hotel_sources: List[str] = Field(default_factory=list)


class ItineraryExtraction(BaseModel):
    cities: List[CityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_itinerary() -> str:
    return """
    Extract at most the first 4 US destination cities presented in the answer that are proposed for a multi‑city trip from Atlanta (ATL).
    For each city, return an object with the following fields (use null if missing, and an empty list [] when URLs are not provided):

    - city: the city name (e.g., "Orlando")
    - state: the two-letter US state code or the full state name (e.g., "FL" or "Florida")
    - airport_code: the destination airport IATA code (e.g., "MCO")
    - flight_sources: an array of URLs specifically intended to support that Frontier operates a direct (non‑stop) route from ATL to this airport
    - lounge_name: the specific name of the lounge or dining facility (e.g., "The Club", "Minute Suites", "Bambuza", "Centurion Lounge")
    - lounge_type: one of: "Centurion Lounge", "Priority Pass lounge", or "Priority Pass dining" (use the best match; for Escape Lounge – The Centurion Studio Partner, classify as "Centurion Lounge")
    - lounge_sources: an array of URLs that confirm the lounge/dining access at the destination airport (Priority Pass page, the lounge’s site, or AmEx lounge list)
    - hotel_name: the name of at least one hotel near the airport
    - hotel_distance: the approximate distance from the airport to that hotel (as stated in the answer), keep as a short string like "2.1 miles"
    - hotel_sources: an array of URLs that confirm BOTH (a) the hotel is within about 5 miles of the airport and (b) the hotel provides complimentary airport shuttle service

    Return a JSON object:
    {
      "cities": [ {CityItem}, {CityItem}, {CityItem}, {CityItem} ]
    }

    Rules:
    - Extract only what appears in the answer. Do not invent data.
    - If more than 4 cities are present, keep only the first 4.
    - Normalize URLs: include complete URLs with http/https when possible.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _first_k(items: List[CityItem], k: int) -> List[CityItem]:
    return items[:k] if items else []


def _pad_to_4(items: List[CityItem]) -> List[CityItem]:
    padded = list(items)
    while len(padded) < 4:
        padded.append(CityItem())
    return padded


def _distinct_count(vals: List[str]) -> int:
    return len({v for v in vals if v is not None and v != ""})


# --------------------------------------------------------------------------- #
# City verification subroutine                                                #
# --------------------------------------------------------------------------- #
async def verify_one_city(evaluator: Evaluator, parent_node, city_item: CityItem, idx: int) -> None:
    """
    Build the verification subtree for one city (City_1..City_4) and run URL-grounded checks.
    """
    city_label = f"City_{idx + 1}"
    pretty_city = city_item.city or f"City #{idx + 1}"
    dest_code = (city_item.airport_code or "").upper()

    # City root (parallel); non-critical at the itinerary level (partial credit allowed per city)
    city_node = evaluator.add_parallel(
        id=city_label,
        desc=[
            "First city in the itinerary meets all requirements",
            "Second city in the itinerary meets all requirements",
            "Third city in the itinerary meets all requirements",
            "Fourth city in the itinerary meets all requirements",
        ][idx],
        parent=parent_node,
        critical=False,
    )

    # Airport code existence (critical for city validity)
    evaluator.add_custom_node(
        result=bool(_norm(city_item.airport_code)),
        id=f"{city_label}_Airport_Code",
        desc=f"Airport code is provided for {city_label}",
        parent=city_node,
        critical=True,
    )

    # Flight availability group (critical)
    flight_group = evaluator.add_parallel(
        id=f"{city_label}_Flight_Availability",
        desc=f"Verify direct Frontier Airlines service from Atlanta to {city_label}",
        parent=city_node,
        critical=True,
    )

    # Route exists (URL-grounded)
    route_leaf = evaluator.add_leaf(
        id=f"{city_label}_Flight_Route_Exists",
        desc="Frontier Airlines operates direct flights from ATL to the destination airport",
        parent=flight_group,
        critical=True,
    )

    route_claim = (
        f"Frontier Airlines operates a direct (non-stop) route from Atlanta (ATL) to {dest_code}."
        if dest_code else
        "Frontier Airlines operates a direct (non-stop) route from Atlanta (ATL) to this destination airport."
    )
    # Existence of flight reference URL(s)
    evaluator.add_custom_node(
        result=bool(city_item.flight_sources),
        id=f"{city_label}_Flight_Reference",
        desc="Valid URL reference provided confirming Frontier route from Atlanta",
        parent=flight_group,
        critical=True,
    )

    # Lounge access group (critical)
    lounge_group = evaluator.add_parallel(
        id=f"{city_label}_Lounge_Access",
        desc=f"Verify lounge or dining access availability at {city_label} airport",
        parent=city_node,
        critical=True,
    )

    lounge_leaf = evaluator.add_leaf(
        id=f"{city_label}_Lounge_Type_Verification",
        desc="Airport offers American Express Centurion Lounge OR Priority Pass access (lounge or dining)",
        parent=lounge_group,
        critical=True,
    )

    # Lounge name provided
    evaluator.add_custom_node(
        result=bool(_norm(city_item.lounge_name)),
        id=f"{city_label}_Lounge_Name_Specified",
        desc="Specific lounge name or dining facility is provided",
        parent=lounge_group,
        critical=True,
    )
    # Lounge reference provided
    evaluator.add_custom_node(
        result=bool(city_item.lounge_sources),
        id=f"{city_label}_Lounge_Reference",
        desc="Valid URL reference provided confirming lounge/dining availability",
        parent=lounge_group,
        critical=True,
    )

    # Hotel shuttle group (critical)
    hotel_group = evaluator.add_parallel(
        id=f"{city_label}_Hotel_Shuttle",
        desc=f"Verify hotel with shuttle service availability at {city_label}",
        parent=city_node,
        critical=True,
    )

    # Hotel name provided
    evaluator.add_custom_node(
        result=bool(_norm(city_item.hotel_name)),
        id=f"{city_label}_Hotel_Name_Specified",
        desc="Name of at least one hotel is provided",
        parent=hotel_group,
        critical=True,
    )

    # Distance specified
    evaluator.add_custom_node(
        result=bool(_norm(city_item.hotel_distance)),
        id=f"{city_label}_Distance_Specified",
        desc="Approximate distance from airport is provided",
        parent=hotel_group,
        critical=True,
    )

    # Hotel within 5 miles (URL-grounded)
    distance_leaf = evaluator.add_leaf(
        id=f"{city_label}_Hotel_Within_Distance",
        desc="The hotel is located within 5 miles of the airport",
        parent=hotel_group,
        critical=True,
    )

    # Shuttle service confirmed (URL-grounded)
    shuttle_leaf = evaluator.add_leaf(
        id=f"{city_label}_Shuttle_Service_Confirmed",
        desc="The hotel provides complimentary shuttle service to/from airport",
        parent=hotel_group,
        critical=True,
    )

    # Hotel reference(s) provided
    evaluator.add_custom_node(
        result=bool(city_item.hotel_sources),
        id=f"{city_label}_Hotel_Reference",
        desc="Valid URL reference provided confirming hotel shuttle service",
        parent=hotel_group,
        critical=True,
    )

    # Batch URL-grounded verifications for this city
    claims_and_sources = []

    # Flight route existence
    claims_and_sources.append((
        route_claim,
        city_item.flight_sources,
        route_leaf,
        (
            "Verify that at least one of the provided webpages explicitly shows a non-stop route operated by Frontier Airlines "
            f"from ATL (Atlanta) to {dest_code if dest_code else 'the destination airport'}. "
            "Accept official Frontier route maps, timetables, booking engine results explicitly showing non-stop, or reputable "
            "route listings. Reject itineraries that require a connection."
        )
    ))

    # Lounge access availability
    lt = city_item.lounge_type or "lounge access (Centurion or Priority Pass)"
    lname = city_item.lounge_name or "the specified facility"
    lounge_claim = (
        f"At {dest_code if dest_code else 'the destination airport'}, {lname} is available and is accessible via {lt}. "
        "If it is a Centurion Lounge or an Escape Lounge – The Centurion Studio Partner, it counts as AmEx Centurion access. "
        "If it is a Priority Pass lounge or a Priority Pass dining partner, it counts as Priority Pass access."
    )
    claims_and_sources.append((
        lounge_claim,
        city_item.lounge_sources,
        lounge_leaf,
        (
            "Confirm that the provided page(s) show this facility at the destination airport and that it is accessible under the "
            "specified program: either (a) American Express Centurion Lounge (or 'Escape Lounge – The Centurion Studio Partner'), "
            "or (b) Priority Pass (including dining partners/restaurants that offer a credit). "
            "Minor name variations are fine, but the airport must match."
        )
    ))

    # Hotel within 5 miles
    hname = city_item.hotel_name or "the hotel"
    distance_claim = (
        f"The hotel '{hname}' is within 5 miles of {dest_code if dest_code else 'the destination airport'}."
    )
    claims_and_sources.append((
        distance_claim,
        city_item.hotel_sources,
        distance_leaf,
        (
            "Use the hotel's official site or a reputable source to confirm the approximate distance from the airport. "
            "If a number is given on the page (e.g., '2 miles from the airport'), treat it as authoritative. "
            "Qualify as 'within 5 miles' if the distance is 5.0 miles or less. If only drive time is shown without "
            "distance, do not assume; the page should reasonably support within 5 miles."
        )
    ))

    # Complimentary shuttle service
    shuttle_claim = (
        f"The hotel '{hname}' provides complimentary (free) airport shuttle service to and from "
        f"{dest_code if dest_code else 'the destination airport'}."
    )
    claims_and_sources.append((
        shuttle_claim,
        city_item.hotel_sources,
        shuttle_leaf,
        (
            "Confirm that the hotel offers a complimentary (free/no-charge) airport shuttle service. "
            "Phrases like 'complimentary airport shuttle' or 'free airport shuttle' qualify. "
            "If the page indicates a fee or only a third-party/paid shuttle, then this does NOT qualify."
        )
    ))

    await evaluator.batch_verify(claims_and_sources)


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Frontier multi-city lounge + shuttle task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: each city independently contributes to the score
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

    # Note: We intentionally set root as non-critical to allow partial credit aggregation across cities.
    evaluator.add_custom_info(
        info={"note": "Root node set non-critical for framework compatibility and partial-credit aggregation."},
        info_type="design_choice",
        info_name="criticality_adjustment"
    )

    # 1) Extract itinerary items
    extraction = await evaluator.extract(
        prompt=prompt_extract_itinerary(),
        template_class=ItineraryExtraction,
        extraction_name="itinerary_extraction",
    )

    # Keep only the first 4 items; pad if fewer
    cities = _first_k(extraction.cities, 4)
    cities = _pad_to_4(cities)

    # 2) Structural requirement checks (parallel)
    structural = evaluator.add_parallel(
        id="Structural_Requirements",
        desc="Verify structural requirements for the itinerary",
        parent=root,
        critical=False,
    )

    # City count verification: exactly 4 distinct city names provided among the first 4 items
    provided_city_names = [c.city for c in cities if _norm(c.city)]
    city_count_ok = (len(provided_city_names) == 4) and (_distinct_count([_norm(n) for n in provided_city_names]) == 4)
    evaluator.add_custom_node(
        result=city_count_ok,
        id="City_Count_Verification",
        desc="Verify exactly 4 distinct cities are provided",
        parent=structural,
        critical=False,
    )

    # State uniqueness verification: states present and all 4 are different
    provided_states = [c.state for c in cities if _norm(c.state)]
    state_unique_ok = (len(provided_states) == 4) and (_distinct_count([_norm(s) for s in provided_states]) == 4)
    evaluator.add_custom_node(
        result=state_unique_ok,
        id="State_Uniqueness_Verification",
        desc="Verify all 4 cities are in different US states",
        parent=structural,
        critical=False,
    )

    # 3) Per-city verification subtrees
    city_tasks = []
    for i in range(4):
        city_tasks.append(verify_one_city(evaluator, root, cities[i], i))

    await asyncio.gather(*city_tasks)

    # 4) Return structured summary
    return evaluator.get_summary()