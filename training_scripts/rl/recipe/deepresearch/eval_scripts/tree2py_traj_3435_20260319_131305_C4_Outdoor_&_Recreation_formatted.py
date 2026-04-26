import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rmnp_budget_airline_2026"
TASK_DESCRIPTION = """
Which budget airline among Allegiant Air, Breeze Airways, Avelo Airlines, and Sun Country Airlines serves the closest commercial airport to Rocky Mountain National Park with nonstop flights in 2026? Provide the airline name, the airport code, at least one origin city for nonstop service, and the approximate travel time from the airport to the park entrance.
"""

ALLOWED_BUDGET_AIRLINES = [
    "Allegiant Air",
    "Breeze Airways",
    "Avelo Airlines",
    "Sun Country Airlines"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RMNPAnswerExtraction(BaseModel):
    """
    Structured extraction of the agent's answer.
    """
    airline: Optional[str] = None                        # Chosen airline name
    airport_name: Optional[str] = None                  # Airport name (e.g., Denver International Airport)
    airport_code: Optional[str] = None                  # Airport code (IATA or FAA LID), e.g., DEN or KDEN
    origin_cities: List[str] = Field(default_factory=list)  # One or more nonstop origins (city or city/airport)
    travel_time: Optional[str] = None                   # Approximate travel time/distance statement to RMNP entrance

    # Source URLs grouped by purpose if the answer provides them
    airport_urls: List[str] = Field(default_factory=list)   # Airport identity/airport code/commercial service evidence
    route_urls: List[str] = Field(default_factory=list)     # Airline route/nonstop/2026 service evidence
    distance_urls: List[str] = Field(default_factory=list)  # Travel time/distance to RMNP evidence (e.g., maps)
    sources: List[str] = Field(default_factory=list)        # Any additional general sources


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
Extract the final recommendation the answer makes in response to:
- Which budget airline (restricted to: Allegiant Air, Breeze Airways, Avelo Airlines, Sun Country Airlines)
- Which airport (name and code)
- At least one origin city for a nonstop route associated with that airline/airport that is operating in 2026
- An approximate travel time and/or distance from the airport to a Rocky Mountain National Park entrance
- All URLs cited that support the above

Return a JSON object with fields:
- airline: string | null
- airport_name: string | null
- airport_code: string | null
- origin_cities: array of string (0 or more)
- travel_time: string | null

Also extract supporting URLs (put only actual URLs):
- airport_urls: array of string (airport facts, code, commercial service classification, etc.)
- route_urls: array of string (pages that show the airline's nonstop route(s) and that the route operates in 2026)
- distance_urls: array of string (pages showing approximate drive time or distance from the airport to an RMNP entrance, e.g., maps/NPS)
- sources: array of string (any other URLs mentioned)

Guidance:
- If multiple airlines/airports are discussed, choose the single one that the answer ultimately recommends as "the closest" per the task framing.
- For URLs, include the full URL; extract from markdown links if needed.
- Do NOT fabricate any information; if a field is missing in the answer, return it as null (or empty array where appropriate).
- Normalize airline names if obvious (e.g., "Avelo" -> "Avelo Airlines", "Sun Country" -> "Sun Country Airlines", "Breeze" -> "Breeze Airways", "Allegiant" -> "Allegiant Air").
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _airport_label(name: Optional[str], code: Optional[str]) -> str:
    if name and code:
        return f"{name} ({code})"
    return (name or "").strip() or (code or "").strip() or "the identified airport"


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: RMNPAnswerExtraction) -> None:
    """
    Build the verification tree exactly as defined in the rubric and run checks.
    Root is parallel; all children are critical.
    """
    root = evaluator.root  # already initialized by caller

    # Consolidate sources
    airport_sources = extraction.airport_urls
    route_sources = extraction.route_urls
    distance_sources = extraction.distance_urls
    all_sources = _merge_unique_urls(airport_sources, route_sources, distance_sources, extraction.sources)

    # 1) The identified airline is in the allowed set
    node_airline_ok = evaluator.add_leaf(
        id="budget_airline_identification",
        desc="The identified airline is one of: Allegiant Air, Breeze Airways, Avelo Airlines, or Sun Country Airlines.",
        parent=root,
        critical=True
    )
    airline_name = extraction.airline or ""
    claim_airline = (
        f"The identified airline '{airline_name}' is one of the following: "
        "Allegiant Air, Breeze Airways, Avelo Airlines, or Sun Country Airlines."
    )
    await evaluator.verify(
        claim=claim_airline,
        node=node_airline_ok,
        additional_instruction=(
            "Approve if the extracted airline name is semantically equivalent to one of the allowed names. "
            "Allow minor naming variants, e.g., 'Avelo' ~ 'Avelo Airlines', 'Breeze' ~ 'Breeze Airways', "
            "'Sun Country' ~ 'Sun Country Airlines', 'Allegiant' ~ 'Allegiant Air'."
        )
    )

    # 2) The identified airport provides commercial passenger service
    node_airport_commercial = evaluator.add_leaf(
        id="airport_is_commercial",
        desc="The identified airport provides commercial passenger service.",
        parent=root,
        critical=True
    )
    ap_label = _airport_label(extraction.airport_name, extraction.airport_code)
    claim_commercial = (
        f"{ap_label} is a commercial passenger service airport with scheduled airline service."
    )
    await evaluator.verify(
        claim=claim_commercial,
        node=node_airport_commercial,
        sources=airport_sources if airport_sources else all_sources,
        additional_instruction=(
            "Support the claim only if the evidence shows it is a commercial passenger service airport "
            "(e.g., FAA primary/non-primary commercial service classification, or explicit scheduled airline service). "
            "Mentions of 'general aviation only' or lack of scheduled passenger service should lead to 'not supported'."
        )
    )

    # 3) Closest airport verification (scoped to the question constraints)
    node_closest_scoped = evaluator.add_leaf(
        id="closest_airport_verification_scoped",
        desc="The identified airport is the closest commercial airport to Rocky Mountain National Park among airports that have qualifying nonstop service in 2026 operated by one of the listed budget airlines (as framed by the question/constraints).",
        parent=root,
        critical=True
    )
    claim_closest = (
        f"Among airports that have nonstop service in 2026 operated by Allegiant Air, Breeze Airways, "
        f"Avelo Airlines, or Sun Country Airlines, the closest commercial airport to Rocky Mountain National Park is {ap_label}."
    )
    await evaluator.verify(
        claim=claim_closest,
        node=node_closest_scoped,
        sources=all_sources,
        additional_instruction=(
            "Judge only within the scope: airports that (a) are commercial and (b) have nonstop service in 2026 by one of the listed budget airlines. "
            "Support if the sources explicitly state this airport is the closest to RMNP under this framing, "
            "or if distances/drive times provided clearly establish it is closer than other qualifying alternatives. "
            "If evidence suggests a different qualifying airport is closer, or if the claim lacks sufficient support, mark as not supported."
        )
    )

    # 4) Nonstop service in 2026
    node_nonstop_2026 = evaluator.add_leaf(
        id="nonstop_service_in_2026",
        desc="The identified airline offers nonstop service to the identified airport, and the service/route is operating or available in 2026.",
        parent=root,
        critical=True
    )
    first_origin = extraction.origin_cities[0] if extraction.origin_cities else None
    if first_origin:
        claim_nonstop = (
            f"In 2026, {airline_name} operates nonstop flights between {first_origin} and {ap_label}."
        )
    else:
        claim_nonstop = (
            f"In 2026, {airline_name} operates nonstop flights to {ap_label}."
        )
    await evaluator.verify(
        claim=claim_nonstop,
        node=node_nonstop_2026,
        sources=route_sources if route_sources else all_sources,
        additional_instruction=(
            "Accept seasonal or limited schedules as long as the route clearly operates at some point in 2026. "
            "Prefer evidence from the airline's official site, schedules/timetables, or reputable news/airport pages. "
            "Reject if only historical (pre-2026) or discontinued service is shown without 2026 relevance."
        )
    )

    # 5) Airport code provided (correctness)
    node_code = evaluator.add_leaf(
        id="airport_code_provided",
        desc="The correct airport code (FAA/IATA designation as applicable) for the identified airport is provided.",
        parent=root,
        critical=True
    )
    code_text = extraction.airport_code or ""
    claim_code = (
        f"The correct IATA or FAA code for {extraction.airport_name or 'the identified airport'} is '{code_text}'."
    )
    await evaluator.verify(
        claim=claim_code,
        node=node_code,
        sources=airport_sources if airport_sources else all_sources,
        additional_instruction=(
            "Allow either IATA (e.g., 'DEN') or FAA LID/ICAO forms (e.g., 'KDEN') to substantiate correctness. "
            "Minor formatting differences (with/without leading 'K') should be considered correct if the code clearly refers to the same airport."
        )
    )

    # 6) Origin city identified (existence in the answer)
    node_origin_exists = evaluator.add_custom_node(
        result=bool(extraction.origin_cities),
        id="origin_city_identified",
        desc="At least one origin city from which nonstop service to the identified airport is available (in 2026) is provided.",
        parent=root,
        critical=True
    )

    # 7) Approximate travel time/distance stated (existence in the answer)
    node_travel_time_exists = evaluator.add_custom_node(
        result=bool((extraction.travel_time or "").strip()),
        id="distance_or_travel_time_to_entrance",
        desc="An approximate travel time and/or distance from the identified airport to a Rocky Mountain National Park entrance is stated.",
        parent=root,
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
    Evaluate an answer for the RMNP closest budget-airline-served airport task.
    """
    # Initialize evaluator with a parallel root (as per rubric)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Record allowed airlines as GT/context (optional)
    evaluator.add_ground_truth(
        {"allowed_budget_airlines": ALLOWED_BUDGET_AIRLINES},
        gt_type="allowed_options"
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=RMNPAnswerExtraction,
        extraction_name="parsed_answer_core"
    )

    # Build verification tree and perform checks
    await build_and_verify_tree(evaluator, extraction)

    # Return standardized summary
    return evaluator.get_summary()