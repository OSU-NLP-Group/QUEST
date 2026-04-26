import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "mlk_budget_ski_2026"
TASK_DESCRIPTION = (
    "You are planning a ski trip for Martin Luther King Jr. Day weekend 2026 (January 17-19, 2026) and want to use budget airlines to minimize travel costs. "
    "Identify 4 different US destination airports that meet ALL of the following criteria:\n"
    "1. The airport is served by direct flights from at least one budget airline (either Allegiant Air or Frontier Airlines) with routes operating during the 2025-2026 winter ski season\n"
    "2. The airport is located within 1 hour driving distance of at least one ski resort\n"
    "3. The nearby ski resort(s) are confirmed to be open and operational during MLK weekend 2026\n\n"
    "For each of the 4 airports, provide:\n"
    "- The airport code and city name\n"
    "- Which budget airline(s) (Allegiant Air or Frontier Airlines) serve this destination\n"
    "- The name of at least one ski resort within 1 hour driving distance\n"
    "- Verification that this ski resort will be open during MLK weekend 2026 (January 17-19, 2026)"
)

EXPECTED_MLK_WEEKEND = "January 17-19, 2026"
EXPECTED_MLK_DAY = "Monday, January 19, 2026"  # Federal holiday date to be recognized
BUDGET_AIRLINES = {"Allegiant Air", "Frontier Airlines"}


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class ResortInfo(BaseModel):
    name: Optional[str] = None
    proximity_statement: Optional[str] = None
    proximity_sources: List[str] = Field(default_factory=list)
    mlk_open_statement: Optional[str] = None
    mlk_open_sources: List[str] = Field(default_factory=list)


class DestinationAirport(BaseModel):
    airport_code: Optional[str] = None
    city: Optional[str] = None
    budget_airlines: List[str] = Field(default_factory=list)
    airline_sources: List[str] = Field(default_factory=list)
    resorts: List[ResortInfo] = Field(default_factory=list)


class DestinationsExtraction(BaseModel):
    airports: List[DestinationAirport] = Field(default_factory=list)


class MLKWeekendExtraction(BaseModel):
    mlk_weekend_dates_mentioned: Optional[str] = None
    mlk_day_mentioned: Optional[str] = None
    date_evidence_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_destinations() -> str:
    return """
Extract up to 4 different US destination airports the answer proposes for an MLK 2026 ski trip, preserving the original order. For each airport, extract the following fields:

- airport_code: IATA code (e.g., DEN). Use exactly what appears in the answer; if absent, return null.
- city: City name for the airport (e.g., Denver, CO). If absent, return null.
- budget_airlines: List of budget airlines that (per the answer) serve this destination by direct flights. Only include exactly 'Allegiant Air' or 'Frontier Airlines' if mentioned. Ignore other airlines.
- airline_sources: URLs cited that support the budget airline service (e.g., airline route maps, seasonal schedules, press releases). Extract only URLs explicitly present in the answer.
- resorts: An array of resorts (at least one is expected), where each resort includes:
  - name: Resort name as stated.
  - proximity_statement: Any statement about drive time or distance to the airport/city, if present.
  - proximity_sources: URL(s) that support the within-1-hour proximity (e.g., resort "getting here" page, Google Maps links, regional tourism pages).
  - mlk_open_statement: Any statement in the answer asserting the resort will be open/operational during MLK weekend 2026.
  - mlk_open_sources: URL(s) that support being open during Jan 17-19, 2026 (e.g., season dates page, operating calendar, hours).

Rules:
- Do not invent information. If any field is missing in the answer, set it to null or an empty list as appropriate.
- Only extract URLs that are explicitly present in the answer (in plain text or Markdown).
- Keep the first 4 unique airports as they appear. Uniqueness should be judged by airport_code if available; if code is missing, use city name.
- If fewer than 4 airports are present, return as many as are given.
"""


def prompt_extract_mlk_weekend_dates() -> str:
    return f"""
Identify how the answer describes MLK weekend 2026 and the holiday date. Extract:
- mlk_weekend_dates_mentioned: The date range for MLK weekend 2026 mentioned in the answer, if any (e.g., "{EXPECTED_MLK_WEEKEND}"). If not explicitly mentioned, return null.
- mlk_day_mentioned: How the answer states the federal holiday date for MLK Day 2026 (e.g., "{EXPECTED_MLK_DAY}"), if any. If not mentioned, return null.
- date_evidence_urls: Any URLs the answer cites to justify these dates (if any). If none, return an empty list.

Only extract text and URLs explicitly present in the answer.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _first_n_unique_airports(airports: List[DestinationAirport], n: int = 4) -> List[DestinationAirport]:
    seen = set()
    picked: List[DestinationAirport] = []
    for ap in airports:
        key = (ap.airport_code or "").strip().upper() or (ap.city or "").strip().lower()
        if not key:
            # If both missing, still include but ensure uniqueness via running index
            key = f"unknown_{len(picked)}"
        if key in seen:
            continue
        seen.add(key)
        picked.append(ap)
        if len(picked) >= n:
            break
    return picked


def _pad_to_k(items: List[DestinationAirport], k: int = 4) -> List[DestinationAirport]:
    out = list(items)
    while len(out) < k:
        out.append(DestinationAirport())
    return out[:k]


def _get_first_resort(airport: DestinationAirport) -> ResortInfo:
    return airport.resorts[0] if airport.resorts else ResortInfo()


# -----------------------------------------------------------------------------
# Verification subroutines
# -----------------------------------------------------------------------------
async def verify_mlk_weekend_dates(evaluator: Evaluator, parent_node) -> None:
    """
    Verify the answer identifies MLK weekend 2026 correctly (Jan 17-19, 2026; holiday Monday Jan 19, 2026).
    """
    node = evaluator.add_leaf(
        id="mlk_weekend_dates",
        desc="The MLK weekend 2026 dates are correctly identified as January 17-19, 2026, with the federal holiday on Monday, January 19, 2026",
        parent=parent_node,
        critical=True,
    )
    claim = (
        "The answer identifies MLK weekend 2026 as January 17-19, 2026 and explicitly (or implicitly) recognizes that MLK Day is Monday, January 19, 2026."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            "Judge based on the answer content. Accept reasonable equivalent phrasings or partial mentions as long as the intended weekend "
            "dates (Jan 17-19, 2026) are correct and the holiday is recognized as Monday Jan 19, 2026."
        ),
    )


async def add_information_completeness_checks(evaluator: Evaluator, parent_node, airports: List[DestinationAirport]) -> None:
    """
    Build a critical completeness subtree ensuring the answer provided all required fields for each airport.
    """
    comp_root = evaluator.add_parallel(
        id="information_completeness",
        desc="For each airport, required information is provided: airport code and city name, budget airline(s), at least one nearby ski resort, and verification of being open during MLK weekend 2026",
        parent=parent_node,
        critical=True,
    )

    for idx in range(4):
        ap = airports[idx] if idx < len(airports) else DestinationAirport()
        res = _get_first_resort(ap)

        # Each child under a critical parent must be critical per framework rules
        sub = evaluator.add_parallel(
            id=f"airport_{idx+1}_completeness",
            desc=f"Airport #{idx+1} completeness checks",
            parent=comp_root,
            critical=True,
        )

        evaluator.add_custom_node(
            result=bool(ap.airport_code and ap.city),
            id=f"airport_{idx+1}_code_city_present",
            desc=f"Airport #{idx+1}: airport code and city name are provided",
            parent=sub,
            critical=True,
        )

        evaluator.add_custom_node(
            result=bool(set(a.strip() for a in ap.budget_airlines) & BUDGET_AIRLINES),
            id=f"airport_{idx+1}_budget_airlines_present",
            desc=f"Airport #{idx+1}: at least one budget airline (Allegiant Air or Frontier Airlines) is listed",
            parent=sub,
            critical=True,
        )

        evaluator.add_custom_node(
            result=bool(res.name),
            id=f"airport_{idx+1}_resort_present",
            desc=f"Airport #{idx+1}: at least one ski resort name within 1 hour is provided",
            parent=sub,
            critical=True,
        )

        evaluator.add_custom_node(
            result=bool((res.mlk_open_statement and res.mlk_open_statement.strip()) or (res.mlk_open_sources)),
            id=f"airport_{idx+1}_mlk_open_verification_present",
            desc=f"Airport #{idx+1}: verification statement or source(s) confirming MLK weekend 2026 opening is provided",
            parent=sub,
            critical=True,
        )


async def verify_single_destination(
    evaluator: Evaluator,
    parent_node,
    airport: DestinationAirport,
    idx: int,
) -> None:
    """
    Build verification checks for a single airport.
    Structure: Sequential node gated by required-info existence, then three factual checks with sources.
    """
    dest = evaluator.add_sequential(
        id=f"destination_airport_{idx+1}",
        desc=(
            f"{['First','Second','Third','Fourth'][idx]} destination airport meets all requirements: "
            "(1) served by Allegiant Air or Frontier Airlines with direct flights during winter 2025-2026 season, "
            "(2) within 1 hour driving distance of at least one ski resort, and "
            "(3) that ski resort is confirmed open during MLK weekend 2026 (January 17-19, 2026)"
        ),
        parent=parent_node,
        critical=False,
    )

    first_resort = _get_first_resort(airport)
    has_required = (
        bool(airport.airport_code and airport.city)
        and bool(set(a.strip() for a in airport.budget_airlines) & BUDGET_AIRLINES)
        and bool(first_resort.name)
    )

    evaluator.add_custom_node(
        result=has_required,
        id=f"dest_{idx+1}_required_info",
        desc=f"Airport #{idx+1} has required fields (airport code, city, at least one eligible budget airline, and at least one resort)",
        parent=dest,
        critical=True,
    )

    # 1) Airline direct service during winter 2025-2026
    airline_leaf = evaluator.add_leaf(
        id=f"dest_{idx+1}_budget_airline_service",
        desc=f"Airport #{idx+1}: At least one of Allegiant Air or Frontier Airlines operates direct flights to this airport during winter 2025-2026",
        parent=dest,
        critical=True,
    )
    airline_list_str = ", ".join(sorted(set(airport.budget_airlines))) if airport.budget_airlines else "none"
    claim_airline = (
        f"At least one of Allegiant Air or Frontier Airlines operates direct/nonstop flights to {airport.airport_code or 'the destination airport'} "
        f"during the winter 2025-2026 season (around January 2026). The answer lists: [{airline_list_str}]."
    )
    await evaluator.verify(
        claim=claim_airline,
        node=airline_leaf,
        sources=airport.airline_sources,
        additional_instruction=(
            "Use the provided URLs (airline route map, seasonal schedule, press release, etc.) to confirm a direct/nonstop service by Allegiant Air or Frontier Airlines "
            "to the destination airport. The service should reasonably cover winter 2025–2026 (around January 2026). "
            "If sources only show legacy/old routes not active in winter 2025–2026, consider it not supported."
        ),
    )

    # 2) Within 1 hour driving distance to a ski resort
    proximity_leaf = evaluator.add_leaf(
        id=f"dest_{idx+1}_proximity_within_1hr",
        desc=f"Airport #{idx+1}: The cited ski resort is within 60 minutes driving distance of the airport/city",
        parent=dest,
        critical=True,
    )
    origin_label = f"{airport.airport_code} airport in {airport.city}" if (airport.airport_code and airport.city) else (airport.city or airport.airport_code or "the airport")
    claim_proximity = (
        f"The driving time from {origin_label} to the ski resort {first_resort.name or 'the cited resort'} is within 60 minutes."
    )
    await evaluator.verify(
        claim=claim_proximity,
        node=proximity_leaf,
        sources=first_resort.proximity_sources,
        additional_instruction=(
            "Rely on provided sources (e.g., Google Maps links, resort 'getting here' pages, or local guides). "
            "Allow reasonable rounding. If typical drive time is approximately an hour (e.g., 55–65 minutes), consider it acceptable "
            "as within 60 minutes, unless clearly exceeding."
        ),
    )

    # 3) Resort open during MLK weekend 2026
    open_leaf = evaluator.add_leaf(
        id=f"dest_{idx+1}_resort_open_mlk",
        desc=f"Airport #{idx+1}: The ski resort is open and operational during MLK weekend 2026 (Jan 17-19, 2026)",
        parent=dest,
        critical=True,
    )
    claim_open = (
        f"The ski resort {first_resort.name or 'the cited resort'} will be open and operational on January 17, 18, and 19, 2026 (MLK weekend)."
    )
    await evaluator.verify(
        claim=claim_open,
        node=open_leaf,
        sources=first_resort.mlk_open_sources,
        additional_instruction=(
            "Use official resort calendars, season dates, or operating hours. It is sufficient if the official season dates clearly include Jan 17–19, 2026, "
            "and there is no notice of a scheduled closure for those dates. If credible sources show the resort is closed, not operating, or on hold for those dates, "
            "consider it not supported."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Entry point for evaluating the MLK 2026 budget airline ski destination task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Airports evaluated independently; global checks in parallel
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

    # Perform extractions (can run in parallel)
    dest_task = evaluator.extract(
        prompt=prompt_extract_destinations(),
        template_class=DestinationsExtraction,
        extraction_name="destinations_extraction",
    )
    mlk_task = evaluator.extract(
        prompt=prompt_extract_mlk_weekend_dates(),
        template_class=MLKWeekendExtraction,
        extraction_name="mlk_weekend_extraction",
    )
    destinations_extraction, mlk_dates_extraction = await asyncio.gather(dest_task, mlk_task)

    # Normalize airport list: first 4 unique, then pad if needed
    picked_airports = _first_n_unique_airports(destinations_extraction.airports, 4)
    picked_airports = _pad_to_k(picked_airports, 4)

    # Add ground-truth reference info for context
    evaluator.add_ground_truth(
        {
            "expected_mlk_weekend": EXPECTED_MLK_WEEKEND,
            "expected_mlk_day": EXPECTED_MLK_DAY,
            "budget_airlines_allowed": sorted(list(BUDGET_AIRLINES)),
            "requirements": [
                "Direct flights by Allegiant Air or Frontier Airlines for winter 2025-2026",
                "Within 1 hour drive to at least one ski resort",
                "Resort open during Jan 17-19, 2026",
            ],
        },
        gt_type="task_requirements",
    )

    # Global check: dates correctly identified in the answer
    await verify_mlk_weekend_dates(evaluator, root)

    # Completeness checks across 4 airports (critical)
    await add_information_completeness_checks(evaluator, root, picked_airports)

    # Per-airport verification subtrees
    for i in range(4):
        await verify_single_destination(evaluator, root, picked_airports[i], i)

    # Return the evaluation summary
    return evaluator.get_summary()