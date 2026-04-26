import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "thanksgiving_2025_theater_perf"
TASK_DESCRIPTION = (
    "Among the three artists who performed at the 2025 NFL Thanksgiving Day halftime shows, identify which performer also performed at a theater venue "
    "(defined as having a seating capacity under 5,000) during 2025. Provide the following details about this theater performance: (1) the performer's name, "
    "(2) the venue name, (3) the venue's seating capacity, (4) the performance date, and (5) the city and state where the venue is located."
)


class HalftimePerformerExtraction(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TheaterPerformanceExtraction(BaseModel):
    performer_name: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state_or_country: Optional[str] = None
    performance_date: Optional[str] = None
    seating_capacity: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)
    performance_source_urls: List[str] = Field(default_factory=list)


def prompt_extract_halftime_performer() -> str:
    return (
        "From the provided answer, extract the single performer that the answer claims is one of the three 2025 NFL Thanksgiving Day halftime show artists.\n"
        "Return:\n"
        "- name: The performer's name exactly as written in the answer.\n"
        "- sources: An array of URLs that the answer cites to support that this person performed at a 2025 Thanksgiving Day NFL halftime show.\n"
        "Rules:\n"
        "1) Only extract what is explicitly present in the answer.\n"
        "2) If multiple Thanksgiving halftime performers are mentioned, pick the one the answer associates with the theater performance.\n"
        "3) If the answer provides no URLs, return an empty array for sources.\n"
    )


def prompt_extract_theater_performance() -> str:
    return (
        "From the provided answer, extract details for a single 2025 theater performance (venue capacity < 5,000) by the same performer.\n"
        "Return:\n"
        "- performer_name: The performer tied to this theater show.\n"
        "- venue_name: Official venue name.\n"
        "- city: City where the venue is located.\n"
        "- state_or_country: State (for U.S.) or country if non-U.S.\n"
        "- performance_date: The specific date of the performance as written in the answer.\n"
        "- seating_capacity: The seating capacity number/value as written in the answer.\n"
        "- capacity_source_urls: An array of URLs the answer cites to document the venue capacity.\n"
        "- performance_source_urls: An array of URLs the answer cites to document the performance event (listing, venue calendar, ticket site, or credible news).\n"
        "Rules:\n"
        "1) Extract only what appears in the answer; do not invent.\n"
        "2) If multiple performances are listed, choose one that fits the theater definition (< 5,000 seats) and took place in 2025.\n"
        "3) If any URLs are missing in the answer, return an empty array for that URL field.\n"
    )


def _extract_year(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    m = re.search(r"\b(20\d{2})\b", date_str)
    try:
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _parse_capacity_int(capacity_str: Optional[str]) -> Optional[int]:
    if not capacity_str:
        return None
    # Try to match comma-formatted or plain integer first
    m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,5})\b", capacity_str.replace(" ", ""))
    if m:
        digits = re.sub(r"[^\d]", "", m.group(1))
        try:
            return int(digits)
        except Exception:
            pass
    # Try patterns like "3k", "4.5k"
    m2 = re.search(r"\b(\d+(?:\.\d+)?)\s*k\b", capacity_str, flags=re.IGNORECASE)
    if m2:
        try:
            val = float(m2.group(1))
            return int(round(val * 1000))
        except Exception:
            return None
    return None


async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    halftime: HalftimePerformerExtraction,
    theater: TheaterPerformanceExtraction,
) -> None:
    # Top-level: critical sequential aggregation of all steps
    main_node = evaluator.add_sequential(
        id="theater_performer_identification_and_details",
        desc="Identify the Thanksgiving 2025 halftime performer who also performed at a <5,000-capacity theater venue in 2025, and provide all requested theater-performance details.",
        parent=root_node,
        critical=True,
    )

    # Part 1: Identify valid Thanksgiving halftime performer (critical sequential)
    id_node = evaluator.add_sequential(
        id="identify_valid_thanksgiving_halftime_performer",
        desc="Identify a performer who is one of the three 2025 NFL Thanksgiving Day halftime show performers.",
        parent=main_node,
        critical=True,
    )

    # Leaf: performer_name_provided
    performer_name_present = bool(halftime.name and halftime.name.strip())
    evaluator.add_custom_node(
        result=performer_name_present,
        id="performer_name_provided",
        desc="Provide the performer's name.",
        parent=id_node,
        critical=True,
    )

    # Leaf: performer_is_one_of_three_halftime_artists
    halftime_verify_leaf = evaluator.add_leaf(
        id="performer_is_one_of_three_halftime_artists",
        desc="The named performer is one of the three artists who performed at the 2025 NFL Thanksgiving Day halftime shows (Nov 27, 2025).",
        parent=id_node,
        critical=True,
    )
    halftime_claim = (
        f"{halftime.name} performed at a 2025 NFL Thanksgiving Day halftime show on Nov 27, 2025 "
        f"(Detroit Lions game, Dallas Cowboys game, or the night game)."
    )
    await evaluator.verify(
        claim=halftime_claim,
        node=halftime_verify_leaf,
        sources=halftime.sources if halftime.sources else None,
        additional_instruction=(
            "Use the provided sources (NFL team pages, credible news articles, or official announcements) to confirm that the person performed "
            "at one of the three Thanksgiving Day NFL halftime shows on Nov 27, 2025. Accept reasonable title/name variants."
        ),
    )

    # Part 2: Theater performance in 2025 with required fields (critical sequential)
    theater_node = evaluator.add_sequential(
        id="theater_performance_in_2025_with_required_fields",
        desc="Provide a 2025 theater performance by that performer at a venue with seating capacity under 5,000, including all requested details.",
        parent=main_node,
        critical=True,
    )

    # Sub-part: Required fields provided (critical parallel)
    details_provided_node = evaluator.add_parallel(
        id="theater_performance_details_provided",
        desc="Provide the required theater-performance fields (venue name, location, date in 2025, and seating capacity value) and ensure capacity documentation is provided.",
        parent=theater_node,
        critical=True,
    )

    # venue_name_provided
    evaluator.add_custom_node(
        result=bool(theater.venue_name and theater.venue_name.strip()),
        id="venue_name_provided",
        desc="Provide the official name of the theater venue.",
        parent=details_provided_node,
        critical=True,
    )

    # venue_location_provided (city and state_or_country)
    location_ok = bool(theater.city and theater.city.strip() and theater.state_or_country and theater.state_or_country.strip())
    evaluator.add_custom_node(
        result=location_ok,
        id="venue_location_provided",
        desc="Provide the city and state (or country) where the venue is located.",
        parent=details_provided_node,
        critical=True,
    )

    # performance_date_in_2025_provided
    year = _extract_year(theater.performance_date)
    date_ok = bool(theater.performance_date and theater.performance_date.strip() and year == 2025)
    evaluator.add_custom_node(
        result=date_ok,
        id="performance_date_in_2025_provided",
        desc="Provide the specific performance date, and it must fall in 2025.",
        parent=details_provided_node,
        critical=True,
    )

    # capacity_value_provided (must have a numeric value)
    capacity_has_number = _parse_capacity_int(theater.seating_capacity) is not None
    evaluator.add_custom_node(
        result=bool(theater.seating_capacity and theater.seating_capacity.strip() and capacity_has_number),
        id="capacity_value_provided",
        desc="State the venue's seating capacity as a value/number.",
        parent=details_provided_node,
        critical=True,
    )

    # capacity_docs_present (explicit existence check to ensure documentation is provided)
    capacity_docs_present_node = evaluator.add_custom_node(
        result=bool(theater.capacity_source_urls and len(theater.capacity_source_urls) > 0),
        id="capacity_docs_present",
        desc="At least one capacity documentation URL is provided in the answer.",
        parent=details_provided_node,
        critical=True,
    )

    # capacity_is_documented_verifiable (LLM verifies capacity against capacity_source_urls)
    capacity_verify_leaf = evaluator.add_leaf(
        id="capacity_is_documented_verifiable",
        desc="The seating capacity claim is backed by verifiable documentation/source (as required by the constraints).",
        parent=details_provided_node,
        critical=True,
    )
    capacity_claim = f"The seating capacity of {theater.venue_name} is {theater.seating_capacity}."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_verify_leaf,
        sources=theater.capacity_source_urls if theater.capacity_source_urls else None,
        additional_instruction=(
            "Confirm that the provided source(s) explicitly state the venue's seating capacity (or seating count). "
            "Allow minor variations (e.g., 'about', rounding). The key is that the cited source(s) substantively support the capacity value claimed."
        ),
        extra_prerequisites=[capacity_docs_present_node],
    )

    # theater_definition_satisfied (capacity under 5,000)
    parsed_capacity = _parse_capacity_int(theater.seating_capacity)
    capacity_under_5000 = bool(parsed_capacity is not None and parsed_capacity < 5000)
    evaluator.add_custom_node(
        result=capacity_under_5000,
        id="theater_definition_satisfied",
        desc="The stated seating capacity is under 5,000 (meets the theater definition).",
        parent=theater_node,
        critical=True,
    )

    # performance_matches_performer_venue_and_date (verify event details match a single performance)
    perf_match_leaf = evaluator.add_leaf(
        id="performance_matches_performer_venue_and_date",
        desc="The described performance is by the identified performer at the identified venue on the identified date (i.e., the performer/venue/date refer to the same event).",
        parent=theater_node,
        critical=True,
    )
    # Use the halftime-extracted performer as the authoritative identity for this check
    perf_claim = (
        f"On {theater.performance_date}, {halftime.name} performed at {theater.venue_name} in {theater.city}, {theater.state_or_country}."
    )
    await evaluator.verify(
        claim=perf_claim,
        node=perf_match_leaf,
        sources=theater.performance_source_urls if theater.performance_source_urls else None,
        additional_instruction=(
            "Verify that the sources show the performer on the specified date at the specified venue. "
            "Sources may include venue calendars, ticketing pages, tour schedules, or credible news posts. "
            "Accept reasonable variants in naming and date formatting; focus on performer, venue, and date matching the same event."
        ),
    )

    # Record helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "parsed_capacity_int": parsed_capacity,
            "performance_year": year,
            "halftime_sources_count": len(halftime.sources or []),
            "capacity_sources_count": len(theater.capacity_source_urls or []),
            "performance_sources_count": len(theater.performance_source_urls or []),
        },
        info_type="debug_meta",
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    halftime_task = evaluator.extract(
        prompt=prompt_extract_halftime_performer(),
        template_class=HalftimePerformerExtraction,
        extraction_name="halftime_performer",
    )
    theater_task = evaluator.extract(
        prompt=prompt_extract_theater_performance(),
        template_class=TheaterPerformanceExtraction,
        extraction_name="theater_performance",
    )

    halftime, theater = await asyncio.gather(halftime_task, theater_task)

    await build_verification_tree(evaluator, root, halftime, theater)

    return evaluator.get_summary()