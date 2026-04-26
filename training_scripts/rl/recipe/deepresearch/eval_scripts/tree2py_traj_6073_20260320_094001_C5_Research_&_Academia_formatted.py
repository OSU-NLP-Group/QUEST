import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eclipse_spain_2026_city_select"
TASK_DESCRIPTION = """
For the total solar eclipse occurring on August 12, 2026, identify a major Spanish city that is suitable for hosting eclipse tourists. The city must satisfy the following criteria:

1. The city must be located within the path of totality for the August 12, 2026 solar eclipse
2. The city must experience at least 90 seconds (1 minute 30 seconds) of totality during the eclipse
3. The city must have a population of at least 100,000 inhabitants
4. The city must have an international airport located either within the city or within 100 kilometers of the city center
5. The city must have hotel accommodation infrastructure, demonstrated by the existence of at least 10 hotels
6. If multiple cities satisfy all the above mandatory criteria, select the city that experiences the longest duration of totality

Provide the name of the city, its totality duration during the eclipse, its population, the name and location of the international airport serving it, and evidence of hotel infrastructure. Include reference URLs supporting each piece of information.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CitySelectionExtraction(BaseModel):
    # Core selection
    city_name: Optional[str] = None

    # Location in Spain
    spain_urls: List[str] = Field(default_factory=list)

    # Eclipse path and duration
    totality_path_urls: List[str] = Field(default_factory=list)
    totality_duration_text: Optional[str] = None  # e.g., "1m 45s", "105 seconds", "1:45"
    totality_urls: List[str] = Field(default_factory=list)

    # Population
    population_text: Optional[str] = None  # e.g., "345,678 (2023)"
    population_urls: List[str] = Field(default_factory=list)

    # Airport access
    airport_name: Optional[str] = None  # e.g., "Bilbao Airport (BIO)"
    airport_location_text: Optional[str] = None  # e.g., "within the city", "in Loiu, 12 km from Bilbao"
    airport_distance_text: Optional[str] = None  # e.g., "12 km", "50 miles"
    airport_urls: List[str] = Field(default_factory=list)

    # Hotel infrastructure
    hotel_count_text: Optional[str] = None  # e.g., "150 hotels", "200+ properties"
    hotel_urls: List[str] = Field(default_factory=list)

    # Cultural/tourist attractions
    attraction_text: Optional[str] = None  # e.g., "Guggenheim Museum Bilbao"
    attraction_urls: List[str] = Field(default_factory=list)

    # Tie-break (if asserted by the answer)
    tie_break_asserted: Optional[bool] = None
    tie_break_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_city_selection() -> str:
    return """
    Extract from the answer the SINGLE selected Spanish city proposed for hosting eclipse tourists for the total solar eclipse on 12 August 2026, together with the supporting details and the exact URLs cited.

    Extraction requirements:
    1) city_name: The single final city selected/recommended. If multiple cities are discussed, choose the one explicitly recommended as the final pick; if unclear, choose the first city that fully meets the constraints in the answer.
    2) spain_urls: URL(s) that support that the city is located in Spain (e.g., a Wikipedia or official page clearly stating the city is in Spain).
    3) totality_path_urls: URL(s) that support that the city lies within the path of totality on 12 August 2026 (not partial).
    4) totality_duration_text: The totality duration reported in the answer for the city (keep the original formatting from the answer, such as "1m 45s", "105 seconds", or "1:45"). If not provided, return null.
    5) totality_urls: URL(s) cited for the totality duration or path for this specific city.
    6) population_text: The population figure reported in the answer for the city (keep punctuation and year if mentioned). If not provided, return null.
    7) population_urls: URL(s) cited for the population.
    8) airport_name: The name of an INTERNATIONAL airport that serves the city.
    9) airport_location_text: The stated location of the airport relative to the city (e.g., “in the city”, “12 km from CITY”, “in TOWN near CITY”).
    10) airport_distance_text: The reported distance if given (e.g., “12 km”, “50 miles”). If not provided, return null.
    11) airport_urls: URL(s) cited for the airport (ideally showing it is an international airport and its location/distance).
    12) hotel_count_text: The reported number of hotels or a phrase indicating a count (e.g., “150 hotels”, “200+ properties”). If not provided, return null.
    13) hotel_urls: URL(s) cited as evidence for hotels in the city (e.g., a listing page, tourism board, hotel association).
    14) attraction_text: One notable cultural/tourist attraction cited for the city (e.g., a UNESCO site, major museum, landmark). If not provided, return null.
    15) attraction_urls: URL(s) cited for that attraction or for a notable cultural/tourist asset in the city.
    16) tie_break_asserted: true if the answer explicitly or implicitly asserts that the selected city has the longest (or longer/longest) totality among qualifying Spanish cities; otherwise false.
    17) tie_break_urls: URL(s) cited to support the tie-break assertion (e.g., a Spain-specific eclipse-duration map/table).

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer (plain URLs or in markdown links). Do not invent or infer URLs.
    - Prefer authoritative sources for eclipse data (e.g., NASA, Xavier Jubier, timeanddate.com) when present in the answer.
    - If a field is not mentioned in the answer, set it to null (or an empty array for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        us = u.strip()
        if not us:
            continue
        if us not in seen:
            seen.add(us)
            result.append(us)
    return result


async def _verify_with_required_sources(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: List[str],
    additional_instruction: str,
) -> bool:
    """
    Verify a claim only if there are non-empty URLs. If URLs are empty, mark the node as failed
    (because these checks explicitly require supporting sources).
    """
    clean_urls = _unique_nonempty(urls)
    if not clean_urls:
        # No sources provided; this leaf requires URL grounding -> fail
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=clean_urls,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_location_and_eclipse_nodes(evaluator: Evaluator, parent, data: CitySelectionExtraction) -> None:
    # Parent: critical + parallel
    loc_node = evaluator.add_parallel(
        id="location_and_eclipse_constraints",
        desc="City satisfies Spain + path-of-totality + totality-duration constraints with supporting evidence.",
        parent=parent,
        critical=True,
    )

    city = data.city_name or "the selected city"

    # 1) Located in Spain (critical leaf)
    spain_leaf = evaluator.add_leaf(
        id="located_in_spain",
        desc="City is located in Spain (with a supporting reference URL).",
        parent=loc_node,
        critical=True,
    )
    await _verify_with_required_sources(
        evaluator=evaluator,
        claim=f"The city of {city} is located in Spain.",
        node=spain_leaf,
        urls=data.spain_urls,
        additional_instruction="Use the provided URL(s) to confirm that the city is in Spain. Accept authoritative sources (e.g., Wikipedia, official portals).",
    )

    # 2) In path of totality (critical leaf)
    in_path_leaf = evaluator.add_leaf(
        id="in_path_of_totality",
        desc="City lies within the path of totality (not merely partial) for Aug 12, 2026 (with a supporting reference URL).",
        parent=loc_node,
        critical=True,
    )
    in_path_sources = _unique_nonempty(data.totality_path_urls or []) + [
        u for u in _unique_nonempty(data.totality_urls or []) if u not in _unique_nonempty(data.totality_path_urls or [])
    ]
    await _verify_with_required_sources(
        evaluator=evaluator,
        claim=f"The city of {city} lies within the path of totality (not partial) for the total solar eclipse on August 12, 2026.",
        node=in_path_leaf,
        urls=in_path_sources,
        additional_instruction="Verify the Spanish location is inside the totality band for the 2026-08-12 eclipse. Do NOT use info for other eclipse years.",
    )

    # 3) Totality duration (critical + sequential: reported first, then threshold)
    duration_node = evaluator.add_sequential(
        id="totality_duration",
        desc="Totality duration is provided with a source and meets the minimum requirement.",
        parent=loc_node,
        critical=True,
    )

    # 3.1) Reported with source (critical leaf)
    duration_report_leaf = evaluator.add_leaf(
        id="totality_duration_reported_with_source",
        desc="Answer reports the city’s totality duration (minutes:seconds) for Aug 12, 2026 and provides a supporting reference URL for that duration.",
        parent=duration_node,
        critical=True,
    )
    duration_text = (data.totality_duration_text or "").strip()
    await _verify_with_required_sources(
        evaluator=evaluator,
        claim=f"On August 12, 2026, the duration of totality in {city} is {duration_text}.",
        node=duration_report_leaf,
        urls=data.totality_urls,
        additional_instruction="Check that the page explicitly states (or allows clear inference of) the totality duration for this city on 2026-08-12. Minor formatting differences (e.g., '1m45s' vs '1:45') are acceptable.",
    )

    # 3.2) Meets minimum (critical leaf)
    duration_min_leaf = evaluator.add_leaf(
        id="totality_duration_meets_minimum",
        desc="Reported totality duration is at least 90 seconds (≥ 1 minute 30 seconds).",
        parent=duration_node,
        critical=True,
    )
    await _verify_with_required_sources(
        evaluator=evaluator,
        claim=f"The duration of totality in {city} on August 12, 2026 is at least 90 seconds (1 minute 30 seconds).",
        node=duration_min_leaf,
        urls=data.totality_urls,
        additional_instruction="Verify the page shows a duration >= 90 seconds for this location on 2026-08-12. Allow small rounding differences.",
    )


async def build_city_capacity_nodes(evaluator: Evaluator, parent, data: CitySelectionExtraction) -> None:
    # Parent: critical + parallel
    cap_node = evaluator.add_parallel(
        id="city_capacity_constraints",
        desc="City satisfies population, airport access, and hotel-infrastructure requirements with supporting evidence.",
        parent=parent,
        critical=True,
    )

    city = data.city_name or "the selected city"

    # A) Population (critical + sequential)
    pop_node = evaluator.add_sequential(
        id="population",
        desc="Population is provided with a source and meets the threshold.",
        parent=cap_node,
        critical=True,
    )

    # A.1) Reported with source (existence check)
    pop_reported = evaluator.add_custom_node(
        result=bool((data.population_text or "").strip()) and len(_unique_nonempty(data.population_urls)) > 0,
        id="population_reported_with_source",
        desc="Answer reports a population figure for the city and provides a supporting reference URL.",
        parent=pop_node,
        critical=True,
    )

    # A.2) Meets threshold (critical verify)
    pop_meets_leaf = evaluator.add_leaf(
        id="population_meets_threshold",
        desc="Reported population is at least 100,000 inhabitants.",
        parent=pop_node,
        critical=True,
    )
    await _verify_with_required_sources(
        evaluator=evaluator,
        claim=f"The population of {city} is at least 100,000 inhabitants.",
        node=pop_meets_leaf,
        urls=data.population_urls,
        additional_instruction="Use the cited page to confirm the population figure is >= 100,000. Allow reasonable rounding or year differences.",
    )

    # B) Airport access (critical + sequential)
    air_node = evaluator.add_sequential(
        id="airport_access",
        desc="International airport is identified with a source and is within 100 km (or in-city).",
        parent=cap_node,
        critical=True,
    )

    # B.1) Reported with source (verify that it's an international airport serving the city, with location text)
    airport_report_leaf = evaluator.add_leaf(
        id="airport_reported_with_source",
        desc="Answer names an international airport serving the city, provides its location, and includes a supporting reference URL.",
        parent=air_node,
        critical=True,
    )
    airport_name = (data.airport_name or "").strip() or "the identified airport"
    airport_loc_text = (data.airport_location_text or "").strip() or "serving the city"
    await _verify_with_required_sources(
        evaluator=evaluator,
        claim=f"The international airport '{airport_name}' serves {city}; location detail: {airport_loc_text}.",
        node=airport_report_leaf,
        urls=data.airport_urls,
        additional_instruction="Confirm from the provided page(s) that the named airport is an international airport and it serves the city, including the stated location relationship.",
    )

    # B.2) Within 100 km (critical verify)
    airport_within_leaf = evaluator.add_leaf(
        id="airport_within_100_km",
        desc="Airport is within the city or within 100 km of the city center (supported by the provided information/source).",
        parent=air_node,
        critical=True,
    )
    await _verify_with_required_sources(
        evaluator=evaluator,
        claim=f"'{airport_name}' is located within the city of {city} or within 100 kilometers of {city}'s city center.",
        node=airport_within_leaf,
        urls=data.airport_urls,
        additional_instruction="Check whether the provided page(s) indicate the airport is in the city or quote a distance <= 100 km (≈ 62 miles). If distance is in miles, allow 62 miles as equivalent to 100 km.",
    )

    # C) Hotel infrastructure (critical + sequential)
    hotel_node = evaluator.add_sequential(
        id="hotel_infrastructure",
        desc="Hotel infrastructure evidence is provided with a source and meets the minimum hotel count.",
        parent=cap_node,
        critical=True,
    )

    # C.1) Evidence reported with source (existence)
    hotel_reported = evaluator.add_custom_node(
        result=len(_unique_nonempty(data.hotel_urls)) > 0,
        id="hotel_evidence_reported_with_source",
        desc="Answer provides evidence for the number/existence of hotels in the city (e.g., a listing/count) and includes a supporting reference URL.",
        parent=hotel_node,
        critical=True,
    )

    # C.2) Meets minimum count (critical verify)
    hotel_meets_leaf = evaluator.add_leaf(
        id="hotel_count_meets_threshold",
        desc="Provided hotel evidence supports that the city has at least 10 hotels.",
        parent=hotel_node,
        critical=True,
    )
    await _verify_with_required_sources(
        evaluator=evaluator,
        claim=f"There are at least 10 hotels in {city}.",
        node=hotel_meets_leaf,
        urls=data.hotel_urls,
        additional_instruction="If the page lists hotels (possibly across multiple pages) or states a count showing >= 10, this criterion is satisfied.",
    )


async def build_cultural_nodes(evaluator: Evaluator, parent, data: CitySelectionExtraction) -> None:
    # Single critical leaf
    cult_leaf = evaluator.add_leaf(
        id="cultural_or_tourist_attractions",
        desc="City has notable cultural/tourist attractions (e.g., UNESCO World Heritage Site, major museum, historic landmark) and provides a supporting reference URL.",
        parent=parent,
        critical=True,
    )
    city = data.city_name or "the selected city"
    attraction_text = (data.attraction_text or "a notable cultural/tourist attraction").strip()
    await _verify_with_required_sources(
        evaluator=evaluator,
        claim=f"{city} has a notable cultural/tourist attraction: {attraction_text}.",
        node=cult_leaf,
        urls=data.attraction_urls,
        additional_instruction="Confirm that the cited page supports the existence/significance of the named attraction in the city.",
    )


async def build_tie_break_nodes(evaluator: Evaluator, parent, data: CitySelectionExtraction) -> None:
    # Make tie-break non-critical because it is conditional (only applicable if the answer asserts it)
    tie_node = evaluator.add_parallel(
        id="tie_break_longest_totality_if_applicable",
        desc="Tie-break rule: if multiple cities satisfy all mandatory constraints, the selected city must be the one with the longest totality duration.",
        parent=parent,
        critical=False,
    )

    # If tie-break was asserted and URLs present, verify that the citation enables the comparison
    tie_leaf = evaluator.add_leaf(
        id="tie_break_verifiability",
        desc="If the answer asserts longest totality among qualifying Spanish cities, it provides sufficient citation to verify the comparison.",
        parent=tie_node,
        critical=False,
    )

    asserted = bool(data.tie_break_asserted)
    tie_urls = _unique_nonempty(data.tie_break_urls)

    if asserted and tie_urls:
        await evaluator.verify(
            claim="This source provides eclipse totality durations (or a duration map/table) for Spanish locations for the August 12, 2026 eclipse, sufficient to compare and determine the city with the longest totality.",
            node=tie_leaf,
            sources=tie_urls,
            additional_instruction="Look for a Spain-focused 2026-08-12 eclipse duration map/table or list enabling cross-city comparison.",
        )
    else:
        # Not asserted or no URLs -> mark as skipped (optional requirement not engaged)
        tie_leaf.score = 0.0
        tie_leaf.status = "skipped"


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
    Evaluate an answer for the Spain eclipse 2026 city selection task.
    """
    # Initialize evaluator (root is non-critical; we'll add a critical child node to aggregate mandatory criteria)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Matches rubric root
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_city_selection(),
        template_class=CitySelectionExtraction,
        extraction_name="city_selection",
    )

    # Add a top-level critical aggregator for mandatory criteria (location/evidence, capacity, culture)
    mandatory = evaluator.add_parallel(
        id="mandatory_constraints",
        desc="All mandatory constraints must be satisfied with supporting evidence.",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_location_and_eclipse_nodes(evaluator, mandatory, extracted)
    await build_city_capacity_nodes(evaluator, mandatory, extracted)
    await build_cultural_nodes(evaluator, mandatory, extracted)
    await build_tie_break_nodes(evaluator, root, extracted)  # tie-break kept outside mandatory

    # Record some custom info for convenience
    evaluator.add_custom_info(
        {
            "selected_city": extracted.city_name,
            "reported_totality_duration": extracted.totality_duration_text,
            "airport_name": extracted.airport_name,
            "hotel_count_reported": extracted.hotel_count_text,
            "tie_break_asserted": bool(extracted.tie_break_asserted),
        },
        info_type="summary",
        info_name="parsed_answer_summary",
    )

    # Return standard summary
    return evaluator.get_summary()