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
TASK_ID = "live_music_performances_2025_2026"
TASK_DESCRIPTION = """Between December 2025 and February 2026, several high-profile live music performances took place at venues of varying sizes across the United States. Identify four specific performances that meet the following criteria:

Performance 1: A halftime show performance by a prominent hip-hop artist at a major NFL stadium in the Upper Midwest during the 2025 holiday season. The venue must be a stadium with a capacity of 30,000 or more.

Performance 2: A concert featuring a married country music couple performing together at a unique golf course venue in Arizona in early February 2025. The venue must have an arena-sized capacity (15,000-25,000).

Performance 3: A solo Las Vegas residency performance by a country music artist at a Roman-themed theater in early February 2025. The venue must be a theater with a capacity under 10,000.

Performance 4: A Mardi Gras celebration performance featuring a married country music couple at a major indoor stadium in Louisiana in February 2026. The venue must be a stadium with a capacity of 70,000 or more.

For each performance, provide:
- The performer(s) name(s)
- The exact venue name and location (city, state)
- The specific date of the performance
- The official event or residency name (if applicable)
- The venue's seating capacity and its category (stadium, arena, or theater)
- Any notable additional details (special guests, performance times, or event-specific information)
- Reference URL(s) supporting the information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Perf1Extraction(BaseModel):
    performer: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    date: Optional[str] = None
    event_type: Optional[str] = None  # e.g., "NFL halftime show"
    capacity_value: Optional[str] = None  # leave as string for flexibility
    capacity_category: Optional[str] = None  # expected: "stadium"
    special_guests: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class Perf2Extraction(BaseModel):
    performers: List[str] = Field(default_factory=list)  # married country couple
    venue_name: Optional[str] = None  # include hole number if given (e.g., "TPC Scottsdale 16th hole")
    city: Optional[str] = None
    state: Optional[str] = None
    date: Optional[str] = None
    event_name: Optional[str] = None
    capacity_value: Optional[str] = None
    capacity_category: Optional[str] = None  # e.g., "arena-sized"
    performance_times: List[str] = Field(default_factory=list)  # start times per act
    reference_urls: List[str] = Field(default_factory=list)


class Perf3Extraction(BaseModel):
    performer: Optional[str] = None
    venue_name: Optional[str] = None  # e.g., "The Colosseum"
    casino_hotel: Optional[str] = None  # e.g., "Caesars Palace"
    city: Optional[str] = None  # should be "Las Vegas"
    state: Optional[str] = None  # "NV" or "Nevada"
    date: Optional[str] = None
    residency_name: Optional[str] = None
    capacity_value: Optional[str] = None
    capacity_category: Optional[str] = None  # expected "theater"
    reference_urls: List[str] = Field(default_factory=list)


class Perf4Extraction(BaseModel):
    performers: List[str] = Field(default_factory=list)  # married country couple
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    date: Optional[str] = None
    event_name: Optional[str] = None  # Mardi Gras celebration name
    capacity_value: Optional[str] = None
    capacity_category: Optional[str] = None  # expected "stadium"
    doors_open_time: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_perf1() -> str:
    return """
    Extract details for "Performance 1": A halftime show performance by a hip-hop artist at a major NFL stadium in the Upper Midwest during the 2025 holiday season.
    Return the following fields:
    - performer: the primary hip-hop artist name (string)
    - venue_name: exact stadium name (string)
    - city: city of the venue (string)
    - state: state of the venue (string)
    - date: specific performance date (month day, year) (string)
    - event_type: should state it's an NFL halftime show (string)
    - capacity_value: the venue's seating capacity as stated (string; keep ranges or approximate text)
    - capacity_category: the category for the venue (expected 'stadium') (string)
    - special_guests: list of special guests if mentioned (array of strings; [] if none)
    - reference_urls: list of URLs that support these details (array). Extract actual URLs given (plain or markdown).
    If any field is not present, set it to null or [] accordingly.
    """


def prompt_extract_perf2() -> str:
    return """
    Extract details for "Performance 2": A concert featuring a married country music couple at a unique golf course venue in Arizona in early February 2025.
    Return the following fields:
    - performers: list with both performer names (array of strings)
    - venue_name: specific venue name including hole number if provided (e.g., 'TPC Scottsdale 16th hole') (string)
    - city: city (string)
    - state: state (string)
    - date: specific performance date (month day, year) (string)
    - event_name: official event name (string)
    - capacity_value: venue capacity as stated (string; keep ranges or approximate text)
    - capacity_category: category label used in the answer (e.g., 'arena-sized') (string)
    - performance_times: list of start times per act or approximate times (array of strings; [] if not provided)
    - reference_urls: list of URLs that support these details (array). Extract actual URLs given (plain or markdown).
    If any field is not present, set it to null or [] accordingly.
    """


def prompt_extract_perf3() -> str:
    return """
    Extract details for "Performance 3": A solo Las Vegas residency performance by a country music artist at a Roman-themed theater in early February 2025.
    Return the following fields:
    - performer: the artist's name (string)
    - venue_name: the theater name (e.g., 'The Colosseum') (string)
    - casino_hotel: the casino/hotel (e.g., 'Caesars Palace') (string)
    - city: city (should be Las Vegas) (string)
    - state: state (e.g., NV or Nevada) (string)
    - date: specific performance date (month day, year) in early February 2025 (string)
    - residency_name: the official residency name (string)
    - capacity_value: venue capacity as stated (string)
    - capacity_category: category label used (expected 'theater') (string)
    - reference_urls: list of URLs that support these details (array). Extract actual URLs given (plain or markdown).
    If any field is not present, set it to null or [] accordingly.
    """


def prompt_extract_perf4() -> str:
    return """
    Extract details for "Performance 4": A Mardi Gras celebration performance featuring a married country music couple at a major indoor stadium in Louisiana in February 2026.
    Return the following fields:
    - performers: list with both performer names (array of strings)
    - venue_name: exact stadium name (string)
    - city: city (string)
    - state: state (string)
    - date: specific performance date (month day, year) (string)
    - event_name: event name (string)
    - capacity_value: venue capacity as stated (string)
    - capacity_category: category label used (expected 'stadium') (string)
    - doors_open_time: doors open time if provided (string or null)
    - reference_urls: list of URLs that support these details (array). Extract actual URLs given (plain or markdown).
    If any field is not present, set it to null or [] accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_join_names(names: List[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_performance_1(evaluator: Evaluator, parent_node, data: Perf1Extraction) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_1",
        desc="Performance 1: A halftime show performance by a hip-hop artist at a major NFL stadium in the Upper Midwest during the 2025 holiday season",
        parent=parent_node,
        critical=False
    )

    # Critical: Reference URLs existence first (gates other verifications)
    urls_ok = bool(data.reference_urls)
    evaluator.add_custom_node(
        result=urls_ok,
        id="perf1_reference_urls",
        desc="Reference URL(s) supporting the performance information are provided",
        parent=perf_node,
        critical=True
    )
    sources = _urls_or_empty(data.reference_urls)

    # Non-critical: Special guests existence
    evaluator.add_custom_node(
        result=bool(data.special_guests),
        id="perf1_special_guests",
        desc="At least one special guest performer is identified",
        parent=perf_node,
        critical=False
    )

    # Build leaf nodes
    nodes_and_claims: List[tuple] = []

    n_performer = evaluator.add_leaf(
        id="perf1_performer",
        desc="The primary performer is correctly identified",
        parent=perf_node,
        critical=True
    )
    claim_performer = f"On {data.date or ''}, during an NFL game halftime at {data.venue_name or ''} in {data.city or ''}, {data.state or ''}, the performer was {data.performer or ''}."
    nodes_and_claims.append((claim_performer, sources, n_performer, "Confirm the named performer for the halftime show; allow minor formatting variations in names and date formatting."))

    n_venue_name = evaluator.add_leaf(
        id="perf1_venue_name",
        desc="The exact venue name is provided",
        parent=perf_node,
        critical=True
    )
    claim_venue = f"The halftime show took place at {data.venue_name or ''}."
    nodes_and_claims.append((claim_venue, sources, n_venue_name, "Verify the exact venue name for the event."))

    n_location = evaluator.add_leaf(
        id="perf1_venue_location",
        desc="The city and state of the venue are correctly provided",
        parent=perf_node,
        critical=True
    )
    claim_loc = f"The venue {data.venue_name or ''} is located in {data.city or ''}, {data.state or ''}."
    nodes_and_claims.append((claim_loc, sources, n_location, "Verify city and state for the venue."))

    n_date = evaluator.add_leaf(
        id="perf1_date",
        desc="The specific date of the performance (month, day, year) is provided",
        parent=perf_node,
        critical=True
    )
    claim_date = f"The performance took place on {data.date or ''}."
    nodes_and_claims.append((claim_date, sources, n_date, "Accept minor variations in date formatting that clearly indicate the same date."))

    n_event_type = evaluator.add_leaf(
        id="perf1_event_type",
        desc="The event type (halftime show during NFL game) is correctly identified",
        parent=perf_node,
        critical=True
    )
    claim_event_type = "This performance was a halftime show during an NFL game."
    nodes_and_claims.append((claim_event_type, sources, n_event_type, "Verify that the described event is an NFL game halftime show."))

    n_capacity_val = evaluator.add_leaf(
        id="perf1_capacity_value",
        desc="The stated venue capacity is 30,000 or higher",
        parent=perf_node,
        critical=True
    )
    claim_capacity_val = f"The seating capacity of {data.venue_name or 'the venue'} is at least 30,000."
    nodes_and_claims.append((claim_capacity_val, sources, n_capacity_val, "Use the provided sources to confirm venue capacity; minor variations acceptable if clearly >= 30,000."))

    n_capacity_cat = evaluator.add_leaf(
        id="perf1_capacity_category",
        desc="The venue is correctly categorized as a stadium",
        parent=perf_node,
        critical=True
    )
    claim_capacity_cat = f"The venue {data.venue_name or 'the venue'} is a stadium."
    nodes_and_claims.append((claim_capacity_cat, sources, n_capacity_cat, "Confirm the venue type is a stadium."))

    await evaluator.batch_verify(nodes_and_claims)


async def verify_performance_2(evaluator: Evaluator, parent_node, data: Perf2Extraction) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_2",
        desc="Performance 2: A concert by a married country music couple at a unique golf course venue in Arizona in early February 2025",
        parent=parent_node,
        critical=False
    )

    # Critical: Reference URLs existence first
    urls_ok = bool(data.reference_urls)
    evaluator.add_custom_node(
        result=urls_ok,
        id="perf2_reference_urls",
        desc="Reference URL(s) supporting the performance information are provided",
        parent=perf_node,
        critical=True
    )
    sources = _urls_or_empty(data.reference_urls)

    # Non-critical: performance times existence
    evaluator.add_custom_node(
        result=bool(data.performance_times),
        id="perf2_performance_times",
        desc="The approximate start times for each performer are provided",
        parent=perf_node,
        critical=False
    )

    nodes_and_claims: List[tuple] = []

    n_performers = evaluator.add_leaf(
        id="perf2_performers",
        desc="Both performers (the couple) are correctly identified",
        parent=perf_node,
        critical=True
    )
    duo = _safe_join_names(data.performers)
    claim_performers = f"On {data.date or ''}, at {data.venue_name or ''} in {data.city or ''}, {data.state or ''}, the concert featured {duo} performing together."
    nodes_and_claims.append((claim_performers, sources, n_performers, "Verify that both named country artists performed together at this event; allow minor name formatting variations."))

    n_venue_name = evaluator.add_leaf(
        id="perf2_venue_name",
        desc="The specific venue name including the hole number is provided",
        parent=perf_node,
        critical=True
    )
    claim_venue = f"The venue for the concert was {data.venue_name or ''}."
    nodes_and_claims.append((claim_venue, sources, n_venue_name, "Verify the exact unique golf course venue name, including hole number if applicable."))

    n_location = evaluator.add_leaf(
        id="perf2_venue_location",
        desc="The city and state are correctly provided",
        parent=perf_node,
        critical=True
    )
    claim_loc = f"The venue {data.venue_name or ''} is located in {data.city or ''}, {data.state or ''}."
    nodes_and_claims.append((claim_loc, sources, n_location, "Verify city and state for the venue; it should be in Arizona."))

    n_date = evaluator.add_leaf(
        id="perf2_date",
        desc="The specific date of the performance (month, day, year) is provided",
        parent=perf_node,
        critical=True
    )
    claim_date = f"The concert took place on {data.date or ''}."
    nodes_and_claims.append((claim_date, sources, n_date, "Accept minor variations in date formatting that indicate the same date."))

    n_event_name = evaluator.add_leaf(
        id="perf2_event_name",
        desc="The official event name is provided",
        parent=perf_node,
        critical=True
    )
    claim_event_name = f"The official event was called '{data.event_name or ''}'."
    nodes_and_claims.append((claim_event_name, sources, n_event_name, "Verify the official event title if referenced in the sources."))

    n_capacity_val = evaluator.add_leaf(
        id="perf2_capacity_value",
        desc="The stated venue capacity falls within the 15,000-25,000 range",
        parent=perf_node,
        critical=True
    )
    claim_capacity_val = f"The venue {data.venue_name or 'the venue'} had an arena-sized capacity between 15,000 and 25,000."
    nodes_and_claims.append((claim_capacity_val, sources, n_capacity_val, "Confirm from the sources that capacity lies in 15,000–25,000 (approximate figures acceptable)."))

    n_capacity_cat = evaluator.add_leaf(
        id="perf2_capacity_category",
        desc="The venue is correctly categorized as arena-sized",
        parent=perf_node,
        critical=True
    )
    claim_capacity_cat = f"The venue {data.venue_name or 'the venue'} is appropriately categorized as arena-sized (around 15,000–25,000 capacity)."
    nodes_and_claims.append((claim_capacity_cat, sources, n_capacity_cat, "Given the capacity range and venue setup, confirm that 'arena-sized' is an appropriate categorization."))

    await evaluator.batch_verify(nodes_and_claims)


async def verify_performance_3(evaluator: Evaluator, parent_node, data: Perf3Extraction) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_3",
        desc="Performance 3: A solo Las Vegas residency show by a country music artist at a Roman-themed theater in early February 2025",
        parent=parent_node,
        critical=False
    )

    # Critical: Reference URLs existence first
    urls_ok = bool(data.reference_urls)
    evaluator.add_custom_node(
        result=urls_ok,
        id="perf3_reference_urls",
        desc="Reference URL(s) supporting the performance information are provided",
        parent=perf_node,
        critical=True
    )
    sources = _urls_or_empty(data.reference_urls)

    nodes_and_claims: List[tuple] = []

    n_performer = evaluator.add_leaf(
        id="perf3_performer",
        desc="The performer is correctly identified",
        parent=perf_node,
        critical=True
    )
    claim_performer = f"The Las Vegas residency show on {data.date or ''} featured {data.performer or ''}."
    nodes_and_claims.append((claim_performer, sources, n_performer, "Verify the named performer for the residency performance; allow minor name format variations."))

    n_venue_name = evaluator.add_leaf(
        id="perf3_venue_name",
        desc="The exact venue name is provided",
        parent=perf_node,
        critical=True
    )
    claim_venue = f"The performance took place at {data.venue_name or ''}."
    nodes_and_claims.append((claim_venue, sources, n_venue_name, "Verify the theater name (e.g., The Colosseum)."))

    n_location = evaluator.add_leaf(
        id="perf3_venue_location",
        desc="The specific casino/hotel and city are correctly provided",
        parent=perf_node,
        critical=True
    )
    casino_hotel = data.casino_hotel or ""
    city = data.city or ""
    state = data.state or ""
    claim_loc = f"The venue is at {casino_hotel} in {city}, {state}."
    nodes_and_claims.append((claim_loc, sources, n_location, "Verify the casino/hotel and city for the venue; should be Las Vegas, NV (Nevada)."))

    n_date = evaluator.add_leaf(
        id="perf3_date",
        desc="A specific performance date in early February 2025 is provided",
        parent=perf_node,
        critical=True
    )
    claim_date = f"The residency performance took place on {data.date or ''}."
    nodes_and_claims.append((claim_date, sources, n_date, "Accept minor date format variations; ensure it is an early February 2025 date."))

    n_residency_name = evaluator.add_leaf(
        id="perf3_residency_name",
        desc="The official name of the residency show is provided",
        parent=perf_node,
        critical=True
    )
    claim_residency = f"The residency is officially titled '{data.residency_name or ''}'."
    nodes_and_claims.append((claim_residency, sources, n_residency_name, "Verify the official residency show title as presented in the sources."))

    n_capacity_val = evaluator.add_leaf(
        id="perf3_capacity_value",
        desc="The stated venue capacity is under 10,000",
        parent=perf_node,
        critical=True
    )
    claim_capacity_val = f"The venue {data.venue_name or 'the venue'} seats fewer than 10,000 people."
    nodes_and_claims.append((claim_capacity_val, sources, n_capacity_val, "Confirm from the sources that the seating capacity is < 10,000; minor variation acceptable."))

    n_capacity_cat = evaluator.add_leaf(
        id="perf3_capacity_category",
        desc="The venue is correctly categorized as a theater",
        parent=perf_node,
        critical=True
    )
    claim_capacity_cat = f"The venue {data.venue_name or 'the venue'} is a theater."
    nodes_and_claims.append((claim_capacity_cat, sources, n_capacity_cat, "Confirm that the venue is a theater (Roman-themed)."))

    await evaluator.batch_verify(nodes_and_claims)


async def verify_performance_4(evaluator: Evaluator, parent_node, data: Perf4Extraction) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_4",
        desc="Performance 4: A Mardi Gras celebration performance by a married country music couple at a major indoor stadium in Louisiana in February 2026",
        parent=parent_node,
        critical=False
    )

    # Critical: Reference URLs existence first
    urls_ok = bool(data.reference_urls)
    evaluator.add_custom_node(
        result=urls_ok,
        id="perf4_reference_urls",
        desc="Reference URL(s) supporting the performance information are provided",
        parent=perf_node,
        critical=True
    )
    sources = _urls_or_empty(data.reference_urls)

    # Non-critical: Doors open time existence
    evaluator.add_custom_node(
        result=bool(data.doors_open_time and data.doors_open_time.strip()),
        id="perf4_doors_open_time",
        desc="The time when doors open for the event is provided",
        parent=perf_node,
        critical=False
    )

    nodes_and_claims: List[tuple] = []

    n_performers = evaluator.add_leaf(
        id="perf4_performers",
        desc="Both performers (the couple) are correctly identified",
        parent=perf_node,
        critical=True
    )
    duo = _safe_join_names(data.performers)
    claim_performers = f"At {data.venue_name or ''} in {data.city or ''}, {data.state or ''} on {data.date or ''}, a Mardi Gras celebration performance featured {duo}."
    nodes_and_claims.append((claim_performers, sources, n_performers, "Verify that the married country music couple performed at this Mardi Gras event; allow minor name formatting variations."))

    n_venue_name = evaluator.add_leaf(
        id="perf4_venue_name",
        desc="The exact venue name is provided",
        parent=perf_node,
        critical=True
    )
    claim_venue = f"The performance took place at {data.venue_name or ''}."
    nodes_and_claims.append((claim_venue, sources, n_venue_name, "Verify the exact stadium name for the event."))

    n_location = evaluator.add_leaf(
        id="perf4_venue_location",
        desc="The city and state are correctly provided",
        parent=perf_node,
        critical=True
    )
    claim_loc = f"The venue {data.venue_name or ''} is located in {data.city or ''}, {data.state or ''}."
    nodes_and_claims.append((claim_loc, sources, n_location, "Verify city and state for the stadium (in Louisiana)."))

    n_date = evaluator.add_leaf(
        id="perf4_date",
        desc="The specific date of the performance (month, day, year) is provided",
        parent=perf_node,
        critical=True
    )
    claim_date = f"The Mardi Gras performance took place on {data.date or ''}."
    nodes_and_claims.append((claim_date, sources, n_date, "Accept minor date format variations that clearly indicate the same date."))

    n_event_name = evaluator.add_leaf(
        id="perf4_event_name",
        desc="The official event name is provided",
        parent=perf_node,
        critical=True
    )
    claim_event_name = f"The event was called '{data.event_name or ''}'."
    nodes_and_claims.append((claim_event_name, sources, n_event_name, "Verify the official event name as listed in the sources."))

    n_capacity_val = evaluator.add_leaf(
        id="perf4_capacity_value",
        desc="The stated venue capacity is 70,000 or higher",
        parent=perf_node,
        critical=True
    )
    claim_capacity_val = f"The venue {data.venue_name or 'the venue'} has a seating capacity of at least 70,000."
    nodes_and_claims.append((claim_capacity_val, sources, n_capacity_val, "Use the provided sources to confirm venue capacity; minor variations acceptable if clearly ≥ 70,000."))

    n_capacity_cat = evaluator.add_leaf(
        id="perf4_capacity_category",
        desc="The venue is correctly categorized as a stadium",
        parent=perf_node,
        critical=True
    )
    claim_capacity_cat = f"The venue {data.venue_name or 'the venue'} is a stadium."
    nodes_and_claims.append((claim_capacity_cat, sources, n_capacity_cat, "Confirm the venue type is a stadium."))

    await evaluator.batch_verify(nodes_and_claims)


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
    Evaluate an answer for four specific live music performances between Dec 2025 and Feb 2026.
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
        default_model=model
    )

    # Extract all four performances in parallel
    perf1_task = evaluator.extract(
        prompt=prompt_extract_perf1(),
        template_class=Perf1Extraction,
        extraction_name="performance_1_extraction"
    )
    perf2_task = evaluator.extract(
        prompt=prompt_extract_perf2(),
        template_class=Perf2Extraction,
        extraction_name="performance_2_extraction"
    )
    perf3_task = evaluator.extract(
        prompt=prompt_extract_perf3(),
        template_class=Perf3Extraction,
        extraction_name="performance_3_extraction"
    )
    perf4_task = evaluator.extract(
        prompt=prompt_extract_perf4(),
        template_class=Perf4Extraction,
        extraction_name="performance_4_extraction"
    )

    perf1, perf2, perf3, perf4 = await asyncio.gather(perf1_task, perf2_task, perf3_task, perf4_task)

    # Build verification tree per rubric and verify each performance
    await verify_performance_1(evaluator, root, perf1)
    await verify_performance_2(evaluator, root, perf2)
    await verify_performance_3(evaluator, root, perf3)
    await verify_performance_4(evaluator, root, perf4)

    # Return structured summary
    return evaluator.get_summary()