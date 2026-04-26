import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "co_historic_concert_venue"
TASK_DESCRIPTION = (
    "Identify the historic concert venue located in Colorado near the Denver metropolitan area that meets all of the "
    "following criteria: (1) has been designated as a National Historic Landmark by the National Park Service, "
    "(2) received its NHL designation in the 21st century (after the year 2000), (3) was officially dedicated or "
    "opened between 1940 and 1945, and (4) has a seating capacity between 9,000 and 10,000 people. Provide the "
    "venue's name, its exact dedication date, its seating capacity, the year it received NHL designation, and a "
    "reference URL supporting this information."
)


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    dedication_date: Optional[str] = None  # exact dedication/opening date (full date if available)
    seating_capacity: Optional[str] = None  # keep as string to allow formats like "9,525"
    nhl_designation_year: Optional[str] = None  # keep as string to allow formats like "2015"
    reference_urls: List[str] = Field(default_factory=list)  # one or more supporting URLs


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the following fields from the answer text (do not invent or infer anything not explicitly present):
    - venue_name: The name of the historic concert venue.
    - dedication_date: The exact official dedication or opening date as a single string, preferably including month, day, and year (e.g., "June 15, 1941" or "1941-06-15"). If only a year is provided in the answer, still return exactly what is written.
    - seating_capacity: The venue's seating capacity value as it appears (e.g., "9,525", "9500", "around 9,500").
    - nhl_designation_year: The year (or year-like string) in which it was designated as a National Historic Landmark.
    - reference_urls: A list of all reference URLs explicitly provided in the answer that support the facts. Extract only valid URLs actually present in the answer text.

    Rules:
    - Return null for any field not present in the answer.
    - For reference_urls, include all URLs that the answer cites for this venue.
    - Keep values as strings exactly as written in the answer; do not normalize numbers.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
MONTH_PATTERN = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?" \
                r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"


def looks_like_full_date(value: Optional[str]) -> bool:
    """
    Heuristic check to see if a value looks like a specific calendar date (not just a year), e.g.:
      - "June 15, 1941"
      - "15 June 1941"
      - "1941-06-15"
      - "06/15/1941" or "6/15/41" (allowing 2 or 4-digit years, but prefer 4)
    """
    if not value:
        return False
    s = value.strip()

    # Patterns that include day, month, and year explicitly
    patterns = [
        # Month Day, Year or Month Day Year
        rf"\b{MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}}\b",
        # Day Month Year
        rf"\b\d{{1,2}}\s+{MONTH_PATTERN}\s+\d{{4}}\b",
        # ISO-like YYYY-MM-DD
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        # MM/DD/YYYY or M/D/YYYY
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    ]
    for pat in patterns:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False


def contains_digits(value: Optional[str]) -> bool:
    """Check whether the string contains at least one digit."""
    if not value:
        return False
    return any(ch.isdigit() for ch in value)


def contains_4digit_year(value: Optional[str]) -> bool:
    """Check whether the string contains a 4-digit year."""
    if not value:
        return False
    return re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", value) is not None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: VenueExtraction,
    root: Any,
) -> None:
    """
    Build the verification tree based on the rubric and execute verifications.
    """

    # Top-level node: venue_identification (critical, parallel)
    venue_node = evaluator.add_parallel(
        id="venue_identification",
        desc="Identify a historic concert venue near the Denver metro area that satisfies all listed constraints and provide the requested fields with supporting reference URL(s).",
        parent=root,
        critical=True
    )

    # Child node 1: requested_information_provided (critical, parallel)
    requested_info_node = evaluator.add_parallel(
        id="requested_information_provided",
        desc="All requested output fields are present in the response.",
        parent=venue_node,
        critical=True
    )

    # Existence/format checks (custom nodes, each critical)
    evaluator.add_custom_node(
        result=bool(extracted.venue_name and extracted.venue_name.strip()),
        id="venue_name_provided",
        desc="Venue name is provided.",
        parent=requested_info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=looks_like_full_date(extracted.dedication_date),
        id="exact_dedication_date_provided",
        desc="Exact dedication/opening date is provided (a specific date, not only a year).",
        parent=requested_info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=contains_digits(extracted.seating_capacity),
        id="seating_capacity_value_provided",
        desc="A seating capacity value is provided.",
        parent=requested_info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=contains_4digit_year(extracted.nhl_designation_year),
        id="nhl_designation_year_provided",
        desc="NHL designation year is provided.",
        parent=requested_info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.reference_urls and len(extracted.reference_urls) > 0),
        id="reference_url_provided",
        desc="At least one reference URL is provided to support the stated information.",
        parent=requested_info_node,
        critical=True
    )

    # Child node 2: constraints_satisfied (critical, parallel)
    constraints_node = evaluator.add_parallel(
        id="constraints_satisfied",
        desc="Venue satisfies all stated constraints.",
        parent=venue_node,
        critical=True
    )

    # Create leaf nodes for each constraint check (binary judging via LLM with provided sources)
    # 1) Geographic location near Denver metro area (Colorado)
    geo_node = evaluator.add_leaf(
        id="geographic_location",
        desc="Venue is located in Colorado near the Denver metropolitan area.",
        parent=constraints_node,
        critical=True
    )

    # 2) NHL designation status by NPS
    nhl_status_node = evaluator.add_leaf(
        id="nhl_designation_status",
        desc="Venue is designated as a National Historic Landmark by the National Park Service.",
        parent=constraints_node,
        critical=True
    )

    # 3) NHL designation year is after 2000 (21st century)
    nhl_year_after_2000_node = evaluator.add_leaf(
        id="nhl_designation_year_after_2000",
        desc="NHL designation occurred after the year 2000 (21st century).",
        parent=constraints_node,
        critical=True
    )

    # 4) Dedication/opening date between 1940 and 1945
    dedication_in_range_node = evaluator.add_leaf(
        id="dedication_or_opening_date_in_range",
        desc="Venue was officially dedicated or opened between 1940 and 1945 (inclusive).",
        parent=constraints_node,
        critical=True
    )

    # 5) Capacity between 9,000 and 10,000
    capacity_in_range_node = evaluator.add_leaf(
        id="capacity_in_range",
        desc="Venue seating capacity is between 9,000 and 10,000 people (inclusive).",
        parent=constraints_node,
        critical=True
    )

    # Prepare claims and run batch verification for constraint checks
    venue_name = extracted.venue_name or "the venue"
    dedication_date = extracted.dedication_date or "UNKNOWN DATE"
    nhl_year_text = extracted.nhl_designation_year or "UNKNOWN YEAR"
    capacity_text = extracted.seating_capacity or "UNKNOWN CAPACITY"
    sources = extracted.reference_urls  # list of URLs; may be empty if user didn't provide any

    claims_and_sources = [
        (
            f"The venue named {venue_name} is in Colorado and is near or part of the Denver metropolitan area.",
            sources,
            geo_node,
            "Verify that the cited page(s) indicate the venue is in Colorado and located near Denver (e.g., Morrison, CO; within the Denver metro area; or described as near/just west of Denver). Minor wording variations are acceptable."
        ),
        (
            f"The venue named {venue_name} is designated as a U.S. National Historic Landmark (NHL) by the National Park Service.",
            sources,
            nhl_status_node,
            "Confirm the source states the site is a National Historic Landmark. The NHL program is administered by the National Park Service; accept mentions of 'National Historic Landmark (NHL)' or equivalent phrasing on the page."
        ),
        (
            f"The venue received its National Historic Landmark designation in {nhl_year_text}, and that year is in the 21st century (2001 or later).",
            sources,
            nhl_year_after_2000_node,
            "Confirm from the source(s) the NHL designation year and judge whether it is 2001 or later (strictly after 2000). If the page lists month/year, extract the year."
        ),
        (
            f"The venue was officially dedicated or opened on {dedication_date}, and that date falls between 1940 and 1945 inclusive.",
            sources,
            dedication_in_range_node,
            "Confirm the official dedication/opening date is within Jan 1, 1940 to Dec 31, 1945. If multiple dates are present, prefer those explicitly labeled 'dedicated' or 'opened'; do not use construction or renovation dates."
        ),
        (
            f"The seating capacity of {venue_name} is {capacity_text}, which is between 9,000 and 10,000 inclusive.",
            sources,
            capacity_in_range_node,
            "Confirm the typical or official seating capacity is within [9,000, 10,000]. Accept common variants like 9,500 or 9,525 and minor formatting differences (commas)."
        ),
    ]

    # Execute all constraint verifications (they will auto-skip if prerequisites fail)
    await evaluator.batch_verify(claims_and_sources)


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Colorado historic concert venue task and return the structured result.
    """
    # Initialize evaluator and root
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Build and verify tree based on extracted info
    await build_and_verify_tree(evaluator, extracted, root)

    # Return summary
    return evaluator.get_summary()