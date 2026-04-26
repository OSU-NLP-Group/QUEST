import asyncio
import logging
import math
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "columbus_venue_ada_wheelchair"
TASK_DESCRIPTION = """
Identify a performing arts center or concert hall in Columbus, Ohio that has a seating capacity between 2,500 and 3,000 seats. Once identified, calculate the minimum number of wheelchair-accessible seats that this venue must provide according to the 2010 ADA Standards (Americans with Disabilities Act).
"""


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    """
    Parse the first integer value from a text string. Handles thousands separators and basic ranges.
    Example inputs:
      "2,791 seats" -> 2791
      "2500-3000" -> 2500
      "approx 2800" -> 2800
    Returns None if no plausible integer is found.
    """
    if not text:
        return None
    # Remove commas and spaces in numbers
    cleaned = re.sub(r"[,\s]", "", text)
    m = re.search(r"(\d{3,6})", cleaned)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def compute_ada_min_wheelchair_spaces(capacity: int) -> int:
    """
    Compute minimum required wheelchair spaces per 2010 ADA Standards, Section 221.2.1.1 (Table 221.2.1.1).
    Rules summary:
      - 1 to 25: 1
      - 26 to 50: 2
      - 51 to 150: 4
      - 151 to 300: 5
      - 301 to 500: 6
      - 501 to 5000: 6 + ceil((seats - 500) / 150)
      - Over 5000: 36 + ceil((seats - 5000) / 200)
    """
    if capacity <= 25:
        return 1
    if capacity <= 50:
        return 2
    if capacity <= 150:
        return 4
    if capacity <= 300:
        return 5
    if capacity <= 500:
        return 6
    if capacity <= 5000:
        return 6 + math.ceil((capacity - 500) / 150.0)
    return 36 + math.ceil((capacity - 5000) / 200.0)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_text: Optional[str] = None
    capacity_number: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    type_urls: List[str] = Field(default_factory=list)


class ADACalcExtraction(BaseModel):
    capacity_used_text: Optional[str] = None
    capacity_used_number: Optional[str] = None
    computed_required_wheelchair_seats: Optional[str] = None
    method_summary: Optional[str] = None
    ada_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract details for a single venue used in the answer that is intended to meet the task requirements:
    - The venue should be in Columbus, Ohio.
    - It should be a performing arts center or concert hall that hosts ticketed live performances.
    - It should have a seating capacity between 2,500 and 3,000 seats.

    If multiple venues are mentioned, extract the one that is actually used for the ADA calculation in the answer.

    Return the following fields:
    - venue_name: The full venue name exactly as mentioned.
    - city: The city for the venue (e.g., "Columbus").
    - state: The state for the venue (e.g., "Ohio" or "OH").
    - capacity_text: The capacity statement as written in the answer (e.g., "2,791 seats" or "about 2,800").
    - capacity_number: Extract only the numeric capacity used in the answer if present (digits only, e.g., "2791"). If the answer gives a range or approximation and then chooses a number for calculation, extract that chosen number as digits.
    - location_urls: All URLs cited that support the venue's location in Columbus, Ohio.
    - capacity_urls: All URLs cited that support the seating capacity for the venue.
    - type_urls: All URLs cited that support that the venue is a performing arts center or concert hall hosting ticketed live performances.

    If any field is missing from the answer, return null or an empty array accordingly.
    """


def prompt_extract_ada_calc() -> str:
    return """
    Extract the ADA wheelchair seating calculation information from the answer.

    Return:
    - capacity_used_text: The capacity value used for the ADA calculation as written (e.g., "2,791").
    - capacity_used_number: The numeric digits for the capacity used for the calculation (e.g., "2791"). If the answer used a rounded value, extract that rounded number.
    - computed_required_wheelchair_seats: The final number of required wheelchair-accessible seats stated in the answer (digits only if possible).
    - method_summary: A brief summary (one or two sentences) of how the 2010 ADA Standards were applied according to the answer (if present).
    - ada_reference_urls: All URLs to the 2010 ADA Standards (or official reproductions) that the answer uses to justify the calculation.

    If a field is not present, return null or an empty array.
    """


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_venue_identification(evaluator: Evaluator, root_node, ve: VenueExtraction) -> None:
    """
    Build and verify the 'venue_identification' subtree.
    """
    venue_node = evaluator.add_parallel(
        id="venue_identification",
        desc="Identify the performing arts venue that meets all specified criteria",
        parent=root_node,
        critical=False
    )

    # Location: Columbus, Ohio
    loc_node = evaluator.add_sequential(
        id="location_columbus_ohio",
        desc="The venue is located in Columbus, Ohio",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ve.location_urls),
        id="location_urls_provided",
        desc="Reference URL(s) provided for venue location",
        parent=loc_node,
        critical=True
    )

    loc_leaf = evaluator.add_leaf(
        id="location_reference_url",
        desc="Provide reference URL confirming the venue's location in Columbus, Ohio",
        parent=loc_node,
        critical=True
    )
    venue_name_disp = ve.venue_name or "the venue"
    loc_claim = f"{venue_name_disp} is located in Columbus, Ohio."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=ve.location_urls,
        additional_instruction="Verify that the URL clearly indicates the venue is in Columbus, Ohio (city and state)."
    )

    # Capacity: 2,500 to 3,000
    cap_node = evaluator.add_sequential(
        id="capacity_2500_to_3000",
        desc="The venue has a seating capacity between 2,500 and 3,000 seats",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ve.capacity_urls),
        id="capacity_urls_provided",
        desc="Reference URL(s) provided for seating capacity",
        parent=cap_node,
        critical=True
    )

    cap_leaf = evaluator.add_leaf(
        id="capacity_reference_url",
        desc="Provide reference URL confirming the venue's seating capacity",
        parent=cap_node,
        critical=True
    )
    cap_claim = f"The seating capacity of {venue_name_disp} is between 2,500 and 3,000 seats."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=ve.capacity_urls,
        additional_instruction="Accept exact capacities within [2500, 3000]. If multiple capacities are given (e.g., seating vs. standing), use the main seated capacity."
    )

    # Venue Type: performing arts center or concert hall hosting ticketed live performances
    type_node = evaluator.add_sequential(
        id="performing_arts_type",
        desc="The venue is a performing arts center or concert hall that hosts ticketed live performances",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ve.type_urls),
        id="type_urls_provided",
        desc="Reference URL(s) provided for venue type and events",
        parent=type_node,
        critical=True
    )

    type_leaf = evaluator.add_leaf(
        id="venue_type_reference_url",
        desc="Provide reference URL confirming the venue type and events hosted",
        parent=type_node,
        critical=True
    )
    type_claim = f"{venue_name_disp} is a performing arts center or concert hall that hosts ticketed live performances."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=ve.type_urls,
        additional_instruction="Verify that the venue is a performing arts center or concert hall and that it hosts ticketed live performances (e.g., concerts, theater, dance)."
    )


async def build_ada_calculation(evaluator: Evaluator, root_node, ve: VenueExtraction, ada: ADACalcExtraction) -> None:
    """
    Build and verify the ADA calculation subtree.
    """
    ada_node = evaluator.add_sequential(
        id="ada_wheelchair_seats_calculation",
        desc="Calculate the minimum number of wheelchair-accessible seats required by the 2010 ADA Standards",
        parent=root_node,
        critical=False
    )

    # Parse capacities and computed result from extractions
    venue_capacity = None
    for candidate in [ve.capacity_number, ve.capacity_text]:
        venue_capacity = parse_first_int(candidate)
        if venue_capacity is not None:
            break

    calc_capacity = None
    for candidate in [ada.capacity_used_number, ada.capacity_used_text]:
        calc_capacity = parse_first_int(candidate)
        if calc_capacity is not None:
            break

    stated_result = parse_first_int(ada.computed_required_wheelchair_seats)

    # Add a custom info block for debugging
    evaluator.add_custom_info(
        {
            "parsed_venue_capacity": venue_capacity,
            "parsed_capacity_used_in_calc": calc_capacity,
            "stated_required_wheelchair_spaces": stated_result
        },
        info_type="parsed_numbers",
        info_name="extracted_numeric_values"
    )

    # correct_venue_capacity_used (sequential)
    cap_used_node = evaluator.add_sequential(
        id="correct_venue_capacity_used",
        desc="The calculation uses the actual seating capacity of the identified venue",
        parent=ada_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(venue_capacity is not None),
        id="venue_capacity_parsed",
        desc="Venue seating capacity parsed from the answer",
        parent=cap_used_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(calc_capacity is not None),
        id="calc_capacity_parsed",
        desc="Capacity used for ADA calculation parsed from the answer",
        parent=cap_used_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(venue_capacity is not None and calc_capacity is not None and venue_capacity == calc_capacity),
        id="capacity_used_matches_identified",
        desc="Capacity used in calculation matches the venue's capacity",
        parent=cap_used_node,
        critical=True
    )

    # ada_formula_correctly_applied (sequential)
    formula_node = evaluator.add_sequential(
        id="ada_formula_correctly_applied",
        desc="The 2010 ADA Standards (sections 221 and 221.2) formula is correctly applied based on venue capacity",
        parent=cap_used_node,
        critical=True
    )

    # Compute expected count per ADA; only if we have capacity
    expected_count = compute_ada_min_wheelchair_spaces(calc_capacity) if calc_capacity is not None else None

    evaluator.add_custom_info(
        {
            "expected_required_wheelchair_spaces": expected_count
        },
        info_type="ada_expected",
        info_name="computed_expected_wheelchair_spaces"
    )

    evaluator.add_custom_node(
        result=(expected_count is not None and stated_result is not None and expected_count == stated_result),
        id="formula_application_matches_expected",
        desc="Computed accessible seats in the answer match the ADA table-based computation",
        parent=formula_node,
        critical=True
    )

    # correct_numerical_result (sequential)
    numerical_node = evaluator.add_sequential(
        id="correct_numerical_result",
        desc="The correct number of required wheelchair-accessible seats is stated",
        parent=formula_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(stated_result is not None),
        id="numerical_result_stated",
        desc="Answer states a numeric result for required wheelchair-accessible seats",
        parent=numerical_node,
        critical=True
    )

    ada_leaf = evaluator.add_leaf(
        id="ada_standards_reference_url",
        desc="Provide reference URL for the 2010 ADA Standards that supports the calculation method",
        parent=numerical_node,
        critical=True
    )
    # Build a general formula claim to validate against ADA sources
    formula_claim = (
        "The 2010 ADA Standards (Section 221.2.1.1) specify wheelchair spaces for assembly areas as follows: "
        "1–25 seats: 1; 26–50: 2; 51–150: 4; 151–300: 5; 301–500: 6; "
        "501–5000: 6 plus 1 for each 150, or fraction thereof, over 500; "
        "over 5000: 36 plus 1 for each 200, or fraction thereof, over 5000."
    )
    await evaluator.verify(
        claim=formula_claim,
        node=ada_leaf,
        sources=ada.ada_reference_urls,
        additional_instruction="Verify that the provided URL(s) correspond to the 2010 ADA Standards (e.g., ADA.gov or U.S. Access Board) or faithful reproductions, and that they include the counts specified in Table 221.2.1.1."
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
    Evaluate an answer for the Columbus venue ADA wheelchair seating task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # First identify a valid venue, then validate ADA computation
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

    # Extract venue info and ADA calculation info concurrently
    venue_extraction_task = evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )
    ada_extraction_task = evaluator.extract(
        prompt=prompt_extract_ada_calc(),
        template_class=ADACalcExtraction,
        extraction_name="ada_calc_extraction"
    )
    ve, ada = await asyncio.gather(venue_extraction_task, ada_extraction_task)

    # Build and verify venue identification subtree
    await build_venue_identification(evaluator, root, ve)

    # Build and verify ADA calculation subtree
    await build_ada_calculation(evaluator, root, ve, ada)

    # Return the evaluation summary
    return evaluator.get_summary()