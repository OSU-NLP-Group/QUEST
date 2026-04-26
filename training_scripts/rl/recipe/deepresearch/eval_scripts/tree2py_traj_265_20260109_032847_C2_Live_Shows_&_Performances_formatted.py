import asyncio
import logging
import math
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_concert_venue_ada"
TASK_DESCRIPTION = (
    "You are organizing a large concert in New York City with an expected audience of at least 5,500 people. "
    "Identify a suitable concert venue in New York City that can accommodate this audience size. Provide the venue's name, "
    "its total seating capacity, and calculate the minimum number of wheelchair-accessible seats required by ADA standards for a venue of that capacity. "
    "Include reference URLs for both the venue information and the ADA standards used."
)

CAPACITY_THRESHOLD = 5500  # Minimum required attendees for the venue
ADA_RULE_TEXT_EXPECTED = "10 wheelchair-accessible seats per 1,000 total seats"  # Rule per rubric
ADA_RULE_RATIO = 10 / 1000  # 1% of total seats
ADA_ROUNDING_STRATEGY = "ceil"  # Round up to whole-seat minimum per rubric


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Information about the selected venue extracted from the answer."""
    venue_name: Optional[str] = None
    venue_city_or_location_text: Optional[str] = None
    capacity_text: Optional[str] = None
    capacity_number_text: Optional[str] = None
    venue_reference_urls: List[str] = Field(default_factory=list)


class ADAExtraction(BaseModel):
    """Information about ADA references and the answer's computation."""
    ada_reference_urls: List[str] = Field(default_factory=list)
    ada_rule_text_used: Optional[str] = None
    wheelchair_min_text: Optional[str] = None
    wheelchair_min_number_text: Optional[str] = None
    dispersion_statement_present: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return (
        "From the answer, extract details about the selected concert venue. Return the following fields:\n"
        "1. venue_name: The name of the venue chosen for the concert.\n"
        "2. venue_city_or_location_text: Any text indicating the venue's location (e.g., 'New York City', 'Manhattan', 'Brooklyn, NY').\n"
        "3. capacity_text: The total seating capacity as stated in the answer (string as written).\n"
        "4. capacity_number_text: The numeric form of the total seating capacity if present (e.g., '18000', '12,500', '18k'); if not explicitly numeric, return null.\n"
        "5. venue_reference_urls: A list of URLs cited for venue information such as capacity and/or location.\n"
        "If any field is not mentioned, return null for strings and an empty list for URLs. Extract exactly what the answer provides and do not infer."
    )


def prompt_extract_ada() -> str:
    return (
        "From the answer, extract ADA standards references and the computation:\n"
        "1. ada_reference_urls: A list of URLs cited that correspond to the 2010 ADA Standards for Accessible Design or official authoritative material explicitly tied to the 2010 ADA Standards.\n"
        "2. ada_rule_text_used: The ADA rule text (as stated in the answer) used to compute wheelchair-accessible seats, e.g., '10 wheelchair-accessible seats per 1,000 total seats'.\n"
        "3. wheelchair_min_text: The minimum number of wheelchair-accessible seats stated in the answer (string form).\n"
        "4. wheelchair_min_number_text: The numeric form of the minimum number if present (e.g., '180'); otherwise return null.\n"
        "5. dispersion_statement_present: A boolean indicating whether the answer states that wheelchair-accessible seats must be dispersed horizontally and vertically throughout the venue. "
        "Return true only if the answer explicitly states dispersion across horizontal and vertical dimensions (or an equivalent phrasing like across sections and levels).\n"
        "If any field is not mentioned, return null for strings and an empty list for URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_int_from_text(text: Optional[str]) -> Optional[int]:
    """
    Best-effort parse of an integer capacity or count from arbitrary text.
    Handles formats like:
    - '18,200' -> 18200
    - '18k' or '18.5k' -> 18000 or 18500
    - '1.2 million' or '1.2m' -> 1200000
    - Ranges '10,000–12,000' -> returns the max number found (12000)
    Returns None if no reasonable number is found.
    """
    if not text:
        return None
    s = text.lower()

    # Replace unicode en dash or em dash with hyphen for consistency
    s = s.replace("–", "-").replace("—", "-")

    # Find all number + optional unit occurrences
    pattern = re.compile(r"(\d{1,3}(?:[,\.]\d{3})+|\d+(?:\.\d+)?)\s*(k|thousand|m|million)?")
    matches = pattern.findall(s)

    if not matches:
        return None

    candidates: List[int] = []
    for num_str, unit in matches:
        # Normalize thousand separators
        if "," in num_str and "." in num_str:
            # ambiguous separators; remove commas
            num_str = num_str.replace(",", "")
        else:
            num_str = num_str.replace(",", "")

        try:
            val = float(num_str)
        except ValueError:
            continue

        multiplier = 1
        if unit in ("k", "thousand"):
            multiplier = 1000
        elif unit in ("m", "million"):
            multiplier = 1_000_000

        candidates.append(int(round(val * multiplier)))

    if not candidates:
        return None

    # Choose the maximum number (more conservative for ranges)
    return max(candidates)


def _ensure_capacity_number(venue: VenueExtraction) -> Optional[int]:
    """
    Try to obtain a numeric capacity from extraction results.
    Priority: capacity_number_text -> parse; else capacity_text -> parse.
    """
    # First try capacity_number_text
    num = _parse_int_from_text(venue.capacity_number_text)
    if num is not None:
        return num
    # Fall back to capacity_text
    return _parse_int_from_text(venue.capacity_text)


def _parse_min_wheelchair_from_text(ada: ADAExtraction) -> Optional[int]:
    """
    Parse the minimum wheelchair-accessible seats number from ADA extraction.
    Priority: wheelchair_min_number_text -> parse; else wheelchair_min_text -> parse.
    """
    num = _parse_int_from_text(ada.wheelchair_min_number_text)
    if num is not None:
        return num
    return _parse_int_from_text(ada.wheelchair_min_text)


def _compute_ada_minimum(capacity: int) -> int:
    """
    Per rubric rule: 10 wheelchair-accessible seats per 1,000 total seats, rounded up to whole seats.
    Equivalent to ceil(capacity * 0.01).
    """
    return int(math.ceil(capacity * ADA_RULE_RATIO))


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_venue_information(
    evaluator: Evaluator,
    parent_node,
    venue: VenueExtraction,
) -> Tuple[Optional[int], str]:
    """
    Build 'venue_information' parallel critical subtree and run verifications.
    Returns (capacity_num, venue_name_for_context).
    """
    venue_node = evaluator.add_parallel(
        id="venue_information",
        desc="Provide a suitable NYC venue and the required venue details with a supporting citation.",
        parent=parent_node,
        critical=True
    )

    # Leaf: venue_name_provided (existence)
    name_ok = bool(venue.venue_name and venue.venue_name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="venue_name_provided",
        desc="Provides the venue name.",
        parent=venue_node,
        critical=True
    )

    # Leaf: venue_location_nyc (answer states NYC)
    loc_leaf = evaluator.add_leaf(
        id="venue_location_nyc",
        desc="Indicates the venue is located in New York City.",
        parent=venue_node,
        critical=True
    )
    loc_claim = (
        "The answer indicates that the selected venue is located in New York City (NYC). "
        "Mentions such as 'New York City', 'NYC', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', or 'Staten Island' "
        "should be considered as indicating an NYC location."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        additional_instruction="Focus on the answer text only. If the answer states the venue is in NYC or any of its boroughs, mark correct. If the location is absent or outside NYC, mark incorrect."
    )

    # Compute capacity number
    capacity_num = _ensure_capacity_number(venue)

    # Leaf: capacity provided and >= 5,500 (existence + threshold)
    capacity_ok = capacity_num is not None and capacity_num >= CAPACITY_THRESHOLD
    evaluator.add_custom_node(
        result=capacity_ok,
        id="venue_capacity_provided_and_meets_threshold",
        desc="Provides the venue's total seating capacity as a numeric value and it is at least 5,500.",
        parent=venue_node,
        critical=True
    )

    # Leaf: capacity supported by reference URL(s)
    cap_ref_leaf = evaluator.add_leaf(
        id="venue_capacity_reference_url",
        desc="Provides a reference URL that supports the venue capacity claim.",
        parent=venue_node,
        critical=True
    )

    # If no URLs provided, fail immediately (this leaf requires a citation)
    if not venue.venue_reference_urls:
        cap_ref_leaf.score = 0.0
        cap_ref_leaf.status = "failed"
    else:
        # Build a claim to verify against provided URL(s)
        approx_capacity_str = venue.capacity_text or (str(capacity_num) if capacity_num is not None else "the stated capacity")
        venue_name_for_claim = venue.venue_name or "the selected venue"
        cap_claim = (
            f"The referenced webpage(s) explicitly support that {venue_name_for_claim} has a total seating capacity "
            f"of approximately {approx_capacity_str} seats."
        )
        await evaluator.verify(
            claim=cap_claim,
            node=cap_ref_leaf,
            sources=venue.venue_reference_urls,
            additional_instruction=(
                "Check whether the page(s) explicitly mention the venue's seating capacity. "
                "Allow reasonable approximations (e.g., ranges or slight differences due to configuration), "
                "but the capacity must clearly correspond to the selected venue."
            )
        )

    return capacity_num, (venue.venue_name or "the selected venue")


async def build_and_verify_ada_compliance(
    evaluator: Evaluator,
    parent_node,
    capacity_num: Optional[int],
    venue_name_for_context: str,
    ada: ADAExtraction,
) -> None:
    """
    Build 'ada_compliance' parallel critical subtree and run verifications.
    """
    ada_node = evaluator.add_parallel(
        id="ada_compliance",
        desc="Provide ADA (2010) wheelchair-accessible seating requirement computation and supporting citation.",
        parent=parent_node,
        critical=True
    )

    # Leaf: ada_2010_reference_url (requires citation)
    ada_ref_leaf = evaluator.add_leaf(
        id="ada_2010_reference_url",
        desc="Provides a reference URL to the 2010 ADA Standards (or official authoritative material explicitly tied to the 2010 ADA Standards) used for the wheelchair-accessible seating requirement.",
        parent=ada_node,
        critical=True
    )

    if not ada.ada_reference_urls:
        ada_ref_leaf.score = 0.0
        ada_ref_leaf.status = "failed"
    else:
        ada_ref_claim = (
            "This webpage corresponds to, or is an official authoritative material explicitly tied to, "
            "the 2010 ADA Standards for Accessible Design that governs assembly areas and wheelchair seating requirements."
        )
        await evaluator.verify(
            claim=ada_ref_claim,
            node=ada_ref_leaf,
            sources=ada.ada_reference_urls,
            additional_instruction=(
                "Accept ADA.gov pages, DOJ/ADA 2010 Standards PDFs, or official materials directly referencing the 2010 ADA Standards. "
                "Focus on verifying that the cited page is indeed the official 2010 ADA Standards (or clearly tied to them)."
            )
        )

    # Leaf: wheelchair_seat_calculation (math correctness based on provided rule)
    calc_leaf = evaluator.add_leaf(
        id="wheelchair_seat_calculation",
        desc="Correctly calculates the minimum number of wheelchair-accessible seats using the stated rule (10 wheelchair-accessible seats per 1,000 total seats) based on the venue's total seating capacity (with appropriate rounding to a whole-seat minimum).",
        parent=ada_node,
        critical=True
    )

    # Determine expected number and answer-stated number
    stated_min = _parse_min_wheelchair_from_text(ada)
    expected_min = _compute_ada_minimum(capacity_num) if capacity_num is not None else None

    if capacity_num is None or stated_min is None or expected_min is None:
        calc_leaf.score = 0.0
        calc_leaf.status = "failed"
    else:
        calc_claim = (
            f"Given a total seating capacity of {capacity_num} for {venue_name_for_context} and the ADA rule "
            f"'{ADA_RULE_TEXT_EXPECTED}' with rounding up to whole seats, the correct minimum number of "
            f"wheelchair-accessible seats is {expected_min}. "
            f"The answer's computed minimum of {stated_min} matches this correct calculation."
        )
        await evaluator.verify(
            claim=calc_claim,
            node=calc_leaf,
            additional_instruction=(
                "Check the arithmetic carefully: minimum = ceil(total_capacity * 10 / 1000). "
                "If the answer's number equals the computed minimum, mark correct; otherwise, incorrect."
            )
        )

    # Leaf: dispersion_requirement_stated (answer explicitly states dispersion)
    dispersion_leaf = evaluator.add_leaf(
        id="dispersion_requirement_stated",
        desc="States that the required wheelchair-accessible seats must be dispersed horizontally and vertically throughout the venue.",
        parent=ada_node,
        critical=True
    )

    dispersion_claim = (
        "The answer explicitly states that wheelchair-accessible seats must be dispersed both horizontally and vertically "
        "throughout the venue (or an equivalent phrasing such as across sections and levels)."
    )
    await evaluator.verify(
        claim=dispersion_claim,
        node=dispersion_leaf,
        additional_instruction=(
            "Focus only on whether the answer includes this requirement. Accept synonymous phrasing that clearly conveys "
            "dispersion across the venue's horizontal sections and vertical levels."
        )
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for the NYC concert venue and ADA seating requirement task.
    Returns a structured summary dictionary containing the verification tree and final score.
    """
    # Initialize evaluator with sequential root; set root critical per rubric
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
        default_model=model
    )
    # Make root critical to enforce no partial credit if any critical child fails
    root.critical = True

    # Extract venue and ADA information (in parallel)
    venue_extraction_task = evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )
    ada_extraction_task = evaluator.extract(
        prompt=prompt_extract_ada(),
        template_class=ADAExtraction,
        extraction_name="ada_extraction"
    )
    venue_info, ada_info = await asyncio.gather(venue_extraction_task, ada_extraction_task)

    # Build and verify venue information subtree
    capacity_num, venue_name_for_context = await build_and_verify_venue_information(
        evaluator=evaluator,
        parent_node=root,
        venue=venue_info
    )

    # Build and verify ADA compliance subtree
    await build_and_verify_ada_compliance(
        evaluator=evaluator,
        parent_node=root,
        capacity_num=capacity_num,
        venue_name_for_context=venue_name_for_context,
        ada=ada_info
    )

    # Record some computed values as custom info for transparency
    ada_expected = _compute_ada_minimum(capacity_num) if capacity_num is not None else None
    evaluator.add_custom_info(
        info={
            "parsed_capacity_number": capacity_num,
            "computed_ada_minimum_by_rule_10_per_1000": ada_expected,
            "ada_rule_text_expected": ADA_RULE_TEXT_EXPECTED,
            "capacity_threshold_required": CAPACITY_THRESHOLD,
        },
        info_type="derived_values",
        info_name="computed_derived_values"
    )

    # Return final structured summary
    return evaluator.get_summary()