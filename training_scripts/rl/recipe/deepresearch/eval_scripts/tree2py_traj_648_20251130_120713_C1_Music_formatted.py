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
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mccartney_got_back_2025_mid_size_venue"
TASK_DESCRIPTION = (
    "I am planning to attend Paul McCartney's Got Back 2025 North America tour and prefer a mid-size venue. "
    "Identify one venue on the tour that has a seating capacity between 15,000 and 25,000 seats. For the identified venue, "
    "provide the concert date, the official venue name, the complete physical address, and the venue's seating capacity."
)
OFFICIAL_TOUR_URL = "https://www.paulmccartney.com/live/got-back-tour-2025"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueSelection(BaseModel):
    """Single venue selected in the answer."""
    venue_name: Optional[str] = None
    concert_date: Optional[str] = None
    address: Optional[str] = None
    seating_capacity: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_selection() -> str:
    return """
    Extract the single venue the answer identifies for Paul McCartney's Got Back 2025 North America tour.
    If the answer mentions multiple venues, select the first one the answer presents as the choice or recommendation.
    Extract the following fields exactly as presented in the answer (do not infer):
    - venue_name: Official venue name (string). If not provided, null.
    - concert_date: The specific concert date (string). If not provided, null.
    - address: Complete physical address of the venue (string). If not provided, null.
    - seating_capacity: The venue's seating capacity (string exactly as written, e.g., "18,200", "around 20k", etc.). If not provided, null.
    - source_urls: All URLs cited in the answer that are directly about this venue or the tour stop (e.g., the official tour page, venue page, press releases). If no URLs are cited, return an empty array.

    Important rules:
    - Only extract information explicitly present in the answer text.
    - Do not invent or normalize values (e.g., keep the original formatting of capacity).
    - For URLs, include complete URLs. If a URL is missing a protocol, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_capacity_to_int(cap_str: Optional[str]) -> Optional[int]:
    """
    Parse a textual seating capacity into an integer.
    Handles formats like "18,000", "18k", "18.5k", "around 20,000", "16,000–18,000" etc.
    Returns first plausible capacity between 1,000 and 200,000 if found; otherwise None.
    """
    if not cap_str:
        return None

    candidates: List[int] = []

    # Pattern: number with optional commas/decimals, optional suffix (k|thousand)
    pattern = re.compile(r'(?i)(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)[\s\-]*?(k|thousand|m|million)?')
    for m in pattern.finditer(cap_str):
        num_str = m.group(1)
        suffix = m.group(2).lower() if m.group(2) else None
        try:
            # Remove commas for numeric conversion
            val = float(num_str.replace(",", ""))
            if suffix in ("k", "thousand"):
                val *= 1000.0
            elif suffix in ("m", "million"):
                val *= 1_000_000.0
            int_val = int(round(val))
            # Only keep plausible venue capacities
            if 1_000 <= int_val <= 200_000:
                candidates.append(int_val)
        except Exception:
            continue

    # Fallback simple digits grouping if pattern fails
    if not candidates:
        digits = re.findall(r'\d+', cap_str)
        for d in digits:
            try:
                iv = int(d)
                if 1_000 <= iv <= 200_000:
                    candidates.append(iv)
            except Exception:
                continue

    return candidates[0] if candidates else None


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_valid_venue_selection_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: VenueSelection
) -> None:
    """
    Build the 'Valid_Venue_Selection' node and all its child leaf checks.
    All children are critical (per rubric).
    """
    valid_node = evaluator.add_parallel(
        id="Valid_Venue_Selection",
        desc="Response identifies one venue on the specified tour meeting the capacity constraint and provides all requested fields.",
        parent=parent_node,
        critical=True
    )

    # Parse capacity for numeric checks
    parsed_capacity = parse_capacity_to_int(extracted.seating_capacity)

    # 1) Official_Venue_Name_Provided (existence)
    evaluator.add_custom_node(
        result=bool(extracted.venue_name and extracted.venue_name.strip()),
        id="Official_Venue_Name_Provided",
        desc="The response provides the official venue name.",
        parent=valid_node,
        critical=True
    )

    # 2) Concert_Date_Provided (existence)
    evaluator.add_custom_node(
        result=bool(extracted.concert_date and extracted.concert_date.strip()),
        id="Concert_Date_Provided",
        desc="The response provides the specific concert date for the identified venue.",
        parent=valid_node,
        critical=True
    )

    # 3) Complete_Physical_Address_Provided (existence)
    evaluator.add_custom_node(
        result=bool(extracted.address and extracted.address.strip()),
        id="Complete_Physical_Address_Provided",
        desc="The response provides the complete physical address of the venue.",
        parent=valid_node,
        critical=True
    )

    # 4) Seating_Capacity_Numeric_Provided (numeric extractability)
    evaluator.add_custom_node(
        result=(parsed_capacity is not None),
        id="Seating_Capacity_Numeric_Provided",
        desc="The response provides the venue's seating capacity as a numerical value.",
        parent=valid_node,
        critical=True
    )

    # 5) Capacity_Requirement (range check)
    evaluator.add_custom_node(
        result=(parsed_capacity is not None and 15000 <= parsed_capacity <= 25000),
        id="Capacity_Requirement",
        desc="The venue's seating capacity is between 15,000 and 25,000 seats inclusive.",
        parent=valid_node,
        critical=True
    )

    # 6) Tour_Participation (verify against official tour page)
    tour_part_leaf = evaluator.add_leaf(
        id="Tour_Participation",
        desc="The identified venue is listed as an official stop on Paul McCartney's Got Back 2025 North America tour per paulmccartney.com/live/got-back-tour-2025.",
        parent=valid_node,
        critical=True
    )

    # Build the claim for verification
    venue_name_for_claim = extracted.venue_name or ""
    claim = f"The official Paul McCartney 'Got Back 2025' North America tour page lists a concert at '{venue_name_for_claim}'."

    add_ins = (
        "Focus on whether the official tour page includes this venue as part of the 2025 North America tour. "
        "Allow reasonable name variants (e.g., corporate suffixes, sponsorship names, abbreviations). "
        f"The answer's provided date is: {extracted.concert_date or 'N/A'}. "
        "The date detail is provided for context but is not strictly required to pass this check."
    )

    await evaluator.verify(
        claim=claim,
        node=tour_part_leaf,
        sources=OFFICIAL_TOUR_URL,
        additional_instruction=add_ins
    )

    # Record helpful custom info for debugging
    evaluator.add_custom_info(
        {
            "parsed_capacity": parsed_capacity,
            "raw_capacity_text": extracted.seating_capacity,
            "official_tour_url_used": OFFICIAL_TOUR_URL
        },
        info_type="debug_info",
        info_name="venue_debug"
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for selecting a mid-size venue (15k–25k capacity) on Paul McCartney's Got Back 2025 North America tour.
    """
    # Initialize evaluator
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

    # Extract the venue selection from the answer
    extracted_venue: VenueSelection = await evaluator.extract(
        prompt=prompt_extract_venue_selection(),
        template_class=VenueSelection,
        extraction_name="venue_selection"
    )

    # Build verification nodes and run checks
    await build_valid_venue_selection_checks(
        evaluator=evaluator,
        parent_node=root,
        extracted=extracted_venue
    )

    # Return final summary
    return evaluator.get_summary()