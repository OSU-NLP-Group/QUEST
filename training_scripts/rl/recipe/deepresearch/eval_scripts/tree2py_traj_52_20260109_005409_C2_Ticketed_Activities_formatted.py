import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "msg_nyc_concert_venue"
TASK_DESCRIPTION = """
Identify the major indoor concert venue in New York City that is operated by Madison Square Garden Entertainment and has a seating capacity between 5,900 and 6,000 seats. Provide the venue name and its exact seating capacity.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured info extracted from the agent's answer."""
    venue_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract the venue information mentioned in the answer.

    Required fields:
    1) venue_name: The name of the venue (e.g., "Radio City Music Hall").
    2) seating_capacity: The exact seating capacity stated in the answer (e.g., "5,960" or "5960 seats").
       - Extract the number as it appears; if multiple capacities are mentioned, extract the one most clearly identified as the venue's seating capacity.
       - If only a range or approximation like "around 6,000" is provided, extract that text verbatim.
    3) sources: All URLs cited in the answer that provide information about the venue (official site, Wikipedia, operator page, credible news or reference sites, etc.).
       - Include plain URLs or URLs embedded in markdown links; extract actual URLs only.
       - If no URLs are provided, return an empty list.

    Rules:
    - Do not invent data. If a field is missing in the answer, set it to null (for strings) or [] (for lists).
    - Keep the seating_capacity as a string exactly as stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip()


def parse_capacity_int(capacity_text: Optional[str]) -> Optional[int]:
    """
    Try to parse an integer seating capacity from an arbitrary text like "5,960 seats" or "5960".
    Returns None if parsing fails.
    """
    if not capacity_text:
        return None
    # Extract the largest integer-like token (digits with optional commas)
    matches = re.findall(r"\d{1,3}(?:,\d{3})+|\d+", capacity_text)
    if not matches:
        return None
    # Prefer the last or the one with most digits; often the main number
    best = max(matches, key=lambda x: len(x.replace(",", "")))
    try:
        return int(best.replace(",", ""))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extracted: VenueExtraction
) -> None:
    """
    Build the verification tree per rubric and run verifications.
    """
    venue_name = _normalize_str(extracted.venue_name)
    capacity_text = _normalize_str(extracted.seating_capacity)
    sources = extracted.sources if extracted.sources else []

    # Critical parent node: Correct_Venue_Identified
    correct_node = evaluator.add_parallel(
        id="Correct_Venue_Identified",
        desc="The answer correctly identifies and provides information about a New York City venue meeting all specified requirements.",
        parent=root_node,
        critical=True
    )

    # Critical existence checks
    name_exists = bool(venue_name)
    evaluator.add_custom_node(
        result=name_exists,
        id="Venue_Name_Provided",
        desc="A venue name is provided.",
        parent=correct_node,
        critical=True
    )

    capacity_exists = bool(capacity_text)
    evaluator.add_custom_node(
        result=capacity_exists,
        id="Seating_Capacity_Provided",
        desc="The exact seating capacity is stated.",
        parent=correct_node,
        critical=True
    )

    # Constraints aggregator (critical)
    constraints_node = evaluator.add_parallel(
        id="Constraints_Satisfied",
        desc="The provided venue satisfies all specified constraints.",
        parent=correct_node,
        critical=True
    )

    # Individual constraint leaves
    nyc_node = evaluator.add_leaf(
        id="NYC_Location",
        desc="The venue is located in New York City.",
        parent=constraints_node,
        critical=True
    )
    operator_node = evaluator.add_leaf(
        id="MSG_Entertainment_Operator",
        desc="The venue is operated by Madison Square Garden Entertainment.",
        parent=constraints_node,
        critical=True
    )
    capacity_range_node = evaluator.add_leaf(
        id="Capacity_Range",
        desc="The stated seating capacity is between 5,900 and 6,000 seats (inclusive).",
        parent=constraints_node,
        critical=True
    )
    concert_node = evaluator.add_leaf(
        id="Concert_Suitability",
        desc="The venue is suitable for concerts and live performances.",
        parent=constraints_node,
        critical=True
    )

    # Build claims and run verifications (batch for constraints)
    nyc_claim = f"The venue '{venue_name or 'the venue'}' is located in New York City (NYC)."
    operator_claim = f"The venue '{venue_name or 'the venue'}' is operated by Madison Square Garden Entertainment."
    capacity_claim = (
        f"The seating capacity of '{venue_name or 'the venue'}' is between 5,900 and 6,000 seats inclusive."
    )
    concert_claim = f"The venue '{venue_name or 'the venue'}' is suitable for concerts and live performances."

    claims_and_sources = [
        (
            nyc_claim,
            sources,
            nyc_node,
            "Accept equivalent phrasing like 'New York, NY', 'New York City', 'Manhattan', "
            "or neighborhood references clearly within NYC (e.g., Midtown Manhattan)."
        ),
        (
            operator_claim,
            sources,
            operator_node,
            "Treat 'Madison Square Garden Entertainment', 'MSG Entertainment', "
            "'Madison Square Garden Entertainment Corp.', and similar corporate variants as equivalent."
        ),
        (
            capacity_claim,
            sources,
            capacity_range_node,
            "If the page states a specific capacity like 5,960 or 5,900–6,000, consider it supported. "
            "Minor textual variations (commas, the word 'seats') are acceptable."
        ),
        (
            concert_claim,
            sources,
            concert_node,
            "Look for indications that the venue hosts concerts, music performances, shows, or live events. "
            "Theater-style live performances also qualify."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)

    # Record some helpful custom info
    evaluator.add_custom_info(
        info={
            "extracted_venue_name": venue_name or None,
            "extracted_seating_capacity_text": capacity_text or None,
            "normalized_capacity_int": parse_capacity_int(capacity_text),
            "sources_count": len(sources),
            "sources": sources,
        },
        info_type="extraction_summary",
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
    Evaluate an answer for identifying the specified NYC concert venue operated by MSG Entertainment
    with seating capacity between 5,900 and 6,000.

    Returns a standardized summary dict with the verification tree and final score.
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
        default_model=model,
    )

    # Extract structured venue info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return structured result
    return evaluator.get_summary()