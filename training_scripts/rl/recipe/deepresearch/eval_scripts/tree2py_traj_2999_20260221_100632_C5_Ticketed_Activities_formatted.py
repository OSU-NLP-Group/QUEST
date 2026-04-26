import asyncio
import logging
import math
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "norcal_arenas_ada_eval"
TASK_DESCRIPTION = (
    "A concert promoter is planning a multi-city tour across Northern California and needs to evaluate four major "
    "indoor arena venues as potential tour stops: Oakland Arena (Oakland), SAP Center (San Jose), Chase Center "
    "(San Francisco), and Golden 1 Center (Sacramento).\n\n"
    "For each of these four venues, provide the following information:\n"
    "1. The exact seating capacity for concerts\n"
    "2. The minimum number of wheelchair-accessible seats required by ADA regulations\n"
    "3. The minimum number of companion seats required by ADA regulations\n\n"
    "Use the ADA formula for calculating wheelchair-accessible seating requirements: for venues with 5,001 or more seats, "
    "36 wheelchair spaces are required, plus 1 additional space for each 200 seats (or fraction thereof) over 5,000. "
    "Each wheelchair space requires at least one adjacent companion seat."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    """Structured info for a single venue extracted from the agent's answer."""
    venue_name: Optional[str] = None
    concert_capacity: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)
    wheelchair_spaces: Optional[str] = None
    companion_seats: Optional[str] = None


class VenuesExtraction(BaseModel):
    """Extraction of all four venues."""
    oakland_arena: Optional[VenueInfo] = None
    sap_center: Optional[VenueInfo] = None
    chase_center: Optional[VenueInfo] = None
    golden1_center: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues_info() -> str:
    return (
        "Extract, from the provided answer, the requested information for each of the following venues: "
        "Oakland Arena (Oakland), SAP Center (San Jose), Chase Center (San Francisco), Golden 1 Center (Sacramento).\n\n"
        "For each venue, extract the following fields exactly as presented in the answer:\n"
        "1) venue_name: The venue name as stated (string)\n"
        "2) concert_capacity: The seating capacity for concerts as stated (string; keep any formatting, units, notes)\n"
        "3) capacity_urls: An array of URL(s) the answer cites to support the capacity (only actual URLs mentioned)\n"
        "4) wheelchair_spaces: The minimum number of wheelchair-accessible spaces the answer reports (string; null if not stated)\n"
        "5) companion_seats: The minimum number of companion seats the answer reports (string; null if not stated)\n\n"
        "Return a JSON object with keys: oakland_arena, sap_center, chase_center, golden1_center; each is a VenueInfo object. "
        "If any venue is not mentioned, set its value to null. If a specific field is missing for a venue, set it to null or an empty list (for capacity_urls).\n"
        "Important: Only include URLs explicitly present in the answer text (plain URLs or within markdown links)."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    """Extract the first integer value from text (handles commas, ranges by taking first number)."""
    if not text:
        return None
    # Remove commas and non-digit separators but keep digits
    # Example: "18,064–18,500 (concerts)" -> matches 18064 first
    cleaned = re.sub(r"[^\d]", " ", text)
    match = re.search(r"\d+", cleaned)
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def compute_ada_wheelchair_spaces(capacity: Optional[int]) -> Optional[int]:
    """
    ADA formula per task:
    For venues with 5,001+ seats: 36 wheelchair spaces + 1 additional space for each 200 seats (or fraction thereof) over 5,000.
    That is: 36 + ceil((capacity - 5000)/200).
    Returns None if capacity is missing or < 5001 (formula explicitly given only for 5001+ in task).
    """
    if capacity is None:
        return None
    if capacity <= 5000:
        return None
    # Fraction thereof => ceiling division
    extra = math.ceil((capacity - 5000) / 200.0)
    return 36 + extra


def parse_int_or_none(text: Optional[str]) -> Optional[int]:
    """Parse integer from the agent-reported value."""
    return parse_first_int(text)


def urls_present(urls: Optional[List[str]]) -> bool:
    """Check if at least one syntactically valid URL is present."""
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification logic per venue                                                #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue_key: str,
    venue_label: str,
    venue_info: Optional[VenueInfo],
) -> None:
    """
    Build verification subtree and run checks for a single venue.
    Structure:
      - {venue_key} (parallel, non-critical)
        - {venue_key}_url (critical): existence of capacity source URL(s)
        - {venue_key}_capacity (critical): capacity claim supported by provided URLs
        - {venue_key}_wheelchair (critical): reported wheelchair spaces match ADA formula based on capacity
        - {venue_key}_companion (critical): reported companion seats are at least one per wheelchair space (minimum equals wheelchair spaces)
    """
    venue_node = evaluator.add_parallel(
        id=venue_key,
        desc=f"Complete evaluation of {venue_label} including seating capacity and ADA accessibility requirements",
        parent=parent_node,
        critical=False,
    )

    # Safely unpack fields
    concert_capacity_str = venue_info.concert_capacity if venue_info else None
    capacity_urls = venue_info.capacity_urls if (venue_info and venue_info.capacity_urls) else []
    reported_wheelchair_str = venue_info.wheelchair_spaces if venue_info else None
    reported_companion_str = venue_info.companion_seats if venue_info else None

    capacity_int = parse_first_int(concert_capacity_str)
    expected_wheelchair = compute_ada_wheelchair_spaces(capacity_int)
    reported_wheelchair_int = parse_int_or_none(reported_wheelchair_str)
    reported_companion_int = parse_int_or_none(reported_companion_str)

    # 1) URL existence (critical sibling)
    evaluator.add_custom_node(
        result=urls_present(capacity_urls),
        id=f"{venue_key}_url",
        desc=f"Provide a valid reference URL from an official or reliable source that confirms {venue_label}'s seating capacity",
        parent=venue_node,
        critical=True,
    )

    # 2) Capacity supported by cited URLs (critical)
    capacity_leaf = evaluator.add_leaf(
        id=f"{venue_key}_capacity",
        desc=f"Provide the exact seating capacity for {venue_label} based on reliable source documentation",
        parent=venue_node,
        critical=True,
    )
    capacity_claim = (
        f"The concert seating capacity for {venue_label} is '{concert_capacity_str}'. "
        f"The cited source(s) explicitly support this concert capacity."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=capacity_urls,
        additional_instruction=(
            "Verify that the provided webpage(s) explicitly state the concert seating capacity matching the answer. "
            "If multiple capacities are listed (e.g., basketball vs. concerts), focus on concerts. "
            "Allow minor formatting differences (commas, wording). If the page does not state an explicit concert capacity "
            "matching the claim, mark as not supported."
        ),
    )

    # 3) Wheelchair spaces per ADA formula (critical)
    wheelchair_leaf = evaluator.add_leaf(
        id=f"{venue_key}_wheelchair",
        desc=(
            "Calculate the minimum number of wheelchair-accessible seats required by ADA regulations "
            f"for {venue_label} using the appropriate formula based on the venue's capacity "
            "(if capacity is 5,001 or more seats: 36 spaces plus 1 for each 200 seats or fraction thereof over 5,000)"
        ),
        parent=venue_node,
        critical=True,
    )
    if expected_wheelchair is None:
        wc_claim = (
            f"The stated concert capacity for {venue_label} ('{concert_capacity_str}') does not yield a computable ADA wheelchair "
            "minimum under the provided 5001+ formula context (either missing or ≤5000). Therefore, the reported minimum cannot be validated."
        )
    else:
        wc_claim = (
            f"Given a concert seating capacity of {capacity_int} seats at {venue_label}, the minimum required number of "
            f"wheelchair-accessible spaces under the ADA formula (36 + 1 per 200 seats or fraction over 5,000) is {expected_wheelchair}. "
            f"The answer reports {reported_wheelchair_int} wheelchair-accessible spaces; this should equal {expected_wheelchair} "
            "as the minimum requirement."
        )
    await evaluator.verify(
        claim=wc_claim,
        node=wheelchair_leaf,
        additional_instruction=(
            "Perform a simple check against the ADA formula for 5001+ seats: minimum = 36 + ceil((capacity - 5000) / 200). "
            "Judge the correctness by comparing the answer's reported minimum to the computed minimum. "
            "If the answer expresses 'at least' or a range, consider it correct if the lower bound is ≥ the computed minimum. "
            "If the answer does not state a number, mark as incorrect."
        ),
    )

    # 4) Companion seats minimum (critical)
    companion_leaf = evaluator.add_leaf(
        id=f"{venue_key}_companion",
        desc=(
            "State the minimum number of companion seats required by ADA regulations (at least one companion seat must be provided for each wheelchair space)"
        ),
        parent=venue_node,
        critical=True,
    )
    if expected_wheelchair is None:
        comp_claim = (
            f"Because the minimum wheelchair spaces for {venue_label} could not be computed under the 5001+ ADA formula context "
            "from the provided concert capacity, the minimum companion seats (≥ one per wheelchair space) cannot be validated."
        )
    else:
        comp_claim = (
            f"Under ADA requirements, the minimum number of companion seats for {venue_label} must be at least one per wheelchair space. "
            f"Given the computed minimum wheelchair spaces {expected_wheelchair}, the minimum companion seats is {expected_wheelchair}. "
            f"The answer reports {reported_companion_int} companion seats; this should be at least {expected_wheelchair} "
            "(and for 'minimum' exactly {expected_wheelchair})."
        )
    await evaluator.verify(
        claim=comp_claim,
        node=companion_leaf,
        additional_instruction=(
            "Check the answer: minimum companion seats must be ≥ the minimum wheelchair spaces. "
            "If the answer reports a number smaller than the computed wheelchair minimum, mark incorrect. "
            "If the answer reports a number equal to the computed minimum, mark correct for the 'minimum' requirement. "
            "If no number is reported, mark incorrect."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Northern California arenas ADA accessibility requirements task.
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
        default_model=model,
    )

    # Extract venue info from the answer
    venues_info = await evaluator.extract(
        prompt=prompt_extract_venues_info(),
        template_class=VenuesExtraction,
        extraction_name="venues_info",
    )

    # Add custom info: ADA formula used
    evaluator.add_custom_info(
        info={
            "ada_formula": "For 5,001+ seats: minimum wheelchair spaces = 36 + ceil((capacity - 5000) / 200). "
                           "Minimum companion seats: at least one per wheelchair space.",
        },
        info_type="ada_policy",
        info_name="ada_formula_applied",
    )

    # Build verification subtrees for each venue
    venues: List[Tuple[str, str, Optional[VenueInfo]]] = [
        ("oakland_arena", "Oakland Arena", venues_info.oakland_arena),
        ("sap_center", "SAP Center", venues_info.sap_center),
        ("chase_center", "Chase Center", venues_info.chase_center),
        ("golden1_center", "Golden 1 Center", venues_info.golden1_center),
    ]

    # Create venue nodes and verify
    for key, label, info in venues:
        await verify_venue(evaluator, root, key, label, info)

    # Return structured evaluation summary
    return evaluator.get_summary()