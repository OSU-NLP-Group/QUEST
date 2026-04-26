import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "msg_ada_requirements"
TASK_DESCRIPTION = """
For Madison Square Garden in New York City, determine its official seating capacity and calculate the ADA accessibility requirements that apply to a venue of this size, including: the minimum number of wheelchair-accessible spaces required, the minimum number of companion seats needed, whether wheelchair space dispersion to multiple locations is required, and whether accessible seating must be distributed across all price levels with at least 20% of accessible seats at each price level.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ADAExtraction(BaseModel):
    # Capacity and sources
    capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)

    # Computed or stated ADA requirements
    wheelchair_spaces: Optional[str] = None
    companion_seats: Optional[str] = None

    # Rule statements from the answer (text snippets)
    dispersion_statement: Optional[str] = None
    price_level_distribution_statement: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ada_info() -> str:
    return """
    Extract the following information from the answer text:

    1) capacity: The official seating capacity of Madison Square Garden as stated in the answer, verbatim as a text snippet. If multiple capacities are mentioned for different configurations, extract the one the answer presents as the official/default capacity (the main value used for subsequent ADA calculations). If the answer does not provide any capacity, return null.

    2) capacity_sources: A list of all URLs the answer cites specifically to support the seating capacity. Include only actual URLs (plain or markdown-linked). If none are present, return an empty list.

    3) wheelchair_spaces: The minimum required wheelchair-accessible spaces the answer states (e.g., "209", "at least 209"), as text. If not provided, return null.

    4) companion_seats: The minimum required companion seats the answer states (e.g., "≥ 209" or "209"), as text. If not provided, return null.

    5) dispersion_statement: A short text snippet from the answer describing whether dispersion of wheelchair spaces is required for this venue size and specifying dispersion across multiple locations both horizontally and vertically (if present). If no such statement appears, return null.

    6) price_level_distribution_statement: A short text snippet from the answer regarding distribution of accessible seating across all price levels and the rule that at least 20% of required accessible seating must be available at each price level. If absent, return null.

    SPECIAL RULES FOR URL EXTRACTION:
    - Only include explicit URLs present in the answer. If the answer mentions sources without actual URLs, do not fabricate any links; just leave the list empty.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    """Extract the first integer found in a text (handles commas/periods)."""
    if not text:
        return None
    match = re.search(r"\d[\d,\.]*", text)
    if not match:
        return None
    raw = match.group(0)
    norm = raw.replace(",", "").replace(".", "")
    try:
        return int(norm)
    except Exception:
        return None


def is_official_msg_domain(url: str) -> bool:
    """Heuristic check for official Madison Square Garden / MSG domains."""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    # Known official/owner/operator related domains (heuristic)
    official_domains = {
        "msg.com",              # Madison Square Garden / MSG umbrella
        "thegarden.com",        # Often used branding for Madison Square Garden
        "madisonsquaregarden.com",
        "msgentertainment.com", # Corporate ownership
    }
    # Allow subdomains of these domains
    for dom in official_domains:
        if netloc == dom or netloc.endswith("." + dom):
            return True
    return False


def compute_min_wheelchair_spaces(capacity: Optional[int]) -> Optional[int]:
    """Apply the provided constraint: for venues with >500 seats, 1% of total capacity + 1 (floor 1%)."""
    if capacity is None:
        return None
    if capacity <= 500:
        # The rubric focuses on >500 seats case; MSG is >> 500. If <=500, return None to indicate N/A.
        return None
    return int(capacity * 0.01) + 1


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_capacity_verification(
    evaluator: Evaluator,
    parent_node,
    ex: ADAExtraction,
) -> None:
    """
    Build the Venue Capacity verification subtree.
    """
    cap_node = evaluator.add_parallel(
        id="Venue_Capacity_Verification",
        desc="State Madison Square Garden’s official seating capacity and support it using official source(s).",
        parent=parent_node,
        critical=True,
    )

    # 1) Capacity is stated
    capacity_stated = evaluator.add_custom_node(
        result=bool(ex.capacity and ex.capacity.strip()),
        id="capacity_stated",
        desc="Capacity is stated in the answer",
        parent=cap_node,
        critical=True,
    )

    # 2) Capacity sources provided
    sources_provided = evaluator.add_custom_node(
        result=bool(ex.capacity_sources and len(ex.capacity_sources) > 0),
        id="capacity_sources_provided",
        desc="Capacity supporting sources (URLs) are provided in the answer",
        parent=cap_node,
        critical=True,
    )

    # 3) Capacity supported by cited sources
    cap_supported_leaf = evaluator.add_leaf(
        id="capacity_supported_by_sources",
        desc="The stated capacity is supported by at least one cited source",
        parent=cap_node,
        critical=True,
    )
    claim = f"The official seating capacity of Madison Square Garden is {ex.capacity}."
    await evaluator.verify(
        claim=claim,
        node=cap_supported_leaf,
        sources=ex.capacity_sources,
        additional_instruction=(
            "Verify that at least one cited webpage explicitly supports the exact capacity number stated. "
            "Event-specific capacities are acceptable if the answer claims that configuration as 'official', "
            "but the page must clearly state the same capacity figure."
        ),
    )

    # 4) At least one official source present (heuristic check)
    has_official = any(is_official_msg_domain(u) for u in ex.capacity_sources)
    evaluator.add_custom_node(
        result=has_official,
        id="capacity_official_source_present",
        desc="At least one cited capacity source appears to be an official MSG domain",
        parent=cap_node,
        critical=True,
    )


async def build_wheelchair_calculation_verification(
    evaluator: Evaluator,
    parent_node,
    ex: ADAExtraction,
) -> None:
    """
    Build the wheelchair-accessible spaces calculation subtree.
    """
    wc_node = evaluator.add_parallel(
        id="Wheelchair_Space_Calculation",
        desc="Compute minimum required wheelchair-accessible spaces (1% of total capacity + 1 for >500 seats) based on the stated capacity.",
        parent=parent_node,
        critical=True,
    )

    # Parse capacity and wheelchair spaces
    capacity_int = parse_first_int(ex.capacity)
    stated_wc = parse_first_int(ex.wheelchair_spaces)
    required_wc = compute_min_wheelchair_spaces(capacity_int)

    # 1) Capacity parsed and valid
    evaluator.add_custom_node(
        result=capacity_int is not None and capacity_int > 0,
        id="capacity_parsed_valid",
        desc=f"Capacity parsed from answer is a valid positive integer (parsed={capacity_int})",
        parent=wc_node,
        critical=True,
    )

    # 2) Capacity exceeds 500 (per rule scope)
    evaluator.add_custom_node(
        result=capacity_int is not None and capacity_int > 500,
        id="capacity_exceeds_500",
        desc="Capacity exceeds 500 seats (rule applicability condition)",
        parent=wc_node,
        critical=True,
    )

    # 3) Wheelchair minimum stated in answer
    evaluator.add_custom_node(
        result=bool(ex.wheelchair_spaces and ex.wheelchair_spaces.strip()),
        id="wheelchair_spaces_stated",
        desc="Answer provides a minimum required number of wheelchair-accessible spaces",
        parent=wc_node,
        critical=True,
    )

    # 4) Wheelchair calculation correctness
    evaluator.add_custom_node(
        result=(required_wc is not None and stated_wc is not None and stated_wc == required_wc),
        id="wheelchair_calc_correct",
        desc=f"Wheelchair spaces minimum equals computed 1% of capacity + 1 (computed={required_wc}, stated={stated_wc})",
        parent=wc_node,
        critical=True,
    )

    # 5) Calculation explanation present (simple verification against the answer)
    calc_explained_leaf = evaluator.add_leaf(
        id="wheelchair_calc_explained",
        desc="Answer explicitly shows or explains the calculation '1% of capacity + 1' to derive wheelchair spaces",
        parent=wc_node,
        critical=True,
    )
    calc_claim = (
        f"The answer explicitly explains the calculation '1% of {capacity_int} + 1' "
        f"to derive the minimum wheelchair-accessible spaces ({required_wc})."
    )
    await evaluator.verify(
        claim=calc_claim,
        node=calc_explained_leaf,
        additional_instruction=(
            "Check the answer text only. It should show or describe the formula 1% of capacity plus 1 and apply it to the stated capacity. "
            "Minor wording variations are acceptable as long as the formula and its application are clear."
        ),
    )

    # Record computed values for transparency
    evaluator.add_custom_info(
        info={
            "parsed_capacity": capacity_int,
            "computed_min_wheelchair_spaces": required_wc,
            "stated_wheelchair_spaces": stated_wc,
        },
        info_type="computed_values",
        info_name="wheelchair_computation"
    )


async def build_companion_requirement_verification(
    evaluator: Evaluator,
    parent_node,
    ex: ADAExtraction,
) -> None:
    """
    Build the companion seats requirement subtree.
    """
    comp_node = evaluator.add_parallel(
        id="Companion_Seat_Requirement",
        desc="Compute minimum required companion seats (≥ wheelchair spaces) and verify the answer states this rule.",
        parent=parent_node,
        critical=True,
    )

    capacity_int = parse_first_int(ex.capacity)
    required_wc = compute_min_wheelchair_spaces(capacity_int)
    stated_comp = parse_first_int(ex.companion_seats)

    # 1) Companion seats stated in answer
    evaluator.add_custom_node(
        result=bool(ex.companion_seats and ex.companion_seats.strip()),
        id="companion_seats_stated",
        desc="Answer provides a minimum required number of companion seats",
        parent=comp_node,
        critical=True,
    )

    # 2) Companion seats meet minimum (≥ wheelchair spaces)
    evaluator.add_custom_node(
        result=(required_wc is not None and stated_comp is not None and stated_comp >= required_wc),
        id="companion_meets_minimum",
        desc=f"Companion seats ≥ wheelchair spaces (required_wc={required_wc}, stated_companion={stated_comp})",
        parent=comp_node,
        critical=True,
    )

    # 3) Rule explanation present
    rule_explained_leaf = evaluator.add_leaf(
        id="companion_rule_explained",
        desc="Answer states that at least one adjacent companion seat is required per wheelchair space",
        parent=comp_node,
        critical=True,
    )
    rule_claim = "The answer explicitly states that at least one adjacent companion seat is required for every wheelchair space."
    await evaluator.verify(
        claim=rule_claim,
        node=rule_explained_leaf,
        additional_instruction=(
            "Judge only based on the answer text. Accept equivalent phrasing (e.g., 'one companion seat per wheelchair space')."
        ),
    )

    evaluator.add_custom_info(
        info={
            "computed_min_companion_seats": required_wc,
            "stated_companion_seats": stated_comp,
        },
        info_type="computed_values",
        info_name="companion_computation"
    )


async def build_dispersion_requirement_verification(
    evaluator: Evaluator,
    parent_node,
    ex: ADAExtraction,
) -> None:
    """
    Build the dispersion requirement subtree.
    """
    disp_node = evaluator.add_parallel(
        id="Dispersion_Requirement",
        desc="Determine whether dispersion of wheelchair spaces is required (>300 seats) and that dispersion must be across multiple locations both horizontally and vertically.",
        parent=parent_node,
        critical=True,
    )

    capacity_int = parse_first_int(ex.capacity)

    # 1) Capacity exceeds 300 (rule applicability)
    evaluator.add_custom_node(
        result=capacity_int is not None and capacity_int > 300,
        id="capacity_exceeds_300",
        desc="Capacity exceeds 300 seats (dispersion rule applicability)",
        parent=disp_node,
        critical=True,
    )

    # 2) Dispersion statement present
    evaluator.add_custom_node(
        result=bool(ex.dispersion_statement and ex.dispersion_statement.strip()),
        id="dispersion_statement_provided",
        desc="Answer provides a dispersion requirement statement",
        parent=disp_node,
        critical=True,
    )

    # 3) Dispersion requirement correctness (horizontal and vertical, multiple locations)
    disp_correct_leaf = evaluator.add_leaf(
        id="dispersion_requirement_correct",
        desc="Answer states dispersion to multiple locations both horizontally and vertically is required",
        parent=disp_node,
        critical=True,
    )
    disp_claim = (
        "The answer explicitly states that wheelchair space dispersion is required to multiple locations "
        "both horizontally and vertically for a venue of this size."
    )
    await evaluator.verify(
        claim=disp_claim,
        node=disp_correct_leaf,
        additional_instruction=(
            "Judge solely from the answer text. Look for both 'horizontal' and 'vertical' dispersion language and that dispersion applies across multiple locations."
        ),
    )


async def build_price_level_distribution_verification(
    evaluator: Evaluator,
    parent_node,
    ex: ADAExtraction,
) -> None:
    """
    Build the price level distribution requirement subtree.
    """
    price_node = evaluator.add_parallel(
        id="Price_Level_Distribution",
        desc="Determine that accessible seating must be distributed across all price levels with at least 20% at each price level.",
        parent=parent_node,
        critical=True,
    )

    # 1) Price-level distribution statement present
    evaluator.add_custom_node(
        result=bool(ex.price_level_distribution_statement and ex.price_level_distribution_statement.strip()),
        id="price_level_statement_provided",
        desc="Answer provides a statement regarding distribution across price levels and the 20% rule",
        parent=price_node,
        critical=True,
    )

    # 2) Price-level distribution rule correctness (all price levels + ≥20%)
    price_rule_leaf = evaluator.add_leaf(
        id="price_level_rule_correct",
        desc="Answer states accessible seating must be distributed across all price levels and ≥20% at each level",
        parent=price_node,
        critical=True,
    )
    price_claim = (
        "The answer explicitly states that accessible seating must be distributed across all price levels "
        "and that at least 20% of the required accessible seating must be available at each price level."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_rule_leaf,
        additional_instruction=(
            "Judge solely from the answer text. Both aspects must be present: distribution across all price levels and the 'at least 20%' requirement for each price level."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Madison Square Garden ADA requirements task.
    """
    # Initialize evaluator with a non-critical root, then add a critical sequential analysis node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Use a simple root; main analysis node will be sequential
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

    # Extract structured information from the answer
    ex = await evaluator.extract(
        prompt=prompt_extract_ada_info(),
        template_class=ADAExtraction,
        extraction_name="ada_extraction",
    )

    # Main analysis node (critical and sequential per rubric)
    analysis_node = evaluator.add_sequential(
        id="ADA_Requirements_Analysis",
        desc="Determine Madison Square Garden’s official seating capacity and derive ADA accessibility requirements from the provided constraints.",
        parent=root,
        critical=True,
    )

    # Build sub-verifications in order (sequential)
    await build_capacity_verification(evaluator, analysis_node, ex)
    await build_wheelchair_calculation_verification(evaluator, analysis_node, ex)
    await build_companion_requirement_verification(evaluator, analysis_node, ex)
    await build_dispersion_requirement_verification(evaluator, analysis_node, ex)
    await build_price_level_distribution_verification(evaluator, analysis_node, ex)

    # Return structured summary
    return evaluator.get_summary()