import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_indoor_arena_capacity"
TASK_DESCRIPTION = (
    "Identify one major indoor arena venue in New York City that has a concert seating capacity of 20,000 or more. "
    "Provide the venue name, the exact concert seating capacity, and include a link to either the official venue "
    "website or the venue's Wikipedia page."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    concert_capacity: Optional[str] = None
    reference_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return (
        "From the answer, extract the details for ONE venue. If multiple venues are mentioned, extract the FIRST one.\n"
        "Return a JSON object with these fields:\n"
        "1) venue_name: The venue name exactly as presented.\n"
        "2) concert_capacity: The exact concert seating capacity value stated in the answer (not a range or vague estimate). "
        "If the answer only gives a range, approximation, or a non-concert capacity, return null.\n"
        "3) reference_url: A single URL provided in the answer that serves as a reference for this venue. "
        "Prefer a Wikipedia page for the venue or the venue's official website if provided. If no URL is provided, return null.\n"
        "Do not invent or infer any information; only extract what is explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_exact_capacity_value(cap_str: Optional[str]) -> Tuple[bool, Optional[int]]:
    """
    Determine whether the provided capacity string represents a single, exact integer value (not a range or approximation),
    and parse it into an integer if possible.

    Returns (is_exact, value_or_none)
    """
    if cap_str is None:
        return False, None

    s = cap_str.strip()
    if not s:
        return False, None

    lower = s.lower()

    # Disallow approximations, inequalities, ranges, or vague phrases
    approx_keywords = ["approx", "approximately", "around", "about", "~", "≈", "more than", "less than", "over ", "under ", "up to", "at most", "at least", "+"]
    range_separators = ["-", "–", "—", " to "]
    vague_keywords = ["varies", "variable", "range", "depending", "configurations", "setups"]

    if any(k in lower for k in approx_keywords):
        return False, None
    if any(sep in lower for sep in range_separators):
        return False, None
    if any(k in lower for k in vague_keywords):
        return False, None

    # Extract a single integer-like token
    import re
    numbers = re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{2,6}\b", s)
    if len(numbers) != 1:
        return False, None

    try:
        val = int(numbers[0].replace(",", ""))
    except Exception:
        return False, None

    return True, val


# --------------------------------------------------------------------------- #
# Verification sub-tree construction                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_nyc_arena_tree(
    evaluator: Evaluator,
    extracted: VenueExtraction,
    parent: VerificationNode,
) -> VerificationNode:
    """
    Construct the verification tree for the NYC indoor arena capacity task and perform verifications.
    All children under this node are critical, as the task requires meeting all criteria.
    """
    # Add the main task node (critical, parallel aggregation)
    nyc_node = evaluator.add_parallel(
        id="NYC_Arena_Venue",
        desc="Identify one indoor arena venue in New York City with concert seating capacity ≥ 20,000, and provide name, exact capacity, and an authoritative reference link.",
        parent=parent,
        critical=True,
    )

    # ---------- Custom existence and numeric parsing checks (critical) ----------
    # Venue name must be provided
    name_provided = bool(extracted.venue_name and extracted.venue_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id="Venue_Name_Provided",
        desc="A specific venue name is provided.",
        parent=nyc_node,
        critical=True
    )

    # Reference URL must be provided
    url_provided = bool(extracted.reference_url and extracted.reference_url.strip())
    evaluator.add_custom_node(
        result=url_provided,
        id="Reference_URL_Provided",
        desc="A URL is provided as a reference.",
        parent=nyc_node,
        critical=True
    )

    # Capacity must be an exact integer value (not a range or approximation)
    exact_cap_ok, cap_val = _is_exact_capacity_value(extracted.concert_capacity)
    evaluator.add_custom_node(
        result=exact_cap_ok,
        id="Exact_Concert_Capacity_Provided",
        desc="The exact concert seating capacity is stated as a specific value (not just a vague estimate/range).",
        parent=nyc_node,
        critical=True
    )

    # Capacity must be at least 20,000
    evaluator.add_custom_node(
        result=bool(exact_cap_ok and cap_val is not None and cap_val >= 20000),
        id="Concert_Capacity_At_Least_20000",
        desc="The stated concert seating capacity is 20,000 or more.",
        parent=nyc_node,
        critical=True
    )

    # Record parsed capacity info for transparency
    evaluator.add_custom_info(
        info={
            "venue_name": extracted.venue_name,
            "concert_capacity_raw": extracted.concert_capacity,
            "concert_capacity_parsed_int": cap_val,
            "reference_url": extracted.reference_url
        },
        info_type="parsed_capacity",
        info_name="parsed_capacity_info"
    )

    # ---------- Evidence-based verifications via the reference URL (critical) ----------
    # 1) Located in NYC
    located_node = evaluator.add_leaf(
        id="Located_In_New_York_City",
        desc="The venue is located in New York City.",
        parent=nyc_node,
        critical=True
    )
    located_claim = f"The venue '{extracted.venue_name}' is located within New York City (NYC)."
    await evaluator.verify(
        claim=located_claim,
        node=located_node,
        sources=extracted.reference_url,
        additional_instruction=(
            "Consider 'New York City' to include the five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. "
            "Confirm via the webpage that the venue's location is in NYC."
        ),
    )

    # 2) Is an indoor arena
    indoor_node = evaluator.add_leaf(
        id="Is_Indoor_Arena",
        desc="The venue is an indoor arena (not an outdoor stadium or amphitheater).",
        parent=nyc_node,
        critical=True
    )
    indoor_claim = (
        f"The venue '{extracted.venue_name}' is an indoor arena, meaning an enclosed multi-purpose arena, "
        "not an outdoor stadium or amphitheater."
    )
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=extracted.reference_url,
        additional_instruction=(
            "Use the webpage to determine whether it is an indoor arena. If the page indicates it is an outdoor stadium "
            "or amphitheater, this should be incorrect."
        ),
    )

    # 3) Regularly hosts concerts
    concerts_node = evaluator.add_leaf(
        id="Regularly_Hosts_Concerts",
        desc="The venue regularly hosts concert events.",
        parent=nyc_node,
        critical=True
    )
    concerts_claim = (
        f"The venue '{extracted.venue_name}' regularly hosts concert events as part of its standard operations."
    )
    await evaluator.verify(
        claim=concerts_claim,
        node=concerts_node,
        sources=extracted.reference_url,
        additional_instruction=(
            "Check the webpage for evidence that the venue hosts concerts (e.g., mentions of past or upcoming concerts, "
            "concert-specific sections). 'Regularly' means multiple events, typical of the venue."
        ),
    )

    # 4) Reference type is allowed (official site or Wikipedia)
    ref_type_node = evaluator.add_leaf(
        id="Reference_Is_Allowed_Source_Type",
        desc="The reference URL points to either the official venue website or the venue's Wikipedia page.",
        parent=nyc_node,
        critical=True
    )
    ref_type_claim = (
        f"This URL for '{extracted.venue_name}' is either (a) the venue's official website, or (b) the venue's Wikipedia page."
    )
    await evaluator.verify(
        claim=ref_type_claim,
        node=ref_type_node,
        sources=extracted.reference_url,
        additional_instruction=(
            "Allowed sources:\n"
            "- Official venue website (the venue's own domain / clearly labeled official site).\n"
            "- Wikipedia page specifically about this venue (typically en.wikipedia.org).\n"
            "If the URL is a different third-party site (news, directory, ticket aggregator, etc.), it is not allowed."
        ),
    )

    # 5) Reference supports both the venue identity and the exact concert capacity value
    ref_support_node = evaluator.add_leaf(
        id="Reference_Supports_Name_And_Capacity",
        desc="The provided reference supports/indicates the venue identity and the concert seating capacity value used in the answer.",
        parent=nyc_node,
        critical=True
    )
    ref_support_claim = (
        f"The webpage clearly identifies the venue as '{extracted.venue_name}' and explicitly states that the concert seating "
        f"capacity is {extracted.concert_capacity}."
    )
    await evaluator.verify(
        claim=ref_support_claim,
        node=ref_support_node,
        sources=extracted.reference_url,
        additional_instruction=(
            "The capacity must be the concert seating capacity explicitly. If the page only provides capacities for sports "
            "configurations (e.g., basketball/hockey) and does not explicitly state the concert capacity, treat it as not supported."
        ),
    )

    return nyc_node


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
    Evaluate an answer for the NYC indoor arena concert capacity task.
    Builds a critical parallel verification tree and verifies all criteria.
    """
    # Initialize evaluator (framework root is non-critical; we add our own critical parent node)
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
    extracted_venue = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_nyc_arena_tree(evaluator, extracted_venue, root)

    # Return the structured summary including the verification tree and final score
    return evaluator.get_summary()