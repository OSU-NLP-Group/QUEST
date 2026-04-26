import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "greensboro_coliseum_concert_capacity"
TASK_DESCRIPTION = "What is the concert capacity of Greensboro Coliseum in Greensboro, North Carolina?"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueCapacityExtraction(BaseModel):
    """Structured information extracted from the agent's answer."""
    venue_name: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    location_country: Optional[str] = None
    is_indoor_venue: Optional[bool] = None

    capacity_value: Optional[str] = None  # Prefer string to handle ranges/approximate values
    capacity_is_for_concerts: Optional[bool] = None
    capacity_context_text: Optional[str] = None

    capacity_sources: List[str] = Field(default_factory=list)  # URLs cited for concert capacity support


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_capacity() -> str:
    return """
    Extract structured information that the answer explicitly provides about Greensboro Coliseum.
    Only extract what is plainly stated in the answer; do not infer or invent anything.

    Fields to extract:
    1) venue_name: The venue name as stated in the answer (e.g., "Greensboro Coliseum"). If not stated, return null.
    2) location_city: The city stated (e.g., "Greensboro"). If not stated, return null.
    3) location_state: The state stated (e.g., "North Carolina" or "NC"). If not stated, return null.
    4) location_country: The country stated (e.g., "United States", "USA", "US", "U.S."). If not stated, return null.
    5) is_indoor_venue: Return true only if the answer clearly indicates the venue is an indoor venue or indoor arena/coliseum. Otherwise, return false or null if not mentioned.
    6) capacity_value: The capacity number stated for Greensboro Coliseum as presented in the answer, preferably the concert capacity. Preserve formatting such as commas, ranges, or words like "approximately" if present. If not stated, return null.
    7) capacity_is_for_concerts: Return true ONLY if the answer explicitly says the capacity is for concert events or concert configuration (e.g., uses words like "concert capacity" or "concerts"). Return false if it is stated as a generic/max capacity or for other event types. Return null if no capacity context is given.
    8) capacity_context_text: A short snippet (a clause or sentence) from the answer around the capacity that indicates the capacity context (e.g., "concert capacity is ..."). If missing, return null.
    9) capacity_sources: All URLs explicitly cited in the answer to support the stated concert capacity. Include direct URLs and URLs in markdown links. If none, return an empty array.

    IMPORTANT:
    - Do NOT infer indoor venue status unless the answer explicitly says so (e.g., "indoor", "indoor arena/coliseum").
    - For URL extraction, list only actual URLs present in the answer. If the answer mentions a source without a URL, do not include it.
    - If any field is missing, return null (or empty array for capacity_sources).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _contains_digits(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(ch.isdigit() for ch in s)


def _matches_greensboro_nc_us(city: Optional[str], state: Optional[str], country: Optional[str]) -> bool:
    city_ok = _norm(city) == "greensboro"
    st = _norm(state)
    state_ok = st in {"north carolina", "nc", "n.c."}
    co = _norm(country)
    country_ok = co in {"united states", "usa", "us", "u.s.", "u.s.a."}
    return city_ok and state_ok and country_ok


def _is_concert_context(capacity_is_for_concerts: Optional[bool], context_text: Optional[str]) -> bool:
    if capacity_is_for_concerts is True:
        return True
    ctx = _norm(context_text)
    return "concert" in ctx if ctx else False


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    info: VenueCapacityExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and perform checks.
    """

    # Parent node: Greensboro_Coliseum_Concert_Capacity
    parent = evaluator.add_parallel(
        id="Greensboro_Coliseum_Concert_Capacity",
        desc="Evaluate whether the answer correctly provides Greensboro Coliseum's concert capacity and meets all stated constraints.",
        parent=root_node,
        critical=False,
    )

    # 1) Venue_Is_Indoor_Concert_Venue (critical) - check that the answer confirms indoor venue
    # Use custom existence/confirmation based on extraction
    evaluator.add_custom_node(
        result=(info.is_indoor_venue is True),
        id="Venue_Is_Indoor_Concert_Venue",
        desc="Answer confirms Greensboro Coliseum qualifies as an indoor concert venue.",
        parent=parent,
        critical=True,
    )

    # 2) Venue_Located_In_Greensboro_NC_USA (critical) - check that the answer confirms Greensboro, NC, USA
    evaluator.add_custom_node(
        result=_matches_greensboro_nc_us(info.location_city, info.location_state, info.location_country),
        id="Venue_Located_In_Greensboro_NC_USA",
        desc="Answer confirms the venue is located in Greensboro, North Carolina, United States.",
        parent=parent,
        critical=True,
    )

    # 3) Concert_Capacity_Value_Stated (critical) - specific numeric capacity value is stated
    evaluator.add_custom_node(
        result=_contains_digits(info.capacity_value),
        id="Concert_Capacity_Value_Stated",
        desc="Answer states a specific numeric capacity value for Greensboro Coliseum.",
        parent=parent,
        critical=True,
    )

    # 4) Capacity_Explicitly_For_Concert_Events (critical) - capacity is explicitly for concerts
    evaluator.add_custom_node(
        result=_is_concert_context(info.capacity_is_for_concerts, info.capacity_context_text),
        id="Capacity_Explicitly_For_Concert_Events",
        desc="Answer makes clear the stated capacity refers specifically to concert events (not a generic/max/other event configuration capacity).",
        parent=parent,
        critical=True,
    )

    # 5) Capacity_Verified_By_Reliable_Source_Citation (critical) – split into existence + support checks under sequential parent
    source_seq = evaluator.add_sequential(
        id="Capacity_Verified_By_Reliable_Source_Citation",
        desc="Answer provides at least one citation (e.g., URL) to a reliable source that supports the stated concert capacity.",
        parent=parent,
        critical=True,
    )

    # 5.a) Existence of at least one citation URL (critical)
    citation_exists = evaluator.add_custom_node(
        result=(bool(info.capacity_sources) and len(info.capacity_sources) > 0),
        id="Capacity_Citation_Provided",
        desc="At least one citation URL is provided for the stated concert capacity.",
        parent=source_seq,
        critical=True,
    )

    # 5.b) Sources support the stated concert capacity (critical)
    support_leaf = evaluator.add_leaf(
        id="Capacity_Citation_Supports_Claim",
        desc="Cited source(s) support the stated concert capacity for concerts.",
        parent=source_seq,
        critical=True,
    )

    # Construct the claim for verification
    capacity_str = info.capacity_value or ""
    claim = f"The concert capacity of Greensboro Coliseum is {capacity_str}."

    # Verify against provided URLs
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=info.capacity_sources if info.capacity_sources else None,
        additional_instruction=(
            "Verify that the cited webpage(s) explicitly support the stated concert capacity number for Greensboro Coliseum. "
            "Focus on concert configuration specifically—pages that only state generic, basketball, or other event capacities "
            "should not be considered sufficient unless they clearly indicate concert capacity. Allow minor numeric variations "
            "(e.g., rounding or approximate phrasing like 'about'/'approximately')."
        ),
    )


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
    Evaluate an answer for Greensboro Coliseum concert capacity.
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venue_capacity(),
        template_class=VenueCapacityExtraction,
        extraction_name="venue_capacity_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()