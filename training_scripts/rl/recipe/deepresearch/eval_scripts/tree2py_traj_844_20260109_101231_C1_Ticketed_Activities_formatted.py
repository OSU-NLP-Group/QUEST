import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_major_concert_arena"
TASK_DESCRIPTION = (
    "Identify a major indoor concert arena in New York City that has a seating capacity of at least 18,000 for concerts "
    "and provides wheelchair accessible seating. Provide the venue name, its concert seating capacity, and a reference "
    "URL confirming this information."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """
    Information extracted from the agent's answer.
    - venue_name: the selected venue name (a single venue)
    - concert_capacity: the concert seating capacity as stated in the answer (text, keep as-is)
    - wheelchair_accessibility: the snippet/statement from the answer indicating wheelchair accessible seating (text; keep as-is)
    - reference_urls: all URLs cited in the answer that are meant to substantiate the claims
    """
    venue_name: Optional[str] = None
    concert_capacity: Optional[str] = None
    wheelchair_accessibility: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract exactly one venue (the first or primary one if multiple are mentioned) from the answer that the responder claims meets the task.
    Return:
    - venue_name: The name of the chosen venue.
    - concert_capacity: The concert seating capacity as explicitly stated in the answer text (keep it verbatim, e.g., "20,000 for concerts", "up to 19,500 for concerts").
    - wheelchair_accessibility: The exact sentence or short phrase from the answer indicating that the venue provides wheelchair-accessible seating / ADA seating (if present). If not mentioned, return null.
    - reference_urls: All URLs cited in the answer intended as references for this venue (e.g., official site, venue page, Wikipedia, accessibility page). Follow the SPECIAL RULES FOR URL SOURCES EXTRACTION from the system.
    
    Rules:
    - Do NOT infer or create any data. Only extract what is explicitly present in the answer.
    - If the answer lists multiple venues, pick the first one that the answer uses to satisfy the task and extract only that venue's details.
    - Keep concert_capacity as text (do not parse into a number).
    - Only include URLs that are explicitly present in the answer (plain URLs or markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _safe_name(name: Optional[str]) -> str:
    return name if _nonempty(name) else "the venue"


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_for_venue(evaluator: Evaluator, parent_node, extracted: VenueExtraction) -> None:
    """
    Build verification nodes for the NYC major concert arena task.
    """
    venue_name = extracted.venue_name or ""
    urls = extracted.reference_urls or []
    capacity_text = extracted.concert_capacity or ""
    access_text = extracted.wheelchair_accessibility or ""

    # ---------------- Major node (critical, parallel) --------------------
    major_node = evaluator.add_parallel(
        id="Major_Concert_Arena_NYC",
        desc="Identify one qualifying major indoor concert arena in New York City meeting capacity and accessibility constraints, and provide the required fields and citation.",
        parent=parent_node,
        critical=True
    )

    # ---------------- Venue name provided (critical leaf via custom) -----
    evaluator.add_custom_node(
        result=_nonempty(venue_name),
        id="Venue_Name_Provided",
        desc="The response provides the venue name.",
        parent=major_node,
        critical=True
    )

    # ---------------- Venue eligibility (critical parallel) --------------
    eligibility_node = evaluator.add_parallel(
        id="Venue_Eligibility",
        desc="The venue meets the location/type/suitability eligibility constraints.",
        parent=major_node,
        critical=True
    )

    # Indoor arena in NYC
    indoor_nyc_leaf = evaluator.add_leaf(
        id="Indoor_Arena_Located_In_NYC",
        desc="The venue is an indoor arena located in New York City.",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{_safe_name(venue_name)}' is an indoor arena located in New York City (i.e., within one of the five NYC boroughs).",
        node=indoor_nyc_leaf,
        sources=urls,
        additional_instruction=(
            "Use the provided webpage(s) to verify both aspects: (1) it is an indoor arena (not an open-air stadium or amphitheater), "
            "(2) it is located within New York City (Manhattan, Brooklyn, Queens, The Bronx, or Staten Island). "
            "If the webpages are irrelevant or do not confirm both, mark as not supported."
        )
    )

    # Suitable for major concert events
    major_concert_leaf = evaluator.add_leaf(
        id="Suitable_For_Major_Concert_Events",
        desc="The venue is suitable for hosting major concert events (i.e., is presented/recognized as a major concert arena/venue rather than a small/local-only space).",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{_safe_name(venue_name)}' is recognized as a major concert venue/arena suitable for hosting large-scale concert events by major artists.",
        node=major_concert_leaf,
        sources=urls,
        additional_instruction=(
            "Look for indications such as being an 'arena', hosting major tours, or being described as a primary/large concert venue. "
            "Small clubs, local bars, or minor theaters should not qualify. Use the provided webpage(s) to make this determination."
        )
    )

    # ---------------- Concert capacity at least 18,000 (critical parallel) ------
    capacity_parent = evaluator.add_parallel(
        id="Concert_Capacity_At_Least_18000",
        desc="The response states the venue's concert seating capacity, and it is at least 18,000 (capacity specifically for concerts, not just a generic/other-event capacity).",
        parent=major_node,
        critical=True
    )

    # The answer states some concert capacity (existence check)
    evaluator.add_custom_node(
        result=_nonempty(capacity_text),
        id="Capacity_Stated_In_Answer",
        desc="The answer states a concert seating capacity value.",
        parent=capacity_parent,
        critical=True
    )

    # The answer indicates the concert seating capacity is at least 18,000
    capacity_le_18000_leaf = evaluator.add_leaf(
        id="Capacity_AtLeast_18000_According_to_Answer",
        desc="According to the answer text, the concert seating capacity is at least 18,000.",
        parent=capacity_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the answer text, the concert seating capacity for '{_safe_name(venue_name)}' is at least 18,000.",
        node=capacity_le_18000_leaf,
        sources=None,
        additional_instruction=(
            "Judge only based on the provided answer text. Interpret formats like 'approximately 20,000', 'around 19,500', or 'up to 20,000' as appropriate. "
            "If the answer gives a specific number or range that is less than 18,000, or no clear concert capacity threshold is given, mark as incorrect."
        )
    )

    # The answer explicitly frames this capacity as 'concert' capacity (not generic or other sport)
    capacity_concert_specific_leaf = evaluator.add_leaf(
        id="Capacity_ConcertSpecific_In_Answer",
        desc="According to the answer text, the capacity mentioned is specifically for concerts.",
        parent=capacity_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the answer text, the capacity stated for '{_safe_name(venue_name)}' is explicitly described as concert seating capacity (not just generic or sport-specific).",
        node=capacity_concert_specific_leaf,
        sources=None,
        additional_instruction=(
            "Only evaluate the answer text. Look for phrases like 'for concerts', 'concert capacity', or equivalent. "
            "If the answer only mentions basketball/hockey capacity or generic capacity without stating it is for concerts, mark as incorrect."
        )
    )

    # ---------------- Wheelchair accessible seating stated in the answer (critical leaf) ----
    wheelchair_leaf = evaluator.add_leaf(
        id="Wheelchair_Accessible_Seating_Provided",
        desc="The response states that the venue provides wheelchair accessible seating (ADA-compliant accessibility as required by the constraints).",
        parent=major_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the answer text, it is stated that '{_safe_name(venue_name)}' provides wheelchair-accessible seating (ADA seating).",
        node=wheelchair_leaf,
        sources=None,
        additional_instruction=(
            "Evaluate based solely on the answer text. Accept phrases such as 'wheelchair accessible seating', 'ADA seating', or 'accessible seating'. "
            "If the answer does not clearly state accessibility, mark as incorrect."
        )
    )

    # ---------------- Reference URL(s) confirm claims (critical parallel) ------
    refs_parent = evaluator.add_parallel(
        id="Reference_URL_Confirms_Claims",
        desc="The response provides at least one valid reference URL, and the cited source(s) substantiate the stated concert seating capacity and the availability of wheelchair accessible seating for the named venue.",
        parent=major_node,
        critical=True
    )

    # Has at least one URL in the answer
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="Has_Valid_Reference_URL",
        desc="The answer includes at least one valid reference URL.",
        parent=refs_parent,
        critical=True
    )

    # Sources substantiate concert capacity >= 18,000 for concerts
    refs_capacity_leaf = evaluator.add_leaf(
        id="Sources_Substantiate_Concert_Capacity",
        desc="Cited source(s) substantiate that the concert seating capacity is at least 18,000.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited source(s) show that '{_safe_name(venue_name)}' has a concert seating capacity of at least 18,000.",
        node=refs_capacity_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit 'concert' capacity on the provided webpage(s). If multiple capacities are listed (e.g., basketball/hockey vs concerts), "
            "ensure the concert configuration is at least 18,000. If the pages do not explicitly indicate 'concert' capacity ≥ 18,000, mark as not supported."
        )
    )

    # Sources substantiate wheelchair-accessible seating availability
    refs_access_leaf = evaluator.add_leaf(
        id="Sources_Substantiate_Wheelchair_Accessibility",
        desc="Cited source(s) substantiate that wheelchair accessible seating is provided at the venue.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited source(s) show that '{_safe_name(venue_name)}' provides wheelchair-accessible seating (ADA seating).",
        node=refs_access_leaf,
        sources=urls,
        additional_instruction=(
            "Look for pages that mention 'wheelchair accessible seating', 'ADA seating', 'accessible seating', or similar language. "
            "Official venue or ticketing accessibility pages are acceptable. If none of the provided pages support this, mark as not supported."
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
    Entry point to evaluate an answer for the NYC major indoor concert arena task.
    Returns a structured summary containing the verification tree and final score.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Build verification nodes
    await build_verification_for_venue(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()