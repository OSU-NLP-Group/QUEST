import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "large_broadway_theaters_dec2025"
TASK_DESCRIPTION = (
    "Identify four large Broadway theaters in New York City that meet the following criteria: "
    "(1) the theater must have a seating capacity of at least 1,700 seats, "
    "(2) the theater must be hosting a musical production (not a play) during December 2025, and "
    "(3) the theater must be a legitimate Broadway venue located in Manhattan's Theater District. "
    "For each theater, provide the theater name, the musical show currently playing, the seating capacity, "
    "and a reference URL from an official or reliable Broadway source that verifies this information."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TheaterEntry(BaseModel):
    theater_name: Optional[str] = None
    show_title: Optional[str] = None
    seating_capacity: Optional[str] = None  # Keep as string; may include commas or ranges
    reference_urls: List[str] = Field(default_factory=list)


class TheatersExtraction(BaseModel):
    theaters: List[TheaterEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theaters() -> str:
    return """
    Extract all theater entries the answer provides that are intended to meet the task criteria.
    For each entry, return the following fields exactly as stated in the answer:

    - theater_name: The full name of the Broadway theater (e.g., "Gershwin Theatre").
    - show_title: The title of the musical the theater is hosting (e.g., "Wicked").
    - seating_capacity: The seating capacity number or text as stated (e.g., "1,933", "about 1,750").
      Do NOT convert to numbers; keep any commas or qualifiers like "about".
    - reference_urls: An array of explicit URLs provided in the answer that are intended to verify this theater's show and/or capacity.
      Only include actual URLs. If URLs are missing, return an empty array.

    IMPORTANT:
    - Preserve the original order of entries from the answer.
    - Include all provided entries; the evaluator will use only the first 4 later.
    - Apply the URL extraction rules: only extract explicit URLs (plaintext or markdown). Do not invent or infer URLs.
    - If a field is missing for an entry, set it to null (or empty array for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _first_k(items: List[TheaterEntry], k: int) -> List[TheaterEntry]:
    if len(items) >= k:
        return items[:k]
    padded = items[:]
    while len(padded) < k:
        padded.append(TheaterEntry())
    return padded


# --------------------------------------------------------------------------- #
# Verification sub-tree for one theater                                       #
# --------------------------------------------------------------------------- #
async def verify_one_theater(
    evaluator: Evaluator,
    parent_node,
    theater: TheaterEntry,
    index: int,
) -> None:
    """
    Build verification sub-tree for a single theater and run checks.
    Mirrors the rubric structure:
    - Theater_X (parallel)
      - Theater_Identification (critical, parallel)
        - Theater_Name_Provided (critical, leaf/custom)
        - Is_Broadway_Theater (critical, leaf, verify by URLs)
      - Capacity_Verification (critical, parallel)
        - Capacity_Stated (critical, custom)
        - Meets_1700_Threshold (critical, leaf, verify by URLs)
      - Show_Requirements (critical, parallel)
        - Show_Name_Provided (critical, custom)
        - Is_Musical (critical, leaf, verify by URLs)
        - December_2025_Performance (critical, leaf, verify by URLs)
      - Reference_URL (critical, leaf, verify by URLs)
    """
    # Top-level theater node
    theater_node = evaluator.add_parallel(
        id=f"Theater_{index}",
        desc=f"{['First','Second','Third','Fourth'][index-1]} large Broadway theater hosting a musical in December 2025",
        parent=parent_node,
        critical=False,
    )

    # Normalize sources to list (can be empty)
    sources = theater.reference_urls or []

    # -------------------- Theater Identification ------------------------- #
    ident_node = evaluator.add_parallel(
        id=f"Theater_{index}_Theater_Identification",
        desc="Valid Broadway theater name and classification",
        parent=theater_node,
        critical=True,
    )

    # Theater_Name_Provided (existence)
    evaluator.add_custom_node(
        result=_safe_nonempty(theater.theater_name),
        id=f"Theater_{index}_Theater_Name_Provided",
        desc="A specific theater name is provided",
        parent=ident_node,
        critical=True,
    )

    # Is_Broadway_Theater (verify via sources)
    is_broadway_leaf = evaluator.add_leaf(
        id=f"Theater_{index}_Is_Broadway_Theater",
        desc="The theater is a legitimate Broadway venue (located in Manhattan's Theater District)",
        parent=ident_node,
        critical=True,
    )
    claim_broadway = (
        f"{theater.theater_name or 'The theater'} is a legitimate Broadway theater located in Manhattan's Theater District."
    )
    await evaluator.verify(
        claim=claim_broadway,
        node=is_broadway_leaf,
        sources=sources,
        additional_instruction=(
            "Judge based on the page content whether the venue is a recognized Broadway house in Manhattan's Theater District. "
            "Accept official theater operator sites (e.g., Shubert, Nederlander, Jujamcyn), the Broadway League, Playbill, "
            "official theater/show websites, Telecharge, or similar reputable Broadway sources."
        ),
    )

    # -------------------- Capacity Verification -------------------------- #
    capacity_node = evaluator.add_parallel(
        id=f"Theater_{index}_Capacity_Verification",
        desc="Theater meets the large venue capacity requirement",
        parent=theater_node,
        critical=True,
    )

    # Capacity_Stated (existence)
    evaluator.add_custom_node(
        result=_safe_nonempty(theater.seating_capacity),
        id=f"Theater_{index}_Capacity_Stated",
        desc="Seating capacity number is provided",
        parent=capacity_node,
        critical=True,
    )

    # Meets_1700_Threshold (verify via sources)
    capacity_leaf = evaluator.add_leaf(
        id=f"Theater_{index}_Meets_1700_Threshold",
        desc="Stated capacity is 1,700 seats or greater",
        parent=capacity_node,
        critical=True,
    )
    cap_text = theater.seating_capacity or "unknown capacity"
    claim_capacity = (
        f"The seating capacity of {theater.theater_name or 'the theater'} is {cap_text}, which is at least 1,700 seats."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=sources,
        additional_instruction=(
            "Use the provided page(s) to check the theater's seating capacity. "
            "Minor discrepancies or approximate wording are acceptable, but the capacity must be reasonably ≥ 1700."
        ),
    )

    # -------------------- Show Requirements ------------------------------ #
    show_node = evaluator.add_parallel(
        id=f"Theater_{index}_Show_Requirements",
        desc="Hosting a musical production in December 2025",
        parent=theater_node,
        critical=True,
    )

    # Show_Name_Provided (existence)
    evaluator.add_custom_node(
        result=_safe_nonempty(theater.show_title),
        id=f"Theater_{index}_Show_Name_Provided",
        desc="A specific show title is provided",
        parent=show_node,
        critical=True,
    )

    # Is_Musical (verify via sources)
    musical_leaf = evaluator.add_leaf(
        id=f"Theater_{index}_Is_Musical",
        desc="The show is classified as a musical (not a play or other type)",
        parent=show_node,
        critical=True,
    )
    claim_musical = f"The show '{theater.show_title or 'the show'}' is a musical."
    await evaluator.verify(
        claim=claim_musical,
        node=musical_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the production type on the page(s). Accept if the show is clearly described as a musical, "
            "even if phrased as 'musical' or 'musical theater'. Do not accept 'play' or other non-musical forms."
        ),
    )

    # December_2025_Performance (verify via sources)
    dec25_leaf = evaluator.add_leaf(
        id=f"Theater_{index}_December_2025_Performance",
        desc="The show is confirmed to be performing during December 2025",
        parent=show_node,
        critical=True,
    )
    claim_dec25 = (
        f"The show '{theater.show_title or 'the show'}' is scheduled to be performing at "
        f"{theater.theater_name or 'the theater'} during December 2025."
    )
    await evaluator.verify(
        claim=claim_dec25,
        node=dec25_leaf,
        sources=sources,
        additional_instruction=(
            "Check the production's run dates, calendar, or schedule on the page(s). "
            "It's sufficient if the page indicates an ongoing run that includes December 2025, "
            "or a calendar/ticketing availability for dates in December 2025."
        ),
    )

    # -------------------- Reference URL ---------------------------------- #
    ref_leaf = evaluator.add_leaf(
        id=f"Theater_{index}_Reference_URL",
        desc="A valid reference URL from an official or reliable Broadway source is provided",
        parent=theater_node,
        critical=True,
    )
    claim_ref = (
        "The provided page(s) are official or reliably authoritative Broadway sources suitable for verifying the theater and show information."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=sources,
        additional_instruction=(
            "Judge whether the URL(s) come from official or widely recognized Broadway sources: "
            "Broadway League, Playbill, official show/theater/operator websites (e.g., Shubert, Jujamcyn, Nederlander), "
            "Telecharge, Ticketmaster (for Broadway shows), etc. If no URL is provided, this should fail."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point: Build the verification tree, extract data from answer,
    and verify four theaters according to the rubric.
    """
    # Initialize evaluator with a parallel root (matches rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four large Broadway theaters (seating capacity ≥1,700) that are hosting musical productions during December 2025",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract theater entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_theaters(),
        template_class=TheatersExtraction,
        extraction_name="theaters_extraction",
    )

    # Record ground truth criteria (informational)
    evaluator.add_ground_truth({
        "required_count": 4,
        "constraints": [
            "Theater capacity ≥ 1,700 seats",
            "Show is a musical (not a play)",
            "Performing during December 2025",
            "Legitimate Broadway venue in Manhattan's Theater District",
            "Provide verifying URL(s) from official or reliable Broadway sources"
        ]
    }, gt_type="task_constraints")

    # Build the rubric tree: top-level node already exists as root
    main_node = evaluator.add_parallel(
        id="Large_Broadway_Theaters_December_2025",
        desc="Identify four large Broadway theaters (seating capacity ≥1,700) that are hosting musical productions during December 2025",
        parent=root,
        critical=False,
    )

    # Use only the first 4 entries; pad if fewer
    theaters4 = _first_k(extracted.theaters, 4)

    # Create subtrees for each theater
    for i, theater in enumerate(theaters4, start=1):
        await verify_one_theater(evaluator, main_node, theater, i)

    # Return summary
    return evaluator.get_summary()