import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nh_thanksgiving_grocery_2025"
TASK_DESCRIPTION = """
On Thanksgiving Day 2025 (Thursday, November 27), if you are in New Hampshire and need to do grocery shopping between 1:00 PM and 2:30 PM, which major grocery store chain can you visit, and what are its operating hours on that day?
"""

THANKSGIVING_DAY_STR = "Thursday, November 27, 2025"
REQUIRED_WINDOW_TEXT = "1:00 PM–2:30 PM"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreCandidate(BaseModel):
    """One candidate store chain mentioned in the answer."""
    name: Optional[str] = None
    thanksgiving_opening_time: Optional[str] = None  # e.g., "6 AM", "7:00 a.m."
    thanksgiving_closing_time: Optional[str] = None  # e.g., "3 PM", "2:30 p.m."
    thanksgiving_hours_text: Optional[str] = None    # free-form hours string for Thanksgiving 2025
    sources_hours: List[str] = Field(default_factory=list)     # URLs specifically supporting Thanksgiving 2025 hours
    sources_location: List[str] = Field(default_factory=list)  # URLs proving NH locations
    sources_general: List[str] = Field(default_factory=list)   # other relevant URLs


class ShoppingAnswerExtraction(BaseModel):
    """Extraction for the overall answer: up to three store candidates."""
    stores: List[StoreCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_store_candidates() -> str:
    return """
    Extract up to three major grocery store chain options that the answer proposes for shopping on Thanksgiving Day 2025 in New Hampshire.
    For each store, extract the following fields exactly as written in the answer:
    - name: The store chain name (not a specific single-location store name unless the answer clearly names the chain).
    - thanksgiving_opening_time: The opening time on Thanksgiving Day 2025 as stated in the answer (string; do not infer).
    - thanksgiving_closing_time: The closing time on Thanksgiving Day 2025 as stated in the answer (string; do not infer).
    - thanksgiving_hours_text: Any free-form hours wording referring to Thanksgiving Day 2025 (e.g., "open 7am–3pm").
    - sources_hours: All URLs that the answer cites to support Thanksgiving Day 2025 hours (chain-level or NH store-specific).
    - sources_location: All URLs that the answer cites proving that the chain has locations in New Hampshire (e.g., store locator or NH location pages).
    - sources_general: Any additional URLs the answer cites that are relevant (e.g., holiday-hours policy pages, news).
    
    Rules:
    - Only extract URLs that are explicitly present in the answer text (including markdown links).
    - Preserve times exactly as written (e.g., "7 AM", "7am", "07:00", etc.). Do not normalize.
    - If a field is not present in the answer, set it to null (for strings) or [] (for URL lists).
    - The "stores" array should contain up to three store objects in the same order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def all_sources(candidate: StoreCandidate) -> List[str]:
    return unique_urls(candidate.sources_hours + candidate.sources_location + candidate.sources_general)


def hour_sources(candidate: StoreCandidate) -> List[str]:
    urls = candidate.sources_hours
    if not urls:
        urls = all_sources(candidate)
    return unique_urls(urls)


def location_sources(candidate: StoreCandidate) -> List[str]:
    urls = candidate.sources_location
    if not urls:
        urls = all_sources(candidate)
    return unique_urls(urls)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_for_store(evaluator: Evaluator, root_node, store: StoreCandidate) -> None:
    """
    Construct and run verification according to the rubric for a single identified store.
    """
    # Top-level: store_identification (critical)
    identify_node = evaluator.add_parallel(
        id="store_identification",
        desc="Correctly identifies appropriate grocery store chain(s)",
        parent=root_node,
        critical=True,
    )

    # Leaf: store_name (critical) - basic existence of a named chain
    has_store_name = bool(store.name and store.name.strip())
    evaluator.add_custom_node(
        result=has_store_name,
        id="store_name",
        desc="Names at least one major grocery store chain that is open on Thanksgiving Day 2025",
        parent=identify_node,
        critical=True,
    )

    # Leaf: new_hampshire_location (critical) - verify chain has NH locations via provided URLs
    nh_loc_node = evaluator.add_leaf(
        id="new_hampshire_location",
        desc="The identified store has locations in or serves New Hampshire",
        parent=identify_node,
        critical=True,
    )
    nh_claim = f"The grocery chain '{store.name or 'UNKNOWN'}' has at least one store location in the state of New Hampshire."
    await evaluator.verify(
        claim=nh_claim,
        node=nh_loc_node,
        sources=location_sources(store),
        additional_instruction="Look for an official store locator, location pages, or explicit NH addresses showing that the chain operates in New Hampshire.",
    )

    # Leaf: operating_status (critical) - verify open (not closed) on Thanksgiving Day 2025
    open_status_node = evaluator.add_leaf(
        id="operating_status",
        desc="The identified store is confirmed to be open (not closed) on Thanksgiving Day 2025",
        parent=identify_node,
        critical=True,
    )
    status_claim = f"The grocery chain '{store.name or 'UNKNOWN'}' is open on Thanksgiving Day {THANKSGIVING_DAY_STR}."
    await evaluator.verify(
        claim=status_claim,
        node=open_status_node,
        sources=hour_sources(store),
        additional_instruction="Check holiday hours pages or official announcements. If the source indicates closure on Thanksgiving 2025, this claim is not supported.",
    )

    # Leaf: time_compatibility (critical) - verify window 1:00–2:30 PM is covered
    time_compat_node = evaluator.add_leaf(
        id="time_compatibility",
        desc="The store's operating hours cover the shopper's required timeframe (1:00 PM - 2:30 PM)",
        parent=identify_node,
        critical=True,
    )
    time_claim = (
        f"Based on the Thanksgiving Day {THANKSGIVING_DAY_STR} hours for '{store.name or 'UNKNOWN'}', "
        f"the store is open during the entire time window {REQUIRED_WINDOW_TEXT} local time."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_compat_node,
        sources=hour_sources(store),
        additional_instruction="From the posted Thanksgiving 2025 hours, determine if the store is open for the whole window 1:00 PM–2:30 PM (i.e., opens no later than 1:00 PM and closes at or after 2:30 PM).",
    )

    # Top-level: hour_details (critical)
    # Note: Parent is critical, so children must also be critical to satisfy framework constraints.
    hours_node = evaluator.add_parallel(
        id="hour_details",
        desc="Provides accurate operating hours for the identified store on Thanksgiving Day 2025",
        parent=root_node,
        critical=True,
    )

    # Leaf: opening_time (critical) - verify opening time if provided
    opening_leaf = evaluator.add_leaf(
        id="opening_time",
        desc="States the store's opening time on Thanksgiving Day 2025",
        parent=hours_node,
        critical=True,
    )
    opening_claim = (
        f"For the grocery chain '{store.name or 'UNKNOWN'}', the opening time on Thanksgiving Day {THANKSGIVING_DAY_STR} "
        f"is '{store.thanksgiving_opening_time or ''}'."
    )
    await evaluator.verify(
        claim=opening_claim,
        node=opening_leaf,
        sources=hour_sources(store),
        additional_instruction="Verify the exact opening time for Thanksgiving Day 2025 as stated on the provided source(s). If the answer doesn't provide a concrete time, this claim should not be considered supported.",
    )

    # Leaf: closing_time (critical) - verify closing time if provided
    closing_leaf = evaluator.add_leaf(
        id="closing_time",
        desc="States the store's closing time on Thanksgiving Day 2025",
        parent=hours_node,
        critical=True,
    )
    closing_claim = (
        f"For the grocery chain '{store.name or 'UNKNOWN'}', the closing time on Thanksgiving Day {THANKSGIVING_DAY_STR} "
        f"is '{store.thanksgiving_closing_time or ''}'."
    )
    await evaluator.verify(
        claim=closing_claim,
        node=closing_leaf,
        sources=hour_sources(store),
        additional_instruction="Verify the exact closing time for Thanksgiving Day 2025 as stated on the provided source(s). If the answer doesn't provide a concrete time, this claim should not be considered supported.",
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
    Evaluate an answer for the New Hampshire Thanksgiving 2025 grocery shopping task.
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

    # Extract store candidates
    extraction = await evaluator.extract(
        prompt=prompt_extract_store_candidates(),
        template_class=ShoppingAnswerExtraction,
        extraction_name="extracted_store_candidates",
    )

    # Record custom info about required window/date
    evaluator.add_custom_info(
        info={
            "required_date": THANKSGIVING_DAY_STR,
            "required_time_window": REQUIRED_WINDOW_TEXT,
        },
        info_type="task_constraints",
    )

    # Choose the first candidate if available; otherwise, create an empty placeholder (will fail critical checks)
    candidate = extraction.stores[0] if extraction.stores else StoreCandidate()

    # Build verification tree for the selected store
    await build_verification_for_store(evaluator, root, candidate)

    # Return evaluation summary
    return evaluator.get_summary()