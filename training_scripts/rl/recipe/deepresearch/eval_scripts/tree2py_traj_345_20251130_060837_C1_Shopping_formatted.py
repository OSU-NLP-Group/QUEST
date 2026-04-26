import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bf_2025_earliest_open"
TASK_DESCRIPTION = "Among the major national retailers Target, Best Buy, Walmart, Kohl's, and Barnes & Noble, which store opens earliest on Black Friday 2025, and what time does it open?"

ALLOWED_RETAILERS = ["Target", "Best Buy", "Walmart", "Kohl's", "Barnes & Noble"]
BLACK_FRIDAY_2025_DATE = "November 28, 2025"  # Day after Thanksgiving in 2025


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RetailerInfo(BaseModel):
    """Information for a single retailer."""
    name: Optional[str] = None
    opening_time: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class EarliestOpeningExtraction(BaseModel):
    """Extraction result capturing the named earliest retailer, its opening time, and any other retailers mentioned."""
    chosen: Optional[RetailerInfo] = None
    others: List[RetailerInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_earliest_opening() -> str:
    return (
        "From the provided answer, extract the single retailer that the answer claims opens earliest on Black Friday 2025, "
        "along with the specific opening time stated and any URLs the answer cites to support that time. Also extract any "
        "other of the five specified retailers that the answer mentions with their opening times and URLs.\n\n"
        "The five retailers of interest are exactly: Target, Best Buy, Walmart, Kohl's, and Barnes & Noble.\n\n"
        "Return a JSON object with the following fields:\n"
        "- chosen: An object with fields:\n"
        "    • name: The name of the retailer the answer claims opens earliest (as written in the answer). If the answer does not clearly name a single earliest retailer, set to null.\n"
        "    • opening_time: The specific time-of-day the answer claims this retailer opens on Black Friday 2025 (e.g., '5 AM', '5:00 a.m.', '05:00', '6 am'). If not provided, set to null.\n"
        "    • source_urls: An array of URLs explicitly cited in the answer that support the claimed opening time for Black Friday 2025 for this retailer. If none are cited, return an empty array.\n"
        "- others: An array of objects, each with fields:\n"
        "    • name: One of the five specified retailers (as written in the answer) that the answer also mentions with respect to Black Friday 2025 opening.\n"
        "    • opening_time: The time-of-day the answer states for this retailer (if any). If not stated, set to null.\n"
        "    • source_urls: An array of URLs explicitly cited for this retailer (if any). If none, return an empty array.\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly stated in the answer; do not invent any times or URLs.\n"
        "2) For URLs, only extract valid URLs present in the answer (including markdown links). If a URL lacks protocol, prepend http://.\n"
        "3) The opening_time should be a specific time-of-day string, not vague phrases like 'early morning'. If the answer uses a vague phrase only, set opening_time to null.\n"
        "4) Do not include retailers outside the five specified.\n"
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _unique_urls(url_lists: List[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for sub in url_lists:
        for u in sub:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification sub-tree construction                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_earliest_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: EarliestOpeningExtraction
) -> None:
    """
    Build the verification nodes and run the checks according to the rubric:
      1) The answer names exactly one retailer and it is one of the five.
      2) The answer provides a specific opening time for that retailer.
      3) The stated opening time matches reliable published store-hours info (via cited sources if available).
      4) None of the other four opens earlier than the named retailer (use published info; rely on cited sources when provided).
    All children under this node are critical, and the node itself is critical.
    """
    node = evaluator.add_parallel(
        id="earliest_black_friday_store",
        desc="Identify which of Target, Best Buy, Walmart, Kohl's, and Barnes & Noble opens earliest on Black Friday 2025, and provide that opening time.",
        parent=parent_node,
        critical=True
    )

    chosen_name = (extracted.chosen.name.strip() if extracted and extracted.chosen and extracted.chosen.name else "")
    chosen_time = (extracted.chosen.opening_time.strip() if extracted and extracted.chosen and extracted.chosen.opening_time else "")
    chosen_sources = (extracted.chosen.source_urls if extracted and extracted.chosen and extracted.chosen.source_urls else [])

    # Leaf 1: Retailer is one of specified five AND exactly one retailer is named
    leaf1 = evaluator.add_leaf(
        id="retailer_is_one_of_specified_five",
        desc="The answer names exactly one retailer and it is one of: Target, Best Buy, Walmart, Kohl's, Barnes & Noble.",
        parent=node,
        critical=True
    )
    claim1 = (
        f"The answer names exactly one retailer as the earliest-opening store on Black Friday 2025 among "
        f"Target, Best Buy, Walmart, Kohl's, and Barnes & Noble. The named retailer is '{chosen_name}'. "
        f"Treat minor name variants as equivalent (e.g., 'Kohls' = 'Kohl's', 'Barnes and Noble' = 'Barnes & Noble'). "
        f"If the answer names more than one retailer (a tie) or names a retailer outside the five, this claim is incorrect."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        additional_instruction="Judge based solely on the answer text content. Confirm there is exactly one named earliest retailer and it is in the allowed set."
    )

    # Leaf 2: Opening time is provided (specific time-of-day)
    leaf2 = evaluator.add_leaf(
        id="opening_time_is_provided",
        desc="The answer provides a specific opening time (time-of-day) for the named retailer for Black Friday 2025.",
        parent=node,
        critical=True
    )
    claim2 = (
        f"The answer explicitly provides a specific time-of-day for when {chosen_name if chosen_name else 'the named retailer'} "
        f"opens on Black Friday 2025: '{chosen_time}'. Vague phrases like 'early morning' do not count as a specific time."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        additional_instruction="Accept standard time formats: '5 AM', '5:00 a.m.', '05:00', '5am', etc. Reject vague/relative timing."
    )

    # Leaf 3: Opening time is correct for the named retailer (verify with URLs if provided)
    leaf3 = evaluator.add_leaf(
        id="opening_time_is_correct_for_named_retailer",
        desc="The stated opening time for the named retailer on Black Friday 2025 matches reliable published store-hours information for that retailer/date.",
        parent=node,
        critical=True
    )
    claim3 = (
        f"According to the provided sources, {chosen_name if chosen_name else 'the named retailer'} opens at "
        f"{chosen_time if chosen_time else '[no time provided]'} on Black Friday 2025 ({BLACK_FRIDAY_2025_DATE}). "
        f"Do not confuse with Thanksgiving Day hours (Nov 27, 2025) or general/regular hours."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=chosen_sources,  # may be empty -> simple verification route
        additional_instruction=(
            "Verify that the page explicitly supports Black Friday 2025 opening time (Nov 28, 2025). "
            "If the page is irrelevant, outdated (e.g., 2024), or only lists generic hours with no BF 2025 specificity, mark as not supported."
        )
    )

    # Leaf 4: Named retailer is earliest among the five (verify using all available URLs)
    leaf4 = evaluator.add_leaf(
        id="named_retailer_is_earliest_among_five",
        desc="Using reliable published store-hours information for Black Friday 2025, none of the other four specified retailers opens earlier than the named retailer.",
        parent=node,
        critical=True
    )

    # Aggregate all available sources from 'others' as well as the chosen retailer
    other_sources_lists = []
    if extracted and extracted.others:
        for r in extracted.others:
            if r and r.source_urls:
                other_sources_lists.append(r.source_urls)
    all_sources_for_earliest_check = _unique_urls([chosen_sources] + other_sources_lists)

    claim4 = (
        f"Among Target, Best Buy, Walmart, Kohl's, and Barnes & Noble, no retailer opens earlier than "
        f"{chosen_name if chosen_name else 'the named retailer'} on Black Friday 2025 ({BLACK_FRIDAY_2025_DATE}). "
        f"Equal opening times are not earlier; they are acceptable as 'no earlier'. "
        f"Use the provided sources to compare opening times across these retailers."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=all_sources_for_earliest_check,  # may be empty -> simple verification route
        additional_instruction=(
            "Focus on Black Friday 2025 (Nov 28, 2025) opening times. Prefer corporate/national announcements or official store-hours pages. "
            "Do not use Thanksgiving Day hours. If any other retailer among the five clearly opens earlier than the named retailer, mark this claim as not supported."
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
    Evaluate an answer for the Black Friday 2025 earliest-opening retailer task.

    Steps:
    1) Initialize evaluator and root.
    2) Extract the named earliest retailer, opening time, and any cited URLs for the five retailers.
    3) Build and verify the rubric tree:
       - one-of-five and exactly-one retailer
       - opening time provided
       - opening time correct (via sources)
       - none of the other four opens earlier (via sources)
    4) Return summary.
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_earliest_opening(),
        template_class=EarliestOpeningExtraction,
        extraction_name="earliest_opening_extraction"
    )

    # Add contextual info (non-scoring)
    evaluator.add_ground_truth(
        {
            "allowed_retailers": ALLOWED_RETAILERS,
            "black_friday_2025_date": BLACK_FRIDAY_2025_DATE
        },
        gt_type="context_info"
    )

    # Build verification
    await build_and_verify_earliest_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()