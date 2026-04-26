import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thanksgiving_grocers_4"
TASK_DESCRIPTION = (
    "Identify 4 different national grocery store chains in the United States that are open on Thanksgiving Day "
    "and remain open past 1:00 PM. For each chain, provide: (1) The specific closing time on Thanksgiving, "
    "(2) Any significant regional exceptions or hour variations that exist, and (3) A reference URL that verifies "
    "this information. Note: Dollar stores, convenience stores, and restaurants do not qualify as grocery store chains."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChainInfo(BaseModel):
    name: Optional[str] = None
    closing_time: Optional[str] = None  # Free-form string (e.g., "2 pm", "3:00 PM local time")
    variations: Optional[str] = None    # Notes about regional exceptions or "hours vary by location"
    urls: List[str] = Field(default_factory=list)  # All URLs cited for this chain


class ChainsExtraction(BaseModel):
    chains: List[ChainInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chains() -> str:
    return """
    Extract all grocery store chain entries that the answer provides for this task. For each chain mentioned,
    extract the following fields exactly as stated in the answer:
    - name: The chain's name (string).
    - closing_time: The specific Thanksgiving Day closing time if provided (string; e.g., "2 pm", "3:00 PM local time").
                    If the answer gives a range (e.g., "2–3 pm") or imprecise phrasing (e.g., "mid-afternoon"), capture that verbatim.
                    If not stated, set to null.
    - variations: Any significant regional exceptions/variations or a general note such as "hours vary by location".
                  If not stated, set to null.
    - urls: All reference URLs that the answer cites for this chain's Thanksgiving hours/closing information.
            Include every explicit URL (plain or in markdown). If none provided, return an empty array.

    Return:
    {
      "chains": [
        {"name": ..., "closing_time": ..., "variations": ..., "urls": [...]},
        ...
      ]
    }

    Rules:
    - Do not invent or infer any values not present in the answer.
    - Include ALL distinct chain entries the answer lists (even if more than 4).
    - Only record URLs explicitly present in the answer text.
    - Keep values as strings; do not normalize or reinterpret times.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_chain_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().casefold()


def first_k_unique_chains(chains: List[ChainInfo], k: int = 4) -> List[ChainInfo]:
    seen = set()
    picked: List[ChainInfo] = []
    for ch in chains:
        key = normalize_chain_name(ch.name)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        picked.append(ch)
        if len(picked) >= k:
            break
    return picked


def valid_url_present(urls: List[str]) -> bool:
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


# --------------------------------------------------------------------------- #
# Verification builder for a single chain                                     #
# --------------------------------------------------------------------------- #
async def verify_store_chain(
    evaluator: Evaluator,
    parent_node,
    chain: ChainInfo,
    store_index: int,
) -> None:
    """
    Build verification subtree and run verifications for a single chain.
    Follows the rubric leaves for one chain under a parallel parent node.
    """
    store_node = evaluator.add_parallel(
        id=f"store_{store_index+1}",
        desc=f"{store_index+1}st qualifying grocery store chain" if store_index == 0 else
             (f"{store_index+1}nd qualifying grocery store chain" if store_index == 1 else
              (f"{store_index+1}rd qualifying grocery store chain" if store_index == 2 else
               f"{store_index+1}th qualifying grocery store chain")),
        parent=parent_node,
        critical=False
    )

    name = chain.name or ""
    urls = chain.urls or []
    closing_time = chain.closing_time or ""
    variations = chain.variations or ""

    # chain_name_provided (critical, existence)
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id=f"store_{store_index+1}_chain_name_provided",
        desc="The chain name is clearly identified.",
        parent=store_node,
        critical=True
    )

    # closing_time_provided (critical, existence)
    evaluator.add_custom_node(
        result=bool(closing_time.strip()),
        id=f"store_{store_index+1}_closing_time_provided",
        desc="A specific Thanksgiving Day closing time is provided.",
        parent=store_node,
        critical=True
    )

    # regional_variations_addressed (critical, answer addresses variations)
    evaluator.add_custom_node(
        result=bool(variations.strip()),
        id=f"store_{store_index+1}_regional_variations_addressed",
        desc="Any significant regional exceptions/variations in Thanksgiving hours are noted if they exist (or the answer clarifies that hours vary by location / no significant exceptions were found).",
        parent=store_node,
        critical=True
    )

    # source_url_provided (critical, existence)
    evaluator.add_custom_node(
        result=valid_url_present(urls),
        id=f"store_{store_index+1}_source_url_provided",
        desc="At least one verifiable source URL is provided that supports the Thanksgiving hours/closing time information given.",
        parent=store_node,
        critical=True
    )

    # qualifies_as_grocery_chain (critical, verify with sources if available)
    qualifies_node = evaluator.add_leaf(
        id=f"store_{store_index+1}_qualifies_as_grocery_chain",
        desc="The chain qualifies as a grocery store chain (not a dollar store, convenience store, or restaurant).",
        parent=store_node,
        critical=True
    )
    qualifies_claim = (
        f"'{name}' is a grocery store chain (i.e., a retailer primarily selling groceries) and is not a dollar store, "
        f"convenience store, or restaurant."
    )

    # is_national_us_chain (critical)
    national_node = evaluator.add_leaf(
        id=f"store_{store_index+1}_is_national_us_chain",
        desc="The chain is a national grocery store chain in the United States.",
        parent=store_node,
        critical=True
    )
    national_claim = (
        f"'{name}' operates as a national grocery store chain in the United States (with locations across multiple U.S. states)."
    )

    # open_on_thanksgiving (critical)
    open_node = evaluator.add_leaf(
        id=f"store_{store_index+1}_open_on_thanksgiving",
        desc="The chain is confirmed open on Thanksgiving Day.",
        parent=store_node,
        critical=True
    )
    open_claim = (
        f"On Thanksgiving Day, '{name}' stores are open (i.e., not closed)."
    )

    # open_past_1pm (critical)
    past1_node = evaluator.add_leaf(
        id=f"store_{store_index+1}_open_past_1pm",
        desc="The Thanksgiving Day hours imply the store remains open past 1:00 PM (e.g., closing time is after 1:00 PM).",
        parent=store_node,
        critical=True
    )
    past1_claim = (
        f"On Thanksgiving Day, '{name}' remains open past 1:00 PM local time (closing time later than 1:00 PM)."
    )

    # Batch verify the four claim-type leaves
    claims_and_sources = [
        (qualifies_claim, urls, qualifies_node,
         "Verify from the provided pages whether this brand is a grocery store chain. "
         "If the source is an official site, Wikipedia, or reputable news article describing it as a grocery chain, "
         "consider it sufficient. If the brand is primarily a dollar store, convenience store, or restaurant, mark incorrect."),
        (national_claim, urls, national_node,
         "Check whether the pages indicate that the chain operates across the United States (multiple states). "
         "Evidence can include mentions of nationwide presence, a multi-state store locator, or a description of operations across many states."),
        (open_claim, urls, open_node,
         "Focus strictly on Thanksgiving Day hours. If the source states the chain is open on Thanksgiving Day, "
         "even with reduced hours or exceptions, mark as supported. If the page says closed on Thanksgiving, mark incorrect."),
        (past1_claim, urls, past1_node,
         "From the Thanksgiving hours on the provided pages, determine whether stores remain open past 1:00 PM local time. "
         "If the closing time is after 1 pm (e.g., 2 pm, 3 pm), or if the hours range covers the afternoon, mark as supported. "
         "If closing is at or before 1 pm, mark as not supported.")
    ]
    await evaluator.batch_verify(claims_and_sources)


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
) -> Dict:
    """
    Evaluate an answer for the '4 national grocers open past 1 PM on Thanksgiving' task.
    """
    # Initialize evaluator/root
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

    # 1) Extract chains from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=ChainsExtraction,
        extraction_name="chains_extraction",
    )

    # Prepare counts for the critical "exactly_four_distinct_chains" node
    all_named = [c for c in extracted.chains if c.name and c.name.strip()]
    unique_keys = {normalize_chain_name(c.name) for c in all_named if normalize_chain_name(c.name)}
    unique_count = len(unique_keys)
    total_named_count = len(all_named)

    # 2) Critical: exactly four distinct chains, no extras beyond four
    evaluator.add_custom_node(
        result=(total_named_count == 4 and unique_count == 4),
        id="exactly_four_distinct_chains",
        desc="Exactly 4 grocery store chains are provided and they are all distinct (no duplicates and no extra chains beyond the four).",
        parent=root,
        critical=True
    )

    # 3) Select the first 4 unique chains (order-preserving) for detailed checking
    selected = first_k_unique_chains(extracted.chains, k=4)

    # Pad to 4 if fewer (placeholders will fail critical checks as appropriate)
    while len(selected) < 4:
        selected.append(ChainInfo())

    # Record custom info for debugging
    evaluator.add_custom_info(
        {
            "all_extracted_names": [c.name for c in extracted.chains],
            "unique_extracted_names": [c.name for c in all_named if normalize_chain_name(c.name) in unique_keys],
            "selected_names_for_verification": [c.name for c in selected]
        },
        info_type="extraction_debug",
        info_name="extraction_debug_info"
    )

    # 4) Build/store verification subtrees for each of the 4
    # Each store node is parallel aggregation as per rubric
    # Children inside each store node are individual critical checks
    for idx in range(4):
        await verify_store_chain(evaluator, root, selected[idx], idx)

    # 5) Return the evaluation summary
    return evaluator.get_summary()