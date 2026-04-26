import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "phx_xmas_eve_craft_shopping_2025"
TASK_DESCRIPTION = (
    "I'm planning a last-minute holiday craft shopping trip in Phoenix, Arizona on Christmas Eve morning, "
    "December 24, 2025. I need to find 4 different major craft or home improvement stores in Phoenix that "
    "will still be open in the late afternoon so I can shop after 4:00 PM. Specifically, find 4 stores that "
    "meet ALL of the following requirements: (1) The store must be a major national chain (Hobby Lobby, "
    "Michaels, Lowe's, Home Depot, Joann Fabrics, or Ace Hardware); (2) The store must be located in Phoenix, "
    "Arizona; (3) The store must close at 5:00 PM or later on Christmas Eve (December 24, 2025); (4) The store "
    "must be closed on Christmas Day (December 25, 2025). For each store, provide the store name, specific "
    "Phoenix location address, Christmas Eve closing time, and a reference URL confirming the holiday hours."
)

ALLOWED_CHAINS = [
    "Hobby Lobby",
    "Michaels",
    "Lowe's",
    "Home Depot",
    "Joann Fabrics",
    "Ace Hardware",
]

# Helpful synonyms/variants accepted for the chain verification
ALLOWED_CHAIN_SYNONYMS = [
    # Hobby Lobby is usually exact
    "Michaels Stores",
    "Lowe's Home Improvement",
    "The Home Depot",
    "Home Depot",
    "JOANN",
    "JOANN Fabrics",
    "JOANN Fabrics and Crafts",
    "Ace",
    "Ace Hardware",
]

XMAS_EVE_DATE_TEXT = "December 24, 2025"
XMAS_DAY_DATE_TEXT = "December 25, 2025"


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class StoreItem(BaseModel):
    """One store item extracted from the answer."""
    chain: Optional[str] = None
    store_name: Optional[str] = None  # The display/store name as mentioned (optional)
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    christmas_eve_close_time: Optional[str] = None  # Keep as string; examples: "5 PM", "6:00 pm"
    christmas_day_status: Optional[str] = None  # e.g., "Closed", "Open", "Varies by location"
    reference_urls: List[str] = Field(default_factory=list)


class StoresExtraction(BaseModel):
    """Extraction of up to four stores."""
    stores: List[StoreItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return f"""
Extract up to the first 4 distinct store entries from the answer that are proposed as meeting the user's requirements.
For each store, return a JSON with the following fields:

- chain: The chain/brand name as presented (e.g., "Hobby Lobby", "Michaels", "Lowe's", "Home Depot", "Joann Fabrics", "Ace Hardware").
- store_name: The specific store name as written (if provided), otherwise null.
- address: The specific Phoenix street address as provided (include street number if present).
- city: The city name for the store location, if provided.
- state: The state abbreviation (e.g., "AZ") if provided.
- christmas_eve_close_time: The closing time stated for Christmas Eve ({XMAS_EVE_DATE_TEXT}) as a string exactly as written in the answer (e.g., "5 PM", "6:00 pm"). If missing, set to null.
- christmas_day_status: The stated status for Christmas Day ({XMAS_DAY_DATE_TEXT}), e.g., "Closed" or "Open", exactly as claimed in the answer. If not mentioned, set to null.
- reference_urls: An array of 1 or more URLs explicitly provided in the answer that are used to support holiday hours for this specific store. Extract only actual URLs present in the answer. If none are provided, return an empty array.

Important:
- Extract only information explicitly present in the answer. Do not invent any information.
- For URLs, follow the URL extraction rules strictly: extract only valid URLs explicitly present.
- If the answer lists more than 4 stores, only extract the first 4 in the order they appear in the answer.
- If the answer lists fewer than 4, extract whatever is available (possibly 0-3).
"""


# --------------------------------------------------------------------------- #
# Helper: Ordinal mapping                                                     #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"Store #{n}")


# --------------------------------------------------------------------------- #
# Verification logic for a single store                                       #
# --------------------------------------------------------------------------- #
async def verify_single_store(
    evaluator: Evaluator,
    root,
    store: StoreItem,
    idx: int,
) -> None:
    """
    Build the subtree for one store with all required verification leaves.
    """
    store_id = idx  # 1-based index
    store_desc = f"{ordinal(store_id)} qualifying craft or home improvement store in Phoenix"

    # Parent node for this store (parallel aggregator; non-critical to allow partial credit across stores)
    parent_node = evaluator.add_parallel(
        id=f"store_{store_id}",
        desc=store_desc,
        parent=root,
        critical=False,
    )

    # 1) Reference URL provided (Critical) — implement as a custom leaf check (existence + basic validity).
    # We rely on hours/location leaves to verify actual content with the URL(s).
    has_valid_ref = bool(store.reference_urls) and any(
        isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))
        for u in store.reference_urls
    )
    reference_node = evaluator.add_custom_node(
        result=has_valid_ref,
        id=f"store_{store_id}_reference",
        desc="Valid reference URL provided confirming store details and holiday hours",
        parent=parent_node,
        critical=True
    )

    # 2) Chain check (Critical)
    chain_node = evaluator.add_leaf(
        id=f"store_{store_id}_chain",
        desc="Store is a major national craft or home improvement chain (Hobby Lobby, Michaels, Lowe's, Home Depot, Joann Fabrics, or Ace Hardware)",
        parent=parent_node,
        critical=True
    )

    chain_name_for_check = store.chain or (store.store_name or "")
    chain_claim = (
        f"The store '{chain_name_for_check}' is one of the allowed chains: "
        f"{', '.join(ALLOWED_CHAINS)}. Accept commonly used synonyms or formatting variants."
    )
    await evaluator.verify(
        claim=chain_claim,
        node=chain_node,
        additional_instruction=(
            "Judge whether the given store/brand name belongs to the allowed major chains. "
            "Allow reasonable synonyms/variants: "
            f"{', '.join(ALLOWED_CHAIN_SYNONYMS)}. "
            "For example, treat 'The Home Depot' as 'Home Depot', 'Lowe's Home Improvement' as 'Lowe's', "
            "'JOANN'/'JOANN Fabrics' as 'Joann Fabrics', and 'Michaels Stores' as 'Michaels'."
        ),
    )

    # 3) Location in Phoenix, AZ (Critical) — verify with provided reference URLs; gate by reference existence
    location_node = evaluator.add_leaf(
        id=f"store_{store_id}_location",
        desc="Store is located in Phoenix, Arizona",
        parent=parent_node,
        critical=True
    )
    location_address = store.address or ""
    location_city = store.city or ""
    location_state = store.state or ""
    location_claim = (
        f"The store location (address: '{location_address}') is in Phoenix, Arizona (Phoenix, AZ). "
        "It should be clear that the city is Phoenix and the state is AZ."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=store.reference_urls if store.reference_urls else None,
        additional_instruction=(
            "Use the provided webpage(s) to confirm the location is in 'Phoenix, AZ' specifically "
            "(not a neighboring city such as Scottsdale, Tempe, Glendale, etc.). "
            "Accept reasonable address formatting variants (e.g., 'Phoenix, AZ 850xx')."
        ),
        extra_prerequisites=[reference_node],
    )

    # 4) Christmas Eve closing time requirement (Critical) — open until 5 PM or later on Dec 24, 2025
    xmas_eve_node = evaluator.add_leaf(
        id=f"store_{store_id}_christmas_eve_hours",
        desc="Store closes at 5:00 PM or later on Christmas Eve (December 24, 2025)",
        parent=parent_node,
        critical=True
    )
    eve_claim = (
        f"On {XMAS_EVE_DATE_TEXT} (Christmas Eve), this store location is open until at least 5:00 PM "
        "(i.e., closing time is 5:00 PM or later)."
    )
    await evaluator.verify(
        claim=eve_claim,
        node=xmas_eve_node,
        sources=store.reference_urls if store.reference_urls else None,
        additional_instruction=(
            "Check the holiday hours on the provided source(s) specifically for Christmas Eve 2025. "
            "Pass if the page indicates the store is open until 5 PM or later (e.g., 'Open until 5 PM', 'closes at 6 PM', "
            "'Hours: 7 AM – 6 PM'). "
            "Fail if it indicates closing earlier than 5 PM (e.g., 4 PM or earlier), closed all day, or if hours are unknown/unspecified for Christmas Eve 2025."
        ),
        extra_prerequisites=[reference_node],
    )

    # 5) Christmas Day closure requirement (Critical) — closed on Dec 25, 2025
    xmas_day_node = evaluator.add_leaf(
        id=f"store_{store_id}_christmas_closure",
        desc="Store is closed on Christmas Day (December 25, 2025)",
        parent=parent_node,
        critical=True
    )
    day_claim = (
        f"On {XMAS_DAY_DATE_TEXT} (Christmas Day), this store location is closed."
    )
    await evaluator.verify(
        claim=day_claim,
        node=xmas_day_node,
        sources=store.reference_urls if store.reference_urls else None,
        additional_instruction=(
            "Use the provided source(s) to verify that the location is closed on Christmas Day 2025. "
            "Chain-level holiday policy pages that clearly state all locations are closed on Christmas Day are acceptable."
        ),
        extra_prerequisites=[reference_node],
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Phoenix Christmas Eve 2025 shopping task.
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

    # Extract stores from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction",
    )

    # Keep only the first 4 items (pad with empty ones if fewer)
    stores: List[StoreItem] = list(extracted.stores[:4])
    while len(stores) < 4:
        stores.append(StoreItem())

    # Record helpful info
    evaluator.add_ground_truth({
        "allowed_chains": ALLOWED_CHAINS,
        "allowed_chain_synonyms": ALLOWED_CHAIN_SYNONYMS,
        "requirements": {
            "city": "Phoenix, AZ",
            "christmas_eve": f"Open until >= 5:00 PM on {XMAS_EVE_DATE_TEXT}",
            "christmas_day": f"Closed on {XMAS_DAY_DATE_TEXT}"
        }
    })

    # Build and verify per-store subtrees
    for i in range(4):
        await verify_single_store(
            evaluator=evaluator,
            root=root,
            store=stores[i],
            idx=i + 1
        )

    return evaluator.get_summary()