import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "store_seattle"
TASK_DESCRIPTION = """
Can you find two physical stores in the Seattle area where I can try on clothing from Acne Studios? Please provide the store names, their addresses, and a link to their product page showing the available Acne Studios items.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreNames(BaseModel):
    """Names of stores extracted from the answer"""
    names: List[str] = Field(default_factory=list)


class StoreDetails(BaseModel):
    """Detailed information about a single store"""
    name: Optional[str] = None
    address: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_store_names() -> str:
    return """
    Extract the names of all physical stores in the Seattle area mentioned in the answer that carry Acne Studios clothing.

    Return a list of store names only. If no stores are mentioned, return an empty list.
    """


def prompt_extract_store_details(store_name: str) -> str:
    return f"""
    Extract detailed information about the store '{store_name}' mentioned in the answer.

    Extract:
    1. The store address in the Seattle area
    2. All URLs mentioned in relation to this store, especially any that might show Acne Studios products

    If the address is not provided, return null for that field.
    If no URLs are provided, return an empty list for urls.
    """


# --------------------------------------------------------------------------- #
# Store verification functions                                                #
# --------------------------------------------------------------------------- #
async def verify_store(
        evaluator: Evaluator,
        parent_node,
        store_index: int,
        store: StoreDetails,
) -> None:
    """
    Verify all aspects of a single store through sequential checks:
    1. Basic info is provided and address is in Seattle
    2. Store name and address are supported by at least one URL
    3. At least one product page shows available Acne Studios items
    """
    # Create store verification parent node
    store_node = evaluator.add_sequential(
        id=f"store_{store_index}_verification",
        desc=f"Store {store_index + 1} '{store.name}' is a valid Seattle store that carries Acne Studios",
        parent=parent_node,
        critical=False,  # Individual stores can fail, we need at least 2 total
    )

    # 1. Basic info check - verify all information is provided
    has_all_info = bool(store.name and store.address and store.urls)
    basic_info_node = evaluator.add_custom_node(
        result=has_all_info,
        id=f"store_{store_index}_basic_info",
        desc=f"Store {store_index + 1} has all required basic information",
        parent=store_node,
        critical=True
    )

    seattle_address_node = evaluator.add_leaf(
        id=f"store_{store_index}_address_seattle",
        desc=f"Store {store_index + 1} has a address in the Seattle area.",
        parent=store_node,
        critical=True
    )
    # Verify address is in Seattle area
    claim = f"The address '{store.address}' is located in the Seattle area."
    await evaluator.verify(
        claim=claim,
        node=seattle_address_node,
        additional_instruction="Verify that this address is in the Seattle area. " +
                                "The Seattle area includes Seattle city proper and nearby cities like Bellevue, " +
                                "Kirkland, Redmond, etc."
    )


    # 2. Store info support check - verify store exists at the address
    info_support_node = evaluator.add_leaf(
        id=f"store_{store_index}_info_support",
        desc=f"Store {store_index + 1} name and address are supported by provided URLs",
        parent=store_node,
        critical=True
    )

    claim = f"The store '{store.name}' located at '{store.address}' is a real store that exists at the address as indicated in the page."
    await evaluator.verify(
        claim=claim,
        node=info_support_node,
        sources=store.urls
    )

    # 3. Acne Studios availability check
    acne_availability_node = evaluator.add_leaf(
        id=f"store_{store_index}_acne_availability",
        desc=f"Store {store_index + 1} URLs show available Acne Studios items",
        parent=store_node,
        critical=True
    )

    claim = f"Acne Studios items are available at the store '{store.name}', as indicated in this webpage."
    await evaluator.verify(
        claim=claim,
        node=acne_availability_node,
        sources=store.urls,
        additional_instruction="If the store is Nordstrom, as long as the webpage shows the Acne Studio items on Nordstrom, the verification should be passed (it does not need to be tied with the specific store)."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.

    This function evaluates whether the answer correctly identifies two physical
    stores in the Seattle area where Acne Studios clothing can be tried on, including
    store names, addresses, and product page URLs.
    """
    # -------- 1. Initialize evaluator ----------------------------------- #
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

    # -------- 2. Extract store information from the answer ---------------- #
    # First extract all store names
    store_names_info = await evaluator.extract(
        prompt=prompt_extract_store_names(),
        template_class=StoreNames,
        extraction_name="store_names"
    )

    # Extract details for each store (pad to ensure we have exactly 2)
    stores_details = []
    for i in range(2):
        if i < len(store_names_info.names):
            store_name = store_names_info.names[i]
            # Extract details for this store
            store_details = await evaluator.extract(
                prompt=prompt_extract_store_details(store_name),
                template_class=StoreDetails,
                extraction_name=f"store_{i}_details"
            )
            # Ensure store name is set correctly
            store_details.name = store_name
        else:
            # Create empty store details for missing stores
            store_details = StoreDetails(name=None, address=None, urls=[])
        
        stores_details.append(store_details)

    # -------- 3. Verify each store ------------------------------------ #
    for i, store in enumerate(stores_details):
        await verify_store(evaluator, root, i, store)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()