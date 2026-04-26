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
TASK_ID = "hou_christmas_eve_stores_2024"
TASK_DESCRIPTION = (
    "You are planning last-minute Christmas Eve shopping in Houston, Texas on December 24, 2024. "
    "To help you plan your shopping route efficiently, find 4 retail stores in Houston that are open on Christmas Eve 2024, "
    "with each store from a different category: (1) a major grocery store chain, (2) a home improvement store, "
    "(3) a pharmacy, and (4) a pet supply store. For each store, provide the following information: the store name "
    "(including the chain name), the complete street address of a specific location in Houston, Texas, the time the store "
    "closes on Christmas Eve 2024 (December 24, 2024), and a reference URL that verifies the store's Christmas Eve hours. "
    "Note: The stores must be from 4 different retail chains, and each must represent one of the 4 required categories."
)

REQUIRED_CATEGORIES = ["grocery", "home_improvement", "pharmacy", "pet_supply"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreInfo(BaseModel):
    chain: Optional[str] = None  # Chain/brand name (e.g., Kroger, Lowe's, CVS, PetSmart)
    store_name: Optional[str] = None  # Store/location name as written in the answer (include chain)
    category: Optional[str] = None  # One of: grocery, home_improvement, pharmacy, pet_supply
    address: Optional[str] = None  # Complete street address in Houston, TX
    closing_time: Optional[str] = None  # Closing time on Dec 24, 2024 (string, e.g., "6 PM")
    reference_url: Optional[str] = None  # URL verifying Christmas Eve hours for this store/location


class StoresExtraction(BaseModel):
    grocery: Optional[StoreInfo] = None
    home_improvement: Optional[StoreInfo] = None
    pharmacy: Optional[StoreInfo] = None
    pet_supply: Optional[StoreInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
Extract exactly four stores mentioned in the answer, one for each of the following categories:
- grocery (must be a major grocery store chain)
- home_improvement (e.g., big-box hardware/home improvement chains)
- pharmacy (retail pharmacy chains)
- pet_supply (pet supply chains)

For each category, return a JSON object with these fields:
- chain: the retail chain/brand name (e.g., "Kroger", "Lowe's", "CVS", "PetSmart"). If only a store name is provided, infer the chain from it.
- store_name: the store/location name as written in the answer (include the chain name if present).
- category: one of exactly ["grocery","home_improvement","pharmacy","pet_supply"] corresponding to the category slot.
- address: the complete street address for a specific location in Houston, Texas (including city and state).
- closing_time: the time this store closes on December 24, 2024 (e.g., "6 PM", "8:00 p.m.", "10pm"). Keep as a string exactly as written or summarized.
- reference_url: the URL cited in the answer that verifies the Christmas Eve 2024 hours for this location or at least for this chain.

Output JSON fields must be:
{
  "grocery": { StoreInfo },
  "home_improvement": { StoreInfo },
  "pharmacy": { StoreInfo },
  "pet_supply": { StoreInfo }
}

Rules:
1) Do not invent information. Only extract what is explicitly present in the answer.
2) If a field is missing in the answer for a category, set it to null.
3) If the chain isn’t explicitly named but can be clearly inferred from the store name, fill it in.
4) If multiple stores are provided for one category, pick the first mentioned for that category.
5) The address must be a specific Houston, TX street address. If missing or incomplete, set to null.
6) The closing_time must reflect December 24, 2024 (Christmas Eve) specifically; if not available, set to null.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_chain_name(chain: Optional[str], fallback: Optional[str]) -> Optional[str]:
    base = chain or fallback
    if not base:
        return None
    return base.strip().lower()


def _store_present(store: Optional[StoreInfo]) -> bool:
    return bool(store and (store.chain or store.store_name) and store.category)


# --------------------------------------------------------------------------- #
# Verification logic per store                                                #
# --------------------------------------------------------------------------- #
async def verify_store(
    evaluator: Evaluator,
    parent_node,
    store: Optional[StoreInfo],
    node_prefix: str,
    category_label: str
) -> None:
    """
    Build verification nodes for a single store and run URL-grounded verifications.

    node_prefix examples:
      - "Store_1" for grocery
      - "Store_2" for home improvement
      - "Store_3" for pharmacy
      - "Store_4" for pet supply

    category_label examples:
      - "grocery", "home_improvement", "pharmacy", "pet_supply"
    """
    # Parent node for this store
    store_parent = evaluator.add_parallel(
        id=f"{node_prefix}_{'Grocery' if category_label=='grocery' else 'Home_Improvement' if category_label=='home_improvement' else 'Pharmacy' if category_label=='pharmacy' else 'Pet_Supply'}",
        desc=f"Verify information for the {'grocery store' if category_label=='grocery' else 'home improvement store' if category_label=='home_improvement' else 'pharmacy' if category_label=='pharmacy' else 'pet supply store'}",
        parent=parent_node,
        critical=False
    )

    # Reference existence check (critical)
    ref_exists_node = evaluator.add_custom_node(
        result=bool(store and store.reference_url and store.reference_url.strip().startswith("http")),
        id=f"{node_prefix}_Reference",
        desc="A reference URL is provided that verifies the store information",
        parent=store_parent,
        critical=True
    )

    # Prepare common variables
    store_name = store.store_name if store else ""
    chain = store.chain if store else ""
    address = store.address if store else ""
    closing_time = store.closing_time if store else ""
    url = (store.reference_url or "") if store else ""

    # 1) Name + Category verification (critical)
    name_cat_node = evaluator.add_leaf(
        id=f"{node_prefix}_Name_Category",
        desc=f"The store is identified by name and is a {'major ' if category_label=='grocery' else ''}{category_label.replace('_',' ')} store chain",
        parent=store_parent,
        critical=True
    )
    name_cat_claim = (
        f"The webpage shows that this location is '{store_name}' and belongs to the chain '{chain}'. "
        f"This chain is a {('major ' if category_label=='grocery' else '')}{category_label.replace('_',' ')} retailer."
    )
    await evaluator.verify(
        claim=name_cat_claim,
        node=name_cat_node,
        sources=url,
        additional_instruction=(
            "Verify the brand/chain identity from the page. "
            "For category confirmation, infer from the brand identity or page content (e.g., grocery, home improvement, pharmacy, pet supply). "
            "Allow reasonable variants in store naming. Focus on whether the brand matches the category."
        ),
        extra_prerequisites=[ref_exists_node]
    )

    # 2) Location verification (critical)
    location_node = evaluator.add_leaf(
        id=f"{node_prefix}_Location",
        desc="A complete street address in Houston, Texas is provided",
        parent=store_parent,
        critical=True
    )
    location_claim = (
        f"The webpage shows the address for this location as '{address}', and it is in Houston, Texas."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=url,
        additional_instruction=(
            "Confirm the address is a specific street address in Houston, TX. "
            "Allow minor formatting differences (e.g., abbreviations like 'TX' vs 'Texas', punctuation, ZIP present or missing). "
            "Fail if the page clearly suggests a different city or lacks a specific address."
        ),
        extra_prerequisites=[ref_exists_node]
    )

    # 3) Open status on Dec 24, 2024 (critical)
    open_status_node = evaluator.add_leaf(
        id=f"{node_prefix}_Open_Status",
        desc="The store is confirmed to be open on Christmas Eve 2024 (December 24, 2024)",
        parent=store_parent,
        critical=True
    )
    open_status_claim = (
        f"The webpage indicates the location is open on December 24, 2024 (Christmas Eve)."
    )
    await evaluator.verify(
        claim=open_status_claim,
        node=open_status_node,
        sources=url,
        additional_instruction=(
            "Look for holiday/Christmas Eve hours for Dec 24, 2024. "
            "If the page explicitly lists special hours (e.g., 'Open until X' or 'Special hours' for Dec 24), consider it open. "
            "If the page indicates 'Closed' for Dec 24, then this claim is incorrect."
        ),
        extra_prerequisites=[ref_exists_node]
    )

    # 4) Closing time on Dec 24, 2024 (critical)
    closing_node = evaluator.add_leaf(
        id=f"{node_prefix}_Closing_Time",
        desc="An accurate Christmas Eve closing time is provided",
        parent=store_parent,
        critical=True
    )
    closing_claim = (
        f"On December 24, 2024 (Christmas Eve), the store closes at '{closing_time}'."
    )
    await evaluator.verify(
        claim=closing_claim,
        node=closing_node,
        sources=url,
        additional_instruction=(
            "Verify the closing time for Dec 24, 2024. "
            "Allow reasonable format variations (e.g., '5 PM', '5:00 p.m.', '17:00'). "
            "If multiple locations/hours are listed, ensure the time corresponds to the specific Houston location."
        ),
        extra_prerequisites=[ref_exists_node]
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Houston Christmas Eve 2024 store-hours task.
    """
    # Initialize evaluator with parallel root
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

    # Extract structured store info
    stores = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction"
    )

    # Add a critical node: Different Chains
    # Compute chain uniqueness across available stores (fallback to store_name if chain missing)
    chains: List[Optional[str]] = []
    for cat in REQUIRED_CATEGORIES:
        store: Optional[StoreInfo] = getattr(stores, cat, None)
        chains.append(_normalize_chain_name(store.chain if store else None, store.store_name if store else None))

    unique_chains = set([c for c in chains if c])
    different_chains_ok = (len(unique_chains) == 4)

    evaluator.add_custom_node(
        result=different_chains_ok,
        id="Different_Chains",
        desc="The 4 stores are from 4 different retail chains (not multiple locations of the same chain)",
        parent=root,
        critical=True
    )

    # Verify each store by category
    await verify_store(
        evaluator=evaluator,
        parent_node=root,
        store=stores.grocery,
        node_prefix="Store_1",
        category_label="grocery"
    )

    await verify_store(
        evaluator=evaluator,
        parent_node=root,
        store=stores.home_improvement,
        node_prefix="Store_2",
        category_label="home_improvement"
    )

    await verify_store(
        evaluator=evaluator,
        parent_node=root,
        store=stores.pharmacy,
        node_prefix="Store_3",
        category_label="pharmacy"
    )

    await verify_store(
        evaluator=evaluator,
        parent_node=root,
        store=stores.pet_supply,
        node_prefix="Store_4",
        category_label="pet_supply"
    )

    # Optional: add custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_chains": chains,
            "required_categories": REQUIRED_CATEGORIES
        },
        info_type="extraction_meta",
        info_name="extraction_overview"
    )

    # Return evaluation summary
    return evaluator.get_summary()