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
TASK_ID = "grocery_senior_services"
TASK_DESCRIPTION = """I am looking for major grocery store chains that offer convenient services for seniors and provide multiple shopping options. Please identify four distinct national or regional grocery store chains that meet ALL of the following requirements:

1. Curbside Pickup Service: The store must offer curbside pickup or order pickup service with a minimum order requirement of $35 or less for free pickup (or no minimum at all).

2. Senior Discount Program: The store must have a senior discount program that:
   - Has an age eligibility requirement of 60 years or younger (such as 55+, 60+, etc.)
   - Offers a discount of at least 5% off

3. Pharmacy Services with Transfer Incentive: The store must have in-store pharmacy services and must offer a prescription transfer incentive or reward program (such as fuel points, discounts, gift cards, or loyalty rewards for transferring prescriptions).

For each of the four stores, provide:
- The name of the grocery store chain
- The curbside pickup minimum order requirement (or state "no minimum")
- The senior discount details (age requirement and discount percentage)
- The pharmacy prescription transfer incentive offered
- Direct URLs to the store's official pages documenting: (a) curbside pickup policy, (b) senior discount program, and (c) pharmacy services/prescription transfer offers
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreRecord(BaseModel):
    name: Optional[str] = None

    # Curbside / Pickup
    pickup_minimum: Optional[str] = None  # e.g., "$35", "no minimum", "35+", etc.
    pickup_urls: List[str] = Field(default_factory=list)

    # Senior discount
    senior_age_requirement: Optional[str] = None  # e.g., "55+", "60+"
    senior_discount_percent: Optional[str] = None  # e.g., "5%", "10%"
    senior_urls: List[str] = Field(default_factory=list)

    # Pharmacy
    pharmacy_transfer_incentive: Optional[str] = None  # e.g., "500 fuel points", "$25 gift card"
    pharmacy_urls: List[str] = Field(default_factory=list)


class StoresExtraction(BaseModel):
    stores: List[StoreRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
You will extract up to FOUR grocery store chains from the answer. For each, capture the key details exactly as stated and any official URLs provided.

Return a JSON object with a field "stores" which is an array of at most 4 objects. For each store object, extract:

- name: The grocery store chain name (not a single local store).
- pickup_minimum: The minimum order requirement (as text) for free curbside/order pickup, or "no minimum" if stated. If the answer mentions a fee waiver threshold (e.g., "free pickup for orders $35+"), capture that threshold text, not just a boolean.
- pickup_urls: A list of direct official URLs that document curbside or order pickup policy/details for that chain (if provided in the answer).
- senior_age_requirement: The minimum age text for the senior discount program (e.g., "55+", "60+"). If absent, set to null.
- senior_discount_percent: The discount amount text (e.g., "5%", "10%") for the senior program. If absent, set to null.
- senior_urls: A list of direct official URLs that document the chain's senior discount program (if provided).
- pharmacy_transfer_incentive: The text describing the prescription transfer incentive/reward (e.g., "500 fuel points", "$25 gift card"). If absent, set to null.
- pharmacy_urls: A list of direct official URLs that document pharmacy services and/or the prescription transfer offer (if provided).

Important:
- Extract ONLY what appears in the answer. Do not invent any data.
- If the answer contains more than four stores, include only the first four.
- If fewer than four are present, just return as many as are available (others will be treated as missing).
- For any field not present in the answer, set null (for strings) or [] (for URL lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]


def _ordinal(idx: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(idx, f"Store #{idx}")


def _union_urls(*lists: List[str]) -> List[str]:
    result: List[str] = []
    for lst in lists:
        for u in lst:
            su = u.strip()
            if su and su not in result:
                result.append(su)
    return result


# --------------------------------------------------------------------------- #
# Verification for a single store                                             #
# --------------------------------------------------------------------------- #
async def verify_store(
    evaluator: Evaluator,
    parent_node,
    store: StoreRecord,
    index_1based: int,
) -> None:
    store_id = f"Store_{index_1based}"
    title = f"{_ordinal(index_1based)} grocery store chain meeting all requirements"

    # Create the node for this store (parallel; non-critical to allow partial credit per store)
    store_node = evaluator.add_parallel(
        id=store_id,
        desc=title,
        parent=parent_node,
        critical=False
    )

    # Prepare sources
    pickup_urls = _nonempty_urls(store.pickup_urls)
    senior_urls = _nonempty_urls(store.senior_urls)
    pharmacy_urls = _nonempty_urls(store.pharmacy_urls)
    any_urls = _union_urls(pickup_urls, senior_urls, pharmacy_urls)
    store_name = store.name or f"Store #{index_1based}"

    # 1) Major chain check (critical leaf under this store)
    is_major_chain_node = evaluator.add_leaf(
        id=f"{store_id}_Is_Major_Chain",
        desc=f"{store_id} must be a major national or regional grocery chain (not a local independent store)",
        parent=store_node,
        critical=True
    )
    major_chain_claim = (
        f"'{store_name}' is a major national or regional grocery store chain brand (not a single local independent store)."
    )
    await evaluator.verify(
        claim=major_chain_claim,
        node=is_major_chain_node,
        sources=any_urls if any_urls else None,
        additional_instruction=(
            "Use the provided official pages to determine if this is a chain brand with multiple locations/regions served. "
            "Evidence such as a store locator, multiple locations, or references to regions served is sufficient."
        ),
    )

    # 2) Curbside Pickup requirements (critical group)
    curbside_group = evaluator.add_parallel(
        id=f"{store_id}_Curbside_Pickup_Services",
        desc=f"{store_id} curbside pickup service requirements",
        parent=store_node,
        critical=True
    )

    # 2.a) URL provided (critical existence)
    evaluator.add_custom_node(
        result=len(pickup_urls) > 0,
        id=f"{store_id}_Pickup_URL",
        desc=f"Provide URL to {store_id}'s official curbside pickup policy or service page",
        parent=curbside_group,
        critical=True
    )

    # 2.b) Offers curbside/order pickup (critical)
    offers_pickup_node = evaluator.add_leaf(
        id=f"{store_id}_Offers_Curbside_Pickup",
        desc=f"{store_id} must offer curbside pickup or order pickup service",
        parent=curbside_group,
        critical=True
    )
    offer_claim = (
        f"The official page(s) show that {store_name} offers curbside pickup or order pickup service."
    )
    # 2.c) Minimum requirement ≤ $35 or no minimum (critical)
    pickup_min_node = evaluator.add_leaf(
        id=f"{store_id}_Pickup_Minimum",
        desc=f"{store_id} curbside pickup must have a minimum order requirement of $35 or less for free pickup (or no minimum)",
        parent=curbside_group,
        critical=True
    )
    min_txt = store.pickup_minimum or ""
    min_claim = (
        f"The official curbside/order pickup page(s) indicate that free pickup is available with orders of $35 or less, "
        f"or that there is no minimum required. The answer states: '{min_txt}'."
    )

    await evaluator.batch_verify([
        (
            offer_claim,
            pickup_urls if pickup_urls else None,
            offers_pickup_node,
            "Accept synonymous terms like 'Order Pickup', 'Drive Up', 'Curbside Pickup', or 'Pickup & Go'."
        ),
        (
            min_claim,
            pickup_urls if pickup_urls else None,
            pickup_min_node,
            "Focus on the threshold for FREE pickup. Accept statements like 'free pickup on orders $35+' or 'no minimum'. "
            "If fees apply below a threshold, it still satisfies the requirement as long as free pickup is available at $35 or less."
        ),
    ])

    # 3) Senior + Pharmacy group (critical wrapper, contains two critical subgroups)
    senior_and_pharmacy_group = evaluator.add_parallel(
        id=f"{store_id}_Senior_And_Pharmacy",
        desc=f"{store_id} senior discount and pharmacy service requirements",
        parent=store_node,
        critical=True
    )

    # 3.1) Senior Discount subgroup (critical)
    senior_group = evaluator.add_parallel(
        id=f"{store_id}_Senior_Discount",
        desc=f"{store_id} senior discount program requirements",
        parent=senior_and_pharmacy_group,
        critical=True
    )

    # Senior URL existence (critical)
    evaluator.add_custom_node(
        result=len(senior_urls) > 0,
        id=f"{store_id}_Senior_URL",
        desc=f"Provide URL to {store_id}'s official senior discount policy page",
        parent=senior_group,
        critical=True
    )

    # Has senior program (critical)
    has_senior_node = evaluator.add_leaf(
        id=f"{store_id}_Has_Senior_Discount",
        desc=f"{store_id} must have a senior discount program",
        parent=senior_group,
        critical=True
    )
    has_senior_claim = f"The official page(s) show that {store_name} offers a senior discount program."
    # Age requirement ≤ 60 (critical)
    senior_age_node = evaluator.add_leaf(
        id=f"{store_id}_Senior_Age_Requirement",
        desc=f"{store_id} senior discount age eligibility is 60 years or younger (e.g., 55+, 60+)",
        parent=senior_group,
        critical=True
    )
    age_txt = store.senior_age_requirement or ""
    senior_age_claim = (
        f"The official page(s) state the senior discount eligibility age is 60 years or younger "
        f"(e.g., 55+, 60+). The answer lists: '{age_txt}'."
    )
    # Discount amount ≥ 5% (critical)
    senior_amt_node = evaluator.add_leaf(
        id=f"{store_id}_Senior_Discount_Amount",
        desc=f"{store_id} senior discount must be at least 5% off",
        parent=senior_group,
        critical=True
    )
    disc_txt = store.senior_discount_percent or ""
    senior_amt_claim = (
        f"The official page(s) state the senior discount amount is at least 5% off. "
        f"The answer lists: '{disc_txt}'."
    )

    await evaluator.batch_verify([
        (
            has_senior_claim,
            senior_urls if senior_urls else None,
            has_senior_node,
            "Look for terms like 'Senior Discount', 'Senior Day', 'Discount for seniors', or similar."
        ),
        (
            senior_age_claim,
            senior_urls if senior_urls else None,
            senior_age_node,
            "Verify that the minimum age requirement is 60 or younger (e.g., 55+, 60+). "
            "If multiple age tiers exist, the lowest qualifying age must be ≤ 60."
        ),
        (
            senior_amt_claim,
            senior_urls if senior_urls else None,
            senior_amt_node,
            "Verify that the discount is at least 5%. Accept day-specific promotions (e.g., 'Senior Day 10% off') "
            "as long as the advertised discount for seniors meets or exceeds 5%."
        ),
    ])

    # 3.2) Pharmacy subgroup (critical)
    pharmacy_group = evaluator.add_parallel(
        id=f"{store_id}_Pharmacy",
        desc=f"{store_id} pharmacy service requirements",
        parent=senior_and_pharmacy_group,
        critical=True
    )

    # Pharmacy URL existence (critical)
    evaluator.add_custom_node(
        result=len(pharmacy_urls) > 0,
        id=f"{store_id}_Pharmacy_URL",
        desc=f"Provide URL to {store_id}'s pharmacy services or prescription transfer offer page",
        parent=pharmacy_group,
        critical=True
    )

    # Has in-store pharmacy (critical)
    has_pharmacy_node = evaluator.add_leaf(
        id=f"{store_id}_Has_Pharmacy",
        desc=f"{store_id} must have in-store pharmacy services",
        parent=pharmacy_group,
        critical=True
    )
    has_pharmacy_claim = (
        f"The official page(s) show that {store_name} operates in-store pharmacy services."
    )

    # Transfer incentive present (critical)
    transfer_incentive_node = evaluator.add_leaf(
        id=f"{store_id}_Pharmacy_Transfer_Incentive",
        desc=f"{store_id} pharmacy must offer a prescription transfer incentive or reward",
        parent=pharmacy_group,
        critical=True
    )
    incentive_txt = store.pharmacy_transfer_incentive or ""
    transfer_incentive_claim = (
        f"The official page(s) indicate that the pharmacy offers an incentive or reward when transferring a prescription "
        f"(e.g., fuel points, discounts, gift cards, or loyalty rewards). The answer notes: '{incentive_txt}'."
    )

    await evaluator.batch_verify([
        (
            has_pharmacy_claim,
            pharmacy_urls if pharmacy_urls else None,
            has_pharmacy_node,
            "Confirm that in-store pharmacy services exist (e.g., pharmacy locations, refill services, immunizations in-store)."
        ),
        (
            transfer_incentive_claim,
            pharmacy_urls if pharmacy_urls else None,
            transfer_incentive_node,
            "Look for 'transfer' or 'new or transferred prescriptions' offers that reward customers (fuel points, discounts, gift cards, loyalty rewards). "
            "Limited-time or location-dependent offers still count if clearly stated."
        ),
    ])


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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'four grocery store chains with services for seniors' task.
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

    # Add a top-level node to match rubric grouping
    top_node = evaluator.add_parallel(
        id="Find_Four_Grocery_Stores",
        desc="Identify four distinct major grocery store chains that each meet all specified service requirements for curbside pickup, senior discounts, and pharmacy services",
        parent=root,
        critical=False  # Non-critical root to allow partial credit across stores
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction",
    )

    stores: List[StoreRecord] = list(extracted.stores or [])
    # Keep only the first 4; pad if fewer
    if len(stores) > 4:
        stores = stores[:4]
    while len(stores) < 4:
        stores.append(StoreRecord())

    evaluator.add_custom_info(
        info={
            "stores_provided_in_answer": len(extracted.stores) if extracted and extracted.stores is not None else 0,
            "stores_evaluated": 4
        },
        info_type="stats",
        info_name="extraction_counts"
    )

    # Build and verify each store subtree
    for idx in range(1, 5):
        await verify_store(evaluator, top_node, stores[idx - 1], idx)

    # Return evaluation summary
    return evaluator.get_summary()