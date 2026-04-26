import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "christmas_eve_il_shopping_2024"
TASK_DESCRIPTION = """
I need to plan my Christmas Eve 2024 shopping across four different retail stores in Illinois, each serving a specific purpose. Please identify four stores that meet the following requirements:

Store 1: A major retailer that has both a tire and automotive service center AND a pharmacy, is open on Christmas Eve 2024, and is located in Illinois.

Store 2: A retailer that offers pickup service with NO minimum order requirement, has a pharmacy, will be open past 6 PM on Christmas Eve 2024, and is located in Illinois.

Store 3: A warehouse club that offers an executive membership program with annual rewards on purchases, opens before 10 AM on Christmas Eve 2024, and is located in Illinois.

Store 4: A grocery retailer that provides delivery service, has a pharmacy, has confirmed operating hours on Christmas Eve 2024, and is located in Illinois.

For each store, provide the store name, specific Illinois location address, and a reference URL that confirms the store's services and Christmas Eve 2024 hours.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreEntry(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None  # Full single-line address (must be in Illinois)
    reference_urls: List[str] = Field(default_factory=list)
    christmas_eve_2024_hours_text: Optional[str] = None  # Any hours text mentioned for Dec 24, 2024 in the answer


class FourStoresExtraction(BaseModel):
    store1: Optional[StoreEntry] = None
    store2: Optional[StoreEntry] = None
    store3: Optional[StoreEntry] = None
    store4: Optional[StoreEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_four_stores() -> str:
    return """
    Extract the four Illinois store entries described in the answer. For each store, extract ONLY what is explicitly stated in the answer text.

    For each of the four stores, return:
    - name: The chain/store name (e.g., "Walmart Supercenter", "Target", "Costco", etc.)
    - address: A single-line full physical street address for the specific Illinois location provided in the answer. Include city and the state (preferably as "IL" or "Illinois") and ZIP if present. If the answer does not clearly provide a full address, return null.
    - reference_urls: An array of all URLs that the answer cites for THIS store to support services and/or Christmas Eve 2024 hours. Include only URLs explicitly present in the answer. Do not invent. Include both location pages and official policy/holiday pages if cited.
    - christmas_eve_2024_hours_text: If the answer states any hours or timing for Dec 24, 2024 (Christmas Eve) for this store/location (e.g., "open until 7 PM", "8am–6pm", etc.), copy that text exactly. Otherwise null.

    Output fields:
    - store1, store2, store3, store4: Each is an object with the above fields.

    Important:
    - Do NOT infer or add URLs or hours; extract only what appears in the answer.
    - If no URLs are provided for a store, return an empty array for reference_urls.
    - If a store is missing in the answer, set its object to null.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_il_address(addr: Optional[str]) -> bool:
    if not addr:
        return False
    txt = addr.strip()
    if not txt:
        return False
    return re.search(r"\b(IL|Illinois)\b", txt, flags=re.IGNORECASE) is not None


async def create_and_verify_leaf(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent: VerificationNode,
    claim: str,
    sources: Optional[List[str] | str],
    additional_instruction: str,
    critical: bool = True,
) -> VerificationNode:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction
    )
    return leaf


def safe_name(value: Optional[str]) -> str:
    return value or "the store"


def safe_addr(value: Optional[str]) -> str:
    return value or "the Illinois location"


# --------------------------------------------------------------------------- #
# Store verifications                                                         #
# --------------------------------------------------------------------------- #
async def verify_store_1(evaluator: Evaluator, parent: VerificationNode, s: Optional[StoreEntry]) -> None:
    store_node = evaluator.add_parallel(
        id="store_1",
        desc=("Store 1 requirements (major retailer; tire/auto service; pharmacy; open on Christmas Eve 2024; "
              "IL address; URL(s) confirming required services and Christmas Eve 2024 hours)."),
        parent=parent,
        critical=False
    )

    name = s.name if s else None
    addr = s.address if s else None
    urls = (s.reference_urls if s else []) or []

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="store_1_name_provided",
        desc="Provides the store name.",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_il_address(addr),
        id="store_1_specific_il_address_provided",
        desc="Provides a specific physical store address located in Illinois.",
        parent=store_node,
        critical=True
    )

    # Major retailer
    await create_and_verify_leaf(
        evaluator,
        node_id="store_1_is_major_retailer",
        desc="The identified store is a major retailer (as required by the question).",
        parent=store_node,
        claim=f"{safe_name(name)} is a major national or big-box retailer (widely recognized, many locations).",
        sources=urls,
        additional_instruction="Use the provided URLs to judge whether this brand is a widely recognized major retailer chain. Consider brands like Walmart, Target, Meijer, etc., as major retailers.",
        critical=True
    )

    # Tire/auto service center
    await create_and_verify_leaf(
        evaluator,
        node_id="store_1_tire_auto_service_center",
        desc="Store offers a tire and automotive service center.",
        parent=store_node,
        claim=f"The location at {safe_addr(addr)} offers a tire and automotive service center (e.g., Auto Care, Tire & Lube).",
        sources=urls,
        additional_instruction="Look for terms like 'Auto Care', 'Tire Center', 'Automotive Services', 'Tire & Lube'. The claim must apply to the specified Illinois location.",
        critical=True
    )

    # Pharmacy
    await create_and_verify_leaf(
        evaluator,
        node_id="store_1_pharmacy",
        desc="Store has a pharmacy.",
        parent=store_node,
        claim=f"The location at {safe_addr(addr)} has an in-store pharmacy.",
        sources=urls,
        additional_instruction="The page should indicate that this store has a pharmacy (Rx) at the specified IL location.",
        critical=True
    )

    # Open on Christmas Eve 2024
    await create_and_verify_leaf(
        evaluator,
        node_id="store_1_open_christmas_eve_2024",
        desc="Store is open on Christmas Eve 2024 (hours or open status stated).",
        parent=store_node,
        claim=f"The location at {safe_addr(addr)} is OPEN on December 24, 2024 (Christmas Eve) with hours listed for that date.",
        sources=urls,
        additional_instruction="Verify specifically for 12/24/2024 (Christmas Eve). A location-specific hours page or an official holiday-hours page that clearly applies to this location is acceptable.",
        critical=True
    )

    # Reference URLs confirmation (split into services + hours, both critical)
    ref_group = evaluator.add_parallel(
        id="store_1_reference_urls_confirm_services_and_hours",
        desc=("Provides verifiable URL reference(s) that confirm the required services (tire/auto service center and pharmacy) "
              "and Christmas Eve 2024 open status/hours for this specific location (or an official holiday-hours source "
              "that unambiguously applies to the location)."),
        parent=store_node,
        critical=True
    )
    await create_and_verify_leaf(
        evaluator,
        node_id="store_1_urls_confirm_services",
        desc="Reference URL(s) confirm both tire/auto service and pharmacy for the specified IL location.",
        parent=ref_group,
        claim=f"The provided pages confirm that the {safe_name(name)} location at {safe_addr(addr)} has BOTH a tire/auto service center and a pharmacy.",
        sources=urls,
        additional_instruction="It's acceptable if confirmation comes from separate provided URLs. This check focuses on service availability at the specified IL location.",
        critical=True
    )
    await create_and_verify_leaf(
        evaluator,
        node_id="store_1_urls_confirm_hours",
        desc="Reference URL(s) confirm Christmas Eve 2024 hours/open status for the specified IL location.",
        parent=ref_group,
        claim=f"The provided pages state hours or open status for the {safe_name(name)} location at {safe_addr(addr)} on December 24, 2024.",
        sources=urls,
        additional_instruction="Look for explicit 12/24/2024 hours or a clearly labeled 'Christmas Eve 2024' hours listing for this location.",
        critical=True
    )


async def verify_store_2(evaluator: Evaluator, parent: VerificationNode, s: Optional[StoreEntry]) -> None:
    store_node = evaluator.add_parallel(
        id="store_2",
        desc=("Store 2 requirements (pickup no-minimum; pharmacy; open past 6 PM Christmas Eve 2024; "
              "IL address; URL(s) confirming required services and Christmas Eve 2024 hours)."),
        parent=parent,
        critical=False
    )

    name = s.name if s else None
    addr = s.address if s else None
    urls = (s.reference_urls if s else []) or []

    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="store_2_name_provided",
        desc="Provides the store name.",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_il_address(addr),
        id="store_2_specific_il_address_provided",
        desc="Provides a specific physical store address located in Illinois.",
        parent=store_node,
        critical=True
    )

    # Pickup service with no minimum order requirement
    await create_and_verify_leaf(
        evaluator,
        node_id="store_2_pickup_no_minimum",
        desc="Store offers pickup service with no minimum order requirement.",
        parent=store_node,
        claim=(f"The {safe_name(name)} location at {safe_addr(addr)} offers order pickup with NO minimum order "
               f"requirement (no minimum purchase for pickup)."),
        sources=urls,
        additional_instruction="Look for 'no order minimum', 'no minimum for pickup', or equivalent wording. It should apply to pickup, not delivery.",
        critical=True
    )

    # Pharmacy
    await create_and_verify_leaf(
        evaluator,
        node_id="store_2_pharmacy",
        desc="Store has a pharmacy.",
        parent=store_node,
        claim=f"The location at {safe_addr(addr)} has an in-store pharmacy.",
        sources=urls,
        additional_instruction="The page should indicate pharmacy service at the specified IL location.",
        critical=True
    )

    # Open past 6 PM on Christmas Eve 2024
    await create_and_verify_leaf(
        evaluator,
        node_id="store_2_open_past_6pm_christmas_eve_2024",
        desc="Store is open past 6 PM on Christmas Eve 2024 (closing time later than 6 PM is stated).",
        parent=store_node,
        claim=f"On December 24, 2024, the {safe_name(name)} location at {safe_addr(addr)} closes AFTER 6:00 PM.",
        sources=urls,
        additional_instruction="Confirm that the closing time for 12/24/2024 is later than 6:00 PM (e.g., 7 PM or later).",
        critical=True
    )

    # Reference URLs confirmation (split into services + hours)
    ref_group = evaluator.add_parallel(
        id="store_2_reference_urls_confirm_services_and_hours",
        desc=("Provides verifiable URL reference(s) that confirm pickup no-minimum, pharmacy, and Christmas Eve 2024 "
              "hours showing open past 6 PM for this specific location (or an official holiday-hours source that "
              "unambiguously applies to the location)."),
        parent=store_node,
        critical=True
    )
    await create_and_verify_leaf(
        evaluator,
        node_id="store_2_urls_confirm_pickup_and_pharmacy",
        desc="Reference URL(s) confirm both pickup with no minimum and pharmacy for the specified IL location.",
        parent=ref_group,
        claim=(f"The provided pages confirm that the {safe_name(name)} location at {safe_addr(addr)} offers pickup with "
               f"no minimum order requirement AND has a pharmacy."),
        sources=urls,
        additional_instruction="It's acceptable if confirmation comes from separate provided URLs; this check focuses on both conditions being supported.",
        critical=True
    )
    await create_and_verify_leaf(
        evaluator,
        node_id="store_2_urls_confirm_hours_past_6pm",
        desc="Reference URL(s) confirm Christmas Eve 2024 hours showing close after 6 PM for the specified IL location.",
        parent=ref_group,
        claim=(f"The provided pages show that on December 24, 2024, the {safe_name(name)} location at {safe_addr(addr)} "
               f"closes after 6:00 PM."),
        sources=urls,
        additional_instruction="Look for hours specifically for 12/24/2024 with a closing time later than 6 PM.",
        critical=True
    )


async def verify_store_3(evaluator: Evaluator, parent: VerificationNode, s: Optional[StoreEntry]) -> None:
    store_node = evaluator.add_parallel(
        id="store_3",
        desc=("Store 3 requirements (warehouse club; executive membership w/ annual rewards; opens before 10 AM Christmas "
              "Eve 2024; IL address; URL(s) confirming executive membership and Christmas Eve 2024 hours)."),
        parent=parent,
        critical=False
    )

    name = s.name if s else None
    addr = s.address if s else None
    urls = (s.reference_urls if s else []) or []

    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="store_3_name_provided",
        desc="Provides the store name.",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_il_address(addr),
        id="store_3_specific_il_address_provided",
        desc="Provides a specific physical store address located in Illinois.",
        parent=store_node,
        critical=True
    )

    # Warehouse club
    await create_and_verify_leaf(
        evaluator,
        node_id="store_3_is_warehouse_club",
        desc="The identified store is a warehouse club (as required by the question).",
        parent=store_node,
        claim=f"{safe_name(name)} is a membership-based warehouse club retailer.",
        sources=urls,
        additional_instruction="Look for wording such as 'membership warehouse club' or similar descriptions for the chain.",
        critical=True
    )

    # Executive membership rewards
    await create_and_verify_leaf(
        evaluator,
        node_id="store_3_executive_membership_rewards",
        desc="Store offers an executive membership program with annual rewards on purchases.",
        parent=store_node,
        claim=f"{safe_name(name)} offers an 'Executive' (or similarly named) membership that earns annual rewards (e.g., 2%) on qualifying purchases.",
        sources=urls,
        additional_instruction="Official membership pages that apply chain-wide are acceptable if clearly applicable to all locations.",
        critical=True
    )

    # Opens before 10 AM on Christmas Eve 2024
    await create_and_verify_leaf(
        evaluator,
        node_id="store_3_opens_before_10am_christmas_eve_2024",
        desc="Store opening time on Christmas Eve 2024 is before 10:00 AM (opening time stated).",
        parent=store_node,
        claim=f"On December 24, 2024, the {safe_name(name)} location at {safe_addr(addr)} opens BEFORE 10:00 AM.",
        sources=urls,
        additional_instruction="Confirm the opening time for 12/24/2024 is earlier than 10:00 AM (e.g., 7 AM, 8 AM, 9 AM).",
        critical=True
    )

    # Reference URLs confirmation (split into membership + hours)
    ref_group = evaluator.add_parallel(
        id="store_3_reference_urls_confirm_membership_and_hours",
        desc=("Provides verifiable URL reference(s) that confirm the executive membership rewards program and "
              "Christmas Eve 2024 opening time (before 10 AM) for this specific location (or an official holiday-hours "
              "source that unambiguously applies to the location)."),
        parent=store_node,
        critical=True
    )
    await create_and_verify_leaf(
        evaluator,
        node_id="store_3_urls_confirm_exec_membership",
        desc="Reference URL(s) confirm the executive membership rewards program.",
        parent=ref_group,
        claim=(f"The provided pages confirm that {safe_name(name)} offers an Executive Membership (or similar) that "
               f"earns annual rewards on purchases."),
        sources=urls,
        additional_instruction="Chain-level membership program pages are acceptable if clearly official and applicable to all locations.",
        critical=True
    )
    await create_and_verify_leaf(
        evaluator,
        node_id="store_3_urls_confirm_hours_before_10am",
        desc="Reference URL(s) confirm Christmas Eve 2024 opening time is before 10:00 AM for the specified IL location.",
        parent=ref_group,
        claim=(f"The provided pages show that on December 24, 2024, the {safe_name(name)} location at {safe_addr(addr)} "
               f"opens before 10:00 AM."),
        sources=urls,
        additional_instruction="Look for explicit hours for 12/24/2024 with opening earlier than 10 AM.",
        critical=True
    )


async def verify_store_4(evaluator: Evaluator, parent: VerificationNode, s: Optional[StoreEntry]) -> None:
    store_node = evaluator.add_parallel(
        id="store_4",
        desc=("Store 4 requirements (grocery retailer; delivery; pharmacy; confirmed Christmas Eve 2024 hours; "
              "IL address; URL(s) confirming required services and Christmas Eve 2024 hours)."),
        parent=parent,
        critical=False
    )

    name = s.name if s else None
    addr = s.address if s else None
    urls = (s.reference_urls if s else []) or []

    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="store_4_name_provided",
        desc="Provides the store name.",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_il_address(addr),
        id="store_4_specific_il_address_provided",
        desc="Provides a specific physical store address located in Illinois.",
        parent=store_node,
        critical=True
    )

    # Grocery retailer
    await create_and_verify_leaf(
        evaluator,
        node_id="store_4_is_grocery_retailer",
        desc="The identified store is a grocery retailer (as required by the question).",
        parent=store_node,
        claim=f"{safe_name(name)} is a grocery retailer (primary business is selling groceries).",
        sources=urls,
        additional_instruction="Use provided URLs to confirm it is a grocery store chain (e.g., Jewel-Osco, Mariano's, ALDI, etc.).",
        critical=True
    )

    # Delivery service
    await create_and_verify_leaf(
        evaluator,
        node_id="store_4_delivery_service",
        desc="Store provides delivery service.",
        parent=store_node,
        claim=f"The {safe_name(name)} location at {safe_addr(addr)} offers grocery delivery service.",
        sources=urls,
        additional_instruction="Look for delivery options (e.g., Delivery, same-day delivery, Instacart, etc.) specifically applicable to the IL location.",
        critical=True
    )

    # Pharmacy
    await create_and_verify_leaf(
        evaluator,
        node_id="store_4_pharmacy",
        desc="Store has a pharmacy.",
        parent=store_node,
        claim=f"The location at {safe_addr(addr)} has an in-store pharmacy.",
        sources=urls,
        additional_instruction="Confirm pharmacy availability for the specific IL location.",
        critical=True
    )

    # Confirmed Christmas Eve 2024 hours
    await create_and_verify_leaf(
        evaluator,
        node_id="store_4_confirmed_christmas_eve_2024_hours",
        desc=("Christmas Eve 2024 operating hours for the store are stated/confirmed by a source "
              "(hours shown for 12/24/2024 or explicitly stated as Christmas Eve 2024 hours)."),
        parent=store_node,
        claim=f"The {safe_name(name)} location at {safe_addr(addr)} has hours explicitly listed for December 24, 2024.",
        sources=urls,
        additional_instruction="Look for explicit 12/24/2024 hours or 'Christmas Eve 2024' hours; location-specific or an official chain-level hours page that clearly applies.",
        critical=True
    )

    # Reference URLs confirmation (split into services + hours)
    ref_group = evaluator.add_parallel(
        id="store_4_reference_urls_confirm_services_and_hours",
        desc=("Provides verifiable URL reference(s) that confirm delivery service, pharmacy, and the Christmas Eve 2024 "
              "operating hours for this specific location (or an official holiday-hours source that unambiguously applies "
              "to the location)."),
        parent=store_node,
        critical=True
    )
    await create_and_verify_leaf(
        evaluator,
        node_id="store_4_urls_confirm_delivery_and_pharmacy",
        desc="Reference URL(s) confirm both delivery service and pharmacy for the specified IL location.",
        parent=ref_group,
        claim=(f"The provided pages confirm that the {safe_name(name)} location at {safe_addr(addr)} offers delivery "
               f"service and has a pharmacy."),
        sources=urls,
        additional_instruction="It's acceptable if confirmation comes from separate provided URLs; this check focuses on both services.",
        critical=True
    )
    await create_and_verify_leaf(
        evaluator,
        node_id="store_4_urls_confirm_hours_12242024",
        desc="Reference URL(s) confirm Christmas Eve 2024 operating hours for the specified IL location.",
        parent=ref_group,
        claim=(f"The provided pages show operating hours on December 24, 2024, for the {safe_name(name)} location at "
               f"{safe_addr(addr)}."),
        sources=urls,
        additional_instruction="Look for explicit 12/24/2024 hours or clearly labeled 'Christmas Eve 2024' hours for the location.",
        critical=True
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'Christmas Eve 2024 four Illinois stores' planning task.
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

    # 1) Extract structured store info from the answer
    extracted: FourStoresExtraction = await evaluator.extract(
        prompt=prompt_extract_four_stores(),
        template_class=FourStoresExtraction,
        extraction_name="four_stores_extraction"
    )

    # 2) Build verification subtrees for each store
    await verify_store_1(evaluator, root, extracted.store1)
    await verify_store_2(evaluator, root, extracted.store2)
    await verify_store_3(evaluator, root, extracted.store3)
    await verify_store_4(evaluator, root, extracted.store4)

    # 3) Return evaluation summary
    return evaluator.get_summary()