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
TASK_ID = "thanksgiving_travel_2025"
TASK_DESCRIPTION = (
    "You are planning a road trip from North Carolina through Virginia to Maryland on Thanksgiving Day 2025 "
    "(November 27, 2025). Identify food-related establishments across these states that are open on that date "
    "and meet specific operational criteria, with proper documentation and URLs."
)

THANKSGIVING_DATE_STR = "November 27, 2025"
THANKSGIVING_DATE_SHORT = "Thanksgiving Day 2025"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NCGroceryStore(BaseModel):
    chain_name: Optional[str] = None
    chain_reference_urls: List[str] = Field(default_factory=list)
    open_on_thanksgiving: Optional[str] = None  # "yes"/"no" or a phrase
    thanksgiving_hours: Optional[str] = None    # e.g., "7 AM - 3 PM"
    hours_reference_urls: List[str] = Field(default_factory=list)
    pharmacy_available: Optional[str] = None    # "yes"/"no" or a phrase
    services_reference_urls: List[str] = Field(default_factory=list)


class VARestaurant(BaseModel):
    chain_name: Optional[str] = None
    chain_reference_urls: List[str] = Field(default_factory=list)
    open_on_thanksgiving: Optional[str] = None  # "yes"/"no" or a phrase
    thanksgiving_hours: Optional[str] = None
    hours_reference_urls: List[str] = Field(default_factory=list)
    lunch_service_available: Optional[str] = None  # "yes"/"no" or a phrase


class MDConvenience(BaseModel):
    chain_name: Optional[str] = None
    chain_reference_urls: List[str] = Field(default_factory=list)
    open_on_thanksgiving: Optional[str] = None  # "yes"/"no" or a phrase
    policy_24_7: Optional[str] = None           # "yes"/"no" or a phrase (24/7 policy)
    thanksgiving_hours: Optional[str] = None
    hours_reference_urls: List[str] = Field(default_factory=list)


class TravelPlanExtraction(BaseModel):
    nc_stores: List[NCGroceryStore] = Field(default_factory=list)
    va_restaurants: List[VARestaurant] = Field(default_factory=list)
    md_convenience: Optional[MDConvenience] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_plan() -> str:
    return (
        f"Extract structured information from the answer to satisfy the Thanksgiving Day 2025 (i.e., {THANKSGIVING_DATE_STR}) "
        "requirements for North Carolina (grocery stores), Virginia (restaurants), and Maryland (convenience/pharmacy).\n\n"
        "Return a JSON with these fields:\n"
        "1) nc_stores: an array of up to 2 different major grocery store chains operating in North Carolina. "
        "   For each store, extract:\n"
        "   - chain_name: the chain name (string)\n"
        "   - chain_reference_urls: URLs (array) confirming the chain operates in North Carolina (e.g., store locator, corporate page). If none, empty array.\n"
        f"   - open_on_thanksgiving: whether the chain is open on {THANKSGIVING_DATE_SHORT} ('yes'/'no' or phrase)\n"
        f"   - thanksgiving_hours: the specific opening/closing hours on {THANKSGIVING_DATE_SHORT} (string; allow ranges or notes like 'varies by location' if stated)\n"
        "   - hours_reference_urls: URLs (array) documenting the Thanksgiving 2025 hours or open/closed status. If none, empty array.\n"
        f"   - pharmacy_available: whether pharmacy services are available on {THANKSGIVING_DATE_SHORT} ('yes'/'no' or phrase)\n"
        "   - services_reference_urls: URLs (array) documenting pharmacy service availability. If none, empty array.\n\n"
        "2) va_restaurants: an array of up to 2 different restaurant chains operating in Virginia. For each, extract:\n"
        "   - chain_name\n"
        "   - chain_reference_urls: URLs confirming the chain operates in Virginia. If none, empty array.\n"
        f"   - open_on_thanksgiving: whether the chain is open on {THANKSGIVING_DATE_SHORT}\n"
        f"   - thanksgiving_hours: specific Thanksgiving 2025 hours (string)\n"
        "   - hours_reference_urls: URLs documenting the Thanksgiving 2025 hours or open/closed status. If none, empty array.\n"
        f"   - lunch_service_available: whether lunch service (11:00 AM - 2:00 PM) is available on {THANKSGIVING_DATE_SHORT} ('yes'/'no' or phrase)\n\n"
        "3) md_convenience: one major convenience store OR pharmacy chain operating in Maryland. Extract:\n"
        "   - chain_name\n"
        "   - chain_reference_urls: URLs confirming the chain operates in Maryland. If none, empty array.\n"
        f"   - open_on_thanksgiving: whether it is open on {THANKSGIVING_DATE_SHORT}\n"
        "   - policy_24_7: whether it operates 24 hours (24/7) on Thanksgiving 2025 ('yes'/'no' or phrase)\n"
        f"   - thanksgiving_hours: the specific Thanksgiving 2025 hours (string, may be '24 hours')\n"
        "   - hours_reference_urls: URLs documenting the Thanksgiving 2025 open status and/or 24/7 policy. If none, empty array.\n\n"
        "GENERAL RULES:\n"
        "- Extract only what is explicitly present in the answer. Use 'null' for any missing field.\n"
        "- For URLs, extract actual links mentioned in the answer. If the answer references sources without explicit URLs, return an empty array.\n"
        "- If the answer mentions more than required items, keep only the first two NC stores and first two VA restaurants; for MD convenience/pharmacy, keep the first one.\n"
        "- Prefer official company sites, store locator pages, newsroom posts, or reliable holiday-hours articles if present in the answer.\n"
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    truthy = {"yes", "true", "y", "open", "available", "operational", "operating", "24/7", "24 hours", "24hr", "24-hours"}
    falsy = {"no", "false", "n", "closed", "not available", "unavailable"}
    if any(tok in v for tok in truthy):
        return True
    if any(tok in v for tok in falsy):
        return False
    return None


def _safe_chain(chain: Optional[str]) -> str:
    return chain.strip() if chain else "the chain"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_nc_store(
    evaluator: Evaluator,
    parent_node,
    store: NCGroceryStore,
    index: int,
    other_store_name: Optional[str] = None,
) -> None:
    # Store-level (sequential, critical)
    store_node = evaluator.add_sequential(
        id=f"NC_Grocery_Store_{index}",
        desc=("First North Carolina grocery store meeting all requirements" if index == 1
              else "Second North Carolina grocery store meeting all requirements (must be a different chain than Store 1)"),
        parent=parent_node,
        critical=True
    )

    # Chain Identification (parallel, critical)
    chain_ident_node = evaluator.add_parallel(
        id=f"NC_Store_{index}_Chain_Identification",
        desc=("Identify a major grocery store chain operating in North Carolina" if index == 1
              else "Identify a different major grocery store chain operating in North Carolina"),
        parent=store_node,
        critical=True
    )

    # Chain Name (existence + distinctness for #2)
    name_ok = _is_nonempty(store.chain_name)
    if index == 2 and _is_nonempty(other_store_name) and _is_nonempty(store.chain_name):
        name_ok = name_ok and (store.chain_name.strip().lower() != other_store_name.strip().lower())

    evaluator.add_custom_node(
        result=name_ok,
        id=f"NC_Store_{index}_Chain_Name",
        desc=("Provide the name of the grocery store chain"
              if index == 1 else "Provide the name of the grocery store chain (must differ from Store 1)"),
        parent=chain_ident_node,
        critical=True
    )

    # Chain Reference (presence of URLs)
    evaluator.add_custom_node(
        result=_urls_present(store.chain_reference_urls),
        id=f"NC_Store_{index}_Chain_Reference",
        desc="Provide URL reference confirming the chain operates in North Carolina",
        parent=chain_ident_node,
        critical=True
    )

    # Holiday Operations (parallel, critical)
    hol_ops_node = evaluator.add_parallel(
        id=f"NC_Store_{index}_Holiday_Operations",
        desc=f"Verify operational status and hours on {THANKSGIVING_DATE_SHORT}",
        parent=store_node,
        critical=True
    )

    # Hours Reference (presence)
    hours_ref_present = evaluator.add_custom_node(
        result=_urls_present(store.hours_reference_urls),
        id=f"NC_Store_{index}_Hours_Reference",
        desc=f"Provide URL reference confirming the {THANKSGIVING_DATE_SHORT} hours",
        parent=hol_ops_node,
        critical=True
    )

    # Open Status (verify with URLs; auto-preconditions include hours reference)
    open_leaf = evaluator.add_leaf(
        id=f"NC_Store_{index}_Open_Status",
        desc=f"Confirm the store is open on {THANKSGIVING_DATE_STR}",
        parent=hol_ops_node,
        critical=True
    )
    open_claim = f"The chain '{_safe_chain(store.chain_name)}' is open on {THANKSGIVING_DATE_STR}."
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=store.hours_reference_urls,
        additional_instruction=(
            "Use the provided URL(s) to verify the chain's Thanksgiving 2025 open/closed status. "
            "If the page explicitly says closed nationwide, mark as Incorrect. "
            "If it indicates special holiday hours or 'open', treat as Correct even if hours vary by location."
        )
    )

    # Operating Hours (verify exact/similar hours with URLs)
    hours_leaf = evaluator.add_leaf(
        id=f"NC_Store_{index}_Operating_Hours",
        desc=f"Provide specific opening and closing times for {THANKSGIVING_DATE_SHORT}",
        parent=hol_ops_node,
        critical=True
    )
    hours_text = store.thanksgiving_hours or ""
    hours_claim = (
        f"On {THANKSGIVING_DATE_STR}, '{_safe_chain(store.chain_name)}' operates with hours '{hours_text}'."
        if _is_nonempty(hours_text)
        else f"The referenced page documents specific Thanksgiving 2025 hours for '{_safe_chain(store.chain_name)}'."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=store.hours_reference_urls,
        additional_instruction=(
            "Verify that the page explicitly states Thanksgiving 2025 opening/closing times. "
            "Allow minor formatting differences. If hours vary by location but the page provides specific example hours "
            "matching the answer, consider Correct."
        )
    )

    # Services (parallel, critical — we require the status to be documented, not necessarily 'available')
    services_node = evaluator.add_parallel(
        id=f"NC_Store_{index}_Services",
        desc=f"Document available services on {THANKSGIVING_DATE_SHORT}",
        parent=store_node,
        critical=True
    )

    # Services Reference (presence)
    evaluator.add_custom_node(
        result=_urls_present(store.services_reference_urls),
        id=f"NC_Store_{index}_Services_Reference",
        desc="Provide URL reference for service availability information",
        parent=services_node,
        critical=True
    )

    # Pharmacy Status (verify claim that matches extracted status)
    pharm_leaf = evaluator.add_leaf(
        id=f"NC_Store_{index}_Pharmacy_Status",
        desc=f"Indicate whether pharmacy services are available on {THANKSGIVING_DATE_SHORT}",
        parent=services_node,
        critical=True
    )
    pharm_flag = _to_bool(store.pharmacy_available)
    if pharm_flag is True:
        pharm_claim = f"Pharmacy services are available on {THANKSGIVING_DATE_STR} for '{_safe_chain(store.chain_name)}'."
    elif pharm_flag is False:
        pharm_claim = f"Pharmacy services are NOT available on {THANKSGIVING_DATE_STR} for '{_safe_chain(store.chain_name)}'."
    else:
        pharm_claim = f"The referenced page documents pharmacy service availability status for '{_safe_chain(store.chain_name)}' on {THANKSGIVING_DATE_STR}."

    await evaluator.verify(
        claim=pharm_claim,
        node=pharm_leaf,
        sources=store.services_reference_urls,
        additional_instruction=(
            "Judge based on explicit statements about pharmacy holiday operations for Thanksgiving 2025. "
            "If the page clearly indicates 'open' or 'closed' for pharmacy on that date, follow it."
        )
    )


async def verify_va_restaurant(
    evaluator: Evaluator,
    parent_node,
    rest: VARestaurant,
    index: int,
    other_rest_name: Optional[str] = None,
) -> None:
    rest_node = evaluator.add_sequential(
        id=f"VA_Restaurant_{index}",
        desc=("First Virginia restaurant meeting all requirements" if index == 1
              else "Second Virginia restaurant meeting all requirements (must be a different chain than Restaurant 1)"),
        parent=parent_node,
        critical=True
    )

    # Chain Identification (parallel, critical)
    chain_ident_node = evaluator.add_parallel(
        id=f"VA_Restaurant_{index}_Chain_Identification",
        desc=("Identify a restaurant chain operating in Virginia" if index == 1
              else "Identify a different restaurant chain operating in Virginia"),
        parent=rest_node,
        critical=True
    )

    # Chain Name (existence + distinctness for #2)
    name_ok = _is_nonempty(rest.chain_name)
    if index == 2 and _is_nonempty(other_rest_name) and _is_nonempty(rest.chain_name):
        name_ok = name_ok and (rest.chain_name.strip().lower() != other_rest_name.strip().lower())

    evaluator.add_custom_node(
        result=name_ok,
        id=f"VA_Restaurant_{index}_Chain_Name",
        desc=("Provide the name of the restaurant chain"
              if index == 1 else "Provide the name of the restaurant chain (must differ from Restaurant 1)"),
        parent=chain_ident_node,
        critical=True
    )

    # Chain Reference (presence)
    evaluator.add_custom_node(
        result=_urls_present(rest.chain_reference_urls),
        id=f"VA_Restaurant_{index}_Chain_Reference",
        desc="Provide URL reference confirming the chain operates in Virginia",
        parent=chain_ident_node,
        critical=True
    )

    # Holiday Operations (parallel, critical)
    hol_ops_node = evaluator.add_parallel(
        id=f"VA_Restaurant_{index}_Holiday_Operations",
        desc=f"Verify operational status and hours on {THANKSGIVING_DATE_SHORT}",
        parent=rest_node,
        critical=True
    )

    # Hours Reference (presence)
    evaluator.add_custom_node(
        result=_urls_present(rest.hours_reference_urls),
        id=f"VA_Restaurant_{index}_Hours_Reference",
        desc=f"Provide URL reference confirming the {THANKSGIVING_DATE_SHORT} hours",
        parent=hol_ops_node,
        critical=True
    )

    # Open Status
    open_leaf = evaluator.add_leaf(
        id=f"VA_Restaurant_{index}_Open_Status",
        desc=f"Confirm the restaurant is open on {THANKSGIVING_DATE_STR}",
        parent=hol_ops_node,
        critical=True
    )
    open_claim = f"The restaurant chain '{_safe_chain(rest.chain_name)}' is open on {THANKSGIVING_DATE_STR}."
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=rest.hours_reference_urls,
        additional_instruction=(
            "Use the provided URL(s) to verify the Thanksgiving 2025 open/closed status for the chain. "
            "If explicitly closed, mark Incorrect; if open or special hours, mark Correct."
        )
    )

    # Operating Hours
    hours_leaf = evaluator.add_leaf(
        id=f"VA_Restaurant_{index}_Operating_Hours",
        desc=f"Provide specific opening and closing times for {THANKSGIVING_DATE_SHORT}",
        parent=hol_ops_node,
        critical=True
    )
    hours_text = rest.thanksgiving_hours or ""
    hours_claim = (
        f"On {THANKSGIVING_DATE_STR}, '{_safe_chain(rest.chain_name)}' operates with hours '{hours_text}'."
        if _is_nonempty(hours_text)
        else f"The referenced page documents specific Thanksgiving 2025 hours for '{_safe_chain(rest.chain_name)}'."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=rest.hours_reference_urls,
        additional_instruction=(
            "Verify the stated time range on Thanksgiving 2025. Allow minor formatting differences; "
            "if the page clearly documents hours, this should pass."
        )
    )

    # Lunch Service (11:00 AM - 2:00 PM)
    lunch_leaf = evaluator.add_leaf(
        id=f"VA_Restaurant_{index}_Lunch_Service",
        desc=f"Confirm the restaurant is open during lunch hours (11:00 AM - 2:00 PM) on {THANKSGIVING_DATE_SHORT}",
        parent=hol_ops_node,
        critical=True
    )
    lunch_claim = (
        f"On {THANKSGIVING_DATE_STR}, '{_safe_chain(rest.chain_name)}' is open during 11:00 AM to 2:00 PM "
        f"(i.e., lunch hours)."
    )
    await evaluator.verify(
        claim=lunch_claim,
        node=lunch_leaf,
        sources=rest.hours_reference_urls,
        additional_instruction=(
            "Confirm that Thanksgiving 2025 operating hours include the 11:00 AM–2:00 PM window. "
            "If open at any time covering this interval, mark Correct; otherwise, mark Incorrect."
        )
    )


async def verify_md_convenience(
    evaluator: Evaluator,
    parent_node,
    conv: MDConvenience,
) -> None:
    md_node = evaluator.add_sequential(
        id="Maryland_Convenience_Requirements",
        desc="Identify one major convenience store or pharmacy chain in Maryland that operates 24/7 including Thanksgiving 2025",
        parent=parent_node,
        critical=True
    )

    # Chain Identification (parallel, critical)
    chain_ident = evaluator.add_parallel(
        id="MD_Convenience_Chain_Identification",
        desc="Identify a convenience store or pharmacy chain operating in Maryland",
        parent=md_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(conv.chain_name),
        id="MD_Convenience_Chain_Name",
        desc="Provide the name of the convenience store or pharmacy chain",
        parent=chain_ident,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(conv.chain_reference_urls),
        id="MD_Convenience_Chain_Reference",
        desc="Provide URL reference confirming the chain operates in Maryland",
        parent=chain_ident,
        critical=True
    )

    # 24/7 Operations (parallel, critical)
    ops_node = evaluator.add_parallel(
        id="MD_Convenience_24_7_Operations",
        desc="Verify 24/7 operational status including Thanksgiving 2025",
        parent=md_node,
        critical=True
    )

    # Hours Reference (presence)
    evaluator.add_custom_node(
        result=_urls_present(conv.hours_reference_urls),
        id="MD_Convenience_Hours_Reference",
        desc=f"Provide URL reference confirming the 24/7 {THANKSGIVING_DATE_SHORT} operations",
        parent=ops_node,
        critical=True
    )

    # Open Status
    open_leaf = evaluator.add_leaf(
        id="MD_Convenience_Open_Status",
        desc=f"Confirm the establishment is open on {THANKSGIVING_DATE_STR}",
        parent=ops_node,
        critical=True
    )
    open_claim = f"The chain '{_safe_chain(conv.chain_name)}' is open on {THANKSGIVING_DATE_STR}."
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=conv.hours_reference_urls,
        additional_instruction=(
            "Use the provided URL(s) to verify open/closed status on Thanksgiving 2025. "
            "If the page indicates closure, mark Incorrect."
        )
    )

    # 24/7 Policy
    policy_leaf = evaluator.add_leaf(
        id="MD_Convenience_24_7_Policy",
        desc=f"Confirm the establishment operates 24 hours on {THANKSGIVING_DATE_SHORT}",
        parent=ops_node,
        critical=True
    )
    policy_claim = f"On {THANKSGIVING_DATE_STR}, '{_safe_chain(conv.chain_name)}' operates 24 hours (24/7)."
    await evaluator.verify(
        claim=policy_claim,
        node=policy_leaf,
        sources=conv.hours_reference_urls,
        additional_instruction=(
            "Verify that the page explicitly indicates 24/7 operations for Thanksgiving 2025 (or a general 24/7 policy "
            "that applies to holidays including Thanksgiving)."
        )
    )


# --------------------------------------------------------------------------- #
# Section orchestrators                                                       #
# --------------------------------------------------------------------------- #
async def build_nc_section(evaluator: Evaluator, root_node, extracted: TravelPlanExtraction) -> None:
    nc_node = evaluator.add_parallel(
        id="North_Carolina_Grocery_Requirements",
        desc="Identify two different major grocery store chains in North Carolina that are open on Thanksgiving 2025, with at least one offering active pharmacy services",
        parent=root_node,
        critical=True
    )

    # Prepare two stores (pad with empty if needed)
    stores = list(extracted.nc_stores[:2])
    while len(stores) < 2:
        stores.append(NCGroceryStore())

    # Verify each store
    store1 = stores[0]
    store2 = stores[1]
    await verify_nc_store(evaluator, nc_node, store1, index=1)
    await verify_nc_store(evaluator, nc_node, store2, index=2, other_store_name=store1.chain_name)

    # Critical requirement: at least one grocery store offers pharmacy services available on Thanksgiving 2025
    pharm1 = _to_bool(store1.pharmacy_available)
    pharm2 = _to_bool(store2.pharmacy_available)
    evaluator.add_custom_node(
        result=bool(pharm1 is True or pharm2 is True),
        id="NC_Pharmacy_Requirement",
        desc="Verify that at least one of the two North Carolina grocery stores offers pharmacy services on Thanksgiving 2025",
        parent=nc_node,
        critical=True
    )


async def build_va_section(evaluator: Evaluator, root_node, extracted: TravelPlanExtraction) -> None:
    va_node = evaluator.add_parallel(
        id="Virginia_Restaurant_Requirements",
        desc="Identify two different restaurant chains in Virginia that are open for lunch service on Thanksgiving 2025",
        parent=root_node,
        critical=True
    )

    restaurants = list(extracted.va_restaurants[:2])
    while len(restaurants) < 2:
        restaurants.append(VARestaurant())

    r1 = restaurants[0]
    r2 = restaurants[1]
    await verify_va_restaurant(evaluator, va_node, r1, index=1)
    await verify_va_restaurant(evaluator, va_node, r2, index=2, other_rest_name=r1.chain_name)


async def build_md_section(evaluator: Evaluator, root_node, extracted: TravelPlanExtraction) -> None:
    # Use provided convenience/pharmacy chain or empty one
    conv = extracted.md_convenience or MDConvenience()
    await verify_md_convenience(evaluator, root_node, conv)


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
    Evaluate an answer for the Thanksgiving travel 2025 requirements across NC, VA, and MD.
    """
    # Initialize evaluator (root non-critical parallel to allow partial credit across sections)
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
        prompt=prompt_extract_travel_plan(),
        template_class=TravelPlanExtraction,
        extraction_name="travel_plan_extraction",
    )

    # Add contextual info
    evaluator.add_ground_truth({
        "holiday": THANKSGIVING_DATE_STR,
        "requirements": {
            "NC": "Two grocery chains open with documented hours; at least one with pharmacy services available.",
            "VA": "Two restaurant chains open with documented hours; open during 11:00 AM–2:00 PM.",
            "MD": "One convenience/pharmacy chain open and operating 24/7 on Thanksgiving 2025 with documented policy."
        }
    })

    # Build and verify sections
    await build_nc_section(evaluator, root, extracted)
    await build_va_section(evaluator, root, extracted)
    await build_md_section(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()