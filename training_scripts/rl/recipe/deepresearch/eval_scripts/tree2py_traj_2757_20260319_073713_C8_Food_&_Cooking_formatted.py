import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fl_grocery_guide"
TASK_DESCRIPTION = """I'm planning to create a comprehensive grocery shopping guide for families in Florida who rely on various grocery store services. I need to find 4 major grocery store chain locations across Florida (each in a different city) that offer a complete range of services and conveniences.

For each of the 4 stores, please provide:

1. Location Information:
   - Full street address
   - A reference URL to the store's official page or a reliable source confirming the location

2. In-Store Services: Verify that the store has:
   - An in-store pharmacy
   - A deli department
   - A bakery department

3. Customer Convenience Features: Verify that the store offers:
   - Curbside pickup service
   - Same-day delivery service
   - Accepts EBT/SNAP payments

4. Additional Programs: Verify that the store has:
   - A customer loyalty/rewards program
   - An attached fuel center/gas station
   - An organic produce section

5. Operating Hours: Provide:
   - Regular weekly operating hours
   - Thanksgiving Day status (open with hours, or closed)
   - Christmas Day status (open with hours, or closed)

The stores should be from major grocery chains (such as Publix, Kroger banners, Albertsons, Walmart Supercenter, Target with grocery, or similar national/regional chains), and each store should be located in a different Florida city.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreLocation(BaseModel):
    chain: Optional[str] = None
    store_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    reference_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class Services(BaseModel):
    pharmacy: Optional[bool] = None
    deli: Optional[bool] = None
    bakery: Optional[bool] = None


class Convenience(BaseModel):
    curbside_pickup: Optional[bool] = None
    same_day_delivery: Optional[bool] = None
    ebt_snap: Optional[bool] = None


class Programs(BaseModel):
    loyalty_program: Optional[bool] = None
    fuel_center: Optional[bool] = None
    organic_produce_section: Optional[bool] = None


class Hours(BaseModel):
    regular_hours: Optional[str] = None
    thanksgiving: Optional[str] = None
    christmas: Optional[str] = None


class StoreInfo(BaseModel):
    location: Optional[StoreLocation] = None
    services: Optional[Services] = None
    convenience: Optional[Convenience] = None
    programs: Optional[Programs] = None
    hours: Optional[Hours] = None


class GroceryExtraction(BaseModel):
    stores: List[StoreInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
    Extract up to FOUR (4) grocery store locations described in the answer, prioritizing the first four mentioned.
    Each store must be a major grocery chain location in Florida (e.g., Publix, Walmart Supercenter, Target with Grocery, Winn-Dixie, Whole Foods, Aldi, Sprouts, The Fresh Market, Albertsons banners, etc.), not a convenience store or standalone pharmacy.
    Return JSON with a top-level field "stores": an array of store objects, each with the following nested structure:

    {
      "location": {
        "chain": string or null,                // e.g., "Publix", "Walmart Supercenter", "Target", "Aldi"
        "store_name": string or null,           // if provided (e.g., "Publix at Midtown")
        "address": string or null,              // full street address, ideally including city, FL, and ZIP if provided
        "city": string or null,                 // Florida city
        "state": string or null,                // "FL" or "Florida" if provided
        "zipcode": string or null,
        "reference_url": string or null,        // primary official store page or reliable locator entry URL explicitly mentioned in the answer
        "additional_urls": [string, ...]        // any other URLs for this store mentioned in the answer (e.g., Instacart, Google Maps, Yelp, chain locator)
      },
      "services": {
        "pharmacy": boolean or null,            // true if the answer asserts the store HAS an in-store pharmacy
        "deli": boolean or null,                // true if the answer asserts the store HAS a deli department
        "bakery": boolean or null               // true if the answer asserts the store HAS a bakery department
      },
      "convenience": {
        "curbside_pickup": boolean or null,     // true if the answer asserts curbside pickup / drive-up / pickup is offered
        "same_day_delivery": boolean or null,   // true if the answer asserts same-day delivery is available (e.g., via Instacart)
        "ebt_snap": boolean or null             // true if the answer asserts SNAP/EBT is accepted at this location
      },
      "programs": {
        "loyalty_program": boolean or null,     // true if the answer asserts a customer loyalty/rewards program (chain-level acceptable)
        "fuel_center": boolean or null,         // true if the answer asserts an attached fuel center/gas station for THIS location
        "organic_produce_section": boolean or null // true if the answer asserts an organic produce section is offered
      },
      "hours": {
        "regular_hours": string or null,        // weekly hours as written or summarized in the answer
        "thanksgiving": string or null,         // "closed" OR "open with hours X–Y" (or similar) as claimed in the answer
        "christmas": string or null             // "closed" OR "open with hours X–Y" (or similar) as claimed in the answer
      }
    }

    Rules:
    - Extract ONLY what is explicitly stated in the answer. Do not infer or invent.
    - Use JSON booleans (true/false) for the yes/no style fields when the answer is explicit; otherwise use null.
    - For URLs, return only well-formed URLs that are explicitly present in the answer text (plain URLs or within markdown).
    - If the answer lists more than four stores, include only the first four in order. If fewer than four are given, extract all available.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def store_sources_list(store: StoreInfo) -> List[str]:
    urls: List[str] = []
    if store and store.location:
        if store.location.reference_url:
            urls.append(store.location.reference_url)
        if store.location.additional_urls:
            urls.extend([u for u in store.location.additional_urls if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        nu = u.strip()
        if nu and nu not in seen:
            out.append(nu)
            seen.add(nu)
    return out


def store_display_name(store: StoreInfo, idx: int) -> str:
    loc = store.location or StoreLocation()
    parts = []
    if loc.chain:
        parts.append(loc.chain)
    if loc.store_name:
        parts.append(f"({loc.store_name})")
    city = loc.city or "Unknown city"
    state = loc.state or "FL"
    label = " ".join(parts).strip() or f"Store #{idx + 1}"
    return f"{label} in {city}, {state}".strip()


def normalize_city(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    # Light normalization
    n = n.replace("  ", " ").replace(",", "")
    return n if n else None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_store(
    evaluator: Evaluator,
    parent_node,
    store: StoreInfo,
    store_index: int,
) -> None:
    """
    Build verification subtree for a single store according to the rubric.
    """
    # Parent node for this store (non-critical to allow partial credit per store)
    store_node = evaluator.add_parallel(
        id=f"store_{store_index + 1}",
        desc=f"{['First','Second','Third','Fourth'][store_index]} grocery store location meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Collect basic info
    loc = store.location or StoreLocation()
    services = store.services or Services()
    convenience = store.convenience or Convenience()
    programs = store.programs or Programs()
    hours = store.hours or Hours()
    sources = store_sources_list(store)
    sources_or_none: List[str] | str | None = sources if sources else (loc.reference_url if loc.reference_url else None)
    label = store_display_name(store, store_index)

    # 1) Major chain check (critical leaf)
    chain_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_major_chain",
        desc=f"Store {store_index + 1} is a major grocery chain, not a convenience store or pharmacy",
        parent=store_node,
        critical=True
    )
    chain_name = loc.chain or "the listed brand"
    claim_chain = (
        f"The page(s) show that {label} is a full-service supermarket/grocery store location belonging to a "
        f"major national or large regional grocery chain (not a standalone pharmacy or convenience store). "
        f"Brand: {chain_name}."
    )
    await evaluator.verify(
        claim=claim_chain,
        node=chain_leaf,
        sources=sources_or_none,
        additional_instruction=(
            "Accept evidence that the page or locator entry clearly indicates this is a 'supermarket', 'grocery store', "
            "'supercenter', or equivalent. Chain-owned store pages or recognized store locator entries are preferred. "
            "If the page explicitly presents itself as a convenience store only or a standalone pharmacy, this should fail."
        ),
        majority_vote=False
    )

    # 2) Location Information (critical parallel group)
    loc_group = evaluator.add_parallel(
        id=f"store_{store_index + 1}_location_info",
        desc=f"Complete location information for Store {store_index + 1}",
        parent=store_node,
        critical=True
    )

    # 2.a) Address provided & supported
    address_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_address",
        desc=f"Full street address provided for Store {store_index + 1}",
        parent=loc_group,
        critical=True
    )
    full_address = (loc.address or "").strip()
    claim_address = (
        f"The official or reliable page confirms the store's street address as '{full_address}'. "
        f"If the answer shortened or formatted it slightly differently (e.g., abbreviations like 'Rd.' vs 'Road'), "
        f"treat it as a match if clearly equivalent."
    )
    await evaluator.verify(
        claim=claim_address,
        node=address_leaf,
        sources=sources_or_none,
        additional_instruction="Verify the address equivalence with tolerance to minor punctuation and common abbreviations. The store must be in Florida.",
        majority_vote=False
    )

    # 2.b) Reference URL validity
    refurl_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_reference_url",
        desc=f"Valid reference URL to store's official page or reliable source",
        parent=loc_group,
        critical=True
    )
    ref_url = loc.reference_url or ""
    claim_refurl = (
        f"This webpage is an official store page or a reliable locator/business listing explicitly about {label} "
        f"(ideally showing the same address). URL: {ref_url}"
    )
    await evaluator.verify(
        claim=claim_refurl,
        node=refurl_leaf,
        sources=ref_url if ref_url else sources_or_none,
        additional_instruction=(
            "Accept chain-owned store locator pages, the store's official detail page, or well-known business listings "
            "with clear, matching details (e.g., Google Maps business profile, Instacart store page). The page should "
            "be specifically about this location (not generic chain info only)."
        ),
        majority_vote=False
    )

    # 3) In-Store Services (critical parallel group)
    svc_group = evaluator.add_parallel(
        id=f"store_{store_index + 1}_in_store_services",
        desc=f"Required in-store service departments for Store {store_index + 1}",
        parent=store_node,
        critical=True
    )
    # Pharmacy
    pharmacy_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_pharmacy",
        desc=f"Store {store_index + 1} has an in-store pharmacy",
        parent=svc_group,
        critical=True
    )
    claim_pharmacy = f"{label} offers an in-store pharmacy at this location."
    await evaluator.verify(
        claim=claim_pharmacy,
        node=pharmacy_leaf,
        sources=sources_or_none,
        additional_instruction="Look for 'Pharmacy' listed as a service/department on the store page or locator entry.",
        majority_vote=False
    )
    # Deli
    deli_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_deli",
        desc=f"Store {store_index + 1} has a deli department",
        parent=svc_group,
        critical=True
    )
    claim_deli = f"{label} has a deli department."
    await evaluator.verify(
        claim=claim_deli,
        node=deli_leaf,
        sources=sources_or_none,
        additional_instruction="Look for 'Deli' among departments/services or similar wording (prepared foods, deli counter).",
        majority_vote=False
    )
    # Bakery
    bakery_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_bakery",
        desc=f"Store {store_index + 1} has a bakery department",
        parent=svc_group,
        critical=True
    )
    claim_bakery = f"{label} has a bakery department."
    await evaluator.verify(
        claim=claim_bakery,
        node=bakery_leaf,
        sources=sources_or_none,
        additional_instruction="Look for 'Bakery' among departments/services.",
        majority_vote=False
    )

    # 4) Customer Convenience Features (critical parallel group)
    conv_group = evaluator.add_parallel(
        id=f"store_{store_index + 1}_customer_convenience",
        desc=f"Required convenience features for Store {store_index + 1}",
        parent=store_node,
        critical=True
    )
    # Curbside pickup
    curb_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_curbside_pickup",
        desc=f"Store {store_index + 1} offers curbside pickup service",
        parent=conv_group,
        critical=True
    )
    claim_curb = f"{label} offers curbside pickup (a.k.a. pickup, drive-up, drive-up & go, or similar)."
    await evaluator.verify(
        claim=claim_curb,
        node=curb_leaf,
        sources=sources_or_none,
        additional_instruction="Accept synonyms: Pickup, Drive Up, DriveUp & Go, Curbside, Order Pickup, Grocery Pickup.",
        majority_vote=False
    )
    # Same-day delivery
    sdd_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_same_day_delivery",
        desc=f"Store {store_index + 1} offers same-day delivery service",
        parent=conv_group,
        critical=True
    )
    claim_sdd = f"{label} offers same-day grocery delivery (including via partners like Instacart or similar)."
    await evaluator.verify(
        claim=claim_sdd,
        node=sdd_leaf,
        sources=sources_or_none,
        additional_instruction="Accept phrases like 'same-day', 'delivery today', 'delivery in as little as 2 hours'.",
        majority_vote=False
    )
    # EBT/SNAP
    ebt_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_ebt_snap",
        desc=f"Store {store_index + 1} accepts EBT/SNAP payments",
        parent=conv_group,
        critical=True
    )
    claim_ebt = f"{label} accepts EBT/SNAP payments."
    await evaluator.verify(
        claim=claim_ebt,
        node=ebt_leaf,
        sources=sources_or_none,
        additional_instruction="Look for 'EBT accepted', 'SNAP/EBT', or similar wording. Acceptance at this location is required.",
        majority_vote=False
    )

    # 5) Additional Programs (critical parallel group)
    prog_group = evaluator.add_parallel(
        id=f"store_{store_index + 1}_additional_programs",
        desc=f"Required additional programs and features for Store {store_index + 1}",
        parent=store_node,
        critical=True
    )
    # Loyalty program
    loyalty_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_loyalty_program",
        desc=f"Store {store_index + 1} has a customer loyalty/rewards program",
        parent=prog_group,
        critical=True
    )
    claim_loyalty = (
        f"{label} participates in a customer loyalty/rewards program from its chain that is usable at this location."
    )
    await evaluator.verify(
        claim=claim_loyalty,
        node=loyalty_leaf,
        sources=sources_or_none,
        additional_instruction="Chain-level loyalty programs are acceptable if they clearly apply to the store's brand/locations.",
        majority_vote=False
    )
    # Fuel center
    fuel_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_fuel_center",
        desc=f"Store {store_index + 1} has an attached fuel center/gas station",
        parent=prog_group,
        critical=True
    )
    claim_fuel = f"{label} has an attached or on-premises fuel center/gas station at or immediately adjacent to this store."
    await evaluator.verify(
        claim=claim_fuel,
        node=fuel_leaf,
        sources=sources_or_none,
        additional_instruction="Look for 'Fuel Center', 'Gas', or explicit mention of a gas station associated with this store's location.",
        majority_vote=False
    )
    # Organic produce section
    organic_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_organic_section",
        desc=f"Store {store_index + 1} has an organic produce section",
        parent=prog_group,
        critical=True
    )
    claim_organic = f"{label} offers an organic produce section (organic fruits and vegetables available)."
    await evaluator.verify(
        claim=claim_organic,
        node=organic_leaf,
        sources=sources_or_none,
        additional_instruction="Look for 'organic produce', 'organic fruits and vegetables', or similar wording on store/chain pages.",
        majority_vote=False
    )

    # 6) Operating Hours (critical parallel group)
    hours_group = evaluator.add_parallel(
        id=f"store_{store_index + 1}_operating_hours",
        desc=f"Required operating hours information for Store {store_index + 1}",
        parent=store_node,
        critical=True
    )
    # Regular weekly hours (text)
    reg_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_regular_hours",
        desc=f"Regular weekly operating hours provided for Store {store_index + 1}",
        parent=hours_group,
        critical=True
    )
    reg_text = (hours.regular_hours or "").strip()
    claim_reg = (
        f"The store's regular weekly operating hours are as stated or summarized: '{reg_text}'. "
        f"Treat as matching if the summary clearly corresponds to the hours displayed on the page."
    )
    await evaluator.verify(
        claim=claim_reg,
        node=reg_leaf,
        sources=sources_or_none,
        additional_instruction="Allow reasonable summarization/formatting differences if the meaning clearly matches the page.",
        majority_vote=False
    )
    # Thanksgiving status
    t_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_thanksgiving",
        desc=f"Thanksgiving Day operating status specified for Store {store_index + 1} (open with hours or closed)",
        parent=hours_group,
        critical=True
    )
    t_text = (hours.thanksgiving or "").strip()
    claim_t = (
        f"On Thanksgiving Day, {label} is as stated: '{t_text}'. "
        f"Chain-level official holiday hours pages applicable to this location are acceptable."
    )
    await evaluator.verify(
        claim=claim_t,
        node=t_leaf,
        sources=sources_or_none,
        additional_instruction="Prefer store-level or chain-official holiday hours information; ensure it applies to Florida and this brand.",
        majority_vote=False
    )
    # Christmas status
    c_leaf = evaluator.add_leaf(
        id=f"store_{store_index + 1}_christmas",
        desc=f"Christmas Day operating status specified for Store {store_index + 1} (open with hours or closed)",
        parent=hours_group,
        critical=True
    )
    c_text = (hours.christmas or "").strip()
    claim_c = (
        f"On Christmas Day, {label} is as stated: '{c_text}'. "
        f"Chain-level official holiday hours pages applicable to this location are acceptable."
    )
    await evaluator.verify(
        claim=claim_c,
        node=c_leaf,
        sources=sources_or_none,
        additional_instruction="Prefer store-level or chain-official holiday hours information; ensure it applies to Florida and this brand.",
        majority_vote=False
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Florida grocery stores task and return a structured summary.
    """
    # Initialize Evaluator (root is non-critical; we add our own main node)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Create a top-level task node (non-critical to allow partial credit)
    task_node = evaluator.add_parallel(
        id="grocery_store_task",
        desc="Find 4 major grocery store locations in Florida that each meet all specified service, convenience, and operational requirements",
        parent=evaluator.root,
        critical=False
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=GroceryExtraction,
        extraction_name="extracted_stores"
    )

    # Select exactly 4 items (pad with empty if needed)
    selected: List[StoreInfo] = list(extracted.stores[:4])
    while len(selected) < 4:
        selected.append(StoreInfo())

    # Different cities check (critical under the main task node)
    cities = [normalize_city((s.location.city if s and s.location else None)) for s in selected]
    unique_cities = len({c for c in cities if c})
    different_cities_ok = (len(cities) == 4) and all(c is not None for c in cities) and unique_cities == 4
    evaluator.add_custom_node(
        result=different_cities_ok,
        id="different_cities_check",
        desc="Verify that all 4 stores are located in different Florida cities",
        parent=task_node,
        critical=True
    )

    # Build verification subtree for each store
    for i, store in enumerate(selected):
        await verify_store(evaluator, task_node, store, i)

    # Optionally record some custom info
    evaluator.add_custom_info(
        info={
            "total_stores_extracted": len(extracted.stores),
            "cities_extracted": [s.location.city if s.location else None for s in selected]
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Return summary
    return evaluator.get_summary()