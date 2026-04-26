import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thanksgiving_philly_groceries_2025"
TASK_DESCRIPTION = (
    "You are spending Thanksgiving Day 2025 (Thursday, November 27, 2025) in the Philadelphia, "
    "Pennsylvania metropolitan area and need to identify places where you can purchase groceries and food items on the holiday. "
    "Since many major retailers like Walmart, Target, Costco, Trader Joe's, and Aldi are closed on Thanksgiving, you need to find alternative options that will be open.\n\n"
    "Identify four grocery stores or convenience stores from different national retail chains that will be open on Thanksgiving Day 2025 in the Philadelphia metro area. For each store, provide:\n"
    "1. The name of the national chain\n"
    "2. Confirmation that this chain will be open on Thanksgiving Day 2025\n"
    "3. A specific street address for one store location within the Philadelphia metropolitan area (Philadelphia city proper or nearby suburbs)\n"
    "4. The store's operating hours on Thanksgiving Day 2025\n"
    "5. A reference URL that verifies the chain's Thanksgiving Day operating status or hours\n\n"
    "Important: Each of the four stores must be from a different national chain. For example, you cannot list two different Kroger locations or two different CVS locations - each must represent a distinct retail chain."
)

THANKSGIVING_DATE_LONG = "Thursday, November 27, 2025"
THANKSGIVING_DATE_SHORT = "November 27, 2025"
EXCLUDED_CHAINS = ["walmart", "target", "costco", "trader joe", "trader joe's", "aldi"]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreItem(BaseModel):
    chain_name: Optional[str] = None
    store_type: Optional[str] = None  # e.g., "grocery", "convenience", "supermarket", "pharmacy"
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    thanksgiving_hours: Optional[str] = None
    open_status_text: Optional[str] = None  # any explicit text stating open/closed status
    reference_urls: List[str] = Field(default_factory=list)


class StoresExtraction(BaseModel):
    stores: List[StoreItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return (
        "Extract up to four store entries from the answer. Each store entry represents one location of a national retail "
        "chain that is open on Thanksgiving Day 2025 in the Philadelphia metro area. For each entry, return a JSON object "
        "with the following fields:\n"
        "- chain_name: The national retail chain name (string)\n"
        "- store_type: The type of store (e.g., 'grocery', 'supermarket', 'convenience', 'pharmacy', etc.)\n"
        "- address: The specific street address for a location (string). If a full address is given in one line, put it here.\n"
        "- city: City name if provided (string or null)\n"
        "- state: State abbreviation or full state name, if provided (string or null)\n"
        "- thanksgiving_hours: Operating hours for Thanksgiving Day 2025 for that location (string or null)\n"
        "- open_status_text: Any explicit text in the answer confirming the chain/location is open on Thanksgiving Day 2025 (string or null)\n"
        "- reference_urls: Array of URLs that the answer cites to support open status or hours for Thanksgiving Day 2025\n\n"
        "Rules:\n"
        "1) Extract only what the answer explicitly provides. Do not invent.\n"
        "2) If more than four stores are provided, extract the first four stores only.\n"
        "3) If the answer provides fewer than four, extract whatever is available. Missing fields must be returned as null.\n"
        "4) For reference_urls, include actual URLs only.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_chain(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    # Remove common suffixes
    for token in ["inc", "llc", "co", "company", "stores", "market", "supermarket", "pharmacy", "drugstore"]:
        s = s.replace(f" {token}", "")
    return s.strip()


def is_excluded_chain(name: Optional[str]) -> bool:
    n = normalize_chain(name)
    return any(n.startswith(ex) or ex in n for ex in EXCLUDED_CHAINS)


def looks_like_address(addr: Optional[str]) -> bool:
    if not addr:
        return False
    # Simple heuristic: contains a number and a street-like word
    has_number = bool(re.search(r"\d", addr))
    street_words = ["st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard", "dr", "drive", "pike", "highway", "pl", "place", "ln", "lane"]
    has_street_word = any(sw in addr.lower() for sw in street_words)
    return has_number and has_street_word


def is_grocery_or_convenience(store_type: Optional[str]) -> bool:
    if not store_type:
        return False
    t = store_type.lower()
    keywords = ["grocery", "supermarket", "convenience", "market", "corner store", "pharmacy", "drugstore"]
    return any(k in t for k in keywords)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_store(
    evaluator: Evaluator,
    parent_node,
    store: StoreItem,
    index: int,
) -> None:
    """
    Build and verify the tree for a single store entry.
    """
    chain = store.chain_name or ""
    addr = store.address or ""
    hours = store.thanksgiving_hours or ""
    refs = store.reference_urls or []

    store_node = evaluator.add_parallel(
        id=f"Store_{index+1}",
        desc=f"Store item #{index + 1} details",
        parent=parent_node,
        critical=False,
    )

    # Store_Type (CRITICAL)
    store_type_node = evaluator.add_custom_node(
        result=is_grocery_or_convenience(store.store_type),
        id=f"store_{index+1}_type",
        desc="Store is a grocery store or convenience store",
        parent=store_node,
        critical=True,
    )

    # Chain_Name_Provided (CRITICAL)
    chain_name_node = evaluator.add_custom_node(
        result=bool(chain.strip()),
        id=f"store_{index+1}_chain_name_provided",
        desc="Provides the national retail chain name for this store",
        parent=store_node,
        critical=True,
    )

    # National_Chain (CRITICAL) - simple verification using claim
    nat_chain_leaf = evaluator.add_leaf(
        id=f"store_{index+1}_national_chain",
        desc="The chain is a national retail chain (not a purely local/independent store)",
        parent=store_node,
        critical=True,
    )
    nat_chain_claim = f"The chain '{chain}' is a national retail chain, not a purely local or independent store."
    await evaluator.verify(
        claim=nat_chain_claim,
        node=nat_chain_leaf,
        sources=refs if refs else None,
        additional_instruction="Consider a 'national retail chain' as a company with multi‑state presence in the U.S. "
                               "If the provided reference page indicates multiple states or nationwide operations, that supports the claim.",
    )

    # Open_On_Thanksgiving_2025 (CRITICAL) - verify by URLs if provided
    open_leaf = evaluator.add_leaf(
        id=f"store_{index+1}_open_thanksgiving_2025",
        desc=f"Confirms the chain will be open on Thanksgiving Day 2025 ({THANKSGIVING_DATE_LONG})",
        parent=store_node,
        critical=True,
    )
    open_claim = (
        f"The chain '{chain}' will be open on Thanksgiving Day 2025 ({THANKSGIVING_DATE_LONG}). "
        "Chain-level statements or location-specific hours that include Thanksgiving Day 2025 both count as confirmation."
    )
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=refs if refs else None,
        additional_instruction="Accept evidence if the page mentions 'Thanksgiving' or explicitly references the date "
                               f"'{THANKSGIVING_DATE_SHORT}', and indicates the store/chain is open or provides hours.",
    )

    # Street_Address_Provided (CRITICAL)
    addr_provided_node = evaluator.add_custom_node(
        result=looks_like_address(addr),
        id=f"store_{index+1}_street_address_provided",
        desc="Provides a specific street address for one location",
        parent=store_node,
        critical=True,
    )

    # Address_In_Philadelphia_Metro (CRITICAL) - verify with claim, use refs when available
    philly_addr_leaf = evaluator.add_leaf(
        id=f"store_{index+1}_address_in_philly_metro",
        desc="The provided address is within the Philadelphia metropolitan area (city or nearby suburbs)",
        parent=store_node,
        critical=True,
    )
    philly_addr_claim = (
        f"The address '{addr}' is located within the Philadelphia, Pennsylvania metropolitan area (including "
        "Philadelphia city and nearby suburbs in Bucks, Montgomery, Delaware, Chester (PA), and Camden, Gloucester, Burlington (NJ))."
    )
    await evaluator.verify(
        claim=philly_addr_claim,
        node=philly_addr_leaf,
        sources=refs if refs else None,
        additional_instruction="If the address clearly includes 'Philadelphia, PA' or a known suburb/county listed above, consider it within the metro.",
    )

    # Thanksgiving_Hours_Provided (CRITICAL)
    hours_provided_node = evaluator.add_custom_node(
        result=bool(hours.strip()),
        id=f"store_{index+1}_thanksgiving_hours_provided",
        desc="Provides operating hours for Thanksgiving Day 2025 for that location",
        parent=store_node,
        critical=True,
    )

    # Reference_URL_Provided (CRITICAL)
    refs_provided_node = evaluator.add_custom_node(
        result=(len(refs) > 0),
        id=f"store_{index+1}_reference_url_provided",
        desc="Provides at least one reference URL",
        parent=store_node,
        critical=True,
    )

    # Reference_URL_Supports_Open_Status_Or_Hours (CRITICAL) - verify by URLs
    refs_support_leaf = evaluator.add_leaf(
        id=f"store_{index+1}_reference_url_supports",
        desc="The reference URL supports the claim that the chain/location is open on Thanksgiving and/or provides Thanksgiving hours",
        parent=store_node,
        critical=True,
    )
    refs_support_claim = (
        f"At least one of the provided reference URLs explicitly states Thanksgiving Day 2025 hours or confirms that '{chain}' "
        "is open on Thanksgiving Day 2025."
    )
    await evaluator.verify(
        claim=refs_support_claim,
        node=refs_support_leaf,
        sources=refs if refs else None,
        additional_instruction="Look for explicit mention of 'Thanksgiving' or the date "
                               f"'{THANKSGIVING_DATE_SHORT}', and either open status or specific operating hours.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for the Thanksgiving 2025 Philadelphia metro grocery/convenience store task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates independent checks and store items
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

    # Extract stores information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction",
    )

    # Record some task-specific info (optional)
    evaluator.add_ground_truth({
        "holiday": THANKSGIVING_DATE_LONG,
        "excluded_closed_retailers": ["Walmart", "Target", "Costco", "Trader Joe's", "Aldi"],
        "requirement": "Exactly four stores, each from a different national chain, with address in Philadelphia metro and Thanksgiving 2025 hours"
    })

    # -------------------- Global critical checks -------------------------- #
    # Response_Format_Item_Count (must be exactly 4 items in the answer)
    item_count_node = evaluator.add_custom_node(
        result=(len(extracted.stores) == 4),
        id="response_format_item_count",
        desc="Provides exactly four store entries/items",
        parent=root,
        critical=True,
    )

    # Distinct_Chains_Global (must be four distinct chains)
    chain_names_for_distinct_check = [normalize_chain(s.chain_name) for s in extracted.stores[:4]]
    non_empty_chains = [c for c in chain_names_for_distinct_check if c]
    distinct_chains_ok = (len(non_empty_chains) == 4) and (len(set(non_empty_chains)) == 4)
    distinct_chains_node = evaluator.add_custom_node(
        result=distinct_chains_ok,
        id="distinct_chains_global",
        desc="All four stores are from four distinct national retail chains (no duplicate chains)",
        parent=root,
        critical=True,
    )

    # Excluded_Closed_Retailers (none of the selected chains can be in the excluded list)
    excluded_present = any(is_excluded_chain(s.chain_name) for s in extracted.stores[:4])
    excluded_check_node = evaluator.add_custom_node(
        result=(not excluded_present),
        id="excluded_closed_retailers",
        desc="None of the selected chains are Walmart, Target, Costco, Trader Joe's, or Aldi",
        parent=root,
        critical=True,
    )

    # -------------------- Per-store verification -------------------------- #
    # Prepare up to 4 stores (slice and pad if fewer than 4)
    stores_to_check: List[StoreItem] = list(extracted.stores[:4])
    while len(stores_to_check) < 4:
        stores_to_check.append(StoreItem())  # pad with empty to allow structured evaluation

    # Create store subtrees (non-critical, allowing partial credit)
    for i, store in enumerate(stores_to_check):
        await verify_store(evaluator, root, store, i)

    # -------------------- Final summary ----------------------------------- #
    return evaluator.get_summary()