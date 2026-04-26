import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "winter_wreath_craft_stores_sa_2026"
TASK_DESCRIPTION = (
    "You are planning to make winter wreaths as a DIY hobby project in early January 2026 during the post-Christmas "
    "clearance period. You live in San Antonio, Texas and want to take advantage of clearance sales on craft supplies.\n\n"
    "Identify at least two major craft stores (Michaels or Hobby Lobby) in San Antonio, Texas where you can purchase the following essential supplies for your winter wreath project:\n"
    "- Grapevine wreath bases in 18-inch size\n"
    "- Wired ribbon in 2.5-inch width\n"
    "- Attachment supplies (pipe cleaners and/or zip ties)\n\n"
    "For each store you identify, provide:\n"
    "1. The store name and specific San Antonio location address\n"
    "2. Confirmation that the store stocks grapevine wreath bases in 18-inch size\n"
    "3. Confirmation that the store carries wired ribbon in 2.5-inch width\n"
    "4. Store operating hours (confirming they are open during standard weekday hours, at least 9 AM - 8 PM)\n"
    "5. Information about post-Christmas clearance availability or policy\n"
    "6. Confirmation that attachment supplies (pipe cleaners and/or zip ties) are available\n"
    "7. A reference URL for verifying the store information\n\n"
    "Note: You should identify stores that meet all the critical requirements for your winter wreath-making project."
)


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class StoreInfo(BaseModel):
    store_name: Optional[str] = None
    store_brand: Optional[str] = None  # e.g., "Michaels" or "Hobby Lobby"
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None

    # General store/location references (e.g., store locator page, Google Maps listing)
    reference_urls: List[str] = Field(default_factory=list)

    # Hours info and related sources
    hours_text: Optional[str] = None
    hours_urls: List[str] = Field(default_factory=list)

    # Clearance info and sources
    clearance_info: Optional[str] = None
    clearance_urls: List[str] = Field(default_factory=list)

    # Product-specific sources
    grapevine_urls: List[str] = Field(default_factory=list)       # URLs that show 18" grapevine wreath bases/forms
    wired_ribbon_urls: List[str] = Field(default_factory=list)    # URLs that show 2.5" wired ribbon
    attachment_urls: List[str] = Field(default_factory=list)      # URLs that show pipe cleaners (chenille stems) or zip ties

    # Optional free-form listing of what attachments were mentioned (e.g., ["pipe cleaners","zip ties"])
    attachment_types: List[str] = Field(default_factory=list)


class StoresExtraction(BaseModel):
    stores: List[StoreInfo] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_stores() -> str:
    return """
Extract up to three San Antonio craft store entries (Michaels or Hobby Lobby only) from the answer. For each store, return:
- store_name: The specific store name as written in the answer (e.g., "Michaels – San Pedro Ave")
- store_brand: "Michaels" or "Hobby Lobby" if stated or clearly implied; otherwise null
- address: The full street address if given (e.g., "1234 San Pedro Ave, San Antonio, TX 78212")
- city: City if given (prefer "San Antonio")
- state: State abbreviation if given (prefer "TX")
- zip: Zip code if given
- reference_urls: All general store/location URLs cited (store locator page, store detail page, Google Maps link, etc.)
- hours_text: Any weekday hours text the answer provided
- hours_urls: URLs that show the store’s hours (store page, Google Maps, etc.)
- clearance_info: Any mention of post-Christmas or after-Christmas clearance/sale/policy in the answer
- clearance_urls: URLs that describe or confirm post-Christmas/holiday/Christmas clearance or promotions (brand-level or location-specific)
- grapevine_urls: URLs showing that the store sells an 18-inch grapevine wreath base/form
- wired_ribbon_urls: URLs showing that the store sells 2.5-inch wired ribbon (wired edges)
- attachment_urls: URLs showing that the store sells pipe cleaners (chenille stems) and/or zip ties
- attachment_types: Strings of the attachment supplies explicitly mentioned (e.g., ["pipe cleaners","zip ties"])

Rules:
- Only extract URLs explicitly present in the answer. Do not invent URLs.
- Include full URLs. If a URL is missing the protocol, prepend "http://".
- If a field is not present for a store, set it to null (for strings) or an empty list (for arrays).
- Return exactly a JSON object: { "stores": [ ... up to 3 objects ... ] }.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _sanitize_urls(urls: List[str]) -> List[str]:
    out = []
    for u in urls or []:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        out.append(u)
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _combine_sources(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        for u in _sanitize_urls(lst):
            if u not in combined:
                combined.append(u)
    return combined


def _infer_brand(store: StoreInfo) -> Optional[str]:
    if store.store_brand:
        b = store.store_brand.strip().lower()
        if "michaels" in b:
            return "Michaels"
        if "hobby lobby" in b or "hobbylobby" in b:
            return "Hobby Lobby"
    if store.store_name:
        n = store.store_name.strip().lower()
        if "michaels" in n:
            return "Michaels"
        if "hobby lobby" in n or "hobbylobby" in n:
            return "Hobby Lobby"
    return None


def _address_string(store: StoreInfo) -> str:
    parts = []
    if store.address:
        parts.append(store.address.strip())
    else:
        # Try to assemble from parts if full address isn't given
        addr_parts = []
        if store.city:
            addr_parts.append(store.city.strip())
        if store.state:
            addr_parts.append(store.state.strip())
        if store.zip:
            addr_parts.append(store.zip.strip())
        if addr_parts:
            parts.append(", ".join(addr_parts))
    return ", ".join(parts) if parts else ""


# -----------------------------------------------------------------------------
# Verification logic per store
# -----------------------------------------------------------------------------
async def verify_one_store(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    store: StoreInfo,
    store_index: int,
) -> VerificationNode:
    """
    Build verification nodes and run checks for one store (Michaels or Hobby Lobby in San Antonio).
    All child checks are critical for a store to be considered "qualifying".
    """
    # Parent node for this store
    store_node = evaluator.add_parallel(
        id=f"store_{store_index}",
        desc=("First qualifying craft store meeting all requirements" if store_index == 1 else
              "Second qualifying craft store meeting all requirements" if store_index == 2 else
              "Third qualifying craft store meeting all requirements"),
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence (gatekeeper)
    primary_refs = _sanitize_urls(store.reference_urls)
    ref_exists = any(primary_refs)
    ref_leaf = evaluator.add_custom_node(
        result=ref_exists,
        id=f"reference_url_store_{store_index}",
        desc=f"Valid reference URL provided for store #{store_index} verification",
        parent=store_node,
        critical=True
    )

    # 1) Store type: Michaels or Hobby Lobby
    brand = _infer_brand(store)
    store_type_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_store_type",
        desc="Store is identified as either Michaels or Hobby Lobby",
        parent=store_node,
        critical=True
    )
    if brand:
        claim_type = f"The referenced page(s) show that this location is a {brand} store."
    else:
        claim_type = "The referenced page(s) show that this location is either a Michaels or a Hobby Lobby store."
    await evaluator.verify(
        claim=claim_type,
        node=store_type_leaf,
        sources=primary_refs,
        additional_instruction="Look for brand identifiers on the page (logo, name, breadcrumb, page title). Accept store locator pages or Google Maps pages that explicitly show Michaels or Hobby Lobby.",
        extra_prerequisites=[ref_leaf],
    )

    # 2) Location in San Antonio with a specific address
    loc_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_location_san_antonio",
        desc="Store is located in San Antonio, Texas with specific address provided",
        parent=store_node,
        critical=True
    )
    address_str = _address_string(store)
    claim_loc = (
        f"The store is located in San Antonio, Texas. The address is: {address_str if address_str else '[not provided]'}."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=primary_refs,
        additional_instruction="Confirm the page indicates a San Antonio, TX location. Allow minor formatting differences in the address (suite numbers, punctuation). If a full street address is given in the answer, check that it matches the page.",
        extra_prerequisites=[ref_leaf],
    )

    # 3) Grapevine wreath base 18-inch available
    grapevine_sources = _combine_sources(store.grapevine_urls or [], primary_refs)
    grape_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_has_grapevine_base",
        desc="Store stocks grapevine wreath bases in 18-inch size",
        parent=store_node,
        critical=True
    )
    claim_grape = "This store sells or stocks an 18-inch grapevine wreath base (also called an 18\" grapevine wreath form)."
    await evaluator.verify(
        claim=claim_grape,
        node=grape_leaf,
        sources=grapevine_sources,
        additional_instruction="Look for product pages or category results indicating 'grapevine wreath' with size 18 inch (18\", 18-inch). Synonyms like 'wreath form' are acceptable.",
        extra_prerequisites=[ref_leaf],
    )

    # 4) Wired ribbon 2.5-inch available
    ribbon_sources = _combine_sources(store.wired_ribbon_urls or [], primary_refs)
    ribbon_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_has_wired_ribbon",
        desc="Store carries wired ribbon in 2.5-inch width",
        parent=store_node,
        critical=True
    )
    claim_ribbon = "This store sells wired ribbon with a 2.5-inch width (wired edges)."
    await evaluator.verify(
        claim=claim_ribbon,
        node=ribbon_leaf,
        sources=ribbon_sources,
        additional_instruction="Look for 'wired ribbon' with width 2.5 in, 2-1/2 in, or 2.5\". Product or category pages suffice.",
        extra_prerequisites=[ref_leaf],
    )

    # 5) Operating hours support weekday 9 AM - 8 PM minimum
    hours_sources = _combine_sources(store.hours_urls or [], primary_refs)
    hours_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_operating_hours",
        desc="Store hours are provided and indicate operation during weekday daytime hours (minimum 9 AM - 8 PM)",
        parent=store_node,
        critical=True
    )
    claim_hours = (
        "On weekdays (Monday–Friday), this store is open by 9:00 AM and closes at or after 8:00 PM."
    )
    await evaluator.verify(
        claim=claim_hours,
        node=hours_leaf,
        sources=hours_sources,
        additional_instruction="Check the store hours table or listing. The requirement is satisfied if the weekday schedule shows opening no later than 9:00 AM and closing no earlier than 8:00 PM.",
        extra_prerequisites=[ref_leaf],
    )

    # 6) Post-Christmas clearance information/policy
    clearance_sources = _combine_sources(store.clearance_urls or [], primary_refs)
    clearance_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_clearance_info",
        desc="Information provided about post-Christmas clearance availability or policy",
        parent=store_node,
        critical=True
    )
    claim_clearance = (
        "The store has post-Christmas or after-Christmas clearance promotions (e.g., 'Holiday Clearance' or 'Christmas Clearance') around late December or early January."
    )
    await evaluator.verify(
        claim=claim_clearance,
        node=clearance_leaf,
        sources=clearance_sources,
        additional_instruction="Accept brand-level clearance pages or promotional pages indicating 'After Christmas Sale', 'Christmas Clearance', 'Holiday Clearance', or similar wording.",
        extra_prerequisites=[ref_leaf],
    )

    # 7) Attachment supplies: pipe cleaners and/or zip ties available
    attach_sources = _combine_sources(store.attachment_urls or [], primary_refs)
    attach_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_attachment_supplies",
        desc="Store stocks essential attachment supplies including pipe cleaners and/or zip ties",
        parent=store_node,
        critical=True
    )
    claim_attach = (
        "This store sells at least one of the following: pipe cleaners (also called chenille stems) or zip ties."
    )
    await evaluator.verify(
        claim=claim_attach,
        node=attach_leaf,
        sources=attach_sources,
        additional_instruction="Look for product or category pages showing either 'pipe cleaners' (aka 'chenille stems') or 'zip ties'. Either one is sufficient.",
        extra_prerequisites=[ref_leaf],
    )

    return store_node


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the San Antonio winter wreath supply store task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Stores are evaluated independently
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

    # Note: Root is set to non-critical to avoid framework constraint that critical parents
    # cannot have non-critical children. We'll enforce the "at least two stores" requirement
    # via a dedicated critical node added after per-store checks.
    root.critical = False

    # Extract structured store info from the answer
    extracted: StoresExtraction = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction",
    )

    # Normalize and cap to 3 stores; pad if fewer
    stores: List[StoreInfo] = list(extracted.stores[:3])
    while len(stores) < 3:
        stores.append(StoreInfo())

    # Build store nodes and run verifications
    store_nodes: List[VerificationNode] = []
    for idx, store in enumerate(stores, start=1):
        node = await verify_one_store(
            evaluator=evaluator,
            parent_node=root,
            store=store,
            store_index=idx,
        )
        store_nodes.append(node)

    # Compute how many stores fully qualify (i.e., all critical checks passed)
    qualified = 0
    per_store_results = []
    for i, sn in enumerate(store_nodes, start=1):
        score = sn.aggregated_score  # triggers compute if needed
        is_qualified = (score == 1.0)
        per_store_results.append({"store_index": i, "qualified": is_qualified, "score": score})
        if is_qualified:
            qualified += 1

    # Add a critical node to enforce "at least two qualifying stores"
    evaluator.add_custom_node(
        result=(qualified >= 2),
        id="at_least_two_qualified",
        desc=f"At least two qualifying craft stores identified (found {qualified})",
        parent=root,
        critical=True
    )

    # Record some helpful custom info
    evaluator.add_custom_info(
        info={
            "qualified_store_count": qualified,
            "per_store_results": per_store_results
        },
        info_type="post_eval_stats",
        info_name="qualification_summary"
    )

    return evaluator.get_summary()