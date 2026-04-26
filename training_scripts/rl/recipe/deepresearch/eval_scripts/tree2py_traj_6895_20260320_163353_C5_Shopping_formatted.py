import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "retail_holiday_hours_2025"
TASK_DESCRIPTION = (
    "Identify four different major retail store chains in the United States that were CLOSED on Christmas Day 2025 "
    "(December 25, 2025) and that opened at 7:00 a.m. or earlier on Black Friday 2025 (November 28, 2025). "
    "The four stores must collectively represent at least three different retail categories (such as general merchandise, "
    "warehouse club, department store, electronics, or sporting goods). For each store, provide: "
    "1. The store's official chain name, "
    "2. Confirmation that it was closed on Christmas Day 2025, "
    "3. Its specific opening time on Black Friday 2025 (November 28, 2025), "
    "4. Its primary retail category."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreItem(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    christmas_closed_statement: Optional[str] = None
    christmas_sources: List[str] = Field(default_factory=list)
    black_friday_open_time: Optional[str] = None  # e.g., "5:00 a.m.", "6 AM", "7am", "07:00"
    black_friday_sources: List[str] = Field(default_factory=list)
    category_sources: List[str] = Field(default_factory=list)


class StoresExtraction(BaseModel):
    stores: List[StoreItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
    Extract up to four different major U.S. retail store chains mentioned in the answer that claim to (a) be closed on
    Christmas Day 2025 and (b) open at 7:00 a.m. or earlier on Black Friday 2025. For each identified store, extract:

    - name: the official chain name as written in the answer
    - category: the store's primary retail category (e.g., "general merchandise", "warehouse club", "department store",
      "electronics", "sporting goods", "grocery", "home improvement", etc.) as stated in the answer
    - christmas_closed_statement: the exact phrasing (or a concise paraphrase) from the answer asserting the chain was
      closed on Christmas Day 2025
    - christmas_sources: all URLs explicitly cited in the answer that support the Christmas 2025 closure claim for this chain
    - black_friday_open_time: the specific opening time on Black Friday 2025 as provided in the answer (e.g., "5:00 a.m.", "6 AM", "7 am")
    - black_friday_sources: all URLs explicitly cited in the answer that support the Black Friday 2025 opening time for this chain
    - category_sources: any URLs explicitly cited in the answer that support the category claim for this chain (e.g., company
      about page, Wikipedia, corporate overview). If no category-specific URLs are provided, leave this as an empty list.

    IMPORTANT RULES:
    - Only extract information explicitly present in the answer text.
    - For all source fields, include only valid URLs explicitly present in the answer (plain URLs or markdown links).
      Do not invent or infer URLs. If none are present for a field, return an empty list.
    - If any textual field is not provided, set it to null.
    - Return exactly four items if available; otherwise, return however many are present (up to four).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
ORDINAL_WORDS = ["first", "second", "third", "fourth"]


def ordinal_word(index: int) -> str:
    return ORDINAL_WORDS[index] if 0 <= index < len(ORDINAL_WORDS) else f"#{index + 1}"


def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    n = name.lower().strip()
    # Remove common suffixes/words/punctuation that don't change the chain
    for token in [" inc.", " inc", " llc", " ltd.", " ltd", " co.", " co", " corporation", " corp.", " corp", " stores", " store", " club"]:
        if n.endswith(token):
            n = n[: -len(token)].strip()
    # Replace multiple spaces
    while "  " in n:
        n = n.replace("  ", " ")
    return n


def canonicalize_category(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    s = cat.lower().strip()
    # Map common keywords to canonical buckets
    if any(k in s for k in ["warehouse", "club", "membership", "wholesale"]):
        return "warehouse club"
    if "electronics" in s or "consumer electronics" in s:
        return "electronics"
    if "department" in s:
        return "department store"
    if "sport" in s:
        return "sporting goods"
    if any(k in s for k in ["general merchandise", "mass merchant", "discount", "variety", "supercenter", "big-box", "big box"]):
        return "general merchandise"
    if any(k in s for k in ["grocery", "supermarket", "food market"]):
        return "grocery"
    if any(k in s for k in ["home improvement", "hardware"]):
        return "home improvement"
    if any(k in s for k in ["pharmacy", "drugstore", "drug store"]):
        return "pharmacy/drugstore"
    if any(k in s for k in ["apparel", "clothing", "fashion"]):
        return "apparel"
    if any(k in s for k in ["home goods", "homegoods", "furnitur", "decor"]):
        return "home goods"
    if "toy" in s:
        return "toy"
    if "beauty" in s or "cosmetic" in s:
        return "beauty"
    # Fallback to the raw category string
    return s


def combine_sources(*args: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in args:
        for url in lst or []:
            u = (url or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification logic per store                                                #
# --------------------------------------------------------------------------- #
async def verify_store(
    evaluator: Evaluator,
    parent_node,
    store: StoreItem,
    index: int,
) -> None:
    """
    Build verification sub-tree for one store, including three critical leaves:
    - Christmas Day 2025 closure
    - Black Friday 2025 opening time at or before 7:00 a.m.
    - Correct retail category
    """
    ord_word = ordinal_word(index)

    # Container node for this store (parallel, non-critical)
    store_node = evaluator.add_parallel(
        id=f"{ord_word}_store",
        desc=f"Evaluate the {ord_word} identified store",
        parent=parent_node,
        critical=False,
    )

    # 1) Christmas closure verification (critical)
    christmas_leaf = evaluator.add_leaf(
        id=f"{ord_word}_store_christmas_closure",
        desc=f"The {ord_word} store was closed on Christmas Day 2025 (December 25, 2025)",
        parent=store_node,
        critical=True,
    )
    store_name = store.name or "the identified store chain"
    christmas_claim = (
        f"The chain '{store_name}' was closed on Christmas Day 2025 (December 25, 2025) in the United States."
    )
    await evaluator.verify(
        claim=christmas_claim,
        node=christmas_leaf,
        sources=store.christmas_sources,
        additional_instruction=(
            "Support the claim only if at least one provided URL explicitly indicates that the national chain "
            "was closed on Christmas Day 2025 (Dec 25, 2025) in the U.S., or an official policy page states the "
            "chain is closed on Christmas Day and is clearly applicable in 2025. Reject sources that refer to "
            "other years, other countries, vague blog posts, or only a single location without chain‑level implication. "
            "If no URLs are provided or all URLs are irrelevant, mark as NOT SUPPORTED."
        ),
    )

    # 2) Black Friday opening time at or before 7:00 a.m. (critical)
    bf_leaf = evaluator.add_leaf(
        id=f"{ord_word}_store_black_friday_time",
        desc=f"The {ord_word} store opened at 7:00 a.m. or earlier on Black Friday 2025 (November 28, 2025)",
        parent=store_node,
        critical=True,
    )
    bf_time_text = store.black_friday_open_time or "at or before 7:00 a.m."
    bf_claim = (
        f"On Black Friday 2025 (November 28, 2025), the chain '{store_name}' opened {bf_time_text} local time, "
        f"which is at or before 7:00 a.m."
    )
    await evaluator.verify(
        claim=bf_claim,
        node=bf_leaf,
        sources=store.black_friday_sources,
        additional_instruction=(
            "Support the claim only if at least one provided URL indicates that on Black Friday 2025 (Nov 28, 2025), "
            "the chain opened at or before 7:00 a.m. (e.g., 7:00 a.m., 6:00 a.m., 5:00 a.m.). Accept official corporate "
            "announcements, press releases, or reputable reporting specifically about 2025. If times vary by location, "
            "a clear corporate/national announcement of at‑or‑before‑7 a.m. qualifies; a single unrelated local store "
            "page does not. If the sources show 8 a.m. or later, or refer to a different year, mark as NOT SUPPORTED. "
            "If no URLs are provided, mark as NOT SUPPORTED."
        ),
    )

    # 3) Category verification (critical)
    cat_leaf = evaluator.add_leaf(
        id=f"{ord_word}_store_category",
        desc=f"The {ord_word} store's retail category is correctly identified",
        parent=store_node,
        critical=True,
    )
    category_text = store.category or "the stated category"
    category_claim = (
        f"The primary retail category of the chain '{store_name}' is '{category_text}'."
    )
    # Use category_sources if provided; otherwise, fall back to any other provided sources
    category_sources = store.category_sources or combine_sources(store.christmas_sources, store.black_friday_sources)
    await evaluator.verify(
        claim=category_claim,
        node=cat_leaf,
        sources=category_sources,
        additional_instruction=(
            "Support the claim if a provided URL (e.g., company About page, Wikipedia overview, reputable business profile) "
            "clearly reflects the chain’s primary category. Allow reasonable synonyms (e.g., 'consumer electronics' ≈ 'electronics', "
            "'membership club' ≈ 'warehouse club'). If no URLs are provided or none speak to category, mark as NOT SUPPORTED."
        ),
    )


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
    Evaluate an answer for the retail holiday hours task and return a structured summary.
    """
    # 1) Initialize evaluator and root node (parallel aggregation)
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

    # 2) Extract structured info for up to four stores
    extracted: StoresExtraction = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction",
    )

    # Ensure exactly four positions (pad with empty StoreItem if fewer)
    stores: List[StoreItem] = list(extracted.stores[:4])
    while len(stores) < 4:
        stores.append(StoreItem())

    # 3) Build per-store verification subtrees
    for idx in range(4):
        await verify_store(evaluator, root, stores[idx], idx)

    # 4) Global constraints: category diversity and store uniqueness (both critical)
    # 4.1) Category diversity: at least three distinct categories among four stores
    canonical_cats: List[Optional[str]] = [canonicalize_category(s.category) for s in stores]
    unique_cats = {c for c in canonical_cats if c}
    has_diverse_categories = len(unique_cats) >= 3

    evaluator.add_custom_node(
        result=has_diverse_categories,
        id="category_diversity",
        desc="The four stores collectively represent at least three different retail categories",
        parent=root,
        critical=True,
    )

    # 4.2) Store uniqueness: all four identified stores are different chains
    normalized_names = [normalize_name(s.name) for s in stores if s.name]
    unique_names_count = len(set([n for n in normalized_names if n]))
    all_four_present = sum(1 for s in stores if (s.name or "").strip() != "") == 4
    all_unique = all_four_present and unique_names_count == 4

    evaluator.add_custom_node(
        result=all_unique,
        id="store_uniqueness",
        desc="All four identified stores are different chains (no duplicates)",
        parent=root,
        critical=True,
    )

    # 5) Record some helpful custom info for debugging
    evaluator.add_custom_info(
        {
            "extracted_store_names": [s.name for s in stores],
            "extracted_categories_raw": [s.category for s in stores],
            "canonical_categories": list(unique_cats),
            "category_diversity_count": len(unique_cats),
            "unique_chain_names_count": unique_names_count,
            "all_four_names_present": all_four_present,
        },
        info_type="debug_stats",
        info_name="extraction_debug_statistics",
    )

    # 6) Return the evaluation summary
    return evaluator.get_summary()