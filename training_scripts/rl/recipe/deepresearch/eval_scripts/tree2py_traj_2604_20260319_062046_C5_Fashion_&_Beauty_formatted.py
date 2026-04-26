import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_beauty_brands_2017_2023"
TASK_DESCRIPTION = """
A beauty industry analyst is researching the recent trend of celebrities launching their own beauty and personal care brands. They need to compile a list of celebrity-founded brands from recent years, where the founder is primarily known as a celebrity (such as an actor, singer, model, or television personality) before launching the brand, rather than simply serving as a brand ambassador or spokesperson for an existing company.

Identify 5 celebrity-founded beauty or personal care brands that meet all of the following criteria:
- The brand was founded/launched between January 1, 2017, and December 31, 2023
- The founder is a person who was primarily known as a celebrity (in entertainment, music, modeling, etc.) prior to launching the brand
- The brand is currently available for purchase (either online or in retail stores) as of March 19, 2026
- The brand focuses on beauty, skincare, haircare, or personal care products

For each brand, provide:
1. The brand name
2. The celebrity founder's full name (first and last name, as they are commonly known)
"""

AS_OF_DATE_STR = "March 19, 2026"
LAUNCH_START_YEAR = 2017
LAUNCH_END_YEAR = 2023


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BrandItem(BaseModel):
    brand_name: Optional[str] = None
    founder_name: Optional[str] = None
    brand_urls: List[str] = Field(default_factory=list)
    founder_urls: List[str] = Field(default_factory=list)


class BrandsExtraction(BaseModel):
    brands: List[BrandItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return """
    From the answer, extract up to 5 celebrity‑founded beauty or personal care brands in the same order they appear.
    For each item, return:
    - brand_name: the brand's name (string)
    - founder_name: the celebrity founder’s full name as commonly known (first and last name; include middle/last initials only if shown)
    - brand_urls: all URLs explicitly cited for this brand that help verify the brand’s existence, product focus, current availability (e.g., official website, shop/product/retailer pages, credible articles). Use only URLs explicitly present in the answer.
    - founder_urls: all URLs explicitly cited that help verify that the named person founded/co‑founded/launched the brand (not merely a spokesperson or ambassador), and/or indicate the person’s celebrity background. Use only URLs explicitly present in the answer.
    
    Rules:
    - Extract only what is explicitly present in the answer text; do not invent.
    - Include URLs exactly as shown (parse Markdown links and plain URLs).
    - If more than 5 items are present, include only the first 5.
    - If fewer than 5 items are present, include all available.
    - If a field is not provided for an item, set it to null (or an empty list for URL fields).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}
    return mapping.get(n, f"#{n}")


def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s) and isinstance(s, str) and s.strip() != ""


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and isinstance(urls, list) and len(urls) > 0


def _merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst or []:
            if isinstance(u, str) and u.strip() and u not in seen:
                merged.append(u)
                seen.add(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification logic per brand                                                #
# --------------------------------------------------------------------------- #
async def verify_single_brand(
    evaluator: Evaluator,
    parent_node,
    brand: BrandItem,
    idx: int,
) -> None:
    order_name = _ordinal(idx + 1)
    brand_label = brand.brand_name or "the brand"
    founder_label = brand.founder_name or "the founder"

    # Create the Brand node (parallel; non-critical to allow partial credit across different brands)
    brand_node = evaluator.add_parallel(
        id=f"Brand_{idx + 1}",
        desc=f"{order_name} celebrity-founded beauty/personal care brand (2017-2023) with brand name and founder name provided",
        parent=parent_node,
        critical=False,
    )

    # ----------------------- Name Identification Group -------------------- #
    name_desc = (
        "A brand name is provided with a reference URL, and the URL verifies that this is a celebrity-founded "
        "beauty or personal care brand that was launched between January 1, 2017 and December 31, 2023, and is "
        f"currently available for purchase as of {AS_OF_DATE_STR}"
    )
    name_node = evaluator.add_parallel(
        id=f"Brand_{idx + 1}_Name_Identification",
        desc=name_desc,
        parent=brand_node,
        critical=True,  # As per rubric, this is critical
    )

    # Existence checks (critical gating)
    evaluator.add_custom_node(
        result=_non_empty_str(brand.brand_name),
        id=f"Brand_{idx + 1}_brand_name_provided",
        desc=f"Brand {idx + 1}: brand name is provided",
        parent=name_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_urls(brand.brand_urls),
        id=f"Brand_{idx + 1}_brand_urls_provided",
        desc=f"Brand {idx + 1}: at least one brand reference URL is provided",
        parent=name_node,
        critical=True,
    )

    # Leaf: Beauty/personal care focus
    focus_leaf = evaluator.add_leaf(
        id=f"Brand_{idx + 1}_beauty_focus",
        desc="URL(s) support that this brand sells beauty, skincare, haircare, cosmetics, or personal care products",
        parent=name_node,
        critical=True,
    )
    focus_claim = (
        f"The brand named '{brand_label}' is a beauty, skincare, haircare, cosmetics, fragrance, or personal care brand "
        "that sells relevant products."
    )

    # Leaf: Launch window 2017–2023
    launch_leaf = evaluator.add_leaf(
        id=f"Brand_{idx + 1}_launch_window",
        desc=f"URL(s) support that the brand was launched/founded between {LAUNCH_START_YEAR} and {LAUNCH_END_YEAR} (inclusive)",
        parent=name_node,
        critical=True,
    )
    launch_claim = (
        f"The brand '{brand_label}' was launched or founded between {LAUNCH_START_YEAR}-01-01 and {LAUNCH_END_YEAR}-12-31 (inclusive)."
    )

    # Leaf: Currently available as of AS_OF_DATE_STR
    available_leaf = evaluator.add_leaf(
        id=f"Brand_{idx + 1}_available_as_of_2026",
        desc=f"URL(s) support that the brand is currently available for purchase as of {AS_OF_DATE_STR}",
        parent=name_node,
        critical=True,
    )
    available_claim = (
        f"As of {AS_OF_DATE_STR}, products from the brand '{brand_label}' are available for purchase (e.g., live product/shop pages or active retailer listings)."
    )

    # Prepare batch verifications for the Name group
    name_batch = [
        (
            focus_claim,
            brand.brand_urls,
            focus_leaf,
            "Confirm the page clearly indicates the brand sells items in beauty/skincare/haircare/cosmetics/"
            "fragrance/bath & body/personal care categories."
        ),
        (
            launch_claim,
            brand.brand_urls,
            launch_leaf,
            f"Look for explicit founding/launch/debut dates in {LAUNCH_START_YEAR}–{LAUNCH_END_YEAR}. "
            "Accept synonyms like 'launched', 'founded', 'debuted'. If evidence shows 2016 or 2024+ only, mark incorrect."
        ),
        (
            available_claim,
            brand.brand_urls,
            available_leaf,
            "Check for signals like active product listings, 'add to cart', 'in stock', pricing on official or retailer pages. "
            "A live brand shop or current retailer PDPs count as available."
        ),
    ]

    # Run Name group verifications in parallel
    await evaluator.batch_verify(name_batch)

    # --------------------- Founder Identification Group ------------------- #
    founder_desc = (
        "The celebrity founder's full name (first and last name, as commonly known) is provided with a reference URL, "
        "and the URL verifies that this person founded the brand (not merely served as ambassador or spokesperson) "
        "and was primarily known as a celebrity before launching it"
    )
    founder_node = evaluator.add_parallel(
        id=f"Brand_{idx + 1}_Founder_Identification",
        desc=founder_desc,
        parent=brand_node,
        critical=True,  # As per rubric, this is critical
    )

    # Existence checks (critical gating)
    evaluator.add_custom_node(
        result=_non_empty_str(brand.founder_name),
        id=f"Brand_{idx + 1}_founder_name_provided",
        desc=f"Brand {idx + 1}: founder name is provided",
        parent=founder_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_urls(brand.founder_urls),
        id=f"Brand_{idx + 1}_founder_urls_provided",
        desc=f"Brand {idx + 1}: at least one founder reference URL is provided",
        parent=founder_node,
        critical=True,
    )

    combined_sources = _merge_sources(brand.founder_urls, brand.brand_urls)

    # Leaf: Founder is actual founder (not ambassador)
    is_founder_leaf = evaluator.add_leaf(
        id=f"Brand_{idx + 1}_founder_is_founder",
        desc="URL(s) support that the named person founded/co-founded/launched the brand (not merely ambassador/spokesperson)",
        parent=founder_node,
        critical=True,
    )
    is_founder_claim = (
        f"{founder_label} founded or co-founded the brand '{brand_label}' (i.e., they are a founder/creator/owner), "
        "not just an ambassador or spokesperson."
    )

    # Leaf: Founder was primarily known as a celebrity before launching
    is_celebrity_leaf = evaluator.add_leaf(
        id=f"Brand_{idx + 1}_founder_is_celebrity",
        desc="URL(s) support that the founder was primarily known as a celebrity (actor, singer, model, TV personality) before launching the brand",
        parent=founder_node,
        critical=True,
    )
    is_celebrity_claim = (
        f"Before launching '{brand_label}', {founder_label} was primarily known as a celebrity in entertainment "
        "(actor, singer, model, or television personality)."
    )

    founder_batch = [
        (
            is_founder_claim,
            combined_sources,
            is_founder_leaf,
            "The source should clearly use terms like 'founder', 'co‑founder', 'launched by', or 'created by'. "
            "Do NOT accept 'ambassador', 'face of', 'partnered with', 'collaboration with a retailer' as proof of founding."
        ),
        (
            is_celebrity_claim,
            combined_sources,
            is_celebrity_leaf,
            "Verify the person is primarily known as a celebrity in entertainment (actor/singer/model/TV personality). "
            "Biographies, Wikipedia, and credible media can demonstrate this. Prefer evidence that predates the brand’s launch."
        ),
    ]

    # Run Founder group verifications in parallel
    await evaluator.batch_verify(founder_batch)


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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across brands for partial credit
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluation of whether 5 celebrity-founded beauty or personal care brands meeting all specified criteria have been correctly identified with required information",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="celebrity_beauty_brands_extraction",
    )

    # Normalize to exactly 5 items (pad with empty BrandItem if needed; take first 5 if more)
    items: List[BrandItem] = list(extracted.brands[:5])
    while len(items) < 5:
        items.append(BrandItem())

    # Build and verify per-brand subtrees
    for idx in range(5):
        await verify_single_brand(evaluator, root, items[idx], idx)

    # Return evaluation summary
    return evaluator.get_summary()