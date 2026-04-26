import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "beauty_brands_2019_2022_sephora_celebrity"
TASK_DESCRIPTION = """
Identify 4 beauty brands that meet ALL of the following criteria:

1. The brand was launched between January 1, 2019 and December 31, 2022 (inclusive)
2. The brand is currently available for purchase at Sephora stores or sephora.com in the United States
3. The brand includes skincare products as part of its product line
4. The brand offers products in at least 2 different product categories (such as skincare + makeup, or skincare + body care, or skincare + haircare)
5. The brand is marketed as cruelty-free
6. The brand is still operating and available for purchase as of December 2025
7. The brand was founded or co-founded by a celebrity who is primarily known as an actress, model, or singer/musician
8. The brand was initially launched in or made available to the United States market
9. The brand operates under its own distinct name (not as a sub-line of an existing major beauty conglomerate's brand)

For each of the 4 brands, provide:
- The founder's full name
- The brand name
- The year the brand was launched
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class URLBucket(BaseModel):
    sephora_urls: List[str] = Field(default_factory=list, description="URLs to sephora.com pages for this brand or its products (US site).")
    founder_urls: List[str] = Field(default_factory=list, description="URLs supporting founder identity/role/background (e.g., brand site, press, Wikipedia).")
    launch_urls: List[str] = Field(default_factory=list, description="URLs supporting launch year details (e.g., press releases, news coverage).")
    us_market_urls: List[str] = Field(default_factory=list, description="URLs supporting US-launch/US-availability context.")
    independence_urls: List[str] = Field(default_factory=list, description="URLs showing brand is a standalone brand (not a sub-line under another brand).")
    skincare_urls: List[str] = Field(default_factory=list, description="URLs proving presence of skincare products.")
    categories_urls: List[str] = Field(default_factory=list, description="URLs showing at least two distinct product categories.")
    cruelty_free_urls: List[str] = Field(default_factory=list, description="URLs supporting cruelty-free status (brand page, PETA, Leaping Bunny, Sephora claim).")
    operating_status_urls: List[str] = Field(default_factory=list, description="URLs indicating the brand is actively operating/selling as of late 2025.")
    general_urls: List[str] = Field(default_factory=list, description="Any other URLs cited in the answer relevant to this brand.")


class BrandItem(BaseModel):
    brand_name: Optional[str] = None
    founder_name: Optional[str] = None
    launch_year: Optional[str] = None
    urls: URLBucket = Field(default_factory=URLBucket)


class BrandsExtraction(BaseModel):
    items: List[BrandItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return """
    Extract up to 4 beauty brands mentioned in the answer that meet the task's constraints.
    For each brand, extract the following fields:

    1) brand_name: the exact brand name string as written in the answer.
    2) founder_name: the celebrity founder/co-founder full name as stated in the answer.
    3) launch_year: the year the brand launched (four-digit year if available; if a full date is given, just provide the year).
    4) urls: Categorize all URLs cited in the answer for this brand into the following buckets (if present). If a URL does not obviously map to a bucket, place it in general_urls.
       - sephora_urls
       - founder_urls
       - launch_urls
       - us_market_urls
       - independence_urls
       - skincare_urls
       - categories_urls
       - cruelty_free_urls
       - operating_status_urls
       - general_urls

    IMPORTANT INSTRUCTIONS:
    - Only extract URLs that are explicitly present in the answer. Do not invent or infer URLs.
    - Include the protocol (http/https). If the protocol is missing, prepend http://.
    - Preserve up to the first 10 URLs overall per brand if too many are present.
    - Return at most 4 brands, in the order they appear in the answer. If more than 4 are present, keep only the first 4.
    - If any field is missing in the answer, set it to null (for strings) or an empty list (for URL arrays).
    - Do not perform validation here; just extract.

    Return a JSON object with a top-level field:
    {
      "items": [ { brand_name, founder_name, launch_year, urls: {...} }, ... up to 4 items ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_four_digit_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(20\d{2})\b", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _in_range_2019_2022(year: Optional[int]) -> bool:
    return year is not None and 2019 <= year <= 2022


def _uniq_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Normalize: ensure protocol
        if not u.startswith("http://") and not u.startswith("https://"):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def aggregate_urls(item: BrandItem, buckets: List[str]) -> List[str]:
    all_urls: List[str] = []
    for b in buckets:
        arr = getattr(item.urls, b, [])
        if isinstance(arr, list):
            all_urls.extend(arr)
    # Always consider general_urls as catch-all
    all_urls.extend(item.urls.general_urls or [])
    # Deduplicate
    return _uniq_urls(all_urls)


def ordinal(n: int) -> str:
    return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])


# --------------------------------------------------------------------------- #
# Verification logic per item                                                 #
# --------------------------------------------------------------------------- #
async def verify_item(evaluator: Evaluator, parent_node, item: BrandItem, index: int) -> None:
    item_no = index + 1
    item_node = evaluator.add_parallel(
        id=f"item_{item_no}",
        desc=f"{ordinal(item_no).capitalize()} qualifying beauty brand with all required information",
        parent=parent_node,
        critical=False  # Each item contributes to partial credit independently
    )

    brand_name = item.brand_name or ""
    founder_name = item.founder_name or ""
    launch_year_str = item.launch_year or ""
    launch_year_int = _first_four_digit_year(launch_year_str)

    # ---------------- Brand Information (Critical group) ---------------- #
    brand_info = evaluator.add_parallel(
        id=f"item_{item_no}_brand_information",
        desc=f"Brand identification and launch details for Item {item_no}",
        parent=item_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(brand_name.strip()),
        id=f"item_{item_no}_brand_name_provided",
        desc="Brand name is provided",
        parent=brand_info,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(launch_year_str.strip()),
        id=f"item_{item_no}_launch_year_provided",
        desc="Launch year is provided",
        parent=brand_info,
        critical=True
    )

    evaluator.add_custom_node(
        result=_in_range_2019_2022(launch_year_int),
        id=f"item_{item_no}_launch_year_range",
        desc="Launch year is between 2019 and 2022 (inclusive)",
        parent=brand_info,
        critical=True
    )

    op_status_leaf = evaluator.add_leaf(
        id=f"item_{item_no}_operating_status",
        desc="Brand is still operating and available for purchase as of December 2025",
        parent=brand_info,
        critical=True
    )
    op_sources = aggregate_urls(item, ["operating_status_urls", "sephora_urls"])
    op_claim = f"As of December 2025, the brand '{brand_name}' is still operating and has products available for purchase (e.g., on its official site or at Sephora in the US)."
    await evaluator.verify(
        claim=op_claim,
        node=op_status_leaf,
        sources=op_sources,
        additional_instruction="Use the provided pages (preferably sephora.com or the official brand shop) to confirm the brand was actively selling products around late 2025. Active brand/product listing or shop page suffices."
    )

    # --------------- Founder Information (Critical group) --------------- #
    founder_info = evaluator.add_parallel(
        id=f"item_{item_no}_founder_information",
        desc=f"Founder identification and professional background for Item {item_no}",
        parent=item_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(founder_name.strip()),
        id=f"item_{item_no}_founder_name_provided",
        desc="Founder's name is provided",
        parent=founder_info,
        critical=True
    )

    founder_role_leaf = evaluator.add_leaf(
        id=f"item_{item_no}_founder_role",
        desc="Celebrity is listed as founder or co-founder of the brand",
        parent=founder_info,
        critical=True
    )
    founder_role_sources = aggregate_urls(item, ["founder_urls", "launch_urls"])
    founder_role_claim = f"{founder_name} is the founder or co-founder of the brand '{brand_name}'."
    await evaluator.verify(
        claim=founder_role_claim,
        node=founder_role_leaf,
        sources=founder_role_sources,
        additional_instruction="Verify that the cited page(s) explicitly state the person is the brand's founder or co-founder (brand 'About' page, press release, or reputable media)."
    )

    founder_bg_leaf = evaluator.add_leaf(
        id=f"item_{item_no}_founder_professional_background",
        desc="Founder is primarily known as an actress, model, or singer/musician",
        parent=founder_info,
        critical=True
    )
    founder_bg_sources = aggregate_urls(item, ["founder_urls"])
    founder_bg_claim = f"{founder_name} is primarily known as an actress, model, or singer/musician."
    await evaluator.verify(
        claim=founder_bg_claim,
        node=founder_bg_leaf,
        sources=founder_bg_sources,
        additional_instruction="Use reputable biography/press pages (e.g., Wikipedia, major outlets) to confirm the person's main professional identity is actress, model, or singer/musician."
    )

    # --------- Distribution and Availability (Critical group) ----------- #
    dist_info = evaluator.add_parallel(
        id=f"item_{item_no}_distribution_and_availability",
        desc=f"Retail distribution and market availability for Item {item_no}",
        parent=item_node,
        critical=True
    )

    sephora_leaf = evaluator.add_leaf(
        id=f"item_{item_no}_sephora_availability",
        desc="Brand is available at Sephora stores or sephora.com in the United States",
        parent=dist_info,
        critical=True
    )
    sephora_sources = aggregate_urls(item, ["sephora_urls"])
    sephora_claim = f"The brand '{brand_name}' is available at Sephora stores or on sephora.com in the United States."
    await evaluator.verify(
        claim=sephora_claim,
        node=sephora_leaf,
        sources=sephora_sources,
        additional_instruction="Prefer a sephora.com (US) brand landing page or product page that clearly shows the brand and purchasable items."
    )

    us_launch_leaf = evaluator.add_leaf(
        id=f"item_{item_no}_us_market_launch",
        desc="Brand was initially launched in or made available to the United States market",
        parent=dist_info,
        critical=True
    )
    us_sources = aggregate_urls(item, ["us_market_urls", "sephora_urls", "launch_urls"])
    us_launch_claim = f"The brand '{brand_name}' was initially launched in or made available to the United States market."
    await evaluator.verify(
        claim=us_launch_claim,
        node=us_launch_leaf,
        sources=us_sources,
        additional_instruction="Evidence could include US press releases, US retail availability at/near launch, or official announcements specifying US market access."
    )

    independence_leaf = evaluator.add_leaf(
        id=f"item_{item_no}_brand_independence",
        desc="Brand operates under its own distinct name, not as a sub-line of an existing major beauty conglomerate's brand",
        parent=dist_info,
        critical=True
    )
    independence_sources = aggregate_urls(item, ["independence_urls", "sephora_urls"])
    independence_claim = f"The brand '{brand_name}' operates as a standalone brand under its own distinct name, not merely a sub-line under another brand."
    await evaluator.verify(
        claim=independence_claim,
        node=independence_leaf,
        sources=independence_sources,
        additional_instruction="It's acceptable if a parent company owns the brand; the key is that this brand is not just a sub-line labeled under another pre-existing brand."
    )

    # ------------- Product Characteristics (Critical group) ------------- #
    prod_info = evaluator.add_parallel(
        id=f"item_{item_no}_product_characteristics",
        desc=f"Product line attributes and certifications for Item {item_no}",
        parent=item_node,
        critical=True
    )

    skincare_leaf = evaluator.add_leaf(
        id=f"item_{item_no}_skincare_products",
        desc="Brand includes skincare products as part of its product line",
        parent=prod_info,
        critical=True
    )
    skincare_sources = aggregate_urls(item, ["skincare_urls", "sephora_urls"])
    skincare_claim = f"The brand '{brand_name}' includes skincare products in its product line."
    await evaluator.verify(
        claim=skincare_claim,
        node=skincare_leaf,
        sources=skincare_sources,
        additional_instruction="Look for explicit 'skincare' category pages or obvious skincare items (e.g., cleanser, serum, moisturizer) on sephora.com or the brand website."
    )

    multi_cat_leaf = evaluator.add_leaf(
        id=f"item_{item_no}_multiple_categories",
        desc="Brand offers products in at least 2 different categories (e.g., skincare + makeup, or skincare + haircare)",
        parent=prod_info,
        critical=True
    )
    categories_sources = aggregate_urls(item, ["categories_urls", "sephora_urls"])
    multi_cat_claim = f"The brand '{brand_name}' offers products in at least two distinct categories, including skincare plus at least one of makeup, haircare, body care, fragrance, or similar."
    await evaluator.verify(
        claim=multi_cat_claim,
        node=multi_cat_leaf,
        sources=categories_sources,
        additional_instruction="Confirm that at least two categories are clearly distinct (e.g., 'Skincare' and 'Makeup'). Category menus or product filters suffice."
    )

    cf_leaf = evaluator.add_leaf(
        id=f"item_{item_no}_cruelty_free_status",
        desc="Brand is marketed as cruelty-free",
        parent=prod_info,
        critical=True
    )
    cf_sources = aggregate_urls(item, ["cruelty_free_urls"])
    cf_claim = f"The brand '{brand_name}' is marketed as cruelty-free (not tested on animals)."
    await evaluator.verify(
        claim=cf_claim,
        node=cf_leaf,
        sources=cf_sources,
        additional_instruction="Accept brand's explicit cruelty-free statement or recognized certifications (Leaping Bunny, PETA). Retailer pages that mark the brand as 'cruelty-free' are also acceptable."
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
    Evaluate an answer for the 'celebrity-founded beauty brands (2019-2022) at Sephora' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Items are evaluated independently; allow partial credit
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

    # Extract structured brand info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="brands_extraction"
    )

    # Normalize to exactly 4 items (pad with empty if fewer)
    items = list(extracted.items[:4])
    while len(items) < 4:
        items.append(BrandItem())

    # Build verification tree per item
    for idx, item in enumerate(items):
        await verify_item(evaluator, root, item, idx)

    # Return standardized summary
    return evaluator.get_summary()