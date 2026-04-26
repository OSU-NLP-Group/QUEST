import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wireless_earbuds_anc_us_market"
TASK_DESCRIPTION = """
I'm shopping for wireless earbuds for daily commuting and travel. Please help me find four different models of true wireless earbuds currently available in the US market that meet ALL of the following requirements:

1. Active Noise Cancellation: Must have active noise cancellation (ANC) technology
2. Battery Life: Must provide at least 8 hours of continuous playback with ANC enabled
3. Water Resistance: Must have at least an IPX4 water resistance rating (or equivalent/better IP rating such as IP54, IP57, etc.)
4. Availability: Must be currently available for purchase from at least two major US retailers (such as Amazon, Best Buy, Walmart, Apple, or the manufacturer's official US website)
5. Price: Current retail price must not exceed $350 (regular price, not temporary sales)

For each of the four models you identify, provide:
- Exact model name and number
- Battery life with ANC enabled (in hours)
- IPX/IP water resistance rating
- Current retail price
- Direct purchase links from at least two different retailers
"""

BUDGET_LIMIT_USD = 350.0

# For analytics / guidance only (not used as a hard filter)
KNOWN_MAJOR_US_RETAILERS = [
    "amazon.com",
    "bestbuy.com",
    "walmart.com",
    "apple.com",
    "store.google.com",
    "target.com",
    "bhphotovideo.com",
    "costco.com",
    "newegg.com",
    "microcenter.com",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EarbudItem(BaseModel):
    """One true wireless earbud entry extracted from the answer."""
    model_name: Optional[str] = None
    battery_life_with_anc_hours: Optional[str] = None
    water_resistance_rating: Optional[str] = None
    current_price_usd: Optional[str] = None
    # Direct purchase links; should include at least two different retailers’ product pages.
    retailer_urls: List[str] = Field(default_factory=list)
    # Reference/specification URLs; manufacturer official product pages or reliable retailer pages that include specs.
    spec_urls: List[str] = Field(default_factory=list)


class EarbudsExtraction(BaseModel):
    """List of earbuds extracted from the answer."""
    items: List[EarbudItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_earbuds() -> str:
    return """
    Extract up to four distinct models of true wireless earbuds described in the answer that the user proposes. For each model, return a JSON object with the following fields:
    - model_name: The exact model name and number as written (e.g., "Sony WF-1000XM5")
    - battery_life_with_anc_hours: The claimed battery life WITH ANC enabled (in hours) as stated in the answer; return the value exactly as written (e.g., "8", "8 hours", "up to 8")
    - water_resistance_rating: The IP rating string exactly as written (e.g., "IPX4", "IP54", "IP57", "IPX5", etc.). If only generic 'water resistant' is mentioned without an IP rating, return the string as-is.
    - current_price_usd: The current retail price in USD as written (include the currency symbol if present; e.g., "$299")
    - retailer_urls: A list of direct purchase URLs explicitly mentioned in the answer for this model (e.g., Amazon, Best Buy, Walmart, Apple, manufacturer official US store). Include only product purchase pages, not category pages.
    - spec_urls: A list of reference/specification URLs explicitly mentioned in the answer for this model (e.g., manufacturer official product pages, or reliable retailer product pages that include specs). If the answer reuses retailer URLs for specs, include them here as well.

    IMPORTANT:
    - Extract ONLY what is explicitly present in the answer text. Do not invent, infer, or add any information.
    - For URLs, include only valid URLs that appear in the answer (plain URLs or markdown links).
    - If any field is missing for a model, set it to null or an empty list (for URL lists).
    - Return at most four models, in the same order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_domain(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url.strip())
        return parsed.netloc.lower()
    except Exception:
        return None


def pick_two_distinct_urls(urls: List[str]) -> List[str]:
    """Pick up to two URLs from different domains."""
    seen = set()
    picked = []
    for u in urls:
        d = get_domain(u)
        if not d:
            continue
        if d not in seen:
            seen.add(d)
            picked.append(u)
        if len(picked) >= 2:
            break
    return picked


def combine_unique_urls(a: List[str], b: List[str]) -> List[str]:
    seen = set()
    combined = []
    for u in a + b:
        uu = u.strip()
        if uu and uu not in seen:
            seen.add(uu)
            combined.append(uu)
    return combined


# --------------------------------------------------------------------------- #
# Verification for a single earbud                                            #
# --------------------------------------------------------------------------- #
async def verify_single_earbud(
    evaluator: Evaluator,
    parent_node,
    index: int,
    item: EarbudItem,
) -> None:
    """
    Build verification sub-tree and run checks for one earbud item.
    All core checks are critical; failure of any core criterion disqualifies the model.
    """

    # Parent node for this earbud
    earbud_node = evaluator.add_parallel(
        id=f"earbud_{index}",
        desc=f"Earbud #{index + 1}: verification against all requirements",
        parent=parent_node,
        critical=False,  # non-critical at the parent level to allow partial credit across items
    )

    model_name = (item.model_name or "").strip()
    spec_urls = item.spec_urls or []
    retailer_urls = item.retailer_urls or []
    combined_refs = combine_unique_urls(spec_urls, retailer_urls)
    two_retailer_urls = pick_two_distinct_urls(retailer_urls)

    # 0) Basic data presence prerequisite (critical)
    evaluator.add_custom_node(
        result=bool(model_name) and len(combined_refs) >= 1,
        id=f"earbud_{index}_model_data_present",
        desc="Model name provided and at least one reference URL present",
        parent=earbud_node,
        critical=True,
    )

    # 1) Model Identification and Reference (critical)
    model_id_leaf = evaluator.add_leaf(
        id=f"earbud_{index}_Model_Identification_and_Reference",
        desc="Provide the exact model name/number and reference URLs that verify the model's existence and specifications",
        parent=earbud_node,
        critical=True,
    )
    claim_model_id = (
        f"The provided URLs are official product pages or reliable retailer product pages for the model '{model_name}', "
        f"and they confirm the model's existence and list its specifications."
    )
    await evaluator.verify(
        claim=claim_model_id,
        node=model_id_leaf,
        sources=combined_refs,
        additional_instruction=(
            "Confirm that the pages explicitly reference the exact model name/number (allowing minor formatting differences). "
            "These should be manufacturer official product pages OR well-known retailer product pages showing detailed specs."
        ),
    )

    # 2) ANC Feature (critical)
    anc_leaf = evaluator.add_leaf(
        id=f"earbud_{index}_Active_Noise_Cancellation_Feature",
        desc="Verify the earbuds have active noise cancellation (ANC) technology as stated in official specifications",
        parent=earbud_node,
        critical=True,
    )
    claim_anc = (
        f"The model '{model_name}' includes Active Noise Cancellation (ANC), as indicated on the official/spec pages."
    )
    await evaluator.verify(
        claim=claim_anc,
        node=anc_leaf,
        sources=spec_urls if spec_urls else combined_refs,
        additional_instruction=(
            "Look for terms like 'Active Noise Cancellation', 'ANC', 'Hybrid ANC', or 'Adaptive Noise Cancelling'. "
            "Noise isolation or passive noise reduction alone is NOT sufficient."
        ),
    )

    # 3) Battery Life with ANC >= 8 hours (critical)
    battery_leaf = evaluator.add_leaf(
        id=f"earbud_{index}_Battery_Life_Specification",
        desc="Verify the earbuds provide at least 8 hours of continuous playback with ANC enabled",
        parent=earbud_node,
        critical=True,
    )
    claim_battery = (
        f"The model '{model_name}' provides at least 8 hours of continuous playback WITH ANC enabled."
    )
    await evaluator.verify(
        claim=claim_battery,
        node=battery_leaf,
        sources=spec_urls if spec_urls else combined_refs,
        additional_instruction=(
            "Prefer manufacturer specifications. The page should explicitly reference playback time with ANC ON. "
            "Accept phrasing like 'up to 8 hours with ANC'. If only non-ANC playback time is provided, treat as not supported."
        ),
    )

    # 4) Water Resistance rating >= IPX4 or equivalent (critical)
    water_leaf = evaluator.add_leaf(
        id=f"earbud_{index}_Water_Resistance_Rating",
        desc="Verify the earbuds have at least an IPX4 water resistance rating (or equivalent/better IP rating)",
        parent=earbud_node,
        critical=True,
    )
    claim_water = (
        f"The model '{model_name}' has an IP rating of at least IPX4 (or equivalent/better such as IP54, IP55, IP57, IPX5, IPX7)."
    )
    await evaluator.verify(
        claim=claim_water,
        node=water_leaf,
        sources=spec_urls if spec_urls else combined_refs,
        additional_instruction=(
            "Check the stated IP rating. Accept IPX4 or any higher/equivalent rating (e.g., IPX5, IP55, IP57, IPX7). "
            "Generic 'water resistant' without an explicit IP rating is insufficient."
        ),
    )

    # 5) Retailer availability prerequisites (critical)
    evaluator.add_custom_node(
        result=len(two_retailer_urls) >= 2,
        id=f"earbud_{index}_retailer_links_two_distinct",
        desc="At least two distinct retailer purchase URLs are provided",
        parent=earbud_node,
        critical=True,
    )

    # 5.a) Retailer Availability (split across two leaves; both must pass)
    retailer_avail_node = evaluator.add_parallel(
        id=f"earbud_{index}_Retailer_Availability",
        desc="Verify availability from at least two major US retailers via direct purchase URLs",
        parent=earbud_node,
        critical=True,
    )

    # Ensure two leaves: first two distinct domains
    for j, url in enumerate(two_retailer_urls[:2]):
        leaf = evaluator.add_leaf(
            id=f"earbud_{index}_retailer_availability_{j+1}",
            desc=f"Retailer availability check #{j+1} for a direct purchase URL",
            parent=retailer_avail_node,
            critical=True,
        )
        claim_avail = (
            f"This page is a direct purchase product page for '{model_name}' from a major US retailer or an official US manufacturer store, "
            f"and the item is currently available to buy (e.g., Add to Cart/Buy/Ships/In Stock)."
        )
        await evaluator.verify(
            claim=claim_avail,
            node=leaf,
            sources=url,
            additional_instruction=(
                "Verify that this is a product detail page (not a category/landing page) and shows availability for purchase in the US. "
                "Look for 'Add to Cart', 'Buy', 'In stock', shipping information to US, or similar signals. "
                "If 'Sold out' or 'Unavailable', treat as not supported."
            ),
        )

    # 6) Price Requirement (<= $350) – check across two retailer URLs (both must pass)
    price_parent = evaluator.add_parallel(
        id=f"earbud_{index}_Price_Requirement",
        desc=f"Verify the current retail price does not exceed ${BUDGET_LIMIT_USD} USD (regular price, not temporary sales)",
        parent=earbud_node,
        critical=True,
    )

    for j, url in enumerate(two_retailer_urls[:2]):
        price_leaf = evaluator.add_leaf(
            id=f"earbud_{index}_price_check_{j+1}",
            desc=f"Price check #{j+1} on retailer page (<= ${BUDGET_LIMIT_USD})",
            parent=price_parent,
            critical=True,
        )
        claim_price = (
            f"The current regular price for '{model_name}' on this page is at most ${int(BUDGET_LIMIT_USD)}."
        )
        await evaluator.verify(
            claim=claim_price,
            node=price_leaf,
            sources=url,
            additional_instruction=(
                "Look for the standard listed price (regular price) on the page and confirm it does not exceed $350. "
                "If the page only shows a temporary sale price significantly below a higher regular price, treat this as not supported."
            ),
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
) -> Dict[str, Any]:
    """
    Entry point for evaluating an agent's answer for the wireless earbuds task.
    Builds a hierarchical verification tree and returns a structured summary.
    """
    # Initialize evaluator with a parallel root strategy
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

    # Record known retailer info (for transparency)
    evaluator.add_custom_info(
        info={"known_major_us_retailers": KNOWN_MAJOR_US_RETAILERS, "budget_limit_usd": BUDGET_LIMIT_USD},
        info_type="context",
        info_name="retailer_and_budget_context"
    )

    # Extract earbuds from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_earbuds(),
        template_class=EarbudsExtraction,
        extraction_name="earbuds_extraction",
    )

    # Create main task node (optional; use root or a child node to group all)
    task_node = evaluator.add_parallel(
        id="Find_Four_Wireless_Earbuds",
        desc="Identify four different models of true wireless earbuds that meet all specified criteria",
        parent=root,
        critical=False,
    )

    # Normalize to exactly four entries (pad with empty items if needed; trim extra if provided)
    items: List[EarbudItem] = extracted.items[:4]
    while len(items) < 4:
        items.append(EarbudItem())

    # Build verification sub-trees for each earbud (sequential checks within each function call)
    for idx, item in enumerate(items):
        await verify_single_earbud(evaluator, task_node, idx, item)

    # Return final structured summary
    return evaluator.get_summary()