import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "two_wireless_anc_overear_headphones_bigspring2026"
TASK_DESCRIPTION = (
    "Find two different models of wireless over-ear headphones with active noise cancellation that are currently on sale "
    "during the Amazon Big Spring Sale 2026 (March 25–31) or available with early deals. Each headphone must meet all of the following requirements:\n"
    "- The sale price must be under $300\n"
    "- A discount from the original/regular price must be shown\n"
    "- The product must be available for store pickup at a retail location in either the San Francisco Bay Area or the Los Angeles metropolitan area, California\n"
    "- The product must be purchasable from one of these retailers: Amazon, Best Buy, or Target\n\n"
    "For each of the two headphones, provide:\n"
    "1. Full product name and model number\n"
    "2. Brand/manufacturer name\n"
    "3. Original/regular price\n"
    "4. Current sale price\n"
    "5. Discount amount (in dollars) or percentage off\n"
    "6. Specific store pickup location (store name and city)\n"
    "7. Direct URL to the product purchase page showing the sale price and pickup availability"
)

ALLOWED_RETAILERS = ["Amazon", "Best Buy", "Target"]
ALLOWED_DOMAINS = ["amazon.com", "bestbuy.com", "target.com"]
BAY_AREA_CITIES = [
    "San Francisco", "Oakland", "Berkeley", "San Jose", "Santa Clara", "Sunnyvale", "Mountain View",
    "Palo Alto", "Redwood City", "San Mateo", "Daly City", "Fremont", "Hayward", "Walnut Creek",
    "Pleasanton", "Milpitas", "Cupertino", "San Leandro", "San Rafael"
]
LA_AREA_CITIES = [
    "Los Angeles", "Santa Monica", "Pasadena", "Glendale", "Burbank", "Long Beach", "Anaheim", "Irvine",
    "Costa Mesa", "Inglewood", "Torrance", "West Hollywood", "Beverly Hills", "Culver City", "Pomona", "Fullerton"
]
BIG_SPRING_2026_WINDOW = "March 25–31, 2026"  # For instruction context

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PickupLocation(BaseModel):
    store: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    raw: Optional[str] = None  # Fallback combined text like "Best Buy San Jose (Stevens Creek), CA"


class HeadphoneItem(BaseModel):
    product_name: Optional[str] = None
    model_number: Optional[str] = None
    brand: Optional[str] = None
    original_price: Optional[str] = None
    sale_price: Optional[str] = None
    discount: Optional[str] = None  # Either "$X off" or "Y% off" or similar
    pickup_location: Optional[PickupLocation] = None
    retailer: Optional[str] = None  # "Amazon", "Best Buy", or "Target" if stated; else null
    product_url: Optional[str] = None  # Direct purchase page showing price and pickup availability
    additional_urls: List[str] = Field(default_factory=list)  # Any extra URLs mentioned for the same item
    sale_timing_label: Optional[str] = None  # e.g., "Big Spring Sale", "Early Deal", etc.


class HeadphoneItemsExtraction(BaseModel):
    items: List[HeadphoneItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_headphones() -> str:
    return """
You will extract up to TWO headphone product entries from the answer. Each entry should correspond to a wireless, over-ear headphone model with active noise cancellation (ANC) that the answer claims meets the task constraints.

For each headphone item you find (in the order they appear), extract these fields:
- product_name: Full product name as written in the answer text (e.g., "Sony WH-1000XM5 Wireless Noise-Canceling Headphones")
- model_number: The specific model number (e.g., "WH-1000XM5"); if stated inline in the product name, still extract it into this field. If not provided, return null.
- brand: Brand or manufacturer name (e.g., Sony, Bose)
- original_price: The regular/original price string exactly as shown in the answer (keep $ sign and formatting; do not convert)
- sale_price: The current sale price string exactly as shown (keep $ sign and formatting; do not convert)
- discount: The discount information string as shown, such as "$70 off" or "20% off". If multiple formats appear, pick the most precise one (prefer exact $ off or % off). If not present, return null.
- pickup_location: an object with:
  - store: pickup store name if present (e.g., "Best Buy", "Target West Hollywood")
  - city: the city if present (e.g., "San Jose", "Los Angeles")
  - state: state if present (e.g., "CA")
  - raw: the raw combined pickup text if the answer provides a single string (e.g., "Best Buy San Jose (Stevens Creek), CA"). If separate fields above are extracted, raw can repeat the combined text; if nothing is provided, set to null.
- retailer: The retailer name if explicitly mentioned (Amazon, Best Buy, or Target). If not mentioned but can be inferred from the product URL domain, set it to that retailer. Otherwise null.
- product_url: The direct product purchase page URL that shows the sale price and pickup availability (if multiple URLs are given, choose the one most likely to show BOTH price and pickup). If no valid URL appears, return null.
- additional_urls: Any other URLs in the answer for this same item (e.g., a second link for store inventory or pickup selection). Include only valid URLs explicitly present; do not invent URLs.
- sale_timing_label: If the answer indicates that the deal is part of "Amazon Big Spring Sale 2026", "Big Spring Deals", "Early Deal", or similar spring-sale phrasing, extract that label text. Otherwise null.

Rules:
- Extract EXACTLY what is written in the answer for text fields; do not infer unseen details.
- Only extract up to two items in the order they appear in the answer (ignore any additional items beyond two).
- If any required field is missing for an item, set it to null for that field.
- For URLs, accept plain URLs or markdown links; always output full absolute URLs including protocol.
- If retailer is not explicit, infer retailer from the product_url domain (amazon.com -> Amazon; bestbuy.com -> Best Buy; target.com -> Target) and fill 'retailer' accordingly.
- Do not include items that are clearly not headphones or not ANC over-ear wireless models according to the answer text.

Return a JSON with a top-level key 'items' as a list of the extracted objects.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _first_two_items(extracted: HeadphoneItemsExtraction) -> List[HeadphoneItem]:
    items = extracted.items[:2]
    while len(items) < 2:
        items.append(HeadphoneItem())
    return items


def _urls_for_item(item: HeadphoneItem) -> List[str]:
    urls: List[str] = []
    if item.product_url:
        urls.append(item.product_url)
    if item.additional_urls:
        for u in item.additional_urls:
            if u and isinstance(u, str):
                urls.append(u)
    return urls


def _pickup_label(item: HeadphoneItem) -> str:
    if item.pickup_location:
        pl = item.pickup_location
        if pl.raw:
            return pl.raw
        parts = []
        if pl.store:
            parts.append(pl.store)
        if pl.city:
            parts.append(pl.city)
        if pl.state:
            parts.append(pl.state)
        return ", ".join(parts) if parts else ""
    return ""


def _full_title_for_diff(item: HeadphoneItem) -> str:
    parts = []
    if item.brand:
        parts.append(item.brand)
    if item.product_name:
        parts.append(item.product_name)
    if item.model_number:
        parts.append(item.model_number)
    return " ".join(parts).strip() or "(missing)"


# --------------------------------------------------------------------------- #
# Per-item verification                                                       #
# --------------------------------------------------------------------------- #
async def verify_headphone_item(evaluator: Evaluator, parent_node, item: HeadphoneItem, idx: int) -> None:
    # Parent node for this headphone (non-critical to allow partial credit per item)
    h_node = evaluator.add_parallel(
        id=f"Headphone_{idx+1}",
        desc=f"{'First' if idx == 0 else 'Second'} headphone model satisfies all constraints and includes all required fields.",
        parent=parent_node,
        critical=False
    )

    urls = _urls_for_item(item)
    pickup_text = _pickup_label(item)

    # Category: Wireless Over-Ear with ANC
    cat_leaf = evaluator.add_leaf(
        id=f"H{idx+1}_Category_Is_Wireless_OverEar_ANC",
        desc=f"Headphone {idx+1} is wireless, over-ear, and has active noise cancellation (ANC).",
        parent=h_node,
        critical=True
    )
    cat_claim = (
        "This product is a pair of wireless, over-ear headphones with active noise cancellation (ANC). "
        "Accept if the page clearly indicates over-ear (also called around-ear or circumaural), Bluetooth/wireless connectivity, "
        "and noise cancelling (ANC/Active Noise Cancelling/Noise Canceling). Reject if on-ear or in-ear."
    )
    await evaluator.verify(
        claim=cat_claim,
        node=cat_leaf,
        sources=urls,
        additional_instruction="Look for keywords like 'over-ear', 'around-ear', 'Bluetooth', 'wireless', 'ANC', 'active noise cancel(l)ing'."
    )

    # Retailer is allowed
    retailer_leaf = evaluator.add_leaf(
        id=f"H{idx+1}_Retailer_Is_Allowed",
        desc=f"Headphone {idx+1} is purchasable from Amazon, Best Buy, or Target.",
        parent=h_node,
        critical=True
    )
    retailer_claim = (
        "The product page belongs to one of the allowed retailers: Amazon (amazon.com), Best Buy (bestbuy.com), or Target (target.com). "
        "Judge based on the page URL domain."
    )
    await evaluator.verify(
        claim=retailer_claim,
        node=retailer_leaf,
        sources=urls,
        additional_instruction="Pass only if the domain is amazon.com (or subdomain), bestbuy.com (or subdomain), or target.com (or subdomain)."
    )

    # Sale timing (Big Spring Sale 2026 or Early Deal)
    timing_leaf = evaluator.add_leaf(
        id=f"H{idx+1}_Sale_Timing_Is_BigSpringOrEarlyDeal",
        desc=f"Headphone {idx+1} is on sale during Amazon Big Spring Sale 2026 (Mar 25–31) or available as an early deal.",
        parent=h_node,
        critical=True
    )
    st_label = item.sale_timing_label or ""
    timing_claim = (
        "The product page indicates the deal is part of 'Amazon Big Spring Sale 2026' (or 'Big Spring Deals') "
        "OR it is explicitly marked as an 'Early Deal' for that event."
    )
    await evaluator.verify(
        claim=timing_claim,
        node=timing_leaf,
        sources=urls,
        additional_instruction=(
            "Accept if the page shows labels like 'Big Spring Sale', 'Big Spring Deals', 'Early Deal', 'Spring Sale'. "
            "If the retailer page explicitly mentions 'Amazon Big Spring Sale 2026' or 'Early Deal', pass."
        )
    )

    # Sale price under $300
    under300_leaf = evaluator.add_leaf(
        id=f"H{idx+1}_Sale_Price_Under_300",
        desc=f"Headphone {idx+1} current sale price is under $300.",
        parent=h_node,
        critical=True
    )
    price_hint = item.sale_price or ""
    under300_claim = (
        "The current sale price shown on the page for this product is less than $300 (USD). "
        "Focus on the sale/discounted price, not the regular/original price."
    )
    await evaluator.verify(
        claim=under300_claim,
        node=under300_leaf,
        sources=urls,
        additional_instruction="If multiple prices appear, use the discounted/sale price. Minor rounding differences are okay; it must be < $300."
    )

    # Discount exists from regular price
    discount_leaf = evaluator.add_leaf(
        id=f"H{idx+1}_Discount_Exists_From_Regular",
        desc=f"Headphone {idx+1} reflects a genuine discount from regular/original pricing.",
        parent=h_node,
        critical=True
    )
    discount_claim = (
        f"The page shows a genuine discount: the original/regular price ({item.original_price or 'original'}) is higher than the current sale price "
        f"({item.sale_price or 'sale'}), and/or a discount amount/percentage (e.g., {item.discount or 'X% off'}) is displayed and > 0."
    )
    await evaluator.verify(
        claim=discount_claim,
        node=discount_leaf,
        sources=urls,
        additional_instruction="Pass if the page clearly shows an original/regular price higher than the sale price OR an explicit amount/percent off."
    )

    # Pickup availability in SF Bay Area or LA metro
    pickup_leaf = evaluator.add_leaf(
        id=f"H{idx+1}_Pickup_Available_In_SF_BayArea_Or_LA_CA",
        desc=f"Headphone {idx+1} is available for store pickup in SF Bay Area or Los Angeles metro, CA.",
        parent=h_node,
        critical=True
    )
    pickup_claim = (
        f"The page indicates pickup availability at a California store location corresponding to the San Francisco Bay Area or Los Angeles metro. "
        f"The answer-provided pickup detail: '{pickup_text}'."
    )
    await evaluator.verify(
        claim=pickup_claim,
        node=pickup_leaf,
        sources=urls,
        additional_instruction=(
            "Check for store pickup at a city in these regions.\n"
            f"Bay Area examples: {', '.join(BAY_AREA_CITIES)}.\n"
            f"Los Angeles metro examples: {', '.join(LA_AREA_CITIES)}.\n"
            "Accept 'Store pickup', 'Curbside pickup', or similar. The page should show a store/city in these regions."
        )
    )

    # Required fields block (critical; each sub-field critical)
    req_node = evaluator.add_parallel(
        id=f"H{idx+1}_Required_Fields",
        desc=f"Headphone {idx+1} includes each required output field.",
        parent=h_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(item.product_name and item.product_name.strip()) and bool(item.model_number and item.model_number.strip()),
        id=f"H{idx+1}_Product_Name_And_Model_Number_Provided",
        desc=f"Provides full product name and model number for Headphone {idx+1}.",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(item.brand and item.brand.strip()),
        id=f"H{idx+1}_Brand_Provided",
        desc=f"Provides brand/manufacturer name for Headphone {idx+1}.",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(item.original_price and item.original_price.strip()),
        id=f"H{idx+1}_Original_Price_Provided",
        desc=f"Provides original/regular price for Headphone {idx+1}.",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(item.sale_price and item.sale_price.strip()),
        id=f"H{idx+1}_Sale_Price_Provided",
        desc=f"Provides current sale price for Headphone {idx+1}.",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(item.discount and item.discount.strip()),
        id=f"H{idx+1}_Discount_Amount_Or_Percent_Provided",
        desc=f"Provides discount amount (in dollars) or percentage off for Headphone {idx+1}.",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(pickup_text.strip()),
        id=f"H{idx+1}_Pickup_Location_Provided",
        desc=f"Provides a specific store pickup location (store name and city) for Headphone {idx+1}.",
        parent=req_node,
        critical=True
    )

    # Direct URL provided AND shows price and pickup (single verification leaf)
    direct_url_leaf = evaluator.add_leaf(
        id=f"H{idx+1}_Direct_URL_Provided_And_Shows_Price_And_Pickup",
        desc=f"Provides a direct URL to the product purchase page that shows the current sale price and pickup availability for Headphone {idx+1}.",
        parent=req_node,
        critical=True
    )
    direct_url_claim = (
        f"The provided product page shows the current sale price and also displays pickup availability or store pickup details for this headphone."
    )
    await evaluator.verify(
        claim=direct_url_claim,
        node=direct_url_leaf,
        sources=urls if urls else None,
        additional_instruction=(
            "Pass only if the URL loads a product page that clearly shows the current sale price and pickup availability (e.g., 'Store pickup', 'Pick Up at', "
            "or a store/city selector showing availability). If there is no valid URL or the page does not show pickup, fail."
        )
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_headphones(),
        template_class=HeadphoneItemsExtraction,
        extraction_name="headphone_items"
    )

    # Keep the first two items for subsequent verification; also compute original count for the "exactly two" check
    original_count = len(extracted.items)
    items = _first_two_items(extracted)

    # Add some context info for transparency
    evaluator.add_ground_truth({
        "allowed_retailers": ALLOWED_RETAILERS,
        "allowed_domains": ALLOWED_DOMAINS,
        "big_spring_2026_window": BIG_SPRING_2026_WINDOW,
        "sf_bay_area_examples": BAY_AREA_CITIES,
        "la_metro_examples": LA_AREA_CITIES
    }, gt_type="policy_context")

    # Main node (kept non-critical to allow nuanced partial scores across sub-criteria and items)
    main_node = evaluator.add_parallel(
        id="Two_Wireless_ANC_OverEar_Headphones_On_Sale_With_Pickup",
        desc="Identify exactly two different wireless over-ear ANC headphone models meeting all constraints (sale timing, price, discount, retailer, CA pickup region) and provide all requested info fields for each.",
        parent=root,
        critical=False
    )

    # Item count must be exactly two (critical)
    evaluator.add_custom_node(
        result=(original_count == 2),
        id="Item_Count_Is_Exactly_Two",
        desc="Response provides exactly 2 headphone product models (not 1, not 3+).",
        parent=main_node,
        critical=True
    )

    # The two models are different (critical) - use LLM logical check for robustness
    model_diff_leaf = evaluator.add_leaf(
        id="Models_Are_Different",
        desc="The two provided headphones are different models (not the same model repeated).",
        parent=main_node,
        critical=True
    )
    title1 = _full_title_for_diff(items[0])
    title2 = _full_title_for_diff(items[1])
    diff_claim = (
        f"The two items are different headphone models: '{title1}' vs '{title2}'. "
        "They should not represent the same underlying model; consider model numbers and naming (minor formatting differences do not count as different)."
    )
    await evaluator.verify(
        claim=diff_claim,
        node=model_diff_leaf,
        additional_instruction="Treat them as the same if model numbers are the same or names clearly refer to the same product variant."
    )

    # Per-item verifications
    await verify_headphone_item(evaluator, main_node, items[0], idx=0)
    await verify_headphone_item(evaluator, main_node, items[1], idx=1)

    return evaluator.get_summary()