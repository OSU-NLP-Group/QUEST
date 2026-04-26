import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wireless_earbuds_selection"
TASK_DESCRIPTION = """I'm looking to purchase wireless earbuds with active noise cancellation for commuting and travel. I want to compare options across different price points before making a decision.

Please find three different pairs of wireless earbuds with active noise cancellation (ANC), one in each price category, to help me understand what features are available at different price levels.

For each pair of earbuds, please provide:
- The specific model name and manufacturer
- Current price (approximate range is acceptable)
- A reference URL that documents the product's specifications (this can be a review site, manufacturer page, or retailer page)

Each pair of earbuds must meet ALL of the following requirements:
1. Battery life: At least 7 hours of continuous playback with ANC enabled
2. Noise cancellation: Must feature active noise cancellation (ANC) technology
3. Bluetooth: Must support Bluetooth 5.3 or higher
4. Audio codec: Must support at least AAC codec for high-quality audio
5. Water resistance: Must have at least IPX4 rating for sweat and light rain protection
6. Total battery: Combined battery life with charging case must be at least 20 hours
7. Controls: Must include touch or button controls for playback and calls
8. Release date: Must be a 2024 or later model (current generation)
9. Availability: Must be currently available for purchase as of March 2026

Additional requirements for mid-range and premium options:
- Multipoint connectivity: Must support connecting to two devices simultaneously (required for mid-range and premium only)

The three pairs must fall into these distinct price categories:
- Budget option: Under $100
- Mid-range option: Between $100 and $200
- Premium option: Over $200

All three pairs must be from different manufacturers.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class EarbudItem(BaseModel):
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None
    price_text: Optional[str] = None
    reference_url: Optional[str] = None
    category_label: Optional[str] = None  # e.g., "budget", "mid-range", "premium"

    # Optional fields if the answer explicitly states them (not required for verification)
    anc: Optional[bool] = None
    bluetooth_version_text: Optional[str] = None
    codecs: List[str] = Field(default_factory=list)
    ip_rating_text: Optional[str] = None
    battery_life_anc_text: Optional[str] = None
    total_battery_text: Optional[str] = None
    controls_text: Optional[str] = None
    release_year_text: Optional[str] = None
    availability_note: Optional[str] = None
    multipoint: Optional[bool] = None


class EarbudsExtraction(BaseModel):
    items: List[EarbudItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_earbuds_list() -> str:
    return """
    Extract all wireless earbuds models mentioned in the answer. For each model, return:
    - manufacturer: Brand/manufacturer name (e.g., Sony, Apple, JBL)
    - model_name: Explicit model identifier (e.g., WF-1000XM5)
    - price_text: The stated current price or approximate price range as text (e.g., "$179", "$89–$99", "about $250")
    - reference_url: A single URL in the answer that documents the product specifications (manufacturer, retailer, or an in-depth review page). If multiple URLs are present, choose the most authoritative/spec-detailed one.
    - category_label: If the answer explicitly labels it as "budget", "mid-range", or "premium", capture that label in lowercase; otherwise null.

    Additionally, if the answer explicitly states them (optional; leave null/empty if not stated):
    - anc: true/false if active noise cancellation is mentioned
    - bluetooth_version_text: The stated Bluetooth version text if mentioned (e.g., "Bluetooth 5.3")
    - codecs: A list of codecs mentioned (e.g., ["SBC", "AAC", "LDAC"])
    - ip_rating_text: The stated IP rating text (e.g., "IPX4", "IP54")
    - battery_life_anc_text: The stated per-charge battery life with ANC on, if provided in the answer text
    - total_battery_text: The stated combined battery life with the case, if provided
    - controls_text: The stated controls (e.g., "touch controls", "button controls")
    - release_year_text: The stated release/launch/announce year if provided
    - availability_note: Any stated note about being available now
    - multipoint: true/false if multipoint is mentioned

    Return a JSON object with:
    {
      "items": [ EarbudItem, ... ]
    }

    Rules:
    - Extract only what is explicitly present in the answer. Do not invent any info.
    - If a field is missing for an item, set it to null (or [] for codecs).
    - Include every earbuds model mentioned in the answer, even if more than three.
    """


# --------------------------------------------------------------------------- #
# Helper functions for selection and parsing                                  #
# --------------------------------------------------------------------------- #
USD_PRICE_REGEX = re.compile(r"\$\s*([\d{1,3}(?:,\d{3})*]+(?:\.\d{1,2})?)", re.IGNORECASE)
USD_SIMPLE_REGEX = re.compile(r"\$\s*([0-9][0-9,]*\.?\d*)")


def _parse_first_usd_amount(price_text: Optional[str]) -> Optional[float]:
    if not price_text:
        return None
    m = USD_SIMPLE_REGEX.search(price_text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except Exception:
        return None


def _normalize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    l = label.strip().lower()
    if "budget" in l or "entry" in l or "under" in l:
        return "budget"
    if "mid" in l:
        return "midrange"
    if "premium" in l or "flagship" in l:
        return "premium"
    return None


def _classify_bucket_from_price(price: Optional[float]) -> Optional[str]:
    if price is None:
        return None
    if price < 100.0:
        return "budget"
    if 100.0 <= price <= 200.0:
        return "midrange"
    if price > 200.0:
        return "premium"
    return None


def _select_items_by_category(all_items: List[EarbudItem]) -> Dict[str, Optional[EarbudItem]]:
    """
    Choose up to one unique-manufacturer item for each category: budget, midrange, premium.
    Prefer explicit category labels; fall back to price-based inference.
    """
    # Decorate with inferred bucket and price
    decorated: List[Tuple[EarbudItem, Optional[str], Optional[float]]] = []
    for it in all_items:
        label_bucket = _normalize_label(it.category_label)
        price_value = _parse_first_usd_amount(it.price_text)
        price_bucket = _classify_bucket_from_price(price_value)
        bucket = label_bucket or price_bucket
        decorated.append((it, bucket, price_value))

    selected: Dict[str, Optional[EarbudItem]] = {"budget": None, "midrange": None, "premium": None}
    used_mfr: set = set()

    # Pass 1: use explicitly labeled items while ensuring unique manufacturers
    for bucket_name in ["budget", "midrange", "premium"]:
        for it, bucket, _ in decorated:
            if bucket == bucket_name and it.manufacturer and it.manufacturer not in used_mfr:
                selected[bucket_name] = it
                used_mfr.add(it.manufacturer)
                break

    # Pass 2: fill remaining by price inference while maintaining unique manufacturers
    for bucket_name in ["budget", "midrange", "premium"]:
        if selected[bucket_name] is None:
            for it, bucket, _ in decorated:
                if bucket == bucket_name:
                    mfr = it.manufacturer or ""
                    if (not mfr) or (mfr in used_mfr):
                        continue
                    selected[bucket_name] = it
                    used_mfr.add(mfr)
                    break

    return selected


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_url_reference(
    evaluator: Evaluator,
    parent,
    node_id: str,
    manufacturer: Optional[str],
    model_name: Optional[str],
    url: Optional[str],
    critical: bool = True,
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc="A valid reference URL documenting the product specifications is provided",
        parent=parent,
        critical=critical,
    )
    product_label = "the earbuds model"
    if manufacturer or model_name:
        product_label = f"the earbuds model '{(manufacturer or '').strip()} {(model_name or '').strip()}'.strip()"
    claim = f"This webpage documents specifications or an in-depth review for {(manufacturer or '')} {(model_name or '')}, i.e., it is clearly about the exact earbuds model."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=url or None,
        additional_instruction="Confirm the page is for the exact same model (brand + model name) and it lists specs or detailed product information.",
    )


async def _verify_common_specs(
    evaluator: Evaluator,
    parent,
    url: Optional[str],
    prefix: str,
    require_multipoint: bool = False,
    price_band: str = "budget",
):
    # Battery life with ANC >= 7h
    battery_leaf = evaluator.add_leaf(
        id=f"{prefix}_Battery_Life",
        desc="The earbuds provide at least 7 hours of playback with ANC enabled",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="According to this page, the per-charge battery life with ANC enabled is at least 7 hours.",
        node=battery_leaf,
        sources=url or None,
        additional_instruction="Look for 'ANC on' or explicit ANC-on battery spec; if only non-ANC battery is stated and ANC spec is absent, do not assume it meets 7h.",
    )

    # ANC feature present
    anc_leaf = evaluator.add_leaf(
        id=f"{prefix}_ANC_Feature",
        desc="The earbuds feature active noise cancellation",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="This product features active noise cancellation (ANC).",
        node=anc_leaf,
        sources=url or None,
        additional_instruction="Accept synonyms like 'active noise cancelling', 'hybrid ANC', etc. Passive isolation does not count.",
    )

    # Bluetooth >= 5.3
    bt_leaf = evaluator.add_leaf(
        id=f"{prefix}_Bluetooth_Version",
        desc="The earbuds support Bluetooth 5.3 or higher",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="This product supports Bluetooth version 5.3 or higher (e.g., 5.3 or 5.4).",
        node=bt_leaf,
        sources=url or None,
        additional_instruction="Accept explicit mentions like 'Bluetooth 5.3' or 'Bluetooth 5.4'. 'Bluetooth 5.2' is not sufficient.",
    )

    # Codec AAC
    codec_leaf = evaluator.add_leaf(
        id=f"{prefix}_Codec_Support",
        desc="The earbuds support at least AAC codec",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="This product supports the AAC audio codec.",
        node=codec_leaf,
        sources=url or None,
        additional_instruction="Look for codec lists; AAC can be alongside SBC/LDAC/aptX. AAC must be present.",
    )

    # Water resistance >= IPX4
    water_leaf = evaluator.add_leaf(
        id=f"{prefix}_Water_Resistance",
        desc="The earbuds have at least IPX4 water resistance rating",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="This product has at least an IPX4 water resistance rating.",
        node=water_leaf,
        sources=url or None,
        additional_instruction="Accept IPX4 or any IP rating with water digit >= 4, e.g., IPX5, IP54, IP55, IPX7. If only case is rated but earbuds are not, then it does not qualify.",
    )

    # Total battery (with case) >= 20h
    total_batt_leaf = evaluator.add_leaf(
        id=f"{prefix}_Total_Battery",
        desc="Combined battery life with case is at least 20 hours",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The combined battery life (earbuds plus charging case) is at least 20 hours.",
        node=total_batt_leaf,
        sources=url or None,
        additional_instruction="Check total playtime with charging case; do not sum unrelated specs.",
    )

    # Controls
    controls_leaf = evaluator.add_leaf(
        id=f"{prefix}_Controls",
        desc="The earbuds include touch or button controls for playback and calls",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="This product includes touch controls or physical button controls for playback and calls.",
        node=controls_leaf,
        sources=url or None,
        additional_instruction="Accept 'touch controls', 'capacitive touch', or 'physical buttons' controlling media/calls.",
    )

    # Release date >= 2024
    release_leaf = evaluator.add_leaf(
        id=f"{prefix}_Release_Date",
        desc="The earbuds are a 2024 or later model",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="This model was released, launched, or first announced in 2024 or later (year >= 2024).",
        node=release_leaf,
        sources=url or None,
        additional_instruction="Look for 'released', 'launched', 'announced' year; model generation year must be >= 2024.",
    )

    # Availability as of March 2026
    avail_leaf = evaluator.add_leaf(
        id=f"{prefix}_Availability",
        desc="The earbuds are currently available for purchase as of March 2026",
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim="As of March 2026, this product is currently available for purchase (in stock/orderable).",
        node=avail_leaf,
        sources=url or None,
        additional_instruction="Accept signals like 'Add to cart', 'Buy now', price shown for new units, or explicit 'available'. 'Discontinued' or 'out of stock' fails.",
    )

    # Multipoint (only for midrange/premium)
    if require_multipoint:
        mp_leaf = evaluator.add_leaf(
            id=f"{prefix}_Multipoint",
            desc="The earbuds support multipoint connectivity to two devices simultaneously",
            parent=parent,
            critical=True,
        )
        await evaluator.verify(
            claim="This product supports Bluetooth multipoint: simultaneous connection to two devices.",
            node=mp_leaf,
            sources=url or None,
            additional_instruction="Look for 'multipoint', 'dual device', 'two devices at once'.",
        )

    # Price compliance per band
    price_leaf = evaluator.add_leaf(
        id=f"{prefix}_Price_Compliance",
        desc=(
            "The earbuds are priced under $100"
            if price_band == "budget"
            else ("The earbuds are priced between $100 and $200" if price_band == "midrange" else "The earbuds are priced over $200")
        ),
        parent=parent,
        critical=True,
    )
    if price_band == "budget":
        price_claim = "The current street price or MSRP shown on this page is under $100 USD."
    elif price_band == "midrange":
        price_claim = "The current street price or MSRP shown on this page is between $100 and $200 USD (inclusive)."
    else:
        price_claim = "The current street price or MSRP shown on this page is over $200 USD."

    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=url or None,
        additional_instruction="Use the current/new product price on the page (manufacturer, retailer, or review). Sale price qualifies. Prefer USD; if only non-USD is present but clearly within the band, it's acceptable.",
    )


async def _verify_category(
    evaluator: Evaluator,
    root,
    cat_node_id: str,
    cat_desc: str,
    item: Optional[EarbudItem],
    price_band: str,
    require_multipoint: bool,
):
    """
    Build and verify the subtree for a price category.
    """
    cat_node = evaluator.add_parallel(
        id=cat_node_id,
        desc=cat_desc,
        parent=root,
        critical=False,  # non-critical: allows partial success if some categories fail
    )

    # Required info (gating)
    required_ok = (
        item is not None
        and item.manufacturer is not None and item.manufacturer.strip() != ""
        and item.model_name is not None and item.model_name.strip() != ""
        and item.reference_url is not None and item.reference_url.strip() != ""
    )
    evaluator.add_custom_node(
        result=required_ok,
        id=f"{price_band.capitalize()}_Required_Info",
        desc="Manufacturer, model name, and a reference URL are provided",
        parent=cat_node,
        critical=True,
    )

    # URL reference validity
    await _verify_url_reference(
        evaluator=evaluator,
        parent=cat_node,
        node_id=f"{price_band.capitalize()}_URL_Reference",
        manufacturer=(item.manufacturer if item else None),
        model_name=(item.model_name if item else None),
        url=(item.reference_url if item else None),
        critical=True,
    )

    # Common spec verifications
    await _verify_common_specs(
        evaluator=evaluator,
        parent=cat_node,
        url=(item.reference_url if item else None),
        prefix=price_band.capitalize(),
        require_multipoint=require_multipoint,
        price_band=price_band,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the wireless earbuds selection task.
    """
    # Initialize evaluator (root is non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find three wireless earbuds with active noise cancellation for different budgets, one from each price category (budget, mid-range, premium), each meeting all specified requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract earbuds list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_earbuds_list(),
        template_class=EarbudsExtraction,
        extraction_name="earbuds_extraction",
    )

    all_items = extracted.items if extracted and extracted.items else []
    selected = _select_items_by_category(all_items)

    budget_item = selected.get("budget")
    mid_item = selected.get("midrange")
    premium_item = selected.get("premium")

    # Record a brief summary of selected items
    evaluator.add_custom_info(
        info={
            "budget": {
                "manufacturer": budget_item.manufacturer if budget_item else None,
                "model": budget_item.model_name if budget_item else None,
                "price_text": budget_item.price_text if budget_item else None,
                "url": budget_item.reference_url if budget_item else None,
            },
            "midrange": {
                "manufacturer": mid_item.manufacturer if mid_item else None,
                "model": mid_item.model_name if mid_item else None,
                "price_text": mid_item.price_text if mid_item else None,
                "url": mid_item.reference_url if mid_item else None,
            },
            "premium": {
                "manufacturer": premium_item.manufacturer if premium_item else None,
                "model": premium_item.model_name if premium_item else None,
                "price_text": premium_item.price_text if premium_item else None,
                "url": premium_item.reference_url if premium_item else None,
            },
        },
        info_type="selection_summary",
    )

    # Add top-level critical check: Different manufacturers
    def _all_distinct_manufacturers(a: Optional[EarbudItem], b: Optional[EarbudItem], c: Optional[EarbudItem]) -> bool:
        try:
            mfrs = [x.manufacturer.strip() for x in [a, b, c] if x and x.manufacturer and x.manufacturer.strip()]
            return len(mfrs) == 3 and len(set(mfrs)) == 3
        except Exception:
            return False

    evaluator.add_custom_node(
        result=_all_distinct_manufacturers(budget_item, mid_item, premium_item),
        id="Different_Manufacturers",
        desc="All three pairs of earbuds must be from different manufacturers",
        parent=root,
        critical=True,  # critical: failing this fails the whole selection usefulness
    )

    # Build category subtrees (parallel)
    await _verify_category(
        evaluator=evaluator,
        root=root,
        cat_node_id="Budget_Earbuds_Under_100",
        cat_desc="Identify one pair of ANC wireless earbuds priced under $100 that meets all technical specifications",
        item=budget_item,
        price_band="budget",
        require_multipoint=False,
    )

    await _verify_category(
        evaluator=evaluator,
        root=root,
        cat_node_id="Midrange_Earbuds_100_to_200",
        cat_desc="Identify one pair of ANC wireless earbuds priced between $100 and $200 that meets all technical specifications",
        item=mid_item,
        price_band="midrange",
        require_multipoint=True,
    )

    await _verify_category(
        evaluator=evaluator,
        root=root,
        cat_node_id="Premium_Earbuds_Over_200",
        cat_desc="Identify one pair of ANC wireless earbuds priced over $200 that meets all technical specifications",
        item=premium_item,
        price_band="premium",
        require_multipoint=True,
    )

    # Return evaluation summary
    return evaluator.get_summary()