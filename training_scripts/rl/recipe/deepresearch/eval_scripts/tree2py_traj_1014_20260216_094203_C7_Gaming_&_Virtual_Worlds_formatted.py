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
TASK_ID = "wireless_mouse_specs_2022_2023"
TASK_DESCRIPTION = (
    "Identify a wireless gaming mouse that meets all of the following specifications: "
    "Uses 2.4GHz wireless connectivity (not Bluetooth-only); Weighs 65 grams or less; "
    "Supports a maximum DPI of at least 26,000; Supports a wireless polling rate of at least 4000Hz; "
    "Provides at least 90 hours of continuous battery life; Was officially released between "
    "January 1, 2022 and December 31, 2023; Has an original MSRP between $140 USD and $170 USD; "
    "Has at least 5 programmable buttons; Uses an optical sensor (not laser); "
    "Is designed specifically for right-handed use with an ergonomic shape; Supports USB-C charging; "
    "Features customizable RGB lighting; Uses switches rated for at least 70 million clicks; "
    "Is manufactured by a recognized gaming peripheral brand (Razer, Logitech, SteelSeries, Corsair, or similar). "
    "Provide the model name and manufacturer of the mouse, along with reference URLs that verify each specification."
)

RECOGNIZED_BRANDS = [
    "Razer", "Logitech", "Logitech G", "SteelSeries", "Corsair", "HyperX", "ASUS", "ASUS ROG",
    "Cooler Master", "Glorious", "Glorious PC Gaming Race", "ROCCAT", "Zowie", "BenQ Zowie",
    "Endgame Gear", "Xtrfy", "Lamzu", "Pulsar", "Finalmouse", "Ninjutso", "Fnatic", "Mountain"
]
RECOGNIZED_BRANDS_DISPLAY = ", ".join(RECOGNIZED_BRANDS)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SpecField(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MouseExtraction(BaseModel):
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None

    connectivity_2_4ghz: Optional[SpecField] = None
    weight_grams: Optional[SpecField] = None
    max_dpi: Optional[SpecField] = None
    wireless_polling_rate_hz: Optional[SpecField] = None
    battery_life_hours: Optional[SpecField] = None
    release_date: Optional[SpecField] = None
    msrp_usd: Optional[SpecField] = None
    programmable_buttons: Optional[SpecField] = None
    sensor_type: Optional[SpecField] = None
    hand_orientation: Optional[SpecField] = None
    charging_interface: Optional[SpecField] = None
    rgb_lighting: Optional[SpecField] = None
    switch_durability_clicks: Optional[SpecField] = None

    # Optional dedicated sources for brand recognition if provided
    brand_recognition: Optional[SpecField] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_mouse_specs() -> str:
    return """
Extract the proposed wireless gaming mouse details exactly as stated in the answer. You must fill the following JSON fields:

- model_name: The exact model name of the mouse.
- manufacturer: The brand/manufacturer.

For each specification below, extract:
- value: The exact value/phrase the answer claims (as written).
- sources: A list of URLs cited in the answer that specifically verify this specification. Only include URLs that appear in the answer text. Do not invent any URLs.

Specifications (fill each as an object with {value, sources}):
- connectivity_2_4ghz: Whether it uses 2.4GHz wireless (not Bluetooth-only).
- weight_grams: The stated weight in grams (as text, e.g., "59g" or "59 grams").
- max_dpi: The maximum DPI (as text, e.g., "30,000 DPI" or "26K DPI").
- wireless_polling_rate_hz: The stated wireless polling rate capability (e.g., "4000 Hz" or "4K wireless polling").
- battery_life_hours: The stated continuous battery life (e.g., "90 hours", "100h").
- release_date: The official release/launch/announcement date/timeframe (e.g., "May 2022", "2023-08-21").
- msrp_usd: The original MSRP in USD (e.g., "$149.99").
- programmable_buttons: The number of programmable buttons (e.g., "6 programmable buttons").
- sensor_type: The sensor type (e.g., "optical").
- hand_orientation: The shape/orientation (e.g., "right-handed ergonomic").
- charging_interface: The charging interface (e.g., "USB-C").
- rgb_lighting: Whether it features customizable RGB lighting (e.g., "customizable RGB").
- switch_durability_clicks: The switch durability rating (e.g., "70 million clicks").

Optionally (if the answer provides it), also extract:
- brand_recognition: sources: URLs used to substantiate the brand/manufacturer identity, if any (often the product page or official brand page). Set value to the brand string if present.

IMPORTANT RULES:
1) Only extract information explicitly present in the answer text. If an item is missing, set its value to null and sources to [].
2) For sources arrays, include all URLs the answer associates with verifying that specific item. If the answer provides a general list of references for the whole product, you may include the same URLs across multiple specs if the answer implies they verify those specs.
3) Do not transform units into numbers—keep them as strings exactly as written (e.g., "59g", "30K DPI").
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_product_ref(extracted: MouseExtraction) -> str:
    parts = []
    if extracted.manufacturer and extracted.manufacturer.strip():
        parts.append(extracted.manufacturer.strip())
    if extracted.model_name and extracted.model_name.strip():
        parts.append(extracted.model_name.strip())
    return " ".join(parts) if parts else "the mouse"


def _get_sources(spec: Optional[SpecField]) -> List[str]:
    if spec and spec.sources:
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for u in spec.sources:
            if isinstance(u, str) and u.strip() and u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq
    return []


def _union_all_sources(extracted: MouseExtraction) -> List[str]:
    all_lists: List[List[str]] = []
    for field_name in [
        "connectivity_2_4ghz", "weight_grams", "max_dpi", "wireless_polling_rate_hz", "battery_life_hours",
        "release_date", "msrp_usd", "programmable_buttons", "sensor_type", "hand_orientation",
        "charging_interface", "rgb_lighting", "switch_durability_clicks"
    ]:
        spec = getattr(extracted, field_name, None)
        all_lists.append(_get_sources(spec))
    # Also include brand_recognition sources if provided
    all_lists.append(_get_sources(extracted.brand_recognition))

    # Flatten & deduplicate
    seen = set()
    merged: List[str] = []
    for lst in all_lists:
        for url in lst:
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_mouse_specs(
    evaluator: Evaluator,
    parent_node,
    extracted: MouseExtraction,
) -> None:
    # Parent critical node aggregating all spec checks
    main_node = evaluator.add_parallel(
        id="Mouse_Identification",
        desc="Identify a wireless gaming mouse that satisfies all specified technical requirements",
        parent=parent_node,
        critical=True
    )

    product_ref = _safe_product_ref(extracted)

    # Prepare batch verifications
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    def add_spec_leaf_or_fail(
        node_id: str,
        desc: str,
        claim: str,
        sources: List[str],
        add_ins: str
    ):
        """
        Create a critical leaf for this spec. If no sources are provided, mark as failed immediately;
        otherwise, queue it for batch verification by URLs.
        """
        if sources:
            leaf = evaluator.add_leaf(
                id=node_id,
                desc=desc,
                parent=main_node,
                critical=True
            )
            claims_and_sources.append((claim, sources, leaf, add_ins))
        else:
            evaluator.add_custom_node(
                result=False,
                id=node_id,
                desc=f"{desc} (failed: no reference URL(s) provided in the answer for this spec)",
                parent=main_node,
                critical=True
            )

    # 1) Connectivity: 2.4GHz (not Bluetooth-only)
    conn_sources = _get_sources(extracted.connectivity_2_4ghz)
    conn_claim = (
        f"{product_ref} supports 2.4GHz wireless connectivity (e.g., via a USB receiver/dongle) and is not Bluetooth-only."
    )
    conn_ins = (
        "Accept evidence that mentions '2.4GHz', '2.4G', or proprietary 2.4GHz tech (e.g., LIGHTSPEED, HyperSpeed, "
        "SpeedNova, Quantum 2.0, etc.). It's okay if Bluetooth is also supported; the key is that 2.4GHz is supported."
    )
    add_spec_leaf_or_fail(
        "Connectivity_Type",
        "The mouse must use wireless connectivity via 2.4GHz technology (not Bluetooth-only)",
        conn_claim, conn_sources, conn_ins
    )

    # 2) Weight <= 65g
    weight_sources = _get_sources(extracted.weight_grams)
    weight_claim = (
        f"The listed weight for {product_ref} (wireless, without cable and excluding optional accessories) is 65 grams or less."
    )
    weight_ins = (
        "Verify the stated weight from the product page or official spec sheet. Ignore cable weight and optional accessories. "
        "Minor rounding differences are acceptable if the official spec is <= 65 g."
    )
    add_spec_leaf_or_fail(
        "Weight_Specification",
        "The mouse must weigh 65 grams or less",
        weight_claim, weight_sources, weight_ins
    )

    # 3) Max DPI >= 26,000
    dpi_sources = _get_sources(extracted.max_dpi)
    dpi_claim = f"The maximum DPI (sensitivity) for {product_ref} is at least 26,000."
    dpi_ins = (
        "Check the maximum DPI on the product page or official specification. Accept minor formatting like '26K DPI' or '26000 DPI'. "
        "Values like 30K or 35K also satisfy this."
    )
    add_spec_leaf_or_fail(
        "DPI_Capability",
        "The mouse must support a maximum DPI of at least 26,000",
        dpi_claim, dpi_sources, dpi_ins
    )

    # 4) Wireless polling rate >= 4000 Hz
    pr_sources = _get_sources(extracted.wireless_polling_rate_hz)
    pr_claim = (
        f"{product_ref} supports a wireless polling rate of at least 4000 Hz (e.g., with a compatible 4K/8K wireless dongle if specified)."
    )
    pr_ins = (
        "If the manufacturer states 4000 Hz (or higher) wireless polling when paired with an optional 4K/8K dongle or receiver, "
        "that counts as support. The page should indicate >= 4000 Hz for wireless mode."
    )
    add_spec_leaf_or_fail(
        "Polling_Rate",
        "The mouse must support a wireless polling rate of at least 4000Hz",
        pr_claim, pr_sources, pr_ins
    )

    # 5) Battery life >= 90 hours
    batt_sources = _get_sources(extracted.battery_life_hours)
    batt_claim = f"{product_ref} provides at least 90 hours of continuous battery life under normal test conditions (often RGB off)."
    batt_ins = (
        "Use the official battery life specification. If multiple modes are listed, accept a mode that mentions >= 90 hours. "
        "Typically RGB-off testing is acceptable."
    )
    add_spec_leaf_or_fail(
        "Battery_Life",
        "The mouse must provide at least 90 hours of continuous battery life",
        batt_claim, batt_sources, batt_ins
    )

    # 6) Release date in 2022-01-01 to 2023-12-31 inclusive
    rel_sources = _get_sources(extracted.release_date)
    rel_claim = (
        f"The official launch/release/announcement date for {product_ref} is between January 1, 2022 and December 31, 2023 (inclusive)."
    )
    rel_ins = (
        "Accept official release/launch/announcement dates from manufacturer press releases or product pages, "
        "or credible review sites that clearly state launch date. The date must fall within 2022-01-01 to 2023-12-31 inclusive."
    )
    add_spec_leaf_or_fail(
        "Release_Date",
        "The mouse must have been officially released between January 1, 2022 and December 31, 2023",
        rel_claim, rel_sources, rel_ins
    )

    # 7) MSRP between $140 and $170 USD
    msrp_sources = _get_sources(extracted.msrp_usd)
    msrp_claim = f"The original MSRP for {product_ref} was between $140 and $170 USD (inclusive)."
    msrp_ins = (
        "Verify the original US MSRP. Accept standard pricing like $149.99 or $169.99. "
        "Ignore promotional discounts or regional prices."
    )
    add_spec_leaf_or_fail(
        "Price_Range",
        "The mouse's original MSRP must be between $140 USD and $170 USD",
        msrp_claim, msrp_sources, msrp_ins
    )

    # 8) At least 5 programmable buttons
    btn_sources = _get_sources(extracted.programmable_buttons)
    btn_claim = f"{product_ref} has at least 5 programmable buttons."
    btn_ins = (
        "Count programmable buttons as indicated by the product page/specifications/software support. "
        "Side buttons and top buttons that can be reassigned count."
    )
    add_spec_leaf_or_fail(
        "Button_Count",
        "The mouse must have at least 5 programmable buttons",
        btn_claim, btn_sources, btn_ins
    )

    # 9) Optical sensor (not laser)
    sensor_sources = _get_sources(extracted.sensor_type)
    sensor_claim = f"{product_ref} uses an optical sensor (not a laser sensor)."
    sensor_ins = (
        "Look for the sensor type on the specification page. Names like Focus Pro 30K, HERO 25K/32K, PixArt 3395/3950, etc., are optical."
    )
    add_spec_leaf_or_fail(
        "Sensor_Type",
        "The mouse must use an optical sensor (not laser)",
        sensor_claim, sensor_sources, sensor_ins
    )

    # 10) Right-handed ergonomic shape (not ambidextrous)
    hand_sources = _get_sources(extracted.hand_orientation)
    hand_claim = f"{product_ref} is designed specifically for right-handed ergonomic use (not ambidextrous)."
    hand_ins = (
        "The page should indicate an ergonomic right-handed shape. "
        "If the product is described as ambidextrous/symmetrical, this does NOT satisfy the requirement."
    )
    add_spec_leaf_or_fail(
        "Hand_Orientation",
        "The mouse must be designed specifically for right-handed use with an ergonomic shape",
        hand_claim, hand_sources, hand_ins
    )

    # 11) USB-C charging
    usb_sources = _get_sources(extracted.charging_interface)
    usb_claim = f"{product_ref} supports USB-C (USB Type-C) charging."
    usb_ins = (
        "Verify that the charging interface is USB-C/Type-C. A removable USB-C cable for charging counts."
    )
    add_spec_leaf_or_fail(
        "Charging_Interface",
        "The mouse must support USB-C charging",
        usb_claim, usb_sources, usb_ins
    )

    # 12) Customizable RGB lighting
    rgb_sources = _get_sources(extracted.rgb_lighting)
    rgb_claim = f"{product_ref} features customizable RGB lighting."
    rgb_ins = (
        "The product page should mention RGB lighting that can be customized (via software or onboard controls). "
        "If there is no RGB, this fails."
    )
    add_spec_leaf_or_fail(
        "RGB_Lighting",
        "The mouse must feature customizable RGB lighting",
        rgb_claim, rgb_sources, rgb_ins
    )

    # 13) Switches rated >= 70M clicks
    sw_sources = _get_sources(extracted.switch_durability_clicks)
    sw_claim = f"{product_ref} uses switches rated for at least 70 million clicks."
    sw_ins = (
        "Look for the switch durability rating (e.g., 70M, 80M). If multiple buttons have different ratings, "
        "at least the primary switches should be >= 70M."
    )
    add_spec_leaf_or_fail(
        "Switch_Durability",
        "The mouse must use switches rated for at least 70 million clicks",
        sw_claim, sw_sources, sw_ins
    )

    # 14) Recognized brand
    # Prefer dedicated brand sources if provided; otherwise fall back to any provided product sources.
    brand_sources = _get_sources(extracted.brand_recognition)
    if not brand_sources:
        brand_sources = _union_all_sources(extracted)

    brand_claim = (
        f"The manufacturer/brand shown on the provided webpage(s) for {product_ref} "
        f"is one of the recognized gaming peripheral brands: {RECOGNIZED_BRANDS_DISPLAY}."
    )
    brand_ins = (
        f"From the provided webpage(s), identify the brand/manufacturer string and check whether it is included in this set: "
        f"[{RECOGNIZED_BRANDS_DISPLAY}]. Treat membership in this explicit set as the criterion."
    )
    add_spec_leaf_or_fail(
        "Brand_Recognition",
        "The mouse must be manufactured by a recognized gaming peripheral brand (Razer, Logitech, SteelSeries, Corsair, or similar established gaming brands)",
        brand_claim, brand_sources, brand_ins
    )

    # Run all available verifications in parallel to avoid cross‑dependency skips
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the wireless gaming mouse specification task.
    """
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_mouse_specs(),
        template_class=MouseExtraction,
        extraction_name="mouse_specs"
    )

    # Record requirement thresholds for transparency
    evaluator.add_ground_truth({
        "requirements": {
            "connectivity": "2.4GHz wireless (not Bluetooth-only)",
            "weight_max_g": 65,
            "dpi_min": 26000,
            "wireless_polling_rate_min_hz": 4000,
            "battery_life_min_hours": 90,
            "release_date_window": "2022-01-01 to 2023-12-31 (inclusive)",
            "msrp_usd_range": "$140 - $170",
            "programmable_buttons_min": 5,
            "sensor_type": "optical",
            "hand_orientation": "right-handed ergonomic",
            "charging_interface": "USB-C",
            "rgb": "customizable RGB",
            "switch_durability_min_clicks": 70_000_000,
            "recognized_brands": RECOGNIZED_BRANDS
        }
    }, gt_type="task_requirements")

    # Build and run verification tree
    await verify_mouse_specs(evaluator, root, extracted)

    return evaluator.get_summary()