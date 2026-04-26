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
TASK_ID = "satellite_sos_phones_2026"
TASK_DESCRIPTION = """You are planning extended international travel in 2026 and need a smartphone with reliable emergency communication capability in remote areas. Identify three different smartphone models that have built-in satellite Emergency SOS capability and meet all of the following requirements:

1. Geographic Availability: The satellite SOS feature must be available in both the United States and Canada, as well as in at least two European countries.

2. Carrier Independence: The satellite SOS functionality must work independently without requiring a subscription to any specific mobile carrier.

3. Service Terms: The device must include at least 2 years of free satellite SOS service included with purchase.

For each smartphone model, provide:
- The manufacturer name and complete model designation
- Confirmation that it has built-in satellite SOS hardware
- The specific European countries (at least two) where satellite SOS is available for this device
- The duration of the included free service period
- A URL reference documenting the device's satellite SOS capability and geographic availability
- A URL reference documenting the carrier requirements (or carrier independence)
- A URL reference documenting the free service period

Additionally, if available, include information about the expected monthly cost after the free period ends, along with a supporting URL reference.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DeviceItem(BaseModel):
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    # Availability and capability statements (strings preferred to maximize compatibility)
    built_in_sos_statement: Optional[str] = None
    us_availability_statement: Optional[str] = None
    canada_availability_statement: Optional[str] = None
    europe_countries: List[str] = Field(default_factory=list)
    carrier_independence_statement: Optional[str] = None
    free_service_duration: Optional[str] = None
    # URLs
    satellite_geo_url: Optional[str] = None
    carrier_requirements_url: Optional[str] = None
    free_service_url: Optional[str] = None
    # Optional post-free pricing
    post_free_cost: Optional[str] = None
    post_free_cost_url: Optional[str] = None
    post_free_cost_note: Optional[str] = None  # e.g., "pricing not available" if stated


class DevicesExtraction(BaseModel):
    devices: List[DeviceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_devices() -> str:
    return """
Extract all smartphone models mentioned in the answer that claim to support built-in satellite Emergency SOS, returning a JSON object with an array field `devices`. For each device, extract the following fields exactly as stated in the answer (use null if missing, and empty list if no countries are named):

- manufacturer: Manufacturer name (e.g., "Apple", "Google")
- model: Complete model designation (e.g., "iPhone 15 Pro Max", "Pixel 9 Pro")
- built_in_sos_statement: The text confirming the device has built-in satellite Emergency SOS (no external accessory required)
- us_availability_statement: Text indicating availability in the United States
- canada_availability_statement: Text indicating availability in Canada
- europe_countries: A list of named European countries where satellite SOS is available for this device (include all named; empty list if none named)
- carrier_independence_statement: Text indicating the satellite SOS feature works without requiring a subscription to any specific mobile carrier
- free_service_duration: The stated free service period included with purchase (e.g., "2 years", "24 months")
- satellite_geo_url: A URL that documents the device's satellite SOS capability and geographic availability (can be official spec page, support page, or press release)
- carrier_requirements_url: A URL that documents the carrier requirements (or independence) for satellite SOS
- free_service_url: A URL that documents the free service period (duration included)
- post_free_cost: If available, the expected monthly cost after the free period ends (text)
- post_free_cost_url: If available, a URL that documents the expected monthly cost after the free period
- post_free_cost_note: If the answer explicitly states that pricing after the free period is not available or not announced, include the text here

Return all devices found (do not limit the count). Do not invent URLs or values not explicitly present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm_str(x: Optional[str]) -> str:
    return (x or "").strip()


def _normalized_model_key(manufacturer: Optional[str], model: Optional[str]) -> str:
    m = _norm_str(manufacturer).lower()
    md = _norm_str(model).lower()
    return f"{m}::{md}" if m or md else ""


def _is_eligible_device_family(manufacturer: Optional[str], model: Optional[str]) -> bool:
    """
    Allowed device families:
    - Apple iPhone 14/15/16 series (any variants)
    - Google Pixel 9 series (excluding Pixel 9a)
    Not allowed:
    - Samsung Galaxy S25 series
    """
    m = _norm_str(manufacturer).lower()
    md = _norm_str(model).lower()

    if not md:
        return False

    # Exclude Samsung Galaxy S25
    if "samsung" in m or "galaxy" in md:
        if "s25" in md or "galaxy s25" in md:
            return False

    # Apple iPhone 14/15/16
    if "apple" in m or "iphone" in md:
        if ("iphone 14" in md) or ("iphone 15" in md) or ("iphone 16" in md):
            return True

    # Google Pixel 9 series excluding 9a
    if "google" in m or "pixel" in md:
        if "pixel 9" in md and "9a" not in md:
            return True

    return False


def _format_device_name(d: DeviceItem) -> str:
    m = _norm_str(d.manufacturer)
    md = _norm_str(d.model)
    combo = " ".join([p for p in [m, md] if p])
    return combo if combo else "the device"


def _countries_text(countries: List[str], min_n: int = 2) -> str:
    c = [c.strip() for c in countries if c and c.strip()]
    if len(c) >= min_n:
        return ", ".join(c)
    # If fewer than required, still return what we have (verification likely fails)
    return ", ".join(c) if c else ""


# --------------------------------------------------------------------------- #
# Verification subroutine per smartphone                                      #
# --------------------------------------------------------------------------- #
async def verify_smartphone(
    evaluator: Evaluator,
    parent_node,
    device: DeviceItem,
    index_1based: int,
) -> None:
    """
    Build verification subtree for a single smartphone.
    """
    dev_label = f"smartphone_{index_1based}"
    device_node = evaluator.add_parallel(
        id=dev_label,
        desc=f"Smartphone model #{index_1based}",
        parent=parent_node,
        critical=False,
    )

    # 1) Manufacturer and Model presence (critical)
    has_manu_model = bool(_norm_str(device.manufacturer)) and bool(_norm_str(device.model))
    evaluator.add_custom_node(
        result=has_manu_model,
        id=f"{dev_label}_manufacturer_and_model",
        desc="Provide manufacturer name and complete model designation",
        parent=device_node,
        critical=True,
    )

    # 2) Eligibility against constraints (critical)
    eligible = _is_eligible_device_family(device.manufacturer, device.model)
    evaluator.add_custom_node(
        result=eligible,
        id=f"{dev_label}_eligible_device_family_per_constraints",
        desc="Model is within allowed device families (Apple iPhone 14/15/16 or Google Pixel 9 excluding 9a) and is not Samsung Galaxy S25 series",
        parent=device_node,
        critical=True,
    )

    # 3) URLs existence (critical)
    evaluator.add_custom_node(
        result=bool(_norm_str(device.satellite_geo_url)),
        id=f"{dev_label}_url_satellite_sos_and_geo",
        desc="Provide a URL reference documenting the device's satellite SOS capability and geographic availability",
        parent=device_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(_norm_str(device.carrier_requirements_url)),
        id=f"{dev_label}_url_carrier_requirements",
        desc="Provide a URL reference documenting the carrier requirements (or carrier independence)",
        parent=device_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(_norm_str(device.free_service_url)),
        id=f"{dev_label}_url_free_service_period",
        desc="Provide a URL reference documenting the free service period",
        parent=device_node,
        critical=True,
    )

    # 4) Core requirement verifications (all critical)
    # Build leaves
    built_in_leaf = evaluator.add_leaf(
        id=f"{dev_label}_built_in_satellite_sos",
        desc="Confirm the device has built-in satellite Emergency SOS hardware (no external accessory required)",
        parent=device_node,
        critical=True,
    )
    us_ca_leaf = evaluator.add_leaf(
        id=f"{dev_label}_geo_availability_us_canada",
        desc="Confirm satellite SOS availability in both the United States and Canada",
        parent=device_node,
        critical=True,
    )
    eu_leaf = evaluator.add_leaf(
        id=f"{dev_label}_europe_availability_countries",
        desc="Name at least two specific European countries where satellite SOS is available for this device",
        parent=device_node,
        critical=True,
    )
    carrier_indep_leaf = evaluator.add_leaf(
        id=f"{dev_label}_carrier_independence",
        desc="Confirm satellite SOS works without requiring a subscription to any specific mobile carrier",
        parent=device_node,
        critical=True,
    )
    free_service_leaf = evaluator.add_leaf(
        id=f"{dev_label}_free_service_period",
        desc="State included free satellite SOS service duration and confirm it is at least 2 years",
        parent=device_node,
        critical=True,
    )

    # Claims and sources
    device_name = _format_device_name(device)
    sat_geo_sources = device.satellite_geo_url if device.satellite_geo_url else None
    carrier_sources = device.carrier_requirements_url if device.carrier_requirements_url else None
    free_service_sources = device.free_service_url if device.free_service_url else None

    built_in_claim = (
        f"The {device_name} has built-in satellite Emergency SOS capability without requiring any external accessory."
    )
    us_ca_claim = (
        f"Satellite SOS for {device_name} is available in both the United States and Canada."
    )
    eu_countries_text = _countries_text(device.europe_countries, min_n=2)
    eu_claim = (
        f"Satellite SOS for {device_name} is available in at least the following European countries: {eu_countries_text}."
    )
    carrier_indep_claim = (
        f"The satellite SOS functionality on {device_name} works without requiring a subscription to any specific mobile carrier."
    )
    free_service_text = _norm_str(device.free_service_duration)
    free_service_claim = (
        f"The {device_name} includes at least two years of free satellite SOS service with purchase"
        + (f" (stated free period: {free_service_text})." if free_service_text else ".")
    )

    # Additional instructions for verifier
    add_ins_common = (
        "Accept minor naming variations (e.g., 'Emergency SOS via satellite'). "
        "Focus strictly on the provided webpage(s) for support. "
        "If numbers are given (e.g., 24 months), treat that as equivalent to 2 years."
    )
    add_ins_built_in = (
        add_ins_common
        + " The page should indicate the device itself supports satellite SOS without external accessories."
    )
    add_ins_us_ca = (
        add_ins_common
        + " Look for explicit availability in the United States and in Canada for this device."
    )
    add_ins_eu = (
        add_ins_common
        + " Verify that at least two specifically named European countries are listed as supported for this device."
    )
    add_ins_carrier = (
        add_ins_common
        + " Confirm that the feature works independently of a mobile carrier subscription (no specific carrier required)."
    )
    add_ins_free = (
        add_ins_common
        + " Confirm that the included free service period is at least 2 years for this device."
    )

    # Use batch verification for core five leaves
    await evaluator.batch_verify(
        [
            (built_in_claim, sat_geo_sources, built_in_leaf, add_ins_built_in),
            (us_ca_claim, sat_geo_sources, us_ca_leaf, add_ins_us_ca),
            (eu_claim, sat_geo_sources, eu_leaf, add_ins_eu),
            (carrier_indep_claim, carrier_sources, carrier_indep_leaf, add_ins_carrier),
            (free_service_claim, free_service_sources, free_service_leaf, add_ins_free),
        ]
    )

    # 5) Optional: Post-free monthly cost
    # If cost and URL provided, verify. Otherwise, pass this optional check by default to avoid penalizing.
    if _norm_str(device.post_free_cost) and _norm_str(device.post_free_cost_url):
        post_cost_leaf = evaluator.add_leaf(
            id=f"{dev_label}_post_free_cost_optional",
            desc="If available, include expected monthly cost after the free period ends with a supporting URL; otherwise indicate pricing is not available",
            parent=device_node,
            critical=False,
        )
        cost_claim = (
            f"After the free period ends, the expected monthly satellite SOS cost for {device_name} is '{device.post_free_cost}'."
        )
        await evaluator.verify(
            claim=cost_claim,
            node=post_cost_leaf,
            sources=device.post_free_cost_url,
            additional_instruction=(
                "Verify the stated monthly price on the provided page; minor currency/formatting variations are acceptable."
            ),
        )
    else:
        # Consider the optional requirement satisfied even if not provided
        evaluator.add_custom_node(
            result=True,
            id=f"{dev_label}_post_free_cost_optional",
            desc="If available, include expected monthly cost after the free period ends with a supporting URL; otherwise indicate pricing is not available",
            parent=device_node,
            critical=False,
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
    Evaluate an answer for the 2026 satellite SOS smartphones task.
    """
    # Initialize evaluator (root node is always non-critical internally; we use PARALLEL per rubric)
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

    # Extract all devices mentioned
    extraction = await evaluator.extract(
        prompt=prompt_extract_devices(),
        template_class=DevicesExtraction,
        extraction_name="devices_extraction",
    )

    devices_all = extraction.devices or []
    # Determine distinctness across all mentioned devices (not just first three)
    keys_all = [_normalized_model_key(d.manufacturer, d.model) for d in devices_all if _normalized_model_key(d.manufacturer, d.model)]
    unique_count = len(set(keys_all))

    # Critical check: we need at least three distinct smartphones (we accept 3 or more, then we will evaluate the first 3)
    evaluator.add_custom_node(
        result=(unique_count >= 3),
        id="three_distinct_models_provided",
        desc="Response provides at least three smartphone models and they are all different complete model designations",
        parent=root,
        critical=True,
    )

    # Keep only the first three devices for per-device checks (as per framework guidance)
    devices_3 = []
    seen = set()
    for d in devices_all:
        key = _normalized_model_key(d.manufacturer, d.model)
        if not key or key in seen:
            continue
        devices_3.append(d)
        seen.add(key)
        if len(devices_3) == 3:
            break

    # If fewer than 3 available after dedup, pad with empty DeviceItem objects to keep structure
    while len(devices_3) < 3:
        devices_3.append(DeviceItem())

    # Build subtrees for 3 smartphones (parallel under root)
    await asyncio.gather(
        verify_smartphone(evaluator, root, devices_3[0], 1),
        verify_smartphone(evaluator, root, devices_3[1], 2),
        verify_smartphone(evaluator, root, devices_3[2], 3),
    )

    return evaluator.get_summary()