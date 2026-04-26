import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "tmobile_denver_smartphone_selection"
TASK_DESCRIPTION = """I'm relocating to Denver, Colorado, and need to purchase a new smartphone that will work optimally with T-Mobile's 5G network in the area. Due to my usage patterns and requirements, the phone must meet the following specifications:

1. Full compatibility with T-Mobile's 5G network, including support for the mid-band 5G frequencies that T-Mobile uses in Denver
2. A battery capacity of at least 5000 mAh to support my heavy daily usage
3. An IP68 water resistance rating with a submersion depth rating of at least 1.5 meters for 30 minutes, as I frequently engage in outdoor activities
4. A primary camera with a sensor size of at least 1/1.5 inches for high-quality low-light photography
5. At least 16GB of RAM to handle my multitasking and productivity applications

Please identify a specific smartphone model (including brand and model name) that meets all of these requirements and is currently available for purchase in the United States market. Include reference URLs for each specification.
"""


# -----------------------------------------------------------------------------
# Data models for structured extraction
# -----------------------------------------------------------------------------
class NetworkSpec(BaseModel):
    tmobile_compatibility_statement: Optional[str] = None
    supported_5g_bands: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class BatterySpec(BaseModel):
    capacity: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class WaterResistanceSpec(BaseModel):
    ip_rating: Optional[str] = None
    submersion_depth: Optional[str] = None
    submersion_time: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CameraSpec(BaseModel):
    primary_sensor_size: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class MemorySpec(BaseModel):
    ram: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class DeviceIdentity(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    availability_statement: Optional[str] = None
    availability_urls: List[str] = Field(default_factory=list)


class SmartphoneExtraction(BaseModel):
    identity: Optional[DeviceIdentity] = None
    network: Optional[NetworkSpec] = None
    battery: Optional[BatterySpec] = None
    water: Optional[WaterResistanceSpec] = None
    camera: Optional[CameraSpec] = None
    memory: Optional[MemorySpec] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_smartphone_info() -> str:
    return """
Extract a single specific smartphone proposed in the answer that is claimed to meet all requirements. If multiple models are mentioned, select the one the answer ultimately recommends; otherwise use the first fully specified model.

Return the following fields (use exactly the text as it appears in the answer; do not invent anything):

identity:
- brand: manufacturer brand name
- model: complete model name/number (e.g., "Galaxy S24 Ultra")
- availability_statement: the sentence/phrase asserting availability for purchase in the U.S., if present
- availability_urls: array of URLs that are cited as evidence of U.S. availability (manufacturer US store page, major US retailer, carrier page, etc.)

network:
- tmobile_compatibility_statement: the sentence/phrase asserting T-Mobile compatibility (or 5G support on T-Mobile), if present
- supported_5g_bands: array of 5G NR band codes exactly as written (e.g., "n41", "n71", "n77", "n78"); include only those explicitly listed in the answer
- source_urls: array of URLs cited as evidence for carrier compatibility and/or supported bands

battery:
- capacity: battery capacity exactly as written (e.g., "5000 mAh", "5100mAh")
- source_urls: array of URLs cited as evidence for the capacity

water:
- ip_rating: e.g., "IP68"
- submersion_depth: depth text exactly as written (e.g., "1.5 meters", "2m")
- submersion_time: time text exactly as written (e.g., "30 minutes")
- source_urls: array of URLs cited as evidence for rating and depth/time

camera:
- primary_sensor_size: primary camera sensor size exactly as written (e.g., "1/1.3\"", "1/1.5 inch")
- source_urls: array of URLs cited as evidence for this spec

memory:
- ram: RAM capacity exactly as written (e.g., "16GB", "24 GB")
- source_urls: array of URLs cited as evidence for RAM capacity

RULES:
- Extract only URLs explicitly present in the answer (plain, markdown, or otherwise). Do not fabricate URLs.
- If a field is not present, set it to null (or [] for URL arrays).
- Keep units and formatting as they appear in the answer text.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def device_name(identity: Optional[DeviceIdentity]) -> str:
    if identity and identity.brand and identity.model:
        return f"{identity.brand.strip()} {identity.model.strip()}".strip()
    if identity and identity.model:
        return identity.model.strip()
    return "the identified smartphone"


def nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def mark_fail_for_missing_sources(node, missing_reason: str = "Missing source URLs"):
    node.score = 0.0
    node.status = "failed"


# -----------------------------------------------------------------------------
# Verification builders for each rubric subtree
# -----------------------------------------------------------------------------
async def build_network_checks(evaluator: Evaluator, parent, data: SmartphoneExtraction):
    grp = evaluator.add_parallel(
        id="network_compatibility",
        desc="Smartphone must be compatible with and optimized for T-Mobile's 5G network in Denver, Colorado",
        parent=parent,
        critical=True,
    )
    net = data.network or NetworkSpec()
    dev = device_name(data.identity)
    urls = nonempty_urls(net.source_urls)

    # carrier_compatibility
    n1 = evaluator.add_leaf(
        id="carrier_compatibility",
        desc="Smartphone must support T-Mobile's network and be available for use with T-Mobile service",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(n1, "No network compatibility/band reference URLs provided")
    else:
        claim = f"The smartphone model {dev} is compatible with T-Mobile's 5G network in the United States."
        await evaluator.verify(
            claim=claim,
            node=n1,
            sources=urls,
            additional_instruction="Pass only if at least one provided source explicitly indicates T-Mobile compatibility (e.g., 'T‑Mobile supported', 'works with T‑Mobile', carrier band listings matching T‑Mobile)."
        )

    # 5g_band_support
    n2 = evaluator.add_leaf(
        id="5g_band_support",
        desc="Smartphone must support mid-band 5G frequencies used by T-Mobile (such as n41, n71, n77, or n78 bands)",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(n2, "No network/band URLs to validate mid-band support")
    else:
        claim = f"The smartphone {dev} supports at least one of these 5G NR bands used by T-Mobile: n41, n71, n77, or n78."
        await evaluator.verify(
            claim=claim,
            node=n2,
            sources=urls,
            additional_instruction="Look for an explicit bands list on the provided sources. Pass if any of n41/n71/n77/n78 is listed for this device."
        )

    # network_compatibility_reference
    n3 = evaluator.add_leaf(
        id="network_compatibility_reference",
        desc="Provide reference URL confirming the smartphone's T-Mobile 5G compatibility and supported bands",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(n3, "No reference URLs supplied for T‑Mobile 5G compatibility/supported bands")
    else:
        claim = f"The provided sources explicitly confirm {dev}'s T‑Mobile 5G compatibility and list supported 5G bands."
        await evaluator.verify(
            claim=claim,
            node=n3,
            sources=urls,
            additional_instruction="Pass only if at least one source both mentions T‑Mobile 5G compatibility or carrier support AND lists specific 5G bands (or links to an official spec table including bands)."
        )


async def build_battery_checks(evaluator: Evaluator, parent, data: SmartphoneExtraction):
    grp = evaluator.add_parallel(
        id="battery_specifications",
        desc="Smartphone must have a battery capacity of at least 5000 mAh",
        parent=parent,
        critical=True,
    )
    bat = data.battery or BatterySpec()
    dev = device_name(data.identity)
    urls = nonempty_urls(bat.source_urls)

    # battery_capacity_value
    b1 = evaluator.add_leaf(
        id="battery_capacity_value",
        desc="Battery capacity must be 5000 mAh or greater",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(b1, "No battery specification source URLs provided")
    else:
        cap_txt = bat.capacity or "unknown capacity"
        claim = f"The {dev} has a battery capacity of {cap_txt}, which is at least 5000 mAh."
        await evaluator.verify(
            claim=claim,
            node=b1,
            sources=urls,
            additional_instruction="Use the source specification table to verify the exact mAh value. Pass only if the value is >= 5000 mAh."
        )

    # battery_capacity_reference
    b2 = evaluator.add_leaf(
        id="battery_capacity_reference",
        desc="Provide reference URL confirming the exact battery capacity specification",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(b2, "Missing battery capacity reference URLs")
    else:
        cap_txt = bat.capacity or "the stated capacity"
        claim = f"The provided sources explicitly confirm the {dev} battery capacity ({cap_txt})."
        await evaluator.verify(
            claim=claim,
            node=b2,
            sources=urls,
            additional_instruction="Pass only if the exact capacity (mAh) is clearly stated on at least one provided source."
        )


async def build_water_checks(evaluator: Evaluator, parent, data: SmartphoneExtraction):
    grp = evaluator.add_parallel(
        id="water_resistance",
        desc="Smartphone must have IP68 water resistance rating with submersion depth of at least 1.5 meters",
        parent=parent,
        critical=True,
    )
    wat = data.water or WaterResistanceSpec()
    dev = device_name(data.identity)
    urls = nonempty_urls(wat.source_urls)

    # ip_rating
    w1 = evaluator.add_leaf(
        id="ip_rating",
        desc="Smartphone must have IP68 rating",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(w1, "No water resistance source URLs provided")
    else:
        claim = f"The {dev} has an IP68 ingress protection (water resistance) rating."
        await evaluator.verify(
            claim=claim,
            node=w1,
            sources=urls,
            additional_instruction="Look for the explicit 'IP68' designation on an official spec page or reputable source."
        )

    # submersion_depth
    w2 = evaluator.add_leaf(
        id="submersion_depth",
        desc="IP68 rating must specify submersion depth of at least 1.5 meters for 30 minutes",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(w2, "No source URLs for submersion depth/time details")
    else:
        claim = f"The {dev} is rated for submersion of at least 1.5 meters for 30 minutes per its IP68 specification."
        await evaluator.verify(
            claim=claim,
            node=w2,
            sources=urls,
            additional_instruction="Pass if a source states 1.5m for 30min or a strictly stronger rating (e.g., 2m/30min)."
        )

    # water_resistance_reference
    w3 = evaluator.add_leaf(
        id="water_resistance_reference",
        desc="Provide reference URL confirming the IP68 rating and specific depth specification",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(w3, "Missing water resistance reference URLs")
    else:
        claim = f"The provided sources explicitly confirm {dev}'s IP68 rating and the specified submersion depth/time."
        await evaluator.verify(
            claim=claim,
            node=w3,
            sources=urls,
            additional_instruction="Pass only if both the IP68 rating and the numeric depth/time (e.g., 1.5m for 30min) are stated."
        )


async def build_camera_checks(evaluator: Evaluator, parent, data: SmartphoneExtraction):
    grp = evaluator.add_parallel(
        id="camera_performance",
        desc="Smartphone must have a primary camera sensor size of at least 1/1.5 inches for good low-light performance",
        parent=parent,
        critical=True,
    )
    cam = data.camera or CameraSpec()
    dev = device_name(data.identity)
    urls = nonempty_urls(cam.source_urls)

    # sensor_size_specification
    c1 = evaluator.add_leaf(
        id="sensor_size_specification",
        desc="Primary camera sensor must be 1/1.5 inches or larger",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(c1, "No camera/sensor size source URLs provided")
    else:
        size_txt = cam.primary_sensor_size or "an unspecified size"
        claim = f"The {dev} primary camera sensor size ({size_txt}) is at least 1/1.5 inches (i.e., as large or larger)."
        await evaluator.verify(
            claim=claim,
            node=c1,
            sources=urls,
            additional_instruction="Consider formats like 1/1.5\", 1/1.4\", 1 inch, etc. Pass if the listed size is >= 1/1.5\"."
        )

    # camera_reference
    c2 = evaluator.add_leaf(
        id="camera_reference",
        desc="Provide reference URL confirming the primary camera sensor size specification",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(c2, "Missing camera sensor size reference URLs")
    else:
        size_txt = cam.primary_sensor_size or "the stated sensor size"
        claim = f"The provided sources explicitly confirm the {dev} primary camera sensor size ({size_txt})."
        await evaluator.verify(
            claim=claim,
            node=c2,
            sources=urls,
            additional_instruction="Pass only if the specific size is explicitly stated on at least one source."
        )


async def build_memory_checks(evaluator: Evaluator, parent, data: SmartphoneExtraction):
    grp = evaluator.add_parallel(
        id="memory_specifications",
        desc="Smartphone must have at least 16GB of RAM for multitasking and productivity",
        parent=parent,
        critical=True,
    )
    mem = data.memory or MemorySpec()
    dev = device_name(data.identity)
    urls = nonempty_urls(mem.source_urls)

    # ram_capacity
    m1 = evaluator.add_leaf(
        id="ram_capacity",
        desc="RAM must be 16GB or greater",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(m1, "No RAM capacity source URLs provided")
    else:
        ram_txt = mem.ram or "unknown RAM"
        claim = f"The {dev} has {ram_txt} RAM, which is at least 16GB."
        await evaluator.verify(
            claim=claim,
            node=m1,
            sources=urls,
            additional_instruction="Pass only if the physical RAM capacity (not just virtual/expanded RAM) is >= 16GB."
        )

    # ram_reference
    m2 = evaluator.add_leaf(
        id="ram_reference",
        desc="Provide reference URL confirming the RAM capacity specification",
        parent=grp,
        critical=True,
    )
    if not urls:
        mark_fail_for_missing_sources(m2, "Missing RAM reference URLs")
    else:
        ram_txt = mem.ram or "the stated RAM capacity"
        claim = f"The provided sources explicitly confirm the {dev} RAM capacity ({ram_txt})."
        await evaluator.verify(
            claim=claim,
            node=m2,
            sources=urls,
            additional_instruction="Pass only if the RAM capacity is clearly stated on at least one provided source."
        )


async def build_device_checks(evaluator: Evaluator, parent, data: SmartphoneExtraction):
    grp = evaluator.add_parallel(
        id="device_identification",
        desc="Provide the specific brand and model name of the identified smartphone",
        parent=parent,
        critical=True,
    )
    ident = data.identity or DeviceIdentity()
    dev = device_name(ident)

    # brand_and_model (simple verification against the answer text)
    d1 = evaluator.add_leaf(
        id="brand_and_model",
        desc="Clearly state the manufacturer brand and complete model name/number",
        parent=grp,
        critical=True,
    )
    claim = f"The answer explicitly identifies a single smartphone with brand and model as '{dev}'."
    await evaluator.verify(
        claim=claim,
        node=d1,
        additional_instruction="Pass only if the exact brand and model are clearly stated in the answer text (not inferred). If either brand or model is missing/ambiguous, mark Incorrect."
    )

    # availability_confirmation
    d2 = evaluator.add_leaf(
        id="availability_confirmation",
        desc="Confirm that the smartphone model is available for purchase in the United States market",
        parent=grp,
        critical=True,
    )
    avail_urls = nonempty_urls(ident.availability_urls)
    if not avail_urls:
        mark_fail_for_missing_sources(d2, "No U.S. availability URLs provided")
    else:
        claim = f"The smartphone model {dev} is available for purchase in the United States market."
        await evaluator.verify(
            claim=claim,
            node=d2,
            sources=avail_urls,
            additional_instruction="Pass if a provided source is a U.S. purchase channel (manufacturer US site, US retailer, US carrier) or clearly indicates US availability."
        )


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
    # Initialize evaluator with a critical root (parallel aggregation)
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
    # The root itself is non-critical by framework default; create a critical wrapper subtree as children
    # However, the rubric defines root as critical; to respect 'critical all through', we'll add critical children only.

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_smartphone_info(),
        template_class=SmartphoneExtraction,
        extraction_name="smartphone_selection",
    )

    # Build rubric-aligned verification subtrees (all critical)
    await build_network_checks(evaluator, root, extraction)
    await build_battery_checks(evaluator, root, extraction)
    await build_water_checks(evaluator, root, extraction)
    await build_camera_checks(evaluator, root, extraction)
    await build_memory_checks(evaluator, root, extraction)
    await build_device_checks(evaluator, root, extraction)

    return evaluator.get_summary()