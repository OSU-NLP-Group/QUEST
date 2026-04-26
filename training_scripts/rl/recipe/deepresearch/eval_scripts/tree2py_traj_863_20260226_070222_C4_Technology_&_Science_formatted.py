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
TASK_ID = "gaming_handheld_2025_2026"
TASK_DESCRIPTION = """
Identify one gaming handheld device that meets all of the following specifications and is released or officially announced for release in 2025 or 2026:

- Display size of at least 7 inches (diagonal measurement)
- Native display resolution of at least 1920x1080 (Full HD) or 1920x1200 (WUXGA)
- Display refresh rate of at least 120Hz
- At least 16GB of RAM
- Battery capacity of at least 50Wh (watt-hours) or 5000mAh (milliamp-hours)
- Runs Windows 11 or SteamOS as its primary operating system

For the device you identify, provide:
1. The specific device name or model
2. The exact specifications for display size, display resolution, refresh rate, RAM, battery capacity, and operating system
3. A URL to an official product page or reliable tech news source that confirms these specifications
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DeviceSpec(BaseModel):
    device_name: Optional[str] = None
    display_size: Optional[str] = None               # e.g., "7.0-inch", "7.4-inch"
    display_resolution: Optional[str] = None         # e.g., "1920x1080", "2560 x 1600"
    refresh_rate: Optional[str] = None               # e.g., "120Hz", "144 Hz"
    ram: Optional[str] = None                        # e.g., "16GB", "32 GB LPDDR5"
    battery_capacity: Optional[str] = None           # e.g., "50Wh", "8000mAh"
    operating_system: Optional[str] = None           # e.g., "Windows 11", "SteamOS 3"
    release_timeline: Optional[str] = None           # e.g., "announced January 2026", "release in 2025"
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_device() -> str:
    return """
    Extract exactly one gaming handheld device (the main one identified in the answer). If multiple devices are mentioned, pick the first one that appears to be the recommended/identified device.
    For this single device, extract the following fields exactly as stated in the answer (do NOT infer or make up values):
    - device_name: The specific device/model name.
    - display_size: The diagonal display size as presented (e.g., "7.0-inch", "7.4-inch", "8 inch").
    - display_resolution: The native display resolution as presented (e.g., "1920x1080", "1920 x 1200", "2560×1600").
    - refresh_rate: The panel refresh rate as presented (e.g., "120Hz", "144 Hz", "up to 144Hz").
    - ram: The RAM capacity as presented (e.g., "16GB", "32GB LPDDR5").
    - battery_capacity: The battery capacity as presented (e.g., "50Wh", "80 Wh", "8000 mAh").
    - operating_system: The primary operating system as presented (e.g., "Windows 11", "SteamOS 3").
    - release_timeline: Any release or official announcement timing info as stated (e.g., "announced January 2026", "releases Q4 2025").
    - source_urls: All URLs explicitly included in the answer that are intended to support the device and its specifications (official product page or reliable tech news). These must be actual URLs (http/https). Return all you find in the answer; if none, return an empty array.
    
    Important rules:
    - Extract values verbatim from the answer text only; do not infer missing values.
    - For URLs: include only valid full URLs (HTTP/HTTPS). If no URLs are provided in the answer, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
async def verify_device_specs(evaluator: Evaluator, parent_node, spec: DeviceSpec) -> None:
    """
    Build the verification tree under the given parent node and verify all requirements using provided sources.
    """
    # Create the main critical node representing the overall identification task
    main_node = evaluator.add_parallel(
        id="Gaming_Handheld_Identification",
        desc="Evaluate whether a gaming handheld device meeting all specified technical requirements has been correctly identified with verifiable specifications and source documentation",
        parent=parent_node,
        critical=True
    )

    # 1) Device name provided (existence check)
    device_name_ok = bool(spec.device_name and spec.device_name.strip())
    device_name_node = evaluator.add_custom_node(
        result=device_name_ok,
        id="Device_Name_Provided",
        desc="A specific gaming handheld device name or model is clearly identified",
        parent=main_node,
        critical=True
    )

    # 2) Specification source presence (existence of at least one URL)
    has_sources = bool(spec.source_urls and len([u for u in spec.source_urls if isinstance(u, str) and u.strip()]) > 0)
    spec_source_node = evaluator.add_custom_node(
        result=has_sources,
        id="Specification_Source",
        desc="A URL to an official product page or reliable tech news source confirming the device specifications is provided",
        parent=main_node,
        critical=True
    )

    # For all spec verifications, require that the source presence node passes; otherwise skip to avoid ungrounded checks
    extra_prereqs = [spec_source_node]

    # Helper for sources argument
    sources = spec.source_urls if has_sources else None

    # 3) Display size requirement (>= 7 inches)
    size_node = evaluator.add_leaf(
        id="Display_Size_Requirement",
        desc="The device has a display size of at least 7 inches (diagonal)",
        parent=main_node,
        critical=True
    )
    size_claim = f"The device {spec.device_name or 'the device'} has a display size of at least 7 inches (diagonal)."
    await evaluator.verify(
        claim=size_claim,
        node=size_node,
        sources=sources,
        additional_instruction="Check the page for the device's panel diagonal size. Accept any value >= 7.0 inches (e.g., 7.0, 7.4, 8.0). Sizes may be written with symbols or units. If the page lists a size in cm/mm, convert approximately.",
        extra_prerequisites=extra_prereqs
    )

    # 4) Display resolution requirement (>= 1920x1080 or >= 1920x1200)
    res_node = evaluator.add_leaf(
        id="Display_Resolution_Requirement",
        desc="The device supports a native display resolution of at least 1920x1080 (Full HD) or 1920x1200 (WUXGA)",
        parent=main_node,
        critical=True
    )
    res_claim = f"The device {spec.device_name or 'the device'} has a native display resolution that meets or exceeds Full HD (1920×1080) or WUXGA (1920×1200)."
    await evaluator.verify(
        claim=res_claim,
        node=res_node,
        sources=sources,
        additional_instruction="Verify the built-in display's native resolution from the page. Treat any resolution >= 1920x1080 (both dimensions) as meeting the requirement (e.g., 1920x1200, 2560x1600). Ignore external display output specs.",
        extra_prerequisites=extra_prereqs
    )

    # 5) Refresh rate requirement (>= 120Hz)
    rr_node = evaluator.add_leaf(
        id="Refresh_Rate_Requirement",
        desc="The device display supports a refresh rate of at least 120Hz",
        parent=main_node,
        critical=True
    )
    rr_claim = f"The device {spec.device_name or 'the device'} has a display refresh rate of at least 120Hz (e.g., 120Hz, 144Hz, 120–144Hz, up to 144Hz)."
    await evaluator.verify(
        claim=rr_claim,
        node=rr_node,
        sources=sources,
        additional_instruction="Confirm the panel refresh rate is at least 120Hz. Accept phrasing like 'up to 120Hz' or 'maximum 144Hz'. Ignore adaptive sync marketing unless it states a concrete Hz threshold.",
        extra_prerequisites=extra_prereqs
    )

    # 6) RAM requirement (>= 16GB)
    ram_node = evaluator.add_leaf(
        id="RAM_Requirement",
        desc="The device has at least 16GB of RAM",
        parent=main_node,
        critical=True
    )
    ram_claim = f"The device {spec.device_name or 'the device'} comes with at least 16GB of RAM."
    await evaluator.verify(
        claim=ram_claim,
        node=ram_node,
        sources=sources,
        additional_instruction="Verify that the RAM capacity is 16GB or higher (e.g., 16GB, 32GB). Accept different RAM types (LPDDR5, LPDDR5X, etc.).",
        extra_prerequisites=extra_prereqs
    )

    # 7) Battery capacity requirement (>= 50Wh OR >= 5000mAh)
    batt_node = evaluator.add_leaf(
        id="Battery_Capacity_Requirement",
        desc="The device has a battery capacity of at least 50Wh (watt-hours) or 5000mAh (milliamp-hours)",
        parent=main_node,
        critical=True
    )
    batt_claim = f"The device {spec.device_name or 'the device'} has a battery capacity that is at least 50 Wh or at least 5000 mAh."
    await evaluator.verify(
        claim=batt_claim,
        node=batt_node,
        sources=sources,
        additional_instruction="Accept either unit threshold: >= 50 Wh OR >= 5000 mAh. If both units are shown, either meeting the threshold qualifies. Consider typical spec sections indicating Wh or mAh.",
        extra_prerequisites=extra_prereqs
    )

    # 8) Operating system requirement (Windows 11 or SteamOS)
    os_node = evaluator.add_leaf(
        id="Operating_System_Requirement",
        desc="The device runs Windows 11 or SteamOS as its primary operating system",
        parent=main_node,
        critical=True
    )
    os_claim = f"The device {spec.device_name or 'the device'} runs either Windows 11 or SteamOS as its primary operating system."
    await evaluator.verify(
        claim=os_claim,
        node=os_node,
        sources=sources,
        additional_instruction="Confirm the shipping/preinstalled/primary OS is Windows 11 (any edition) or SteamOS (e.g., SteamOS 3). Accept phrasing like 'ships with Windows 11' or 'preinstalled SteamOS'. Do not count optional dual-boot unless it states it ships with one of these as the main OS.",
        extra_prerequisites=extra_prereqs
    )

    # 9) Release timeline requirement (released or officially announced in 2025 or 2026)
    rel_node = evaluator.add_leaf(
        id="Release_Timeline_Requirement",
        desc="The device is released or officially announced for release in 2025 or 2026",
        parent=main_node,
        critical=True
    )
    rel_claim = f"The device {spec.device_name or 'the device'} was released or officially announced in 2025 or 2026."
    await evaluator.verify(
        claim=rel_claim,
        node=rel_node,
        sources=sources,
        additional_instruction="Look for publication date, announcement date, press release date, or explicit 'announced in 2025/2026' statements. Accept 'announced for release in 2025/2026' as satisfying the condition.",
        extra_prerequisites=extra_prereqs
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the gaming handheld identification task and return a structured evaluation summary.
    """
    # Initialize evaluator
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
        default_model=model
    )

    # Extract device specs from the answer
    extracted_device = await evaluator.extract(
        prompt=prompt_extract_device(),
        template_class=DeviceSpec,
        extraction_name="extracted_device_spec"
    )

    # Build verification tree and verify
    await verify_device_specs(evaluator, root, extracted_device)

    # Return the evaluation summary
    return evaluator.get_summary()