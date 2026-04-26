import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "smartphone_2026_specs"
TASK_DESCRIPTION = """Identify a smartphone commercially available in 2026 that meets the following technical specifications:

1. Neural Processing: Must have a dedicated Neural Processing Unit (NPU) or Neural Engine with documented AI processing performance of at least 35 trillion operations per second (TOPS).

2. Ultra Wideband Connectivity: Must support Ultra Wideband (UWB) technology with a second-generation or equivalent advanced UWB chip that enables spatial awareness features such as Precision Finding.

3. Battery Specifications: Must have a battery capacity of at least 4,000 mAh, with official documentation of the battery capacity.

4. Fast Charging: Must support fast charging technology with documented charging specifications.

5. Release Timeline: Must be officially released or commercially available during the year 2026.

6. Documentation: All technical specifications must be verifiable through official manufacturer documentation, press releases, technical specification pages, or credible technology news sources.

For your answer, provide:
- The complete model name and manufacturer
- The specific Neural Engine/NPU model and its TOPS performance
- The UWB chip model and generation
- The battery capacity in mAh
- The fast charging specifications
- Official reference URLs supporting each specification
"""


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class SmartphoneExtraction(BaseModel):
    # Device identification
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None

    # Release / availability in 2026
    release_statement: Optional[str] = None  # e.g., "Released Jan 2026" or "Available 2026"
    release_urls: List[str] = Field(default_factory=list)

    # NPU / Neural Engine
    npu_present: Optional[bool] = None  # True if clearly says "dedicated NPU/Neural Engine"
    npu_model: Optional[str] = None     # e.g., "A20 Neural Engine", "Hexagon NPU"
    npu_tops: Optional[str] = None      # e.g., "45 TOPS" (free-form string)
    npu_presence_urls: List[str] = Field(default_factory=list)
    npu_tops_urls: List[str] = Field(default_factory=list)

    # UWB
    uwb_supported: Optional[bool] = None
    uwb_chip_model: Optional[str] = None  # e.g., "U1", "U2", "Qorvo DW3000", etc.
    uwb_generation: Optional[str] = None  # e.g., "second-generation", "U2", etc.
    uwb_spatial_features: Optional[str] = None  # e.g., "Precision Finding"
    uwb_support_urls: List[str] = Field(default_factory=list)
    uwb_generation_urls: List[str] = Field(default_factory=list)
    uwb_spatial_urls: List[str] = Field(default_factory=list)

    # Battery
    battery_capacity_mAh: Optional[str] = None  # free-form (e.g., "5000 mAh", "typical 4,400 mAh")
    battery_official_urls: List[str] = Field(default_factory=list)  # should be official manufacturer URLs

    # Fast charging
    fast_charging_supported: Optional[bool] = None
    fast_charging_spec: Optional[str] = None  # e.g., "80W wired", "USB-PD 3.1 45W"
    fast_charging_support_urls: List[str] = Field(default_factory=list)
    fast_charging_spec_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_smartphone() -> str:
    return """
Extract the single primary smartphone identified in the answer and all specification fields listed below. If multiple devices are mentioned, extract the first main device that the answer proposes as meeting the requirements. Return null for any missing field and return empty arrays where URLs are not provided.

Required fields to extract (use exact keys):

Device identification:
- model_name: The complete and specific model name (e.g., "Galaxy S26 Ultra", "iPhone 18 Pro Max")
- manufacturer: The manufacturer/OEM name (e.g., "Samsung", "Apple")

Release / availability:
- release_statement: A short phrase the answer uses about release/availability timing (if present)
- release_urls: An array of URLs that support that the device was released or commercially available during 2026 (official pages or credible publications)

NPU / Neural Engine:
- npu_present: true/false if the device explicitly includes a dedicated NPU/Neural Engine (do not assume)
- npu_model: The specific NPU/Neural Engine model/name if provided (or chip/SoC NPU designation)
- npu_tops: The claimed TOPS figure text as written in the answer (e.g., "35 TOPS", "45 TOPS" or "at least 35 TOPS")
- npu_presence_urls: URLs that support the presence of a dedicated NPU/Neural Engine
- npu_tops_urls: URLs that support the claimed TOPS value for the NPU/Neural Engine

UWB:
- uwb_supported: true/false if the device explicitly supports UWB
- uwb_chip_model: The UWB chip model/name if given (e.g., "U2", "Qorvo DW3000", etc.)
- uwb_generation: The generation wording if provided (e.g., "second-generation", "U2", "Gen 2")
- uwb_spatial_features: Spatial awareness feature name(s) if provided (e.g., "Precision Finding", "Find My-like")
- uwb_support_urls: URLs that support UWB support
- uwb_generation_urls: URLs that support that the UWB chip is second-generation or equivalent advanced
- uwb_spatial_urls: URLs that support that UWB enables spatial awareness features (e.g., Precision Finding)

Battery:
- battery_capacity_mAh: Battery capacity as written (do not normalize; keep units if present)
- battery_official_urls: An array of official manufacturer URLs (product spec page, press release, etc.) that explicitly state the battery capacity

Fast charging:
- fast_charging_supported: true/false if the device supports fast charging
- fast_charging_spec: The specific fast charging spec as written (e.g., "80W", "USB PD 3.1 45W", "MagSafe 15W")
- fast_charging_support_urls: URLs supporting that the device supports fast charging
- fast_charging_spec_urls: URLs supporting the specific fast charging specification

Special URL extraction rules:
- Extract only URLs explicitly present in the answer (plain or markdown). Do not invent URLs.
- For battery_official_urls, prefer URLs on the official manufacturer domain (e.g., samsung.com, apple.com, google.com/pixel).
- If a field is not present in the answer, set it to null (or empty array for URLs).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _device_label(spec: SmartphoneExtraction) -> str:
    if _nonempty(spec.manufacturer) and _nonempty(spec.model_name):
        return f"{spec.manufacturer} {spec.model_name}"
    if _nonempty(spec.model_name):
        return spec.model_name or "the device"
    return "the device"


def _merge_urls(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        if lst:
            for u in lst:
                if isinstance(u, str) and u.strip():
                    merged.append(u)
    # deduplicate while preserving order
    seen = set()
    uniq = []
    for u in merged:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, spec: SmartphoneExtraction) -> None:
    """
    Build the rubric-driven verification tree and perform all checks.
    The top-level child node under root is critical, forcing "all requirements must pass".
    """
    # ---- Top-level aggregation (critical) ----
    qual = evaluator.add_parallel(
        id="Qualifying_Smartphone_Evaluation",
        desc="Evaluate whether the identified smartphone meets all specified technical requirements and provides verifiable documentation URLs for each required specification.",
        parent=evaluator.root,
        critical=True,
    )

    # ===================== Device Identification ===================== #
    dev = evaluator.add_parallel(
        id="Device_Identification",
        desc="Verify the device is clearly identified.",
        parent=qual,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(spec.model_name),
        id="Device_Model_Name_Provided",
        desc="Provides the complete model name that uniquely identifies a specific smartphone.",
        parent=dev,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(spec.manufacturer),
        id="Manufacturer_Provided",
        desc="Provides the manufacturer name for the smartphone.",
        parent=dev,
        critical=True,
    )

    # ===================== Release Timeline (2026) ===================== #
    rel = evaluator.add_parallel(
        id="Release_Timeline_2026",
        desc="Verify the device was officially released or commercially available during 2026, with support.",
        parent=qual,
        critical=True,
    )

    # Existence of at least one supporting URL
    evaluator.add_custom_node(
        result=len(spec.release_urls) > 0,
        id="Reference_URL_For_2026_Availability",
        desc="Provides at least one reference URL (official manufacturer documentation, press release, official spec page, or credible tech publication) supporting the 2026 availability/release claim.",
        parent=rel,
        critical=True,
    )

    # Verify 2026 availability via provided URLs
    released_2026 = evaluator.add_leaf(
        id="Released_or_Commercially_Available_in_2026",
        desc="Device was officially released or commercially available during the year 2026.",
        parent=rel,
        critical=True,
    )
    claim_release = f"{_device_label(spec)} was officially released or commercially available during 2026."
    await evaluator.verify(
        claim=claim_release,
        node=released_2026,
        sources=spec.release_urls,
        additional_instruction=(
            "Check the provided pages for explicit evidence that the device launched, was released, "
            "went on sale, or was otherwise commercially available at any time during calendar year 2026. "
            "Accept: official manufacturer pages or credible technology news with clear 2026 date and explicit device availability. "
            "Reject rumor-only pages or pages that only predict future availability without confirmation."
        ),
    )

    # ===================== AI / NPU Requirements ===================== #
    ai = evaluator.add_parallel(
        id="AI_Processing_Requirements",
        desc="Verify the device meets the dedicated NPU/Neural Engine requirement and the ≥35 TOPS performance requirement, with support.",
        parent=qual,
        critical=True,
    )

    # Existence of NPU presence URLs
    evaluator.add_custom_node(
        result=len(spec.npu_presence_urls) > 0,
        id="Reference_URL_For_NPU_Presence",
        desc="Provides at least one reference URL (official manufacturer documentation, press release, official spec page, or credible tech publication) supporting the presence of a dedicated NPU/Neural Engine.",
        parent=ai,
        critical=True,
    )

    # Dedicated NPU/Neural Engine presence verified by URLs
    npu_presence = evaluator.add_leaf(
        id="Dedicated_NPU_or_Neural_Engine",
        desc="Device includes a dedicated NPU/Neural Engine.",
        parent=ai,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_device_label(spec)} includes a dedicated Neural Processing Unit (NPU) or Neural Engine.",
        node=npu_presence,
        sources=spec.npu_presence_urls,
        additional_instruction=(
            "Look for explicit references to a dedicated NPU/Neural Engine (not just generic 'AI features'). "
            "Accept equivalent terms like 'Neural Engine', 'AI accelerator', or 'dedicated AI processor' "
            "if clearly part of the SoC and distinct from general-purpose CPU/GPU."
        ),
    )

    # NPU/Engine model identified (existence check)
    evaluator.add_custom_node(
        result=_nonempty(spec.npu_model),
        id="NPU_or_Engine_Model_Identified",
        desc="Specifies the NPU/Neural Engine model/name (or clearly identifies the chip/SoC component corresponding to the NPU/Neural Engine).",
        parent=ai,
        critical=True,
    )

    # Existence of NPU TOPS URLs
    evaluator.add_custom_node(
        result=len(spec.npu_tops_urls) > 0,
        id="Reference_URL_For_NPU_TOPS",
        desc="Provides at least one reference URL (official manufacturer documentation, press release, official spec page, or credible tech publication) supporting the claimed TOPS performance figure.",
        parent=ai,
        critical=True,
    )

    # Verify >=35 TOPS via URLs (judge reads numbers from page)
    npu_tops = evaluator.add_leaf(
        id="NPU_TOPS_at_Least_35",
        desc="Documented AI processing performance for the NPU/Neural Engine is at least 35 TOPS.",
        parent=ai,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The dedicated NPU/Neural Engine in {_device_label(spec)} provides at least 35 TOPS of AI processing performance."
        ),
        node=npu_tops,
        sources=spec.npu_tops_urls,
        additional_instruction=(
            "Verify the numeric TOPS figure and ensure it refers specifically to the NPU/Neural Engine (or equivalent dedicated AI accelerator), "
            "not to unrelated components (e.g., GPU-only) and not merely to total system, unless the wording clearly attributes the TOPS to the NPU/Neural engine. "
            "Accept '>= 35 TOPS', '35+ TOPS', or any value >= 35 TOPS."
        ),
    )

    # ===================== UWB Requirements ===================== #
    uwb = evaluator.add_parallel(
        id="UWB_Connectivity_Requirements",
        desc="Verify the device supports UWB, includes a second-generation (or equivalent) UWB chip, enables spatial-awareness features, and provides support.",
        parent=qual,
        critical=True,
    )

    # Existence of URLs for each UWB aspect
    evaluator.add_custom_node(
        result=len(spec.uwb_support_urls) > 0,
        id="Reference_URL_For_UWB_Support",
        desc="Provides at least one reference URL (official manufacturer documentation, press release, official spec page, or credible tech publication) supporting UWB support.",
        parent=uwb,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(spec.uwb_generation_urls) > 0,
        id="Reference_URL_For_UWB_Generation",
        desc="Provides at least one reference URL (official manufacturer documentation, press release, official spec page, or credible tech publication) supporting the second-generation/equivalent UWB chip claim.",
        parent=uwb,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(spec.uwb_spatial_urls) > 0,
        id="Reference_URL_For_UWB_Spatial_Features",
        desc="Provides at least one reference URL (official manufacturer documentation, press release, official spec page, or credible tech publication) supporting the spatial-awareness feature claim enabled by UWB (e.g., Precision Finding or equivalent).",
        parent=uwb,
        critical=True,
    )

    # UWB supported (verify by URL)
    uwb_supported_leaf = evaluator.add_leaf(
        id="UWB_Supported",
        desc="Device supports Ultra Wideband (UWB).",
        parent=uwb,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_device_label(spec)} supports Ultra Wideband (UWB) connectivity.",
        node=uwb_supported_leaf,
        sources=spec.uwb_support_urls,
        additional_instruction="Look for 'UWB' explicitly in official spec lists or credible specification summaries.",
    )

    # UWB chip model identified (existence check)
    evaluator.add_custom_node(
        result=_nonempty(spec.uwb_chip_model),
        id="UWB_Chip_Model_Identified",
        desc="Identifies the UWB chip model/name (or otherwise uniquely identifies the UWB hardware used).",
        parent=uwb,
        critical=True,
    )

    # UWB generation is second-gen or equivalent (verify by URL)
    uwb_gen_leaf = evaluator.add_leaf(
        id="UWB_Chip_Second_Gen_or_Equivalent",
        desc="Documentation indicates the UWB chip is second-generation or equivalent advanced UWB.",
        parent=uwb,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The UWB hardware in {_device_label(spec)} is second-generation or an equivalent advanced UWB implementation.",
        node=uwb_gen_leaf,
        sources=spec.uwb_generation_urls,
        additional_instruction=(
            "Accept phrasings such as 'second-generation', '2nd gen', 'U2', 'Gen 2', "
            "or vendor-specific naming that clearly denotes the second generation or a directly comparable next-gen improvement."
        ),
    )

    # Spatial awareness features enabled (verify by URL)
    uwb_spatial_leaf = evaluator.add_leaf(
        id="Spatial_Awareness_Features_Enabled",
        desc="Documentation indicates UWB enables spatial awareness features (e.g., Precision Finding or equivalent).",
        parent=uwb,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"UWB on {_device_label(spec)} enables spatial awareness features such as 'Precision Finding' or an equivalent feature."
        ),
        node=uwb_spatial_leaf,
        sources=spec.uwb_spatial_urls,
        additional_instruction=(
            "Look for features like 'Precision Finding', device-to-device direction finding, spatial tracking, or similar UWB-enabled locating features."
        ),
    )

    # ===================== Battery Requirements ===================== #
    bat = evaluator.add_parallel(
        id="Battery_Requirements",
        desc="Verify battery capacity meets the minimum and is officially documented by the manufacturer.",
        parent=qual,
        critical=True,
    )

    # Official manufacturer URL provided (existence)
    evaluator.add_custom_node(
        result=len(spec.battery_official_urls) > 0,
        id="Official_Manufacturer_Reference_URL_For_Battery_Capacity",
        desc="Provides at least one official manufacturer URL (spec page, press release, or equivalent official documentation) explicitly supporting the stated battery capacity in mAh.",
        parent=bat,
        critical=True,
    )

    # Verify >= 4000 mAh via official URLs
    bat_cap_leaf = evaluator.add_leaf(
        id="Battery_Capacity_At_Least_4000mAh",
        desc="Battery capacity is at least 4,000 mAh.",
        parent=bat,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The battery capacity of {_device_label(spec)} is at least 4000 mAh.",
        node=bat_cap_leaf,
        sources=spec.battery_official_urls,
        additional_instruction=(
            "Only accept evidence from official manufacturer pages (product spec page, official press release, support docs) that explicitly show a capacity ≥ 4000 mAh. "
            "If the provided URL is not an official manufacturer domain or does not clearly state the capacity, mark as not supported."
        ),
    )

    # ===================== Fast Charging Requirements ===================== #
    fc = evaluator.add_parallel(
        id="Fast_Charging_Requirements",
        desc="Verify fast charging is supported, provide charging specs, and provide support.",
        parent=qual,
        critical=True,
    )

    # Existence of URLs
    evaluator.add_custom_node(
        result=len(spec.fast_charging_support_urls) > 0,
        id="Reference_URL_For_Fast_Charging_Support",
        desc="Provides at least one reference URL (official manufacturer documentation, press release, official spec page, or credible tech publication) supporting that the device supports fast charging.",
        parent=fc,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(spec.fast_charging_spec_urls) > 0,
        id="Reference_URL_For_Fast_Charging_Specs",
        desc="Provides at least one reference URL (official manufacturer documentation, press release, official spec page, or credible tech publication) supporting the stated fast charging specifications.",
        parent=fc,
        critical=True,
    )

    # Fast charging supported - verify by URL
    fc_supported_leaf = evaluator.add_leaf(
        id="Fast_Charging_Supported",
        desc="Device supports fast charging technology.",
        parent=fc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_device_label(spec)} supports fast charging.",
        node=fc_supported_leaf,
        sources=spec.fast_charging_support_urls,
        additional_instruction=(
            "Look for explicit mention of 'fast charging', 'quick charge', 'SuperCharge', 'USB-PD fast charging', etc. "
            "Rumor-only pages are not acceptable."
        ),
    )

    # Fast charging specifications provided (existence)
    evaluator.add_custom_node(
        result=_nonempty(spec.fast_charging_spec),
        id="Fast_Charging_Specifications_Provided",
        desc="Provides specific fast charging specifications (e.g., wattage and/or charging standard).",
        parent=fc,
        critical=True,
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
    Evaluate an answer for the 2026 smartphone qualification task.
    """
    # Initialize evaluator (root is non-critical; we create a critical top-level child node)
    evaluator = Evaluator()
    evaluator.initialize(
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
    extracted: SmartphoneExtraction = await evaluator.extract(
        prompt=prompt_extract_smartphone(),
        template_class=SmartphoneExtraction,
        extraction_name="smartphone_extraction",
    )

    # Add some contextual info into the summary (optional, helpful for debugging)
    evaluator.add_custom_info(
        {
            "device_label": _device_label(extracted),
            "urls_count": {
                "release": len(extracted.release_urls),
                "npu_presence": len(extracted.npu_presence_urls),
                "npu_tops": len(extracted.npu_tops_urls),
                "uwb_support": len(extracted.uwb_support_urls),
                "uwb_generation": len(extracted.uwb_generation_urls),
                "uwb_spatial": len(extracted.uwb_spatial_urls),
                "battery_official": len(extracted.battery_official_urls),
                "fc_support": len(extracted.fast_charging_support_urls),
                "fc_spec": len(extracted.fast_charging_spec_urls),
            },
        },
        info_type="extraction_summary",
        info_name="extraction_overview",
    )

    # Build and run verification checks
    await build_verification_tree(evaluator, extracted)

    # Return final structured result
    return evaluator.get_summary()