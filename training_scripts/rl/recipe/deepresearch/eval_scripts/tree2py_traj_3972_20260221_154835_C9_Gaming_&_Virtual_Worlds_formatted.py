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
TASK_ID = "vr_headsets_2026_multicriteria"
TASK_DESCRIPTION = (
    "Identify three VR headsets announced or released in 2026 that meet all of the following criteria: "
    "(1) Must support both standalone VR operation and wireless PC VR streaming capability, "
    "(2) Must include integrated eye tracking, "
    "(3) Must have a horizontal field of view of at least 105 degrees, "
    "(4) Must support 6DoF inside-out tracking without requiring external base stations for basic operation, "
    "(5) Must include modern pancake lens technology or equivalent, "
    "(6) Must include integrated audio, "
    "(7) Must include a passthrough camera system, and "
    "(8) Must support WiFi 6E or newer. "
    "For each headset, provide the model name, manufacturer, detailed technical specifications including per-eye resolution, "
    "exact horizontal field of view, panel type, refresh rate, weight without headstrap, operating system/platform, "
    "WiFi standard, pricing information if publicly available, and reference URLs for display specifications, "
    "hardware specifications, connectivity features, and release information."
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class URLRefs(BaseModel):
    display_spec_urls: List[str] = Field(default_factory=list)
    hardware_spec_urls: List[str] = Field(default_factory=list)
    connectivity_feature_urls: List[str] = Field(default_factory=list)
    release_info_urls: List[str] = Field(default_factory=list)
    pricing_urls: List[str] = Field(default_factory=list)


class HeadsetInfo(BaseModel):
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None

    # Display
    per_eye_resolution: Optional[str] = None
    horizontal_fov: Optional[str] = None
    panel_type: Optional[str] = None
    lens_type: Optional[str] = None
    refresh_rate: Optional[str] = None

    # Hardware
    weight_without_headstrap: Optional[str] = None
    eye_tracking: Optional[str] = None
    tracking_dof: Optional[str] = None
    tracking_type: Optional[str] = None  # e.g., "inside-out"
    integrated_audio: Optional[str] = None

    # Connectivity/operation
    supports_standalone: Optional[str] = None  # "yes"/"no"/"unknown" preferred
    supports_wireless_pc_streaming: Optional[str] = None  # "yes"/"no"/"unknown"
    wifi_standard: Optional[str] = None
    passthrough_camera: Optional[str] = None

    # Release/platform
    platform_os: Optional[str] = None
    announced_or_released_year: Optional[str] = None

    # Pricing (optional)
    pricing_info: Optional[str] = None

    # Reference URLs
    urls: URLRefs = Field(default_factory=URLRefs)


class VRHeadsetsExtraction(BaseModel):
    headsets: List[HeadsetInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_headsets() -> str:
    return """
Extract up to the first three VR headsets described in the answer that the author claims meet the stated 2026 criteria.

For each headset, extract the following fields exactly as stated in the answer (strings are preferred; do not coerce to numbers). If a field is not clearly provided in the answer text, set it to null:

- model_name: The commercial model name (e.g., "X Brand VR Pro")
- manufacturer: Manufacturer or brand

DISPLAY
- per_eye_resolution: Per-eye resolution (e.g., "2160 x 2160")
- horizontal_fov: Exact horizontal field of view (e.g., "110°", "106 degrees")
- panel_type: Display panel type (e.g., "LCD", "Micro-OLED", "Sony Micro-OLED")
- lens_type: Lens technology (e.g., "pancake", "folded optics", or equivalent)
- refresh_rate: Supported max refresh rate (e.g., "120Hz", "90 Hz")

HARDWARE
- weight_without_headstrap: Weight of the visor/headset body without the headstrap if explicitly specified (string, e.g., "420 g")
- eye_tracking: "yes" / "no" / "unknown"
- tracking_dof: e.g., "6DoF", "3DoF"
- tracking_type: e.g., "inside-out", "outside-in"
- integrated_audio: "yes" / "no" / "unknown"

CONNECTIVITY / OPERATION
- supports_standalone: "yes" / "no" / "unknown" (i.e., operates without a PC)
- supports_wireless_pc_streaming: "yes" / "no" / "unknown" (e.g., via Air Link / Steam Link / Virtual Desktop)
- wifi_standard: e.g., "Wi‑Fi 6E", "Wi‑Fi 7 (802.11be)", "Wi‑Fi 6"
- passthrough_camera: "yes" / "no" / "unknown"

RELEASE / PLATFORM
- platform_os: OS/platform (e.g., "Android-based", "Horizon OS", "Windows Mixed Reality")
- announced_or_released_year: the year (e.g., "2026") as stated

PRICING
- pricing_info: Publicly available pricing info if disclosed in the answer (string, e.g., "$499", "TBD", "not announced")

REFERENCE URLS
- urls.display_spec_urls: URLs cited in the answer that support display specs
- urls.hardware_spec_urls: URLs cited in the answer that support hardware specs
- urls.connectivity_feature_urls: URLs cited in the answer that support connectivity/operation features
- urls.release_info_urls: URLs cited in the answer that support release/manufacturer/platform details
- urls.pricing_urls: URLs cited in the answer that support price information

RULES:
1) Only extract URLs explicitly present in the answer. Do not invent URLs.
2) Keep the raw strings from the answer (e.g., "about 110°", "up to 120 Hz"), do not normalize.
3) If more than three headsets are mentioned, extract the first three in the order they appear.
4) If fewer than three are mentioned, return fewer; missing ones will be padded later by the evaluator.
"""


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _src_or_none(urls: List[str]) -> Optional[List[str]]:
    if urls and len(urls) > 0:
        return urls
    return None


def _name_for_claim(h: HeadsetInfo) -> str:
    if _nonempty(h.model_name) and _nonempty(h.manufacturer):
        return f"{h.manufacturer} {h.model_name}"
    if _nonempty(h.model_name):
        return h.model_name  # type: ignore
    if _nonempty(h.manufacturer):
        return f"{h.manufacturer} headset"
    return "the headset"


# --------------------------------------------------------------------------- #
# Verification Builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_display(evaluator: Evaluator, parent_node, h: HeadsetInfo, idx: int) -> None:
    # Parent: headset_{i}_display (critical, parallel)
    disp_node = evaluator.add_parallel(
        id=f"headset_{idx}_display",
        desc=f"Display and visual specifications for {'first' if idx==1 else ('second' if idx==2 else 'third')} headset",
        parent=parent_node,
        critical=True
    )

    # Child: headset_{i}_display_specs (critical, parallel)
    disp_specs_node = evaluator.add_parallel(
        id=f"headset_{idx}_display_specs",
        desc="Technical display specifications",
        parent=disp_node,
        critical=True
    )

    name = _name_for_claim(h)
    disp_srcs = h.urls.display_spec_urls if h and h.urls else []
    disp_sources = _src_or_none(disp_srcs)

    # Resolution (critical leaf) - if missing, mark failed immediately
    if not _nonempty(h.per_eye_resolution):
        evaluator.add_leaf(
            id=f"headset_{idx}_resolution",
            desc="Per-eye resolution is specified and verified",
            parent=disp_specs_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        res_node = evaluator.add_leaf(
            id=f"headset_{idx}_resolution",
            desc="Per-eye resolution is specified and verified",
            parent=disp_specs_node,
            critical=True
        )
        claim = f"The per-eye resolution of {name} is {h.per_eye_resolution}."
        await evaluator.verify(
            claim=claim,
            node=res_node,
            sources=disp_sources,
            additional_instruction="Verify the per-eye resolution exactly or approximately matches the page; accept formatting variants (e.g., '2160 × 2160' vs '2160 x 2160')."
        )

    # Horizontal FOV >= 105° (critical leaf) - allow verification even if value not given in answer
    fov_node = evaluator.add_leaf(
        id=f"headset_{idx}_fov",
        desc="Horizontal FOV is at least 105 degrees",
        parent=disp_specs_node,
        critical=True
    )
    if _nonempty(h.horizontal_fov):
        claim_fov = f"{name} has a horizontal field of view of {h.horizontal_fov}, which is at least 105°."
    else:
        claim_fov = f"{name} has a horizontal field of view of at least 105°."
    await evaluator.verify(
        claim=claim_fov,
        node=fov_node,
        sources=disp_sources,
        additional_instruction="Check the page for explicit horizontal FOV. If a range is provided, use the lower bound. If only diagonal/vertical is provided without horizontal, do not consider it a pass."
    )

    # Panel type (critical) - if missing, fail immediately
    if not _nonempty(h.panel_type):
        evaluator.add_leaf(
            id=f"headset_{idx}_panel_type",
            desc="Panel type is specified (LCD, Micro-OLED, or Sony Micro-OLED)",
            parent=disp_specs_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        pt_node = evaluator.add_leaf(
            id=f"headset_{idx}_panel_type",
            desc="Panel type is specified (LCD, Micro-OLED, or Sony Micro-OLED)",
            parent=disp_specs_node,
            critical=True
        )
        claim_pt = f"{name} uses {h.panel_type} display panels."
        await evaluator.verify(
            claim=claim_pt,
            node=pt_node,
            sources=disp_sources,
            additional_instruction="Verify the panel type (e.g., LCD, Micro‑OLED, 'Sony Micro‑OLED'). Allow reasonable naming variants."
        )

    # Lens type modern pancake or equivalent (critical)
    lens_node = evaluator.add_leaf(
        id=f"headset_{idx}_lens_type",
        desc="Lens technology is modern pancake lenses or equivalent",
        parent=disp_specs_node,
        critical=True
    )
    if _nonempty(h.lens_type):
        claim_lens = f"{name} uses modern pancake (folded) lens technology or an equivalent; the lens type is '{h.lens_type}'."
    else:
        claim_lens = f"{name} uses modern pancake (folded) lens technology or an equivalent."
    await evaluator.verify(
        claim=claim_lens,
        node=lens_node,
        sources=disp_sources,
        additional_instruction="Confirm that the headset uses pancake/folded optics or an equivalent modern folded lens design. Accept synonyms like 'pancake', 'folded optics', 'freeform pancake', 'polarized stack'."
    )

    # Refresh rate supports at least 90 Hz (critical)
    rr_node = evaluator.add_leaf(
        id=f"headset_{idx}_refresh_rate",
        desc="Refresh rate supports at least 90Hz",
        parent=disp_specs_node,
        critical=True
    )
    if _nonempty(h.refresh_rate):
        claim_rr = f"{name} supports a refresh rate of {h.refresh_rate}, which is at least 90 Hz."
    else:
        claim_rr = f"{name} supports a refresh rate of at least 90 Hz."
    await evaluator.verify(
        claim=claim_rr,
        node=rr_node,
        sources=disp_sources,
        additional_instruction="Check the max/available refresh rate. If a range is given, ensure the max or a selectable mode is ≥ 90 Hz."
    )

    # URL existence check (critical)
    evaluator.add_custom_node(
        result=bool(disp_srcs),
        id=f"headset_{idx}_display_url",
        desc="URL reference provided for display specifications",
        parent=disp_node,
        critical=True
    )


async def _verify_hardware(evaluator: Evaluator, parent_node, h: HeadsetInfo, idx: int) -> None:
    # Parent: headset_{i}_hardware (critical, parallel)
    hw_node = evaluator.add_parallel(
        id=f"headset_{idx}_hardware",
        desc=f"Hardware and tracking specifications for {'first' if idx==1 else ('second' if idx==2 else 'third')} headset",
        parent=parent_node,
        critical=True
    )

    # Child: headset_{i}_hardware_specs (critical, parallel)
    hw_specs_node = evaluator.add_parallel(
        id=f"headset_{idx}_hardware_specs",
        desc="Technical hardware specifications",
        parent=hw_node,
        critical=True
    )

    name = _name_for_claim(h)
    hw_srcs = h.urls.hardware_spec_urls if h and h.urls else []
    hw_sources = _src_or_none(hw_srcs)

    # Weight without headstrap is specified (critical) - if missing, fail immediately
    if not _nonempty(h.weight_without_headstrap):
        evaluator.add_leaf(
            id=f"headset_{idx}_weight",
            desc="Weight without headstrap is specified",
            parent=hw_specs_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        w_node = evaluator.add_leaf(
            id=f"headset_{idx}_weight",
            desc="Weight without headstrap is specified",
            parent=hw_specs_node,
            critical=True
        )
        claim_w = f"The weight of {name} without the headstrap (visor/body only) is {h.weight_without_headstrap}."
        await evaluator.verify(
            claim=claim_w,
            node=w_node,
            sources=hw_sources,
            additional_instruction="Only pass if the page explicitly states 'without headstrap', 'visor/body only', or equivalent. If only total weight with strap is given, it should not pass."
        )

    # Eye tracking present (critical)
    et_node = evaluator.add_leaf(
        id=f"headset_{idx}_eye_tracking",
        desc="Integrated eye tracking capability is present",
        parent=hw_specs_node,
        critical=True
    )
    claim_et = f"{name} includes integrated eye tracking."
    await evaluator.verify(
        claim=claim_et,
        node=et_node,
        sources=hw_sources,
        additional_instruction="Look for 'eye tracking' integrated into the headset. If optional module only and not integrated, do not pass."
    )

    # 6DoF tracking (critical)
    dof_node = evaluator.add_leaf(
        id=f"headset_{idx}_tracking_dof",
        desc="Supports 6DoF tracking",
        parent=hw_specs_node,
        critical=True
    )
    claim_dof = f"{name} supports 6DoF tracking."
    await evaluator.verify(
        claim=claim_dof,
        node=dof_node,
        sources=hw_sources,
        additional_instruction="Confirm six degrees of freedom head/hand tracking is supported."
    )

    # Inside-out tracking without required external base stations (critical)
    it_node = evaluator.add_leaf(
        id=f"headset_{idx}_tracking_type",
        desc="Inside-out tracking without required external base stations",
        parent=hw_specs_node,
        critical=True
    )
    claim_it = f"{name} uses inside-out tracking and does not require external base stations for basic operation."
    await evaluator.verify(
        claim=claim_it,
        node=it_node,
        sources=hw_sources,
        additional_instruction="Accept inside-out camera-based tracking. If base stations (like Lighthouse) are required for basic operation, do not pass; if optional but inside-out works standalone, pass."
    )

    # Integrated audio present (critical)
    aud_node = evaluator.add_leaf(
        id=f"headset_{idx}_audio",
        desc="Integrated audio solution is present",
        parent=hw_specs_node,
        critical=True
    )
    claim_audio = f"{name} includes integrated audio (e.g., built-in speakers/temple transducers)."
    await evaluator.verify(
        claim=claim_audio,
        node=aud_node,
        sources=hw_sources,
        additional_instruction="Confirm built-in/on-headset audio. Earbuds included in box but not integrated do not count as integrated audio."
    )

    # URL existence (critical)
    evaluator.add_custom_node(
        result=bool(hw_srcs),
        id=f"headset_{idx}_hardware_url",
        desc="URL reference provided for hardware specifications",
        parent=hw_node,
        critical=True
    )


async def _verify_connectivity(evaluator: Evaluator, parent_node, h: HeadsetInfo, idx: int) -> None:
    # Parent: headset_{i}_connectivity (critical, parallel)
    conn_node = evaluator.add_parallel(
        id=f"headset_{idx}_connectivity",
        desc=f"Connectivity and operation features for {'first' if idx==1 else ('second' if idx==2 else 'third')} headset",
        parent=parent_node,
        critical=True
    )

    # Child: headset_{i}_connectivity_features (critical, parallel)
    conn_feat_node = evaluator.add_parallel(
        id=f"headset_{idx}_connectivity_features",
        desc="Connectivity capabilities",
        parent=conn_node,
        critical=True
    )

    name = _name_for_claim(h)
    conn_srcs = h.urls.connectivity_feature_urls if h and h.urls else []
    conn_sources = _src_or_none(conn_srcs)

    # Standalone support (critical)
    so_node = evaluator.add_leaf(
        id=f"headset_{idx}_standalone",
        desc="Supports standalone operation without PC",
        parent=conn_feat_node,
        critical=True
    )
    claim_so = f"{name} supports standalone VR operation without requiring a PC."
    await evaluator.verify(
        claim=claim_so,
        node=so_node,
        sources=conn_sources,
        additional_instruction="Look for on-board SoC/OS indicating standalone usage."
    )

    # Wireless PC VR streaming (critical)
    ws_node = evaluator.add_leaf(
        id=f"headset_{idx}_wireless_streaming",
        desc="Supports wireless PC VR streaming capability",
        parent=conn_feat_node,
        critical=True
    )
    claim_ws = f"{name} supports wireless PC VR streaming (e.g., Air Link, Steam Link, or Virtual Desktop)."
    await evaluator.verify(
        claim=claim_ws,
        node=ws_node,
        sources=conn_sources,
        additional_instruction="Accept official or well-documented wireless PC VR streaming solutions. If only wired streaming is supported, do not pass."
    )

    # Wi‑Fi 6E or newer (critical)
    wifi_node = evaluator.add_leaf(
        id=f"headset_{idx}_wifi",
        desc="WiFi standard is WiFi 6E or newer",
        parent=conn_feat_node,
        critical=True
    )
    if _nonempty(h.wifi_standard):
        claim_wifi = f"{name} supports {h.wifi_standard}, which is Wi‑Fi 6E or newer."
    else:
        claim_wifi = f"{name} supports Wi‑Fi 6E or newer (e.g., Wi‑Fi 6E or Wi‑Fi 7)."
    await evaluator.verify(
        claim=claim_wifi,
        node=wifi_node,
        sources=conn_sources,
        additional_instruction="Pass only if the page indicates Wi‑Fi 6E or Wi‑Fi 7 (802.11be). Wi‑Fi 6 (802.11ax) without E should not pass."
    )

    # Passthrough camera system (critical)
    pt_node = evaluator.add_leaf(
        id=f"headset_{idx}_passthrough",
        desc="Includes passthrough camera system",
        parent=conn_feat_node,
        critical=True
    )
    claim_pt = f"{name} includes a passthrough camera system for seeing the real world while wearing the headset."
    await evaluator.verify(
        claim=claim_pt,
        node=pt_node,
        sources=conn_sources,
        additional_instruction="Look for 'passthrough', 'see-through', MR cameras enabling color or black‑and‑white passthrough. Depth sensors alone without video passthrough should not pass."
    )

    # URL existence (critical)
    evaluator.add_custom_node(
        result=bool(conn_srcs),
        id=f"headset_{idx}_connectivity_url",
        desc="URL reference provided for connectivity features",
        parent=conn_node,
        critical=True
    )


async def _verify_release(evaluator: Evaluator, parent_node, h: HeadsetInfo, idx: int) -> None:
    # Parent: headset_{i}_release (critical, parallel)
    rel_node = evaluator.add_parallel(
        id=f"headset_{idx}_release",
        desc=f"Release and availability information for {'first' if idx==1 else ('second' if idx==2 else 'third')} headset",
        parent=parent_node,
        critical=True
    )

    # Child: headset_{i}_release_info (critical, parallel)
    rel_info_node = evaluator.add_parallel(
        id=f"headset_{idx}_release_info",
        desc="Release details",
        parent=rel_node,
        critical=True
    )

    name = _name_for_claim(h)
    rel_srcs = h.urls.release_info_urls if h and h.urls else []
    rel_sources = _src_or_none(rel_srcs)

    # Year is 2026 (critical)
    yr_node = evaluator.add_leaf(
        id=f"headset_{idx}_year",
        desc="Announced or released in 2026",
        parent=rel_info_node,
        critical=True
    )
    claim_year = f"{name} was announced or released in 2026."
    await evaluator.verify(
        claim=claim_year,
        node=yr_node,
        sources=rel_sources,
        additional_instruction="Pass if the page shows launch/announcement in 2026 or availability in 2026. Rumored/forecast without credible source should not pass."
    )

    # Manufacturer (critical) - if missing manufacturer in answer, fail leaf to enforce 'provided'
    if not _nonempty(h.manufacturer):
        evaluator.add_leaf(
            id=f"headset_{idx}_manufacturer",
            desc="From recognized VR headset manufacturer",
            parent=rel_info_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        man_node = evaluator.add_leaf(
            id=f"headset_{idx}_manufacturer",
            desc="From recognized VR headset manufacturer",
            parent=rel_info_node,
            critical=True
        )
        claim_man = f"The {name} is manufactured by {h.manufacturer}."
        await evaluator.verify(
            claim=claim_man,
            node=man_node,
            sources=rel_sources,
            additional_instruction="Verify manufacturer attribution on the page. The brand should be responsible for the headset."
        )

    # Platform/OS specified (critical) - if missing, fail leaf
    if not _nonempty(h.platform_os):
        evaluator.add_leaf(
            id=f"headset_{idx}_platform",
            desc="Operating system or platform is specified",
            parent=rel_info_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        plat_node = evaluator.add_leaf(
            id=f"headset_{idx}_platform",
            desc="Operating system or platform is specified",
            parent=rel_info_node,
            critical=True
        )
        claim_pl = f"{name} runs on the {h.platform_os} platform/OS."
        await evaluator.verify(
            claim=claim_pl,
            node=plat_node,
            sources=rel_sources,
            additional_instruction="Confirm platform/OS on the page (e.g., Android/Meta Horizon OS, proprietary OS)."
        )

    # URL existence (critical)
    evaluator.add_custom_node(
        result=bool(rel_srcs),
        id=f"headset_{idx}_release_url",
        desc="URL reference provided for release information",
        parent=rel_node,
        critical=True
    )


async def _verify_pricing(evaluator: Evaluator, parent_node, h: HeadsetInfo, idx: int) -> None:
    # Parent: headset_{i}_pricing (non-critical, parallel)
    price_node = evaluator.add_parallel(
        id=f"headset_{idx}_pricing",
        desc=f"Pricing information for {'first' if idx==1 else ('second' if idx==2 else 'third')} headset",
        parent=parent_node,
        critical=False
    )

    name = _name_for_claim(h)
    price_srcs = h.urls.pricing_urls if h and h.urls else []
    price_sources = _src_or_none(price_srcs)

    # Pricing info present/verified (non-critical leaf)
    price_leaf = evaluator.add_leaf(
        id=f"headset_{idx}_price",
        desc="Pricing information is publicly available or disclosed",
        parent=price_node,
        critical=False
    )
    if _nonempty(h.pricing_info):
        claim_price = f"The publicly available price for {name} is {h.pricing_info}."
    else:
        claim_price = f"There is publicly available pricing information for {name}."
    await evaluator.verify(
        claim=claim_price,
        node=price_leaf,
        sources=price_sources,
        additional_instruction="Pass if the page discloses pricing (MSRP, starting price, or pre‑order price). If no pricing appears on the page, do not pass."
    )

    # Pricing URL existence (non-critical)
    evaluator.add_custom_node(
        result=bool(price_srcs),
        id=f"headset_{idx}_price_url",
        desc="URL reference provided for pricing information",
        parent=price_node,
        critical=False
    )


async def verify_headset(evaluator: Evaluator, root, h: HeadsetInfo, index_one_based: int) -> None:
    """
    Build the verification subtree for one headset.
    """
    # Parent: headset_{i} (non-critical, parallel)
    hs_node = evaluator.add_parallel(
        id=f"headset_{index_one_based}",
        desc=f"{'First' if index_one_based==1 else ('Second' if index_one_based==2 else 'Third')} VR headset identification and specifications",
        parent=root,
        critical=False
    )

    # Display (critical group)
    await _verify_display(evaluator, hs_node, h, index_one_based)

    # Hardware (critical group)
    await _verify_hardware(evaluator, hs_node, h, index_one_based)

    # Connectivity / operation (critical group)
    await _verify_connectivity(evaluator, hs_node, h, index_one_based)

    # Release/platform (critical group)
    await _verify_release(evaluator, hs_node, h, index_one_based)

    # Pricing (non-critical)
    await _verify_pricing(evaluator, hs_node, h, index_one_based)


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an answer for the 2026 VR headsets criteria task.
    """
    # Initialize evaluator with root parallel strategy
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

    # Extract up to 3 headsets from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_headsets(),
        template_class=VRHeadsetsExtraction,
        extraction_name="vr_headsets_extraction"
    )

    # Normalize to exactly 3 items for evaluation
    headsets: List[HeadsetInfo] = list(extracted.headsets[:3]) if extracted and extracted.headsets else []
    while len(headsets) < 3:
        headsets.append(HeadsetInfo())

    # Add minimal custom info for diagnostics
    evaluator.add_custom_info(
        info={"extracted_count": len(extracted.headsets) if extracted and extracted.headsets else 0},
        info_type="extraction_statistics",
        info_name="extraction_stats"
    )

    # Build subtree for each of the three headsets
    for i in range(3):
        await verify_headset(evaluator, root, headsets[i], i + 1)

    # Return evaluator summary
    return evaluator.get_summary()