import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "vr_headset_ue5_ps5_vrs"
TASK_DESCRIPTION = (
    "A VR game developer is planning to create an immersive simulation using Unreal Engine 5 that requires the following capabilities:\n"
    "- OLED display technology for superior contrast ratios and black levels\n"
    "- Native eye tracking capability for implementing foveated rendering optimization\n"
    "- Console-based VR platform (the headset must connect to a gaming console)\n"
    "- Refresh rate support of at least 120Hz\n\n"
    "Identify the commercially available VR headset released before 2025 that meets ALL these requirements. Then provide:\n"
    "1. Its exact per-eye display resolution (width x height)\n"
    "2. The minimum GPU architecture series required to support Variable Rate Shading (VRS) for foveated rendering when using this headset on PC\n"
    "3. The minimum physical room dimensions required for room-scale VR experiences"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HeadsetSelection(BaseModel):
    """Information about the chosen headset and supporting sources."""
    headset_name: Optional[str] = None
    release_year_or_date: Optional[str] = None
    display_tech: Optional[str] = None
    eye_tracking_capability: Optional[str] = None
    console_platform: Optional[str] = None
    refresh_rate_spec: Optional[str] = None
    ue5_platform_support: Optional[str] = None

    # General/global sources that support the headset meeting constraints
    selection_sources: List[str] = Field(default_factory=list)

    # Property-specific sources if provided in the answer (optional)
    release_sources: List[str] = Field(default_factory=list)
    display_sources: List[str] = Field(default_factory=list)
    eye_tracking_sources: List[str] = Field(default_factory=list)
    console_sources: List[str] = Field(default_factory=list)
    refresh_sources: List[str] = Field(default_factory=list)
    ue5_sources: List[str] = Field(default_factory=list)


class SpecOutputs(BaseModel):
    """Three requested outputs with sources."""
    per_eye_resolution: Optional[str] = None
    resolution_sources: List[str] = Field(default_factory=list)

    min_gpu_arch_series_for_vrs: Optional[str] = None
    vrs_sources: List[str] = Field(default_factory=list)

    min_room_dimensions: Optional[str] = None
    room_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_headset_selection() -> str:
    return (
        "From the answer, extract the single commercially available VR headset that the answer claims meets ALL of the following:\n"
        "• OLED display technology\n"
        "• Native/built-in eye tracking capability\n"
        "• Console-based platform (connects to a gaming console, specifically PlayStation 5 per constraint)\n"
        "• Supports refresh rate of 120Hz or higher\n"
        "• UE5 compatibility on the headset platform (Unreal Engine 5 can target/deploy to the platform)\n\n"
        "Return the following fields:\n"
        "1. headset_name: exact name of the headset\n"
        "2. release_year_or_date: release year or full date string as stated\n"
        "3. display_tech: the display technology stated for the headset\n"
        "4. eye_tracking_capability: a short phrase confirming native eye tracking\n"
        "5. console_platform: the console the headset connects to (e.g., PlayStation 5)\n"
        "6. refresh_rate_spec: the refresh rate support claim (e.g., 'up to 120Hz')\n"
        "7. ue5_platform_support: a short phrase indicating UE5 supports the headset’s platform (e.g., 'UE5 supports PS5')\n"
        "8. selection_sources: URLs that, collectively, support the headset meeting the above constraints; extract all URLs mentioned for this purpose\n"
        "9. release_sources: URLs specifically supporting the release date/year\n"
        "10. display_sources: URLs specifically supporting OLED display claim\n"
        "11. eye_tracking_sources: URLs specifically supporting native eye tracking claim\n"
        "12. console_sources: URLs specifically supporting console connection (PS5)\n"
        "13. refresh_sources: URLs specifically supporting 120Hz or higher refresh capability\n"
        "14. ue5_sources: URLs specifically supporting UE5 platform compatibility\n\n"
        "Only extract URLs explicitly present in the answer (including markdown links). If a specific set of sources is not given, return an empty list for that field."
    )


def prompt_extract_specs() -> str:
    return (
        "From the answer, extract the three requested outputs for the chosen headset:\n"
        "1) per_eye_resolution: exact per-eye resolution string formatted as 'width x height' (numbers only, e.g., '2000 x 2040')\n"
        "2) resolution_sources: URLs that support the per-eye resolution\n"
        "3) min_gpu_arch_series_for_vrs: the minimum GPU architecture series that supports Variable Rate Shading (VRS) on PC\n"
        "4) vrs_sources: URLs that justify/explain VRS is needed for foveated rendering and that the stated architecture series supports VRS\n"
        "5) min_room_dimensions: minimum physical room dimensions for room-scale VR experiences (e.g., '2.0 m x 2.0 m' or '6.7 ft x 5.0 ft')\n"
        "6) room_sources: URLs supporting the minimum room-scale space requirement\n\n"
        "Only extract URLs explicitly present in the answer. If the answer provides multiple candidates, pick the one that clearly corresponds to the chosen headset and return those details."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "the selected headset"


def _pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    if primary and len(primary) > 0:
        return primary
    return fallback or []


def _has_wxh(res_str: Optional[str]) -> bool:
    if not res_str:
        return False
    # Accept patterns like "2000 x 2040", "2040×2000"
    return bool(re.search(r"\b\d{3,4}\s*(?:x|×)\s*\d{3,4}\b", res_str))


def _contains_architecture_keyword(s: Optional[str]) -> bool:
    if not s:
        return False
    s_low = s.lower()
    # Recognized architecture series keywords for VRS-capable GPUs
    keywords = [
        "turing", "rtx 20", "geforce rtx 20",
        "ampere", "rtx 30", "geforce rtx 30",
        "ada", "ada lovelace", "rtx 40", "geforce rtx 40",
        "rdna2", "rdna 2", "radeon rx 6000",
        "rdna3", "rdna 3", "radeon rx 7000",
        "xe", "intel arc", "arc a", "xe hpg",
    ]
    return any(k in s_low for k in keywords)


def _convert_to_meters(value: float, unit: str) -> Optional[float]:
    u = unit.lower()
    if u in {"m", "meter", "metre", "meters", "metres"}:
        return value
    if u in {"ft", "feet", "foot"}:
        return value * 0.3048
    if u in {"cm", "centimeter", "centimetre", "centimeters", "centimetres"}:
        return value / 100.0
    if u in {"in", "inch", "inches"}:
        return value * 0.0254
    return None


def _parse_dimensions_to_meters(dim_str: Optional[str]) -> Optional[Tuple[float, float]]:
    if not dim_str:
        return None
    text = dim_str.strip()
    # Pattern: "<num><unit> (x|×|by) <num><unit?>"
    pattern = re.compile(
        r"(?i)(?P<a>\d+(?:\.\d+)?)\s*(?P<ua>m|meter|metre|meters|metres|ft|feet|foot|cm|centimeter|centimetre|centimeters|centimetres|in|inch|inches)\s*(?:x|×|by)\s*(?P<b>\d+(?:\.\d+)?)\s*(?P<ub>m|meter|metre|meters|metres|ft|feet|foot|cm|centimeter|centimetre|centimeters|centimetres|in|inch|inches)?"
    )
    m = pattern.search(text)
    if not m:
        return None
    a = float(m.group("a"))
    b = float(m.group("b"))
    ua = m.group("ua")
    ub = m.group("ub") or ua
    am = _convert_to_meters(a, ua)
    bm = _convert_to_meters(b, ub)
    if am is None or bm is None:
        return None
    # Return ordered (max, min)
    return (am, bm) if am >= bm else (bm, am)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_identify_headset(
    evaluator: Evaluator,
    parent_node,
    selection: HeadsetSelection,
) -> None:
    """
    Build and verify the 'Identify_Eligible_Headset' parallel node with critical checks.
    """
    node = evaluator.add_parallel(
        id="Identify_Eligible_Headset",
        desc="Select a headset that satisfies all stated constraints.",
        parent=parent_node,
        critical=True,
    )

    # Source presence check (Critical)
    evaluator.add_custom_node(
        result=bool(selection.selection_sources and len(selection.selection_sources) > 0),
        id="Headset_Selection_Is_Cited",
        desc="Provides at least one official or otherwise reliable source supporting the headset meeting the above requirements (URL or bibliographic citation acceptable).",
        parent=node,
        critical=True,
    )

    name = _safe_name(selection.headset_name)

    # Prepare leaves
    commercially_available = evaluator.add_leaf(
        id="Headset_Is_Commercially_Available",
        desc="Headset identified is commercially available (not a prototype/concept-only device).",
        parent=node,
        critical=True,
    )
    released_before_2025 = evaluator.add_leaf(
        id="Headset_Released_Before_2025",
        desc="Headset release date is before 2025.",
        parent=node,
        critical=True,
    )
    has_oled = evaluator.add_leaf(
        id="Headset_Has_OLED_Display",
        desc="Headset uses OLED display technology.",
        parent=node,
        critical=True,
    )
    has_eye_tracking = evaluator.add_leaf(
        id="Headset_Has_Native_Eye_Tracking",
        desc="Headset has native eye tracking capability.",
        parent=node,
        critical=True,
    )
    console_based_ps5 = evaluator.add_leaf(
        id="Headset_Is_Console_Based_PS5",
        desc="Headset is console-based and connects to the specified gaming console (PlayStation 5, per constraints).",
        parent=node,
        critical=True,
    )
    supports_120hz = evaluator.add_leaf(
        id="Headset_Supports_120Hz_Or_Higher",
        desc="Headset supports refresh rate of at least 120Hz.",
        parent=node,
        critical=True,
    )
    ue5_compatible = evaluator.add_leaf(
        id="Headset_Platform_Is_Compatible_With_UE5_Development",
        desc="The headset/platform is compatible with Unreal Engine 5 development (i.e., UE5 can target/deploy to the platform).",
        parent=node,
        critical=True,
    )

    # Claims and sources
    claims_and_sources = [
        (
            f"{name} is a commercially available consumer VR headset (not just a prototype or concept).",
            _pick_sources(selection.selection_sources, []),
            commercially_available,
            "Confirm the product is a released, purchasable consumer headset using official or reliable product pages.",
        ),
        (
            f"{name} was released before 2025.",
            _pick_sources(selection.release_sources, selection.selection_sources),
            released_before_2025,
            "Verify the release year/date on the cited page; any date before January 1, 2025 qualifies.",
        ),
        (
            f"{name} uses OLED display technology.",
            _pick_sources(selection.display_sources, selection.selection_sources),
            has_oled,
            "Check the stated display technology; accept 'OLED' or 'HDR OLED' as OLED.",
        ),
        (
            f"{name} includes native (built-in) eye tracking.",
            _pick_sources(selection.eye_tracking_sources, selection.selection_sources),
            has_eye_tracking,
            "Confirm that the headset itself has built-in eye tracking (not only optional accessories).",
        ),
        (
            f"{name} is a console-based VR headset that connects to PlayStation 5.",
            _pick_sources(selection.console_sources, selection.selection_sources),
            console_based_ps5,
            "Confirm PS5 connectivity (cable or standard interface) as platform requirement.",
        ),
        (
            f"{name} supports a refresh rate of at least 120 Hz.",
            _pick_sources(selection.refresh_sources, selection.selection_sources),
            supports_120hz,
            "Verify the stated refresh rate capability is 120Hz or higher.",
        ),
        (
            f"Unreal Engine 5 supports deploying to PlayStation 5.",
            _pick_sources(selection.ue5_sources, selection.selection_sources),
            ue5_compatible,
            "Use official Unreal Engine or platform documentation to confirm UE5 target/deploy support for PS5.",
        ),
    ]

    # Execute verifications (in parallel)
    await evaluator.batch_verify(claims_and_sources)


async def verify_specifications(
    evaluator: Evaluator,
    parent_node,
    selection: HeadsetSelection,
    specs: SpecOutputs,
) -> None:
    """
    Build and verify the 'Provide_Required_Specifications' parallel node with three critical subtrees.
    """
    node = evaluator.add_parallel(
        id="Provide_Required_Specifications",
        desc="Provide the three requested outputs for the chosen headset with verifiable sourcing.",
        parent=parent_node,
        critical=True,
    )

    # 1) Per-eye display resolution
    res_node = evaluator.add_parallel(
        id="Per_Eye_Display_Resolution",
        desc="Gives the exact per-eye display resolution as width x height, consistent with official specs.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_wxh(specs.per_eye_resolution),
        id="Resolution_Value_Provided_In_WxH",
        desc="Provides a concrete numeric per-eye resolution in the format width x height.",
        parent=res_node,
        critical=True,
    )

    res_verify = evaluator.add_leaf(
        id="Resolution_Is_Verifiably_Sourced",
        desc="Cites an official or reliable source for the per-eye resolution.",
        parent=res_node,
        critical=True,
    )
    name = _safe_name(selection.headset_name)
    res_claim = f"The per-eye resolution of {name} is {specs.per_eye_resolution}."
    await evaluator.verify(
        claim=res_claim,
        node=res_verify,
        sources=_pick_sources(specs.resolution_sources, selection.selection_sources),
        additional_instruction="Verify the exact numeric per-eye resolution from official spec pages; minor formatting variations are acceptable (e.g., spaces or multiplication symbol).",
    )

    # 2) Minimum GPU architecture for VRS on PC
    gpu_node = evaluator.add_parallel(
        id="Minimum_GPU_Architecture_For_VRS_On_PC",
        desc="States the minimum GPU architecture series required to support Variable Rate Shading (VRS) for foveated rendering when using the headset on PC.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_contains_architecture_keyword(specs.min_gpu_arch_series_for_vrs),
        id="Minimum_GPU_Architecture_Stated",
        desc="Provides a minimum GPU architecture series (not just a general list) that supports VRS.",
        parent=gpu_node,
        critical=True,
    )

    vrs_verify = evaluator.add_leaf(
        id="VRS_Requirement_Is_Justified_And_Sourced",
        desc="Explains/justifies that VRS support is needed for foveated rendering in this context and cites a reliable source for the VRS capability/requirements.",
        parent=gpu_node,
        critical=True,
    )
    arch = specs.min_gpu_arch_series_for_vrs or "the stated minimum GPU architecture series"
    vrs_claim = (
        f"Variable Rate Shading (VRS) is used to implement foveated rendering on PC, and {arch} supports VRS."
    )
    await evaluator.verify(
        claim=vrs_claim,
        node=vrs_verify,
        sources=_pick_sources(specs.vrs_sources, []),
        additional_instruction="Confirm from reliable sources that VRS enables foveated rendering (lower shading rate outside the fovea) and that the named GPU architecture series supports VRS (via DirectX/Vulkan or vendor features).",
    )

    # 3) Minimum room-scale dimensions
    room_node = evaluator.add_parallel(
        id="Minimum_Room_Scale_Dimensions",
        desc="States the minimum physical room dimensions required for room-scale VR experiences.",
        parent=node,
        critical=True,
    )

    dims_m = _parse_dimensions_to_meters(specs.min_room_dimensions)
    meets_min = False
    if dims_m is not None:
        larger, smaller = dims_m
        meets_min = (larger >= 2.0 and smaller >= 1.5)

    evaluator.add_custom_node(
        result=meets_min,
        id="Room_Dimensions_Provided",
        desc="Provides minimum room dimensions and they meet the constraint minimum of at least 2.0m x 1.5m (or equivalent units).",
        parent=room_node,
        critical=True,
    )

    room_verify = evaluator.add_leaf(
        id="Room_Dimensions_Are_Verifiably_Sourced",
        desc="Cites an official or reliable source for the room-scale minimum space requirement.",
        parent=room_node,
        critical=True,
    )
    room_claim = f"The minimum room-scale space required for {_safe_name(selection.headset_name)} is {specs.min_room_dimensions}."
    await evaluator.verify(
        claim=room_claim,
        node=room_verify,
        sources=_pick_sources(specs.room_sources, selection.selection_sources),
        additional_instruction="Verify the stated minimum play area/room-scale dimensions from official headset documentation or reliable platform guidance.",
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
    Evaluate the answer for selecting a qualifying console-based VR headset (PS5) and providing specifications.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Create a critical top-level node to mirror rubric's root
    main = evaluator.add_sequential(
        id="Complete_VR_Development_Solution",
        desc="Identify a commercially available VR headset released before 2025 that meets all stated headset requirements, then provide the three requested specifications with verifiable sourcing.",
        parent=root,
        critical=True,
    )

    # Extract information concurrently
    selection_task = evaluator.extract(
        prompt=prompt_extract_headset_selection(),
        template_class=HeadsetSelection,
        extraction_name="headset_selection",
    )
    specs_task = evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=SpecOutputs,
        extraction_name="spec_outputs",
    )
    selection, specs = await asyncio.gather(selection_task, specs_task)

    # Build verification subtrees
    await verify_identify_headset(evaluator, main, selection)
    await verify_specifications(evaluator, main, selection, specs)

    # Return structured summary
    return evaluator.get_summary()