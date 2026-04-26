import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nintendo_switch_2_specs"
TASK_DESCRIPTION = (
    "I am considering purchasing the Nintendo Switch 2 gaming console and need comprehensive information to make an informed decision. "
    "Please provide the following specifications and details about the Nintendo Switch 2: "
    "(1) When was the console officially revealed by Nintendo? "
    "(2) When were the full specifications and release details announced? "
    "(3) What is the official launch date? "
    "(4) What is the suggested retail price in US dollars? "
    "(5) What is the screen size (diagonal measurement in inches)? "
    "(6) What is the native screen resolution? "
    "(7) Does the screen support HDR, and if so, which version? "
    "(8) What is the maximum screen refresh rate and does it support variable refresh rate (VRR)? "
    "(9) What type of display technology is used and does it have touch capability? "
    "(10) What is the internal storage capacity and technology? "
    "(11) What type of expandable storage is supported? "
    "(12) What are the memory (RAM) specifications including capacity and type? "
    "(13) What is the maximum resolution supported when the console is docked to a TV? "
    "(14) What is the maximum frame rate supported at 1080p resolution in handheld mode? "
    "(15) What graphics processing technology is used, including the provider and key AI feature? "
    "Each specification should be cited with reference URLs from official Nintendo sources or reliable gaming news outlets."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WithSources(BaseModel):
    sources: List[str] = Field(default_factory=list)


class RevealInfo(WithSources):
    date: Optional[str] = None


class FullSpecsInfo(WithSources):
    date: Optional[str] = None


class LaunchInfo(WithSources):
    date: Optional[str] = None


class PriceInfo(WithSources):
    usd_price: Optional[str] = None


class ScreenSizeInfo(WithSources):
    size_inches: Optional[str] = None


class ScreenResolutionInfo(WithSources):
    resolution: Optional[str] = None


class HDRInfo(WithSources):
    hdr_supported: Optional[str] = None
    hdr_version: Optional[str] = None


class RefreshVRRInfo(WithSources):
    max_refresh_rate: Optional[str] = None
    vrr_supported: Optional[str] = None


class DisplayTouchInfo(WithSources):
    display_tech: Optional[str] = None
    touch_capability: Optional[str] = None


class InternalStorageInfo(WithSources):
    capacity: Optional[str] = None
    technology: Optional[str] = None


class ExpandableStorageInfo(WithSources):
    storage_type: Optional[str] = None


class MemoryInfo(WithSources):
    ram_capacity: Optional[str] = None
    ram_type: Optional[str] = None
    memory_bandwidth: Optional[str] = None


class TVModeInfo(WithSources):
    tv_max_resolution: Optional[str] = None


class HandheldFPSInfo(WithSources):
    handheld_1080p_max_fps: Optional[str] = None


class GPUInfo(WithSources):
    provider: Optional[str] = None
    ai_feature: Optional[str] = None


class Switch2Specs(BaseModel):
    official_reveal: Optional[RevealInfo] = None
    full_specs_announcement: Optional[FullSpecsInfo] = None
    launch: Optional[LaunchInfo] = None
    us_price: Optional[PriceInfo] = None
    screen_size: Optional[ScreenSizeInfo] = None
    screen_resolution: Optional[ScreenResolutionInfo] = None
    hdr: Optional[HDRInfo] = None
    refresh_vrr: Optional[RefreshVRRInfo] = None
    display_touch: Optional[DisplayTouchInfo] = None
    internal_storage: Optional[InternalStorageInfo] = None
    expandable_storage: Optional[ExpandableStorageInfo] = None
    memory: Optional[MemoryInfo] = None
    tv_mode: Optional[TVModeInfo] = None
    handheld_fps: Optional[HandheldFPSInfo] = None
    gpu_ai: Optional[GPUInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_switch2_specs() -> str:
    return """
    Extract from the answer text the Nintendo Switch 2 specifications below. For each subsection, extract the exact value(s) as stated in the answer and any associated reference URLs that the answer cites for that subsection.
    Important rules:
    - Only extract values that are explicitly present in the answer text.
    - Extract URLs exactly as written (plain or markdown links). Do not invent or infer URLs.
    - If a value is missing, set it to null. If no URLs are cited for that subsection, return an empty array for 'sources'.
    - Prefer strings for values (do not coerce to numeric types).

    Return a single JSON object with these fields and subfields:

    official_reveal:
      - date: string or null (official reveal date by Nintendo)
      - sources: array of URL strings

    full_specs_announcement:
      - date: string or null (date when full specs/release details were announced)
      - sources: array of URL strings

    launch:
      - date: string or null (official launch date)
      - sources: array of URL strings

    us_price:
      - usd_price: string or null (suggested retail price in USD)
      - sources: array of URL strings

    screen_size:
      - size_inches: string or null (diagonal inches)
      - sources: array of URL strings

    screen_resolution:
      - resolution: string or null (native screen resolution)
      - sources: array of URL strings

    hdr:
      - hdr_supported: string or null (e.g., "Yes", "No", "Supported", "Not supported")
      - hdr_version: string or null (e.g., "HDR10", "Dolby Vision", or "N/A")
      - sources: array of URL strings

    refresh_vrr:
      - max_refresh_rate: string or null (e.g., "60Hz", "120Hz")
      - vrr_supported: string or null (e.g., "Yes", "No")
      - sources: array of URL strings

    display_touch:
      - display_tech: string or null (e.g., "OLED", "LCD", "Mini-LED")
      - touch_capability: string or null (e.g., "Yes, capacitive multi-touch", "No")
      - sources: array of URL strings

    internal_storage:
      - capacity: string or null (e.g., "256GB")
      - technology: string or null (e.g., "UFS 3.1", "eMMC")
      - sources: array of URL strings

    expandable_storage:
      - storage_type: string or null (e.g., "microSD", "microSDXC")
      - sources: array of URL strings

    memory:
      - ram_capacity: string or null (e.g., "12GB")
      - ram_type: string or null (e.g., "LPDDR5")
      - memory_bandwidth: string or null (include if the answer states it; otherwise null)
      - sources: array of URL strings

    tv_mode:
      - tv_max_resolution: string or null (e.g., "4K", "1440p", "1080p")
      - sources: array of URL strings

    handheld_fps:
      - handheld_1080p_max_fps: string or null (maximum frame rate at 1080p in handheld mode)
      - sources: array of URL strings

    gpu_ai:
      - provider: string or null (e.g., "NVIDIA", "AMD")
      - ai_feature: string or null (e.g., "DLSS", "AI upscaling")
      - sources: array of URL strings
    """


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _sanitize_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned: List[str] = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        cleaned.append(s)
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _allowed_sources_instruction() -> str:
    return (
        "Decide if ALL the listed URLs are from acceptable sources: "
        "either (A) official Nintendo properties OR (B) reputable editorial gaming/tech outlets. "
        "Official Nintendo includes domains such as: nintendo.com, support.nintendo.com, nintendo.co.jp, nintendo.co.uk, "
        "and official Nintendo YouTube or social channels. "
        "Reputable editorial outlets include, for example: The Verge, IGN, GameSpot, Eurogamer, Polygon, VGC (Video Games Chronicle), "
        "Digital Foundry, Ars Technica, Engadget, TechRadar, Tom's Hardware, Digital Trends, PCMag, Wired, CNET, Bloomberg, Reuters, AP, BBC, The Guardian, "
        "Game Informer, Kotaku (editorial pieces). "
        "Not acceptable: user-generated wikis (e.g., Fandom), forums (e.g., ResetEra), Reddit, personal blogs with no editorial oversight, "
        "random marketplaces, or content farms. "
        "Judge using domain identities; you do not need to read the pages. "
        "Return Correct only if ALL provided URLs are acceptable as defined above."
    )


def _value_verification_instruction(spec_label: str) -> str:
    return (
        f"Verify that the cited page(s) explicitly support the claim about '{spec_label}' for the Nintendo Switch 2. "
        "Accept reasonable naming variants referring clearly to the same console (e.g., 'Switch 2', 'next‑gen Nintendo Switch', "
        "'Switch successor') if the context clearly indicates the same device. "
        "Prefer explicit announcements or clearly stated specs. If the page only speculates or uses uncertain language, consider it not supported."
    )


async def _add_source_checks(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    human_readable: str,
    urls: List[str],
):
    # Citation present
    present_node = evaluator.add_custom_node(
        result=(len(urls) > 0),
        id=f"{base_id}_Citation_Present",
        desc=f"Provides at least one reference URL for {human_readable}.",
        parent=parent_node,
        critical=True
    )

    # Citation source allowed
    allowed_node = evaluator.add_leaf(
        id=f"{base_id}_Citation_Source_Allowed",
        desc=f"The provided {human_readable} reference URL(s) are from an official Nintendo source OR a reliable gaming/tech news outlet (editorial, not user-generated).",
        parent=parent_node,
        critical=True
    )
    url_list_str = "\n".join(f"- {u}" for u in urls) if urls else "- (none)"
    claim = (
        "All of the following URLs are from official Nintendo or reputable editorial gaming/tech outlets (not user-generated):\n"
        f"{url_list_str}"
    )
    await evaluator.verify(
        claim=claim,
        node=allowed_node,
        sources=None,
        additional_instruction=_allowed_sources_instruction()
    )

    return present_node, allowed_node


async def _verify_value_leaf(
    evaluator: Evaluator,
    parent_node,
    leaf_id: str,
    leaf_desc: str,
    claim: str,
    sources: List[str],
    prereq_nodes: List[Any],
    spec_label: str,
):
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources if sources else None,
        additional_instruction=_value_verification_instruction(spec_label),
        extra_prerequisites=prereq_nodes
    )
    return node


# --------------------------------------------------------------------------- #
# Group verifications (build subtrees matching rubric)                        #
# --------------------------------------------------------------------------- #
async def verify_official_reveal(evaluator: Evaluator, parent, info: Optional[RevealInfo]):
    node = evaluator.add_parallel(
        id="Official_Reveal_Date",
        desc="When the console was officially revealed by Nintendo.",
        parent=parent,
        critical=True
    )
    value = (info.date if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Reveal_Date", "the reveal date", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Reveal_Date_Value_Matches_Constraints",
        "States the official reveal date and it matches the constraints.",
        claim=f"The official reveal date of the Nintendo Switch 2 is '{value}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="official reveal date"
    )


async def verify_full_specs_announcement(evaluator: Evaluator, parent, info: Optional[FullSpecsInfo]):
    node = evaluator.add_parallel(
        id="Full_Specs_Announcement_Date",
        desc="When full specifications and release details were announced.",
        parent=parent,
        critical=True
    )
    value = (info.date if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Full_Specs_Date", "the full specs/release details announcement date", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Full_Specs_Date_Value_Matches_Constraints",
        "States the full specs/release details announcement date and it matches the constraints.",
        claim=f"The date when Nintendo announced full specifications and release details for the Nintendo Switch 2 is '{value}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="full specifications/release details announcement date"
    )


async def verify_launch_date(evaluator: Evaluator, parent, info: Optional[LaunchInfo]):
    node = evaluator.add_parallel(
        id="Launch_Date",
        desc="Official launch date.",
        parent=parent,
        critical=True
    )
    value = (info.date if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Launch_Date", "the launch date", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Launch_Date_Value_Matches_Constraints",
        "States the official launch date and it matches the constraints.",
        claim=f"The official launch date of the Nintendo Switch 2 is '{value}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="official launch date"
    )


async def verify_us_price(evaluator: Evaluator, parent, info: Optional[PriceInfo]):
    node = evaluator.add_parallel(
        id="US_Retail_Price",
        desc="Suggested retail price in USD.",
        parent=parent,
        critical=True
    )
    value = (info.usd_price if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Price", "the suggested retail price (USD)", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Price_Value_Matches_Constraints",
        "States the suggested retail price (USD) and it matches the constraints.",
        claim=f"The suggested retail price (USD) for the Nintendo Switch 2 is '{value}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="suggested retail price (USD)"
    )


async def verify_screen_size(evaluator: Evaluator, parent, info: Optional[ScreenSizeInfo]):
    node = evaluator.add_parallel(
        id="Screen_Size",
        desc="Screen size (diagonal, inches).",
        parent=parent,
        critical=True
    )
    value = (info.size_inches if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Screen_Size", "the screen size", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Screen_Size_Value_Matches_Constraints",
        "States the screen size (diagonal inches) and it matches the constraints.",
        claim=f"The Nintendo Switch 2 screen size (diagonal) is '{value}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="screen size"
    )


async def verify_screen_resolution(evaluator: Evaluator, parent, info: Optional[ScreenResolutionInfo]):
    node = evaluator.add_parallel(
        id="Screen_Resolution",
        desc="Native screen resolution.",
        parent=parent,
        critical=True
    )
    value = (info.resolution if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Screen_Resolution", "the native screen resolution", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Screen_Resolution_Value_Matches_Constraints",
        "States the native screen resolution and it matches the constraints.",
        claim=f"The native handheld screen resolution of the Nintendo Switch 2 is '{value}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="native screen resolution"
    )


async def verify_hdr(evaluator: Evaluator, parent, info: Optional[HDRInfo]):
    node = evaluator.add_parallel(
        id="Screen_HDR_Support",
        desc="HDR support and (if supported) HDR version.",
        parent=parent,
        critical=True
    )
    supported = (info.hdr_supported if info else None) or ""
    version = (info.hdr_version if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "HDR", "HDR support/version", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "HDR_Support_And_Version_Matches_Constraints",
        "States whether HDR is supported and the HDR version (if supported); matches the constraints.",
        claim=f"The Nintendo Switch 2 HDR support is '{supported}', and the HDR version is '{version}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="HDR support and version"
    )


async def verify_refresh_vrr(evaluator: Evaluator, parent, info: Optional[RefreshVRRInfo]):
    node = evaluator.add_parallel(
        id="Screen_Refresh_Rate_and_VRR",
        desc="Maximum screen refresh rate and VRR support.",
        parent=parent,
        critical=True
    )
    max_rr = (info.max_refresh_rate if info else None) or ""
    vrr = (info.vrr_supported if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Refresh_VRR", "refresh rate and/or VRR support", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Max_Refresh_Rate_Matches_Constraints",
        "States the maximum screen refresh rate and it matches the constraints.",
        claim=f"The maximum handheld screen refresh rate of the Nintendo Switch 2 is '{max_rr}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="maximum screen refresh rate"
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "VRR_Support_Matches_Constraints",
        "States whether VRR is supported and it matches the constraints.",
        claim=f"The Nintendo Switch 2 VRR (variable refresh rate) support is '{vrr}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="VRR support"
    )


async def verify_display_touch(evaluator: Evaluator, parent, info: Optional[DisplayTouchInfo]):
    node = evaluator.add_parallel(
        id="Display_Technology_and_Touch",
        desc="Display technology and touch capability.",
        parent=parent,
        critical=True
    )
    tech = (info.display_tech if info else None) or ""
    touch = (info.touch_capability if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Display_Touch", "display technology and/or touch capability", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Display_Technology_Matches_Constraints",
        "States the display technology and it matches the constraints.",
        claim=f"The Nintendo Switch 2 display technology is '{tech}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="display technology"
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Touch_Capability_Matches_Constraints",
        "States whether touch capability is supported and it matches the constraints.",
        claim=f"The Nintendo Switch 2 touch capability is '{touch}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="touch capability"
    )


async def verify_internal_storage(evaluator: Evaluator, parent, info: Optional[InternalStorageInfo]):
    node = evaluator.add_parallel(
        id="Internal_Storage_Capacity_and_Tech",
        desc="Internal storage capacity and storage technology.",
        parent=parent,
        critical=True
    )
    cap = (info.capacity if info else None) or ""
    tech = (info.technology if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Storage", "internal storage capacity/technology", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Internal_Storage_Capacity_Matches_Constraints",
        "States internal storage capacity and it matches the constraints.",
        claim=f"The Nintendo Switch 2 internal storage capacity is '{cap}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="internal storage capacity"
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Internal_Storage_Technology_Matches_Constraints",
        "States internal storage technology and it matches the constraints.",
        claim=f"The Nintendo Switch 2 internal storage technology is '{tech}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="internal storage technology"
    )


async def verify_expandable_storage(evaluator: Evaluator, parent, info: Optional[ExpandableStorageInfo]):
    node = evaluator.add_parallel(
        id="Expandable_Storage_Type",
        desc="Type of expandable storage supported.",
        parent=parent,
        critical=True
    )
    stype = (info.storage_type if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Expandable_Storage", "expandable storage type", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Expandable_Storage_Type_Matches_Constraints",
        "States the type of expandable storage supported and it matches the constraints.",
        claim=f"The Nintendo Switch 2 supports expandable storage type '{stype}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="expandable storage type"
    )


async def verify_memory(evaluator: Evaluator, parent, info: Optional[MemoryInfo]):
    node = evaluator.add_parallel(
        id="Memory_Specifications",
        desc="Memory (RAM) specifications, including required attributes from constraints.",
        parent=parent,
        critical=True
    )
    cap = (info.ram_capacity if info else None) or ""
    rtype = (info.ram_type if info else None) or ""
    bw = (info.memory_bandwidth if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Memory", "memory specifications", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "RAM_Capacity_Matches_Constraints",
        "States RAM capacity and it matches the constraints.",
        claim=f"The Nintendo Switch 2 RAM capacity is '{cap}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="RAM capacity"
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "RAM_Type_Matches_Constraints",
        "States RAM type and it matches the constraints.",
        claim=f"The Nintendo Switch 2 RAM type is '{rtype}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="RAM type"
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Memory_Bandwidth_Matches_Constraints",
        "States memory bandwidth and it matches the constraints.",
        claim=f"The Nintendo Switch 2 memory bandwidth is '{bw}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="memory bandwidth"
    )


async def verify_tv_mode(evaluator: Evaluator, parent, info: Optional[TVModeInfo]):
    node = evaluator.add_parallel(
        id="TV_Mode_Max_Resolution",
        desc="Maximum resolution supported when docked to a TV.",
        parent=parent,
        critical=True
    )
    res = (info.tv_max_resolution if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "TV_Mode", "max docked/TV mode resolution", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "TV_Mode_Max_Resolution_Matches_Constraints",
        "States the maximum docked/TV mode resolution and it matches the constraints.",
        claim=f"The maximum TV/docked resolution supported by the Nintendo Switch 2 is '{res}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="TV/docked maximum resolution"
    )


async def verify_handheld_fps(evaluator: Evaluator, parent, info: Optional[HandheldFPSInfo]):
    node = evaluator.add_parallel(
        id="Handheld_1080p_Max_Frame_Rate",
        desc="Maximum frame rate supported at 1080p in handheld mode.",
        parent=parent,
        critical=True
    )
    fps = (info.handheld_1080p_max_fps if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "Handheld_FPS", "handheld 1080p maximum frame rate", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Handheld_1080p_Max_FPS_Matches_Constraints",
        "States the maximum handheld 1080p frame rate and it matches the constraints.",
        claim=f"The maximum handheld frame rate at 1080p for the Nintendo Switch 2 is '{fps}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="handheld 1080p maximum frame rate"
    )


async def verify_gpu_ai(evaluator: Evaluator, parent, info: Optional[GPUInfo]):
    node = evaluator.add_parallel(
        id="GPU_Technology_Provider_and_AI_Feature",
        desc="Graphics processing technology, including provider and key AI feature.",
        parent=parent,
        critical=True
    )
    provider = (info.provider if info else None) or ""
    aifeat = (info.ai_feature if info else None) or ""
    sources = _sanitize_sources(info.sources if info else [])
    present_node, allowed_node = await _add_source_checks(
        evaluator, node, "GPU_AI", "GPU/provider and AI feature", sources
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "GPU_Technology_Provider_Matches_Constraints",
        "States the GPU technology provider and it matches the constraints.",
        claim=f"The Nintendo Switch 2 GPU technology provider is '{provider}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="GPU technology provider"
    )
    await _verify_value_leaf(
        evaluator,
        node,
        "Key_AI_Feature_Matches_Constraints",
        "States the key AI feature and it matches the constraints.",
        claim=f"The key AI graphics feature for the Nintendo Switch 2 is '{aifeat}'.",
        sources=sources,
        prereq_nodes=[present_node, allowed_node],
        spec_label="key AI feature"
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_switch2_specs(),
        template_class=Switch2Specs,
        extraction_name="switch2_specs_extraction"
    )

    # Top-level critical node
    top = evaluator.add_parallel(
        id="Nintendo_Switch_2_Specifications",
        desc="Provide all requested Nintendo Switch 2 specifications. For each specification: (1) the value matches the provided constraints and (2) at least one supporting reference URL is provided from an official Nintendo source or a reliable gaming/tech news outlet.",
        parent=root,
        critical=True
    )

    # Build verification subtrees per rubric
    await verify_official_reveal(evaluator, top, extracted.official_reveal)
    await verify_full_specs_announcement(evaluator, top, extracted.full_specs_announcement)
    await verify_launch_date(evaluator, top, extracted.launch)
    await verify_us_price(evaluator, top, extracted.us_price)
    await verify_screen_size(evaluator, top, extracted.screen_size)
    await verify_screen_resolution(evaluator, top, extracted.screen_resolution)
    await verify_hdr(evaluator, top, extracted.hdr)
    await verify_refresh_vrr(evaluator, top, extracted.refresh_vrr)
    await verify_display_touch(evaluator, top, extracted.display_touch)
    await verify_internal_storage(evaluator, top, extracted.internal_storage)
    await verify_expandable_storage(evaluator, top, extracted.expandable_storage)
    await verify_memory(evaluator, top, extracted.memory)
    await verify_tv_mode(evaluator, top, extracted.tv_mode)
    await verify_handheld_fps(evaluator, top, extracted.handheld_fps)
    await verify_gpu_ai(evaluator, top, extracted.gpu_ai)

    return evaluator.get_summary()