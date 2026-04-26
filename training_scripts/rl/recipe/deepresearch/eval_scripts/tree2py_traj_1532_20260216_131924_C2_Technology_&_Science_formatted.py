import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nswitch2_specs"
TASK_DESCRIPTION = """Provide the key hardware specifications for the Nintendo Switch 2 console across three categories: display, storage/memory, and controllers. For each category, include the following details with supporting evidence from official Nintendo sources:

1. Display Specifications: Report the screen size (in inches) and native resolution, and describe the HDR and variable refresh rate capabilities.

2. Storage and Memory: Report the internal storage capacity (in GB) and storage technology type, as well as the system memory (RAM) capacity (in GB) and RAM type.

3. Controller Features: Describe the functionality of the C Button on the Joy-Con 2 controllers, the mouse control capability, and the attachment mechanism used to connect the Joy-Con 2 controllers to the console.

For each category, provide at least one reference URL from an official Nintendo website (nintendo.com domain) that supports the specifications you report.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class DisplaySpec(BaseModel):
    screen_size_inches: Optional[str] = None
    native_resolution: Optional[str] = None
    hdr_capabilities: Optional[str] = None
    vrr_capabilities: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class StorageMemorySpec(BaseModel):
    internal_storage_capacity_gb: Optional[str] = None
    internal_storage_type: Optional[str] = None
    ram_capacity_gb: Optional[str] = None
    ram_type: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ControllerSpec(BaseModel):
    c_button_functionality: Optional[str] = None
    mouse_control_capability: Optional[str] = None
    attachment_mechanism: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Switch2SpecsExtraction(BaseModel):
    display: Optional[DisplaySpec] = None
    storage_memory: Optional[StorageMemorySpec] = None
    controllers: Optional[ControllerSpec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_switch2_specs() -> str:
    return """
Extract the Nintendo Switch 2 hardware information the answer provides, organized into three categories. Only extract what is explicitly stated in the answer text. Do not infer.

Return JSON with this structure:
{
  "display": {
    "screen_size_inches": string | null,
    "native_resolution": string | null,
    "hdr_capabilities": string | null,
    "vrr_capabilities": string | null,
    "urls": string[]   // All URLs cited in the answer that are meant to support display specs
  },
  "storage_memory": {
    "internal_storage_capacity_gb": string | null,
    "internal_storage_type": string | null,
    "ram_capacity_gb": string | null,
    "ram_type": string | null,
    "urls": string[]   // All URLs cited in the answer that are meant to support storage/memory specs
  },
  "controllers": {
    "c_button_functionality": string | null,
    "mouse_control_capability": string | null,
    "attachment_mechanism": string | null,
    "urls": string[]   // All URLs cited in the answer that are meant to support controller features
  }
}

Rules:
- Keep values as strings exactly as presented (e.g., “7.0-inch”, “1280x720”, “HDR10”, “VRR up to 120 Hz”, “UFS 3.1”, “16 GB LPDDR5”, etc.).
- For any missing field, set it to null.
- For each category, collect ALL URLs explicitly mentioned for that category (including both official and third-party), in the order they appear.
- Do not invent URLs. If no URLs are provided for a category, return an empty array for that category’s "urls".
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_official_nintendo_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        # Accept subdomains of nintendo.com and nintendo.com itself
        return host == "nintendo.com" or host.endswith(".nintendo.com")
    except Exception:
        return False


def filter_official(urls: List[str]) -> List[str]:
    return [u for u in urls if is_official_nintendo_url(u)]


def nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_display_specs(evaluator: Evaluator, parent_node, extracted: Switch2SpecsExtraction) -> None:
    """
    Build and verify the display specifications subtree.
    """
    disp = extracted.display or DisplaySpec()
    disp_node = evaluator.add_parallel(
        id="Display_Specifications",
        desc="Verify display-related specifications are provided",
        parent=parent_node,
        critical=True
    )

    official_urls = filter_official(disp.urls)
    all_urls = disp.urls if disp.urls else []
    urls_for_claims = official_urls if official_urls else all_urls

    # Group: Screen Size & Resolution
    sr_group = evaluator.add_sequential(
        id="Screen_Size_and_Resolution",
        desc="Verify that the screen size (in inches) and native resolution are provided",
        parent=disp_node,
        critical=True
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=nonempty(disp.screen_size_inches),
        id="display_screen_size_provided",
        desc="Screen size is provided in the answer",
        parent=sr_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(disp.native_resolution),
        id="display_resolution_provided",
        desc="Native resolution is provided in the answer",
        parent=sr_group,
        critical=True
    )

    # Support checks (use official URLs if present; else fallback to provided URLs)
    size_leaf = evaluator.add_leaf(
        id="display_screen_size_supported",
        desc="The stated screen size is supported by cited sources",
        parent=sr_group,
        critical=True
    )
    size_claim = f"The Nintendo Switch 2 has a screen size of {disp.screen_size_inches} (inches)."
    await evaluator.verify(
        claim=size_claim,
        node=size_leaf,
        sources=urls_for_claims,
        additional_instruction=(
            "Verify that the official page explicitly mentions the screen size. "
            "Accept reasonable formatting differences (e.g., “7 inch” vs “7.0-inch”). "
            "If multiple sizes appear for different modes, ensure the claim matches the handheld screen size stated in the answer."
        )
    )

    res_leaf = evaluator.add_leaf(
        id="display_resolution_supported",
        desc="The stated native resolution is supported by cited sources",
        parent=sr_group,
        critical=True
    )
    res_claim = f"The Nintendo Switch 2 display has a native resolution of {disp.native_resolution}."
    await evaluator.verify(
        claim=res_claim,
        node=res_leaf,
        sources=urls_for_claims,
        additional_instruction=(
            "Verify that the page states this native resolution for the device display. "
            "Allow simple formatting differences (e.g., ‘1280 x 720’ vs ‘1280x720’)."
        )
    )

    # Group: Display Features (HDR & VRR)
    feat_group = evaluator.add_sequential(
        id="Display_Features",
        desc="Verify that HDR support and variable refresh rate capabilities are mentioned",
        parent=disp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty(disp.hdr_capabilities),
        id="display_hdr_provided",
        desc="HDR support/capabilities are described in the answer",
        parent=feat_group,
        critical=True
    )
    hdr_leaf = evaluator.add_leaf(
        id="display_hdr_supported",
        desc="The stated HDR capability is supported by cited sources",
        parent=feat_group,
        critical=True
    )
    hdr_claim = f"The Nintendo Switch 2 display supports HDR as described: {disp.hdr_capabilities}."
    await evaluator.verify(
        claim=hdr_claim,
        node=hdr_leaf,
        sources=urls_for_claims,
        additional_instruction=(
            "Verify that the page mentions HDR for the device display, consistent with the answer’s description. "
            "Accept minor paraphrases (e.g., 'HDR10 support') as long as meaning matches."
        )
    )

    evaluator.add_custom_node(
        result=nonempty(disp.vrr_capabilities),
        id="display_vrr_provided",
        desc="Variable refresh rate (VRR) capability is described in the answer",
        parent=feat_group,
        critical=True
    )
    vrr_leaf = evaluator.add_leaf(
        id="display_vrr_supported",
        desc="The stated VRR capability is supported by cited sources",
        parent=feat_group,
        critical=True
    )
    vrr_claim = f"The Nintendo Switch 2 display supports variable refresh rate as described: {disp.vrr_capabilities}."
    await evaluator.verify(
        claim=vrr_claim,
        node=vrr_leaf,
        sources=urls_for_claims,
        additional_instruction=(
            "Verify that the page mentions VRR (variable refresh rate) and the capability described by the answer. "
            "Allow minor paraphrases (e.g., 'variable refresh rate' vs 'VRR')."
        )
    )

    # Group: Display Source URL (official URL presence & relevance)
    src_group = evaluator.add_parallel(
        id="Display_Source_URL",
        desc="Verify that display specifications are sourced from an official Nintendo URL",
        parent=disp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(official_urls) >= 1,
        id="display_official_url_present",
        desc="At least one official Nintendo URL (nintendo.com domain) is provided for display specs",
        parent=src_group,
        critical=True
    )
    src_rel_leaf = evaluator.add_leaf(
        id="display_official_url_relevance",
        desc="The official Nintendo URL is relevant to display specifications (e.g., screen size, resolution, HDR or VRR)",
        parent=src_group,
        critical=True
    )
    rel_claim = "This official Nintendo page describes Nintendo Switch 2 display specifications such as screen size, resolution, HDR, or VRR."
    await evaluator.verify(
        claim=rel_claim,
        node=src_rel_leaf,
        sources=official_urls if official_urls else all_urls,
        additional_instruction=(
            "Judge relevance strictly: the page should explicitly mention at least one of screen size, resolution, HDR, or VRR for the Nintendo Switch 2."
        )
    )


async def verify_storage_memory_specs(evaluator: Evaluator, parent_node, extracted: Switch2SpecsExtraction) -> None:
    """
    Build and verify the storage and memory specifications subtree.
    """
    sm = extracted.storage_memory or StorageMemorySpec()
    sm_node = evaluator.add_parallel(
        id="Storage_and_Memory",
        desc="Verify storage and memory specifications are provided",
        parent=parent_node,
        critical=True
    )

    official_urls = filter_official(sm.urls)
    all_urls = sm.urls if sm.urls else []
    urls_for_claims = official_urls if official_urls else all_urls

    # Group: Internal Storage
    int_store = evaluator.add_sequential(
        id="Internal_Storage",
        desc="Verify that the internal storage capacity (in GB) and storage type are provided",
        parent=sm_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(sm.internal_storage_capacity_gb),
        id="storage_capacity_provided",
        desc="Internal storage capacity (GB) is provided in the answer",
        parent=int_store,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(sm.internal_storage_type),
        id="storage_type_provided",
        desc="Internal storage technology/type is provided in the answer",
        parent=int_store,
        critical=True
    )
    cap_leaf = evaluator.add_leaf(
        id="storage_capacity_supported",
        desc="The stated internal storage capacity is supported by cited sources",
        parent=int_store,
        critical=True
    )
    cap_claim = f"The Nintendo Switch 2 internal storage capacity is {sm.internal_storage_capacity_gb}."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=urls_for_claims,
        additional_instruction="Verify that the page states the internal storage capacity (in GB) matching the answer."
    )

    stype_leaf = evaluator.add_leaf(
        id="storage_type_supported",
        desc="The stated internal storage type/technology is supported by cited sources",
        parent=int_store,
        critical=True
    )
    stype_claim = f"The Nintendo Switch 2 uses {sm.internal_storage_type} for its internal storage."
    await evaluator.verify(
        claim=stype_claim,
        node=stype_leaf,
        sources=urls_for_claims,
        additional_instruction="Verify that the page names this storage technology for the device’s internal storage."
    )

    # Group: System Memory (RAM)
    ram_group = evaluator.add_sequential(
        id="System_Memory",
        desc="Verify that the RAM capacity (in GB) and RAM type are provided",
        parent=sm_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(sm.ram_capacity_gb),
        id="ram_capacity_provided",
        desc="System memory (RAM) capacity (GB) is provided in the answer",
        parent=ram_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(sm.ram_type),
        id="ram_type_provided",
        desc="RAM type is provided in the answer",
        parent=ram_group,
        critical=True
    )
    rcap_leaf = evaluator.add_leaf(
        id="ram_capacity_supported",
        desc="The stated RAM capacity is supported by cited sources",
        parent=ram_group,
        critical=True
    )
    rcap_claim = f"The Nintendo Switch 2 has {sm.ram_capacity_gb} of system memory (RAM)."
    await evaluator.verify(
        claim=rcap_claim,
        node=rcap_leaf,
        sources=urls_for_claims,
        additional_instruction="Verify that the page states the RAM capacity matching the answer."
    )
    rtype_leaf = evaluator.add_leaf(
        id="ram_type_supported",
        desc="The stated RAM type is supported by cited sources",
        parent=ram_group,
        critical=True
    )
    rtype_claim = f"The Nintendo Switch 2 uses {sm.ram_type} RAM."
    await evaluator.verify(
        claim=rtype_claim,
        node=rtype_leaf,
        sources=urls_for_claims,
        additional_instruction="Verify that the page names this RAM type for the device."
    )

    # Group: Source URL presence/relevance
    src_group = evaluator.add_parallel(
        id="Storage_Memory_Source_URL",
        desc="Verify that storage and memory specifications are sourced from an official Nintendo URL",
        parent=sm_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(official_urls) >= 1,
        id="storage_official_url_present",
        desc="At least one official Nintendo URL (nintendo.com domain) is provided for storage/memory specs",
        parent=src_group,
        critical=True
    )
    src_rel_leaf = evaluator.add_leaf(
        id="storage_official_url_relevance",
        desc="The official Nintendo URL is relevant to storage and/or memory specifications",
        parent=src_group,
        critical=True
    )
    rel_claim = "This official Nintendo page describes Nintendo Switch 2 storage or memory specifications."
    await evaluator.verify(
        claim=rel_claim,
        node=src_rel_leaf,
        sources=official_urls if official_urls else all_urls,
        additional_instruction="Judge relevance strictly: the page should explicitly mention storage capacity/type and/or RAM capacity/type for the Nintendo Switch 2."
    )


async def verify_controller_specs(evaluator: Evaluator, parent_node, extracted: Switch2SpecsExtraction) -> None:
    """
    Build and verify the controller features subtree.
    """
    ctl = extracted.controllers or ControllerSpec()
    ctl_node = evaluator.add_parallel(
        id="Controller_Features",
        desc="Verify controller-related features are provided",
        parent=parent_node,
        critical=True
    )

    official_urls = filter_official(ctl.urls)
    all_urls = ctl.urls if ctl.urls else []
    urls_for_claims = official_urls if official_urls else all_urls

    # Group: Joy-Con 2 Features (C button & mouse control)
    joy_group = evaluator.add_sequential(
        id="Joy_Con_2_Features",
        desc="Verify that the C Button functionality and mouse control capability are described",
        parent=ctl_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(ctl.c_button_functionality),
        id="c_button_provided",
        desc="C Button functionality is described in the answer",
        parent=joy_group,
        critical=True
    )
    cbtn_leaf = evaluator.add_leaf(
        id="c_button_supported",
        desc="The stated C Button functionality is supported by cited sources",
        parent=joy_group,
        critical=True
    )
    cbtn_claim = f"The Joy-Con 2 includes a C Button with functionality described as: {ctl.c_button_functionality}."
    await evaluator.verify(
        claim=cbtn_claim,
        node=cbtn_leaf,
        sources=urls_for_claims,
        additional_instruction=(
            "Verify the page mentions a 'C Button' on Joy-Con 2 and that its described functionality matches the answer (allow minor paraphrase)."
        )
    )

    evaluator.add_custom_node(
        result=nonempty(ctl.mouse_control_capability),
        id="mouse_control_provided",
        desc="Mouse control capability is described in the answer",
        parent=joy_group,
        critical=True
    )
    mouse_leaf = evaluator.add_leaf(
        id="mouse_control_supported",
        desc="The stated mouse control capability is supported by cited sources",
        parent=joy_group,
        critical=True
    )
    mouse_claim = f"The controllers provide mouse control capability as described: {ctl.mouse_control_capability}."
    await evaluator.verify(
        claim=mouse_claim,
        node=mouse_leaf,
        sources=urls_for_claims,
        additional_instruction=(
            "Verify that the page mentions any mouse control capability for Joy-Con 2 or the controller setup, consistent with the answer’s wording."
        )
    )

    # Group: Attachment Method
    attach_group = evaluator.add_sequential(
        id="Attachment_Method",
        desc="Verify that the attachment mechanism used to connect Joy-Con 2 controllers to the console is described",
        parent=ctl_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(ctl.attachment_mechanism),
        id="attachment_mechanism_provided",
        desc="Attachment mechanism is described in the answer",
        parent=attach_group,
        critical=True
    )
    attach_leaf = evaluator.add_leaf(
        id="attachment_mechanism_supported",
        desc="The stated attachment mechanism is supported by cited sources",
        parent=attach_group,
        critical=True
    )
    attach_claim = f"The Joy-Con 2 attach to the console using this mechanism: {ctl.attachment_mechanism}."
    await evaluator.verify(
        claim=attach_claim,
        node=attach_leaf,
        sources=urls_for_claims,
        additional_instruction=(
            "Verify that the official page (or provided sources) describe the attachment method (e.g., rail system, magnetic, etc.) matching the answer."
        )
    )

    # Group: Controller Source URL (official presence & relevance)
    src_group = evaluator.add_parallel(
        id="Controller_Source_URL",
        desc="Verify that controller features are sourced from an official Nintendo URL",
        parent=ctl_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(official_urls) >= 1,
        id="controller_official_url_present",
        desc="At least one official Nintendo URL (nintendo.com domain) is provided for controller features",
        parent=src_group,
        critical=True
    )
    src_rel_leaf = evaluator.add_leaf(
        id="controller_official_url_relevance",
        desc="The official Nintendo URL is relevant to controller features (C Button, mouse control, or attachment)",
        parent=src_group,
        critical=True
    )
    rel_claim = "This official Nintendo page describes Joy-Con 2 features such as the C Button, mouse control capability, or the attachment mechanism."
    await evaluator.verify(
        claim=rel_claim,
        node=src_rel_leaf,
        sources=official_urls if official_urls else all_urls,
        additional_instruction="Judge relevance strictly: the page should explicitly mention at least one of these controller features."
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
    """
    Evaluate an answer for Nintendo Switch 2 hardware specs (display, storage/memory, controllers)
    with official Nintendo source requirements.
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_switch2_specs(),
        template_class=Switch2SpecsExtraction,
        extraction_name="switch2_specs_extraction"
    )

    # Add a top-level critical node (as per rubric)
    main_node = evaluator.add_parallel(
        id="Nintendo_Switch_2_Hardware_Specifications",
        desc="Verify that the solution provides accurate hardware specifications for the Nintendo Switch 2 as officially announced by Nintendo",
        parent=root,
        critical=True
    )

    # Record custom info about official URL availability for transparency
    display_urls = (extracted.display.urls if extracted.display and extracted.display.urls else [])
    storage_urls = (extracted.storage_memory.urls if extracted.storage_memory and extracted.storage_memory.urls else [])
    controller_urls = (extracted.controllers.urls if extracted.controllers and extracted.controllers.urls else [])

    evaluator.add_custom_info(
        info={
            "display_urls_total": len(display_urls),
            "display_urls_official": len(filter_official(display_urls)),
            "storage_urls_total": len(storage_urls),
            "storage_urls_official": len(filter_official(storage_urls)),
            "controller_urls_total": len(controller_urls),
            "controller_urls_official": len(filter_official(controller_urls)),
        },
        info_type="url_statistics",
        info_name="category_url_stats"
    )

    # Build category subtrees (all critical)
    await verify_display_specs(evaluator, main_node, extracted)
    await verify_storage_memory_specs(evaluator, main_node, extracted)
    await verify_controller_specs(evaluator, main_node, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()