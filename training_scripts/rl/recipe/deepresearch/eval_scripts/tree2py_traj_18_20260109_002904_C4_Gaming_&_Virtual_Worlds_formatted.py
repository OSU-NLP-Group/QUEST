import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ---------------------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------------------
TASK_ID = "vr_headset_selection_2025"
TASK_DESCRIPTION = (
    "As a PC gamer interested in entering the world of virtual reality in 2025, I'm looking for a VR headset that offers "
    "flexibility and good value. I need a standalone VR headset that can also connect to my gaming PC for more demanding VR games, "
    "with a budget of up to $700. The headset must provide at least 2000×2000 pixels per eye resolution to ensure a clear visual experience. "
    "Which VR headset(s) released since 2023 meet these requirements? Please provide the headset name(s), key specifications "
    "(including resolution, refresh rate, field of view, and display type), price, and reference URL(s) to verify the information."
)


# ---------------------------------------------------------------------------------------
# Extraction models
# ---------------------------------------------------------------------------------------
class VRHeadsetItem(BaseModel):
    name: Optional[str] = None
    resolution_per_eye: Optional[str] = None  # e.g., "2064×2208 per eye", "2160 x 2160 per-eye"
    refresh_rate: Optional[str] = None        # e.g., "90 Hz", "90–120 Hz", "up to 120Hz"
    field_of_view: Optional[str] = None       # e.g., "110°", "95° horizontal", "120° diagonal"
    display_type: Optional[str] = None        # e.g., "LCD", "OLED", "micro-OLED", "QLED"
    price_usd: Optional[str] = None           # e.g., "$499", "US$699", "Starts at $299"
    release_date: Optional[str] = None        # free text, may include month/year; extractor can also include just a year
    release_year: Optional[str] = None        # e.g., "2023", "2024", "2025"
    standalone_note: Optional[str] = None     # free text note about standalone capability (if explicitly stated)
    pc_connectivity_note: Optional[str] = None  # free text note about PC connectivity (wired/wireless/Link/Virtual Desktop)
    urls: List[str] = Field(default_factory=list)  # reference URLs mentioned in the answer for this headset


class VRHeadsetList(BaseModel):
    headsets: List[VRHeadsetItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------------------
def prompt_extract_headsets() -> str:
    return """
    You are extracting structured information about VR headset recommendations from the provided answer text.

    Extract up to 5 VR headset entries that the answer proposes or discusses for the user's request.
    For each headset, extract the following fields if present in the answer (otherwise use null or empty list):
    - name: the headset name or model
    - resolution_per_eye: the per-eye resolution as written (e.g., "2064×2208 per eye", "2160 x 2160 per-eye"); if only a combined/both-eyes resolution is provided, still extract it verbatim
    - refresh_rate: the refresh rate as written (e.g., "90 Hz", "up to 120 Hz", "90–120 Hz")
    - field_of_view: the field-of-view as written, including degree symbol or qualifiers if any (e.g., "110°", "96° horizontal", "120° diagonal")
    - display_type: panel type as written (e.g., LCD, OLED, micro-OLED, QLED, fast-switch LCD)
    - price_usd: the base model price in USD as written in the answer (e.g., "$499", "US$699", "starts at $299")
    - release_date: free-form release date text as written in the answer (e.g., "released October 2023", "launched in 2024")
    - release_year: a 4-digit year if the answer states one (e.g., "2023", "2024"); otherwise null
    - standalone_note: any text snippet indicating it is standalone / all-in-one (if explicitly mentioned in the answer)
    - pc_connectivity_note: any text snippet indicating PC connectivity (Link cable, Air Link, Wi-Fi streaming, Virtual Desktop, DP Alt Mode, etc.) if explicitly mentioned
    - urls: an array of all URLs provided in the answer that are relevant to this headset; extract actual URLs only (including those inside markdown links)
    
    IMPORTANT:
    - Extract only what is present in the answer. Do not invent new values.
    - For URLs, return only valid-looking URLs (add http:// if protocol is missing).
    - It's fine if some fields are missing (null). Do not infer missing fields.

    Return a JSON object with a single field:
    {
      "headsets": [ ... up to 5 headset objects as described ... ]
    }
    """


# ---------------------------------------------------------------------------------------
# Helper: Build claims and additional instructions
# ---------------------------------------------------------------------------------------
def claim_standalone(name: str) -> str:
    return f"{name} is a standalone/all-in-one VR headset that can run VR apps by itself without needing a PC or console for basic operation."

def addins_standalone() -> str:
    return (
        "Verify the headset is capable of standalone operation (a built-in processor/SoC and OS). "
        "Phrases like 'standalone', 'all-in-one', 'no PC required', 'runs apps on-device', 'XR2' (for Meta Quest devices) count."
    )

def claim_pc_connectivity(name: str) -> str:
    return f"{name} supports connecting to a Windows PC for PC VR gaming (either via a wired link or wireless streaming)."

def addins_pc_connectivity() -> str:
    return (
        "Look for official or supported PC VR connectivity: USB-C Link cable, DisplayPort/DP Alt Mode, SteamVR via Wi‑Fi/Air Link, "
        "or recognized solutions like 'Virtual Desktop'. Wireless streaming also qualifies as PC connectivity."
    )

def claim_release_2023_plus(name: str) -> str:
    return f"{name} was released in 2023 or later."

def addins_release_timeframe() -> str:
    return (
        "Confirm the consumer release (availability) year is 2023 or later. If the page mentions multiple dates, "
        "use the first consumer availability or general release year rather than announcement or dev-kit dates."
    )

def claim_price_under_700(name: str, price_mention: Optional[str]) -> str:
    if price_mention and price_mention.strip():
        return f"The base model price of {name} is {price_mention} (USD), which is at most $700."
    return f"The base model price of {name} is at most $700 (USD)."

def addins_price() -> str:
    return (
        "Use MSRP or typical current base model price for the US market (2025 pricing). "
        "If multiple storage SKUs/bundles exist, use the lowest/base SKU. Ignore temporary sales."
    )

def claim_resolution_2000_per_eye(name: str) -> str:
    return f"{name} has a per-eye resolution of at least 2000 by 2000 pixels."

def addins_resolution() -> str:
    return (
        "Confirm per-eye resolution is >= 2000×2000. If the page provides exact per-eye numbers like 2064×2208, that qualifies. "
        "If only a combined resolution for both eyes is shown (e.g., 3664×1920 total), it does not directly meet 'per-eye >= 2000×2000'. "
        "Accept equivalent phrasing (e.g., 'per-eye 2160×2160')."
    )

def claim_refresh_90hz(name: str) -> str:
    return f"{name} supports a refresh rate of at least 90 Hz."

def addins_refresh_90hz() -> str:
    return (
        "Check specs for refresh rate support. If ranges or 'up to' values are provided, verify that 90 Hz or higher is supported "
        "(e.g., 90/120 Hz)."
    )


# ---------------------------------------------------------------------------------------
# Per-item verification
# ---------------------------------------------------------------------------------------
async def verify_headset_item(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    item: VRHeadsetItem,
    idx: int
) -> Dict[str, VerificationNode]:
    """
    Build and verify the subtree for one headset.
    Returns a dict of leaf nodes for later aggregation.
    """
    # A per-item sequential node: if name/URLs missing, later checks skip automatically
    item_node = evaluator.add_sequential(
        id=f"headset_{idx}",
        desc=f"Headset #{idx+1} verification: {item.name or 'Unnamed'}",
        parent=parent_node,
        critical=False
    )

    # Name provided (critical within this chain)
    name_exists_node = evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id=f"headset_{idx}_name_provided",
        desc=f"Headset #{idx+1}: name is provided",
        parent=item_node,
        critical=True
    )

    # Reference URLs provided (critical within this chain)
    urls_provided_node = evaluator.add_custom_node(
        result=bool(item.urls and len(item.urls) > 0),
        id=f"headset_{idx}_urls_provided",
        desc=f"Headset #{idx+1}: at least one reference URL is provided",
        parent=item_node,
        critical=True
    )

    # Standalone capability (critical)
    standalone_node = evaluator.add_leaf(
        id=f"headset_{idx}_standalone",
        desc=f"Headset #{idx+1}: standalone capability verified",
        parent=item_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_standalone(item.name or "This headset"),
        node=standalone_node,
        sources=item.urls,
        additional_instruction=addins_standalone()
    )

    # PC connectivity (critical)
    pc_node = evaluator.add_leaf(
        id=f"headset_{idx}_pc_connectivity",
        desc=f"Headset #{idx+1}: PC connectivity verified",
        parent=item_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_pc_connectivity(item.name or "This headset"),
        node=pc_node,
        sources=item.urls,
        additional_instruction=addins_pc_connectivity()
    )

    # Release timeframe (critical)
    release_node = evaluator.add_leaf(
        id=f"headset_{idx}_release_2023_plus",
        desc=f"Headset #{idx+1}: released in 2023 or later",
        parent=item_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_release_2023_plus(item.name or "This headset"),
        node=release_node,
        sources=item.urls,
        additional_instruction=addins_release_timeframe()
    )

    # Price constraint (critical)
    price_node = evaluator.add_leaf(
        id=f"headset_{idx}_price_under_700",
        desc=f"Headset #{idx+1}: base price is $700 or less",
        parent=item_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_price_under_700(item.name or "This headset", item.price_usd),
        node=price_node,
        sources=item.urls,
        additional_instruction=addins_price()
    )

    # Resolution requirement (critical)
    resolution_node = evaluator.add_leaf(
        id=f"headset_{idx}_resolution_requirement",
        desc=f"Headset #{idx+1}: per-eye resolution >= 2000×2000",
        parent=item_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_resolution_2000_per_eye(item.name or "This headset"),
        node=resolution_node,
        sources=item.urls,
        additional_instruction=addins_resolution()
    )

    # Refresh rate provided (critical: must provide some refresh rate spec)
    refresh_provided_node = evaluator.add_custom_node(
        result=bool(item.refresh_rate and item.refresh_rate.strip()),
        id=f"headset_{idx}_refresh_rate_provided",
        desc=f"Headset #{idx+1}: refresh rate value is provided in the answer",
        parent=item_node,
        critical=True
    )

    # Refresh rate minimum 90 Hz (non-critical)
    refresh_90_node = evaluator.add_leaf(
        id=f"headset_{idx}_refresh_90hz_min",
        desc=f"Headset #{idx+1}: refresh rate >= 90 Hz",
        parent=item_node,
        critical=False
    )
    await evaluator.verify(
        claim=claim_refresh_90hz(item.name or "This headset"),
        node=refresh_90_node,
        sources=item.urls,
        additional_instruction=addins_refresh_90hz()
    )

    # Field of view provided (critical)
    fov_provided_node = evaluator.add_custom_node(
        result=bool(item.field_of_view and item.field_of_view.strip()),
        id=f"headset_{idx}_fov_provided",
        desc=f"Headset #{idx+1}: field-of-view value is provided in the answer",
        parent=item_node,
        critical=True
    )

    # Display type provided (critical)
    display_provided_node = evaluator.add_custom_node(
        result=bool(item.display_type and item.display_type.strip()),
        id=f"headset_{idx}_display_type_provided",
        desc=f"Headset #{idx+1}: display type is provided in the answer",
        parent=item_node,
        critical=True
    )

    return {
        "name": name_exists_node,
        "urls": urls_provided_node,
        "standalone": standalone_node,
        "pc": pc_node,
        "release": release_node,
        "price": price_node,
        "resolution": resolution_node,
        "refresh_provided": refresh_provided_node,
        "refresh_min": refresh_90_node,
        "fov": fov_provided_node,
        "display_type": display_provided_node,
        "item_node": item_node
    }


# ---------------------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------------------
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
    Entry point for evaluating the VR headset selection task.
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
        default_model=model
    )

    # Extract headset items from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_headsets(),
        template_class=VRHeadsetList,
        extraction_name="vr_headsets_extraction"
    )

    # Limit number of evaluated headsets to keep evaluation bounded
    headsets = extracted.headsets[:5] if extracted and extracted.headsets else []

    # Per-item verification parent (non-critical and parallel)
    items_parent = evaluator.add_parallel(
        id="per_item_checks",
        desc="Per-headset detailed verification",
        parent=root,
        critical=False
    )

    per_item_results: List[Dict[str, VerificationNode]] = []
    for i, item in enumerate(headsets):
        res = await verify_headset_item(evaluator, items_parent, item, i)
        per_item_results.append(res)

    # Add an empty placeholder if no items were provided, to keep tree understandable
    if not per_item_results:
        empty_node = evaluator.add_sequential(
            id="headset_0",
            desc="Headset #1 verification: Unspecified",
            parent=items_parent,
            critical=False
        )
        # Create explicit failed leaves to show missing info
        evaluator.add_custom_node(
            result=False,
            id="headset_0_name_provided",
            desc="Headset #1: name is provided",
            parent=empty_node,
            critical=True
        )
        evaluator.add_custom_node(
            result=False,
            id="headset_0_urls_provided",
            desc="Headset #1: at least one reference URL is provided",
            parent=empty_node,
            critical=True
        )
        # Store minimal result map to allow aggregations
        per_item_results.append({
            "name": evaluator.find_node("headset_0_name_provided"),
            "urls": evaluator.find_node("headset_0_urls_provided"),
        })

    # -----------------------------------------------------------------------------------
    # Aggregate checks aligned with rubric (top-level CRITICAL node)
    # Note: In the framework, a critical parent cannot have non-critical children.
    #       The rubric specifies one non-critical child ("Refresh_Rate_Minimum").
    #       Therefore, we place that non-critical check outside this critical parent.
    # -----------------------------------------------------------------------------------
    vr_selection_main = evaluator.add_parallel(
        id="VR_Headset_Selection",
        desc="Evaluate whether the answer identifies at least one qualifying VR headset and provides required info while satisfying constraints.",
        parent=root,
        critical=True
    )

    def any_passed(key: str) -> bool:
        hits = [m.get(key) for m in per_item_results if m.get(key) is not None]
        return any(node is not None and node.status == "passed" for node in hits)

    # Critical rubric children expressed as custom aggregation of per-item leaves
    evaluator.add_custom_node(
        result=any_passed("name"),
        id="Headset_Name_Provided",
        desc="At least one qualifying VR headset name is provided.",
        parent=vr_selection_main,
        critical=True
    )

    evaluator.add_custom_node(
        result=any_passed("standalone"),
        id="Standalone_Platform",
        desc="At least one proposed headset is a standalone VR headset.",
        parent=vr_selection_main,
        critical=True
    )

    evaluator.add_custom_node(
        result=any_passed("pc"),
        id="PC_Connectivity",
        desc="At least one proposed headset supports PCVR connectivity (wired or wireless streaming).",
        parent=vr_selection_main,
        critical=True
    )

    evaluator.add_custom_node(
        result=any_passed("release"),
        id="Release_Timeframe",
        desc="At least one proposed headset was released in 2023 or later.",
        parent=vr_selection_main,
        critical=True
    )

    evaluator.add_custom_node(
        result=any_passed("price"),
        id="Price_Constraint",
        desc="At least one proposed headset has a base model price of $700 or less (USD, 2025 pricing).",
        parent=vr_selection_main,
        critical=True
    )

    evaluator.add_custom_node(
        result=any_passed("resolution"),
        id="Resolution_Requirement",
        desc="At least one proposed headset has per-eye resolution of at least 2000×2000 pixels.",
        parent=vr_selection_main,
        critical=True
    )

    evaluator.add_custom_node(
        result=any_passed("refresh_provided"),
        id="Refresh_Rate_Provided",
        desc="At least one proposed headset has a refresh rate specification provided.",
        parent=vr_selection_main,
        critical=True
    )

    evaluator.add_custom_node(
        result=any_passed("fov"),
        id="Field_of_View_Provided",
        desc="At least one proposed headset has a field-of-view specification provided.",
        parent=vr_selection_main,
        critical=True
    )

    evaluator.add_custom_node(
        result=any_passed("display_type"),
        id="Display_Type_Provided",
        desc="At least one proposed headset has a display panel type specified.",
        parent=vr_selection_main,
        critical=True
    )

    evaluator.add_custom_node(
        result=any_passed("urls"),
        id="Reference_URLs",
        desc="At least one proposed headset includes official or reliable reference URL(s).",
        parent=vr_selection_main,
        critical=True
    )

    # Final critical gate to ensure at least one SINGLE headset satisfies all critical constraints together
    # (name, urls, standalone, pc, release, price, resolution, refresh_provided, fov, display_type)
    def at_least_one_fully_qualifies() -> bool:
        required_keys = [
            "name", "urls", "standalone", "pc", "release",
            "price", "resolution", "refresh_provided", "fov", "display_type"
        ]
        for m in per_item_results:
            ok = True
            for k in required_keys:
                node = m.get(k)
                if node is None or node.status != "passed":
                    ok = False
                    break
            if ok:
                return True
        return False

    evaluator.add_custom_node(
        result=at_least_one_fully_qualifies(),
        id="At_Least_One_Headset_Fully_Qualifies",
        desc="There is at least one headset that simultaneously satisfies all critical constraints.",
        parent=vr_selection_main,
        critical=True
    )

    # -----------------------------------------------------------------------------------
    # Non-critical rubric child: Refresh_Rate_Minimum (>= 90 Hz)
    # Kept outside the critical parent due to framework constraint (critical parent cannot have non-critical child).
    # -----------------------------------------------------------------------------------
    evaluator.add_custom_node(
        result=any_passed("refresh_min"),
        id="Refresh_Rate_Minimum",
        desc="At least one proposed headset supports a refresh rate of 90 Hz or higher.",
        parent=root,
        critical=False
    )

    return evaluator.get_summary()