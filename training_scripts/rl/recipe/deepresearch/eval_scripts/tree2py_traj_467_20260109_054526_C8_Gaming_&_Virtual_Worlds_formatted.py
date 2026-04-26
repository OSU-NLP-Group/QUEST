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
TASK_ID = "vr_headsets_vrsetup_2025"
TASK_DESCRIPTION = """
I am planning a VR gaming setup and need to identify high-quality standalone VR headsets that meet specific technical requirements for optimal performance and comfort. Please find three commercially available standalone VR headsets that satisfy all of the following criteria:

1. Resolution: Each headset must have a per-eye resolution of at least 2000 × 2000 pixels.
2. Refresh Rate: Each headset must support a refresh rate of at least 90 Hz.
3. Field of View: Each headset must offer a horizontal field of view (FOV) of at least 100 degrees.
4. Weight: Each headset (with headstrap included) must weigh no more than 600 grams.
5. IPD Range: Each headset's interpupillary distance (IPD) adjustment range must include the range from 60mm to 68mm (i.e., the adjustable range must span from at least some value ≤ 60mm to at least some value ≥ 68mm).
6. Lens Type: Each headset must use pancake lenses (not fresnel lenses).
7. Connectivity: Each headset must function as a standalone VR device, meaning it does not require a connection to a PC or gaming console for basic operation.
8. Availability: Each headset must be commercially available for purchase (released and not discontinued) as of January 2025.

For each of the three headsets, provide the following information:
- Headset Model Name
- Manufacturer
- Per-Eye Resolution
- Refresh Rate
- Horizontal Field of View
- Weight (with headstrap)
- IPD Range
- Lens Type
- Connectivity Type
- Current Retail Price (USD if available)
- Reference URL(s): Official product/spec page or reputable VR specification database entry.
"""

AS_OF_DATE = "January 2025"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VRHeadsetItem(BaseModel):
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None
    per_eye_resolution: Optional[str] = None  # e.g., "2160 × 2160"
    refresh_rate_hz: Optional[str] = None     # e.g., "120 Hz"
    horizontal_fov_deg: Optional[str] = None  # e.g., "110°"
    weight_grams: Optional[str] = None        # e.g., "515 g"
    ipd_range_mm: Optional[str] = None        # e.g., "58–71 mm"
    lens_type: Optional[str] = None           # e.g., "pancake"
    connectivity_type: Optional[str] = None   # e.g., "standalone"
    current_retail_price_usd: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VRHeadsetsExtraction(BaseModel):
    headsets: List[VRHeadsetItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_vr_headsets() -> str:
    return """
    Extract up to three standalone VR headsets listed in the answer along with their key specifications and reference URLs.
    For each headset, return an object with the fields:
    - model_name: Full commercial name of the headset
    - manufacturer: Company that produces the headset
    - per_eye_resolution: Resolution per eye, as written (e.g., "2160 × 2160")
    - refresh_rate_hz: Maximum or standard refresh rate, as written (e.g., "120 Hz")
    - horizontal_fov_deg: Horizontal FOV in degrees, as written (e.g., "110°")
    - weight_grams: Total weight with headstrap, as written (e.g., "515 g" or "515 grams")
    - ipd_range_mm: Adjustable IPD range in millimeters, as written (e.g., "58–71 mm")
    - lens_type: Lens type description (e.g., "pancake", "fresnel")
    - connectivity_type: Connectivity description (e.g., "standalone", "PC VR")
    - current_retail_price_usd: Current retail price in USD if provided in the answer (e.g., "$299", "USD 299", "299 USD")
    - reference_urls: An array of direct URLs cited in the answer that point to official manufacturer product/spec pages OR reputable VR specification databases (e.g., meta.com product pages, vive.com, pico-interactive.com, VRCompare, etc.). Extract only actual URLs mentioned in the answer (plain or markdown links). If none are provided for a headset, return an empty list.

    Rules:
    - Extract information exactly as presented in the answer; do not invent values.
    - If more than three headsets are mentioned, include only the first three.
    - If a field is not mentioned for a headset, set it to null (or empty list for reference_urls).
    - For URLs missing protocol, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_n_items(items: List[VRHeadsetItem], n: int = 3) -> List[VRHeadsetItem]:
    selected = items[:n]
    if len(selected) < n:
        # pad with empty entries
        for _ in range(n - len(selected)):
            selected.append(VRHeadsetItem())
    return selected


def _urls_or_none(urls: List[str]) -> Optional[List[str]]:
    return urls if urls else None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_headset(
    evaluator: Evaluator,
    parent_node,
    item: VRHeadsetItem,
    idx: int,
) -> None:
    """
    Build verification nodes for one headset and perform checks.
    The order ensures that critical prerequisites (like references) are verified early.
    """
    # Create headset-level parallel node (non-critical to allow partial credit across headsets)
    hs_node = evaluator.add_parallel(
        id=f"headset_{idx}",
        desc=f"{idx}st qualifying VR headset (meets constraints and required fields)" if idx == 1 else (
             f"{idx}nd qualifying VR headset (meets constraints and required fields)" if idx == 2 else
             f"{idx}rd qualifying VR headset (meets constraints and required fields)"
        ),
        parent=parent_node,
        critical=False,
    )

    # 1) Required simple fields existence (critical)
    evaluator.add_custom_node(
        result=bool(item.model_name and item.model_name.strip()),
        id=f"h{idx}_model_name",
        desc="Headset model name (full commercial name) is provided",
        parent=hs_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(item.manufacturer and item.manufacturer.strip()),
        id=f"h{idx}_manufacturer",
        desc="Manufacturer is provided",
        parent=hs_node,
        critical=True,
    )

    # 2) Reference quality check (critical) — do this early as a prerequisite for other URL-based verifications
    ref_node = evaluator.add_leaf(
        id=f"h{idx}_reference",
        desc=("Reference URL(s) are provided from an official manufacturer product/spec page or an established VR "
              "specification database entry sufficient to verify the stated specifications"),
        parent=hs_node,
        critical=True,
    )
    # Verify references' quality — pass if at least one is official/spec or reputable database and contains specs
    await evaluator.verify(
        claim=("At least one of the provided URLs is an official manufacturer product/spec page or a reputable VR "
               "specification database entry for this headset, and it includes technical specifications."),
        node=ref_node,
        sources=_urls_or_none(item.reference_urls),
        additional_instruction=("Check domains and page content. Accept official manufacturer product/spec pages "
                                "(e.g., meta.com, vive.com, pico-interactive.com) or reputable VR spec databases "
                                "(e.g., VRCompare). The page should list technical specs like resolution or refresh rate."),
    )

    # Prepare contextual strings for claims
    hs_name = item.model_name or "the headset"
    urls = _urls_or_none(item.reference_urls)

    # 3) Resolution ≥ 2000 × 2000 (critical)
    res_node = evaluator.add_leaf(
        id=f"h{idx}_resolution",
        desc="Per-eye resolution is at least 2000 × 2000 pixels",
        parent=hs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The headset '{hs_name}' has a per-eye resolution that is at least 2000 × 2000 pixels."),
        node=res_node,
        sources=urls,
        additional_instruction=("From the specification page(s), confirm that both width and height per eye are "
                                "≥ 2000. Ensure it refers to per-eye resolution (not combined total)."),
    )

    # 4) Refresh rate ≥ 90 Hz (critical)
    rr_node = evaluator.add_leaf(
        id=f"h{idx}_refresh_rate",
        desc="Refresh rate is at least 90 Hz",
        parent=hs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The headset '{hs_name}' supports a refresh rate of at least 90 Hz."),
        node=rr_node,
        sources=urls,
        additional_instruction=("Confirm max or standard refresh rate from the specs. If multiple modes exist, "
                                "accept if any officially supported mode is ≥ 90 Hz."),
    )

    # 5) Horizontal FOV ≥ 100° (critical)
    fov_node = evaluator.add_leaf(
        id=f"h{idx}_fov",
        desc="Horizontal field of view is at least 100 degrees",
        parent=hs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The headset '{hs_name}' offers a horizontal field of view of at least 100 degrees."),
        node=fov_node,
        sources=urls,
        additional_instruction=("Prefer horizontal FOV. If only 'FOV' is provided without orientation and it is widely "
                                "recognized as horizontal for the device, it's acceptable. Do not count diagonal FOV as horizontal."),
    )

    # 6) Weight ≤ 600 g with headstrap (critical)
    wt_node = evaluator.add_leaf(
        id=f"h{idx}_weight",
        desc="Weight with headstrap is no more than 600 grams",
        parent=hs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The headset '{hs_name}' weighs no more than 600 grams including the headstrap."),
        node=wt_node,
        sources=urls,
        additional_instruction=("Check the listed weight and whether it includes the headstrap. If the page lists weight "
                                "without strap or excludes key components, do not count; the requirement is with headstrap."),
    )

    # 7) IPD range includes 60–68 mm (min ≤ 60 and max ≥ 68) (critical)
    ipd_node = evaluator.add_leaf(
        id=f"h{idx}_ipd",
        desc="IPD adjustment range includes 60mm to 68mm (min ≤ 60 and max ≥ 68)",
        parent=hs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The headset '{hs_name}' has an adjustable IPD range that includes 60 mm to 68 mm (i.e., min ≤ 60 mm and max ≥ 68 mm)."),
        node=ipd_node,
        sources=urls,
        additional_instruction=("Confirm the IPD adjustment range from the specifications. Mechanical or continuous ranges are acceptable. "
                                "If min > 60 or max < 68, this fails."),
    )

    # 8) Pancake lenses (not fresnel) (critical)
    lens_node = evaluator.add_leaf(
        id=f"h{idx}_lens_type",
        desc="Uses pancake lenses (not fresnel lenses)",
        parent=hs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The headset '{hs_name}' uses pancake lenses (not fresnel lenses)."),
        node=lens_node,
        sources=urls,
        additional_instruction=("Look for references to 'pancake lenses'. If the page references 'fresnel lenses', this fails."),
    )

    # 9) Standalone connectivity (critical)
    conn_node = evaluator.add_leaf(
        id=f"h{idx}_connectivity",
        desc="Functions as a standalone VR headset without requiring PC or console for basic operation",
        parent=hs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The headset '{hs_name}' functions as a standalone VR device and does not require a PC or console for basic operation."),
        node=conn_node,
        sources=urls,
        additional_instruction=("Confirm that it can operate independently (apps/games run on-device). "
                                "Optional link modes (PC VR streaming or wired link) are fine as long as standalone is supported."),
    )

    # 10) Availability as of January 2025 (critical)
    avail_node = evaluator.add_leaf(
        id=f"h{idx}_availability",
        desc="Commercially available for purchase as of January 2025 (released and not discontinued)",
        parent=hs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"As of {AS_OF_DATE}, the headset '{hs_name}' is commercially available for purchase (released and not discontinued)."),
        node=avail_node,
        sources=urls,
        additional_instruction=("Look for current product listing status, purchase options, or official availability statements. "
                                "If the product is discontinued or no longer available for sale as of the specified date, this fails."),
    )

    # 11) Price documented (critical)
    price_node = evaluator.add_leaf(
        id=f"h{idx}_price",
        desc="Current retail price is documented (in USD or with a clear USD conversion/value stated)",
        parent=hs_node,
        critical=True,
    )
    price_text = item.current_retail_price_usd or "a documented current retail price in USD (or convertible to USD)"
    await evaluator.verify(
        claim=(f"The headset '{hs_name}' has {price_text} documented on the provided reference pages or official store."),
        node=price_node,
        sources=urls,
        additional_instruction=("Verify that a current retail price is shown in USD or a regional currency that can be clearly converted to USD. "
                                "If multiple variants exist, any clear price for the headset is acceptable."),
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
    Evaluate an answer for the VR headset specification task.
    """
    # Initialize evaluator with parallel root (three headsets evaluated independently)
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

    # Extract headsets info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_vr_headsets(),
        template_class=VRHeadsetsExtraction,
        extraction_name="vr_headsets_extraction",
    )

    # Record simple custom info
    evaluator.add_custom_info(
        {"extracted_count": len(extracted.headsets)},
        info_type="extraction_stats",
        info_name="extraction_summary"
    )

    # Take first 3 items (pad if fewer)
    headsets = _first_n_items(extracted.headsets, n=3)

    # Build verification tree and run checks for each headset
    for i, hs in enumerate(headsets, start=1):
        await verify_single_headset(evaluator, root, hs, i)

    # Return standardized summary
    return evaluator.get_summary()