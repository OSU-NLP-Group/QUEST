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
TASK_ID = "vr_arcade_vr_headsets_2024_2025"
TASK_DESCRIPTION = """
A company in California is opening a commercial VR arcade and needs to purchase VR headsets for their gaming stations. Identify 3 different commercially available VR headset models (as of 2024-2025) that meet ALL of the following technical and business requirements:

Technical Requirements:
- Minimum refresh rate of 90Hz
- Minimum resolution of 1800 pixels × 1900 pixels per eye
- Field of view of at least 100 degrees
- Standalone/wireless capability (no PC tether required)

Business Requirements:
- Available in a commercial/business version or licensing option
- Commercial version price not exceeding $700 per unit

For each of the 3 headsets, provide:
1. Manufacturer name and exact model name
2. Technical specifications (refresh rate, resolution per eye, field of view)
3. Confirmation of standalone/wireless capability
4. Commercial version price
5. At least one reference URL documenting these specifications
""".strip()

REQUIRED_HEADSET_COUNT = 3

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HeadsetItem(BaseModel):
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    refresh_rate: Optional[str] = None
    resolution_per_eye: Optional[str] = None
    field_of_view: Optional[str] = None
    standalone_wireless: Optional[bool] = None
    commercial_option: Optional[str] = None  # e.g., "Business Edition", "For Business", "Commercial license"
    price: Optional[str] = None              # Keep as string to allow "$699", "USD 649", etc.
    reference_urls: List[str] = Field(default_factory=list)  # All URLs cited for this headset


class HeadsetExtraction(BaseModel):
    headsets: List[HeadsetItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_headsets() -> str:
    return """
    Extract up to the first 3 VR headset entries the answer proposes for the commercial VR arcade. 
    For each headset, extract the following fields as they appear in the answer:
    - manufacturer: Manufacturer name (string).
    - model: Exact model name (string).
    - refresh_rate: The stated refresh rate or range (string, e.g., "90Hz", "90–120Hz", "up to 120 Hz").
    - resolution_per_eye: Resolution per eye as stated (string, e.g., "1832 x 1920 per eye", or if only total is provided, include that total string like "3664 x 1920 total").
    - field_of_view: Field of view (string, e.g., "100°", "110 degrees", "104° horizontal").
    - standalone_wireless: true if the answer claims standalone/wireless capability without a PC tether; false otherwise; null if unspecified.
    - commercial_option: The business/commercial/enterprise/licensing mention (e.g., "Business Edition", "For Business", "Commercial license"); null if not mentioned.
    - price: The commercial/business version per-unit price string as stated (e.g., "$699", "USD 649", "£599"); if multiple prices, choose the business/commercial per‑unit price; if unavailable, set null.
    - reference_urls: An array of all URLs in the answer that document or support the specs/pricing/availability for this headset. These may be manufacturer pages, business program pages, spec sheets, retailer pages, or press/coverage pages linked by the answer. If none are provided, return an empty array.

    Notes:
    - Do NOT invent or infer information; only extract what is explicitly present in the answer.
    - Prefer extracting strings for specs to maximize compatibility (e.g., keep "90–120 Hz" as a string).
    - For URLs: extract only valid URLs explicitly present in the answer (including markdown links). Return the full URLs including protocol.
    - If the answer lists more than 3 headsets, include only the first three in the same order. 
    - If fewer than 3 are mentioned, include those available; missing items will be handled downstream.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _full_model_name(item: HeadsetItem, index: int) -> str:
    mfr = (item.manufacturer or "").strip()
    mdl = (item.model or "").strip()
    if mfr and mdl:
        return f"{mfr} {mdl}"
    if mdl:
        return mdl
    if mfr:
        return mfr
    return f"Headset #{index + 1}"


# --------------------------------------------------------------------------- #
# Verification for a single headset                                           #
# --------------------------------------------------------------------------- #
async def verify_single_headset(
    evaluator: Evaluator,
    parent_node,
    item: HeadsetItem,
    index: int
) -> None:
    """
    Build verification subtree and execute checks for one headset.
    All requirement checks are critical under this headset node; failing any will nullify the headset.
    """
    # Create container node for this headset (non-critical to allow partial credit across different headsets)
    headset_node = evaluator.add_parallel(
        id=f"headset_{index+1}",
        desc=f"Headset #{index+1} verification - meets all specified technical and business requirements",
        parent=parent_node,
        critical=False
    )

    # Precondition 1: Identification present
    identification_ok = bool(item.manufacturer and item.manufacturer.strip()) and bool(item.model and item.model.strip())
    evaluator.add_custom_node(
        result=identification_ok,
        id=f"headset_{index+1}_identification",
        desc="Manufacturer name and exact model name are provided",
        parent=headset_node,
        critical=True
    )

    # Precondition 2: At least one reference URL is provided
    references_ok = bool(item.reference_urls and len(item.reference_urls) > 0)
    evaluator.add_custom_node(
        result=references_ok,
        id=f"headset_{index+1}_reference_url",
        desc="At least one reference URL documenting the specifications is provided",
        parent=headset_node,
        critical=True
    )

    # Prepare common strings and sources
    model_full = _full_model_name(item, index)
    sources = item.reference_urls if item.reference_urls else []

    # Business requirement: commercial or business availability/licensing
    node_commercial = evaluator.add_leaf(
        id=f"headset_{index+1}_commercial_availability",
        desc="Headset is available in a commercial or business version or licensing option",
        parent=headset_node,
        critical=True
    )
    claim_commercial = (
        f"The {model_full} headset has a commercial/business offering or licensing option suitable for commercial/arcade use."
    )
    await evaluator.verify(
        claim=claim_commercial,
        node=node_commercial,
        sources=sources,
        additional_instruction=(
            "Look for explicit indications of 'For Business', 'Business Edition', 'Enterprise', 'Commercial', "
            "'Commercial license', 'XR for Business', or a business/enterprise program page tied to this model. "
            "It must clearly indicate commercial/enterprise usage or licensing terms."
        )
    )

    # Business requirement: commercial version price <= $700 per unit
    node_price = evaluator.add_leaf(
        id=f"headset_{index+1}_price",
        desc="Commercial version price does not exceed $700 per unit",
        parent=headset_node,
        critical=True
    )
    if item.price and item.price.strip():
        claim_price = (
            f"The commercial/business per-unit price for {model_full} is {item.price.strip()} and does not exceed $700 USD per unit (excluding taxes/shipping)."
        )
    else:
        claim_price = (
            f"The commercial/business per-unit price for {model_full} does not exceed $700 USD per unit (excluding taxes/shipping)."
        )
    await evaluator.verify(
        claim=claim_price,
        node=node_price,
        sources=sources,
        additional_instruction=(
            "Verify from the referenced page(s) that the business/commercial per-unit hardware price is at most $700 USD. "
            "If the page only shows a standard retail price but explicitly applies to business/commercial purchases, that's acceptable. "
            "If no price is shown or only higher than $700, mark as not supported."
        )
    )

    # Year availability requirement: commercially available as of 2024-2025
    node_year = evaluator.add_leaf(
        id=f"headset_{index+1}_year_availability",
        desc="Headset is commercially available as of 2024-2025",
        parent=headset_node,
        critical=True
    )
    claim_year = (
        f"The {model_full} headset is commercially available for purchase or business deployment as of 2024 or 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_year,
        node=node_year,
        sources=sources,
        additional_instruction=(
            "Treat official product or business purchase pages with a current 'Buy', 'Add to Cart', or 'Contact Sales' option as evidence of availability in 2024–2025. "
            "An explicit 2024/2025 date is helpful but not required if the page clearly offers current purchase for that time frame."
        )
    )

    # Technical requirement: refresh rate >= 90Hz
    node_refresh = evaluator.add_leaf(
        id=f"headset_{index+1}_refresh_rate",
        desc="Refresh rate is at least 90Hz",
        parent=headset_node,
        critical=True
    )
    claim_refresh = f"The {model_full} headset supports a display refresh rate of at least 90 Hz."
    await evaluator.verify(
        claim=claim_refresh,
        node=node_refresh,
        sources=sources,
        additional_instruction=(
            "Check the specifications on the referenced page(s). Accept if the page states '90 Hz', '90–120 Hz', 'up to 120 Hz', etc. "
            "If multiple modes exist, confirm that at least one officially supported mode is ≥ 90 Hz."
        )
    )

    # Technical requirement: resolution per eye >= 1800 x 1900
    node_resolution = evaluator.add_leaf(
        id=f"headset_{index+1}_resolution",
        desc="Resolution is at least 1800x1900 pixels per eye",
        parent=headset_node,
        critical=True
    )
    claim_resolution = (
        f"The per-eye resolution for {model_full} is at least 1800 by 1900 pixels (or equivalent/higher)."
    )
    await evaluator.verify(
        claim=claim_resolution,
        node=node_resolution,
        sources=sources,
        additional_instruction=(
            "If the page lists per-eye resolution (e.g., 1832×1920), that satisfies the requirement. "
            "If only a combined (both-eyes) resolution is provided (e.g., 3664×1920 total), interpret per-eye as half the horizontal resolution if clearly symmetrical (e.g., 1832×1920 per eye). "
            "Pass if the resulting per-eye dimensions are each ≥ the required thresholds."
        )
    )

    # Technical requirement: FOV >= 100 degrees
    node_fov = evaluator.add_leaf(
        id=f"headset_{index+1}_fov",
        desc="Field of view is at least 100 degrees",
        parent=headset_node,
        critical=True
    )
    claim_fov = f"The {model_full} headset has a field of view of at least 100 degrees."
    await evaluator.verify(
        claim=claim_fov,
        node=node_fov,
        sources=sources,
        additional_instruction=(
            "Accept horizontal, vertical, or diagonal FOV claims if the stated number is ≥ 100°. "
            "If the page states '≈100°' or 'about 100°', treat as meeting the threshold."
        )
    )

    # Technical requirement: standalone/wireless (no PC tether required)
    node_wireless = evaluator.add_leaf(
        id=f"headset_{index+1}_wireless",
        desc="Standalone/wireless capability without requiring a PC tether",
        parent=headset_node,
        critical=True
    )
    claim_wireless = (
        f"The {model_full} headset can operate wirelessly/standalone without requiring a PC tether (built‑in compute and power)."
    )
    await evaluator.verify(
        claim=claim_wireless,
        node=node_wireless,
        sources=sources,
        additional_instruction=(
            "Pass if the referenced page indicates a standalone headset (on‑device apps/compute) or explicitly supports untethered operation. "
            "Optional PC streaming is acceptable; the requirement is that a PC tether is not required."
        )
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
    Evaluate an answer for the VR arcade headset selection task.
    """
    # Initialize evaluator (root: parallel to allow partial scoring across the three headsets)
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

    # Extract proposed headsets from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_headsets(),
        template_class=HeadsetExtraction,
        extraction_name="headset_candidates"
    )

    # Keep only the first REQUIRED_HEADSET_COUNT items and pad if necessary
    items = list(extracted.headsets[:REQUIRED_HEADSET_COUNT])
    while len(items) < REQUIRED_HEADSET_COUNT:
        items.append(HeadsetItem())

    # Add ground truth-style reference: requirements summary (for context in output)
    evaluator.add_ground_truth({
        "required_count": REQUIRED_HEADSET_COUNT,
        "technical_requirements": {
            "refresh_rate": ">= 90 Hz",
            "resolution_per_eye": ">= 1800 x 1900 pixels",
            "field_of_view": ">= 100 degrees",
            "standalone_wireless": True
        },
        "business_requirements": {
            "commercial_option": True,
            "commercial_price_per_unit_usd": "<= 700",
            "availability_year": "2024-2025"
        }
    }, gt_type="requirements")

    # Verify each headset independently under the root
    for idx, item in enumerate(items):
        await verify_single_headset(evaluator, root, item, idx)

    # Return the evaluation summary
    return evaluator.get_summary()