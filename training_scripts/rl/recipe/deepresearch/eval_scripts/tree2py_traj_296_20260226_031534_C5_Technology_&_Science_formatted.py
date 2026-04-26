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
TASK_ID = "wireless_earbuds_2025_early_2026_eval"
TASK_DESCRIPTION = """Identify three different models of wireless earbuds that were released in 2025 or early 2026 (by February 26, 2026) and meet all of the following specifications:

1. The retail price must be under $350 USD
2. Must support at least one high-resolution audio codec such as LDAC, LHDC, aptX Lossless, or equivalent technology
3. Must provide at least 8 hours of continuous playback time on a single charge with ANC enabled
4. Must feature active noise cancellation (ANC) technology
5. Must support multipoint Bluetooth connectivity to connect to at least 2 devices simultaneously

For each of the three earbuds models you identify, provide:
- The exact model name and manufacturer
- The retail price in USD
- The specific high-resolution codec(s) supported
- The battery life specification (hours of continuous playback with ANC on)
- Confirmation of ANC and multipoint connectivity features
- A reference URL from either the manufacturer's official website or a reputable technology review site that confirms these specifications

The three models must be distinct products from different product lines (not just different colors or storage variants of the same base model)."""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EarbudsItem(BaseModel):
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None
    release_date: Optional[str] = None  # free-text date as given in the answer (e.g., "Jan 2026", "2025-10-12")
    price_usd: Optional[str] = None  # keep as string to allow "$299", "USD 299", "299"
    hi_res_codecs: List[str] = Field(default_factory=list)  # e.g., ["LDAC", "aptX Lossless"]
    battery_life_anc_hours: Optional[str] = None  # free-text like "8 hours with ANC on"
    anc: Optional[bool] = None
    multipoint: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)  # at least one URL


class EarbudsExtraction(BaseModel):
    items: List[EarbudsItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_earbuds() -> str:
    return """
Extract up to three distinct wireless earbuds models that the answer claims satisfy the task. If the answer lists more than three, only include the first three in the same order. For each model, extract the following fields exactly as stated in the answer:

- model_name: the exact product model name
- manufacturer: the company/brand that makes the earbuds
- release_date: the claimed release or announcement date in free text (e.g., "January 2026", "2025-09-20"); if not clearly stated, return null
- price_usd: the claimed retail price in USD as written (e.g., "$299", "USD 249", "249 USD"); if not clearly given, return null
- hi_res_codecs: array of the specific high-resolution Bluetooth codec names the answer claims (e.g., ["LDAC", "LHDC", "aptX Lossless", "Samsung Seamless Codec (SSC) HiFi"]); if none described, return an empty list
- battery_life_anc_hours: the claimed continuous playback time on a single charge with ANC enabled (e.g., "8 hours with ANC on"); if the answer only provides ANC-off battery life or it is unclear, return null
- anc: true if the answer explicitly claims the earbuds have Active Noise Cancellation (ANC); false if the answer says they do not; null if not specified
- multipoint: true if the answer explicitly claims Bluetooth multipoint connectivity to at least 2 devices; false if the answer says they do not; null if not specified
- reference_urls: array of one or more URLs provided in the answer that purportedly substantiate the specs for this model. Include manufacturer official pages or reputable tech review sites if present. If none given, return an empty array.

Return a JSON object { "items": [ ... ] } with up to three objects (one per model). Do not invent information that is not present in the answer. Ensure URLs are valid; if a URL lacks protocol, prepend http:// as per the general rules.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_k(items: List[EarbudsItem], k: int) -> List[EarbudsItem]:
    return items[:k] if items else []

def _all_urls(models: List[EarbudsItem]) -> List[str]:
    urls: List[str] = []
    for m in models:
        urls.extend(m.reference_urls or [])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification for a single model                                             #
# --------------------------------------------------------------------------- #
async def verify_one_model(evaluator: Evaluator, parent_node, item: EarbudsItem, idx: int) -> None:
    """
    Build verification subtree and run checks for a single earbuds model.
    The subtree follows the rubric's children for one model; each leaf is a single binary check.
    """
    model_node = evaluator.add_parallel(
        id=f"model_{idx}",
        desc=f"Model #{idx + 1}: Qualifying wireless earbuds meeting all requirements",
        parent=parent_node,
        critical=False  # non-critical so partial scoring across models is allowed
    )

    # 1) Reference URL existence (critical)
    has_refs = bool(item.reference_urls) and len(item.reference_urls) > 0
    ref_node = evaluator.add_custom_node(
        result=has_refs,
        id=f"model_{idx}_Reference_URL",
        desc="Valid reference URL(s) provided (manufacturer or reputable review sites expected)",
        parent=model_node,
        critical=True
    )

    # Prepare common sources for downstream verifications
    sources = item.reference_urls if item.reference_urls else None

    # We will verify the remaining 7 leaves; all are critical. We will batch-verify for efficiency.
    # Each verification will be gated by the Reference_URL leaf via extra_prerequisites.
    leaves_and_claims: List[tuple] = []

    # 2) Model Identification (exact model name + manufacturer; page is about this model)
    model_id_node = evaluator.add_leaf(
        id=f"model_{idx}_Model_Identification",
        desc="Provides the exact model name and manufacturer",
        parent=model_node,
        critical=True
    )
    model_name = item.model_name or ""
    manufacturer = item.manufacturer or ""
    claim_model = (
        f"The referenced page(s) is/are about wireless earbuds model '{model_name}' made by '{manufacturer}'. "
        f"Minor formatting differences are acceptable, but it must be the same product."
    )
    leaves_and_claims.append(
        (claim_model, sources, model_id_node,
         "Confirm the page explicitly references the same model and brand as claimed in the answer. "
         "Allow minor variations in hyphens, spacing, suffixes like 'TWS', and capitalization.")
    )

    # 3) Release Date in 2025 or by Feb 26, 2026
    release_leaf = evaluator.add_leaf(
        id=f"model_{idx}_Release_Date",
        desc="Product was released in 2025 or early 2026 (by February 26, 2026)",
        parent=model_node,
        critical=True
    )
    claimed_release = item.release_date or "unspecified"
    claim_release = (
        "This earbuds model was officially announced or released in the year 2025, "
        "or no later than February 26, 2026."
    )
    leaves_and_claims.append(
        (claim_release, sources, release_leaf,
         "Look for explicit release/announcement/availability dates on the page. "
         "If the source is a reputable review or press release from that time, "
         "a clear publication/announcement date within 2025 or on/before 2026-02-26 is acceptable. "
         f"The answer claimed release date: {claimed_release}. "
         "If the page provides a different date outside the window, mark as not supported.")
    )

    # 4) Price under $350 USD
    price_leaf = evaluator.add_leaf(
        id=f"model_{idx}_Model_Price",
        desc="Price is under $350 USD",
        parent=model_node,
        critical=True
    )
    claim_price = "The regular retail price (MSRP) for this model is under $350 USD."
    leaves_and_claims.append(
        (claim_price, sources, price_leaf,
         "Check MSRP/retail price on the page. $349.99 qualifies. "
         "If only non-USD currency is shown, use reasonable interpretation to determine if it is below $350 USD. "
         "Prioritize MSRP/regular price over limited-time sales. If unclear, treat as not supported.")
    )

    # 5) High-resolution codec support
    codec_leaf = evaluator.add_leaf(
        id=f"model_{idx}_High_Resolution_Codec",
        desc="Supports high-resolution audio codec (LDAC, LHDC, aptX Lossless, or equivalent)",
        parent=model_node,
        critical=True
    )
    claimed_codecs = ", ".join(item.hi_res_codecs) if item.hi_res_codecs else "unspecified"
    claim_codec = (
        "These earbuds support at least one high-resolution Bluetooth codec: "
        "LDAC, LHDC, aptX Lossless, or an equivalent hi-fidelity codec (e.g., 'Samsung Seamless Codec (SSC) HiFi')."
    )
    leaves_and_claims.append(
        (claim_codec, sources, codec_leaf,
         "Verify the page mentions support for LDAC, LHDC, aptX Lossless, or an equivalent hi-res codec. "
         "Do NOT accept 'aptX Adaptive' unless it explicitly indicates a Lossless mode. "
         f"The answer listed: {claimed_codecs}.")
    )

    # 6) Battery life with ANC on >= 8 hours
    battery_leaf = evaluator.add_leaf(
        id=f"model_{idx}_Battery_Life",
        desc="Offers at least 8 hours of continuous playback on a single charge with ANC enabled",
        parent=model_node,
        critical=True
    )
    claimed_batt = item.battery_life_anc_hours or "unspecified"
    claim_battery = "With ANC enabled, a single charge provides at least 8 hours of continuous playback."
    leaves_and_claims.append(
        (claim_battery, sources, battery_leaf,
         "Confirm that the stated battery life is with ANC ON (enabled). "
         "If only ANC-OFF time is available or ambiguous, do not count it. "
         f"The answer's claim: {claimed_batt}. If the page shows < 8 hours with ANC on, mark as not supported.")
    )

    # 7) ANC feature
    anc_leaf = evaluator.add_leaf(
        id=f"model_{idx}_Active_Noise_Cancellation",
        desc="Features active noise cancellation (ANC) technology",
        parent=model_node,
        critical=True
    )
    claim_anc = "These earbuds feature active noise cancellation (ANC)."
    leaves_and_claims.append(
        (claim_anc, sources, anc_leaf,
         "Verify that the page explicitly states 'Active Noise Cancellation' or similar functionality.")
    )

    # 8) Multipoint connectivity
    multipoint_leaf = evaluator.add_leaf(
        id=f"model_{idx}_Multipoint_Connectivity",
        desc="Supports multipoint Bluetooth connection to at least 2 devices simultaneously",
        parent=model_node,
        critical=True
    )
    claim_multipoint = "These earbuds support Bluetooth multipoint connection to at least two devices simultaneously."
    leaves_and_claims.append(
        (claim_multipoint, sources, multipoint_leaf,
         "The page should mention 'multipoint' or 'connect to two devices at the same time'. "
         "If the feature is branded differently, it still qualifies if simultaneous two-device connection is supported.")
    )

    # Run batch verifications with gating on the reference URL custom node
    await evaluator.batch_verify(
        [(c, s, n, ins) for (c, s, n, ins) in leaves_and_claims],
        extra_prerequisites=[ref_node]
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
    Evaluate an answer for the wireless earbuds 2025/early-2026 task.
    """
    # Initialize evaluator (root is non-critical parallel to allow partial scoring across models)
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

    # Build top-level "Task_Completion" node (parallel); set non-critical to satisfy framework constraints
    # while still allowing critical children within.
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify three different wireless earbuds models released in 2025 or early 2026 that meet all specified criteria",
        parent=root,
        critical=False
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_earbuds(),
        template_class=EarbudsExtraction,
        extraction_name="earbuds_extraction"
    )

    # Keep only first three items; pad with placeholders if fewer than three
    items = _first_k(extracted.items, 3)
    while len(items) < 3:
        items.append(EarbudsItem())

    # Add parallel sub-nodes for each of the three earbuds and verify each
    # Each model subtree enforces its own critical checks
    # First model
    first_model_node = evaluator.add_parallel(
        id="First_Earbuds_Model",
        desc="First qualifying wireless earbuds model meeting all requirements",
        parent=task_node,
        critical=False
    )
    await verify_one_model(evaluator, first_model_node, items[0], 0)

    # Second model
    second_model_node = evaluator.add_parallel(
        id="Second_Earbuds_Model",
        desc="Second qualifying wireless earbuds model meeting all requirements (must be different from first model)",
        parent=task_node,
        critical=False
    )
    await verify_one_model(evaluator, second_model_node, items[1], 1)

    # Third model
    third_model_node = evaluator.add_parallel(
        id="Third_Earbuds_Model",
        desc="Third qualifying wireless earbuds model meeting all requirements (must be different from first and second models)",
        parent=task_node,
        critical=False
    )
    await verify_one_model(evaluator, third_model_node, items[2], 2)

    # Distinct product lines check (critical at top-level task completion node)
    distinct_leaf = evaluator.add_leaf(
        id="Distinct_Product_Lines",
        desc="The three models are distinct products from different product lines (not just different colors or storage variants of the same base model)",
        parent=task_node,
        critical=True
    )
    trio_names = [
        f"{(it.manufacturer or 'Unknown')} {(it.model_name or 'Unknown')}".strip()
        for it in items
    ]
    claim_distinct = (
        "The following three earbuds are distinct product lines and not merely color or storage variants of the same base model: "
        f"1) {trio_names[0]}; 2) {trio_names[1]}; 3) {trio_names[2]}."
    )
    # For this cross-item logical check, use a simple verification.
    await evaluator.verify(
        claim=claim_distinct,
        node=distinct_leaf,
        additional_instruction=(
            "Judge distinctness based on model naming and typical industry conventions. "
            "Different colors, small storage variants, or minor SKU suffixes are NOT distinct product lines. "
            "Different tiers (e.g., 'Pro' vs 'Lite' vs 'Sport') generally ARE distinct lines."
        )
    )

    # Return structured result
    return evaluator.get_summary()