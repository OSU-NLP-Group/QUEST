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
TASK_ID = "apple_watch_airtag_pf_v2_models"
TASK_DESCRIPTION = (
    "Apple announced the second-generation AirTag in January 2026 with new Precision Finding capabilities for Apple Watch. "
    "Identify two different Apple Watch models that support this Precision Finding feature with the AirTag (2nd generation). "
    "For each model, provide: (1) The complete model name, (2) The minimum watchOS version required for this feature, "
    "(3) The iPhone compatibility requirement for pairing, and (4) A direct link to the model's official specifications page "
    "on Apple's support website (support.apple.com)."
)

EXPECTED_MIN_WATCHOS = "watchOS 26.2.1 or later"
EXPECTED_IPHONE_REQ = (
    "Apple Intelligence–enabled iPhone (iPhone 15 Pro models or iPhone 16 models or later)"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WatchModelItem(BaseModel):
    """One Apple Watch model entry with required fields and any supporting URLs mentioned in the answer."""
    model_name: Optional[str] = None
    min_watchos: Optional[str] = None
    iphone_pairing: Optional[str] = None
    spec_url: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class WatchModelsExtraction(BaseModel):
    """Collection of up to two Apple Watch models as presented in the answer."""
    items: List[WatchModelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_watch_models() -> str:
    return """
    Extract up to two Apple Watch models from the answer that the author claims support Precision Finding with AirTag (2nd generation).
    For each model, extract the following fields exactly as presented in the answer:
    - model_name: The full Apple Watch model name (e.g., "Apple Watch Series 9", "Apple Watch Series 10", "Apple Watch Ultra 2", "Apple Watch Ultra (2nd generation)").
    - min_watchos: The minimum watchOS version the answer claims is required for this feature (for AirTag 2nd generation Precision Finding on Apple Watch). Example target wording: "watchOS 26.2.1 or later".
    - iphone_pairing: The iPhone compatibility requirement for pairing that the answer claims is required for this feature. Example target wording: "paired with an Apple Intelligence–enabled iPhone (iPhone 15 Pro models or iPhone 16 models or later)".
    - spec_url: A direct link to the model's official specifications page on Apple's support website (must be from https://support.apple.com). If multiple URLs are present, pick the one that is the most direct Apple Support specs/tech-specs page for that model. If none are present, set to null.
    - supporting_urls: A list of any additional URLs cited in the answer that support the feature prerequisites or claims (e.g., Apple Support articles, Apple Newsroom, feature pages). Include only valid, complete URLs. Do not duplicate spec_url here.

    Rules:
    - Extract only what is explicitly present in the answer text. Do not infer new URLs or change wording.
    - If the answer mentions more than two compatible models, include only the first two mentioned.
    - If any field is missing for a given model, set it to null (or empty list for supporting_urls).
    - Ensure all URLs are absolute and valid; ignore malformed links.

    Return a JSON object:
    {
      "items": [
        {
          "model_name": ...,
          "min_watchos": ...,
          "iphone_pairing": ...,
          "spec_url": ...,
          "supporting_urls": [...]
        },
        ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _combine_sources(item: WatchModelItem) -> List[str]:
    urls: List[str] = []
    for u in (item.supporting_urls or []):
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Optionally include spec_url as supplemental evidence
    if item.spec_url and item.spec_url.strip():
        urls.append(item.spec_url.strip())
    # Deduplicate while preserving order
    seen = set()
    dedup: List[str] = []
    for u in urls:
        if u not in seen:
            dedup.append(u)
            seen.add(u)
    return dedup


def _is_support_apple_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower()
    return "support.apple.com" in u


def _ordinal(idx: int) -> str:
    return "First" if idx == 0 else "Second"


# --------------------------------------------------------------------------- #
# Verification sub-tree per watch model                                       #
# --------------------------------------------------------------------------- #
async def verify_watch_model(
    evaluator: Evaluator,
    parent_node,
    item: WatchModelItem,
    index: int,
) -> None:
    """
    Build and verify the sub-tree for one Apple Watch model.
    """
    node_id = "first_watch_model" if index == 0 else "second_watch_model"
    watch_node = evaluator.add_parallel(
        id=node_id,
        desc=f"{_ordinal(index)} compatible Apple Watch model with complete information",
        parent=parent_node,
        critical=False,  # Allow partial credit per model
    )

    # ---------------- Model name: provided + meets hardware requirement ----------------
    name = _norm(item.model_name)

    name_exists_node = evaluator.add_custom_node(
        result=bool(name),
        id=f"{node_id}_model_name_provided",
        desc="Apple Watch model name is provided",
        parent=watch_node,
        critical=True,
    )

    name_hw_leaf = evaluator.add_leaf(
        id=f"{node_id}_model_name",
        desc="Apple Watch model name meets the hardware requirement (Series 9 or later, OR Ultra 2 or later)",
        parent=watch_node,
        critical=True,
    )
    claim_hw = (
        f"The Apple Watch model name '{name}' satisfies this condition: it is Apple Watch Series 9 or later, "
        f"or Apple Watch Ultra 2 or later (treat 'Ultra (2nd generation)' as Ultra 2). "
        f"If the name is empty or missing, consider the condition not satisfied."
    )
    await evaluator.verify(
        claim=claim_hw,
        node=name_hw_leaf,
        additional_instruction="Be lenient to minor formatting or punctuation variations; focus on whether it clearly denotes Series 9+ or Ultra 2+.",
    )

    # ---------------- watchOS requirement: provided + correctness ----------------
    min_watchos = _norm(item.min_watchos)

    watchos_provided = evaluator.add_custom_node(
        result=bool(min_watchos),
        id=f"{node_id}_watchos_requirement_provided",
        desc="Minimum watchOS requirement is provided",
        parent=watch_node,
        critical=True,
    )

    watchos_leaf = evaluator.add_leaf(
        id=f"{node_id}_watchos_requirement",
        desc="Specifies that watchOS 26.2.1 or later is required for Precision Finding feature",
        parent=watch_node,
        critical=True,
    )
    claim_watchos = (
        "The minimum watchOS version required for Precision Finding with AirTag (2nd generation) on Apple Watch "
        f"is exactly '{EXPECTED_MIN_WATCHOS}'. Allow equivalent phrasing like 'or newer' or 'or later'."
    )
    await evaluator.verify(
        claim=claim_watchos,
        node=watchos_leaf,
        sources=_combine_sources(item),
        additional_instruction="Rely on the cited Apple sources when available. If none are provided, judge based on the answer text.",
    )

    # ---------------- iPhone pairing requirement: provided + correctness ----------------
    iphone_req = _norm(item.iphone_pairing)

    iphone_provided = evaluator.add_custom_node(
        result=bool(iphone_req),
        id=f"{node_id}_iphone_pairing_provided",
        desc="iPhone pairing requirement is provided",
        parent=watch_node,
        critical=True,
    )

    iphone_leaf = evaluator.add_leaf(
        id=f"{node_id}_iphone_pairing",
        desc="States that the watch must be paired with an Apple Intelligence-enabled iPhone (iPhone 15 Pro models or iPhone 16 models or later)",
        parent=watch_node,
        critical=True,
    )
    claim_iphone = (
        "This feature requires pairing the Apple Watch with an Apple Intelligence–enabled iPhone, "
        "defined as iPhone 15 Pro or iPhone 15 Pro Max, or any iPhone 16 model (or later)."
    )
    await evaluator.verify(
        claim=claim_iphone,
        node=iphone_leaf,
        sources=_combine_sources(item),
        additional_instruction="Verify that the requirement matches the Apple Intelligence iPhone requirement wording or its clear equivalent.",
    )

    # ---------------- Specifications reference: support.apple.com + matches model ----------------
    spec_ref_node = evaluator.add_parallel(
        id=f"{node_id}_specifications_reference",
        desc="Provides official Apple support or specifications page for the watch model",
        parent=watch_node,
        critical=True,
    )

    spec_url_present = evaluator.add_custom_node(
        result=bool(item.spec_url) and _is_support_apple_url(item.spec_url),
        id=f"{node_id}_spec_url_present",
        desc="URL links to an official Apple support or product specifications page (support.apple.com)",
        parent=spec_ref_node,
        critical=True,
    )

    spec_leaf = evaluator.add_leaf(
        id=f"{node_id}_spec_url",
        desc="URL links to an official Apple support or product specifications page",
        parent=spec_ref_node,
        critical=True,
    )
    spec_claim = (
        f"This page is an official Apple Support specifications (or equivalent Tech Specs) page for the watch model '{name}', "
        "or a directly relevant Apple Support page that clearly corresponds to the same model and includes specs/technical details."
    )
    await evaluator.verify(
        claim=spec_claim,
        node=spec_leaf,
        sources=item.spec_url if item.spec_url else None,
        additional_instruction="Check that the page is on support.apple.com and that the page content/title indicates the correct Apple Watch model and its specs.",
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
    Evaluate an answer for the Apple Watch + AirTag (2nd gen) Precision Finding compatibility task.
    """
    # Initialize evaluator (root is non-critical PARALLEL aggregator by design)
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

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth(
        {
            "expected_min_watchOS": EXPECTED_MIN_WATCHOS,
            "expected_iPhone_requirement": EXPECTED_IPHONE_REQ,
            "hardware_requirement": "Apple Watch Series 9 or later, or Apple Watch Ultra 2 or later",
            "notes": "Two different Apple Watch models should be identified.",
        },
        gt_type="expected_requirements",
    )

    # Extract structured data from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_watch_models(),
        template_class=WatchModelsExtraction,
        extraction_name="watch_models_extraction",
    )

    items: List[WatchModelItem] = list(extracted.items or [])
    # Keep exactly two slots (pad if needed)
    if len(items) > 2:
        items = items[:2]
    while len(items) < 2:
        items.append(WatchModelItem())

    # Critical check: two distinct model names (if provided)
    m1 = _norm(items[0].model_name)
    m2 = _norm(items[1].model_name)
    two_distinct = bool(m1 and m2 and m1.lower() != m2.lower())
    evaluator.add_custom_node(
        result=two_distinct,
        id="two_distinct_models",
        desc="Two different Apple Watch models are provided (distinct model names)",
        parent=root,
        critical=True,
    )

    # Build verification for both models
    await verify_watch_model(evaluator, root, items[0], 0)
    await verify_watch_model(evaluator, root, items[1], 1)

    # Return structured summary
    return evaluator.get_summary()