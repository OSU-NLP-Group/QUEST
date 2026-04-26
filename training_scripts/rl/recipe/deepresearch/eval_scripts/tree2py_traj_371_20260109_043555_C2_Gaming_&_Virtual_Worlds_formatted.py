import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "vr_headset_2025_specs"
TASK_DESCRIPTION = (
    "Identify a standalone VR headset currently available in 2025 that meets the following technical "
    "specifications for an immersive gaming setup: (1) display resolution of at least 2064x2208 pixels per eye, "
    "(2) refresh rate capability of at least 120Hz, and (3) IPD (interpupillary distance) adjustment range that "
    "accommodates at least 58-71mm. Provide the headset name and manufacturer, along with verification URLs "
    "confirming each of these three technical specifications."
)


# ==============================
# Extraction Models
# ==============================
class HeadsetExtraction(BaseModel):
    headset_name: Optional[str] = None
    manufacturer: Optional[str] = None

    # Statements/claims as written in the answer (strings allow flexible phrasing)
    standalone_statement: Optional[str] = None
    available_2025_statement: Optional[str] = None

    # Spec-specific verification URLs (should be URLs explicitly present in the answer)
    resolution_urls: List[str] = Field(default_factory=list)
    refresh_rate_urls: List[str] = Field(default_factory=list)
    ipd_urls: List[str] = Field(default_factory=list)

    # Catch-all: every URL found in the answer (helps fallback if the answer uses one page for multiple specs)
    all_urls: List[str] = Field(default_factory=list)

    # Optional, not required for verification but useful if the answer included values
    resolution_per_eye_text: Optional[str] = None
    refresh_rate_text: Optional[str] = None
    ipd_range_text: Optional[str] = None


# ==============================
# Extraction Prompt
# ==============================
def prompt_extract_headset_info() -> str:
    return """
Extract the following information exactly as it appears in the provided answer text. Do not invent anything. If an item is missing, return null (for strings) or [] (for URL lists).

Required fields:
- headset_name: The specific name/model of the VR headset.
- manufacturer: The manufacturer/brand of the headset.

- standalone_statement: The exact sentence/phrase from the answer that asserts the headset is standalone (no external PC required). If not stated, return null.
- available_2025_statement: The exact sentence/phrase from the answer that asserts the headset is currently available for purchase in 2025 (not just “coming soon”). If not stated, return null.

- resolution_urls: A list of URL(s) explicitly provided in the answer that are intended to verify the per-eye display resolution requirement (≥ 2064×2208 per eye). If the answer uses a single product/spec page as evidence for multiple specs, include that URL here as well.
- refresh_rate_urls: A list of URL(s) explicitly provided in the answer that verify the refresh rate requirement (≥ 120 Hz). If a single page is used for multiple specs, include that URL here too.
- ipd_urls: A list of URL(s) explicitly provided in the answer that verify the IPD adjustment range includes at least 58–71 mm. If a single page is used for multiple specs, include that URL here too.

- all_urls: A list of every URL explicitly present in the answer (including all of the above and any others).

Optional fields (only if the answer explicitly states them):
- resolution_per_eye_text: The per-eye resolution string as written (e.g., "2064 x 2208 per eye"). If not provided, return null.
- refresh_rate_text: The refresh rate string as written (e.g., "up to 120Hz"). If not provided, return null.
- ipd_range_text: The IPD range string as written (e.g., "58–71 mm"). If not provided, return null.

Special rules for URL extraction:
- Only include URLs explicitly mentioned in the answer (plain URLs or in markdown links). Do not infer or fabricate URLs.
- If the answer provides one general product/spec page used to support multiple specs, duplicate it into each relevant spec URL list.
- Ensure all URLs are full and valid; if the protocol is missing, prepend http://.

Return a single JSON object matching the expected schema.
    """


# ==============================
# Helper Functions
# ==============================
def _has_text(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip())


def _display_name(manufacturer: Optional[str], name: Optional[str]) -> str:
    if _has_text(manufacturer) and _has_text(name):
        return f"{manufacturer.strip()} {name.strip()}"
    if _has_text(name):
        return name.strip()
    if _has_text(manufacturer):
        return f"{manufacturer.strip()} headset"
    return "the headset"


def _fallback_urls(primary: List[str], fallback: List[str]) -> List[str]:
    # Use primary if available, else fallback; also ensure uniqueness and preserve order
    seen = set()
    chosen = primary if primary else fallback
    result = []
    for u in chosen:
        if isinstance(u, str) and u.strip():
            if u not in seen:
                result.append(u)
                seen.add(u)
    return result


# ==============================
# Tree Construction + Verification
# ==============================
async def _verify_main_tree(evaluator: Evaluator, extracted: HeadsetExtraction) -> None:
    # Root rubric node (critical, parallel) under framework root
    main = evaluator.add_parallel(
        id="VR_Headset_Answer_Evaluation",
        desc="Evaluate whether the response identifies a qualifying standalone VR headset available in 2025 and provides required spec-verification URLs.",
        parent=evaluator.root,
        critical=True,
    )

    # Headset Identification (critical, parallel)
    ident = evaluator.add_parallel(
        id="Headset_Identification",
        desc="Response provides the headset identity information requested.",
        parent=main,
        critical=True,
    )

    # Headset name provided (critical leaf via custom node)
    evaluator.add_custom_node(
        result=_has_text(extracted.headset_name),
        id="Headset_Name_Provided",
        desc="Provides a specific headset name.",
        parent=ident,
        critical=True,
    )

    # Manufacturer provided (critical leaf via custom node)
    evaluator.add_custom_node(
        result=_has_text(extracted.manufacturer),
        id="Manufacturer_Provided",
        desc="Provides the manufacturer of the headset.",
        parent=ident,
        critical=True,
    )

    # Standalone requirement (critical leaf; evaluate statement presence in answer)
    standalone_leaf = evaluator.add_leaf(
        id="Standalone_Requirement",
        desc="Headset is standalone (no external PC required for operation).",
        parent=main,
        critical=True,
    )
    # We verify purely against the answer text (no external URLs)
    await evaluator.verify(
        claim=f"According to the answer, {_display_name(extracted.manufacturer, extracted.headset_name)} is a standalone VR headset that does not require a PC or external console for normal operation.",
        node=standalone_leaf,
        sources=None,
        additional_instruction="Judge only based on the answer content. Accept clear statements like 'standalone', 'no PC required', or equivalent phrasing. Do not use your own knowledge or external facts.",
    )

    # Availability in 2025 (critical leaf; evaluate statement presence in answer)
    availability_leaf = evaluator.add_leaf(
        id="Availability_2025",
        desc="Headset is stated to be currently available for purchase in 2025.",
        parent=main,
        critical=True,
    )
    await evaluator.verify(
        claim=f"According to the answer, {_display_name(extracted.manufacturer, extracted.headset_name)} is currently available for purchase in 2025 (not merely announced or pre-order without availability).",
        node=availability_leaf,
        sources=None,
        additional_instruction="Judge only based on the answer content. The answer must clearly state availability in 2025 (e.g., 'available now in 2025'). If only 'pre-order' or 'coming soon' is mentioned, consider it not available.",
    )

    # Technical specifications with verification URLs (critical, parallel)
    specs = evaluator.add_parallel(
        id="Technical_Specifications_With_Verification_URLs",
        desc="Each required technical specification is met and is supported by a verification URL.",
        parent=main,
        critical=True,
    )

    # Prepare claims and sources for each spec
    product_label = _display_name(extracted.manufacturer, extracted.headset_name)

    # Resolution ≥ 2064×2208 per eye
    res_node = evaluator.add_leaf(
        id="Resolution_Verified",
        desc="Provides a URL that confirms the headset's per-eye resolution is at least 2064x2208 pixels per eye.",
        parent=specs,
        critical=True,
    )
    resolution_urls = _fallback_urls(extracted.resolution_urls, extracted.all_urls)
    if resolution_urls:
        await evaluator.verify(
            claim=f"{product_label} has a per-eye display resolution of at least 2064 x 2208 pixels (per eye).",
            node=res_node,
            sources=resolution_urls,
            additional_instruction=(
                "Verify strictly from the webpage(s): the per-eye resolution must meet or exceed 2064 by 2208 pixels. "
                "Order can be '2064 x 2208' or '2208 x 2064'. Accept 'per eye', 'per-eye', or equivalent wording. "
                "If only a combined (both-eyes) resolution is reported and implies per-eye below the threshold, do not pass. "
                "Minor rounding/formatting variations are acceptable."
            ),
        )
    else:
        # Missing URLs => cannot support claim; mark as failed
        res_node.score = 0.0
        res_node.status = "failed"
        evaluator.add_custom_info(
            info={"reason": "No resolution verification URL(s) found in the answer."},
            info_type="missing_urls",
            info_name="resolution_urls_missing",
        )

    # Refresh rate ≥ 120 Hz
    rr_node = evaluator.add_leaf(
        id="Refresh_Rate_Verified",
        desc="Provides a URL that confirms the headset supports a refresh rate of at least 120Hz.",
        parent=specs,
        critical=True,
    )
    refresh_urls = _fallback_urls(extracted.refresh_rate_urls, extracted.all_urls)
    if refresh_urls:
        await evaluator.verify(
            claim=f"{product_label} supports a refresh rate of at least 120 Hz.",
            node=rr_node,
            sources=refresh_urls,
            additional_instruction=(
                "Verify from the webpage(s) that the headset supports 120 Hz or higher. "
                "Accept phrasings like 'up to 120 Hz', '120 Hz mode', '120 Hz (beta/experimental)'. "
                "If the maximum refresh rate shown is below 120 Hz, do not pass."
            ),
        )
    else:
        rr_node.score = 0.0
        rr_node.status = "failed"
        evaluator.add_custom_info(
            info={"reason": "No refresh rate verification URL(s) found in the answer."},
            info_type="missing_urls",
            info_name="refresh_rate_urls_missing",
        )

    # IPD range includes at least 58–71 mm
    ipd_node = evaluator.add_leaf(
        id="IPD_Range_Verified",
        desc="Provides a URL that confirms the headset's IPD adjustment range includes at least 58–71mm.",
        parent=specs,
        critical=True,
    )
    ipd_urls = _fallback_urls(extracted.ipd_urls, extracted.all_urls)
    if ipd_urls:
        await evaluator.verify(
            claim=f"{product_label} has an IPD adjustment range that includes at least 58 mm through 71 mm (inclusive).",
            node=ipd_node,
            sources=ipd_urls,
            additional_instruction=(
                "Verify from the webpage(s) that the adjustable IPD range covers both 58 mm and 71 mm inclusively. "
                "Examples that PASS: '58–71 mm', '56–72 mm', '58 to 73 mm'. "
                "Examples that FAIL: '59–71 mm', '58–70 mm', or ranges that do not include either 58 or 71. "
                "Accept equivalent phrasing like 'fits IPD 58-71 mm'."
            ),
        )
    else:
        ipd_node.score = 0.0
        ipd_node.status = "failed"
        evaluator.add_custom_info(
            info={"reason": "No IPD verification URL(s) found in the answer."},
            info_type="missing_urls",
            info_name="ipd_urls_missing",
        )


# ==============================
# Main Evaluation Entry Point
# ==============================
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

    # Extract structured information from the answer
    extracted: HeadsetExtraction = await evaluator.extract(
        prompt=prompt_extract_headset_info(),
        template_class=HeadsetExtraction,
        extraction_name="headset_extraction",
    )

    # Build verification tree and run checks
    await _verify_main_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()