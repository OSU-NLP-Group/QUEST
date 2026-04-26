import asyncio
import logging
import re
from typing import Any, Optional, List, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "psvr2_specs_eval"
TASK_DESCRIPTION = "What is the per-eye panel resolution and display technology used in the PlayStation VR2 headset? Provide both the horizontal and vertical resolution values, as well as the specific display technology type."


class PSVR2SpecsExtraction(BaseModel):
    resolution_horizontal: Optional[str] = None
    resolution_vertical: Optional[str] = None
    resolution_combined_text: Optional[str] = None
    display_technology: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


def prompt_extract_specs() -> str:
    return """
    Extract the PlayStation VR2 specifications mentioned in the answer. Focus specifically on:
    1) resolution_horizontal: the horizontal resolution per eye (e.g., "2000"). If the answer provides "2000 x 2040 per eye", parse and fill horizontal as "2000".
    2) resolution_vertical: the vertical resolution per eye (e.g., "2040"). If the answer provides "2000 x 2040 per eye", parse and fill vertical as "2040".
    3) resolution_combined_text: the exact resolution expression as stated for per-eye resolution (e.g., "2000 × 2040 per eye" or "2000 x 2040 per eye").
    4) display_technology: the display technology name (e.g., "OLED", "Organic Light-Emitting Diode").
    5) sources: collect all URLs explicitly cited in the answer. Include any URLs under playstation.com as well as any other links mentioned.

    Notes:
    - Return string values exactly as they appear in the answer; do not normalize beyond splitting for horizontal/vertical.
    - If any value is missing, return null for that field. If no URLs are provided, return an empty list for sources.
    - Do not invent URLs; only extract those present in the answer text (including markdown links).
    """


def parse_resolution(horizontal: Optional[str], vertical: Optional[str], combined: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    h = horizontal.strip() if horizontal else None
    v = vertical.strip() if vertical else None
    if h and v:
        return h, v
    if combined:
        text = combined.strip()
        # Prefer explicit pair near "x" or "×"
        m = re.search(r'(\d{3,4})\s*[×x]\s*(\d{3,4})', text)
        if m:
            return m.group(1), m.group(2)
        # Fallback: pick first two 3–4 digit numbers if separated in text
        nums = re.findall(r'\b(\d{3,4})\b', text)
        if len(nums) >= 2:
            return nums[0], nums[1]
    return None, None


def filter_official_ps_urls(urls: List[str]) -> List[str]:
    official = []
    for u in urls:
        lu = u.lower()
        if "playstation.com" in lu and ("vr2" in lu or "ps-vr2" in lu or "playstation-vr2" in lu or "psvr2" in lu):
            official.append(u)
    return official


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

    specs = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=PSVR2SpecsExtraction,
        extraction_name="psvr2_specs_extraction",
    )

    # Ground truth reference
    evaluator.add_ground_truth({
        "expected_resolution_per_eye": {"horizontal": "2000", "vertical": "2040"},
        "expected_display_technology": "OLED (Organic Light-Emitting Diode)",
        "official_source_requirement": "Must cite official PlayStation (playstation.com) VR2 technical/product specs page"
    }, gt_type="ground_truth_specs")

    # Create critical parent node matching rubric "Root"
    task_main = evaluator.add_parallel(
        id="Root",
        desc="Correctly provides both the per-eye panel resolution (with horizontal/vertical values) and the display technology for PlayStation VR2, using the official PlayStation VR2 technical specifications as the source",
        parent=root,
        critical=True
    )

    # Compute helpers
    h_val, v_val = parse_resolution(specs.resolution_horizontal, specs.resolution_vertical, specs.resolution_combined_text)
    official_urls = filter_official_ps_urls(specs.sources)
    all_urls = specs.sources if specs.sources else []

    # Leaf 1: Panel_Resolution_Per_Eye
    res_leaf = evaluator.add_leaf(
        id="Panel_Resolution_Per_Eye",
        desc="States the per-eye panel resolution as 2000 (horizontal) × 2040 (vertical)",
        parent=task_main,
        critical=True
    )
    # Construct claim from the answer's extracted values (so we judge what the agent actually asserted)
    if h_val and v_val:
        res_claim = f"The PlayStation VR2 panel resolution per eye is {h_val} (horizontal) × {v_val} (vertical)."
    else:
        # If answer didn't provide usable numbers, make the claim reflect that deficiency to encourage a fail
        res_claim = "The PlayStation VR2 panel resolution per eye is not clearly provided in the answer."
    res_sources = official_urls if official_urls else all_urls if all_urls else None
    await evaluator.verify(
        claim=res_claim,
        node=res_leaf,
        sources=res_sources,
        additional_instruction=(
            "Confirm the per-eye panel resolution values on the official PlayStation VR2 specifications page. "
            "Focus on per-eye resolution (not total across both eyes). Allow minor formatting variants like 'x' vs '×' or thousands separators."
        ),
    )

    # Leaf 2: Display_Technology
    display_leaf = evaluator.add_leaf(
        id="Display_Technology",
        desc="Identifies the display technology as OLED (Organic Light-Emitting Diode)",
        parent=task_main,
        critical=True
    )
    display_val = specs.display_technology.strip() if specs.display_technology else ""
    if display_val:
        display_claim = f"The PlayStation VR2 uses {display_val} display technology."
    else:
        display_claim = "The PlayStation VR2's display technology is not clearly provided in the answer."
    display_sources = official_urls if official_urls else all_urls if all_urls else None
    await evaluator.verify(
        claim=display_claim,
        node=display_leaf,
        sources=display_sources,
        additional_instruction=(
            "Verify the display panel technology on the official PlayStation VR2 specifications page. "
            "Treat 'OLED' and 'Organic Light-Emitting Diode' as equivalent; allow minor case differences."
        ),
    )

    # Leaf 3: Official_Source_Used (custom existence check based on extracted URLs)
    official_used = evaluator.add_custom_node(
        result=(len(official_urls) > 0),
        id="Official_Source_Used",
        desc="Cites/uses the official PlayStation VR2 technical specifications as the source for the provided specs",
        parent=task_main,
        critical=True
    )

    return evaluator.get_summary()