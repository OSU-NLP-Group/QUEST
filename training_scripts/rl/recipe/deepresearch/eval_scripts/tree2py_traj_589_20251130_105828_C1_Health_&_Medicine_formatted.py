import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "orforglipron_fda_timeline_dosing"
TASK_DESCRIPTION = """
Eli Lilly is developing an oral GLP-1 receptor agonist called Orforglipron for weight loss treatment.
Based on the medication's selection for the FDA Commissioner's National Priority Review Voucher pilot program, when is the expected FDA regulatory decision timeline?
Additionally, what is the dosing regimen for this medication in terms of frequency and administration requirements?
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OrforglipronExtraction(BaseModel):
    """Structured information extracted from the agent's answer about Orforglipron."""
    expected_timeline_text: Optional[str] = None
    dosing_frequency_text: Optional[str] = None
    administration_requirements_text: Optional[str] = None

    # Source URLs cited in the answer for each claim (if any)
    expected_timeline_sources: List[str] = Field(default_factory=list)
    dosing_frequency_sources: List[str] = Field(default_factory=list)
    administration_requirements_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_orforglipron_info() -> str:
    return """
    Extract the key factual information about Orforglipron from the provided answer.

    Required fields:
    1) expected_timeline_text: The answer's stated expected FDA regulatory decision timeline for Orforglipron
       (e.g., "first half of 2026", "H1 2026", "early to mid-2026", etc.). If not stated, return null.
    2) dosing_frequency_text: The answer's stated dosing frequency (e.g., "once daily", "QD", etc.). If not stated, return null.
    3) administration_requirements_text: The answer's stated administration requirements regarding food or water restrictions
       (e.g., "can be taken without food/water restrictions", "no fasting required", etc.). If not stated, return null.

    For each of the above three fields, also extract any URLs cited in the answer that specifically support that field:
    4) expected_timeline_sources: An array of URLs cited for the expected timeline claim. If none are cited, return an empty array.
    5) dosing_frequency_sources: An array of URLs cited for the dosing frequency claim. If none are cited, return an empty array.
    6) administration_requirements_sources: An array of URLs cited for the administration requirements claim. If none are cited, return an empty array.

    SPECIAL RULES FOR URL SOURCES:
    - Only include actual URLs explicitly present in the answer (plain URLs or in markdown links).
    - Do not invent or infer URLs; if missing, leave the array empty.
    - Include full URLs; if the protocol is missing, prepend "http://".

    Return a single JSON object containing all the fields above.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_timeline_checks(evaluator: Evaluator, parent_node, extracted: OrforglipronExtraction) -> None:
    """
    Build verification nodes for the expected FDA decision timeline.
    Critical group: requires existence in the answer, correctness (matches 'first half of 2026'),
    and support by cited sources if provided.
    """
    timeline_group = evaluator.add_parallel(
        id="Expected_FDA_Decision_Timeline",
        desc="Expected FDA regulatory decision timeline verification",
        parent=parent_node,
        critical=True
    )

    # Existence check: the answer must state a timeline
    has_timeline = extracted.expected_timeline_text is not None and extracted.expected_timeline_text.strip() != ""
    evaluator.add_custom_node(
        result=has_timeline,
        id="timeline_claim_present",
        desc="The answer includes an expected FDA decision timeline for Orforglipron",
        parent=timeline_group,
        critical=True
    )

    # Correctness check: answer states first half of 2026
    timeline_correct_leaf = evaluator.add_leaf(
        id="timeline_matches_expected_h1_2026",
        desc="The answer identifies the timeline as the first half of 2026 (accept equivalents: H1 2026, early to mid-2026)",
        parent=timeline_group,
        critical=True
    )
    await evaluator.verify(
        claim="The expected FDA regulatory decision timeline for Orforglipron is in the first half of 2026.",
        node=timeline_correct_leaf,
        sources=None,  # Check consistency against the answer text itself
        additional_instruction=(
            "Check whether the answer text indicates 'first half of 2026' or reasonable equivalents "
            "such as 'H1 2026', 'by mid-2026', or 'early to mid-2026'. The claim should be considered correct "
            "only if the answer explicitly matches these variants."
        ),
    )

    # Source support check: verify by cited URLs if available
    timeline_supported_leaf = evaluator.add_leaf(
        id="timeline_supported_by_sources",
        desc="The cited sources support that the expected FDA decision timeline is in the first half of 2026",
        parent=timeline_group,
        critical=True
    )
    await evaluator.verify(
        claim="The cited sources indicate that the expected FDA decision timeline for Orforglipron is in the first half of 2026.",
        node=timeline_supported_leaf,
        sources=extracted.expected_timeline_sources,  # May be empty; will route to simple_verify
        additional_instruction=(
            "Evaluate the cited sources (if any). Consider synonyms such as 'H1 2026' or 'by mid-2026'. "
            "The sources should explicitly support this timeframe. Mention of selection for the FDA Commissioner's "
            "National Priority Review Voucher pilot program can provide context for expedited review, "
            "but the source must still substantively support the first-half-of-2026 timeline."
        ),
    )


async def build_dosing_checks(evaluator: Evaluator, parent_node, extracted: OrforglipronExtraction) -> None:
    """
    Build verification nodes for dosing frequency.
    Critical group: requires existence in the answer, and correctness (once daily) supported by sources if available.
    """
    dosing_group = evaluator.add_parallel(
        id="Dosing_Frequency",
        desc="Dosing frequency verification",
        parent=parent_node,
        critical=True
    )

    # Existence check: the answer must state dosing frequency
    has_dosing = extracted.dosing_frequency_text is not None and extracted.dosing_frequency_text.strip() != ""
    evaluator.add_custom_node(
        result=has_dosing,
        id="dosing_claim_present",
        desc="The answer includes a dosing frequency for Orforglipron",
        parent=dosing_group,
        critical=True
    )

    # Correctness check: once daily
    dosing_correct_leaf = evaluator.add_leaf(
        id="dosing_once_daily",
        desc="The answer states that Orforglipron is administered once daily",
        parent=dosing_group,
        critical=True
    )
    await evaluator.verify(
        claim="Orforglipron is administered once daily.",
        node=dosing_correct_leaf,
        sources=None,  # Check against the answer itself
        additional_instruction=(
            "Accept synonyms like 'QD', 'once-daily', or '1x per day' as equivalent phrasing."
        ),
    )

    # Source support check
    dosing_supported_leaf = evaluator.add_leaf(
        id="dosing_supported_by_sources",
        desc="The cited sources support that Orforglipron is dosed once daily",
        parent=dosing_group,
        critical=True
    )
    await evaluator.verify(
        claim="The cited sources indicate that Orforglipron is administered once daily.",
        node=dosing_supported_leaf,
        sources=extracted.dosing_frequency_sources,
        additional_instruction=(
            "Check the dosing instructions, regimen, or label-like materials on the cited sources "
            "to verify once-daily dosing (QD)."
        ),
    )


async def build_admin_checks(evaluator: Evaluator, parent_node, extracted: OrforglipronExtraction) -> None:
    """
    Build verification nodes for administration requirements about food/water restrictions.
    Critical group: requires existence in the answer, and correctness (no restrictions) supported by sources if available.
    """
    admin_group = evaluator.add_parallel(
        id="Administration_Requirements",
        desc="Administration requirements verification (food/water restrictions)",
        parent=parent_node,
        critical=True
    )

    # Existence check: the answer must state administration requirements
    has_admin = extracted.administration_requirements_text is not None and extracted.administration_requirements_text.strip() != ""
    evaluator.add_custom_node(
        result=has_admin,
        id="admin_claim_present",
        desc="The answer includes administration requirements about food/water restrictions",
        parent=admin_group,
        critical=True
    )

    # Correctness check: can be taken without food/water restrictions
    admin_correct_leaf = evaluator.add_leaf(
        id="admin_no_food_water_restrictions",
        desc="The answer states that Orforglipron can be taken without food and water restrictions",
        parent=admin_group,
        critical=True
    )
    await evaluator.verify(
        claim="Orforglipron can be taken without food and water restrictions.",
        node=admin_correct_leaf,
        sources=None,
        additional_instruction=(
            "Interpret 'without food and water restrictions' as no requirements to take on an empty stomach, "
            "no fasting window, and no specific water volume timing constraints. Accept equivalent phrasing "
            "such as 'no fasting required' or 'can be taken without regard to meals'."
        ),
    )

    # Source support check
    admin_supported_leaf = evaluator.add_leaf(
        id="admin_supported_by_sources",
        desc="The cited sources support that Orforglipron has no food/water administration restrictions",
        parent=admin_group,
        critical=True
    )
    await evaluator.verify(
        claim="The cited sources indicate that Orforglipron can be taken without food and water restrictions.",
        node=admin_supported_leaf,
        sources=extracted.administration_requirements_sources,
        additional_instruction=(
            "Verify that the sources explicitly state no fasting or meal-related restrictions, "
            "and no special water intake requirements for dosing."
        ),
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
    Evaluate an answer for Orforglipron key information:
    - Expected FDA decision timeline (first half of 2026)
    - Dosing frequency (once daily)
    - Administration requirements (no food/water restrictions)
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
        default_model=model,
    )

    # Extract structured information from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_orforglipron_info(),
        template_class=OrforglipronExtraction,
        extraction_name="orforglipron_extraction",
    )

    # Add a critical wrapper node reflecting the rubric's "Orforglipron_Key_Information"
    key_info_node = evaluator.add_parallel(
        id="Orforglipron_Key_Information",
        desc="Evaluation of key factual information about Orforglipron, Eli Lilly's investigational oral GLP-1 medication",
        parent=root,
        critical=True
    )

    # Build verification subtrees (all critical under the wrapper)
    await build_timeline_checks(evaluator, key_info_node, extracted)
    await build_dosing_checks(evaluator, key_info_node, extracted)
    await build_admin_checks(evaluator, key_info_node, extracted)

    # Add ground truth reference (for transparency)
    evaluator.add_ground_truth({
        "expected_timeline": "First half of 2026 (H1 2026)",
        "dosing_frequency": "Once daily (QD)",
        "administration_requirements": "No food/water restrictions"
    }, gt_type="ground_truth_orforglipron")

    # Return structured evaluation summary
    return evaluator.get_summary()