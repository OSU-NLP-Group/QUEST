import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "teacher_ce_hours_max"
TASK_DESCRIPTION = """
Among Pennsylvania, Texas, and Illinois, which state requires the most continuing education hours for regular classroom teacher license renewal, and how many hours does that state require?
"""

# Ground truth expectations for reference (not used to auto-judge; just logged)
GROUND_TRUTH = {
    "states_considered": ["Pennsylvania", "Texas", "Illinois"],
    "max_state": "Pennsylvania",
    "max_hours": "180"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    """
    Structured extraction of key fields from the answer text.
    """
    # The set of states the answer explicitly uses for the comparison; record exactly as written in the answer
    states_compared: List[str] = Field(default_factory=list)

    # The state identified by the answer as having the highest continuing education requirement
    reported_max_state: Optional[str] = None

    # The hours value the answer states for the maximum state; keep exactly as shown in the answer (e.g., "180", "180 hours", "180 hrs")
    reported_max_hours: Optional[str] = None

    # Short phrase quoted from the answer indicating the teacher certificate scope (e.g., "regular classroom teachers", "Instructional II certificate")
    scope_description: Optional[str] = None

    # Short phrase quoted from the answer indicating the metric used (e.g., "180 hours every 5 years", "CPE hours per renewal cycle")
    metric_description: Optional[str] = None

    # All URLs cited in the answer (including markdown links; extract the actual URL targets)
    cited_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_fields() -> str:
    return """
    Extract the following fields from the answer text. Copy values exactly as they appear in the answer (do not invent or normalize beyond what is written). If a field is not present, return null for single fields or [] for lists.

    Required fields:
    - states_compared: an array of the state names or abbreviations that the answer explicitly uses in the comparison set (e.g., ["Pennsylvania","Texas","Illinois"] or ["PA","TX","IL"]). Record exactly as written.
    - reported_max_state: the state the answer claims has the highest continuing education requirement among Pennsylvania, Texas, and Illinois. Record exactly as written.
    - reported_max_hours: the number of hours (and possibly unit text) the answer states for the maximum state (e.g., "180", "180 hours"). Record exactly as written.
    - scope_description: a short phrase quoted from the answer indicating the scope (e.g., "regular classroom teachers", "standard teaching certificate", "Instructional II"). If not specified, return null.
    - metric_description: a short phrase quoted from the answer indicating the metric used for comparison (e.g., "hours per renewal cycle", "180 hours every 5 years", "CPE hours"). If not specified, return null.
    - cited_urls: all URLs present in the answer (including markdown links; extract the link targets). Only include valid http/https URLs.

    Return a single JSON object with these fields.
    """


# --------------------------------------------------------------------------- #
# Verification helper (claims)                                                #
# --------------------------------------------------------------------------- #
def build_sources_claim_for_pa_180() -> str:
    return (
        "At least one of the provided citations/links is an official or otherwise authoritative source and "
        "explicitly supports that Pennsylvania requires 180 hours of continuing professional education "
        "per renewal cycle for regular classroom teachers (equivalently phrased as '180 hours every 5 years')."
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Entry point for evaluating an answer to the teacher CE hours maximum question.
    """

    # 1) Initialize evaluator and root
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

    # 2) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_answer_fields(),
        template_class=AnswerExtraction,
        extraction_name="answer_key_fields",
    )

    # 3) Log ground truth (for reference only)
    evaluator.add_ground_truth({"expected": GROUND_TRUTH}, gt_type="ground_truth")

    # 4) Build the evaluation tree according to the rubric
    # Create a critical parent node that aggregates all criteria in parallel
    answer_eval_node = evaluator.add_parallel(
        id="Answer_Evaluation",
        desc=(
            "Evaluate whether the answer correctly determines which of Pennsylvania, Texas, and Illinois has the "
            "highest continuing education/professional development hour requirement for regular classroom teacher "
            "license renewal, and reports the required hours, with traceable sourcing."
        ),
        parent=root,
        critical=True,
    )

    # Leaf: Scope_Includes_Exactly_Three_States
    scope_three_states_node = evaluator.add_leaf(
        id="Scope_Includes_Exactly_Three_States",
        desc="The answer’s comparison is explicitly among Pennsylvania, Texas, and Illinois (no other states substituted/added for the comparison).",
        parent=answer_eval_node,
        critical=True,
    )
    claim_scope_three_states = (
        "The answer's comparison set is explicitly and only Pennsylvania, Texas, and Illinois. "
        "Allow abbreviations like PA, TX, IL to count as those states. "
        "If any other state is added or substituted as part of the comparison or decision, this is incorrect."
    )
    await evaluator.verify(
        claim=claim_scope_three_states,
        node=scope_three_states_node,
        additional_instruction=(
            "Judge solely based on the provided answer text. It's sufficient if the answer clearly limits the comparison "
            "to Pennsylvania, Texas, and Illinois only, even if it mentions other states elsewhere for unrelated context."
        ),
    )

    # Leaf: Scope_Regular_Classroom_Teacher
    scope_regular_node = evaluator.add_leaf(
        id="Scope_Regular_Classroom_Teacher",
        desc="The answer addresses renewal requirements for regular classroom teachers (not administrative or specialized certificates).",
        parent=answer_eval_node,
        critical=True,
    )
    claim_scope_regular = (
        "The answer addresses renewal requirements for regular classroom teachers (standard teaching certificate), "
        "not for administrators or specialized endorsements. Accept equivalent phrases such as 'standard teaching certificate' "
        "or Pennsylvania's 'Instructional II' if used clearly to mean regular classroom teachers."
    )
    await evaluator.verify(
        claim=claim_scope_regular,
        node=scope_regular_node,
        additional_instruction="Base this judgment only on what the answer states.",
    )

    # Leaf: Metric_Is_Hours_Per_Renewal_Cycle
    metric_hours_node = evaluator.add_leaf(
        id="Metric_Is_Hours_Per_Renewal_Cycle",
        desc="The answer compares continuing education/professional development/CPE hour requirements per renewal cycle.",
        parent=answer_eval_node,
        critical=True,
    )
    claim_metric_hours = (
        "The answer makes the comparison specifically in terms of continuing education/professional development hours "
        "per renewal cycle (e.g., 'X hours every Y years'), not other metrics like credits-only, fees, tests, or initial licensure."
    )
    await evaluator.verify(
        claim=claim_metric_hours,
        node=metric_hours_node,
        additional_instruction="Look for explicit mention of hours per cycle; phrases like 'PD hours', 'CPE hours', or '180 hours every 5 years' qualify.",
    )

    # Leaf: Correct_State_With_Max_Hours
    correct_state_node = evaluator.add_leaf(
        id="Correct_State_With_Max_Hours",
        desc="The answer identifies Pennsylvania as the state with the highest hour requirement among Pennsylvania, Texas, and Illinois.",
        parent=answer_eval_node,
        critical=True,
    )
    claim_correct_state = (
        "Within the answer, the state identified as having the highest continuing education hour requirement among "
        "Pennsylvania, Texas, and Illinois is Pennsylvania."
    )
    await evaluator.verify(
        claim=claim_correct_state,
        node=correct_state_node,
        additional_instruction="Do not rely on external knowledge; verify only the answer's stated identification.",
    )

    # Leaf: Correct_Hour_Value_For_Max_State
    correct_hours_node = evaluator.add_leaf(
        id="Correct_Hour_Value_For_Max_State",
        desc="The answer states that the maximum-state requirement is 180 hours (Pennsylvania’s requirement).",
        parent=answer_eval_node,
        critical=True,
    )
    claim_correct_hours = (
        "The answer explicitly states that the highest requirement is 180 hours (for Pennsylvania). "
        "Accept equivalent phrasing like '180 hours over 5 years' or 'Act 48 requires 180 hours every cycle'."
    )
    await evaluator.verify(
        claim=claim_correct_hours,
        node=correct_hours_node,
        additional_instruction="Minor wording variations are fine as long as it clearly communicates 180 hours for Pennsylvania.",
    )

    # Leaf: Sources_Are_Traceable_And_Authoritative
    sources_auth_node = evaluator.add_leaf(
        id="Sources_Are_Traceable_And_Authoritative",
        desc="The answer provides traceable citations/links to official state education department sources or other authoritative educational resources supporting the stated requirement.",
        parent=answer_eval_node,
        critical=True,
    )
    sources_claim = build_sources_claim_for_pa_180()
    urls = extracted.cited_urls if extracted and extracted.cited_urls else []
    await evaluator.verify(
        claim=sources_claim,
        node=sources_auth_node,
        sources=urls,
        additional_instruction=(
            "Consider official state education department domains (e.g., *.pa.gov, Pennsylvania Department of Education) as authoritative. "
            "Also acceptable: clearly authoritative education bodies or state teacher certification resources. "
            "The page must explicitly support that Pennsylvania requires 180 hours per renewal cycle (phrases like "
            "'180 hours every 5 years' or 'Act 48: 180 hours' qualify). If no links are provided in the answer, this should fail."
        ),
    )

    # 5) Return summary
    return evaluator.get_summary()