import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "teaching_cert_comparison"
TASK_DESCRIPTION = (
    "Compare the initial teaching certification requirements for secondary education (grades 6-12) across four U.S. "
    "states: Texas, Florida, California, and North Carolina. For each state, provide the following specific information:\n\n"
    "1. Minimum degree requirement for initial certification\n"
    "2. Required examination(s) that must be passed\n"
    "3. Student teaching or clinical practice hours requirement\n"
    "4. Whether a background check is required\n"
    "5. Application fee amount\n"
    "6. Continuing professional education or professional development hours required for the first certification renewal\n\n"
    "Your answer must include all six categories of information for each of the four states, with appropriate reference "
    "URLs supporting each piece of information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CategoryInfo(BaseModel):
    content: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StateRequirements(BaseModel):
    degree_requirement: Optional[CategoryInfo] = None
    exam_requirements: Optional[CategoryInfo] = None
    student_teaching: Optional[CategoryInfo] = None
    background_check: Optional[CategoryInfo] = None
    application_fee: Optional[CategoryInfo] = None
    continuing_education: Optional[CategoryInfo] = None


class CertificationExtraction(BaseModel):
    texas: Optional[StateRequirements] = None
    florida: Optional[StateRequirements] = None
    california: Optional[StateRequirements] = None
    north_carolina: Optional[StateRequirements] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the initial teacher certification requirements for secondary education (grades 6–12) from the provided answer text
    for each of the four states: Texas, Florida, California, and North Carolina.

    For each state, extract the following six categories exactly as stated in the answer:
    1) degree_requirement.content: The stated minimum degree requirement for initial certification (e.g., "Bachelor's degree").
       degree_requirement.sources: An array of all URLs cited in the answer that support this degree requirement.
    2) exam_requirements.content: The required examination(s) that must be passed for initial certification (e.g., list or description).
       exam_requirements.sources: An array of all URLs cited for the exams requirement.
    3) student_teaching.content: The student teaching or clinical practice requirement (include hours if stated).
       student_teaching.sources: An array of all URLs cited for the student teaching/clinical practice requirement.
    4) background_check.content: Whether a background check/fingerprinting is required (e.g., "Yes, fingerprint-based criminal history check").
       background_check.sources: An array of all URLs cited for this requirement.
    5) application_fee.content: The application or initial certification fee amount (e.g., "$75").
       application_fee.sources: An array of all URLs cited for this fee.
    6) continuing_education.content: The continuing professional education/professional development required for the first renewal
       (e.g., "150 CPE hours over 5 years" or "120 in-service points").
       continuing_education.sources: An array of all URLs cited for this item.

    Return a JSON object with the following top-level keys: "texas", "florida", "california", "north_carolina".
    Each of these should be an object with the six categories above. For any missing category, set the 'content' to null and 'sources' to [].
    Only include URLs that are explicitly present in the answer. If URLs are in markdown links, extract the actual link.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_content_and_sources(cat: Optional[CategoryInfo]) -> bool:
    return bool(cat and cat.content and str(cat.content).strip() and cat.sources and len(cat.sources) > 0)


async def _add_and_verify(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]],
    additional_instruction: str,
    critical: bool = True
):
    """
    Create a leaf node and verify the given claim against sources. If sources are missing, fail the node.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )
    # Enforce source-grounding: if missing sources, fail this factual check
    if not sources:
        leaf.score = 0.0
        leaf.status = "failed"
        return leaf

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction
    )
    return leaf


async def verify_state_requirements(
    evaluator: Evaluator,
    root,
    state_name: str,
    state_node_id: str,
    node_name_prefix: str,
    data: Optional[StateRequirements]
):
    """
    Build the verification subtree for one state and execute all leaf verifications as specified in the rubric.
    """
    # Parent node for the state (parallel; non-critical to allow partial credit within a state)
    state_node = evaluator.add_parallel(
        id=state_node_id,
        desc=f"Identify all required initial teaching certification requirements for secondary education (grades 6-12) in {state_name}",
        parent=root,
        critical=False
    )

    # Per-category additional instruction scaffold
    common_ctx = f"Focus on INITIAL teacher certification for secondary (grades 6–12) in {state_name}. "\
                 f"If a statewide/general teacher certification requirement applies across grade bands, it's acceptable as long as it applies to secondary candidates."

    # 1) Degree requirement
    degree = data.degree_requirement if data else None
    deg_desc = f"State the minimum degree requirement for initial secondary teaching certification in {state_name}"
    deg_claim = f"In {state_name}, the minimum degree requirement for initial secondary (grades 6–12) teacher certification is: {degree.content if degree and degree.content else ''}"
    await _add_and_verify(
        evaluator=evaluator,
        parent_node=state_node,
        node_id=f"{node_name_prefix}_Degree_Requirement",
        desc=deg_desc,
        claim=deg_claim,
        sources=degree.sources if degree else None,
        additional_instruction=f"{common_ctx} Verify that the page states the minimum degree (e.g., Bachelor's degree). Allow paraphrases and formatting variations."
    )

    # 2) Exam requirements
    exams = data.exam_requirements if data else None
    ex_desc = f"List all required examination(s) for initial secondary teaching certification in {state_name}"
    ex_claim = f"In {state_name}, the required examination(s) that must be passed for initial secondary (grades 6–12) teacher certification are: {exams.content if exams and exams.content else ''}"
    await _add_and_verify(
        evaluator=evaluator,
        parent_node=state_node,
        node_id=f"{node_name_prefix}_Exam_Requirements",
        desc=ex_desc,
        claim=ex_claim,
        sources=exams.sources if exams else None,
        additional_instruction=f"{common_ctx} Confirm the exam(s) truly required for INITIAL certification (e.g., subject-area/content tests, pedagogy). "
                               f"Accept naming variants and family names (e.g., FTCE, TExES, CSET). Judge incorrect if a distinct required exam type is missing."
    )

    # 3) Student teaching / clinical practice
    stu = data.student_teaching if data else None
    st_desc = f"Specify the student teaching or clinical hours requirement for initial secondary teaching certification in {state_name}"
    st_claim = f"In {state_name}, the student teaching or clinical practice requirement for initial secondary (grades 6–12) certification is: {stu.content if stu and stu.content else ''}"
    await _add_and_verify(
        evaluator=evaluator,
        parent_node=state_node,
        node_id=f"{node_name_prefix}_Student_Teaching",
        desc=st_desc,
        claim=st_claim,
        sources=stu.sources if stu else None,
        additional_instruction=f"{common_ctx} Verify that supervised student teaching/clinical practice is required; if hours are stated, ensure the figure matches. "
                               f"Accept synonyms like practicum, clinical residency, internship when equivalent."
    )

    # 4) Background check
    bg = data.background_check if data else None
    bg_desc = f"Confirm whether a background check is required for initial secondary teaching certification in {state_name}"
    bg_claim = f"In {state_name}, background check/fingerprinting requirement for initial secondary (grades 6–12) certification: {bg.content if bg and bg.content else ''}"
    await _add_and_verify(
        evaluator=evaluator,
        parent_node=state_node,
        node_id=f"{node_name_prefix}_Background_Check",
        desc=bg_desc,
        claim=bg_claim,
        sources=bg.sources if bg else None,
        additional_instruction=f"{common_ctx} Determine if a criminal history background check and/or fingerprinting is required. Accept synonyms."
    )

    # 5) Application fee
    fee = data.application_fee if data else None
    fee_desc = f"State the application fee for initial secondary teaching certification in {state_name}"
    fee_claim = f"In {state_name}, the application/issuance fee for initial secondary (grades 6–12) teacher certification is: {fee.content if fee and fee.content else ''}"
    await _add_and_verify(
        evaluator=evaluator,
        parent_node=state_node,
        node_id=f"{node_name_prefix}_Application_Fee",
        desc=fee_desc,
        claim=fee_claim,
        sources=fee.sources if fee else None,
        additional_instruction=f"{common_ctx} Verify the base application or certificate issuance fee for initial certification. "
                               f"Ignore vendor processing fees unless explicitly included in the stated amount. Minor rounding differences are acceptable."
    )

    # 6) Continuing education / professional development for first renewal
    ce = data.continuing_education if data else None
    ce_desc = f"Specify the continuing education or professional development hours required for the first renewal of secondary teaching certification in {state_name}"
    ce_claim = f"In {state_name}, the requirement for the FIRST certification renewal (continuing education/professional development) is: {ce.content if ce and ce.content else ''}"
    await _add_and_verify(
        evaluator=evaluator,
        parent_node=state_node,
        node_id=f"{node_name_prefix}_Continuing_Education",
        desc=ce_desc,
        claim=ce_claim,
        sources=ce.sources if ce else None,
        additional_instruction=f"{common_ctx} Verify the requirement for the first renewal (hours, points, or credits). Accept equivalent units if numbers align and the policy clearly applies to the first renewal."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the teaching certification requirements comparison task.
    """
    # Initialize evaluator (root is always non-critical by framework design)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=CertificationExtraction,
        extraction_name="requirements_extraction"
    )

    # Build verification subtrees per state according to the rubric
    state_meta = [
        # (Display Name, State node ID in rubric, Node name prefix in rubric, attribute key in extraction)
        ("Texas", "Texas_Requirements", "Texas", "texas"),
        ("Florida", "Florida_Requirements", "Florida", "florida"),
        ("California", "California_Requirements", "California", "california"),
        ("North Carolina", "North_Carolina_Requirements", "North_Carolina", "north_carolina"),
    ]

    for display, state_node_id, node_prefix, attr_key in state_meta:
        state_data: Optional[StateRequirements] = getattr(extracted, attr_key, None)
        await verify_state_requirements(
            evaluator=evaluator,
            root=root,
            state_name=display,
            state_node_id=state_node_id,
            node_name_prefix=node_prefix,
            data=state_data
        )

    # Return structured evaluation summary
    return evaluator.get_summary()