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
TASK_ID = "psu_head_coach_requirements"
TASK_DESCRIPTION = """
What are the minimum education and experience requirements stated in Penn State's official job posting for their Division I Head Coach position?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoachPostingExtraction(BaseModel):
    """Structured extraction of the user's answer for PSU Head Coach posting requirements."""
    job_code: Optional[str] = None  # e.g., "PSU1405"
    position_title: Optional[str] = None  # e.g., "Head Coach (Division I)"
    institution_name: Optional[str] = None  # e.g., "Penn State" or "Pennsylvania State University"
    education_min_requirement: Optional[str] = None  # quoted or paraphrased minimum education requirement
    experience_min_requirement: Optional[str] = None  # quoted or paraphrased minimum experience requirement
    official_posting_urls: List[str] = Field(default_factory=list)  # URLs claimed as the official posting
    other_urls: List[str] = Field(default_factory=list)  # any other URLs (news, third-party, etc.)
    mentions_official_posting: Optional[bool] = None  # whether the answer explicitly says "official posting"
    mentions_division_I: Optional[bool] = None  # whether the answer explicitly says "Division I"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_posting() -> str:
    return """
    Extract from the provided answer the structured information related to Penn State's Division I Head Coach official posting.

    Required fields to extract:
    1) job_code: The job code if explicitly mentioned (e.g., "PSU1405"). If not in the answer, return null.
    2) position_title: The role title as stated in the answer (e.g., "Head Coach (Division I)"). If absent, return null.
    3) institution_name: The institution specified in the answer (e.g., "Penn State", "Pennsylvania State University"). If absent, return null.
    4) education_min_requirement: The stated minimum education requirement exactly as quoted or paraphrased in the answer (do not invent). If not present, return null.
    5) experience_min_requirement: The stated minimum experience requirement exactly as quoted or paraphrased in the answer (do not invent). If not present, return null.
    6) official_posting_urls: All URLs in the answer that appear to be Penn State's official job posting for Head Coach (Division I). Extract only actual URLs explicitly present in the answer text (including markdown links). Do not invent URLs. If none are present, return an empty list.
       Note: These should ideally be on official Penn State domains (e.g., psu.edu, hr.psu.edu). Still, extract whatever the answer provides.
    7) other_urls: Any other URLs mentioned in the answer that are not the official posting (e.g., third-party job boards, news articles). Extract only actual URLs present. Return an empty list if none.
    8) mentions_official_posting: Return true if the answer explicitly indicates using the "official job posting" (even without a URL), otherwise false or null.
    9) mentions_division_I: Return true if the answer explicitly mentions "Division I" in relation to the Head Coach position, otherwise false or null.

    Extraction rules:
    - Do not add, infer, or normalize beyond the answer text. Use the exact text provided by the answer.
    - For URLs, include the full URL string; if protocol is missing, prepend "http://".
    - If a field is absent, return null (for scalar fields) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_head_coach_requirements(
    evaluator: Evaluator,
    root_node,
    extracted: CoachPostingExtraction,
) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    Root node is critical and aggregates children in parallel.
    """

    # Root node representing the entire rubric item
    psu_root = evaluator.add_parallel(
        id="Penn_State_Head_Coach_Requirements",
        desc="Correctly identifies the minimum education and experience requirements for Penn State's Division I Head Coach position, sourced from the official job posting (job code PSU1405).",
        parent=root_node,
        critical=True,
    )

    # 1) Uses Official Posting PSU1405
    uses_official_node = evaluator.add_leaf(
        id="Uses_Official_Posting_PSU1405",
        desc="Answer is sourced from Penn State's official job posting for Head Coach (Division I), job code PSU1405 (e.g., provides a citation/link or otherwise clearly indicates the official posting as the source).",
        parent=psu_root,
        critical=True,
    )

    # Determine claim & sources for official posting usage
    official_urls = extracted.official_posting_urls if extracted and extracted.official_posting_urls else []
    extracted_job_code = (extracted.job_code or "").strip() if extracted else ""
    job_code_display = extracted_job_code if extracted_job_code else "PSU1405"

    if official_urls:
        claim_official = (
            f"This webpage is Penn State's official job posting for Head Coach (Division I), "
            f"and it corresponds to job code '{job_code_display}'."
        )
        add_ins_official = (
            "Verify the page is clearly an official Penn State posting for Head Coach (Division I). "
            "Prefer confirming the job code 'PSU1405' if visible; if the job code is not present but the page "
            "is unambiguously the official Penn State posting for Head Coach (Division I), consider it acceptable. "
            "If the page is irrelevant, third-party, or not obviously official, return NOT SUPPORTED."
        )
        await evaluator.verify(
            claim=claim_official,
            node=uses_official_node,
            sources=official_urls,
            additional_instruction=add_ins_official,
        )
    else:
        # No URLs: judge based on answer text only
        claim_official = (
            "The answer explicitly cites or clearly indicates Penn State's official job posting for Head Coach (Division I), "
            f"with job code '{job_code_display}', as its source."
        )
        add_ins_official = (
            "Use only the answer text provided. "
            "If the answer does not clearly mention using the official posting (e.g., says 'official job posting' or shows 'PSU1405'), "
            "return INCORRECT."
        )
        await evaluator.verify(
            claim=claim_official,
            node=uses_official_node,
            additional_instruction=add_ins_official,
        )

    # 2) Education Minimum Requirement Stated Accurately
    edu_node = evaluator.add_leaf(
        id="Education_Minimum_Requirement_Stated_Accurately",
        desc="States the minimum education requirement exactly as specified in the official job posting.",
        parent=psu_root,
        critical=True,
    )
    education_text = (extracted.education_min_requirement or "").strip() if extracted else ""
    if official_urls:
        claim_edu = (
            f"According to the official Penn State posting for Head Coach (Division I), "
            f"the minimum education requirement is exactly: '{education_text}'."
        )
        add_ins_edu = (
            "Check the official posting content to confirm the stated minimum education requirement. "
            "Be strict: the claim should refer to the baseline minimum requirement (not preferred qualifications), "
            "and should not omit or alter critical qualifiers (e.g., 'or equivalent combination of education and experience'). "
            "Minor punctuation or casing differences are acceptable, but substantive differences should fail."
        )
        await evaluator.verify(
            claim=claim_edu,
            node=edu_node,
            sources=official_urls,
            additional_instruction=add_ins_edu,
        )
    else:
        # No official URLs -> cannot verify accuracy with evidence
        claim_edu = (
            "The stated minimum education requirement exactly matches the official job posting."
        )
        add_ins_edu = (
            "No official posting URLs were provided for verification. "
            "Without evidence from the posting, treat this claim as NOT SUPPORTED and return INCORRECT."
        )
        await evaluator.verify(
            claim=claim_edu,
            node=edu_node,
            additional_instruction=add_ins_edu,
        )

    # 3) Experience Minimum Requirement Stated Accurately
    exp_node = evaluator.add_leaf(
        id="Experience_Minimum_Requirement_Stated_Accurately",
        desc="States the minimum experience requirement exactly as specified in the official job posting.",
        parent=psu_root,
        critical=True,
    )
    experience_text = (extracted.experience_min_requirement or "").strip() if extracted else ""
    if official_urls:
        claim_exp = (
            f"According to the official Penn State posting for Head Coach (Division I), "
            f"the minimum experience requirement is exactly: '{experience_text}'."
        )
        add_ins_exp = (
            "Check the official posting content to confirm the stated minimum experience requirement. "
            "Be strict: the claim should refer to the baseline minimum requirement. "
            "Minor punctuation or casing differences are acceptable, but substantive differences should fail."
        )
        await evaluator.verify(
            claim=claim_exp,
            node=exp_node,
            sources=official_urls,
            additional_instruction=add_ins_exp,
        )
    else:
        # No official URLs -> cannot verify accuracy with evidence
        claim_exp = (
            "The stated minimum experience requirement exactly matches the official job posting."
        )
        add_ins_exp = (
            "No official posting URLs were provided for verification. "
            "Without evidence from the posting, treat this claim as NOT SUPPORTED and return INCORRECT."
        )
        await evaluator.verify(
            claim=claim_exp,
            node=exp_node,
            additional_instruction=add_ins_exp,
        )

    # 4) Position Scope Matches Division I Head Coach
    scope_node = evaluator.add_leaf(
        id="Position_Scope_Matches_Division_I_Head_Coach",
        desc="The stated requirements are explicitly tied to Penn State University's Division I Head Coach position (not a different role or institution).",
        parent=psu_root,
        critical=True,
    )
    pos_title = (extracted.position_title or "").strip() if extracted else ""
    inst_name = (extracted.institution_name or "").strip() if extracted else ""
    if official_urls:
        claim_scope = (
            "This job posting is for Head Coach (Division I) at Penn State University."
        )
        add_ins_scope = (
            "Confirm the job title includes 'Head Coach (Division I)' (or an equivalent phrasing clearly indicating Division I), "
            "and that the institution is Penn State/Pennsylvania State University. "
            "If the page is for a different role or institution, return NOT SUPPORTED."
        )
        await evaluator.verify(
            claim=claim_scope,
            node=scope_node,
            sources=official_urls,
            additional_instruction=add_ins_scope,
        )
    else:
        claim_scope = (
            "The answer explicitly ties the stated requirements to Penn State University's Division I Head Coach position, "
            "and not to a different role or institution."
        )
        add_ins_scope = (
            "Use only the answer text. "
            "If the answer does not clearly say 'Penn State' and 'Division I Head Coach' for the scope, return INCORRECT."
        )
        await evaluator.verify(
            claim=claim_scope,
            node=scope_node,
            additional_instruction=add_ins_scope,
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
    Evaluate an answer for PSU Division I Head Coach minimum requirements.
    """

    # Initialize evaluator and root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root matches rubric: parallel aggregation
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_coach_posting(),
        template_class=CoachPostingExtraction,
        extraction_name="coach_posting_extraction",
    )

    # Add ground-truth-like context info (non-binding)
    evaluator.add_ground_truth({
        "expected_job_code": "PSU1405",
        "position": "Head Coach (Division I)",
        "institution": "Penn State (Pennsylvania State University)",
        "note": "Verification relies on the official posting URLs provided in the answer when available."
    })

    # Build tree and verify
    await verify_head_coach_requirements(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()