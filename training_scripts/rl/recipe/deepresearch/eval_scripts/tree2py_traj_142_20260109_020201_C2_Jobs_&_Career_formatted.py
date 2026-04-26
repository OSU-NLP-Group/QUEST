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
TASK_ID = "cpa_state_requirements"
TASK_DESCRIPTION = """A prospective CPA candidate is researching state-specific requirements for taking the Uniform CPA Examination and obtaining licensure. They have found a state with the following specific requirements:

1. The state board requires candidates to maintain continuous physical presence (residency) in the state for at least 120 days within the one-year period preceding the date of their initial CPA examination.

2. For the CPA exam application, the state requires a minimum of three character references, and these references must be residents of the same state who have known the applicant for at least 12 months.

3. After successfully passing the CPA Exam, candidates must complete the 150-semester-hour education requirement by December 31 of the fifth calendar year following successful completion of the examination, or their examination scores will be voided.

Which U.S. state has this specific combination of CPA requirements? Provide the state name and include reference URLs from the state's board of accountancy that confirm each of the three requirements listed above.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CPARequirementsExtraction(BaseModel):
    """Extracted state identification and evidence URLs from the agent's answer."""
    state_name: Optional[str] = None
    residency_urls: List[str] = Field(default_factory=list)
    character_reference_urls: List[str] = Field(default_factory=list)
    deadline_rule_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cpa_state_and_urls() -> str:
    return """
    From the provided answer, extract:
    1. state_name: The name of the U.S. state identified as having the specified CPA requirements.
    2. residency_urls: All URLs cited that specifically support the residency/continuous physical presence requirement (120 days within the one-year period preceding the initial CPA examination).
    3. character_reference_urls: All URLs cited that specifically support the character reference requirement (minimum 3 references, must be residents of the same state, and must have known the applicant for at least 12 months).
    4. deadline_rule_urls: All URLs cited that specifically support the deadline rule (complete the 150 semester hours by December 31 of the fifth calendar year after passing the CPA exam, otherwise exam scores are voided).

    IMPORTANT:
    - Extract URLs exactly as they appear in the answer (including markdown links). Do NOT invent URLs.
    - Include only URLs that the answer provides as evidence. If the answer gives a description without a URL, return an empty list for that category.
    - Prefer official pages from the state's Board of Accountancy or its licensing division, but still extract whatever URLs are present in the answer text.
    - If any field is missing in the answer, set it to null (for state_name) or an empty list (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions to build claims                                            #
# --------------------------------------------------------------------------- #
def build_residency_claim(state_name: Optional[str]) -> str:
    st = (state_name or "the state").strip()
    return (
        f"The official {st} state board of accountancy policy requires candidates to maintain continuous physical "
        f"presence (residency) in {st} for at least 120 days within the one-year period preceding the date of their "
        f"initial CPA examination."
    )


def build_character_refs_claim(state_name: Optional[str]) -> str:
    st = (state_name or "the state").strip()
    return (
        f"For the CPA exam application in {st}, the policy requires a minimum of three character references. "
        f"These references must be residents of {st} and must have known the applicant for at least 12 months."
    )


def build_deadline_rule_claim(state_name: Optional[str]) -> str:
    st = (state_name or "the state").strip()
    return (
        f"After passing the CPA Exam in {st}, candidates must complete the 150-semester-hour education requirement by "
        f"December 31 of the fifth calendar year following successful completion of the examination; otherwise, "
        f"their examination scores will be voided."
    )


def common_additional_instruction(state_name: Optional[str]) -> str:
    st = (state_name or "the state").strip()
    return (
        "Verify that the provided webpage(s) are official pages from the state's Board of Accountancy or its licensing "
        f"authority for {st}. The statement must be explicitly supported by the page content (text or displayed "
        "policy PDFs). Allow minor wording variations, but the numeric/time conditions must match exactly. "
        "If the URLs are not official board pages or do not clearly state the requirement, judge as not supported."
    )


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root: Any,
    extracted: CPARequirementsExtraction,
) -> None:
    """
    Construct the verification tree according to the rubric.
    """
    # Top-level sequential critical node: "State_Identification_and_Verification"
    top_node = evaluator.add_sequential(
        id="State_Identification_and_Verification",
        desc="Correctly identify the U.S. state and verify that its CPA requirements match all specified criteria",
        parent=root,
        critical=True
    )

    # Leaf/custom: "State_Name_Provided"
    state_name_present = bool(extracted.state_name and extracted.state_name.strip())
    evaluator.add_custom_node(
        result=state_name_present,
        id="State_Name_Provided",
        desc="The answer provides the name of a specific U.S. state",
        parent=top_node,
        critical=True
    )

    # Parallel critical node: "Requirements_Verification"
    reqs_node = evaluator.add_parallel(
        id="Requirements_Verification",
        desc="Verify that the identified state's CPA requirements match all three specified criteria",
        parent=top_node,
        critical=True
    )

    # 1) Residency Requirement Verification (leaf)
    residency_leaf = evaluator.add_leaf(
        id="Residency_Requirement_Verification",
        desc="Verify with URL evidence that the identified state requires 120 days of continuous physical presence within one year preceding the initial CPA examination",
        parent=reqs_node,
        critical=True
    )
    residency_claim = build_residency_claim(extracted.state_name)
    await evaluator.verify(
        claim=residency_claim,
        node=residency_leaf,
        sources=extracted.residency_urls,
        additional_instruction=common_additional_instruction(extracted.state_name)
    )

    # 2) Character Reference Requirements Verification (leaf)
    char_ref_leaf = evaluator.add_leaf(
        id="Character_Reference_Requirements_Verification",
        desc="Verify with URL evidence that the identified state requires minimum 3 character references who must be state residents and have known the applicant for at least 12 months",
        parent=reqs_node,
        critical=True
    )
    char_ref_claim = build_character_refs_claim(extracted.state_name)
    await evaluator.verify(
        claim=char_ref_claim,
        node=char_ref_leaf,
        sources=extracted.character_reference_urls,
        additional_instruction=common_additional_instruction(extracted.state_name)
    )

    # 3) Deadline Rule Verification (leaf)
    deadline_leaf = evaluator.add_leaf(
        id="Deadline_Rule_Verification",
        desc="Verify with URL evidence that the identified state requires candidates to complete the 150-semester-hour requirement by December 31 of the fifth calendar year after passing the CPA exam, or scores will be voided",
        parent=reqs_node,
        critical=True
    )
    deadline_claim = build_deadline_rule_claim(extracted.state_name)
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=extracted.deadline_rule_urls,
        additional_instruction=common_additional_instruction(extracted.state_name)
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
    model: str = "o4-mini"
) -> Dict:
    """
    Entry point to evaluate an agent's answer for the CPA state requirements task.
    """
    # Initialize evaluator (framework root node is non-critical; we add our critical task node under it)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root-level aggregation; internal task node handles sequential gating
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_cpa_state_and_urls(),
        template_class=CPARequirementsExtraction,
        extraction_name="cpa_requirements_extraction"
    )

    # Optionally record counts for diagnostic purposes
    evaluator.add_custom_info(
        {
            "state_name": extracted.state_name,
            "residency_url_count": len(extracted.residency_urls),
            "character_reference_url_count": len(extracted.character_reference_urls),
            "deadline_rule_url_count": len(extracted.deadline_rule_urls),
        },
        info_type="diagnostics",
        info_name="extraction_summary"
    )

    # Build verification tree according to rubric and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()