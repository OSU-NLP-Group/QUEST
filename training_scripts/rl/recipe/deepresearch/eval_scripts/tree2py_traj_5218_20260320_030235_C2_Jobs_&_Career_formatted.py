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
TASK_ID = "iar_requirements"
TASK_DESCRIPTION = """
What are the mandatory requirements that an individual must fulfill to become qualified as an Investment Adviser Representative (IAR) who can legally provide financial advice and manage client investments in the United States? Your answer should include all essential requirements related to education, licensing examinations, regulatory registration, and professional training.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementCheck(BaseModel):
    present: Optional[bool] = None
    snippet: Optional[str] = None


class IARRequirementsExtraction(BaseModel):
    # Education
    bachelors_required: Optional[RequirementCheck] = None
    fields_examples: Optional[RequirementCheck] = None

    # Series 65
    series65_pass: Optional[RequirementCheck] = None
    series65_format_time: Optional[RequirementCheck] = None
    series65_passing_score: Optional[RequirementCheck] = None
    series65_no_sponsor_u10: Optional[RequirementCheck] = None

    # Regulatory registration
    register_state_or_sec: Optional[RequirementCheck] = None

    # Professional training
    training_duration: Optional[RequirementCheck] = None
    training_content: Optional[RequirementCheck] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_iar_requirements() -> str:
    return """
    From the answer text, determine whether each of the following specific statements is explicitly asserted as a mandatory requirement (i.e., the answer says a person "must", "is required to", or an equivalent strong requirement), not merely recommended or optional. For each item, set 'present' to true only if the answer clearly asserts the requirement as mandatory. If present, also provide the most relevant exact quotation snippet from the answer.

    Items to check:
    1) bachelors_required: The answer states that a person must hold a bachelor's degree from an accredited institution to qualify as an IAR.
    2) fields_examples: The answer mentions common relevant fields as examples (e.g., business, social science, or mathematics), without asserting a strict major requirement.

    3) series65_pass: The answer states the person must pass the Series 65 (Uniform Investment Adviser Law Examination).
    4) series65_format_time: The answer states the Series 65 consists of 130 multiple-choice questions and is to be completed in 3 hours (180 minutes).
    5) series65_passing_score: The answer states the passing requirement as at least 92 correct out of 130 (about 70%).
    6) series65_no_sponsor_u10: The answer states that no firm sponsorship is required to register for the Series 65 exam, and Form U10 can be used if the individual is not Form U4 registered.

    7) register_state_or_sec: The answer states that, to manage client investments as an IAR, one must register with state securities regulators (for smaller advisers) or with the U.S. SEC (for larger advisers).

    8) training_duration: The answer states that one must complete long-term supervised on-the-job training under senior advisors, typically exceeding one year.
    9) training_content: The answer states that during training one must learn to build client networks, develop investment portfolios, and perform other essential advisory duties (at least two of these elements must be mentioned).

    Output a JSON object matching the IARRequirementsExtraction schema, with each field containing:
    - present: true/false or null if not mentioned
    - snippet: the exact supporting sentence/phrase from the answer if present, else null

    Notes:
    - Consider synonyms and equivalent phrasings (e.g., "four-year degree" for bachelor's; "no sponsor required" for sponsorship).
    - If the answer only says something is "recommended," "helpful," or "common," do NOT mark present=true unless it is clearly framed as a must/requirement (except for the 'fields_examples' item which only checks if examples are mentioned).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_education(evaluator: Evaluator, parent_node) -> None:
    edu_node = evaluator.add_parallel(
        id="Education",
        desc="Education requirements per constraints",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Bachelor's degree from accredited institution stated as mandatory
    bachelors_leaf = evaluator.add_leaf(
        id="Bachelors_Degree_Accredited",
        desc="States that the individual must hold a bachelor's degree from an accredited institution",
        parent=edu_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly states that to qualify as an Investment Adviser Representative (IAR), "
            "an individual must hold a bachelor's degree from an accredited institution."
        ),
        node=bachelors_leaf,
        additional_instruction=(
            "Judge only whether the answer text makes this a mandatory requirement (look for 'must', "
            "'required', or equivalent). Accept synonyms like 'four-year degree' and 'accredited college/university'. "
            "If the answer frames a degree as merely 'recommended' or 'preferred', mark Incorrect."
        ),
    )

    # Leaf: Mentions common relevant fields as examples (not strict major)
    fields_leaf = evaluator.add_leaf(
        id="Common_Relevant_Fields_Mentioned",
        desc="Mentions that common relevant fields include business, social science, or mathematics (as examples, not necessarily as a strict major requirement)",
        parent=edu_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer mentions common relevant fields as examples such as business, social science, or mathematics, "
            "without claiming that a specific major is strictly required."
        ),
        node=fields_leaf,
        additional_instruction=(
            "Accept if at least two of the example areas (business, social science, mathematics, finance, economics, accounting) "
            "are mentioned as examples or common backgrounds. If the answer instead asserts a strict major requirement, mark Incorrect."
        ),
    )


async def verify_series65(evaluator: Evaluator, parent_node) -> None:
    s65_node = evaluator.add_parallel(
        id="Series_65",
        desc="Series 65 exam and registration-to-test constraints",
        parent=parent_node,
        critical=True,
    )

    # Must pass Series 65
    pass_leaf = evaluator.add_leaf(
        id="Pass_Series_65",
        desc="States that the individual must pass the Series 65 (Uniform Investment Adviser Law Examination)",
        parent=s65_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that an individual must pass the Series 65 (Uniform Investment Adviser Law Examination).",
        node=pass_leaf,
        additional_instruction="Look for explicit 'must pass Series 65' or equivalent mandatory phrasing.",
    )

    # Exam format and time
    fmt_leaf = evaluator.add_leaf(
        id="Exam_Format_Time",
        desc="States that the Series 65 consists of 130 multiple-choice questions to be completed within 3 hours",
        parent=s65_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the Series 65 consists of 130 multiple-choice questions and must be completed in 3 hours (180 minutes).",
        node=fmt_leaf,
        additional_instruction="Allow 'multiple-choice' synonyms; accept '3 hours' or '180 minutes'. Both 130 items and 3 hours must be present.",
    )

    # Passing score
    pass_score_leaf = evaluator.add_leaf(
        id="Passing_Score",
        desc="States that the passing requirement is at least 92 correct out of 130 (approximately 70%)",
        parent=s65_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the Series 65 passing requirement is at least 92 correct out of 130 (about 70%).",
        node=pass_score_leaf,
        additional_instruction=(
            "Accept any clear equivalent such as '92/130' or '~70%'. If the answer provides a different cut score (e.g., 94/130 or 72%), mark Incorrect."
        ),
    )

    # Sponsorship and U10
    u10_leaf = evaluator.add_leaf(
        id="No_Sponsorship_And_U10_Option",
        desc="States that no firm sponsorship is required to register for the Series 65 exam, and that Form U10 can be used if the individual is not Form U4 registered",
        parent=s65_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that no firm sponsorship is required to register for the Series 65 exam and that Form U10 may be used "
            "if the person is not registered on Form U4."
        ),
        node=u10_leaf,
        additional_instruction="Both parts must be present: 'no sponsorship required' and 'U10 option if not U4-registered'.",
    )


async def verify_regulatory_registration(evaluator: Evaluator, parent_node) -> None:
    reg_node = evaluator.add_parallel(
        id="Regulatory_Registration",
        desc="Regulatory registration requirement per constraints",
        parent=parent_node,
        critical=True,
    )

    reg_leaf = evaluator.add_leaf(
        id="Register_State_or_SEC",
        desc="States that the individual must register with state securities regulators (for small firms) or with the U.S. SEC (for large firms) when managing client investments",
        parent=reg_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that to manage client investments as an IAR, registration is required with state securities regulators "
            "for smaller advisers, or with the U.S. SEC for larger advisers."
        ),
        node=reg_leaf,
        additional_instruction=(
            "Accept equivalent phrasing about state registration versus SEC registration based on firm size or AUM threshold. "
            "The idea that registration with the appropriate regulator is mandatory must be explicit."
        ),
    )


async def verify_professional_training(evaluator: Evaluator, parent_node) -> None:
    train_node = evaluator.add_parallel(
        id="Professional_Training",
        desc="Supervised training requirements per constraints",
        parent=parent_node,
        critical=True,
    )

    duration_leaf = evaluator.add_leaf(
        id="Supervised_OJT_Duration",
        desc="States that the individual must complete long-term supervised on-the-job training under senior advisors, typically exceeding one year",
        parent=train_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that the individual must complete long-term supervised on-the-job training under senior advisors, "
            "typically exceeding one year."
        ),
        node=duration_leaf,
        additional_instruction="Accept 'more than a year', 'at least one year', or 'typically >1 year' phrasing. It must be framed as a requirement.",
    )

    content_leaf = evaluator.add_leaf(
        id="Training_Content",
        desc="States that during training the individual must learn to build client networks, develop investment portfolios, and perform other essential advisory duties",
        parent=train_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that during training the individual must learn to: build client networks, develop investment portfolios, "
            "and perform other essential advisory duties."
        ),
        node=content_leaf,
        additional_instruction=(
            "Accept if at least two of the three elements are explicitly included (client networking/acquisition, portfolio construction, "
            "advisory duties such as compliance, planning, or client service) and framed as required learning."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the IAR mandatory requirements task.
    """
    # 1) Initialize evaluator
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

    # 2) Extraction (for record-keeping/analysis; verification uses answer directly)
    extracted = await evaluator.extract(
        prompt=prompt_extract_iar_requirements(),
        template_class=IARRequirementsExtraction,
        extraction_name="iar_requirements_extraction",
    )

    # 3) Add ground-truth style checklist (as context of expected assertions)
    evaluator.add_ground_truth(
        {
            "expected_assertions": [
                "Bachelor's degree from an accredited institution (mandatory).",
                "Mentions common fields as examples (business, social science, mathematics).",
                "Must pass the Series 65 exam.",
                "Series 65: 130 MCQs in 3 hours.",
                "Series 65 passing: 92/130 (~70%).",
                "No sponsorship required; can use Form U10 if not U4-registered.",
                "Registration with state regulators or SEC (as applicable).",
                "Long-term supervised OJT typically > 1 year (mandatory).",
                "Training content includes networking, portfolio development, and advisory duties.",
            ]
        },
        gt_type="rubric_expectations",
    )

    # 4) Build rubric tree
    main_node = evaluator.add_parallel(
        id="IAR_Career_Requirements",
        desc="Meets all stated constraints for becoming qualified as an Investment Adviser Representative (IAR) per the provided constraints",
        parent=root,
        critical=True,
    )

    # 5) Run structured verifications
    await verify_education(evaluator, main_node)
    await verify_series65(evaluator, main_node)
    await verify_regulatory_registration(evaluator, main_node)
    await verify_professional_training(evaluator, main_node)

    # 6) Return structured summary
    return evaluator.get_summary()