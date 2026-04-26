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
TASK_ID = "eng_edu_licensure_us"
TASK_DESCRIPTION = (
    "Engineering education and professional licensure in the United States are governed by specific standards and requirements. "
    "Provide the following information about engineering education accreditation and professional engineer licensure requirements:\n\n"
    "1. According to ABET (Accreditation Board for Engineering and Technology) criteria for accrediting engineering programs, what is the minimum number of semester credit hours (or equivalent) of mathematics and basic science required in the curriculum?\n\n"
    "2. According to ABET criteria, what is the minimum number of semester credit hours (or equivalent) of engineering topics required in the curriculum?\n\n"
    "3. What is the standard minimum number of years of post-degree engineering experience typically required for Professional Engineer (PE) licensure in the United States?\n\n"
    "4. In Missouri, how many Professional Development Hours (PDH) must licensed professional engineers complete for license renewal?\n\n"
    "5. In Missouri, what is the length of the renewal period for professional engineer continuing education requirements?\n\n"
    "6. In Missouri, for how many years must professional engineers retain documentation of their continuing education for potential audit purposes?\n\n"
    "For each requirement, provide the specific numerical value and include a reference URL from an official source (such as ABET's official criteria documents, NCEES resources, or Missouri's state licensing board) that verifies the requirement."
)

# Ground truth expectations (used for context recording)
GROUND_TRUTH_EXPECTATIONS = {
    "abet_math_basic_science_minimum": "30 semester credit hours (or equivalent) of mathematics and basic science",
    "abet_engineering_topics_minimum": "45 semester credit hours (or equivalent) of engineering topics",
    "pe_experience_minimum": "4 years of qualifying post-degree engineering experience",
    "missouri_pdh_amount": "30 PDH per renewal period",
    "missouri_renewal_period": "2 years (biennial)",
    "missouri_retention_period": "4 years following license renewal"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementItem(BaseModel):
    value_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class RequirementsExtraction(BaseModel):
    abet_math_basic_science: Optional[RequirementItem] = None
    abet_engineering_topics: Optional[RequirementItem] = None
    pe_experience_years: Optional[RequirementItem] = None
    mo_pdh: Optional[RequirementItem] = None
    mo_renewal_period: Optional[RequirementItem] = None
    mo_retention_years: Optional[RequirementItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return (
        "Extract the specific values and the reference URLs for each requirement stated in the answer. "
        "Only extract what is explicitly present in the answer text.\n\n"
        "For each item, return two fields:\n"
        "- value_text: A concise text capturing the numeric value with appropriate units (e.g., '30 semester credit hours', '45 credits', '4 years', '30 PDH', '2 years', '4 years'). If missing, return null.\n"
        "- urls: A list of all URLs the answer cites to support that specific item (official sources preferred). If none are cited, return an empty list.\n\n"
        "Items to extract and their JSON keys:\n"
        "- abet_math_basic_science: The ABET minimum for mathematics and basic science credit hours.\n"
        "- abet_engineering_topics: The ABET minimum for engineering topics credit hours.\n"
        "- pe_experience_years: The typical minimum years of post-degree engineering experience for PE licensure in the U.S.\n"
        "- mo_pdh: The number of PDH required for Missouri PE license renewal.\n"
        "- mo_renewal_period: The renewal period length for Missouri PE continuing education requirements.\n"
        "- mo_retention_years: The number of years Missouri PEs must retain CE documentation for audit purposes.\n\n"
        "Rules:\n"
        "1) Extract only what the answer states; do not infer or invent numbers or URLs.\n"
        "2) URLs may appear as raw links or markdown links; include the actual URL strings.\n"
        "3) If a numeric value appears in words (e.g., 'thirty'), convert it to a clear numeric phrase like '30' followed by the appropriate unit.\n"
        "4) If the answer provides multiple URLs for an item, include them all in the 'urls' list.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(item: Optional[RequirementItem]) -> List[str]:
    return item.urls if (item and item.urls) else []


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_abet_minimums(evaluator: Evaluator, parent_node, extracted: RequirementsExtraction) -> None:
    abet_node = evaluator.add_parallel(
        id="ABET_Curriculum_Minimums",
        desc="ABET minimum credit-hour requirements (math/basic science; engineering topics), each with an official ABET citation",
        parent=parent_node,
        critical=True,
    )

    # Math & Basic Science
    math_node = evaluator.add_parallel(
        id="ABET_Math_and_Basic_Science_Minimum",
        desc="Minimum semester credit hours (or equivalent) of mathematics and basic science required by ABET",
        parent=abet_node,
        critical=True,
    )

    # Leaf: Value stated as 30
    math_value_leaf = evaluator.add_leaf(
        id="ABET_Math_Basic_Credit_Hours_Value",
        desc="States the minimum math/basic science credit-hour requirement as 30 semester credit hours (or equivalent)",
        parent=math_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that ABET requires a minimum of 30 semester credit hours (or equivalent) "
            "of mathematics and basic science in the engineering curriculum."
        ),
        node=math_value_leaf,
        additional_instruction=(
            "Check the answer text for an explicit statement of '30' credits/hours for math and basic science. "
            "Accept equivalent phrasings like '30 credit hours', 'at least 30 credits', or similar."
        ),
    )

    # Leaf: Reference URL supports 30 requirement
    math_ref_leaf = evaluator.add_leaf(
        id="ABET_Math_Basic_Reference_URL",
        desc="Provides an official ABET criteria document URL supporting the math/basic science minimum",
        parent=math_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "At least one of the provided URLs is an official ABET criteria document (on abet.org) and clearly supports "
            "that the minimum mathematics and basic science requirement is 30 semester credit hours (or equivalent)."
        ),
        node=math_ref_leaf,
        sources=_safe_urls(extracted.abet_math_basic_science),
        additional_instruction=(
            "Verify within the provided webpage(s) that the ABET criteria explicitly mention a minimum of 30 semester credit hours "
            "for mathematics and basic sciences (combined). Prefer official ABET criteria pages or PDFs on abet.org."
        ),
    )

    # Engineering Topics
    topics_node = evaluator.add_parallel(
        id="ABET_Engineering_Topics_Minimum",
        desc="Minimum semester credit hours (or equivalent) of engineering topics required by ABET",
        parent=abet_node,
        critical=True,
    )

    # Leaf: Value stated as 45
    topics_value_leaf = evaluator.add_leaf(
        id="ABET_Engineering_Topics_Credit_Hours_Value",
        desc="States the minimum engineering topics credit-hour requirement as 45 semester credit hours (or equivalent)",
        parent=topics_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that ABET requires a minimum of 45 semester credit hours (or equivalent) of engineering topics in the curriculum."
        ),
        node=topics_value_leaf,
        additional_instruction=(
            "Check the answer text for an explicit statement of '45' credits/hours for engineering topics. "
            "Accept equivalent phrasings like '45 credit hours', 'at least 45 credits', or similar."
        ),
    )

    # Leaf: Reference URL supports 45 requirement
    topics_ref_leaf = evaluator.add_leaf(
        id="ABET_Engineering_Topics_Reference_URL",
        desc="Provides an official ABET criteria document URL supporting the engineering topics minimum",
        parent=topics_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "At least one of the provided URLs is an official ABET criteria document (on abet.org) and clearly supports "
            "that the minimum engineering topics requirement is 45 semester credit hours (or equivalent)."
        ),
        node=topics_ref_leaf,
        sources=_safe_urls(extracted.abet_engineering_topics),
        additional_instruction=(
            "Verify within the provided webpage(s) that the ABET criteria explicitly mention a minimum of 45 semester credit hours for engineering topics. "
            "Prefer official ABET criteria pages or PDFs on abet.org."
        ),
    )


async def verify_pe_experience(evaluator: Evaluator, parent_node, extracted: RequirementsExtraction) -> None:
    pe_node = evaluator.add_parallel(
        id="PE_Licensure_Experience",
        desc="Typical minimum post-degree engineering experience required for PE licensure in the U.S., with an authoritative citation",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Value stated as 4 years
    exp_value_leaf = evaluator.add_leaf(
        id="PE_Experience_Duration_Value",
        desc="States the typical minimum experience requirement as 4 years of qualifying post-degree engineering experience",
        parent=pe_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that the typical minimum experience required for PE licensure in the United States is 4 years of qualifying post-degree engineering experience."
        ),
        node=exp_value_leaf,
        additional_instruction=(
            "Check the answer text for an explicit '4 years' experience requirement. Accept equivalent phrasing like "
            "'four years of progressive engineering experience.'"
        ),
    )

    # Leaf: Reference URL supports 4-year requirement
    exp_ref_leaf = evaluator.add_leaf(
        id="PE_Experience_Duration_Reference_URL",
        desc="Provides a reference URL from an authoritative source (e.g., NCEES or another recognized licensing authority) supporting the experience requirement",
        parent=pe_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "At least one of the provided URLs from an authoritative licensing source (e.g., NCEES or a state licensing board) "
            "explicitly supports that 4 years of engineering experience is the typical minimum requirement for PE licensure in the U.S."
        ),
        node=exp_ref_leaf,
        sources=_safe_urls(extracted.pe_experience_years),
        additional_instruction=(
            "Prefer sources such as ncees.org or official state board sites. The page should clearly state that 4 years of qualifying/progressive engineering experience is typically required."
        ),
    )


async def verify_missouri_requirements(evaluator: Evaluator, parent_node, extracted: RequirementsExtraction) -> None:
    mo_node = evaluator.add_parallel(
        id="Missouri_PE_Renewal_and_CE_Requirements",
        desc="Missouri PE renewal/continuing education requirements (PDH amount, renewal period length, and retention period), each with an official Missouri source citation",
        parent=parent_node,
        critical=True,
    )

    # Missouri PDH Amount
    pdh_node = evaluator.add_parallel(
        id="Missouri_PDH_Amount",
        desc="Number of PDH required for Missouri PE license renewal",
        parent=mo_node,
        critical=True,
    )

    pdh_value_leaf = evaluator.add_leaf(
        id="Missouri_PDH_Value",
        desc="States the Missouri PDH requirement as 30 PDH per renewal period",
        parent=pdh_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that Missouri requires 30 Professional Development Hours (PDH) per renewal period for licensed professional engineers."
        ),
        node=pdh_value_leaf,
        additional_instruction=(
            "Check the answer text for an explicit '30 PDH' requirement per renewal period for Missouri PE license renewal."
        ),
    )

    pdh_ref_leaf = evaluator.add_leaf(
        id="Missouri_PDH_Reference_URL",
        desc="Provides a reference URL from an official Missouri state licensing/regulatory source supporting the PDH requirement",
        parent=pdh_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "At least one of the provided URLs from an official Missouri state licensing/regulatory source clearly supports that Missouri requires 30 PDH per renewal period for PE license renewal."
        ),
        node=pdh_ref_leaf,
        sources=_safe_urls(extracted.mo_pdh),
        additional_instruction=(
            "Prefer official Missouri domains (e.g., pr.mo.gov or other state regulatory sites). The page should explicitly mention the 30 PDH requirement."
        ),
    )

    # Missouri Renewal Period Length
    period_node = evaluator.add_parallel(
        id="Missouri_Renewal_Period_Length",
        desc="Length of the Missouri renewal period for PE continuing education requirements",
        parent=mo_node,
        critical=True,
    )

    period_value_leaf = evaluator.add_leaf(
        id="Missouri_Renewal_Period_Value",
        desc="States the Missouri renewal period length as 2 years (biennial/every two years)",
        parent=period_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that Missouri's renewal period length for PE continuing education requirements is 2 years (biennial)."
        ),
        node=period_value_leaf,
        additional_instruction=(
            "Check the answer text for '2 years', 'biennial', or 'every two years' referring to the Missouri renewal cycle."
        ),
    )

    period_ref_leaf = evaluator.add_leaf(
        id="Missouri_Renewal_Period_Reference_URL",
        desc="Provides a reference URL from an official Missouri state licensing/regulatory source supporting the renewal period length",
        parent=period_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "At least one of the provided URLs from an official Missouri state licensing/regulatory source clearly supports that the PE renewal period in Missouri is every two years (biennial)."
        ),
        node=period_ref_leaf,
        sources=_safe_urls(extracted.mo_renewal_period),
        additional_instruction=(
            "Prefer official Missouri domains (e.g., pr.mo.gov). The page should explicitly state that the renewal period/cycle is two years."
        ),
    )

    # Missouri Documentation Retention Period
    retention_node = evaluator.add_parallel(
        id="Missouri_Documentation_Retention_Period",
        desc="How long Missouri PEs must retain continuing-education documentation for audit purposes",
        parent=mo_node,
        critical=True,
    )

    retention_value_leaf = evaluator.add_leaf(
        id="Missouri_Retention_Period_Value",
        desc="States the required documentation retention period as 4 years following license renewal (or equivalent official wording indicating a 4-year retention requirement)",
        parent=retention_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that Missouri PEs must retain continuing-education documentation for 4 years following license renewal (or equivalent official wording indicating a 4-year retention requirement)."
        ),
        node=retention_value_leaf,
        additional_instruction=(
            "Check the answer text for '4 years' retention following license renewal, or equivalent phrasing indicating the same duration."
        ),
    )

    retention_ref_leaf = evaluator.add_leaf(
        id="Missouri_Retention_Period_Reference_URL",
        desc="Provides a reference URL from an official Missouri state licensing/regulatory source supporting the retention requirement",
        parent=retention_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "At least one of the provided URLs from an official Missouri state licensing/regulatory source clearly supports that Missouri PEs must retain CE documentation for 4 years following license renewal."
        ),
        node=retention_ref_leaf,
        sources=_safe_urls(extracted.mo_retention_years),
        additional_instruction=(
            "Prefer official Missouri domains (e.g., pr.mo.gov). The page should explicitly state that CE documentation must be retained for 4 years following renewal."
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
) -> Dict[str, Any]:
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

    # Extract structured requirements and URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_extraction",
    )

    # Add a critical top-level node under root (root is non-critical by design)
    main_node = evaluator.add_parallel(
        id="Engineering_Education_and_Licensure_Requirements",
        desc="Evaluation of ABET curriculum minimums and PE licensure/renewal requirements with official citations",
        parent=root,
        critical=True,
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH_EXPECTATIONS,
            "notes": "These are the expected values typically cited for ABET and licensure/renewal requirements."
        },
        gt_type="expected_requirements",
    )

    # Build and verify subtrees
    await verify_abet_minimums(evaluator, main_node, extracted)
    await verify_pe_experience(evaluator, main_node, extracted)
    await verify_missouri_requirements(evaluator, main_node, extracted)

    return evaluator.get_summary()