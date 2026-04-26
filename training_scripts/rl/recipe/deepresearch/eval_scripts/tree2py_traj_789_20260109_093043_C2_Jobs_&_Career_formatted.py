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
TASK_ID = "pmp_pathway_requirements"
TASK_DESCRIPTION = (
    "Alex graduated with a Bachelor's degree in Business Administration and has accumulated 48 months of experience "
    "leading marketing projects over the past 6 years. Alex is planning to apply for the PMP (Project Management "
    "Professional) certification. Based on Alex's educational background: (1) Which specific PMP certification pathway "
    "(four-year degree pathway or high school/associate degree pathway) does Alex qualify for? (2) What is the minimum "
    "project management experience requirement (stated in months) for that specific pathway? (3) How many hours of formal "
    "project management education or training must Alex complete to be eligible for the PMP exam?"
)

# Ground-truth information for reference (recorded in summary)
GROUND_TRUTH = {
    "expected_pathway_for_bachelor": "four-year degree pathway",
    "min_experience_months_four_year": "36",  # 36 months (3 years)
    "pm_education_hours": "35",               # 35 contact hours
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PMRequirementsExtraction(BaseModel):
    """
    Extracted fields from the agent's answer regarding PMP eligibility requirements.
    """
    pathway: Optional[str] = None  # e.g., "four-year degree pathway", "Bachelor's pathway", etc.
    min_experience_months: Optional[str] = None  # stated in months (string). If stated in years, keep textual form.
    training_hours: Optional[str] = None  # stated in hours (string)
    source_urls: List[str] = Field(default_factory=list)  # URLs explicitly listed in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return (
        "From the answer, extract the PMP pathway and the minimum requirements stated. "
        "Return the following fields:\n"
        "1) pathway: The PMP educational pathway identified in the answer for Alex. Use the exact wording from the answer. "
        "   Examples of acceptable wordings include 'four-year degree pathway', 'Bachelor’s pathway', 'high school/associate pathway', etc.\n"
        "2) min_experience_months: The minimum required project management experience stated for Alex’s applicable pathway, in months. "
        "   If the answer only states the requirement in years (e.g., '3 years'), you may keep that textual form (e.g., '3 years'). "
        "   Do not invent numbers. If the answer provides a range or is unclear, return the exact phrase from the answer; if entirely missing, return null.\n"
        "3) training_hours: The required number of hours of formal project management education/training stated in the answer. "
        "   Use the exact number/phrase the answer provides (e.g., '35 hours', '35 contact hours'). If missing, return null.\n"
        "4) source_urls: Extract all URLs explicitly present in the answer that pertain to PMP eligibility or requirements. "
        "   Only include actual URLs. If none are present, return an empty list."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_pmp_requirements(
    evaluator: Evaluator,
    extracted: PMRequirementsExtraction,
) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """
    # Create a critical parallel parent node to mirror the rubric
    top_node = evaluator.add_parallel(
        id="PMP_Certification_Pathway_and_Requirements",
        desc="Determine the correct PMP educational pathway for Alex and report the minimum experience (months) and training hours required for that pathway.",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Educational_Pathway (Critical leaf)
    pathway_leaf = evaluator.add_leaf(
        id="Educational_Pathway",
        desc="Correctly identifies the PMP educational pathway applicable to someone with a Bachelor's degree (i.e., the four-year degree pathway, not the high school/associate pathway).",
        parent=top_node,
        critical=True,
    )
    # Construct claim strictly focusing on the correct pathway given the Bachelor's degree
    # We allow synonyms for 'four-year degree pathway'
    pathway_claim = (
        "Given that the candidate holds a Bachelor's degree (a four-year degree), the correct PMP educational pathway "
        "is the four-year degree pathway (not the high school/associate pathway)."
    )
    await evaluator.verify(
        claim=pathway_claim,
        node=pathway_leaf,
        additional_instruction=(
            "Judge based on the task description (Bachelor's degree). "
            "Pass if the answer explicitly identifies the four-year degree pathway (or synonymous wording such as "
            "'Bachelor’s pathway', 'degree holder pathway'), and does not incorrectly select the high school/associate pathway. "
            "Allow minor wording variations; focus on the pathway semantics."
        ),
    )

    # Prepare sources for subsequent numeric checks
    sources = extracted.source_urls if extracted and extracted.source_urls else None

    # 2) Minimum_PM_Experience_Months (Critical leaf)
    min_exp_leaf = evaluator.add_leaf(
        id="Minimum_PM_Experience_Months",
        desc="States the minimum required project management experience in months for the identified pathway.",
        parent=top_node,
        critical=True,
    )
    # Claim uses the value stated in the answer; verification checks correctness against sources if available,
    # otherwise uses general knowledge (explicitly permitted here).
    min_exp_value = (extracted.min_experience_months or "").strip() if extracted else ""
    min_exp_claim = (
        f"For candidates on the four-year degree pathway, the minimum required project management experience is "
        f"{min_exp_value} months."
    )
    await evaluator.verify(
        claim=min_exp_claim,
        node=min_exp_leaf,
        sources=sources,
        additional_instruction=(
            "Verify whether the number stated in the answer matches the correct PMI requirement for the four-year degree pathway. "
            "As of 2024, the correct minimum is 36 months (3 years). If the answer uses years, treat '3 years' as equivalent to '36 months'. "
            "If source URLs are provided, rely on them primarily. If no sources are available, you may use your general knowledge "
            "of current PMI PMP prerequisites to judge this claim."
        ),
    )

    # 3) Formal_PM_Education_Hours (Critical leaf)
    training_leaf = evaluator.add_leaf(
        id="Formal_PM_Education_Hours",
        desc="States the required number of hours of formal project management education/training needed for PMP eligibility.",
        parent=top_node,
        critical=True,
    )
    training_value = (extracted.training_hours or "").strip() if extracted else ""
    training_claim = (
        f"The required formal project management education/training for PMP eligibility is {training_value} hours."
    )
    await evaluator.verify(
        claim=training_claim,
        node=training_leaf,
        sources=sources,
        additional_instruction=(
            "Verify whether the number stated in the answer matches the correct PMI requirement. "
            "As of 2024, PMP requires 35 contact hours (or a CAPM certification). Accept synonymous phrasing like '35 contact hours'. "
            "If source URLs are provided, rely on them primarily. If none are available, you may use general knowledge "
            "of PMI PMP prerequisites to judge this claim."
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
    Evaluate an agent's answer for the PMP pathway and requirements task using the Mind2Web2 framework.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root node (non-critical), rubric main node will be critical child
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

    # Extract structured fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=PMRequirementsExtraction,
        extraction_name="pmp_requirements_extraction",
    )

    # Record ground-truth info for transparency (not used to auto-judge)
    evaluator.add_ground_truth(
        {
            "expected_pathway_for_bachelor": GROUND_TRUTH["expected_pathway_for_bachelor"],
            "min_experience_months_four_year": GROUND_TRUTH["min_experience_months_four_year"],
            "pm_education_hours": GROUND_TRUTH["pm_education_hours"],
        },
        gt_type="ground_truth",
    )

    # Add custom info for debugging
    evaluator.add_custom_info(
        {
            "extracted_pathway": extracted.pathway,
            "extracted_min_experience_months": extracted.min_experience_months,
            "extracted_training_hours": extracted.training_hours,
            "extracted_source_urls": extracted.source_urls,
        },
        info_type="extraction_debug",
        info_name="extracted_values",
    )

    # Build tree and verify
    await verify_pmp_requirements(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()