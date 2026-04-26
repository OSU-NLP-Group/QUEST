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
TASK_ID = "ohio_advanced_license_eligibility_2026"
TASK_DESCRIPTION = (
    "Determine which Ohio advanced professional educator license the educator is eligible for based on their "
    "qualifications and career history, and provide an Ohio State Board of Education (sboe.ohio.gov) reference URL "
    "supporting the eligibility requirements."
)

EXPECTED_LICENSE = "Lead Professional Educator License"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LicenseExtraction(BaseModel):
    license_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_license_and_urls() -> str:
    return """
    Extract from the answer:
    1) license_name: The exact name of the advanced professional educator license the answer claims the educator is eligible to obtain. If multiple licenses are mentioned, choose the one presented as the final or recommended eligibility outcome. If unclear, choose the first clearly stated as the answer.
    2) reference_urls: A list of all URLs cited as references for eligibility requirements in the answer text. Include every URL string exactly as it appears (plain or in markdown). Do not infer URLs.

    If license_name is not clearly stated, set it to null.
    If no URLs are provided, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _infer_claimed_license_type(license_name: Optional[str]) -> str:
    """
    Infer the claimed license type from the extracted license_name.
    Returns 'lead', 'senior', or 'unknown'.
    """
    if not license_name:
        return "unknown"
    lname = license_name.strip().lower()
    if "lead" in lname:
        return "lead"
    if "senior" in lname:
        return "senior"
    return "unknown"


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def add_prerequisite_checks(evaluator: Evaluator, parent_node) -> None:
    """
    Build and verify prerequisite requirement nodes (all critical) under parent_node.
    Uses simple logical verification based on the scenario facts provided in the task description.
    """
    prereq_node = evaluator.add_parallel(
        id="Prerequisite_Requirements",
        desc="Verify that the educator meets all prerequisite requirements for Ohio advanced professional educator licenses",
        parent=parent_node,
        critical=True
    )

    # Degree requirement (critical leaf)
    degree_leaf = evaluator.add_leaf(
        id="Degree_Requirement",
        desc="The educator holds a bachelor's degree from a regionally accredited institution (or equivalent for foreign degrees)",
        parent=prereq_node,
        critical=True
    )
    degree_claim = (
        "According to the task scenario, the educator holds a bachelor's degree in education from Ohio State University, "
        "which is a regionally accredited institution."
    )
    await evaluator.verify(
        claim=degree_claim,
        node=degree_leaf,
        additional_instruction=(
            "This is a logical verification based solely on the task description. "
            "Treat the scenario facts as given and true; do not require external sources."
        )
    )

    # Experience requirements (parallel, both critical sub-leaves)
    exp_node = evaluator.add_parallel(
        id="Experience_Requirements",
        desc="Verify that the educator meets all teaching experience requirements",
        parent=prereq_node,
        critical=True
    )

    # Total experience requirement
    total_exp_leaf = evaluator.add_leaf(
        id="Total_Experience_Requirement",
        desc="The educator has completed 9 years of teaching experience under a standard teaching license or certificate",
        parent=exp_node,
        critical=True
    )
    total_exp_claim = (
        "According to the task scenario, the educator has at least 9 years of teaching experience under a standard "
        "teaching license or certificate: specifically nine years (2015–2024) under a Five-Year Professional Teaching License, "
        "after two years on a Resident Educator License."
    )
    await evaluator.verify(
        claim=total_exp_claim,
        node=total_exp_leaf,
        additional_instruction=(
            "This is a logical verification based on the scenario facts. "
            "Interpret 'standard teaching license or certificate' to include the Five-Year Professional Teaching License."
        )
    )

    # Professional license experience requirement
    pro_exp_leaf = evaluator.add_leaf(
        id="Professional_License_Experience",
        desc="At least 5 of the 9 years of teaching experience are under a professional or permanent license or certificate",
        parent=exp_node,
        critical=True
    )
    pro_exp_claim = (
        "According to the task scenario, at least five of the nine required years were under a professional or permanent "
        "license: the educator held a Five-Year Professional Teaching License from 2015 through 2024 (nine years)."
    )
    await evaluator.verify(
        claim=pro_exp_claim,
        node=pro_exp_leaf,
        additional_instruction=(
            "This is a logical verification based on the scenario facts. "
            "Treat the time on the Five-Year Professional Teaching License (2015–2024) as satisfying the professional/permanent requirement."
        )
    )

    # Distinguished or accomplished-level performance credential
    distinguished_leaf = evaluator.add_leaf(
        id="Distinguished_Performance_Credential",
        desc="The educator holds credentials demonstrating distinguished or accomplished level performance (Master Teacher designation, National Board Certification, or Master Teacher designation with Teacher Leader Endorsement)",
        parent=prereq_node,
        critical=True
    )
    distinguished_claim = (
        "According to the task scenario, the educator holds an active Master Teacher designation (renewed in 2025) "
        "and also holds a Teacher Leader Endorsement from an accredited Ohio university."
    )
    await evaluator.verify(
        claim=distinguished_claim,
        node=distinguished_leaf,
        additional_instruction=(
            "This is a logical verification based on the scenario facts. "
            "Recognize that Master Teacher designation together with a Teacher Leader Endorsement constitutes a qualifying distinguished pathway."
        )
    )


async def add_license_determination_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: LicenseExtraction
) -> None:
    """
    Build and verify the license determination nodes under parent_node (all critical).
    - Check that the license identified in the answer is the correct one for the scenario.
    - Check that the provided URL(s) from the answer include an sboe.ohio.gov page that supports the eligibility determination.
    """
    license_node = evaluator.add_parallel(
        id="License_Determination",
        desc=(
            "Correctly identifies which specific advanced professional educator license the educator qualifies for "
            "based on their credentials: Senior Professional Educator License (requires Master Teacher designation) or "
            "Lead Professional Educator License (requires either active National Board Certification OR both Master "
            "Teacher designation AND Teacher Leader Endorsement)"
        ),
        parent=parent_node,
        critical=True
    )

    # Leaf: Correct license determination (critical)
    license_name_leaf = evaluator.add_leaf(
        id="License_Name_Correct",
        desc="Correct license is identified based on the scenario (should be Lead Professional Educator License for this profile)",
        parent=license_node,
        critical=True
    )

    provided_name = extracted.license_name or ""
    claimed_type = _infer_claimed_license_type(provided_name)

    # Formulate a robust claim that connects the expected outcome to the stated license name
    license_name_claim = (
        f"The answer states the educator is eligible for '{provided_name}'. "
        f"Given the scenario facts (active Master Teacher designation and a Teacher Leader Endorsement, "
        f"plus sufficient experience and a degree), the correct Ohio advanced professional educator license "
        f"is the '{EXPECTED_LICENSE}'. The stated license matches this (allowing reasonable naming variants)."
    )
    await evaluator.verify(
        claim=license_name_claim,
        node=license_name_leaf,
        additional_instruction=(
            "Judge this as Correct only if the provided license name is effectively the 'Lead Professional Educator License' "
            "(allowing minor naming variants like 'Lead Professional License' or 'Lead Professional Educator teaching license'). "
            "If the provided license name indicates 'Senior' instead of 'Lead' or is missing/ambiguous, mark Incorrect."
        )
    )

    # Leaf: Reference URL support (critical)
    reference_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provides a reference URL from the Ohio State Board of Education (sboe.ohio.gov) that supports the license eligibility determination",
        parent=license_node,
        critical=True
    )

    # Build a claim tailored to the provided license name (lead vs senior). If unknown, still require sboe page about lead.
    if claimed_type == "senior":
        url_claim = (
            "This page is hosted on sboe.ohio.gov and it describes eligibility requirements for the Senior Professional Educator License in Ohio, "
            "including that a Master Teacher designation is required for eligibility."
        )
    else:
        # Default to Lead (expected)
        url_claim = (
            "This page is hosted on sboe.ohio.gov and it describes eligibility requirements for the Lead Professional Educator License in Ohio, "
            "including that candidates may qualify with either an active National Board Certification OR with both a Master Teacher designation "
            "and a Teacher Leader Endorsement."
        )

    # Use all extracted URLs; the verification will pass if any is an sboe.ohio.gov page that supports the claim.
    urls_to_check: List[str] = extracted.reference_urls or []

    # If no URLs are present, the verification should fail. Direct the judge accordingly.
    add_ins = (
        "Verify BOTH of the following:\n"
        "1) The URL is on the domain sboe.ohio.gov.\n"
        "2) The page content supports the described eligibility for the specified license.\n"
        "If the answer provides multiple URLs, pass if at least one URL satisfies both conditions. "
        "If no URL is provided or none are from sboe.ohio.gov, mark as not supported."
    )

    await evaluator.verify(
        claim=url_claim,
        node=reference_leaf,
        sources=urls_to_check if urls_to_check else None,
        additional_instruction=add_ins
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the answer for Ohio advanced professional educator license eligibility.
    """
    # Initialize evaluator (root is non-critical container)
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

    # Extract the claimed license and reference URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_license_and_urls(),
        template_class=LicenseExtraction,
        extraction_name="license_extraction"
    )

    # Add ground truth context (for transparency)
    evaluator.add_ground_truth(
        {
            "expected_license": EXPECTED_LICENSE,
            "scenario_summary": {
                "degree": "Bachelor's from a regionally accredited institution (Ohio State University)",
                "experience": "11 years total; 9 years under Five-Year Professional Teaching License (2015–2024)",
                "distinguished": "Active Master Teacher designation (renewed 2025) and Teacher Leader Endorsement"
            }
        },
        gt_type="expected_outcome"
    )

    # Build the main evaluation branch as a sequential critical node
    adv_node = evaluator.add_sequential(
        id="Advanced_License_Eligibility",
        desc="Determine which Ohio advanced professional educator license the educator is eligible for based on their qualifications and career history",
        parent=root,
        critical=True
    )

    # 1) Prerequisite checks (parallel, all critical)
    await add_prerequisite_checks(evaluator, adv_node)

    # 2) License determination checks (parallel, all critical)
    await add_license_determination_checks(evaluator, adv_node, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()