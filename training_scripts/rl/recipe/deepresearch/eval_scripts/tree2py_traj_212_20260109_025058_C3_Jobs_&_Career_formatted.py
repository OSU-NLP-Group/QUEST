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
TASK_ID = "aws_csa_pro_path"
TASK_DESCRIPTION = (
    "You are advising a recent high school graduate who is interested in becoming an AWS Certified Solutions Architect - Professional. "
    "They want to understand the complete career pathway, including educational requirements, work experience milestones, and key knowledge "
    "about the certification exam itself.\n\n"
    "Please provide a comprehensive career roadmap that includes:\n\n"
    "1. Educational Foundation: Specify the type of undergraduate degree typically required or recommended for cloud architecture positions, "
    "and state how many years of full-time study are typically needed to complete such a degree.\n\n"
    "2. Professional Experience: Indicate the minimum number of years of hands-on experience with AWS services that is recommended before "
    "attempting the AWS Certified Solutions Architect - Professional certification, as stated in the official AWS certification guidelines.\n\n"
    "3. Certification Achievement: Provide the full official name of the professional-level AWS Solutions Architect certification and include "
    "a reference URL to its official AWS certification page.\n\n"
    "4. Exam Structure Knowledge: For the AWS Certified Solutions Architect - Professional exam:\n"
    "   - Provide a reference URL to the official exam guide that documents the exam's content domains and weightings\n"
    "   - Identify which of the exam's four content domains has the highest percentage weighting\n"
    "   - State the exact number of task statements defined within that highest-weighted domain according to the official exam guide\n\n"
    "Your response should demonstrate that this career path is achievable through structured progression and that certification candidates must "
    "understand both the prerequisites and the detailed exam structure."
)

# Expected constants (used for simple verifications)
EXPECTED_CERT_NAME = "AWS Certified Solutions Architect - Professional"
EXPECTED_HIGHEST_DOMAIN_NAME = "Domain 2: Design for New Solutions"
EXPECTED_HIGHEST_DOMAIN_WEIGHTING = "29%"
EXPECTED_HIGHEST_DOMAIN_TASK_COUNT = "6"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CareerRoadmapExtraction(BaseModel):
    """
    Unified extraction of the key fields from the answer.
    Note: prefer strings to maximize compatibility with varied answer formats.
    """
    degree_type: Optional[str] = None  # e.g., "Bachelor's in Computer Science/IT or related field"
    degree_duration_years: Optional[str] = None  # e.g., "4 years", "four years"
    experience_years_recommended: Optional[str] = None  # e.g., "2+ years", "two or more years"
    certification_official_name: Optional[str] = None  # e.g., "AWS Certified Solutions Architect - Professional"
    certification_official_page_url: Optional[str] = None  # official AWS page URL
    exam_guide_url: Optional[str] = None  # official Exam Guide URL (can be a PDF)
    highest_domain_name: Optional[str] = None  # e.g., "Domain 2: Design for New Solutions"
    highest_domain_weighting: Optional[str] = None  # e.g., "29%"
    highest_domain_task_count: Optional[str] = None  # e.g., "6"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_roadmap() -> str:
    return (
        "Extract the following fields from the answer exactly as they are stated. If a field is not explicitly present, return null.\n"
        "Fields to extract:\n"
        "1. degree_type: The type of undergraduate degree that is typically required or strongly recommended for cloud architect roles. "
        "   Prefer phrasing like \"Bachelor's in Computer Science, Information Technology, or a related field\" if present.\n"
        "2. degree_duration_years: The typical number of years of full-time study to complete the degree (use a short string, e.g., \"4 years\").\n"
        "3. experience_years_recommended: The minimum years of hands-on AWS experience recommended before attempting the AWS CSA-Pro (short string such as \"2+ years\").\n"
        "4. certification_official_name: The full official name of the professional-level AWS Solutions Architect certification.\n"
        "5. certification_official_page_url: The URL to the official AWS certification page for the professional-level Solutions Architect certification.\n"
        "6. exam_guide_url: The URL to the official AWS Certified Solutions Architect - Professional exam guide that documents domains and weightings (often a PDF).\n"
        "7. highest_domain_name: The name of the exam domain with the highest percentage weighting (e.g., \"Domain 2: Design for New Solutions\").\n"
        "8. highest_domain_weighting: The percentage weighting of the highest-weighted domain (e.g., \"29%\" as a string).\n"
        "9. highest_domain_task_count: The exact number of task statements in that highest-weighted domain (return as a short string like \"6\").\n\n"
        "URL extraction rules:\n"
        "- Only extract valid URLs explicitly present in the answer text. Do not invent URLs.\n"
        "- Include full URLs including protocol. If missing, prepend http://.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _prefer_exam_guide_or_cert_url(extracted: CareerRoadmapExtraction) -> Optional[str]:
    """Return the exam guide URL if present; otherwise return the certification page URL."""
    return extracted.exam_guide_url or extracted.certification_official_page_url


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_education_nodes(evaluator: Evaluator, parent_node, extracted: CareerRoadmapExtraction) -> None:
    """
    Build and verify the 'Educational_Foundation' parallel node:
    - Undergraduate_Degree_Type (critical leaf)
    - Undergraduate_Degree_Duration (critical leaf)
    """
    edu_node = evaluator.add_parallel(
        id="Educational_Foundation",
        desc="Checks the required educational foundation details.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Undergraduate_Degree_Type
    degree_type_node = evaluator.add_leaf(
        id="Undergraduate_Degree_Type",
        desc="States that a bachelor's degree in computer science, information technology, or a related field is typically required or strongly recommended for cloud architect positions.",
        parent=edu_node,
        critical=True
    )
    # Claim is general; allow synonyms and reasonable equivalents
    degree_type_claim = (
        "The response states that a bachelor's degree in computer science, information technology (IT), "
        "software engineering, information systems, or a closely related field is typically required or strongly "
        "recommended for cloud architect positions."
    )
    await evaluator.verify(
        claim=degree_type_claim,
        node=degree_type_node,
        additional_instruction=(
            "Judge based on the answer text. Accept reasonable synonyms (e.g., CS, IT, IS, software engineering) and phrasing like "
            "'recommended/strongly recommended' or 'typically required'. Minor variations are acceptable."
        ),
    )

    # Leaf: Undergraduate_Degree_Duration
    degree_duration_node = evaluator.add_leaf(
        id="Undergraduate_Degree_Duration",
        desc="States that a traditional bachelor's degree typically takes four years of full-time study to complete.",
        parent=edu_node,
        critical=True
    )
    degree_duration_claim = (
        "The response states that a traditional bachelor's degree typically takes four years of full-time study to complete."
    )
    await evaluator.verify(
        claim=degree_duration_claim,
        node=degree_duration_node,
        additional_instruction=(
            "Judge based on the answer text. Accept variants like 'approximately four years' or 'around 4 years'. "
            "If the answer indicates a different duration without qualifying 'typical', mark incorrect."
        ),
    )


async def build_experience_node(evaluator: Evaluator, parent_node, extracted: CareerRoadmapExtraction) -> None:
    """
    Build and verify the single critical leaf 'Professional_Experience_Requirement'
    which checks that the answer states AWS recommends 2+ years of hands-on AWS experience
    before attempting CSA-Pro (as per official guidelines).
    """
    exp_node = evaluator.add_leaf(
        id="Professional_Experience_Requirement",
        desc="States that AWS Certified Solutions Architect - Professional recommends 2 or more years of hands-on experience using AWS services to design and implement cloud solutions (per official AWS guidelines).",
        parent=parent_node,
        critical=True
    )
    exp_claim = (
        "The response states that the AWS Certified Solutions Architect - Professional recommends at least two (2) years "
        "of hands-on experience with AWS services to design and implement solutions before attempting the exam."
    )
    # Provide an official source if available, but verify the statement primarily against the answer text.
    exp_source = _prefer_exam_guide_or_cert_url(extracted)
    await evaluator.verify(
        claim=exp_claim,
        node=exp_node,
        sources=exp_source,
        additional_instruction=(
            "First, confirm the answer explicitly states a recommendation of two or more years of hands-on AWS experience. "
            "If a source URL is provided, cross-check it to ensure this recommendation aligns with the official AWS guidelines; "
            "otherwise, judge based on the answer alone."
        ),
    )


async def build_certification_nodes(evaluator: Evaluator, parent_node, extracted: CareerRoadmapExtraction) -> None:
    """
    Build and verify the 'Certification_Achievement' parallel node:
    - Certification_Official_Name (critical leaf)
    - Certification_Official_Page_URL (critical leaf)
    """
    cert_node = evaluator.add_parallel(
        id="Certification_Achievement",
        desc="Checks the required certification identification details.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Certification_Official_Name
    cert_name_node = evaluator.add_leaf(
        id="Certification_Official_Name",
        desc="Provides the full official name: AWS Certified Solutions Architect - Professional.",
        parent=cert_node,
        critical=True
    )
    provided_name = extracted.certification_official_name or ""
    cert_name_claim = (
        f"The certification name '{provided_name}' matches the official name '{EXPECTED_CERT_NAME}'. "
        "Minor punctuation or case variations are acceptable if they refer to the same certification."
    )
    await evaluator.verify(
        claim=cert_name_claim,
        node=cert_name_node,
        additional_instruction=(
            "Verify that the response's certification name corresponds to the official name. "
            "Allow minor punctuation/case variants (e.g., hyphens, capitalization). "
            "Do not accept different certifications (e.g., Associate level)."
        ),
    )

    # Leaf: Certification_Official_Page_URL
    cert_url_node = evaluator.add_leaf(
        id="Certification_Official_Page_URL",
        desc="Provides a valid URL to the official AWS certification page for AWS Certified Solutions Architect - Professional.",
        parent=cert_node,
        critical=True
    )
    cert_url = extracted.certification_official_page_url or ""
    cert_url_claim = (
        "This webpage is the official AWS certification page for the AWS Certified Solutions Architect - Professional."
    )
    await evaluator.verify(
        claim=cert_url_claim,
        node=cert_url_node,
        sources=cert_url,
        additional_instruction=(
            "Confirm the URL is on an official AWS domain (e.g., aws.amazon.com) and the page clearly references the "
            "AWS Certified Solutions Architect - Professional certification."
        ),
    )


async def build_exam_structure_nodes(evaluator: Evaluator, parent_node, extracted: CareerRoadmapExtraction) -> None:
    """
    Build and verify the 'Exam_Structure_Knowledge' parallel node:
    - Official_Exam_Guide_URL (critical leaf)
    - Highest_Weighted_Domain_Name (critical leaf)
    - Highest_Weighted_Domain_Weighting (critical leaf)
    - Highest_Weighted_Domain_Task_Statements_Count (critical leaf)
    """
    exam_node = evaluator.add_parallel(
        id="Exam_Structure_Knowledge",
        desc="Checks the required exam guide reference and highest-weighted domain details.",
        parent=parent_node,
        critical=True
    )

    exam_guide_url = extracted.exam_guide_url or ""

    # Leaf: Official_Exam_Guide_URL
    guide_leaf = evaluator.add_leaf(
        id="Official_Exam_Guide_URL",
        desc="Provides a valid URL to the official AWS Certified Solutions Architect - Professional exam guide that documents the exam domains and weightings.",
        parent=exam_node,
        critical=True
    )
    guide_claim = (
        "This URL points to the official AWS Certified Solutions Architect - Professional exam guide that documents the exam domains and their weightings."
    )

    # Other leaves that depend on the exam guide
    domain_name_leaf = evaluator.add_leaf(
        id="Highest_Weighted_Domain_Name",
        desc="Identifies Domain 2 (Design for New Solutions) as the highest-weighted exam domain.",
        parent=exam_node,
        critical=True
    )
    domain_name_claim = (
        "According to the official exam guide, the highest-weighted domain is Domain 2: Design for New Solutions."
    )

    weighting_leaf = evaluator.add_leaf(
        id="Highest_Weighted_Domain_Weighting",
        desc="States that the highest-weighted domain has a weighting of 29%.",
        parent=exam_node,
        critical=True
    )
    weighting_claim = (
        "According to the official exam guide, the highest-weighted domain has a weighting of 29%."
    )

    task_count_leaf = evaluator.add_leaf(
        id="Highest_Weighted_Domain_Task_Statements_Count",
        desc="States that the highest-weighted domain (Domain 2) contains exactly 6 task statements, per the official exam guide.",
        parent=exam_node,
        critical=True
    )
    task_count_claim = (
        "According to the official exam guide, Domain 2 (Design for New Solutions) contains exactly 6 task statements."
    )

    # Batch verify for efficiency; all use the exam guide URL as source
    claims_and_sources: List[tuple[str, Optional[str], Any, Optional[str]]] = [
        (guide_claim, exam_guide_url, guide_leaf, "Confirm the page or PDF is the official AWS exam guide for CSA-Pro and includes the domain weightings."),
        (domain_name_claim, exam_guide_url, domain_name_leaf, "Check the 'Content outline' or domain listing; verify Domain 2 is the highest-weighted."),
        (weighting_claim, exam_guide_url, weighting_leaf, "Verify the percentage shown for the highest-weighted domain is 29%. Allow minor formatting differences."),
        (task_count_claim, exam_guide_url, task_count_leaf, "Count the task statements listed under Domain 2 in the official guide and confirm there are six."),
    ]
    await evaluator.batch_verify(claims_and_sources)


async def build_structure_check_node(evaluator: Evaluator, parent_node) -> None:
    """
    Build and verify the non-critical leaf 'Roadmap_Is_Structured' which checks that
    the response presents the pathway as a structured progression across the four required areas.
    """
    structured_leaf = evaluator.add_leaf(
        id="Roadmap_Is_Structured",
        desc="Presents the pathway as a structured progression (e.g., clearly separated steps/sections covering education, experience, certification, and exam structure).",
        parent=parent_node,
        critical=False
    )
    structured_claim = (
        "The response presents the pathway as a structured progression with clearly separated sections or steps covering "
        "Educational Foundation, Professional Experience, Certification Achievement, and Exam Structure Knowledge."
    )
    await evaluator.verify(
        claim=structured_claim,
        node=structured_leaf,
        additional_instruction=(
            "Judge based on formatting and organization in the answer. Accept headings, numbered steps, or clearly separated bullet sections "
            "that distinctly cover the four categories."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the AWS Professional Solutions Architect career pathway task.
    """
    # Initialize evaluator with a parallel root to match rubric tree structure
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluate whether the response provides the required education, experience, certification, and exam-structure details for the AWS Certified Solutions Architect - Professional pathway.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract key information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_career_roadmap(),
        template_class=CareerRoadmapExtraction,
        extraction_name="career_roadmap_extraction",
    )

    # Optionally record expected constants as GT info for reference
    evaluator.add_ground_truth({
        "expected_certification_name": EXPECTED_CERT_NAME,
        "expected_highest_domain_name": EXPECTED_HIGHEST_DOMAIN_NAME,
        "expected_highest_domain_weighting": EXPECTED_HIGHEST_DOMAIN_WEIGHTING,
        "expected_highest_domain_task_count": EXPECTED_HIGHEST_DOMAIN_TASK_COUNT,
    }, gt_type="expected_values")

    # Build verification tree following rubric
    await build_education_nodes(evaluator, root, extracted)
    await build_experience_node(evaluator, root, extracted)
    await build_certification_nodes(evaluator, root, extracted)
    await build_exam_structure_nodes(evaluator, root, extracted)
    await build_structure_check_node(evaluator, root)

    # Return structured evaluation summary
    return evaluator.get_summary()