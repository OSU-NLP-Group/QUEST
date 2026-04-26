import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nsf_career_2026_requirements"
TASK_DESCRIPTION = (
    "A tenure-track assistant professor in the Computer Science department at a U.S. university is planning to submit "
    "an NSF Faculty Early Career Development (CAREER) award proposal for the 2026 competition. Provide a comprehensive "
    "specification of the NSF CAREER award requirements, including: the award duration and funding structure, "
    "applicant eligibility criteria, required proposal components with their page/character limits, and any submission restrictions."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ClaimWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareerRequirementsExtraction(BaseModel):
    # Award duration and funding
    award_duration: Optional[ClaimWithSources] = None
    min_funding_standard_directorates: Optional[ClaimWithSources] = None
    min_funding_bio_eng_opp: Optional[ClaimWithSources] = None

    # Eligibility
    doctoral_degree_eligibility: Optional[ClaimWithSources] = None
    tenure_track_position_requirement: Optional[ClaimWithSources] = None
    nsf_supported_research_area: Optional[ClaimWithSources] = None

    # Proposal components and limits
    project_description_page_limit: Optional[ClaimWithSources] = None
    project_summary_page_limit: Optional[ClaimWithSources] = None
    project_summary_character_limit: Optional[ClaimWithSources] = None
    departmental_letter_requirement: Optional[ClaimWithSources] = None
    data_management_plan: Optional[ClaimWithSources] = None
    mentoring_plan_requirement_condition: Optional[ClaimWithSources] = None  # if requesting funding for postdocs or graduate students
    mentoring_plan_page_limit: Optional[ClaimWithSources] = None  # 1 page limit

    # Submission restriction
    submission_limit_per_pi: Optional[ClaimWithSources] = None

    # General sources used in the answer (e.g., solicitation, PAPPG, program pages)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the NSF CAREER award requirements as explicitly provided in the answer. For each item below, return:
    - value: the textual statement or number exactly as stated in the answer (do not invent or normalize; keep original phrasing).
    - sources: an array of explicit URLs cited in the answer that support this specific item (include only actual URLs mentioned in the answer; markdown links are allowed; do not invent URLs).

    Fields to extract:
    1) award_duration
       Example of value: "5 years" or "five-year award".
    2) min_funding_standard_directorates
       Example of value: "$400,000 minimum total funding including indirect costs (over 5 years) for most directorates."
    3) min_funding_bio_eng_opp
       Example of value: "$500,000 minimum (approximately $100,000 per year) for BIO, ENG, or OPP."
    4) doctoral_degree_eligibility
       Example of value: "Proposers must hold a doctoral degree at time of submission in a field supported by NSF."
    5) tenure_track_position_requirement
       Example of value: "At least a 50% tenure-track (or equivalent) position at time of submission."
    6) nsf_supported_research_area
       Example of value: "Research must be in an area of science, engineering, or education supported by NSF."
    7) project_description_page_limit
       Example of value: "Project Description is limited to 15 pages (including Results from Prior NSF Support)."
    8) project_summary_page_limit
       Example of value: "Project Summary must be no more than 1 page."
    9) project_summary_character_limit
       Example of value: "Project Summary must not exceed 4,600 characters."
    10) departmental_letter_requirement
        Example of value: "A departmental letter verifying eligibility is required."
    11) data_management_plan
        Example of value: "A Data Management Plan is required."
    12) mentoring_plan_requirement_condition
        Example of value: "A Mentoring Plan is required if requesting funding for postdoctoral researchers or graduate students."
    13) mentoring_plan_page_limit
        Example of value: "Mentoring Plan: 1-page limit."
    14) submission_limit_per_pi
        Example of value: "Each eligible PI may submit only one CAREER proposal per annual competition."
    15) general_sources
        Collect any general URLs cited in the answer that pertain to CAREER, the solicitation, program pages, or NSF PAPPG, etc.

    Rules:
    - Extract only information explicitly present in the answer. If an item is not mentioned, set its 'value' to null and its 'sources' to [].
    - For URLs, include only valid URLs explicitly present in the answer (plain or markdown). If absent, 'sources' should be [].
    - For 'general_sources', include all URLs cited in the answer that are relevant but not tied to a specific item.

    Return a JSON object matching the specified fields exactly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _pick_sources(field: Optional[ClaimWithSources], fallback: List[str]) -> List[str]:
    """
    Choose per-field sources if available; otherwise fallback to general sources.
    """
    if field and field.sources:
        return field.sources
    return fallback if fallback else []


async def _verify_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim_text: str,
    sources: List[str],
    critical: bool = True,
    additional_instruction: str = "None",
) -> None:
    """
    Create a leaf node and run verification on the claim, using provided sources (if any).
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    # If no sources are available, allow simple verification; otherwise verify by provided URLs
    srcs: Optional[List[str]] = sources if sources else None
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=srcs,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_requirements_tree(
    evaluator: Evaluator,
    root: Any,
    extracted: CareerRequirementsExtraction,
) -> None:
    """
    Build the verification tree and verify each requirement against cited sources.
    """
    # Create main requirements node (non-critical to allow optional elements under it).
    # Critical gating will still apply via child nodes marked as critical.
    req_root = evaluator.add_parallel(
        id="NSF_CAREER_Requirements",
        desc="Comprehensive specification of NSF CAREER award requirements covering duration, funding, eligibility, proposal components, and submission restrictions",
        parent=root,
        critical=False,
    )

    # Convenience fallback sources (general from the answer)
    general_sources = extracted.general_sources if extracted and extracted.general_sources else []

    # 1) Award Duration: 5 years
    await _verify_leaf(
        evaluator,
        req_root,
        "Award_Duration",
        "Specifies that the NSF CAREER award provides funding for a 5-year period",
        claim_text="NSF CAREER awards provide funding for a five-year (5-year) period.",
        sources=_pick_sources(extracted.award_duration, general_sources),
        critical=True,
        additional_instruction="Confirm on the cited NSF CAREER solicitation/program pages or PAPPG that CAREER awards are 5 years in duration.",
    )

    # 2) Minimum funding for standard directorates: $400,000 total including indirect over 5 years
    await _verify_leaf(
        evaluator,
        req_root,
        "Minimum_Funding_Standard_Directorates",
        "Specifies the minimum total funding of $400,000 (including indirect costs) for the 5-year duration for most NSF directorates",
        claim_text="For most NSF directorates (excluding BIO, ENG, and OPP), the CAREER award minimum total funding is $400,000 over 5 years, including indirect costs.",
        sources=_pick_sources(extracted.min_funding_standard_directorates, general_sources),
        critical=True,
        additional_instruction="Verify that the cited CAREER solicitation/program guidance states a minimum total of $400,000 (including indirects) over 5 years for directorates other than BIO, ENG, and OPP.",
    )

    # 3) Minimum funding for BIO/ENG/OPP: $500,000 total (~$100k/year)
    await _verify_leaf(
        evaluator,
        req_root,
        "Minimum_Funding_BIO_ENG_OPP",
        "Specifies the higher minimum funding requirement of $500,000 (approximately $100,000 per year) for proposals to BIO, ENG, or OPP directorates",
        claim_text="For BIO, ENG, or OPP directorates, the CAREER award minimum total funding is $500,000 (approximately $100,000 per year).",
        sources=_pick_sources(extracted.min_funding_bio_eng_opp, general_sources),
        critical=True,
        additional_instruction="Check the CAREER solicitation/program guidance for BIO, ENG, and OPP showing a $500,000 minimum total.",
    )

    # 4) Doctoral degree eligibility
    await _verify_leaf(
        evaluator,
        req_root,
        "Doctoral_Degree_Eligibility",
        "Specifies that proposers must hold a doctoral degree in a field supported by NSF at the time of submission",
        claim_text="Proposers must hold a doctoral degree in a field supported by NSF at the time of submission.",
        sources=_pick_sources(extracted.doctoral_degree_eligibility, general_sources),
        critical=True,
        additional_instruction="Confirm eligibility statements indicating the doctoral degree requirement in an NSF-supported field.",
    )

    # 5) Tenure track position requirement (≥50% appointment) at time of submission
    await _verify_leaf(
        evaluator,
        req_root,
        "Tenure_Track_Position_Requirement",
        "Specifies that proposers must hold at least a 50% tenure-track or equivalent position at the time of submission",
        claim_text="Proposers must hold at least a 50% tenure-track (or equivalent) position at the time of submission.",
        sources=_pick_sources(extracted.tenure_track_position_requirement, general_sources),
        critical=True,
        additional_instruction="Verify statements that specify a ≥50% tenure-track (or equivalent) appointment at the time of submission.",
    )

    # 6) NSF-supported research area
    await _verify_leaf(
        evaluator,
        req_root,
        "NSF_Supported_Research_Area",
        "Specifies that proposers must be engaged in research in an area of science, engineering, or education supported by NSF",
        claim_text="Proposers must be engaged in research in an area of science, engineering, or education supported by NSF.",
        sources=_pick_sources(extracted.nsf_supported_research_area, general_sources),
        critical=True,
        additional_instruction="Check that the eligibility criteria state that the PI's research area must be one supported by NSF.",
    )

    # 7) Project Description page limit: 15 pages (including Results from Prior NSF Support)
    await _verify_leaf(
        evaluator,
        req_root,
        "Project_Description_Page_Limit",
        "Specifies the 15-page maximum limit for the Project Description (including Results from Prior NSF Support)",
        claim_text="The CAREER Project Description has a maximum of 15 pages, and the 'Results from Prior NSF Support' must be included within that limit when applicable.",
        sources=_pick_sources(extracted.project_description_page_limit, general_sources),
        critical=True,
        additional_instruction="Confirm that the Project Description is limited to 15 pages and that 'Results from Prior NSF Support' (if applicable) are included within those 15 pages.",
    )

    # 8) Project Summary constraints: split into page limit and character limit under a dedicated node
    proj_sum_node = evaluator.add_parallel(
        id="Project_Summary_Constraints",
        desc="Project Summary constraints: 1 page and 4,600 characters maximum",
        parent=req_root,
        critical=True,
    )
    await _verify_leaf(
        evaluator,
        proj_sum_node,
        "Project_Summary_Page_Limit",
        "Specifies the Project Summary limit of 1 page",
        claim_text="The CAREER Project Summary must not exceed 1 page.",
        sources=_pick_sources(extracted.project_summary_page_limit, general_sources),
        critical=True,
        additional_instruction="Verify the Project Summary page limit (1 page) per NSF guidance.",
    )
    await _verify_leaf(
        evaluator,
        proj_sum_node,
        "Project_Summary_Character_Limit",
        "Specifies the Project Summary limit of 4,600 characters maximum",
        claim_text="The CAREER Project Summary must not exceed 4,600 characters.",
        sources=_pick_sources(extracted.project_summary_character_limit, general_sources),
        critical=True,
        additional_instruction="Some NSF systems enforce a 4,600 character maximum for the Project Summary; confirm this on the cited page.",
    )

    # 9) Departmental letter requirement
    await _verify_leaf(
        evaluator,
        req_root,
        "Departmental_Letter_Requirement",
        "Specifies that a departmental letter verifying eligibility requirements is required",
        claim_text="A departmental letter verifying eligibility requirements is required for CAREER proposals.",
        sources=_pick_sources(extracted.departmental_letter_requirement, general_sources),
        critical=True,
        additional_instruction="Confirm that the CAREER solicitation requires a departmental (or organizational) letter attesting to eligibility and support.",
    )

    # 10) Data Management Plan requirement
    await _verify_leaf(
        evaluator,
        req_root,
        "Data_Management_Plan",
        "Specifies that a Data Management Plan is a required proposal component",
        claim_text="A Data Management Plan (DMP) is required for CAREER proposals.",
        sources=_pick_sources(extracted.data_management_plan, general_sources),
        critical=True,
        additional_instruction="Verify that NSF requires a Data Management Plan for proposals (including CAREER).",
    )

    # 11) Mentoring Plan requirement (non-critical node with two checks)
    mentoring_node = evaluator.add_parallel(
        id="Mentoring_Plan_Requirement",
        desc="Mentoring Plan requirements (conditional requirement and 1-page limit)",
        parent=req_root,
        critical=False,
    )
    await _verify_leaf(
        evaluator,
        mentoring_node,
        "Mentoring_Plan_Required_Condition",
        "Specifies that a Mentoring Plan is required if requesting funding for postdoctoral researchers or graduate students",
        claim_text="A Mentoring Plan is required if the proposal requests funding for postdoctoral researchers or graduate students.",
        sources=_pick_sources(extracted.mentoring_plan_requirement_condition, general_sources),
        critical=True,
        additional_instruction=(
            "Confirm whether the cited guidance requires a Mentoring Plan when funding is requested for postdoctoral researchers "
            "and/or graduate students. If only postdocs are mentioned, determine if the claim including graduate students is supported."
        ),
    )
    await _verify_leaf(
        evaluator,
        mentoring_node,
        "Mentoring_Plan_Page_Limit",
        "Specifies that the Mentoring Plan has a 1 page limit",
        claim_text="The Mentoring Plan is limited to 1 page.",
        sources=_pick_sources(extracted.mentoring_plan_page_limit, general_sources),
        critical=True,
        additional_instruction="Verify that the Mentoring Plan page limit is 1 page in the cited guidance.",
    )

    # 12) Submission limit per PI: one CAREER proposal per annual competition
    await _verify_leaf(
        evaluator,
        req_root,
        "Submission_Limit_Per_PI",
        "Specifies that each eligible PI may submit only one CAREER proposal per annual competition",
        claim_text="Each eligible PI may submit only one CAREER proposal per annual competition.",
        sources=_pick_sources(extracted.submission_limit_per_pi, general_sources),
        critical=True,
        additional_instruction="Verify that CAREER imposes a submission limit of one proposal per PI in each annual competition.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the NSF CAREER 2026 requirements task.
    """
    # Initialize evaluator with a parallel root; we'll add a dedicated requirement node under it.
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

    # Extract structured requirements from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=CareerRequirementsExtraction,
        extraction_name="nsf_career_requirements_extraction",
    )

    # Build verification tree and check each requirement with evidence
    await build_and_verify_requirements_tree(evaluator, root, extracted)

    # Return structured result
    return evaluator.get_summary()