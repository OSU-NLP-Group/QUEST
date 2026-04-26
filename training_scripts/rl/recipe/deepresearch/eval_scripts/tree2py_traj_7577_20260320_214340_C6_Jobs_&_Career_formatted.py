import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_principal_cert_pathway"
TASK_DESCRIPTION = (
    "A current classroom teacher in Texas with three years of teaching experience and a bachelor's degree in education "
    "is planning to advance their career to become a school principal. They want to document the complete pathway to "
    "principal certification in Texas, including: (1) the graduate degree requirements and approved program completion "
    "needed for eligibility, (2) the teaching certification and experience prerequisites, (3) all required state "
    "certification examinations and performance assessments with their specific formats, and (4) identification of the "
    "school district in Texas that was recognized by Forbes as the top education employer in 2025 or 2026 rankings, "
    "where they could pursue principal positions after certification. Document the comprehensive requirements for each "
    "of these four components, providing reference URLs to support each major requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DegreeProgramExtraction(BaseModel):
    masters_degree_accreditation_statement: Optional[str] = None
    masters_degree_accreditation_urls: List[str] = Field(default_factory=list)
    approved_program_requirement_statement: Optional[str] = None
    approved_program_urls: List[str] = Field(default_factory=list)


class TeachingPrereqsExtraction(BaseModel):
    valid_texas_classroom_certificate_statement: Optional[str] = None
    teaching_certificate_urls: List[str] = Field(default_factory=list)
    minimum_teaching_experience_statement: Optional[str] = None
    teaching_experience_urls: List[str] = Field(default_factory=list)


class ExamsAssessmentsExtraction(BaseModel):
    texes_268_requirement_statement: Optional[str] = None
    texes_268_format_statement: Optional[str] = None
    texes_268_urls: List[str] = Field(default_factory=list)
    pasl_requirement_statement: Optional[str] = None
    pasl_format_statement: Optional[str] = None
    pasl_urls: List[str] = Field(default_factory=list)


class ForbesDistrictExtraction(BaseModel):
    district_name: Optional[str] = None
    forbes_ranking_claim_statement: Optional[str] = None
    forbes_ranking_urls: List[str] = Field(default_factory=list)


class TxPrincipalPathwayExtraction(BaseModel):
    degree: Optional[DegreeProgramExtraction] = None
    teaching: Optional[TeachingPrereqsExtraction] = None
    exams: Optional[ExamsAssessmentsExtraction] = None
    district: Optional[ForbesDistrictExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pathway() -> str:
    return """
Extract the following fields exactly as they appear in the answer. Do not invent or infer facts. If a field is not
present, set it to null (for strings) or [] (for URL lists).

Return a JSON object with the following schema:

{
  "degree": {
    "masters_degree_accreditation_statement": string | null,
    "masters_degree_accreditation_urls": string[],

    "approved_program_requirement_statement": string | null,
    "approved_program_urls": string[]
  },
  "teaching": {
    "valid_texas_classroom_certificate_statement": string | null,
    "teaching_certificate_urls": string[],

    "minimum_teaching_experience_statement": string | null,
    "teaching_experience_urls": string[]
  },
  "exams": {
    "texes_268_requirement_statement": string | null,
    "texes_268_format_statement": string | null,
    "texes_268_urls": string[],

    "pasl_requirement_statement": string | null,
    "pasl_format_statement": string | null,
    "pasl_urls": string[]
  },
  "district": {
    "district_name": string | null,
    "forbes_ranking_claim_statement": string | null,
    "forbes_ranking_urls": string[]
  }
}

Field guidance:
- masters_degree_accreditation_statement: The exact text in the answer stating a master's degree is required and that it
  must be from an accredited institution (preferably noting an accrediting agency recognized by the Texas Higher Education Coordinating Board).
- masters_degree_accreditation_urls: All URLs in the answer that support the master's/accreditation requirement.

- approved_program_requirement_statement: The exact text in the answer stating completion of a Texas-approved principal
  preparation program is required (optionally noting typical 30–36 credit hours in educational leadership/administration
  if the answer states that).
- approved_program_urls: All URLs in the answer that support the approved program requirement and/or the typical credit-hour range.

- valid_texas_classroom_certificate_statement: The exact text in the answer stating a valid Texas classroom teaching certificate is required.
- teaching_certificate_urls: All URLs that support the teaching certificate requirement.

- minimum_teaching_experience_statement: The exact text in the answer stating a minimum of two years of creditable teaching experience is required.
- teaching_experience_urls: All URLs that support the minimum two years of experience requirement.

- texes_268_requirement_statement: The exact text stating TExES Principal as Instructional Leader (268) is required.
- texes_268_format_statement: The exact text describing TExES 268 format (e.g., 70 selected-response + 4 constructed-response in a 5-hour session) if stated.
- texes_268_urls: All URLs that support the 268 requirement and/or format.

- pasl_requirement_statement: The exact text stating the Performance Assessment for School Leaders (PASL) must be completed in addition to 268.
- pasl_format_statement: The exact text describing PASL’s performance-based format (e.g., multi-task submission with artifacts).
- pasl_urls: All URLs that support the PASL requirement and/or format.

- district_name: The district the answer identifies as the Forbes-recognized top education employer in Texas in 2025 or 2026.
- forbes_ranking_claim_statement: The exact text of the Forbes recognition claim (e.g., "#1 education employer in Texas (2025)" or "top K–12 employer among large institutions in 2026").
- forbes_ranking_urls: All URLs provided to verify the Forbes recognition for the identified district.

URL rules:
- Only extract URLs explicitly present in the answer (plain URLs or within markdown).
- Include full URLs with protocol. Ignore malformed URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic cleaning: strip whitespace and drop empties/obvious non-URLs
    cleaned = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        cleaned.append(s)
    return cleaned


async def _add_url_existence_leaf(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    urls: List[str],
    critical: bool = True
):
    has_any_url = len(_safe_urls(urls)) > 0
    evaluator.add_custom_node(
        result=has_any_url,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )


async def _verify_claim_by_urls(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    additional_instruction: str,
    critical: bool = True
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_safe_urls(urls),
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_degree_requirements_subtree(
    evaluator: Evaluator,
    parent,
    degree: Optional[DegreeProgramExtraction]
):
    node = evaluator.add_parallel(
        id="Graduate_Degree_Requirements",
        desc="Graduate degree and approved program completion requirements for eligibility, with citations.",
        parent=parent,
        critical=True
    )

    degree = degree or DegreeProgramExtraction()

    # Group: Master's degree + accreditation
    masters_group = evaluator.add_parallel(
        id="Masters_Degree_Accreditation_Group",
        desc="Master's degree and accreditation requirement group (with supporting URL).",
        parent=node,
        critical=True
    )

    await _add_url_existence_leaf(
        evaluator,
        masters_group,
        "Masters_Degree_Accreditation_URL",
        "Provides a reference URL supporting the master's degree/accreditation requirement.",
        degree.masters_degree_accreditation_urls
    )

    masters_claim = (
        "For Texas principal certification eligibility, a master's degree is required, and the degree must be from a "
        "university accredited by an accrediting agency recognized by the Texas Higher Education Coordinating Board (THECB) "
        "or an equivalently recognized accrediting body."
    )
    await _verify_claim_by_urls(
        evaluator,
        masters_group,
        "Masters_Degree_Accreditation_Requirement",
        "States that a master's degree is required and that it must be from a university accredited by an accrediting agency recognized by THECB.",
        masters_claim,
        degree.masters_degree_accreditation_urls,
        additional_instruction=(
            "Verify that the cited webpage explicitly indicates: (1) a master's degree is required for principal "
            "certification in Texas and (2) the degree must be from an accredited institution (ideally recognized by THECB). "
            "Allow minor wording variations like 'accredited institution of higher education'."
        )
    )

    # Group: Approved principal preparation program + typical credits
    program_group = evaluator.add_parallel(
        id="Approved_Principal_Preparation_Program_Group",
        desc="Approved principal preparation program requirement group (with supporting URL).",
        parent=node,
        critical=True
    )

    await _add_url_existence_leaf(
        evaluator,
        program_group,
        "Approved_Principal_Preparation_Program_URL",
        "Provides a reference URL supporting the approved principal preparation program requirement (including the typical credit-hour range, if asserted).",
        degree.approved_program_urls
    )

    program_claim = (
        "To be eligible for Texas principal certification, completion of a Texas-approved principal preparation program "
        "in educational leadership/administration is required; such programs are commonly around 30–36 graduate credit hours."
    )
    await _verify_claim_by_urls(
        evaluator,
        program_group,
        "Approved_Principal_Preparation_Program_Requirement",
        "States that completion of a Texas-approved principal preparation program is required, and notes the typical program size of 30–36 credit hours in educational leadership/administration (as specified).",
        program_claim,
        degree.approved_program_urls,
        additional_instruction=(
            "The page should indicate that a Texas-approved EPP (principal/EDL preparation) is required. "
            "For the credit-hour range, it is acceptable if the supporting evidence comes from representative Texas university "
            "program pages or TEA/EPP resources that commonly indicate totals in the ~30–36 credit hour range. "
            "Small deviations are acceptable; focus on showing that programs are roughly in that band."
        )
    )


async def build_teaching_prereqs_subtree(
    evaluator: Evaluator,
    parent,
    teaching: Optional[TeachingPrereqsExtraction]
):
    node = evaluator.add_parallel(
        id="Teaching_Prerequisites",
        desc="Teaching certification and experience prerequisites for principal certification eligibility, with citations.",
        parent=parent,
        critical=True
    )

    teaching = teaching or TeachingPrereqsExtraction()

    # Group: Valid Texas classroom teaching certificate
    cert_group = evaluator.add_parallel(
        id="Valid_Texas_Certificate_Group",
        desc="Valid Texas classroom teaching certificate requirement (with supporting URL).",
        parent=node,
        critical=True
    )

    await _add_url_existence_leaf(
        evaluator,
        cert_group,
        "Teaching_Certificate_URL",
        "Provides a reference URL supporting the valid Texas classroom teaching certificate requirement.",
        teaching.teaching_certificate_urls
    )

    cert_claim = "A valid Texas classroom teaching certificate is required to be eligible for Texas principal certification."
    await _verify_claim_by_urls(
        evaluator,
        cert_group,
        "Valid_Texas_Classroom_Teaching_Certificate",
        "States that a valid Texas classroom teaching certificate is required.",
        cert_claim,
        teaching.teaching_certificate_urls,
        additional_instruction=(
            "Accept TEA or official state/ETS sources indicating that candidates must hold a valid classroom teaching "
            "certificate for principal certification eligibility in Texas."
        )
    )

    # Group: Minimum creditable teaching experience (2 years)
    exp_group = evaluator.add_parallel(
        id="Minimum_Teaching_Experience_Group",
        desc="Minimum two years of creditable teaching experience requirement (with supporting URL).",
        parent=node,
        critical=True
    )

    await _add_url_existence_leaf(
        evaluator,
        exp_group,
        "Teaching_Experience_URL",
        "Provides a reference URL supporting the minimum two years of creditable teaching experience requirement.",
        teaching.teaching_experience_urls
    )

    exp_claim = "At least two years of creditable classroom teaching experience is required for Texas principal certification eligibility."
    await _verify_claim_by_urls(
        evaluator,
        exp_group,
        "Minimum_Creditable_Teaching_Experience",
        "States that a minimum of two years of creditable teaching experience is required.",
        exp_claim,
        teaching.teaching_experience_urls,
        additional_instruction=(
            "Look for language like 'two years of creditable teaching experience' or equivalent on TEA or official guidance pages."
        )
    )


async def build_exams_assessments_subtree(
    evaluator: Evaluator,
    parent,
    exams: Optional[ExamsAssessmentsExtraction]
):
    node = evaluator.add_parallel(
        id="Certification_Examinations_and_Assessments",
        desc="All required state certification examinations and performance assessments, including their formats, with citations.",
        parent=parent,
        critical=True
    )

    exams = exams or ExamsAssessmentsExtraction()

    # Subgroup: TExES 268
    texes_group = evaluator.add_parallel(
        id="TExES_268_Group",
        desc="TExES Principal as Instructional Leader (268) requirement and format (with supporting URL).",
        parent=node,
        critical=True
    )

    await _add_url_existence_leaf(
        evaluator,
        texes_group,
        "TExES_268_URL",
        "Provides a reference URL supporting the TExES 268 requirement and/or its format specifics.",
        exams.texes_268_urls
    )

    texes_req_claim = "Passing the TExES Principal as Instructional Leader (268) exam is required for Texas principal certification."
    await _verify_claim_by_urls(
        evaluator,
        texes_group,
        "TExES_268_Requirement",
        "States that passing the TExES Principal as Instructional Leader (268) exam is required.",
        texes_req_claim,
        exams.texes_268_urls,
        additional_instruction=(
            "Verify on TEA or ETS Texas program testing information that 268 is required for principal certification."
        )
    )

    texes_format_claim = (
        "The TExES 268 exam format comprises approximately 70 selected-response (multiple‑choice) questions and 4 "
        "constructed‑response questions in a single testing session of about five hours."
    )
    await _verify_claim_by_urls(
        evaluator,
        texes_group,
        "TExES_268_Format_Specifics",
        "Specifies the TExES 268 exam format as 70 selected-response questions plus 4 constructed-response questions in a five-hour testing session.",
        texes_format_claim,
        exams.texes_268_urls,
        additional_instruction=(
            "Confirm that the page explicitly mentions the structure (70 selected-response + 4 constructed-response) and "
            "an approximately five-hour session (allow minor wording/time variations)."
        )
    )

    # Subgroup: PASL
    pasl_group = evaluator.add_parallel(
        id="PASL_Group",
        desc="Performance Assessment for School Leaders (PASL) requirement and format (with supporting URL).",
        parent=node,
        critical=True
    )

    await _add_url_existence_leaf(
        evaluator,
        pasl_group,
        "PASL_URL",
        "Provides a reference URL supporting the PASL requirement and/or its format description.",
        exams.pasl_urls
    )

    pasl_req_claim = "Completion of the Performance Assessment for School Leaders (PASL) is required in addition to the TExES 268 exam."
    await _verify_claim_by_urls(
        evaluator,
        pasl_group,
        "PASL_Requirement",
        "States that the Performance Assessment for School Leaders (PASL) must be completed in addition to the TExES 268 exam.",
        pasl_req_claim,
        exams.pasl_urls,
        additional_instruction=(
            "Verify on TEA or ETS sources that PASL is an additional requirement alongside the 268 exam."
        )
    )

    pasl_format_claim = (
        "PASL is a performance‑based assessment requiring candidates to complete multi‑part tasks that include written "
        "commentary, artifacts/evidence from practice, and (when applicable) video or other submissions aligned to "
        "leadership standards."
    )
    await _verify_claim_by_urls(
        evaluator,
        pasl_group,
        "PASL_Format_Description",
        "Describes PASL’s assessment format (i.e., performance-based with tasks and submitted artifacts/evidence).",
        pasl_format_claim,
        exams.pasl_urls,
        additional_instruction=(
            "Look for ETS/TEA descriptions indicating PASL consists of multiple tasks with written commentary and artifacts "
            "collected from the candidate’s leadership practice. Precise task counts not required; focus on performance-based nature."
        )
    )


async def build_district_subtree(
    evaluator: Evaluator,
    parent,
    district_info: Optional[ForbesDistrictExtraction]
):
    node = evaluator.add_parallel(
        id="Forbes_Recognized_Top_Education_Employer_District",
        desc="Identifies the Texas school district recognized by Forbes as the top education employer in 2025 or 2026 (per constraints) and provides verification URLs.",
        parent=parent,
        critical=True
    )

    district_info = district_info or ForbesDistrictExtraction()

    # Subgroup: Identification (name match check - internal answer consistency)
    ident_group = evaluator.add_parallel(
        id="District_Identification_Group",
        desc="Answer identifies the target district correctly.",
        parent=node,
        critical=True
    )

    district_name_in_answer = (district_info.district_name or "").strip()
    target_names = ["Klein ISD", "Klein Independent School District"]

    # Leaf: District identified as Klein ISD (simple name equivalence check)
    leaf_ident = evaluator.add_leaf(
        id="District_Identified_As_Klein_ISD",
        desc="Identifies Klein ISD (Klein Independent School District) as the target district (as specified in the constraints).",
        parent=ident_group,
        critical=True
    )
    name_match_claim = f"The identified district '{district_name_in_answer}' refers to Klein ISD (Klein Independent School District)."
    await evaluator.verify(
        claim=name_match_claim,
        node=leaf_ident,
        additional_instruction=(
            "Judge whether the provided district name is equivalent to 'Klein ISD' (i.e., 'Klein Independent School District'). "
            "Allow minor variations and letter casing."
        )
    )

    # Subgroup: Forbes ranking verification (web-grounded)
    forbes_group = evaluator.add_parallel(
        id="Forbes_Ranking_Group",
        desc="Forbes recognition verification (with supporting URL).",
        parent=node,
        critical=True
    )

    await _add_url_existence_leaf(
        evaluator,
        forbes_group,
        "Forbes_Ranking_URL",
        "Provides a reference URL verifying the Forbes ranking/recognition claim for the identified district.",
        district_info.forbes_ranking_urls
    )

    forbes_claim = (
        "Forbes recognized Klein ISD as the top education employer in Texas in either 2025 or 2026—e.g., #1 education "
        "employer in Texas in 2025, or the top K–12 education employer among large institutions in 2026."
    )
    await _verify_claim_by_urls(
        evaluator,
        forbes_group,
        "Forbes_Ranking_Claim",
        "States that Klein ISD was recognized by Forbes as the #1 Education Employer in Texas in 2025 and/or the top K–12 education employer among large institutions in 2026.",
        forbes_claim,
        district_info.forbes_ranking_urls,
        additional_instruction=(
            "Accept verification from Forbes pages or authoritative district/education pages quoting Forbes' lists. "
            "Either 2025 (#1 education employer in Texas) or 2026 (top K–12 education employer among large institutions) "
            "satisfies the claim."
        )
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Texas principal certification pathway task.
    """
    # Initialize evaluator (root is non-critical by framework design)
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_pathway(),
        template_class=TxPrincipalPathwayExtraction,
        extraction_name="tx_principal_pathway_extraction",
    )

    # Top-level rubric node (critical aggregator)
    top = evaluator.add_parallel(
        id="Texas_Principal_Certification_Pathway",
        desc="Complete pathway to become a certified school principal in Texas, including: (1) graduate/program eligibility, (2) teaching prerequisites, (3) required exams/assessments with formats, and (4) the Forbes-recognized target district, with supporting URLs.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_degree_requirements_subtree(evaluator, top, extraction.degree)
    await build_teaching_prereqs_subtree(evaluator, top, extraction.teaching)
    await build_exams_assessments_subtree(evaluator, top, extraction.exams)
    await build_district_subtree(evaluator, top, extraction.district)

    # Return evaluation summary
    return evaluator.get_summary()