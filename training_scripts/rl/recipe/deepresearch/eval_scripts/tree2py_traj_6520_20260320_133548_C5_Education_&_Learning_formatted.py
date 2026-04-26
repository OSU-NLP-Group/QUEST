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
TASK_ID = "cwru_eng_grad_fall2025"
TASK_DESCRIPTION = """
I am an international student interested in applying to Case Western Reserve University's graduate programs in Engineering for Fall 2025. Please provide comprehensive information including: 
1. Admission Requirements: What are the minimum GPA requirements and what is the GRE policy for graduate engineering programs? 
2. English Proficiency Requirements: What is the minimum TOEFL score required, and what alternative English proficiency tests are accepted with their minimum scores? 
3. Funding Opportunities: What types of graduate assistantships and fellowships are available for engineering graduate students, including details about stipends and coverage? 
4. Program Offerings: What types of graduate degrees are offered in Engineering (MS, PhD, certificates, etc.), and what is the university's accreditation status? 
5. Application Process: What is the application deadline for Fall 2025 admission (or is it rolling admission), and what are the required application materials? 
For each of these five categories, please provide a reference URL to the relevant official Case Western Reserve University webpage.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AdmissionReq(BaseModel):
    gpa_requirement: Optional[str] = None
    gre_policy: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class EnglishReq(BaseModel):
    toefl_min: Optional[str] = None
    alternative_tests: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class FundingInfo(BaseModel):
    assistantships_info: Optional[str] = None
    fellowships_info: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProgramInfo(BaseModel):
    degree_types: List[str] = Field(default_factory=list)
    accreditation_status: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ApplicationInfo(BaseModel):
    deadline_fall_2025: Optional[str] = None
    materials: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class CWRUGradEngExtraction(BaseModel):
    admission: Optional[AdmissionReq] = None
    english: Optional[EnglishReq] = None
    funding: Optional[FundingInfo] = None
    program: Optional[ProgramInfo] = None
    application: Optional[ApplicationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cwru_info() -> str:
    return """
Extract the following structured information exactly as stated in the answer for Case Western Reserve University (CWRU) graduate engineering programs, targeting Fall 2025 admission for international students. 
Only extract what is explicitly present in the answer. For URLs, only include URLs actually shown in the answer text.

Return a JSON with the following structure:

{
  "admission": {
    "gpa_requirement": string | null,                  // Verbatim minimum GPA text as claimed (e.g., "3.0/4.0", "B average", "upper-third of class")
    "gre_policy": string | null,                       // Verbatim GRE policy (e.g., "required", "optional", "waived", "not required", with any notes)
    "urls": string[]                                   // URLs in the answer that document graduate engineering admission requirements (official CWRU pages)
  },
  "english": {
    "toefl_min": string | null,                        // Verbatim minimum TOEFL requirement (e.g., "TOEFL iBT 90", include new vs old system if mentioned)
    "alternative_tests": string[],                     // Verbatim entries like "IELTS: 7.0", "Duolingo: 115", "PTE: 61", etc.
    "urls": string[]                                   // URLs in the answer documenting English proficiency requirements
  },
  "funding": {
    "assistantships_info": string | null,              // Verbatim summary mentioning assistantships (TA/RA/GA) and whether stipend and coverage (tuition/insurance) are discussed
    "fellowships_info": string | null,                 // Verbatim summary of fellowships/scholarships if mentioned
    "urls": string[]                                   // URLs in the answer about graduate funding/assistantships/fellowships
  },
  "program": {
    "degree_types": string[],                          // Degree types listed (e.g., "MS", "MEng", "PhD", "Graduate Certificate", "Dual-degree")
    "accreditation_status": string | null,             // Verbatim institutional accreditation (e.g., "Higher Learning Commission")
    "urls": string[]                                   // URLs listing engineering graduate programs (or Case School of Engineering pages); accreditation page if provided
  },
  "application": {
    "deadline_fall_2025": string | null,               // Verbatim deadline info for Fall 2025 (date, priority date, or "rolling"; if varies by department, include that text)
    "materials": string[],                             // Required materials list (e.g., "online application", "transcripts", "CV", "recommendation letters", "statement of purpose", "English scores", etc.)
    "urls": string[]                                   // URLs for official application instructions or the graduate application portal
  }
}

Important rules:
- For any missing info, use null (for a single field) or [] for lists.
- Maintain verbatim wording for fields like "gpa_requirement", "gre_policy", "toefl_min", and "accreditation_status".
- For URLs, include only those explicitly shown in the answer; do not infer or fabricate URLs.
"""


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def has_nonempty_text(s: Optional[str]) -> bool:
    return bool(s) and isinstance(s, str) and s.strip() != ""


def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and isinstance(urls, list) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_admission_requirements_tree(evaluator: Evaluator, parent, data: Optional[AdmissionReq]) -> None:
    node_adm = evaluator.add_parallel(
        id="Admission_Requirements",
        desc="Information about academic admission requirements for graduate engineering programs",
        parent=parent,
        critical=False
    )

    # Academic_Standards (critical)
    node_academic = evaluator.add_parallel(
        id="Academic_Standards",
        desc="Academic qualifications required for admission",
        parent=node_adm,
        critical=True
    )

    # GPA requirement exists
    evaluator.add_custom_node(
        result=has_nonempty_text(data.gpa_requirement) if data else False,
        id="GPA_Requirement_exists",
        desc="GPA requirement text is provided in the answer",
        parent=node_academic,
        critical=True
    )

    # GPA requirement supported by URLs
    leaf_gpa = evaluator.add_leaf(
        id="GPA_Requirement",
        desc="Minimum GPA requirement is provided (e.g., B-average, 3.0 GPA, or upper third of class)",
        parent=node_academic,
        critical=True
    )
    await evaluator.verify(
        claim=f'The minimum GPA requirement for admission to CWRU graduate engineering programs is stated as: "{(data.gpa_requirement or "").strip()}".',
        node=leaf_gpa,
        sources=(data.urls if data else []),
        additional_instruction="Verify on an official Case Western Reserve University site (case.edu domain) that the graduate engineering admission page documents this minimum GPA or equivalent wording (e.g., B average ≈ 3.0/4.0). Allow reasonable wording variations."
    )

    # GRE policy exists
    evaluator.add_custom_node(
        result=has_nonempty_text(data.gre_policy) if data else False,
        id="GRE_Policy_exists",
        desc="GRE policy text is provided in the answer",
        parent=node_academic,
        critical=True
    )

    # GRE policy supported by URLs
    leaf_gre = evaluator.add_leaf(
        id="GRE_Policy",
        desc="GRE requirement status is clearly stated (required, optional, or waived for engineering programs)",
        parent=node_academic,
        critical=True
    )
    await evaluator.verify(
        claim=f'The GRE policy for CWRU graduate engineering programs is: "{(data.gre_policy or "").strip()}".',
        node=leaf_gre,
        sources=(data.urls if data else []),
        additional_instruction="Check on an official CWRU page whether GRE is required/optional/waived/recommended for graduate engineering. Accept department-level notes if the page specifies exceptions."
    )

    # Admission reference URL(s)
    leaf_adm_url = evaluator.add_leaf(
        id="Admission_Reference_URL",
        desc="Valid URL to official CWRU admissions or engineering program page is provided",
        parent=node_adm,
        critical=True
    )
    await evaluator.verify(
        claim="This is an official Case Western Reserve University page that documents graduate engineering admission requirements.",
        node=leaf_adm_url,
        sources=(data.urls if data else []),
        additional_instruction="The URL should be on a CWRU-owned domain (e.g., case.edu, engineering.case.edu). The page should discuss graduate admission requirements."
    )


async def build_english_proficiency_tree(evaluator: Evaluator, parent, data: Optional[EnglishReq]) -> None:
    node_eng = evaluator.add_parallel(
        id="English_Proficiency",
        desc="English language proficiency requirements for international applicants",
        parent=parent,
        critical=False
    )

    node_tests = evaluator.add_parallel(
        id="Test_Requirements",
        desc="Required English proficiency test scores",
        parent=node_eng,
        critical=True
    )

    # TOEFL minimum exists
    evaluator.add_custom_node(
        result=has_nonempty_text(data.toefl_min) if data else False,
        id="TOEFL_Minimum_exists",
        desc="TOEFL minimum text is provided in the answer",
        parent=node_tests,
        critical=True
    )

    # TOEFL minimum supported
    leaf_toefl = evaluator.add_leaf(
        id="TOEFL_Minimum",
        desc="Minimum TOEFL score requirement is provided with specific numerical value (including new vs. old scoring system if applicable)",
        parent=node_tests,
        critical=True
    )
    await evaluator.verify(
        claim=f'The minimum required TOEFL score for graduate (engineering) admission at CWRU is: "{(data.toefl_min or "").strip()}".',
        node=leaf_toefl,
        sources=(data.urls if data else []),
        additional_instruction="Verify on an official CWRU page that the listed minimum TOEFL total (iBT or other formats) matches. If the page mentions multiple formats or updates (new vs old scale), accept equivalent descriptions."
    )

    # Alternative tests exist
    evaluator.add_custom_node(
        result=(bool(data.alternative_tests) if data else False),
        id="Alternative_Tests_exists",
        desc="Alternative English tests with minimum scores are provided in the answer",
        parent=node_tests,
        critical=True
    )

    # Alternative tests supported
    alt_summary = ", ".join(data.alternative_tests) if data and data.alternative_tests else ""
    leaf_alt = evaluator.add_leaf(
        id="Alternative_Tests",
        desc="Information about alternative English tests accepted (IELTS, PTE, Duolingo, etc.) with minimum scores is provided",
        parent=node_tests,
        critical=True
    )
    await evaluator.verify(
        claim=f'CWRU accepts the following alternative English proficiency tests and minimum scores: {alt_summary}.',
        node=leaf_alt,
        sources=(data.urls if data else []),
        additional_instruction="Confirm on an official CWRU page that tests such as IELTS, PTE, Duolingo (or others) are accepted and that minimum scores are specified. Minor wording variants are fine."
    )

    # English proficiency reference URL(s)
    leaf_eng_url = evaluator.add_leaf(
        id="English_Reference_URL",
        desc="Valid URL to official page documenting English proficiency requirements is provided",
        parent=node_eng,
        critical=True
    )
    await evaluator.verify(
        claim="This is an official CWRU page that documents graduate English language proficiency requirements for international applicants.",
        node=leaf_eng_url,
        sources=(data.urls if data else []),
        additional_instruction="The page should be on the CWRU domain and explicitly describe English proficiency requirements and accepted tests."
    )


async def build_funding_tree(evaluator: Evaluator, parent, data: Optional[FundingInfo]) -> None:
    node_funding = evaluator.add_parallel(
        id="Funding_Opportunities",
        desc="Available financial support options for graduate engineering students",
        parent=parent,
        critical=False
    )

    node_types = evaluator.add_parallel(
        id="Funding_Types",
        desc="Types of financial support available to graduate engineering students",
        parent=node_funding,
        critical=True
    )

    # Assistantships exist
    evaluator.add_custom_node(
        result=has_nonempty_text(data.assistantships_info) if data else False,
        id="Assistantships_exists",
        desc="Assistantship information is provided in the answer",
        parent=node_types,
        critical=True
    )

    # Assistantships supported with stipend/coverage detail
    leaf_assist = evaluator.add_leaf(
        id="Assistantships",
        desc="Information about graduate assistantship opportunities (teaching, research, or other) including stipend and coverage details is provided",
        parent=node_types,
        critical=True
    )
    await evaluator.verify(
        claim="Official CWRU pages indicate that graduate assistantships (e.g., TA/RA/GA) are available to engineering graduate students and include details about stipends and tuition and/or health insurance coverage.",
        node=leaf_assist,
        sources=(data.urls if data else []),
        additional_instruction="Confirm that assistantships are described and that compensation details (stipend and some coverage like tuition remission or insurance) are mentioned on CWRU sites."
    )

    # Fellowships exist
    evaluator.add_custom_node(
        result=has_nonempty_text(data.fellowships_info) if data else False,
        id="Fellowships_exists",
        desc="Fellowship/scholarship information is provided in the answer",
        parent=node_types,
        critical=True
    )

    # Fellowships supported
    leaf_fellow = evaluator.add_leaf(
        id="Fellowships",
        desc="Information about fellowships or scholarships available to graduate engineering students is provided",
        parent=node_types,
        critical=True
    )
    await evaluator.verify(
        claim="Official CWRU pages describe fellowships or scholarships available to graduate engineering students.",
        node=leaf_fellow,
        sources=(data.urls if data else []),
        additional_instruction="Look for CWRU pages listing internal or external fellowships/scholarships relevant to graduate engineering students."
    )

    # Funding reference URL(s)
    leaf_fund_url = evaluator.add_leaf(
        id="Funding_Reference_URL",
        desc="Valid URL to official page about graduate funding, assistantships, or financial aid is provided",
        parent=node_funding,
        critical=True
    )
    await evaluator.verify(
        claim="This is an official CWRU page discussing graduate funding, assistantships, fellowships, or financial aid.",
        node=leaf_fund_url,
        sources=(data.urls if data else []),
        additional_instruction="The URL should be on a CWRU domain and the content should clearly be about graduate funding opportunities."
    )


async def build_program_details_tree(evaluator: Evaluator, parent, data: Optional[ProgramInfo]) -> None:
    node_prog = evaluator.add_parallel(
        id="Program_Details",
        desc="Information about engineering graduate program offerings at CWRU",
        parent=parent,
        critical=False
    )

    node_offerings = evaluator.add_parallel(
        id="Program_Offerings",
        desc="Details about available engineering graduate programs",
        parent=node_prog,
        critical=True
    )

    # Degree types exist
    evaluator.add_custom_node(
        result=(bool(data.degree_types) if data else False),
        id="Degree_Types_exists",
        desc="Degree types are listed in the answer",
        parent=node_offerings,
        critical=True
    )

    # Degree types supported
    deg_list = ", ".join(data.degree_types) if data and data.degree_types else ""
    leaf_deg = evaluator.add_leaf(
        id="Degree_Types",
        desc="Types of graduate degrees offered in engineering are listed (MS, PhD, MEng, certificates, etc.)",
        parent=node_offerings,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official CWRU engineering/program pages indicate the following graduate degree types are offered in engineering: {deg_list}.",
        node=leaf_deg,
        sources=(data.urls if data else []),
        additional_instruction="Verify that the program pages list the degree types (e.g., MS, MEng, PhD, Graduate Certificate). Minor naming variations (e.g., M.S. vs MS) are acceptable."
    )

    # Accreditation status exists
    evaluator.add_custom_node(
        result=has_nonempty_text(data.accreditation_status) if data else False,
        id="Accreditation_Status_exists",
        desc="Accreditation status text is provided in the answer",
        parent=node_offerings,
        critical=True
    )

    # Accreditation status supported
    leaf_accred = evaluator.add_leaf(
        id="Accreditation_Status",
        desc="University's regional accreditation status is verified (e.g., Higher Learning Commission)",
        parent=node_offerings,
        critical=True
    )
    await evaluator.verify(
        claim=f'Case Western Reserve University is institutionally (regionally) accredited by "{(data.accreditation_status or "").strip()}".',
        node=leaf_accred,
        sources=(data.urls if data else []),
        additional_instruction="Confirm on official CWRU pages (or linked official accreditation statements on CWRU domain) that the institutional accreditor (e.g., HLC) is as stated. Focus on university-level accreditation, not ABET program accreditation."
    )

    # Program reference URL(s)
    leaf_prog_url = evaluator.add_leaf(
        id="Program_Reference_URL",
        desc="Valid URL to official page listing engineering graduate programs or Case School of Engineering is provided",
        parent=node_prog,
        critical=True
    )
    await evaluator.verify(
        claim="This is an official CWRU page listing engineering graduate programs or a Case School of Engineering page describing graduate offerings.",
        node=leaf_prog_url,
        sources=(data.urls if data else []),
        additional_instruction="The URL should be on the case.edu domain and clearly list/describe graduate engineering programs."
    )


async def build_application_process_tree(evaluator: Evaluator, parent, data: Optional[ApplicationInfo]) -> None:
    node_app = evaluator.add_parallel(
        id="Application_Process",
        desc="Information about how to apply and application timeline",
        parent=parent,
        critical=False
    )

    node_app_info = evaluator.add_parallel(
        id="Application_Information",
        desc="Key application process details",
        parent=node_app,
        critical=True
    )

    # Deadline exists
    evaluator.add_custom_node(
        result=has_nonempty_text(data.deadline_fall_2025) if data else False,
        id="Application_Deadline_exists",
        desc="Application deadline info for Fall 2025 is provided in the answer",
        parent=node_app_info,
        critical=True
    )

    # Deadline supported
    leaf_deadline = evaluator.add_leaf(
        id="Application_Deadline",
        desc="Application deadline information for Fall 2025 admission is provided (specific date or rolling admission status)",
        parent=node_app_info,
        critical=True
    )
    await evaluator.verify(
        claim=f'The application timeline for Fall 2025 is stated as: "{(data.deadline_fall_2025 or "").strip()}" (either a specific deadline or rolling).',
        node=leaf_deadline,
        sources=(data.urls if data else []),
        additional_instruction="Confirm on an official CWRU page that the Fall 2025 deadline (or rolling/priority scheme) is as stated. Departmental variation is acceptable if the answer reflects that."
    )

    # Materials exist
    evaluator.add_custom_node(
        result=(bool(data.materials) if data else False),
        id="Application_Materials_exists",
        desc="Required application materials list is provided in the answer",
        parent=node_app_info,
        critical=True
    )

    # Materials supported
    materials_list = ", ".join(data.materials) if data and data.materials else ""
    leaf_materials = evaluator.add_leaf(
        id="Application_Materials",
        desc="Required application materials are listed (e.g., online application, transcripts, letters of recommendation, etc.)",
        parent=node_app_info,
        critical=True
    )
    await evaluator.verify(
        claim=f"The required application materials include: {materials_list}.",
        node=leaf_materials,
        sources=(data.urls if data else []),
        additional_instruction="Verify on official CWRU pages that the listed items (e.g., transcripts, letters of recommendation, statement of purpose, CV/resume, English scores) are required or commonly required for graduate engineering."
    )

    # Application reference URL(s)
    leaf_app_url = evaluator.add_leaf(
        id="Application_Reference_URL",
        desc="Valid URL to official application instructions or graduate application portal is provided",
        parent=node_app,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is an official CWRU application instruction page or the graduate application portal.",
        node=leaf_app_url,
        sources=(data.urls if data else []),
        additional_instruction="The URL should be on the case.edu domain and clearly be about how to apply or the application portal for graduate programs."
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
    Evaluate an answer for comprehensive CWRU graduate engineering information (Fall 2025).
    """
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

    # 1) Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_cwru_info(),
        template_class=CWRUGradEngExtraction,
        extraction_name="cwru_grad_eng_info",
    )

    # 2) Build verification trees per rubric
    await build_admission_requirements_tree(evaluator, root, extracted.admission)
    await build_english_proficiency_tree(evaluator, root, extracted.english)
    await build_funding_tree(evaluator, root, extracted.funding)
    await build_program_details_tree(evaluator, root, extracted.program)
    await build_application_process_tree(evaluator, root, extracted.application)

    # 3) Return summary
    return evaluator.get_summary()