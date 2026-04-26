import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_superintendent_career_plan"
TASK_DESCRIPTION = """
A high school principal in Texas with a master's degree in educational leadership and 4 years of principal experience is planning to pursue superintendent certification in Texas. Research the Texas Education Agency requirements and accredited Texas university superintendent certification programs to provide a comprehensive career advancement plan.

Your answer must include:

1. Educational Prerequisites: List all required educational credentials (degree level, field of study, and any additional certifications) needed for Texas superintendent certification.

2. Experience Requirements: Specify the minimum years and type of administrative/managerial experience required for Texas superintendent certification. Based on the principal's current 4 years of principal experience, determine if additional experience is needed.

3. Certification Process: Identify the specific certification coursework or training programs required (including course numbers or program names if specified by Texas), and state whether a certification examination is required and what it is called.

4. Career Timeline Analysis: Calculate the minimum total number of years of professional education experience typically required to progress from beginning teacher (with no prior experience) to becoming eligible for superintendent certification in Texas. This should account for: prerequisite teaching experience before becoming an administrator, the required managerial experience, and any additional time needed for certification coursework.

5. Program Reference: Provide at least one specific accredited Texas university that offers a superintendent certification program, including the program name and relevant details about how it meets Texas Education Agency requirements.

For each section (1-5), include supporting URL references from official Texas Education Agency sources or accredited Texas university program websites.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EducationalPrereqsExtraction(BaseModel):
    masters_degree_requirement_statement: Optional[str] = None
    degree_field_specification_statement: Optional[str] = None
    principal_certificate_requirement_statement: Optional[str] = None
    accreditation_requirement_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ExperienceRequirementsExtraction(BaseModel):
    required_years_statement: Optional[str] = None
    experience_type_statement: Optional[str] = None
    current_experience_assessment_statement: Optional[str] = None
    principal_experience_qualification_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CertificationProcessExtraction(BaseModel):
    coursework_requirement_statement: Optional[str] = None
    field_experience_requirement_statement: Optional[str] = None
    examination_requirement_statement: Optional[str] = None
    coursework_content_examples_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TimelineAnalysisExtraction(BaseModel):
    teaching_prerequisite_years_statement: Optional[str] = None
    administrative_experience_years_statement: Optional[str] = None
    minimum_total_calculation_statement: Optional[str] = None
    realistic_timeline_context_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProgramReferenceExtraction(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    tea_alignment_statement: Optional[str] = None
    additional_program_details_statement: Optional[str] = None
    program_url: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareerPlanExtraction(BaseModel):
    educational: Optional[EducationalPrereqsExtraction] = None
    experience: Optional[ExperienceRequirementsExtraction] = None
    certification: Optional[CertificationProcessExtraction] = None
    timeline: Optional[TimelineAnalysisExtraction] = None
    program: Optional[ProgramReferenceExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_plan() -> str:
    return """
    Extract the specific statements and URLs the answer uses to support each of the five required sections for Texas superintendent certification.

    For each section, return the following fields. If an item is not explicitly stated in the answer, set it to null. For URLs, only include URLs shown in the answer text; do not invent.

    educational:
      - masters_degree_requirement_statement: the sentence or bullet indicating that a master's degree or higher is required.
      - degree_field_specification_statement: the sentence indicating acceptable degree fields (e.g., educational leadership/administration or related).
      - principal_certificate_requirement_statement: the sentence indicating a current Texas Principal Certificate (or equivalent) is required.
      - accreditation_requirement_statement: the sentence indicating the degree must be from a regionally accredited institution.
      - sources: list all TEA or accredited Texas university URLs the answer cites for educational prerequisites.

    experience:
      - required_years_statement: the sentence stating a minimum of (e.g.,) 3 years creditable managerial/administrative experience.
      - experience_type_statement: the sentence stating the experience must be in a public or private school setting.
      - current_experience_assessment_statement: the sentence assessing whether 4 years of principal experience meets the requirement.
      - principal_experience_qualification_statement: the sentence stating that principal experience counts toward the managerial requirement.
      - sources: list all TEA or accredited Texas university URLs the answer cites for experience requirements.

    certification:
      - coursework_requirement_statement: the sentence stating completion of designated superintendent preparation program coursework is required.
      - field_experience_requirement_statement: the sentence indicating a practicum/field experience/internship is required.
      - examination_requirement_statement: the sentence stating passing the TExES Superintendent Certification Examination is required.
      - coursework_content_examples_statement: the sentence listing example coursework content areas (e.g., legal, finance, leadership).
      - sources: list TEA or accredited Texas university URLs cited for certification steps.

    timeline:
      - teaching_prerequisite_years_statement: the sentence stating typical prerequisite teaching experience years before becoming an administrator (e.g., ~2 years).
      - administrative_experience_years_statement: the sentence stating 3 years of administrative experience requirement.
      - minimum_total_calculation_statement: the sentence that computes/claims minimum total years from beginning teacher to superintendent eligibility (e.g., ~5 years) considering teaching + admin + coursework time.
      - realistic_timeline_context_statement: the sentence acknowledging realistic progression commonly takes 15–20 years.
      - sources: list TEA or university URLs cited for timeline items (e.g., principal certification prerequisites, superintendent requirements).

    program:
      - university_name: the specific accredited Texas university name that offers superintendent certification (choose the first if multiple are given).
      - program_name: the specific program name or certification designation at that university.
      - tea_alignment_statement: the sentence that indicates how the program meets TEA requirements (e.g., TEA-approved, includes practicum, prepares for TExES).
      - additional_program_details_statement: any sentence with extra details (e.g., online format, GPA minimum, cohort structure).
      - program_url: the specific URL of the university’s superintendent certification program page.
      - sources: list any other URLs cited in the answer relevant to the program or TEA requirements.

    Return a single JSON object with keys: educational, experience, certification, timeline, program.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def nonempty(text: Optional[str]) -> bool:
    return bool(text and text.strip())


def combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            val = u.strip()
            if not val:
                continue
            if val not in seen:
                seen.add(val)
                combined.append(val)
    return combined


def as_list(possible_url: Optional[str]) -> List[str]:
    return [possible_url] if nonempty(possible_url) else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_educational_prereqs(evaluator: Evaluator, root: VerificationNode, data: CareerPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="educational_prerequisites",
        desc="Verify that educational credential requirements for Texas superintendent certification are correctly identified",
        parent=root,
        critical=False  # Adjusted to allow mixed critical leaves within this category
    )
    edu = data.educational or EducationalPrereqsExtraction()

    # URL presence check (critical)
    edu_urls_present = evaluator.add_custom_node(
        result=len(edu.sources) > 0,
        id="educational_url_reference",
        desc="Provides URL reference from Texas Education Agency or accredited Texas university supporting educational requirements",
        parent=node,
        critical=True
    )

    # Masters degree requirement existence (critical precondition)
    masters_exists = evaluator.add_custom_node(
        result=nonempty(edu.masters_degree_requirement_statement),
        id="masters_degree_requirement_exists",
        desc="Answer includes a statement that a master's degree or higher is required",
        parent=node,
        critical=True
    )

    # Verify masters degree requirement (critical)
    masters_leaf = evaluator.add_leaf(
        id="masters_degree_requirement",
        desc="Answer states that a master's degree or higher from a regionally accredited institution is required",
        parent=node,
        critical=True
    )
    masters_claim = "Texas superintendent certification requires a master's degree or higher."
    await evaluator.verify(
        claim=masters_claim,
        node=masters_leaf,
        sources=edu.sources,
        additional_instruction="Check TEA policy pages or accredited Texas university program pages that explicitly state a master's degree (or higher) is required.",
        extra_prerequisites=[masters_exists, edu_urls_present]
    )

    # Principal certificate requirement existence (critical precondition)
    principal_cert_exists = evaluator.add_custom_node(
        result=nonempty(edu.principal_certificate_requirement_statement),
        id="principal_certificate_requirement_exists",
        desc="Answer includes a statement that a current Texas Principal Certificate (or equivalent) is required",
        parent=node,
        critical=True
    )

    # Verify principal certificate requirement (critical)
    principal_cert_leaf = evaluator.add_leaf(
        id="principal_certificate_requirement",
        desc="Answer states that a current Texas Principal Certificate, Mid-management Certificate, or equivalent administrative certificate is required",
        parent=node,
        critical=True
    )
    principal_cert_claim = "Texas superintendent certification requires holding a current Texas Principal Certificate or equivalent administrative certificate."
    await evaluator.verify(
        claim=principal_cert_claim,
        node=principal_cert_leaf,
        sources=edu.sources,
        additional_instruction="Look for TEA certification requirement pages indicating that superintendent candidates must hold a principal certificate or equivalent.",
        extra_prerequisites=[principal_cert_exists, edu_urls_present]
    )

    # Degree fields existence (non-critical precondition)
    fields_exist = evaluator.add_custom_node(
        result=nonempty(edu.degree_field_specification_statement),
        id="degree_field_specification_exists",
        desc="Answer includes acceptable fields of study for the master's degree",
        parent=node,
        critical=False
    )

    # Verify acceptable fields (non-critical)
    fields_leaf = evaluator.add_leaf(
        id="degree_field_specification",
        desc="Answer specifies acceptable fields of study for the master's degree (educational leadership, educational administration, or related field)",
        parent=node,
        critical=False
    )
    fields_claim = "Acceptable master's degree fields for superintendent certification include educational leadership, educational administration, or a closely related field."
    await evaluator.verify(
        claim=fields_claim,
        node=fields_leaf,
        sources=edu.sources,
        additional_instruction="Check program admission or certification requirement pages that accept education-related master's fields or related disciplines.",
        extra_prerequisites=[fields_exist]
    )

    # Accreditation requirement existence (non-critical precondition)
    accred_exists = evaluator.add_custom_node(
        result=nonempty(edu.accreditation_requirement_statement),
        id="accreditation_requirement_exists",
        desc="Answer includes that the degree must be from a regionally accredited institution",
        parent=node,
        critical=False
    )

    # Verify accreditation (non-critical)
    accred_leaf = evaluator.add_leaf(
        id="accreditation_requirement",
        desc="Answer specifies that the degree must be from a regionally accredited institution",
        parent=node,
        critical=False
    )
    accred_claim = "The master's degree must be from a regionally accredited institution for superintendent certification."
    await evaluator.verify(
        claim=accred_claim,
        node=accred_leaf,
        sources=edu.sources,
        additional_instruction="Look for TEA or program pages that specify degrees must be earned from regionally accredited institutions.",
        extra_prerequisites=[accred_exists]
    )


async def verify_experience_requirements(evaluator: Evaluator, root: VerificationNode, data: CareerPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="experience_requirements",
        desc="Verify that experience requirements are correctly identified and properly assessed for the scenario",
        parent=root,
        critical=False
    )
    exp = data.experience or ExperienceRequirementsExtraction()

    # URL presence check (critical)
    exp_urls_present = evaluator.add_custom_node(
        result=len(exp.sources) > 0,
        id="experience_url_reference",
        desc="Provides URL reference supporting experience requirements",
        parent=node,
        critical=True
    )

    # Required years existence (critical precondition)
    years_exists = evaluator.add_custom_node(
        result=nonempty(exp.required_years_statement),
        id="minimum_experience_years_exists",
        desc="Answer includes the minimum years of managerial/administrative experience",
        parent=node,
        critical=True
    )

    # Verify minimum years (critical)
    years_leaf = evaluator.add_leaf(
        id="minimum_experience_years",
        desc="Answer correctly states that 3 years of creditable managerial/administrative experience is required",
        parent=node,
        critical=True
    )
    years_claim = "Texas superintendent certification requires at least 3 years of creditable managerial/administrative experience."
    await evaluator.verify(
        claim=years_claim,
        node=years_leaf,
        sources=exp.sources,
        additional_instruction="Confirm on TEA pages that superintendent certification requires a minimum of 3 years of managerial/administrative experience (e.g., principal, assistant principal).",
        extra_prerequisites=[years_exists, exp_urls_present]
    )

    # Experience setting existence (critical precondition)
    setting_exists = evaluator.add_custom_node(
        result=nonempty(exp.experience_type_statement),
        id="experience_type_specification_exists",
        desc="Answer includes the type of setting required for experience (public/private school)",
        parent=node,
        critical=True
    )

    # Verify experience type (critical)
    setting_leaf = evaluator.add_leaf(
        id="experience_type_specification",
        desc="Answer specifies that experience must be in a public or private school setting",
        parent=node,
        critical=True
    )
    setting_claim = "The required managerial/administrative experience must be in a public or private school setting."
    await evaluator.verify(
        claim=setting_claim,
        node=setting_leaf,
        sources=exp.sources,
        additional_instruction="Check whether TEA or program pages specify the experience context as in a school district or school setting.",
        extra_prerequisites=[setting_exists, exp_urls_present]
    )

    # Assessment existence (critical precondition)
    assess_exists = evaluator.add_custom_node(
        result=nonempty(exp.current_experience_assessment_statement),
        id="current_experience_assessment_exists",
        desc="Answer assesses whether 4 years of principal experience meets the requirement",
        parent=node,
        critical=True
    )

    # Verify assessment that 4 years meets requirement (critical but logical check)
    assess_leaf = evaluator.add_leaf(
        id="current_experience_assessment",
        desc="Answer correctly determines that the principal with 4 years of experience meets the requirement",
        parent=node,
        critical=True
    )
    assess_claim = "Given a requirement of 3 years of creditable managerial experience, a principal with 4 years of principal experience meets the experience requirement for superintendent certification in Texas."
    await evaluator.verify(
        claim=assess_claim,
        node=assess_leaf,
        sources=None,  # logical check; sources validated above
        additional_instruction="This is a logical assessment: if the requirement is >=3 years of managerial experience, then 4 years as a principal satisfies it.",
        extra_prerequisites=[assess_exists, years_leaf]  # depend on years verification passing
    )

    # Principal qualification existence (non-critical precondition)
    principal_qual_exists = evaluator.add_custom_node(
        result=nonempty(exp.principal_experience_qualification_statement),
        id="principal_experience_qualification_exists",
        desc="Answer clarifies principal experience counts as managerial experience",
        parent=node,
        critical=False
    )

    # Verify principal counts (non-critical)
    principal_qual_leaf = evaluator.add_leaf(
        id="principal_experience_qualification",
        desc="Answer clarifies that principal experience counts as managerial experience for superintendent certification",
        parent=node,
        critical=False
    )
    principal_qual_claim = "Principal experience counts as managerial/administrative experience toward Texas superintendent certification."
    await evaluator.verify(
        claim=principal_qual_claim,
        node=principal_qual_leaf,
        sources=exp.sources,
        additional_instruction="Look for TEA language or program pages identifying principal roles as managerial/administrative for superintendent requirements.",
        extra_prerequisites=[principal_qual_exists]
    )


async def verify_certification_process(evaluator: Evaluator, root: VerificationNode, data: CareerPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="certification_process",
        desc="Verify that the certification process steps and requirements are correctly identified",
        parent=root,
        critical=False
    )
    cert = data.certification or CertificationProcessExtraction()

    # URL presence check (critical)
    cert_urls_present = evaluator.add_custom_node(
        result=len(cert.sources) > 0,
        id="certification_url_reference",
        desc="Provides URL reference from TEA or Texas university describing certification process requirements",
        parent=node,
        critical=True
    )

    # Coursework existence (critical precondition)
    coursework_exists = evaluator.add_custom_node(
        result=nonempty(cert.coursework_requirement_statement),
        id="coursework_requirement_exists",
        desc="Answer includes that completion of designated superintendent coursework is required",
        parent=node,
        critical=True
    )

    # Verify coursework requirement (critical)
    coursework_leaf = evaluator.add_leaf(
        id="coursework_requirement",
        desc="Answer identifies that completion of designated superintendent certification coursework is required",
        parent=node,
        critical=True
    )
    coursework_claim = "Completion of an approved superintendent preparation program's required coursework is required for superintendent certification in Texas."
    await evaluator.verify(
        claim=coursework_claim,
        node=coursework_leaf,
        sources=cert.sources,
        additional_instruction="Verify on TEA or TEA-approved program pages that superintendent candidates must complete designated program coursework.",
        extra_prerequisites=[coursework_exists, cert_urls_present]
    )

    # Field experience existence (critical precondition)
    field_exists = evaluator.add_custom_node(
        result=nonempty(cert.field_experience_requirement_statement),
        id="field_experience_requirement_exists",
        desc="Answer includes requirement for practicum/field experiences/internship",
        parent=node,
        critical=True
    )

    # Verify field experience (critical)
    field_leaf = evaluator.add_leaf(
        id="field_experience_requirement",
        desc="Answer identifies that field experiences, practicum, or internship components are required",
        parent=node,
        critical=True
    )
    field_claim = "Superintendent certification programs in Texas require field experiences such as a practicum or internship."
    await evaluator.verify(
        claim=field_claim,
        node=field_leaf,
        sources=cert.sources,
        additional_instruction="Confirm that superintendent programs include structured field experiences (e.g., practicum/internship) per TEA or program descriptions.",
        extra_prerequisites=[field_exists, cert_urls_present]
    )

    # Exam existence (critical precondition)
    exam_exists = evaluator.add_custom_node(
        result=nonempty(cert.examination_requirement_statement),
        id="examination_requirement_exists",
        desc="Answer includes requirement for passing the TExES Superintendent Certification Examination",
        parent=node,
        critical=True
    )

    # Verify examination requirement (critical)
    exam_leaf = evaluator.add_leaf(
        id="examination_requirement",
        desc="Answer states that passing the TExES Superintendent Certification Examination is required",
        parent=node,
        critical=True
    )
    exam_claim = "Passing the TExES Superintendent Certification Examination is required for superintendent certification in Texas."
    await evaluator.verify(
        claim=exam_claim,
        node=exam_leaf,
        sources=cert.sources,
        additional_instruction="Check TEA test requirements pages or TEA-approved program pages referencing the TExES Superintendent exam.",
        extra_prerequisites=[exam_exists, cert_urls_present]
    )

    # Coursework content examples existence (non-critical precondition)
    content_exists = evaluator.add_custom_node(
        result=nonempty(cert.coursework_content_examples_statement),
        id="coursework_content_description_exists",
        desc="Answer includes examples of coursework content areas",
        parent=node,
        critical=False
    )

    # Verify coursework content examples (non-critical)
    content_leaf = evaluator.add_leaf(
        id="coursework_content_description",
        desc="Answer provides examples of coursework content areas (such as legal aspects, finance, leadership)",
        parent=node,
        critical=False
    )
    content_claim = "Superintendent certification coursework commonly includes legal aspects, school finance, and leadership."
    await evaluator.verify(
        claim=content_claim,
        node=content_leaf,
        sources=cert.sources,
        additional_instruction="Check program course lists or handbooks for courses/topics like school law, finance, and district leadership.",
        extra_prerequisites=[content_exists]
    )


async def verify_timeline_analysis(evaluator: Evaluator, root: VerificationNode, data: CareerPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="career_timeline_analysis",
        desc="Verify that career progression timeline is accurately calculated and contextualized",
        parent=root,
        critical=False
    )
    tl = data.timeline or TimelineAnalysisExtraction()

    # URL presence check (critical)
    tl_urls_present = evaluator.add_custom_node(
        result=len(tl.sources) > 0,
        id="timeline_url_reference",
        desc="Provides URL reference supporting timeline information",
        parent=node,
        critical=True
    )

    # Teaching prerequisite years existence (non-critical precondition)
    teach_exists = evaluator.add_custom_node(
        result=nonempty(tl.teaching_prerequisite_years_statement),
        id="teaching_prerequisite_years_exists",
        desc="Answer includes typical prerequisite teaching experience years",
        parent=node,
        critical=False
    )

    # Verify teaching prerequisite (non-critical)
    teach_leaf = evaluator.add_leaf(
        id="teaching_prerequisite_years",
        desc="Answer identifies that approximately 2 years of teaching experience is typically required before becoming an administrator",
        parent=node,
        critical=False
    )
    teach_claim = "Principal certification in Texas typically requires around two years of successful classroom teaching experience beforehand."
    await evaluator.verify(
        claim=teach_claim,
        node=teach_leaf,
        sources=tl.sources,
        additional_instruction="Look for TEA principal certification prerequisites or university program pages indicating ~2 years of successful teaching experience.",
        extra_prerequisites=[teach_exists, tl_urls_present]
    )

    # Administrative experience years existence (non-critical precondition)
    admin_exists = evaluator.add_custom_node(
        result=nonempty(tl.administrative_experience_years_statement),
        id="administrative_experience_years_exists",
        desc="Answer includes the 3 years administrative experience requirement",
        parent=node,
        critical=False
    )

    # Verify administrative experience years (non-critical)
    admin_leaf = evaluator.add_leaf(
        id="administrative_experience_years",
        desc="Answer correctly identifies the 3 years of required administrative experience",
        parent=node,
        critical=False
    )
    admin_claim = "Eligibility for Texas superintendent certification includes a minimum of 3 years of administrative/managerial experience."
    await evaluator.verify(
        claim=admin_claim,
        node=admin_leaf,
        sources=tl.sources,
        additional_instruction="Confirm TEA superintendent requirements indicating at least 3 years of managerial/administrative experience.",
        extra_prerequisites=[admin_exists, tl_urls_present]
    )

    # Minimum total calculation existence (critical precondition)
    total_exists = evaluator.add_custom_node(
        result=nonempty(tl.minimum_total_calculation_statement),
        id="minimum_total_calculation_exists",
        desc="Answer includes a minimum total years calculation",
        parent=node,
        critical=True
    )

    # Verify minimum total calculation ~5 years (critical; logical check)
    total_leaf = evaluator.add_leaf(
        id="minimum_total_calculation",
        desc="Answer correctly calculates minimum total years from beginning teacher to superintendent eligibility (approximately 5 years)",
        parent=node,
        critical=True
    )
    total_claim = "A minimal pathway is approximately 5 years (e.g., ~2 years teaching + 3 years administrative experience), excluding additional time to complete certification coursework."
    await evaluator.verify(
        claim=total_claim,
        node=total_leaf,
        sources=None,
        additional_instruction="This is a logical synthesis: combine typical teaching prerequisite (~2 years) with the superintendent managerial requirement (3 years) to reach ~5 years minimum, without counting time to finish certification coursework.",
        extra_prerequisites=[total_exists, admin_leaf, teach_leaf]
    )

    # Realistic timeline existence (non-critical precondition)
    realistic_exists = evaluator.add_custom_node(
        result=nonempty(tl.realistic_timeline_context_statement),
        id="realistic_timeline_context_exists",
        desc="Answer acknowledges realistic career progression timeline",
        parent=node,
        critical=False
    )

    # Verify realistic timeline 15–20 years (non-critical; logical/contextual)
    realistic_leaf = evaluator.add_leaf(
        id="realistic_timeline_context",
        desc="Answer acknowledges that typical career progression realistically takes 15-20 years",
        parent=node,
        critical=False
    )
    realistic_claim = "In practice, progressing from teacher to superintendent often spans 15–20 years when accounting for experience accrual, advanced degrees, and program completion."
    await evaluator.verify(
        claim=realistic_claim,
        node=realistic_leaf,
        sources=tl.sources,
        additional_instruction="Accept reasonable contextual acknowledgments on program pages or TEA-related guidance; allow general profession data if clearly relevant.",
        extra_prerequisites=[realistic_exists]
    )


async def verify_program_reference(evaluator: Evaluator, root: VerificationNode, data: CareerPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="program_reference",
        desc="Verify that at least one specific accredited Texas university superintendent program is correctly identified with supporting details",
        parent=root,
        critical=False
    )
    prog = data.program or ProgramReferenceExtraction()

    # Program URL presence check (critical)
    program_url_present = evaluator.add_custom_node(
        result=nonempty(prog.program_url),
        id="program_url_reference",
        desc="Provides URL reference to the specific university program website",
        parent=node,
        critical=True
    )

    # University identification existence (critical precondition)
    uni_exists = evaluator.add_custom_node(
        result=nonempty(prog.university_name),
        id="university_identification_exists",
        desc="Answer includes a specific accredited Texas university",
        parent=node,
        critical=True
    )

    # Verify university identification (critical)
    uni_leaf = evaluator.add_leaf(
        id="university_identification",
        desc="Answer identifies at least one specific accredited Texas university offering superintendent certification",
        parent=node,
        critical=True
    )
    uni_claim = f"This page is from a Texas university and it offers a superintendent certification program."
    await evaluator.verify(
        claim=uni_claim,
        node=uni_leaf,
        sources=prog.program_url,
        additional_instruction="Verify that the page belongs to a Texas-based university and explicitly describes a superintendent certification program.",
        extra_prerequisites=[uni_exists, program_url_present]
    )

    # Program name existence (critical precondition)
    program_name_exists = evaluator.add_custom_node(
        result=nonempty(prog.program_name),
        id="program_name_exists",
        desc="Answer includes the specific program name or certification designation",
        parent=node,
        critical=True
    )

    # Verify program name (critical)
    program_name_leaf = evaluator.add_leaf(
        id="program_name",
        desc="Answer provides the specific program name or certification designation",
        parent=node,
        critical=True
    )
    program_name_claim = f"The program name is '{prog.program_name}' (or a close variant)."
    await evaluator.verify(
        claim=program_name_claim,
        node=program_name_leaf,
        sources=prog.program_url,
        additional_instruction="Allow reasonable variants or minor formatting differences when matching the program name as shown on the university page.",
        extra_prerequisites=[program_name_exists, program_url_present]
    )

    # TEA alignment existence (critical precondition)
    tea_align_exists = evaluator.add_custom_node(
        result=nonempty(prog.tea_alignment_statement),
        id="tea_alignment_exists",
        desc="Answer indicates how the program meets TEA requirements",
        parent=node,
        critical=True
    )

    # Verify TEA alignment (critical)
    tea_align_leaf = evaluator.add_leaf(
        id="tea_alignment",
        desc="Answer indicates how the program meets Texas Education Agency requirements",
        parent=node,
        critical=True
    )
    tea_align_claim = "This superintendent certification program meets TEA requirements (e.g., TEA-approved, includes required practicum/field experience, prepares candidates for the TExES Superintendent exam)."
    await evaluator.verify(
        claim=tea_align_claim,
        node=tea_align_leaf,
        sources=combine_sources(as_list(prog.program_url), prog.sources),
        additional_instruction="Look for explicit statements about TEA approval/requirements, practicum/internship inclusion, and exam preparation on the program page.",
        extra_prerequisites=[tea_align_exists, program_url_present]
    )

    # Additional program details existence (non-critical precondition)
    details_exist = evaluator.add_custom_node(
        result=nonempty(prog.additional_program_details_statement),
        id="additional_program_details_exists",
        desc="Answer includes additional relevant program details",
        parent=node,
        critical=False
    )

    # Verify additional program details (non-critical)
    details_leaf = evaluator.add_leaf(
        id="additional_program_details",
        desc="Answer provides additional relevant details such as delivery format, GPA requirements, or program structure",
        parent=node,
        critical=False
    )
    details_claim = f"The program page provides additional relevant details such as delivery format, GPA requirements, or program structure."
    await evaluator.verify(
        claim=details_claim,
        node=details_leaf,
        sources=prog.program_url,
        additional_instruction="Check if the program page includes any of: online/face-to-face format, GPA minimums, cohort/credit structure, or similar.",
        extra_prerequisites=[details_exist, program_url_present]
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Texas superintendent certification career advancement plan.
    Converts rubric criteria into a verification tree and returns the evaluation summary.
    """
    # Initialize evaluator (Note: set root non-critical to allow mixed children; criticality enforced at leaves)
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_career_plan(),
        template_class=CareerPlanExtraction,
        extraction_name="career_plan_extraction",
    )

    # Build verification subtrees for each section
    await verify_educational_prereqs(evaluator, root, extraction)
    await verify_experience_requirements(evaluator, root, extraction)
    await verify_certification_process(evaluator, root, extraction)
    await verify_timeline_analysis(evaluator, root, extraction)
    await verify_program_reference(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()