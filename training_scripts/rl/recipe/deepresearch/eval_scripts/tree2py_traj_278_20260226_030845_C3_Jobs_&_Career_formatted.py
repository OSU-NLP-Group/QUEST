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
TASK_ID = "ohio_educator_career_path"
TASK_DESCRIPTION = """
In Ohio, an educator wants to progress from being a new teacher to becoming a school district superintendent. What is the complete sequential path of licensure requirements and mandatory experience this educator must fulfill to achieve this career goal?

Your answer must provide the following information for each stage:
1. Initial Teaching License Stage: The educational degree required, the type of program that must be completed, and the exams that must be passed to obtain an initial teaching license in Ohio.
2. Teaching Experience Stage: The minimum number of years of teaching experience required under a standard or professional teaching license before being eligible to pursue a principal license, and any requirements about the grade levels where this experience must be obtained.
3. Principal License Stage: The educational degree required, the type of preparation program that must be completed, any recommendation requirements from the preparation institution, and the specific licensure exam that must be passed to obtain a principal license in Ohio.
4. Administrative Experience Stage: The minimum number of years of administrative experience required as a principal or administrative specialist (while holding the respective license) before being eligible to pursue a superintendent license.
5. Superintendent License Stage: Confirmation of the educational degree requirement, the specific licensure exams that must be passed (including both the educational leadership exam and the superintendent-specific assessment), and any additional requirements such as background checks.

Each stage must be documented with specific details about Ohio's requirements, and all information must be verifiable through official Ohio State Board of Education sources or accredited Ohio universities' licensure program information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class InitialTeachingStage(BaseModel):
    degree_required: Optional[str] = None
    preparation_program_type: Optional[str] = None
    exams: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class TeachingExperienceStage(BaseModel):
    min_years: Optional[str] = None
    grade_levels_requirement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PrincipalPreparationRequirements(BaseModel):
    preparation_program_type: Optional[str] = None
    program_recommendation_requirement: Optional[str] = None
    exam_required: Optional[str] = None  # Expect "OAE 015" or equivalent phrasing
    sources: List[str] = Field(default_factory=list)


class PrincipalLicenseStage(BaseModel):
    degree_required: Optional[str] = None
    preparation: Optional[PrincipalPreparationRequirements] = None
    sources: List[str] = Field(default_factory=list)


class AdministrativeExperienceStage(BaseModel):
    min_years: Optional[str] = None
    license_condition: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SuperintendentLicenseStage(BaseModel):
    degree_required: Optional[str] = None
    licensure_exams: List[str] = Field(default_factory=list)  # Should include OAE 015 and Praxis 6991
    additional_requirements: List[str] = Field(default_factory=list)  # e.g., background checks
    sources: List[str] = Field(default_factory=list)


class EducatorPathExtraction(BaseModel):
    initial_stage: Optional[InitialTeachingStage] = None
    teaching_experience_stage: Optional[TeachingExperienceStage] = None
    principal_stage: Optional[PrincipalLicenseStage] = None
    administrative_experience_stage: Optional[AdministrativeExperienceStage] = None
    superintendent_stage: Optional[SuperintendentLicenseStage] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_educator_path() -> str:
    return """
    Extract the complete sequential Ohio educator licensure path as presented in the answer and return a structured JSON object with the following stages and fields. Extract ONLY information explicitly stated in the answer. Also extract all stage-specific source URLs cited in the answer (Ohio Department/State Board official pages or accredited Ohio university licensure pages). If something is missing in the answer, return null for that field or an empty list for arrays.

    Required JSON structure:
    {
      "initial_stage": {
        "degree_required": string | null,
        "preparation_program_type": string | null,
        "exams": [string, ...],
        "sources": [url, ...]
      },
      "teaching_experience_stage": {
        "min_years": string | null,
        "grade_levels_requirement": string | null,
        "sources": [url, ...]
      },
      "principal_stage": {
        "degree_required": string | null,
        "preparation": {
          "preparation_program_type": string | null,
          "program_recommendation_requirement": string | null,
          "exam_required": string | null,
          "sources": [url, ...]
        },
        "sources": [url, ...]
      },
      "administrative_experience_stage": {
        "min_years": string | null,
        "license_condition": string | null,
        "sources": [url, ...]
      },
      "superintendent_stage": {
        "degree_required": string | null,
        "licensure_exams": [string, ...],
        "additional_requirements": [string, ...],
        "sources": [url, ...]
      }
    }

    Notes:
    - exams and licensure_exams should include exam names/codes mentioned (e.g., "OAE 015 Educational Leadership", "Praxis 6991 School Superintendent Assessment").
    - sources must be actual URLs explicitly present in the answer (plain or markdown). Do not invent URLs.
    - If the answer mentions general sources without URLs, do not include them.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_string(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


def _has_nonempty_list(lst: Optional[List[str]]) -> bool:
    return bool(lst) and len(lst) > 0


def _collect_sources(*lists: Optional[List[str]]) -> List[str]:
    sources: List[str] = []
    for lst in lists:
        if lst:
            sources.extend(lst)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in sources:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_initial_stage(evaluator: Evaluator, parent_node, data: EducatorPathExtraction) -> None:
    stage = data.initial_stage or InitialTeachingStage()
    node = evaluator.add_parallel(
        id="Initial_Teaching_Credential",
        desc="Validates the requirements for obtaining an initial teaching license in Ohio",
        parent=parent_node,
        critical=False
    )

    # Existence checks for required fields and sources (critical gates)
    evaluator.add_custom_node(
        result=_has_nonempty_string(stage.degree_required) and
               _has_nonempty_string(stage.preparation_program_type) and
               _has_nonempty_list(stage.exams),
        id="Initial_Stage_Fields_Provided",
        desc="Answer provides degree, program type, and required exams for the initial license stage",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_nonempty_list(stage.sources),
        id="Initial_Stage_Sources_Present",
        desc="Stage has at least one official/source URL cited for verification",
        parent=node,
        critical=True
    )

    # Bachelor degree requirement
    leaf_degree = evaluator.add_leaf(
        id="Bachelor_Degree_Requirement",
        desc="Candidate must hold a bachelor's degree from an accredited institution",
        parent=node,
        critical=True
    )
    claim_degree = ("Ohio requirements for obtaining an initial teaching license include holding a bachelor's "
                    "degree from an accredited institution.")
    await evaluator.verify(
        claim=claim_degree,
        node=leaf_degree,
        sources=stage.sources,
        additional_instruction="Verify on Ohio official/licensure pages or accredited Ohio university licensure pages that a bachelor’s degree is required for initial teacher licensure."
    )

    # Teacher preparation program requirement
    leaf_program = evaluator.add_leaf(
        id="Teacher_Preparation_Program",
        desc="Candidate must complete a state-approved teacher preparation program",
        parent=node,
        critical=True
    )
    claim_program = ("To obtain an initial Ohio teaching license, completion of a state-approved teacher preparation "
                     "program is required.")
    await evaluator.verify(
        claim=claim_program,
        node=leaf_program,
        sources=stage.sources,
        additional_instruction="Confirm the requirement for completing a state-approved teacher preparation program for initial Ohio licensure."
    )

    # Teaching certification exams requirement
    leaf_exams = evaluator.add_leaf(
        id="Teaching_Certification_Exams",
        desc="Candidate must pass required Ohio teaching certification exams",
        parent=node,
        critical=True
    )
    claim_exams = ("Initial Ohio teacher licensure requires passing the required Ohio teaching certification exams "
                   "(such as OAE content/assessment or other required Ohio licensure exams).")
    await evaluator.verify(
        claim=claim_exams,
        node=leaf_exams,
        sources=stage.sources,
        additional_instruction="Check that Ohio's initial teacher licensure includes passing required certification exams; allow variation in exact exam names by area."
    )


async def verify_teaching_experience_stage(evaluator: Evaluator, parent_node, data: EducatorPathExtraction) -> None:
    stage = data.teaching_experience_stage or TeachingExperienceStage()
    node = evaluator.add_parallel(
        id="Teaching_Experience_Stage",
        desc="Validates the required teaching experience before pursuing principal license",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_nonempty_string(stage.min_years) and _has_nonempty_string(stage.grade_levels_requirement),
        id="Teaching_Stage_Fields_Provided",
        desc="Answer provides min years and grade level applicability for teaching experience stage",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_nonempty_list(stage.sources),
        id="Teaching_Stage_Sources_Present",
        desc="Stage has at least one official/source URL cited for verification",
        parent=node,
        critical=True
    )

    # Two years of teaching experience under standard/professional license
    leaf_years = evaluator.add_leaf(
        id="Two_Years_Teaching",
        desc="Candidate must have two years of successful teaching experience under a standard or professional teaching license",
        parent=node,
        critical=True
    )
    claim_years = ("Eligibility for an Ohio principal license requires at least two years of successful teaching "
                   "experience under a standard or professional teaching license.")
    await evaluator.verify(
        claim=claim_years,
        node=leaf_years,
        sources=stage.sources,
        additional_instruction="Confirm the minimum of two years successful teaching experience under the appropriate license before principal license eligibility."
    )

    # Appropriate grade levels requirement
    leaf_grades = evaluator.add_leaf(
        id="Appropriate_Grade_Levels",
        desc="Teaching experience must be in the ages and grade levels for which the principal license is sought",
        parent=node,
        critical=True
    )
    claim_grades = ("The teaching experience must be obtained in the ages/grade levels for which the principal "
                    "license is sought.")
    await evaluator.verify(
        claim=claim_grades,
        node=leaf_grades,
        sources=stage.sources,
        additional_instruction="Verify that Ohio requires the teaching experience to align with the ages/grade levels of the intended principal license."
    )


async def verify_principal_license_stage(evaluator: Evaluator, parent_node, data: EducatorPathExtraction) -> None:
    stage = data.principal_stage or PrincipalLicenseStage()
    node = evaluator.add_parallel(
        id="Principal_License_Stage",
        desc="Validates the requirements for obtaining a principal license in Ohio",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_nonempty_string(stage.degree_required) and (stage.preparation is not None),
        id="Principal_Stage_Fields_Provided",
        desc="Answer provides master’s degree and preparation details for the principal license stage",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_nonempty_list(stage.sources) or (_has_nonempty_list(stage.preparation.sources) if stage.preparation else False),
        id="Principal_Stage_Sources_Present",
        desc="Principal stage has at least one official/source URL cited for verification",
        parent=node,
        critical=True
    )

    # Master's degree requirement
    leaf_masters = evaluator.add_leaf(
        id="Master_Degree_Requirement",
        desc="Candidate must hold a master's degree from an accredited university",
        parent=node,
        critical=True
    )
    claim_masters = "Ohio principal licensure requires holding a master’s degree from an accredited university."
    await evaluator.verify(
        claim=claim_masters,
        node=leaf_masters,
        sources=_collect_sources(stage.sources, stage.preparation.sources if stage.preparation else []),
        additional_instruction="Verify that a master’s degree is required for the Ohio principal license."
    )

    # Preparation requirements (critical aggregate)
    prep_node = evaluator.add_parallel(
        id="Principal_Preparation_Requirements",
        desc="Validates completion of principal preparation program requirements",
        parent=node,
        critical=True
    )

    # Approved principal preparation program
    leaf_approved = evaluator.add_leaf(
        id="Approved_Principal_Program",
        desc="Candidate must complete an approved principal preparation program",
        parent=prep_node,
        critical=True
    )
    claim_approved = "Completion of an approved principal preparation program is required for Ohio principal licensure."
    await evaluator.verify(
        claim=claim_approved,
        node=leaf_approved,
        sources=_collect_sources(stage.sources, stage.preparation.sources if stage.preparation else []),
        additional_instruction="Confirm that Ohio requires completion of an approved principal preparation program."
    )

    # Program recommendation
    leaf_rec = evaluator.add_leaf(
        id="Program_Recommendation",
        desc="Candidate must receive a recommendation from the dean or head of teacher education at the institution where the principal preparation program was completed",
        parent=prep_node,
        critical=True
    )
    claim_rec = ("A recommendation from the dean or head of teacher education at the institution where the principal "
                 "program was completed is required for Ohio principal licensure.")
    await evaluator.verify(
        claim=claim_rec,
        node=leaf_rec,
        sources=_collect_sources(stage.sources, stage.preparation.sources if stage.preparation else []),
        additional_instruction="Verify the institutional recommendation requirement (dean/head of teacher education) for principal licensure."
    )

    # OAE 015 Educational Leadership exam
    leaf_oae015 = evaluator.add_leaf(
        id="OAE_015_Exam",
        desc="Candidate must pass the Ohio Assessment for Educators (OAE) 015 Educational Leadership licensure exam",
        parent=prep_node,
        critical=True
    )
    claim_oae015 = "Passing the OAE 015 Educational Leadership licensure exam is required for Ohio principal licensure."
    await evaluator.verify(
        claim=claim_oae015,
        node=leaf_oae015,
        sources=_collect_sources(stage.sources, stage.preparation.sources if stage.preparation else []),
        additional_instruction="Verify that Ohio principal licensure requires the OAE 015 Educational Leadership exam."
    )


async def verify_admin_experience_stage(evaluator: Evaluator, parent_node, data: EducatorPathExtraction) -> None:
    stage = data.administrative_experience_stage or AdministrativeExperienceStage()
    node = evaluator.add_parallel(
        id="Administrative_Experience_Stage",
        desc="Validates the required administrative experience before pursuing superintendent license",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_nonempty_string(stage.min_years) and _has_nonempty_string(stage.license_condition),
        id="Administrative_Stage_Fields_Provided",
        desc="Answer provides min years and license-condition details for administrative experience stage",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_nonempty_list(stage.sources),
        id="Administrative_Stage_Sources_Present",
        desc="Administrative stage has at least one official/source URL cited for verification",
        parent=node,
        critical=True
    )

    # Three years administrative experience
    leaf_three_years = evaluator.add_leaf(
        id="Three_Years_Administrative",
        desc="Candidate must have three years of successful experience serving as a principal or administrative specialist",
        parent=node,
        critical=True
    )
    claim_three_years = ("Ohio superintendent licensure eligibility requires at least three years of successful "
                         "experience as a principal or administrative specialist.")
    await evaluator.verify(
        claim=claim_three_years,
        node=leaf_three_years,
        sources=stage.sources,
        additional_instruction="Confirm the minimum of three years successful administrative experience (principal or administrative specialist) for superintendent eligibility."
    )

    # Licensed administrative position condition
    leaf_license_cond = evaluator.add_leaf(
        id="Licensed_Administrative_Position",
        desc="The three years of experience must be while holding the respective license (principal license or administrative specialist license)",
        parent=node,
        critical=True
    )
    claim_license_cond = ("The required administrative experience must be obtained while holding the relevant license "
                          "(principal or administrative specialist) in Ohio.")
    await evaluator.verify(
        claim=claim_license_cond,
        node=leaf_license_cond,
        sources=stage.sources,
        additional_instruction="Verify that Ohio requires the administrative experience to be obtained while the candidate holds the appropriate principal/administrative specialist license."
    )


async def verify_superintendent_license_stage(evaluator: Evaluator, parent_node, data: EducatorPathExtraction) -> None:
    stage = data.superintendent_stage or SuperintendentLicenseStage()
    node = evaluator.add_parallel(
        id="Superintendent_License_Stage",
        desc="Validates the requirements for obtaining a superintendent license in Ohio",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_nonempty_string(stage.degree_required) and
               _has_nonempty_list(stage.licensure_exams) and
               _has_nonempty_list(stage.additional_requirements),
        id="Superintendent_Stage_Fields_Provided",
        desc="Answer provides degree, exam list, and additional requirements for superintendent stage",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_nonempty_list(stage.sources),
        id="Superintendent_Stage_Sources_Present",
        desc="Superintendent stage has at least one official/source URL cited for verification",
        parent=node,
        critical=True
    )

    # Master's degree maintained
    leaf_masters_super = evaluator.add_leaf(
        id="Master_Degree_Maintained",
        desc="Candidate must hold a master's degree from an accredited university (continuing requirement)",
        parent=node,
        critical=True
    )
    claim_masters_super = "Ohio superintendent licensure requires holding a master’s degree from an accredited university."
    await evaluator.verify(
        claim=claim_masters_super,
        node=leaf_masters_super,
        sources=stage.sources,
        additional_instruction="Verify that a master’s degree remains a requirement for Ohio superintendent licensure."
    )

    # Superintendent exam requirements (critical aggregate)
    exam_node = evaluator.add_parallel(
        id="Superintendent_Exam_Requirements",
        desc="Candidate must pass required superintendent licensure exams",
        parent=node,
        critical=True
    )

    # OAE 015 confirmation
    leaf_oae015_conf = evaluator.add_leaf(
        id="OAE_015_Confirmation",
        desc="Candidate must have passed the Ohio Assessment for Educators (OAE) 015 Educational Leadership licensure exam",
        parent=exam_node,
        critical=True
    )
    claim_oae015_conf = ("Ohio superintendent licensure requires that the candidate has passed the OAE 015 "
                         "Educational Leadership licensure exam.")
    await evaluator.verify(
        claim=claim_oae015_conf,
        node=leaf_oae015_conf,
        sources=stage.sources,
        additional_instruction="Confirm that OAE 015 Educational Leadership is required/recognized for superintendent licensure per Ohio official or accredited university sources."
    )

    # Praxis 6991 exam
    leaf_praxis6991 = evaluator.add_leaf(
        id="Praxis_6991_Exam",
        desc="Candidate must pass the Praxis 6991 School Superintendent Assessment",
        parent=exam_node,
        critical=True
    )
    claim_praxis6991 = "Passing the Praxis 6991 School Superintendent Assessment is required for Ohio superintendent licensure."
    await evaluator.verify(
        claim=claim_praxis6991,
        node=leaf_praxis6991,
        sources=stage.sources,
        additional_instruction="Verify that Praxis 6991 School Superintendent Assessment is required for Ohio superintendent licensure."
    )

    # Background check requirement
    leaf_bg = evaluator.add_leaf(
        id="Background_Check",
        desc="Candidate must have current background checks on file with the Ohio State Board of Education",
        parent=node,
        critical=True
    )
    claim_bg = "Ohio superintendent licensure requires having current background checks on file with the Ohio State Board of Education."
    await evaluator.verify(
        claim=claim_bg,
        node=leaf_bg,
        sources=stage.sources,
        additional_instruction="Confirm the background check requirement for Ohio superintendent licensure."
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
    Evaluate an answer for the Ohio educator career path licensure task.
    """
    # Initialize evaluator (root sequential, set non-critical to allow partial credit across stages)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Validates the complete sequential career progression path from teaching to superintendent in Ohio",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured educator path information
    extraction = await evaluator.extract(
        prompt=prompt_extract_educator_path(),
        template_class=EducatorPathExtraction,
        extraction_name="ohio_educator_path"
    )

    # Build verification tree following rubric structure
    await verify_initial_stage(evaluator, root, extraction)
    await verify_teaching_experience_stage(evaluator, root, extraction)
    await verify_principal_license_stage(evaluator, root, extraction)
    await verify_admin_experience_stage(evaluator, root, extraction)
    await verify_superintendent_license_stage(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()