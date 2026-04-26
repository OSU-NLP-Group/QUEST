import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nevada_cpa_big4"
TASK_DESCRIPTION = """
I have recently graduated with a bachelor's degree (120 semester hours) in accounting from an accredited U.S. university. My goal is to obtain a Certified Public Accountant (CPA) license in Nevada and start my career at a Big Four accounting firm (Deloitte, PwC, EY, or KPMG). Please provide comprehensive information about: (1) Nevada CPA License Requirements - What total education hours are required for Nevada CPA licensure? What specific accounting coursework requirements must be met? What examinations must I pass and what are the minimum passing scores? What work experience is required, including the number of hours and supervision requirements? (2) Big Four Career Path - What is the typical entry-level position title at Big Four accounting firms? How long does it typically take to be promoted from entry-level to Senior Associate at Big Four firms? For each requirement, please include specific details such as total semester hours, accounting course hour requirements and levels, examination names and minimum scores, work experience hours and supervision qualifications, and typical promotion timeframes in years.
"""

# Ground truth reference for Nevada-specific requirements and Big Four timeline
GROUND_TRUTH_INFO = {
    "nevada_cpa": {
        "total_hours": "150 semester credit hours",
        "upper_level_accounting_hours": "30 semester hours of upper-level (non-introductory) accounting coursework",
        "cpa_exam": "Pass all 4 sections of the Uniform CPA Examination with a minimum score of 75 on each section",
        "ethics_exam": "Pass the AICPA Professional Ethics Examination",
        "work_experience": "2,000 hours (approx. 1 year) of verified experience supervised by an active CPA license holder",
    },
    "big4": {
        "starting_position_examples": "Associate (often Audit/Tax/Advisory Associate), Staff Accountant, or Analyst depending on service line",
        "senior_promotion_timeline": "Typically 2–3 years from entry-level to Senior Associate",
    }
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ValueWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EducationExtraction(BaseModel):
    total_hours: Optional[ValueWithSources] = None
    upper_level_accounting_hours: Optional[ValueWithSources] = None


class CPAExamExtraction(BaseModel):
    sections_required: Optional[str] = None
    minimum_passing_score: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EthicsExamExtraction(BaseModel):
    exam_name: Optional[str] = None
    minimum_passing_score: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WorkExperienceExtraction(BaseModel):
    required_hours: Optional[str] = None
    supervision_requirement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BigFourStartingExtraction(BaseModel):
    title: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BigFourPromotionExtraction(BaseModel):
    timeline_years: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NevadaCPAAndBig4Extraction(BaseModel):
    education: Optional[EducationExtraction] = None
    cpa_exam: Optional[CPAExamExtraction] = None
    ethics_exam: Optional[EthicsExamExtraction] = None
    work_experience: Optional[WorkExperienceExtraction] = None
    big4_starting_position: Optional[BigFourStartingExtraction] = None
    big4_senior_promotion_timeline: Optional[BigFourPromotionExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nevada_cpa_and_big4() -> str:
    return """
    Extract the Nevada CPA licensure requirements and Big Four career path information as presented in the answer. Return a single JSON object with the following structure. Only extract information explicitly stated in the answer.

    {
      "education": {
        "total_hours": {
          "value": "string or null",
          "sources": ["array of URLs explicitly cited for total hours, can be empty"]
        },
        "upper_level_accounting_hours": {
          "value": "string or null",
          "sources": ["array of URLs explicitly cited for accounting coursework, can be empty"]
        }
      },
      "cpa_exam": {
        "sections_required": "string or null",
        "minimum_passing_score": "string or null",
        "sources": ["array of URLs for CPA exam requirements, can be empty"]
      },
      "ethics_exam": {
        "exam_name": "string or null",
        "minimum_passing_score": "string or null",
        "sources": ["array of URLs for ethics exam requirement, can be empty"]
      },
      "work_experience": {
        "required_hours": "string or null",
        "supervision_requirement": "string or null",
        "sources": ["array of URLs for work experience requirement, can be empty"]
      },
      "big4_starting_position": {
        "title": "string or null",
        "sources": ["array of URLs for Big Four entry-level title, can be empty"]
      },
      "big4_senior_promotion_timeline": {
        "timeline_years": "string or null",
        "sources": ["array of URLs for promotion timeline, can be empty"]
      }
    }

    Rules:
    - Use strings for values (e.g., "150 semester hours", "30 upper-level hours", "75").
    - Sources must be actual URLs explicitly present in the answer (plain URLs or markdown links). If none are provided, return an empty array for sources.
    - Do not invent or infer any information not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_sources(lst: Optional[List[str]]) -> List[str]:
    return lst if lst else []


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_nevada_cpa_tree(
    evaluator: Evaluator,
    root_node,
    ext: NevadaCPAAndBig4Extraction,
) -> None:
    # Nevada_CPA_Licensure (critical, parallel)
    nv_node = evaluator.add_parallel(
        id="Nevada_CPA_Licensure",
        desc="Complete all requirements to obtain Nevada CPA license",
        parent=root_node,
        critical=True
    )

    # Education (critical, parallel)
    edu_node = evaluator.add_parallel(
        id="Education",
        desc="Meet education requirements including total hours and accounting coursework",
        parent=nv_node,
        critical=True
    )

    # Total_Hours (critical leaf)
    total_hours_leaf = evaluator.add_leaf(
        id="Total_Hours",
        desc="Complete 150 semester hours of education (candidate currently has 120 semester hours per question context)",
        parent=edu_node,
        critical=True
    )
    total_hours_sources = _safe_sources(
        ext.education.total_hours.sources if ext and ext.education and ext.education.total_hours else []
    )
    total_hours_claim = "Nevada CPA licensure requires a total of 150 semester credit hours."
    await evaluator.verify(
        claim=total_hours_claim,
        node=total_hours_leaf,
        sources=total_hours_sources,
        additional_instruction=(
            "Confirm Nevada's official total education requirement is 150 semester/credit hours. "
            "Treat 'semester hours' and 'credit hours' as equivalent terminology."
        ),
    )

    # Accounting_Coursework (critical, parallel)
    acct_node = evaluator.add_parallel(
        id="Accounting_Coursework",
        desc="Complete required accounting coursework",
        parent=edu_node,
        critical=True
    )

    # Upper_Level_Accounting (critical leaf)
    upper_leaf = evaluator.add_leaf(
        id="Upper_Level_Accounting",
        desc="Complete 30 semester hours of upper-level accounting courses (above introductory level)",
        parent=acct_node,
        critical=True
    )
    upper_sources = _safe_sources(
        ext.education.upper_level_accounting_hours.sources if ext and ext.education and ext.education.upper_level_accounting_hours else []
    )
    upper_claim = (
        "Nevada requires 30 semester hours of upper-level (non-introductory) accounting coursework for CPA licensure."
    )
    await evaluator.verify(
        claim=upper_claim,
        node=upper_leaf,
        sources=upper_sources,
        additional_instruction=(
            "Check Nevada's requirement for accounting coursework units and level; "
            "upper-level/non-introductory equivalences are acceptable."
        ),
    )

    # Examinations (critical, parallel)
    exams_node = evaluator.add_parallel(
        id="Examinations",
        desc="Pass required examinations for CPA licensure",
        parent=nv_node,
        critical=True
    )

    # CPA_Exam (critical leaf)
    cpa_leaf = evaluator.add_leaf(
        id="CPA_Exam",
        desc="Pass all 4 sections of the Uniform CPA Examination with a minimum score of 75 on each section",
        parent=exams_node,
        critical=True
    )
    cpa_sources = _safe_sources(ext.cpa_exam.sources if ext and ext.cpa_exam else [])
    cpa_claim = (
        "Nevada requires candidates to pass all four sections of the Uniform CPA Examination, "
        "with a minimum score of 75 on each section."
    )
    await evaluator.verify(
        claim=cpa_claim,
        node=cpa_leaf,
        sources=cpa_sources,
        additional_instruction=(
            "Verify the pass standard for each CPA Exam section is 75 and four sections must be passed."
        ),
    )

    # Ethics_Exam (critical leaf)
    ethics_leaf = evaluator.add_leaf(
        id="Ethics_Exam",
        desc="Pass the AICPA Professional Ethics Examination",
        parent=exams_node,
        critical=True
    )
    ethics_sources = _safe_sources(ext.ethics_exam.sources if ext and ext.ethics_exam else [])
    ethics_claim = "Nevada requires passing the AICPA Professional Ethics Examination."
    await evaluator.verify(
        claim=ethics_claim,
        node=ethics_leaf,
        sources=ethics_sources,
        additional_instruction=(
            "Confirm Nevada mandates the AICPA Professional Ethics Exam as part of licensure."
        ),
    )

    # Work_Experience (critical leaf)
    work_leaf = evaluator.add_leaf(
        id="Work_Experience",
        desc="Complete 2,000 hours (equivalent to 1 year) of verified work experience supervised by an active CPA license holder",
        parent=nv_node,
        critical=True
    )
    work_sources = _safe_sources(ext.work_experience.sources if ext and ext.work_experience else [])
    work_claim = (
        "Nevada requires 2,000 hours (approximately one year) of verified work experience supervised by an active CPA license holder."
    )
    await evaluator.verify(
        claim=work_claim,
        node=work_leaf,
        sources=work_sources,
        additional_instruction=(
            "Confirm the hours requirement (2,000) and supervision by an active CPA license holder; "
            "phrases like 'one year' equating to 2,000 hours are acceptable."
        ),
    )


async def build_big4_career_tree(
    evaluator: Evaluator,
    root_node,
    ext: NevadaCPAAndBig4Extraction,
) -> None:
    # Big4_Career_Entry (critical, parallel)
    big4_node = evaluator.add_parallel(
        id="Big4_Career_Entry",
        desc="Entry-level position and progression timeline at Big Four accounting firm",
        parent=root_node,
        critical=True
    )

    # Starting_Position (critical leaf)
    start_leaf = evaluator.add_leaf(
        id="Starting_Position",
        desc="Identify typical entry-level position title at Big Four firms (e.g., Staff Accountant, Associate, or Analyst)",
        parent=big4_node,
        critical=True
    )
    start_sources = _safe_sources(ext.big4_starting_position.sources if ext and ext.big4_starting_position else [])
    start_title = ext.big4_starting_position.title if ext and ext.big4_starting_position else None
    if start_title and start_title.strip():
        start_claim = (
            f"At Big Four accounting firms, the typical entry-level position title is '{start_title}', "
            "which is equivalent to common titles like Associate, Staff Accountant, or Analyst depending on service line."
        )
    else:
        start_claim = (
            "Big Four accounting firms typically hire entry-level candidates into roles titled Associate, Staff Accountant, "
            "or Analyst depending on service line."
        )

    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=start_sources,
        additional_instruction=(
            "Accept service-line variations such as Audit Associate, Tax Associate, Advisory Analyst, or Staff Accountant; "
            "consider these titles equivalent entry-level roles."
        ),
    )

    # Senior_Promotion_Timeline (critical leaf)
    promo_leaf = evaluator.add_leaf(
        id="Senior_Promotion_Timeline",
        desc="Typical timeline for promotion from entry-level to Senior Associate position (2-3 years)",
        parent=big4_node,
        critical=True
    )
    promo_sources = _safe_sources(ext.big4_senior_promotion_timeline.sources if ext and ext.big4_senior_promotion_timeline else [])
    promo_claim = (
        "At Big Four firms, the typical timeline to be promoted from entry-level Associate/Staff to Senior Associate is about 2–3 years."
    )
    await evaluator.verify(
        claim=promo_claim,
        node=promo_leaf,
        sources=promo_sources,
        additional_instruction=(
            "Minor phrasing variations (e.g., 'around 2 years' or '2 to 3 years') are acceptable; "
            "use Big Four HR/career pages if provided."
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
    # Initialize evaluator as root (parallel aggregation)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Complete requirements for Nevada CPA license and entry into Big Four accounting career",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_nevada_cpa_and_big4(),
        template_class=NevadaCPAAndBig4Extraction,
        extraction_name="nevada_cpa_big4_extraction"
    )

    # Add ground truth reference information
    evaluator.add_ground_truth(
        gt_info=GROUND_TRUTH_INFO,
        gt_type="expected_requirements"
    )

    # Build verification subtrees
    await build_nevada_cpa_tree(evaluator, root, extraction)
    await build_big4_career_tree(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()