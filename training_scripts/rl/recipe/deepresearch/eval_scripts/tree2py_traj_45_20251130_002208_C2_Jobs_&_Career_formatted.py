import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "rutgers_run_director_athletics_recreation_25ST2446"
TASK_DESCRIPTION = """I'm interested in applying for a Director of Athletics & Recreation position at Rutgers University-Newark that I heard was recently posted. The position involves overseeing NCAA Division III athletics programs and reports to the Chancellor.

Please help me find this job posting and provide the following information:

1. Position Verification: Confirm the job posting number and verify that this is indeed the Director of Athletics & Recreation position at Rutgers University-Newark.

2. Minimum Qualifications: What are the minimum education and experience requirements for this position?

3. Application Requirements: What documents are required to apply for this position?

4. Salary Information: What is the salary range for this position?

For each piece of information, please provide the source URL where you found it.
"""


# -----------------------------------------------------------------------------
# Extraction data models
# -----------------------------------------------------------------------------
class PositionInfo(BaseModel):
    posting_number: Optional[str] = None
    title: Optional[str] = None
    institution: Optional[str] = None
    campus: Optional[str] = None
    reports_to: Optional[str] = None
    oversees_ncaad3_text: Optional[str] = None
    position_urls: List[str] = Field(default_factory=list)


class MinimumQualificationsInfo(BaseModel):
    education_requirement: Optional[str] = None
    education_urls: List[str] = Field(default_factory=list)
    experience_requirement: Optional[str] = None
    experience_urls: List[str] = Field(default_factory=list)


class ApplicationRequirementsInfo(BaseModel):
    required_documents: List[str] = Field(default_factory=list)
    documents_urls: List[str] = Field(default_factory=list)


class SalaryInfo(BaseModel):
    salary_range: Optional[str] = None
    salary_urls: List[str] = Field(default_factory=list)


class JobPostingExtraction(BaseModel):
    position: Optional[PositionInfo] = None
    minimum_qualifications: Optional[MinimumQualificationsInfo] = None
    application_requirements: Optional[ApplicationRequirementsInfo] = None
    salary: Optional[SalaryInfo] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_job_posting() -> str:
    return """
Extract structured information about the Rutgers University–Newark Director of Athletics & Recreation job posting as presented in the answer.

Return a JSON object with the following structure and fields:

position:
  - posting_number: The job posting or requisition number exactly as written in the answer (e.g., 25ST2446). If missing, null.
  - title: The position title as written (e.g., Director of Athletics & Recreation). If missing, null.
  - institution: The institution or university in the answer (e.g., Rutgers University–Newark). If missing, null.
  - campus: The specific campus name if mentioned (e.g., Rutgers University–Newark). If missing, null.
  - reports_to: The role the position reports to if specified (e.g., Chancellor). If missing, null.
  - oversees_ncaad3_text: The phrase or sentence indicating NCAA Division III oversight if present; otherwise null.
  - position_urls: All URLs the answer cites as source(s) for the main posting or position verification details.

minimum_qualifications:
  - education_requirement: The minimum education requirement text as given (e.g., "Bachelor's degree in sports management, higher education administration, or a related field"). If absent, null.
  - education_urls: All URLs cited for the minimum education requirement.
  - experience_requirement: The minimum experience requirement text as given (e.g., "5–7 years of progressively responsible experience in collegiate athletics administration"). If absent, null.
  - experience_urls: All URLs cited for the minimum experience requirement.

application_requirements:
  - required_documents: A list of the application documents specified in the answer (e.g., ["Resume/CV", "Cover Letter/Letter of Application", "List of Professional References"]). If none mentioned, return an empty list.
  - documents_urls: All URLs cited for the application document requirements.

salary:
  - salary_range: The salary range text exactly as stated in the answer (e.g., "$120,000–$140,000" or "Minimum $120,000, Mid $130,000, Max $140,000"). If the answer does not provide one, null.
  - salary_urls: All URLs cited for the salary information.

Rules:
- Extract only what is explicitly in the answer text. Do not invent values.
- For URLs, include only valid, explicit URLs shown in the answer (including those in markdown links). If a URL lacks protocol, prepend http://.
- If a section is not addressed in the answer, set its fields to null or empty lists as appropriate.
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _has_any_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip():
            return True
    return False


def _pick_sources(preferred: Optional[List[str]], fallback: Optional[List[str]]) -> Optional[List[str]]:
    if preferred and len(preferred) > 0:
        return preferred
    if fallback and len(fallback) > 0:
        return fallback
    return None


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def build_position_identification(
    evaluator: Evaluator,
    parent_node,
    extraction: JobPostingExtraction,
) -> None:
    """
    Build and verify the PositionIdentification subtree.
    """
    pos = extraction.position or PositionInfo()
    pos_urls = pos.position_urls or []

    node = evaluator.add_parallel(
        id="PositionIdentification",
        desc="Correctly identify the specific job posting and verify required position characteristics",
        parent=parent_node,
        critical=True
    )

    # PostingNumber
    leaf_posting = evaluator.add_leaf(
        id="PostingNumber",
        desc="Job posting number matches the required posting number (25ST2446)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting (or requisition) number for this position is 25ST2446.",
        node=leaf_posting,
        sources=pos_urls,
        additional_instruction=(
            "Verify on the provided job posting page(s) that the posting number equals 25ST2446. "
            "Treat 'Posting number', 'Job number', or 'Requisition number' as equivalent. "
            "Allow minor formatting variations (e.g., spaces or punctuation), but the digits/letters must match."
        )
    )

    # PositionIsDirectorAthleticsRecreationRUNewark
    leaf_title_campus = evaluator.add_leaf(
        id="PositionIsDirectorAthleticsRecreationRUNewark",
        desc="Position is Director of Athletics & Recreation at Rutgers University–Newark",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This posting is for the 'Director of Athletics & Recreation' at Rutgers University–Newark.",
        node=leaf_title_campus,
        sources=pos_urls,
        additional_instruction=(
            "Check the position title and the institution/campus on the page. "
            "Allow '&' vs 'and', as well as hyphen or en dash variations in 'Rutgers University–Newark' "
            "(e.g., 'Rutgers University-Newark' or 'Rutgers University Newark')."
        )
    )

    # ReportsToChancellor
    leaf_reports = evaluator.add_leaf(
        id="ReportsToChancellor",
        desc="Posting states the position reports to the Chancellor",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting states that this position reports to the Chancellor.",
        node=leaf_reports,
        sources=pos_urls,
        additional_instruction=(
            "Look for a 'Reports to' field or similar phrasing (e.g., 'reporting to the Chancellor' or 'reports directly to the Chancellor')."
        )
    )

    # OverseesNCAADivisionIII
    leaf_ncaa = evaluator.add_leaf(
        id="OverseesNCAADivisionIII",
        desc="Posting states the position oversees NCAA Division III intercollegiate athletics",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting indicates the position oversees NCAA Division III intercollegiate athletics programs.",
        node=leaf_ncaa,
        sources=pos_urls,
        additional_instruction=(
            "Accept reasonable variants such as 'NCAA Division III' or 'NCAA DIII'. "
            "The statement must clearly indicate oversight/leadership responsibility for NCAA Division III athletics."
        )
    )

    # PositionVerificationSourceURL (existence check for supporting URL)
    evaluator.add_custom_node(
        result=_has_any_url(pos_urls),
        id="PositionVerificationSourceURL",
        desc="Provide a source URL supporting the posting number and position verification details",
        parent=node,
        critical=True
    )


async def build_information_extraction(
    evaluator: Evaluator,
    parent_node,
    extraction: JobPostingExtraction,
) -> None:
    """
    Build and verify the InformationExtraction subtree, including
    MinimumQualifications, ApplicationRequirements, and SalaryInformation.
    """
    pos = extraction.position or PositionInfo()
    pos_urls = pos.position_urls or []

    info_node = evaluator.add_parallel(
        id="InformationExtraction",
        desc="Extract minimum qualifications, application requirements, and salary information with source URLs",
        parent=parent_node,
        critical=True
    )

    # -------------------- Minimum Qualifications --------------------
    minq = extraction.minimum_qualifications or MinimumQualificationsInfo()
    minq_node = evaluator.add_parallel(
        id="MinimumQualifications",
        desc="Extract minimum education and experience requirements (with source URLs)",
        parent=info_node,
        critical=True
    )

    # EducationRequirement
    edu_leaf = evaluator.add_leaf(
        id="EducationRequirement",
        desc="Minimum education requirement matches: Bachelor's degree in sports management, higher education administration, or a related field",
        parent=minq_node,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum education requirement is a Bachelor's degree in sports management, higher education administration, or a related field.",
        node=edu_leaf,
        sources=_pick_sources(minq.education_urls, pos_urls),
        additional_instruction=(
            "Focus on the minimum requirement. Accept reasonable formulations like 'Bachelor’s degree', 'BA/BS', or 'undergraduate degree', "
            "and allow mention of 'or a related field'."
        )
    )

    # EducationSourceURL (existence)
    evaluator.add_custom_node(
        result=_has_any_url(minq.education_urls),
        id="EducationSourceURL",
        desc="Provide the source URL where the minimum education requirement is stated",
        parent=minq_node,
        critical=True
    )

    # ExperienceRequirement
    exp_leaf = evaluator.add_leaf(
        id="ExperienceRequirement",
        desc="Minimum experience requirement matches: 5–7 years of progressively responsible experience in collegiate athletics administration",
        parent=minq_node,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum experience requirement is 5–7 years of progressively responsible experience in collegiate athletics administration.",
        node=exp_leaf,
        sources=_pick_sources(minq.experience_urls, pos_urls),
        additional_instruction=(
            "Treat '5–7 years' equivalently to 'five to seven years' or '5 to 7 years'. "
            "The wording should clearly indicate progressively responsible experience in collegiate athletics administration."
        )
    )

    # ExperienceSourceURL (existence)
    evaluator.add_custom_node(
        result=_has_any_url(minq.experience_urls),
        id="ExperienceSourceURL",
        desc="Provide the source URL where the minimum experience requirement is stated",
        parent=minq_node,
        critical=True
    )

    # -------------------- Application Requirements --------------------
    appreq = extraction.application_requirements or ApplicationRequirementsInfo()
    app_node = evaluator.add_parallel(
        id="ApplicationRequirements",
        desc="Extract required application documents (with source URL)",
        parent=info_node,
        critical=True
    )

    docs_leaf = evaluator.add_leaf(
        id="RequiredDocuments",
        desc="Identify the three required application documents: Resume/CV, Cover Letter/Letter of Application, and List of Professional References",
        parent=app_node,
        critical=True
    )
    await evaluator.verify(
        claim="The required application documents include a Resume/CV, a Cover Letter (or Letter of Application), and a list of professional references.",
        node=docs_leaf,
        sources=_pick_sources(appreq.documents_urls, pos_urls),
        additional_instruction=(
            "Verify that these documents are clearly listed as required. Accept 'Curriculum Vitae' as equivalent to 'CV', "
            "'Cover Letter' equivalent to 'Letter of Application', and 'List of Professional References' possibly with a specified number (e.g., three). "
            "It's acceptable if additional documents are also required; this claim only checks that these three are included."
        )
    )

    evaluator.add_custom_node(
        result=_has_any_url(appreq.documents_urls),
        id="ApplicationDocumentsSourceURL",
        desc="Provide the source URL where required application documents are stated",
        parent=app_node,
        critical=True
    )

    # -------------------- Salary Information --------------------
    sal = extraction.salary or SalaryInfo()
    sal_node = evaluator.add_parallel(
        id="SalaryInformation",
        desc="Extract salary range information (with source URL)",
        parent=info_node,
        critical=True
    )

    salary_text = sal.salary_range or ""
    salary_leaf = evaluator.add_leaf(
        id="SalaryRange",
        desc="Provide the salary range values as stated in the posting (may include minimum, midrange, and/or maximum)",
        parent=sal_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The salary range for this position is {salary_text}.",
        node=salary_leaf,
        sources=_pick_sources(sal.salary_urls, pos_urls),
        additional_instruction=(
            "Confirm that the page states this salary range. Allow reasonable formatting differences such as currency symbols, commas, "
            "en dashes vs hyphens, and labels like 'minimum', 'midpoint', 'maximum'. If only a min–max range is provided, that's acceptable."
        )
    )

    evaluator.add_custom_node(
        result=_has_any_url(sal.salary_urls),
        id="SalarySourceURL",
        desc="Provide the source URL where the salary range is stated",
        parent=sal_node,
        critical=True
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Rutgers University–Newark Director of Athletics & Recreation job posting task.
    """

    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
        prompt=prompt_extract_job_posting(),
        template_class=JobPostingExtraction,
        extraction_name="job_posting_extraction"
    )

    # Optional: record expected ground-truth anchors from rubric for transparency
    evaluator.add_ground_truth({
        "expected_posting_number": "25ST2446",
        "expected_position_title": "Director of Athletics & Recreation",
        "expected_institution": "Rutgers University–Newark",
        "expected_reports_to": "Chancellor",
        "expected_ncaadivision": "NCAA Division III"
    }, gt_type="rubric_expectations")

    # Build the rubric tree as specified
    job_node = evaluator.add_sequential(
        id="JobPostingAnalysis",
        desc="Complete identification and information extraction for the Rutgers University–Newark Director of Athletics & Recreation job posting",
        parent=root,
        critical=True
    )

    # Subtree 1: PositionIdentification
    await build_position_identification(evaluator, job_node, extraction)

    # Subtree 2: InformationExtraction
    await build_information_extraction(evaluator, job_node, extraction)

    # Return the final summarized evaluation
    return evaluator.get_summary()