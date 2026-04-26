import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "astro_phd_us_top_program_2025"
TASK_DESCRIPTION = (
    "Identify a university in the United States that offers a PhD program in astronomy or astrophysics and meets all of "
    "the following criteria: (1) Ranked in the top 10 globally for Physics & Astronomy according to the QS World University "
    "Rankings 2025; (2) Has an astronomy department with at least 10 full-time equivalent (FTE) faculty members; "
    "(3) Provides institutional access to at least one 6-meter class or larger ground-based optical/infrared telescope; "
    "(4) Requires PhD students to complete at least 4 graduate-level core courses in astronomy/astrophysics; "
    "(5) Has faculty members conducting research in observational astronomy; "
    "(6) Has faculty members conducting research in theoretical astrophysics; "
    "(7) Provides students with access to space-based telescope data archives (such as HST, JWST, or Chandra); "
    "(8) Graduates at least 5 PhD students per year on average from the astronomy program; "
    "(9) Includes a comprehensive examination or qualifying examination requirement in the PhD program; "
    "(10) Has participation in major astronomical surveys or collaborations; "
    "(11) Has astronomy faculty who publish research in top-tier journals (ApJ, AJ, MNRAS, A&A, or Nature Astronomy). "
    "Provide the name of the university and supporting evidence with reference URLs for each criterion."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class CriterionEvidence(BaseModel):
    """
    Evidence item for one criterion.
    - statement: a concise claim as stated/implied in the answer, tailored to the criterion.
    - urls: all URLs explicitly cited in the answer to support this criterion.
    """
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AstroPhDExtraction(BaseModel):
    """
    Extracted structured info from the agent's answer.
    """
    university_name: Optional[str] = None
    department_name: Optional[str] = None
    phd_program_url: Optional[str] = None

    qs_ranking_top_10: Optional[CriterionEvidence] = None
    faculty_size_minimum: Optional[CriterionEvidence] = None
    large_telescope_access: Optional[CriterionEvidence] = None
    core_course_requirement: Optional[CriterionEvidence] = None
    observational_research: Optional[CriterionEvidence] = None
    theoretical_research: Optional[CriterionEvidence] = None
    space_telescope_data_access: Optional[CriterionEvidence] = None
    phd_graduation_rate: Optional[CriterionEvidence] = None
    us_location: Optional[CriterionEvidence] = None
    comprehensive_exam_requirement: Optional[CriterionEvidence] = None
    survey_collaboration_participation: Optional[CriterionEvidence] = None
    top_tier_publication_record: Optional[CriterionEvidence] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_university_evidence() -> str:
    return """
Extract the SINGLE primary U.S. university proposed in the answer and the supporting evidence (URLs) for each required criterion.

Return a JSON object with the following fields:
- university_name: string | null
- department_name: string | null
- phd_program_url: string | null

For each criterion below, return an object with:
- statement: a concise, self-contained claim tailored to the named university that directly addresses the criterion; if not clearly stated in the answer, set to null
- urls: an array of all URLs explicitly provided in the answer that support this criterion; if none are present, return an empty array

Criteria fields (use EXACT field names):
- qs_ranking_top_10
- faculty_size_minimum
- large_telescope_access
- core_course_requirement
- observational_research
- theoretical_research
- space_telescope_data_access
- phd_graduation_rate
- us_location
- comprehensive_exam_requirement
- survey_collaboration_participation
- top_tier_publication_record

Important rules:
1) Only extract URLs that are explicitly present in the answer (plain links or markdown). Do not fabricate or infer URLs.
2) If multiple universities are mentioned, choose the main one the answer ultimately recommends or focuses on.
3) Do not deduplicate, rewrite, or expand the URLs—return them exactly as provided (fix protocol if missing by prepending http://).
4) If the answer does not clearly provide a statement for a criterion, set statement to null; if no URLs, return an empty array for that criterion.
"""


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _get_urls(e: Optional[CriterionEvidence]) -> List[str]:
    return e.urls if (e and e.urls) else []


def _get_statement_or_default(e: Optional[CriterionEvidence], default_text: str) -> str:
    return e.statement if (e and e.statement) else default_text


def _source_required_instruction(extra: str = "") -> str:
    base = (
        "Important: Use ONLY the provided URL sources to judge this claim. "
        "If there are no valid URL sources provided for this check (empty or missing), "
        "you must judge the claim as Incorrect (not supported). "
    )
    return base + (extra or "")


async def _verify_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    claim: str,
    urls: List[str],
    add_ins: str,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls if urls else None,
        additional_instruction=add_ins
    )


# -----------------------------------------------------------------------------
# Build verification tree for all criteria
# -----------------------------------------------------------------------------
async def verify_university_criteria(evaluator: Evaluator, root_parent, ext: AstroPhDExtraction) -> None:
    """
    Build and evaluate the rubric tree according to the provided JSON rubric.
    """
    uni = ext.university_name or "the university"

    # Create the rubric root node (critical parallel)
    rubric_root = evaluator.add_parallel(
        id="Qualifying_University_Identification",
        desc="Identify a U.S. university with an astronomy PhD program that meets all specified criteria for rankings, faculty size, research facilities, program structure, and research output.",
        parent=root_parent,
        critical=True
    )

    # 1) QS_Ranking_Top_10
    qs_default = f"According to the QS World University Rankings 2025 by Subject (Physics & Astronomy), {uni} is ranked within the global top 10."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "QS_Ranking_Top_10",
        "The university is ranked in the top 10 globally for Physics & Astronomy according to QS World University Rankings 2025.",
        _get_statement_or_default(ext.qs_ranking_top_10, qs_default),
        _get_urls(ext.qs_ranking_top_10),
        _source_required_instruction(
            "The evidence should clearly indicate QS 2025 Physics & Astronomy subject ranking and show the university is within positions 1–10."
        ),
    )

    # 2) Faculty_Size_Minimum
    fac_default = f"The astronomy department at {uni} has at least 10 full-time equivalent (FTE) faculty members."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "Faculty_Size_Minimum",
        "The astronomy department has at least 10 full-time equivalent (FTE) faculty members.",
        _get_statement_or_default(ext.faculty_size_minimum, fac_default),
        _get_urls(ext.faculty_size_minimum),
        _source_required_instruction(
            "Department roster pages, official counts, or similar pages should make it reasonable to conclude FTE ≥ 10; allow small reasonable inference across multiple official pages."
        ),
    )

    # 3) Large_Telescope_Access
    tel_default = (
        f"{uni} has institutional access to at least one ground-based optical/infrared telescope with primary mirror diameter ≥ 6 meters."
    )
    await _verify_leaf(
        evaluator,
        rubric_root,
        "Large_Telescope_Access",
        "The university has institutional access to at least one 6-meter class or larger ground-based optical/infrared telescope.",
        _get_statement_or_default(ext.large_telescope_access, tel_default),
        _get_urls(ext.large_telescope_access),
        _source_required_instruction(
            "Examples include 6.5m Magellan, 6.5m MMT, 8–10m Gemini/Keck/Subaru/VLT/LBT/HET, etc. "
            "Membership, partnership, or guaranteed institutional time counts as access."
        ),
    )

    # 4) Core_Course_Requirement
    core_default = f"The astronomy/astrophysics PhD program at {uni} requires completion of at least 4 graduate-level core courses."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "Core_Course_Requirement",
        "The PhD program requires completion of at least 4 graduate-level core courses in astronomy/astrophysics.",
        _get_statement_or_default(ext.core_course_requirement, core_default),
        _get_urls(ext.core_course_requirement),
        _source_required_instruction(
            "Graduate handbook or official program requirements must explicitly or clearly imply ≥ 4 core courses."
        ),
    )

    # 5) Observational_Research
    obs_default = f"There are faculty members at {uni} conducting research in observational astronomy."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "Observational_Research",
        "The university has faculty members conducting research in observational astronomy.",
        _get_statement_or_default(ext.observational_research, obs_default),
        _get_urls(ext.observational_research),
        _source_required_instruction(
            "Faculty profiles, group pages, or publication lists should clearly indicate observational work (e.g., imaging, spectroscopy, time-domain observations)."
        ),
    )

    # 6) Theoretical_Research
    th_default = f"There are faculty members at {uni} conducting research in theoretical astrophysics."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "Theoretical_Research",
        "The university has faculty members conducting research in theoretical astrophysics.",
        _get_statement_or_default(ext.theoretical_research, th_default),
        _get_urls(ext.theoretical_research),
        _source_required_instruction(
            "Faculty pages or research descriptions should indicate theoretical/computational modeling, simulations, or analytic theory."
        ),
    )

    # 7) Space_Telescope_Data_Access
    space_default = f"Students at {uni} have access to space-based telescope data archives (e.g., HST/MAST, JWST/MAST, Chandra)."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "Space_Telescope_Data_Access",
        "The university provides access to space-based telescope data archives such as HST, JWST, or Chandra.",
        _get_statement_or_default(ext.space_telescope_data_access, space_default),
        _get_urls(ext.space_telescope_data_access),
        _source_required_instruction(
            "Evidence can include official pages pointing students to MAST (HST/JWST), Chandra, etc., or program pages indicating usage of these archives."
        ),
    )

    # 8) PhD_Graduation_Rate
    grad_default = f"The astronomy department at {uni} graduates at least 5 PhD students per year on average."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "PhD_Graduation_Rate",
        "The astronomy department graduates at least 5 PhD students per year on average.",
        _get_statement_or_default(ext.phd_graduation_rate, grad_default),
        _get_urls(ext.phd_graduation_rate),
        _source_required_instruction(
            "Use thesis lists, commencement data, or program statistics across multiple years to infer an average; if multi-year totals are given, compute average per year."
        ),
    )

    # 9) US_Location
    us_default = f"{uni} is located in the United States."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "US_Location",
        "The university is located in the United States.",
        _get_statement_or_default(ext.us_location, us_default),
        _get_urls(ext.us_location),
        _source_required_instruction(
            "Official university pages or authoritative references (e.g., Wikipedia infobox) should clearly indicate the U.S. location."
        ),
    )

    # 10) Comprehensive_Exam_Requirement
    comp_default = f"The astronomy/astrophysics PhD program at {uni} includes a comprehensive or qualifying examination requirement."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "Comprehensive_Exam_Requirement",
        "The PhD program includes a comprehensive examination or qualifying examination requirement.",
        _get_statement_or_default(ext.comprehensive_exam_requirement, comp_default),
        _get_urls(ext.comprehensive_exam_requirement),
        _source_required_instruction(
            "Graduate handbook or program pages should explicitly mention comprehensive/qualifying exams (written/oral)."
        ),
    )

    # 11) Survey_Collaboration_Participation
    survey_default = f"{uni} participates in major astronomical surveys or collaborations."
    await _verify_leaf(
        evaluator,
        rubric_root,
        "Survey_Collaboration_Participation",
        "The university has participation in major astronomical surveys or collaborations.",
        _get_statement_or_default(ext.survey_collaboration_participation, survey_default),
        _get_urls(ext.survey_collaboration_participation),
        _source_required_instruction(
            "Examples include SDSS, LSST/Rubin, DES, HSC, DESI, ZTF, etc.; faculty/group memberships or institutional memberships count."
        ),
    )

    # 12) Top_Tier_Publication_Record
    pubs_default = (
        f"Astronomy faculty at {uni} publish research in top-tier journals such as ApJ, AJ, MNRAS, A&A, or Nature Astronomy."
    )
    await _verify_leaf(
        evaluator,
        rubric_root,
        "Top_Tier_Publication_Record",
        "The astronomy department publishes research in top-tier journals including ApJ, AJ, MNRAS, A&A, or Nature Astronomy.",
        _get_statement_or_default(ext.top_tier_publication_record, pubs_default),
        _get_urls(ext.top_tier_publication_record),
        _source_required_instruction(
            "Accept journal abbreviations (ApJ/ApJL/ApJS, AJ, MNRAS, A&A) and Nature Astronomy; sample publication pages are sufficient."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
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
    Entry point to evaluate an agent's answer for the astronomy PhD university criteria task.
    """
    # Initialize evaluator (root is a non-critical container)
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

    # 1) Extraction
    extraction: AstroPhDExtraction = await evaluator.extract(
        prompt=prompt_extract_university_evidence(),
        template_class=AstroPhDExtraction,
        extraction_name="astro_phd_extraction",
    )

    # Record selected university for transparency
    evaluator.add_custom_info(
        info={"selected_university": extraction.university_name, "department_name": extraction.department_name},
        info_type="extraction_summary",
        info_name="selected_entity"
    )

    # 2) Build and run verification tree
    await verify_university_criteria(evaluator, root, extraction)

    # 3) Return evaluation summary
    return evaluator.get_summary()