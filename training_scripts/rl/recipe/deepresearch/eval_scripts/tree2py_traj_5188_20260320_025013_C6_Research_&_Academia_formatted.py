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
TASK_ID = "assistant_professor_nsf_career_2024_2026"
TASK_DESCRIPTION = (
    "Identify an assistant professor in Computer Science at an R1-classified university in the United States who "
    "received an NSF CAREER award between 2024 and 2026 (inclusive). The faculty member must have completed both a "
    "PhD in Computer Science or a closely related field and subsequent postdoctoral training prior to their current "
    "appointment. Provide the following information about this faculty member: (1) their full name, (2) their current "
    "institutional affiliation, (3) verification that their institution holds R1 classification under the 2025 "
    "Carnegie Classification system, (4) their current academic rank and tenure status, (5) details about their PhD "
    "(institution and year), (6) details about their postdoctoral training (institution and approximate duration), "
    "(7) details about their NSF CAREER award (year and brief research focus), and (8) evidence of active research "
    "participation via conference presentations or peer-reviewed publications. Each piece of information must be "
    "supported by publicly accessible reference URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacultyExtraction(BaseModel):
    # Identity
    name: Optional[str] = None
    institution: Optional[str] = None
    institution_urls: List[str] = Field(default_factory=list)

    # Institutional qualification
    r1_urls: List[str] = Field(default_factory=list)
    department_name: Optional[str] = None
    department_urls: List[str] = Field(default_factory=list)

    # Position characteristics
    rank: Optional[str] = None                    # e.g., "Assistant Professor"
    tenure_status: Optional[str] = None           # e.g., "tenure-track"
    rank_urls: List[str] = Field(default_factory=list)

    # Educational background
    phd_field: Optional[str] = None               # e.g., "Computer Science"
    phd_institution: Optional[str] = None
    phd_year: Optional[str] = None
    phd_urls: List[str] = Field(default_factory=list)

    postdoc_institution: Optional[str] = None
    postdoc_duration: Optional[str] = None        # e.g., "2 years", "2019–2021"
    postdoc_urls: List[str] = Field(default_factory=list)

    # NSF CAREER award
    award_year: Optional[str] = None              # e.g., "2025"
    award_focus: Optional[str] = None             # brief research topic/title
    award_urls: List[str] = Field(default_factory=list)

    # Research activity
    conference_evidence: Optional[str] = None     # a representative talk/paper claim
    conference_urls: List[str] = Field(default_factory=list)
    publication_evidence: Optional[str] = None    # a representative publication claim
    publication_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_faculty_info() -> str:
    return """
Extract the following structured information exactly as presented in the answer. Do not infer or invent anything.

Required fields (return null for any missing string field; return [] for any missing URL list):

Identity
- name: Full name of the faculty member.
- institution: Current institutional affiliation (university name).
- institution_urls: All URLs in the answer that support the institutional affiliation.

Institutional qualification
- r1_urls: All URLs in the answer that verify the institution is classified as R1 (Very High Research Activity) per the 2025 Carnegie Classification (official Carnegie site preferred; credible institutional pages acceptable if they explicitly state R1 status).
- department_name: The specific department or academic unit (e.g., "Department of Computer Science", "Electrical and Computer Engineering").
- department_urls: All URLs that reference/confirm the department or academic unit and the faculty member's affiliation with it.

Position characteristics
- rank: The current academic rank/title (e.g., "Assistant Professor"; include qualifiers like "Teaching", "Research", "Adjunct" if present).
- tenure_status: The tenure status string if stated (e.g., "tenure-track", "non-tenure track", "tenured", "clinical", etc.).
- rank_urls: All URLs that verify the rank/position.

Educational background
- phd_field: The PhD degree field/discipline as stated.
- phd_institution: The awarding institution for the PhD.
- phd_year: The (approximate) year of PhD completion as stated in the answer.
- phd_urls: All URLs that verify the PhD details (biography page, CV, department profile, etc.).

- postdoc_institution: The institution for postdoctoral training.
- postdoc_duration: The approximate duration (e.g., "2 years", "2019–2021") as stated.
- postdoc_urls: All URLs that verify the postdoctoral training details.

NSF CAREER award
- award_year: The year of the NSF CAREER award (as stated).
- award_focus: A brief research focus or project title associated with the CAREER award.
- award_urls: All URLs that verify the NSF CAREER award details (NSF database entry, university announcement, news, etc.).

Research activity
- conference_evidence: A short text snippet identifying a specific conference presentation or peer-reviewed conference paper (as stated).
- conference_urls: All URLs that verify the above conference evidence.
- publication_evidence: A short text snippet identifying a specific peer-reviewed publication (as stated).
- publication_urls: All URLs that verify the above publication evidence.

Rules for URL fields:
- Extract only actual URLs explicitly present in the answer (plain links or markdown links).
- Include full URLs with protocol; if missing, prepend http://
- Do not invent URLs.

Return a single JSON object conforming to the FacultyExtraction schema.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _combine_urls(*lists: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for l in lists:
        if l:
            out.extend([u for u in l if _nonempty_str(u)])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification sections                                                       #
# --------------------------------------------------------------------------- #
async def add_identity_section(evaluator: Evaluator, parent, ex: FacultyExtraction) -> None:
    node = evaluator.add_parallel(
        id="identity_verification",
        desc="The faculty member's identity and basic information are provided (full name and current institutional affiliation).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(ex.name),
        id="name_provided",
        desc="The faculty member's full name is provided.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(ex.institution),
        id="institution_named",
        desc="The faculty member's current institutional affiliation is clearly identified.",
        parent=node,
        critical=True
    )


async def add_institution_section(evaluator: Evaluator, parent, ex: FacultyExtraction) -> None:
    inst_node = evaluator.add_parallel(
        id="institutional_qualification",
        desc="The faculty member's institution meets R1 classification requirements and has a CS (or related) academic unit.",
        parent=parent,
        critical=True
    )

    # R1 classification
    r1_node = evaluator.add_parallel(
        id="r1_classification",
        desc="The institution is verified to hold R1 classification under the Carnegie Classification system (2025).",
        parent=inst_node,
        critical=True
    )

    r1_urls = _safe_list(ex.r1_urls)
    r1_claim = f"{ex.institution or 'The institution'} is classified as R1 (Very High Research Activity) under the 2025 Carnegie Classification."
    r1_leaf = evaluator.add_leaf(
        id="r1_verification_evidence",
        desc="Evidence confirms the university holds R1 status (Carnegie Classification, 2025).",
        parent=r1_node,
        critical=True
    )
    await evaluator.verify(
        claim=r1_claim,
        node=r1_leaf,
        sources=r1_urls,
        additional_instruction="Prefer the official 2025 Carnegie Classification listing. Accept credible institutional pages that explicitly state 'R1: Very High Research Activity' and clearly reference Carnegie 2025."
    )

    evaluator.add_custom_node(
        result=len(r1_urls) > 0,
        id="r1_reference_url",
        desc="A reference URL is provided that verifies the institution's R1 classification status.",
        parent=r1_node,
        critical=True
    )

    # Department existence (CS or closely related)
    dept_node = evaluator.add_parallel(
        id="cs_department_exists",
        desc="The institution has a CS department or related academic unit where the faculty member holds their appointment.",
        parent=inst_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(ex.department_name),
        id="department_identification",
        desc="The specific department or academic unit is identified.",
        parent=dept_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(ex.department_urls)) > 0,
        id="department_reference_url",
        desc="A reference URL is provided that confirms the department/unit and the faculty member's affiliation.",
        parent=dept_node,
        critical=True
    )


async def add_position_section(evaluator: Evaluator, parent, ex: FacultyExtraction) -> None:
    pos_node = evaluator.add_parallel(
        id="position_characteristics",
        desc="The faculty member holds the appropriate academic position: assistant professor rank on a tenure-track appointment in CS or a closely related field.",
        parent=parent,
        critical=True
    )

    # Assistant Professor rank + URL
    rank_group = evaluator.add_parallel(
        id="assistant_professor_rank",
        desc="Assistant Professor rank verified with supporting URL(s).",
        parent=pos_node,
        critical=True
    )

    rank_urls = _safe_list(ex.rank_urls)
    # Verify rank = Assistant Professor (allow variants)
    rank_leaf = evaluator.add_leaf(
        id="rank_verification",
        desc="The assistant professor rank is explicitly stated or clearly indicated.",
        parent=rank_group,
        critical=True
    )
    await evaluator.verify(
        claim="The faculty member currently holds the title of Assistant Professor (or equivalent phrasing such as 'Asst. Professor').",
        node=rank_leaf,
        sources=rank_urls or _safe_list(ex.department_urls),
        additional_instruction="Confirm the page indicates 'Assistant Professor'. Allow minor variants like 'Asst. Professor' or 'Assistant Professor of X'."
    )

    evaluator.add_custom_node(
        result=len(rank_urls) > 0,
        id="rank_reference_url",
        desc="A reference URL is provided that verifies the faculty member's current rank.",
        parent=rank_group,
        critical=True
    )

    # Tenure-track status
    tenure_leaf = evaluator.add_leaf(
        id="tenure_track_status",
        desc="The position is tenure-track.",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim="The assistant professor position is a tenure-track appointment.",
        node=tenure_leaf,
        sources=_combine_urls(rank_urls, _safe_list(ex.department_urls)),
        additional_instruction="Treat 'Assistant Professor' as tenure-track unless the page explicitly indicates non-tenure-track variants (e.g., 'Teaching/Clinical/Research/Adjunct/Visiting Assistant Professor'). If non-tenure-track wording appears, this claim is not supported."
    )

    # Appointment in CS or closely related field
    cs_related_leaf = evaluator.add_leaf(
        id="cs_related_appointment",
        desc="The appointment is in Computer Science or a closely related field.",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim="The faculty member's appointment is in Computer Science or a closely related field (e.g., Electrical and Computer Engineering, Information/Computer/Computational Science).",
        node=cs_related_leaf,
        sources=_combine_urls(_safe_list(ex.department_urls), rank_urls),
        additional_instruction="Look for department/program names indicating CS or a close cognate (e.g., 'Electrical and Computer Engineering', 'Informatics', 'Information Science'). Consider clear CS-adjacent designations as valid."
    )


async def add_education_section(evaluator: Evaluator, parent, ex: FacultyExtraction) -> None:
    edu_node = evaluator.add_sequential(
        id="educational_background",
        desc="The faculty member has completed both PhD and postdoctoral training meeting the specified requirements.",
        parent=parent,
        critical=True
    )

    # PhD completion details
    phd_node = evaluator.add_parallel(
        id="phd_completion",
        desc="The faculty member completed a PhD in CS or a closely related field from an accredited institution.",
        parent=edu_node,
        critical=True
    )

    phd_urls = _safe_list(ex.phd_urls)

    phd_field_leaf = evaluator.add_leaf(
        id="phd_field",
        desc="The PhD degree is in Computer Science or a closely related field.",
        parent=phd_node,
        critical=True
    )
    phd_field_claim = "The PhD degree is in Computer Science or a closely related field (e.g., Computer/Electrical Engineering, Information Science)."
    if _nonempty_str(ex.phd_field):
        phd_field_claim = f"The PhD degree field is '{ex.phd_field}', which is Computer Science or a closely related field."
    await evaluator.verify(
        claim=phd_field_claim,
        node=phd_field_leaf,
        sources=phd_urls,
        additional_instruction="Verify discipline/department text supports that the doctorate is in CS or an obviously close cognate."
    )

    phd_inst_leaf = evaluator.add_leaf(
        id="phd_institution",
        desc="The institution where the PhD was earned is identified.",
        parent=phd_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The PhD was earned at {ex.phd_institution or 'the stated institution'}.",
        node=phd_inst_leaf,
        sources=phd_urls,
        additional_instruction="Confirm the page explicitly names the PhD awarding institution."
    )

    phd_year_leaf = evaluator.add_leaf(
        id="phd_completion_year",
        desc="The approximate year of PhD completion is provided.",
        parent=phd_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The PhD was completed around {ex.phd_year or 'the stated year'}.",
        node=phd_year_leaf,
        sources=phd_urls,
        additional_instruction="Verify the completion year (allow approximate phrasing or ±1 year tolerance if the page clearly implies the completion time)."
    )

    evaluator.add_custom_node(
        result=len(phd_urls) > 0,
        id="phd_reference_url",
        desc="A reference URL is provided that verifies the PhD degree details.",
        parent=phd_node,
        critical=True
    )

    # Postdoctoral training details
    postdoc_node = evaluator.add_parallel(
        id="postdoc_completion",
        desc="The faculty member completed postdoctoral training after the PhD and before the current appointment.",
        parent=edu_node,
        critical=True
    )

    postdoc_urls = _safe_list(ex.postdoc_urls)

    postdoc_inst_leaf = evaluator.add_leaf(
        id="postdoc_institution",
        desc="The institution where postdoctoral training was completed is identified.",
        parent=postdoc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The faculty member completed postdoctoral training at {ex.postdoc_institution or 'the stated institution'}.",
        node=postdoc_inst_leaf,
        sources=postdoc_urls,
        additional_instruction="Confirm that the page lists the postdoctoral institution (not the PhD institution)."
    )

    postdoc_duration_leaf = evaluator.add_leaf(
        id="postdoc_duration",
        desc="Information about approximate postdoctoral training duration is provided.",
        parent=postdoc_node,
        critical=True  # Set to True to satisfy critical-parent constraint
    )
    await evaluator.verify(
        claim=f"The postdoctoral training duration was approximately {ex.postdoc_duration or 'the stated duration'}.",
        node=postdoc_duration_leaf,
        sources=postdoc_urls,
        additional_instruction="Accept approximate durations (e.g., '2 years', '2019–2021'). If multiple postdocs are listed, any one that precedes the current appointment is acceptable."
    )

    evaluator.add_custom_node(
        result=len(postdoc_urls) > 0,
        id="postdoc_reference_url",
        desc="A reference URL is provided that verifies the postdoctoral training details.",
        parent=postdoc_node,
        critical=True
    )


async def add_award_section(evaluator: Evaluator, parent, ex: FacultyExtraction) -> None:
    award_node = evaluator.add_parallel(
        id="nsf_career_award",
        desc="The faculty member received an NSF CAREER award (2024–2026 inclusive) supporting CS or a related area.",
        parent=parent,
        critical=True
    )

    award_urls = _safe_list(ex.award_urls)

    # Subgroup for award received details
    received_node = evaluator.add_parallel(
        id="award_received",
        desc="Evidence confirms the faculty member received an NSF CAREER award.",
        parent=award_node,
        critical=True
    )

    # Explicit confirmation that it's an NSF CAREER award
    award_received_leaf = evaluator.add_leaf(
        id="award_received_confirm",
        desc="The faculty member received an NSF CAREER award.",
        parent=received_node,
        critical=True
    )
    await evaluator.verify(
        claim="The faculty member received an NSF CAREER (Faculty Early Career Development) award.",
        node=award_received_leaf,
        sources=award_urls,
        additional_instruction="Look for 'NSF CAREER' or 'Faculty Early Career Development (CAREER) Program' explicitly associated with the faculty member."
    )

    # Year (must be 2024, 2025, or 2026)
    award_year_leaf = evaluator.add_leaf(
        id="award_year",
        desc="The award year is within 2024–2026 (inclusive).",
        parent=received_node,
        critical=True
    )
    year_str = ex.award_year or "the stated year"
    await evaluator.verify(
        claim=f"The NSF CAREER award year is {year_str}, which lies between 2024 and 2026 inclusive.",
        node=award_year_leaf,
        sources=award_urls,
        additional_instruction="Confirm both the actual year on the page and that it falls in the 2024–2026 window."
    )

    # Project title / brief research focus
    award_focus_leaf = evaluator.add_leaf(
        id="award_project_description",
        desc="A brief description of the research focus or project title supported by the CAREER award is provided.",
        parent=received_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The CAREER award's research focus/title is described as: {ex.award_focus or 'the stated focus/title'}.",
        node=award_focus_leaf,
        sources=award_urls,
        additional_instruction="It is sufficient if the page gives a clear project title or short description of the funded research."
    )

    evaluator.add_custom_node(
        result=len(award_urls) > 0,
        id="award_reference_url",
        desc="A reference URL is provided that verifies the NSF CAREER award.",
        parent=received_node,
        critical=True
    )

    # CS research focus
    cs_focus_leaf = evaluator.add_leaf(
        id="cs_research_focus",
        desc="The NSF CAREER award supports research in Computer Science or a closely related computational field.",
        parent=award_node,
        critical=True
    )
    await evaluator.verify(
        claim="The CAREER project focuses on Computer Science or a closely related computational field.",
        node=cs_focus_leaf,
        sources=award_urls,
        additional_instruction="Check the project title/abstract/description for CS/computing-related themes. Accept obvious close cognates (e.g., ML/AI, systems, HCI, ECE with computing focus)."
    )


async def add_research_activity_section(evaluator: Evaluator, parent, ex: FacultyExtraction) -> None:
    ra_node = evaluator.add_parallel(
        id="research_activity",
        desc="Active participation in the research community (conference presentations and/or peer-reviewed publications).",
        parent=parent,
        critical=False
    )

    # Conferences
    conf_node = evaluator.add_parallel(
        id="conference_participation",
        desc="Evidence of conference participation (presentations or peer-reviewed conference papers).",
        parent=ra_node,
        critical=False
    )

    # Gate conference verification by requiring URLs
    evaluator.add_custom_node(
        result=len(_safe_list(ex.conference_urls)) > 0,
        id="conference_reference_url",
        desc="A reference URL is provided that verifies conference participation.",
        parent=conf_node,
        critical=True
    )

    conf_leaf = evaluator.add_leaf(
        id="conference_evidence",
        desc="Specific conference presentations or papers are identified and supported by sources.",
        parent=conf_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The faculty member's conference activity is evidenced by: {ex.conference_evidence or 'the stated conference evidence'}.",
        node=conf_leaf,
        sources=_safe_list(ex.conference_urls),
        additional_instruction="Confirm that at least one of the provided URLs shows a peer-reviewed conference publication or an official conference talk/presentation by the faculty member."
    )

    # Publications
    pub_node = evaluator.add_parallel(
        id="scholarly_publications",
        desc="Evidence of scholarly publications in peer-reviewed venues.",
        parent=ra_node,
        critical=False
    )

    # Gate publication verification by requiring URLs
    evaluator.add_custom_node(
        result=len(_safe_list(ex.publication_urls)) > 0,
        id="publication_reference_url",
        desc="A reference URL is provided that verifies scholarly publications.",
        parent=pub_node,
        critical=True
    )

    pub_leaf = evaluator.add_leaf(
        id="publication_evidence",
        desc="Specific peer-reviewed publications are identified and supported by sources.",
        parent=pub_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The faculty member has peer-reviewed publications; for example: {ex.publication_evidence or 'the stated publication evidence'}.",
        node=pub_leaf,
        sources=_safe_list(ex.publication_urls),
        additional_instruction="Accept pages like Google Scholar, DBLP, publisher pages, or institutional publication lists that clearly show peer-reviewed outputs."
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
    Evaluate an answer for the Assistant Professor NSF CAREER (2024–2026) task.
    """
    # Initialize evaluator (root kept non-critical to allow partial credit and avoid critical-child constraint issues)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at top level
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_faculty_info(),
        template_class=FacultyExtraction,
        extraction_name="faculty_extraction"
    )

    # Add a summary of constraints as custom info (for transparency)
    evaluator.add_custom_info(
        info={
            "timeframe_years_inclusive": [2024, 2025, 2026],
            "required_position": "Assistant Professor (tenure-track), CS or closely related field",
            "institution_requirement": "R1 under 2025 Carnegie Classification",
            "degree_requirements": "PhD completed (CS or closely related) + subsequent postdoctoral training",
            "evidence_policy": "Each factual claim should be supported by public reference URL(s)"
        },
        info_type="constraint_summary"
    )

    # Build verification tree sections
    await add_identity_section(evaluator, root, extracted)
    await add_institution_section(evaluator, root, extracted)
    await add_position_section(evaluator, root, extracted)
    await add_education_section(evaluator, root, extracted)
    await add_award_section(evaluator, root, extracted)
    await add_research_activity_section(evaluator, root, extracted)

    # Return unified summary
    return evaluator.get_summary()