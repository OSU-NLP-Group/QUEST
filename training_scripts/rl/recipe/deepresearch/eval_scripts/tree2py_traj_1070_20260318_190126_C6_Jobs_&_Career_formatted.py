import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "public_universities_career_services_eval"
TASK_DESCRIPTION = """
Identify three public universities in the United States that meet ALL of the following career services criteria:

1. The university must report a career outcomes rate of at least 90% (students employed, continuing education, or in volunteer/service positions) within six months of graduation for their most recent graduating class
2. The university's career outcomes reporting must follow NACE (National Association of Colleges and Employers) First Destination Survey standards and must report a knowledge rate
3. The university must host regular career and internship fairs at minimum during both fall and spring semesters
4. The university must have had at least 350 different employers or organizations either attend career fairs, recruit on campus, or engage with students during the most recent academic year reported
5. The university's career center must provide all four core services: resume/CV review, mock interview preparation, career advising appointments, and a dedicated job/internship search platform (specify platform name)

For each university, provide:
- University name
- The specific career outcomes rate percentage and the graduating class year it represents
- The knowledge rate percentage
- Evidence of NACE standards compliance
- The number or description of employers/organizations that engaged with the university
- The name of the job/internship platform used
- Direct URLs to: (a) the career outcomes or first destination survey data page, (b) the career center main services page, and (c) a page showing employer engagement or career fair information
"""


# --------------------------------------------------------------------------- #
# Data Models for Extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityInfo(BaseModel):
    # Identity
    name: Optional[str] = None

    # Public status sources (if any were cited in the answer beyond required URLs)
    public_status_source_urls: List[str] = Field(default_factory=list)

    # Outcomes (FDS)
    outcomes_rate_percent: Optional[str] = None
    outcomes_time_window: Optional[str] = None  # e.g., "within six months"
    outcomes_class_year: Optional[str] = None
    outcomes_data_url: Optional[str] = None

    # NACE compliance + knowledge rate
    knowledge_rate_percent: Optional[str] = None
    nace_compliance_evidence: Optional[str] = None
    nace_evidence_url: Optional[str] = None  # can be same as outcomes page or a methods page

    # Career fairs (schedule)
    career_fair_url: Optional[str] = None
    fair_schedule_evidence: Optional[str] = None  # textual evidence snippet in the answer

    # Employer engagement
    employer_engagement_count: Optional[str] = None  # keep as string for flexibility (e.g., "400+")
    employer_engagement_desc: Optional[str] = None
    employer_engagement_url: Optional[str] = None

    # Core services
    job_platform_name: Optional[str] = None
    services_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract information for up to three distinct public U.S. universities exactly as presented in the answer.
    For each university mentioned (in order of appearance), extract the following fields. If any field is not present in the answer, set it to null (for strings) or [] (for URL arrays). Do NOT invent or infer values not present in the answer.

    Required fields per university:
    - name: The university name as written in the answer.
    - public_status_source_urls: An array of any URLs in the answer that directly support that the institution is a public university in the United States (e.g., official "About", system pages, Wikipedia). If none are provided, return [].
    - outcomes_rate_percent: The career outcomes (or placement/success) percentage cited for the most recent class (string as written, e.g., "92%").
    - outcomes_time_window: The time window for measuring outcomes (e.g., "within six months of graduation"). If stated differently (e.g., "six months"), extract exactly as written.
    - outcomes_class_year: The graduating class year the outcomes rate corresponds to (e.g., "Class of 2024").
    - outcomes_data_url: The direct URL to the career outcomes / first destination survey data page.
    - knowledge_rate_percent: The knowledge rate percentage as written (e.g., "78%").
    - nace_compliance_evidence: A short phrase as written in the answer that indicates NACE FDS standards/compliance (e.g., "per NACE First Destination Survey standards"). If not explicitly stated, set to null.
    - nace_evidence_url: A URL in the answer that supports NACE compliance or methodology (can be the same outcomes page). If none is provided, set to null.
    - career_fair_url: A URL to a page about career or internship fairs (schedule, events, or overview).
    - fair_schedule_evidence: Evidence text in the answer suggesting at least fall and spring career/internship fairs occur (extract the phrasing used).
    - employer_engagement_count: The number or descriptor of employers/organizations engaged in the most recent year (e.g., "400+", "over 500", "at least 350").
    - employer_engagement_desc: Any additional phrasing used in the answer describing this employer engagement (e.g., "employers attending fairs, on-campus recruiting, info sessions").
    - employer_engagement_url: A URL in the answer that supports the employer engagement statistic or description.
    - job_platform_name: The dedicated job/internship platform name mentioned (e.g., "Handshake", "Symplicity", "12twenty").
    - services_url: The URL to the career center main services page (where resume reviews, mock interviews, advising, and the platform are described).

    Return JSON with a top-level key "universities" as a list of up to 3 UniversityInfo objects in the order they appear in the answer text. Do not include more than 3.
    """


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_urls(*parts: Optional[str | List[str]]) -> List[str]:
    """Flatten inputs into a unique list of non-empty URLs."""
    seen = set()
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        if isinstance(p, list):
            for u in p:
                if isinstance(u, str) and u.strip() and u.strip() not in seen:
                    out.append(u.strip())
                    seen.add(u.strip())
        elif isinstance(p, str):
            if p.strip() and p.strip() not in seen:
                out.append(p.strip())
                seen.add(p.strip())
    return out


def _has_text(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


# --------------------------------------------------------------------------- #
# Verification Subtree for One University                                     #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    uni_idx: int,
) -> None:
    """
    Build and verify the rubric subtree for a single university (indexed 1..3).
    """
    uid = f"u{uni_idx}"
    uni_name = uni.name or f"University #{uni_idx}"

    # Top-level node for this university (non-critical to allow partial credit across different schools)
    uni_node = evaluator.add_parallel(
        id=f"{uid}_node",
        desc=f"{['First','Second','Third'][uni_idx-1]} public university meeting all criteria",
        parent=parent_node,
        critical=False,
    )

    # Collect a general pool of URLs that may help verify identity/public status
    general_urls = _norm_urls(
        uni.public_status_source_urls,
        uni.outcomes_data_url,
        uni.services_url,
        uni.employer_engagement_url,
        uni.career_fair_url,
        uni.nace_evidence_url,
    )

    # 1) Public university status (leaf, critical)
    public_leaf = evaluator.add_leaf(
        id=f"{uid}_public_university_status",
        desc="University is a public institution in the United States",
        parent=uni_node,
        critical=True,
    )
    public_claim = f"The institution '{uni_name}' is a public university located in the United States."
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=general_urls,
        additional_instruction=(
            "Accept phrases like 'public university', 'public research university', "
            "'state university', 'public institution', or being part of a state university system. "
            "The page(s) should clearly indicate it is public and in the U.S."
        ),
    )

    # 2) Career outcomes rate (critical group)
    outcomes_node = evaluator.add_parallel(
        id=f"{uid}_career_outcomes_rate",
        desc="Reports career outcomes rate of at least 90% within six months of graduation",
        parent=uni_node,
        critical=True,
    )

    # outcomes_data_url must be provided
    outcomes_url_exists = evaluator.add_custom_node(
        result=_has_text(uni.outcomes_data_url),
        id=f"{uid}_outcomes_data_url",
        desc="Direct URL to career outcomes data page is provided",
        parent=outcomes_node,
        critical=True,
    )

    # outcomes >= 90%
    outcomes_value_leaf = evaluator.add_leaf(
        id=f"{uid}_outcomes_rate_value",
        desc="Specific percentage is at least 90%",
        parent=outcomes_node,
        critical=True,
    )
    outcomes_value_claim = (
        "The most recent graduating class has a career outcomes (or placement/success) rate of at least 90%."
    )
    await evaluator.verify(
        claim=outcomes_value_claim,
        node=outcomes_value_leaf,
        sources=uni.outcomes_data_url,
        additional_instruction=(
            "Look for an overall career outcomes/placement/success rate defined as employed, "
            "continuing education, or service/volunteer within the stated time window. "
            "Values like '90%', '90.0%', '≥90%', or 'at least 90%' should qualify."
        ),
    )

    # measured within six months
    outcomes_timeline_leaf = evaluator.add_leaf(
        id=f"{uid}_outcomes_rate_timeline",
        desc="Outcome measured within six months of graduation",
        parent=outcomes_node,
        critical=True,
    )
    timeline_claim = "The career outcomes metric is explicitly measured within six months of graduation."
    await evaluator.verify(
        claim=timeline_claim,
        node=outcomes_timeline_leaf,
        sources=uni.outcomes_data_url,
        additional_instruction=(
            "Accept phrases like 'within six months of graduation', 'six months after graduation', "
            "or '6 months post-graduation'. If the page uses another standard NACE timeframe but explicitly states it, that also qualifies."
        ),
    )

    # class year specified
    outcomes_year_leaf = evaluator.add_leaf(
        id=f"{uid}_outcomes_rate_class_year",
        desc="Graduating class year is specified",
        parent=outcomes_node,
        critical=True,
    )
    if _has_text(uni.outcomes_class_year):
        class_year_claim = (
            f"The outcomes data page specifies that the graduating class year is {uni.outcomes_class_year}."
        )
    else:
        class_year_claim = (
            "The outcomes data page specifies the graduating class year for the most recent class reported."
        )
    await evaluator.verify(
        claim=class_year_claim,
        node=outcomes_year_leaf,
        sources=uni.outcomes_data_url,
        additional_instruction=(
            "Look for explicit references such as 'Class of 2023', 'Class of 2024', or similar."
        ),
    )

    # 3) NACE compliance + knowledge rate (critical group)
    nace_node = evaluator.add_parallel(
        id=f"{uid}_nace_compliance",
        desc="Career center follows NACE First Destination Survey standards and reports knowledge rate",
        parent=uni_node,
        critical=True,
    )

    # URL to support NACE (either a dedicated NACE/methods URL or the outcomes page if it states NACE)
    nace_url_exists = evaluator.add_custom_node(
        result=_has_text(uni.nace_evidence_url) or _has_text(uni.outcomes_data_url),
        id=f"{uid}_nace_evidence_url",
        desc="URL reference supporting NACE compliance is provided",
        parent=nace_node,
        critical=True,
    )
    nace_sources = _norm_urls(uni.nace_evidence_url, uni.outcomes_data_url)

    # NACE standards followed
    nace_followed_leaf = evaluator.add_leaf(
        id=f"{uid}_nace_standards_followed",
        desc="Evidence that NACE standards are followed in outcomes reporting",
        parent=nace_node,
        critical=True,
    )
    nace_claim = (
        "The outcomes reporting states that it follows NACE (National Association of Colleges and Employers) "
        "First Destination Survey (FDS) standards or methodology."
    )
    await evaluator.verify(
        claim=nace_claim,
        node=nace_followed_leaf,
        sources=nace_sources,
        additional_instruction=(
            "Look for phrases like 'NACE standards', 'NACE First Destination Survey (FDS)', "
            "'aligned with NACE', or a methodology section citing NACE."
        ),
    )

    # Knowledge rate reported
    knowledge_leaf = evaluator.add_leaf(
        id=f"{uid}_knowledge_rate_reported",
        desc="Knowledge rate percentage is reported",
        parent=nace_node,
        critical=True,
    )
    if _has_text(uni.knowledge_rate_percent):
        knowledge_claim = f"The page reports a knowledge rate of {uni.knowledge_rate_percent}."
    else:
        knowledge_claim = "The page reports a knowledge rate (knowledge response rate) as a percentage."
    await evaluator.verify(
        claim=knowledge_claim,
        node=knowledge_leaf,
        sources=nace_sources,
        additional_instruction=(
            "Accept terms like 'knowledge rate', 'knowledge response rate', or similar. "
            "It should be a percentage related to the share of graduates for whom outcomes data is known."
        ),
    )

    # 4) Career fairs (critical group)
    fairs_node = evaluator.add_parallel(
        id=f"{uid}_career_fairs",
        desc="University hosts regular career and internship fairs (fall and spring minimum)",
        parent=uni_node,
        critical=True,
    )

    # Career fair URL provided
    fair_url_exists = evaluator.add_custom_node(
        result=_has_text(uni.career_fair_url),
        id=f"{uid}_career_fair_url",
        desc="URL reference to career fair information is provided",
        parent=fairs_node,
        critical=True,
    )

    # Evidence of fall and spring fairs
    fair_schedule_leaf = evaluator.add_leaf(
        id=f"{uid}_regular_fair_schedule",
        desc="Evidence of career fairs held at least in fall and spring semesters",
        parent=fairs_node,
        critical=True,
    )
    fair_schedule_claim = (
        "The university hosts career and/or internship fairs at least in both the fall and spring semesters each academic year."
    )
    await evaluator.verify(
        claim=fair_schedule_claim,
        node=fair_schedule_leaf,
        sources=uni.career_fair_url,
        additional_instruction=(
            "Look for references like 'Fall Career & Internship Fair' and 'Spring Career & Internship Fair', "
            "or a recurring schedule clearly mentioning both semesters. Calendars or event pages qualify."
        ),
    )

    # 5) Employer engagement (critical group)
    employers_node = evaluator.add_parallel(
        id=f"{uid}_employer_engagement",
        desc="At least 350 employers/organizations engaged during most recent academic year",
        parent=uni_node,
        critical=True,
    )

    # Employer engagement URL provided
    employers_url_exists = evaluator.add_custom_node(
        result=_has_text(uni.employer_engagement_url),
        id=f"{uid}_employer_engagement_url",
        desc="URL reference to employer engagement statistics is provided",
        parent=employers_node,
        critical=True,
    )

    # At least 350 employers engaged
    employer_count_leaf = evaluator.add_leaf(
        id=f"{uid}_employer_count",
        desc="Specific number or description showing at least 350 employers/organizations engaged",
        parent=employers_node,
        critical=True,
    )
    employer_count_claim = (
        "At least 350 distinct employers or organizations engaged with students in the most recent academic year, "
        "including attending career fairs, recruiting on campus, or similar engagement."
    )
    await evaluator.verify(
        claim=employer_count_claim,
        node=employer_count_leaf,
        sources=uni.employer_engagement_url,
        additional_instruction=(
            "Accept phrasings like '350+', 'over 350', 'more than 350', or any number >= 350. "
            "The context may include employers attending fairs, on-campus recruiting, info sessions, or other engagement."
        ),
    )

    # 6) Core services (critical group)
    services_node = evaluator.add_parallel(
        id=f"{uid}_core_services",
        desc="Career center provides all four required core services",
        parent=uni_node,
        critical=True,
    )

    # Services URL provided
    services_url_exists = evaluator.add_custom_node(
        result=_has_text(uni.services_url),
        id=f"{uid}_services_url",
        desc="URL reference to career center services page is provided",
        parent=services_node,
        critical=True,
    )

    # Resume/CV review
    resume_leaf = evaluator.add_leaf(
        id=f"{uid}_resume_cv_review",
        desc="Resume/CV review service is offered",
        parent=services_node,
        critical=True,
    )
    resume_claim = "The career center offers resume review or CV review services to students."
    await evaluator.verify(
        claim=resume_claim,
        node=resume_leaf,
        sources=uni.services_url,
        additional_instruction=(
            "Accept phrases like 'resume reviews', 'CV critiques', 'document reviews', or similar."
        ),
    )

    # Mock interviews
    mock_leaf = evaluator.add_leaf(
        id=f"{uid}_mock_interviews",
        desc="Mock interview preparation service is offered",
        parent=services_node,
        critical=True,
    )
    mock_claim = "The career center offers mock interviews or interview practice/preparation services."
    await evaluator.verify(
        claim=mock_claim,
        node=mock_leaf,
        sources=uni.services_url,
        additional_instruction=(
            "Accept 'mock interviews', 'practice interviews', 'interview prep', or platform-based practice such as 'Big Interview'."
        ),
    )

    # Career advising
    advising_leaf = evaluator.add_leaf(
        id=f"{uid}_career_advising",
        desc="Career advising/counseling appointments are available",
        parent=services_node,
        critical=True,
    )
    advising_claim = "The career center provides career advising or counseling through appointments or drop-ins."
    await evaluator.verify(
        claim=advising_claim,
        node=advising_leaf,
        sources=uni.services_url,
        additional_instruction=(
            "Accept 'career advising', 'career coaching', 'one-on-one appointments', or 'drop-in advising'."
        ),
    )

    # Job/Internship platform (with name)
    platform_leaf = evaluator.add_leaf(
        id=f"{uid}_job_platform",
        desc="Dedicated job/internship search platform is provided (platform name specified)",
        parent=services_node,
        critical=True,
    )
    if _has_text(uni.job_platform_name):
        platform_claim = (
            f"The career center provides a dedicated job/internship search platform named {uni.job_platform_name}."
        )
    else:
        platform_claim = (
            "The career center provides a dedicated job/internship search platform (e.g., Handshake, Symplicity, or 12twenty)."
        )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=uni.services_url,
        additional_instruction=(
            "Verify that a named platform is provided (e.g., Handshake, Symplicity, 12twenty, WayUp). "
            "Minor name variants are acceptable."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point for evaluating an agent's answer for the public universities career services task.
    """
    # Initialize evaluator (root remains non-critical to allow partial credit across the three universities)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Universities are evaluated independently
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Prepare exactly three entries (truncate or pad)
    universities: List[UniversityInfo] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityInfo())

    # Build verification tree per university (parallel under root)
    for i in range(3):
        await verify_university(
            evaluator=evaluator,
            parent_node=root,
            uni=universities[i],
            uni_idx=i + 1,
        )

    # Return standard summary
    return evaluator.get_summary()