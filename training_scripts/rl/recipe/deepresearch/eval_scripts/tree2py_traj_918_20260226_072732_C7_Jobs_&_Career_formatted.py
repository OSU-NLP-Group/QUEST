import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "nv_state_position_it_criteria"
TASK_DESCRIPTION = """Find a current Nevada state government position that meets ALL of the following criteria:
- Pay grade between 30 and 40 (inclusive)
- Minimum annual salary of at least $50,000
- Requires a bachelor's degree as minimum education qualification
- Requires at least 2 years of relevant work experience
- Has an active job posting with an application deadline after February 26, 2026
- Is a classified position (not unclassified)
- Is located in either Carson City or Las Vegas, Nevada
- Is in the field of Information Technology, Data Science, Cybersecurity, or Information Security
- Is full-time employment
- Is posted on the official Nevada state jobs website (nvjobs.nv.gov or the NEATS system)
- Includes a complete minimum qualifications section listing education and experience requirements

Provide the following information for the position you identify:
1. The official job title exactly as stated in the posting
2. The specific recruitment or announcement ID number
3. The direct URL to the job posting page
"""

CUTOFF_DATE_TEXT = "February 26, 2026"
CUTOFF_DATE_ISO = "2026-02-26"

# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class NVPositionExtraction(BaseModel):
    job_title: Optional[str] = None
    recruitment_id: Optional[str] = None
    posting_url: Optional[str] = None

    pay_grade: Optional[str] = None
    minimum_salary: Optional[str] = None
    education_requirement: Optional[str] = None
    experience_requirement: Optional[str] = None
    application_deadline: Optional[str] = None
    classification_status: Optional[str] = None
    location: Optional[str] = None
    field: Optional[str] = None
    employment_type: Optional[str] = None

    additional_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_position() -> str:
    return """
    Extract from the answer the details for the identified Nevada state government position. Return fields exactly as presented in the answer text.

    Required fields (return null if missing):
    - job_title: The official job title exactly as stated in the posting
    - recruitment_id: The recruitment/announcement ID number as written (e.g., "Recruitment: 12345", "Announcement # 14-XYZ"). Extract only the ID string/number, not the label.
    - posting_url: The direct URL to the job posting page
    
    Additional fields (return null if missing):
    - pay_grade: The pay grade text (e.g., "Grade 36") as given in the answer
    - minimum_salary: The minimum annual salary value or text (e.g., "$56,628", "$27.50/hour") as given in the answer
    - education_requirement: The minimum education requirement text (e.g., "Bachelor's degree in ...") from the answer
    - experience_requirement: The minimum years/experience text (e.g., "two (2) years of ...") from the answer
    - application_deadline: The application deadline/close date text as presented in the answer (e.g., "Closes March 10, 2026")
    - classification_status: Text indicating "classified" or "unclassified" as per the answer
    - location: The location city text from the answer (e.g., "Carson City, NV" or "Las Vegas, NV")
    - field: The field/discipline text from the answer (e.g., "Information Technology", "Cybersecurity")
    - employment_type: The employment type text from the answer (e.g., "Full-Time")
    - additional_urls: Any additional URLs cited in the answer for this posting (exclude duplicates of posting_url)

    Rules:
    - Do not infer or invent values.
    - Extract only what the answer explicitly states.
    - For URLs, extract complete valid URLs. If missing a protocol, prepend http://.
    """


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _combine_sources(extracted: NVPositionExtraction) -> List[str]:
    urls = []
    if extracted.posting_url:
        urls.append(extracted.posting_url)
    if extracted.additional_urls:
        for u in extracted.additional_urls:
            if u and u not in urls:
                urls.append(u)
    return urls


def _is_official_nv_source(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        # Official criteria per rubric: nvjobs.nv.gov or NEATS system (commonly nvapps.state.nv.us under /NEATS/)
        if "nvjobs.nv.gov" in host:
            return True
        if "nvapps.state.nv.us" in host:
            return True  # NEATS lives here (e.g., /NEATS/Recruiting/...)
        return False
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Verification Tree Construction and Checks
# -----------------------------------------------------------------------------
async def verify_nv_position(evaluator: Evaluator, parent_node, info: NVPositionExtraction) -> None:
    """
    Build verification leaves under the given parent node for all rubric criteria.
    All children under this parent are critical per rubric.
    """
    sources = _combine_sources(info)

    # 1) Posting URL existence and basic validity (acts as gating prerequisite for other URL-backed checks)
    posting_url_exists = bool(info.posting_url) and isinstance(info.posting_url, str) and info.posting_url.startswith(("http://", "https://"))
    posting_url_node = evaluator.add_custom_node(
        result=posting_url_exists,
        id="Posting_URL",
        desc="Provide the direct URL to the specific job posting page",
        parent=parent_node,
        critical=True
    )

    # 2) Official Source domain check (must be nvjobs.nv.gov or NEATS [nvapps.state.nv.us])
    official_source_ok = _is_official_nv_source(info.posting_url)
    official_source_node = evaluator.add_custom_node(
        result=official_source_ok,
        id="Official_Source",
        desc="Position must be posted on official Nevada state jobs website (nvjobs.nv.gov or NEATS system)",
        parent=parent_node,
        critical=True
    )

    extra_prereqs = [posting_url_node, official_source_node]

    # 3) Job Title exact match against posting page
    job_title_leaf = evaluator.add_leaf(
        id="Job_Title",
        desc="Provide the official job title exactly as stated in the posting",
        parent=parent_node,
        critical=True
    )
    title_text = info.job_title or ""
    await evaluator.verify(
        claim=f"The official job title shown on the job posting page is exactly '{title_text}'. Do not accept paraphrases. The comparison should be character-for-character identical, ignoring only trivial leading/trailing whitespace.",
        node=job_title_leaf,
        sources=info.posting_url,
        additional_instruction="Be strict. Consider differences in punctuation, hyphenation, or casing as not exact unless clearly identical on the page.",
        extra_prerequisites=extra_prereqs
    )

    # 4) Recruitment/Announcement ID matches page
    recruitment_leaf = evaluator.add_leaf(
        id="Recruitment_ID",
        desc="Provide the specific recruitment or announcement ID number for the position",
        parent=parent_node,
        critical=True
    )
    rid_text = info.recruitment_id or ""
    await evaluator.verify(
        claim=f"The job posting page explicitly lists the recruitment or announcement ID as '{rid_text}', matching exactly (ignoring the label such as 'Recruitment:' or 'Announcement #').",
        node=recruitment_leaf,
        sources=info.posting_url,
        additional_instruction="Accept minor formatting like presence/absence of leading '#', spaces, or label words, but the core alphanumeric ID must match.",
        extra_prerequisites=extra_prereqs
    )

    # 5) Pay grade between 30 and 40 inclusive
    pay_grade_leaf = evaluator.add_leaf(
        id="Pay_Grade_Range",
        desc="Position must have a pay grade between 30 and 40 (inclusive)",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The job posting indicates a pay/classification grade between 30 and 40 inclusive (e.g., 'Grade 30' through 'Grade 40').",
        node=pay_grade_leaf,
        sources=info.posting_url,
        additional_instruction="Look for 'Grade', 'Pay Grade', or classification series grade on the page. If multiple grades are shown, accept if any applicable grade for this recruitment is between 30 and 40 inclusive.",
        extra_prerequisites=extra_prereqs
    )

    # 6) Minimum salary threshold at least $50,000
    min_salary_leaf = evaluator.add_leaf(
        id="Minimum_Salary_Threshold",
        desc="Position's minimum annual salary must be at least $50,000",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum annual salary for the position is at least $50,000. If a salary range is provided, the lower bound is >= $50,000. If only an hourly rate is shown, the implied annualized minimum (hourly * 2080) is >= $50,000.",
        node=min_salary_leaf,
        sources=sources,
        additional_instruction="Consider base salary only (ignore employer-paid benefits like PERS). If pay is hourly or monthly, perform a reasonable conversion to annual to assess the >= $50k threshold.",
        extra_prerequisites=extra_prereqs
    )

    # 7) Education requirement: bachelor's degree as minimum
    edu_leaf = evaluator.add_leaf(
        id="Education_Requirement",
        desc="Position must require a bachelor's degree as minimum education qualification",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The posting explicitly requires a Bachelor's degree as a minimum education qualification.",
        node=edu_leaf,
        sources=info.posting_url,
        additional_instruction="Be strict: If the posting says 'Bachelor's degree OR equivalent experience', that is not strictly requiring a Bachelor's degree; mark as not satisfied.",
        extra_prerequisites=extra_prereqs
    )

    # 8) Experience requirement: at least 2 years
    exp_leaf = evaluator.add_leaf(
        id="Experience_Requirement",
        desc="Position must require at least 2 years of relevant experience",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The posting requires at least two (2) years of relevant work experience.",
        node=exp_leaf,
        sources=info.posting_url,
        additional_instruction="Accept wording like 'two years', '2 years', or '24 months'. If experience can substitute for education, ensure that minimally 2 years are required for eligibility.",
        extra_prerequisites=extra_prereqs
    )

    # 9) Active posting with application deadline after February 26, 2026
    active_leaf = evaluator.add_leaf(
        id="Active_Posting",
        desc="Position must have an active job posting with application deadline after February 26, 2026",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The posting shows an application deadline/close date strictly after {CUTOFF_DATE_TEXT}, and the posting is currently active/open.",
        node=active_leaf,
        sources=info.posting_url,
        additional_instruction=f"If the page shows 'Open Until Filled', 'Continuous', or no close date, treat this as not satisfying 'deadline after {CUTOFF_DATE_TEXT}'. If the close date is exactly {CUTOFF_DATE_TEXT} or earlier, mark incorrect.",
        extra_prerequisites=extra_prereqs
    )

    # 10) Classified position (not unclassified)
    class_leaf = evaluator.add_leaf(
        id="Classification_Status",
        desc="Position must be a classified position (not unclassified)",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The posting indicates the position is classified (not unclassified).",
        node=class_leaf,
        sources=info.posting_url,
        additional_instruction="Look for explicit text such as 'Classified' or statements indicating the position is part of the classified service.",
        extra_prerequisites=extra_prereqs
    )

    # 11) Location: Carson City or Las Vegas
    location_leaf = evaluator.add_leaf(
        id="Location_Requirement",
        desc="Position must be located in either Carson City or Las Vegas",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The job location listed on the posting is in Carson City, Nevada or Las Vegas, Nevada.",
        node=location_leaf,
        sources=info.posting_url,
        additional_instruction="If multiple locations are listed, accept if any definitive duty station is Carson City or Las Vegas. Remote/hybrid is acceptable only if a listed duty station city is Carson City or Las Vegas.",
        extra_prerequisites=extra_prereqs
    )

    # 12) Field requirement: IT, Data Science, Cybersecurity, or Information Security
    field_leaf = evaluator.add_leaf(
        id="Field_Requirement",
        desc="Position must be in Information Technology, Data Science, Cybersecurity, or Information Security field",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="This position is in the domain of Information Technology, Data Science, Cybersecurity, or Information Security.",
        node=field_leaf,
        sources=info.posting_url,
        additional_instruction="Use the class specification/series, job title, and duties to determine. Accept terms like 'IT Professional', 'Information Security', 'Cybersecurity Analyst', 'Data Scientist', 'Data Engineer', etc.",
        extra_prerequisites=extra_prereqs
    )

    # 13) Employment type: Full-time
    fulltime_leaf = evaluator.add_leaf(
        id="Employment_Type",
        desc="Position must be full-time employment",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The position is full-time employment.",
        node=fulltime_leaf,
        sources=info.posting_url,
        additional_instruction="Look for 'Full-Time', 'FTE 1.0', or equivalent indications. Do not accept part-time, hourly intermittent, seasonal, or temporary unless explicitly full-time.",
        extra_prerequisites=extra_prereqs
    )

    # 14) Minimum Qualifications section present and complete (education + experience)
    quals_leaf = evaluator.add_leaf(
        id="Qualifications_Section",
        desc="Position posting must include a complete minimum qualifications section listing education and experience requirements",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The posting contains a 'Minimum Qualifications' section that explicitly lists both the required education and required experience.",
        node=quals_leaf,
        sources=info.posting_url,
        additional_instruction="Do not accept if only duties are listed. The section must clearly state both education and experience minima.",
        extra_prerequisites=extra_prereqs
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Nevada state position task and return a structured summary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall tree aggregation
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

    # Extract structured info from the answer
    extracted: NVPositionExtraction = await evaluator.extract(
        prompt=prompt_extract_position(),
        template_class=NVPositionExtraction,
        extraction_name="position_extraction",
    )

    # Record cutoff-date and helpful info
    evaluator.add_custom_info(
        info={"cutoff_date_text": CUTOFF_DATE_TEXT, "cutoff_date_iso": CUTOFF_DATE_ISO},
        info_type="metadata",
        info_name="cutoff_policy"
    )

    # Build a critical aggregator node representing the rubric root
    main = evaluator.add_parallel(
        id="Nevada_State_Position",
        desc="Find a current Nevada state government position that meets all specified criteria",
        parent=root,
        critical=True
    )

    # Verify all rubric criteria under the critical node
    await verify_nv_position(evaluator, main, extracted)

    # Return evaluation summary
    return evaluator.get_summary()