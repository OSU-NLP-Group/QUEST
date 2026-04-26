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
TASK_ID = "spring2026_career_services_universities"
TASK_DESCRIPTION = """Identify three U.S. universities that meet all of the following career services criteria for Spring 2026:

Career Fair Requirements:
1. The university must host a Spring 2026 in-person All-Majors Career Fair (or general/broad-scope career fair open to students from all majors) between February 1, 2026 and March 31, 2026.
2. The career fair date, start time, end time, and physical campus location must be publicly listed on the university's official website or official university career center pages.

Resume Services Requirements:
3. The university's career center must offer resume review or critique services during Spring 2026.
4. The resume service must have a publicly available method for students to access it (such as scheduled appointments, drop-in hours, or workshops).

Professional Development Workshop Requirements:
5. The university's career center must host at least one professional development workshop during the Spring 2026 semester (January-May 2026) that focuses on career fair preparation, resume writing, or networking.
6. The workshop must have a specific scheduled date and time that is publicly listed.

Career Counseling Requirements:
7. The university's career center must provide career counseling or career advising services.
8. The career counseling service must have publicly available contact information or an online scheduling system.

Employer Engagement Requirements:
9. The university must host or facilitate employer information sessions during Spring 2026.

Career Center Operations Requirements:
10. The university's career center must have publicly listed operating hours.
11. The university's career center must have publicly available contact information (email, phone, or physical address).

For each of the three universities you identify, provide:
- University name
- Career fair date, time, and location
- Link to the official career fair information page
- Description of resume services and how students can access them
- Link to resume services information
- At least one specific workshop with its topic, date, and time
- Link to workshop information
- Description of career counseling services and how to contact or schedule
- Link to career counseling information
- Description of employer information session availability
- Link to employer session information
- Career center operating hours and contact information
- Link to career center contact page

All information must be verifiable through official university websites or official university-affiliated platforms.
"""

# Timeframe constants
CAREER_FAIR_START = "2026-02-01"
CAREER_FAIR_END = "2026-03-31"
SPRING_2026_START = "2026-01-01"
SPRING_2026_END = "2026-05-31"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CareerFairInfo(BaseModel):
    date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None


class ResumeInfo(BaseModel):
    description: Optional[str] = None
    access_method: Optional[str] = None
    url: Optional[str] = None


class WorkshopInfo(BaseModel):
    topic: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    url: Optional[str] = None


class CounselingInfo(BaseModel):
    description: Optional[str] = None
    contact_or_scheduling: Optional[str] = None
    url: Optional[str] = None


class EmployerInfo(BaseModel):
    description: Optional[str] = None
    url: Optional[str] = None


class CareerCenterInfo(BaseModel):
    operating_hours: Optional[str] = None
    contact_info: Optional[str] = None
    contact_page_url: Optional[str] = None


class UniversityEntry(BaseModel):
    name: Optional[str] = None
    career_fair: Optional[CareerFairInfo] = None
    resume: Optional[ResumeInfo] = None
    workshop: Optional[WorkshopInfo] = None
    counseling: Optional[CounselingInfo] = None
    employer: Optional[EmployerInfo] = None
    career_center: Optional[CareerCenterInfo] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to the first three universities and their requested Spring 2026 career services details from the answer.

For each university, extract the following fields as text exactly as presented in the answer. If any field is missing, set it to null. For URLs, extract the actual URL string exactly as shown (accept plain URLs or markdown links), and include the protocol (http/https).

For each university (in order of appearance in the answer):
- name

- career_fair:
  - date (e.g., "February 12, 2026")
  - start_time (e.g., "10:00 AM")
  - end_time (e.g., "3:00 PM")
  - location (e.g., "Student Union Ballroom")
  - url (link to official career fair information page)

- resume:
  - description (resume review/critique services summary)
  - access_method (appointments/drop-ins/workshops description, if provided)
  - url (official resume services info page)

- workshop:
  - topic (e.g., "Resume Writing", "Career Fair Prep", or "Networking")
  - date (e.g., "March 5, 2026")
  - time (e.g., "1:00–2:00 PM")
  - url (official workshop page)

- counseling:
  - description (career counseling/advising services summary)
  - contact_or_scheduling (email/phone/office info or online scheduling instructions/link)
  - url (official counseling/advising info page)

- employer:
  - description (employer information sessions availability summary)
  - url (official employer info sessions/events page)

- career_center:
  - operating_hours (career center hours as text)
  - contact_info (email/phone/address as text)
  - contact_page_url (official career center contact page URL)

Only consider the first three universities if more are present; ignore additional ones beyond three.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _gather_official_urls(u: UniversityEntry) -> List[str]:
    urls: List[str] = []
    if u.career_fair and _nonempty(u.career_fair.url):
        urls.append(u.career_fair.url)  # type: ignore
    if u.resume and _nonempty(u.resume.url):
        urls.append(u.resume.url)  # type: ignore
    if u.workshop and _nonempty(u.workshop.url):
        urls.append(u.workshop.url)  # type: ignore
    if u.counseling and _nonempty(u.counseling.url):
        urls.append(u.counseling.url)  # type: ignore
    if u.employer and _nonempty(u.employer.url):
        urls.append(u.employer.url)  # type: ignore
    if u.career_center and _nonempty(u.career_center.contact_page_url):
        urls.append(u.career_center.contact_page_url)  # type: ignore
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityEntry,
    index: int,
) -> None:
    """
    Build and verify the subtree for a single university with partial credit allowed.
    All concrete checks are leaf nodes with binary outcomes.
    """

    # Parent node for this university (parallel aggregation, non-critical for partial credit)
    uni_node = evaluator.add_parallel(
        id=f"university_{index+1}",
        desc=f"University #{index+1} verification (partial credit allowed).",
        parent=parent_node,
        critical=False
    )

    # 1) University name provided (critical)
    name_provided = evaluator.add_custom_node(
        result=_nonempty(uni.name),
        id=f"u{index+1}_name_provided",
        desc="University name is provided.",
        parent=uni_node,
        critical=True
    )

    # 2) US 4-year institution (critical) - verify via any official URLs available
    us4yr_leaf = evaluator.add_leaf(
        id=f"u{index+1}_us_4yr",
        desc="University is a 4-year institution in the United States.",
        parent=uni_node,
        critical=True
    )
    us4yr_claim = (
        f"The institution '{uni.name or ''}' is a U.S.-based four-year university (offers bachelor's degree programs)."
    )
    await evaluator.verify(
        claim=us4yr_claim,
        node=us4yr_leaf,
        sources=_gather_official_urls(uni),
        additional_instruction=(
            "Use the provided official university/career pages to determine whether this is a U.S. four-year university. "
            "Accept terms like university/college; evidence can include mention of bachelor's/undergraduate programs or U.S. location. "
            "If no official page supports this, mark as not supported."
        ),
    )

    # ------- Career Fair checks -------
    fair_url = uni.career_fair.url if (uni.career_fair and _nonempty(uni.career_fair.url)) else None

    # 3) Career fair official link provided (critical)
    fair_link_provided = evaluator.add_custom_node(
        result=_nonempty(fair_url),
        id=f"u{index+1}_fair_link_provided",
        desc="Provides a link to the official career fair information page.",
        parent=uni_node,
        critical=True
    )

    # 4) Career fair qualifies (critical)
    fair_qualifies_leaf = evaluator.add_leaf(
        id=f"u{index+1}_fair_qualifies",
        desc="In-person All-Majors career fair between Feb 1, 2026 and Mar 31, 2026.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This official career fair page shows an in-person general/all-majors career fair scheduled between "
            "February 1, 2026 and March 31, 2026."
        ),
        node=fair_qualifies_leaf,
        sources=[fair_url] if fair_url else None,
        additional_instruction=(
            "Confirm BOTH of the following on the page: "
            "(a) the fair is in-person (not virtual) with a physical campus venue, and "
            "(b) it is a general/all-majors/broad-scope fair open to all students (accept synonyms like 'All Majors', 'All Students', "
            "'University-wide', 'Career Expo' for all majors). Also confirm the date is within 2026-02-01 to 2026-03-31."
        ),
    )

    # 5) Career fair details listed (critical)
    fair_details_leaf = evaluator.add_leaf(
        id=f"u{index+1}_fair_details_listed",
        desc="Career fair page lists date, start time, end time, and a physical on-campus location.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This page lists the career fair date, start time, end time, and the physical on-campus location (e.g., a building or address)."
        ),
        node=fair_details_leaf,
        sources=[fair_url] if fair_url else None,
        additional_instruction=(
            "All four elements must be present on the page: date, start time, end time, and a physical location. "
            "Time ranges like '10:00 AM – 3:00 PM' satisfy start and end time. "
            "If any are missing, mark as not supported."
        ),
    )

    # ------- Resume Services checks -------
    resume_url = uni.resume.url if (uni.resume and _nonempty(uni.resume.url)) else None

    # 6) Resume services official link provided (critical)
    resume_link_provided = evaluator.add_custom_node(
        result=_nonempty(resume_url),
        id=f"u{index+1}_resume_link_provided",
        desc="Provides a link to official resume-services information.",
        parent=uni_node,
        critical=True
    )

    # 7) Resume review offered Spring 2026 (critical)
    resume_offered_leaf = evaluator.add_leaf(
        id=f"u{index+1}_resume_offered_spring2026",
        desc="Career center offers resume review/critique services during Spring 2026.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="During Spring 2026 (Jan–May 2026), the career center offers resume review or critique services.",
        node=resume_offered_leaf,
        sources=[resume_url] if resume_url else None,
        additional_instruction=(
            "General services pages count if they reasonably indicate ongoing resume reviews available to students "
            "and are not limited to a different term. If clearly out of date or restricted to another term, do not support."
        ),
    )

    # 8) Resume access method publicly described (critical)
    resume_access_leaf = evaluator.add_leaf(
        id=f"u{index+1}_resume_access_public",
        desc="Public method for students to access resume review is provided (appointments, drop-ins, or workshops).",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This page provides a public method for students to access resume reviews (e.g., appointment system, "
            "drop-in hours, or workshops), with sufficient instructions."
        ),
        node=resume_access_leaf,
        sources=[resume_url] if resume_url else None,
        additional_instruction=(
            "Look for appointment links/systems, stated drop-in hours, or workshop sign-up details. "
            "If no clear access method is listed, mark as not supported."
        ),
    )

    # ------- Workshop checks -------
    workshop_url = uni.workshop.url if (uni.workshop and _nonempty(uni.workshop.url)) else None

    # 9) Workshop official link provided (critical)
    workshop_link_provided = evaluator.add_custom_node(
        result=_nonempty(workshop_url),
        id=f"u{index+1}_workshop_link_provided",
        desc="Provides a link to official workshop information.",
        parent=uni_node,
        critical=True
    )

    # 10) Workshop exists Spring 2026 and topic qualified (critical)
    workshop_exists_leaf = evaluator.add_leaf(
        id=f"u{index+1}_workshop_exists_spring2026",
        desc="At least one Spring 2026 workshop focuses on career fair prep, resume writing, or networking.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This page lists at least one professional development workshop during Spring 2026 (Jan–May 2026) "
            "that focuses on career fair preparation, resume writing, or networking."
        ),
        node=workshop_exists_leaf,
        sources=[workshop_url] if workshop_url else None,
        additional_instruction=(
            "Accept reasonable synonyms (e.g., 'Career Fair Prep', 'Resume Workshop', 'Networking Strategies'). "
            "Ensure the date is within Jan–May 2026."
        ),
    )

    # 11) Workshop date and time provided (critical)
    workshop_datetime_leaf = evaluator.add_leaf(
        id=f"u{index+1}_workshop_datetime_provided",
        desc="The identified workshop includes a specific scheduled date and time that is publicly listed.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This workshop page lists a specific scheduled date and time for the event."
        ),
        node=workshop_datetime_leaf,
        sources=[workshop_url] if workshop_url else None,
        additional_instruction=(
            "Both a specific date and a specific time (or time range) must be present on the page for at least one relevant workshop."
        ),
    )

    # ------- Career Counseling checks -------
    counseling_url = uni.counseling.url if (uni.counseling and _nonempty(uni.counseling.url)) else None

    # 12) Counseling official link provided (critical)
    counseling_link_provided = evaluator.add_custom_node(
        result=_nonempty(counseling_url),
        id=f"u{index+1}_counseling_link_provided",
        desc="Provides a link to official career counseling/advising information.",
        parent=uni_node,
        critical=True
    )

    # 13) Counseling offered (critical)
    counseling_offered_leaf = evaluator.add_leaf(
        id=f"u{index+1}_counseling_offered",
        desc="Career center provides career counseling or advising services.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="The career center offers career counseling/advising services for students.",
        node=counseling_offered_leaf,
        sources=[counseling_url] if counseling_url else None,
        additional_instruction=(
            "Verify language indicating counseling/advising services are provided (appointments, advising sessions, etc.)."
        ),
    )

    # 14) Counseling contact or scheduling publicly available (critical)
    counseling_contact_leaf = evaluator.add_leaf(
        id=f"u{index+1}_counseling_contact_or_scheduling",
        desc="Career counseling includes publicly available contact information or an online scheduling system.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This page provides publicly available contact information (email/phone/address) or an online scheduling system "
            "for career counseling/advising."
        ),
        node=counseling_contact_leaf,
        sources=[counseling_url] if counseling_url else None,
        additional_instruction=(
            "Accept appointment links, calendaring systems, or clearly stated contact details for counseling/advising."
        ),
    )

    # ------- Employer Information Sessions checks -------
    employer_url = uni.employer.url if (uni.employer and _nonempty(uni.employer.url)) else None

    # 15) Employer sessions official link provided (critical)
    employer_link_provided = evaluator.add_custom_node(
        result=_nonempty(employer_url),
        id=f"u{index+1}_employer_link_provided",
        desc="Provides a link to official employer session information.",
        parent=uni_node,
        critical=True
    )

    # 16) Employer info sessions available Spring 2026 (critical)
    employer_sessions_leaf = evaluator.add_leaf(
        id=f"u{index+1}_employer_sessions_spring2026",
        desc="University hosts or facilitates employer information sessions during Spring 2026.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This page shows employer information sessions occurring during Spring 2026 (Jan–May 2026)."
        ),
        node=employer_sessions_leaf,
        sources=[employer_url] if employer_url else None,
        additional_instruction=(
            "Look for event listings or descriptions indicating employer info sessions or employer-hosted talks during Spring 2026."
        ),
    )

    # ------- Career Center Operations & Contact checks -------
    contact_url = (
        uni.career_center.contact_page_url
        if (uni.career_center and _nonempty(uni.career_center.contact_page_url))
        else None
    )

    # 17) Career center contact page link provided (critical)
    contact_link_provided = evaluator.add_custom_node(
        result=_nonempty(contact_url),
        id=f"u{index+1}_contact_link_provided",
        desc="Provides a link to the official career center contact page (or official page containing the contact information).",
        parent=uni_node,
        critical=True
    )

    # 18) Career center operating hours publicly listed (critical)
    hours_public_leaf = evaluator.add_leaf(
        id=f"u{index+1}_hours_public",
        desc="Career center operating hours are publicly listed on an official page.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page lists the career center's operating hours.",
        node=hours_public_leaf,
        sources=[contact_url] if contact_url else None,
        additional_instruction=(
            "Accept an hours section, business hours, or open office hours on the career center page. "
            "If hours are not present on the provided official page, mark as not supported."
        ),
    )

    # 19) Career center contact info publicly listed (critical)
    contact_info_public_leaf = evaluator.add_leaf(
        id=f"u{index+1}_contact_info_public",
        desc="Career center contact information (email, phone, or physical address) is publicly listed on an official page.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This page lists the career center's contact information, such as an email address, phone number, or physical address."
        ),
        node=contact_info_public_leaf,
        sources=[contact_url] if contact_url else None,
        additional_instruction=(
            "At least one of email, phone, or physical address must be present. If none are present on the provided official page, "
            "mark as not supported."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Spring 2026 career services university task.
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
        default_model=model,
    )

    # Extract up to first three universities and their info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Keep at most three; pad to three entries if fewer provided
    universities: List[UniversityEntry] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityEntry())

    # Top-level node: sequential gating to enforce count, then evaluate items
    main_node = evaluator.add_sequential(
        id="three_qualifying_universities",
        desc="Provide three U.S. universities with required Spring 2026 career-services evidence and official links.",
        parent=root,
        critical=False  # Keep non-critical to allow detailed breakdown even if gating fails
    )

    # Gate: Exactly three provided (considering only the first three; extras ignored)
    # We consider the evaluation to focus on the first three per instruction. Pass if at least three distinct names available.
    distinct_names = [u.name.strip() for u in universities if _nonempty(u.name)]
    exactly_three_ok = len(distinct_names) >= 3
    evaluator.add_custom_node(
        result=exactly_three_ok,
        id="exactly_three_universities_provided",
        desc="Response provides at least three distinct university entries (extras beyond three are ignored).",
        parent=main_node,
        critical=True
    )

    # Parallel block for three universities (evaluated only if gate passes; otherwise skipped by sequential logic)
    unis_block = evaluator.add_parallel(
        id="universities_block",
        desc="Verification for University 1, 2, and 3 (parallel).",
        parent=main_node,
        critical=False
    )

    # Build per-university subtrees
    for idx in range(3):
        await verify_university(evaluator, unis_block, universities[idx], idx)

    # Return evaluation summary
    return evaluator.get_summary()