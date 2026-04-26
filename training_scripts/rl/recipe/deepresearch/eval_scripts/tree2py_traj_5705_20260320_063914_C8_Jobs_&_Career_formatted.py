import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "northeast_career_universities"
TASK_DESCRIPTION = """
Identify four universities located in the Northeastern United States (Maine, New Hampshire, Vermont, Massachusetts, Rhode Island, Connecticut, New York, New Jersey, or Pennsylvania) that demonstrate comprehensive career development programs meeting the following criteria:

Experiential Learning & Outcomes:
- Reports an overall co-op or internship participation rate of at least 85% of graduates
- Reports a post-graduation employment or continuing education rate of at least 90% within 9 months of graduation
- Maintains partnerships with at least 1,000 employer organizations

Career Center Services & Resources:
- Holds at least 3 in-person career fairs per academic year
- Has a career center with at least 5 full-time equivalent (FTE) staff positions
- Offers career development workshops on multiple topics throughout the academic year
- Utilizes technology platforms for career services (such as online job boards, virtual appointments, or career management systems)

Student Support Programs:
- Offers financial support (stipends, scholarships, or funding) for unpaid or underpaid internships
- Offers a formal alumni-to-student mentorship program
- Actively implements NACE (National Association of Colleges and Employers) career readiness competencies through structured programs or initiatives
- Provides one-on-one career advising appointments (virtual or in-person) to students

For each university, provide the university name, its location (city and state), and reference URLs that verify the university meets all the specified criteria.
"""

NE_STATES = {
    "Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island",
    "Connecticut", "New York", "New Jersey", "Pennsylvania"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Experiential(BaseModel):
    coop_participation: Optional[str] = None
    employment_rate_9mo: Optional[str] = None
    employer_partnerships: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class Services(BaseModel):
    in_person_career_fairs_per_year: Optional[str] = None
    career_center_fte_staff: Optional[str] = None
    workshops_offered: Optional[str] = None
    technology_platforms: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class SupportPrograms(BaseModel):
    internship_financial_support: Optional[str] = None
    alumni_to_student_mentorship: Optional[str] = None
    nace_competencies_implementation: Optional[str] = None
    one_on_one_career_advising: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class UniversityItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    experiential: Experiential = Field(default_factory=Experiential)
    services: Services = Field(default_factory=Services)
    support: SupportPrograms = Field(default_factory=SupportPrograms)


class UniversityList(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four universities listed in the answer that purportedly meet the specified criteria.
    For each university, extract the following fields strictly from the answer content:

    - name: University name
    - city: City of the main campus (if provided)
    - state: State of the main campus (if provided)
    - location_urls: All URLs cited that support the location (about/contact page, Wikipedia, etc.). If none provided, return an empty list.

    experiential:
      - coop_participation: The stated overall co-op or internship participation rate (as text, e.g., "91% of graduates complete an internship or co-op").
      - employment_rate_9mo: The stated post-graduation employment or continuing education rate within 9 months of graduation (as text).
      - employer_partnerships: The stated size of the employer partner network (as text, e.g., "over 3,000 employer partners").
      - reference_urls: All URLs in the answer that support the experiential learning and outcomes data.

    services:
      - in_person_career_fairs_per_year: The stated count of in-person career fairs per academic year (as text, e.g., "4 in-person career fairs annually").
      - career_center_fte_staff: The stated count of FTE staff in the career center (as text).
      - workshops_offered: A textual description indicating recurring, multi-topic career workshops during the academic year (if present).
      - technology_platforms: Names of any career technology platforms mentioned (e.g., Handshake, Symplicity, 12Twenty, virtual appointment systems).
      - reference_urls: All URLs in the answer that support the services/resources information.

    support:
      - internship_financial_support: Text indicating stipends, scholarships, or funding for unpaid/underpaid internships.
      - alumni_to_student_mentorship: Text indicating a formal alumni-to-student mentorship program.
      - nace_competencies_implementation: Text indicating explicit implementation of NACE career readiness competencies.
      - one_on_one_career_advising: Text indicating availability of one-on-one career advising appointments (virtual or in-person).
      - reference_urls: All URLs in the answer that support the student support programs.

    Rules:
    - Only extract information explicitly present in the answer.
    - All URL fields should contain only URLs explicitly shown in the answer (including markdown links).
    - If any field is missing, set it to null (or empty list for URLs).
    - Return a JSON object with a single field "universities", which is an array of up to four UniversityItem objects in the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities and additional-instruction texts                           #
# --------------------------------------------------------------------------- #
def union_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def uni_display_name(uni: UniversityItem, idx: int) -> str:
    return uni.name or f"University #{idx + 1}"


ADD_INS_LOCATION = (
    "Focus only on verifying the institution's location (city/state or state). It is acceptable if the page confirms "
    "the state or city+state; do not require the page to mention 'Northeastern US'. Minor naming variations are acceptable."
)

ADD_INS_COOP = (
    "Verify that the page explicitly indicates an overall co-op or internship participation rate of at least 85% for "
    "students/graduates. Accept phrasing like '≥85%', 'over 85%', or 'the vast majority (≥85%)'. Prefer the most recent data."
)

ADD_INS_EMPLOYMENT = (
    "Verify that the page reports a post-graduation outcome rate (employed or continuing education) within 9 months of "
    "graduation of at least 90%. Accept synonyms like 'career outcomes rate' or 'outcomes within 6-12 months'; prefer the most recent year."
)

ADD_INS_PARTNERS = (
    "Verify that the page states the university maintains partnerships with at least 1,000 employer organizations. "
    "Accept variants such as 'employer partners', 'hiring partners', or 'employer network ≥ 1,000'."
)

ADD_INS_FAIRS = (
    "Verify that the university holds at least three in-person career fairs per academic year. Sum separate in-person fairs "
    "(e.g., Fall + Spring + specialized). Virtual-only fairs do not count."
)

ADD_INS_STAFF = (
    "Verify that the career center team has at least five full-time equivalent (FTE) staff. Accept statements like '5+ staff', "
    "'team of 8 advisors', or org charts listing ≥5 FTE."
)

ADD_INS_WORKSHOPS = (
    "Verify that the career center offers career development workshops on multiple topics throughout the academic year. "
    "Accept programming calendars or statements indicating recurring workshops across topics."
)

ADD_INS_TECH = (
    "Verify that the career center utilizes technology platforms for career services such as an online job board, "
    "virtual appointments, or a career management system (e.g., Handshake, Symplicity, 12Twenty). One or more platforms suffice."
)

ADD_INS_FIN_SUPPORT = (
    "Verify that the university offers financial support such as stipends, scholarships, or funding for unpaid or underpaid internships."
)

ADD_INS_MENTOR = (
    "Verify that the university runs a formal alumni-to-student mentorship program administered by the career center or alumni office."
)

ADD_INS_NACE = (
    "Verify that the university explicitly references and implements NACE (National Association of Colleges and Employers) "
    "career readiness competencies through structured programs, learning outcomes, or initiatives."
)

ADD_INS_ADVISING = (
    "Verify that the university provides one-on-one career advising or coaching appointments to students, in-person or virtual."
)


# --------------------------------------------------------------------------- #
# Verification helpers per university                                         #
# --------------------------------------------------------------------------- #
async def verify_geographic_location(evaluator: Evaluator, parent_node, uni: UniversityItem, idx: int) -> None:
    """
    Leaf: University is located in one of the Northeastern US states (by verifying its state/city via a cited URL).
    """
    node = evaluator.add_leaf(
        id=f"univ_{idx}_geographic_location",
        desc="University is located in one of the Northeastern US states: Maine, New Hampshire, Vermont, Massachusetts, Rhode Island, Connecticut, New York, New Jersey, or Pennsylvania",
        parent=parent_node,
        critical=True,
    )

    name = uni_display_name(uni, idx)
    if uni.city and uni.state:
        claim = f"The university {name} is located in {uni.city}, {uni.state}."
    elif uni.state:
        claim = f"The university {name} is located in {uni.state}."
    else:
        claim = f"The university {name} has a main campus located in one of the Northeastern US states."

    # Prefer location URLs; otherwise fall back to any references provided
    location_sources = union_urls(
        uni.location_urls,
        uni.experiential.reference_urls,
        uni.services.reference_urls,
        uni.support.reference_urls,
    )

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=location_sources if location_sources else None,
        additional_instruction=ADD_INS_LOCATION,
    )


async def verify_experiential_group(evaluator: Evaluator, parent_node, uni: UniversityItem, idx: int) -> None:
    """
    Parallel critical group: Experiential Learning & Outcomes
    """
    group = evaluator.add_parallel(
        id=f"univ_{idx}_experiential_group",
        desc="University's experiential learning and employment outcomes meet specified thresholds",
        parent=parent_node,
        critical=True,
    )

    # Create leaves
    coop_node = evaluator.add_leaf(
        id=f"univ_{idx}_coop_participation",
        desc="University reports an overall co-op or internship participation rate of at least 85% of graduates",
        parent=group,
        critical=True,
    )
    employment_node = evaluator.add_leaf(
        id=f"univ_{idx}_employment_rate",
        desc="University reports post-graduation employment or continuing education rate of at least 90% within 9 months of graduation",
        parent=group,
        critical=True,
    )
    partners_node = evaluator.add_leaf(
        id=f"univ_{idx}_employer_partnerships",
        desc="University maintains partnerships with at least 1,000 employer organizations",
        parent=group,
        critical=True,
    )

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=len(uni.experiential.reference_urls) > 0,
        id=f"univ_{idx}_experiential_reference_url",
        desc="Provide reference URL(s) supporting the experiential learning and outcomes data",
        parent=group,
        critical=True,
    )

    # Build batch verifications
    name = uni_display_name(uni, idx)
    exp_sources = uni.experiential.reference_urls or union_urls(
        uni.location_urls, uni.services.reference_urls, uni.support.reference_urls
    )

    claims_and_sources = [
        (
            f"{name} reports an overall co-op or internship participation rate of at least 85% of graduates.",
            exp_sources if exp_sources else None,
            coop_node,
            ADD_INS_COOP,
        ),
        (
            f"{name} reports a post-graduation employment or continuing education rate within 9 months of graduation of at least 90%.",
            exp_sources if exp_sources else None,
            employment_node,
            ADD_INS_EMPLOYMENT,
        ),
        (
            f"{name} maintains partnerships with at least 1,000 employer organizations.",
            exp_sources if exp_sources else None,
            partners_node,
            ADD_INS_PARTNERS,
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_services_group(evaluator: Evaluator, parent_node, uni: UniversityItem, idx: int) -> None:
    """
    Parallel critical group: Career Center Services & Resources
    """
    group = evaluator.add_parallel(
        id=f"univ_{idx}_services_group",
        desc="University's career center services and resources meet specified standards",
        parent=parent_node,
        critical=True,
    )

    fairs_node = evaluator.add_leaf(
        id=f"univ_{idx}_career_fairs",
        desc="University holds at least 3 in-person career fairs per academic year",
        parent=group,
        critical=True,
    )
    staffing_node = evaluator.add_leaf(
        id=f"univ_{idx}_career_staffing",
        desc="University career center has at least 5 full-time equivalent (FTE) staff positions",
        parent=group,
        critical=True,
    )
    workshops_node = evaluator.add_leaf(
        id=f"univ_{idx}_career_workshops",
        desc="University offers career development workshops on multiple topics throughout the academic year",
        parent=group,
        critical=True,
    )
    technology_node = evaluator.add_leaf(
        id=f"univ_{idx}_career_technology",
        desc="Career center utilizes technology platforms for career services (e.g., online job boards, virtual appointments, career management systems)",
        parent=group,
        critical=True,
    )

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=len(uni.services.reference_urls) > 0,
        id=f"univ_{idx}_services_reference_url",
        desc="Provide reference URL(s) supporting the career center services data",
        parent=group,
        critical=True,
    )

    name = uni_display_name(uni, idx)
    services_sources = uni.services.reference_urls or union_urls(
        uni.location_urls, uni.experiential.reference_urls, uni.support.reference_urls
    )

    tech_list = ", ".join(uni.services.technology_platforms) if uni.services.technology_platforms else None
    tech_claim = (
        f"The career center at {name} utilizes technology platforms for career services"
        + (f", such as {tech_list}." if tech_list else ".")
    )

    claims_and_sources = [
        (
            f"{name} holds at least three in-person career fairs per academic year.",
            services_sources if services_sources else None,
            fairs_node,
            ADD_INS_FAIRS,
        ),
        (
            f"The career center at {name} has at least five full-time equivalent (FTE) staff positions.",
            services_sources if services_sources else None,
            staffing_node,
            ADD_INS_STAFF,
        ),
        (
            f"{name} offers career development workshops on multiple topics throughout the academic year.",
            services_sources if services_sources else None,
            workshops_node,
            ADD_INS_WORKSHOPS,
        ),
        (
            tech_claim,
            services_sources if services_sources else None,
            technology_node,
            ADD_INS_TECH,
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_support_group(evaluator: Evaluator, parent_node, uni: UniversityItem, idx: int) -> None:
    """
    Parallel critical group: Student Support Programs
    """
    group = evaluator.add_parallel(
        id=f"univ_{idx}_support_group",
        desc="University's student support programs for career development meet specified standards",
        parent=parent_node,
        critical=True,
    )

    fin_support_node = evaluator.add_leaf(
        id=f"univ_{idx}_internship_financial_support",
        desc="University offers financial support (stipends, scholarships, or funding) for unpaid or underpaid internships",
        parent=group,
        critical=True,
    )
    mentorship_node = evaluator.add_leaf(
        id=f"univ_{idx}_alumni_mentorship",
        desc="University offers a formal alumni-to-student mentorship program",
        parent=group,
        critical=True,
    )
    nace_node = evaluator.add_leaf(
        id=f"univ_{idx}_career_readiness",
        desc="University actively implements NACE career readiness competencies through structured programs or initiatives",
        parent=group,
        critical=True,
    )
    advising_node = evaluator.add_leaf(
        id=f"univ_{idx}_career_advising",
        desc="University provides one-on-one career advising appointments (virtual or in-person) to students",
        parent=group,
        critical=True,
    )

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=len(uni.support.reference_urls) > 0,
        id=f"univ_{idx}_support_reference_url",
        desc="Provide reference URL(s) supporting the student support programs data",
        parent=group,
        critical=True,
    )

    name = uni_display_name(uni, idx)
    support_sources = uni.support.reference_urls or union_urls(
        uni.location_urls, uni.experiential.reference_urls, uni.services.reference_urls
    )

    claims_and_sources = [
        (
            f"{name} offers financial support such as stipends, scholarships, or funding for unpaid or underpaid internships.",
            support_sources if support_sources else None,
            fin_support_node,
            ADD_INS_FIN_SUPPORT,
        ),
        (
            f"{name} runs a formal alumni-to-student mentorship program.",
            support_sources if support_sources else None,
            mentorship_node,
            ADD_INS_MENTOR,
        ),
        (
            f"{name} actively implements NACE career readiness competencies through structured programs or initiatives.",
            support_sources if support_sources else None,
            nace_node,
            ADD_INS_NACE,
        ),
        (
            f"{name} provides one-on-one career advising or coaching appointments to students, in-person or virtual.",
            support_sources if support_sources else None,
            advising_node,
            ADD_INS_ADVISING,
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_university(evaluator: Evaluator, root, uni: UniversityItem, idx: int) -> None:
    """
    Build the verification subtree for a single university.
    """
    uni_node = evaluator.add_parallel(
        id=f"univ_{idx}",
        desc=["First", "Second", "Third", "Fourth"][idx] + " university meeting all specified career development criteria",
        parent=root,
        critical=False,
    )

    # Geographic location (critical leaf)
    await verify_geographic_location(evaluator, uni_node, uni, idx)

    # Experiential Learning & Outcomes (critical group)
    await verify_experiential_group(evaluator, uni_node, uni, idx)

    # Career Center Services & Resources (critical group)
    await verify_services_group(evaluator, uni_node, uni, idx)

    # Student Support Programs (critical group)
    await verify_support_group(evaluator, uni_node, uni, idx)


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
    Evaluate an answer for the Northeastern US universities with comprehensive career development programs.
    """
    # Initialize evaluator
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

    # Extract structured university info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversityList,
        extraction_name="universities_extraction",
    )

    # Limit to first four universities; pad if fewer
    universities: List[UniversityItem] = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Record ground truth context (Northeastern state set)
    evaluator.add_ground_truth(
        {
            "northeastern_states": sorted(list(NE_STATES)),
            "requirement_summary": "All three critical groups (experiential, services, support) must be satisfied with URL evidence for each university."
        },
        gt_type="constraints",
    )

    # Build verification subtrees for each university
    for idx in range(4):
        await verify_university(evaluator, root, universities[idx], idx)

    # Return the structured summary (includes extraction and verification tree)
    return evaluator.get_summary()