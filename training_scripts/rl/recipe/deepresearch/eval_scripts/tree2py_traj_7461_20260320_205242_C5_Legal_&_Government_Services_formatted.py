import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sotu_2026_moh_recipient"
TASK_DESCRIPTION = (
    "During the longest State of the Union address in U.S. history, which took place in February 2026, the President "
    "awarded the Medal of Honor to a military service member. This individual is notable for holding both a bachelor's "
    "degree and a master's degree in strategic studies-related fields from the same university.\n\n"
    "Identify this service member and provide the following information:\n\n"
    "1. The individual's full name and complete military rank\n"
    "2. The specific military unit or regiment to which they belong\n"
    "3. The name of the university where they earned both degrees\n"
    "4. The exact titles of both degree programs (bachelor's and master's)\n"
    "5. Supporting URL references for the Medal of Honor award, military service, and educational credentials"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PersonInfo(BaseModel):
    full_name: Optional[str] = None
    complete_rank: Optional[str] = None
    active_duty_status: Optional[str] = None  # e.g., "active duty", "Active-Duty", True/False as text


class MilitaryInfo(BaseModel):
    unit_or_regiment: Optional[str] = None
    special_operations_unit_name: Optional[str] = None  # If same as unit, repeat; otherwise specify the SOF entity


class EducationInfo(BaseModel):
    university_name: Optional[str] = None
    college_or_school: Optional[str] = None  # internal college/school within the university
    bachelors_program_title: Optional[str] = None
    masters_program_title: Optional[str] = None
    bachelors_eligibility_requirement: Optional[str] = None  # text describing eligibility related to military service


class SupportingURLs(BaseModel):
    award_urls: List[str] = Field(default_factory=list)
    military_service_urls: List[str] = Field(default_factory=list)
    education_urls: List[str] = Field(default_factory=list)
    event_urls: List[str] = Field(default_factory=list)  # URLs covering SOTU specifics (date/length/record)


class RecipientExtraction(BaseModel):
    person: Optional[PersonInfo] = None
    military: Optional[MilitaryInfo] = None
    education: Optional[EducationInfo] = None
    urls: Optional[SupportingURLs] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_recipient_info() -> str:
    return """
    Extract the requested structured information about the Medal of Honor recipient mentioned in the answer.

    You must extract ONLY what is explicitly stated in the answer. Do not invent or infer facts not present.

    Return a JSON object with this structure:
    {
      "person": {
        "full_name": string|null,
        "complete_rank": string|null,
        "active_duty_status": string|null
      },
      "military": {
        "unit_or_regiment": string|null,
        "special_operations_unit_name": string|null
      },
      "education": {
        "university_name": string|null,
        "college_or_school": string|null,
        "bachelors_program_title": string|null,
        "masters_program_title": string|null,
        "bachelors_eligibility_requirement": string|null
      },
      "urls": {
        "award_urls": string[]  // pages that specifically support the Medal of Honor award and/or its SOTU context
        "military_service_urls": string[]  // pages that support unit/regiment and service details
        "education_urls": string[]  // pages that support the degrees, program titles, university, and college/school
        "event_urls": string[]  // pages that support SOTU 2026 date (Feb 24, 2026), its length (108 minutes), and/or record status
      }
    }

    Rules and clarifications:
    - complete_rank should include any modifiers (e.g., "Staff Sergeant", "Master Sergeant", "Sergeant First Class", "U.S. Army Captain").
    - active_duty_status should reflect what the answer says (e.g., "active duty"). If unclear or unspecified, return null.
    - unit_or_regiment must be the specific unit reference used in the answer (e.g., "75th Ranger Regiment", "1st Battalion, 75th Ranger Regiment", "Naval Special Warfare Development Group").
    - special_operations_unit_name should be provided if the answer explicitly names a SOF entity; if the same as unit_or_regiment, just repeat it; otherwise set null if not mentioned.
    - university_name must be the single university where BOTH degrees were earned (as per the answer).
    - college_or_school is the internal college/school (e.g., "College of Graduate and Continuing Studies") if stated.
    - bachelors_program_title and masters_program_title must be the exact program titles as written in the answer.
    - bachelors_eligibility_requirement should describe any special eligibility tied to military service (e.g., restricted to certain units), if present.
    - For all URL arrays: extract only valid URLs that are explicitly present in the answer text. Include markdown link targets if applicable.

    If any field is not found in the answer, set it to null (or [] for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


def _combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str) and u.strip():
                combined.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_supporting_urls_nodes(
    evaluator: Evaluator,
    parent,
    person: PersonInfo,
    military: MilitaryInfo,
    education: EducationInfo,
    urls: SupportingURLs,
):
    """
    Build 'SupportingURLs' node and verify each category has at least one URL that supports the corresponding claim.
    All leaves are critical under a critical parent.
    """
    sup_node = evaluator.add_parallel(
        id="SupportingURLs",
        desc="Provide publicly available URL references covering award, military service/unit, and educational credentials.",
        parent=parent,
        critical=True,
    )

    # 1) Award URL support
    award_leaf = evaluator.add_leaf(
        id="AwardURL",
        desc="Provide at least one publicly available URL supporting the Medal of Honor award (recipient and/or award context).",
        parent=sup_node,
        critical=True,
    )
    full_name = _safe(person.full_name)
    claim_award_support = (
        f"This page supports that {full_name} received the Medal of Honor in February 2026, "
        f"and/or documents the award context (e.g., State of the Union presentation)."
    )
    await evaluator.verify(
        claim=claim_award_support,
        node=award_leaf,
        sources=urls.award_urls,  # verify_by_urls will require non-empty; empty -> fail
        additional_instruction="Look for explicit mention of the named recipient and the Medal of Honor award details. "
                               "A White House transcript, DoD release, or reputable news page is acceptable."
    )

    # 2) Military service URL support
    mil_leaf = evaluator.add_leaf(
        id="MilitaryServiceURL",
        desc="Provide at least one publicly available URL supporting the recipient’s military service details, including unit/regiment.",
        parent=sup_node,
        critical=True,
    )
    unit = _safe(military.unit_or_regiment)
    claim_mil_support = f"This page supports the recipient's military service details, including their unit/regiment: '{unit}'."
    await evaluator.verify(
        claim=claim_mil_support,
        node=mil_leaf,
        sources=urls.military_service_urls,
        additional_instruction="Verify the page states or clearly implies the service member's unit/regiment or organization."
    )

    # 3) Education URL support
    edu_leaf = evaluator.add_leaf(
        id="EducationURL",
        desc="Provide at least one publicly available URL supporting the recipient’s educational credentials (degrees/program titles/university/college).",
        parent=sup_node,
        critical=True,
    )
    uni = _safe(education.university_name)
    b_title = _safe(education.bachelors_program_title)
    m_title = _safe(education.masters_program_title)
    college = _safe(education.college_or_school)
    claim_edu_support = (
        f"This page supports that the recipient earned both degrees at {uni}, including the bachelor's program '{b_title}' "
        f"and master's program '{m_title}' (and college/school '{college}' if provided)."
    )
    await evaluator.verify(
        claim=claim_edu_support,
        node=edu_leaf,
        sources=urls.education_urls,
        additional_instruction="Check the page for the university affiliation and the exact program titles; a university site, "
                               "official program page, or authoritative bio is preferred."
    )

    return {
        "award_leaf": award_leaf,
        "mil_leaf": mil_leaf,
        "edu_leaf": edu_leaf,
        "node": sup_node,
    }


async def verify_event_constraints(
    evaluator: Evaluator,
    parent,
    person: PersonInfo,
    urls: SupportingURLs,
):
    """
    Event constraints tied to the Medal of Honor award context:
    - Award occurred during SOTU on Feb 24, 2026.
    - 2026 SOTU lasted 108 minutes and is described as the longest on record.
    """
    ev_node = evaluator.add_parallel(
        id="EventConstraints",
        desc="Verify the State of the Union event constraints tied to the Medal of Honor award context.",
        parent=parent,
        critical=True,
    )

    # Award during Feb 24, 2026 SOTU
    award_during_leaf = evaluator.add_leaf(
        id="AwardDuringFeb24_2026SOTU",
        desc="Verify the Medal of Honor was awarded during the State of the Union address that occurred on February 24, 2026.",
        parent=ev_node,
        critical=True,
    )
    claim_award_during = (
        f"{_safe(person.full_name)} was awarded the Medal of Honor during the State of the Union address held on February 24, 2026."
    )
    await evaluator.verify(
        claim=claim_award_during,
        node=award_during_leaf,
        sources=_combine_sources(urls.award_urls, urls.event_urls),
        additional_instruction="Look for explicit mention that the President presented/awarded the Medal of Honor during the "
                               "Feb 24, 2026 State of the Union address."
    )

    # SOTU length and record
    length_leaf = evaluator.add_leaf(
        id="SOTULengthRecord",
        desc="Verify the 2026 State of the Union address lasted 108 minutes and is described as the longest on record.",
        parent=ev_node,
        critical=True,
    )
    claim_length = "The 2026 State of the Union address lasted 108 minutes and is described as the longest State of the Union on record."
    await evaluator.verify(
        claim=claim_length,
        node=length_leaf,
        sources=_combine_sources(urls.event_urls, urls.award_urls),
        additional_instruction="The page should clearly indicate a 108-minute duration and characterize it as the longest SOTU on record."
    )

    return ev_node


async def verify_person_identification(
    evaluator: Evaluator,
    parent,
    person: PersonInfo,
    urls: SupportingURLs,
):
    """
    Verify identity details: full name, complete rank, active-duty status.
    """
    p_node = evaluator.add_parallel(
        id="PersonIdentification",
        desc="Provide the recipient’s identity details.",
        parent=parent,
        critical=True,
    )

    # Full name
    name_leaf = evaluator.add_leaf(
        id="FullName",
        desc="Provide the individual’s full name.",
        parent=p_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Medal of Honor recipient's full name is '{_safe(person.full_name)}'.",
        node=name_leaf,
        sources=_combine_sources(urls.award_urls, urls.military_service_urls),
        additional_instruction="Confirm the page explicitly states the recipient's full name as provided."
    )

    # Complete rank
    rank_leaf = evaluator.add_leaf(
        id="CompleteRank",
        desc="Provide the individual’s complete military rank.",
        parent=p_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The recipient's complete military rank is '{_safe(person.complete_rank)}'.",
        node=rank_leaf,
        sources=_combine_sources(urls.award_urls, urls.military_service_urls),
        additional_instruction="Accept reasonable variants or abbreviations if they clearly refer to the same rank."
    )

    # Active-duty status
    active_leaf = evaluator.add_leaf(
        id="ActiveDutyStatus",
        desc="The individual is an active-duty military service member.",
        parent=p_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Medal of Honor recipient is an active-duty military service member.",
        node=active_leaf,
        sources=_combine_sources(urls.award_urls, urls.military_service_urls),
        additional_instruction="The page should indicate active-duty status or otherwise make clear the service member is on active duty."
    )

    return p_node


async def verify_military_service_details(
    evaluator: Evaluator,
    parent,
    military: MilitaryInfo,
    urls: SupportingURLs,
):
    """
    Verify unit/regiment and that it is a special operations unit.
    """
    m_node = evaluator.add_parallel(
        id="MilitaryServiceDetails",
        desc="Provide and verify the service member’s unit/regiment and special operations affiliation.",
        parent=parent,
        critical=True,
    )

    # Unit or regiment
    unit_leaf = evaluator.add_leaf(
        id="MilitaryUnitOrRegiment",
        desc="Identify the specific military unit or regiment to which they belong.",
        parent=m_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The recipient belongs to the military unit/regiment '{_safe(military.unit_or_regiment)}'.",
        node=unit_leaf,
        sources=_combine_sources(urls.military_service_urls, urls.award_urls),
        additional_instruction="Names may appear with battalion/company qualifiers. Consider reasonable formatting variants equivalent."
    )

    # Special operations unit verification
    sof_leaf = evaluator.add_leaf(
        id="SpecialOperationsUnit",
        desc="Verify the identified unit is a special operations unit within the U.S. military.",
        parent=m_node,
        critical=True,
    )
    unit_name_for_sof = _safe(military.special_operations_unit_name) or _safe(military.unit_or_regiment)
    await evaluator.verify(
        claim=f"The unit/regiment '{unit_name_for_sof}' is a special operations unit within the U.S. military.",
        node=sof_leaf,
        sources=_combine_sources(urls.military_service_urls),
        additional_instruction="The page should clearly indicate that the named unit is part of U.S. special operations."
    )

    return m_node


async def verify_educational_credentials(
    evaluator: Evaluator,
    parent,
    education: EducationInfo,
    urls: SupportingURLs,
):
    """
    Verify university, same college/school, exact program titles, strategic-studies relation,
    and bachelor's eligibility requirement tied to military service.
    """
    e_node = evaluator.add_parallel(
        id="EducationalCredentials",
        desc="Provide and verify the individual’s bachelor’s and master’s degrees in strategic studies-related fields from the same university and the same internal college/school, including the bachelor’s eligibility requirement.",
        parent=parent,
        critical=True,
    )

    # University name - same for both degrees
    uni_leaf = evaluator.add_leaf(
        id="UniversityName",
        desc="Provide the name of the university where both degrees were earned (same university for both).",
        parent=e_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The recipient earned both the bachelor's and master's degrees at '{_safe(education.university_name)}'.",
        node=uni_leaf,
        sources=_combine_sources(urls.education_urls),
        additional_instruction="The page should indicate that both degrees were earned at the same university."
    )

    # Same college/school for both
    college_leaf = evaluator.add_leaf(
        id="SameCollegeOrSchool",
        desc="Identify the college/school within the university through which both degree programs were completed (same for both).",
        parent=e_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Both degrees were completed through the same internal college/school at '{_safe(education.university_name)}', specifically '{_safe(education.college_or_school)}'.",
        node=college_leaf,
        sources=_combine_sources(urls.education_urls),
        additional_instruction="The page should explicitly or clearly imply the same internal college/school granted both degrees."
    )

    # Bachelor's program title
    b_leaf = evaluator.add_leaf(
        id="BachelorsProgramTitle",
        desc="Provide the exact title of the bachelor’s degree program.",
        parent=e_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The exact title of the bachelor's degree program is '{_safe(education.bachelors_program_title)}'.",
        node=b_leaf,
        sources=_combine_sources(urls.education_urls),
        additional_instruction="Match exact program title text; minor punctuation/case differences are acceptable."
    )

    # Master's program title
    m_leaf = evaluator.add_leaf(
        id="MastersProgramTitle",
        desc="Provide the exact title of the master’s degree program.",
        parent=e_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The exact title of the master's degree program is '{_safe(education.masters_program_title)}'.",
        node=m_leaf,
        sources=_combine_sources(urls.education_urls),
        additional_instruction="Match exact program title text; minor punctuation/case differences are acceptable."
    )

    # Programs are strategic-studies-related
    strat_leaf = evaluator.add_leaf(
        id="ProgramsStrategicStudiesRelated",
        desc="Verify that both degree programs are in strategic studies-related fields.",
        parent=e_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Both degree programs ('{_safe(education.bachelors_program_title)}' and '{_safe(education.masters_program_title)}') are in strategic-studies-related fields.",
        node=strat_leaf,
        sources=_combine_sources(urls.education_urls),
        additional_instruction="Confirm that both programs are clearly related to strategic/defense/security studies or closely aligned strategic domains."
    )

    # Bachelor's eligibility requirement (tied to military service)
    elig_leaf = evaluator.add_leaf(
        id="BachelorsEligibilityRequirement",
        desc="State the bachelor’s program special eligibility requirements related to military service.",
        parent=e_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The bachelor's program has a special military service-related eligibility requirement: {_safe(education.bachelors_eligibility_requirement)}",
        node=elig_leaf,
        sources=_combine_sources(urls.education_urls),
        additional_instruction="Verify that the bachelor's program specifies eligibility tied to military status, unit, MOS, or similar."
    )

    return e_node


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
    Evaluate an answer for the 2026 SOTU Medal of Honor recipient identification and details task.
    """
    # 1) Initialize evaluator
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

    # 2) Extraction
    extracted: RecipientExtraction = await evaluator.extract(
        prompt=prompt_extract_recipient_info(),
        template_class=RecipientExtraction,
        extraction_name="recipient_extraction",
    )

    # Normalize sub-objects to avoid None checks downstream
    person = extracted.person or PersonInfo()
    military = extracted.military or MilitaryInfo()
    education = extracted.education or EducationInfo()
    urls = extracted.urls or SupportingURLs()

    # 3) Build main critical node mirroring rubric root
    main_node = evaluator.add_parallel(
        id="ServiceMemberIdentification",
        desc="Identify the service member who received the Medal of Honor during the record-long Feb 24, 2026 State of the Union address and provide required military, education, and source URL information.",
        parent=root,
        critical=True,
    )

    # 4) Build and verify SupportingURLs first (so that other checks can rely on sources)
    supporting_nodes = await build_supporting_urls_nodes(
        evaluator=evaluator,
        parent=main_node,
        person=person,
        military=military,
        education=education,
        urls=urls,
    )

    # 5) Event constraints
    await verify_event_constraints(
        evaluator=evaluator,
        parent=main_node,
        person=person,
        urls=urls,
    )

    # 6) Person identification
    await verify_person_identification(
        evaluator=evaluator,
        parent=main_node,
        person=person,
        urls=urls,
    )

    # 7) Military service details
    await verify_military_service_details(
        evaluator=evaluator,
        parent=main_node,
        military=military,
        urls=urls,
    )

    # 8) Educational credentials
    await verify_educational_credentials(
        evaluator=evaluator,
        parent=main_node,
        education=education,
        urls=urls,
    )

    # 9) Final summary
    return evaluator.get_summary()