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
TASK_ID = "edu_leadership_bigeast_texas"
TASK_DESCRIPTION = """
In the field of educational leadership, identify two key individuals and institutions meeting specific criteria:

Part A: Identify a superintendent of a Texas school district who meets ALL of the following requirements:
1. Was officially appointed or began serving in their current superintendent role between January 1, 2022, and December 31, 2025 (inclusive)
2. Holds a doctoral degree (Ed.D. or Ph.D.)
3. The doctoral degree was earned from a university located in Texas

Provide: (a) the superintendent's full name, (b) the name of the Texas school district they lead, (c) the type of doctoral degree they hold, (d) the name of the Texas university that awarded their doctoral degree, and (e) the date they were appointed or began serving as superintendent.

Part B: Identify a university that meets ALL of the following requirements:
1. Is a current member of the Big East Conference (as of the 2024-2025 academic year)
2. Was one of the seven founding members of the Big East Conference in 1979
3. Was founded before the year 1900

Provide: (a) the university's name, (b) the year it was founded, (c) the full name of the current Director of Athletics & Recreation, and (d) the full name of the men's basketball head coach for the 2024-2025 season.

For all information provided, include URL references that support your answers.
"""

DATE_RANGE_START = "2022-01-01"
DATE_RANGE_END = "2025-12-31"


# --------------------------------------------------------------------------- #
# Extraction Data Models                                                      #
# --------------------------------------------------------------------------- #
class SuperintendentEntry(BaseModel):
    name: Optional[str] = None
    district: Optional[str] = None
    appointment_date: Optional[str] = None
    appointment_urls: List[str] = Field(default_factory=list)
    degree_type: Optional[str] = None           # e.g., "Ph.D.", "Ed.D.", "Doctor of Philosophy"
    degree_university: Optional[str] = None     # university name
    degree_urls: List[str] = Field(default_factory=list)      # URLs confirming the degree
    location_urls: List[str] = Field(default_factory=list)    # URLs confirming the university is in Texas


class BigEastEntry(BaseModel):
    university_name: Optional[str] = None
    founding_year: Optional[str] = None
    founding_year_urls: List[str] = Field(default_factory=list)
    membership_urls: List[str] = Field(default_factory=list)
    founding_member_urls: List[str] = Field(default_factory=list)
    ad_director_name: Optional[str] = None
    ad_director_urls: List[str] = Field(default_factory=list)
    coach_name: Optional[str] = None   # Men's basketball head coach (2024-2025)
    coach_urls: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    superintendent: Optional[SuperintendentEntry] = None
    university: Optional[BigEastEntry] = None


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return f"""
    Extract structured information for two parts (Part A: Texas school superintendent, Part B: Big East university) exactly as presented in the provided answer text.

    GENERAL URL RULES:
    - Only include URLs explicitly present in the answer text (including markdown link URLs).
    - Ignore malformed URLs. If multiple URLs are cited for the same fact, include all of them.
    - Do not fabricate URLs.

    PART A (Texas Superintendent):
    If the answer mentions a superintendent who satisfies the task, extract the first clearly identified candidate. Return:
      - superintendent.name: Full name of the superintendent.
      - superintendent.district: Full name of the Texas school district they lead.
      - superintendent.appointment_date: The specific appointment or start date string (any human-readable date format).
      - superintendent.appointment_urls: URLs that explicitly support the appointment/start date and role at that district.
      - superintendent.degree_type: The doctoral degree type (e.g., "Ph.D.", "Ed.D.", or spelled-out equivalents).
      - superintendent.degree_university: University that awarded the doctoral degree.
      - superintendent.degree_urls: URLs that explicitly support the doctoral degree (and ideally the type and university).
      - superintendent.location_urls: URLs that explicitly support that the degree-granting university is located in Texas.
    If any field is not present in the answer, set it to null (or empty list for URL arrays).

    PART B (Big East University):
    Extract the first university in the answer that fits the criteria and provide:
      - university.university_name: University name.
      - university.founding_year: The founding year as a string (e.g., "1789"). If a range or multi-part founding history is given, extract the primary year used by the answer.
      - university.founding_year_urls: URLs that support the founding year.
      - university.membership_urls: URLs that confirm current Big East membership (as of 2024-2025).
      - university.founding_member_urls: URLs that confirm it was one of the seven founding members in 1979.
      - university.ad_director_name: Full name of the current Director of Athletics & Recreation (or equivalent title).
      - university.ad_director_urls: URLs that support the AD & Recreation director identification.
      - university.coach_name: Full name of the men's basketball head coach for the 2024-2025 season.
      - university.coach_urls: URLs that support the men's basketball head coach identification.
    If any field is not present in the answer, set it to null (or empty list for URL arrays).

    IMPORTANT:
    - Do not infer or add any information that is not explicitly provided in the answer text.
    - If multiple candidates are listed, extract only the first one that appears complete.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str):
            us = u.strip()
            if us and (us.startswith("http://") or us.startswith("https://")):
                if us not in cleaned:
                    cleaned.append(us)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_texas_superintendent(evaluator: Evaluator, parent_node, sup: Optional[SuperintendentEntry]) -> None:
    sup = sup or SuperintendentEntry()

    # Part A main node (parallel; non-critical overall to allow partial credit)
    a_node = evaluator.add_parallel(
        id="texas_superintendent",
        desc="Identify Texas superintendent with doctoral degree from Texas university, appointed 2022-2025",
        parent=parent_node,
        critical=False
    )

    # Basic Identification (critical group)
    basic_node = evaluator.add_parallel(
        id="basic_identification",
        desc="Provide superintendent and district names",
        parent=a_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sup.name and sup.name.strip()),
        id="superintendent_name",
        desc="Provide full name of superintendent",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sup.district and sup.district.strip()),
        id="district_name",
        desc="Provide name of Texas school district",
        parent=basic_node,
        critical=True
    )

    # Appointment Verification (sequential, critical)
    ap_node = evaluator.add_sequential(
        id="appointment_verification",
        desc="Verify appointment January 1, 2022 - December 31, 2025",
        parent=a_node,
        critical=True
    )

    # 1) Appointment Documentation (parallel under sequential; make critical to satisfy parent)
    ap_doc = evaluator.add_parallel(
        id="appointment_documentation",
        desc="Provide appointment date and supporting URL",
        parent=ap_node,
        critical=True
    )
    # 1.a) Appointment Date present (critical)
    evaluator.add_custom_node(
        result=bool(sup.appointment_date and sup.appointment_date.strip()),
        id="appointment_date",
        desc="State specific appointment or start date",
        parent=ap_doc,
        critical=True
    )
    # 1.b) Appointment URL supports appointment (critical)
    ap_urls = _clean_urls(sup.appointment_urls)
    if ap_urls:
        ap_url_leaf = evaluator.add_leaf(
            id="appointment_url",
            desc="Provide URL supporting appointment date",
            parent=ap_doc,
            critical=True
        )
        date_phrase = f" on {sup.appointment_date}" if (sup.appointment_date and sup.appointment_date.strip()) else ""
        claim = f"{sup.name} was appointed or began serving as superintendent of {sup.district}{date_phrase}."
        await evaluator.verify(
            claim=claim,
            node=ap_url_leaf,
            sources=ap_urls,
            additional_instruction="Confirm the page explicitly states the person was appointed, named, approved, or began serving as superintendent for the specified district on the stated date (or a clearly equivalent phrasing)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="appointment_url",
            desc="Provide URL supporting appointment date",
            parent=ap_doc,
            critical=True
        )

    # 2) Date Range Check (critical; runs only if documentation step passes)
    date_range_leaf = evaluator.add_leaf(
        id="date_range_check",
        desc="Confirm date falls within Jan 2022 - Dec 2025",
        parent=ap_node,
        critical=True
    )
    dr_date = sup.appointment_date or ""
    claim = f"The appointment or start date '{dr_date}' falls within {DATE_RANGE_START} and {DATE_RANGE_END} inclusive."
    await evaluator.verify(
        claim=claim,
        node=date_range_leaf,
        additional_instruction="Accept common date formats (e.g., 'Jan 10, 2023', '2023-01-10', 'July 2024'). If only month/year is given, consider it in range if the month-year is between Jan 2022 and Dec 2025."
    )

    # Doctoral Education (parallel, critical)
    edu_node = evaluator.add_parallel(
        id="doctoral_education",
        desc="Verify doctoral degree from Texas university",
        parent=a_node,
        critical=True
    )

    # Degree Verification (parallel, critical)
    deg_ver = evaluator.add_parallel(
        id="degree_verification",
        desc="Verify doctoral degree with type and university",
        parent=edu_node,
        critical=True
    )

    # Degree Type (critical) – require Ed.D or Ph.D (or spelled equivalents)
    if sup.degree_type and sup.degree_type.strip():
        deg_type_leaf = evaluator.add_leaf(
            id="degree_type",
            desc="State degree type (Ed.D. or Ph.D.)",
            parent=deg_ver,
            critical=True
        )
        claim = f"The degree type '{sup.degree_type}' indicates a doctoral degree that is either an Ed.D. (Doctor of Education) or a Ph.D. (Doctor of Philosophy)."
        await evaluator.verify(
            claim=claim,
            node=deg_type_leaf,
            additional_instruction="Treat common variants as acceptable: 'PhD', 'Ph.D.', 'Doctor of Philosophy', 'EdD', 'Ed.D.', 'Doctor of Education'."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="degree_type",
            desc="State degree type (Ed.D. or Ph.D.)",
            parent=deg_ver,
            critical=True
        )

    # University Name present (critical)
    evaluator.add_custom_node(
        result=bool(sup.degree_university and sup.degree_university.strip()),
        id="university_name",
        desc="State name of university that granted the degree",
        parent=deg_ver,
        critical=True
    )

    # Degree URL supports degree claim (critical)
    deg_urls = _clean_urls(sup.degree_urls)
    if deg_urls:
        deg_url_leaf = evaluator.add_leaf(
            id="degree_url",
            desc="Provide URL confirming doctoral degree",
            parent=deg_ver,
            critical=True
        )
        # Build a robust claim even if type is missing
        if sup.degree_type and sup.degree_type.strip():
            claim = f"{sup.name} holds a {sup.degree_type} from {sup.degree_university}."
        else:
            claim = f"{sup.name} holds a doctoral degree from {sup.degree_university}."
        await evaluator.verify(
            claim=claim,
            node=deg_url_leaf,
            sources=deg_urls,
            additional_instruction="Page should clearly indicate that the person earned a doctoral degree (and, if available, the type) from the specified university."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="degree_url",
            desc="Provide URL confirming doctoral degree",
            parent=deg_ver,
            critical=True
        )

    # Texas Location Verification (sequential, critical)
    loc_node = evaluator.add_sequential(
        id="texas_location_verification",
        desc="Verify university location in Texas",
        parent=edu_node,
        critical=True
    )
    # Location URL present (critical)
    evaluator.add_custom_node(
        result=bool(_clean_urls(sup.location_urls)),
        id="location_url",
        desc="Provide URL confirming Texas location",
        parent=loc_node,
        critical=True
    )
    # Location confirmed (critical)
    loc_urls = _clean_urls(sup.location_urls)
    loc_confirm_leaf = evaluator.add_leaf(
        id="location_confirmation",
        desc="Confirm university is in Texas",
        parent=loc_node,
        critical=True
    )
    uni_name = sup.degree_university or ""
    await evaluator.verify(
        claim=f"{uni_name} is located in the state of Texas, United States.",
        node=loc_confirm_leaf,
        sources=loc_urls,
        additional_instruction="Accept authoritative sources (official university pages, Wikipedia infobox, state pages) that clearly place the institution in Texas."
    )


async def verify_big_east_university(evaluator: Evaluator, parent_node, uni: Optional[BigEastEntry]) -> None:
    uni = uni or BigEastEntry()

    b_node = evaluator.add_parallel(
        id="big_east_university",
        desc="Identify Big East founding member founded before 1900",
        parent=parent_node,
        critical=False
    )

    # University Identification (critical)
    evaluator.add_custom_node(
        result=bool(uni.university_name and uni.university_name.strip()),
        id="university_identification",
        desc="Provide university name meeting all criteria",
        parent=b_node,
        critical=True
    )

    # Conference Affiliation (parallel, critical)
    conf_node = evaluator.add_parallel(
        id="conference_affiliation",
        desc="Verify Big East membership and founding status",
        parent=b_node,
        critical=True
    )

    # Current Membership (sequential, critical)
    mem_node = evaluator.add_sequential(
        id="current_membership",
        desc="Verify current Big East membership (2024-2025)",
        parent=conf_node,
        critical=True
    )
    # Membership URL presence (critical)
    evaluator.add_custom_node(
        result=bool(_clean_urls(uni.membership_urls)),
        id="membership_url",
        desc="Provide URL confirming Big East membership",
        parent=mem_node,
        critical=True
    )
    # Membership Confirmation (critical)
    membership_leaf = evaluator.add_leaf(
        id="membership_confirmation",
        desc="Confirm university is on Big East member list",
        parent=mem_node,
        critical=True
    )
    mem_urls = _clean_urls(uni.membership_urls)
    await evaluator.verify(
        claim=f"As of the 2024-2025 academic year, {uni.university_name} is a member of the Big East Conference.",
        node=membership_leaf,
        sources=mem_urls,
        additional_instruction="Use official Big East or university sources, or reliable media/encyclopedic sources listing current Big East members for 2024–25."
    )

    # Founding Member Status (sequential, critical)
    fnd_node = evaluator.add_sequential(
        id="founding_member_status",
        desc="Verify founding member status (1979)",
        parent=conf_node,
        critical=True
    )
    # Founding Member URL present (critical)
    evaluator.add_custom_node(
        result=bool(_clean_urls(uni.founding_member_urls)),
        id="founding_member_url",
        desc="Provide URL confirming founding member status",
        parent=fnd_node,
        critical=True
    )
    # Founding Member Confirmation (critical)
    found_leaf = evaluator.add_leaf(
        id="founding_member_confirmation",
        desc="Confirm as one of seven 1979 founding members",
        parent=fnd_node,
        critical=True
    )
    fnd_urls = _clean_urls(uni.founding_member_urls)
    await evaluator.verify(
        claim=f"{uni.university_name} was one of the seven founding members of the Big East Conference in 1979.",
        node=found_leaf,
        sources=fnd_urls,
        additional_instruction="Accept sources that enumerate the seven founding institutions in 1979 and include this university."
    )

    # Historical Foundation (sequential, critical)
    hist_node = evaluator.add_sequential(
        id="historical_foundation",
        desc="Verify founding year before 1900",
        parent=b_node,
        critical=True
    )
    # Founding Year Documentation (parallel, critical)
    fy_doc = evaluator.add_parallel(
        id="founding_year_documentation",
        desc="Provide founding year and supporting URL",
        parent=hist_node,
        critical=True
    )
    # Founding Year present (critical)
    evaluator.add_custom_node(
        result=bool(uni.founding_year and uni.founding_year.strip()),
        id="founding_year",
        desc="State specific founding year",
        parent=fy_doc,
        critical=True
    )
    # Founding Year URL supports year (critical)
    fy_urls = _clean_urls(uni.founding_year_urls)
    if fy_urls:
        fy_url_leaf = evaluator.add_leaf(
            id="founding_year_url",
            desc="Provide URL supporting founding year",
            parent=fy_doc,
            critical=True
        )
        claim = f"{uni.university_name} was founded in {uni.founding_year}."
        await evaluator.verify(
            claim=claim,
            node=fy_url_leaf,
            sources=fy_urls,
            additional_instruction="Accept pages that explicitly state the university's founding year (institutional history pages, reliable encyclopedias, etc.)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="founding_year_url",
            desc="Provide URL supporting founding year",
            parent=fy_doc,
            critical=True
        )

    # Pre-1900 Confirmation (critical)
    pre1900_leaf = evaluator.add_leaf(
        id="pre1900_confirmation",
        desc="Confirm year is before 1900",
        parent=hist_node,
        critical=True
    )
    fy_str = uni.founding_year or ""
    await evaluator.verify(
        claim=f"The year '{fy_str}' is before 1900.",
        node=pre1900_leaf,
        additional_instruction="Treat the statement as true only if the numeric year is strictly less than 1900."
    )

    # Athletic Leadership (parallel, critical)
    lead_node = evaluator.add_parallel(
        id="athletic_leadership",
        desc="Provide athletic department leadership names",
        parent=b_node,
        critical=True
    )

    # Athletic Director Info (parallel, critical)
    ad_node = evaluator.add_parallel(
        id="athletic_director_info",
        desc="Provide athletic director name and URL",
        parent=lead_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.ad_director_name and uni.ad_director_name.strip()),
        id="director_name",
        desc="State full name of Director of Athletics & Recreation",
        parent=ad_node,
        critical=True
    )
    ad_urls = _clean_urls(uni.ad_director_urls)
    if ad_urls:
        dir_leaf = evaluator.add_leaf(
            id="director_url",
            desc="Provide URL for athletic director",
            parent=ad_node,
            critical=True
        )
        claim = f"The current Director of Athletics & Recreation (or equivalent 'Athletic Director' title) at {uni.university_name} is {uni.ad_director_name}."
        await evaluator.verify(
            claim=claim,
            node=dir_leaf,
            sources=ad_urls,
            additional_instruction="Allow reasonable title variants like 'Athletic Director', 'Vice President/Director of Athletics', or similar."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="director_url",
            desc="Provide URL for athletic director",
            parent=ad_node,
            critical=True
        )

    # Basketball Coach Info (parallel, critical)
    coach_node = evaluator.add_parallel(
        id="basketball_coach_info",
        desc="Provide basketball coach name and URL (2024-2025)",
        parent=lead_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.coach_name and uni.coach_name.strip()),
        id="coach_name",
        desc="State full name of men's basketball head coach",
        parent=coach_node,
        critical=True
    )
    coach_urls = _clean_urls(uni.coach_urls)
    if coach_urls:
        coach_leaf = evaluator.add_leaf(
            id="coach_url",
            desc="Provide URL for basketball coach",
            parent=coach_node,
            critical=True
        )
        claim = f"The men's basketball head coach for the 2024-2025 season at {uni.university_name} is {uni.coach_name}."
        await evaluator.verify(
            claim=claim,
            node=coach_leaf,
            sources=coach_urls,
            additional_instruction="Confirm that the page indicates this person is (or was) the men's basketball head coach specifically for the 2024–25 season (or clearly current for that season)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="coach_url",
            desc="Provide URL for basketball coach",
            parent=coach_node,
            critical=True
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
    # Initialize evaluator with PARALLEL root (two independent parts)
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
        default_model=model
    )

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=ExtractionResult,
        extraction_name="extracted_info"
    )

    # Build verification tree per rubric
    await verify_texas_superintendent(evaluator, root, extraction.superintendent)
    await verify_big_east_university(evaluator, root, extraction.university)

    # Record reference parameters used in evaluation as custom info
    evaluator.add_custom_info(
        info={
            "date_range_start": DATE_RANGE_START,
            "date_range_end": DATE_RANGE_END,
            "conference_season_checked": "2024-2025"
        },
        info_type="evaluation_parameters",
        info_name="parameters"
    )

    return evaluator.get_summary()