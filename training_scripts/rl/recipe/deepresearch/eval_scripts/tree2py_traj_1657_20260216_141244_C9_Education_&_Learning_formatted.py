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
TASK_ID = "ivy_ny_president_2026"
TASK_DESCRIPTION = (
    "Identify the name of the person who meets all of the following criteria as of January 31, 2026:\n"
    "1. The person was announced as the next president of an Ivy League university located in New York State between January 1, 2026, and January 31, 2026, with the appointment taking effect on July 1, 2026.\n"
    "2. The person holds a PhD from MIT in History and Social Study of Science and Technology (or its predecessor program name), received in 1999.\n"
    "3. The person holds a JD from Yale Law School and an AB (undergraduate degree) from Harvard University.\n"
    "4. The person served as a law professor at the University of Virginia School of Law for at least 5 years during the period 1998-2005.\n"
    "5. The person served as Dean of UCLA School of Law for at least 5 consecutive years, with the deanship beginning in August 2015 and ending in June 2022.\n"
    "6. Immediately before being appointed as university president, the person served as Chancellor of the University of Wisconsin-Madison, a Big Ten Conference institution, starting in 2022.\n"
    "7. The university where the person was appointed as president maintains a test-optional admissions policy for the 2025-2026 admissions cycle.\n"
    "8. The person's appointment was specifically announced on January 25, 2026.\n"
    "9. The person is one of three former law school deans who were appointed to university presidencies in January 2026, as reported by Reuters in a January 29, 2026 article.\n"
    "Provide the person's full name."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DegreeInfo(BaseModel):
    degree: Optional[str] = None
    program: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EmploymentInfo(BaseModel):
    role: Optional[str] = None
    institution: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AppointmentInfo(BaseModel):
    university: Optional[str] = None
    announcement_date: Optional[str] = None
    effective_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class UniversityPolicyInfo(BaseModel):
    ivy_league_urls: List[str] = Field(default_factory=list)
    ny_location_urls: List[str] = Field(default_factory=list)
    test_optional_policy_urls: List[str] = Field(default_factory=list)


class ReutersInfo(BaseModel):
    urls: List[str] = Field(default_factory=list)


class PersonExtraction(BaseModel):
    person_name: Optional[str] = None

    appointment: Optional[AppointmentInfo] = None

    mit_phd: Optional[DegreeInfo] = None
    yale_jd: Optional[DegreeInfo] = None
    harvard_ab: Optional[DegreeInfo] = None

    uva_professor: Optional[EmploymentInfo] = None
    ucla_dean: Optional[EmploymentInfo] = None
    uw_chancellor: Optional[EmploymentInfo] = None

    university_policies: Optional[UniversityPolicyInfo] = None
    uw_big_ten_urls: List[str] = Field(default_factory=list)

    reuters: Optional[ReutersInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person() -> str:
    return """
Extract structured information about the single person named as the university president candidate in the answer. Only extract what is explicitly present in the answer text. If any item is not mentioned, return null or an empty list for URLs.

Return a JSON object with these fields:

- person_name: The full name of the person identified as meeting the criteria.

- appointment: Object with
  - university: The name of the university where the person was appointed president.
  - announcement_date: Date string for the appointment announcement (as written in the answer).
  - effective_date: Date string for when the appointment takes effect (as written).
  - sources: Array of URLs that directly support the announcement and effective date details (press releases, official news, credible media, etc.).

- mit_phd: Object with
  - degree: Should be "PhD" if present.
  - program: Program name as written (e.g., "History and Social Study of Science and Technology" or "Science, Technology, and Society (STS)").
  - institution: Institution name; should be "Massachusetts Institute of Technology" or "MIT" if present.
  - year: Year the PhD was received (e.g., "1999").
  - sources: Array of URLs supporting the MIT PhD credential and year/program.

- yale_jd: Object with
  - degree: Should be "JD" or "J.D." if present.
  - institution: Should be "Yale Law School" if present.
  - year: Year if mentioned (otherwise null).
  - sources: Array of URLs supporting the JD credential.

- harvard_ab: Object with
  - degree: Should be "AB", "A.B.", or "BA" if presented that way in the answer.
  - institution: Should be "Harvard University" if present.
  - year: Year if mentioned (otherwise null).
  - sources: Array of URLs supporting the Harvard undergraduate credential.

- uva_professor: Object with
  - role: Title as written (e.g., "Professor of Law").
  - institution: Should be "University of Virginia School of Law" (or equivalent naming) if present.
  - start_year: Start year if given; otherwise null.
  - end_year: End year if given; otherwise null.
  - sources: Array of URLs supporting that the person served as a law professor at UVA Law and the approximate timeframe (1998-2005).

- ucla_dean: Object with
  - role: Should be "Dean of UCLA School of Law" if present.
  - start: Start month-year string (e.g., "August 2015") if present.
  - end: End month-year string (e.g., "June 2022") if present.
  - sources: Array of URLs supporting the UCLA deanship and its start/end dates.

- uw_chancellor: Object with
  - role: Should be "Chancellor" if present.
  - institution: Should be "University of Wisconsin-Madison" (or equivalent hyphenation) if present.
  - start_year: Start year if present (should be "2022" for this task if mentioned).
  - end_year: End year if mentioned; otherwise null.
  - sources: Array of URLs supporting the chancellorship and start year.

- university_policies: Object with
  - ivy_league_urls: URLs supporting that the appointed university is in the Ivy League.
  - ny_location_urls: URLs supporting that the appointed university is located in New York State.
  - test_optional_policy_urls: URLs supporting a test-optional admissions policy for the 2025-2026 cycle.

- uw_big_ten_urls: Array of URLs that explicitly confirm the University of Wisconsin-Madison is a Big Ten Conference institution.

- reuters: Object with
  - urls: URLs for a Reuters article (dated January 29, 2026) reporting that the person is one of three former law school deans appointed to university presidencies in January 2026.

Rules:
- Extract only URLs actually provided in the answer text. Do not invent URLs.
- Keep all strings exactly as written in the answer (do not normalize).
- If something is missing, use null (for strings) or [] (for URL arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _merge_sources(*args: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in args:
        if lst:
            merged.extend([u for u in lst if _non_empty_str(u)])
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for u in merged:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_academic_background(evaluator: Evaluator, parent_node, data: PersonExtraction) -> None:
    name = data.person_name or "the person"
    acad_node = evaluator.add_parallel(
        id="academic_background",
        desc="Verify the person's complete academic background meets all educational requirements",
        parent=parent_node,
        critical=False
    )

    # MIT PhD 1999 in History and Social Study of Science and Technology (or equivalent STS)
    phd_node = evaluator.add_parallel(
        id="phd_mit",
        desc="Confirm PhD from MIT in History and Social Study of Science and Technology received in 1999",
        parent=acad_node,
        critical=False
    )
    phd_sources = data.mit_phd.sources if data.mit_phd else []
    phd_exist_gate = evaluator.add_custom_node(
        result=_has_sources(phd_sources),
        id="phd_mit_exists_sources",
        desc="MIT PhD info has supporting sources",
        parent=phd_node,
        critical=True
    )
    phd_leaf = evaluator.add_leaf(
        id="phd_mit_verify",
        desc="Verify MIT PhD program and 1999 year",
        parent=phd_node,
        critical=True
    )
    phd_claim = (
        f"{name} holds a PhD from the Massachusetts Institute of Technology (MIT) in "
        f"History and Social Study of Science and Technology (or an equivalent MIT STS program name), "
        f"awarded in 1999."
    )
    await evaluator.verify(
        claim=phd_claim,
        node=phd_leaf,
        sources=phd_sources,
        additional_instruction=(
            "Accept program names such as 'History and Social Study of Science and Technology' or "
            "'Science, Technology, and Society (STS)' as equivalent if the page clearly indicates the MIT PhD "
            "in that field. Focus on confirming MIT, the PhD, and the 1999 year."
        )
    )
    evaluator.add_custom_node(
        result=_has_sources(phd_sources),
        id="phd_mit_references",
        desc="Provide URL references supporting MIT PhD credential",
        parent=phd_node,
        critical=False
    )

    # JD from Yale Law School
    jd_node = evaluator.add_parallel(
        id="jd_yale",
        desc="Confirm JD degree from Yale Law School",
        parent=acad_node,
        critical=False
    )
    jd_sources = data.yale_jd.sources if data.yale_jd else []
    evaluator.add_custom_node(
        result=_has_sources(jd_sources),
        id="jd_yale_exists_sources",
        desc="Yale JD info has supporting sources",
        parent=jd_node,
        critical=True
    )
    jd_leaf = evaluator.add_leaf(
        id="jd_yale_verify",
        desc="Verify Yale Law School JD",
        parent=jd_node,
        critical=True
    )
    jd_claim = f"{name} holds a Juris Doctor (JD) from Yale Law School."
    await evaluator.verify(
        claim=jd_claim,
        node=jd_leaf,
        sources=jd_sources,
        additional_instruction="Minor formatting variants like 'J.D.' are acceptable; confirm Yale Law School JD credential."
    )
    evaluator.add_custom_node(
        result=_has_sources(jd_sources),
        id="jd_yale_references",
        desc="Provide URL references supporting Yale JD credential",
        parent=jd_node,
        critical=False
    )

    # AB from Harvard University
    ab_node = evaluator.add_parallel(
        id="ab_harvard",
        desc="Confirm AB (undergraduate degree) from Harvard University",
        parent=acad_node,
        critical=False
    )
    ab_sources = data.harvard_ab.sources if data.harvard_ab else []
    evaluator.add_custom_node(
        result=_has_sources(ab_sources),
        id="ab_harvard_exists_sources",
        desc="Harvard AB info has supporting sources",
        parent=ab_node,
        critical=True
    )
    ab_leaf = evaluator.add_leaf(
        id="ab_harvard_verify",
        desc="Verify Harvard AB undergraduate degree",
        parent=ab_node,
        critical=True
    )
    ab_claim = f"{name} holds an AB (A.B., Bachelor of Arts) from Harvard University."
    await evaluator.verify(
        claim=ab_claim,
        node=ab_leaf,
        sources=ab_sources,
        additional_instruction="Allow 'AB', 'A.B.', or 'BA' as equivalent labels if the page clearly indicates a Harvard undergraduate degree."
    )
    evaluator.add_custom_node(
        result=_has_sources(ab_sources),
        id="ab_harvard_references",
        desc="Provide URL references supporting Harvard AB credential",
        parent=ab_node,
        critical=False
    )


async def verify_career_progression(evaluator: Evaluator, parent_node, data: PersonExtraction) -> None:
    name = data.person_name or "the person"
    career_node = evaluator.add_parallel(
        id="career_progression",
        desc="Verify the person's complete career trajectory meets all position and timeline requirements",
        parent=parent_node,
        critical=False
    )

    # UVA law professor for at least 5 years during 1998-2005
    uva_node = evaluator.add_parallel(
        id="law_professor_virginia",
        desc="Confirm service as law professor at University of Virginia School of Law for at least 5 years during 1998-2005",
        parent=career_node,
        critical=False
    )
    uva_sources = data.uva_professor.sources if data.uva_professor else []
    evaluator.add_custom_node(
        result=_has_sources(uva_sources),
        id="virginia_exists_sources",
        desc="UVA law professor info has supporting sources",
        parent=uva_node,
        critical=True
    )
    uva_leaf = evaluator.add_leaf(
        id="law_professor_virginia_verify",
        desc="Verify UVA law professor service and 5-year span within 1998-2005",
        parent=uva_node,
        critical=True
    )
    uva_claim = (
        f"{name} served as a law professor at the University of Virginia School of Law for at least five years "
        f"during the period 1998–2005."
    )
    await evaluator.verify(
        claim=uva_claim,
        node=uva_leaf,
        sources=uva_sources,
        additional_instruction=(
            "Confirm that the UVA Law faculty bio or credible sources indicate the person held a law professor "
            "role spanning at least five years within 1998–2005 (inclusive). The exact title may be "
            "Assistant/Associate/Full Professor of Law; those are acceptable."
        )
    )
    evaluator.add_custom_node(
        result=_has_sources(uva_sources),
        id="virginia_references",
        desc="Provide URL references supporting UVA law professor position",
        parent=uva_node,
        critical=False
    )

    # UCLA Law Dean, start Aug 2015 and end June 2022 (at least 5 consecutive years)
    ucla_node = evaluator.add_parallel(
        id="ucla_law_dean",
        desc="Confirm service as Dean of UCLA School of Law for at least 5 consecutive years",
        parent=career_node,
        critical=False
    )
    ucla_sources = data.ucla_dean.sources if data.ucla_dean else []
    evaluator.add_custom_node(
        result=_has_sources(ucla_sources),
        id="ucla_dean_exists_sources",
        desc="UCLA deanship info has supporting sources",
        parent=ucla_node,
        critical=True
    )
    ucla_start_leaf = evaluator.add_leaf(
        id="ucla_dean_start",
        desc="Verify deanship began in August 2015",
        parent=ucla_node,
        critical=True
    )
    ucla_start_claim = f"{name}'s service as Dean of UCLA School of Law began in August 2015."
    await evaluator.verify(
        claim=ucla_start_claim,
        node=ucla_start_leaf,
        sources=ucla_sources,
        additional_instruction="Confirm the start month as August 2015 from official UCLA announcements or equivalent credible sources."
    )
    evaluator.add_custom_node(
        result=_has_sources(ucla_sources),
        id="ucla_start_references",
        desc="Provide URL references supporting UCLA deanship start date",
        parent=ucla_node,
        critical=False
    )

    ucla_end_leaf = evaluator.add_leaf(
        id="ucla_dean_end",
        desc="Verify deanship ended in June 2022",
        parent=ucla_node,
        critical=True
    )
    ucla_end_claim = f"{name}'s service as Dean of UCLA School of Law ended in June 2022."
    await evaluator.verify(
        claim=ucla_end_claim,
        node=ucla_end_leaf,
        sources=ucla_sources,
        additional_instruction="Confirm the end month as June 2022 from official UCLA or other credible sources."
    )
    evaluator.add_custom_node(
        result=_has_sources(ucla_sources),
        id="ucla_end_references",
        desc="Provide URL references supporting UCLA deanship end date",
        parent=ucla_node,
        critical=False
    )

    # UW–Madison Chancellor, starting 2022; institution is UW–Madison; UW–Madison is Big Ten
    uw_node = evaluator.add_parallel(
        id="uw_madison_chancellor",
        desc="Confirm service as Chancellor at University of Wisconsin-Madison starting in 2022",
        parent=career_node,
        critical=False
    )
    uw_sources = data.uw_chancellor.sources if data.uw_chancellor else []
    evaluator.add_custom_node(
        result=_has_sources(uw_sources),
        id="uw_chancellor_exists_sources",
        desc="UW–Madison chancellorship info has supporting sources",
        parent=uw_node,
        critical=True
    )

    uw_start_leaf = evaluator.add_leaf(
        id="uw_start_year",
        desc="Verify chancellorship began in 2022",
        parent=uw_node,
        critical=True
    )
    uw_start_claim = f"{name} began serving as Chancellor of the University of Wisconsin–Madison in 2022."
    await evaluator.verify(
        claim=uw_start_claim,
        node=uw_start_leaf,
        sources=uw_sources,
        additional_instruction="Confirm that the start year for the UW–Madison chancellorship is 2022."
    )
    evaluator.add_custom_node(
        result=_has_sources(uw_sources),
        id="uw_start_references",
        desc="Provide URL references supporting UW-Madison chancellorship start",
        parent=uw_node,
        critical=False
    )

    uw_name_leaf = evaluator.add_leaf(
        id="uw_institution_name",
        desc="Confirm institution is University of Wisconsin-Madison",
        parent=uw_node,
        critical=True
    )
    uw_name_claim = "The chancellorship is at the University of Wisconsin–Madison (UW–Madison)."
    await evaluator.verify(
        claim=uw_name_claim,
        node=uw_name_leaf,
        sources=uw_sources,
        additional_instruction="Minor hyphenation variations are acceptable; confirm the institution is UW–Madison."
    )
    evaluator.add_custom_node(
        result=_has_sources(uw_sources),
        id="uw_name_references",
        desc="Provide URL references confirming institution name",
        parent=uw_node,
        critical=False
    )

    # Big Ten membership
    big_ten_sources = _merge_sources(data.uw_big_ten_urls, uw_sources)
    uw_big_ten_leaf = evaluator.add_leaf(
        id="uw_big_ten",
        desc="Verify University of Wisconsin-Madison is a Big Ten Conference institution",
        parent=uw_node,
        critical=True
    )
    uw_big_ten_claim = "The University of Wisconsin–Madison is a member of the Big Ten Conference."
    await evaluator.verify(
        claim=uw_big_ten_claim,
        node=uw_big_ten_leaf,
        sources=big_ten_sources,
        additional_instruction="Use official Big Ten sites, UW–Madison pages, or other authoritative sources to confirm membership."
    )
    evaluator.add_custom_node(
        result=_has_sources(big_ten_sources),
        id="uw_big_ten_references",
        desc="Provide URL references confirming Big Ten membership",
        parent=uw_node,
        critical=False
    )


async def verify_university_characteristics(evaluator: Evaluator, parent_node, data: PersonExtraction) -> None:
    uni_ctx = data.university_policies or UniversityPolicyInfo()
    university = (data.appointment.university if data.appointment else None) or "the university"

    uni_node = evaluator.add_parallel(
        id="university_characteristics",
        desc="Verify the university where appointed as president meets all institutional requirements",
        parent=parent_node,
        critical=False
    )

    # Ivy League status
    ivy_leaf = evaluator.add_leaf(
        id="ivy_league",
        desc="Confirm the university is an Ivy League institution",
        parent=uni_node,
        critical=True
    )
    ivy_claim = f"{university} is an Ivy League institution."
    await evaluator.verify(
        claim=ivy_claim,
        node=ivy_leaf,
        sources=uni_ctx.ivy_league_urls,
        additional_instruction="Use authoritative sources (Ivy League official site, the university page, or reputable references) to confirm Ivy League membership."
    )
    evaluator.add_custom_node(
        result=_has_sources(uni_ctx.ivy_league_urls),
        id="ivy_league_references",
        desc="Provide URL references confirming Ivy League status",
        parent=uni_node,
        critical=False
    )

    # New York State location
    ny_leaf = evaluator.add_leaf(
        id="new_york_location",
        desc="Confirm the university is located in New York State",
        parent=uni_node,
        critical=True
    )
    ny_claim = f"{university} is located in New York State."
    await evaluator.verify(
        claim=ny_claim,
        node=ny_leaf,
        sources=uni_ctx.ny_location_urls,
        additional_instruction="Confirm the main campus location is within New York State."
    )
    evaluator.add_custom_node(
        result=_has_sources(uni_ctx.ny_location_urls),
        id="ny_location_references",
        desc="Provide URL references confirming New York State location",
        parent=uni_node,
        critical=False
    )

    # Test-optional policy for 2025-2026
    test_opt_leaf = evaluator.add_leaf(
        id="test_optional_2025_26",
        desc="Verify the university has a test-optional admissions policy for the 2025-2026 admissions cycle",
        parent=uni_node,
        critical=True
    )
    test_opt_claim = f"For the 2025–2026 admissions cycle, {university} maintained a test-optional undergraduate admissions policy."
    await evaluator.verify(
        claim=test_opt_claim,
        node=test_opt_leaf,
        sources=uni_ctx.test_optional_policy_urls,
        additional_instruction="Use official admissions pages or reliable announcements clearly indicating test-optional policy for the 2025–2026 cycle."
    )
    evaluator.add_custom_node(
        result=_has_sources(uni_ctx.test_optional_policy_urls),
        id="test_optional_references",
        desc="Provide URL references confirming test-optional policy for 2025-2026",
        parent=uni_node,
        critical=False
    )


async def verify_appointment_details(evaluator: Evaluator, parent_node, data: PersonExtraction) -> None:
    name = data.person_name or "the person"
    appt = data.appointment or AppointmentInfo()
    app_node = evaluator.add_parallel(
        id="appointment_details",
        desc="Verify all details of the presidential appointment meet specified requirements",
        parent=parent_node,
        critical=False
    )

    # Announcement date: Jan 25, 2026
    ann_sources = appt.sources or []
    evaluator.add_custom_node(
        result=_has_sources(ann_sources),
        id="announcement_sources_exist",
        desc="Appointment announcement details have supporting sources",
        parent=app_node,
        critical=True
    )
    ann_leaf = evaluator.add_leaf(
        id="announcement_date",
        desc="Confirm appointment was announced on January 25, 2026",
        parent=app_node,
        critical=True
    )
    ann_claim = f"The appointment of {name} as president of {appt.university or 'the university'} was announced on January 25, 2026."
    await evaluator.verify(
        claim=ann_claim,
        node=ann_leaf,
        sources=ann_sources,
        additional_instruction="Verify the explicit announcement date is January 25, 2026 on an official or reputable page."
    )
    evaluator.add_custom_node(
        result=_has_sources(ann_sources),
        id="announcement_date_references",
        desc="Provide URL references confirming announcement date",
        parent=app_node,
        critical=False
    )

    # Effective date: July 1, 2026
    eff_leaf = evaluator.add_leaf(
        id="effective_date",
        desc="Confirm appointment takes effect on July 1, 2026",
        parent=app_node,
        critical=True
    )
    eff_claim = f"The appointment takes effect on July 1, 2026."
    await evaluator.verify(
        claim=eff_claim,
        node=eff_leaf,
        sources=ann_sources,
        additional_instruction="Confirm that the page clearly states the start/effective date is July 1, 2026."
    )
    evaluator.add_custom_node(
        result=_has_sources(ann_sources),
        id="effective_date_references",
        desc="Provide URL references confirming effective date",
        parent=app_node,
        critical=False
    )

    # Reuters cohort: three former law school deans appointed in January 2026 (Reuters 2026-01-29)
    reuters_urls = (data.reuters.urls if data.reuters else []) or []
    cohort_leaf = evaluator.add_leaf(
        id="january_2026_cohort",
        desc="Verify person is one of three former law school deans appointed to university presidencies in January 2026 as reported by Reuters",
        parent=app_node,
        critical=True
    )
    cohort_claim = (
        f"A Reuters article published on January 29, 2026 reports that {name} is one of three former law school deans "
        f"appointed to university presidencies in January 2026."
    )
    await evaluator.verify(
        claim=cohort_claim,
        node=cohort_leaf,
        sources=reuters_urls,
        additional_instruction="Verify the Reuters article date (January 29, 2026) and that it lists exactly three such former law school deans including the person."
    )
    evaluator.add_custom_node(
        result=_has_sources(reuters_urls),
        id="cohort_references",
        desc="Provide URL references to Reuters article or other sources confirming the three-dean cohort",
        parent=app_node,
        critical=False
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
    """
    Evaluate an answer for the Ivy League New York university president identification task.
    """
    # Initialize evaluator and root
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

    # Extract structured information from the answer
    extracted: PersonExtraction = await evaluator.extract(
        prompt=prompt_extract_person(),
        template_class=PersonExtraction,
        extraction_name="structured_person_info"
    )

    # Add an initial critical gate: the answer must provide the person's full name
    evaluator.add_custom_node(
        result=_non_empty_str(extracted.person_name),
        id="person_name_present",
        desc="The answer provides the person's full name",
        parent=root,
        critical=True
    )

    # Build and verify all rubrics
    await verify_academic_background(evaluator, root, extracted)
    await verify_career_progression(evaluator, root, extracted)
    await verify_university_characteristics(evaluator, root, extracted)
    await verify_appointment_details(evaluator, root, extracted)

    # Optional: add custom info about timing constraints (for transparency)
    evaluator.add_custom_info(
        {
            "timeframe": "Announcement in January 2026; effective date July 1, 2026",
            "location_requirement": "Ivy League university in New York State",
            "admissions_policy_window": "2025–2026 test-optional"
        },
        info_type="constraints_summary"
    )

    # Return evaluation summary
    return evaluator.get_summary()