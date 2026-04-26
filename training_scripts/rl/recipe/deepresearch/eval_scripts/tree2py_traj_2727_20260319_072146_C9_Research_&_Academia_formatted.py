import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_universities_eclipse_2026"
TASK_DESCRIPTION = """
Identify three universities in the United States that meet all of the following criteria: (1) The university has an astronomy department that offers a PhD program in astronomy, astrophysics, or a closely related field. (2) The university hosted a public viewing event specifically for the total lunar eclipse that occurred on March 3, 2026, and this event was scheduled to include at least part of the totality phase (which occurred in the early morning hours of March 3, 2026, approximately between 3:00-6:30 AM across various North American time zones). (3) The university has on-campus astronomical observation infrastructure—specifically a planetarium or an observatory (or both)—that is operated by or officially affiliated with the university and is used for public outreach or educational purposes. (4) The university is located in a region of the United States where the March 3, 2026 total lunar eclipse was visible, with particular preference for locations in western North America where the eclipse was best observed. (5) All of the above information (PhD program, eclipse event, and infrastructure) must be documented through publicly accessible official sources such as university websites, official social media accounts, university event calendars, or established news outlets. For each university, provide: the name of the university, the name of the astronomy/astrophysics department, a direct link to the department's webpage showing the PhD program, the name of the planetarium or observatory, a direct link to the planetarium/observatory webpage, details about the March 3, 2026 eclipse viewing event (event name, date/time, location), a direct link to the eclipse event announcement or documentation, and the city and state where the university is located.
"""

TOTALITY_WINDOW_NOTE = "The totality phase was in the early morning of March 3, 2026, approximately 3:00–6:30 AM across North American time zones."

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Identity & location
    university_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    university_homepage_url: Optional[str] = None
    location_url: Optional[str] = None  # any official page that states the city/state

    # Department / program
    department_name: Optional[str] = None
    department_url: Optional[str] = None
    phd_program_url: Optional[str] = None
    phd_program_statement: Optional[str] = None  # excerpt if provided in the answer

    # Faculty (any example astronomy/astrophysics faculty)
    faculty_example: Optional[str] = None
    faculty_url: Optional[str] = None

    # Infrastructure (planetarium / observatory)
    facility_name: Optional[str] = None
    facility_type: Optional[str] = None  # "planetarium" or "observatory" or similar
    facility_url: Optional[str] = None
    facility_affiliation_statement: Optional[str] = None
    facility_outreach_statement: Optional[str] = None

    # Eclipse event
    event_name: Optional[str] = None
    event_date: Optional[str] = None  # e.g., "March 2–3, 2026"
    event_time_range: Optional[str] = None  # e.g., "3:00–5:00 AM PT"
    event_location: Optional[str] = None  # on-campus location
    event_url: Optional[str] = None
    event_alt_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to the first three universities that the answer claims satisfy ALL requested criteria.
For each university, extract the following fields exactly as presented in the answer. If a field is missing, set it to null (or [] for arrays). Extract only explicit URLs shown in the answer.

Return JSON of the form:
{
  "universities": [
    {
      "university_name": str|null,
      "city": str|null,
      "state": str|null,
      "university_homepage_url": str|null,
      "location_url": str|null,

      "department_name": str|null,
      "department_url": str|null,
      "phd_program_url": str|null,
      "phd_program_statement": str|null,

      "faculty_example": str|null,
      "faculty_url": str|null,

      "facility_name": str|null,
      "facility_type": str|null,
      "facility_url": str|null,
      "facility_affiliation_statement": str|null,
      "facility_outreach_statement": str|null,

      "event_name": str|null,
      "event_date": str|null,
      "event_time_range": str|null,
      "event_location": str|null,
      "event_url": str|null,
      "event_alt_urls": [str, ...]
    }
  ]
}

Guidance:
- department_url: the department’s main page (Astronomy/Astrophysics/Physics).
- phd_program_url: the specific page that shows a PhD program in astronomy/astrophysics or closely related field.
- faculty_url: a page demonstrating at least one astronomy/astrophysics faculty member (e.g., people/faculty page or a faculty profile).
- facility_url: the page for the planetarium/observatory affiliated with the university.
- event_url and event_alt_urls: pages documenting the March 3, 2026 lunar eclipse viewing event (official site, official social media, university calendar, or established news).
- location_url: an official page that states the university’s city and state (can be the university homepage, an About page, or similar).
- Only include URLs explicitly present in the answer. If the answer mentions a source but does not provide a URL, leave the corresponding URL field as null.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nz(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())

def _collect_urls(*args: Optional[str], extra: Optional[List[str]] = None) -> List[str]:
    urls: List[str] = []
    for x in args:
        if _nz(x):
            urls.append(str(x).strip())
    if extra:
        for x in extra:
            if _nz(x):
                urls.append(str(x).strip())
    # de-duplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

def _domain_type(url: str) -> str:
    """
    Coarse source-type classifier used for 'Multiple_Source_Verification'.
    Categories: edu_university, social, news_media, gov, other
    """
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return "other"

    if any(k in netloc for k in ["facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com", "eventbrite.com", "linkedin.com"]):
        return "social"
    if netloc.endswith(".edu") or "university" in netloc or "college" in netloc:
        return "edu_university"
    if netloc.endswith(".gov") or "nasa.gov" in netloc or "noaa.gov" in netloc:
        return "gov"
    # naive heuristic for news media; not exhaustive
    if any(k in netloc for k in ["news", "times", "chronicle", "daily", "tribune", "guardian", "wsj.com", "reuters.com", "apnews.com", "npr.org", "bbc.com"]):
        return "news_media"
    return "other"

ADD_INS_MISSING_SOURCES_FAIL = "If no valid webpage URL is provided, you must judge the claim as NOT supported."
ADD_INS_GENERAL = f"{ADD_INS_MISSING_SOURCES_FAIL} Allow reasonable wording variants (e.g., 'Doctor of Philosophy' vs 'PhD'). Focus on whether the provided webpage(s) explicitly support the claim."

# --------------------------------------------------------------------------- #
# Verification builders per university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, parent_node, u: UniversityItem, idx: int) -> None:
    """
    Build and verify the full rubric subtree for one university.
    """
    univ_node = evaluator.add_parallel(
        id=f"univ_{idx}",
        desc=f"University #{idx + 1} verification",
        parent=parent_node,
        critical=False
    )

    # Pre-compute commonly used sources
    dept_sources = _collect_urls(u.department_url, u.phd_program_url)
    program_sources = _collect_urls(u.phd_program_url)
    faculty_sources = _collect_urls(u.faculty_url, u.department_url, u.university_homepage_url)
    facility_sources = _collect_urls(u.facility_url)
    event_sources = _collect_urls(u.event_url, extra=u.event_alt_urls)
    location_sources = _collect_urls(u.location_url, u.university_homepage_url, u.department_url)

    # --------------------- PhD_Program_Verification --------------------- #
    phd_node = evaluator.add_parallel(
        id=f"univ_{idx}_phd_program",
        desc="Verification that the university offers a PhD program in astronomy or astrophysics",
        parent=univ_node,
        critical=True
    )

    # Department_Existence
    dept_exist_node = evaluator.add_parallel(
        id=f"univ_{idx}_dept_existence",
        desc="The university has an astronomy/astrophysics/physics department offering astronomy/astrophysics programs",
        parent=phd_node,
        critical=True
    )

    # Astronomy_Department_Listed
    dep_listed_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_astronomy_dept_listed",
        desc="The department is listed as offering astronomy or astrophysics programs on an official page",
        parent=dept_exist_node,
        critical=True
    )
    claim_dep_listed = (
        f"This page shows that {u.university_name or 'the university'} has a department/unit that offers "
        f"astronomy or astrophysics programs (undergraduate or graduate)."
    )
    await evaluator.verify(
        claim=claim_dep_listed,
        node=dep_listed_leaf,
        sources=dept_sources if dept_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # Department_URL_Reference
    dep_url_ref_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_dept_url_ref",
        desc="URL reference to the department's official page",
        parent=dept_exist_node,
        critical=True
    )
    claim_dep_url_ref = (
        f"This page is an official page for the {u.department_name or 'astronomy/astrophysics/physics department'} "
        f"at {u.university_name or 'the university'}."
    )
    await evaluator.verify(
        claim=claim_dep_url_ref,
        node=dep_url_ref_leaf,
        sources=dept_sources if dept_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # Doctoral_Degree_Offered
    doc_node = evaluator.add_parallel(
        id=f"univ_{idx}_doctoral_degree",
        desc="The department explicitly offers a PhD degree in astronomy/astrophysics or closely related field",
        parent=phd_node,
        critical=True
    )

    phd_listed_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_phd_listed",
        desc="PhD program is listed on an official page",
        parent=doc_node,
        critical=True
    )
    claim_phd_listed = (
        "This page explicitly shows that a PhD (Doctor of Philosophy) program in astronomy, astrophysics, "
        "or a closely related field is offered."
    )
    await evaluator.verify(
        claim=claim_phd_listed,
        node=phd_listed_leaf,
        sources=program_sources if program_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    program_url_ref_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_phd_url_ref",
        desc="URL reference to the PhD program’s official page",
        parent=doc_node,
        critical=True
    )
    claim_prog_url_ref = (
        "This page is an official university or department webpage describing the PhD program."
    )
    await evaluator.verify(
        claim=claim_prog_url_ref,
        node=program_url_ref_leaf,
        sources=program_sources if program_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # Faculty_Presence
    faculty_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_faculty_presence",
        desc="At least one verifiable astronomy/astrophysics faculty member exists",
        parent=phd_node,
        critical=True
    )
    claim_faculty = (
        "This page shows at least one faculty member conducting research in astronomy or astrophysics."
    )
    await evaluator.verify(
        claim=claim_faculty,
        node=faculty_leaf,
        sources=faculty_sources if faculty_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # --------------------- Eclipse_Event_Verification ------------------- #
    eclipse_node = evaluator.add_parallel(
        id=f"univ_{idx}_eclipse_event",
        desc="University hosted a public viewing event for the March 3, 2026 total lunar eclipse",
        parent=univ_node,
        critical=True
    )

    # Event_Announcement
    event_ann_node = evaluator.add_parallel(
        id=f"univ_{idx}_event_announcement",
        desc="The university announced/hosted a public event specifically for the March 3, 2026 lunar eclipse",
        parent=eclipse_node,
        critical=True
    )

    event_doc_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_event_documented",
        desc="Event documented with specific date March 2–3, 2026",
        parent=event_ann_node,
        critical=True
    )
    claim_event_doc = (
        "This page documents a public viewing event for the total lunar eclipse on March 3, 2026 "
        "(possibly starting late on March 2 and continuing into early March 3)."
    )
    await evaluator.verify(
        claim=claim_event_doc,
        node=event_doc_leaf,
        sources=event_sources if event_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    event_url_ref_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_event_url_ref",
        desc="URL reference to the eclipse event announcement/documentation (official or established outlet)",
        parent=event_ann_node,
        critical=True
    )
    claim_event_url_ref = (
        "This page is an official university source (e.g., .edu site or official university social media) "
        "or an established news outlet documenting the event."
    )
    await evaluator.verify(
        claim=claim_event_url_ref,
        node=event_url_ref_leaf,
        sources=event_sources if event_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # Totality_Phase_Coverage
    totality_node = evaluator.add_parallel(
        id=f"univ_{idx}_totality_coverage",
        desc="The viewing event was scheduled to include at least part of the totality phase",
        parent=eclipse_node,
        critical=True
    )

    timing_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_eclipse_timing_overlap",
        desc="Event timing overlaps with totality (≈3:00–6:30 AM local on March 3, 2026)",
        parent=totality_node,
        critical=True
    )
    claim_timing = (
        "This page shows the event schedule includes at least part of the totality phase in the early morning of "
        "March 3, 2026 (approximately 3:00–6:30 AM local time). If explicit times are not listed but the page "
        "explicitly mentions 'totality' or 'total lunar eclipse' viewing, consider it covered."
    )
    await evaluator.verify(
        claim=claim_timing,
        node=timing_leaf,
        sources=event_sources if event_sources else None,
        additional_instruction=ADD_INS_GENERAL + f" Treat this as supported if the page clearly indicates viewing of totality. {TOTALITY_WINDOW_NOTE}"
    )

    visibility_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_visibility_confirmation",
        desc="Event description indicates viewing of the total eclipse/blood moon (not just partial)",
        parent=totality_node,
        critical=True
    )
    claim_visibility = (
        "This page indicates that the viewing included the total eclipse (e.g., mentions 'totality' or 'blood moon'), "
        "not just partial phases."
    )
    await evaluator.verify(
        claim=claim_visibility,
        node=visibility_leaf,
        sources=event_sources if event_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # Public_Accessibility
    public_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_public_accessibility",
        desc="The event was open to the public (not restricted to students/faculty only)",
        parent=eclipse_node,
        critical=True
    )
    claim_public = "This page indicates that the event was open to the general public (community members welcome)."
    await evaluator.verify(
        claim=claim_public,
        node=public_leaf,
        sources=event_sources if event_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # --------------------- Infrastructure_Verification ------------------ #
    infra_node = evaluator.add_parallel(
        id=f"univ_{idx}_infrastructure",
        desc="University has on-campus planetarium or observatory used for public outreach/education",
        parent=univ_node,
        critical=True
    )

    facility_exist_node = evaluator.add_parallel(
        id=f"univ_{idx}_facility_existence",
        desc="The university has either a planetarium or an observatory (or both) on campus",
        parent=infra_node,
        critical=True
    )

    facility_type_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_facility_type_identified",
        desc="Facility is explicitly a planetarium or an observatory (or similar astronomical facility)",
        parent=facility_exist_node,
        critical=True
    )
    claim_facility_type = (
        "This page describes a facility that is a planetarium or an observatory affiliated with the university."
    )
    await evaluator.verify(
        claim=claim_facility_type,
        node=facility_type_leaf,
        sources=facility_sources if facility_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    facility_url_ref_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_facility_url_ref",
        desc="URL reference to the planetarium/observatory information (official page)",
        parent=facility_exist_node,
        critical=True
    )
    claim_facility_url_ref = "This page is the official webpage for the named facility at the university."
    await evaluator.verify(
        claim=claim_facility_url_ref,
        node=facility_url_ref_leaf,
        sources=facility_sources if facility_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    affiliation_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_facility_affiliation",
        desc="Facility is operated by or officially affiliated with the university",
        parent=infra_node,
        critical=True
    )
    claim_affiliation = "This page indicates the facility is operated by or officially affiliated with the university."
    await evaluator.verify(
        claim=claim_affiliation,
        node=affiliation_leaf,
        sources=facility_sources if facility_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    outreach_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_facility_outreach",
        desc="Facility is used for educational purposes or public outreach (e.g., public shows/tours/programs)",
        parent=infra_node,
        critical=True
    )
    claim_outreach = (
        "This page indicates the facility is used for public outreach and/or education (e.g., public shows, tours, "
        "school programs, student training)."
    )
    await evaluator.verify(
        claim=claim_outreach,
        node=outreach_leaf,
        sources=facility_sources if facility_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # --------------------- Location_Verification ------------------------ #
    loc_node = evaluator.add_sequential(
        id=f"univ_{idx}_location",
        desc="University located in a region where the March 3, 2026 total lunar eclipse was visible",
        parent=univ_node,
        critical=True
    )

    geo_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_geo_location",
        desc="The university is located in the United States (city, state match)",
        parent=loc_node,
        critical=True
    )
    claim_geo = (
        f"This page shows that {u.university_name or 'the university'} is located in "
        f"{(u.city or 'the listed city')}, {(u.state or 'the listed state')}, United States."
    )
    await evaluator.verify(
        claim=claim_geo,
        node=geo_leaf,
        sources=location_sources if location_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    vis_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_eclipse_visibility_region",
        desc="The university's location is in a region where the March 3, 2026 eclipse was visible",
        parent=loc_node,
        critical=True
    )
    claim_visibility_region = (
        "This page indicates that from this location the total lunar eclipse on March 3, 2026 was visible, "
        "or the university hosted/announced viewing on campus implying visibility at that locale."
    )
    await evaluator.verify(
        claim=claim_visibility_region,
        node=vis_leaf,
        sources=event_sources if event_sources else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # --------------------- Documentation_Verification ------------------- #
    # NOTE: Parent is critical; framework requires children under a critical parent to be critical as well.
    # Therefore we set both children to critical=True for consistency (even though JSON marked one non-critical).
    docs_node = evaluator.add_parallel(
        id=f"univ_{idx}_documentation",
        desc="All information is documented through reliable, publicly accessible sources",
        parent=univ_node,
        critical=True
    )

    official_sources_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_official_sources",
        desc="Information comes from official university websites/social or established news outlets",
        parent=docs_node,
        critical=True
    )
    claim_official_sources = (
        "This page is either an official university webpage (e.g., .edu or official department/facility page), "
        "an official university social media account, or an established news outlet reporting the information."
    )
    # We prioritize validating the event source here since department/facility officialness are already checked above
    official_sources_all = event_sources if event_sources else _collect_urls(u.department_url, u.phd_program_url, u.facility_url)
    await evaluator.verify(
        claim=claim_official_sources,
        node=official_sources_leaf,
        sources=official_sources_all if official_sources_all else None,
        additional_instruction=ADD_INS_GENERAL
    )

    # Multiple_Source_Verification (custom binary check)
    # At least two of (PhD program, eclipse event, infrastructure) from different source types
    types_present: List[str] = []
    if program_sources:
        types_present.append(_domain_type(program_sources[0]))
    if event_sources:
        types_present.append(_domain_type(event_sources[0]))
    if facility_sources:
        types_present.append(_domain_type(facility_sources[0]))
    distinct_types = len(set([t for t in types_present if t])) >= 2

    evaluator.add_custom_node(
        result=distinct_types,
        id=f"univ_{idx}_multiple_source_verification",
        desc="At least two categories (PhD program, eclipse event, infrastructure) are verifiable from different source types",
        parent=docs_node,
        critical=True  # see note above for critical consistency under a critical parent
    )

    # --------------------- Output_Completeness --------------------------- #
    out_node = evaluator.add_parallel(
        id=f"univ_{idx}_output_completeness",
        desc="All required output fields are provided in the solution",
        parent=univ_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nz(u.university_name),
        id=f"univ_{idx}_name_provided",
        desc="University name provided",
        parent=out_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nz(u.department_name),
        id=f"univ_{idx}_dept_name_provided",
        desc="Department name provided",
        parent=out_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nz(u.phd_program_url),
        id=f"univ_{idx}_dept_link_provided",
        desc="Direct link to department/PhD program page provided",
        parent=out_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nz(u.facility_name),
        id=f"univ_{idx}_facility_name_provided",
        desc="Facility (planetarium/observatory) name provided",
        parent=out_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nz(u.facility_url),
        id=f"univ_{idx}_facility_link_provided",
        desc="Facility webpage link provided",
        parent=out_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nz(u.event_name) and (_nz(u.event_date) or _nz(u.event_time_range)) and _nz(u.event_location),
        id=f"univ_{idx}_event_details_provided",
        desc="Event details (name, date/time, location) provided",
        parent=out_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nz(u.event_url) or (bool(u.event_alt_urls)),
        id=f"univ_{idx}_event_link_provided",
        desc="Link to eclipse event announcement/documentation provided",
        parent=out_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nz(u.city) and _nz(u.state),
        id=f"univ_{idx}_location_provided",
        desc="City and state provided",
        parent=out_node,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'US Universities hosting Mar 3, 2026 eclipse events with PhD programs and facilities' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Three universities evaluated independently
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Keep only the first 3; pad with placeholders if fewer than 3
    universities: List[UniversityItem] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Add a small note as custom info
    evaluator.add_custom_info(
        info={
            "totality_window_reference": TOTALITY_WINDOW_NOTE,
            "universities_count_extracted": len(extracted.universities),
        },
        info_type="context_note",
        info_name="eclipse_context"
    )

    # Build verification subtrees for each university
    for i in range(3):
        await verify_university(evaluator, root, universities[i], i)

    return evaluator.get_summary()