import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_astronomy_within_60_miles_philadelphia"
TASK_DESCRIPTION = """
I'm a prospective undergraduate student interested in studying astronomy or astrophysics in Pennsylvania. I want to find universities within 60 miles of Philadelphia that offer undergraduate degree programs in astronomy or astrophysics. Please identify at least three such universities and provide the following information for each: (1) The official name of the university, (2) The name of the astronomy/astrophysics undergraduate program, (3) The degree type(s) offered (BA and/or BS), (4) The distance from Philadelphia to the university campus (in miles), (5) The total credit hours required to complete the astronomy/astrophysics major, (6) The annual undergraduate tuition for the 2025-2026 academic year, (7) Whether the university has an observatory and/or planetarium facility, (8) The public transportation options available from Philadelphia to the campus, (9) Whether first-year students are required to live on campus, (10) The prerequisite courses required for the astronomy/astrophysics major (such as calculus and physics), (11) Whether undergraduate research opportunities are available in astronomy/astrophysics, (12) The number of faculty members in the astronomy/astrophysics department, (13) Whether meal plans are required for first-year students, and (14) Reference URLs to official university pages for each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversitySourceURLs(BaseModel):
    university_url: Optional[str] = None
    program_url: Optional[str] = None
    degree_url: Optional[str] = None
    distance_url: Optional[str] = None
    credits_url: Optional[str] = None
    tuition_url: Optional[str] = None
    facilities_url: Optional[str] = None
    transportation_url: Optional[str] = None
    housing_url: Optional[str] = None
    prerequisites_url: Optional[str] = None
    research_url: Optional[str] = None
    faculty_url: Optional[str] = None
    dining_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    degree_types: List[str] = Field(default_factory=list)
    distance_miles: Optional[str] = None
    total_credits: Optional[str] = None
    annual_tuition_2025_2026: Optional[str] = None
    has_observatory_planetarium: Optional[str] = None
    public_transportation_options: Optional[str] = None
    first_year_housing_required: Optional[str] = None
    prerequisite_courses: Optional[str] = None
    research_opportunities_available: Optional[str] = None
    faculty_count: Optional[str] = None
    meal_plan_required_first_year: Optional[str] = None
    # Optional helpful field for location (city/state) if the answer provided it
    location_city_state: Optional[str] = None

    sources: Optional[UniversitySourceURLs] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to the first 5 Pennsylvania universities mentioned in the answer that are claimed to be within ~60 miles of Philadelphia and that offer undergraduate astronomy or astrophysics programs. For each university, extract the following fields exactly as presented in the answer:

- university_name: Official university name.
- program_name: Official name of the undergraduate astronomy/astrophysics program (e.g., "B.S. in Astrophysics", "Astronomy Major").
- degree_types: A list of degree types offered for the undergrad astronomy/astrophysics program, e.g., ["BS", "BA"]. Use standard abbreviations BA/BS when possible.
- distance_miles: The distance in miles from Philadelphia to the university campus as stated in the answer (keep it as text exactly as provided, e.g., "28 miles", "~45 mi", "about 32").
- total_credits: Total credit hours required to complete the astronomy/astrophysics major as stated in the answer.
- annual_tuition_2025_2026: The undergraduate annual tuition for the 2025–2026 academic year as stated in the answer. If not specified for 2025–2026, extract what is provided (e.g., "2025-26 tuition TBD; 2024-25: $XX,XXX").
- has_observatory_planetarium: Whether an observatory and/or planetarium is available (e.g., "observatory", "planetarium", "both", "none"). Use the description provided in the answer.
- public_transportation_options: Summary of public transit options from Philadelphia to campus as stated (e.g., "SEPTA regional rail to X station + campus shuttle").
- first_year_housing_required: Whether first-year students are required to live on campus (e.g., "required", "not required", "varies").
- prerequisite_courses: The prerequisite courses for the astronomy/astrophysics major (e.g., "Calculus I–III, Physics I–II").
- research_opportunities_available: Whether undergraduate research opportunities are available in astronomy/astrophysics (e.g., "yes", "no", brief description).
- faculty_count: The number of astronomy/astrophysics faculty (extract as a string exactly as shown; do not convert to a number).
- meal_plan_required_first_year: Whether a meal plan is required for first-year students (e.g., "required", "optional").
- location_city_state: If the answer provides city/state, extract it.

- sources: Provide URLs explicitly cited in the answer for each piece of information when available. Do not invent any URLs. Use the following mapping fields when the answer provides them:
  - university_url
  - program_url
  - degree_url
  - distance_url
  - credits_url
  - tuition_url
  - facilities_url
  - transportation_url
  - housing_url
  - prerequisites_url
  - research_url
  - faculty_url
  - dining_url
  - additional_urls (array of any other relevant official URLs cited)

Important:
- Only extract URLs that appear in the answer. If a specific URL for a field is not provided, set that field to null.
- If a value is missing in the answer, set the field to null (or empty list for degree_types).
- Do not infer or fabricate values or URLs.
- Preserve wording/format of values as stated in the answer (e.g., keep "~" or "about" if present).

Return JSON with a top-level "universities" array of UniversityItem objects as defined.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(*vals: Optional[str]) -> List[str]:
    return [v for v in vals if isinstance(v, str) and v.strip()]

def _gather_all_sources(u: UniversityItem) -> List[str]:
    s = u.sources or UniversitySourceURLs()
    urls = [
        s.university_url, s.program_url, s.degree_url, s.distance_url, s.credits_url,
        s.tuition_url, s.facilities_url, s.transportation_url, s.housing_url,
        s.prerequisites_url, s.research_url, s.faculty_url, s.dining_url,
    ]
    all_urls = [x for x in urls if x]
    all_urls.extend(s.additional_urls or [])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped

def _has_any_official_url(u: UniversityItem) -> bool:
    return len(_gather_all_sources(u)) > 0

def _value_exists(val: Optional[str] | List[str]) -> bool:
    if isinstance(val, list):
        return len([x for x in val if isinstance(x, str) and x.strip()]) > 0
    return bool(val and str(val).strip())

def _urls_exist(urls: List[str]) -> bool:
    return len(urls) > 0


# --------------------------------------------------------------------------- #
# Attribute check helper                                                      #
# --------------------------------------------------------------------------- #
async def add_attribute_with_source_check(
    evaluator: Evaluator,
    parent_node,
    *,
    node_prefix: str,
    value: Optional[str] | List[str],
    url_candidates: List[str],
    claim: str,
    provided_desc: str,
    supported_desc: str,
    add_ins: Optional[str] = None,
    critical_supported: bool = True,
) -> None:
    """
    For a single attribute:
      - Add a main parallel node to group checks
      - Add a 'value provided' critical custom node
      - Add a 'source provided' critical custom node
      - Add a 'supported by cited URL(s)' critical leaf verification
    """
    main = evaluator.add_parallel(
        id=f"{node_prefix}_main",
        desc=supported_desc,
        parent=parent_node,
        critical=False
    )

    # Provided?
    evaluator.add_custom_node(
        result=_value_exists(value),
        id=f"{node_prefix}_provided",
        desc=provided_desc,
        parent=main,
        critical=True
    )

    # Source provided?
    evaluator.add_custom_node(
        result=_urls_exist(url_candidates),
        id=f"{node_prefix}_source_provided",
        desc=f"{supported_desc} - source URL(s) provided",
        parent=main,
        critical=True
    )

    # Supported by the URL(s)?
    leaf = evaluator.add_leaf(
        id=f"{node_prefix}_supported",
        desc=supported_desc,
        parent=main,
        critical=critical_supported
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=url_candidates,
        additional_instruction=add_ins or "None"
    )


# --------------------------------------------------------------------------- #
# Per-university verification                                                 #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    u: UniversityItem,
    idx: int
) -> None:
    """
    Build verification sub-tree for one university.
    """
    uni_label = u.university_name or f"University #{idx + 1}"
    uni_node = evaluator.add_sequential(
        id=f"university_{idx}",
        desc=f"Verification for {uni_label}",
        parent=parent_node,
        critical=False
    )

    all_sources = _gather_all_sources(u)

    # 1) Required basic info (critical gate)
    required_node = evaluator.add_custom_node(
        result=(_value_exists(u.university_name) and _value_exists(u.program_name) and _has_any_official_url(u)),
        id=f"university_{idx}_required_info",
        desc="University has required identifying information (official name, program name, and at least one official URL)",
        parent=uni_node,
        critical=True
    )

    # 2) Qualifying constraints (parallel): in PA, within 60 miles, offers undergrad astronomy/astrophysics
    constraints = evaluator.add_parallel(
        id=f"university_{idx}_constraints",
        desc="Qualifying constraints (PA location, within 60 miles of Philadelphia, and offers undergrad astronomy/astrophysics)",
        parent=uni_node,
        critical=False
    )

    # 2.1 Located in Pennsylvania
    loc_leaf = evaluator.add_leaf(
        id=f"university_{idx}_in_pa",
        desc="University is located in Pennsylvania",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_label} is located in Pennsylvania.",
        node=loc_leaf,
        sources=_safe_list(u.sources.university_url if u.sources else None, u.sources.program_url if u.sources else None, *(_gather_all_sources(u))),
        additional_instruction="Use official university or program pages to confirm the state. Allow common abbreviations like 'PA'."
    )

    # 2.2 Within 60 miles of Philadelphia
    within_leaf = evaluator.add_leaf(
        id=f"university_{idx}_within_60_miles",
        desc="Main campus is within 60 miles of Philadelphia",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The distance from Philadelphia to {uni_label}'s main campus is within 60 miles.",
        node=within_leaf,
        sources=_safe_list(u.sources.distance_url if u.sources else None, *all_sources),
        additional_instruction="If a specific miles figure is provided, allow ±5 miles tolerance for rounding or route variation. Prefer official campus directions pages, program pages, or cited map links."
    )

    # 2.3 Offers undergraduate astronomy/astrophysics degree
    offers_leaf = evaluator.add_leaf(
        id=f"university_{idx}_offers_undergrad_astronomy",
        desc="Offers an undergraduate astronomy/astrophysics degree program",
        parent=constraints,
        critical=True
    )
    offer_sources = _safe_list(u.sources.program_url if u.sources else None, u.sources.degree_url if u.sources else None, *all_sources)
    await evaluator.verify(
        claim=f"{uni_label} offers an undergraduate astronomy or astrophysics degree program named '{u.program_name}'.",
        node=offers_leaf,
        sources=offer_sources,
        additional_instruction="Confirm that the program is undergraduate-level and in astronomy/astrophysics (may include 'astrophysics', 'astronomy & astrophysics', 'physics with astronomy concentration', etc.). Minor name variations acceptable."
    )

    # 3) Attribute verifications (non-critical group, each sub-attribute has its own critical checks)
    attributes = evaluator.add_parallel(
        id=f"university_{idx}_attributes",
        desc="Attribute verifications for the university",
        parent=uni_node,
        critical=False
    )

    s = u.sources or UniversitySourceURLs()

    # 3.0 Official university name support
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_name",
        value=u.university_name,
        url_candidates=_safe_list(s.university_url, s.program_url, *all_sources),
        claim=f"The official university name is '{u.university_name}'.",
        provided_desc="Official university name is provided",
        supported_desc="Official university name is supported by the cited source",
        add_ins="Use the university homepage or an official page to verify the official institution name; minor punctuation/casing differences are acceptable."
    )

    # 3.1 Program name
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_program_name",
        value=u.program_name,
        url_candidates=_safe_list(s.program_url, s.degree_url, *all_sources),
        claim=f"The official undergraduate program name is '{u.program_name}'.",
        provided_desc="Program name is provided",
        supported_desc="Program name is supported by the cited source",
        add_ins="Verify the exact or very similar program name on the official program/department page; allow minor formatting differences."
    )

    # 3.2 Degree types (BA/BS)
    deg_value = ", ".join([d for d in u.degree_types if d.strip()]) if u.degree_types else ""
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_degree_types",
        value=u.degree_types,
        url_candidates=_safe_list(s.degree_url, s.program_url, *all_sources),
        claim=f"The program offers the following undergraduate degree type(s): {deg_value}.",
        provided_desc="Degree type(s) are provided",
        supported_desc="Degree type(s) are supported by the cited source",
        add_ins="Confirm whether BA and/or BS (or equivalents) are offered; closely related naming acceptable (e.g., 'B.S.')."
    )

    # 3.3 Distance (value)
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_distance_value",
        value=u.distance_miles,
        url_candidates=_safe_list(s.distance_url, *all_sources),
        claim=f"The distance from Philadelphia to {uni_label}'s campus is {u.distance_miles}.",
        provided_desc="Distance in miles is provided",
        supported_desc="Distance value is supported by the cited source",
        add_ins="If a numeric value is provided, allow approximate/rounded figures; verify with cited campus directions page or map link."
    )

    # 3.4 Total credits
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_credits",
        value=u.total_credits,
        url_candidates=_safe_list(s.credits_url, s.program_url, s.degree_url, *all_sources),
        claim=f"The astronomy/astrophysics major requires {u.total_credits} total credit hours (or equivalent) to complete.",
        provided_desc="Total credits are provided",
        supported_desc="Total credits are supported by the cited source",
        add_ins="Check official curriculum or catalog pages. Accept course-unit equivalents if the program uses units instead of credit hours."
    )

    # 3.5 Annual tuition (2025–2026)
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_tuition",
        value=u.annual_tuition_2025_2026,
        url_candidates=_safe_list(s.tuition_url, *all_sources),
        claim=f"The annual undergraduate tuition for the 2025–2026 academic year is {u.annual_tuition_2025_2026}.",
        provided_desc="Annual undergraduate tuition (2025–2026) is provided",
        supported_desc="Annual undergraduate tuition (2025–2026) is supported by the cited source",
        add_ins="If 2025–2026 is not posted yet, the answer may indicate a provisional or prior-year figure; verify consistency with the cited tuition/fees page."
    )

    # 3.6 Observatory/Planetarium
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_facilities",
        value=u.has_observatory_planetarium,
        url_candidates=_safe_list(s.facilities_url, s.program_url, *all_sources),
        claim=f"The university has the following facility availability for students: {u.has_observatory_planetarium}.",
        provided_desc="Observatory/planetarium facility information is provided",
        supported_desc="Observatory/planetarium facility information is supported by the cited source",
        add_ins="Confirm if an on-campus observatory and/or planetarium exists and is accessible for undergraduates."
    )

    # 3.7 Public transportation options from Philadelphia
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_public_transport",
        value=u.public_transportation_options,
        url_candidates=_safe_list(s.transportation_url, s.university_url, *all_sources),
        claim=f"Public transportation options from Philadelphia to {uni_label} include: {u.public_transportation_options}.",
        provided_desc="Public transportation options are provided",
        supported_desc="Public transportation options are supported by the cited source",
        add_ins="Accept official campus directions/transit guidance pages; references to SEPTA/Amtrak pages are acceptable if cited in the answer."
    )

    # 3.8 First-year housing requirement
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_first_year_housing",
        value=u.first_year_housing_required,
        url_candidates=_safe_list(s.housing_url, *all_sources),
        claim=f"For {uni_label}, first-year students are {u.first_year_housing_required} to live on campus.",
        provided_desc="First-year housing requirement is provided",
        supported_desc="First-year housing requirement is supported by the cited source",
        add_ins="Verify on official housing/residential life pages or student policies."
    )

    # 3.9 Prerequisite courses
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_prerequisites",
        value=u.prerequisite_courses,
        url_candidates=_safe_list(s.prerequisites_url, s.program_url, s.degree_url, *all_sources),
        claim=f"Prerequisite courses for the astronomy/astrophysics major include: {u.prerequisite_courses}.",
        provided_desc="Prerequisite courses are provided",
        supported_desc="Prerequisite courses are supported by the cited source",
        add_ins="Look for calculus and physics sequences or equivalent; verify on program/major requirements pages."
    )

    # 3.10 Undergraduate research opportunities
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_research",
        value=u.research_opportunities_available,
        url_candidates=_safe_list(s.research_url, s.program_url, *all_sources),
        claim=f"Undergraduate research opportunities in astronomy/astrophysics at {uni_label} are: {u.research_opportunities_available}.",
        provided_desc="Undergraduate research opportunities information is provided",
        supported_desc="Undergraduate research opportunities are supported by the cited source",
        add_ins="Confirm mentions of undergraduate research, faculty-mentored projects, REUs, or departmental research programs."
    )

    # 3.11 Faculty count
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_faculty_count",
        value=u.faculty_count,
        url_candidates=_safe_list(s.faculty_url, s.program_url, *all_sources),
        claim=f"The astronomy/astrophysics department (or relevant unit) lists {u.faculty_count} faculty members.",
        provided_desc="Faculty count is provided",
        supported_desc="Faculty count is supported by the cited source",
        add_ins="Use the department faculty page; reasonable interpretation allowed (e.g., counting core astronomy faculty within physics)."
    )

    # 3.12 Dining/meal plan requirement for first-year
    await add_attribute_with_source_check(
        evaluator,
        attributes,
        node_prefix=f"university_{idx}_dining_requirements",
        value=u.meal_plan_required_first_year,
        url_candidates=_safe_list(s.dining_url, *all_sources),
        claim=f"Meal plans for first-year students at {uni_label} are {u.meal_plan_required_first_year}.",
        provided_desc="Meal plan requirement information is provided",
        supported_desc="Meal plan requirement information is supported by the cited source",
        add_ins="Verify on official dining or student life pages."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Pennsylvania astronomy/astrophysics universities task.
    """
    # 1) Initialize evaluator and root (parallel aggregation across universities)
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

    # 2) Extract structured university data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # 3) Global requirement: at least three universities identified
    num_unis = len(extracted.universities)
    evaluator.add_custom_node(
        result=(num_unis >= 3),
        id="at_least_three_universities",
        desc="Identify at least three qualifying universities",
        parent=root,
        critical=True
    )

    # 4) Verify up to first 3 universities; pad with empty items if fewer
    universities = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Build per-university verification subtrees
    tasks = []
    for idx, uni in enumerate(universities):
        tasks.append(verify_university(evaluator, root, uni, idx))
    await asyncio.gather(*tasks)

    # 5) Return evaluation summary
    return evaluator.get_summary()