import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohio_public_universities_comprehensive_info"
TASK_DESCRIPTION = """
Find comprehensive information about four specific Ohio public universities: Ohio University (main campus in Athens), Miami University (main campus in Oxford), University of Cincinnati, and Kent State University.

For each of these four universities, you must provide the following information:

1. Founding Year: The year the university was founded or chartered

2. Current President: Full name of the current president, the year they assumed office, and any notable distinction about their presidency (e.g., first female president, etc.)

3. Current Chief Academic Officer: Full name of the current provost or executive vice president for academic affairs, official title, and year they assumed the position (or note if serving in interim capacity)

4. Main Campus Location: City where the main campus is located and complete street address of the main campus or central administration

5. Academic Structure: Total number of degree-granting colleges or academic divisions, and names of at least three colleges/schools

6. Regional Campus System: Whether the university operates regional or branch campuses, and if so, the number of regional campuses and their names/locations

7. Current Enrollment: Total student enrollment from Fall 2024 or 2025 (specify which semester/year)

8. Reference URLs: For each university, provide at least three reference URLs from official university sources supporting the information (covering leadership, location/structure, and enrollment/campuses)

All information must be current as of 2024-2026 and verifiable through official university websites or authoritative sources.
"""

# Expected official domains per university (used to validate "official source" references)
UNIVERSITY_DOMAINS = {
    "ohio": ["ohio.edu"],
    "miami": ["miamioh.edu"],
    "cincinnati": ["uc.edu"],
    "kent": ["kent.edu"],
}

UNIVERSITY_NAMES = {
    "ohio": "Ohio University",
    "miami": "Miami University",
    "cincinnati": "University of Cincinnati",
    "kent": "Kent State University",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityInfo(BaseModel):
    # 1. Founding
    founding_year: Optional[str] = None

    # 2. President
    president_name: Optional[str] = None
    president_start_year: Optional[str] = None
    president_distinction: Optional[str] = None  # optional

    # 3. Provost / Chief Academic Officer
    provost_name: Optional[str] = None
    provost_title: Optional[str] = None
    provost_start_or_status: Optional[str] = None  # could be year or "interim"

    # 4. Main Campus Location
    campus_city: Optional[str] = None
    campus_address: Optional[str] = None

    # 5. Academic Structure
    number_colleges: Optional[str] = None
    college_names: List[str] = Field(default_factory=list)

    # 6. Regional campus system
    has_regional: Optional[str] = None  # yes/no/unknown (keep as free text for flexibility)
    regional_count: Optional[str] = None
    regional_names: List[str] = Field(default_factory=list)

    # 7. Enrollment
    enrollment_figure: Optional[str] = None
    enrollment_year: Optional[str] = None  # e.g., "Fall 2024" or "Fall 2025"

    # 8. References (URLs)
    leadership_reference_urls: List[str] = Field(default_factory=list)          # leadership-related official pages
    location_structure_reference_urls: List[str] = Field(default_factory=list)  # campus location/colleges pages
    enrollment_campus_reference_urls: List[str] = Field(default_factory=list)   # enrollment/regional campuses pages


class FourUniversitiesExtraction(BaseModel):
    ohio_university: Optional[UniversityInfo] = None
    miami_university: Optional[UniversityInfo] = None
    university_of_cincinnati: Optional[UniversityInfo] = None
    kent_state_university: Optional[UniversityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract structured information for each of the four universities explicitly mentioned in the answer:
    - Ohio University (Athens)
    - Miami University (Oxford)
    - University of Cincinnati
    - Kent State University

    For each university, extract these fields as strings or lists (do not infer or add information not present in the answer; if missing, return null or empty list):
    1) founding_year
    2) president_name
    3) president_start_year
    4) president_distinction  (if not mentioned or not applicable, set to null)
    5) provost_name
    6) provost_title
    7) provost_start_or_status
    8) campus_city
    9) campus_address
    10) number_colleges
    11) college_names (list of at least three, if provided; otherwise list what is given)
    12) has_regional (e.g., "Yes", "No", "No regional campuses", "Unknown")
    13) regional_count (string if a number or descriptor is provided; else null)
    14) regional_names (list; include all names/locations if provided; else empty list)
    15) enrollment_figure (e.g., "40,000", "40,200 total", or similar)
    16) enrollment_year (e.g., "Fall 2024", "Fall 2025", etc.)
    17) leadership_reference_urls (list of URLs cited for leadership/president/provost info)
    18) location_structure_reference_urls (list of URLs cited for campus location/addresses/college structure)
    19) enrollment_campus_reference_urls (list of URLs cited for enrollment or regional campus info)

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer text. Do not invent URLs.
    - Include full URLs (with http/https). If protocol is missing, prepend http://.
    - Keep values as strings as they may contain qualifiers (e.g., "circa 1804", "Executive Vice President and Provost").
    - If an item is not present for a university, set it to null (or [] for list fields).

    Output JSON must have these top-level keys exactly:
    {
      "ohio_university": {...},
      "miami_university": {...},
      "university_of_cincinnati": {...},
      "kent_state_university": {...}
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s is not None and str(s).strip() != "")


def _normalize_yes_no(s: Optional[str]) -> Optional[str]:
    if not _nonempty(s):
        return None
    t = str(s).strip().lower()
    if any(x in t for x in ["yes", "has", "operate", "operates", "maintain", "maintains", "regional campuses", "branch campuses"]):
        # If string says "no" explicitly, treat as no
        if "no " in t or t == "no" or "none" in t or "does not" in t or "do not" in t:
            return "no"
        return "yes"
    if t in ["no", "none", "0", "does not", "do not"]:
        return "no"
    return t  # unknown/other


def _is_official_url(url: str, allowed_domains: List[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if not host:
            return False
        return any(host.endswith(d) for d in allowed_domains)
    except Exception:
        return False


def _has_at_least_one_official(urls: List[str], allowed_domains: List[str]) -> bool:
    if not urls:
        return False
    return any(_is_official_url(u, allowed_domains) for u in urls)


def _first_n(items: List[str], n: int) -> List[str]:
    return items[:n] if items else []


# --------------------------------------------------------------------------- #
# Verification builder for one university                                     #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni_key: str,
    uni_name: str,
    info: Optional[UniversityInfo],
    allowed_domains: List[str],
) -> None:
    """
    Build verification tree and checks for a single university.
    """
    # Create a container node per university (non-critical, parallel as per rubric)
    uni_node = evaluator.add_parallel(
        id=f"{uni_key}_info",
        desc=f"Complete information set for {uni_name}",
        parent=parent_node,
        critical=False,
    )

    # If nothing extracted, create placeholder to avoid attribute errors
    info = info or UniversityInfo()

    # 8. References (Create first so other verifications can depend on them)
    refs_node = evaluator.add_parallel(
        id=f"{uni_key}_references",
        desc="Reference URLs from official sources",
        parent=uni_node,
        critical=True
    )
    # leadership_reference
    leader_ref_ok = _has_at_least_one_official(info.leadership_reference_urls, allowed_domains)
    ref_lead_node = evaluator.add_custom_node(
        result=leader_ref_ok,
        id=f"{uni_key}_leadership_reference",
        desc="URL for leadership information from official university website",
        parent=refs_node,
        critical=True
    )
    # location/structure reference
    loc_ref_ok = _has_at_least_one_official(info.location_structure_reference_urls, allowed_domains)
    ref_loc_node = evaluator.add_custom_node(
        result=loc_ref_ok,
        id=f"{uni_key}_location_structure_reference",
        desc="URL for location or academic structure from official university website",
        parent=refs_node,
        critical=True
    )
    # enrollment/campus reference
    enr_ref_ok = _has_at_least_one_official(info.enrollment_campus_reference_urls, allowed_domains)
    ref_enr_node = evaluator.add_custom_node(
        result=enr_ref_ok,
        id=f"{uni_key}_enrollment_campus_reference",
        desc="URL for enrollment or campus system from official university website",
        parent=refs_node,
        critical=True
    )

    # Convenience: prerequisites for verifications that should rely on these references
    reference_leaves = [ref_lead_node, ref_loc_node, ref_enr_node]

    # 1) Founding year (presence per rubric)
    evaluator.add_custom_node(
        result=_nonempty(info.founding_year),
        id=f"{uni_key}_founding_year",
        desc="Founding year is provided",
        parent=uni_node,
        critical=True
    )

    # 2) President info (critical group; all children must be critical to satisfy framework constraint)
    pres_node = evaluator.add_parallel(
        id=f"{uni_key}_president_info",
        desc="Presidential information",
        parent=uni_node,
        critical=True
    )

    # 2.1 President name (verify with leadership refs)
    pres_name_leaf = evaluator.add_leaf(
        id=f"{uni_key}_president_name",
        desc="President's full name is provided",
        parent=pres_node,
        critical=True
    )
    pres_name_claim = f"The current president of {uni_name} is '{info.president_name}'."
    await evaluator.verify(
        claim=pres_name_claim,
        node=pres_name_leaf,
        sources=info.leadership_reference_urls,
        additional_instruction="Verify from the official leadership page(s) that the stated person is the current president. Allow minor name formatting variations.",
        extra_prerequisites=[ref_lead_node]
    )

    # 2.2 President start year
    pres_start_leaf = evaluator.add_leaf(
        id=f"{uni_key}_president_start",
        desc="Year president assumed office is provided",
        parent=pres_node,
        critical=True
    )
    pres_start_claim = f"The president of {uni_name} assumed office in {info.president_start_year}."
    await evaluator.verify(
        claim=pres_start_claim,
        node=pres_start_leaf,
        sources=info.leadership_reference_urls,
        additional_instruction="Check official bio or leadership page for the year the president began their term. Allow minor phrasing differences (e.g., 'appointed in', 'started in').",
        extra_prerequisites=[ref_lead_node]
    )

    # 2.3 President distinction (treat as 'mentioned if applicable'; pass if empty; otherwise require non-empty)
    evaluator.add_custom_node(
        result=(info.president_distinction is None) or _nonempty(info.president_distinction),
        id=f"{uni_key}_president_distinction",
        desc="Notable distinction about presidency is mentioned if applicable",
        parent=pres_node,
        critical=True  # Keep critical to satisfy framework constraint for critical parent
    )

    # 3) Provost / Chief Academic Officer info (critical group)
    prov_node = evaluator.add_parallel(
        id=f"{uni_key}_provost_info",
        desc="Provost/chief academic officer information",
        parent=uni_node,
        critical=True
    )

    prov_name_leaf = evaluator.add_leaf(
        id=f"{uni_key}_provost_name",
        desc="Provost's full name is provided",
        parent=prov_node,
        critical=True
    )
    prov_name_claim = f"The current chief academic officer (provost) of {uni_name} is '{info.provost_name}'."
    await evaluator.verify(
        claim=prov_name_claim,
        node=prov_name_leaf,
        sources=info.leadership_reference_urls,
        additional_instruction="Confirm from official leadership/academic affairs pages. Allow minor naming or title formatting differences.",
        extra_prerequisites=[ref_lead_node]
    )

    prov_title_leaf = evaluator.add_leaf(
        id=f"{uni_key}_provost_title",
        desc="Official title is provided",
        parent=prov_node,
        critical=True
    )
    prov_title_claim = f"The official title for the chief academic officer at {uni_name} is '{info.provost_title}'."
    await evaluator.verify(
        claim=prov_title_claim,
        node=prov_title_leaf,
        sources=info.leadership_reference_urls,
        additional_instruction="Verify that the stated title (e.g., 'Provost and Executive Vice President for Academic Affairs') matches the official leadership page.",
        extra_prerequisites=[ref_lead_node]
    )

    prov_start_leaf = evaluator.add_leaf(
        id=f"{uni_key}_provost_start_or_status",
        desc="Year assumed position or interim status is noted",
        parent=prov_node,
        critical=True
    )
    prov_start_claim = f"The provost at {uni_name} has the start year or status as stated: '{info.provost_start_or_status}'."
    await evaluator.verify(
        claim=prov_start_claim,
        node=prov_start_leaf,
        sources=info.leadership_reference_urls,
        additional_instruction="Verify from official sources whether the stated year/status (including interim status if applicable) is correct.",
        extra_prerequisites=[ref_lead_node]
    )

    # 4) Main campus location (critical)
    loc_node = evaluator.add_parallel(
        id=f"{uni_key}_location_info",
        desc="Campus location information",
        parent=uni_node,
        critical=True
    )

    city_leaf = evaluator.add_leaf(
        id=f"{uni_key}_campus_city",
        desc="City of main campus is provided",
        parent=loc_node,
        critical=True
    )
    city_claim = f"The main campus of {uni_name} is located in '{info.campus_city}'."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=info.location_structure_reference_urls,
        additional_instruction="Verify the city of the main campus from official campus/location pages. Allow minor variants such as including the state (e.g., 'Athens, Ohio').",
        extra_prerequisites=[ref_loc_node]
    )

    addr_leaf = evaluator.add_leaf(
        id=f"{uni_key}_campus_address",
        desc="Complete street address is provided",
        parent=loc_node,
        critical=True
    )
    addr_claim = f"The primary main campus or central administration address for {uni_name} is '{info.campus_address}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=info.location_structure_reference_urls,
        additional_instruction="Verify the street address from official 'Contact', 'Visit', or 'Campus' pages. Minor formatting or abbreviation differences are acceptable.",
        extra_prerequisites=[ref_loc_node]
    )

    # 5) Academic structure (critical)
    acad_node = evaluator.add_parallel(
        id=f"{uni_key}_academic_structure",
        desc="Academic structure information",
        parent=uni_node,
        critical=True
    )

    num_colleges_leaf = evaluator.add_leaf(
        id=f"{uni_key}_number_colleges",
        desc="Total number of colleges/divisions is provided",
        parent=acad_node,
        critical=True
    )
    num_colleges_claim = f"The total number of degree‑granting colleges or academic divisions at {uni_name} is '{info.number_colleges}'."
    await evaluator.verify(
        claim=num_colleges_claim,
        node=num_colleges_leaf,
        sources=info.location_structure_reference_urls,
        additional_instruction="Verify any explicit counts from official academic structure pages. Accept reasonable phrasing variations (e.g., 'colleges/schools').",
        extra_prerequisites=[ref_loc_node]
    )

    colleges_to_check = _first_n(info.college_names, 6)
    colleges_list_str = ", ".join(colleges_to_check) if colleges_to_check else ""
    college_names_leaf = evaluator.add_leaf(
        id=f"{uni_key}_college_names",
        desc="Names of at least 3 colleges/schools are listed",
        parent=acad_node,
        critical=True
    )
    college_names_claim = (
        f"Some of the colleges/schools at {uni_name} include: {colleges_list_str}. "
        f"At least three of the listed names are accurate official colleges/schools."
    )
    await evaluator.verify(
        claim=college_names_claim,
        node=college_names_leaf,
        sources=info.location_structure_reference_urls,
        additional_instruction="Confirm that at least 3 of the listed names are present on the official colleges/schools pages. Allow minor naming variations (e.g., '&' vs 'and').",
        extra_prerequisites=[ref_loc_node]
    )

    # 6) Regional campus system (critical)
    regional_node = evaluator.add_parallel(
        id=f"{uni_key}_regional_campuses",
        desc="Regional campus system information",
        parent=uni_node,
        critical=True
    )
    has_reg_norm = _normalize_yes_no(info.has_regional) or (info.has_regional or "unknown")
    has_reg_leaf = evaluator.add_leaf(
        id=f"{uni_key}_has_regional",
        desc="Indicates whether university has regional campuses",
        parent=regional_node,
        critical=True
    )
    has_reg_claim = f"The university {uni_name} operates regional or branch campuses: '{has_reg_norm}'."
    await evaluator.verify(
        claim=has_reg_claim,
        node=has_reg_leaf,
        sources=(info.enrollment_campus_reference_urls or info.location_structure_reference_urls),
        additional_instruction="Judge 'yes' if official pages list regional/branch campuses or multi-campus system; judge 'no' if pages clearly indicate none. Allow synonyms like 'regional', 'branch', 'satellite'.",
        extra_prerequisites=[ref_enr_node] if info.enrollment_campus_reference_urls else [ref_loc_node]
    )

    # If has_regional is 'no' or 'unknown', treat count-and-names as trivially satisfied (presence 'if applicable')
    has_any_regional = (_normalize_yes_no(info.has_regional) == "yes")
    regional_count_names_ok = (not has_any_regional) or (_nonempty(info.regional_count) or len(info.regional_names) > 0)
    evaluator.add_custom_node(
        result=regional_count_names_ok,
        id=f"{uni_key}_regional_count_and_names",
        desc="If applicable, number and names of regional campuses are provided",
        parent=regional_node,
        critical=True
    )

    # 7) Enrollment (critical)
    enroll_node = evaluator.add_parallel(
        id=f"{uni_key}_enrollment",
        desc="Current enrollment information",
        parent=uni_node,
        critical=True
    )

    enrollment_fig_leaf = evaluator.add_leaf(
        id=f"{uni_key}_enrollment_figure",
        desc="Total enrollment number is provided",
        parent=enroll_node,
        critical=True
    )
    enrollment_fig_claim = f"The total student enrollment for {uni_name} is stated as '{info.enrollment_figure}'."
    await evaluator.verify(
        claim=enrollment_fig_claim,
        node=enrollment_fig_leaf,
        sources=info.enrollment_campus_reference_urls,
        additional_instruction="Verify the total enrollment figure from official facts/enrollment pages. Minor rounding or formatting variations are acceptable.",
        extra_prerequisites=[ref_enr_node]
    )

    enrollment_year_leaf = evaluator.add_leaf(
        id=f"{uni_key}_enrollment_year",
        desc="Semester/year of enrollment data is specified",
        parent=enroll_node,
        critical=True
    )
    enrollment_year_claim = f"The enrollment figure for {uni_name} corresponds to '{info.enrollment_year}'."
    await evaluator.verify(
        claim=enrollment_year_claim,
        node=enrollment_year_leaf,
        sources=info.enrollment_campus_reference_urls,
        additional_instruction="Verify that the cited enrollment corresponds to Fall 2024 or Fall 2025 (or equivalent academic term). Accept 'Autumn' synonym for 'Fall'.",
        extra_prerequisites=[ref_enr_node]
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
    Evaluate an answer for the comprehensive Ohio public universities information task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # parallel across universities
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=FourUniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Build verification trees for each university
    await verify_university(
        evaluator,
        root,
        "ohio_university",
        UNIVERSITY_NAMES["ohio"],
        extracted.ohio_university,
        UNIVERSITY_DOMAINS["ohio"],
    )

    await verify_university(
        evaluator,
        root,
        "miami_university",
        UNIVERSITY_NAMES["miami"],
        extracted.miami_university,
        UNIVERSITY_DOMAINS["miami"],
    )

    await verify_university(
        evaluator,
        root,
        "university_of_cincinnati",
        UNIVERSITY_NAMES["cincinnati"],
        extracted.university_of_cincinnati,
        UNIVERSITY_DOMAINS["cincinnati"],
    )

    await verify_university(
        evaluator,
        root,
        "kent_state_university",
        UNIVERSITY_NAMES["kent"],
        extracted.kent_state_university,
        UNIVERSITY_DOMAINS["kent"],
    )

    # Optionally, record domains used for official-check diagnostics
    evaluator.add_custom_info(
        info={
            "official_domains": UNIVERSITY_DOMAINS,
            "universities": UNIVERSITY_NAMES
        },
        info_type="official_domain_policy",
        info_name="domain_expectations"
    )

    return evaluator.get_summary()