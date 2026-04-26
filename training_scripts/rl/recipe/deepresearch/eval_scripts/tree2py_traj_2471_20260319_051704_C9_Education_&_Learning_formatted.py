import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "three_public_universities_sacscoc_r1_criteria"
TASK_DESCRIPTION = """
Identify three distinct public universities in the United States that meet ALL of the following criteria:

General Requirements (all three universities must meet these):
- Founded or established between 1870 and 1940 (inclusive)
- Located in a southern U.S. state within the SACSCOC (Southern Association of Colleges and Schools Commission on Colleges) accreditation region
- Hold current SACSCOC accreditation
- Hold Carnegie R1 classification for very high research activity
- Offer doctoral degree programs (PhD and/or EdD)
- Offer online or distance learning graduate programs
- Main campus is at least 200 acres in size
- Total student enrollment of at least 20,000
- Offer at least 50 graduate degree programs (master's and doctoral combined)

Special Distinguishing Requirements (each university must meet at least ONE of these):
- Was founded in 1930, OR
- Is a land-grant institution founded in 1876, OR
- Has a main campus larger than 5,000 acres, OR
- Is located in Virginia, OR
- Is located in Texas, OR
- Had a former University of Michigan quarterback serve as a coach (in any capacity) at that institution during the period 2020-2022

For each of the three universities, provide:
1. The university's full name
2. Founding year
3. Location (city and state)
4. Main campus size in acres
5. Total enrollment (with reference year)
6. Number of graduate programs offered
7. Which special requirement(s) it satisfies
8. Reference URLs supporting each claim
"""

# SACSCOC U.S. states (allow both names and postal abbreviations)
SACSCOC_STATE_NAMES = {
    "alabama", "florida", "georgia", "kentucky", "louisiana",
    "mississippi", "north carolina", "south carolina",
    "tennessee", "texas", "virginia"
}
SACSCOC_STATE_ABBR = {
    "al": "alabama", "fl": "florida", "ga": "georgia", "ky": "kentucky",
    "la": "louisiana", "ms": "mississippi", "nc": "north carolina",
    "sc": "south carolina", "tn": "tennessee", "tx": "texas", "va": "virginia"
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Core identity
    name: Optional[str] = None
    founding_year: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    public_institution: Optional[bool] = None

    # Accreditation & classification
    sacscoc_accredited: Optional[bool] = None
    carnegie_r1: Optional[bool] = None

    # Programs
    has_doctoral_programs: Optional[bool] = None
    online_grad_programs: Optional[bool] = None
    graduate_program_count: Optional[str] = None

    # Size & enrollment
    campus_acres: Optional[str] = None
    total_enrollment: Optional[str] = None
    enrollment_year: Optional[str] = None

    # Special flags
    is_land_grant: Optional[bool] = None
    michigan_qb_coach_2020_2022: Optional[bool] = None
    special_requirements: List[str] = Field(default_factory=list)

    # Per-attribute source URLs (any that appear in the answer)
    founding_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    public_status_urls: List[str] = Field(default_factory=list)
    sacscoc_urls: List[str] = Field(default_factory=list)
    carnegie_urls: List[str] = Field(default_factory=list)
    doctoral_urls: List[str] = Field(default_factory=list)
    online_urls: List[str] = Field(default_factory=list)
    program_count_urls: List[str] = Field(default_factory=list)
    campus_acres_urls: List[str] = Field(default_factory=list)
    enrollment_urls: List[str] = Field(default_factory=list)
    special_urls: List[str] = Field(default_factory=list)
    general_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to THREE distinct U.S. public universities from the answer and provide structured details for each. Only extract what is explicitly stated in the answer. If a field is missing, set it to null or an empty array as appropriate. For each university, extract:

Identity
- name: full university name
- founding_year: the founding or establishment year (string, keep as written, e.g., "1912")
- city: main campus city
- state: main campus state name or two-letter abbreviation
- public_institution: boolean (true if public university)

Accreditation & classification
- sacscoc_accredited: boolean (true if currently accredited by SACSCOC)
- carnegie_r1: boolean (true if designated R1: Very High Research Activity)

Programs
- has_doctoral_programs: boolean (true if PhD and/or EdD offered)
- online_grad_programs: boolean (true if online/distance graduate programs offered)
- graduate_program_count: string for the total number of graduate programs (master's + doctoral) if stated

Size & enrollment
- campus_acres: string for main campus acreage (keep formatting like "5,200")
- total_enrollment: string for total student enrollment (keep as written, e.g., "39,000", "over 20,000")
- enrollment_year: string for the reference year of the enrollment if provided

Special flags and notes
- is_land_grant: boolean (true if described as a land‑grant institution)
- michigan_qb_coach_2020_2022: boolean (true if a former University of Michigan quarterback served as a coach there during 2020–2022)
- special_requirements: array of strings summarizing which special requirement(s) (from the task list) the answer claims this university satisfies

For each university, also extract any URLs explicitly cited that support the specific claims. Put each URL in the most specific field(s) that apply; if a URL supports multiple items or is general, put it into general_urls.
- founding_urls: list
- location_urls: list
- public_status_urls: list
- sacscoc_urls: list
- carnegie_urls: list
- doctoral_urls: list
- online_urls: list
- program_count_urls: list
- campus_acres_urls: list
- enrollment_urls: list
- special_urls: list
- general_urls: list

Return an object:
{
  "universities": [UniversityItem, UniversityItem, UniversityItem]
}

If the answer provides more than three universities, only extract the first three. If fewer than three, return what exists (others will be null).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _strip_or_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def parse_int_loose(text: Optional[str]) -> Optional[int]:
    """
    Extract a reasonable integer from a possibly formatted string like:
    "5,200", "5200+", "approx. 21,000", "20,000–22,000", "18k" (not supported well), etc.
    Strategy: find all digit groups and return the largest integer found.
    """
    if not text:
        return None
    nums = re.findall(r"\d[\d,]*", text)
    candidates: List[int] = []
    for n in nums:
        try:
            candidates.append(int(n.replace(",", "")))
        except Exception:
            continue
    if candidates:
        return max(candidates)
    return None


def normalize_state_to_name(state: Optional[str]) -> Optional[str]:
    """
    Normalize a state input (full name or 2-letter abbr) to lowercase full name if recognized.
    """
    if not state:
        return None
    s = state.strip().lower()
    if s in SACSCOC_STATE_NAMES:
        return s
    if s in SACSCOC_STATE_ABBR:
        return SACSCOC_STATE_ABBR[s]
    # Try title-case mapping for unusual casing (e.g., "North Carolina")
    s_norm = " ".join(s.split())
    if s_norm in SACSCOC_STATE_NAMES:
        return s_norm
    return None


def is_in_sacscoc_region(state: Optional[str]) -> bool:
    full = normalize_state_to_name(state)
    return (full in SACSCOC_STATE_NAMES) if full else False


def gather_urls(u: UniversityItem, fields: List[str]) -> List[str]:
    """
    Collect URLs from multiple list fields on the UniversityItem, plus fallback to general_urls.
    """
    out: List[str] = []
    for f in fields:
        urls = getattr(u, f, None)
        if urls and isinstance(urls, list):
            out.extend(urls)
    # Fallback to general urls
    if u.general_urls:
        out.extend(u.general_urls)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for url in out:
        if not url:
            continue
        if url not in seen:
            uniq.append(url)
            seen.add(url)
    return uniq


def detect_specials(u: UniversityItem) -> List[str]:
    """
    Compute which of the special requirements are met based on extracted fields.
    """
    specials: List[str] = []
    year = parse_int_loose(u.founding_year)
    acres = parse_int_loose(u.campus_acres)
    st_name = normalize_state_to_name(u.state)

    if year == 1930:
        specials.append("founded in 1930")
    if (year == 1876) and (u.is_land_grant is True):
        specials.append("land-grant institution founded in 1876")
    if acres is not None and acres > 5000:
        specials.append("main campus larger than 5,000 acres")
    if st_name == "virginia":
        specials.append("located in Virginia")
    if st_name == "texas":
        specials.append("located in Texas")
    if u.michigan_qb_coach_2020_2022 is True:
        specials.append("had a former University of Michigan QB serve as a coach during 2020–2022")

    # Also include any explicit entries listed by the answer (dedupe after)
    for s in (u.special_requirements or []):
        s_clean = s.strip()
        if s_clean and s_clean not in specials:
            specials.append(s_clean)

    return specials


# --------------------------------------------------------------------------- #
# Verification sub-tree builder for a single university                       #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, root: VerificationNode, u: UniversityItem, idx: int) -> None:
    """
    Build and evaluate the verification tree for one university based on the rubric.
    We slightly adjust criticality for 'Source' leaves to ensure evidence-grounded checks gate success.
    """
    uni_n = idx + 1
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_n}",
        desc=f"{['First','Second','Third'][idx]} qualifying university",
        parent=root,
        critical=True  # All three universities are required; failing any should fail the overall task
    )

    # -------------------- Institutional Characteristics (critical) --------------------
    inst_node = evaluator.add_parallel(
        id=f"U{uni_n}_Institutional_Characteristics",
        desc="Basic institutional attributes verified",
        parent=uni_node,
        critical=True
    )

    # Founding period (sequential)
    founding_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Founding_Period",
        desc="Founded between 1870-1940 inclusive",
        parent=inst_node,
        critical=True
    )

    # Founding year check (custom, critical)
    year_val = parse_int_loose(u.founding_year)
    founding_year_ok = (year_val is not None) and (1870 <= year_val <= 1940)
    evaluator.add_custom_node(
        result=founding_year_ok,
        id=f"U{uni_n}_Founding_Year",
        desc="Specific founding year documented and within 1870-1940",
        parent=founding_seq,
        critical=True
    )

    # Founding source (verify by URLs, critical to enforce evidence)
    founding_source_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_Founding_Source",
        desc="URL reference for founding information supports the year and range requirement",
        parent=founding_seq,
        critical=True
    )
    founding_claim = (
        f"The university '{_strip_or_none(u.name) or 'the institution'}' was founded in "
        f"{_strip_or_none(u.founding_year) or 'the cited year'}, which lies between 1870 and 1940 inclusive."
    )
    await evaluator.verify(
        claim=founding_claim,
        node=founding_source_leaf,
        sources=gather_urls(u, ["founding_urls"])
    )

    # Location info (sequential)
    loc_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Location_Info",
        desc="Located in SACSCOC region southern state",
        parent=inst_node,
        critical=True
    )
    state_city_ok = bool(_strip_or_none(u.city)) and is_in_sacscoc_region(u.state)
    evaluator.add_custom_node(
        result=state_city_ok,
        id=f"U{uni_n}_State_City",
        desc="Specific state and city documented and state is within SACSCOC region",
        parent=loc_seq,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_Location_Source",
        desc="URL reference for location information",
        parent=loc_seq,
        critical=True
    )
    loc_claim = (
        f"The main campus of '{_strip_or_none(u.name) or 'the institution'}' is in "
        f"{_strip_or_none(u.city) or 'the stated city'}, "
        f"{_strip_or_none(u.state) or 'the stated state'} (United States)."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=gather_urls(u, ["location_urls"])
    )

    # Public institution (sequential)
    public_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Public_Institution",
        desc="Public university status verified",
        parent=inst_node,
        critical=True
    )
    public_ok = (u.public_institution is True)
    evaluator.add_custom_node(
        result=public_ok,
        id=f"U{uni_n}_Public_Status_Check",
        desc="Confirmed as public institution (boolean from answer)",
        parent=public_seq,
        critical=True
    )
    public_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_Public_Source",
        desc="URL reference for public status",
        parent=public_seq,
        critical=True
    )
    public_claim = f"'{_strip_or_none(u.name) or 'The institution'}' is a public university."
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=gather_urls(u, ["public_status_urls"])
    )

    # -------------------- Accreditation & Classification (critical) --------------------
    accred_node = evaluator.add_parallel(
        id=f"U{uni_n}_Accreditation_Classification",
        desc="Accreditation and research classification verified",
        parent=uni_node,
        critical=True
    )

    # SACSCOC Accreditation (sequential)
    sacscoc_seq = evaluator.add_sequential(
        id=f"U{uni_n}_SACSCOC_Accreditation",
        desc="SACSCOC accreditation confirmed",
        parent=accred_node,
        critical=True
    )
    sacscoc_ok = (u.sacscoc_accredited is True)
    evaluator.add_custom_node(
        result=sacscoc_ok,
        id=f"U{uni_n}_SACSCOC_Status",
        desc="Current SACSCOC accreditation status (boolean from answer)",
        parent=sacscoc_seq,
        critical=True
    )
    sacscoc_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_SACSCOC_Source",
        desc="URL reference for SACSCOC accreditation",
        parent=sacscoc_seq,
        critical=True
    )
    sacscoc_claim = (
        f"'{_strip_or_none(u.name) or 'The institution'}' is currently accredited by the "
        f"Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)."
    )
    await evaluator.verify(
        claim=sacscoc_claim,
        node=sacscoc_leaf,
        sources=gather_urls(u, ["sacscoc_urls"])
    )

    # Carnegie R1 (sequential)
    r1_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Research_Classification",
        desc="Carnegie R1 classification confirmed",
        parent=accred_node,
        critical=True
    )
    r1_ok = (u.carnegie_r1 is True)
    evaluator.add_custom_node(
        result=r1_ok,
        id=f"U{uni_n}_R1_Status",
        desc="Carnegie R1 very high research activity status (boolean from answer)",
        parent=r1_seq,
        critical=True
    )
    r1_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_R1_Source",
        desc="URL reference for Carnegie classification",
        parent=r1_seq,
        critical=True
    )
    r1_claim = (
        f"'{_strip_or_none(u.name) or 'The institution'}' holds the Carnegie R1 "
        f"(Very High Research Activity) classification."
    )
    await evaluator.verify(
        claim=r1_claim,
        node=r1_leaf,
        sources=gather_urls(u, ["carnegie_urls"])
    )

    # -------------------- Academic Programs (critical) --------------------
    acad_node = evaluator.add_parallel(
        id=f"U{uni_n}_Academic_Programs",
        desc="Academic program offerings verified",
        parent=uni_node,
        critical=True
    )

    # Doctoral offerings (sequential)
    doc_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Doctoral_Offerings",
        desc="Doctoral programs (PhD/EdD) offered",
        parent=acad_node,
        critical=True
    )
    doc_ok = (u.has_doctoral_programs is True)
    evaluator.add_custom_node(
        result=doc_ok,
        id=f"U{uni_n}_Doctoral_Programs_Check",
        desc="PhD and/or EdD programs confirmed (boolean from answer)",
        parent=doc_seq,
        critical=True
    )
    doc_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_Doctoral_Source",
        desc="URL reference for doctoral programs",
        parent=doc_seq,
        critical=True
    )
    doc_claim = f"'{_strip_or_none(u.name) or 'The institution'}' offers doctoral degree programs (PhD and/or EdD)."
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=gather_urls(u, ["doctoral_urls"])
    )

    # Online graduate programs (sequential)
    online_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Online_Graduate",
        desc="Online/distance graduate programs offered",
        parent=acad_node,
        critical=True
    )
    online_ok = (u.online_grad_programs is True)
    evaluator.add_custom_node(
        result=online_ok,
        id=f"U{uni_n}_Online_Programs_Check",
        desc="Online graduate programs confirmed (boolean from answer)",
        parent=online_seq,
        critical=True
    )
    online_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_Online_Source",
        desc="URL reference for online programs",
        parent=online_seq,
        critical=True
    )
    online_claim = (
        f"'{_strip_or_none(u.name) or 'The institution'}' offers online or distance learning graduate programs."
    )
    await evaluator.verify(
        claim=online_claim,
        node=online_leaf,
        sources=gather_urls(u, ["online_urls"])
    )

    # Graduate program count >= 50 (sequential)
    count_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Program_Count",
        desc="At least 50 graduate programs offered",
        parent=acad_node,
        critical=True
    )
    grad_count_val = parse_int_loose(u.graduate_program_count)
    count_ok = (grad_count_val is not None) and (grad_count_val >= 50)
    evaluator.add_custom_node(
        result=count_ok,
        id=f"U{uni_n}_Graduate_Count_Check",
        desc="Minimum 50 master's + doctoral programs confirmed (from extracted count)",
        parent=count_seq,
        critical=True
    )
    count_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_Count_Source",
        desc="URL reference for program count",
        parent=count_seq,
        critical=True
    )
    count_claim = (
        f"'{_strip_or_none(u.name) or 'The institution'}' offers at least 50 graduate degree programs "
        f"(master's and doctoral combined)."
    )
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=gather_urls(u, ["program_count_urls"])
    )

    # -------------------- Size & Enrollment (critical) --------------------
    size_node = evaluator.add_parallel(
        id=f"U{uni_n}_Size_Enrollment",
        desc="Campus size and enrollment verified",
        parent=uni_node,
        critical=True
    )

    # Campus acreage >= 200 (sequential)
    acres_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Campus_Acreage",
        desc="Main campus at least 200 acres",
        parent=size_node,
        critical=True
    )
    acres_val = parse_int_loose(u.campus_acres)
    acres_ok = (acres_val is not None) and (acres_val >= 200)
    evaluator.add_custom_node(
        result=acres_ok,
        id=f"U{uni_n}_Acres_Check",
        desc="Campus size meets minimum requirement (>= 200 acres)",
        parent=acres_seq,
        critical=True
    )
    acres_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_Acres_Source",
        desc="URL reference for campus size",
        parent=acres_seq,
        critical=True
    )
    acres_claim = (
        f"The main campus of '{_strip_or_none(u.name) or 'the institution'}' is at least 200 acres "
        f"(reported size: {_strip_or_none(u.campus_acres) or 'as cited'})."
    )
    await evaluator.verify(
        claim=acres_claim,
        node=acres_leaf,
        sources=gather_urls(u, ["campus_acres_urls"])
    )

    # Total enrollment >= 20,000 (sequential)
    enroll_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Student_Enrollment",
        desc="Total enrollment at least 20,000 students",
        parent=size_node,
        critical=True
    )
    enrollment_val = parse_int_loose(u.total_enrollment)
    enrollment_ok = (enrollment_val is not None) and (enrollment_val >= 20000)
    evaluator.add_custom_node(
        result=enrollment_ok,
        id=f"U{uni_n}_Enrollment_Check",
        desc="Enrollment meets minimum requirement (>= 20,000)",
        parent=enroll_seq,
        critical=True
    )
    enroll_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_Enrollment_Source",
        desc="URL reference for enrollment data",
        parent=enroll_seq,
        critical=True
    )
    enroll_claim = (
        f"'{_strip_or_none(u.name) or 'The institution'}' has total student enrollment of at least 20,000 "
        f"(reported: {_strip_or_none(u.total_enrollment) or 'as cited'}; reference year: "
        f"{_strip_or_none(u.enrollment_year) or 'as cited'})."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=gather_urls(u, ["enrollment_urls"])
    )

    # -------------------- Special Distinguishing (critical) --------------------
    special_seq = evaluator.add_sequential(
        id=f"U{uni_n}_Special_Distinguishing",
        desc="Meets at least one special requirement",
        parent=uni_node,
        critical=True
    )
    specials_detected = detect_specials(u)
    specials_ok = len(specials_detected) > 0
    evaluator.add_custom_node(
        result=specials_ok,
        id=f"U{uni_n}_Special_Met",
        desc="At least one special requirement satisfied",
        parent=special_seq,
        critical=True
    )
    special_leaf = evaluator.add_leaf(
        id=f"U{uni_n}_Special_Source",
        desc="URL reference for special requirement",
        parent=special_seq,
        critical=True
    )
    specials_text = ", ".join(specials_detected) if specials_detected else "the stated special requirement(s)"
    special_claim = (
        f"'{_strip_or_none(u.name) or 'The institution'}' satisfies at least one of the required special "
        f"distinguishing conditions: {specials_text}."
    )
    await evaluator.verify(
        claim=special_claim,
        node=special_leaf,
        sources=gather_urls(u, ["special_urls", "founding_urls", "campus_acres_urls", "location_urls"])
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
    Evaluate an answer for the 'three public universities meeting SACSCOC/R1/etc.' task.

    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator (root is non-critical by framework design; we enforce "all 3" via child criticality)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # As per rubric's Task_Completion aggregation
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

    # Extract up to three universities
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize to exactly three entries (pad with empty if fewer)
    universities: List[UniversityItem] = list(extracted.universities or [])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Add simple info about recognized SACSCOC states to the report
    evaluator.add_custom_info(
        info={"sacscoc_states": sorted(list(SACSCOC_STATE_NAMES))},
        info_type="reference",
        info_name="sacscoc_region_reference"
    )

    # Build and verify for each of the three required universities
    for i in range(3):
        await verify_university(evaluator, root, universities[i], i)

    # Return the final structured evaluation summary
    return evaluator.get_summary()