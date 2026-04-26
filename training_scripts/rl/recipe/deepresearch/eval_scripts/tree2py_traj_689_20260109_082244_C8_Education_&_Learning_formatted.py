import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "public_universities_stem_teacher_cert_midwest"
TASK_DESCRIPTION = """Identify 4 public universities located in Pennsylvania, Ohio, Michigan, or Illinois that meet all of the following requirements:

1. The institution must be a public university (state-funded institution, not a private university or community college)
2. The institution must hold current regional accreditation from the appropriate regional accrediting agency (Higher Learning Commission for Illinois, Michigan, and Ohio; or Middle States Commission on Higher Education for Pennsylvania)
3. The institution must have at least one educator preparation program that is currently accredited by the Council for the Accreditation of Educator Preparation (CAEP)
4. The institution must offer an undergraduate (bachelor's degree level) teacher certification program in at least one STEM subject area, specifically: Mathematics Education, Science Education (Biology, Chemistry, Physics), or Computer Science Education
5. The teacher certification program must prepare teachers for secondary education (middle school and/or high school level, typically covering grades 6-12)

For each of the 4 universities you identify, provide:
- The full official name of the institution
- The state in which it is located
- The specific STEM teaching certification program(s) offered at the undergraduate level
- The grade levels the program prepares teachers to teach
- Reference URLs documenting: (a) the institution's regional accreditation status, (b) the institution's CAEP accreditation status, and (c) the specific STEM teacher certification program details
"""

ALLOWED_STATES = {"pennsylvania", "ohio", "michigan", "illinois"}
STATE_ABBR_MAP = {
    "pa": "pennsylvania",
    "oh": "ohio",
    "mi": "michigan",
    "il": "illinois",
}
STATE_TO_AGENCY = {
    "pennsylvania": "Middle States Commission on Higher Education (MSCHE)",
    "ohio": "Higher Learning Commission (HLC)",
    "michigan": "Higher Learning Commission (HLC)",
    "illinois": "Higher Learning Commission (HLC)",
}
STEM_KEYWORDS = [
    "mathematics", "math", "biology", "chemistry", "physics", "computer science", "cs", "computing"
]
SECONDARY_KEYWORDS = [
    "secondary", "grades 6-12", "6-12", "grades 7-12", "7-12", "grades 8-12", "8-12", "middle school", "high school",
    "adolescent/young adult", "adolescence to young adulthood", "aya", "secondary education"
]
UNDERGRAD_KEYWORDS = [
    "b.s.", "bs", "b.a.", "ba", "b.s.ed", "bsed", "b.a.ed", "baed", "b.ed", "bed", "bachelor", "undergraduate"
]
ACCEPTABLE_ACCRED_SOURCES = [
    "hlcommission.org", "msche.org"
]
ACCEPTABLE_CAEP_SOURCE = "caepnet.org"
ACCEPTABLE_FEDERAL_SOURCES = [
    "nces.ed.gov", "studentaid.gov", "ifap.ed.gov", "fsapartners.ed.gov", "collegenavigator.gov"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramInfo(BaseModel):
    name: Optional[str] = None
    grade_levels: Optional[str] = None


class UniversityItem(BaseModel):
    official_name: Optional[str] = None
    state: Optional[str] = None
    programs: List[ProgramInfo] = Field(default_factory=list)

    regional_accreditation_urls: List[str] = Field(default_factory=list)
    caep_urls: List[str] = Field(default_factory=list)
    program_details_urls: List[str] = Field(default_factory=list)
    federal_aid_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract all universities mentioned in the answer along with the required fields.
    For each university, return:
    - official_name: full official institution name as written
    - state: the state where the institution is located (full name or postal abbreviation)
    - programs: a list of undergraduate STEM teacher certification programs, each with:
        * name: the program name as written (e.g., "B.S.Ed. in Mathematics Education (Grades 7–12)")
        * grade_levels: the grade range or level stated (e.g., "Grades 7–12", "Secondary Education")
    - regional_accreditation_urls: URLs that document the institution's regional accreditation and status (HLC for OH/MI/IL; MSCHE for PA) or an official institutional accreditation page that clearly indicates the accreditor and current status
    - caep_urls: URLs that document current/active CAEP accreditation for at least one educator preparation program (official CAEP database or institutional accreditation page)
    - program_details_urls: URLs that document the specific STEM teacher certification program details and grade levels (official program/college catalog pages)
    - federal_aid_urls: URLs that document federal student aid eligibility (official federal sources like College Navigator (nces.ed.gov) or studentaid.gov, or official institutional disclosures)

    Notes:
    - Extract only URLs actually present in the answer (plain or markdown).
    - Do not invent URLs.
    - Include all URLs given for each category.
    - If a specific field is not present in the answer, return null (for strings) or an empty list (for arrays).
    - Do not deduplicate universities; list them in the order the answer uses.

    Return a JSON object with a single field:
    {
      "universities": [ ... up to all in the answer ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().lower()
    s = re.sub(r'[^a-z]', '', s)
    # map abbreviations
    if s in STATE_ABBR_MAP:
        return STATE_ABBR_MAP[s]
    # common full names
    if s in ALLOWED_STATES:
        return s
    # try partials like "commonwealthofpennsylvania" -> contains "pennsylvania"
    for st in ALLOWED_STATES:
        if st in s:
            return st
    return None


def is_state_allowed(state: Optional[str]) -> bool:
    norm = normalize_state_name(state)
    return norm in ALLOWED_STATES if norm else False


def determine_agency(state: Optional[str]) -> Optional[str]:
    norm = normalize_state_name(state)
    if not norm:
        return None
    return STATE_TO_AGENCY.get(norm)


def has_any_keyword(text: Optional[str], keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in keywords)


def program_list_has_stem_undergrad(programs: List[ProgramInfo]) -> bool:
    # At least one program name exists with a STEM subject; undergrad check is looser here (verified by URL)
    for p in programs:
        if p and p.name and has_any_keyword(p.name, STEM_KEYWORDS):
            return True
    return False


def programs_grade_levels_indicate_secondary(programs: List[ProgramInfo]) -> bool:
    # At least one program includes secondary-grade-level hints
    for p in programs:
        if p and p.grade_levels and has_any_keyword(p.grade_levels, SECONDARY_KEYWORDS):
            return True
    return False


def normalize_institution_name(name: Optional[str]) -> str:
    if not name:
        return ""
    n = name.lower()
    n = re.sub(r'&', 'and', n)
    n = re.sub(r'[^a-z0-9 ]+', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    # remove common words to reduce alias collisions
    stopwords = [
        "the", "university", "univ", "of", "at", "state", "college", "and", "system", "campus", "main", "campus"
    ]
    tokens = [tok for tok in n.split() if tok not in stopwords]
    return " ".join(tokens)


def count_distinct_universities(unis: List[UniversityItem]) -> int:
    keys = set()
    for u in unis:
        k = normalize_institution_name(u.official_name)
        if k:
            keys.add(k)
    return len(keys)


def provided_university_count(unis: List[UniversityItem]) -> int:
    return sum(1 for u in unis if (u.official_name and u.official_name.strip()))


def merge_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if u and (u not in seen):
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_global_requirements(evaluator: Evaluator, parent_node, extracted: UniversitiesExtraction) -> None:
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Submission satisfies global requirements about number/distinctness of universities",
        parent=parent_node,
        critical=True
    )

    # Exactly 4 universities provided in the answer (based on extracted items with a name)
    provided_count = provided_university_count(extracted.universities)
    evaluator.add_custom_node(
        result=(provided_count == 4),
        id="Provides_Exactly_4_Universities",
        desc="Response provides exactly 4 universities (not fewer or more)",
        parent=global_node,
        critical=True
    )

    # Universities are distinct (based on normalized names), and count is 4
    distinct_count = count_distinct_universities(extracted.universities)
    evaluator.add_custom_node(
        result=(distinct_count == 4 and provided_count == 4),
        id="Universities_Are_Distinct",
        desc="All 4 universities are distinct institutions (no duplicates/aliases of the same institution)",
        parent=global_node,
        critical=True
    )

    # Add some custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_total": len(extracted.universities),
            "provided_with_name": provided_count,
            "distinct_normalized": distinct_count
        },
        info_type="extraction_stats",
        info_name="global_counts"
    )


async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int
) -> None:
    uni_node = evaluator.add_parallel(
        id=f"University_{idx+1}",
        desc=f"University #{idx+1} meets all criteria and required reporting fields are provided",
        parent=parent_node,
        critical=False
    )

    # U{idx}_Official_Name_Provided
    evaluator.add_custom_node(
        result=(bool(uni.official_name) and uni.official_name.strip() != ""),
        id=f"U{idx+1}_Official_Name_Provided",
        desc="Provides the full official name of the institution",
        parent=uni_node,
        critical=True
    )

    # U{idx}_State_Provided_And_Allowed
    evaluator.add_custom_node(
        result=is_state_allowed(uni.state),
        id=f"U{idx+1}_State_Provided_And_Allowed",
        desc="Provides the state and it is one of: Pennsylvania, Ohio, Michigan, Illinois",
        parent=uni_node,
        critical=True
    )

    # U{idx}_Public_University (verify using regional acc URLs and/or program URLs and/or federal aid URLs)
    public_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Public_University",
        desc="Institution is a public university (not private and not a community college)",
        parent=uni_node,
        critical=True
    )
    public_sources = merge_sources(uni.regional_accreditation_urls, uni.program_details_urls, uni.federal_aid_urls)
    public_claim = f"{uni.official_name or 'The institution'} is a public university (state-funded), not a private university and not a community college."
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=public_sources,
        additional_instruction="Use only the provided URLs. Prefer official institutional or accreditor pages. If the accreditor directory lists 'Control: Public' or similar, that is sufficient. Do not accept Wikipedia or third-party aggregator sites."
    )

    # U{idx}_Regional_Accreditation_Current_And_Appropriate
    reg_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Regional_Accreditation_Current_And_Appropriate",
        desc="Holds current regional accreditation with the appropriate agency (HLC for IL/MI/OH; MSCHE for PA) and is not on probation/warning",
        parent=uni_node,
        critical=True
    )
    agency = determine_agency(uni.state)
    if agency is None:
        # If state not normalized, verification will likely fail
        reg_claim = f"{uni.official_name or 'The institution'} holds current regional accreditation appropriate for its state and is not on probation or warning."
    else:
        reg_claim = f"{uni.official_name or 'The institution'} holds current regional accreditation with {agency} and is not on probation or warning."
    await evaluator.verify(
        claim=reg_claim,
        node=reg_leaf,
        sources=uni.regional_accreditation_urls,
        additional_instruction="Acceptable sources: official institutional accreditation page or accreditor directory (HLC or MSCHE). The page should clearly indicate current/active status; if it explicitly lists 'Public' control and current status, that is sufficient. Reject Wikipedia or random third-party pages."
    )

    # U{idx}_CAEP_Accreditation_Current_Active
    caep_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_CAEP_Accreditation_Current_Active",
        desc="Has at least one educator preparation program with current/active CAEP accreditation",
        parent=uni_node,
        critical=True
    )
    caep_claim = f"{uni.official_name or 'The institution'} has at least one educator preparation program with current/active CAEP accreditation."
    await evaluator.verify(
        claim=caep_claim,
        node=caep_leaf,
        sources=uni.caep_urls,
        additional_instruction="Acceptable sources: CAEP database (caepnet.org) or an official institutional accreditation page that explicitly states CAEP accreditation. The status should be current/active."
    )

    # U{idx}_STEM_Teacher_Cert_Program_Provided
    evaluator.add_custom_node(
        result=program_list_has_stem_undergrad(uni.programs),
        id=f"U{idx+1}_STEM_Teacher_Cert_Program_Provided",
        desc="Provides the specific undergraduate (bachelor’s) teacher certification program name(s) in at least one STEM area: Mathematics Education, Science Education (Biology/Chemistry/Physics), or Computer Science Education",
        parent=uni_node,
        critical=True
    )

    # U{idx}_Grade_Levels_Provided_And_Secondary
    evaluator.add_custom_node(
        result=programs_grade_levels_indicate_secondary(uni.programs),
        id=f"U{idx+1}_Grade_Levels_Provided_And_Secondary",
        desc="Provides the grade levels the program prepares teachers to teach, and they correspond to secondary education (middle/high school; typically grades 6–12)",
        parent=uni_node,
        critical=True
    )

    # U{idx}_Federal_Student_Aid_Eligible
    federal_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Federal_Student_Aid_Eligible",
        desc="Institution is eligible to participate in federal student aid programs (per the stated constraints)",
        parent=uni_node,
        critical=True
    )
    federal_claim = f"{uni.official_name or 'The institution'} is eligible to participate in U.S. federal student aid programs (Title IV)."
    await evaluator.verify(
        claim=federal_claim,
        node=federal_leaf,
        sources=uni.federal_aid_urls,
        additional_instruction="Acceptable sources include official federal databases (e.g., College Navigator at nces.ed.gov, studentaid.gov) or official institutional disclosures that explicitly indicate participation in Title IV federal student aid programs."
    )

    # U{idx}_Clinical_Experience_Documented
    clinical_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Clinical_Experience_Documented",
        desc="Program includes required clinical experience components (e.g., field experience/student teaching) as per the stated constraints, and this is documented",
        parent=uni_node,
        critical=True
    )
    clinical_claim = "The cited program page(s) explicitly document required clinical experience components such as field experiences and/or student teaching for the teacher certification program."
    await evaluator.verify(
        claim=clinical_claim,
        node=clinical_leaf,
        sources=uni.program_details_urls,
        additional_instruction="Look for terms like 'clinical experience', 'field experience', 'practicum', 'student teaching', or equivalent on official program/institutional pages."
    )

    # U{idx}_Evidence_URLs (critical group)
    evidence_node = evaluator.add_parallel(
        id=f"U{idx+1}_Evidence_URLs",
        desc="Provides acceptable-source URLs supporting each required claim",
        parent=uni_node,
        critical=True
    )

    # Individual URL presence + acceptability verified with URL-based judgments
    reg_url_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Regional_Accreditation_URL",
        desc="Provides a URL from an acceptable source (official institutional site or HLC/MSCHE directory) documenting regional accreditation and status",
        parent=evidence_node,
        critical=True
    )
    reg_url_claim = "This page is an acceptable source (official institution or HLC/MSCHE) and documents the university's regional accreditation and current status."
    caep_url_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_CAEP_URL",
        desc="Provides a URL from an acceptable source (CAEP database or official institutional accreditation page) documenting CAEP accreditation",
        parent=evidence_node,
        critical=True
    )
    caep_url_claim = "This page is an acceptable source (CAEP database or official institutional accreditation page) and documents current/active CAEP accreditation."
    prog_url_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Program_Details_URL",
        desc="Provides a URL from an acceptable source (official institutional/program page) documenting the specific STEM program(s), grade levels, and relevant details",
        parent=evidence_node,
        critical=True
    )
    prog_url_claim = "This is an official institutional program or catalog page that documents the specific STEM teacher certification program details and grade levels."
    fed_url_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Federal_Aid_URL",
        desc="Provides a URL from an acceptable source (official federal database or official institutional disclosure) documenting federal student aid eligibility",
        parent=evidence_node,
        critical=True
    )
    fed_url_claim = "This is an acceptable official source (e.g., College Navigator (nces.ed.gov), studentaid.gov) or an official institutional disclosure page that documents federal student aid eligibility."

    # Batch verify all evidence URLs presence/acceptability
    claims_and_sources: List[Tuple[str, List[str], Any, Optional[str]]] = [
        (
            reg_url_claim,
            uni.regional_accreditation_urls,
            reg_url_leaf,
            "Accept only official institutional pages that clearly state the regional accreditor and status, or accreditor directories (HLC or MSCHE). Reject Wikipedia or informal blogs."
        ),
        (
            caep_url_claim,
            uni.caep_urls,
            caep_url_leaf,
            "Accept only CAEP database (caepnet.org) or official institutional accreditation page clearly indicating CAEP accreditation."
        ),
        (
            prog_url_claim,
            uni.program_details_urls,
            prog_url_leaf,
            "Must be an official institutional page (program site, catalog, college site) that explicitly describes the specific STEM teacher certification program and grade levels."
        ),
        (
            fed_url_claim,
            uni.federal_aid_urls,
            fed_url_leaf,
            "Accept official federal databases (nces.ed.gov, studentaid.gov, etc.) or official institutional disclosures indicating Title IV participation."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Extract universities from the answer
    extracted: UniversitiesExtraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Global requirements checks (based on what the answer actually provided)
    await build_global_requirements(evaluator, root, extracted)

    # Build per-university verification for the first 4 entries (pad if fewer)
    selected_unis: List[UniversityItem] = list(extracted.universities[:4])
    while len(selected_unis) < 4:
        selected_unis.append(UniversityItem())

    # Create university subtrees
    tasks = []
    for i, uni in enumerate(selected_unis):
        tasks.append(verify_university(evaluator, root, uni, i))
    await asyncio.gather(*tasks)

    # Add ground-truth/meta context for transparency
    evaluator.add_ground_truth({
        "allowed_states": sorted(list(ALLOWED_STATES)),
        "state_to_expected_agency": STATE_TO_AGENCY,
        "requirements_summary": {
            "public_university": True,
            "regional_accreditation": "MSCHE for PA; HLC for OH/MI/IL",
            "caep": "At least one current CAEP-accredited educator preparation program",
            "undergrad_stem_teacher_cert": ["Mathematics", "Biology", "Chemistry", "Physics", "Computer Science"],
            "secondary_grade_levels": "middle school/high school (e.g., grades 6–12, 7–12, 8–12)",
            "evidence_urls_required": ["regional accreditation", "CAEP", "program details", "federal aid eligibility"]
        }
    })

    return evaluator.get_summary()