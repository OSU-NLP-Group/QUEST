import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eastern_flagship_engineering"
TASK_DESCRIPTION = """I am a high school student planning to major in engineering at a large public university in the Eastern United States. I want to identify three public flagship state universities that meet all of the following criteria:

1. The university must be located in a state east of the Mississippi River
2. The university must be ranked in the top 30 public universities in the United States according to the US News 2026 rankings
3. The university must offer at least one ABET-accredited engineering program
4. The university must have at least three of the following engineering departments: Aerospace Engineering, Biomedical Engineering, Chemical Engineering, Civil Engineering, Computer Engineering, Electrical Engineering, or Mechanical Engineering
5. The university must have a published minimum or average GPA requirement for freshman admission of at least 3.0 on a 4.0 scale
6. The in-state tuition and fees for the 2024-2025 or 2025-2026 academic year must be under $15,000 per year
7. The total undergraduate enrollment must be at least 15,000 students
8. The university must offer study abroad or international exchange programs
9. The university must provide on-campus housing for undergraduate students
10. The university must have campus recreation or fitness center facilities
11. The university must offer an undergraduate honors program or honors college

For each of the three universities you identify, please provide:
- The university name
- Verification that it meets each of the 11 criteria listed above
- A reference URL supporting each criterion
"""

ALLOWED_ENGINEERING_DEPARTMENTS = [
    "Aerospace Engineering",
    "Biomedical Engineering",
    "Chemical Engineering",
    "Civil Engineering",
    "Computer Engineering",
    "Electrical Engineering",
    "Mechanical Engineering"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DepartmentEntry(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None


class UniversityEntry(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None

    # Basic/Institutional
    flagship_urls: List[str] = Field(default_factory=list)
    geographic_urls: List[str] = Field(default_factory=list)
    ranking_urls: List[str] = Field(default_factory=list)

    # Engineering
    abet_urls: List[str] = Field(default_factory=list)
    engineering_urls: List[str] = Field(default_factory=list)
    departments: List[DepartmentEntry] = Field(default_factory=list)

    # Admissions & Finance
    gpa: Optional[str] = None
    gpa_urls: List[str] = Field(default_factory=list)
    tuition: Optional[str] = None
    tuition_year: Optional[str] = None  # Expected values like "2024-2025" or "2025-2026"
    tuition_urls: List[str] = Field(default_factory=list)

    # Enrollment
    enrollment: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    # Student programs/facilities
    study_abroad_urls: List[str] = Field(default_factory=list)
    housing_urls: List[str] = Field(default_factory=list)
    recreation_urls: List[str] = Field(default_factory=list)
    honors_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to the first three universities from the answer and the supporting information for each of the 11 criteria. For each university, extract the following fields exactly as they appear in the answer:

- name: University name
- state: The U.S. state where the university is located (e.g., "Virginia", "Florida")
- flagship_urls: URLs (array) that explicitly support that the school is the state's public flagship university (e.g., official pages, system pages, Wikipedia)
- geographic_urls: URLs (array) that support the state/location (e.g., university 'About' page, Wikipedia, state page)
- ranking_urls: URLs (array) that support the U.S. News 2026 public university ranking claim for this university (should point to a ranking page or credible summary page that explicitly states the ranking/year)

Engineering:
- abet_urls: URLs (array) that show the school has at least one ABET-accredited engineering program (prefer abet.org program search or official accreditation pages)
- engineering_urls: URLs (array) that list engineering departments or programs (college of engineering overview page, departments page)
- departments: an array of objects for specific departments from the allowed set only (Aerospace Engineering, Biomedical Engineering, Chemical Engineering, Civil Engineering, Computer Engineering, Electrical Engineering, Mechanical Engineering). Each object should be:
  { "name": "<DEPARTMENT NAME>", "url": "<URL explicitly showing that department/program>" }
  Only include departments that are explicitly mentioned in the answer and clearly belong to the allowed set. If a combined department (e.g., "Electrical and Computer Engineering") appears, use that exact name and its page URL.

Admissions & Finance:
- gpa: The published minimum or average GPA for freshman admission as written (string, e.g., "3.4 average GPA", "minimum 3.0 GPA")
- gpa_urls: URLs (array) that support the GPA figure/requirement
- tuition: The in-state tuition and fees figure as written (string, e.g., "$13,500 per year")
- tuition_year: The academic year the tuition applies to (string, e.g., "2024-2025", "2025-2026") if present
- tuition_urls: URLs (array) that support the tuition and fees figure (tuition & fees pages; exclude room & board pages if possible)

Enrollment:
- enrollment: The total undergraduate enrollment as written (string, e.g., "25,000 undergraduates")
- enrollment_urls: URLs (array) supporting the undergraduate enrollment figure

Student programs & facilities:
- study_abroad_urls: URLs (array) that support existence of study abroad or international exchange programs
- housing_urls: URLs (array) that support on-campus housing availability for undergraduates
- recreation_urls: URLs (array) that support campus recreation/fitness centers
- honors_urls: URLs (array) that support the existence of an honors college or undergraduate honors program

Rules:
- Provide all URLs exactly as shown in the answer text. Do not invent or infer URLs.
- If a specific field is not present in the answer, set it to null (for strings) or an empty array (for arrays of URLs).
- Only include departments that are in the allowed set.
- Return a JSON object with a top-level "universities" array, each element having the fields listed above.
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _first_n_universities(extracted: UniversitiesExtraction, n: int = 3) -> List[UniversityEntry]:
    items = extracted.universities[:n]
    # Pad with empty entries if fewer than n
    while len(items) < n:
        items.append(UniversityEntry())
    return items


def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]


def _select_allowed_departments(departments: List[DepartmentEntry], max_count: int = 3) -> List[DepartmentEntry]:
    """
    Select up to max_count departments whose names align with the allowed set.
    We match liberally, allowing combined department names like "Electrical and Computer Engineering".
    """
    selected: List[DepartmentEntry] = []
    for d in departments:
        if not d or not d.name:
            continue
        name = d.name.strip()
        low = name.lower()
        # Accept if any allowed department keyword appears alongside "engineering"
        # e.g., "Electrical and Computer Engineering" should match both electrical and computer engineering
        for allowed in ALLOWED_ENGINEERING_DEPARTMENTS:
            allowed_key = allowed.lower().replace(" engineering", "")
            if "engineering" in low and allowed_key in low:
                selected.append(d)
                break
        if len(selected) >= max_count:
            break
    return selected


def _union_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged = []
    for urls in url_lists:
        for u in urls:
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# University verification builders                                            #
# --------------------------------------------------------------------------- #
async def _build_basic_institutional_criteria(
    evaluator: Evaluator,
    parent,
    uni: UniversityEntry,
    idx: int
):
    """
    Build 'U{idx}_Basic_Institutional_Criteria' subtree with 4 children:
    - Public flagship status (leaf, critical)
    - Geographic location east of Mississippi (leaf, critical)
    - Top 30 public ranking by US News 2026 (leaf, critical)
    - Ranking URL reference existence (custom, critical)
    """
    ulabel = f"U{idx}"
    uname = uni.name or f"University #{idx}"

    basic_node = evaluator.add_parallel(
        id=f"{ulabel}_Basic_Institutional_Criteria",
        desc="Verification of institution type, location, and ranking",
        parent=parent,
        critical=True
    )

    # Public Flagship Status
    flagship_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Public_Flagship_Status",
        desc="University is a public flagship state university",
        parent=basic_node,
        critical=True
    )
    flagship_claim = (
        f"{uname} is the public flagship state university (or flagship campus) of its state."
    )
    await evaluator.verify(
        claim=flagship_claim,
        node=flagship_leaf,
        sources=_non_empty_urls(uni.flagship_urls),
        additional_instruction="Verify that the page explicitly states the university is the state's flagship public university, or uses equivalent phrasing such as 'flagship campus'."
    )

    # Geographic Location (east of Mississippi)
    geo_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Geographic_Location",
        desc="University is located in a state east of the Mississippi River",
        parent=basic_node,
        critical=True
    )
    state_str = uni.state or "the relevant state"
    geo_claim = (
        f"{uname} is located in {state_str}, which is east of the Mississippi River."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=_non_empty_urls(uni.geographic_urls),
        additional_instruction="Use the page to confirm the state where the university is located. If the page indicates the state or region commonly understood to be in the Eastern United States (e.g., Northeast, Mid-Atlantic, Southeast), that supports being east of the Mississippi River."
    )

    # Top 30 Public Ranking (US News 2026)
    ranking_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Top_30_Public_Ranking",
        desc="University is ranked in top 30 public universities by US News 2026",
        parent=basic_node,
        critical=True
    )
    ranking_claim = (
        f"According to the U.S. News 2026 rankings of public universities, {uname} is in the top 30 public universities."
    )
    await evaluator.verify(
        claim=ranking_claim,
        node=ranking_leaf,
        sources=_non_empty_urls(uni.ranking_urls),
        additional_instruction="Verify the 2026 public university ranking. The claim is satisfied if {uname} ranks between 1 and 30 inclusive among public universities. Use official U.S. News pages or credible summaries that clearly state the ranking and year."
    )

    # Ranking URL Reference existence
    ranking_url_exists = evaluator.add_custom_node(
        result=len(_non_empty_urls(uni.ranking_urls)) > 0,
        id=f"{ulabel}_Ranking_URL_Reference",
        desc="Provide URL reference supporting the ranking claim",
        parent=basic_node,
        critical=True
    )

    return basic_node


async def _build_engineering_programs(
    evaluator: Evaluator,
    parent,
    uni: UniversityEntry,
    idx: int
):
    """
    Build 'U{idx}_Engineering_Programs' subtree with:
    - ABET accreditation (leaf, critical)
    - Has three departments (leaf, critical)
    - Engineering URL Reference existence (custom, critical)

    NOTE: The JSON placed Department_Listing as a non-critical child under a critical parent,
    which violates the framework constraint that critical parents cannot have non-critical children.
    To honor both the rubric intent and framework rules, we implement Department_Listing as a
    separate non-critical sibling group under the university node (see _build_department_listing()).
    """
    ulabel = f"U{idx}"
    uname = uni.name or f"University #{idx}"

    eng_node = evaluator.add_parallel(
        id=f"{ulabel}_Engineering_Programs",
        desc="Verification of engineering accreditation and department offerings",
        parent=parent,
        critical=True
    )

    # ABET Accreditation
    abet_leaf = evaluator.add_leaf(
        id=f"{ulabel}_ABET_Accreditation",
        desc="University offers at least one ABET-accredited engineering program",
        parent=eng_node,
        critical=True
    )
    abet_claim = (
        f"{uname} has at least one ABET-accredited engineering program."
    )
    await evaluator.verify(
        claim=abet_claim,
        node=abet_leaf,
        sources=_non_empty_urls(uni.abet_urls),
        additional_instruction="Confirm at least one ABET-accredited engineering (EAC) program. Prefer abet.org or official accreditation pages. Engineering technology (ETAC) alone does not satisfy the requirement unless the page clearly indicates it counts as an engineering program."
    )

    # Has at least three of specified departments
    selected_depts = _select_allowed_departments(uni.departments, max_count=3)
    dept_names = [d.name for d in selected_depts if d.name]
    dept_urls = _non_empty_urls([d.url for d in selected_depts if d.url])

    three_depts_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Has_Three_Departments",
        desc="University has at least three of the specified engineering departments (Aerospace, Biomedical, Chemical, Civil, Computer, Electrical, or Mechanical Engineering)",
        parent=eng_node,
        critical=True
    )
    depts_list_str = ", ".join([n for n in dept_names if n]) if dept_names else "no departments listed"
    three_depts_claim = (
        f"{uname} offers at least three of the specified engineering departments. The identified departments include: {depts_list_str}."
    )
    await evaluator.verify(
        claim=three_depts_claim,
        node=three_depts_leaf,
        sources=_union_sources(dept_urls, _non_empty_urls(uni.engineering_urls)),
        additional_instruction="Check official engineering college/department pages. Combined departments (e.g., 'Electrical and Computer Engineering', 'Mechanical and Aerospace Engineering') satisfy the relevant categories."
    )

    # Engineering URL Reference existence (accept any credible engineering program/department URLs)
    engineering_sources_union = _union_sources(
        _non_empty_urls(uni.engineering_urls),
        _non_empty_urls(uni.abet_urls),
        dept_urls
    )
    evaluator.add_custom_node(
        result=len(engineering_sources_union) > 0,
        id=f"{ulabel}_Engineering_URL_Reference",
        desc="Provide URL reference supporting engineering program information",
        parent=eng_node,
        critical=True
    )

    return eng_node


async def _build_department_listing(
    evaluator: Evaluator,
    parent,
    uni: UniversityEntry,
    idx: int
):
    """
    Build 'U{idx}_Department_Listing' as a non-critical sibling group under the university node,
    containing Department_1, Department_2, Department_3 as non-critical leaves.
    """
    ulabel = f"U{idx}"
    uname = uni.name or f"University #{idx}"

    dept_node = evaluator.add_parallel(
        id=f"{ulabel}_Department_Listing",
        desc="List of specific engineering departments offered",
        parent=parent,
        critical=False
    )

    selected_depts = _select_allowed_departments(uni.departments, max_count=3)
    # Ensure exactly 3 items for node creation
    while len(selected_depts) < 3:
        selected_depts.append(DepartmentEntry())

    for j in range(3):
        dep = selected_depts[j]
        dep_name = dep.name or f"Department #{j+1}"
        dep_url = dep.url

        leaf = evaluator.add_leaf(
            id=f"{ulabel}_Department_{j+1}",
            desc=f"{['First','Second','Third'][j]} engineering department from the specified list",
            parent=dept_node,
            critical=False
        )

        if dep_url:
            claim = f"{uname} offers {dep_name} as a department or program (possibly within a combined department)."
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=dep_url,
                additional_instruction="Verify on the linked department/program page that the program exists and is part of engineering. Combined department titles are acceptable."
            )
        else:
            # No URL provided; mark as failed via a direct verification without sources (will likely fail to support).
            # To avoid unsupported simple verification without sources, use a custom node to reflect missing info.
            # Replace the leaf with a deterministic failure using custom node:
            # Since leaf already added, set it failed explicitly by a simple unsupported claim:
            await evaluator.verify(
                claim=f"{uname} offers {dep_name} as a department or program.",
                node=leaf,
                sources=None,
                additional_instruction="There is no supporting URL provided; this claim should be considered unsupported."
            )

    return dept_node


async def _build_admission_financial(
    evaluator: Evaluator,
    parent,
    uni: UniversityEntry,
    idx: int
):
    """
    Build 'U{idx}_Admission_Financial_Criteria' with:
    - GPA_Minimum_3_0 (leaf, critical)
    - GPA_URL_Reference (custom, critical)
    - Tuition_Under_15000 (leaf, critical)
    - Tuition_URL_Reference (custom, critical)
    """
    ulabel = f"U{idx}"
    uname = uni.name or f"University #{idx}"

    adm_node = evaluator.add_parallel(
        id=f"{ulabel}_Admission_Financial_Criteria",
        desc="Verification of admission standards and tuition affordability",
        parent=parent,
        critical=True
    )

    # GPA >= 3.0 (4.0 scale)
    gpa_leaf = evaluator.add_leaf(
        id=f"{ulabel}_GPA_Minimum_3_0",
        desc="Published minimum or average GPA requirement is at least 3.0",
        parent=adm_node,
        critical=True
    )
    gpa_claim = (
        f"The published minimum or average GPA requirement for freshman admission at {uname} is at least 3.0 on a 4.0 scale."
    )
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=_non_empty_urls(uni.gpa_urls),
        additional_instruction="Accept either a stated minimum GPA >= 3.0 or an average admitted GPA >= 3.0 for freshmen on a 4.0 scale. If the page uses another scale without clear conversion, consider it not supported."
    )

    evaluator.add_custom_node(
        result=len(_non_empty_urls(uni.gpa_urls)) > 0,
        id=f"{ulabel}_GPA_URL_Reference",
        desc="Provide URL reference supporting GPA requirement",
        parent=adm_node,
        critical=True
    )

    # Tuition under $15,000 (2024-2025 or 2025-2026)
    tuition_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Tuition_Under_15000",
        desc="In-state tuition and fees for 2024-2025 or 2025-2026 is under $15,000",
        parent=adm_node,
        critical=True
    )
    year_str = uni.tuition_year or "2024-2025 or 2025-2026"
    tuition_claim = (
        f"The in-state undergraduate tuition and fees at {uname} for {year_str} are under $15,000 per year (excluding room and board)."
    )
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_leaf,
        sources=_non_empty_urls(uni.tuition_urls),
        additional_instruction="Verify the in-state undergraduate tuition and mandatory fees for either 2024-2025 or 2025-2026 are < $15,000 for the full academic year. Exclude room and board. If amounts are per semester/term, compute an annual total."
    )

    evaluator.add_custom_node(
        result=len(_non_empty_urls(uni.tuition_urls)) > 0,
        id=f"{ulabel}_Tuition_URL_Reference",
        desc="Provide URL reference supporting tuition information",
        parent=adm_node,
        critical=True
    )

    return adm_node


async def _build_enrollment_size(
    evaluator: Evaluator,
    parent,
    uni: UniversityEntry,
    idx: int
):
    """
    Build 'U{idx}_Enrollment_Size' with:
    - Undergrad >= 15,000 (leaf, critical)
    - Enrollment URL Reference (custom, critical)
    """
    ulabel = f"U{idx}"
    uname = uni.name or f"University #{idx}"

    enr_node = evaluator.add_parallel(
        id=f"{ulabel}_Enrollment_Size",
        desc="Verification of undergraduate enrollment size",
        parent=parent,
        critical=True
    )

    enroll_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Undergrad_At_Least_15000",
        desc="Total undergraduate enrollment is at least 15,000 students",
        parent=enr_node,
        critical=True
    )
    enroll_claim = (
        f"The total undergraduate enrollment at {uname} is at least 15,000 students."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=_non_empty_urls(uni.enrollment_urls),
        additional_instruction="Use official fact books, Common Data Set, or university stats pages that clearly identify undergraduate enrollment (not total headcount). Allow reasonable rounding."
    )

    evaluator.add_custom_node(
        result=len(_non_empty_urls(uni.enrollment_urls)) > 0,
        id=f"{ulabel}_Enrollment_URL_Reference",
        desc="Provide URL reference supporting enrollment data",
        parent=enr_node,
        critical=True
    )

    return enr_node


async def _build_student_programs_facilities(
    evaluator: Evaluator,
    parent,
    uni: UniversityEntry,
    idx: int
):
    """
    Build 'U{idx}_Student_Programs_Facilities' with the following pairs:
    - Study Abroad Existence (leaf, critical) + Study Abroad URL Reference (custom, critical)
    - Housing Availability (leaf, critical) + Housing URL Reference (custom, critical)
    - Recreation Existence (leaf, critical) + Recreation URL Reference (custom, critical)
    - Honors Existence (leaf, critical) + Honors URL Reference (custom, critical)
    """
    ulabel = f"U{idx}"
    uname = uni.name or f"University #{idx}"

    spf_node = evaluator.add_parallel(
        id=f"{ulabel}_Student_Programs_Facilities",
        desc="Verification of student programs and campus facilities",
        parent=parent,
        critical=True
    )

    # Study Abroad
    sa_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Study_Abroad_Existence",
        desc="University offers study abroad or international exchange programs",
        parent=spf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uname} offers study abroad or international exchange programs.",
        node=sa_leaf,
        sources=_non_empty_urls(uni.study_abroad_urls),
        additional_instruction="Confirm presence of study abroad or international exchange opportunities via global education/abroad office pages."
    )
    evaluator.add_custom_node(
        result=len(_non_empty_urls(uni.study_abroad_urls)) > 0,
        id=f"{ulabel}_Study_Abroad_URL_Reference",
        desc="Provide URL reference supporting study abroad programs",
        parent=spf_node,
        critical=True
    )

    # Housing
    housing_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Housing_Availability",
        desc="University provides on-campus housing for undergraduate students",
        parent=spf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uname} provides on-campus housing for undergraduate students.",
        node=housing_leaf,
        sources=_non_empty_urls(uni.housing_urls),
        additional_instruction="Confirm via housing/residential life pages that on-campus housing is available for undergraduates."
    )
    evaluator.add_custom_node(
        result=len(_non_empty_urls(uni.housing_urls)) > 0,
        id=f"{ulabel}_Housing_URL_Reference",
        desc="Provide URL reference supporting housing information",
        parent=spf_node,
        critical=True
    )

    # Recreation
    rec_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Recreation_Existence",
        desc="University has campus recreation or fitness center facilities",
        parent=spf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uname} has campus recreation or fitness center facilities.",
        node=rec_leaf,
        sources=_non_empty_urls(uni.recreation_urls),
        additional_instruction="Verify via campus recreation/fitness center pages that such facilities exist."
    )
    evaluator.add_custom_node(
        result=len(_non_empty_urls(uni.recreation_urls)) > 0,
        id=f"{ulabel}_Recreation_URL_Reference",
        desc="Provide URL reference supporting recreation facilities",
        parent=spf_node,
        critical=True
    )

    # Honors
    honors_leaf = evaluator.add_leaf(
        id=f"{ulabel}_Honors_Existence",
        desc="University offers an undergraduate honors program or honors college",
        parent=spf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uname} offers an undergraduate honors program or an honors college.",
        node=honors_leaf,
        sources=_non_empty_urls(uni.honors_urls),
        additional_instruction="Verify via an official honors college or undergraduate honors program page."
    )
    evaluator.add_custom_node(
        result=len(_non_empty_urls(uni.honors_urls)) > 0,
        id=f"{ulabel}_Honors_URL_Reference",
        desc="Provide URL reference supporting honors program",
        parent=spf_node,
        critical=True
    )

    return spf_node


async def _verify_university(
    evaluator: Evaluator,
    root_node,
    uni: UniversityEntry,
    idx1_based: int
):
    """
    Build and verify the subtree for one university.
    """
    # University container node (parallel, non-critical)
    uni_node = evaluator.add_parallel(
        id=f"University_{idx1_based}",
        desc=f"{['First','Second','Third'][idx1_based-1]} university meeting all specified criteria",
        parent=root_node,
        critical=False
    )

    idx_tag = idx1_based  # 1,2,3

    # Build all subtrees
    await _build_basic_institutional_criteria(evaluator, uni_node, uni, idx_tag)
    await _build_engineering_programs(evaluator, uni_node, uni, idx_tag)
    await _build_department_listing(evaluator, uni_node, uni, idx_tag)
    await _build_admission_financial(evaluator, uni_node, uni, idx_tag)
    await _build_enrollment_size(evaluator, uni_node, uni, idx_tag)
    await _build_student_programs_facilities(evaluator, uni_node, uni, idx_tag)


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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the answer for the 'eastern_flagship_engineering' task.
    Note: The JSON rubric marks the Root as critical, but the framework enforces that
    critical parents cannot have non-critical children. Since university nodes are
    non-critical (to allow partial credit across universities), we set the root as
    non-critical here to satisfy framework constraints while preserving rubric intent.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Task completion: Identify three qualifying public universities in the Eastern United States",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Record allowed departments info
    evaluator.add_custom_info(
        info={"allowed_engineering_departments": ALLOWED_ENGINEERING_DEPARTMENTS},
        info_type="policy",
        info_name="allowed_departments"
    )

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    universities = _first_n_universities(extraction, n=3)

    # Build and verify each university subtree
    for idx, uni in enumerate(universities, start=1):
        await _verify_university(evaluator, root, uni, idx)

    # Return evaluator summary
    return evaluator.get_summary()