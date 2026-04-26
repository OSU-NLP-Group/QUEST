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
TASK_ID = "msche_universities_4"
TASK_DESCRIPTION = """
Identify four universities in the United States that meet all of the following criteria for undergraduate education:

1. The university must be located in Michigan, Pennsylvania, New Jersey, Maryland, Delaware, or Washington, D.C.

2. The university must be accredited by the Middle States Commission on Higher Education (MSCHE)

3. The university must have an undergraduate enrollment between 8,000 and 18,000 students

4. The university must provide on-campus housing capacity for at least 3,000 students

5. The university must require a minimum of 120 credit hours for bachelor's degree completion

6. The university must require a minimum cumulative GPA of 2.00 for graduation

7. The university must require at least 4 years of English and at least 3 years of mathematics for undergraduate admission

8. The university must offer degree programs across at least 5 different academic colleges or schools

9. The university's main campus must be at least 200 acres in size

10. The university must have a published acceptance rate between 30% and 80%

11. The university's annual out-of-state tuition and fees for the 2025-26 academic year must be between $25,000 and $45,000

12. Can be either a public or private university

For each university, provide its name, location (city and state), and reference URLs that verify the key requirements (especially MSCHE accreditation status, enrollment figures, and campus size).
"""

ALLOWED_REGIONS = [
    "Michigan",
    "Pennsylvania",
    "New Jersey",
    "Maryland",
    "Delaware",
    "Washington, D.C.",
    "District of Columbia",
    "Washington DC",
    "DC",
]

ORDINALS = ["First", "Second", "Third", "Fourth"]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversitySources(BaseModel):
    location_urls: List[str] = Field(default_factory=list)
    accreditation_urls: List[str] = Field(default_factory=list)
    enrollment_urls: List[str] = Field(default_factory=list)
    housing_urls: List[str] = Field(default_factory=list)
    credits_urls: List[str] = Field(default_factory=list)
    graduation_gpa_urls: List[str] = Field(default_factory=list)
    hs_requirements_urls: List[str] = Field(default_factory=list)
    academic_colleges_urls: List[str] = Field(default_factory=list)
    campus_size_urls: List[str] = Field(default_factory=list)
    acceptance_rate_urls: List[str] = Field(default_factory=list)
    tuition_urls: List[str] = Field(default_factory=list)
    general_urls: List[str] = Field(default_factory=list)


class UniversityItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    sources: UniversitySources = Field(default_factory=UniversitySources)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four universities listed in the answer that the responder claims meet the specified criteria. For each university, return the following fields:

    - name: The official university name as provided in the answer.
    - city: The city of the university's main campus as stated.
    - state: The state (or Washington, D.C.) of the university's main campus.
    - sources: A nested object containing URL arrays relevant to each criterion. Extract only explicit URLs mentioned in the answer. If a specific type of source URL is not provided, return an empty array for that type.
        • location_urls: URLs that can verify the location (city/state) of the main campus.
        • accreditation_urls: URLs that can verify MSCHE accreditation (prefer the MSCHE institution directory page or official accreditation page).
        • enrollment_urls: URLs that state undergraduate enrollment numbers.
        • housing_urls: URLs that mention on-campus housing capacity (beds).
        • credits_urls: URLs that state the minimum credits required for a bachelor's degree.
        • graduation_gpa_urls: URLs that state the minimum cumulative GPA required for graduation.
        • hs_requirements_urls: URLs that list high-school course requirements for freshman admission (English and Math years).
        • academic_colleges_urls: URLs that enumerate academic colleges/schools (used to count if there are at least 5).
        • campus_size_urls: URLs that state main campus size in acres.
        • acceptance_rate_urls: URLs that provide a published acceptance rate.
        • tuition_urls: URLs that list annual out-of-state tuition and fees for the 2025–26 academic year.
        • general_urls: Any other URLs cited for the university (e.g., main facts page, Wikipedia, admissions page) that may help support multiple criteria.

    Rules:
    - Return null for missing name/city/state values if the answer does not provide them.
    - For each URL field, include all URLs explicitly cited in the answer text (including those in markdown link format).
    - Do not invent URLs. If none are provided for a field, return an empty array.
    - If the answer lists more than four universities, include only the first four.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Use primary URLs if present; otherwise, use fallback list."""
    return primary if (primary and len(primary) > 0) else fallback


def _safe(val: Optional[str]) -> str:
    return val or ""


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
) -> None:
    """
    Build verification subtree for one university and execute checks.
    All leaves are critical under the university node because the rubric requires
    each university to meet all specified criteria.
    """
    ordinal = ORDINALS[idx] if idx < len(ORDINALS) else f"University #{idx + 1}"

    # University aggregator (parallel, critical under critical root)
    uni_node = evaluator.add_parallel(
        id=f"university_{idx + 1}",
        desc=f"{ordinal} university meeting all requirements",
        parent=parent_node,
        critical=True,
    )

    name = _safe(uni.name)
    city = _safe(uni.city)
    state = _safe(uni.state)
    s = uni.sources

    # Prepare leaf nodes
    # 1) Location check
    loc_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_location",
        desc="University is located in Michigan, Pennsylvania, New Jersey, Maryland, Delaware, or Washington, D.C.",
        parent=uni_node,
        critical=True,
    )
    loc_claim = (
        f'The university "{name}" is located in {city}, {state}, and the state/region '
        f'is one of the allowed regions: Michigan, Pennsylvania, New Jersey, Maryland, Delaware, or Washington, D.C.'
    )
    loc_sources = _merge_sources(s.location_urls, s.general_urls)

    # 2) MSCHE accreditation
    accred_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_msche_accreditation",
        desc="University is accredited by the Middle States Commission on Higher Education (MSCHE)",
        parent=uni_node,
        critical=True,
    )
    accred_claim = f'The university "{name}" is accredited by the Middle States Commission on Higher Education (MSCHE).'
    accred_sources = _merge_sources(s.accreditation_urls, s.general_urls)

    # 3) MSCHE accreditation reference check
    accred_ref_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_msche_accreditation_reference",
        desc="Provide URL reference confirming MSCHE accreditation status",
        parent=uni_node,
        critical=True,
    )
    accred_ref_claim = (
        f'The provided source(s) explicitly confirm that "{name}" is accredited by MSCHE (e.g., MSCHE institution directory or official accreditation page).'
    )
    accred_ref_sources = _merge_sources(s.accreditation_urls, s.general_urls)

    # 4) Undergraduate enrollment range
    enroll_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_undergraduate_enrollment",
        desc="University has undergraduate enrollment between 8,000 and 18,000 students",
        parent=uni_node,
        critical=True,
    )
    enroll_claim = f'The university "{name}" has undergraduate enrollment between 8,000 and 18,000 students.'
    enroll_sources = _merge_sources(s.enrollment_urls, s.general_urls)

    # 5) Housing capacity
    housing_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_housing_capacity",
        desc="University provides on-campus housing capacity for at least 3,000 students",
        parent=uni_node,
        critical=True,
    )
    housing_claim = f'The university "{name}" provides on-campus housing capacity for at least 3,000 students.'
    housing_sources = _merge_sources(s.housing_urls, s.general_urls)

    # 6) Bachelor's minimum credit hours
    credits_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_bachelor_credit_requirement",
        desc="University requires a minimum of 120 credit hours for bachelor's degree completion",
        parent=uni_node,
        critical=True,
    )
    credits_claim = f'The university "{name}" requires at least 120 credit hours to earn a bachelor\'s degree.'
    credits_sources = _merge_sources(s.credits_urls, s.general_urls)

    # 7) Graduation GPA minimum
    gpa_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_graduation_gpa",
        desc="University requires a minimum cumulative GPA of 2.00 for graduation",
        parent=uni_node,
        critical=True,
    )
    gpa_claim = f'The university "{name}" requires a minimum cumulative GPA of 2.00 for graduation.'
    gpa_sources = _merge_sources(s.graduation_gpa_urls, s.general_urls)

    # 8) HS requirements (English & Math)
    hs_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_high_school_requirements",
        desc="University requires at least 4 years of English and 3 years of mathematics for undergraduate admission",
        parent=uni_node,
        critical=True,
    )
    hs_claim = (
        f'For freshman admission, the university "{name}" requires at least 4 years of English and at least 3 years of mathematics.'
    )
    hs_sources = _merge_sources(s.hs_requirements_urls, s.general_urls)

    # 9) Academic colleges >= 5
    colleges_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_academic_colleges",
        desc="University offers degree programs across at least 5 different academic colleges or schools",
        parent=uni_node,
        critical=True,
    )
    colleges_claim = (
        f'The university "{name}" offers degree programs across at least 5 distinct academic colleges or schools.'
    )
    colleges_sources = _merge_sources(s.academic_colleges_urls, s.general_urls)

    # 10) Campus size >= 200 acres
    campus_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_campus_size",
        desc="University has a main campus of at least 200 acres",
        parent=uni_node,
        critical=True,
    )
    campus_claim = f'The main campus of the university "{name}" is at least 200 acres in size.'
    campus_sources = _merge_sources(s.campus_size_urls, s.general_urls)

    # 11) Acceptance rate 30%–80%
    accept_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_acceptance_rate",
        desc="University has a published acceptance rate between 30% and 80%",
        parent=uni_node,
        critical=True,
    )
    accept_claim = f'The university "{name}" has a published acceptance rate between 30% and 80%.'
    accept_sources = _merge_sources(s.acceptance_rate_urls, s.general_urls)

    # 12) Out-of-state tuition & fees for 2025–26 between $25k and $45k
    tuition_node = evaluator.add_leaf(
        id=f"university_{idx + 1}_tuition_range",
        desc="Annual out-of-state tuition and fees for 2025-26 academic year are between $25,000 and $45,000",
        parent=uni_node,
        critical=True,
    )
    tuition_claim = (
        f'For the 2025–26 academic year, the university "{name}" lists annual out-of-state tuition and mandatory fees '
        f'between $25,000 and $45,000.'
    )
    tuition_sources = _merge_sources(s.tuition_urls, s.general_urls)

    # Prepare batch verifications for parallel execution within the university node
    claims_and_sources = [
        (
            loc_claim,
            loc_sources,
            loc_node,
            "Verify the main campus location (city/state). Accept state abbreviations and DC synonyms (Washington, D.C., District of Columbia, DC). "
            "Besides confirming the location, ensure the state/region is one of the allowed: Michigan, Pennsylvania, New Jersey, Maryland, Delaware, Washington, D.C.",
        ),
        (
            accred_claim,
            accred_sources,
            accred_node,
            "Prefer official MSCHE directory or the university's accreditation page. The page must explicitly indicate MSCHE accreditation.",
        ),
        (
            accred_ref_claim,
            accred_ref_sources,
            accred_ref_node,
            "This specifically checks that the cited URL(s) confirm MSCHE accreditation (e.g., MSCHE institution directory entry for this university).",
        ),
        (
            enroll_claim,
            enroll_sources,
            enroll_node,
            "Confirm UNDERGRADUATE enrollment. If multiple numbers are shown, focus on undergraduate headcount. "
            "Ranges, approximations, or recent-year figures are acceptable if clearly between 8,000 and 18,000.",
        ),
        (
            housing_claim,
            housing_sources,
            housing_node,
            "Confirm on-campus housing capacity (beds). Statements like 'housing capacity of X beds' or 'can house X students' should be used.",
        ),
        (
            credits_claim,
            credits_sources,
            credits_node,
            "Confirm bachelor's degree minimum credits (often 120 credits/units/hours). Policy pages, catalogs, or registrar pages are suitable.",
        ),
        (
            gpa_claim,
            gpa_sources,
            gpa_node,
            "Confirm graduation GPA threshold (minimum cumulative GPA of 2.00). Use catalog, registrar, or policy pages.",
        ),
        (
            hs_claim,
            hs_sources,
            hs_node,
            "Confirm high-school course requirements for freshman admission: at least 4 years (or equivalent units/credits) of English and at least 3 years of mathematics.",
        ),
        (
            colleges_claim,
            colleges_sources,
            colleges_node,
            "Confirm there are at least 5 distinct academic colleges/schools (e.g., College of Engineering, Arts & Sciences, Business, Education, Health, etc.). "
            "Department lists are not sufficient; count colleges/schools or equivalent units conferring degrees.",
        ),
        (
            campus_claim,
            campus_sources,
            campus_node,
            "Confirm main campus size (in acres). Wikipedia or official campus facts pages are acceptable.",
        ),
        (
            accept_claim,
            accept_sources,
            accept_node,
            "Confirm a published acceptance rate between 30% and 80%. Use credible sources (institutional reports, recognized aggregators). Allow minor rounding differences.",
        ),
        (
            tuition_claim,
            tuition_sources,
            tuition_node,
            "Confirm ANNUAL out-of-state tuition and mandatory fees for the 2025–26 academic year fall between $25,000 and $45,000. "
            "Do not include room/board. If only a different academic year is listed and 2025–26 cannot be confirmed, consider this not supported.",
        ),
    ]

    # Execute batch verification
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'four MSCHE universities meeting comprehensive criteria' task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates four universities independently
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
    # Make root critical to reflect rubric requirement (all four must meet all criteria)
    evaluator.root.critical = True

    # Extract universities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Record ground-truth policy info (constraints to be checked)
    evaluator.add_ground_truth({
        "allowed_regions": ALLOWED_REGIONS,
        "criteria_summary": [
            "MSCHE accreditation required",
            "Undergraduate enrollment: 8,000–18,000",
            "On-campus housing capacity: >= 3,000",
            "Bachelor's minimum credits: >= 120",
            "Graduation GPA minimum: 2.00",
            "HS course requirements: >= 4 years English and >= 3 years Mathematics",
            "Academic colleges/schools: >= 5",
            "Main campus size: >= 200 acres",
            "Acceptance rate: 30%–80%",
            "Out-of-state tuition & fees (2025–26): $25,000–$45,000",
        ]
    })

    # Use only the first four universities; pad with empty placeholders if fewer
    universities = extracted.universities[:4]
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Build and verify each university subtree (critical under root)
    tasks = []
    for i in range(4):
        tasks.append(verify_university(evaluator, root, universities[i], i))
    await asyncio.gather(*tasks)

    # Return structured summary
    return evaluator.get_summary()