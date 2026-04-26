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
TASK_ID = "big_ten_midwest_public_uni_3"
TASK_DESCRIPTION = (
    "Identify 3 public universities that are current members of the Big Ten Conference, have NCAA Division I FBS football programs, "
    "and have undergraduate enrollment between 30,000 and 50,000 students. For each university, provide the following information: "
    "(1) University name and location (city, state), (2) Confirmation of Big Ten Conference membership with reference URL, "
    "(3) Confirmation of public institution status, (4) Verification of NCAA Division I FBS football program with reference URL, "
    "(5) Current undergraduate enrollment count, (6) Current total enrollment (undergraduate + graduate), "
    "(7) Confirmation of state flagship status with reference URL, (8) Verification of at least one ABET-accredited engineering program with reference URL, "
    "(9) Confirmation that the university offers graduate degree programs, (10) Confirmation that the university provides on-campus housing, "
    "(11) Fall 2026 regular decision application deadline, and (12) In-state undergraduate tuition and fees for the 2025-2026 academic year. "
    "Each piece of information must be supported by reference URLs from official university websites or reliable sources."
)

# Midwest states we accept for the “Midwestern United States” constraint
MIDWEST_STATES = {
    "illinois", "indiana", "iowa", "kansas", "michigan", "minnesota",
    "missouri", "nebraska", "north dakota", "ohio", "south dakota", "wisconsin"
}

# Allowed official authority domains (besides .edu university domains) for source policy objective
ALLOWED_AUTHORITY_DOMAINS = {
    "bigten.org",
    "ncaa.com", "ncaa.org",
    "abet.org",
    "carnegieclassifications.acenet.edu",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Basic identification
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    name_location_urls: List[str] = Field(default_factory=list)

    # Constraints and attributes with sources
    big_ten_member_urls: List[str] = Field(default_factory=list)
    public_status_urls: List[str] = Field(default_factory=list)
    ncaa_fbs_urls: List[str] = Field(default_factory=list)

    undergrad_enrollment: Optional[str] = None
    undergrad_enrollment_urls: List[str] = Field(default_factory=list)

    total_enrollment: Optional[str] = None
    total_enrollment_urls: List[str] = Field(default_factory=list)

    flagship_urls: List[str] = Field(default_factory=list)
    abet_engineering_urls: List[str] = Field(default_factory=list)
    grad_degrees_urls: List[str] = Field(default_factory=list)
    housing_urls: List[str] = Field(default_factory=list)
    research_class_urls: List[str] = Field(default_factory=list)

    active_fall_2026_admissions_urls: List[str] = Field(default_factory=list)

    regular_deadline_fall_2026: Optional[str] = None
    regular_deadline_urls: List[str] = Field(default_factory=list)

    in_state_tuition_2025_2026: Optional[str] = None
    tuition_urls: List[str] = Field(default_factory=list)

    # Convenience: union of all reference URLs cited for this university
    all_reference_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract all universities mentioned in the answer. For each university, provide the following fields exactly as present in the answer:

    - name: university name (string)
    - city: city location (string)
    - state: state location (string)
    - name_location_urls: array of URLs that support the stated name/location (prefer official university .edu pages)

    - big_ten_member_urls: array of URLs that support current Big Ten Conference membership (prefer bigten.org or the university’s official athletics site)
    - public_status_urls: array of URLs that explicitly state the university is public (e.g., "public research university") from official .edu or trusted sources
    - ncaa_fbs_urls: array of URLs that support an NCAA Division I FBS football program (prefer ncaa.com/.org or official athletics pages)

    - undergrad_enrollment: the current undergraduate enrollment count as a string (do not normalize; extract exactly as written; include thousands separators if present)
    - undergrad_enrollment_urls: array of URLs that support the undergraduate enrollment figure
    - total_enrollment: the current total enrollment (undergraduate + graduate) as a string
    - total_enrollment_urls: array of URLs that support the total enrollment figure

    - flagship_urls: array of URLs that state the university is the state’s flagship public university
    - abet_engineering_urls: array of URLs that show at least one ABET-accredited engineering program (prefer abet.org or the school’s accreditation page)
    - grad_degrees_urls: array of URLs that confirm the university offers graduate degree programs (master's and/or doctoral)
    - housing_urls: array of URLs that confirm the university provides on-campus housing
    - research_class_urls: array of URLs that confirm classification as a research university (e.g., Carnegie R1). Prefer official university pages or Carnegie Classification site.

    - active_fall_2026_admissions_urls: array of URLs that show undergraduate admissions for Fall 2026 are active/accepting
    - regular_deadline_fall_2026: the regular decision application deadline string for Fall 2026 (extract exactly as written)
    - regular_deadline_urls: array of URLs that support the stated Fall 2026 regular decision application deadline

    - in_state_tuition_2025_2026: the in-state undergraduate tuition and fees for the 2025–2026 academic year as a string (extract exactly as written)
    - tuition_urls: array of URLs that support the 2025–2026 in-state tuition and fees

    - all_reference_urls: array containing all URLs cited for the above fields for this university (duplicates allowed)

    Return a JSON object:
    {
      "universities": [ { ... }, { ... }, ... ]
    }

    IMPORTANT:
    - Only extract URLs explicitly present in the answer. Do not invent or infer any URLs.
    - If a field is not provided for a university, set it to null (for strings) or [] (for arrays).
    - Include all universities listed by the answer, even if more than 3; we will evaluate only the first 3.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def normalize_university_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(ch.lower() for ch in name if ch.isalnum() or ch.isspace()).strip()


def is_midwest_state(state_name: Optional[str]) -> bool:
    if not state_name:
        return False
    return state_name.strip().lower() in MIDWEST_STATES


def get_all_urls_for_university(u: UniversityItem) -> List[str]:
    urls = []
    urls.extend(u.name_location_urls)
    urls.extend(u.big_ten_member_urls)
    urls.extend(u.public_status_urls)
    urls.extend(u.ncaa_fbs_urls)
    urls.extend(u.undergrad_enrollment_urls)
    urls.extend(u.total_enrollment_urls)
    urls.extend(u.flagship_urls)
    urls.extend(u.abet_engineering_urls)
    urls.extend(u.grad_degrees_urls)
    urls.extend(u.housing_urls)
    urls.extend(u.research_class_urls)
    urls.extend(u.active_fall_2026_admissions_urls)
    urls.extend(u.regular_deadline_urls)
    urls.extend(u.tuition_urls)
    # If extractor already provided union, include it too
    urls.extend(u.all_reference_urls)
    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for url in urls:
        if url not in seen:
            dedup.append(url)
            seen.add(url)
    return dedup


def domain_from_url(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def is_allowed_source_domain(domain: str) -> bool:
    if not domain:
        return False
    # Accept all .edu domains (including subdomains)
    if domain.endswith(".edu"):
        return True
    # Accept explicitly allowed authority domains
    if domain in ALLOWED_AUTHORITY_DOMAINS:
        return True
    # Accept subdomains of allowed domains (e.g., sub.ncaa.org)
    for allowed in ALLOWED_AUTHORITY_DOMAINS:
        if domain.endswith("." + allowed):
            return True
    return False


def check_source_policy(urls: List[str]) -> bool:
    if not urls:
        return False
    return all(is_allowed_source_domain(domain_from_url(u)) for u in urls)


# --------------------------------------------------------------------------- #
# Verification logic for one university                                       #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    u: UniversityItem,
    index: int,
) -> None:
    uni_node = evaluator.add_parallel(
        id=f"university_{index+1}",
        desc=f"University #{index+1} meets all constraints and includes all required details with supporting reference URLs.",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit across different universities
    )

    uni_name = u.name or ""
    city = u.city or ""
    state = u.state or ""
    all_urls = get_all_urls_for_university(u)

    # 1. Name and location with source
    name_loc_leaf = evaluator.add_leaf(
        id=f"u{index+1}_name_and_location_with_source",
        desc="Provides the university name and location (city, state) with a supporting reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"The university '{uni_name}' is located in {city}, {state}."
    await evaluator.verify(
        claim=claim,
        node=name_loc_leaf,
        sources=u.name_location_urls,
        additional_instruction="Confirm the name and the city/state on the official university page or a reliable source."
    )

    # 2. Midwestern state constraint (programmatic check)
    midwest_leaf = evaluator.add_custom_node(
        result=is_midwest_state(u.state),
        id=f"u{index+1}_midwestern_state_constraint",
        desc="Confirms the university is located in a Midwestern United States state.",
        parent=uni_node,
        critical=True
    )

    # 3. Big Ten membership
    bigten_leaf = evaluator.add_leaf(
        id=f"u{index+1}_big_ten_membership",
        desc="Confirms current Big Ten Conference membership with a reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"'{uni_name}' is a current member of the Big Ten Conference."
    await evaluator.verify(
        claim=claim,
        node=bigten_leaf,
        sources=u.big_ten_member_urls,
        additional_instruction="Prefer bigten.org membership pages or official athletics pages that explicitly state Big Ten membership."
    )

    # 4. Public institution status
    public_leaf = evaluator.add_leaf(
        id=f"u{index+1}_public_institution_status",
        desc="Confirms the university is public (not private) with a reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"'{uni_name}' is a public university."
    await evaluator.verify(
        claim=claim,
        node=public_leaf,
        sources=u.public_status_urls,
        additional_instruction="Look for phrases like 'public research university' on official .edu pages (About/Facts)."
    )

    # 5. NCAA Division I FBS football
    fbs_leaf = evaluator.add_leaf(
        id=f"u{index+1}_ncaa_division_i_fbs_football",
        desc="Verifies an NCAA Division I FBS football program with a reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"'{uni_name}' fields a football program competing in NCAA Division I FBS."
    await evaluator.verify(
        claim=claim,
        node=fbs_leaf,
        sources=u.ncaa_fbs_urls,
        additional_instruction="Prefer NCAA pages or official athletics pages explicitly stating 'NCAA Division I FBS' for the football program."
    )

    # 6. Undergraduate enrollment and range [30,000, 50,000]
    ug_leaf = evaluator.add_leaf(
        id=f"u{index+1}_undergrad_enrollment_and_range",
        desc="Provides current undergraduate enrollment count with a reference URL, and the value is between 30,000 and 50,000 (inclusive).",
        parent=uni_node,
        critical=True
    )
    ug_val = u.undergrad_enrollment or ""
    claim = (
        f"The current undergraduate enrollment at '{uni_name}' is '{ug_val}', and this figure lies between 30,000 and 50,000 (inclusive)."
    )
    await evaluator.verify(
        claim=claim,
        node=ug_leaf,
        sources=u.undergrad_enrollment_urls,
        additional_instruction=(
            "Confirm the undergraduate enrollment figure (not total) from official sources; also check that the numeric value falls within [30,000, 50,000]."
        )
    )

    # 7. Total enrollment exceeds 40,000
    total_leaf = evaluator.add_leaf(
        id=f"u{index+1}_total_enrollment_and_threshold",
        desc="Provides current total enrollment (undergraduate + graduate) with a reference URL, and the value exceeds 40,000.",
        parent=uni_node,
        critical=True
    )
    total_val = u.total_enrollment or ""
    claim = (
        f"The current total enrollment (undergraduate + graduate) at '{uni_name}' is '{total_val}', and this value exceeds 40,000."
    )
    await evaluator.verify(
        claim=claim,
        node=total_leaf,
        sources=u.total_enrollment_urls,
        additional_instruction="Confirm total enrollment from official sources and check it is greater than 40,000."
    )

    # 8. State flagship status
    flagship_leaf = evaluator.add_leaf(
        id=f"u{index+1}_state_flagship_status",
        desc="Confirms state flagship public university status with a reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"'{uni_name}' is the flagship public university of {state}."
    await evaluator.verify(
        claim=claim,
        node=flagship_leaf,
        sources=u.flagship_urls,
        additional_instruction="Look for explicit 'flagship' designation on official pages or credible state/university resources."
    )

    # 9. ABET-accredited engineering program
    abet_leaf = evaluator.add_leaf(
        id=f"u{index+1}_abet_accredited_engineering",
        desc="Verifies at least one ABET-accredited engineering program with a reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"At least one engineering program at '{uni_name}' is accredited by ABET."
    await evaluator.verify(
        claim=claim,
        node=abet_leaf,
        sources=u.abet_engineering_urls,
        additional_instruction="Prefer abet.org accreditation search entries or the school's accreditation page stating ABET accreditation."
    )

    # 10. Offers graduate degrees
    grad_leaf = evaluator.add_leaf(
        id=f"u{index+1}_offers_graduate_degrees",
        desc="Confirms the university offers graduate degree programs (master's and/or doctoral) with a reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"'{uni_name}' offers graduate degree programs."
    await evaluator.verify(
        claim=claim,
        node=grad_leaf,
        sources=u.grad_degrees_urls,
        additional_instruction="Graduate school or programs pages should explicitly confirm master's and/or doctoral offerings."
    )

    # 11. On-campus housing
    housing_leaf = evaluator.add_leaf(
        id=f"u{index+1}_on_campus_housing",
        desc="Confirms the university provides on-campus housing with a reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"'{uni_name}' provides on-campus housing."
    await evaluator.verify(
        claim=claim,
        node=housing_leaf,
        sources=u.housing_urls,
        additional_instruction="Residence life/housing pages should explicitly confirm on-campus housing availability."
    )

    # 12. Research university classification
    research_leaf = evaluator.add_leaf(
        id=f"u{index+1}_research_university_classification",
        desc="Confirms the university is classified as a research university with a reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"'{uni_name}' is classified as a research university (e.g., Carnegie R1 or similar)."
    await evaluator.verify(
        claim=claim,
        node=research_leaf,
        sources=u.research_class_urls,
        additional_instruction="Prefer Carnegie Classification pages or official university pages explicitly stating 'research university' classification."
    )

    # 13. Active Fall 2026 admissions
    active_fall_leaf = evaluator.add_leaf(
        id=f"u{index+1}_active_fall_2026_admissions",
        desc="Confirms an active undergraduate admissions process accepting applications for Fall 2026 with a reference URL.",
        parent=uni_node,
        critical=True
    )
    claim = f"'{uni_name}' is currently accepting undergraduate applications for Fall 2026."
    await evaluator.verify(
        claim=claim,
        node=active_fall_leaf,
        sources=u.active_fall_2026_admissions_urls,
        additional_instruction="Admissions pages should indicate active intake or application windows for Fall 2026."
    )

    # 14. Fall 2026 regular decision deadline
    deadline_leaf = evaluator.add_leaf(
        id=f"u{index+1}_fall_2026_regular_decision_deadline",
        desc="Provides the Fall 2026 regular decision application deadline with a reference URL.",
        parent=uni_node,
        critical=True
    )
    deadline_val = u.regular_deadline_fall_2026 or ""
    claim = f"The Fall 2026 regular decision application deadline at '{uni_name}' is '{deadline_val}'."
    await evaluator.verify(
        claim=claim,
        node=deadline_leaf,
        sources=u.regular_deadline_urls,
        additional_instruction="Admissions/application deadlines page should explicitly list the regular decision deadline for Fall 2026."
    )

    # 15. In-state tuition 2025–2026
    tuition_leaf = evaluator.add_leaf(
        id=f"u{index+1}_in_state_tuition_2025_2026",
        desc="Provides in-state undergraduate tuition and fees for the 2025–2026 academic year with a reference URL.",
        parent=uni_node,
        critical=True
    )
    tuition_val = u.in_state_tuition_2025_2026 or ""
    claim = f"The in-state undergraduate tuition and fees for the 2025–2026 academic year at '{uni_name}' are '{tuition_val}'."
    await evaluator.verify(
        claim=claim,
        node=tuition_leaf,
        sources=u.tuition_urls,
        additional_instruction="Tuition & fees page should list 2025–2026 in-state undergraduate tuition and fees; confirm the figure exactly."
    )

    # 16. Source policy objective (programmatic domain check)
    source_policy_leaf = evaluator.add_custom_node(
        result=check_source_policy(all_urls),
        id=f"u{index+1}_source_policy_objective",
        desc="All reference URLs for this university are from official university-controlled .edu sites or official bodies (Big Ten, NCAA, ABET, Carnegie).",
        parent=uni_node,
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
    """
    Evaluate an answer for the Big Ten Midwest public universities task.

    Notes:
    - The provided rubric marks the root node as critical, but the framework enforces that
      critical parents must have all-critical children. To allow partial credit across the
      three universities (non-critical by rubric), we initialize a non-critical root.
    - If the answer lists more than 3 universities, we only evaluate the first 3 (per guidance).
    """
    # Initialize evaluator (root is non-critical by design in Evaluator.initialize)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks across universities + count check
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Record Midwest policy and allowed domains for transparency
    evaluator.add_custom_info(
        info={"midwest_states_considered": sorted(list(MIDWEST_STATES))},
        info_type="policy",
        info_name="midwest_states_policy"
    )
    evaluator.add_custom_info(
        info={"allowed_authority_domains": sorted(list(ALLOWED_AUTHORITY_DOMAINS))},
        info_type="policy",
        info_name="source_policy_allowed_domains"
    )

    # University count and distinctness (custom check; len>=3 and first-3 unique)
    uni_count_node = evaluator.add_custom_node(
        result=(
            len(extraction.universities) >= 3 and
            len({normalize_university_name(u.name) for u in extraction.universities[:3] if u.name}) == 3
        ),
        id="university_count_and_distinctness",
        desc="At least 3 universities are provided and the first 3 are distinct (no duplicates).",
        parent=root,
        critical=True
    )

    # Verify first 3 universities only
    selected_unis = extraction.universities[:3]
    for idx, uni in enumerate(selected_unis):
        await verify_university(evaluator, root, uni, idx)

    # Return summary
    return evaluator.get_summary()