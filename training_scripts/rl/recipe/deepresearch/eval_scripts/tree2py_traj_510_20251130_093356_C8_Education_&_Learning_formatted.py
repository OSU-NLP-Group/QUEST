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
TASK_ID = "largest_public_universities_mid_atlantic_midwest_states"
TASK_DESCRIPTION = """An international graduate student is researching large public research universities in the Mid-Atlantic and Midwest regions of the United States. They are particularly interested in universities that combine academic breadth, strong athletic traditions, and substantial enrollment that ensures diverse peer networks and extensive resources.

Identify the largest public university by total enrollment in each of the following four states: Ohio, Pennsylvania, Michigan, and New Jersey. For each university identified, it must meet ALL of the following criteria:

1. Be located in the specified state
2. Be the single largest public university by total enrollment within that state
3. Have a total enrollment exceeding 40,000 students
4. Be a current member of the Big Ten Conference
5. Hold regional accreditation from a recognized regional accrediting body
6. Offer at least 150 undergraduate major programs or degree options
7. Have undergraduate students comprising at least 60% of the total enrollment

For each of the four universities, provide the following information with supporting reference URLs:
- The university name
- Main campus city location
- Current total enrollment figure (with academic year specified)
- Current undergraduate enrollment figure (with academic year specified)
- The calculated percentage of undergraduate students
- The specific name of the regional accrediting body
- The number of undergraduate majors or programs offered
- Reference URLs supporting each data point

Present your findings in a structured format with clear citations for all factual claims.
"""

# Recognized regional accrediting bodies for instruction
RECOGNIZED_REGIONAL_ACCREDITORS = [
    "Middle States Commission on Higher Education",
    "MSCHE",
    "Higher Learning Commission",
    "HLC",
    "Southern Association of Colleges and Schools Commission on Colleges",
    "SACSCOC",
    "WASC Senior College and University Commission",
    "WSCUC",
    "New England Commission of Higher Education",
    "NECHE",
    "Northwest Commission on Colleges and Universities",
    "NWCCU",
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversitySources(BaseModel):
    """Per-data-point source URLs explicitly mentioned in the answer."""
    general_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    largest_enrollment_urls: List[str] = Field(default_factory=list)
    total_enrollment_urls: List[str] = Field(default_factory=list)
    undergraduate_enrollment_urls: List[str] = Field(default_factory=list)
    big_ten_membership_urls: List[str] = Field(default_factory=list)
    accreditation_urls: List[str] = Field(default_factory=list)
    majors_urls: List[str] = Field(default_factory=list)


class UniversityReport(BaseModel):
    """Structured report for a single state/university entry."""
    state: Optional[str] = None
    university_name: Optional[str] = None
    main_campus_city: Optional[str] = None
    total_enrollment: Optional[str] = None
    total_enrollment_year: Optional[str] = None
    undergraduate_enrollment: Optional[str] = None
    undergraduate_enrollment_year: Optional[str] = None
    undergraduate_percentage: Optional[str] = None
    accrediting_body: Optional[str] = None
    major_count: Optional[str] = None
    sources: Optional[UniversitySources] = None


class UniversitiesExtraction(BaseModel):
    """Top-level extraction container."""
    universities: List[UniversityReport] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract structured information from the answer for exactly four U.S. states: Ohio, Pennsylvania, Michigan, and New Jersey.
    Produce up to one university entry per state (if the answer provides multiple, keep the first for that state).
    If a state does not have an identified university in the answer, still include an entry for that state with the 'state' filled and other fields as null.

    For each university entry, extract the following fields as they appear in the answer:
    - state: One of "Ohio", "Pennsylvania", "Michigan", "New Jersey"
    - university_name: The university's name
    - main_campus_city: City of the main campus
    - total_enrollment: The total enrollment figure (as stated, keep exact text; do not normalize)
    - total_enrollment_year: The academic year associated with the total enrollment figure (e.g., "2024–25" or "2023-2024")
    - undergraduate_enrollment: The undergraduate enrollment figure (as stated)
    - undergraduate_enrollment_year: The academic year associated with the undergraduate enrollment figure
    - undergraduate_percentage: The percentage of undergraduates out of total enrollment (if reported directly). If not explicitly reported, you may calculate it from the answer's numbers and return it as a percentage string (e.g., "62%"); if not possible, return null.
    - accrediting_body: The name of the regional accrediting body (exact as stated)
    - major_count: The number of undergraduate majors or programs (as stated)
    - sources: Provide URL lists for each data point. Include only URLs explicitly present in the answer (plain URLs or within markdown). If no URL is provided for a field, the list for that field should be empty.
      • general_urls
      • location_urls
      • largest_enrollment_urls
      • total_enrollment_urls
      • undergraduate_enrollment_urls
      • big_ten_membership_urls
      • accreditation_urls
      • majors_urls

    IMPORTANT:
    - Do not invent or infer any information that is not present in the answer.
    - Preserve the exact textual form of numbers and years.
    - Include all URLs exactly as shown in the answer for the relevant field lists.
    - Ensure at most one entry per specified state.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return [u for u in (lst or []) if isinstance(u, str) and u.strip()]

def _pick_sources(report: UniversityReport, attr: str) -> List[str]:
    if not report.sources:
        return []
    lst = getattr(report.sources, attr, None)
    return _safe_list(lst) or _safe_list(report.sources.general_urls)

def _normalize_state_name(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = s.strip().lower()
    if "ohio" in m:
        return "Ohio"
    if "penn" in m:
        return "Pennsylvania"
    if "michigan" in m:
        return "Michigan"
    if "jersey" in m:
        return "New Jersey"
    return s.strip()

def _parse_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    t = text.strip().lower()
    # Handle k-suffix (e.g., "40k", "50K")
    k_match = re.search(r'(\d+(?:\.\d+)?)\s*[kK]\b', t)
    if k_match:
        try:
            val = float(k_match.group(1))
            return int(round(val * 1000))
        except Exception:
            pass
    # Extract the largest plausible integer with commas allowed
    nums = re.findall(r'\d{1,3}(?:,\d{3})+|\d+', t)
    if not nums:
        return None
    # Choose the largest numeric value present
    parsed_vals = []
    for n in nums:
        try:
            parsed_vals.append(int(n.replace(",", "")))
        except Exception:
            continue
    return max(parsed_vals) if parsed_vals else None

def _parse_percentage(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    t = text.strip().lower()
    # Prefer explicit %
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # Fallback: plain number that likely represents percent
    m2 = re.search(r'(\d+(?:\.\d+)?)', t)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None
    return None

def _compute_undergrad_share(undergrad_text: Optional[str], total_text: Optional[str], percent_text: Optional[str]) -> Optional[float]:
    ug = _parse_int(undergrad_text)
    tot = _parse_int(total_text)
    if ug is not None and tot is not None and tot > 0:
        return (ug / tot) * 100.0
    # Fallback to provided percentage if computation not possible
    pct = _parse_percentage(percent_text)
    return pct

def _has_any_sources_for_required(report: UniversityReport) -> bool:
    if not report.sources:
        return False
    src = report.sources
    total_count = sum(len(_safe_list(getattr(src, a))) for a in [
        "location_urls", "largest_enrollment_urls", "total_enrollment_urls",
        "undergraduate_enrollment_urls", "big_ten_membership_urls",
        "accreditation_urls", "majors_urls", "general_urls",
    ])
    return total_count > 0

def _all_required_fields_have_sources(report: UniversityReport) -> bool:
    """Require that each present data field has at least one source URL in its corresponding list."""
    if not report.sources:
        return False

    checks: List[Tuple[Optional[str], List[str]]] = [
        (report.main_campus_city, _pick_sources(report, "location_urls")),
        (report.total_enrollment, _pick_sources(report, "total_enrollment_urls")),
        (report.undergraduate_enrollment, _pick_sources(report, "undergraduate_enrollment_urls")),
        ("BigTenMarker", _pick_sources(report, "big_ten_membership_urls")),
        (report.accrediting_body, _pick_sources(report, "accreditation_urls")),
        (report.major_count, _pick_sources(report, "majors_urls")),
        ("LargestEnrollmentClaim", _pick_sources(report, "largest_enrollment_urls")),
    ]
    # For each present field (non-empty string except synthetic markers), require >=1 URL
    for value, urls in checks:
        if value and isinstance(value, str) and value.strip():
            if len(_safe_list(urls)) == 0:
                return False
    return _has_any_sources_for_required(report)


# --------------------------------------------------------------------------- #
# Verification logic per state                                                #
# --------------------------------------------------------------------------- #
async def verify_university_for_state(
    evaluator: Evaluator,
    parent_node,
    state_key: str,          # "ohio", "pa", "mi", "nj"
    state_full: str,         # "Ohio", "Pennsylvania", "Michigan", "New Jersey"
    report: UniversityReport
) -> None:
    """
    Build verification nodes and run checks for one state's university.
    """
    # 0) Create parent node for the state (parallel aggregator)
    state_node_id = f"university_{('ohio' if state_key=='ohio' else ('pennsylvania' if state_key=='pa' else ('michigan' if state_key=='mi' else 'new_jersey')))}"
    state_node_desc = f"Public university from {state_full} meeting all criteria"
    state_node = evaluator.add_parallel(
        id=state_node_id,
        desc=state_node_desc,
        parent=parent_node,
        critical=False,  # Allow partial credit on the overall state block
    )

    # Normalize state name in the extracted report for consistent claims
    normalized_state = _normalize_state_name(report.state) if report else None
    uni_name = (report.university_name or "").strip()
    city_name = (report.main_campus_city or "").strip()

    # --------------------------------------------------------------------- #
    # A) Data reporting sub-group                                           #
    # --------------------------------------------------------------------- #
    data_group_id = f"{state_key}_data_reporting"
    data_group = evaluator.add_parallel(
        id=data_group_id,
        desc="All required data points are provided with proper citations",
        parent=state_node,
        critical=False  # Non-critical group allowing partial credit
    )

    # A1) University name provided
    name_leaf = evaluator.add_custom_node(
        result=bool(uni_name),
        id=f"{state_key}_university_name",
        desc="University name is provided",
        parent=data_group,
        critical=True
    )

    # A2) Source URLs provided supporting the data (per data point)
    sources_provided = _all_required_fields_have_sources(report)
    sources_leaf = evaluator.add_custom_node(
        result=sources_provided,
        id=f"{state_key}_source_urls",
        desc="Reference URLs are provided supporting the data",
        parent=data_group,
        critical=True
    )

    # A3) Total enrollment figure with academic year stated
    total_enr_present = bool((report.total_enrollment or "").strip()) and bool((report.total_enrollment_year or "").strip())
    evaluator.add_custom_node(
        result=total_enr_present,
        id=f"{state_key}_total_enrollment_reported",
        desc="Specific total enrollment figure with academic year is stated",
        parent=data_group,
        critical=True
    )

    # A4) Undergraduate enrollment figure with academic year stated
    ug_enr_present = bool((report.undergraduate_enrollment or "").strip()) and bool((report.undergraduate_enrollment_year or "").strip())
    evaluator.add_custom_node(
        result=ug_enr_present,
        id=f"{state_key}_undergraduate_enrollment_reported",
        desc="Specific undergraduate enrollment figure with academic year is stated",
        parent=data_group,
        critical=True
    )

    # A5) Main campus city location identified
    evaluator.add_custom_node(
        result=bool(city_name),
        id=f"{state_key}_main_campus_city",
        desc="Main campus city location is identified",
        parent=data_group,
        critical=True
    )

    # A6) Accrediting body name provided
    evaluator.add_custom_node(
        result=bool((report.accrediting_body or "").strip()),
        id=f"{state_key}_accrediting_body_name",
        desc="Specific regional accrediting body name is provided",
        parent=data_group,
        critical=True
    )

    # A7) Specific number of undergraduate majors stated
    evaluator.add_custom_node(
        result=bool((report.major_count or "").strip()),
        id=f"{state_key}_major_count",
        desc="Specific number of undergraduate majors is stated",
        parent=data_group,
        critical=True
    )

    # A8) Calculated undergraduate percentage provided
    evaluator.add_custom_node(
        result=bool((report.undergraduate_percentage or "").strip()),
        id=f"{state_key}_undergraduate_percentage_calculated",
        desc="Calculated undergraduate percentage is provided",
        parent=data_group,
        critical=True
    )

    # --------------------------------------------------------------------- #
    # B) Core criteria verifications                                        #
    # --------------------------------------------------------------------- #

    # B1) Location in specified state
    loc_leaf = evaluator.add_leaf(
        id=f"{state_key}_state_location",
        desc=f"University is located in the state of {state_full}",
        parent=state_node,
        critical=True
    )
    loc_claim = f"{uni_name} is located in {state_full}. Its main campus is in {city_name}, {state_full}."
    loc_sources = _pick_sources(report, "location_urls")
    loc_instruction = (
        "Verify the university's location in the specified state using the provided URL(s). "
        "If the answer does not include any valid reference URLs supporting this claim, mark it as not supported."
    )

    # B2) Largest public university by total enrollment within the state
    largest_leaf = evaluator.add_leaf(
        id=f"{state_key}_enrollment_leadership",
        desc=f"University is the largest public university by total enrollment in {state_full}",
        parent=state_node,
        critical=True
    )
    largest_claim = f"{uni_name} is the largest public university by total enrollment in {state_full}."
    largest_sources = _pick_sources(report, "largest_enrollment_urls")
    largest_instruction = (
        "Confirm that the claim refers to the largest public university by total enrollment (undergraduate + graduate) "
        "within the specified state. Use the provided URL(s). If no valid URLs are provided, mark as not supported."
    )

    # B3) Total enrollment exceeds 40,000 students (threshold check via custom node)
    total_enr_val = _parse_int(report.total_enrollment)
    threshold_ok = (total_enr_val is not None) and (total_enr_val > 40000)
    evaluator.add_custom_node(
        result=threshold_ok,
        id=f"{state_key}_enrollment_threshold",
        desc="Total enrollment exceeds 40,000 students",
        parent=state_node,
        critical=True
    )

    # B4) Current Big Ten Conference membership
    bigten_leaf = evaluator.add_leaf(
        id=f"{state_key}_big_ten_membership",
        desc="University is a member of the Big Ten Conference",
        parent=state_node,
        critical=True
    )
    bigten_claim = f"{uni_name} is a current member institution of the Big Ten Conference."
    bigten_sources = _pick_sources(report, "big_ten_membership_urls")
    bigten_instruction = (
        "Verify Big Ten Conference membership using the provided URL(s) (e.g., the Big Ten official site or the university athletics page). "
        "If membership is not evidenced by the provided URLs, or no URLs are included, mark as not supported."
    )

    # B5) Regional accreditation (recognized)
    accred_leaf = evaluator.add_leaf(
        id=f"{state_key}_regional_accreditation",
        desc="University holds regional accreditation from a recognized accrediting body",
        parent=state_node,
        critical=True
    )
    accred_body = (report.accrediting_body or "").strip()
    accred_claim = f"{uni_name} holds regional accreditation from {accred_body}."
    accred_sources = _pick_sources(report, "accreditation_urls")
    accred_instruction = (
        "Verify that the university is accredited by a recognized U.S. regional accrediting body. "
        f"Recognized examples include: {', '.join(RECOGNIZED_REGIONAL_ACCREDITORS)}. "
        "Use the provided URL(s). If no valid URLs are provided in the answer, mark as not supported."
    )

    # B6) Undergraduate majors/programs >= 150 (threshold via custom node)
    major_count_val = _parse_int(report.major_count)
    majors_ok = (major_count_val is not None) and (major_count_val >= 150)
    evaluator.add_custom_node(
        result=majors_ok,
        id=f"{state_key}_undergraduate_majors",
        desc="University offers at least 150 undergraduate majors or programs",
        parent=state_node,
        critical=True
    )

    # B7) Undergraduate students represent at least 60% of total enrollment (threshold via custom node)
    ug_share = _compute_undergrad_share(report.undergraduate_enrollment, report.total_enrollment, report.undergraduate_percentage)
    ug_pct_ok = (ug_share is not None) and (ug_share >= 60.0)
    evaluator.add_custom_node(
        result=ug_pct_ok,
        id=f"{state_key}_undergraduate_percentage",
        desc="Undergraduate students represent at least 60% of total enrollment",
        parent=state_node,
        critical=True
    )

    # --------------------------------------------------------------------- #
    # C) Execute source-based verifications in parallel                     #
    #    We gate them logically with the data reporting leaves via instruction|
    # --------------------------------------------------------------------- #
    claims_and_sources: List[Tuple[str, List[str] | None, Any, Optional[str]]] = [
        (loc_claim, loc_sources if loc_sources else None, loc_leaf, loc_instruction),
        (largest_claim, largest_sources if largest_sources else None, largest_leaf, largest_instruction),
        (bigten_claim, bigten_sources if bigten_sources else None, bigten_leaf, bigten_instruction),
        (accred_claim, accred_sources if accred_sources else None, accred_leaf, accred_instruction),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the largest public university task across four states.
    """
    # Initialize evaluator (Root is parallel across states)
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Index extracted entries by normalized state name (keep first per state)
    by_state: Dict[str, UniversityReport] = {}
    for item in extraction.universities:
        st = _normalize_state_name(item.state)
        if st in ("Ohio", "Pennsylvania", "Michigan", "New Jersey"):
            if st not in by_state:
                by_state[st] = item

    # Ensure all four states have an entry (create empty placeholder if missing)
    for st in ("Ohio", "Pennsylvania", "Michigan", "New Jersey"):
        if st not in by_state:
            by_state[st] = UniversityReport(state=st, sources=UniversitySources())

    # Build and run verification for each state
    await verify_university_for_state(evaluator, root, "ohio", "Ohio", by_state["Ohio"])
    await verify_university_for_state(evaluator, root, "pa", "Pennsylvania", by_state["Pennsylvania"])
    await verify_university_for_state(evaluator, root, "mi", "Michigan", by_state["Michigan"])
    await verify_university_for_state(evaluator, root, "nj", "New Jersey", by_state["New Jersey"])

    # Return summary
    return evaluator.get_summary()