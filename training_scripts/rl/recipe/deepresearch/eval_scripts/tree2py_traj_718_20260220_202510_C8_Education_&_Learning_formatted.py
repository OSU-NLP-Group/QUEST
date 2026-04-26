import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_catholic_university_criteria"
TASK_DESCRIPTION = (
    "What is the name of the Catholic university in Pennsylvania that was founded between 1840 and 1850 by a religious order, "
    "is located in a township within 15 miles of Philadelphia, has a campus of at least 200 acres, has at least 5 colleges or schools, "
    "offers PhD programs in both Philosophy and Theology as well as graduate programs in Engineering, is accredited by a regional accrediting agency, "
    "and had a Fall 2024 total enrollment between 9,000 and 11,000 students with at least 6,500 undergraduate students?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CriterionEvidence(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class UniversityExtraction(BaseModel):
    university_name: Optional[str] = None

    location_pennsylvania: CriterionEvidence = Field(default_factory=CriterionEvidence)
    catholic_affiliation: CriterionEvidence = Field(default_factory=CriterionEvidence)
    founding_period: CriterionEvidence = Field(default_factory=CriterionEvidence)
    township_location: CriterionEvidence = Field(default_factory=CriterionEvidence)
    campus_size: CriterionEvidence = Field(default_factory=CriterionEvidence)
    distance_from_philadelphia: CriterionEvidence = Field(default_factory=CriterionEvidence)
    doctoral_programs: CriterionEvidence = Field(default_factory=CriterionEvidence)
    colleges_count: CriterionEvidence = Field(default_factory=CriterionEvidence)
    total_enrollment_fall_2024: CriterionEvidence = Field(default_factory=CriterionEvidence)
    undergraduate_enrollment_fall_2024: CriterionEvidence = Field(default_factory=CriterionEvidence)
    phd_philosophy: CriterionEvidence = Field(default_factory=CriterionEvidence)
    phd_theology: CriterionEvidence = Field(default_factory=CriterionEvidence)
    regional_accreditation: CriterionEvidence = Field(default_factory=CriterionEvidence)
    religious_order_founding: CriterionEvidence = Field(default_factory=CriterionEvidence)
    engineering_graduate_programs: CriterionEvidence = Field(default_factory=CriterionEvidence)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_profile() -> str:
    return (
        "Extract the single university identified in the answer along with criterion-specific evidence.\n"
        "Return a JSON object with the following fields:\n"
        "- university_name: The name of the university identified to meet all criteria.\n"
        "- For each criterion below, extract:\n"
        "  • value: The key fact claimed in the answer (number, name, or short phrase as stated).\n"
        "  • urls: A list of all explicit URLs cited in the answer that support this specific criterion.\n"
        "Criteria fields to extract:\n"
        "  • location_pennsylvania\n"
        "  • catholic_affiliation\n"
        "  • founding_period\n"
        "  • township_location\n"
        "  • campus_size\n"
        "  • distance_from_philadelphia\n"
        "  • doctoral_programs\n"
        "  • colleges_count\n"
        "  • total_enrollment_fall_2024\n"
        "  • undergraduate_enrollment_fall_2024\n"
        "  • phd_philosophy\n"
        "  • phd_theology\n"
        "  • regional_accreditation\n"
        "  • religious_order_founding\n"
        "  • engineering_graduate_programs\n"
        "General rules:\n"
        "1) Do not invent URLs; include only URLs explicitly present in the answer (plain or markdown links). If none are provided for a criterion, return an empty list.\n"
        "2) Preserve numbers and names exactly as stated in the answer for 'value'. If a range or qualitative description is provided (e.g., 'around 10,000'), include it verbatim.\n"
        "3) If a criterion is not mentioned, set its 'value' to null and 'urls' to an empty list.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: List[str]) -> List[str]:
    """Normalize and deduplicate URLs; ensure protocol if missing."""
    seen = set()
    normalized = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # If markdown-like [text](url), try to extract url inside parentheses
        m = re.search(r"\((https?://[^\s)]+)\)", u)
        if m:
            u = m.group(1)
        if not u.startswith("http://") and not u.startswith("https://"):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            normalized.append(u)
    return normalized


def _parse_number(text: Optional[str]) -> Optional[float]:
    """Extract the first numeric value from a string; handles commas, decimals, and 'k' shorthand."""
    if not text:
        return None
    t = text.lower().strip()
    # Handle shorthand like "10k", "9.5k"
    mk = re.search(r"(\d+(?:\.\d+)?)\s*k\b", t)
    if mk:
        try:
            return float(mk.group(1)) * 1000.0
        except Exception:
            pass

    # General number with optional commas/decimals
    m = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)", t)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except Exception:
        return None


def _parse_int(text: Optional[str]) -> Optional[int]:
    n = _parse_number(text)
    return int(round(n)) if n is not None else None


def _parse_year(text: Optional[str]) -> Optional[int]:
    """Extract a plausible 4-digit year."""
    if not text:
        return None
    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _regional_agency_match(name: Optional[str]) -> bool:
    """Heuristic check that the accreditor is a recognized US regional accrediting agency."""
    if not name:
        return False
    s = name.lower()
    # Common regional agencies and acronyms
    agencies = [
        "middle states commission on higher education", "msche",
        "new england commission of higher education", "neche",
        "higher learning commission", "hlc",
        "northwest commission on colleges and universities", "nwccu",
        "southern association of colleges and schools commission on colleges", "sacscoc",
        "wasc senior college and university commission", "wscuc",
    ]
    return any(a in s for a in agencies)


def _add_sources_provided_node(
    evaluator: Evaluator,
    parent,
    base_id: str,
    desc_suffix: str,
    urls: List[str],
) -> Any:
    return evaluator.add_custom_node(
        result=bool(urls),
        id=f"{base_id}_sources_provided",
        desc=f"URLs provided for {desc_suffix}",
        parent=parent,
        critical=True
    )


async def _add_reference_verification(
    evaluator: Evaluator,
    parent,
    base_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    add_ins: str,
) -> Any:
    node = evaluator.add_leaf(
        id=f"{base_id}_reference",
        desc=desc,
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction=add_ins
    )
    return node


def _add_numeric_check_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    value_str: Optional[str],
    check_kind: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Any:
    """Add a custom numeric check node: min/max/range."""
    num = None
    if check_kind == "year_range":
        num = _parse_year(value_str)
    else:
        num = _parse_number(value_str)

    result = False
    if num is not None:
        if check_kind == "min":
            result = (min_val is not None) and (num >= float(min_val))
        elif check_kind == "max":
            result = (max_val is not None) and (num <= float(max_val))
        elif check_kind == "range":
            result = (min_val is not None and max_val is not None) and (float(min_val) <= num <= float(max_val))
        elif check_kind == "year_range":
            result = (min_val is not None and max_val is not None) and (float(min_val) <= num <= float(max_val))

    return evaluator.add_custom_node(
        result=result,
        id=node_id,
        desc=f"{desc} (parsed '{value_str}' → {num})",
        parent=parent,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_university_identification(
    evaluator: Evaluator,
    root,
    data: UniversityExtraction
) -> None:
    # Parent identification (critical: all required constraints must pass)
    uni_node = evaluator.add_parallel(
        id="university_identification",
        desc="Identify a university that meets all specified criteria",
        parent=root,
        critical=True
    )

    # University name existence check (critical)
    evaluator.add_custom_node(
        result=bool(data.university_name and data.university_name.strip()),
        id="university_name_provided",
        desc="University name is provided",
        parent=uni_node,
        critical=True
    )
    uni = data.university_name or "the university"

    # Helper additional instruction for URL-grounded checks
    base_add_ins = (
        "Use only the provided URLs to judge this claim. If no URLs are provided or the URLs do not explicitly support the claim, "
        "mark the claim as not supported. Allow minor name variants and reasonable numeric rounding."
    )

    # 1) Location: Pennsylvania
    loc_pa_node = evaluator.add_parallel(
        id="location_pennsylvania",
        desc="The university must be located in Pennsylvania",
        parent=uni_node,
        critical=True
    )
    urls_pa = _clean_urls(data.location_pennsylvania.urls)
    _add_sources_provided_node(evaluator, loc_pa_node, "location_pennsylvania", "Pennsylvania location", urls_pa)
    await _add_reference_verification(
        evaluator, loc_pa_node, "location_pennsylvania",
        "Provide URL evidence confirming Pennsylvania location",
        f"{uni} is located in Pennsylvania.",
        urls_pa,
        base_add_ins
    )

    # 2) Catholic affiliation
    catholic_node = evaluator.add_parallel(
        id="catholic_affiliation",
        desc="The university must be a Catholic university",
        parent=uni_node,
        critical=True
    )
    urls_cath = _clean_urls(data.catholic_affiliation.urls)
    _add_sources_provided_node(evaluator, catholic_node, "catholic_affiliation", "Catholic affiliation", urls_cath)
    await _add_reference_verification(
        evaluator, catholic_node, "catholic_affiliation",
        "Provide URL evidence confirming Catholic affiliation",
        f"{uni} is a Catholic university or is affiliated with the Catholic Church.",
        urls_cath,
        base_add_ins
    )

    # 3) Founding between 1840 and 1850 inclusive
    founding_node = evaluator.add_parallel(
        id="founding_period",
        desc="The university must have been founded between 1840 and 1850 (inclusive)",
        parent=uni_node,
        critical=True
    )
    urls_found = _clean_urls(data.founding_period.urls)
    _add_sources_provided_node(evaluator, founding_node, "founding_period", "founding year", urls_found)
    await _add_reference_verification(
        evaluator, founding_node, "founding_period",
        "Provide URL evidence confirming founding year",
        f"{uni} was founded in {data.founding_period.value}.",
        urls_found,
        base_add_ins
    )
    _add_numeric_check_node(
        evaluator, founding_node,
        "founding_year_in_range",
        "Founding year is between 1840 and 1850 inclusive",
        data.founding_period.value,
        check_kind="year_range", min_val=1840, max_val=1850
    )

    # 4) Township location (not a city) within Philadelphia metro area (the 15-mile constraint will be checked separately)
    township_node = evaluator.add_parallel(
        id="township_location",
        desc="The university must be located in a township (not a city) within the Philadelphia metropolitan area",
        parent=uni_node,
        critical=True
    )
    urls_town = _clean_urls(data.township_location.urls)
    _add_sources_provided_node(evaluator, township_node, "township_location", "township municipality", urls_town)
    township_val = data.township_location.value or ""
    await _add_reference_verification(
        evaluator, township_node, "township_location",
        "Provide URL evidence confirming township location",
        f"{uni} is located in {township_val} Township.",
        urls_town,
        base_add_ins + " Confirm that the municipality type is 'Township' (not 'City' or 'Borough')."
    )

    # 5) Campus size at least 200 acres
    campus_node = evaluator.add_parallel(
        id="campus_size",
        desc="The university must have a campus size of at least 200 acres",
        parent=uni_node,
        critical=True
    )
    urls_campus = _clean_urls(data.campus_size.urls)
    _add_sources_provided_node(evaluator, campus_node, "campus_size", "campus acreage", urls_campus)
    await _add_reference_verification(
        evaluator, campus_node, "campus_size",
        "Provide URL evidence confirming campus acreage",
        f"The campus size of {uni} is {data.campus_size.value} acres.",
        urls_campus,
        base_add_ins
    )
    _add_numeric_check_node(
        evaluator, campus_node,
        "campus_size_min_200",
        "Campus size is at least 200 acres",
        data.campus_size.value,
        check_kind="min", min_val=200
    )

    # 6) Distance within 15 miles of Philadelphia
    distance_node = evaluator.add_parallel(
        id="distance_from_philadelphia",
        desc="The university must be located within 15 miles of Philadelphia",
        parent=uni_node,
        critical=True
    )
    urls_dist = _clean_urls(data.distance_from_philadelphia.urls)
    _add_sources_provided_node(evaluator, distance_node, "distance_from_philadelphia", "distance from Philadelphia", urls_dist)
    await _add_reference_verification(
        evaluator, distance_node, "distance_from_philadelphia",
        "Provide URL evidence confirming distance from Philadelphia",
        f"{uni} is approximately {data.distance_from_philadelphia.value} miles from Philadelphia.",
        urls_dist,
        base_add_ins + " If a distance range or multiple values are given, use the most direct statement from the source."
    )
    _add_numeric_check_node(
        evaluator, distance_node,
        "distance_within_15_miles",
        "Distance from Philadelphia is at most 15 miles",
        data.distance_from_philadelphia.value,
        check_kind="max", max_val=15
    )

    # 7) Offers doctoral (PhD) programs (general)
    doctoral_node = evaluator.add_parallel(
        id="doctoral_programs",
        desc="The university must offer doctoral (PhD) programs",
        parent=uni_node,
        critical=True
    )
    urls_phd_general = _clean_urls(data.doctoral_programs.urls)
    _add_sources_provided_node(evaluator, doctoral_node, "doctoral_programs", "doctoral offerings", urls_phd_general)
    await _add_reference_verification(
        evaluator, doctoral_node, "doctoral_programs",
        "Provide URL evidence confirming doctoral program offerings",
        f"{uni} offers doctoral (PhD) programs.",
        urls_phd_general,
        base_add_ins
    )

    # 8) At least 5 colleges or schools
    colleges_node = evaluator.add_parallel(
        id="colleges_count",
        desc="The university must have at least 5 colleges or schools",
        parent=uni_node,
        critical=True
    )
    urls_colleges = _clean_urls(data.colleges_count.urls)
    _add_sources_provided_node(evaluator, colleges_node, "colleges_count", "number of colleges/schools", urls_colleges)
    await _add_reference_verification(
        evaluator, colleges_node, "colleges_count",
        "Provide URL evidence confirming number of colleges/schools",
        f"{uni} has {data.colleges_count.value} colleges or schools.",
        urls_colleges,
        base_add_ins
    )
    _add_numeric_check_node(
        evaluator, colleges_node,
        "colleges_count_min_5",
        "Number of colleges/schools is at least 5",
        data.colleges_count.value,
        check_kind="min", min_val=5
    )

    # 9) Total enrollment Fall 2024: between 9,000 and 11,000
    total_enr_node = evaluator.add_parallel(
        id="total_enrollment_fall_2024",
        desc="The university's Fall 2024 total enrollment must be between 9,000 and 11,000 students",
        parent=uni_node,
        critical=True
    )
    urls_total = _clean_urls(data.total_enrollment_fall_2024.urls)
    _add_sources_provided_node(evaluator, total_enr_node, "total_enrollment_fall_2024", "Fall 2024 total enrollment", urls_total)
    await _add_reference_verification(
        evaluator, total_enr_node, "total_enrollment_fall_2024",
        "Provide URL evidence confirming Fall 2024 total enrollment",
        f"The Fall 2024 total enrollment at {uni} was {data.total_enrollment_fall_2024.value}.",
        urls_total,
        base_add_ins + " Focus on Fall 2024 specifically."
    )
    _add_numeric_check_node(
        evaluator, total_enr_node,
        "total_enrollment_between_9000_11000",
        "Fall 2024 total enrollment is between 9,000 and 11,000",
        data.total_enrollment_fall_2024.value,
        check_kind="range", min_val=9000, max_val=11000
    )

    # 10) Undergraduate enrollment Fall 2024: at least 6,500
    undergrad_enr_node = evaluator.add_parallel(
        id="undergraduate_enrollment_fall_2024",
        desc="The university must have at least 6,500 undergraduate students in Fall 2024",
        parent=uni_node,
        critical=True
    )
    urls_undergrad = _clean_urls(data.undergraduate_enrollment_fall_2024.urls)
    _add_sources_provided_node(evaluator, undergrad_enr_node, "undergraduate_enrollment_fall_2024", "Fall 2024 undergraduate enrollment", urls_undergrad)
    await _add_reference_verification(
        evaluator, undergrad_enr_node, "undergraduate_enrollment_fall_2024",
        "Provide URL evidence confirming Fall 2024 undergraduate enrollment",
        f"The Fall 2024 undergraduate enrollment at {uni} was {data.undergraduate_enrollment_fall_2024.value}.",
        urls_undergrad,
        base_add_ins + " Focus on undergraduate headcount for Fall 2024."
    )
    _add_numeric_check_node(
        evaluator, undergrad_enr_node,
        "undergraduate_enrollment_min_6500",
        "Fall 2024 undergraduate enrollment is at least 6,500",
        data.undergraduate_enrollment_fall_2024.value,
        check_kind="min", min_val=6500
    )

    # 11) PhD in Philosophy
    phd_phil_node = evaluator.add_parallel(
        id="phd_philosophy",
        desc="The university must offer a PhD program in Philosophy",
        parent=uni_node,
        critical=True
    )
    urls_phil = _clean_urls(data.phd_philosophy.urls)
    _add_sources_provided_node(evaluator, phd_phil_node, "phd_philosophy", "PhD in Philosophy offering", urls_phil)
    await _add_reference_verification(
        evaluator, phd_phil_node, "phd_philosophy",
        "Provide URL evidence confirming PhD in Philosophy offering",
        f"{uni} offers a PhD program in Philosophy.",
        urls_phil,
        base_add_ins + " Prefer official departmental or graduate catalog sources."
    )

    # 12) PhD in Theology
    phd_theol_node = evaluator.add_parallel(
        id="phd_theology",
        desc="The university must offer a PhD program in Theology",
        parent=uni_node,
        critical=True
    )
    urls_theol = _clean_urls(data.phd_theology.urls)
    _add_sources_provided_node(evaluator, phd_theol_node, "phd_theology", "PhD in Theology offering", urls_theol)
    await _add_reference_verification(
        evaluator, phd_theol_node, "phd_theology",
        "Provide URL evidence confirming PhD in Theology offering",
        f"{uni} offers a PhD program in Theology.",
        urls_theol,
        base_add_ins + " Prefer official departmental or graduate catalog sources."
    )

    # 13) Regional accreditation
    accred_node = evaluator.add_parallel(
        id="regional_accreditation",
        desc="The university must be accredited by a regional accrediting agency",
        parent=uni_node,
        critical=True
    )
    urls_accred = _clean_urls(data.regional_accreditation.urls)
    _add_sources_provided_node(evaluator, accred_node, "regional_accreditation", "regional accreditation", urls_accred)
    accred_val = data.regional_accreditation.value or ""
    await _add_reference_verification(
        evaluator, accred_node, "regional_accreditation",
        "Provide URL evidence confirming regional accreditation status",
        f"{uni} is accredited by {accred_val}.",
        urls_accred,
        base_add_ins + " Prefer official accreditor directories or university accreditation pages."
    )
    evaluator.add_custom_node(
        result=_regional_agency_match(data.regional_accreditation.value),
        id="regional_accreditation_is_recognized",
        desc=f"Accrediting agency '{data.regional_accreditation.value}' is a recognized US regional accreditor",
        parent=accred_node,
        critical=True
    )

    # 14) Founded by a religious order
    order_node = evaluator.add_parallel(
        id="religious_order_founding",
        desc="The university must have been founded by a religious order",
        parent=uni_node,
        critical=True
    )
    urls_order = _clean_urls(data.religious_order_founding.urls)
    _add_sources_provided_node(evaluator, order_node, "religious_order_founding", "founding by a religious order", urls_order)
    order_val = data.religious_order_founding.value or ""
    await _add_reference_verification(
        evaluator, order_node, "religious_order_founding",
        "Provide URL evidence confirming founding by a religious order",
        f"{uni} was founded by {order_val}.",
        urls_order,
        base_add_ins + " Prefer official institutional history pages or reputable sources."
    )

    # 15) Graduate programs in Engineering
    eng_node = evaluator.add_parallel(
        id="engineering_graduate_programs",
        desc="The university must offer graduate programs in Engineering",
        parent=uni_node,
        critical=True
    )
    urls_eng = _clean_urls(data.engineering_graduate_programs.urls)
    _add_sources_provided_node(evaluator, eng_node, "engineering_graduate_programs", "graduate Engineering programs", urls_eng)
    await _add_reference_verification(
        evaluator, eng_node, "engineering_graduate_programs",
        "Provide URL evidence confirming graduate Engineering programs",
        f"{uni} offers graduate programs in Engineering.",
        urls_eng,
        base_add_ins + " Prefer official College/School of Engineering graduate program pages."
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
    Evaluate an answer for the Pennsylvania Catholic university criteria task.
    Builds a verification tree where all listed criteria are critical under a single identification node.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured evidence from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_profile(),
        template_class=UniversityExtraction,
        extraction_name="university_profile_evidence"
    )

    # Build and verify the identification tree
    await build_university_identification(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()