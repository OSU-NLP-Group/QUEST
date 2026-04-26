import asyncio
import logging
import re
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "north_texas_6a_schools"
TASK_DESCRIPTION = (
    "Identify three distinct public high schools located in North Texas (Dallas-Fort Worth metropolitan area or "
    "surrounding counties) that meet ALL of the following criteria:\n\n"
    "1. UIL Classification: The school must be classified as UIL Conference 6A for the 2026-2028 realignment period.\n"
    "2. Enrollment Requirement: The school must have a total enrollment of at least 4,500 students (based on official UIL enrollment numbers).\n"
    "3. School District Information: For each school's district, provide:\n"
    "   - The full official name of the school district\n"
    "   - The current superintendent's full name with title (Dr./Mr./Mrs.)\n"
    "   - The official school district website URL\n"
    "4. Graduation Requirements: The school must follow the Texas Foundation Plan graduation requirements, which mandate 22 total credits for graduation.\n"
    "5. Public School Status: The school must be a traditional public high school (not a charter school, private school, or specialized academy).\n\n"
    "For each of the three schools, provide:\n"
    "- Full official name of the high school\n"
    "- City location\n"
    "- Exact enrollment number\n"
    "- UIL 6A classification confirmation\n"
    "- School district name\n"
    "- Current superintendent's name with title\n"
    "- Official district website URL\n"
    "- Confirmation that the school follows the Texas Foundation Plan (22 credits)\n"
    "- Source URLs for all factual claims (enrollment, classification, superintendent information, graduation requirements)"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SchoolItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None

    # Enrollment and sources (UIL or district)
    enrollment: Optional[str] = None
    enrollment_sources: List[str] = Field(default_factory=list)

    # UIL classification and sources (official UIL)
    uil_classification: Optional[str] = None
    classification_sources: List[str] = Field(default_factory=list)

    # District information
    district_name: Optional[str] = None
    superintendent_name_with_title: Optional[str] = None
    superintendent_sources: List[str] = Field(default_factory=list)
    district_website: Optional[str] = None

    # Graduation requirements (22 credits) and sources (district/school)
    grad_plan_credits: Optional[str] = None
    graduation_sources: List[str] = Field(default_factory=list)


class SchoolsExtraction(BaseModel):
    schools: List[SchoolItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_schools() -> str:
    return (
        "Extract up to three qualifying schools from the answer. For each school, return the following fields:\n"
        "1) name: Full official name of the high school.\n"
        "2) city: City location of the high school.\n"
        "3) enrollment: The exact official UIL enrollment number (string as written in the answer).\n"
        "4) enrollment_sources: Array of URLs cited for enrollment data (UIL or district sources).\n"
        "5) uil_classification: UIL classification string for 2026-2028 (e.g., '6A').\n"
        "6) classification_sources: Array of official UIL URLs cited for classification.\n"
        "7) district_name: Full official name of the school district.\n"
        "8) superintendent_name_with_title: Current superintendent's full name including title (e.g., 'Dr. Jane Doe').\n"
        "9) superintendent_sources: Array of district URLs cited that confirm superintendent information.\n"
        "10) district_website: The official district website URL (home page preferred).\n"
        "11) grad_plan_credits: The graduation requirement credits value (e.g., '22' or '22 credits').\n"
        "12) graduation_sources: Array of district or school URLs cited for graduation requirements.\n\n"
        "Rules:\n"
        "- Extract only information explicitly present in the answer text.\n"
        "- Include only URLs actually shown in the answer for the source arrays.\n"
        "- If any field is missing, set it to null; if a source list is missing, return an empty array.\n"
        "- Return a JSON object with a 'schools' array of up to three items following the schema."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _extract_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return None


def _any_url_matches_domain(urls: List[str], domain: Optional[str]) -> bool:
    if not domain:
        return False
    for u in urls:
        d = _extract_domain(u)
        if d and d.endswith(domain):
            return True
    return False


def _has_uil_domain(urls: List[str]) -> bool:
    for u in urls:
        d = _extract_domain(u)
        if not d:
            continue
        if "uiltexas.org" in d or d.endswith("uiltexas.org") or d.endswith("files.uiltexas.org"):
            return True
    return False


def _clean_sources(*source_lists: List[str]) -> List[str]:
    merged = []
    for lst in source_lists:
        for u in lst:
            if isinstance(u, str) and u.strip():
                merged.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in merged:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _numeric_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Find first integer-like token (support commas)
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", text)
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    try:
        return int(num_str)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification for a single school                                            #
# --------------------------------------------------------------------------- #
async def verify_one_school(
    evaluator: Evaluator,
    parent_node,
    school: SchoolItem,
    index: int,
) -> None:
    idx = index + 1
    school_id_prefix = f"school_{idx}"

    # Parent node for this school (sequential to gate later checks if identification fails)
    school_node = evaluator.add_sequential(
        id=school_id_prefix,
        desc=f"{['First','Second','Third'][index]} qualifying high school identified and verified",
        parent=parent_node,
        critical=False,
    )

    # 1) Identification
    ident_node = evaluator.add_parallel(
        id=f"{school_id_prefix}_identification",
        desc="School name and location provided",
        parent=school_node,
        critical=True,
    )

    # 1.1 Name existence (critical)
    name_exists = bool(school.name and school.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"{school_id_prefix}_name",
        desc="Full official name of the high school provided",
        parent=ident_node,
        critical=True,
    )

    # 1.2 City in North Texas (critical) – simple logical verification
    city_leaf = evaluator.add_leaf(
        id=f"{school_id_prefix}_city",
        desc="City location in North Texas provided",
        parent=ident_node,
        critical=True,
    )
    city_claim = (
        f"The city '{school.city or ''}' is located in the Dallas-Fort Worth metropolitan area "
        f"or surrounding North Texas counties."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        additional_instruction=(
            "Treat 'North Texas' as the DFW metro area and adjacent counties (e.g., Dallas, Tarrant, Collin, Denton, "
            "Rockwall, Kaufman, Ellis, Johnson, Parker, Wise). Allow common-sense geographic knowledge."
        ),
    )

    # 2) Verifications (all required)
    verifs_node = evaluator.add_parallel(
        id=f"{school_id_prefix}_verifications",
        desc="All required verifications for the identified school",
        parent=school_node,
        critical=True,
    )

    # 2.1 Enrollment verification
    enroll_node = evaluator.add_parallel(
        id=f"{school_id_prefix}_enrollment_verification",
        desc="Enrollment meets the 4,500+ student threshold",
        parent=verifs_node,
        critical=True,
    )

    # 2.1.a Enrollment source presence (critical, custom)
    # Must be UIL or district source URL(s)
    has_enrollment_source = bool(school.enrollment_sources)
    # Prefer UIL domain or matching district domain
    district_domain = _extract_domain(school.district_website)
    enrollment_source_valid = has_enrollment_source and (
        _has_uil_domain(school.enrollment_sources) or _any_url_matches_domain(school.enrollment_sources, district_domain)
    )
    enroll_source_node = evaluator.add_custom_node(
        result=enrollment_source_valid,
        id=f"{school_id_prefix}_enrollment_source",
        desc="Official UIL or district source URL provided for enrollment data",
        parent=enroll_node,
        critical=True,
    )

    # 2.1.b Enrollment value is provided and >= 4,500, and supported by sources (critical, leaf)
    enroll_value_leaf = evaluator.add_leaf(
        id=f"{school_id_prefix}_enrollment_value",
        desc="Specific enrollment number provided and is 4,500 or higher",
        parent=enroll_node,
        critical=True,
    )
    enroll_num = _numeric_from_text(school.enrollment)
    enroll_value_claim = (
        f"The official enrollment for {school.name or ''} used for UIL realignment is {school.enrollment or ''}, "
        f"and it is at least 4,500."
    )
    await evaluator.verify(
        claim=enroll_value_claim,
        node=enroll_value_leaf,
        sources=school.enrollment_sources,
        additional_instruction=(
            "Check the cited UIL or district source to confirm the exact enrollment number. "
            "Treat the verification as FAILED if the enrollment is missing or below 4,500."
        ),
        extra_prerequisites=[enroll_source_node],  # Gate by source existence
    )

    # 2.2 UIL classification verification (2026-2028 realignment)
    uil_node = evaluator.add_parallel(
        id=f"{school_id_prefix}_uil_classification",
        desc="UIL 6A classification verified for 2026-2028 period",
        parent=verifs_node,
        critical=True,
    )

    # 2.2.a Classification source presence (critical, custom) – should be official UIL URL
    class_source_ok = bool(school.classification_sources) and _has_uil_domain(school.classification_sources)
    class_source_node = evaluator.add_custom_node(
        result=class_source_ok,
        id=f"{school_id_prefix}_classification_source",
        desc="Official UIL source URL provided for classification",
        parent=uil_node,
        critical=True,
    )

    # 2.2.b Classification confirmed (critical, leaf)
    class_confirm_leaf = evaluator.add_leaf(
        id=f"{school_id_prefix}_classification_confirmed",
        desc="School confirmed to be UIL Conference 6A",
        parent=uil_node,
        critical=True,
    )
    class_confirm_claim = (
        f"For the 2026-2028 realignment period, {school.name or ''} is classified as UIL Conference 6A."
    )
    await evaluator.verify(
        claim=class_confirm_claim,
        node=class_confirm_leaf,
        sources=school.classification_sources,
        additional_instruction=(
            "Use the official UIL 2026-2028 realignment directory or PDF. "
            "Confirm the school's classification is '6A'. Allow minor formatting differences."
        ),
        extra_prerequisites=[class_source_node],
    )

    # 2.3 District info verification
    district_node = evaluator.add_parallel(
        id=f"{school_id_prefix}_district_info",
        desc="School district administrative information provided",
        parent=verifs_node,
        critical=True,
    )

    # 2.3.a District name confirmed (critical, leaf)
    district_name_leaf = evaluator.add_leaf(
        id=f"{school_id_prefix}_district_name",
        desc="Full official school district name provided",
        parent=district_node,
        critical=True,
    )
    district_name_claim = (
        f"The high school {school.name or ''} belongs to the school district '{school.district_name or ''}'."
    )
    district_name_sources = _clean_sources(
        [school.district_website] if school.district_website else [],
        school.classification_sources,
    )
    await evaluator.verify(
        claim=district_name_claim,
        node=district_name_leaf,
        sources=district_name_sources,
        additional_instruction=(
            "Verify using the district website or UIL alignment pages that this school is part of the given district."
        ),
    )

    # 2.3.b Superintendent sub-tree (critical, parallel)
    super_node = evaluator.add_parallel(
        id=f"{school_id_prefix}_superintendent",
        desc="Current superintendent's name provided",
        parent=district_node,
        critical=True,
    )

    # 2.3.b.i Superintendent source presence (critical, custom) – must be district domain
    super_source_ok = bool(school.superintendent_sources) and _any_url_matches_domain(
        school.superintendent_sources, district_domain
    )
    super_source_node = evaluator.add_custom_node(
        result=super_source_ok,
        id=f"{school_id_prefix}_superintendent_source",
        desc="Official district website URL confirming superintendent information",
        parent=super_node,
        critical=True,
    )

    # 2.3.b.ii Superintendent name verification (critical, leaf)
    super_name_leaf = evaluator.add_leaf(
        id=f"{school_id_prefix}_superintendent_name",
        desc="Superintendent's full name with title provided",
        parent=super_node,
        critical=True,
    )
    super_name_claim = (
        f"The current superintendent of {school.district_name or ''} is {school.superintendent_name_with_title or ''}."
    )
    await evaluator.verify(
        claim=super_name_claim,
        node=super_name_leaf,
        sources=school.superintendent_sources,
        additional_instruction=(
            "Use the district leadership/superintendent page. Allow minor variations in title formatting "
            "(e.g., Dr., Mr., Mrs., Ms.). Confirm the person is the current superintendent."
        ),
        extra_prerequisites=[super_source_node],
    )

    # 2.3.c District website verification (critical, leaf)
    district_site_leaf = evaluator.add_leaf(
        id=f"{school_id_prefix}_district_website",
        desc="Official district website URL provided",
        parent=district_node,
        critical=True,
    )
    district_site_claim = f"This URL is the official website of the school district '{school.district_name or ''}'."
    await evaluator.verify(
        claim=district_site_claim,
        node=district_site_leaf,
        sources=school.district_website,
        additional_instruction=(
            "Check header/footer branding, contact info, and domain to confirm it is the district's official site."
        ),
    )

    # 2.4 Graduation requirements (22 credits)
    grad_node = evaluator.add_parallel(
        id=f"{school_id_prefix}_graduation_requirements",
        desc="Texas Foundation Plan graduation requirements (22 credits) confirmed",
        parent=verifs_node,
        critical=True,
    )

    # 2.4.a Graduation source presence (critical, custom) – should be district/school domain
    grad_source_ok = bool(school.graduation_sources) and _any_url_matches_domain(
        school.graduation_sources, district_domain
    )
    grad_source_node = evaluator.add_custom_node(
        result=grad_source_ok,
        id=f"{school_id_prefix}_grad_source",
        desc="District or school website URL confirming graduation requirements",
        parent=grad_node,
        critical=True,
    )

    # 2.4.b Graduation credits confirmed (critical, leaf)
    grad_credits_leaf = evaluator.add_leaf(
        id=f"{school_id_prefix}_grad_credits",
        desc="School follows Texas Foundation Plan requiring 22 total credits",
        parent=grad_node,
        critical=True,
    )
    grad_claim = (
        f"The district/school policy confirms the Foundation High School Program (Foundation Plan) requires "
        f"22 total credits to graduate."
    )
    await evaluator.verify(
        claim=grad_claim,
        node=grad_credits_leaf,
        sources=school.graduation_sources,
        additional_instruction=(
            "Look for district/school graduation requirements pages stating the Foundation Plan (FHSP) has 22 credits."
        ),
        extra_prerequisites=[grad_source_node],
    )

    # 2.5 Public school designation (critical, leaf)
    public_leaf = evaluator.add_leaf(
        id=f"{school_id_prefix}_public_designation",
        desc="School verified as a public (non-charter, non-private) high school",
        parent=verifs_node,
        critical=True,
    )
    public_sources = _clean_sources(
        [school.district_website] if school.district_website else [],
        school.classification_sources,
    )
    public_claim = (
        f"The high school {school.name or ''} is a traditional public high school (not charter, not private) within "
        f"its Texas public school district."
    )
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=public_sources,
        additional_instruction=(
            "Confirm the school is listed on a Texas public school district site or UIL alignment listings. "
            "Presence within an ISD and absence of charter/private descriptors indicate public status."
        ),
    )


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
    Evaluate an answer for the North Texas UIL 6A schools task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates three independent schools
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

    # Important: Root set to non-critical to allow partial credit if only some schools are correct.
    # The JSON root was critical, but obj_task_eval enforces that critical parents must have all critical children.
    # Changing root to non-critical avoids structural conflicts and allows partial scoring.

    # Extract schools
    extracted = await evaluator.extract(
        prompt=prompt_extract_schools(),
        template_class=SchoolsExtraction,
        extraction_name="schools_extraction",
    )

    # Take first three schools; pad if fewer
    schools: List[SchoolItem] = list(extracted.schools[:3])
    while len(schools) < 3:
        schools.append(SchoolItem())

    # Build and verify for each school
    for i, sch in enumerate(schools):
        await verify_one_school(evaluator, root, sch, i)

    # Return summary
    return evaluator.get_summary()