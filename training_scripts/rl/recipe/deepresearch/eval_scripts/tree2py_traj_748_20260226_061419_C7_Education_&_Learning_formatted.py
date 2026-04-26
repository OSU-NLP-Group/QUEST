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
TASK_ID = "pa_private_university_fcs_2025"
TASK_DESCRIPTION = (
    "Identify a private university in Pennsylvania that meets all of the following requirements: "
    "(1) The university must be located in the state of Pennsylvania; "
    "(2) The university must be a private institution (not a public university); "
    "(3) The university must be regionally accredited by a recognized accrediting body; "
    "(4) The university must have an active NCAA Division I Football Championship Subdivision (FCS) football program; "
    "(5) The university's football program must have been affiliated with an NCAA Division I FCS conference during the 2025 season; "
    "(6) The university must have a test-optional admissions policy for SAT/ACT scores; "
    "(7) The middle 50% SAT score range for admitted students must have a lower bound of at least 1300; "
    "(8) Total undergraduate enrollment must be between 5,000 and 10,000 students; "
    "(9) The university must have been founded before the year 1900; "
    "(10) Annual undergraduate tuition must exceed $60,000; "
    "(11) The university must be located within 20 miles of a major metropolitan area; "
    "(12) The university must offer bachelor's degree programs. "
    "Provide the name of the university and reference URLs that verify these characteristics."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    # Core identification
    university_name: Optional[str] = None

    # General URLs mentioned anywhere in the answer
    urls_general: List[str] = Field(default_factory=list)

    # Location
    location_state: Optional[str] = None
    location_city_or_town: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)

    # Institution type
    institution_type: Optional[str] = None  # e.g., "private", "public"
    institution_sources: List[str] = Field(default_factory=list)

    # Accreditation
    accreditation_body: Optional[str] = None  # e.g., "MSCHE", "NECHE", "HLC", etc.
    accreditation_sources: List[str] = Field(default_factory=list)

    # Athletics (FCS)
    fcs_program_status: Optional[str] = None  # e.g., "NCAA Division I FCS", "active FCS football"
    fcs_sources: List[str] = Field(default_factory=list)

    # FCS conference (2025 season)
    fcs_conference_2025: Optional[str] = None  # e.g., "Patriot League", "CAA", "Ivy League"
    fcs_conference_sources: List[str] = Field(default_factory=list)

    # Test-optional policy
    test_optional_policy: Optional[str] = None  # e.g., "test-optional", "SAT/ACT optional"
    test_optional_sources: List[str] = Field(default_factory=list)

    # SAT middle 50% range
    sat_middle_50_range: Optional[str] = None  # e.g., "1300-1480", "1410–1520"
    sat_sources: List[str] = Field(default_factory=list)

    # Undergraduate enrollment
    undergrad_enrollment: Optional[str] = None  # keep as string (e.g., "6,800", "~7k")
    enrollment_sources: List[str] = Field(default_factory=list)

    # Founding year
    founding_year: Optional[str] = None  # keep as string (e.g., "1842")
    founding_sources: List[str] = Field(default_factory=list)

    # Undergraduate tuition (annual)
    undergrad_tuition: Optional[str] = None  # keep as string (e.g., "$65,120")
    tuition_sources: List[str] = Field(default_factory=list)

    # Proximity to major metropolitan area
    metro_area: Optional[str] = None  # e.g., "Philadelphia", "Pittsburgh", "Allentown-Bethlehem-Easton"
    metro_distance_miles: Optional[str] = None  # keep as string (e.g., "10 miles")
    metro_sources: List[str] = Field(default_factory=list)

    # Bachelor's degree programs
    offers_bachelors: Optional[str] = None  # e.g., "offers bachelor's degrees"
    program_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_info() -> str:
    return """
    Extract the single university identified in the answer and the evidence URLs that support each required criterion.

    Return a JSON object with the following fields:

    - university_name: The name of the university mentioned.
    - urls_general: All URLs referenced anywhere in the answer (include every URL you can find).
    - location_state: The state where the university is located (e.g., "Pennsylvania", "PA").
    - location_city_or_town: The city/town or locality of the university's main campus (e.g., "Villanova", "Bethlehem").
    - location_sources: URLs that support the location details.

    - institution_type: The institution type explicitly stated (e.g., "private", "public").
    - institution_sources: URLs that support the institution type classification.

    - accreditation_body: The regional institutional accreditor (e.g., "Middle States Commission on Higher Education", "MSCHE").
    - accreditation_sources: URLs that support accreditation.

    - fcs_program_status: A phrase showing the university has an active NCAA Division I FCS football program (e.g., "NCAA Division I FCS").
    - fcs_sources: URLs (e.g., athletics site, NCAA page, conference page) that support the FCS program status.

    - fcs_conference_2025: The FCS conference in which the football program competed during the 2025 season (e.g., "Patriot League", "CAA", "Ivy League").
    - fcs_conference_sources: URLs that support 2025 FCS conference affiliation.

    - test_optional_policy: A phrase indicating the school is test-optional for SAT/ACT (e.g., "test-optional", "does not require SAT/ACT").
    - test_optional_sources: URLs (e.g., admissions policy page) that support test-optional status.

    - sat_middle_50_range: The middle 50% SAT score range (e.g., "1300-1480").
    - sat_sources: URLs that provide the middle 50% SAT range.

    - undergrad_enrollment: Undergrad enrollment (as a string, e.g., "6,800").
    - enrollment_sources: URLs that provide undergraduate enrollment numbers.

    - founding_year: The year the university was founded (as a string, e.g., "1842").
    - founding_sources: URLs that support the founding year.

    - undergrad_tuition: Annual undergraduate tuition (as a string, e.g., "$65,120" or "USD 65,120").
    - tuition_sources: URLs that provide undergraduate tuition for one academic year (exclude room and board).

    - metro_area: The major metropolitan area the university is near or within (e.g., "Philadelphia", "Pittsburgh").
    - metro_distance_miles: Distance in miles to that metro area if explicitly stated (as a string, e.g., "10 miles").
    - metro_sources: URLs that support proximity to a major metro area.

    - offers_bachelors: A phrase indicating bachelor's degree programs are offered (e.g., "offers bachelor's degrees" or "undergraduate programs").
    - program_sources: URLs that support that the university offers bachelor's degrees.

    SPECIAL RULES:
    - Only extract what is explicitly present in the answer. If something is not present, use null for that field, and use an empty list for sources fields.
    - For all URL lists, include every URL present for that attribute. If no attribute-specific URL is provided, leave the list empty.
    - Preserve the exact text of fields like sat_middle_50_range, undergrad_enrollment, undergrad_tuition, and founding_year as strings.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_all_urls(info: UniversityExtraction) -> List[str]:
    all_lists = [
        info.urls_general or [],
        info.location_sources or [],
        info.institution_sources or [],
        info.accreditation_sources or [],
        info.fcs_sources or [],
        info.fcs_conference_sources or [],
        info.test_optional_sources or [],
        info.sat_sources or [],
        info.enrollment_sources or [],
        info.founding_sources or [],
        info.tuition_sources or [],
        info.metro_sources or [],
        info.program_sources or [],
    ]
    merged = []
    for lst in all_lists:
        merged.extend(lst)
    return _dedup_preserve_order(merged)


def prefer_sources(primary: List[str], fallback_all: List[str]) -> List[str]:
    if primary and len(primary) > 0:
        return primary
    return fallback_all


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(evaluator: Evaluator, parent_node, info: UniversityExtraction) -> None:
    uni_name = info.university_name or "the university"
    all_urls = collect_all_urls(info)

    # 1) URL reference provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(all_urls),
        id="url_reference_provided",
        desc="A reference URL is provided that supports the identification and key attributes of the university",
        parent=parent_node,
        critical=True
    )

    # 2) Located in Pennsylvania
    node_loc_pa = evaluator.add_leaf(
        id="located_in_pennsylvania",
        desc="The university is located in the state of Pennsylvania",
        parent=parent_node,
        critical=True
    )
    claim_loc_pa = f"{uni_name} is located in Pennsylvania (PA)."
    await evaluator.verify(
        claim=claim_loc_pa,
        node=node_loc_pa,
        sources=prefer_sources(info.location_sources, all_urls),
        additional_instruction="Accept 'Pennsylvania' or 'PA' and campus location within Pennsylvania. Verify using official or reputable sources."
    )

    # 3) Private institution
    node_private = evaluator.add_leaf(
        id="private_institution",
        desc="The university is a private institution, not a public university",
        parent=parent_node,
        critical=True
    )
    claim_private = f"{uni_name} is a private university (not a public university)."
    await evaluator.verify(
        claim=claim_private,
        node=node_private,
        sources=prefer_sources(info.institution_sources, all_urls),
        additional_instruction="Check classification on the school's website or reputable directories; accept 'private research university' or 'private university' wording."
    )

    # 4) Regionally accredited
    node_accred = evaluator.add_leaf(
        id="regionally_accredited",
        desc="The university is regionally accredited by a recognized accrediting body",
        parent=parent_node,
        critical=True
    )
    accred_body = info.accreditation_body or ""
    claim_accred = (
        f"{uni_name} is institutionally (regionally) accredited by a recognized accrediting body"
        + (f" (e.g., {accred_body})." if accred_body else ".")
    )
    await evaluator.verify(
        claim=claim_accred,
        node=node_accred,
        sources=prefer_sources(info.accreditation_sources, all_urls),
        additional_instruction="Accept recognized institutional accreditors (e.g., MSCHE, NECHE, HLC, SACSCOC, WSCUC, NWCCU) recognized by US Dept. of Education or CHEA."
    )

    # 5) Active NCAA Division I FCS football program
    node_fcs_active = evaluator.add_leaf(
        id="has_ncaa_d1_fcs_football",
        desc="The university has an active NCAA Division I Football Championship Subdivision (FCS) football program",
        parent=parent_node,
        critical=True
    )
    claim_fcs_active = f"{uni_name} fields an active NCAA Division I FCS football program."
    await evaluator.verify(
        claim=claim_fcs_active,
        node=node_fcs_active,
        sources=prefer_sources(info.fcs_sources, all_urls),
        additional_instruction="Verify that the school's varsity football competes in NCAA Division I FCS (not FBS or lower divisions). Use athletics, NCAA, or conference sources."
    )

    # 6) FCS conference affiliation during 2025 season
    node_fcs_2025 = evaluator.add_leaf(
        id="football_conference_2025",
        desc="The university's football program was affiliated with an NCAA Division I FCS conference during the 2025 season",
        parent=parent_node,
        critical=True
    )
    conf_2025 = info.fcs_conference_2025 or ""
    claim_fcs_2025 = (
        f"In the 2025 season, {uni_name}'s football program competed in an NCAA Division I FCS conference"
        + (f" (e.g., {conf_2025})." if conf_2025 else ".")
    )
    await evaluator.verify(
        claim=claim_fcs_2025,
        node=node_fcs_2025,
        sources=prefer_sources(info.fcs_conference_sources, all_urls),
        additional_instruction="Confirm the program's 2025 season conference affiliation (FCS). Accept official athletics, conference, or NCAA pages showing 2025 membership/schedule."
    )

    # 7) Test-optional policy
    node_test_opt = evaluator.add_leaf(
        id="test_optional_policy",
        desc="The university has a test-optional admissions policy for SAT/ACT scores",
        parent=parent_node,
        critical=True
    )
    claim_test_opt = f"{uni_name} has a test-optional admissions policy for SAT/ACT scores."
    await evaluator.verify(
        claim=claim_test_opt,
        node=node_test_opt,
        sources=prefer_sources(info.test_optional_sources, all_urls),
        additional_instruction="Verify on admissions policy pages that first-year applicants may apply without SAT/ACT scores; accept terms like 'test-optional' or 'does not require SAT/ACT'."
    )

    # 8) SAT lower bound at least 1300 (middle 50% range)
    node_sat_lb = evaluator.add_leaf(
        id="sat_lower_bound_1300_plus",
        desc="The middle 50% SAT score range for admitted students has a lower bound of at least 1300",
        parent=parent_node,
        critical=True
    )
    sat_text = info.sat_middle_50_range or ""
    claim_sat_lb = (
        f"The middle 50% SAT score range for admitted students at {uni_name} has a lower bound of at least 1300"
        + (f" (reported as '{sat_text}')." if sat_text else ".")
    )
    await evaluator.verify(
        claim=claim_sat_lb,
        node=node_sat_lb,
        sources=prefer_sources(info.sat_sources, all_urls),
        additional_instruction="Check official admissions or common data set pages. The lower bound must be ≥1300; allow minor formatting variants and superscoring notes."
    )

    # 9) Undergraduate enrollment 5,000–10,000
    node_enroll = evaluator.add_leaf(
        id="enrollment_5000_to_10000",
        desc="Total undergraduate enrollment is between 5,000 and 10,000 students",
        parent=parent_node,
        critical=True
    )
    enroll_str = info.undergrad_enrollment or ""
    claim_enroll = (
        f"Total undergraduate enrollment at {uni_name} is between 5,000 and 10,000 students"
        + (f" (reported as '{enroll_str}')." if enroll_str else ".")
    )
    await evaluator.verify(
        claim=claim_enroll,
        node=node_enroll,
        sources=prefer_sources(info.enrollment_sources, all_urls),
        additional_instruction="Verify undergraduate (not total university) enrollment falls within 5,000–10,000; accept approximate numbers and minor rounding."
    )

    # 10) Founded before 1900
    node_founded = evaluator.add_leaf(
        id="founded_before_1900",
        desc="The university was founded before the year 1900",
        parent=parent_node,
        critical=True
    )
    founding_str = info.founding_year or ""
    claim_founded = (
        f"{uni_name} was founded before the year 1900"
        + (f" (founded in {founding_str})." if founding_str else ".")
    )
    await evaluator.verify(
        claim=claim_founded,
        node=node_founded,
        sources=prefer_sources(info.founding_sources, all_urls),
        additional_instruction="Verify the founding year is < 1900 using official or reputable historical sources."
    )

    # 11) Annual undergraduate tuition exceeds $60,000
    node_tuition = evaluator.add_leaf(
        id="tuition_exceeds_60000",
        desc="Annual undergraduate tuition exceeds $60,000",
        parent=parent_node,
        critical=True
    )
    tuition_str = info.undergrad_tuition or ""
    claim_tuition = (
        f"Annual undergraduate tuition at {uni_name} exceeds $60,000"
        + (f" (listed as '{tuition_str}')." if tuition_str else ".")
    )
    await evaluator.verify(
        claim=claim_tuition,
        node=node_tuition,
        sources=prefer_sources(info.tuition_sources, all_urls),
        additional_instruction="Use official tuition pages; consider base tuition for one academic year (exclude room/board and fees). Accept amounts ≥ $60,000."
    )

    # 12) Within 20 miles of a major metropolitan area
    node_metro = evaluator.add_leaf(
        id="within_20_miles_major_city",
        desc="The university is located within 20 miles of a major metropolitan area",
        parent=parent_node,
        critical=True
    )
    metro_area = info.metro_area or "a major metropolitan area"
    metro_dist = info.metro_distance_miles or ""
    claim_metro = (
        f"{uni_name} is located within 20 miles of {metro_area}"
        + (f" (reported distance: {metro_dist})." if metro_dist else ".")
    )
    await evaluator.verify(
        claim=claim_metro,
        node=node_metro,
        sources=prefer_sources(info.metro_sources, all_urls),
        additional_instruction="Accept explicit statements of proximity or clear campus location in a suburb within the metro area; if a distance is stated, it must be ≤ 20 miles."
    )

    # 13) Offers bachelor's degree programs
    node_bachelors = evaluator.add_leaf(
        id="offers_bachelors_programs",
        desc="The university offers bachelor's degree programs",
        parent=parent_node,
        critical=True
    )
    claim_bachelors = f"{uni_name} offers bachelor's degree programs."
    await evaluator.verify(
        claim=claim_bachelors,
        node=node_bachelors,
        sources=prefer_sources(info.program_sources, all_urls),
        additional_instruction="Verify undergraduate programs/majors or language indicating bachelor's degree offerings on official academic or admissions pages."
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
    Evaluate an answer for the Pennsylvania private university criteria task.
    """
    # Initialize evaluator with parallel aggregation at root
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

    # Extract structured university info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityExtraction,
        extraction_name="university_extraction",
    )

    # Record some custom info: all URLs collected
    all_urls = collect_all_urls(extracted_info)
    evaluator.add_custom_info(
        info={"total_urls_collected": len(all_urls), "all_urls": all_urls},
        info_type="url_collection",
        info_name="collected_urls"
    )

    # Build verification nodes and run checks under a single critical parallel node
    # To reflect rubric root criticality, we add a critical parallel child under root
    main_criteria_node = evaluator.add_parallel(
        id="main_criteria",
        desc="All specified criteria for the identified private Pennsylvania university",
        parent=root,
        critical=True
    )

    await build_and_verify_criteria(evaluator, main_criteria_node, extracted_info)

    # Return structured summary
    return evaluator.get_summary()