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
TASK_ID = "educational_leaders_edd_mba"
TASK_DESCRIPTION = """Identify two educational leaders who meet ALL of the following criteria: (1) Hold a Doctor of Education (EdD) degree in a field related to educational administration, leadership, management, or policy from a U.S. university; (2) Hold a Master of Business Administration (MBA) degree from a U.S. university; (3) Transitioned to a senior educational leadership position between January 1, 2015 and December 31, 2025, where senior leadership position is defined as: university president, business school dean, or superintendent of a school district serving 100,000 or more students; (4) The institution they currently lead or led during this period is located in a U.S. state along the East Coast or in the Mid-Atlantic region, specifically: Maine, New Hampshire, Vermont, Massachusetts, Rhode Island, Connecticut, New York, New Jersey, Pennsylvania, Delaware, Maryland, Virginia, North Carolina, South Carolina, Georgia, or Florida; (5) Completed their EdD degree no more than 2 years after their appointment to the senior leadership position (the EdD may have been completed before or up to 2 years after the appointment); (6) Met the prior experience requirement: either spent at least 10 years in the business/consulting sector before transitioning to educational leadership, OR held at least one prior educational leadership or administrative role before receiving their senior appointment. For each of the two individuals, provide: full name, current or most recent senior leadership position title, institution name, EdD degree field of study and granting institution, MBA granting institution, year of appointment to the senior leadership position, reference URL documenting their educational credentials (EdD and MBA), reference URL documenting their leadership appointment, and reference URL documenting their current institution and position."""


ALLOWED_STATES = [
    "Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut",
    "New York", "New Jersey", "Pennsylvania", "Delaware", "Maryland", "Virginia",
    "North Carolina", "South Carolina", "Georgia", "Florida"
]

ALLOWED_ROLE_CATEGORIES = ["president", "dean", "superintendent"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Individual(BaseModel):
    name: Optional[str] = None
    position_title: Optional[str] = None
    institution_name: Optional[str] = None
    institution_state: Optional[str] = None
    role_category: Optional[str] = None  # expected: "president" | "dean" | "superintendent"
    appointment_year: Optional[str] = None
    appointment_date: Optional[str] = None

    edd_degree_label: Optional[str] = None  # e.g., "EdD", "Doctor of Education"
    edd_field: Optional[str] = None
    edd_institution: Optional[str] = None
    edd_completion_year: Optional[str] = None

    mba_degree_label: Optional[str] = None  # e.g., "MBA"
    mba_institution: Optional[str] = None

    district_student_population: Optional[str] = None  # if superintendent
    prior_experience_summary: Optional[str] = None

    credentials_urls: List[str] = Field(default_factory=list)
    appointment_urls: List[str] = Field(default_factory=list)
    institution_urls: List[str] = Field(default_factory=list)


class LeadersExtraction(BaseModel):
    individuals: List[Individual] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_leaders() -> str:
    return """
Extract up to TWO individuals described in the answer who are educational leaders. For EACH individual, extract the following fields exactly as stated in the answer text:

- name: Full name of the individual.
- position_title: Current or most recent senior leadership position title as presented (e.g., "President", "Dean of the School of Business", "Superintendent").
- institution_name: The institution they currently lead or led during the relevant period.
- institution_state: The U.S. state of the institution (if explicitly stated).
- role_category: Normalize based on the title in the answer into one of ["president", "dean", "superintendent"] using the answer's wording (do not invent); if unclear from the answer, return null.
- appointment_year: A 4-digit year of appointment to the senior leadership position if present.
- appointment_date: Any appointment date string if present (e.g., "June 2018"). Return null if absent.

- edd_degree_label: The exact degree label for the doctorate (e.g., "EdD", "Ed.D.", "Doctor of Education").
- edd_field: The field/area of the EdD (e.g., "Educational Leadership", "Education Policy") as stated.
- edd_institution: The granting institution for the EdD.
- edd_completion_year: A 4-digit year of EdD completion if provided (or a year mentioned; otherwise null).

- mba_degree_label: The exact degree label for MBA (e.g., "MBA") as stated.
- mba_institution: The granting institution for the MBA.

- district_student_population: If the role is a superintendent and the answer mentions student count for the district, extract that number or phrase; otherwise null.

- prior_experience_summary: Any summary of prior experience relevant to the requirement (e.g., "15 years at McKinsey", "served as provost", etc.), exactly as mentioned in the answer. If not mentioned, return null.

- credentials_urls: List of URLs cited in the answer that document educational credentials (EdD and/or MBA). Only include URLs explicitly present in the answer.
- appointment_urls: List of URLs cited that document the leadership appointment (e.g., official announcement, reliable news). Only include URLs explicitly present in the answer.
- institution_urls: List of URLs cited that document the current institution and position (e.g., official bio page). Only include URLs explicitly present in the answer.

Rules:
- Do not invent any information. Only extract what is explicitly present in the answer.
- If the answer provides more than two individuals, extract only the first two in order of appearance.
- If any field is missing, set it to null or an empty list as appropriate.
- Ensure URLs are valid and include protocol. If protocol is missing, prepend http://.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_year(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.findall(r"\b(19|20)\d{2}\b", s)
    if not m:
        # try alternative findall capturing full year text
        m2 = re.findall(r"\b((?:19|20)\d{2})\b", s)
        if m2:
            try:
                y = int(m2[0])
                return y
            except Exception:
                return None
        return None
    # m contains group matches for first two digits; try matching full year
    m3 = re.search(r"\b((?:19|20)\d{2})\b", s)
    if m3:
        try:
            return int(m3.group(1))
        except Exception:
            return None
    return None


def has_edd_label(label: Optional[str]) -> bool:
    if not label:
        return False
    t = label.strip().lower()
    return ("edd" in t) or ("ed.d" in t) or ("doctor of education" in t)


def has_mba_label(label: Optional[str]) -> bool:
    if not label:
        return False
    return "mba" in label.strip().lower()


def normalize_sources_list(a: Optional[List[str]], b: Optional[List[str]] = None) -> List[str]:
    items = []
    if a:
        items.extend([u for u in a if isinstance(u, str) and u.strip() != ""])
    if b:
        items.extend([u for u in b if isinstance(u, str) and u.strip() != ""])
    # de-duplicate preserving order
    seen = set()
    uniq = []
    for u in items:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def allowed_states_text() -> str:
    return ", ".join(ALLOWED_STATES)


# --------------------------------------------------------------------------- #
# Verification logic for a single individual                                  #
# --------------------------------------------------------------------------- #
async def verify_individual(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    person: Individual,
) -> None:
    # Individual Node (Non-critical; parallel)
    indiv_node = evaluator.add_parallel(
        id=f"individual_{idx+1}",
        desc=f"{'First' if idx == 0 else 'Second'} educational leader meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Documentation Node (Non-critical; contains presence and support checks)
    docs_node = evaluator.add_parallel(
        id=f"ind_{idx+1}_documentation",
        desc="Educational credentials, appointment, and current position are publicly documented with reference URLs",
        parent=indiv_node,
        critical=False
    )

    # URL presence checks (non-critical under documentation)
    cred_urls_present_node = evaluator.add_custom_node(
        result=bool(person.credentials_urls),
        id=f"ind_{idx+1}_credentials_urls_present",
        desc="Credentials reference URL(s) provided",
        parent=docs_node,
        critical=False
    )
    appt_urls_present_node = evaluator.add_custom_node(
        result=bool(person.appointment_urls),
        id=f"ind_{idx+1}_appointment_urls_present",
        desc="Appointment reference URL(s) provided",
        parent=docs_node,
        critical=False
    )
    inst_urls_present_node = evaluator.add_custom_node(
        result=bool(person.institution_urls),
        id=f"ind_{idx+1}_institution_urls_present",
        desc="Institution/position reference URL(s) provided",
        parent=docs_node,
        critical=False
    )
    leadership_sources_present_node = evaluator.add_custom_node(
        result=bool(person.appointment_urls or person.institution_urls),
        id=f"ind_{idx+1}_leadership_sources_present",
        desc="At least one leadership-related reference URL (appointment or institution) provided",
        parent=docs_node,
        critical=False
    )

    # Appointment reference support
    appt_support_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_appointment_reference_supported",
        desc="Leadership appointment documented with reference URL from official announcement or reliable news source",
        parent=docs_node,
        critical=False
    )
    appt_claim = f"This page documents the appointment of {person.name or 'the individual'} as {person.position_title or 'the stated position'} at {person.institution_name or 'the institution'} in or around {person.appointment_year or 'the stated year'}."
    await evaluator.verify(
        claim=appt_claim,
        node=appt_support_node,
        sources=person.appointment_urls if person.appointment_urls else None,
        additional_instruction="Confirm the page is an official announcement or reliable news source that explicitly states the appointment and, if available, the year.",
        extra_prerequisites=[appt_urls_present_node] if appt_urls_present_node else None
    )

    # Institution/position reference support
    inst_support_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_institution_reference_supported",
        desc="Current institution and position documented with reference URL",
        parent=docs_node,
        critical=False
    )
    inst_claim = f"This page shows that {person.name or 'the individual'} currently (or most recently) holds the position '{person.position_title or 'the stated position'}' at {person.institution_name or 'the institution'}."
    await evaluator.verify(
        claim=inst_claim,
        node=inst_support_node,
        sources=person.institution_urls if person.institution_urls else None,
        additional_instruction="Look for explicit confirmation of the individual's current (or most recent) position and the institution.",
        extra_prerequisites=[inst_urls_present_node] if inst_urls_present_node else None
    )

    # Helper gate: Is Superintendent? (non-critical; used as prerequisite)
    is_superintendent_gate = evaluator.add_custom_node(
        result=((person.role_category or "").strip().lower() == "superintendent") or (
            isinstance(person.position_title, str) and "superintendent" in person.position_title.lower()
        ),
        id=f"ind_{idx+1}_is_superintendent_gate",
        desc="Role is superintendent (gate for population check)",
        parent=docs_node,
        critical=False
    )

    # Educational Credentials (Critical; parallel)
    edu_node = evaluator.add_parallel(
        id=f"ind_{idx+1}_educational_credentials",
        desc="Holds required doctoral and business degrees from U.S. institutions",
        parent=indiv_node,
        critical=True
    )

    # EdD Degree group (Critical; parallel)
    edd_group = evaluator.add_parallel(
        id=f"ind_{idx+1}_edd_degree",
        desc="Holds EdD in educational administration, leadership, management, or policy from a U.S. university",
        parent=edu_node,
        critical=True
    )

    # EdD present (Critical existence check)
    edd_present_node = evaluator.add_custom_node(
        result=bool(person.edd_institution) and bool(person.edd_field) and has_edd_label(person.edd_degree_label),
        id=f"ind_{idx+1}_edd_present",
        desc="EdD degree information present in the answer",
        parent=edd_group,
        critical=True
    )

    # EdD supported by credentials URLs
    edd_supported_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_edd_supported_by_credentials",
        desc="Credentials reference supports EdD credential and awarding institution",
        parent=edd_group,
        critical=True
    )
    edd_claim = f"{person.name or 'The individual'} holds a Doctor of Education (EdD) in {person.edd_field or '[field]'} from {person.edd_institution or '[EdD institution]'}."
    await evaluator.verify(
        claim=edd_claim,
        node=edd_supported_node,
        sources=person.credentials_urls if person.credentials_urls else None,
        additional_instruction="Verify that the page explicitly states the EdD degree (or equivalent label like 'Doctor of Education'), the field/area, and the awarding institution.",
        extra_prerequisites=[cred_urls_present_node, edd_present_node] if cred_urls_present_node and edd_present_node else None
    )

    # EdD field relevance (simple logical check)
    edd_field_relevance_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_edd_field_relevance",
        desc="EdD field is related to educational administration, leadership, management, or policy",
        parent=edd_group,
        critical=True
    )
    field_rel_claim = f"The EdD field of study '{person.edd_field or ''}' is related to educational administration, leadership, management, or policy."
    await evaluator.verify(
        claim=field_rel_claim,
        node=edd_field_relevance_node,
        additional_instruction="Treat fields such as 'Educational Leadership', 'Education Administration', 'Education Policy', 'Higher Education Administration', or similar as related. If field is missing or clearly unrelated, mark incorrect."
    )

    # EdD in US
    edd_us_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_edd_in_us",
        desc="EdD awarding institution is a U.S. university",
        parent=edd_group,
        critical=True
    )
    edd_us_claim = f"{person.edd_institution or '[EdD institution]'} is a U.S. university."
    await evaluator.verify(
        claim=edd_us_claim,
        node=edd_us_node,
        sources=person.credentials_urls if person.credentials_urls else None,
        additional_instruction="Verify the awarding institution is U.S.-based using the provided credentials page(s). Look for city/state or country on the page.",
        extra_prerequisites=[cred_urls_present_node, edd_present_node] if cred_urls_present_node and edd_present_node else None
    )

    # EdD completion year supported
    edd_year_supported_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_edd_year_supported",
        desc="EdD completion year supported by credentials reference",
        parent=edd_group,
        critical=True
    )
    edd_year_claim = f"The EdD was completed in {person.edd_completion_year or '[year]'}."
    await evaluator.verify(
        claim=edd_year_claim,
        node=edd_year_supported_node,
        sources=person.credentials_urls if person.credentials_urls else None,
        additional_instruction="Confirm a year is indicated for the EdD completion on the credentials page. If no year is present on the page, this should be incorrect.",
        extra_prerequisites=[cred_urls_present_node, edd_present_node] if cred_urls_present_node and edd_present_node else None
    )

    # MBA Degree group (Critical; parallel)
    mba_group = evaluator.add_parallel(
        id=f"ind_{idx+1}_mba_degree",
        desc="Holds MBA degree from a U.S. university",
        parent=edu_node,
        critical=True
    )

    # MBA present
    mba_present_node = evaluator.add_custom_node(
        result=bool(person.mba_institution) and has_mba_label(person.mba_degree_label),
        id=f"ind_{idx+1}_mba_present",
        desc="MBA degree information present in the answer",
        parent=mba_group,
        critical=True
    )

    # MBA supported by credentials URLs
    mba_supported_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_mba_supported_by_credentials",
        desc="Credentials reference supports MBA credential and awarding institution",
        parent=mba_group,
        critical=True
    )
    mba_claim = f"{person.name or 'The individual'} holds an MBA from {person.mba_institution or '[MBA institution]'}."
    await evaluator.verify(
        claim=mba_claim,
        node=mba_supported_node,
        sources=person.credentials_urls if person.credentials_urls else None,
        additional_instruction="Verify the MBA credential and awarding institution are explicitly stated on the referenced page(s).",
        extra_prerequisites=[cred_urls_present_node, mba_present_node] if cred_urls_present_node and mba_present_node else None
    )

    # MBA in US
    mba_us_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_mba_in_us",
        desc="MBA awarding institution is a U.S. university",
        parent=mba_group,
        critical=True
    )
    mba_us_claim = f"{person.mba_institution or '[MBA institution]'} is a U.S. university."
    await evaluator.verify(
        claim=mba_us_claim,
        node=mba_us_node,
        sources=person.credentials_urls if person.credentials_urls else None,
        additional_instruction="Verify the MBA awarding institution is U.S.-based using the provided credentials page(s). Look for city/state or country on the page.",
        extra_prerequisites=[cred_urls_present_node, mba_present_node] if cred_urls_present_node and mba_present_node else None
    )

    # EdD Timing (Critical) - computed logical check, depends on verified years
    edd_year_int = parse_year(person.edd_completion_year)
    appt_year_int = parse_year(person.appointment_year)
    edd_timing_ok = (edd_year_int is not None and appt_year_int is not None and (edd_year_int <= appt_year_int + 2))
    evaluator.add_custom_node(
        result=edd_timing_ok,
        id=f"ind_{idx+1}_edd_timing",
        desc="EdD completed no more than 2 years after appointment to senior leadership position",
        parent=edu_node,
        critical=True
    )

    # Leadership Position Details (Critical; parallel)
    lead_node = evaluator.add_parallel(
        id=f"ind_{idx+1}_leadership_details",
        desc="Currently serves or served in qualifying senior leadership role with proper timing and location",
        parent=indiv_node,
        critical=True
    )

    # Role type validity
    role_type_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_role_type_valid",
        desc="Position is university president, business school dean, or superintendent",
        parent=lead_node,
        critical=True
    )
    # Construct sources for role validation
    role_sources = normalize_sources_list(person.institution_urls, person.appointment_urls)
    role_claim = f"The role '{person.position_title or ''}' at '{person.institution_name or ''}' qualifies as one of: university president, business school dean, or superintendent."
    await evaluator.verify(
        claim=role_claim,
        node=role_type_node,
        sources=role_sources if role_sources else None,
        additional_instruction="Confirm from the page that the role is a university president, a dean of a business school (i.e., a business school or school of business), or a school district superintendent.",
        extra_prerequisites=[leadership_sources_present_node] if leadership_sources_present_node else None
    )

    # Superintendent population requirement (only if superintendent)
    sup_pop_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_superintendent_population",
        desc="If superintendent, the school district serves 100,000+ students",
        parent=lead_node,
        critical=True
    )
    sup_pop_claim = "The school district serves at least 100,000 students."
    await evaluator.verify(
        claim=sup_pop_claim,
        node=sup_pop_node,
        sources=role_sources if role_sources else None,
        additional_instruction="Verify, if the role is a superintendent, that the page indicates the district serves >= 100,000 students. If the role is not superintendent, this check should be skipped.",
        extra_prerequisites=[leadership_sources_present_node, is_superintendent_gate] if leadership_sources_present_node and is_superintendent_gate else None
    )

    # Appointment year supported by appointment URL
    appt_year_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_appointment_year_supported",
        desc="Appointment year is supported by appointment reference",
        parent=lead_node,
        critical=True
    )
    appt_year_claim = f"The individual was appointed in {person.appointment_year or '[year]' }."
    await evaluator.verify(
        claim=appt_year_claim,
        node=appt_year_node,
        sources=person.appointment_urls if person.appointment_urls else None,
        additional_instruction="Confirm the appointment year on the appointment announcement page.",
        extra_prerequisites=[appt_urls_present_node] if appt_urls_present_node else None
    )

    # Appointment timeframe (2015-2025 inclusive) logical check
    appt_year_ok = appt_year_int is not None and 2015 <= appt_year_int <= 2025
    evaluator.add_custom_node(
        result=appt_year_ok,
        id=f"ind_{idx+1}_appointment_timeframe",
        desc="Appointed to senior leadership position between January 1, 2015 and December 31, 2025",
        parent=lead_node,
        critical=True
    )

    # Geographic Location check
    geo_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_geographic_location",
        desc="Institution located in East Coast or Mid-Atlantic state",
        parent=lead_node,
        critical=True
    )
    geo_claim = f"The institution {person.institution_name or 'the institution'} is located in one of the allowed states: {allowed_states_text()}."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=person.institution_urls if person.institution_urls else None,
        additional_instruction=f"Use the page to determine the institution's state. It must be one of: {allowed_states_text()}.",
        extra_prerequisites=[inst_urls_present_node] if inst_urls_present_node else None
    )

    # Prior Experience Requirement (Critical)
    prior_exp_node = evaluator.add_leaf(
        id=f"ind_{idx+1}_prior_experience",
        desc="Met prior experience requirement: 10+ years in business/consulting OR prior educational leadership/administrative role",
        parent=indiv_node,
        critical=True
    )
    prior_claim = (
        f"Before appointment in {person.appointment_year or '[year]'}, "
        f"{person.name or 'the individual'} either spent at least 10 years in the business/consulting sector, "
        f"or held at least one prior educational leadership or administrative role."
    )
    all_urls = normalize_sources_list(person.credentials_urls, normalize_sources_list(person.appointment_urls, person.institution_urls))
    await evaluator.verify(
        claim=prior_claim,
        node=prior_exp_node,
        sources=all_urls if all_urls else None,
        additional_instruction="Pass if the page(s) show either 10+ years in business/consulting OR any prior educational leadership/administrative role (e.g., dean, provost, principal, associate dean) prior to the senior appointment."
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation across two individuals
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

    # Extract individuals
    extraction = await evaluator.extract(
        prompt=prompt_extract_leaders(),
        template_class=LeadersExtraction,
        extraction_name="extracted_leaders"
    )

    # Ensure exactly two individuals (pad with empty if fewer)
    individuals: List[Individual] = extraction.individuals[:2]
    while len(individuals) < 2:
        individuals.append(Individual())

    # Top-level task node (to mirror rubric naming; optional parallel container)
    task_node = evaluator.add_parallel(
        id="educational_leaders_task",
        desc="Identify two educational leaders who hold both EdD and MBA degrees and transitioned to senior leadership positions in East Coast/Mid-Atlantic institutions between 2015-2025",
        parent=root,
        critical=False
    )

    # Verify first and second individuals
    await verify_individual(evaluator, task_node, 0, individuals[0])
    await verify_individual(evaluator, task_node, 1, individuals[1])

    # Return summary
    return evaluator.get_summary()