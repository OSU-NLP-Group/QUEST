import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_public_universities_multi_criteria_v1"
TASK_DESCRIPTION = """
Identify three public universities in the United States that meet ALL of the following criteria:

1. Must be a public (state-funded) university
2. Must hold regional institutional accreditation from a recognized accrediting body
3. Must have total enrollment between 15,000 and 40,000 students as of Fall 2025 or Fall 2026
4. Must operate on a semester system (not quarter, trimester, or other alternative calendar system)
5. Must define full-time undergraduate status as a minimum of 12 semester credit hours
6. Must require 120 semester credit hours (or equivalent) for bachelor's degree completion
7. Must offer at least one ABET-accredited bachelor's degree program in engineering or computer science
8. Must have NCAA Division I athletic programs
9. Must NOT have a mandatory on-campus housing requirement for all first-year students (either no requirement or exceptions must be available)
10. Must accept transfer credits from regionally accredited institutions with clearly documented policies
11. Must have a minimum transfer GPA requirement of 2.5 or lower for general admission consideration
12. Must be located in a state that borders at least one other US state (i.e., not Alaska or Hawaii)

For each university you identify, provide:
- The full official name of the institution
- The state where it is located
- A brief confirmation that it meets each criterion
- Reference URLs that verify each criterion

Your response should identify exactly three universities that satisfy all twelve criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Core identity
    name: Optional[str] = None
    state: Optional[str] = None

    # Criterion 1: Public institution
    public_status_urls: List[str] = Field(default_factory=list)

    # Criterion 2: Regional accreditation
    accreditation_body: Optional[str] = None
    accreditation_urls: List[str] = Field(default_factory=list)

    # Criterion 3: Enrollment and date
    enrollment_total: Optional[str] = None  # Keep as string to allow ranges/phrases
    enrollment_term: Optional[str] = None   # Expect "Fall 2025" or "Fall 2026"
    enrollment_urls: List[str] = Field(default_factory=list)

    # Criterion 4: Academic calendar
    calendar_system: Optional[str] = None   # Expect "semester", but keep free-form
    calendar_urls: List[str] = Field(default_factory=list)

    # Criterion 5: Full-time undergraduate definition
    full_time_ug_credits_min: Optional[str] = None  # Expect "12" but keep as string
    full_time_urls: List[str] = Field(default_factory=list)

    # Criterion 6: Degree requirements (120 credits)
    degree_credit_hours: Optional[str] = None       # Expect "120" or equivalent
    degree_req_urls: List[str] = Field(default_factory=list)

    # Criterion 7: ABET program
    abet_program_name: Optional[str] = None
    abet_program_area: Optional[str] = None  # "engineering" or "computer science"
    abet_urls: List[str] = Field(default_factory=list)

    # Criterion 8: NCAA Division I
    ncaa_division: Optional[str] = None
    ncaa_urls: List[str] = Field(default_factory=list)

    # Criterion 9: Housing policy (no universal mandate for all first-year)
    housing_policy_summary: Optional[str] = None
    housing_urls: List[str] = Field(default_factory=list)

    # Criterion 10: Transfer credit acceptance from regionally accredited institutions
    transfer_policy_summary: Optional[str] = None
    transfer_policy_urls: List[str] = Field(default_factory=list)

    # Criterion 11: Minimum transfer GPA <= 2.5
    transfer_min_gpa: Optional[str] = None
    transfer_gpa_urls: List[str] = Field(default_factory=list)

    # Criterion 12: Geographic location and bordering states
    location_urls: List[str] = Field(default_factory=list)  # State location/overview
    borders_urls: List[str] = Field(default_factory=list)   # Evidence about bordering states (can reuse state page)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to three universities that the answer explicitly identifies to satisfy the task. Return at most three.

    For each university, extract:
    1) name: Full official institution name as written in the answer
    2) state: U.S. state where the institution is located (full name or 2-letter abbreviation exactly as in the answer)

    For each criterion, extract all URLs that the answer cites to verify that criterion (do not invent URLs):
    3) public_status_urls: URLs confirming the institution is public/state-funded
    4) accreditation_body: Name of the regional institutional accrediting body if stated
       accreditation_urls: URLs confirming the institutional accreditation (HLC, MSCHE, SACSCOC, NWCCU, NECHE, WSCUC, etc.)
    5) enrollment_total: The total enrollment text/number the answer used
       enrollment_term: The term of the figure, e.g., "Fall 2025" or "Fall 2026", exactly as given
       enrollment_urls: URLs referencing the enrollment figure and term
    6) calendar_system: Academic calendar system text (e.g., "semester")
       calendar_urls: URLs confirming the calendar system
    7) full_time_ug_credits_min: Minimum credit hours that define full-time undergraduate status (text, e.g., "12")
       full_time_urls: URLs confirming the definition
    8) degree_credit_hours: Required bachelor's degree credit hours (text, e.g., "120")
       degree_req_urls: URLs confirming the 120-credit bachelor’s requirement (or equivalent)
    9) abet_program_name: Name of at least one ABET-accredited bachelor's program, if stated
       abet_program_area: The area such as "engineering" or "computer science", if stated
       abet_urls: URLs confirming ABET accreditation (ABET or official university/college accreditation page)
    10) ncaa_division: Division text if stated (e.g., "NCAA Division I")
        ncaa_urls: URLs confirming Division I status
    11) housing_policy_summary: Brief phrase indicating no universal mandatory on-campus housing (or exceptions exist)
        housing_urls: URLs confirming the housing policy
    12) transfer_policy_summary: Phrase indicating acceptance of transfer credits from regionally accredited institutions
        transfer_policy_urls: URLs confirming transfer credit acceptance policy
    13) transfer_min_gpa: Minimum transfer GPA for general admission consideration (text, e.g., "2.0", "2.25", "2.5")
        transfer_gpa_urls: URLs confirming the minimum transfer GPA
    14) location_urls: URLs confirming the institution's state location (e.g., official "About" page)
        borders_urls: URLs confirming that the state borders at least one other U.S. state (may reuse a state facts page)

    Rules:
    - Only extract URLs explicitly presented in the answer (plain links or markdown). Do not invent or infer new URLs.
    - If a field is missing in the answer, set it to null (for single value) or [] (for URL lists).
    - Return JSON with a "universities" array of up to three UniversityItem objects in the same order as the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_any_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _merge_urls(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for ul in url_lists:
        if ul:
            merged.extend(ul)
    # Keep order, remove empties and duplicates conservatively
    seen = set()
    deduped: List[str] = []
    for u in merged:
        if not u or not isinstance(u, str):
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Criterion verification builders                                             #
# --------------------------------------------------------------------------- #
async def _verify_public_status(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_institution_type",
        desc="The identified institution is a public university (state-funded institution)",
        parent=parent,
        critical=True
    )

    # Verification leaf
    leaf = evaluator.add_leaf(
        id=f"u{idx+1}_verify_public_status",
        desc="Institution is documented as a public, state-supported university",
        parent=crit,
        critical=True
    )
    claim = f"The institution '{u.name}' is a public (state-supported) university."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=u.public_status_urls,
        additional_instruction="Accept phrases like 'public university', 'public research university', 'state university', or explicit membership in a state university system."
    )

    # Reference existence leaf
    evaluator.add_custom_node(
        result=_has_any_urls(u.public_status_urls),
        id=f"u{idx+1}_public_reference_provided",
        desc="A valid reference URL is provided that confirms the public status",
        parent=crit,
        critical=True
    )


async def _verify_regional_accreditation(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_regional_accreditation",
        desc="The institution holds regional institutional accreditation from a recognized accrediting body",
        parent=parent,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"u{idx+1}_verify_accreditation",
        desc="Institution is accredited by HLC, MSCHE, SACSCOC, NWCCU, NECHE, WSCUC (WASC Senior), or another recognized regional accreditor",
        parent=crit,
        critical=True
    )
    # Keep the claim to what can be matched on-page (the accreditor's name)
    if u.accreditation_body:
        claim = f"The institution '{u.name}' is institutionally accredited by {u.accreditation_body}."
    else:
        claim = f"The institution '{u.name}' is institutionally accredited by a recognized U.S. regional accrediting body (e.g., HLC, MSCHE, SACSCOC, NWCCU, NECHE, WSCUC)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=u.accreditation_urls,
        additional_instruction="Pass if the page shows institutional accreditation by one of the main regional accreditors: HLC, MSCHE, SACSCOC, NWCCU, NECHE, or WSCUC (WASC Senior). Programmatic accreditations alone are not sufficient."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.accreditation_urls),
        id=f"u{idx+1}_accreditation_reference_provided",
        desc="A valid reference URL is provided that confirms the accreditation status",
        parent=crit,
        critical=True
    )


async def _verify_enrollment(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_enrollment_size",
        desc="The institution has total enrollment between 15,000 and 40,000 students as of Fall 2025 or Fall 2026",
        parent=parent,
        critical=True
    )

    # Verify range
    leaf_range = evaluator.add_leaf(
        id=f"u{idx+1}_verify_enrollment_range",
        desc="Total enrollment is documented as being within the 15,000-40,000 range",
        parent=crit,
        critical=True
    )
    claim_range = "The total student enrollment (overall headcount) is between 15,000 and 40,000."
    await evaluator.verify(
        claim=claim_range,
        node=leaf_range,
        sources=u.enrollment_urls,
        additional_instruction="Use the page's stated total enrollment number (overall headcount across all levels). Accept if the total is in the 15k–40k inclusive range."
    )

    # Verify date/term
    leaf_date = evaluator.add_leaf(
        id=f"u{idx+1}_verify_enrollment_date",
        desc="The enrollment figure is from Fall 2025 or Fall 2026",
        parent=crit,
        critical=True
    )
    term_text = u.enrollment_term or "Fall 2025 or Fall 2026"
    claim_date = f"The cited enrollment figure is specifically from Fall 2025 or Fall 2026 (e.g., '{term_text}')."
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=u.enrollment_urls,
        additional_instruction="Pass if the page clearly labels the enrollment figure as 'Fall 2025' or 'Fall 2026' (or an unmistakable equivalent like 'Autumn 2025/2026')."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.enrollment_urls),
        id=f"u{idx+1}_enrollment_reference_provided",
        desc="A valid reference URL is provided that confirms the enrollment data",
        parent=crit,
        critical=True
    )


async def _verify_calendar(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_academic_calendar",
        desc="The institution operates on a semester system (not quarter, trimester, or other alternative)",
        parent=parent,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"u{idx+1}_verify_semester_system",
        desc="Institution's academic calendar is documented as a semester system",
        parent=crit,
        critical=True
    )
    claim = f"The institution '{u.name}' operates on a semester academic calendar (not quarter or trimester)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=u.calendar_urls,
        additional_instruction="Look for explicit mention of 'semester' calendars in academic calendar/registrar pages; do not pass if quarter or trimester."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.calendar_urls),
        id=f"u{idx+1}_calendar_reference_provided",
        desc="A valid reference URL is provided that confirms the semester system",
        parent=crit,
        critical=True
    )


async def _verify_full_time(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_full_time_definition",
        desc="The institution defines full-time undergraduate status as a minimum of 12 semester credit hours",
        parent=parent,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"u{idx+1}_verify_credit_hour_requirement",
        desc="Full-time undergraduate status is documented as 12+ semester credit hours",
        parent=crit,
        critical=True
    )
    claim = "Full-time undergraduate status is defined as at least 12 semester credit hours."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=u.full_time_urls,
        additional_instruction="Pass if the page explicitly states that 12 (or more) credit hours constitute full-time status for undergraduates."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.full_time_urls),
        id=f"u{idx+1}_full_time_reference_provided",
        desc="A valid reference URL is provided that confirms this definition",
        parent=crit,
        critical=True
    )


async def _verify_degree_requirements(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_degree_requirements",
        desc="The institution requires 120 semester credit hours (or equivalent) for bachelor's degree completion",
        parent=parent,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"u{idx+1}_verify_credit_hour_total",
        desc="Bachelor's degree requirements are documented as 120 semester credit hours or equivalent",
        parent=crit,
        critical=True
    )
    claim = "A bachelor's degree requires 120 semester credit hours (or an explicit equivalent total for graduation)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=u.degree_req_urls,
        additional_instruction="Pass if the general undergraduate degree requirement is 120 credits (or an explicitly equivalent total such as 120 semester hours)."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.degree_req_urls),
        id=f"u{idx+1}_degree_req_reference_provided",
        desc="A valid reference URL is provided that confirms degree requirements",
        parent=crit,
        critical=True
    )


async def _verify_abet(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_abet_accreditation",
        desc="The institution offers at least one ABET-accredited bachelor's degree program in engineering or computer science",
        parent=parent,
        critical=True
    )

    # ABET program existence
    leaf_prog = evaluator.add_leaf(
        id=f"u{idx+1}_verify_abet_program",
        desc="At least one bachelor's program is documented as ABET-accredited",
        parent=crit,
        critical=True
    )
    prog_name = u.abet_program_name or "at least one bachelor's program"
    claim_prog = f"The institution offers {prog_name} that is ABET-accredited."
    await evaluator.verify(
        claim=claim_prog,
        node=leaf_prog,
        sources=u.abet_urls,
        additional_instruction="Pass if the ABET database or an official college/program page confirms ABET accreditation for any bachelor's program."
    )

    # ABET program type (engineering or CS)
    leaf_type = evaluator.add_leaf(
        id=f"u{idx+1}_verify_program_type",
        desc="The ABET-accredited program is in engineering or computer science",
        parent=crit,
        critical=True
    )
    area_text = u.abet_program_area or "engineering or computer science"
    claim_type = f"The ABET-accredited bachelor's program is in {area_text}."
    await evaluator.verify(
        claim=claim_type,
        node=leaf_type,
        sources=u.abet_urls,
        additional_instruction="Pass if the accredited bachelor's program is clearly within engineering (any discipline) or computer science/computing (e.g., CS, CE, SE, etc.)."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.abet_urls),
        id=f"u{idx+1}_abet_reference_provided",
        desc="A valid reference URL is provided that confirms ABET accreditation",
        parent=crit,
        critical=True
    )


async def _verify_ncaa_division(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_ncaa_division",
        desc="The institution has NCAA Division I athletic programs",
        parent=parent,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"u{idx+1}_verify_division_status",
        desc="Institution's NCAA Division I status is documented",
        parent=crit,
        critical=True
    )
    claim = f"The institution '{u.name}' competes in NCAA Division I athletics."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=u.ncaa_urls,
        additional_instruction="Pass if the page (e.g., NCAA profile, athletics site, or Wikipedia athletics section) clearly states NCAA Division I participation."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.ncaa_urls),
        id=f"u{idx+1}_ncaa_reference_provided",
        desc="A valid reference URL is provided that confirms NCAA Division I status",
        parent=crit,
        critical=True
    )


async def _verify_housing_policy(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_housing_policy",
        desc="The institution does NOT have a mandatory on-campus housing requirement for all first-year students",
        parent=parent,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"u{idx+1}_verify_no_mandatory_housing",
        desc="Housing policy is documented as either having no requirement or allowing exceptions for first-year students",
        parent=crit,
        critical=True
    )
    claim = ("The institution does not universally mandate on-campus housing for all first-year students; "
             "either no requirement exists or published, commonly-available exemptions allow living off campus.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=u.housing_urls,
        additional_instruction="Pass if the policy states no universal mandate OR lists standard exemptions (e.g., age, distance/commuter, marital status, dependents, military, financial hardship, etc.) that allow first-year students to live off campus."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.housing_urls),
        id=f"u{idx+1}_housing_reference_provided",
        desc="A valid reference URL is provided that confirms the housing policy",
        parent=crit,
        critical=True
    )


async def _verify_transfer_policy(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_transfer_credit_policy",
        desc="The institution accepts transfer credits from regionally accredited institutions with clearly documented policies",
        parent=parent,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"u{idx+1}_verify_accepts_transfer",
        desc="Institution's transfer credit acceptance policy is documented and includes credits from regionally accredited institutions",
        parent=crit,
        critical=True
    )
    claim = "The institution's transfer credit policy accepts credits from regionally accredited institutions."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=u.transfer_policy_urls,
        additional_instruction="Pass if the policy explicitly mentions accepting transfer credits from regionally accredited colleges/universities. Program/course-specific limitations are acceptable."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.transfer_policy_urls),
        id=f"u{idx+1}_transfer_policy_reference_provided",
        desc="A valid reference URL is provided that confirms the transfer credit policy",
        parent=crit,
        critical=True
    )


async def _verify_transfer_gpa(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_transfer_gpa",
        desc="The institution has a minimum transfer GPA requirement of 2.5 or lower for general admission consideration",
        parent=parent,
        critical=True
    )

    # Combine URLs if GPA and policy are the same page
    urls = _merge_urls(u.transfer_gpa_urls, u.transfer_policy_urls)

    leaf = evaluator.add_leaf(
        id=f"u{idx+1}_verify_gpa_requirement",
        desc="Minimum transfer GPA is documented as 2.5 or lower",
        parent=crit,
        critical=True
    )
    claim = "For general transfer admission consideration, the minimum GPA requirement is 2.5 or lower."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Pass if the general/minimum transfer admission GPA threshold is 2.5, 2.25, 2.0, etc. If multiple colleges have higher thresholds, still pass if the baseline university-wide minimum is ≤ 2.5."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(urls),
        id=f"u{idx+1}_transfer_gpa_reference_provided",
        desc="A valid reference URL is provided that confirms the minimum transfer GPA",
        parent=crit,
        critical=True
    )


async def _verify_geography(evaluator: Evaluator, parent, idx: int, u: UniversityItem):
    crit = evaluator.add_parallel(
        id=f"u{idx+1}_criterion_geographic_location",
        desc="The institution is located in a state that borders at least one other US state",
        parent=parent,
        critical=True
    )

    # Verify state location (US state)
    leaf_loc = evaluator.add_leaf(
        id=f"u{idx+1}_verify_state_location",
        desc="Institution's state location is documented",
        parent=crit,
        critical=True
    )
    claim_loc = f"The institution '{u.name}' is located in the U.S. state of {u.state}."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=u.location_urls,
        additional_instruction="Pass if the official site or authoritative page shows the campus location in the stated U.S. state."
    )

    # Verify that the state borders at least one other U.S. state (i.e., not AK or HI)
    leaf_border = evaluator.add_leaf(
        id=f"u{idx+1}_verify_borders_state",
        desc="The state borders at least one other US state (not Alaska or Hawaii)",
        parent=crit,
        critical=True
    )
    border_sources = _merge_urls(u.borders_urls, u.location_urls)
    claim_border = f"The state of {u.state} borders at least one other U.S. state (i.e., it is not Alaska or Hawaii)."
    await evaluator.verify(
        claim=claim_border,
        node=leaf_border,
        sources=border_sources,
        additional_instruction="Pass if the state is one of the contiguous 48 (or otherwise borders another state, e.g., Michigan). A general state facts page listing neighboring states suffices."
    )

    evaluator.add_custom_node(
        result=_has_any_urls(u.location_urls),
        id=f"u{idx+1}_geography_reference_provided",
        desc="A valid reference URL is provided that confirms the institution's location",
        parent=crit,
        critical=True
    )


# --------------------------------------------------------------------------- #
# University verification orchestrator                                        #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, root_parent, idx: int, u: UniversityItem):
    uni_node = evaluator.add_parallel(
        id=f"university_{idx+1}",
        desc=f"The {'first' if idx==0 else 'second' if idx==1 else 'third'} identified university meets all 12 criteria",
        parent=root_parent,
        critical=False
    )

    # Build all 12 criteria subtrees (each critical internally)
    await _verify_public_status(evaluator, uni_node, idx, u)
    await _verify_regional_accreditation(evaluator, uni_node, idx, u)
    await _verify_enrollment(evaluator, uni_node, idx, u)
    await _verify_calendar(evaluator, uni_node, idx, u)
    await _verify_full_time(evaluator, uni_node, idx, u)
    await _verify_degree_requirements(evaluator, uni_node, idx, u)
    await _verify_abet(evaluator, uni_node, idx, u)
    await _verify_ncaa_division(evaluator, uni_node, idx, u)
    await _verify_housing_policy(evaluator, uni_node, idx, u)
    await _verify_transfer_policy(evaluator, uni_node, idx, u)
    await _verify_transfer_gpa(evaluator, uni_node, idx, u)
    await _verify_geography(evaluator, uni_node, idx, u)


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
    Evaluate an answer for the 'three US public universities meeting 12 criteria' task.
    """
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

    # Extract up to three universities with per-criterion URLs
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Normalize to exactly three entries (pad with empty if fewer; truncate if more)
    universities: List[UniversityItem] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Optional info for debugging/visibility
    evaluator.add_custom_info(
        info={
            "extracted_university_count": len(extracted.universities),
            "used_university_count": 3
        },
        info_type="extraction_stats",
        info_name="extraction_overview"
    )

    # Build verification for each of the three universities
    for i in range(3):
        await verify_university(evaluator, root, i, universities[i])

    return evaluator.get_summary()