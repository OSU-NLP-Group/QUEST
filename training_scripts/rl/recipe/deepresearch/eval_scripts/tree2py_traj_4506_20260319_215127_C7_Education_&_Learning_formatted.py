import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nc_sc_school_cal_2025_26"
TASK_DESCRIPTION = (
    "Identify two public school districts in the southeastern United States (specifically North Carolina or South Carolina) "
    "where BOTH districts meet ALL of the following criteria for their 2025-2026 traditional calendar schools: "
    "(1) Both are public K-12 school systems located in either North Carolina or South Carolina; "
    "(2) Both offer traditional calendar schools; "
    "(3) Both have their first day of school for the 2025-26 academic year in August 2025; "
    "(4) Both provide at least 180 instructional days for students; "
    "(5) Both schedule teacher workdays within state-mandated limits; "
    "(6) Both have at least 10 designated teacher workdays in 2025-26; "
    "(7) Both include a winter break spanning late December 2025 into early January 2026; "
    "(8) Both designate specific days for teacher professional development; "
    "(9) Both operate high schools with state-sanctioned athletic programs; "
    "(10) Both make calendars publicly available on official websites; "
    "(11) Both end the academic year in June 2026; "
    "(12) Both comply with state education department calendar requirements. "
    "Provide the official name, state, and reference URL(s) for each district."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class District(BaseModel):
    # Identifying info
    name: Optional[str] = None
    state: Optional[str] = None  # Accept "NC"/"SC" or "North Carolina"/"South Carolina"
    # Evidence URLs (from the answer; must be official where possible)
    calendar_urls: List[str] = Field(default_factory=list, description="Official district URL(s) to the 2025–26 traditional academic calendar (webpage or PDF)")
    official_site_urls: List[str] = Field(default_factory=list, description="Other official district webpages (About, Schools, Calendar hub, Board policy, etc.)")
    athletics_urls: List[str] = Field(default_factory=list, description="Official district or high school athletics pages indicating participation in state-sanctioned associations (NCHSAA or SCHSL)")
    # Optional details that might be present in the answer
    first_day_text: Optional[str] = None
    last_day_text: Optional[str] = None
    instructional_days_text: Optional[str] = None
    teacher_workdays_count_text: Optional[str] = None
    pd_days_text: Optional[str] = None
    winter_break_text: Optional[str] = None
    calendar_type_text: Optional[str] = None  # e.g., "Traditional"
    compliance_note: Optional[str] = None


class DistrictsExtraction(BaseModel):
    districts: List[District] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    From the answer, extract exactly the first two public school districts that the answer is presenting as meeting the 2025–26 traditional-calendar criteria in North Carolina or South Carolina. 
    If the answer lists more than two, extract only the first two mentioned; if fewer than two are present, extract what is available and leave missing fields as null or empty lists as appropriate.

    For each district, extract:
    - name: The official district name (e.g., "Wake County Public School System").
    - state: The state as provided (accept "NC"/"SC" or "North Carolina"/"South Carolina").
    - calendar_urls: All official district URL(s) in the answer that point to the 2025–26 traditional academic calendar page or PDF.
    - official_site_urls: Any additional official district webpages cited in the answer (homepage, calendar hub, policies, school listings, etc.).
    - athletics_urls: Any official district or high school athletics pages cited that indicate state-sanctioned participation (e.g., NCHSAA in NC, SCHSL in SC).
    - first_day_text: Any text in the answer indicating first day of school for 2025–26.
    - last_day_text: Any text in the answer indicating last day/end of year for students for 2025–26.
    - instructional_days_text: Any text indicating the total instructional days for students (e.g., "180 student days", "1,025 hours equivalent to 180 days").
    - teacher_workdays_count_text: Any text indicating how many teacher workdays are scheduled (e.g., "12 teacher workdays").
    - pd_days_text: Any text indicating professional development/professional learning designated days.
    - winter_break_text: Any text indicating winter break dates spanning late December 2025 to early January 2026.
    - calendar_type_text: Any text indicating the presence of a "Traditional" calendar (even if other tracks also exist).
    - compliance_note: Any text or note in the answer explicitly stating compliance with state education department calendar requirements.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - Prefer official district domains for calendar_urls and other official pages. If a calendar is hosted on a 3rd-party platform but clearly branded as the district’s official file, still include it.
    - Do not invent or infer any URLs.

    Return a JSON object with a 'districts' array of length up to 2, each element having the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().lower()
    if s in {"nc", "north carolina"}:
        return "NC"
    if s in {"sc", "south carolina"}:
        return "SC"
    return None


def full_state_name(abbrev: Optional[str]) -> Optional[str]:
    if abbrev == "NC":
        return "North Carolina"
    if abbrev == "SC":
        return "South Carolina"
    return None


def safe_name(d: District, fallback: str) -> str:
    return (d.name or fallback).strip()


def dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        uu = u.strip()
        if uu and uu not in seen:
            seen.add(uu)
            result.append(uu)
    return result


def assemble_sources(d: District) -> List[str]:
    return dedup_urls((d.calendar_urls or []) + (d.official_site_urls or []) + (d.athletics_urls or []))


def mark_leaf_failed(node) -> None:
    node.score = 0.0
    node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification for a single district                                          #
# --------------------------------------------------------------------------- #
async def verify_single_district(
    evaluator: Evaluator,
    parent_node,
    d: District,
    idx: int
) -> None:
    """
    Build and run the verification sub-tree for one district (idx = 1 or 2).
    All children under this node are critical per rubric.
    """
    dn = idx  # 1-based index
    node_id_prefix = f"d{dn}"

    district_node = evaluator.add_parallel(
        id=f"district_{dn}_evaluation",
        desc=f"District {dn}: provided with required identifying info and meets all stated criteria.",
        parent=parent_node,
        critical=True,
    )

    # Normalized/state strings
    state_abbrev = normalize_state(d.state)
    state_full = full_state_name(state_abbrev)
    district_display = safe_name(d, f"District {dn}")

    # 1) Reporting: name & state provided (existence check)
    name_state_exists = evaluator.add_custom_node(
        result=(bool(d.name) and bool(state_abbrev)),
        id=f"{node_id_prefix}_reporting_name_state",
        desc=f"Provides District {dn} official name and state (NC or SC).",
        parent=district_node,
        critical=True,
    )

    # 2) Calendar URL exists and is an official public calendar page for 2025–26 (verify with URL)
    cal_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_calendar_url_public_official",
        desc=f"Provides at least one URL on an official District {dn} website where the 2025–26 school calendar is publicly available.",
        parent=district_node,
        critical=True,
    )
    if d.calendar_urls:
        await evaluator.verify(
            claim=f"This webpage is an official page of {district_display} and publicly provides the district's 2025–26 traditional academic calendar (webpage or PDF).",
            node=cal_leaf,
            sources=d.calendar_urls,
            additional_instruction=(
                "Confirm the page is on the district's official site (or an official PDF/file) and that it clearly refers to the 2025–26 school year. "
                "If multiple calendars/tracks exist, prioritize the 'Traditional' calendar. "
                "Reject 3rd-party news or unofficial sites unless the file is clearly an official district document."
            ),
        )
    else:
        mark_leaf_failed(cal_leaf)

    # Prepare commonly used source sets
    calendar_sources = dedup_urls(d.calendar_urls)
    general_sources = assemble_sources(d)

    # 3) Criterion 1: Public K-12 system in NC/SC
    c1 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_1_geographic_location",
        desc=f"District {dn} is a public K-12 school system located in either North Carolina or South Carolina.",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{district_display} is a public K-12 school district located in {state_full or 'North Carolina or South Carolina'}.",
        node=c1,
        sources=general_sources or calendar_sources or None,
        additional_instruction=(
            "Verify that the entity is a public school district/LEA (not a single school or private district) and located in NC or SC as specified. "
            "Allow synonyms like 'Public Schools', 'School District', 'County Schools', or 'School System'."
        ),
    )

    # 4) Criterion 2: Offers traditional calendar schools
    c2 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_2_traditional_calendar",
        desc=f"District {dn} offers traditional calendar schools (not exclusively year-round).",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{district_display} offers traditional-calendar schools for 2025–26 (even if other tracks also exist).",
        node=c2,
        sources=calendar_sources or general_sources or None,
        additional_instruction=(
            "Look for 'Traditional' labels or language indicating a standard/traditional calendar. "
            "If multiple calendars exist (e.g., year-round, early colleges), confirm that at least one is the traditional calendar used by most schools."
        ),
    )

    # 5) Criterion 3: First day in August 2025
    c3 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_3_august_start",
        desc=f"District {dn} first day of school for students for the 2025-26 academic year is in August 2025.",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The first student day for {district_display} in 2025–26 occurs in August 2025.",
        node=c3,
        sources=calendar_sources or general_sources or None,
        additional_instruction=(
            "Check the 2025–26 traditional calendar; accept any student first day within August 2025. "
            "If multiple tracks exist, consider the traditional track."
        ),
    )

    # 6) Criterion 4: At least 180 instructional days
    c4 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_4_minimum_instructional_days",
        desc=f"District {dn} provides at least 180 instructional days for students in its 2025-26 traditional calendar.",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 2025–26 traditional calendar for {district_display} provides at least 180 instructional/student days (or an officially stated hours-equivalency meeting/ exceeding the 180-day standard).",
        node=c4,
        sources=calendar_sources or general_sources or None,
        additional_instruction=(
            "Many NC/SC calendars state '180 student days' explicitly; some state hours (e.g., 1,025 hours) meeting the same requirement. "
            "If hours are provided that satisfy the state's equivalent requirement, consider this supported."
        ),
    )

    # 7) Criterion 5: Teacher workdays within state-mandated limits
    c5 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_5_teacher_workday_limits",
        desc=f"District {dn} schedules teacher workdays within applicable state-mandated limits (NC: ≤195 days for 10-month teachers; SC: 190-day annual term, as applicable).",
        parent=district_node,
        critical=True,
    )
    if state_abbrev == "NC":
        limit_text = "no more than 195 days for 10-month teachers (NC G.S. 115C-84.2 and related guidance)"
    elif state_abbrev == "SC":
        limit_text = "a 190-day annual term (per SC state policy)"
    else:
        limit_text = "the applicable state-mandated limits for teacher calendars"
    await evaluator.verify(
        claim=f"The 2025–26 calendar for {district_display} schedules teacher workdays within {limit_text}.",
        node=c5,
        sources=calendar_sources or general_sources or None,
        additional_instruction=(
            "Use the calendar to infer compliance based on the number of listed teacher workdays and overall teacher contract days. "
            "If the calendar (or official page) explicitly states compliance with state limits, accept it. "
            "If totals shown are clearly within the specified limits, accept as compliant."
        ),
    )

    # 8) Criterion 6: At least 10 designated teacher workdays
    c6 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_6_teacher_workday_count",
        desc=f"District {dn} has at least 10 designated teacher workdays in the 2025-26 traditional calendar.",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 2025–26 traditional calendar for {district_display} designates at least 10 teacher workdays (teacher-only days).",
        node=c6,
        sources=calendar_sources or general_sources or None,
        additional_instruction=(
            "Count explicit teacher workdays such as 'Teacher Workday', 'Workday (No Students)', 'Planning Day', etc. "
            "If the calendar totals or legend indicates ≥10 teacher workdays, accept."
        ),
    )

    # 9) Criterion 7: Winter break spanning late Dec 2025 into early Jan 2026
    c7 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_7_winter_break",
        desc=f"District {dn} includes a winter break spanning late December 2025 into early January 2026.",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 2025–26 traditional calendar for {district_display} shows a winter break spanning late December 2025 into early January 2026.",
        node=c7,
        sources=calendar_sources or general_sources or None,
        additional_instruction=(
            "Look for a consecutive no-school period covering the last week(s) of December 2025 and at least one day in early January 2026."
        ),
    )

    # 10) Criterion 8: Professional development days designated
    c8 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_8_professional_development",
        desc=f"District {dn} designates specific teacher workdays for professional development/professional learning purposes.",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 2025–26 calendar for {district_display} designates specific teacher days for professional development or professional learning.",
        node=c8,
        sources=calendar_sources or general_sources or None,
        additional_instruction=(
            "Look for 'PD', 'Professional Development', 'PL', 'Staff Development', or similar labels on teacher-only days."
        ),
    )

    # 11) Criterion 9: High school athletics in state-sanctioned association
    c9 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_9_high_school_athletics",
        desc=f"District {dn} operates high schools with athletic programs competing in state-sanctioned athletics associations.",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"{district_display} operates high schools whose athletic programs compete in the state's sanctioned association "
            f"({'NCHSAA' if state_abbrev=='NC' else 'SCHSL' if state_abbrev=='SC' else 'NCHSAA or SCHSL'})."
        ),
        node=c9,
        sources=dedup_urls(d.athletics_urls) or general_sources or None,
        additional_instruction=(
            "Accept evidence from official district or high school athletics pages indicating NCHSAA (NC) or SCHSL (SC) membership, "
            "or other authoritative official references confirming participation."
        ),
    )

    # 12) Criterion 11: Academic year ends in June 2026
    c11 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_11_academic_year_end",
        desc=f"District {dn} academic year for students concludes in June 2026.",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 2025–26 traditional calendar for {district_display} ends the student academic year in June 2026.",
        node=c11,
        sources=calendar_sources or general_sources or None,
        additional_instruction=(
            "Look for the last student day, last day of classes/exams, or graduation dates in June 2026 on the traditional calendar."
        ),
    )

    # 13) Criterion 12: Complies with state education department requirements
    c12 = evaluator.add_leaf(
        id=f"{node_id_prefix}_criterion_12_state_compliance",
        desc=f"District {dn} calendar complies with applicable state education department requirements for school calendar structure.",
        parent=district_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 2025–26 traditional calendar for {district_display} complies with {state_full or 'state'} education department calendar requirements.",
        node=c12,
        sources=calendar_sources or general_sources or None,
        additional_instruction=(
            "Look for explicit statements of compliance or evidence that required elements are satisfied (e.g., student days/hours, workdays, PD, holidays), "
            "per NC or SC state rules. If the calendar is a standard district-approved calendar meeting typical state constraints, consider this supported."
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
    Evaluate an answer for the NC/SC 2025–26 traditional calendar districts task.

    Returns a standardized summary dictionary from Evaluator.get_summary().
    """
    # Initialize evaluator with a non-critical framework root, then add a critical task root under it.
    evaluator = Evaluator()
    framework_root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level rubric root is parallel
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

    # Create a critical task root to mirror rubric's critical root
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify exactly two distinct public school districts in NC or SC that each meet all specified 2025–26 traditional-calendar criteria, and provide required identifying info and official calendar reference URL(s) for each.",
        parent=framework_root,
        critical=True,
    )

    # 1) Extract structured district info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction",
    )

    # Select only the first two districts; pad if fewer
    selected: List[District] = list(extraction.districts[:2])
    while len(selected) < 2:
        selected.append(District())

    d1, d2 = selected[0], selected[1]

    # 2) Root-level: verify two distinct districts are provided (by name)
    def _distinct_two() -> bool:
        n1 = (d1.name or "").strip().lower()
        n2 = (d2.name or "").strip().lower()
        return bool(n1) and bool(n2) and (n1 != n2)

    evaluator.add_custom_node(
        result=_distinct_two(),
        id="two_distinct_districts_provided",
        desc="Answer identifies at least two distinct public school districts (the first two extracted are distinct).",
        parent=task_root,
        critical=True,
    )

    # 3) District subtrees (all children critical under each district evaluation node)
    await verify_single_district(evaluator, task_root, d1, idx=1)
    await verify_single_district(evaluator, task_root, d2, idx=2)

    # Return the full evaluation summary
    return evaluator.get_summary()