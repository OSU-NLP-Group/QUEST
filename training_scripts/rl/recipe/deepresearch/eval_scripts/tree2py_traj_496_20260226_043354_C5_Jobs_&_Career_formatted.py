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
TASK_ID = "coach_ahc_transition_2025_2026"
TASK_DESCRIPTION = (
    "A college football coach recently transitioned to a new assistant head coach position in 2025 or 2026. "
    "This coach has the following career background: earned a bachelor's degree from a university and played college "
    "football at the NCAA level; started their coaching career as an administrative assistant at a college football program, "
    "serving in this role for at least 3 years; advanced to become a position coach (coaching a specific position group), "
    "serving in position coaching role(s) for at least 3 years total; was promoted to a coordinator position (offensive, "
    "defensive, or special teams coordinator) and held this role for at least 5 years; recently accepted an assistant head coach "
    "position at a different university than where they served as coordinator; the new assistant head coach position includes "
    "coordinator responsibilities in addition to the assistant head coach title; and the coach's total coaching career, from their "
    "first coaching position to the assistant head coach promotion, spans at least 15 years. Identify this coach by providing their "
    "full name, the university where they currently serve as assistant head coach, and the specific coordinator role included in "
    "their current position."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class RoleEntry(BaseModel):
    title: Optional[str] = None
    organization: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    duration_years: Optional[str] = None


class CoachExtraction(BaseModel):
    coach_full_name: Optional[str] = None

    # Current position info
    current_university: Optional[str] = None
    current_coordinator_role: Optional[str] = None  # e.g., "offensive coordinator", "defensive coordinator", "special teams coordinator"
    current_position_title: Optional[str] = None  # e.g., "Assistant Head Coach and Offensive Coordinator"
    promotion_year: Optional[str] = None  # Expected 2025 or 2026

    # Education and playing background
    bachelors_institution: Optional[str] = None
    played_ncaa_football: Optional[str] = None  # "yes"/"no"/details

    # Coaching trajectory
    coaching_start_year: Optional[str] = None
    first_coaching_role: Optional[RoleEntry] = None  # Expected to be administrative assistant
    position_coach_roles: List[RoleEntry] = Field(default_factory=list)
    coordinator_roles: List[RoleEntry] = Field(default_factory=list)
    most_recent_coordinator_organization: Optional[str] = None

    # Optional totals (if answer provided)
    total_years_position_coach: Optional[str] = None
    total_years_coordinator: Optional[str] = None
    total_career_span_years: Optional[str] = None

    # All URLs explicitly present in the answer
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_profile() -> str:
    return """
    Extract the requested structured information about the identified college football coach from the provided answer text.
    The answer is expected to name a specific coach and summarize their career path and current position in 2025 or 2026.

    REQUIRED FIELDS:
    - coach_full_name: The full name of the coach.
    - current_university: The university where the coach currently serves as Assistant Head Coach (AHC).
    - current_coordinator_role: The specific coordinator role included in their current AHC position; must be one of:
      "offensive coordinator", "defensive coordinator", or "special teams coordinator" (use lowercase; allow obvious synonyms like "OC", "DC", "STC" but normalize).
    - current_position_title: The title string if provided (e.g., "Assistant Head Coach and Offensive Coordinator").
    - promotion_year: The year (four digits) when the coach accepted/was appointed to the new AHC role (should be 2025 or 2026 if provided).

    EDUCATION & PLAYING:
    - bachelors_institution: The university from which the coach earned a bachelor's degree (return the name if provided).
    - played_ncaa_football: "yes" if the answer states they played NCAA college football; otherwise "no" or null.

    COACHING TRAJECTORY:
    - coaching_start_year: The year the coaching career began (four digits), if provided in the answer.
    - first_coaching_role: The FIRST coaching role, as an object: title, organization, start_year, end_year, duration_years (if any).
      Note: The prompt expects the first role to be an "administrative assistant" or similarly named administrative staff role.
    - position_coach_roles: An array of roles where the coach served as a position coach for a specific group (e.g., WR Coach, RB Coach).
      Each element is an object: title, organization, start_year, end_year, duration_years (if available).
    - coordinator_roles: An array of roles where the coach served as a coordinator (offensive, defensive, or special teams).
      Each element is an object: title, organization, start_year, end_year, duration_years (if available).
    - most_recent_coordinator_organization: The organization/university of the most recent coordinator role (string) if stated.

    OPTIONAL TOTALS:
    - total_years_position_coach: Total number of years served as a position coach (string as written in the answer if present).
    - total_years_coordinator: Total number of years served as a coordinator (string as written in the answer if present).
    - total_career_span_years: Total years from first coaching position to the assistant head coach promotion (string if present).

    SOURCES:
    - source_urls: Extract all URLs explicitly present in the answer. Include press releases, bios, news articles, etc.

    EXTRACTION RULES:
    1) Extract exactly what is stated in the answer. Do not invent information. If any field is not mentioned, return null (or empty array for lists).
    2) Normalize the "current_coordinator_role" to one of: "offensive coordinator", "defensive coordinator", "special teams coordinator" when possible.
    3) For years, prefer four-digit numeric strings if provided (e.g., "2012"). For durations, keep them as text if the answer provides text (e.g., "3 years").
    4) For role arrays, include as many distinct role entries as the answer lists (don't deduplicate beyond obvious duplicates).
    5) For URLs, include only valid full URLs that appear in the answer text (or markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_int_year(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    # find a 4-digit year in the string
    m = re.search(r"(20\d{2}|19\d{2})", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    try:
        val = int(s.strip())
        if 1900 <= val <= 2100:
            return val
        return None
    except Exception:
        return None


def parse_int_from_text(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def role_duration_years(role: RoleEntry) -> Optional[int]:
    # Prefer explicit duration_years if parseable
    yrs = parse_int_from_text(role.duration_years) if role.duration_years else None
    if yrs is not None:
        return yrs
    # Try compute from start/end years
    sy = parse_int_year(role.start_year)
    ey = parse_int_year(role.end_year)
    if sy is not None and ey is not None and ey >= sy:
        return ey - sy + 1  # inclusive if roles listed as ranges
    return None


def sum_roles_years(roles: List[RoleEntry]) -> int:
    total = 0
    for r in roles:
        yrs = role_duration_years(r)
        if yrs:
            total += yrs
    return total


def first_nonempty_str(*args: Optional[str]) -> Optional[str]:
    for a in args:
        if a and a.strip():
            return a.strip()
    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: CoachExtraction):
    root = evaluator.root  # Root is initialized as SEQUENTIAL (non-critical root)

    # 1) Identify coach (leaf - existence check)
    identify_node = evaluator.add_custom_node(
        result=bool(extraction.coach_full_name and extraction.coach_full_name.strip()),
        id="identify_coach",
        desc="Answer provides a specific coach's full name",
        parent=root,
        critical=True
    )

    # 2) Required outputs present (parallel aggregator)
    required_outputs = evaluator.add_parallel(
        id="required_outputs_present",
        desc="Answer provides all requested outputs: current assistant head coach university and the specific coordinator role included in the current position",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extraction.current_university and extraction.current_university.strip()),
        id="current_university_provided",
        desc="Answer states the university where the coach currently serves as assistant head coach",
        parent=required_outputs,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extraction.current_coordinator_role and extraction.current_coordinator_role.strip()),
        id="current_coordinator_role_provided",
        desc="Answer states the specific coordinator role included in the coach’s current position (offensive/defensive/special teams coordinator)",
        parent=required_outputs,
        critical=True
    )

    # 3) Constraint verification (parallel aggregator)
    constraints = evaluator.add_parallel(
        id="constraint_verification",
        desc="Verify the identified coach satisfies all constraints in the prompt",
        parent=root,
        critical=True
    )

    # Prepare handy variables
    sources = extraction.source_urls or []
    name = extraction.coach_full_name or "the coach"
    current_uni = extraction.current_university or "the current university"
    coord_role = extraction.current_coordinator_role or "coordinator"
    bachelors_inst = extraction.bachelors_institution
    most_recent_coord_org = extraction.most_recent_coordinator_organization

    # 3.1 Bachelor's degree
    bachelors_node = evaluator.add_leaf(
        id="bachelors_degree",
        desc="Coach earned a bachelor's degree from a university",
        parent=constraints,
        critical=True
    )
    if bachelors_inst and bachelors_inst.strip():
        bachelors_claim = f"{name} earned a bachelor's degree from {bachelors_inst}."
    else:
        bachelors_claim = f"{name} earned a bachelor's degree."
    await evaluator.verify(
        claim=bachelors_claim,
        node=bachelors_node,
        sources=sources,
        additional_instruction="Accept standard equivalents such as BA/BS or similar language clearly indicating a bachelor's degree."
    )

    # 3.2 Played NCAA football
    played_node = evaluator.add_leaf(
        id="played_ncaa_football",
        desc="Coach played college football at the NCAA level",
        parent=constraints,
        critical=True
    )
    played_claim = f"{name} played college football at the NCAA level."
    await evaluator.verify(
        claim=played_claim,
        node=played_node,
        sources=sources,
        additional_instruction="Look for roster/bio lines or articles stating that he played NCAA college football; allow reasonable synonyms."
    )

    # 3.3 First role administrative assistant 3+ years
    admin_node = evaluator.add_leaf(
        id="admin_assistant_first_role_3plus_years",
        desc="Coach started their coaching career as an administrative assistant at a college football program and served in that role for at least 3 years",
        parent=constraints,
        critical=True
    )
    first_title = extraction.first_coaching_role.title if extraction.first_coaching_role else None
    first_org = extraction.first_coaching_role.organization if extraction.first_coaching_role else None
    admin_claim = (
        f"{name} began his coaching career as an administrative assistant"
        f"{f' at {first_org}' if first_org else ''} and served in that role for at least 3 years."
    )
    await evaluator.verify(
        claim=admin_claim,
        node=admin_node,
        sources=sources,
        additional_instruction="Consider reasonably equivalent admin staff titles (e.g., 'administrative assistant', 'football administrative assistant', 'operations/administrative assistant'). "
                              "The page(s) should clearly indicate this was the FIRST coaching role and that it lasted 3 or more years in total."
    )

    # 3.4 Position coach 3+ years total
    pos_node = evaluator.add_leaf(
        id="position_coach_specific_group_3plus_years",
        desc="Coach served as a position coach for a specific position group for at least 3 years total",
        parent=constraints,
        critical=True
    )
    pos_claim = f"{name} served as a college football position coach (e.g., RB/WR/OL/DL/etc.) for at least 3 total years."
    await evaluator.verify(
        claim=pos_claim,
        node=pos_node,
        sources=sources,
        additional_instruction="Look across the career summary for multiple stints adding up to 3+ years as a position coach."
    )

    # 3.5 Coordinator position held
    coord_held_node = evaluator.add_leaf(
        id="coordinator_position_held",
        desc="Coach was promoted to/served as a coordinator (offensive coordinator, defensive coordinator, or special teams coordinator)",
        parent=constraints,
        critical=True
    )
    coord_held_claim = f"{name} served as a coordinator (offensive, defensive, or special teams) at the college level."
    await evaluator.verify(
        claim=coord_held_claim,
        node=coord_held_node,
        sources=sources,
        additional_instruction="Coordinator variants such as co-coordinator (e.g., co-DC, co-OC) also count."
    )

    # 3.6 Coordinator 5+ years total
    coord_5yrs_node = evaluator.add_leaf(
        id="coordinator_5plus_years",
        desc="Coach held the coordinator position for at least 5 years",
        parent=constraints,
        critical=True
    )
    coord_5yrs_claim = f"{name} accumulated at least 5 total years as a coordinator (offensive/defensive/special teams)."
    await evaluator.verify(
        claim=coord_5yrs_claim,
        node=coord_5yrs_node,
        sources=sources,
        additional_instruction="The evidence can come from a single bio page or multiple sources; if a page summarizes total years, that suffices."
    )

    # 3.7 Assistant head coach promotion in 2025 or 2026
    ahc_year_node = evaluator.add_leaf(
        id="assistant_head_coach_promotion_2025_or_2026",
        desc="Promotion/transition to assistant head coach occurred in 2025 or 2026",
        parent=constraints,
        critical=True
    )
    ahc_year_claim = f"In 2025 or 2026, {name} accepted or was appointed to an Assistant Head Coach role."
    if extraction.promotion_year and extraction.current_university:
        # More specific if available
        ahc_year_claim = f"In {extraction.promotion_year}, {name} accepted/was appointed Assistant Head Coach at {extraction.current_university}."
    await evaluator.verify(
        claim=ahc_year_claim,
        node=ahc_year_node,
        sources=sources,
        additional_instruction="The page should show a press release or bio update indicating the AHC appointment in 2025 or 2026."
    )

    # 3.8 AHC at different university than where served as coordinator
    ahc_diff_uni_node = evaluator.add_leaf(
        id="assistant_head_coach_at_different_university_than_coordinator",
        desc="Assistant head coach position is at a different university than where the coach served as coordinator",
        parent=constraints,
        critical=True
    )
    if most_recent_coord_org and extraction.current_university:
        diff_uni_claim = (
            f"{name}'s Assistant Head Coach position at {extraction.current_university} "
            f"is at a different university than where he served as a coordinator (e.g., {most_recent_coord_org})."
        )
    else:
        diff_uni_claim = (
            f"{name}'s current Assistant Head Coach university is different from his prior coordinator university."
        )
    await evaluator.verify(
        claim=diff_uni_claim,
        node=ahc_diff_uni_node,
        sources=sources,
        additional_instruction="If multiple coordinator stints exist, use the most recent coordinator stop before the AHC move."
    )

    # 3.9 AHC role includes coordinator duties
    ahc_includes_coord_node = evaluator.add_leaf(
        id="assistant_head_coach_role_includes_coordinator_duties",
        desc="Current assistant head coach position includes coordinator responsibilities in addition to the assistant head coach title",
        parent=constraints,
        critical=True
    )
    ahc_includes_coord_claim = (
        f"In the current role at {current_uni}, {name} holds the title Assistant Head Coach and also serves as {coord_role}."
    )
    await evaluator.verify(
        claim=ahc_includes_coord_claim,
        node=ahc_includes_coord_node,
        sources=sources,
        additional_instruction="The job title or press release should explicitly include both 'Assistant Head Coach' and a coordinator duty."
    )

    # 3.10 Total coaching career span 15+ years (custom calculation if possible)
    # Compute years from coaching_start_year to promotion_year
    start_year = parse_int_year(first_nonempty_str(extraction.coaching_start_year, extraction.first_coaching_role.start_year if extraction.first_coaching_role else None))
    promo_year = parse_int_year(extraction.promotion_year)
    total_span_ok = False
    if start_year is not None and promo_year is not None:
        total_span_ok = (promo_year - start_year) >= 15

    evaluator.add_custom_node(
        result=total_span_ok,
        id="total_career_span_15plus_years",
        desc="Total coaching career (from first coaching position to assistant head coach promotion) spans at least 15 years",
        parent=constraints,
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
    Evaluate an answer for the 'assistant head coach transition 2025/2026' task.
    """

    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # As per rubric: sequential: identify -> outputs -> constraints
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_coach_profile(),
        template_class=CoachExtraction,
        extraction_name="coach_profile"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return evaluation summary
    return evaluator.get_summary()