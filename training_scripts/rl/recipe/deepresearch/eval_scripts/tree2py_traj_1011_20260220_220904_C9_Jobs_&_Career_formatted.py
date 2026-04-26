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
TASK_ID = "ivy_head_coach_2020_2025"
TASK_DESCRIPTION = """Identify an Ivy League head football coach who was hired between January 1, 2020 and December 31, 2025, and provide comprehensive documentation showing they meet the following career progression and qualification requirements:

1. Coach Identification: Full name, current institution, official position title, hire date confirming appointment between 2020-2025, and confirmation of Ivy League Conference membership.

2. Educational Credentials: Bachelor's degree from an accredited institution (required), institution where degree was earned, and field of study (if available).

3. Coaching Experience Timeline: Minimum of 15 years of coaching experience before head coaching appointment, year coaching career began, and year of head coach appointment.

4. Multi-Level Coaching Experience: Evidence of coaching at at least two different competitive levels (Division III, FCS, FBS, or NFL) with specific institutions and levels where coaching occurred.

5. Position-Specific Coaching: At least two distinct position coaching roles (e.g., offensive line, tight ends, running backs, quarterbacks) and any special teams coordination experience.

6. Coordinator-Level Experience: At least one coordinator role (offensive coordinator, defensive coordinator, or recruiting coordinator) held before becoming head coach.

7. Senior Assistant Experience (preferred): Associate head coach or assistant head coach experience.

8. Ivy League Background (preferred): Playing experience at an Ivy League institution, OR previous coaching experience at an Ivy League institution.

9. Championship Achievements (preferred): Participation in conference championship teams with years and institutions where championships occurred.

10. First Season Performance (if applicable): Overall record from first season as head coach, conference record from first season, and notable achievements (conference titles, playoff berths, national rankings).

11. Player Development Record (preferred): Development of All-Conference or All-America players.

12. Current Program Status: Most recent or current season performance and evidence of program improvement or sustained success.

For each requirement, provide specific details and URL references that verify the information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Identification(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None
    position_title: Optional[str] = None
    hire_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    ivy_membership_urls: List[str] = Field(default_factory=list)


class Education(BaseModel):
    bachelors_degree_held: Optional[str] = None
    undergraduate_institution: Optional[str] = None
    degree_field: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ExperienceTimeline(BaseModel):
    career_start_year: Optional[str] = None
    head_coach_appointment_year: Optional[str] = None
    minimum_years_claim: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MultiLevelExperience(BaseModel):
    levels: List[str] = Field(default_factory=list)
    institutions_by_level: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class PositionExperience(BaseModel):
    primary_role: Optional[str] = None
    secondary_role: Optional[str] = None
    additional_roles: List[str] = Field(default_factory=list)
    special_teams_coordination: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CoordinatorExperience(BaseModel):
    roles: List[str] = Field(default_factory=list)
    requirement_met_claim: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SeniorAssistantExperience(BaseModel):
    role_title: Optional[str] = None
    institution: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class IvyBackground(BaseModel):
    ivy_player: Optional[str] = None
    ivy_previous_coach: Optional[str] = None
    connection_present: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ChampionshipAchievements(BaseModel):
    conference_title: Optional[str] = None
    championship_years_institutions: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FirstSeasonPerformance(BaseModel):
    overall_record: Optional[str] = None
    conference_record: Optional[str] = None
    notable_achievements: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class PlayerDevelopment(BaseModel):
    all_conference_players: List[str] = Field(default_factory=list)
    all_america_players: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class CurrentProgramTrajectory(BaseModel):
    recent_season_record: Optional[str] = None
    national_fcs_ranking: Optional[str] = None
    improvement_evidence: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CoachProfileExtraction(BaseModel):
    identification: Optional[Identification] = None
    education: Optional[Education] = None
    experience_timeline: Optional[ExperienceTimeline] = None
    multi_level_experience: Optional[MultiLevelExperience] = None
    position_experience: Optional[PositionExperience] = None
    coordinator_experience: Optional[CoordinatorExperience] = None
    senior_assistant_experience: Optional[SeniorAssistantExperience] = None
    ivy_background: Optional[IvyBackground] = None
    championship_achievements: Optional[ChampionshipAchievements] = None
    first_season_performance: Optional[FirstSeasonPerformance] = None
    player_development: Optional[PlayerDevelopment] = None
    current_program_trajectory: Optional[CurrentProgramTrajectory] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_profile() -> str:
    return """
Extract a single Ivy League head football coach's profile as presented in the answer. Return a JSON object matching the provided schema. Follow these instructions carefully:

1) identification:
   - name: full name of the coach.
   - institution: the current Ivy League institution (e.g., Harvard, Yale, Princeton, Penn, Brown, Columbia, Cornell, Dartmouth).
   - position_title: official role title (e.g., "Head Football Coach").
   - hire_date: the (announced) date the coach was hired/appointed.
   - sources: ALL URL(s) cited in the answer that confirm the hiring/appointment and role at the institution.
   - ivy_membership_urls: URL(s) cited in the answer that confirm Ivy League membership of the institution (if any are provided).

2) education:
   - bachelors_degree_held: a short phrase like "yes", "holds a bachelor's degree", or similar text from the answer (if provided).
   - undergraduate_institution: institution where the bachelor's degree was earned.
   - degree_field: field/major (if mentioned).
   - urls: ALL URL(s) cited that document the educational background.

3) experience_timeline:
   - career_start_year: year coaching career began (if mentioned).
   - head_coach_appointment_year: year appointed as head coach at the current institution.
   - minimum_years_claim: any explicit claim indicating at least 15 years before head coach appointment (if stated).
   - urls: ALL URL(s) cited that document the coaching career start and timeline.

4) multi_level_experience:
   - levels: list of distinct competitive levels explicitly mentioned (choose from: "Division III", "FCS", "FBS", "NFL").
   - institutions_by_level: list of strings pairing institution and level (e.g., "Delaware (FCS)", "UCLA (FBS)"), as presented in the answer.
   - urls: ALL URL(s) cited that support multi-level experience.

5) position_experience:
   - primary_role: first distinct position group coached (e.g., "offensive line", "tight ends", "running backs", "quarterbacks").
   - secondary_role: second distinct position group coached.
   - additional_roles: any additional position groups coached (array; may be empty).
   - special_teams_coordination: text indicating special teams coordination experience if present (e.g., "special teams coordinator").
   - urls: ALL URL(s) cited that support position coaching roles.

6) coordinator_experience:
   - roles: list of coordinator roles held before becoming head coach (e.g., "offensive coordinator", "defensive coordinator", "recruiting coordinator").
   - requirement_met_claim: any explicit statement that coach held at least one coordinator role before head coach (if present).
   - urls: ALL URL(s) cited that support coordinator experience.

7) senior_assistant_experience:
   - role_title: "associate head coach" or "assistant head coach" (if present).
   - institution: institution where this senior role was held.
   - urls: ALL URL(s) cited for the senior assistant role.

8) ivy_background:
   - ivy_player: text indicating playing experience at an Ivy League school (if present).
   - ivy_previous_coach: text indicating prior coaching at an Ivy League school (if present).
   - connection_present: a short phrase like "yes" if any Ivy connection is indicated.
   - urls: ALL URL(s) cited for Ivy background.

9) championship_achievements:
   - conference_title: text indicating being part of a conference-championship staff (if present).
   - championship_years_institutions: years and institutions for those titles (if present).
   - urls: ALL URL(s) cited that document championship achievements.

10) first_season_performance:
    - overall_record: win-loss record for first season as head coach (if applicable).
    - conference_record: Ivy League conference record from first season (if applicable).
    - notable_achievements: list of notable outcomes (e.g., title, playoff berth, rankings).
    - urls: ALL URL(s) cited documenting first season performance.

11) player_development:
    - all_conference_players: list of players developed who earned All-Conference honors (if present).
    - all_america_players: list of players developed who earned All-America honors (if present).
    - urls: ALL URL(s) cited documenting player development.

12) current_program_trajectory:
    - recent_season_record: most recent or current season record (if present).
    - national_fcs_ranking: current or recent FCS national ranking (if present).
    - improvement_evidence: text describing improvement or sustained success (if present).
    - urls: ALL URL(s) cited documenting current status.

GENERAL RULES:
- Extract only what is explicitly present in the answer.
- If a field is not mentioned, set it to null (or empty list for arrays/URLs).
- For all URLs, include complete URLs exactly as shown in the answer. If none are provided for a section, return an empty list for that section's urls.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _year_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None


async def _verify_leaf_with_urls(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    claim: str,
    urls: List[str],
    critical: bool,
    additional_instruction: str = "None",
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls if urls else None,
        additional_instruction=additional_instruction,
    )
    return leaf


def _add_url_presence_node(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    urls: List[str],
    critical: bool,
):
    result = bool(urls and len(urls) > 0)
    return evaluator.add_custom_node(
        result=result,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )


# --------------------------------------------------------------------------- #
# Verification builders (subtrees)                                            #
# --------------------------------------------------------------------------- #
async def build_identification_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    ident = data.identification or Identification()
    id_urls = _safe_list(ident.sources)
    ivy_urls = _safe_list(ident.ivy_membership_urls) or id_urls

    coach_ident_node = evaluator.add_parallel(
        id="Coach_Identification",
        desc="Identification and verification of coach and institution",
        parent=parent,
        critical=True  # All children in this subtree are critical
    )

    # Basic Information (critical group)
    basic_info = evaluator.add_parallel(
        id="Basic_Information",
        desc="Coach's name and current institution",
        parent=coach_ident_node,
        critical=True
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Coach_Full_Name",
        desc="Full name of the head coach",
        parent=basic_info,
        claim=f"This page confirms that the head coach appointed is '{ident.name}'.",
        urls=id_urls,
        critical=True,
        additional_instruction="Verify the coach's full name as presented on the hiring/appointment page. Allow minor formatting or middle initial differences.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Current_Institution",
        desc="Name of current Ivy League institution",
        parent=basic_info,
        claim=f"This page indicates the coach's current institution is '{ident.institution}'.",
        urls=id_urls,
        critical=True,
        additional_instruction="Confirm the institution associated with the head coaching appointment.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Position_Title",
        desc="Official title of head coaching position",
        parent=basic_info,
        claim=f"The page states the official position title is '{ident.position_title}'.",
        urls=id_urls,
        critical=True,
        additional_instruction="Confirm that the role is for Head Football Coach (or an equivalent official head coach title).",
    )

    # Appointment Details (critical group)
    appt = evaluator.add_parallel(
        id="Appointment_Details",
        desc="Details of head coaching appointment",
        parent=coach_ident_node,
        critical=True
    )

    # Hire Date within range 2020-2025 (inclusive)
    await _verify_leaf_with_urls(
        evaluator,
        node_id="Hire_Date",
        desc="Coach was hired between January 1, 2020 and December 31, 2025",
        parent=appt,
        claim=f"The hire/appointment date is '{ident.hire_date}', and this date falls between January 1, 2020 and December 31, 2025.",
        urls=id_urls,
        critical=True,
        additional_instruction="Verify the hire/appointment date on the page and confirm the date lies within the 2020–2025 inclusive range.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Ivy_League_Member",
        desc="Institution is a current member of the Ivy League Conference",
        parent=appt,
        claim=f"'{ident.institution}' is a current member of the Ivy League.",
        urls=ivy_urls,
        critical=True,
        additional_instruction="Confirm Ivy League membership of the institution using the provided sources. If the page is the school's or conference site, look for explicit Ivy League references.",
    )

    _add_url_presence_node(
        evaluator,
        node_id="Identification_URL",
        desc="URL reference confirming coach's appointment",
        parent=appt,
        urls=id_urls,
        critical=True,
    )


async def build_education_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    edu = data.education or Education()
    edu_urls = _safe_list(edu.urls)

    edu_root = evaluator.add_parallel(
        id="Educational_Credentials",
        desc="Verification of coach's educational background",
        parent=parent,
        critical=False  # Mixed criticality at leaves; keep parent non-critical to satisfy framework constraints
    )

    deg_req = evaluator.add_parallel(
        id="Degree_Requirements",
        desc="Verification of required educational degrees",
        parent=edu_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Bachelors_Degree_Held",
        desc="Coach holds a bachelor's degree from an accredited institution",
        parent=deg_req,
        claim="This page states the coach holds a bachelor's degree.",
        urls=edu_urls,
        critical=True,
        additional_instruction="Look for statements confirming a bachelor's degree. Minor wording differences are okay.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Undergraduate_Institution_Name",
        desc="Institution where bachelor's degree was earned",
        parent=deg_req,
        claim=f"The bachelor's degree was earned at '{edu.undergraduate_institution}'.",
        urls=edu_urls,
        critical=True,
        additional_instruction="Confirm the undergraduate institution as stated.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Degree_Field_of_Study",
        desc="Academic field or major of bachelor's degree",
        parent=deg_req,
        claim=f"The bachelor's degree field/major is '{edu.degree_field}'.",
        urls=edu_urls,
        critical=False,
        additional_instruction="If the field/major is stated, verify it; otherwise this may not be supported.",
    )

    edu_doc = evaluator.add_parallel(
        id="Education_Documentation",
        desc="URL reference supporting educational credentials",
        parent=edu_root,
        critical=False
    )
    _add_url_presence_node(
        evaluator,
        node_id="Education_URL_Reference",
        desc="URL documenting educational background",
        parent=edu_doc,
        urls=edu_urls,
        critical=True,
    )


async def build_experience_timeline_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    tl = data.experience_timeline or ExperienceTimeline()
    tl_urls = _safe_list(tl.urls)

    exp_root = evaluator.add_parallel(
        id="Coaching_Experience_Timeline",
        desc="Verification of total coaching experience and career timeline",
        parent=parent,
        critical=True  # All leaves in this subtree are critical
    )

    dur = evaluator.add_parallel(
        id="Experience_Duration",
        desc="Length of coaching career prior to head coaching role",
        parent=exp_root,
        critical=True
    )

    # Compute appointment year preference
    appt_year = _year_from_text(tl.head_coach_appointment_year) or _year_from_text(tl.head_coach_appointment_year)
    # Fallback to try to extract from a year-like pattern in the field itself
    start_year = _year_from_text(tl.career_start_year)

    computed_years_text = ""
    if start_year and appt_year:
        try:
            years_diff = int(appt_year) - int(start_year)
            computed_years_text = f" (approximately {years_diff} years)"
        except Exception:
            computed_years_text = ""

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Minimum_Years_Met",
        desc="Coach had at least 15 years of coaching experience before becoming head coach",
        parent=dur,
        claim=f"The coach began in {tl.career_start_year} and became head coach in {tl.head_coach_appointment_year}{computed_years_text}, indicating at least 15 years of coaching experience before the head coach appointment.",
        urls=tl_urls,
        critical=True,
        additional_instruction="Use the timeline information to confirm ≥15 years passed between the career start and the head coach appointment.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Career_Start_Year",
        desc="Year when coach began coaching career",
        parent=dur,
        claim=f"The coaching career began in {tl.career_start_year}.",
        urls=tl_urls,
        critical=True,
        additional_instruction="Verify the initial coaching year from the provided timeline source(s).",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Head_Coach_Appointment_Year",
        desc="Year of appointment as head coach at current institution",
        parent=dur,
        claim=f"The head coach appointment year is {tl.head_coach_appointment_year}.",
        urls=tl_urls,
        critical=True,
        additional_instruction="Confirm the year the coach was appointed head coach at the current institution.",
    )

    exp_doc = evaluator.add_parallel(
        id="Experience_Documentation",
        desc="URL reference supporting experience timeline",
        parent=exp_root,
        critical=True
    )
    _add_url_presence_node(
        evaluator,
        node_id="Timeline_URL_Reference",
        desc="URL documenting coaching experience timeline",
        parent=exp_doc,
        urls=tl_urls,
        critical=True,
    )


async def build_multi_level_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    ml = data.multi_level_experience or MultiLevelExperience()
    ml_urls = _safe_list(ml.urls)
    levels_text = ", ".join(ml.levels) if ml.levels else ""

    ml_root = evaluator.add_parallel(
        id="Multi_Level_Coaching_Experience",
        desc="Verification of coaching experience across different competitive levels",
        parent=parent,
        critical=False  # Mixed critical leaves
    )

    comp_levels = evaluator.add_parallel(
        id="Competitive_Levels",
        desc="Identification of coaching levels where coach has experience",
        parent=ml_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Division_III_Coaching",
        desc="Coach has NCAA Division III coaching experience",
        parent=comp_levels,
        claim="This page shows the coach has NCAA Division III coaching experience.",
        urls=ml_urls,
        critical=False,
        additional_instruction="Verify if Division III experience is indicated on the sources.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="FCS_Coaching",
        desc="Coach has FCS (including Ivy League) coaching experience",
        parent=comp_levels,
        claim="This page shows the coach has FCS coaching experience (Ivy League is FCS).",
        urls=ml_urls,
        critical=False,
        additional_instruction="Verify if FCS experience is indicated on the sources.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="FBS_Coaching",
        desc="Coach has FBS coaching experience",
        parent=comp_levels,
        claim="This page shows the coach has FBS coaching experience.",
        urls=ml_urls,
        critical=False,
        additional_instruction="Verify if FBS experience is indicated on the sources.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="NFL_Coaching",
        desc="Coach has NFL coaching experience",
        parent=comp_levels,
        claim="This page shows the coach has NFL coaching experience.",
        urls=ml_urls,
        critical=False,
        additional_instruction="Verify if NFL coaching experience is indicated on the sources.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Multi_Level_Requirement",
        desc="Coach has experience at at least two different competitive levels",
        parent=comp_levels,
        claim=f"The coach has experience at at least two distinct competitive levels: {levels_text}.",
        urls=ml_urls,
        critical=True,
        additional_instruction="Confirm there are at least two different levels (e.g., FCS and FBS, or NCAA and NFL).",
    )

    lvl_doc = evaluator.add_parallel(
        id="Level_Documentation",
        desc="URL reference supporting multi-level experience",
        parent=ml_root,
        critical=True
    )
    _add_url_presence_node(
        evaluator,
        node_id="Levels_URL_Reference",
        desc="URL documenting coaching across levels",
        parent=lvl_doc,
        urls=ml_urls,
        critical=True,
    )


async def build_position_roles_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    pos = data.position_experience or PositionExperience()
    pos_urls = _safe_list(pos.urls)

    pos_root = evaluator.add_parallel(
        id="Position_Specific_Coaching",
        desc="Verification of coaching roles for specific position groups",
        parent=parent,
        critical=False
    )

    pos_roles = evaluator.add_parallel(
        id="Position_Roles_Held",
        desc="Specific position coaching roles during career",
        parent=pos_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Primary_Position_Role",
        desc="First distinct position group coached (e.g., offensive line, tight ends)",
        parent=pos_roles,
        claim=f"The coach has coached the '{pos.primary_role}' position group.",
        urls=pos_urls,
        critical=True,
        additional_instruction="Verify at least one specific position group coached by the coach.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Secondary_Position_Role",
        desc="Second distinct position group coached",
        parent=pos_roles,
        claim=f"The coach has also coached the '{pos.secondary_role}' position group.",
        urls=pos_urls,
        critical=True,
        additional_instruction="Verify a second distinct position coaching assignment.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Additional_Position_Roles",
        desc="Any additional position groups coached",
        parent=pos_roles,
        claim=f"Additional position roles coached include: {', '.join(pos.additional_roles)}.",
        urls=pos_urls,
        critical=False,
        additional_instruction="If listed, verify additional position roles; otherwise this may not be supported.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Special_Teams_Coordination",
        desc="Experience coordinating special teams units",
        parent=pos_roles,
        claim=f"The coach has special teams coordination experience: '{pos.special_teams_coordination}'.",
        urls=pos_urls,
        critical=False,
        additional_instruction="Verify whether the coach served as a special teams coordinator.",
    )

    pos_doc = evaluator.add_parallel(
        id="Position_Documentation",
        desc="URL reference supporting position coaching roles",
        parent=pos_root,
        critical=True
    )
    _add_url_presence_node(
        evaluator,
        node_id="Positions_URL_Reference",
        desc="URL documenting position coaching experience",
        parent=pos_doc,
        urls=pos_urls,
        critical=True,
    )


async def build_coordinator_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    coord = data.coordinator_experience or CoordinatorExperience()
    coord_urls = _safe_list(coord.urls)

    coord_root = evaluator.add_parallel(
        id="Coordinator_Level_Experience",
        desc="Verification of coordinator-level responsibilities",
        parent=parent,
        critical=False
    )

    coord_roles = evaluator.add_parallel(
        id="Coordinator_Roles",
        desc="Coordinator positions held during career",
        parent=coord_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Offensive_Coordinator_Role",
        desc="Served as offensive coordinator",
        parent=coord_roles,
        claim="This page states the coach served as an offensive coordinator.",
        urls=coord_urls,
        critical=False,
        additional_instruction="Check for offensive coordinator experience.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Defensive_Coordinator_Role",
        desc="Served as defensive coordinator",
        parent=coord_roles,
        claim="This page states the coach served as a defensive coordinator.",
        urls=coord_urls,
        critical=False,
        additional_instruction="Check for defensive coordinator experience.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Recruiting_Coordinator_Role",
        desc="Served as recruiting coordinator",
        parent=coord_roles,
        claim="This page states the coach served as a recruiting coordinator.",
        urls=coord_urls,
        critical=False,
        additional_instruction="Check for recruiting coordinator experience.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Coordinator_Requirement_Met",
        desc="Held at least one coordinator position before becoming head coach",
        parent=coord_roles,
        claim=f"The coach held at least one coordinator role before becoming head coach. Roles listed: {', '.join(coord.roles)}.",
        urls=coord_urls,
        critical=True,
        additional_instruction="Confirm at least one coordinator role (offensive, defensive, or recruiting) pre-dates the head coaching appointment.",
    )

    coord_doc = evaluator.add_parallel(
        id="Coordinator_Documentation",
        desc="URL reference supporting coordinator experience",
        parent=coord_root,
        critical=True
    )
    _add_url_presence_node(
        evaluator,
        node_id="Coordinator_URL_Reference",
        desc="URL documenting coordinator-level experience",
        parent=coord_doc,
        urls=coord_urls,
        critical=True,
    )


async def build_senior_assistant_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    sa = data.senior_assistant_experience or SeniorAssistantExperience()
    sa_urls = _safe_list(sa.urls)

    sa_root = evaluator.add_parallel(
        id="Senior_Assistant_Experience",
        desc="Verification of associate/assistant head coach experience",
        parent=parent,
        critical=False
    )

    sa_details = evaluator.add_parallel(
        id="Senior_Role_Details",
        desc="Details of associate or assistant head coach role",
        parent=sa_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Associate_AHC_Role",
        desc="Served as associate head coach or assistant head coach",
        parent=sa_details,
        claim=f"The coach served as '{sa.role_title}'.",
        urls=sa_urls,
        critical=False,
        additional_instruction="Verify if the coach held an associate/assistant head coach title.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Senior_Role_Institution",
        desc="Institution where senior assistant role was held",
        parent=sa_details,
        claim=f"The senior assistant role was held at '{sa.institution}'.",
        urls=sa_urls,
        critical=False,
        additional_instruction="Verify the institution where the senior assistant role occurred.",
    )

    sa_doc = evaluator.add_parallel(
        id="Senior_Role_Documentation",
        desc="URL reference supporting senior assistant experience",
        parent=sa_root,
        critical=False
    )
    _add_url_presence_node(
        evaluator,
        node_id="Senior_Role_URL_Reference",
        desc="URL documenting associate/assistant head coach experience",
        parent=sa_doc,
        urls=sa_urls,
        critical=False,
    )


async def build_ivy_background_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    ivy = data.ivy_background or IvyBackground()
    ivy_urls = _safe_list(ivy.urls)

    ivy_root = evaluator.add_parallel(
        id="Ivy_League_Background",
        desc="Verification of prior Ivy League connection",
        parent=parent,
        critical=False
    )

    ivy_type = evaluator.add_parallel(
        id="Ivy_Connection_Type",
        desc="Nature of prior Ivy League involvement",
        parent=ivy_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Ivy_Player",
        desc="Played football at an Ivy League institution",
        parent=ivy_type,
        claim=f"The coach played at an Ivy League institution: '{ivy.ivy_player}'.",
        urls=ivy_urls,
        critical=False,
        additional_instruction="Verify prior playing experience at an Ivy League school if claimed.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Ivy_Previous_Coach",
        desc="Previously coached at an Ivy League institution",
        parent=ivy_type,
        claim=f"The coach previously coached at an Ivy League institution: '{ivy.ivy_previous_coach}'.",
        urls=ivy_urls,
        critical=False,
        additional_instruction="Verify prior coaching experience at an Ivy League school if claimed.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Ivy_Connection_Present",
        desc="Has at least one form of prior Ivy League connection",
        parent=ivy_type,
        claim=f"The coach has at least one prior Ivy League connection: '{ivy.connection_present}'.",
        urls=ivy_urls,
        critical=False,
        additional_instruction="Confirm at least one Ivy League connection (playing or coaching) if indicated.",
    )

    ivy_doc = evaluator.add_parallel(
        id="Ivy_Background_Documentation",
        desc="URL reference supporting Ivy League background",
        parent=ivy_root,
        critical=False
    )
    _add_url_presence_node(
        evaluator,
        node_id="Ivy_Background_URL_Reference",
        desc="URL documenting Ivy League connection",
        parent=ivy_doc,
        urls=ivy_urls,
        critical=False,
    )


async def build_championship_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    champ = data.championship_achievements or ChampionshipAchievements()
    champ_urls = _safe_list(champ.urls)

    champ_root = evaluator.add_parallel(
        id="Championship_Achievements",
        desc="Verification of championship success in previous roles",
        parent=parent,
        critical=False
    )

    champ_details = evaluator.add_parallel(
        id="Championship_Details",
        desc="Specific championship accomplishments",
        parent=champ_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Conference_Title",
        desc="Was part of staff that won conference championship",
        parent=champ_details,
        claim=f"The coach was part of a staff that won a conference championship: '{champ.conference_title}'.",
        urls=champ_urls,
        critical=False,
        additional_instruction="Verify any stated conference championship involvement.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Championship_Years_Institutions",
        desc="Year(s) and institution(s) where championships were won",
        parent=champ_details,
        claim=f"Championships occurred in the following years/institutions: '{champ.championship_years_institutions}'.",
        urls=champ_urls,
        critical=False,
        additional_instruction="Verify the years and institutions for the championships if provided.",
    )

    champ_doc = evaluator.add_parallel(
        id="Championship_Documentation",
        desc="URL reference supporting championship success",
        parent=champ_root,
        critical=False
    )
    _add_url_presence_node(
        evaluator,
        node_id="Championship_URL_Reference",
        desc="URL documenting championship achievements",
        parent=champ_doc,
        urls=champ_urls,
        critical=False,
    )


async def build_first_season_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    fst = data.first_season_performance or FirstSeasonPerformance()
    fst_urls = _safe_list(fst.urls)

    first_root = evaluator.add_parallel(
        id="First_Season_Performance",
        desc="Verification of results from inaugural season as head coach",
        parent=parent,
        critical=False
    )

    season_results = evaluator.add_parallel(
        id="Season_Results",
        desc="Statistical and competitive results from first season",
        parent=first_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Overall_Record",
        desc="Win-loss record from first season",
        parent=season_results,
        claim=f"The overall record in the first season was '{fst.overall_record}'.",
        urls=fst_urls,
        critical=False,
        additional_instruction="Verify the overall win-loss record for the coach's first season.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Conference_Record",
        desc="Ivy League conference record from first season",
        parent=season_results,
        claim=f"The Ivy League conference record in the first season was '{fst.conference_record}'.",
        urls=fst_urls,
        critical=False,
        additional_instruction="Verify the conference record for the coach's first season.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Notable_First_Season_Achievements",
        desc="Conference title, playoff berth, or rankings achieved in first season",
        parent=season_results,
        claim=f"Notable first-season achievements include: {', '.join(fst.notable_achievements)}.",
        urls=fst_urls,
        critical=False,
        additional_instruction="Verify any notable achievements (e.g., titles, playoff berths, rankings) in the first season.",
    )

    first_doc = evaluator.add_parallel(
        id="First_Season_Documentation",
        desc="URL reference supporting first season results",
        parent=first_root,
        critical=False
    )
    _add_url_presence_node(
        evaluator,
        node_id="First_Season_URL_Reference",
        desc="URL documenting first season performance",
        parent=first_doc,
        urls=fst_urls,
        critical=False,
    )


async def build_player_development_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    dev = data.player_development or PlayerDevelopment()
    dev_urls = _safe_list(dev.urls)

    dev_root = evaluator.add_parallel(
        id="Player_Development_Record",
        desc="Verification of player development and recognition achievements",
        parent=parent,
        critical=False
    )

    dev_players = evaluator.add_parallel(
        id="Player_Honors",
        desc="Players developed who earned individual recognition",
        parent=dev_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="All_Conference_Development",
        desc="Developed All-Conference players in current or previous roles",
        parent=dev_players,
        claim=f"Developed All-Conference players: {', '.join(dev.all_conference_players)}.",
        urls=dev_urls,
        critical=False,
        additional_instruction="Verify if the coach developed any All-Conference players as cited.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="All_America_Development",
        desc="Developed All-America players in current or previous roles",
        parent=dev_players,
        claim=f"Developed All-America players: {', '.join(dev.all_america_players)}.",
        urls=dev_urls,
        critical=False,
        additional_instruction="Verify if the coach developed any All-America players as cited.",
    )

    dev_doc = evaluator.add_parallel(
        id="Development_Documentation",
        desc="URL reference supporting player development success",
        parent=dev_root,
        critical=False
    )
    _add_url_presence_node(
        evaluator,
        node_id="Development_URL_Reference",
        desc="URL documenting player development achievements",
        parent=dev_doc,
        urls=dev_urls,
        critical=False,
    )


async def build_current_program_tree(evaluator: Evaluator, parent, data: CoachProfileExtraction):
    cur = data.current_program_trajectory or CurrentProgramTrajectory()
    cur_urls = _safe_list(cur.urls)

    cur_root = evaluator.add_parallel(
        id="Current_Program_Trajectory",
        desc="Verification of current program status and competitive standing",
        parent=parent,
        critical=False
    )

    perf = evaluator.add_parallel(
        id="Program_Performance",
        desc="Current season and recent performance indicators",
        parent=cur_root,
        critical=False
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Recent_Season_Record",
        desc="Most recent or current season record",
        parent=perf,
        claim=f"The most recent/current season record is '{cur.recent_season_record}'.",
        urls=cur_urls,
        critical=False,
        additional_instruction="Verify the stated recent or current season record.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="National_FCS_Ranking",
        desc="Current or recent FCS national ranking",
        parent=perf,
        claim=f"The current/recent FCS national ranking is '{cur.national_fcs_ranking}'.",
        urls=cur_urls,
        critical=False,
        additional_instruction="Verify the cited FCS ranking, if present.",
    )

    await _verify_leaf_with_urls(
        evaluator,
        node_id="Program_Improvement_Evidence",
        desc="Evidence of program improvement or sustained excellence",
        parent=perf,
        claim=f"Evidence of program improvement or sustained success: '{cur.improvement_evidence}'.",
        urls=cur_urls,
        critical=False,
        additional_instruction="Verify qualitative or quantitative evidence of improvement/sustained excellence.",
    )

    cur_doc = evaluator.add_parallel(
        id="Current_Status_Documentation",
        desc="URL reference supporting current program status",
        parent=cur_root,
        critical=False
    )
    _add_url_presence_node(
        evaluator,
        node_id="Program_Status_URL_Reference",
        desc="URL documenting current program trajectory",
        parent=cur_doc,
        urls=cur_urls,
        critical=False,
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Ivy League head coach (2020–2025) task.
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

    # Extract structured profile
    extracted_profile = await evaluator.extract(
        prompt=prompt_extract_coach_profile(),
        template_class=CoachProfileExtraction,
        extraction_name="coach_profile",
    )

    # Top-level evaluation node (non-critical to allow mixed criticality in children)
    top = evaluator.add_parallel(
        id="Ivy_League_Head_Coach_Evaluation",
        desc="Comprehensive evaluation of an Ivy League head football coach hired between 2020-2025 who meets all specified career and qualification requirements",
        parent=root,
        critical=False
    )

    # Build subtrees
    await build_identification_tree(evaluator, top, extracted_profile)
    await build_education_tree(evaluator, top, extracted_profile)
    await build_experience_timeline_tree(evaluator, top, extracted_profile)
    await build_multi_level_tree(evaluator, top, extracted_profile)
    await build_position_roles_tree(evaluator, top, extracted_profile)
    await build_coordinator_tree(evaluator, top, extracted_profile)
    await build_senior_assistant_tree(evaluator, top, extracted_profile)
    await build_ivy_background_tree(evaluator, top, extracted_profile)
    await build_championship_tree(evaluator, top, extracted_profile)
    await build_first_season_tree(evaluator, top, extracted_profile)
    await build_player_development_tree(evaluator, top, extracted_profile)
    await build_current_program_tree(evaluator, top, extracted_profile)

    return evaluator.get_summary()