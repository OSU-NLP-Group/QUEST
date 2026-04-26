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
TASK_ID = "coach_michigan_career_path_2026"
TASK_DESCRIPTION = (
    "Identify a college football coach who meets ALL of the following criteria:\n\n"
    "1. Played college football as a quarterback at the University of Michigan\n"
    "2. Was born in Michigan\n"
    "3. Started their coaching career at a high school in Michigan\n"
    "4. Served as a graduate assistant at a Division I FBS program\n"
    "5. Was promoted to a non-graduate assistant on-field coaching position (position coach or coordinator) within three years of starting their collegiate coaching career\n"
    "6. Held an offensive coordinator position at a Power Five conference school\n"
    "7. Worked under the same head coach at multiple (at least two) different universities\n"
    "8. As of February 2026, holds the position of offensive coordinator at a Big Ten conference school\n"
    "9. Has coaching experience at a minimum of five different Division I FBS universities\n\n"
    "Provide the coach's full name and supporting URLs that verify each of these career milestones and current position."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PlayingCareer(BaseModel):
    school: Optional[str] = None
    position: Optional[str] = None
    years: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BirthInfo(BaseModel):
    birth_place: Optional[str] = None  # e.g., city, state
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FirstCoachingPosition(BaseModel):
    hs_name: Optional[str] = None
    state: Optional[str] = None
    role_title: Optional[str] = None
    year_or_season: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GraduateAssistantRole(BaseModel):
    school: Optional[str] = None
    role_title: Optional[str] = None  # should include "Graduate Assistant"
    years_or_seasons: Optional[str] = None
    collegiate_start_year: Optional[str] = None  # first year in collegiate coaching (often first GA season)
    sources: List[str] = Field(default_factory=list)


class PromotionInfo(BaseModel):
    promotion_school: Optional[str] = None
    promotion_role_title: Optional[str] = None  # position coach or coordinator (non-GA)
    promotion_year_or_season: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PowerFiveOC(BaseModel):
    oc_school: Optional[str] = None
    conference_at_the_time: Optional[str] = None
    oc_title: Optional[str] = None  # offensive coordinator or co-offensive coordinator
    years_or_seasons: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SameHeadCoach(BaseModel):
    head_coach_name: Optional[str] = None
    universities: List[str] = Field(default_factory=list)  # universities where worked under same head coach
    sources: List[str] = Field(default_factory=list)


class CurrentOC(BaseModel):
    current_school: Optional[str] = None
    current_title: Optional[str] = None  # should be offensive coordinator
    as_of_date_or_note: Optional[str] = None  # e.g., "announced Jan 2026", "as of Feb 2026"
    sources: List[str] = Field(default_factory=list)


class FBSExperience(BaseModel):
    universities: List[str] = Field(default_factory=list)  # distinct FBS universities coached at
    sources: List[str] = Field(default_factory=list)


class CoachExtraction(BaseModel):
    coach_name: Optional[str] = None

    playing: Optional[PlayingCareer] = None
    birth: Optional[BirthInfo] = None

    first_coaching: Optional[FirstCoachingPosition] = None
    ga_role: Optional[GraduateAssistantRole] = None
    early_promotion: Optional[PromotionInfo] = None

    power5_oc: Optional[PowerFiveOC] = None
    same_head_coach: Optional[SameHeadCoach] = None

    current_oc: Optional[CurrentOC] = None
    fbs_experience: Optional[FBSExperience] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach() -> str:
    return """
Extract the single coach proposed in the answer and all referenced facts as structured data. Only extract what is explicitly stated in the answer. If any field is missing, return null for that field or an empty list for list fields.

Required fields:
- coach_name: The coach’s full name.

College playing career:
- playing.school: The college where the coach played (expected: University of Michigan)
- playing.position: The position played (expected: quarterback)
- playing.years: The years or seasons listed (if present)
- playing.sources: All URLs provided that substantiate the Michigan quarterback playing career

Geographic origins:
- birth.birth_place: The city/state if given
- birth.state: The state of birth (expected: Michigan)
- birth.sources: All URLs provided that substantiate the Michigan birth

First coaching position (high school in Michigan):
- first_coaching.hs_name: High school name
- first_coaching.state: State for the high school (expected: Michigan)
- first_coaching.role_title: Role/title
- first_coaching.year_or_season: Year or season when held (if given)
- first_coaching.sources: URLs supporting this being the first coaching position and its Michigan location

Graduate assistant role (FBS):
- ga_role.school: University where the GA role was held
- ga_role.role_title: Should include "Graduate Assistant" or equivalent
- ga_role.years_or_seasons: The timeframe (if present)
- ga_role.collegiate_start_year: The first year of collegiate coaching (if extractable)
- ga_role.sources: URLs showing GA role and institution context

Early promotion (non-GA on-field) within three years of starting collegiate coaching:
- early_promotion.promotion_school: School of the promoted position
- early_promotion.promotion_role_title: Position coach or coordinator title (non-GA)
- early_promotion.promotion_year_or_season: Year/season (if given)
- early_promotion.sources: URLs for the promotion and timing

Offensive coordinator at a Power Five school:
- power5_oc.oc_school: School name
- power5_oc.conference_at_the_time: The conference at the time of employment (ACC, Big Ten, Big 12, Pac-12, SEC)
- power5_oc.oc_title: Title (e.g., Offensive Coordinator, Co-OC)
- power5_oc.years_or_seasons: Timeframe (if given)
- power5_oc.sources: URLs supporting the OC role at a Power Five school

Working under the same head coach across multiple universities:
- same_head_coach.head_coach_name: The head coach’s name
- same_head_coach.universities: List of at least two different universities where the coach worked under this head coach
- same_head_coach.sources: URLs supporting both stints under the same head coach

Current Big Ten OC as of February 2026:
- current_oc.current_school: The Big Ten school
- current_oc.current_title: Should be offensive coordinator
- current_oc.as_of_date_or_note: Any phrase indicating timing (e.g., "as of Feb 2026", announcement date, etc.)
- current_oc.sources: URLs supporting current role and Big Ten membership as of Feb 2026

FBS coaching experience breadth:
- fbs_experience.universities: Distinct Division I FBS universities where the coach has worked (5+ expected)
- fbs_experience.sources: URLs supporting the career stops and FBS status

Rules:
- Collect all URLs explicitly present in the answer for each section. Do not invent URLs.
- Preserve text exactly where feasible. Use nulls for missing single-value fields and empty arrays for missing list fields.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nz(s: Optional[str]) -> str:
    return s or ""


def _nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        for u in (lst or []):
            if isinstance(u, str) and u.strip():
                combined.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for u in combined:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Tree-building and verification functions                                    #
# --------------------------------------------------------------------------- #
async def build_college_playing_career(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_sequential(
        id="College_Playing_Career",
        desc="Verify the coach played college football as a quarterback at the University of Michigan",
        parent=parent_node,
        critical=True,
    )

    verification_group = evaluator.add_parallel(
        id="Playing_Career_Verification",
        desc="Verify the basic playing career facts",
        parent=node,
        critical=True,
    )

    coach_name = _nz(data.coach_name)
    playing = data.playing or PlayingCareer()
    psrc = _nonempty_urls(playing.sources)

    # University_of_Michigan_QB
    leaf_um_qb = evaluator.add_leaf(
        id="University_of_Michigan_QB",
        desc="The coach played quarterback at the University of Michigan",
        parent=verification_group,
        critical=True,
    )
    claim_um_qb = f"{coach_name} played quarterback for the University of Michigan football team."
    await evaluator.verify(
        claim=claim_um_qb,
        node=leaf_um_qb,
        sources=psrc,
        additional_instruction="Confirm the page(s) explicitly indicate the person played as a quarterback at the University of Michigan (college football)."
    )

    # College_Football_Level
    leaf_college_level = evaluator.add_leaf(
        id="College_Football_Level",
        desc="The playing career was at the college football level",
        parent=verification_group,
        critical=True,
    )
    claim_college_level = f"The documented playing career for {coach_name} at the University of Michigan is collegiate (NCAA) football."
    await evaluator.verify(
        claim=claim_college_level,
        node=leaf_college_level,
        sources=psrc,
        additional_instruction="Confirm that the playing experience referenced is college football (University of Michigan is an NCAA Division I FBS program)."
    )

    # College_Playing_References (require URLs present)
    evaluator.add_custom_node(
        result=len(psrc) > 0,
        id="College_Playing_References",
        desc="URLs provided that verify the college playing career at Michigan as a quarterback",
        parent=node,
        critical=True
    )


async def build_michigan_origins(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_sequential(
        id="Michigan_Origins",
        desc="Verify the coach was born in Michigan",
        parent=parent_node,
        critical=True,
    )

    verify_grp = evaluator.add_parallel(
        id="Birth_Location_Verification",
        desc="Verify the birth location",
        parent=node,
        critical=True,
    )

    birth = data.birth or BirthInfo()
    bsrc = _nonempty_urls(birth.sources)
    coach_name = _nz(data.coach_name)

    # Birth_State
    leaf_birth_state = evaluator.add_leaf(
        id="Birth_State",
        desc="The coach was born in the state of Michigan",
        parent=verify_grp,
        critical=True,
    )
    claim_birth_state = f"{coach_name} was born in the state of Michigan."
    await evaluator.verify(
        claim=claim_birth_state,
        node=leaf_birth_state,
        sources=bsrc,
        additional_instruction="Confirm the page explicitly indicates a birth place in Michigan (e.g., a Michigan city)."
    )

    # Michigan_Origins_References (require URLs present)
    evaluator.add_custom_node(
        result=len(bsrc) > 0,
        id="Michigan_Origins_References",
        desc="URLs provided that verify Michigan birth or hometown",
        parent=node,
        critical=True
    )


async def build_first_coaching_position(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_sequential(
        id="First_Coaching_Position",
        desc="Verify the coach's first coaching position was at a high school in Michigan",
        parent=parent_node,
        critical=True,
    )

    verify_grp = evaluator.add_parallel(
        id="High_School_Position_Details",
        desc="Verify the details of the high school coaching position",
        parent=node,
        critical=True,
    )

    fc = data.first_coaching or FirstCoachingPosition()
    fsrc = _nonempty_urls(fc.sources)
    coach_name = _nz(data.coach_name)
    hs_name = _nz(fc.hs_name)
    hs_state = _nz(fc.state)

    # High_School_Location
    leaf_hs_loc = evaluator.add_leaf(
        id="High_School_Location",
        desc="The first coaching position was at a high school located in Michigan",
        parent=verify_grp,
        critical=True,
    )
    claim_hs_loc = f"{coach_name}'s first coaching position was at {hs_name}, a high school in Michigan."
    await evaluator.verify(
        claim=claim_hs_loc,
        node=leaf_hs_loc,
        sources=fsrc,
        additional_instruction="Confirm the page(s) indicate the coach began at a high school in Michigan; the high school's location should be in Michigan."
    )

    # First_Position_Timing
    leaf_first_timing = evaluator.add_leaf(
        id="First_Position_Timing",
        desc="This was the coach's first coaching position",
        parent=verify_grp,
        critical=True,
    )
    claim_first_timing = f"{coach_name} began his coaching career at {hs_name} (first coaching position)."
    await evaluator.verify(
        claim=claim_first_timing,
        node=leaf_first_timing,
        sources=fsrc,
        additional_instruction="Look for phrases like 'began his coaching career' or 'first coaching job' indicating this was the first role."
    )

    # First_Position_References (require URLs)
    evaluator.add_custom_node(
        result=len(fsrc) > 0,
        id="First_Position_References",
        desc="URLs provided that verify the first coaching position at a Michigan high school",
        parent=node,
        critical=True
    )


async def build_graduate_assistant_role(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_sequential(
        id="Graduate_Assistant_Role",
        desc="Verify the coach served as a graduate assistant at a Division I FBS program",
        parent=parent_node,
        critical=True,
    )

    verify_grp = evaluator.add_parallel(
        id="GA_Position_Details",
        desc="Verify the graduate assistant position details",
        parent=node,
        critical=True,
    )

    ga = data.ga_role or GraduateAssistantRole()
    gsrc = _nonempty_urls(ga.sources)
    coach_name = _nz(data.coach_name)
    ga_school = _nz(ga.school)

    # FBS_Level
    leaf_fbs = evaluator.add_leaf(
        id="FBS_Level",
        desc="The graduate assistant position was at a Division I FBS university",
        parent=verify_grp,
        critical=True,
    )
    claim_fbs = f"{coach_name} served as a graduate assistant at {ga_school}, which competes in NCAA Division I FBS."
    await evaluator.verify(
        claim=claim_fbs,
        node=leaf_fbs,
        sources=gsrc,
        additional_instruction="Confirm that the GA role was at a university that is an NCAA Division I FBS program."
    )

    # GA_Position_Title
    leaf_ga_title = evaluator.add_leaf(
        id="GA_Position_Title",
        desc="The role was explicitly identified as graduate assistant",
        parent=verify_grp,
        critical=True,
    )
    claim_ga_title = f"{coach_name}'s role at {ga_school} is explicitly listed as Graduate Assistant."
    await evaluator.verify(
        claim=claim_ga_title,
        node=leaf_ga_title,
        sources=gsrc,
        additional_instruction="Look for 'Graduate Assistant' or clear equivalent in the title."
    )

    # GA_References (require URLs)
    evaluator.add_custom_node(
        result=len(gsrc) > 0,
        id="GA_References",
        desc="URLs provided that verify the graduate assistant position at an FBS program",
        parent=node,
        critical=True
    )


async def build_early_promotion(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_sequential(
        id="Early_Promotion",
        desc="Verify the coach was promoted to a non-GA on-field coaching position within three years of starting collegiate coaching",
        parent=parent_node,
        critical=True,
    )

    # Details
    details_grp = evaluator.add_parallel(
        id="Promotion_Position_Details",
        desc="Verify the details of the promoted position",
        parent=node,
        critical=True,
    )

    promo = data.early_promotion or PromotionInfo()
    ga = data.ga_role or GraduateAssistantRole()
    psrc = _nonempty_urls(promo.sources)
    coach_name = _nz(data.coach_name)
    prom_school = _nz(promo.promotion_school)
    prom_title = _nz(promo.promotion_role_title)

    # Non_GA_Role
    leaf_non_ga = evaluator.add_leaf(
        id="Non_GA_Role",
        desc="The promoted position was not a graduate assistant role",
        parent=details_grp,
        critical=True,
    )
    claim_non_ga = f"{coach_name} was promoted to a non-graduate assistant on-field role at {prom_school} (title: {prom_title})."
    await evaluator.verify(
        claim=claim_non_ga,
        node=leaf_non_ga,
        sources=psrc,
        additional_instruction="Verify that the promoted role is not a GA, and is an on-field coaching role."
    )

    # Position_Coach_Level
    leaf_pos_level = evaluator.add_leaf(
        id="Position_Coach_Level",
        desc="The promoted position was a position coach or coordinator level",
        parent=details_grp,
        critical=True,
    )
    claim_pos_level = f"The promoted role for {coach_name} at {prom_school} was a position coach or coordinator (on-field) position."
    await evaluator.verify(
        claim=claim_pos_level,
        node=leaf_pos_level,
        sources=psrc,
        additional_instruction="Confirm that the role is clearly an on-field coaching position (position coach or coordinator)."
    )

    # Timing
    timing_grp = evaluator.add_parallel(
        id="Promotion_Timing",
        desc="Verify the timing of the promotion",
        parent=node,
        critical=True,
    )
    leaf_timeline = evaluator.add_leaf(
        id="Timeline_Within_Three_Years",
        desc="The promotion occurred within three years of the first collegiate coaching position",
        parent=timing_grp,
        critical=True,
    )
    start_year = _nz(ga.collegiate_start_year)
    promo_year = _nz(promo.promotion_year_or_season)
    timeline_sources = _combine_sources(psrc, (ga.sources if ga else []))
    claim_timeline = (
        f"The time from {coach_name}'s first collegiate coaching role (start year: {start_year}) "
        f"to the first non-GA on-field role (year: {promo_year}) was three years or less."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=leaf_timeline,
        sources=timeline_sources,
        additional_instruction="Use the dates or seasons indicated to verify the elapsed time is <= 3 years."
    )

    # Promotion_References (require URLs)
    evaluator.add_custom_node(
        result=len(psrc) > 0,
        id="Promotion_References",
        desc="URLs provided that verify the promotion timeline and position details",
        parent=node,
        critical=True
    )


async def build_power5_oc(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_sequential(
        id="Offensive_Coordinator_Power_Five",
        desc="Verify the coach held an offensive coordinator position at a Power Five conference school",
        parent=parent_node,
        critical=True,
    )

    details_grp = evaluator.add_parallel(
        id="Power_Five_OC_Details",
        desc="Verify the Power Five OC position details",
        parent=node,
        critical=True,
    )

    p5 = data.power5_oc or PowerFiveOC()
    p5src = _nonempty_urls(p5.sources)
    coach_name = _nz(data.coach_name)
    oc_school = _nz(p5.oc_school)
    oc_title = _nz(p5.oc_title)

    # Power_Five_Conference
    leaf_p5_conf = evaluator.add_leaf(
        id="Power_Five_Conference",
        desc="The school was a member of a Power Five conference (ACC, Big Ten, Big 12, Pac-12, or SEC) at the time of employment",
        parent=details_grp,
        critical=True,
    )
    claim_p5_conf = f"{oc_school} is a member of one of the Power Five conferences (ACC, Big Ten, Big 12, Pac-12, SEC) at the time of {coach_name}'s OC role."
    await evaluator.verify(
        claim=claim_p5_conf,
        node=leaf_p5_conf,
        sources=p5src,
        additional_instruction="Confirm the institution is/was in a Power Five conference during the tenure described."
    )

    # OC_Title
    leaf_oc_title = evaluator.add_leaf(
        id="OC_Title",
        desc="The position title included offensive coordinator responsibilities",
        parent=details_grp,
        critical=True,
    )
    claim_oc_title = f"{coach_name} served as an offensive coordinator (or co-offensive coordinator) at {oc_school}."
    await evaluator.verify(
        claim=claim_oc_title,
        node=leaf_oc_title,
        sources=p5src,
        additional_instruction="Confirm that the title includes offensive coordinator responsibilities."
    )

    # OC_Power_Five_References (require URLs)
    evaluator.add_custom_node(
        result=len(p5src) > 0,
        id="OC_Power_Five_References",
        desc="URLs provided that verify the offensive coordinator position at a Power Five school",
        parent=node,
        critical=True
    )


async def build_same_head_coach(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_sequential(
        id="Same_Head_Coach_Multiple_Schools",
        desc="Verify the coach worked under the same head coach at multiple different universities",
        parent=parent_node,
        critical=True,
    )

    details_grp = evaluator.add_parallel(
        id="Multiple_Schools_Details",
        desc="Verify the details of working under the same head coach",
        parent=node,
        critical=True,
    )

    shc = data.same_head_coach or SameHeadCoach()
    shcsrc = _nonempty_urls(shc.sources)
    coach_name = _nz(data.coach_name)
    head_name = _nz(shc.head_coach_name)
    universities = shc.universities or []
    universities_str = ", ".join(universities) if universities else ""

    # Minimum_Two_Universities
    leaf_min_two = evaluator.add_leaf(
        id="Minimum_Two_Universities",
        desc="The coach worked at a minimum of two different universities under the same head coach",
        parent=details_grp,
        critical=True,
    )
    claim_min_two = f"{coach_name} worked under head coach {head_name} at at least two different universities: {universities_str}."
    await evaluator.verify(
        claim=claim_min_two,
        node=leaf_min_two,
        sources=shcsrc,
        additional_instruction="Confirm the pages show the coach worked under the same head coach at two or more distinct universities."
    )

    # Same_Head_Coach_Identified
    leaf_same_hc = evaluator.add_leaf(
        id="Same_Head_Coach_Identified",
        desc="The specific head coach is identified and verified at both institutions",
        parent=details_grp,
        critical=True,
    )
    claim_same_hc = f"The head coach at each of those universities was {head_name}, and {coach_name} served on that head coach's staff."
    await evaluator.verify(
        claim=claim_same_hc,
        node=leaf_same_hc,
        sources=shcsrc,
        additional_instruction="Confirm that the same named head coach is the head coach during the coach's stints at both institutions."
    )

    # Same_Head_Coach_References (require URLs)
    evaluator.add_custom_node(
        result=len(shcsrc) > 0,
        id="Same_Head_Coach_References",
        desc="URLs provided that verify working under the same head coach at multiple universities",
        parent=node,
        critical=True
    )


async def build_current_status_and_experience(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_parallel(
        id="Current_Status_and_Experience_Breadth",
        desc="Verify the coach's current position as of February 2026 and overall FBS coaching experience",
        parent=parent_node,
        critical=True,
    )

    # Current Big Ten OC
    cur_node = evaluator.add_sequential(
        id="Current_Big_Ten_OC_Position",
        desc="Verify the coach's current position (as of February 2026) is offensive coordinator at a Big Ten conference school",
        parent=node,
        critical=True,
    )

    cur_details = evaluator.add_parallel(
        id="Current_Position_Details",
        desc="Verify the details of the current position",
        parent=cur_node,
        critical=True,
    )

    curr = data.current_oc or CurrentOC()
    csrc = _nonempty_urls(curr.sources)
    coach_name = _nz(data.coach_name)
    cur_school = _nz(curr.current_school)
    cur_title = _nz(curr.current_title)

    # Current_OC_Title
    leaf_cur_title = evaluator.add_leaf(
        id="Current_OC_Title",
        desc="The current position title is offensive coordinator",
        parent=cur_details,
        critical=True,
    )
    claim_cur_title = f"As of February 2026, {coach_name} is the offensive coordinator at {cur_school}."
    await evaluator.verify(
        claim=claim_cur_title,
        node=leaf_cur_title,
        sources=csrc,
        additional_instruction="Confirm that the current title includes 'offensive coordinator' and is current/announced by February 2026."
    )

    # Big_Ten_Conference_Member
    leaf_b1g = evaluator.add_leaf(
        id="Big_Ten_Conference_Member",
        desc="The current employer is a Big Ten conference member school",
        parent=cur_details,
        critical=True,
    )
    claim_b1g = f"{cur_school} is a member of the Big Ten Conference."
    await evaluator.verify(
        claim=claim_b1g,
        node=leaf_b1g,
        sources=csrc,
        additional_instruction="Confirm the institution is a Big Ten member as of 2026."
    )

    # Current_Position_Timing
    cur_timing = evaluator.add_parallel(
        id="Current_Position_Timing",
        desc="Verify the timing of the current position",
        parent=cur_node,
        critical=True,
    )
    leaf_as_of = evaluator.add_leaf(
        id="As_of_February_2026",
        desc="The position was held or announced as of February 2026",
        parent=cur_timing,
        critical=True,
    )
    as_of_note = _nz(curr.as_of_date_or_note)
    claim_as_of = f"{coach_name}'s offensive coordinator position at {cur_school} was held or announced by February 2026. Evidence or note: {as_of_note}"
    await evaluator.verify(
        claim=claim_as_of,
        node=leaf_as_of,
        sources=csrc,
        additional_instruction="Confirm that the page shows the role existed or was officially announced by February 2026 (publication/update date or explicit wording)."
    )

    # Current_Position_References (require URLs)
    evaluator.add_custom_node(
        result=len(csrc) > 0,
        id="Current_Position_References",
        desc="URLs provided that verify the current position as OC at a Big Ten school in February 2026",
        parent=cur_node,
        critical=True
    )

    # FBS Experience breadth
    exp_node = evaluator.add_sequential(
        id="FBS_Coaching_Experience_Breadth",
        desc="Verify the coach has worked at a minimum of five different FBS universities in any coaching capacity",
        parent=node,
        critical=True,
    )

    exp_count = evaluator.add_parallel(
        id="FBS_Experience_Count",
        desc="Verify the number of FBS universities",
        parent=exp_node,
        critical=True,
    )

    exp = data.fbs_experience or FBSExperience()
    esrc = _nonempty_urls(exp.sources)
    schools = exp.universities or []
    schools_str = ", ".join(schools) if schools else ""
    # Minimum_Five_FBS_Schools
    leaf_min5 = evaluator.add_leaf(
        id="Minimum_Five_FBS_Schools",
        desc="The coach has been employed by at least five different Division I FBS universities",
        parent=exp_count,
        critical=True,
    )
    claim_min5 = f"The coach has coached at least five distinct NCAA Division I FBS universities: {schools_str}."
    await evaluator.verify(
        claim=claim_min5,
        node=leaf_min5,
        sources=esrc,
        additional_instruction="Confirm that there are 5 or more distinct FBS institutions listed in the coach's career history."
    )

    # FBS_Schools_Identified (set critical True to satisfy parent constraint)
    leaf_listed = evaluator.add_leaf(
        id="FBS_Schools_Identified",
        desc="All five (or more) FBS universities are specifically identified",
        parent=exp_count,
        critical=True,
    )
    claim_listed = f"The answer specifically identifies at least five distinct FBS universities for { _nz(data.coach_name) }'s coaching career."
    await evaluator.verify(
        claim=claim_listed,
        node=leaf_listed,
        sources=esrc,
        additional_instruction="Verify that the institutions are explicitly named and are FBS programs."
    )

    # Experience_Breadth_References (require URLs)
    evaluator.add_custom_node(
        result=len(esrc) > 0,
        id="Experience_Breadth_References",
        desc="URLs provided that verify coaching positions at a minimum of five different FBS universities",
        parent=exp_node,
        critical=True
    )


async def build_educational_and_geographic(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_parallel(
        id="Educational_and_Geographic_Background",
        desc="Verify the coach's educational background and geographic origins meet Michigan-specific requirements",
        parent=parent_node,
        critical=True,
    )
    await build_college_playing_career(evaluator, node, data)
    await build_michigan_origins(evaluator, node, data)


async def build_early_coaching_path(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_sequential(
        id="Early_Coaching_Career_Path",
        desc="Verify the coach's early career progression starting from high school coaching through early collegiate positions",
        parent=parent_node,
        critical=True,
    )
    await build_first_coaching_position(evaluator, node, data)
    await build_graduate_assistant_role(evaluator, node, data)
    await build_early_promotion(evaluator, node, data)


async def build_career_advancement_and_relationships(evaluator: Evaluator, parent_node, data: CoachExtraction):
    node = evaluator.add_parallel(
        id="Career_Advancement_and_Relationships",
        desc="Verify the coach's career advancement to coordinator level and professional relationships",
        parent=parent_node,
        critical=True,
    )
    await build_power5_oc(evaluator, node, data)
    await build_same_head_coach(evaluator, node, data)


async def build_full_tree(evaluator: Evaluator, extracted: CoachExtraction):
    # Top-level critical sequential node
    top = evaluator.add_sequential(
        id="Coach_Identification",
        desc="Identify a college football coach who meets all specified career path criteria related to Michigan connections, career progression, and current position as of February 2026",
        parent=evaluator.root,
        critical=True,
    )

    await build_educational_and_geographic(evaluator, top, extracted)
    await build_early_coaching_path(evaluator, top, extracted)
    await build_career_advancement_and_relationships(evaluator, top, extracted)
    await build_current_status_and_experience(evaluator, top, extracted)


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
    Evaluate an answer for the Michigan-connected coach career path task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_coach(),
        template_class=CoachExtraction,
        extraction_name="coach_profile_extraction"
    )

    # Build and run verification tree
    await build_full_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()