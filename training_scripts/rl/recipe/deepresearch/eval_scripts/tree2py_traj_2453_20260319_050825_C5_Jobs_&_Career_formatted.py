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
TASK_ID = "coach_dec2024_fbs_appointment"
TASK_DESCRIPTION = """
Identify the college football head coach who was appointed to a new head coaching position at an FBS university in December 2024 and meets ALL of the following criteria:

1. Previously served as head coach at an FCS institution from 2022 to 2024, compiling an overall record of 26-13
2. Led their FCS team to NCAA FCS playoff appearances in all three seasons as head coach (2022, 2023, and 2024)
3. Before their FCS head coaching tenure, previously served as an assistant coach at the same FCS institution from 2004 to 2006
4. Began their coaching career as a graduate assistant at a Big Ten Conference university from 1999 to 2001, earning their master's degree during that time
5. Served under a head coach who won three Rose Bowl games during their tenure at that Big Ten university

Provide the coach's full name and their current head coaching position (institution name).
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AppointmentInfo(BaseModel):
    coach_name: Optional[str] = None
    current_fbs_institution: Optional[str] = None
    appointment_month: Optional[str] = None  # Expect "December" or similar
    appointment_year: Optional[str] = None   # Expect "2024"
    appointment_urls: List[str] = Field(default_factory=list)


class FCSHeadCoachInfo(BaseModel):
    fcs_institution: Optional[str] = None
    overall_record_2022_2024: Optional[str] = None  # Expect "26-13" or similar
    seasons_listed: List[str] = Field(default_factory=list)  # e.g., ["2022","2023","2024"]
    playoff_appearances_years: List[str] = Field(default_factory=list)  # Expect 2022,2023,2024
    fcs_record_urls: List[str] = Field(default_factory=list)


class AssistantCoachInfo(BaseModel):
    institution: Optional[str] = None
    years: Optional[str] = None  # Expect "2004-2006" or equivalent
    assistant_urls: List[str] = Field(default_factory=list)


class GraduateAssistantInfo(BaseModel):
    big_ten_university: Optional[str] = None
    years: Optional[str] = None  # Expect "1999-2001" or equivalent
    masters_degree: Optional[str] = None  # e.g., "Master's in Education"
    ga_urls: List[str] = Field(default_factory=list)


class MentorCoachInfo(BaseModel):
    mentor_name: Optional[str] = None
    rose_bowl_wins: Optional[str] = None  # Expect "3" or "three"
    rose_bowl_urls: List[str] = Field(default_factory=list)


class CoachExtraction(BaseModel):
    appointment: Optional[AppointmentInfo] = None
    fcs_head: Optional[FCSHeadCoachInfo] = None
    assistant: Optional[AssistantCoachInfo] = None
    ga: Optional[GraduateAssistantInfo] = None
    mentor: Optional[MentorCoachInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_info() -> str:
    return """
    Extract the following structured information about the coach identified in the answer. Extract ONLY what is explicitly present in the answer text.

    1) appointment (current FBS head coach position, appointed Dec 2024)
       - coach_name: full name of the coach
       - current_fbs_institution: the FBS university where they were appointed head coach
       - appointment_month: the month mentioned for the appointment (e.g., "December")
       - appointment_year: the year mentioned for the appointment (e.g., "2024")
       - appointment_urls: list of URLs cited that support this appointment

    2) fcs_head (FCS head coach record 2022–2024)
       - fcs_institution: the FCS university where they were head coach (2022–2024)
       - overall_record_2022_2024: overall record over 2022–2024 (e.g., "26-13")
       - seasons_listed: list of seasons explicitly mentioned for this FCS tenure (e.g., ["2022","2023","2024"])
       - playoff_appearances_years: which seasons are claimed as FCS playoff appearances (e.g., ["2022","2023","2024"])
       - fcs_record_urls: list of URLs cited that support the FCS record and playoff appearances

    3) assistant (assistant coaching before that at the same FCS institution)
       - institution: name of the institution where they were an assistant (should match fcs_institution if stated)
       - years: the years they were assistant (e.g., "2004-2006")
       - assistant_urls: list of URLs cited that support the assistant coaching tenure

    4) ga (graduate assistant background at a Big Ten university)
       - big_ten_university: name of the Big Ten university
       - years: the years of GA tenure (e.g., "1999-2001")
       - masters_degree: the master's degree earned there (if the degree/field is stated, include the text; otherwise null)
       - ga_urls: list of URLs cited that support the GA tenure and/or master's degree

    5) mentor (the head coach they served under; three Rose Bowl wins)
       - mentor_name: the head coach they served under as a GA
       - rose_bowl_wins: the number of Rose Bowl wins stated for the mentor (e.g., "3", "three")
       - rose_bowl_urls: list of URLs cited that support the mentor's Rose Bowl wins

    Return a JSON object for these five sections. If any field is missing from the answer, set it to null (or an empty list for URL lists).
    The 'appointment_urls', 'fcs_record_urls', 'assistant_urls', 'ga_urls', and 'rose_bowl_urls' must be URLs explicitly present in the answer.
    """.strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _safe(val: Optional[str], fallback: str = "unknown") -> str:
    return (val or "").strip() if val else fallback


def _urls(lst: Optional[List[str]]) -> List[str]:
    return [u for u in (lst or []) if isinstance(u, str) and u.strip()]


def _combine_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in _urls(lst):
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_current_fbs_position(evaluator: Evaluator, parent_node, ex: CoachExtraction) -> None:
    node = evaluator.add_parallel(
        id="current_fbs_head_coaching_position",
        desc="Verify the coach was appointed to an FBS head coaching position in December 2024",
        parent=parent_node,
        critical=True,
    )

    appt = ex.appointment or AppointmentInfo()
    coach = _safe(appt.coach_name)
    school = _safe(appt.current_fbs_institution)
    month = _safe(appt.appointment_month)
    year = _safe(appt.appointment_year)
    appt_urls = _urls(appt.appointment_urls)

    # Reference presence (treat as critical to enforce source-grounding)
    evaluator.add_custom_node(
        result=len(appt_urls) > 0,
        id="reference_url_current_position",
        desc="Valid reference URL supporting the appointment information",
        parent=node,
        critical=True,
    )

    # Appointment details (Dec 2024 + head coach role at institution)
    leaf_appt = evaluator.add_leaf(
        id="appointment_details",
        desc="Coach was appointed as head coach at an FBS institution in December 2024",
        parent=node,
        critical=True,
    )
    claim_appt = (
        f"In {month} {year}, {coach} was appointed as the head football coach at {school}."
    )
    await evaluator.verify(
        claim=claim_appt,
        node=leaf_appt,
        sources=appt_urls,
        additional_instruction=(
            "Confirm the announcement month and year are December 2024 (allow minor variants like 'Dec. 2024'). "
            "The page should clearly indicate a head coach appointment at the named institution."
        ),
    )

    # Institution is FBS
    leaf_fbs = evaluator.add_leaf(
        id="institution_is_fbs",
        desc="The appointed institution competes in NCAA FBS (Football Bowl Subdivision)",
        parent=node,
        critical=True,
    )
    claim_fbs = f"{school} competes in NCAA FBS (Football Bowl Subdivision) football."
    await evaluator.verify(
        claim=claim_fbs,
        node=leaf_fbs,
        sources=appt_urls,
        additional_instruction=(
            "Verify that the institution is an FBS program. If the page references a clear FBS conference (e.g., Big Ten, SEC, Big 12, ACC, Pac-12/MWC/AAC/CUSA/SBC), that is sufficient."
        ),
    )


async def verify_fcs_record(evaluator: Evaluator, parent_node, ex: CoachExtraction) -> None:
    node = evaluator.add_parallel(
        id="fcs_head_coaching_record",
        desc="Verify the coach's FCS head coaching record from 2022-2024",
        parent=parent_node,
        critical=True,
    )

    fcs = ex.fcs_head or FCSHeadCoachInfo()
    appt = ex.appointment or AppointmentInfo()
    coach = _safe(appt.coach_name)
    fcs_school = _safe(fcs.fcs_institution)
    record = _safe(fcs.overall_record_2022_2024)
    fcs_urls = _urls(fcs.fcs_record_urls)

    # Reference presence
    evaluator.add_custom_node(
        result=len(fcs_urls) > 0,
        id="reference_url_fcs_record",
        desc="Valid reference URL supporting the FCS coaching record and playoff appearances",
        parent=node,
        critical=True,
    )

    # Overall record 26-13 over 2022–2024
    leaf_record = evaluator.add_leaf(
        id="overall_record_26_13",
        desc="Coach compiled a 26-13 overall record as FCS head coach from 2022 to 2024",
        parent=node,
        critical=True,
    )
    claim_record = (
        f"From 2022 through 2024, as head coach of {fcs_school}, {coach} compiled an overall record of 26-13."
    )
    await evaluator.verify(
        claim=claim_record,
        node=leaf_record,
        sources=fcs_urls,
        additional_instruction=(
            "Confirm that the combined overall record across the 2022, 2023, and 2024 FCS seasons totals 26–13. "
            "Allow minor formatting variants like '26–13' with an en dash."
        ),
    )

    # Playoff appearances in all three seasons
    leaf_playoffs = evaluator.add_leaf(
        id="three_fcs_playoff_appearances",
        desc="Coach led their FCS team to NCAA FCS playoff appearances in all three seasons",
        parent=node,
        critical=True,
    )
    claim_playoffs = (
        f"{fcs_school} made the NCAA Division I FCS playoffs in 2022, 2023, and 2024 under head coach {coach}."
    )
    await evaluator.verify(
        claim=claim_playoffs,
        node=leaf_playoffs,
        sources=fcs_urls,
        additional_instruction="Verify each of the listed seasons includes a playoff appearance.",
    )


async def verify_previous_assistant_experience(evaluator: Evaluator, parent_node, ex: CoachExtraction) -> None:
    node = evaluator.add_parallel(
        id="previous_assistant_coaching_experience",
        desc="Verify prior assistant coaching experience at the same FCS institution",
        parent=parent_node,
        critical=True,
    )

    asst = ex.assistant or AssistantCoachInfo()
    fcs = ex.fcs_head or FCSHeadCoachInfo()
    appt = ex.appointment or AppointmentInfo()

    coach = _safe(appt.coach_name)
    asst_school = _safe(asst.institution)
    fcs_school = _safe(fcs.fcs_institution)
    asst_urls = _urls(asst.assistant_urls)

    # Reference presence
    evaluator.add_custom_node(
        result=len(asst_urls) > 0,
        id="reference_url_assistant_coaching",
        desc="Valid reference URL supporting the assistant coaching tenure",
        parent=node,
        critical=True,
    )

    # Assistant coach 2004–2006 at the institution
    leaf_asst_years = evaluator.add_leaf(
        id="assistant_coach_2004_2006",
        desc="Coach served as assistant coach at the same FCS institution from 2004 to 2006",
        parent=node,
        critical=True,
    )
    claim_asst_years = f"From 2004 to 2006, {coach} served as an assistant coach at {asst_school}."
    await evaluator.verify(
        claim=claim_asst_years,
        node=leaf_asst_years,
        sources=asst_urls,
        additional_instruction="Confirm role (assistant coach) and years 2004–2006 at the named institution.",
    )

    # Logical check: assistant institution matches the FCS head coach institution
    leaf_match_school = evaluator.add_leaf(
        id="assistant_same_institution_match",
        desc="Assistant institution matches the FCS head-coaching institution",
        parent=node,
        critical=True,
    )
    claim_match = f"The institutions '{asst_school}' and '{fcs_school}' refer to the same school."
    await evaluator.verify(
        claim=claim_match,
        node=leaf_match_school,
        additional_instruction=(
            "Treat reasonable variants and abbreviations as matches (e.g., 'Idaho' vs 'University of Idaho'). "
            "This is a simple logical/name-matching check; rely on equivalence of naming."
        ),
    )


async def verify_graduate_assistant_background(evaluator: Evaluator, parent_node, ex: CoachExtraction) -> None:
    node = evaluator.add_parallel(
        id="graduate_assistant_background",
        desc="Verify graduate assistant experience and education at Big Ten university",
        parent=parent_node,
        critical=True,
    )

    ga = ex.ga or GraduateAssistantInfo()
    appt = ex.appointment or AppointmentInfo()

    coach = _safe(appt.coach_name)
    ga_school = _safe(ga.big_ten_university)
    ga_urls = _urls(ga.ga_urls)

    # Reference presence
    evaluator.add_custom_node(
        result=len(ga_urls) > 0,
        id="reference_url_graduate_assistant",
        desc="Valid reference URL supporting the graduate assistant tenure and master's degree",
        parent=node,
        critical=True,
    )

    # GA years and role at the Big Ten university
    leaf_ga_years = evaluator.add_leaf(
        id="ga_1999_2001",
        desc="Coach served as graduate assistant from 1999 to 2001 at the stated Big Ten university",
        parent=node,
        critical=True,
    )
    claim_ga_years = f"From 1999 to 2001, {coach} served as a graduate assistant at {ga_school}."
    await evaluator.verify(
        claim=claim_ga_years,
        node=leaf_ga_years,
        sources=ga_urls,
        additional_instruction="Confirm GA role and the 1999–2001 timeframe at the named university.",
    )

    # Master's degree earned during that time
    leaf_masters = evaluator.add_leaf(
        id="ga_masters_degree",
        desc="Coach earned a master's degree during the GA tenure",
        parent=node,
        critical=True,
    )
    claim_masters = (
        f"While serving as a GA at {ga_school} (around 1999–2001), {coach} earned a master's degree."
    )
    await evaluator.verify(
        claim=claim_masters,
        node=leaf_masters,
        sources=ga_urls,
        additional_instruction="Confirm that a master's degree was earned at that university during or around the GA tenure.",
    )

    # Big Ten membership of the university
    leaf_bigten = evaluator.add_leaf(
        id="ga_big_ten_membership",
        desc="The GA university is a Big Ten Conference member",
        parent=node,
        critical=True,
    )
    claim_bigten = f"{ga_school} is (or was at that time) a member of the Big Ten Conference."
    await evaluator.verify(
        claim=claim_bigten,
        node=leaf_bigten,
        sources=ga_urls,
        additional_instruction="Verify that the named university is a Big Ten Conference member (explicit statement preferred).",
    )


async def verify_head_coach_rose_bowl(evaluator: Evaluator, parent_node, ex: CoachExtraction) -> None:
    node = evaluator.add_parallel(
        id="head_coach_rose_bowl_achievement",
        desc="Verify the coach served under a head coach with three Rose Bowl wins",
        parent=parent_node,
        critical=True,
    )

    mentor = ex.mentor or MentorCoachInfo()
    ga = ex.ga or GraduateAssistantInfo()
    appt = ex.appointment or AppointmentInfo()

    coach = _safe(appt.coach_name)
    ga_school = _safe(ga.big_ten_university)
    mentor_name = _safe(mentor.mentor_name)
    rb_urls = _combine_urls(mentor.rose_bowl_urls, ga.ga_urls)

    # Reference presence (allow either explicit rose bowl URLs or GA URLs if they contain the info)
    evaluator.add_custom_node(
        result=len(rb_urls) > 0,
        id="reference_url_rose_bowl",
        desc="Valid reference URL supporting the Rose Bowl wins of the head coach they served under",
        parent=node,
        critical=True,
    )

    # Single composite check per rubric: served under mentor who won three Rose Bowls
    leaf_under_three_rb = evaluator.add_leaf(
        id="under_three_rose_bowl_coach",
        desc="Coach served as graduate assistant under a head coach who won three Rose Bowl games",
        parent=node,
        critical=True,
    )
    claim_under_three = (
        f"While at {ga_school}, {coach} served under head coach {mentor_name}, who won three Rose Bowl games."
    )
    await evaluator.verify(
        claim=claim_under_three,
        node=leaf_under_three_rb,
        sources=rb_urls,
        additional_instruction=(
            "Two parts must be supported: (1) the coach served under the named head coach at the GA school; "
            "(2) that head coach has three Rose Bowl wins. A single page containing both is ideal; "
            "however, verification can rely on any of the provided pages that clearly support the full statement."
        ),
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

    # Extract structured information from the answer
    extracted: CoachExtraction = await evaluator.extract(
        prompt=prompt_extract_coach_info(),
        template_class=CoachExtraction,
        extraction_name="coach_extraction",
    )

    # Top-level critical aggregator representing the overall identification
    coach_id_node = evaluator.add_parallel(
        id="coach_identification",
        desc="Correctly identify the college football head coach meeting all specified criteria",
        parent=root,
        critical=True,  # Enforce 'all criteria must be met'
    )

    # Build verification subtrees
    await verify_current_fbs_position(evaluator, coach_id_node, extracted)
    await verify_fcs_record(evaluator, coach_id_node, extracted)
    await verify_previous_assistant_experience(evaluator, coach_id_node, extracted)
    await verify_graduate_assistant_background(evaluator, coach_id_node, extracted)
    await verify_head_coach_rose_bowl(evaluator, coach_id_node, extracted)

    # Return summary
    return evaluator.get_summary()