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
TASK_ID = "edu_leadership_feb2026"
TASK_DESCRIPTION = (
    "In February 2026, two significant educational leadership appointments were announced in the United States: "
    "one for a head football coach position at an Ivy League university and another for an interim superintendent "
    "position at a Texas school district.\n\n"
    "Identify both individuals appointed to these positions and provide the following detailed information about each:\n\n"
    "For the football coach appointment:\n"
    "- The individual's full name\n"
    "- The exact date the appointment was announced\n"
    "- The institution where the individual previously served as head coach immediately before this appointment\n"
    "- The individual's win-loss record during the 2025 season at that previous institution\n"
    "- Any national coaching awards or honors the individual received for the 2025 season\n"
    "- The individual's previous employment at the same institution (if applicable), including the specific role title and years of service\n"
    "- The individual's position number in the chronological history of head coaches for this program\n\n"
    "For the superintendent appointment:\n"
    "- The individual's full name (including any professional titles such as Dr.)\n"
    "- The exact date the individual was appointed to the position\n"
    "- The total number of years of experience the individual has in public education\n"
    "- The names of all previous Texas school districts where the individual served as superintendent (not interim)\n"
    "- Any state-level superintendent recognition or awards the individual has received, including the specific year\n"
    "- Any previous interim superintendent position the individual held at another Texas school district, including the duration of that service\n"
    "- The position title the individual held immediately before this appointment\n\n"
    "Provide reference URLs that support each piece of information."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CoachExtraction(BaseModel):
    name: Optional[str] = None
    announcement_date: Optional[str] = None
    position_title: Optional[str] = None
    previous_institution: Optional[str] = None
    record_2025: Optional[str] = None
    awards_2025: List[str] = Field(default_factory=list)
    previous_yale_role_title: Optional[str] = None
    previous_yale_years: Optional[str] = None
    program_history_position: Optional[str] = None

    urls_identity: List[str] = Field(default_factory=list)
    urls_announcement_date: List[str] = Field(default_factory=list)
    urls_position_title: List[str] = Field(default_factory=list)
    urls_previous_institution: List[str] = Field(default_factory=list)
    urls_record_2025: List[str] = Field(default_factory=list)
    urls_awards_2025: List[str] = Field(default_factory=list)
    urls_previous_yale_experience: List[str] = Field(default_factory=list)
    urls_program_history_position: List[str] = Field(default_factory=list)
    urls_any: List[str] = Field(default_factory=list)


class SuperintendentExtraction(BaseModel):
    name: Optional[str] = None
    appointment_date: Optional[str] = None
    position_type: Optional[str] = None  # e.g., "Interim Superintendent of Judson ISD"
    total_experience_years: Optional[str] = None
    previous_superintendent_districts: List[str] = Field(default_factory=list)  # e.g., ["Harlandale ISD", "Victoria ISD"]
    saisd_interim_duration: Optional[str] = None  # e.g., "seven months"
    state_recognition: Optional[str] = None  # e.g., "2017 TASB Superintendent of the Year Finalist"
    most_recent_position: Optional[str] = None  # e.g., "ESC-20 Senior Field Service Agent"

    urls_identity: List[str] = Field(default_factory=list)
    urls_appointment_date: List[str] = Field(default_factory=list)
    urls_position_type: List[str] = Field(default_factory=list)
    urls_total_experience: List[str] = Field(default_factory=list)
    urls_prev_superintendent_roles: List[str] = Field(default_factory=list)
    urls_saisd_interim: List[str] = Field(default_factory=list)
    urls_state_recognition: List[str] = Field(default_factory=list)
    urls_recent_position: List[str] = Field(default_factory=list)
    urls_any: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    yale_coach: Optional[CoachExtraction] = None
    judson_superintendent: Optional[SuperintendentExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information for two February 2026 appointments mentioned in the answer:
    1) The appointment of a head football coach at an Ivy League university (Yale Football).
    2) The appointment of an interim superintendent at a Texas school district (Judson ISD).

    For the football coach (object: yale_coach):
      - name: full name of the appointed individual.
      - announcement_date: the exact date the appointment was announced (as written in the answer).
      - position_title: the official position title at the new institution (e.g., "Joel E. Smilow '54 Head Coach of Yale Football").
      - previous_institution: the institution where the individual previously served as head coach immediately before this appointment.
      - record_2025: the win-loss record during the 2025 season at that previous institution (e.g., "12-1").
      - awards_2025: list of any national coaching awards/honors for the 2025 season (e.g., "Eddie Robinson Award").
      - previous_yale_role_title: the specific previous role title at Yale (if applicable).
      - previous_yale_years: the years of service at Yale (if applicable), e.g., "2012-2022".
      - program_history_position: position number in the chronological history of head coaches for the program (e.g., "35th").
      - For each of the following, extract all URLs explicitly present in the answer that support the specific claim; keep them in the corresponding arrays:
        urls_identity, urls_announcement_date, urls_position_title, urls_previous_institution, urls_record_2025,
        urls_awards_2025, urls_previous_yale_experience, urls_program_history_position, urls_any.
      Notes:
        * Only extract URLs that are explicitly present in the answer text. Do not invent URLs.
        * If the answer gives sources collectively, assign them to urls_any. If a source clearly maps to a specific claim, also list it under that specific urls_* field.
        * If a field is not present in the answer, set it to null (or an empty list for arrays).

    For the superintendent (object: judson_superintendent):
      - name: full name including professional titles (e.g., "Dr. Robert Jaklich").
      - appointment_date: the exact date the individual was appointed.
      - position_type: the position type/title (e.g., "Interim Superintendent of Judson ISD").
      - total_experience_years: total number of years of experience in public education (as written; keep the phrase like "more than 40 years" if used).
      - previous_superintendent_districts: list all previous Texas school districts where the individual served as superintendent (not interim).
      - saisd_interim_duration: duration of any prior Interim Superintendent service at another district (e.g., "seven months" at San Antonio ISD).
      - state_recognition: any state-level superintendent recognition or awards with year (e.g., "2017 TASB Superintendent of the Year Finalist").
      - most_recent_position: the position held immediately before this appointment (e.g., "ESC-20 Senior Field Service Agent").
      - For each of the following, extract all URLs explicitly present in the answer that support the specific claim; keep them in the corresponding arrays:
        urls_identity, urls_appointment_date, urls_position_type, urls_total_experience, urls_prev_superintendent_roles,
        urls_saisd_interim, urls_state_recognition, urls_recent_position, urls_any.
      Notes:
        * Only extract URLs that are explicitly present in the answer text. Do not invent URLs.
        * If the answer gives sources collectively, assign them to urls_any. If a source clearly maps to a specific claim, also list it under that specific urls_* field.

    Return a single JSON object with fields:
      - yale_coach: object as defined above (or null if missing).
      - judson_superintendent: object as defined above (or null if missing).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            key = u.strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(key)
    return merged


def _ensure_sources(preferred: List[str], fallback: List[str]) -> List[str]:
    if preferred and len(preferred) > 0:
        return preferred
    return fallback


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_yale_coach(evaluator: Evaluator, parent_node, coach: Optional[CoachExtraction]) -> None:
    # Parent node for Yale coach appointment (parallel)
    yale_node = evaluator.add_parallel(
        id="yale_coach",
        desc="Yale Football Head Coach Appointment: February 2026",
        parent=parent_node,
        critical=False
    )

    # Prepare URL pools
    coach = coach or CoachExtraction()
    all_urls = _merge_urls(
        coach.urls_identity,
        coach.urls_announcement_date,
        coach.urls_position_title,
        coach.urls_previous_institution,
        coach.urls_record_2025,
        coach.urls_awards_2025,
        coach.urls_previous_yale_experience,
        coach.urls_program_history_position,
        coach.urls_any,
    )

    # 1) Individual Identity (critical) – use simple verification against the answer text
    identity_leaf = evaluator.add_leaf(
        id="yale_identity",
        desc="Individual Identity: The appointed individual is Kevin Cahill",
        parent=yale_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer identifies the individual appointed as Yale's head football coach as Kevin Cahill.",
        node=identity_leaf,
        sources=None,
        additional_instruction="Check the answer text to see if it clearly names Kevin Cahill as the appointee."
    )

    # 2) Appointment Details (parent non-critical due to framework critical-child rule)
    app_details = evaluator.add_parallel(
        id="yale_appointment_details",
        desc="Appointment Details",
        parent=yale_node,
        critical=False
    )

    # 2a) Announcement Date (critical)
    ann_date_leaf = evaluator.add_leaf(
        id="yale_announcement_date",
        desc="Announcement Date: February 23, 2026",
        parent=app_details,
        critical=True
    )
    await evaluator.verify(
        claim="The coaching appointment announcement was made on February 23, 2026.",
        node=ann_date_leaf,
        sources=_ensure_sources(coach.urls_announcement_date, all_urls),
        additional_instruction="Accept equivalent formats (e.g., Feb. 23, 2026 or 2/23/2026). Confirm it's the announcement date for Yale's head football coach appointment."
    )

    # 2b) Position Title (non-critical)
    pos_title_leaf = evaluator.add_leaf(
        id="yale_position_title",
        desc="Position Title: Joel E. Smilow '54 Head Coach of Yale Football",
        parent=app_details,
        critical=False
    )
    await evaluator.verify(
        claim="The position title is Joel E. Smilow '54 Head Coach of Yale Football.",
        node=pos_title_leaf,
        sources=_ensure_sources(coach.urls_position_title, all_urls),
        additional_instruction="Verify the official title language on Yale's announcement or official sources."
    )

    # 3) Previous Institution Background (critical)
    prev_inst_node = evaluator.add_parallel(
        id="yale_previous_institution",
        desc="Previous Institution Background",
        parent=yale_node,
        critical=True
    )
    # 3a) Institution Name (critical)
    prev_inst_leaf = evaluator.add_leaf(
        id="yale_prev_inst_name",
        desc="Came from Lehigh University as head coach",
        parent=prev_inst_node,
        critical=True
    )
    await evaluator.verify(
        claim="Immediately before this appointment, he served as head coach at Lehigh University.",
        node=prev_inst_leaf,
        sources=_ensure_sources(coach.urls_previous_institution, all_urls),
        additional_instruction="Confirm he held the head coach title at Lehigh directly prior to Yale."
    )

    # 3b) 2025 Season Record (critical)
    record_leaf = evaluator.add_leaf(
        id="yale_2025_record",
        desc="2025 Season Record: 12-1",
        parent=prev_inst_node,
        critical=True
    )
    await evaluator.verify(
        claim="During the 2025 season at Lehigh University, his team had a 12-1 record.",
        node=record_leaf,
        sources=_ensure_sources(coach.urls_record_2025, all_urls),
        additional_instruction="Allow en-dash or hyphen variants (12–1 or 12-1). Count should reflect the 2025 season (including playoffs if the cited source states so)."
    )

    # 4) Achievements and History (critical)
    achieve_node = evaluator.add_parallel(
        id="yale_achievements_history",
        desc="Achievements and Program History",
        parent=yale_node,
        critical=True
    )

    # 4a) National Coaching Award (critical)
    award_leaf = evaluator.add_leaf(
        id="yale_eddie_robinson_award",
        desc="Won the Eddie Robinson Award (National FCS Coach of the Year) for 2025",
        parent=achieve_node,
        critical=True
    )
    await evaluator.verify(
        claim="He won the Eddie Robinson Award (National FCS Coach of the Year) for the 2025 season.",
        node=award_leaf,
        sources=_ensure_sources(coach.urls_awards_2025, all_urls),
        additional_instruction="Sometimes referred to as the 'Stats Perform Eddie Robinson Award'. Confirm the award year is 2025."
    )

    # 4b) Previous Yale Experience (critical)
    prior_yale_leaf = evaluator.add_leaf(
        id="yale_prior_experience",
        desc="Previously at Yale (2012-2022) as Associate Head Coach and Offensive Coordinator",
        parent=achieve_node,
        critical=True
    )
    await evaluator.verify(
        claim="He previously worked at Yale from 2012 to 2022 as Associate Head Coach and Offensive Coordinator.",
        node=prior_yale_leaf,
        sources=_ensure_sources(coach.urls_previous_yale_experience, all_urls),
        additional_instruction="Minor variations in capitalization or punctuation are acceptable as long as the roles and years are clear."
    )

    # 4c) Program History Position (critical)
    program_pos_leaf = evaluator.add_leaf(
        id="yale_program_history_position",
        desc="He is the 35th head coach in Yale football program history",
        parent=achieve_node,
        critical=True
    )
    await evaluator.verify(
        claim="He is the 35th head coach in Yale football program history.",
        node=program_pos_leaf,
        sources=_ensure_sources(coach.urls_program_history_position, all_urls),
        additional_instruction="Confirm any official count of head coaches; minor formatting differences (e.g., '35th') are acceptable."
    )

    # 5) Reference URLs presence (critical)
    # Require that each critical claim above has at least one associated URL provided in the answer
    critical_sources_ok = all([
        bool(coach.urls_identity),
        bool(coach.urls_announcement_date),
        bool(coach.urls_previous_institution),
        bool(coach.urls_record_2025),
        bool(coach.urls_awards_2025),
        bool(coach.urls_previous_yale_experience),
        bool(coach.urls_program_history_position),
    ])
    evaluator.add_custom_node(
        result=critical_sources_ok,
        id="yale_reference_urls",
        desc="Reference URLs: At least one URL is provided for each critical claim about the Yale appointment",
        parent=yale_node,
        critical=True
    )


async def verify_judson_superintendent(evaluator: Evaluator, parent_node, sup: Optional[SuperintendentExtraction]) -> None:
    # Parent node for Judson ISD appointment (parallel)
    judson_node = evaluator.add_parallel(
        id="judson_superintendent",
        desc="Judson ISD Interim Superintendent Appointment: February 2026",
        parent=parent_node,
        critical=False
    )

    # Prepare URL pools
    sup = sup or SuperintendentExtraction()
    all_urls = _merge_urls(
        sup.urls_identity,
        sup.urls_appointment_date,
        sup.urls_position_type,
        sup.urls_total_experience,
        sup.urls_prev_superintendent_roles,
        sup.urls_saisd_interim,
        sup.urls_state_recognition,
        sup.urls_recent_position,
        sup.urls_any,
    )

    # 1) Individual Identity (critical) – use simple verification against the answer text
    identity_leaf = evaluator.add_leaf(
        id="judson_identity",
        desc="Individual Identity: The appointee is Dr. Robert Jaklich",
        parent=judson_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer identifies the appointed Interim Superintendent of Judson ISD as Dr. Robert Jaklich.",
        node=identity_leaf,
        sources=None,
        additional_instruction="Check the answer text to see if it clearly names Dr. Robert Jaklich as the appointee."
    )

    # 2) Appointment Details (critical, parallel)
    app_details = evaluator.add_parallel(
        id="judson_appointment_details",
        desc="Appointment Details",
        parent=judson_node,
        critical=True
    )
    # 2a) Appointment Date (critical)
    app_date_leaf = evaluator.add_leaf(
        id="judson_appointment_date",
        desc="Appointment Date: February 16, 2026",
        parent=app_details,
        critical=True
    )
    await evaluator.verify(
        claim="The appointment date was February 16, 2026.",
        node=app_date_leaf,
        sources=_ensure_sources(sup.urls_appointment_date, all_urls),
        additional_instruction="Accept equivalent formats (e.g., Feb. 16, 2026 or 2/16/2026). Confirm it's the date for Judson ISD's interim superintendent appointment."
    )

    # 2b) Position Type (critical)
    pos_type_leaf = evaluator.add_leaf(
        id="judson_position_type",
        desc="Position Type: Interim Superintendent of Judson ISD",
        parent=app_details,
        critical=True
    )
    await evaluator.verify(
        claim="The position is Interim Superintendent of Judson ISD.",
        node=pos_type_leaf,
        sources=_ensure_sources(sup.urls_position_type, all_urls),
        additional_instruction="Confirm the appointment is specifically an Interim Superintendent position at Judson ISD."
    )

    # 3) Professional Experience (critical, parallel)
    prof_exp_node = evaluator.add_parallel(
        id="judson_professional_experience",
        desc="Professional Experience and Prior Superintendent Roles",
        parent=judson_node,
        critical=True
    )
    # 3a) Total Experience (critical)
    experience_leaf = evaluator.add_leaf(
        id="judson_total_experience",
        desc="More than 40 years of experience in public education",
        parent=prof_exp_node,
        critical=True
    )
    await evaluator.verify(
        claim="He has more than 40 years of experience in public education.",
        node=experience_leaf,
        sources=_ensure_sources(sup.urls_total_experience, all_urls),
        additional_instruction="Phrasing like 'over 40 years' or 'more than 40 years' is acceptable."
    )

    # 3b) Previous Superintendent Roles (critical)
    prev_sup_leaf = evaluator.add_leaf(
        id="judson_prev_superintendent_roles",
        desc="Previously served as superintendent at Harlandale ISD and Victoria ISD (not interim)",
        parent=prof_exp_node,
        critical=True
    )
    await evaluator.verify(
        claim="He previously served as superintendent at both Harlandale ISD and Victoria ISD (not in an interim capacity).",
        node=prev_sup_leaf,
        sources=_ensure_sources(sup.urls_prev_superintendent_roles, all_urls),
        additional_instruction="Confirm both districts and that the roles were full superintendent roles, not interim."
    )

    # 3c) SAISD Interim Role (critical)
    saisd_leaf = evaluator.add_leaf(
        id="judson_saisd_interim",
        desc="Served as Interim Superintendent of San Antonio ISD for seven months",
        parent=prof_exp_node,
        critical=True
    )
    await evaluator.verify(
        claim="He served as Interim Superintendent of San Antonio ISD for seven months.",
        node=saisd_leaf,
        sources=_ensure_sources(sup.urls_saisd_interim, all_urls),
        additional_instruction="Look for explicit mention of 'Interim Superintendent' at San Antonio ISD and a duration of seven months."
    )

    # 4) Recognition and Recent Position (critical, parallel)
    recog_recent_node = evaluator.add_parallel(
        id="judson_recognition_recent",
        desc="State recognition and most recent position prior to appointment",
        parent=judson_node,
        critical=True
    )
    # 4a) State Recognition (critical)
    recognition_leaf = evaluator.add_leaf(
        id="judson_state_recognition",
        desc="2017 TASB Superintendent of the Year Finalist",
        parent=recog_recent_node,
        critical=True
    )
    await evaluator.verify(
        claim="He was recognized as a 2017 TASB (Texas Association of School Boards) Superintendent of the Year Finalist.",
        node=recognition_leaf,
        sources=_ensure_sources(sup.urls_state_recognition, all_urls),
        additional_instruction="Confirm the year 2017 and that the recognition is a TASB Superintendent of the Year Finalist."
    )

    # 4b) Most Recent Position (critical)
    recent_pos_leaf = evaluator.add_leaf(
        id="judson_recent_position",
        desc="Immediately before this appointment, served as ESC-20 Senior Field Service Agent",
        parent=recog_recent_node,
        critical=True
    )
    await evaluator.verify(
        claim="Immediately before the Judson ISD appointment, he served as ESC-20 Senior Field Service Agent.",
        node=recent_pos_leaf,
        sources=_ensure_sources(sup.urls_recent_position, all_urls),
        additional_instruction="ESC-20 refers to Education Service Center, Region 20. Confirm the exact role title."
    )

    # 5) Reference URLs presence (critical)
    critical_sources_ok = all([
        bool(sup.urls_identity),
        bool(sup.urls_appointment_date),
        bool(sup.urls_position_type),
        bool(sup.urls_total_experience),
        bool(sup.urls_prev_superintendent_roles),
        bool(sup.urls_saisd_interim),
        bool(sup.urls_state_recognition),
        bool(sup.urls_recent_position),
    ])
    evaluator.add_custom_node(
        result=critical_sources_ok,
        id="judson_reference_urls",
        desc="Reference URLs: At least one URL is provided for each critical claim about the Judson ISD appointment",
        parent=judson_node,
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
    # Initialize evaluator and root node
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="appointments_extraction"
    )

    # Add ground truth info for transparency (not used to gate verification, only for report)
    evaluator.add_ground_truth({
        "Yale Football Head Coach": {
            "expected_identity": "Kevin Cahill",
            "expected_announcement_date": "February 23, 2026",
            "expected_position_title": "Joel E. Smilow '54 Head Coach of Yale Football",
            "expected_previous_institution": "Lehigh University (Head Coach)",
            "expected_2025_record": "12-1",
            "expected_2025_award": "Eddie Robinson Award (National FCS Coach of the Year)",
            "expected_previous_yale_experience": "Associate Head Coach and Offensive Coordinator (2012-2022)",
            "expected_program_history_position": "35th"
        },
        "Judson ISD Interim Superintendent": {
            "expected_identity": "Dr. Robert Jaklich",
            "expected_appointment_date": "February 16, 2026",
            "expected_position_type": "Interim Superintendent of Judson ISD",
            "expected_total_experience": "More than 40 years in public education",
            "expected_previous_superintendent_districts": ["Harlandale ISD", "Victoria ISD"],
            "expected_saisd_interim": "Interim Superintendent of SAISD for seven months",
            "expected_state_recognition": "2017 TASB Superintendent of the Year Finalist",
            "expected_recent_position": "ESC-20 Senior Field Service Agent"
        }
    })

    # Build top-level task node
    top_node = evaluator.add_parallel(
        id="recent_edu_leadership_appointments",
        desc="Identify two recent educational/athletic leadership appointments in February 2026 and verify specific details about each appointee",
        parent=root,
        critical=False
    )

    # Verify Yale coach
    await verify_yale_coach(evaluator, top_node, extraction.yale_coach)

    # Verify Judson ISD superintendent
    await verify_judson_superintendent(evaluator, top_node, extraction.judson_superintendent)

    # Return evaluation summary
    return evaluator.get_summary()