import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "athletic_director_identification"
TASK_DESCRIPTION = """
Identify the athletic director at an NCAA Division I institution in the northeastern United States who meets ALL of the following criteria:
(1) Currently holds the position of Director of Athletics or Athletic Director;
(2) The institution is classified as NCAA Division I;
(3) Was appointed to the athletic director position between June 2020 and June 2022;
(4) Previously served as a head coach at the same institution where they currently serve as athletic director;
(5) Earned a bachelor's degree from the same institution where they currently serve as athletic director;
(6) Won conference or regional championships during their tenure as a head coach;
(7) The institution competes in an athletic conference that sponsors ice hockey as a varsity sport;
(8) Has head coaching experience specifically in ice hockey;
(9) Served as head coach at the institution for at least 8 years before becoming athletic director;
(10) Was promoted internally to the athletic director position rather than hired externally from another institution;
(11) The institution is located in a major metropolitan area with a population exceeding 500,000;
(12) The appointment to athletic director was announced during the spring or summer months (April through August);
(13) Was a student-athlete at the same institution as an undergraduate;
(14) The institution is classified as a research university.
Provide the name of the athletic director, the institution, and the specific date of their appointment.
"""


class ADSources(BaseModel):
    """URLs cited in the answer that support each criterion."""
    position_title: List[str] = Field(default_factory=list)
    ncaa_division: List[str] = Field(default_factory=list)
    geographic_region: List[str] = Field(default_factory=list)
    appointment_date: List[str] = Field(default_factory=list)
    previous_coaching_same_inst: List[str] = Field(default_factory=list)
    alumni_status: List[str] = Field(default_factory=list)
    coaching_championships: List[str] = Field(default_factory=list)
    conference_hockey_sponsorship: List[str] = Field(default_factory=list)
    ice_hockey_coaching: List[str] = Field(default_factory=list)
    coaching_tenure_duration: List[str] = Field(default_factory=list)
    internal_promotion: List[str] = Field(default_factory=list)
    metropolitan_location: List[str] = Field(default_factory=list)
    appointment_season: List[str] = Field(default_factory=list)
    student_athlete_background: List[str] = Field(default_factory=list)
    research_university_classification: List[str] = Field(default_factory=list)


class ADCandidate(BaseModel):
    """Structured extraction for the identified athletic director and criteria."""
    ad_name: Optional[str] = None
    institution: Optional[str] = None
    position_title: Optional[str] = None
    appointment_date: Optional[str] = None  # e.g., "2021-06-15" or "June 15, 2021"
    ncaa_division: Optional[str] = None
    region: Optional[str] = None  # e.g., state or "northeastern US"
    previous_head_coach_same_inst: Optional[bool] = None
    alumni_bachelors_same_inst: Optional[bool] = None
    coaching_championships_desc: Optional[str] = None
    conference_name: Optional[str] = None
    conference_sponsors_ice_hockey: Optional[bool] = None
    head_coaching_sport: Optional[str] = None  # expect "ice hockey"
    head_coaching_years_at_inst: Optional[str] = None  # e.g., "10 years" or "2011–2021"
    internal_promotion: Optional[bool] = None
    metro_area: Optional[str] = None
    metro_population: Optional[str] = None  # e.g., "4,900,000"
    appointment_month: Optional[str] = None  # e.g., "June" or "06"
    student_athlete_same_inst: Optional[bool] = None
    research_university_classification: Optional[str] = None  # e.g., "R1"
    sources: ADSources = ADSources()


def prompt_extract_candidate() -> str:
    return """
    Extract the SINGLE athletic director candidate identified in the answer who satisfies the task.
    Return a JSON object with the following fields (use null if missing). Also extract supporting URLs explicitly cited in the answer for each criterion.

    Core identity:
    - ad_name: The full name of the athletic director.
    - institution: The institution's name.
    - position_title: The exact position title (e.g., "Director of Athletics", "Athletic Director").
    - appointment_date: The specific date of appointment (any reasonable format).
    - appointment_month: The month of appointment announcement (e.g., "April", "05", "June").

    Classification & geography:
    - ncaa_division: e.g., "NCAA Division I".
    - region: State or region information indicating northeastern US.

    Coaching & alumni:
    - previous_head_coach_same_inst: true/false if they previously served as HEAD coach at the SAME institution.
    - head_coaching_sport: the sport coached as head coach (e.g., "ice hockey").
    - head_coaching_years_at_inst: string describing duration (e.g., "10 years", "2011–2021").
    - alumni_bachelors_same_inst: true/false if they earned a bachelor's degree from the same institution.
    - student_athlete_same_inst: true/false if they were a student-athlete as an undergraduate at the same institution.
    - coaching_championships_desc: brief description of conference or regional championships won as head coach.

    Conference & research:
    - conference_name: the athletic conference the institution competes in (for ice hockey).
    - conference_sponsors_ice_hockey: true/false if that conference sponsors varsity ice hockey.
    - research_university_classification: e.g., "R1", "R2", "Research University".

    Promotion & metro:
    - internal_promotion: true/false if promoted internally (not hired from another institution).
    - metro_area: name of the major metropolitan area (e.g., "Boston", "New York City").
    - metro_population: approximate population number for the metro area (string ok).

    sources: For each of the following keys, list ONLY the URLs explicitly cited in the answer that support the criterion. If no URL is cited for that criterion, return an empty list.
    - position_title
    - ncaa_division
    - geographic_region
    - appointment_date
    - previous_coaching_same_inst
    - alumni_status
    - coaching_championships
    - conference_hockey_sponsorship
    - ice_hockey_coaching
    - coaching_tenure_duration
    - internal_promotion
    - metropolitan_location
    - appointment_season
    - student_athlete_background
    - research_university_classification

    STRICT RULES:
    - Extract ONLY information explicitly present in the answer.
    - For URLs, include only full URLs (plain or markdown).
    - If a required field is missing, set it to null.
    """


def _require_sources_instruction(extra: str = "") -> str:
    base = (
        "You must judge support ONLY from the provided URL(s). If no URL is provided for this criterion, "
        "conclude the claim is NOT supported. Do not rely on the answer text alone."
    )
    if extra:
        return f"{base} {extra}"
    return base


async def _verify_candidate(evaluator: Evaluator, parent_node, c: ADCandidate) -> None:
    """
    Build verification leaves for all criteria under a critical parallel node.
    """
    # Existence checks (provided fields)
    evaluator.add_custom_node(
        result=bool(c.ad_name and c.ad_name.strip()),
        id="AD_Name_Provided",
        desc="The solution provides the name of the athletic director",
        parent=parent_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(c.institution and c.institution.strip()),
        id="Institution_Name_Provided",
        desc="The solution provides the name of the institution",
        parent=parent_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(c.appointment_date and c.appointment_date.strip()),
        id="Appointment_Date_Provided",
        desc="The solution provides the specific date of the appointment",
        parent=parent_node,
        critical=True
    )

    # Position Title
    node_position = evaluator.add_leaf(
        id="Position_Title",
        desc="The individual holds the position of Director of Athletics or Athletic Director",
        parent=parent_node,
        critical=True,
    )
    claim_position = (
        f"{c.ad_name or 'The individual'} holds the position of Director of Athletics (Athletic Director) "
        f"at {c.institution or 'the institution'}."
    )
    await evaluator.verify(
        claim=claim_position,
        node=node_position,
        sources=c.sources.position_title,
        additional_instruction=_require_sources_instruction(
            "Allow reasonable synonyms such as 'Director of Athletics', 'Athletic Director', or 'AD'."
        ),
    )

    # NCAA Division I
    node_division = evaluator.add_leaf(
        id="NCAA_Division",
        desc="The institution is classified as NCAA Division I",
        parent=parent_node,
        critical=True,
    )
    claim_division = f"{c.institution or 'The institution'} is classified as NCAA Division I."
    await evaluator.verify(
        claim=claim_division,
        node=node_division,
        sources=c.sources.ncaa_division,
        additional_instruction=_require_sources_instruction("Accept 'Division I', 'D-I', or 'NCAA Division I'."),
    )

    # Geographic Region (Northeastern US)
    node_region = evaluator.add_leaf(
        id="Geographic_Region",
        desc="The institution is located in the northeastern United States",
        parent=parent_node,
        critical=True,
    )
    claim_region = f"{c.institution or 'The institution'} is located in the northeastern United States."
    await evaluator.verify(
        claim=claim_region,
        node=node_region,
        sources=c.sources.geographic_region,
        additional_instruction=_require_sources_instruction(
            "Consider the northeastern US to include states like CT, ME, MA, NH, NJ, NY, PA, RI, VT (and commonly DC). "
            "Accept if the institution is clearly in one of these states or metro areas."
        ),
    )

    # Appointment timeframe (June 2020 – June 2022)
    node_timeframe = evaluator.add_leaf(
        id="Appointment_Timeframe",
        desc="The athletic director was appointed to the position between June 2020 and June 2022",
        parent=parent_node,
        critical=True,
    )
    claim_timeframe = (
        f"{c.ad_name or 'The individual'} was appointed as athletic director on {c.appointment_date or 'the given date'}, "
        f"and this date falls between June 1, 2020 and June 30, 2022."
    )
    await evaluator.verify(
        claim=claim_timeframe,
        node=node_timeframe,
        sources=c.sources.appointment_date,
        additional_instruction=_require_sources_instruction(
            "Verify the appointment announcement date is within 2020-06-01 to 2022-06-30 inclusive."
        ),
    )

    # Previous Coaching at same institution (head coach)
    node_prev_coach = evaluator.add_leaf(
        id="Previous_Coaching_Same_Institution",
        desc="The athletic director previously served as a head coach at the same institution",
        parent=parent_node,
        critical=True,
    )
    claim_prev_coach = (
        f"{c.ad_name or 'The individual'} previously served as a HEAD coach at {c.institution or 'the institution'}."
    )
    await evaluator.verify(
        claim=claim_prev_coach,
        node=node_prev_coach,
        sources=c.sources.previous_coaching_same_inst,
        additional_instruction=_require_sources_instruction("It must be HEAD coaching, not assistant."),
    )

    # Alumni bachelor's degree from same institution
    node_alumni = evaluator.add_leaf(
        id="Alumni_Status",
        desc="The athletic director earned a bachelor's degree from the same institution where they currently serve",
        parent=parent_node,
        critical=True,
    )
    claim_alumni = (
        f"{c.ad_name or 'The individual'} earned a bachelor's degree from {c.institution or 'the institution'}."
    )
    await evaluator.verify(
        claim=claim_alumni,
        node=node_alumni,
        sources=c.sources.alumni_status,
        additional_instruction=_require_sources_instruction("Bachelor's or equivalent undergraduate degree is acceptable."),
    )

    # Coaching championships (conference or regional)
    node_champs = evaluator.add_leaf(
        id="Coaching_Championships",
        desc="The athletic director won conference or regional championships as a head coach",
        parent=parent_node,
        critical=True,
    )
    claim_champs = (
        f"{c.ad_name or 'The individual'} won at least one conference or regional championship as a head coach."
    )
    await evaluator.verify(
        claim=claim_champs,
        node=node_champs,
        sources=c.sources.coaching_championships,
        additional_instruction=_require_sources_instruction(
            "Accept conference tournament championships (e.g., Hockey East Tournament) or regional titles recognized in NCAA contexts."
        ),
    )

    # Conference sponsors ice hockey (varsity)
    node_conf_hockey = evaluator.add_leaf(
        id="Conference_Hockey_Sponsorship",
        desc="The institution competes in an athletic conference that sponsors ice hockey as a varsity sport",
        parent=parent_node,
        critical=True,
    )
    conf_name = c.conference_name or "the athletic conference"
    claim_conf_hockey = (
        f"{c.institution or 'The institution'} competes in {conf_name}, and this conference sponsors varsity ice hockey."
    )
    await evaluator.verify(
        claim=claim_conf_hockey,
        node=node_conf_hockey,
        sources=c.sources.conference_hockey_sponsorship,
        additional_instruction=_require_sources_instruction(
            "Ensure it is ICE hockey (men's or women's), not field hockey."
        ),
    )

    # Ice hockey head coaching experience
    node_hockey_coach = evaluator.add_leaf(
        id="Ice_Hockey_Coaching",
        desc="The athletic director has head coaching experience in ice hockey",
        parent=parent_node,
        critical=True,
    )
    claim_hockey_coach = f"{c.ad_name or 'The individual'} has HEAD coaching experience in ice hockey."
    await evaluator.verify(
        claim=claim_hockey_coach,
        node=node_hockey_coach,
        sources=c.sources.ice_hockey_coaching,
        additional_instruction=_require_sources_instruction("It must be head coach in ICE hockey."),
    )

    # Coaching tenure duration (>=8 years)
    node_tenure = evaluator.add_leaf(
        id="Coaching_Tenure_Duration",
        desc="The athletic director's previous head coaching tenure at the institution lasted at least 8 years",
        parent=parent_node,
        critical=True,
    )
    claim_tenure = (
        f"{c.ad_name or 'The individual'} served as head coach at {c.institution or 'the institution'} for at least 8 years."
    )
    await evaluator.verify(
        claim=claim_tenure,
        node=node_tenure,
        sources=c.sources.coaching_tenure_duration,
        additional_instruction=_require_sources_instruction("Verify tenure length is >= 8 years."),
    )

    # Internal promotion (not hired externally)
    node_internal = evaluator.add_leaf(
        id="Internal_Promotion",
        desc="The athletic director was promoted from within the institution rather than hired externally",
        parent=parent_node,
        critical=True,
    )
    claim_internal = (
        f"{c.ad_name or 'The individual'} was promoted internally to athletic director at {c.institution or 'the institution'}, "
        f"not hired from another institution."
    )
    await evaluator.verify(
        claim=claim_internal,
        node=node_internal,
        sources=c.sources.internal_promotion,
        additional_instruction=_require_sources_instruction(
            "Confirm the person moved from an internal role (e.g., head coach or admin at the same institution) directly into the AD role."
        ),
    )

    # Metropolitan location with population > 500,000
    node_metro = evaluator.add_leaf(
        id="Metropolitan_Location",
        desc="The institution is located in a major metropolitan area with a population exceeding 500,000",
        parent=parent_node,
        critical=True,
    )
    claim_metro = (
        f"{c.institution or 'The institution'} is located in the {c.metro_area or 'named'} metropolitan area "
        f"and that metro's population exceeds 500,000."
    )
    await evaluator.verify(
        claim=claim_metro,
        node=node_metro,
        sources=c.sources.metropolitan_location,
        additional_instruction=_require_sources_instruction(
            "Use reliable sources (e.g., census/Wikipedia/official regional data). Accept if metro population > 500,000."
        ),
    )

    # Appointment season (April–August)
    node_season = evaluator.add_leaf(
        id="Appointment_Season",
        desc="The appointment was announced during the spring or summer months (April through August)",
        parent=parent_node,
        critical=True,
    )
    claim_season = (
        f"The appointment announcement month ({c.appointment_month or 'given'}) is April, May, June, July, or August."
    )
    await evaluator.verify(
        claim=claim_season,
        node=node_season,
        sources=c.sources.appointment_season or c.sources.appointment_date,
        additional_instruction=_require_sources_instruction(
            "Check the announcement date month; accept April (4) through August (8), inclusive."
        ),
    )

    # Student-athlete background at same institution
    node_student = evaluator.add_leaf(
        id="Student_Athlete_Background",
        desc="The athletic director was a student-athlete at the same institution as an undergraduate",
        parent=parent_node,
        critical=True,
    )
    claim_student = (
        f"{c.ad_name or 'The individual'} was a student-athlete at {c.institution or 'the institution'} as an undergraduate."
    )
    await evaluator.verify(
        claim=claim_student,
        node=node_student,
        sources=c.sources.student_athlete_background,
        additional_instruction=_require_sources_instruction(
            "Look for evidence they competed as an undergraduate at the same institution."
        ),
    )

    # Research university classification
    node_research = evaluator.add_leaf(
        id="Research_University_Classification",
        desc="The institution is classified as a research university",
        parent=parent_node,
        critical=True,
    )
    claim_research = f"{c.institution or 'The institution'} is classified as a research university."
    await evaluator.verify(
        claim=claim_research,
        node=node_research,
        sources=c.sources.research_university_classification,
        additional_instruction=_require_sources_instruction(
            "Carnegie Classification (e.g., R1/R2) or equivalent recognized classification should support this."
        ),
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Entry point for evaluating the athletic director identification task.
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

    # Build critical root node (as per rubric) under evaluator's root
    main_node = evaluator.add_parallel(
        id="Athletic_Director_Identification",
        desc="Identify the athletic director at an NCAA Division I institution who meets all specified criteria",
        parent=root,
        critical=True
    )

    # Extract candidate data from the answer
    candidate: ADCandidate = await evaluator.extract(
        prompt=prompt_extract_candidate(),
        template_class=ADCandidate,
        extraction_name="ad_candidate_extraction"
    )

    # Perform verification for all criteria/leaves
    await _verify_candidate(evaluator, main_node, candidate)

    return evaluator.get_summary()