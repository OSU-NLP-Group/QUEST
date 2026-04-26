import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nfl_head_coach_wr_fl_oc"
TASK_DESCRIPTION = (
    "Identify the current NFL head coach (as of the 2024-2025 season) whose immediately previous position was serving "
    "as offensive coordinator for an NFL team based in Florida, and who played college football as a wide receiver at "
    "a university in California that was founded in the 19th century. Provide their complete career receiving "
    "statistics (total receptions, receiving yards, and touchdowns) from their college playing career, as well as "
    "their undergraduate degree information (degree type and major field of study)."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CoachProfileExtraction(BaseModel):
    # Identification
    coach_name: Optional[str] = None
    current_head_coach_team: Optional[str] = None
    profile_urls: List[str] = Field(default_factory=list)

    # Previous coordinator role
    previous_role_title: Optional[str] = None
    previous_role_team: Optional[str] = None
    previous_role_urls: List[str] = Field(default_factory=list)

    # College playing background
    college_university: Optional[str] = None
    college_playing_position: Optional[str] = None
    college_playing_urls: List[str] = Field(default_factory=list)

    # University info (location + founding year)
    university_location_state: Optional[str] = None
    university_founded_year: Optional[str] = None
    university_info_urls: List[str] = Field(default_factory=list)

    # Career receiving statistics (complete totals)
    total_receptions: Optional[str] = None
    total_receiving_yards: Optional[str] = None
    total_receiving_touchdowns: Optional[str] = None
    stats_urls: List[str] = Field(default_factory=list)

    # Undergraduate degree information
    degree_university: Optional[str] = None
    degree_type: Optional[str] = None
    major_field: Optional[str] = None
    degree_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_profile() -> str:
    return """
    Extract structured information about the single person identified as the coach in the answer. Return a single JSON object with the following fields. Extract exactly as written in the answer; do not infer.

    Required fields:
    - coach_name: The full name of the identified coach.
    - current_head_coach_team: The NFL team of which the person is the current head coach (as stated in the answer).
    - profile_urls: Array of URLs cited in the answer that support the person's identity and current head coach status.

    - previous_role_title: The title of the immediately previous position before becoming an NFL head coach (e.g., "offensive coordinator").
    - previous_role_team: The NFL team for that immediately previous role (e.g., "Tampa Bay Buccaneers").
    - previous_role_urls: Array of URLs cited that support the immediately previous role information.

    - college_university: The name of the university where the person played college football.
    - college_playing_position: The playing position stated for their college career (e.g., "wide receiver").
    - college_playing_urls: Array of URLs cited that support the college playing information.

    - university_location_state: The U.S. state of that university (as stated in the answer).
    - university_founded_year: The founding year of that university (as stated in the answer).
    - university_info_urls: Array of URLs cited that support the university location and founding year.

    - total_receptions: The total career receptions from the person's college playing career (as stated).
    - total_receiving_yards: The total career receiving yards from the college playing career (as stated).
    - total_receiving_touchdowns: The total career receiving touchdowns from the college playing career (as stated).
    - stats_urls: Array of URLs cited that support the college receiving statistics totals.

    - degree_university: The university from which the person earned their bachelor's degree.
    - degree_type: The bachelor's degree type (e.g., BA, BS, Bachelor’s in ...).
    - major_field: The major field of study for that bachelor’s degree.
    - degree_urls: Array of URLs cited that support the undergraduate degree info.

    Rules:
    - If a field is not explicitly mentioned in the answer, set it to null (for a string field) or [] for arrays.
    - For URLs, extract only explicit URLs present in the answer text (including markdown links).
    - Do not invent or infer URLs or values.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s is not None and str(s).strip() != "")


def _combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for u in lst or []:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    combined.append(uu)
    return combined


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: CoachProfileExtraction) -> None:
    # Root aggregation for this rubric (critical, parallel)
    profile_root = evaluator.add_parallel(
        id="Complete_Coach_Profile",
        desc="Verify the identified person meets all constraints and required outputs in the prompt.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Identification
    identification_node = evaluator.add_parallel(
        id="Identification",
        desc="The answer identifies a specific person as the coach.",
        parent=profile_root,
        critical=True
    )

    # 1.1) Coach name provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(extracted.coach_name),
        id="Coach_Name_Provided",
        desc="Coach's name is provided.",
        parent=identification_node,
        critical=True
    )

    # 1.2) Active Head Coach in 2024-2025 season (verification via URLs)
    active_hc_leaf = evaluator.add_leaf(
        id="Active_Head_Coach_2024_2025",
        desc="Person is a current NFL head coach during the 2024-2025 season.",
        parent=identification_node,
        critical=True
    )
    hc_claim = (
        f"As of the 2024-2025 NFL season, {_safe_name(extracted.coach_name)} is a current NFL head coach."
    )
    await evaluator.verify(
        claim=hc_claim,
        node=active_hc_leaf,
        sources=extracted.profile_urls,
        additional_instruction=(
            "Verify that the person is serving as an NFL head coach during the 2024-2025 season. "
            "Sources may include team press releases, official team pages, or reliable profiles. "
            "Minor wording differences are acceptable as long as the role and timeframe are clear."
        )
    )

    # 2) Previous coordinator role (OC in Florida)
    prev_role_node = evaluator.add_parallel(
        id="Previous_Coordinator_Role",
        desc="Immediately prior job matches the offensive coordinator / Florida-based NFL team constraint.",
        parent=profile_root,
        critical=True
    )

    # 2.1) Immediately previous was NFL OC
    prev_oc_leaf = evaluator.add_leaf(
        id="Immediately_Previous_Was_NFL_OC",
        desc="Person's immediately previous position was offensive coordinator for an NFL team.",
        parent=prev_role_node,
        critical=True
    )
    prev_role_claim = (
        f"Immediately prior to becoming an NFL head coach, {_safe_name(extracted.coach_name)} served as an "
        f"offensive coordinator for an NFL team."
    )
    await evaluator.verify(
        claim=prev_role_claim,
        node=prev_oc_leaf,
        sources=_combine_sources(extracted.previous_role_urls, extracted.profile_urls),
        additional_instruction=(
            "Confirm that the person's immediate prior role before being hired as an NFL head coach was "
            "specifically 'offensive coordinator' (OC) in the NFL (not college). The page should clearly "
            "indicate it was the immediate previous role."
        )
    )

    # 2.2) That NFL team is based in Florida
    florida_team_leaf = evaluator.add_leaf(
        id="OC_Team_Based_In_Florida",
        desc="That NFL team is based in the state of Florida.",
        parent=prev_role_node,
        critical=True
    )
    fl_claim = (
        "The NFL team for which this person served as offensive coordinator immediately prior to becoming "
        "a head coach is based in the state of Florida."
    )
    await evaluator.verify(
        claim=fl_claim,
        node=florida_team_leaf,
        sources=_combine_sources(extracted.previous_role_urls, extracted.profile_urls),
        additional_instruction=(
            "Acceptable Florida-based teams include Tampa Bay Buccaneers (Tampa, FL), Jacksonville Jaguars "
            "(Jacksonville, FL), and Miami Dolphins (Miami Gardens, FL). Verify the page explicitly or clearly "
            "implies Florida as the team's location."
        )
    )

    # 3) College playing criteria
    college_node = evaluator.add_parallel(
        id="College_Playing_Criteria",
        desc="College playing background matches position and university constraints.",
        parent=profile_root,
        critical=True
    )

    # 3.1) Played college football as WR
    wr_leaf = evaluator.add_leaf(
        id="Played_College_Football_As_WR",
        desc="Person played college football as a wide receiver (not solely coached).",
        parent=college_node,
        critical=True
    )
    wr_claim = f"{_safe_name(extracted.coach_name)} played college football as a wide receiver."
    await evaluator.verify(
        claim=wr_claim,
        node=wr_leaf,
        sources=extracted.college_playing_urls,
        additional_instruction=(
            "Verify playing history as an on-field player in college, specifically the position 'wide receiver' "
            "(WR). Coaching roles do not satisfy this."
        )
    )

    # 3.2) University located in California
    ca_leaf = evaluator.add_leaf(
        id="University_Located_In_California",
        desc="The university where they played is located in California.",
        parent=college_node,
        critical=True
    )
    ca_claim = (
        f"The university where {_safe_name(extracted.coach_name)} played college football "
        f"({_safe_value(extracted.college_university, 'the university')}) is located in California."
    )
    await evaluator.verify(
        claim=ca_claim,
        node=ca_leaf,
        sources=_combine_sources(extracted.university_info_urls, extracted.college_playing_urls),
        additional_instruction=(
            "Confirm that the university is in the state of California. Minor name variants are acceptable."
        )
    )

    # 3.3) University founded in 19th century
    founded_leaf = evaluator.add_leaf(
        id="University_Founded_In_19th_Century",
        desc="The university was founded between 1800 and 1899 (inclusive).",
        parent=college_node,
        critical=True
    )
    founded_claim = (
        f"The university where {_safe_name(extracted.coach_name)} played "
        f"({_safe_value(extracted.college_university, 'the university')}) was founded between 1800 and 1899 (inclusive)."
    )
    await evaluator.verify(
        claim=founded_claim,
        node=founded_leaf,
        sources=extracted.university_info_urls,
        additional_instruction=(
            "Verify the founding year is within 1800–1899 inclusive. If the page lists a founding year like 1899, "
            "that satisfies the condition."
        )
    )

    # 4) College receiving statistics provided (existence checks only as specified)
    stats_node = evaluator.add_parallel(
        id="College_Receiving_Statistics_Provided",
        desc="Complete career receiving statistics from college playing career are provided.",
        parent=profile_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.total_receptions),
        id="Total_Receptions_Provided",
        desc="Total career receptions is provided.",
        parent=stats_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.total_receiving_yards),
        id="Total_Receiving_Yards_Provided",
        desc="Total career receiving yards is provided.",
        parent=stats_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.total_receiving_touchdowns),
        id="Total_Receiving_TDs_Provided",
        desc="Total career receiving touchdowns is provided.",
        parent=stats_node,
        critical=True
    )

    # 5) Undergraduate degree information
    degree_node = evaluator.add_parallel(
        id="Undergraduate_Degree_Information",
        desc="Undergraduate degree details meeting prompt constraints are provided.",
        parent=profile_root,
        critical=True
    )

    # 5.1) Bachelor's from same university as where they played
    same_uni_leaf = evaluator.add_leaf(
        id="Bachelors_From_Same_University",
        desc="Person earned a bachelor's degree from the same university where they played.",
        parent=degree_node,
        critical=True
    )
    same_uni_claim = (
        f"{_safe_name(extracted.coach_name)} earned a bachelor's degree from "
        f"{_safe_value(extracted.degree_university, 'the same university they played for')}, "
        f"which is the same institution where they played college football "
        f"({_safe_value(extracted.college_university, 'the university')})."
    )
    await evaluator.verify(
        claim=same_uni_claim,
        node=same_uni_leaf,
        sources=_combine_sources(extracted.degree_urls, extracted.college_playing_urls),
        additional_instruction=(
            "Confirm two things: (1) it is a bachelor's degree (e.g., BA, BS, Bachelor's), and (2) the granting "
            "university is the same institution as where the person played college football. The confirmation may "
            "come from multiple sources; if both pages together establish the claim, consider it supported."
        )
    )

    # 5.2) Degree type specified (existence)
    evaluator.add_custom_node(
        result=_non_empty(extracted.degree_type),
        id="Degree_Type_Specified",
        desc="Bachelor's degree type is specified (e.g., BA/BS).",
        parent=degree_node,
        critical=True
    )

    # 5.3) Major field specified (existence)
    evaluator.add_custom_node(
        result=_non_empty(extracted.major_field),
        id="Major_Field_Specified",
        desc="Major field of study is specified.",
        parent=degree_node,
        critical=True
    )


def _safe_name(name: Optional[str]) -> str:
    return name if _non_empty(name) else "the identified person"


def _safe_value(value: Optional[str], fallback: str) -> str:
    return value if _non_empty(value) else fallback


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_coach_profile(),
        template_class=CoachProfileExtraction,
        extraction_name="coach_profile_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()