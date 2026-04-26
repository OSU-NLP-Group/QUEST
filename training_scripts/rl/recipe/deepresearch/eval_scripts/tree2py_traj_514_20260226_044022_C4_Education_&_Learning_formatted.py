import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fcs_coach_departure_2024"
TASK_DESCRIPTION = (
    "Identify the NCAA Division I FCS university where a head football coach with the following career characteristics "
    "most recently served before accepting an FBS head coaching position in December 2024: "
    "(1) The coach began their coaching career as a graduate assistant at the University of Wisconsin, working under "
    "legendary head coach Barry Alvarez; "
    "(2) The coach was appointed as head football coach at the university in question in December 2021; "
    "(3) The coach served exactly three seasons (2022-2024) as head coach at this university; "
    "(4) During their tenure, the coach compiled an overall winning record with more than 20 total victories; "
    "(5) Under this coach's leadership, the football team made multiple appearances in the FCS playoffs; "
    "(6) After the 2024 season, the coach departed to accept a head coaching position at a Mountain West Conference university. "
    "Provide the full name of the university and include reference URLs that support your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityCoachExtraction(BaseModel):
    """
    Structured information extracted from the answer.
    - university: Full official name of the NCAA Division I FCS university identified in the answer.
    - coach: Full name of the head coach identified in the answer.
    - appointment_month_year: Month and year when the coach was appointed at the identified university (e.g., 'December 2021').
    - tenure_seasons: Seasons explicitly claimed (e.g., ['2022', '2023', '2024']).
    - total_wins: Total number of wins claimed across the tenure (string to accommodate formats like '25' or '25-?').
    - fcs_playoff_appearances: Claimed playoff appearances text or count (e.g., 'multiple', '2', 'two').
    - departure_destination: The FBS university the coach departed to in December 2024 (if provided).
    - departure_month_year: Month and year of departure (e.g., 'December 2024') if provided.
    - sources: All URLs explicitly cited in the answer that support any part of the claim.
    """
    university: Optional[str] = None
    coach: Optional[str] = None
    appointment_month_year: Optional[str] = None
    tenure_seasons: List[str] = Field(default_factory=list)
    total_wins: Optional[str] = None
    fcs_playoff_appearances: Optional[str] = None
    departure_destination: Optional[str] = None
    departure_month_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_university_and_coach() -> str:
    return (
        "Extract the key information the answer provides about the identified university and the head coach. "
        "Return a JSON object with the following fields:\n"
        "1) university: The full official name of the NCAA Division I FCS university identified in the answer.\n"
        "2) coach: The full name of the head coach identified in the answer.\n"
        "3) appointment_month_year: The month and year when this coach was appointed head coach at the university "
        "(e.g., 'December 2021') if mentioned; otherwise null.\n"
        "4) tenure_seasons: A list of distinct seasons explicitly claimed for the coach's tenure at this university. "
        "For example, if the answer says '2022–2024' or 'the 2022, 2023, and 2024 seasons', return ['2022','2023','2024'] when possible. "
        "If not clearly stated, return an empty array.\n"
        "5) total_wins: The total number of wins claimed for the coach during the tenure (as text). If not mentioned, null.\n"
        "6) fcs_playoff_appearances: The answer's claim about FCS playoff appearances under this coach (text such as 'multiple', 'two', or a number). If not mentioned, null.\n"
        "7) departure_destination: The name of the FBS university the coach moved to in December 2024, if explicitly provided; otherwise null.\n"
        "8) departure_month_year: The departure month and year if explicitly provided (e.g., 'December 2024'); otherwise null.\n"
        "9) sources: An array of every explicit URL cited anywhere in the answer (including markdown links). "
        "Only include valid URLs that are clearly associated with this topic in the answer.\n"
        "Do not hallucinate. If a field is missing from the answer, return null (or empty array for tenure_seasons)."
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_university(evaluator: Evaluator, parent_node, data: UniversityCoachExtraction) -> None:
    """
    Build verification subtree for 'University_Identification' and run verifications.
    All child nodes are critical to align with the rubric's mandatory criteria.
    """
    # Create the main critical node for this rubric section
    uni_node = evaluator.add_parallel(
        id="University_Identification",
        desc="Identify the university where a head football coach with specific career characteristics most recently served before moving to another institution in December 2024",
        parent=parent_node,
        critical=True
    )

    # Convenience values with graceful fallbacks for claim readability
    university = (data.university or "the identified university").strip()
    coach = (data.coach or "the coach").strip()
    dest = (data.departure_destination or "a Mountain West Conference university").strip()
    all_sources = data.sources or []

    # Gatekeeper: Ensure the answer provided a university name and at least one source URL
    provided_uni_sources_node = evaluator.add_custom_node(
        result=bool(data.university and data.university.strip()) and (len(all_sources) > 0),
        id="Provided_University_And_Sources",
        desc="Answer provides a specific university name and at least one supporting URL",
        parent=uni_node,
        critical=True
    )

    # 1) NCAA FCS classification
    fcs_class_node = evaluator.add_leaf(
        id="NCAA_FCS_Classification",
        desc="The identified university competes at the NCAA Division I FCS level",
        parent=uni_node,
        critical=True
    )
    claim_fcs = f"{university} competes in NCAA Division I FCS (Football Championship Subdivision) in football."
    await evaluator.verify(
        claim=claim_fcs,
        node=fcs_class_node,
        sources=all_sources,
        additional_instruction=(
            "Verify that the football program is part of the FCS (not FBS). "
            "If league membership changed historically, judge based on the period 2022–2024."
        ),
    )

    # 2) Graduate assistant under Barry Alvarez at Wisconsin
    ga_under_alvarez_node = evaluator.add_leaf(
        id="Graduate_Assistant_Under_Alvarez",
        desc="The head coach began their coaching career as a graduate assistant at the University of Wisconsin under head coach Barry Alvarez",
        parent=uni_node,
        critical=True
    )
    claim_ga = (
        f"{coach} began his coaching career as a graduate assistant at the University of Wisconsin under head coach Barry Alvarez."
    )
    await evaluator.verify(
        claim=claim_ga,
        node=ga_under_alvarez_node,
        sources=all_sources,
        additional_instruction=(
            "Look for biographical summaries, media guides, or articles stating that the coach's first coaching role "
            "was a GA at Wisconsin under Barry Alvarez. Allow minor wording variations like 'GA at Wisconsin'."
        ),
    )

    # 3) Appointed in December 2021
    dec_2021_node = evaluator.add_leaf(
        id="December_2021_Appointment",
        desc="The coach was appointed as head football coach at the identified university in December 2021",
        parent=uni_node,
        critical=True
    )
    claim_appt = f"In December 2021, {coach} was appointed head football coach at {university}."
    await evaluator.verify(
        claim=claim_appt,
        node=dec_2021_node,
        sources=all_sources,
        additional_instruction=(
            "Confirm that the hiring/appointment announcement is dated in December 2021 for the specified university."
        ),
    )

    # 4) Served exactly 3 seasons (2022-2024)
    three_seasons_node = evaluator.add_leaf(
        id="Three_Season_Tenure",
        desc="The coach served exactly three consecutive seasons (2022-2024) as head coach at this university",
        parent=uni_node,
        critical=True
    )
    claim_tenure = f"{coach} served exactly three seasons (2022, 2023, and 2024) as head coach at {university}."
    await evaluator.verify(
        claim=claim_tenure,
        node=three_seasons_node,
        sources=all_sources,
        additional_instruction=(
            "Check season-by-season summaries, bio pages, or news reports stating the coach led the team during 2022, 2023, and 2024 (three seasons total)."
        ),
    )

    # 5) Winning record with more than 20 total victories
    winning_record_node = evaluator.add_leaf(
        id="Winning_Record_Achievement",
        desc="During their tenure at this university, the coach achieved an overall winning record with more than 20 total victories",
        parent=uni_node,
        critical=True
    )
    claim_record = (
        f"From 2022 through 2024 at {university}, {coach} compiled an overall winning record and more than 20 total victories."
    )
    await evaluator.verify(
        claim=claim_record,
        node=winning_record_node,
        sources=all_sources,
        additional_instruction=(
            "Use official record aggregates or trusted reports (e.g., media guides, season summaries) to confirm total wins exceed 20 and the record is overall winning."
        ),
    )

    # 6) Multiple FCS playoff appearances under this coach
    playoffs_node = evaluator.add_leaf(
        id="FCS_Playoff_Appearances",
        desc="Under this coach's leadership, the university's football team made multiple appearances in the NCAA FCS playoffs",
        parent=uni_node,
        critical=True
    )
    claim_playoffs = (
        f"Under {coach}, {university}'s football team made multiple NCAA FCS Playoffs appearances (at least two) between 2022 and 2024."
    )
    await evaluator.verify(
        claim=claim_playoffs,
        node=playoffs_node,
        sources=all_sources,
        additional_instruction=(
            "Confirm that the team qualified for the FCS playoffs two or more times during the coach's 2022–2024 tenure."
        ),
    )

    # 7) Departure after 2024 season to a Mountain West Conference university
    mwc_departure_node = evaluator.add_leaf(
        id="Mountain_West_Departure",
        desc="After the 2024 season, the coach departed to accept a head coaching position at a Mountain West Conference university",
        parent=uni_node,
        critical=True
    )
    claim_departure = (
        f"After the 2024 season, {coach} departed to accept a head coaching position at {dest}, which is a Mountain West Conference university."
        if data.departure_destination
        else f"After the 2024 season, {coach} departed to accept a head coaching position at a Mountain West Conference university."
    )
    await evaluator.verify(
        claim=claim_departure,
        node=mwc_departure_node,
        sources=all_sources,
        additional_instruction=(
            "Prefer December 2024 reports and official announcements. If a destination university is named, "
            "also verify that institution is a member of the Mountain West Conference."
        ),
    )


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
    """
    Evaluate an answer for the NCAA Division I FCS coach-to-MWC departure identification task.
    """
    # Initialize evaluator with a parallel root (we'll add a critical parallel subtree under it)
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

    # 1) Extract structured info from the answer
    extracted: UniversityCoachExtraction = await evaluator.extract(
        prompt=prompt_extract_university_and_coach(),
        template_class=UniversityCoachExtraction,
        extraction_name="university_coach_extraction",
    )

    # 2) Build verification tree and run checks
    await build_and_verify_university(evaluator, root, extracted)

    # 3) Return structured evaluation summary
    return evaluator.get_summary()