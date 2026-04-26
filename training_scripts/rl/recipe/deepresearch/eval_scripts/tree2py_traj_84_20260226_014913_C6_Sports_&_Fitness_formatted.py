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
TASK_ID = "coach_identification_acc_uva"
TASK_DESCRIPTION = (
    "Identify the full name of the college football head coach who meets all of the following criteria:\n\n"
    "1. Previously served as a co-offensive coordinator at a university\n"
    "2. During their tenure as co-offensive coordinator, their team won national championships in both 2016 and 2018\n"
    "3. Was hired as a head coach at an ACC conference institution in December 2021\n"
    "4. Is entering their 4th season as head coach in 2025\n"
    "5. Their current program has exactly 2 conference championships in its history, both of which were shared titles (not outright championships)\n"
    "6. Those two conference championships occurred in 1989 and 1995\n"
    "7. Their program participated in the 2025 ACC Championship Game\n"
    "8. Their program lost the 2025 ACC Championship Game to Duke with a final score of 27-20 in overtime on December 7, 2025\n\n"
    "Provide the coach's full name and reference URLs supporting each criterion."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LossDetailsEvidence(BaseModel):
    """URLs supporting detailed facts of the 2025 ACC Championship Game outcome."""
    lost_game_urls: List[str] = Field(default_factory=list)
    opponent_duke_urls: List[str] = Field(default_factory=list)
    final_score_27_20_urls: List[str] = Field(default_factory=list)
    overtime_urls: List[str] = Field(default_factory=list)
    date_dec_7_2025_urls: List[str] = Field(default_factory=list)


class CoachCriteriaExtraction(BaseModel):
    """Structured extraction of the coach name and URLs per criterion from the answer text."""
    coach_full_name: Optional[str] = None
    # Optional helper fields for clearer claims (use when available)
    program_name: Optional[str] = None  # e.g., "Virginia Cavaliers" or "University of Virginia"
    co_oc_university: Optional[str] = None  # e.g., "Clemson University"

    # URLs per criterion
    urls_co_offensive_coordinator_experience: List[str] = Field(default_factory=list)
    urls_nat_championships_2016_2018_during_cooc: List[str] = Field(default_factory=list)
    urls_current_head_coach_uva_acc: List[str] = Field(default_factory=list)
    urls_hired_december_2021: List[str] = Field(default_factory=list)
    urls_entering_4th_season_in_2025: List[str] = Field(default_factory=list)
    urls_program_exactly_2_shared_titles: List[str] = Field(default_factory=list)
    urls_conference_title_years_1989_and_1995: List[str] = Field(default_factory=list)
    urls_participated_in_2025_acc_ccg: List[str] = Field(default_factory=list)

    loss_details: LossDetailsEvidence = Field(default_factory=LossDetailsEvidence)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_evidence() -> str:
    return (
        "From the answer text, extract the coach's full name and the URLs explicitly cited to support each criterion.\n"
        "Return a JSON object with the following fields:\n"
        "1) coach_full_name: The full name of the coach (string). If not provided, return null.\n"
        "2) program_name: The current program name associated with the coach (e.g., 'Virginia Cavaliers' or 'University of Virginia'). If unclear or not stated, return null.\n"
        "3) co_oc_university: If the answer mentions the university where the coach served as co-offensive coordinator, extract it (string). Otherwise, null.\n"
        "4) urls_co_offensive_coordinator_experience: Array of all URLs supporting that the coach previously served as a co-offensive coordinator at a university.\n"
        "5) urls_nat_championships_2016_2018_during_cooc: Array of all URLs supporting that during the coach's co-offensive coordinator tenure, the team won national championships in 2016 and 2018.\n"
        "6) urls_current_head_coach_uva_acc: Array of all URLs supporting that the coach is the current head coach at the University of Virginia (an ACC institution). Include any links that help establish UVA's ACC membership if present in the answer.\n"
        "7) urls_hired_december_2021: Array of all URLs supporting that the coach was hired into the current head coach position in December 2021.\n"
        "8) urls_entering_4th_season_in_2025: Array of all URLs supporting that the coach is entering their 4th season as head coach in 2025.\n"
        "9) urls_program_exactly_2_shared_titles: Array of all URLs supporting that the program has exactly 2 conference championships and both were shared titles (co-championships).\n"
        "10) urls_conference_title_years_1989_and_1995: Array of all URLs supporting that the two conference championships occurred in 1989 and 1995.\n"
        "11) urls_participated_in_2025_acc_ccg: Array of all URLs supporting that the program participated in the 2025 ACC Championship Game.\n"
        "12) loss_details: An object with arrays of URLs for the ACC Championship Game details:\n"
        "    - lost_game_urls: URLs supporting that the program lost the game.\n"
        "    - opponent_duke_urls: URLs supporting that the opponent was Duke.\n"
        "    - final_score_27_20_urls: URLs supporting that the final score was 27-20.\n"
        "    - overtime_urls: URLs supporting that the game went to overtime.\n"
        "    - date_dec_7_2025_urls: URLs supporting that the game date was December 7, 2025.\n\n"
        "Important:\n"
        "- Extract only actual URLs explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.\n"
        "- Deduplicate exact duplicate URLs within each array.\n"
        "- If the answer does not provide any URL for a criterion, return an empty array for that field.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) else ""


def _safe_program_name(program_name: Optional[str]) -> str:
    return program_name.strip() if isinstance(program_name, str) and program_name.strip() else "the coach's current program"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    ext: CoachCriteriaExtraction,
) -> None:
    """
    Build the verification tree and perform verifications based on the extracted evidence.
    """
    coach_name = _safe_name(ext.coach_full_name)
    program_label = _safe_program_name(ext.program_name)
    co_oc_univ = _safe_name(ext.co_oc_university)

    # Top-level critical parallel node
    main_node = evaluator.add_parallel(
        id="Coach_Identification",
        desc="Identify the coach and provide reference URL(s) supporting each stated criterion/constraint.",
        parent=root_node,
        critical=True,
    )

    # 1) Coach full name provided (existence check)
    evaluator.add_custom_node(
        result=bool(coach_name),
        id="Coach_Full_Name_Provided",
        desc="Provide the coach's full name.",
        parent=main_node,
        critical=True,
    )

    # 2) Co-offensive coordinator experience
    node_cooc = evaluator.add_leaf(
        id="Co_Offensive_Coordinator_Experience",
        desc="Provide URL(s) supporting that the coach previously served as a co-offensive coordinator at a university.",
        parent=main_node,
        critical=True,
    )
    if co_oc_univ:
        claim_cooc = f"{coach_name} previously served as a co-offensive coordinator at {co_oc_univ}."
    else:
        claim_cooc = f"{coach_name} previously served as a co-offensive coordinator at a university."
    await evaluator.verify(
        claim=claim_cooc,
        node=node_cooc,
        sources=ext.urls_co_offensive_coordinator_experience,
        additional_instruction="Confirm the page states the person served in the role 'co-offensive coordinator' (accept synonyms like 'co-OC' or 'co-offense coordinator').",
    )

    # 3) National championships (2016, 2018) during co-OC tenure
    node_nat = evaluator.add_leaf(
        id="National_Championships_2016_2018_During_CoOC",
        desc="Provide URL(s) supporting championships in 2016 and 2018 occurred during co-offensive coordinator tenure.",
        parent=main_node,
        critical=True,
    )
    claim_nat = (
        f"During {coach_name}'s tenure as co-offensive coordinator, the team won national championships in 2016 and 2018."
    )
    await evaluator.verify(
        claim=claim_nat,
        node=node_nat,
        sources=ext.urls_nat_championships_2016_2018_during_cooc,
        additional_instruction="Accept 'College Football Playoff national championship' or 'NCAA Division I FBS national champion' phrasing; both years 2016 AND 2018 must be supported.",
    )

    # 4) Current head coach at UVA (ACC) — split into two critical checks under a parallel node
    node_uva_acc = evaluator.add_parallel(
        id="Current_Head_Coach_at_UVA_ACC",
        desc="Provide URL(s) supporting that the coach is currently the head coach at the University of Virginia (an ACC institution).",
        parent=main_node,
        critical=True,
    )
    # 4a) Current head coach at UVA
    node_uva_hc = evaluator.add_leaf(
        id="Current_Head_Coach_UVA",
        desc="Coach is the current head coach at the University of Virginia (Virginia Cavaliers football).",
        parent=node_uva_acc,
        critical=True,
    )
    claim_uva_hc = f"{coach_name} is the current head coach of the Virginia Cavaliers football program at the University of Virginia."
    await evaluator.verify(
        claim=claim_uva_hc,
        node=node_uva_hc,
        sources=ext.urls_current_head_coach_uva_acc,
        additional_instruction="Verify the page clearly states the coach is the current head coach of UVA/Virginia Cavaliers football.",
    )
    # 4b) UVA is an ACC institution
    node_uva_is_acc = evaluator.add_leaf(
        id="UVA_Is_ACC_Institution",
        desc="University of Virginia competes in the ACC (is an ACC institution).",
        parent=node_uva_acc,
        critical=True,
    )
    claim_uva_is_acc = "The University of Virginia (Virginia Cavaliers football) competes in the Atlantic Coast Conference (ACC)."
    await evaluator.verify(
        claim=claim_uva_is_acc,
        node=node_uva_is_acc,
        sources=ext.urls_current_head_coach_uva_acc,
        additional_instruction="The evidence may be on a conference or team page; confirm UVA's ACC affiliation.",
    )

    # 5) Hired December 2021
    node_hired = evaluator.add_leaf(
        id="Hired_December_2021",
        desc="Provide URL(s) supporting that the coach was hired into the current head coach position in December 2021.",
        parent=main_node,
        critical=True,
    )
    claim_hired = f"{coach_name} was hired as head coach in December 2021."
    await evaluator.verify(
        claim=claim_hired,
        node=node_hired,
        sources=ext.urls_hired_december_2021,
        additional_instruction="Look for an official announcement or reputable news sources dated December 2021.",
    )

    # 6) Entering 4th season in 2025
    node_season = evaluator.add_leaf(
        id="Entering_4th_Season_in_2025",
        desc="Provide URL(s) supporting that the coach is entering their 4th season as head coach in 2025.",
        parent=main_node,
        critical=True,
    )
    claim_season = f"In the 2025 season, {coach_name} is entering his fourth season as head coach."
    await evaluator.verify(
        claim=claim_season,
        node=node_season,
        sources=ext.urls_entering_4th_season_in_2025,
        additional_instruction="Pages may say 'entering year 4' or similar phrasing; verify that 2025 corresponds to his fourth season.",
    )

    # 7) Program has exactly 2 shared conference titles
    node_shared_titles = evaluator.add_leaf(
        id="Program_Has_Exactly_2_Shared_Conference_Titles",
        desc="Provide URL(s) supporting exactly 2 conference championships and both were shared titles (co-championships).",
        parent=main_node,
        critical=True,
    )
    claim_shared_titles = (
        f"{program_label} has exactly two conference championships in its history, and both were shared titles (co-championships), not outright."
    )
    await evaluator.verify(
        claim=claim_shared_titles,
        node=node_shared_titles,
        sources=ext.urls_program_exactly_2_shared_titles,
        additional_instruction="Allow 'co-champions' wording to count as 'shared titles'. Ensure the count is exactly two.",
    )

    # 8) Conference title years 1989 and 1995
    node_title_years = evaluator.add_leaf(
        id="Conference_Title_Years_1989_and_1995",
        desc="Provide URL(s) supporting that the program’s two conference championships occurred in 1989 and 1995.",
        parent=main_node,
        critical=True,
    )
    claim_title_years = f"The two conference championships for {program_label} occurred in 1989 and 1995."
    await evaluator.verify(
        claim=claim_title_years,
        node=node_title_years,
        sources=ext.urls_conference_title_years_1989_and_1995,
        additional_instruction="Confirm the specific years listed are 1989 and 1995.",
    )

    # 9) Program participated in 2025 ACC Championship Game
    node_participated = evaluator.add_leaf(
        id="Program_Participated_in_2025_ACC_CCG",
        desc="Provide URL(s) supporting that the program participated in the 2025 ACC Championship Game.",
        parent=main_node,
        critical=True,
    )
    claim_participated = f"{program_label} participated in the 2025 ACC Championship Game."
    await evaluator.verify(
        claim=claim_participated,
        node=node_participated,
        sources=ext.urls_participated_in_2025_acc_ccg,
        additional_instruction="Verify that the team appeared in the ACC Championship Game in 2025.",
    )

    # 10) 2025 ACC Championship Game loss details (parallel/critical sub-checks)
    node_loss_details = evaluator.add_parallel(
        id="2025_ACC_CCG_Loss_Details",
        desc="Provide URL(s) supporting the specified 2025 ACC Championship Game loss details.",
        parent=main_node,
        critical=True,
    )

    # 10a) Lost the game
    node_lost = evaluator.add_leaf(
        id="Lost_The_Game",
        desc="Provide URL(s) supporting that the program lost the 2025 ACC Championship Game.",
        parent=node_loss_details,
        critical=True,
    )
    claim_lost = f"{program_label} lost the 2025 ACC Championship Game."
    await evaluator.verify(
        claim=claim_lost,
        node=node_lost,
        sources=ext.loss_details.lost_game_urls,
        additional_instruction="Confirm the outcome indicates a loss for the program in the 2025 ACC Championship Game.",
    )

    # 10b) Opponent was Duke
    node_opp = evaluator.add_leaf(
        id="Opponent_Duke",
        desc="Provide URL(s) supporting that the opponent was Duke.",
        parent=node_loss_details,
        critical=True,
    )
    claim_opp = "The opponent in the 2025 ACC Championship Game was Duke."
    await evaluator.verify(
        claim=claim_opp,
        node=node_opp,
        sources=ext.loss_details.opponent_duke_urls,
        additional_instruction="Verify that Duke was the opposing team in the 2025 ACC Championship Game.",
    )

    # 10c) Final score was 27-20
    node_score = evaluator.add_leaf(
        id="Final_Score_27_20",
        desc="Provide URL(s) supporting that the final score was 27-20.",
        parent=node_loss_details,
        critical=True,
    )
    claim_score = "The final score of the 2025 ACC Championship Game was 27-20."
    await evaluator.verify(
        claim=claim_score,
        node=node_score,
        sources=ext.loss_details.final_score_27_20_urls,
        additional_instruction="Confirm the page states a 27–20 final score for the 2025 ACC Championship Game.",
    )

    # 10d) Overtime
    node_ot = evaluator.add_leaf(
        id="Overtime",
        desc="Provide URL(s) supporting that the game ended in overtime.",
        parent=node_loss_details,
        critical=True,
    )
    claim_ot = "The 2025 ACC Championship Game went to overtime."
    await evaluator.verify(
        claim=claim_ot,
        node=node_ot,
        sources=ext.loss_details.overtime_urls,
        additional_instruction="Confirm the page states the game was decided in overtime.",
    )

    # 10e) Date was December 7, 2025
    node_date = evaluator.add_leaf(
        id="Date_December_7_2025",
        desc="Provide URL(s) supporting that the game was played on December 7, 2025.",
        parent=node_loss_details,
        critical=True,
    )
    claim_date = "The 2025 ACC Championship Game was played on December 7, 2025."
    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        sources=ext.loss_details.date_dec_7_2025_urls,
        additional_instruction="Confirm the page lists the game date as December 7, 2025 (accept reasonable date format variations).",
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
    """
    Evaluate an answer for the coach identification task using the Mind2Web2 framework.
    """
    # Initialize evaluator with a parallel root (we'll add a critical child node under it)
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

    # Extract structured evidence from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_coach_evidence(),
        template_class=CoachCriteriaExtraction,
        extraction_name="coach_criteria_evidence",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()