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
TASK_ID = "nfl_coach_identification"
TASK_DESCRIPTION = """
Identify the current NFL head coach who meets all of the following career and educational criteria:

Educational Background:
- Graduated from Azusa Pacific University in 2003 with a Bachelor of Arts degree in Business Administration
- Played college football as a wide receiver at Azusa Pacific University from 1999 to 2003

Early Coaching Career:
- Began coaching career at Carson High School from 2004 to 2005 as offensive coordinator
- Coached at El Camino College from 2006 to 2008, during which time the team won the California Community College State Championship in 2006

NFL Coaching Career:
- Joined the Seattle Seahawks in 2010 and remained with the team for 13 years until 2022, serving in various offensive coaching roles
- Served as offensive coordinator for the Tampa Bay Buccaneers in 2023

Current Position:
- Was hired as an NFL head coach on January 25, 2024
- Currently serves as head coach of the Carolina Panthers
- Achieved his first win as head coach on September 22, 2024, with a 36-22 victory over the Las Vegas Raiders

Provide the coach's name and include reference URLs that verify the educational background, career progression, and current position.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoachExtraction(BaseModel):
    """Extracted coach identification and references."""
    coach_name: Optional[str] = None
    references: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_info() -> str:
    return """
    Extract the identified NFL head coach's name and all reference URLs provided in the answer.

    Return a JSON object with:
    - coach_name: the full name of the coach identified in the answer (string; null if missing).
    - references: an array of URLs explicitly present in the answer that are used as sources/evidence. Include all URLs, regardless of whether they are markdown links or plain text. If none are present, return an empty array.

    Rules for URL extraction:
    - Only include valid URLs explicitly present in the answer text.
    - If a URL is missing the protocol (http:// or https://), prepend http://.
    - If the answer references a site without a direct URL (e.g., "according to Wikipedia") and no actual URL is provided, do not add a URL—just leave the field empty.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return (name or "").strip()


def _build_extra_prereqs(name_node, refs_node) -> List:
    """Ensure verification claims depend on name existence and references existence."""
    return [name_node, refs_node]


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_coach_verification_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: CoachExtraction,
) -> None:
    """
    Build the verification tree and run verifications according to the rubric.
    """
    name = _safe_name(extracted.coach_name)
    refs = extracted.references or []

    # Top-level critical node aggregating all checks
    top_node = evaluator.add_parallel(
        id="Coach_Identification",
        desc="Answer identifies the NFL head coach who satisfies all stated constraints and includes reference URLs that collectively verify the required claims.",
        parent=parent_node,
        critical=True
    )

    # Existence: Coach name provided (critical leaf)
    name_exists_node = evaluator.add_custom_node(
        result=bool(name),
        id="Coach_Name_Provided",
        desc="Provides the coach's name (the identified NFL head coach).",
        parent=top_node,
        critical=True
    )

    # Existence: Reference URLs provided (at least one) (critical leaf)
    refs_provided_node = evaluator.add_custom_node(
        result=(len(refs) > 0),
        id="References_Provided",
        desc="Provides reference URLs that collectively verify the educational background, career progression, and current position claims listed in the rubric.",
        parent=top_node,
        critical=True
    )

    prereqs = _build_extra_prereqs(name_exists_node, refs_provided_node)

    # ----------------------- Educational Background ----------------------- #
    edu_node = evaluator.add_parallel(
        id="Educational_Background",
        desc="Educational background matches all stated constraints.",
        parent=top_node,
        critical=True
    )

    # APU Graduation in 2003
    apu_grad_leaf = evaluator.add_leaf(
        id="Graduated_From_APU_In_2003",
        desc="Coach graduated from Azusa Pacific University in 2003.",
        parent=edu_node,
        critical=True
    )
    claim_apu_grad = f"{name} graduated from Azusa Pacific University in 2003."
    await evaluator.verify(
        claim=claim_apu_grad,
        node=apu_grad_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm the graduation year (2003) and institution (Azusa Pacific University) from the provided source(s)."
    )

    # BA in Business Administration
    ba_leaf = evaluator.add_leaf(
        id="Degree_BA_Business_Administration",
        desc="Coach earned a Bachelor of Arts degree in Business Administration.",
        parent=edu_node,
        critical=True
    )
    claim_ba = f"{name} earned a Bachelor of Arts degree in Business Administration (at Azusa Pacific University)."
    await evaluator.verify(
        claim=claim_ba,
        node=ba_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Allow reasonable variants such as 'Bachelor's in Business Administration' or 'BA in Business Administration'. Verify the degree field and level."
    )

    # College WR 1999–2003 at APU
    wr_leaf = evaluator.add_leaf(
        id="College_Football_WR_1999_2003_APU",
        desc="Coach played college football as a wide receiver at Azusa Pacific University from 1999 to 2003.",
        parent=edu_node,
        critical=True
    )
    claim_wr = f"{name} played college football as a wide receiver at Azusa Pacific University from 1999 to 2003."
    await evaluator.verify(
        claim=claim_wr,
        node=wr_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm both the position (wide receiver) and the span (1999–2003). Minor phrasing variants are acceptable."
    )

    # ----------------------- Early Coaching Career ------------------------ #
    early_node = evaluator.add_parallel(
        id="Early_Coaching_Career",
        desc="Early coaching career matches all stated constraints.",
        parent=top_node,
        critical=True
    )

    # Carson High School OC 2004–2005
    carson_leaf = evaluator.add_leaf(
        id="Carson_HS_OC_2004_2005",
        desc="Coach began coaching at Carson High School from 2004 to 2005 as offensive coordinator.",
        parent=early_node,
        critical=True
    )
    claim_carson = f"From 2004 to 2005, {name} served as offensive coordinator at Carson High School."
    await evaluator.verify(
        claim=claim_carson,
        node=carson_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Verify the role (offensive coordinator) and the timeframe (2004–2005) at Carson High School."
    )

    # El Camino College coach 2006–2008
    el_camino_leaf = evaluator.add_leaf(
        id="El_Camino_College_Coach_2006_2008",
        desc="Coach coached at El Camino College from 2006 to 2008 (as a coach/position coach).",
        parent=early_node,
        critical=True
    )
    claim_el_camino = f"From 2006 to 2008, {name} coached at El Camino College."
    await evaluator.verify(
        claim=claim_el_camino,
        node=el_camino_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm tenure at El Camino College spanning 2006–2008 (position coach/coach). Exact position title variations are acceptable."
    )

    # El Camino Championship 2006
    el_camino_ch_leaf = evaluator.add_leaf(
        id="El_Camino_Championship_2006",
        desc="During the coach's tenure at El Camino College, the team won the California Community College State Championship in 2006.",
        parent=early_node,
        critical=True
    )
    claim_champ = f"In 2006, during {name}'s tenure at El Camino College, the team won the California Community College State Championship."
    await evaluator.verify(
        claim=claim_champ,
        node=el_camino_ch_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Verify that El Camino College won the California Community College State Championship in 2006 and that this coincided with the coach's tenure."
    )

    # ----------------------- NFL Coaching Career -------------------------- #
    nfl_node = evaluator.add_parallel(
        id="NFL_Coaching_Career",
        desc="NFL coaching career matches all stated constraints.",
        parent=top_node,
        critical=True
    )

    # Joined Seahawks in 2010
    seahawks_join_leaf = evaluator.add_leaf(
        id="Seahawks_Joined_2010",
        desc="Coach joined the Seattle Seahawks in 2010.",
        parent=nfl_node,
        critical=True
    )
    claim_join = f"{name} joined the Seattle Seahawks in 2010."
    await evaluator.verify(
        claim=claim_join,
        node=seahawks_join_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm the initial year with the Seattle Seahawks as 2010."
    )

    # Seahawks 13 years 2010–2022
    seahawks_span_leaf = evaluator.add_leaf(
        id="Seahawks_13_Years_2010_2022",
        desc="Coach spent 13 years with the Seattle Seahawks (2010-2022).",
        parent=nfl_node,
        critical=True
    )
    claim_span = f"{name} worked for the Seattle Seahawks from 2010 through 2022 (13 seasons/years)."
    await evaluator.verify(
        claim=claim_span,
        node=seahawks_span_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="If the source confirms tenure from 2010 to 2022, consider '13 years' accurate. Minor phrasing variants (e.g., '13 seasons') are acceptable."
    )

    # Buccaneers OC in 2023
    bucs_leaf = evaluator.add_leaf(
        id="Buccaneers_OC_2023",
        desc="Coach served as offensive coordinator for the Tampa Bay Buccaneers in 2023.",
        parent=nfl_node,
        critical=True
    )
    claim_bucs = f"In 2023, {name} served as offensive coordinator for the Tampa Bay Buccaneers."
    await evaluator.verify(
        claim=claim_bucs,
        node=bucs_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm the role (offensive coordinator) and year (2023) with the Tampa Bay Buccaneers."
    )

    # ----------------------- Current Position and Result ------------------ #
    current_node = evaluator.add_parallel(
        id="Current_Position_and_Result",
        desc="Current position and first-win details match all stated constraints.",
        parent=top_node,
        critical=True
    )

    # Hired head coach on Jan 25, 2024
    hired_leaf = evaluator.add_leaf(
        id="Hired_Head_Coach_Jan_25_2024",
        desc="Coach was hired as an NFL head coach on January 25, 2024.",
        parent=current_node,
        critical=True
    )
    claim_hired = f"On January 25, 2024, {name} was hired as an NFL head coach."
    await evaluator.verify(
        claim=claim_hired,
        node=hired_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Verify the hiring date (January 25, 2024) and head coach appointment."
    )

    # Current head coach of Carolina Panthers
    panthers_leaf = evaluator.add_leaf(
        id="Current_Head_Coach_Carolina_Panthers",
        desc="Coach currently serves as head coach of the Carolina Panthers.",
        parent=current_node,
        critical=True
    )
    claim_panthers = f"{name} is currently the head coach of the Carolina Panthers."
    await evaluator.verify(
        claim=claim_panthers,
        node=panthers_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm current role as head coach of the Carolina Panthers."
    )

    # First win Sept 22, 2024 vs Raiders 36–22
    first_win_leaf = evaluator.add_leaf(
        id="First_Win_Sept_22_2024_vs_Raiders_36_22",
        desc="Coach earned first win as head coach on September 22, 2024, defeating the Las Vegas Raiders 36–22.",
        parent=current_node,
        critical=True
    )
    claim_first_win = f"On September 22, 2024, {name} earned his first win as head coach with a 36–22 victory over the Las Vegas Raiders."
    await evaluator.verify(
        claim=claim_first_win,
        node=first_win_leaf,
        sources=refs,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm the game date (Sept 22, 2024), opponent (Las Vegas Raiders), and score (36–22)."
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
    Evaluate an answer for the NFL head coach identification task.
    """
    # Initialize evaluator
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

    # Extract coach name and references from the answer
    extracted_coach = await evaluator.extract(
        prompt=prompt_extract_coach_info(),
        template_class=CoachExtraction,
        extraction_name="coach_extraction",
    )

    # Build verification tree and run all checks
    await build_coach_verification_tree(
        evaluator=evaluator,
        parent_node=root,
        extracted=extracted_coach
    )

    # Return structured summary
    return evaluator.get_summary()