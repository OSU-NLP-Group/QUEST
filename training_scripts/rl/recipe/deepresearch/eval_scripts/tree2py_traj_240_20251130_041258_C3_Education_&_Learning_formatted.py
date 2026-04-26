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
TASK_ID = "yale_qb_research_2025"
TASK_DESCRIPTION = (
    "Research Yale's starting quarterback from their November 22, 2025 football game against Harvard, which Yale won 45-28. "
    "Provide the following information: (1) The quarterback's name and current class year at Yale, "
    "(2) His passing statistics from the Harvard game (total passing yards and touchdown passes thrown), "
    "(3) His transfer history (which university he transferred from and his status there) and his family connection to Yale's football program, "
    "(4) His performance in Yale's first-round FCS playoff game (the opponent faced and his total passing yards in that game)."
)

# Expected facts embedded from rubric
EXPECTED = {
    "qb_name": "Dante Reno",
    "qb_jersey_number": "2",
    "qb_class_year": "Sophomore",
    "harvard_game_date": "November 22, 2025",
    "harvard_game_venue": "Yale Bowl",
    "harvard_game_location": "New Haven, Connecticut",
    "harvard_game_opponent": "Harvard",
    "harvard_game_final_score_canonical": "45-28",  # Accept 45–28 as equivalent
    "harvard_passing_yards": "273",
    "harvard_passing_tds": "3",
    "transfer_from": "University of South Carolina",
    "prior_status": "redshirt freshman in 2024",
    "father_name": "Tony Reno",
    "father_role": "Yale head football coach",
    "playoff_opponent": "Youngstown State",
    "playoff_date_allowed": ["November 29, 2025", "November 30, 2025"],
    "playoff_passing_yards": "260",
    "playoff_result_canonical": "43-42",  # Accept 43–42 as equivalent
    "playoff_comeback_detail": "rallied from a 42-14 deficit",
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class IdentityInfo(BaseModel):
    qb_name: Optional[str] = None
    jersey_number: Optional[str] = None  # Keep exactly as in the answer, e.g., "2", "#2", "No. 2"
    class_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HarvardAnchorInfo(BaseModel):
    game_date: Optional[str] = None  # e.g., "November 22, 2025", "Nov 22, 2025", "11/22/2025"
    venue: Optional[str] = None      # e.g., "Yale Bowl"
    location: Optional[str] = None   # e.g., "New Haven, Connecticut", "New Haven, CT"
    opponent: Optional[str] = None   # e.g., "Harvard"
    final_score: Optional[str] = None  # e.g., "45-28", "45–28", "45 to 28"
    sources: List[str] = Field(default_factory=list)


class HarvardStatsInfo(BaseModel):
    passing_yards: Optional[str] = None  # e.g., "273", "273 yards", "273 yds"
    passing_tds: Optional[str] = None    # e.g., "3", "three"
    sources: List[str] = Field(default_factory=list)


class TransferInfo(BaseModel):
    transferred_from: Optional[str] = None  # e.g., "University of South Carolina", "South Carolina"
    prior_status: Optional[str] = None      # e.g., "redshirt freshman in 2024"
    sources: List[str] = Field(default_factory=list)


class FamilyInfo(BaseModel):
    father_name: Optional[str] = None    # e.g., "Tony Reno"
    father_role: Optional[str] = None    # e.g., "Yale head football coach"
    sources: List[str] = Field(default_factory=list)


class PlayoffInfo(BaseModel):
    opponent: Optional[str] = None           # e.g., "Youngstown State"
    game_date: Optional[str] = None          # e.g., "November 29, 2025" or "November 30, 2025"
    passing_yards: Optional[str] = None      # e.g., "260", "260 yards"
    result: Optional[str] = None             # e.g., "43-42", "43–42"
    comeback_detail: Optional[str] = None    # e.g., "rallied from a 42-14 deficit"
    sources: List[str] = Field(default_factory=list)


class QBEvalExtraction(BaseModel):
    identity: Optional[IdentityInfo] = None
    harvard_anchor: Optional[HarvardAnchorInfo] = None
    harvard_stats: Optional[HarvardStatsInfo] = None
    transfer: Optional[TransferInfo] = None
    family: Optional[FamilyInfo] = None
    playoff: Optional[PlayoffInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_qb_research() -> str:
    return """
    Extract the following structured information EXACTLY as presented in the answer text (do not infer or normalize; keep strings as they appear).

    1) identity:
       - qb_name: the starting quarterback’s name as stated
       - jersey_number: the jersey identifier as written (e.g., "2", "#2", "No. 2")
       - class_year: the class year as written (e.g., "Sophomore", "So.", "second-year")
       - sources: a list of URLs cited in the answer that support QB identity/class/jersey; include only explicit URLs from the answer

    2) harvard_anchor (about the Harvard–Yale game mentioned in the task):
       - game_date: the date string as written (e.g., "November 22, 2025", "Nov. 22, 2025", "11/22/2025")
       - venue: the venue as written (e.g., "Yale Bowl")
       - location: the location as written (e.g., "New Haven, Connecticut", "New Haven, CT")
       - opponent: the opponent as written (e.g., "Harvard")
       - final_score: the score string as written (e.g., "45-28", "45–28", "45 to 28")
       - sources: URLs cited that support date/venue/location/opponent/final score

    3) harvard_stats (QB passing stats for the Harvard game):
       - passing_yards: the yards value string as written (e.g., "273", "273 yards", "273 yds")
       - passing_tds: the touchdown passes value as written (e.g., "3", "three")
       - sources: URLs cited that support these Harvard-game stats

    4) transfer:
       - transferred_from: the prior school as written (e.g., "University of South Carolina", "South Carolina")
       - prior_status: the status at that school as written (e.g., "redshirt freshman in 2024")
       - sources: URLs cited that support transfer origin and status

    5) family:
       - father_name: the father's name as written
       - father_role: the father's role as written (e.g., "Yale head football coach")
       - sources: URLs cited that support this family connection information

    6) playoff (first-round FCS playoff details for Yale and the QB):
       - opponent: the opponent as written (e.g., "Youngstown State")
       - game_date: the date string as written (e.g., "November 29, 2025" or "November 30, 2025")
       - passing_yards: the QB’s playoff passing yards value as written (e.g., "260", "260 yards")
       - result: the score/result string as written (e.g., "43-42", "43–42")
       - comeback_detail: the comeback phrasing as written (e.g., "rallied from a 42-14 deficit")
       - sources: URLs cited that support these playoff details

    Rules:
    - Return null for any field not present in the answer.
    - For numbers and dates, keep them as strings EXACTLY as they appear.
    - For sources, only include URLs explicitly present in the answer; if none are present for a section, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nz(s: Optional[str]) -> str:
    return s if s is not None else ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_qb_identity_and_class(evaluator: Evaluator, parent, ex: QBEvalExtraction) -> None:
    node = evaluator.add_parallel(
        id="QB_Identity_And_Class",
        desc="Correctly identifies Yale's starting QB and his current class year.",
        parent=parent,
        critical=True,
    )

    # QB Name: must be Dante Reno
    qb_name_leaf = evaluator.add_leaf(
        id="QB_Name",
        desc="States the starting QB's name as Dante Reno.",
        parent=node,
        critical=True,
    )
    qb_name_val = _nz(ex.identity.qb_name) if ex.identity else ""
    claim = f"The QB name stated in the answer ('{qb_name_val}') matches the expected starting quarterback 'Dante Reno' (allowing minor spelling/casing variations)."
    await evaluator.verify(
        claim=claim,
        node=qb_name_leaf,
        additional_instruction="Treat minor spelling/casing variations and middle initials as equivalent if they clearly refer to the same person."
    )

    # Jersey Number: must indicate #2
    jersey_leaf = evaluator.add_leaf(
        id="QB_Jersey_Number",
        desc="States the QB wears jersey #2.",
        parent=node,
        critical=True,
    )
    jersey_val = _nz(ex.identity.jersey_number) if ex.identity else ""
    claim = f"The jersey identifier provided in the answer ('{jersey_val}') indicates that the quarterback wears number 2 (acceptable forms include '2', '#2', or 'No. 2')."
    await evaluator.verify(
        claim=claim,
        node=jersey_leaf,
        additional_instruction="Interpret common jersey notation like '2', '#2', 'No. 2', or 'number 2' as equivalent."
    )

    # Class Year: must be Sophomore
    class_leaf = evaluator.add_leaf(
        id="QB_Class_Year",
        desc="States Dante Reno is listed as a Sophomore.",
        parent=node,
        critical=True,
    )
    class_val = _nz(ex.identity.class_year) if ex.identity else ""
    claim = f"The class year stated in the answer ('{class_val}') corresponds to 'Sophomore' (e.g., 'Sophomore', 'So.', or equivalent)."
    await evaluator.verify(
        claim=claim,
        node=class_leaf,
        additional_instruction="Accept reasonable abbreviations like 'Soph.', 'So.', or synonymous phrasing clearly indicating sophomore status."
    )


async def verify_harvard_yale_game_anchor(evaluator: Evaluator, parent, ex: QBEvalExtraction) -> None:
    node = evaluator.add_parallel(
        id="Harvard_Yale_Game_Anchor",
        desc="Correctly anchors the answer to the specified Harvard–Yale game (date/location/opponent/result).",
        parent=parent,
        critical=True,
    )

    anchor = ex.harvard_anchor or HarvardAnchorInfo()

    # Game Date
    date_leaf = evaluator.add_leaf(
        id="Game_Date",
        desc="States the game was played on November 22, 2025.",
        parent=node,
        critical=True,
    )
    date_val = _nz(anchor.game_date)
    claim = f"The date stated in the answer ('{date_val}') corresponds to November 22, 2025 (allow equivalences like 'Nov 22, 2025' or '11/22/2025')."
    await evaluator.verify(
        claim=claim,
        node=date_leaf,
        additional_instruction="Treat 'November 22, 2025', 'Nov. 22, 2025', 'Nov 22, 2025', or '11/22/2025' as equivalent representations of the same date."
    )

    # Venue
    venue_leaf = evaluator.add_leaf(
        id="Game_Venue",
        desc="States the game was held at Yale Bowl.",
        parent=node,
        critical=True,
    )
    venue_val = _nz(anchor.venue)
    claim = f"The venue stated in the answer ('{venue_val}') corresponds to 'Yale Bowl' (accept 'the Yale Bowl' as equivalent)."
    await evaluator.verify(
        claim=claim,
        node=venue_leaf,
        additional_instruction="Consider 'Yale Bowl' and 'the Yale Bowl' equivalent."
    )

    # Location
    loc_leaf = evaluator.add_leaf(
        id="Game_Location",
        desc="States the game took place in New Haven, Connecticut.",
        parent=node,
        critical=True,
    )
    loc_val = _nz(anchor.location)
    claim = f"The location stated in the answer ('{loc_val}') corresponds to 'New Haven, Connecticut' (accept 'New Haven, CT')."
    await evaluator.verify(
        claim=claim,
        node=loc_leaf,
        additional_instruction="Treat 'New Haven, Connecticut' and 'New Haven, CT' as equivalent."
    )

    # Opponent
    opp_leaf = evaluator.add_leaf(
        id="Opponent",
        desc="States the opponent was Harvard.",
        parent=node,
        critical=True,
    )
    opp_val = _nz(anchor.opponent)
    claim = f"The opponent stated in the answer ('{opp_val}') corresponds to 'Harvard' (accept 'Harvard University' or 'the Crimson' as equivalent)."
    await evaluator.verify(
        claim=claim,
        node=opp_leaf,
        additional_instruction="Treat 'Harvard', 'Harvard University', and 'the Crimson' as equivalent identifications of the opponent."
    )

    # Final Score
    score_leaf = evaluator.add_leaf(
        id="Final_Score",
        desc="States Yale won with a final score of 45–28.",
        parent=node,
        critical=True,
    )
    score_val = _nz(anchor.final_score)
    claim = f"The final score stated in the answer ('{score_val}') corresponds to Yale 45, Harvard 28 (treat '45-28', '45–28', or '45 to 28' as equivalent)."
    await evaluator.verify(
        claim=claim,
        node=score_leaf,
        additional_instruction="Accept minor punctuation or formatting differences such as hyphen vs en dash or 'to' phrasing."
    )


async def verify_harvard_game_passing_stats(evaluator: Evaluator, parent, ex: QBEvalExtraction) -> None:
    node = evaluator.add_parallel(
        id="Harvard_Game_Passing_Stats",
        desc="Provides the QB's passing yards and passing TDs from the Harvard game.",
        parent=parent,
        critical=True,
    )

    stats = ex.harvard_stats or HarvardStatsInfo()

    # Passing Yards
    yards_leaf = evaluator.add_leaf(
        id="Passing_Yards",
        desc="States Dante Reno threw for 273 passing yards vs Harvard.",
        parent=node,
        critical=True,
    )
    yards_val = _nz(stats.passing_yards)
    claim = f"The Harvard-game passing yards stated in the answer ('{yards_val}') correspond to 273 yards (accept forms like '273', '273 yards', '273 yds')."
    await evaluator.verify(
        claim=claim,
        node=yards_leaf,
        additional_instruction="Consider numeric-only and unit-suffixed forms (e.g., '273', '273 yards', '273 yds') equivalent."
    )

    # Passing TDs
    tds_leaf = evaluator.add_leaf(
        id="Passing_TDs",
        desc="States Dante Reno threw 3 touchdown passes vs Harvard.",
        parent=node,
        critical=True,
    )
    tds_val = _nz(stats.passing_tds)
    claim = f"The Harvard-game passing TDs stated in the answer ('{tds_val}') correspond to 3 touchdown passes (accept numeric or word forms like '3' or 'three')."
    await evaluator.verify(
        claim=claim,
        node=tds_leaf,
        additional_instruction="Accept numeric or word forms that clearly indicate three touchdown passes."
    )


async def verify_transfer_history(evaluator: Evaluator, parent, ex: QBEvalExtraction) -> None:
    node = evaluator.add_parallel(
        id="Transfer_History",
        desc="Provides transfer origin school and prior status there.",
        parent=parent,
        critical=True,
    )

    transfer = ex.transfer or TransferInfo()

    # Transferred From
    trans_from_leaf = evaluator.add_leaf(
        id="Transferred_From",
        desc="States Dante Reno transferred from the University of South Carolina.",
        parent=node,
        critical=True,
    )
    trans_from_val = _nz(transfer.transferred_from)
    claim = (
        f"The transfer origin stated in the answer ('{trans_from_val}') corresponds to the University of South Carolina "
        f"(accept 'South Carolina' or 'South Carolina Gamecocks' as equivalent)."
    )
    await evaluator.verify(
        claim=claim,
        node=trans_from_leaf,
        additional_instruction="Treat 'University of South Carolina', 'South Carolina', and 'South Carolina Gamecocks' as equivalent identifications of the prior school."
    )

    # Prior Status at South Carolina
    prior_status_leaf = evaluator.add_leaf(
        id="Prior_School_Status",
        desc="States that at South Carolina he was a redshirt freshman in 2024.",
        parent=node,
        critical=True,
    )
    prior_status_val = _nz(transfer.prior_status)
    claim = (
        f"The status at South Carolina stated in the answer ('{prior_status_val}') corresponds to being a redshirt freshman in 2024 "
        f"(accept equivalent phrasing like 'redshirted in 2024' or 'RS freshman (2024)')."
    )
    await evaluator.verify(
        claim=claim,
        node=prior_status_leaf,
        additional_instruction="Accept reasonable paraphrases like 'redshirted in 2024', 'RS freshman in 2024', or 'redshirt freshman (2024)'."
    )


async def verify_family_connection(evaluator: Evaluator, parent, ex: QBEvalExtraction) -> None:
    node = evaluator.add_parallel(
        id="Family_Connection_To_Yale_Football",
        desc="Provides the specified family connection to Yale football.",
        parent=parent,
        critical=True,
    )

    family = ex.family or FamilyInfo()

    # Father Identity
    father_id_leaf = evaluator.add_leaf(
        id="Father_Identity",
        desc="States Dante Reno's father is Tony Reno.",
        parent=node,
        critical=True,
    )
    father_name_val = _nz(family.father_name)
    claim = f"The father stated in the answer ('{father_name_val}') corresponds to 'Tony Reno' (allowing minor variations like 'Anthony Reno' if clearly the same person)."
    await evaluator.verify(
        claim=claim,
        node=father_id_leaf,
        additional_instruction="Treat 'Tony Reno' and 'Anthony Reno' as equivalent if clearly the same person."
    )

    # Father Role
    father_role_leaf = evaluator.add_leaf(
        id="Father_Role",
        desc="States Tony Reno serves as Yale's head football coach.",
        parent=node,
        critical=True,
    )
    father_role_val = _nz(family.father_role)
    claim = f"The role stated in the answer ('{father_role_val}') corresponds to Tony Reno being Yale's head football coach."
    await evaluator.verify(
        claim=claim,
        node=father_role_leaf,
        additional_instruction="Accept equivalent phrasing like 'Yale head coach', 'head football coach at Yale', or 'Yale football head coach'."
    )


async def verify_playoff_game(evaluator: Evaluator, parent, ex: QBEvalExtraction) -> None:
    node = evaluator.add_parallel(
        id="First_Round_FCS_Playoff_Game",
        desc="Provides opponent and QB passing yards in Yale's first-round FCS playoff game (and related constrained details).",
        parent=parent,
        critical=True,
    )

    playoff = ex.playoff or PlayoffInfo()

    # Opponent
    opp_leaf = evaluator.add_leaf(
        id="Playoff_Opponent",
        desc="States Yale played Youngstown State in the first round.",
        parent=node,
        critical=True,
    )
    opp_val = _nz(playoff.opponent)
    claim = f"The playoff opponent stated in the answer ('{opp_val}') corresponds to 'Youngstown State' (accept 'Youngstown State Penguins' or 'YSU' as equivalent)."
    await evaluator.verify(
        claim=claim,
        node=opp_leaf,
        additional_instruction="Treat 'Youngstown State', 'Youngstown State Penguins', and 'YSU' as equivalent identifications of the opponent."
    )

    # Game Date (either November 29 or November 30, 2025)
    date_leaf = evaluator.add_leaf(
        id="Playoff_Game_Date",
        desc="States the game date as November 29 or November 30, 2025.",
        parent=node,
        critical=True,
    )
    date_val = _nz(playoff.game_date)
    allowed = "', '".join(EXPECTED["playoff_date_allowed"])
    claim = (
        f"The playoff game date stated in the answer ('{date_val}') is one of the allowed dates: "
        f"'{allowed}' (accept reasonable abbreviations like 'Nov. 29, 2025')."
    )
    await evaluator.verify(
        claim=claim,
        node=date_leaf,
        additional_instruction="The statement should indicate either November 29, 2025 or November 30, 2025 (abbreviations like 'Nov. 29, 2025' are acceptable)."
    )

    # Passing Yards
    py_leaf = evaluator.add_leaf(
        id="Playoff_Passing_Yards",
        desc="States Dante Reno threw for 260 passing yards vs Youngstown State.",
        parent=node,
        critical=True,
    )
    py_val = _nz(playoff.passing_yards)
    claim = f"The playoff passing yards stated in the answer ('{py_val}') correspond to 260 yards (accept forms like '260', '260 yards', '260 yds')."
    await evaluator.verify(
        claim=claim,
        node=py_leaf,
        additional_instruction="Consider numeric-only and unit-suffixed forms (e.g., '260', '260 yards', '260 yds') equivalent."
    )

    # Result (43–42)
    result_leaf = evaluator.add_leaf(
        id="Playoff_Result",
        desc="States Yale won 43–42 vs Youngstown State.",
        parent=node,
        critical=True,
    )
    result_val = _nz(playoff.result)
    claim = f"The playoff result stated in the answer ('{result_val}') corresponds to Yale 43, Youngstown State 42 (treat '43-42' and '43–42' as equivalent)."
    await evaluator.verify(
        claim=claim,
        node=result_leaf,
        additional_instruction="Accept minor punctuation or formatting differences such as hyphen vs en dash or 'to' phrasing."
    )

    # Comeback detail
    comeback_leaf = evaluator.add_leaf(
        id="Playoff_Comeback_Detail",
        desc="States Yale rallied from a 42–14 deficit.",
        parent=node,
        critical=True,
    )
    comeback_val = _nz(playoff.comeback_detail)
    claim = f"The playoff comeback detail stated in the answer ('{comeback_val}') corresponds to Yale rallying from a 42-14 deficit (accept '42–14' punctuation)."
    await evaluator.verify(
        claim=claim,
        node=comeback_leaf,
        additional_instruction="Accept '42-14' and '42–14' as equivalent punctuation; reasonable paraphrases like 'trailed 42-14 before rallying' are acceptable."
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
    Evaluate an answer for the Yale QB research task (Harvard game Nov 22, 2025 and first-round FCS playoff).
    Returns a structured summary containing the verification tree and final score.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Create a critical top-level task node (to enforce overall failure if any essential part fails)
    task_root = evaluator.add_parallel(
        id="Yale_QB_Research_Task",
        desc="Provide the requested QB identity/class year, Harvard-game passing stats, transfer+family connection, and first-round FCS playoff opponent+passing yards, consistent with the given constraints.",
        parent=root,
        critical=True,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_qb_research(),
        template_class=QBEvalExtraction,
        extraction_name="extracted_qb_research",
    )

    # Add ground truth info to summary (for transparency)
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED,
            "notes": "Expected values are embedded based on the rubric for this specific task."
        },
        gt_type="expected_facts"
    )

    # Build verification subtrees
    await verify_qb_identity_and_class(evaluator, task_root, extracted)
    await verify_harvard_yale_game_anchor(evaluator, task_root, extracted)
    await verify_harvard_game_passing_stats(evaluator, task_root, extracted)
    await verify_transfer_history(evaluator, task_root, extracted)
    await verify_family_connection(evaluator, task_root, extracted)
    await verify_playoff_game(evaluator, task_root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()