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
TASK_ID = "identify_nfl_qb_2024_packers"
TASK_DESCRIPTION = (
    "Identify the NFL quarterback who meets all of the following criteria: "
    "(1) Born on November 2, 1998; "
    "(2) Born in Bakersfield, California; "
    "(3) Played college football at Utah State University; "
    "(4) As a sophomore in the 2018 season, passed for 3,567 yards (a Utah State single-season record); "
    "(5) As a sophomore in the 2018 season, threw 32 touchdown passes (a Utah State single-season record); "
    "(6) Was selected with the 26th overall pick in the first round of the 2020 NFL Draft; "
    "(7) Was drafted by the Green Bay Packers; "
    "(8) Is currently the starting quarterback for the Green Bay Packers in the 2024 NFL season. "
    "Provide the quarterback's full name along with reference URLs that verify the key biographical information, "
    "college statistics, and NFL career details."
)

# Ground truth expectation (for info only; verification will rely on provided sources)
GROUND_TRUTH = {
    "expected_name": "Jordan Love",
    "birth_date": "November 2, 1998",
    "birth_place": "Bakersfield, California",
    "college": "Utah State University",
    "stats_2018": {
        "passing_yards": "3,567",
        "touchdowns": "32",
        "records": ["single-season passing yards", "single-season touchdown passes"]
    },
    "draft": {
        "year": "2020",
        "round": "first",
        "pick_overall": "26",
        "team": "Green Bay Packers"
    },
    "status_2024": "starting quarterback for Green Bay Packers"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QuarterbackExtraction(BaseModel):
    # Identity
    full_name: Optional[str] = None

    # Bio
    birth_date: Optional[str] = None
    birth_place: Optional[str] = None

    # College
    college_team: Optional[str] = None

    # Sophomore (2018) stats
    stats_2018_passing_yards: Optional[str] = None
    stats_2018_touchdowns: Optional[str] = None
    stats_2018_yards_record_flag: Optional[bool] = None
    stats_2018_tds_record_flag: Optional[bool] = None

    # NFL Draft details
    draft_year: Optional[str] = None
    draft_round: Optional[str] = None
    draft_pick_overall: Optional[str] = None
    drafted_by_team: Optional[str] = None

    # Current status (2024 season)
    current_2024_starter_statement: Optional[str] = None  # e.g., "starting QB", "starter", or null

    # Reference URLs grouped by purpose
    ref_bio_urls: List[str] = Field(default_factory=list)
    ref_college_stats_urls: List[str] = Field(default_factory=list)
    ref_nfl_career_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_quarterback_info() -> str:
    return """
    Extract the quarterback information and reference URLs exactly as presented in the answer.

    Required fields (return null when missing):
    - full_name: the quarterback’s full name.
    - birth_date: birth date string as it appears (e.g., "November 2, 1998").
    - birth_place: city and state (e.g., "Bakersfield, California").
    - college_team: college program/school name (e.g., "Utah State University" or "Utah State").
    - stats_2018_passing_yards: the stated passing yards for the 2018 sophomore season (string; allow comma formatting).
    - stats_2018_touchdowns: the stated TD passes for the 2018 sophomore season (string).
    - stats_2018_yards_record_flag: true/false if the answer explicitly claims this is a Utah State single-season record for passing yards; null if not stated.
    - stats_2018_tds_record_flag: true/false if the answer explicitly claims this is a Utah State single-season record for touchdown passes; null if not stated.
    - draft_year: e.g., "2020".
    - draft_round: e.g., "first", "1st", "round 1".
    - draft_pick_overall: e.g., "26" or "26th".
    - drafted_by_team: e.g., "Green Bay Packers".
    - current_2024_starter_statement: a short phrase indicating the 2024 starting QB status (e.g., "starter", "starting quarterback"); return null if not specified.

    Reference URLs (extract only actual URLs mentioned in the answer):
    - ref_bio_urls: URLs that verify birth date and/or birth place.
    - ref_college_stats_urls: URLs that verify 2018 sophomore season statistics and record claims at Utah State.
    - ref_nfl_career_urls: URLs that verify NFL draft details and/or 2024 starting status.

    Rules:
    - Return only information explicitly present in the answer.
    - Extract URLs in any format (plain URL or markdown link); capture the actual URL string.
    - Do not infer missing values; use nulls or empty arrays as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(extracted: QuarterbackExtraction) -> str:
    return (extracted.full_name or "").strip() or "the quarterback"

def _norm_num_str(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.strip()

def _merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst:
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, data: QuarterbackExtraction) -> Dict[str, Any]:
    """
    Build the verification tree according to the rubric and run all checks.
    Returns a dict with some handles in case extra prerequisites are needed.
    """
    # Create critical parent node for the entire task under root
    task_node = evaluator.add_parallel(
        id="Identify_Quarterback",
        desc="Identify the NFL quarterback satisfying all specified biographical and career constraints and provide verifying reference URLs.",
        parent=root_node,
        critical=True
    )

    # 1) Quarterback Full Name (existence check)
    evaluator.add_custom_node(
        result=bool((data.full_name or "").strip()),
        id="Quarterback_Full_Name",
        desc="Provide the quarterback's full name.",
        parent=task_node,
        critical=True
    )

    # 2) Biographical Constraints (with gating on bio sources)
    bio_node = evaluator.add_parallel(
        id="Biographical_Constraints",
        desc="Verify required birth details.",
        parent=task_node,
        critical=True
    )

    bio_sources_provided = evaluator.add_custom_node(
        result=(len(data.ref_bio_urls) > 0),
        id="Bio_Sources_Provided",
        desc="At least one biographical reference URL is provided.",
        parent=bio_node,
        critical=True
    )

    # Birth Date
    birth_date_leaf = evaluator.add_leaf(
        id="Birth_Date",
        desc="Quarterback was born on November 2, 1998.",
        parent=bio_node,
        critical=True
    )
    bd_name = _safe_name(data)
    birth_date_claim = f"{bd_name} was born on November 2, 1998."
    await evaluator.verify(
        claim=birth_date_claim,
        node=birth_date_leaf,
        sources=data.ref_bio_urls,
        additional_instruction="Verify the birth date on the referenced biographical page(s). Allow standard date formatting variants.",
    )

    # Birth Place
    birth_place_leaf = evaluator.add_leaf(
        id="Birth_Place",
        desc="Quarterback was born in Bakersfield, California.",
        parent=bio_node,
        critical=True
    )
    birth_place_claim = f"{bd_name} was born in Bakersfield, California."
    await evaluator.verify(
        claim=birth_place_claim,
        node=birth_place_leaf,
        sources=data.ref_bio_urls,
        additional_instruction="Verify the birthplace on the referenced biographical page(s). Minor wording variations are acceptable.",
    )

    # 3) College Constraint (played at Utah State)
    college_node = evaluator.add_parallel(
        id="College_Constraint",
        desc="Verify required college team/school.",
        parent=task_node,
        critical=True
    )

    college_sources_provided = evaluator.add_custom_node(
        result=(len(data.ref_college_stats_urls) > 0 or len(data.ref_bio_urls) > 0 or len(data.ref_nfl_career_urls) > 0),
        id="College_Sources_Provided",
        desc="At least one reference URL that could verify college affiliation is provided.",
        parent=college_node,
        critical=True
    )

    college_leaf = evaluator.add_leaf(
        id="Played_At_Utah_State",
        desc="Quarterback played college football at Utah State University.",
        parent=college_node,
        critical=True
    )
    college_claim = f"{bd_name} played college football at Utah State University."
    college_sources = _merge_sources(data.ref_college_stats_urls, data.ref_bio_urls, data.ref_nfl_career_urls)
    await evaluator.verify(
        claim=college_claim,
        node=college_leaf,
        sources=college_sources,
        additional_instruction="Accept 'Utah State' or 'Utah State University' as equivalent. Verify via college, team bio, or reliable career pages.",
    )

    # 4) College 2018 Sophomore Stats Constraints
    stats_node = evaluator.add_parallel(
        id="College_2018_Sophomore_Stats_Constraints",
        desc="Verify required 2018 sophomore season passing statistics and record claims at Utah State.",
        parent=task_node,
        critical=True
    )

    stats_sources_provided = evaluator.add_custom_node(
        result=(len(data.ref_college_stats_urls) > 0),
        id="Stats_Sources_Provided",
        desc="At least one college statistics reference URL is provided.",
        parent=stats_node,
        critical=True
    )

    # Passed for 3,567 yards (record)
    yards_leaf = evaluator.add_leaf(
        id="Passed_For_3567_Yards_Record",
        desc="As a sophomore in the 2018 season, quarterback passed for 3,567 yards and it was a Utah State single-season record.",
        parent=stats_node,
        critical=True
    )
    yards_str = _norm_num_str(data.stats_2018_passing_yards) or "3,567"
    yards_claim = (
        f"As a sophomore in the 2018 season at Utah State, {bd_name} passed for {yards_str} yards, "
        "which was a Utah State single-season record."
    )
    await evaluator.verify(
        claim=yards_claim,
        node=yards_leaf,
        sources=data.ref_college_stats_urls,
        additional_instruction="Verify the 2018 passing yards total and that it is stated as a Utah State single-season record. Allow comma/spacing variations in the number.",
    )

    # Threw 32 TDs (record)
    tds_leaf = evaluator.add_leaf(
        id="Threw_32_TDs_Record",
        desc="As a sophomore in the 2018 season, quarterback threw 32 touchdown passes and it was a Utah State single-season record.",
        parent=stats_node,
        critical=True
    )
    tds_str = _norm_num_str(data.stats_2018_touchdowns) or "32"
    tds_claim = (
        f"As a sophomore in the 2018 season at Utah State, {bd_name} threw {tds_str} touchdown passes, "
        "which was a Utah State single-season record."
    )
    await evaluator.verify(
        claim=tds_claim,
        node=tds_leaf,
        sources=data.ref_college_stats_urls,
        additional_instruction="Verify the 2018 touchdown passes total and that it is stated as a Utah State single-season record.",
    )

    # 5) NFL Draft Constraints
    draft_node = evaluator.add_parallel(
        id="NFL_Draft_Constraints",
        desc="Verify required NFL Draft details.",
        parent=task_node,
        critical=True
    )

    nfl_sources_provided = evaluator.add_custom_node(
        result=(len(data.ref_nfl_career_urls) > 0),
        id="NFL_Sources_Provided",
        desc="At least one NFL career/draft reference URL is provided.",
        parent=draft_node,
        critical=True
    )

    # Draft year 2020
    draft_year_leaf = evaluator.add_leaf(
        id="Draft_Year_2020",
        desc="Quarterback was drafted in the 2020 NFL Draft.",
        parent=draft_node,
        critical=True
    )
    draft_year_claim = f"{bd_name} was drafted in the 2020 NFL Draft."
    await evaluator.verify(
        claim=draft_year_claim,
        node=draft_year_leaf,
        sources=data.ref_nfl_career_urls,
        additional_instruction="Verify the draft year on reliable NFL draft or team pages."
    )

    # Draft round first
    draft_round_leaf = evaluator.add_leaf(
        id="Draft_Round_First",
        desc="Quarterback was selected in the first round of the NFL Draft.",
        parent=draft_node,
        critical=True
    )
    draft_round_claim = f"{bd_name} was selected in the first round of the NFL Draft."
    await evaluator.verify(
        claim=draft_round_claim,
        node=draft_round_leaf,
        sources=data.ref_nfl_career_urls,
        additional_instruction="Allow variants like 'Round 1' or '1st round'. Verify via draft records."
    )

    # Draft pick 26th overall
    draft_pick_leaf = evaluator.add_leaf(
        id="Draft_Pick_26th_Overall",
        desc="Quarterback was selected with the 26th overall pick.",
        parent=draft_node,
        critical=True
    )
    draft_pick_claim = f"{bd_name} was selected with the 26th overall pick."
    await evaluator.verify(
        claim=draft_pick_claim,
        node=draft_pick_leaf,
        sources=data.ref_nfl_career_urls,
        additional_instruction="Verify the overall pick number on draft records."
    )

    # Drafted by Packers
    drafted_by_leaf = evaluator.add_leaf(
        id="Drafted_By_Packers",
        desc="Quarterback was drafted by the Green Bay Packers.",
        parent=draft_node,
        critical=True
    )
    drafted_by_claim = f"{bd_name} was drafted by the Green Bay Packers."
    await evaluator.verify(
        claim=drafted_by_claim,
        node=drafted_by_leaf,
        sources=data.ref_nfl_career_urls,
        additional_instruction="Verify the drafting team on draft records or team announcements."
    )

    # 6) Current 2024 Status (starting QB for Packers)
    current_status_leaf = evaluator.add_leaf(
        id="Current_2024_Status",
        desc="As of the 2024 NFL season, quarterback is the starting quarterback for the Green Bay Packers.",
        parent=task_node,
        critical=True
    )
    current_claim = f"As of the 2024 NFL season, {bd_name} is the starting quarterback for the Green Bay Packers."
    # Gate by NFL sources provided via extra prerequisites
    await evaluator.verify(
        claim=current_claim,
        node=current_status_leaf,
        sources=data.ref_nfl_career_urls,
        additional_instruction="Verify starter status using reliable team depth chart, official team pages, or credible season previews/recaps.",
        extra_prerequisites=[nfl_sources_provided]
    )

    # 7) Reference URLs existence checks (explicit)
    refs_node = evaluator.add_parallel(
        id="Reference_URLs",
        desc="Provide reference URLs that verify key biographical information, college statistics, and NFL career details.",
        parent=task_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(data.ref_bio_urls) > 0),
        id="Reference_URL_Biographical",
        desc="Provide at least one reference URL verifying birth date and/or birth place.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(data.ref_college_stats_urls) > 0),
        id="Reference_URL_College_Stats",
        desc="Provide at least one reference URL verifying the 2018 college statistics/records.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(data.ref_nfl_career_urls) > 0),
        id="Reference_URL_NFL_Career",
        desc="Provide at least one reference URL verifying NFL draft and/or 2024 starting status.",
        parent=refs_node,
        critical=True
    )

    return {
        "task_node_id": task_node.id,
        "bio_sources_provided_id": bio_sources_provided.id,
        "nfl_sources_provided_id": nfl_sources_provided.id,
        "stats_sources_provided_id": stats_sources_provided.id,
        "college_sources_provided_id": college_sources_provided.id,
    }


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
    Evaluate an answer for the 'identify_nfl_qb_2024_packers' task.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_quarterback_info(),
        template_class=QuarterbackExtraction,
        extraction_name="quarterback_info"
    )

    # Record ground truth (for reference only)
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH
    }, gt_type="expected_info")

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return standard summary
    return evaluator.get_summary()