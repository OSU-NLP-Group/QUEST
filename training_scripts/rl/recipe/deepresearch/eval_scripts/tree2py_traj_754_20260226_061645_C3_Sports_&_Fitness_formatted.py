import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "unc_qb_2022_sb_lx_record"
TASK_DESCRIPTION = (
    "Identify the quarterback who, during their 2022 college season at the University of North Carolina, "
    "threw for more than 4,000 passing yards and at least 35 passing touchdowns, won the ACC Player of the Year "
    "award in 2022, and finished in the top 10 of the Heisman Trophy voting in 2022. This quarterback was then "
    "selected within the top 5 picks of the first round of the 2024 NFL Draft, was selected to the Pro Bowl in "
    "their rookie season (2024-2025), led their team to Super Bowl LX on February 8, 2026, and set an NFL record "
    "during that Super Bowl for most passing yards in a single quarter of a Super Bowl. What is the name of this "
    "quarterback, and how many passing yards did they throw for in the 4th quarter of Super Bowl LX to set this record?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QBExtraction(BaseModel):
    quarterback_name: Optional[str] = None
    fourth_quarter_passing_yards: Optional[str] = None

    # Per-criterion cited source URLs (if any) explicitly provided in the answer
    sources_unc_2022: List[str] = Field(default_factory=list)
    sources_passing_yards_2022: List[str] = Field(default_factory=list)
    sources_passing_tds_2022: List[str] = Field(default_factory=list)
    sources_acc_poy_2022: List[str] = Field(default_factory=list)
    sources_heisman_top10_2022: List[str] = Field(default_factory=list)
    sources_drafted_top5_2024: List[str] = Field(default_factory=list)
    sources_pro_bowl_rookie_2024_2025: List[str] = Field(default_factory=list)
    sources_led_super_bowl_lx: List[str] = Field(default_factory=list)
    sources_record_4th_qtr_sb_lx: List[str] = Field(default_factory=list)

    # Optional: any global sources section provided by the answer; extractor may use this
    all_other_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_qb_info() -> str:
    return """
    You must extract exactly the following fields from the answer text. Do not invent values.
    1) quarterback_name: The quarterback's full name given in the answer. If not present, return null.
    2) fourth_quarter_passing_yards: The number of passing yards the quarterback threw specifically in the 4th quarter of Super Bowl LX, exactly as stated in the answer (keep it as a string; do not add units; if missing, return null).

    3) For each of the following criteria, extract all URLs (if any) explicitly cited in the answer that support that specific claim. These must be actual URLs present in the answer (including URLs in markdown links). If the answer doesn't provide any URLs for a given criterion, return an empty list for that field.
       - sources_unc_2022: URLs supporting that the QB played for the University of North Carolina (UNC) in the 2022 college season.
       - sources_passing_yards_2022: URLs supporting that the QB threw more than 4,000 passing yards in 2022.
       - sources_passing_tds_2022: URLs supporting that the QB threw at least 35 passing TDs in 2022.
       - sources_acc_poy_2022: URLs supporting that the QB won ACC Player of the Year in 2022.
       - sources_heisman_top10_2022: URLs supporting that the QB finished in the top 10 of the 2022 Heisman voting.
       - sources_drafted_top5_2024: URLs supporting that the QB was selected within the top 5 picks of Round 1 in the 2024 NFL Draft.
       - sources_pro_bowl_rookie_2024_2025: URLs supporting that the QB was selected to the Pro Bowl in his rookie season (2024–2025).
       - sources_led_super_bowl_lx: URLs supporting that the QB led his team to Super Bowl LX (played on February 8, 2026).
       - sources_record_4th_qtr_sb_lx: URLs supporting that during Super Bowl LX the QB set an NFL record for most passing yards in a single quarter of a Super Bowl, and that the record quarter was the 4th quarter.

    4) all_other_sources: If the answer provides a general sources section or other supporting URLs that are not clearly mapped to the above criteria, include them here.

    Rules:
    - Extract only URLs that appear in the answer; do not infer or fabricate links.
    - If a single URL appears to support multiple criteria, you may include it in multiple lists.
    - If URLs are given without protocol, prepend http://.
    - Return exactly these fields in JSON.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _union_all_sources(ex: QBExtraction) -> List[str]:
    all_sources = []
    all_sources.extend(ex.sources_unc_2022 or [])
    all_sources.extend(ex.sources_passing_yards_2022 or [])
    all_sources.extend(ex.sources_passing_tds_2022 or [])
    all_sources.extend(ex.sources_acc_poy_2022 or [])
    all_sources.extend(ex.sources_heisman_top10_2022 or [])
    all_sources.extend(ex.sources_drafted_top5_2024 or [])
    all_sources.extend(ex.sources_pro_bowl_rookie_2024_2025 or [])
    all_sources.extend(ex.sources_led_super_bowl_lx or [])
    all_sources.extend(ex.sources_record_4th_qtr_sb_lx or [])
    all_sources.extend(ex.all_other_sources or [])
    return _dedup(all_sources)


def _pick_sources(primary: List[str], fallback: List[str]) -> Optional[List[str]]:
    if primary and len(primary) > 0:
        return _dedup(primary)
    if fallback and len(fallback) > 0:
        return _dedup(fallback)
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(
    evaluator: Evaluator,
    parent_node,
    ex: QBExtraction,
) -> None:
    """
    Build the 'criteria_verification' parallel critical node and verify each criterion as leaf nodes.
    """
    criteria_node = evaluator.add_parallel(
        id="criteria_verification",
        desc="Verify the identified quarterback satisfies all stated college/NFL/Super Bowl criteria from the question.",
        parent=parent_node,
        critical=True
    )

    name_for_claim = ex.quarterback_name or "the quarterback identified in the answer"
    all_sources = _union_all_sources(ex)

    # Prepare leaf nodes
    node_unc = evaluator.add_leaf(
        id="unc_2022_participation",
        desc="Quarterback played for the University of North Carolina during the 2022 college football season.",
        parent=criteria_node,
        critical=True,
    )
    node_yds = evaluator.add_leaf(
        id="passing_yards_over_4000_2022",
        desc="During the 2022 season, quarterback threw for more than 4,000 passing yards.",
        parent=criteria_node,
        critical=True,
    )
    node_tds = evaluator.add_leaf(
        id="passing_tds_at_least_35_2022",
        desc="During the 2022 season, quarterback threw for at least 35 passing touchdowns.",
        parent=criteria_node,
        critical=True,
    )
    node_acc = evaluator.add_leaf(
        id="acc_player_of_year_2022",
        desc="Quarterback won the ACC Player of the Year award in 2022.",
        parent=criteria_node,
        critical=True,
    )
    node_heisman = evaluator.add_leaf(
        id="heisman_top_10_2022",
        desc="Quarterback finished in the top 10 of the Heisman Trophy voting in 2022.",
        parent=criteria_node,
        critical=True,
    )
    node_draft = evaluator.add_leaf(
        id="drafted_top_5_first_round_2024",
        desc="Quarterback was selected within the top 5 picks of the first round of the 2024 NFL Draft.",
        parent=criteria_node,
        critical=True,
    )
    node_pro_bowl = evaluator.add_leaf(
        id="pro_bowl_rookie_season_2024_2025",
        desc="Quarterback was selected to the Pro Bowl during their rookie season (2024–2025).",
        parent=criteria_node,
        critical=True,
    )
    node_sb_lx = evaluator.add_leaf(
        id="led_team_to_super_bowl_lx",
        desc="Quarterback led their team to Super Bowl LX (played on February 8, 2026).",
        parent=criteria_node,
        critical=True,
    )
    node_record_q4 = evaluator.add_leaf(
        id="set_single_quarter_passing_yards_record_in_4th_quarter_sb_lx",
        desc="During Super Bowl LX, quarterback set an NFL record for most passing yards in a single quarter, and this record-setting quarter was the 4th quarter.",
        parent=criteria_node,
        critical=True,
    )

    # Build claims and sources
    claims_and_sources = [
        (
            f"{name_for_claim} played for the University of North Carolina (UNC) during the 2022 college football season.",
            _pick_sources(ex.sources_unc_2022, all_sources),
            node_unc,
            "Allow 'UNC', 'North Carolina', or 'UNC Tar Heels' as equivalent. Confirm the 2022 season team affiliation."
        ),
        (
            f"In the 2022 season, {name_for_claim} threw for more than 4,000 passing yards (strictly greater than 4,000).",
            _pick_sources(ex.sources_passing_yards_2022, all_sources),
            node_yds,
            "Confirm the 2022 passing yards and ensure the value is > 4000 (not equal)."
        ),
        (
            f"In the 2022 season, {name_for_claim} threw at least 35 passing touchdowns.",
            _pick_sources(ex.sources_passing_tds_2022, all_sources),
            node_tds,
            "Confirm the 2022 passing touchdowns and ensure it is >= 35."
        ),
        (
            f"{name_for_claim} won the ACC Player of the Year award in 2022.",
            _pick_sources(ex.sources_acc_poy_2022, all_sources),
            node_acc,
            "Accept equivalent naming such as 'ACC Player of the Year' or 'ACC POY' in 2022."
        ),
        (
            f"{name_for_claim} finished in the top ten of the Heisman Trophy voting in 2022.",
            _pick_sources(ex.sources_heisman_top10_2022, all_sources),
            node_heisman,
            "Check the final 2022 Heisman voting results; rank 1–10 qualifies."
        ),
        (
            f"{name_for_claim} was selected within the top 5 picks (1–5) of the first round of the 2024 NFL Draft.",
            _pick_sources(ex.sources_drafted_top5_2024, all_sources),
            node_draft,
            "Verify that it was Round 1 in the 2024 NFL Draft and the overall pick number was 1–5."
        ),
        (
            f"{name_for_claim} was selected to the Pro Bowl in his rookie season (the 2024–2025 NFL year).",
            _pick_sources(ex.sources_pro_bowl_rookie_2024_2025, all_sources),
            node_pro_bowl,
            "Confirm that the player earned a Pro Bowl selection tied to his rookie season."
        ),
        (
            f"{name_for_claim} led his team to Super Bowl LX.",
            _pick_sources(ex.sources_led_super_bowl_lx, all_sources),
            node_sb_lx,
            "Confirm that his team reached and played in Super Bowl LX (held on February 8, 2026)."
        ),
        (
            f"During Super Bowl LX, {name_for_claim} set an NFL record for most passing yards in a single quarter of a Super Bowl, and the record came in the 4th quarter.",
            _pick_sources(ex.sources_record_4th_qtr_sb_lx, all_sources),
            node_record_q4,
            "Verify both parts: (1) it is an NFL record for most passing yards in a single Super Bowl quarter; (2) the record quarter was the 4th quarter."
        ),
    ]

    # Parallel verification for all criteria leaves
    await evaluator.batch_verify(claims_and_sources)


async def build_and_verify_answer_provision(
    evaluator: Evaluator,
    parent_node,
    ex: QBExtraction,
) -> None:
    """
    Build the 'answer_provision' parallel critical node that checks the answer explicitly includes
    the quarterback's name and the numeric 4th-quarter passing yards value.
    """
    answer_node = evaluator.add_parallel(
        id="answer_provision",
        desc="Provide the requested outputs (name and 4th-quarter passing yards).",
        parent=parent_node,
        critical=True
    )

    # Check name provided
    evaluator.add_custom_node(
        result=(ex.quarterback_name is not None and str(ex.quarterback_name).strip() != ""),
        id="quarterback_name",
        desc="Answer includes the quarterback's name.",
        parent=answer_node,
        critical=True
    )

    # Check 4th-quarter passing yards value provided (must contain at least one digit)
    yards_val = (ex.fourth_quarter_passing_yards or "").strip()
    has_digits = any(ch.isdigit() for ch in yards_val)
    evaluator.add_custom_node(
        result=bool(yards_val) and has_digits,
        id="fourth_quarter_passing_yards_value",
        desc="Answer includes the number of passing yards thrown in the 4th quarter of Super Bowl LX.",
        parent=answer_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point to evaluate an agent's answer for the UNC 2022 QB to Super Bowl LX record task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # We'll add the real task node beneath
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
    ex = await evaluator.extract(
        prompt=prompt_extract_qb_info(),
        template_class=QBExtraction,
        extraction_name="qb_extraction"
    )

    # Build top-level sequential critical node as specified by the rubric
    qb_task_node = evaluator.add_sequential(
        id="quarterback_identification_and_answer",
        desc="Identify the quarterback who satisfies all stated criteria and provide the quarterback name and the 4th-quarter passing yards in Super Bowl LX.",
        parent=root,
        critical=True
    )

    # First: verify all criteria (parallel, critical)
    await build_and_verify_criteria(evaluator, qb_task_node, ex)

    # Second: verify that the answer provides the requested outputs (parallel, critical)
    await build_and_verify_answer_provision(evaluator, qb_task_node, ex)

    # Return result summary with the verification tree and scores
    return evaluator.get_summary()