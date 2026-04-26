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
TASK_ID = "march_2026_total_lunar_eclipse"
TASK_DESCRIPTION = (
    "In 2026, a total lunar eclipse will be visible from North America in early March. "
    "What is the duration of totality (the period when the Moon is fully within Earth's umbra) for this eclipse?"
)

# Ground truth expectations (for summary only; not directly used for scoring)
GROUND_TRUTH = {
    "duration_totality_expected": "58 minutes",
    "eclipse_date_expected": "March 3, 2026 (UTC)",
    "visibility_expected": "Visible from North America; Not visible from Europe or Africa",
    "next_total_lunar_eclipse_after": "Dec 31, 2028 – Jan 1, 2029",
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class EclipseAnswerExtraction(BaseModel):
    """
    Extract key statements and all cited URLs from the answer.
    """
    totality_duration: Optional[str] = None
    eclipse_date: Optional[str] = None
    visibility_na_statement: Optional[str] = None
    visibility_eu_africa_statement: Optional[str] = None
    last_total_eclipse_until: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_info() -> str:
    return """
    Extract the following fields exactly as stated in the answer (use null if missing):
    - totality_duration: the stated duration of totality for the March 2026 total lunar eclipse (e.g., "58 minutes", "~58 min", "0 h 58 m").
    - eclipse_date: the stated date of the eclipse (e.g., "March 3, 2026", "3 March 2026").
    - visibility_na_statement: the exact phrase/sentence that indicates the eclipse is visible from North America, if present; else null.
    - visibility_eu_africa_statement: the exact phrase/sentence that indicates the eclipse is NOT visible from Europe or Africa, if present; else null.
    - last_total_eclipse_until: the exact phrase/sentence that indicates this is the last total lunar eclipse until Dec 31, 2028–Jan 1, 2029, if present; else null.
    - source_urls: an array of all URLs explicitly included in the answer text (markdown links or plain URLs). Return only valid, absolute URLs with protocol.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, urls: List[str]) -> None:
    """
    Create verification nodes and run checks based on the rubric.
    This function assumes the evaluator has been initialized.
    """
    # Create a top-level critical node to mirror the rubric "root"
    root_criteria = evaluator.add_parallel(
        id="March_2026_Lunar_Eclipse_Totality_Duration",
        desc="Answer satisfies all stated constraints for the March 3, 2026 total lunar eclipse and provides the duration of totality.",
        parent=evaluator.root,
        critical=True,
    )

    # ------------------ Totality Duration (58 minutes) ------------------ #
    totality_group = evaluator.add_parallel(
        id="Totality_Duration",
        desc="States that the duration of totality is 58 minutes.",
        parent=root_criteria,
        critical=True
    )

    # Check that the answer explicitly states "58 minutes"
    totality_answer_leaf = evaluator.add_leaf(
        id="totality_answer_states_58",
        desc="The answer explicitly states that the duration of totality is 58 minutes (allowing minor paraphrase, e.g., ~58 min, 0h 58m).",
        parent=totality_group,
        critical=True
    )

    # Gate: at least one source URL is provided for totality support
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="totality_urls_present",
        desc="At least one source URL is provided for the totality duration claim.",
        parent=totality_group,
        critical=True
    )

    # Sources support "58 minutes"
    totality_sources_leaf = evaluator.add_leaf(
        id="totality_supported_by_sources",
        desc="Cited sources support that totality duration is 58 minutes.",
        parent=totality_group,
        critical=True
    )

    # ------------------------- Eclipse Date ----------------------------- #
    date_group = evaluator.add_parallel(
        id="Eclipse_Date",
        desc="Identifies the eclipse date as March 3, 2026.",
        parent=root_criteria,
        critical=True
    )

    date_answer_leaf = evaluator.add_leaf(
        id="date_answer_states_march_3_2026",
        desc="The answer identifies the eclipse date as March 3, 2026.",
        parent=date_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="date_urls_present",
        desc="At least one source URL is provided for the eclipse date claim.",
        parent=date_group,
        critical=True
    )

    date_sources_leaf = evaluator.add_leaf(
        id="date_supported_by_sources",
        desc="Cited sources support that the eclipse occurs on March 3, 2026 (UTC).",
        parent=date_group,
        critical=True
    )

    # -------------------- Visibility Constraint ------------------------- #
    visibility_group = evaluator.add_parallel(
        id="Visibility_Constraint",
        desc="States the eclipse is visible from North America and NOT visible from Europe or Africa.",
        parent=root_criteria,
        critical=True
    )

    # Global visibility gating by sources for this group
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="visibility_urls_present",
        desc="At least one source URL is provided for the visibility claims.",
        parent=visibility_group,
        critical=True
    )

    # Subgroup: Visible from North America
    vis_na_group = evaluator.add_parallel(
        id="Visibility_North_America",
        desc="Visibility in North America.",
        parent=visibility_group,
        critical=True
    )

    vis_na_answer_leaf = evaluator.add_leaf(
        id="visibility_na_answer_states",
        desc="The answer states that the eclipse is visible from North America.",
        parent=vis_na_group,
        critical=True
    )
    vis_na_sources_leaf = evaluator.add_leaf(
        id="visibility_na_supported_by_sources",
        desc="Cited sources support that the eclipse is visible from North America.",
        parent=vis_na_group,
        critical=True
    )

    # Subgroup: NOT visible from Europe or Africa
    vis_eu_af_group = evaluator.add_parallel(
        id="Not_Visible_Europe_Africa",
        desc="Non-visibility in Europe and Africa.",
        parent=visibility_group,
        critical=True
    )

    vis_eu_af_answer_leaf = evaluator.add_leaf(
        id="visibility_not_eu_af_answer_states",
        desc="The answer states that the eclipse is NOT visible from Europe or Africa.",
        parent=vis_eu_af_group,
        critical=True
    )
    vis_eu_af_sources_leaf = evaluator.add_leaf(
        id="visibility_not_eu_af_supported_by_sources",
        desc="Cited sources support that the eclipse is NOT visible from Europe or Africa.",
        parent=vis_eu_af_group,
        critical=True
    )

    # ------------- Last total eclipse until 2028–2029 ------------------- #
    last_total_group = evaluator.add_parallel(
        id="Last_Total_Eclipse_Until_2028_2029",
        desc="States this is the last total lunar eclipse until December 31, 2028–January 1, 2029.",
        parent=root_criteria,
        critical=True
    )

    last_total_answer_leaf = evaluator.add_leaf(
        id="last_total_answer_states_until_2028_2029",
        desc="The answer states this is the last total lunar eclipse until Dec 31, 2028 – Jan 1, 2029.",
        parent=last_total_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="last_total_urls_present",
        desc="At least one source URL is provided for the 'last total eclipse until 2028–2029' claim.",
        parent=last_total_group,
        critical=True
    )

    last_total_sources_leaf = evaluator.add_leaf(
        id="last_total_supported_by_sources",
        desc="Cited sources support that the next total lunar eclipse after March 3, 2026 occurs on Dec 31, 2028 – Jan 1, 2029.",
        parent=last_total_group,
        critical=True
    )

    # ------------------------- Run verifications ------------------------- #
    claims_and_sources: List[tuple] = []

    # Totality - answer states
    claims_and_sources.append((
        "The answer explicitly states that the duration of totality (the time the Moon is fully within Earth's umbra) is 58 minutes. "
        "Allow minor paraphrases like '~58 minutes', '≈58 min', or '0h 58m'. If the answer does not assert this, mark Incorrect.",
        None,
        totality_answer_leaf,
        "Focus solely on whether the statement is asserted in the answer text. Ignore external facts here."
    ))
    # Totality - sources support
    claims_and_sources.append((
        "For the March 3, 2026 total lunar eclipse, the duration of totality (the time fully within Earth's umbra) is 58 minutes.",
        urls,
        totality_sources_leaf,
        "Rely only on the provided URLs. Accept minor rounding or equivalent expressions (e.g., ~58 minutes, 0h 58m). "
        "If none of the URLs clearly support 58 minutes, mark as not supported."
    ))

    # Date - answer states
    claims_and_sources.append((
        "The answer identifies the eclipse date as March 3, 2026.",
        None,
        date_answer_leaf,
        "Judge only whether the answer text asserts March 3, 2026 as the date (format variations like '3 March 2026' are acceptable)."
    ))
    # Date - sources support
    claims_and_sources.append((
        "The total lunar eclipse occurs on March 3, 2026 in UTC.",
        urls,
        date_sources_leaf,
        "Rely only on the provided URLs. Accept formatting variations like '3 March 2026' or '2026-03-03'. "
        "Local timezones may span adjacent calendar dates, but the canonical/UTC date should be March 3, 2026."
    ))

    # Visibility NA - answer states
    claims_and_sources.append((
        "The answer states that the eclipse is visible from North America.",
        None,
        vis_na_answer_leaf,
        "Check only if the answer text claims visibility in North America (allow paraphrases)."
    ))
    # Visibility NA - sources support
    claims_and_sources.append((
        "This March 3, 2026 total lunar eclipse is visible from North America.",
        urls,
        vis_na_sources_leaf,
        "Rely only on the provided URLs. Use visibility maps/regions on the page to determine if North America can view the eclipse."
    ))

    # Visibility NOT EU/Africa - answer states
    claims_and_sources.append((
        "The answer states that the eclipse is not visible from Europe or Africa.",
        None,
        vis_eu_af_answer_leaf,
        "Check only if the answer text claims non-visibility in Europe and in Africa (allow paraphrases)."
    ))
    # Visibility NOT EU/Africa - sources support
    claims_and_sources.append((
        "This March 3, 2026 total lunar eclipse is not visible from Europe or Africa.",
        urls,
        vis_eu_af_sources_leaf,
        "Rely only on the provided URLs. Use the visibility coverage described or mapped on the page. "
        "If any substantial portion of Europe or Africa can see the eclipse, mark as not supported."
    ))

    # Last total eclipse until 2028–2029 - answer states
    claims_and_sources.append((
        "The answer states that this is the last total lunar eclipse until December 31, 2028 – January 1, 2029.",
        None,
        last_total_answer_leaf,
        "Check only whether the answer asserts this specific 'last until' claim (allow minor formatting differences in the date range)."
    ))
    # Last total eclipse until 2028–2029 - sources support
    claims_and_sources.append((
        "The next total lunar eclipse after the March 3, 2026 event occurs on December 31, 2028 – January 1, 2029.",
        urls,
        last_total_sources_leaf,
        "Rely only on the provided URLs. If the timeline on the page lists the next total lunar eclipse at the 2028–2029 new year boundary, mark supported."
    ))

    # Execute all verifications (parallelized)
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the March 2026 total lunar eclipse totality duration task.
    """
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

    # Extract structured information and sources from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_eclipse_info(),
        template_class=EclipseAnswerExtraction,
        extraction_name="eclipse_answer_extraction",
    )

    # Add ground truth info (for transparency in summary)
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH,
            "notes": "These expectations are used only for human-readable context in the summary, not as automatic scoring."
        },
        gt_type="ground_truth_expectations"
    )

    # Add some custom info (e.g., number of URLs)
    evaluator.add_custom_info(
        {
            "total_extracted_urls": len(extraction.source_urls or []),
            "sample_urls": (extraction.source_urls or [])[:5],
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction.source_urls or [])

    # Return final summary
    return evaluator.get_summary()