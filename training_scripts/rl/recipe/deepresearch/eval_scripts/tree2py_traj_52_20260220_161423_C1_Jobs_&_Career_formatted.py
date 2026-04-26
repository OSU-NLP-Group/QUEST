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
TASK_ID = "school_closing_decision_process"
TASK_DESCRIPTION = "What is the typical time range during which school district superintendents make decisions about school closures or delays due to weather, and which key personnel do they consult with as part of this decision-making process?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DecisionProcessExtraction(BaseModel):
    typical_time_range: Optional[str] = None
    timeline_sources: List[str] = Field(default_factory=list)
    consultation_roles: List[str] = Field(default_factory=list)
    consultation_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_decision_process() -> str:
    return """
    Extract the following from the answer:

    1) typical_time_range:
       - The explicit time of day/hours window during which superintendents typically make weather-related closure or delay decisions (e.g., "between 4:00–6:00 a.m.", "by 5 a.m.", "early morning hours before 6 a.m.").
       - Return the phrase exactly as it appears in the answer. If not stated, return null.

    2) timeline_sources:
       - All URLs that the answer cites as evidence for the timing of the decision (when the decision is made).
       - These can be inline links or listed in a sources/references section.
       - If none are provided for the timing claim, return an empty list.

    3) consultation_roles:
       - The roles/titles of key personnel that the answer says the superintendent consults with as part of the decision-making process.
       - Extract each role as a string (e.g., "transportation supervisor", "transportation director", "transportation department", "police", "road crews", etc.).
       - If none are stated, return an empty list.

    4) consultation_sources:
       - All URLs that the answer cites as evidence for the consultation with personnel (e.g., transportation supervisor/director).
       - Return an empty list if none are provided.

    Be precise and do not invent information not present in the answer. Only include URLs that are actually present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    unique = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _has_transportation_role(roles: List[str]) -> bool:
    """
    Check if any extracted role clearly indicates transportation leadership/staff,
    such as 'transportation supervisor', 'transportation director', 'transportation department', etc.
    """
    if not roles:
        return False
    keywords_any = ["transport", "transportation"]
    leadership_synonyms = [
        "supervisor", "director", "manager", "coordinator", "chief",
        "head", "department", "services", "operations", "office", "team", "staff"
    ]
    for r in roles:
        rl = r.lower()
        if any(k in rl for k in keywords_any) and any(s in rl for s in leadership_synonyms):
            return True
    return False


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
    Evaluate an answer for the school closing decision process task.
    Builds a verification tree with two critical checks:
    - Decision_Timeline
    - Consultation_Party
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

    # Add a named top-level node per rubric
    process_node = evaluator.add_parallel(
        id="School_Closing_Decision_Process",
        desc="Identifies the typical time range when school superintendents make closing decisions and the key personnel they consult with during this process",
        parent=root,
        critical=False
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_decision_process(),
        template_class=DecisionProcessExtraction,
        extraction_name="decision_process_extraction"
    )

    # Clean up URL lists
    timeline_urls = _dedupe_urls(extracted.timeline_sources or [])
    consult_urls = _dedupe_urls(extracted.consultation_sources or [])

    # --------------------------------------------------------------------- #
    # Leaf 1: Decision_Timeline (CRITICAL)
    # --------------------------------------------------------------------- #
    has_time_text = extracted.typical_time_range is not None and extracted.typical_time_range.strip() != ""
    has_time_sources = len(timeline_urls) > 0

    if has_time_text and has_time_sources:
        dt_node = evaluator.add_leaf(
            id="Decision_Timeline",
            desc="Provides the typical time range (early morning hours) when superintendents make school closing or delay decisions",
            parent=process_node,
            critical=True
        )
        claim_timeline = f"School district superintendents typically make weather-related school closing or delay decisions during {extracted.typical_time_range}."
        await evaluator.verify(
            claim=claim_timeline,
            node=dt_node,
            sources=timeline_urls,
            additional_instruction=(
                "Verify that the provided webpage(s) explicitly indicate the decision time window for weather-related school closures/delays. "
                "Allow reasonable wording variations (e.g., 'by 5 a.m.', 'between 4 and 6 a.m.', 'before 6 a.m.', 'early morning hours'). "
                f"If the sources do not clearly support the specific time window '{extracted.typical_time_range}', mark as not supported."
            )
        )
    else:
        # Fail the leaf if the answer lacks explicit time or lacks supporting URLs (source-grounding policy)
        evaluator.add_custom_node(
            result=False,
            id="Decision_Timeline",
            desc="Provides the typical time range (early morning hours) when superintendents make school closing or delay decisions",
            parent=process_node,
            critical=True
        )

    # --------------------------------------------------------------------- #
    # Leaf 2: Consultation_Party (CRITICAL)
    # --------------------------------------------------------------------- #
    mentions_transportation = _has_transportation_role(extracted.consultation_roles or [])
    has_consult_sources = len(consult_urls) > 0

    if mentions_transportation and has_consult_sources:
        cp_node = evaluator.add_leaf(
            id="Consultation_Party",
            desc="Identifies that superintendents consult with the transportation supervisor/director as part of the decision-making process",
            parent=process_node,
            critical=True
        )
        claim_consult = (
            "As part of making weather-related school closure or delay decisions, school district superintendents consult with the "
            "transportation supervisor or transportation director (i.e., the leader of the district transportation department)."
        )
        await evaluator.verify(
            claim=claim_consult,
            node=cp_node,
            sources=consult_urls,
            additional_instruction=(
                "Accept synonyms such as 'transportation director', 'transportation supervisor', 'director of transportation', "
                "'transportation department leadership'. The webpage(s) should clearly indicate that the superintendent consults "
                "with transportation leadership/staff as part of the decision-making process."
            )
        )
    else:
        # Fail the leaf if the answer doesn't state transportation consultation or lacks sources
        evaluator.add_custom_node(
            result=False,
            id="Consultation_Party",
            desc="Identifies that superintendents consult with the transportation supervisor/director as part of the decision-making process",
            parent=process_node,
            critical=True
        )

    # Return the evaluation summary
    return evaluator.get_summary()