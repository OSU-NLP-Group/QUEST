import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yale_head_coach_2026"
TASK_DESCRIPTION = (
    "Who was hired as the head football coach at Yale University in February 2026, and can you verify that this person "
    "won the Eddie Robinson Award in 2025 and served as head football coach at Lehigh University immediately before this appointment?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoachExtraction(BaseModel):
    coach_name: Optional[str] = None
    yale_hire_sources: List[str] = Field(default_factory=list)
    award_sources: List[str] = Field(default_factory=list)
    lehigh_sources: List[str] = Field(default_factory=list)
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_info() -> str:
    return """
    From the answer, extract:
    1) coach_name: The full name of the person the answer claims Yale University hired as its head football coach in February 2026.
    2) yale_hire_sources: All URLs explicitly cited that support or announce Yale University hiring this person as head football coach in February 2026. Include official announcements, credible news coverage, or team/athletics pages.
    3) award_sources: All URLs explicitly cited that support that this person won the Eddie Robinson Award (FCS National Coach of the Year) in 2025. Include official award pages, press releases, or credible coverage.
    4) lehigh_sources: All URLs explicitly cited that support that immediately before joining Yale, this person served as the head football coach at Lehigh University. Include Yale announcements that mention 'comes from Lehigh', Lehigh releases, or credible coverage.
    5) all_urls: A list of all URLs mentioned anywhere in the answer (deduplicate but keep order). This serves as a fallback when specific lists are empty.

    Rules:
    - Only extract URLs that are explicitly present in the answer. Do not invent or infer URLs.
    - If a single URL supports multiple claims, include it in each relevant list.
    - If a field is missing from the answer, set it to null (for coach_name) or an empty list (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _gather_sources(primary: List[str], fallback_all: List[str]) -> List[str]:
    if primary:
        return _dedup_preserve_order(primary)
    return _dedup_preserve_order(fallback_all)


def _safe_name(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "the identified coach"


# --------------------------------------------------------------------------- #
# Verification building                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(evaluator: Evaluator, root: Any, info: CoachExtraction) -> None:
    """
    Build verification nodes according to the rubric and perform verifications.
    Root tree structure:
    - root (framework root, non-critical)
      - Coach_Identification_and_Verification (critical parallel)
         - Yale_Head_Coach_February_2026 (critical leaf)
         - Eddie_Robinson_Award_Winner (critical leaf)
         - Previous_Lehigh_Head_Coach (critical leaf)
    """
    # Parent node per rubric
    parent = evaluator.add_parallel(
        id="Coach_Identification_and_Verification",
        desc="Correctly identifies the newly hired Yale head football coach in February 2026 and verifies the required qualifications",
        parent=root,
        critical=True,
    )

    person = _safe_name(info.coach_name)

    # 1) Yale hire in February 2026
    yale_sources = _gather_sources(info.yale_hire_sources, info.all_urls)
    yale_desc = "Verifies that the identified person was hired as the head football coach at Yale University in February 2026"

    if not yale_sources:
        # No sources → treat as quality failure (source-grounding policy)
        evaluator.add_custom_node(
            result=False,
            id="Yale_Head_Coach_February_2026",
            desc=yale_desc,
            parent=parent,
            critical=True
        )
    else:
        yale_leaf = evaluator.add_leaf(
            id="Yale_Head_Coach_February_2026",
            desc=yale_desc,
            parent=parent,
            critical=True
        )
        yale_claim = f"In February 2026, Yale University hired {person} as its head football coach."
        await evaluator.verify(
            claim=yale_claim,
            node=yale_leaf,
            sources=yale_sources,
            additional_instruction=(
                "Verify that the sources explicitly indicate that Yale University (Yale Bulldogs) named/appointed "
                f"{person} as its head football coach in February 2026. Accept phrasing like 'named head coach', "
                "'appointed head football coach', or 'hired as head coach'. The timing must be in February 2026—"
                "use the article/publication date or explicit mention to confirm."
            )
        )

    # 2) Eddie Robinson Award (FCS National Coach of the Year) in 2025
    award_sources = _gather_sources(info.award_sources, info.all_urls)
    award_desc = "Verifies that the identified coach won the Eddie Robinson Award (FCS National Coach of the Year) in 2025"

    if not award_sources:
        evaluator.add_custom_node(
            result=False,
            id="Eddie_Robinson_Award_Winner",
            desc=award_desc,
            parent=parent,
            critical=True
        )
    else:
        award_leaf = evaluator.add_leaf(
            id="Eddie_Robinson_Award_Winner",
            desc=award_desc,
            parent=parent,
            critical=True
        )
        award_claim = f"{person} won the Eddie Robinson Award (FCS National Coach of the Year) in 2025."
        await evaluator.verify(
            claim=award_claim,
            node=award_leaf,
            sources=award_sources,
            additional_instruction=(
                "Confirm that the sources clearly state that the person won the Eddie Robinson Award for the 2025 season. "
                "This award is commonly referred to as the Stats Perform Eddie Robinson Award (FCS National Coach of the Year). "
                "Do not confuse with finalists or other awards; it must indicate 'won' in 2025."
            )
        )

    # 3) Immediately prior role: Lehigh head coach before Yale
    # Use union of lehigh_sources and yale_hire_sources because Yale hire announcements often mention prior role.
    lehigh_primary = list(info.lehigh_sources or [])
    lehigh_sources_union = _dedup_preserve_order(lehigh_primary + (info.yale_hire_sources or []) + (info.all_urls or []))
    lehigh_desc = "Verifies that the identified coach was the head football coach at Lehigh University immediately before joining Yale"

    if not lehigh_sources_union:
        evaluator.add_custom_node(
            result=False,
            id="Previous_Lehigh_Head_Coach",
            desc=lehigh_desc,
            parent=parent,
            critical=True
        )
    else:
        lehigh_leaf = evaluator.add_leaf(
            id="Previous_Lehigh_Head_Coach",
            desc=lehigh_desc,
            parent=parent,
            critical=True
        )
        lehigh_claim = (
            f"Immediately prior to being hired by Yale in February 2026, {person} served as the head football coach at Lehigh University."
        )
        await evaluator.verify(
            claim=lehigh_claim,
            node=lehigh_leaf,
            sources=lehigh_sources_union,
            additional_instruction=(
                "Confirm that the sources indicate the coach's most recent head coaching job before Yale was at Lehigh University. "
                "Accept phrasings like 'comes to Yale from Lehigh', 'previously served as Lehigh's head coach', or similar language "
                "that clearly establishes Lehigh as the head coaching role immediately prior to Yale."
            )
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
    Entry point for evaluating an answer to the Yale head coach (Feb 2026) task.
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_coach_info(),
        template_class=CoachExtraction,
        extraction_name="coach_extraction",
    )

    # Record some helpful custom info
    evaluator.add_custom_info(
        info={
            "extracted_name": extraction.coach_name,
            "counts": {
                "yale_hire_sources": len(extraction.yale_hire_sources or []),
                "award_sources": len(extraction.award_sources or []),
                "lehigh_sources": len(extraction.lehigh_sources or []),
                "all_urls": len(extraction.all_urls or []),
            }
        },
        info_type="extraction_stats",
    )

    # Build tree and verify claims
    await build_and_verify_nodes(evaluator, root, extraction)

    # Return summary
    return evaluator.get_summary()