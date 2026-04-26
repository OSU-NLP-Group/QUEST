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
TASK_ID = "yale_head_coach_degree_202602"
TASK_DESCRIPTION = (
    "What is the minimum educational degree requirement for the Head Coach, Football position at Yale University "
    "that was posted in February 2026?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DegreeExtraction(BaseModel):
    """
    Structured extraction from the agent's answer:
    - minimum_degree: the phrase the answer claims as the minimum educational degree requirement
    - support_urls: all URLs cited as supporting references for the job posting/announcement
    """
    minimum_degree: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_degree_and_sources() -> str:
    return """
    From the answer, extract the following information:

    1) minimum_degree: The exact phrase the answer uses to describe the minimum educational degree requirement for the Yale University "Head Coach, Football" position (posted in February 2026). This should be a short phrase like "Bachelor's degree", "BA/BS", or similar. If the answer implies a bachelor's degree using synonyms (e.g., "BA/BS", "undergraduate degree"), extract the exact phrasing used in the answer. Do not invent or normalize beyond what is written in the answer.

    2) support_urls: A list of all URLs cited in the answer as references/sources for the job posting or official announcement. Extract actual URLs only (including those in markdown links). If no URLs are provided, return an empty list.

    Return a JSON object with these two fields. If a field is missing, set it to null (for strings) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _dedupe_and_clean_urls(urls: List[str]) -> List[str]:
    """Remove empty/invalid strings and de-duplicate while preserving order."""
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            cleaned.append(u2)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extracted: DegreeExtraction
) -> None:
    """
    Build the verification tree per rubric and run verifications.
    """
    # Parent (critical, parallel) node mirroring the rubric
    parent_node = evaluator.add_parallel(
        id="Minimum_Educational_Degree_Identification",
        desc="Correctly identifies the minimum educational degree requirement for the Yale Head Coach, Football position posted in February 2026",
        parent=root_node,
        critical=True
    )

    # Clean URLs
    support_urls = _dedupe_and_clean_urls(extracted.support_urls or [])

    # ------------------------------------------------------------------ #
    # Leaf 1: Identifies_Bachelor_Degree (Critical)
    # Check that the answer explicitly states Bachelor's degree
    # (or equivalent phrasing like BA/BS) as the minimum requirement.
    # This is a check against the answer text itself (no URL evidence required).
    # ------------------------------------------------------------------ #
    leaf_bachelor = evaluator.add_leaf(
        id="Identifies_Bachelor_Degree",
        desc="States that a Bachelor's degree is the minimum educational degree requirement",
        parent=parent_node,
        critical=True
    )

    degree_text = extracted.minimum_degree or ""
    claim_about_answer = (
        "The answer explicitly indicates that the minimum educational degree requirement for Yale's Head Coach, "
        "Football position (posted in February 2026) is a Bachelor's degree or an equivalent phrasing (e.g., BA/BS, "
        "baccalaureate, undergraduate degree). "
        f"Extracted minimum_degree from the answer: '{degree_text}'."
    )
    await evaluator.verify(
        claim=claim_about_answer,
        node=leaf_bachelor,
        additional_instruction=(
            "Judge based on the provided answer text only. Consider common equivalent phrasings for a bachelor's degree "
            "(e.g., BA/BS, baccalaureate, undergraduate degree). If the answer implies a higher minimum (e.g., Master's), "
            "or does not clearly state Bachelor's (or an equivalent phrasing) as the minimum, mark this as incorrect."
        ),
    )

    # ------------------------------------------------------------------ #
    # Leaf 2: Provides_Supporting_Reference (Critical)
    # Must provide at least one valid reference URL that is an official Yale
    # University job posting or official announcement page relevant to the role.
    # We'll verify that among the provided URLs, at least one qualifies.
    # ------------------------------------------------------------------ #
    if not support_urls:
        # No URLs provided -> immediate failure for this critical criterion.
        evaluator.add_leaf(
            id="Provides_Supporting_Reference",
            desc="Provides a valid reference URL from the Yale University job posting or official announcement",
            parent=parent_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        leaf_reference = evaluator.add_leaf(
            id="Provides_Supporting_Reference",
            desc="Provides a valid reference URL from the Yale University job posting or official announcement",
            parent=parent_node,
            critical=True
        )

        claim_reference_validity = (
            "At least one of these URLs is an official Yale University job posting or an official Yale announcement page "
            "for the Head Coach, Football position posted in February 2026 (or a direct/print-friendly version of that posting). "
            "The page should clearly be owned by Yale (e.g., on yale.edu, hr.yale.edu, or Yale Athletics/yalebulldogs.com), "
            "and it should be relevant to the Head Coach, Football job at Yale. If an explicit posting date is shown, it should "
            "be in February 2026; if the date is not clearly visible but it is evidently the official posting/announcement for "
            "this specific role around that period, consider it acceptable. Do not consider third-party aggregators as valid."
        )

        await evaluator.verify(
            claim=claim_reference_validity,
            node=leaf_reference,
            sources=support_urls,
            additional_instruction=(
                "Focus on whether the URL(s) are clearly official Yale properties and specifically about the Yale Head Coach, "
                "Football position. Prefer pages that show the posting date or are clearly the official job listing/announcement. "
                "Reject non-official sites (job boards, news scrapers, or unrelated pages)."
            ),
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
    Entry point to evaluate an answer for the Yale Head Coach, Football minimum degree requirement task.
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_degree_and_sources(),
        template_class=DegreeExtraction,
        extraction_name="degree_and_sources"
    )

    # Record some custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_minimum_degree": extracted.minimum_degree,
            "support_url_count": len(extracted.support_urls or []),
            "support_urls": _dedupe_and_clean_urls(extracted.support_urls or []),
        },
        info_type="extraction_summary",
        info_name="extraction_overview"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()