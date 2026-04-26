import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fisd_min_hours_life_insurance"
TASK_DESCRIPTION = (
    "What is the minimum number of hours per week an employee must work at Frisco ISD "
    "(Frisco Independent School District in Texas) to qualify for the $10,000 group term "
    "life and AD&D insurance policy?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FISDHoursExtraction(BaseModel):
    """
    Information extracted from the agent's answer.
    - min_weekly_hours: The minimum weekly hours string as claimed in the answer
      (e.g., '20', '20 hours', '20 hours per week', '20+ hours', 'half-time (20 hours)', etc.).
    - sources: All URLs cited in the answer that are intended to support the claim.
    """
    min_weekly_hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_min_hours() -> str:
    return """
    Extract the minimum weekly hours requirement for Frisco ISD (Frisco Independent School District, Texas) employees to qualify
    for the $10,000 group term life and AD&D (basic life) insurance, as stated in the provided answer.

    Return:
    - min_weekly_hours: a short string capturing the minimum weekly hours threshold exactly as stated in the answer text.
      Examples: "20", "20 hours", "20 hours per week", "20+ hours", "half-time (20 hours)".
      If not stated, return null.
    - sources: an array of all URLs explicitly cited in the answer as sources or references for this claim.
      Include any links provided (plain URLs or markdown links).
    """

# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def filter_official_fisd_urls(urls: List[str]) -> List[str]:
    """
    Keep only URLs that appear to be official Frisco ISD resources.
    Currently uses a simple domain check for 'friscoisd.org'.
    """
    official = []
    for u in urls:
        try:
            lu = u.lower().strip()
            if "friscoisd.org" in lu:
                official.append(u)
        except Exception:
            continue
    return official

# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_minimum_hours_requirement(
    evaluator: Evaluator,
    parent_node,
    extracted: FISDHoursExtraction
) -> None:
    """
    Build the verification tree corresponding to the rubric and run verifications.
    Tree:
      Minimum_Weekly_Hours_Requirement (parallel, non-critical)
        ├─ Hours_Value (leaf, critical)
        └─ Source_Reference (leaf, critical)
    """
    # Parent node mirroring the rubric root (non-critical; parallel)
    main_node = evaluator.add_parallel(
        id="Minimum_Weekly_Hours_Requirement",
        desc="Identifies the minimum weekly hours an employee must work to qualify for Frisco ISD's $10,000 group term life and AD&D policy",
        parent=parent_node,
        critical=False
    )

    hours_str = (extracted.min_weekly_hours or "").strip()
    all_sources = extracted.sources or []
    official_sources = filter_official_fisd_urls(all_sources)

    # Leaf 1: Hours_Value (critical)
    hours_leaf = evaluator.add_leaf(
        id="Hours_Value",
        desc="Provides the correct minimum weekly hours value as stated in Frisco ISD's official employment information",
        parent=main_node,
        critical=True
    )

    # Construct the claim; ensure it ties specifically to $10,000 group term life & AD&D
    if hours_str:
        hours_claim = (
            f"According to an official Frisco ISD resource, employees must work at least {hours_str} per week "
            f"to qualify for the $10,000 group term life and AD&D (basic life) insurance benefit."
        )
    else:
        # If the answer didn't provide a value, this claim should be judged unsupported
        hours_claim = (
            "According to an official Frisco ISD resource, there is a clearly stated minimum weekly hours threshold "
            "to qualify for the $10,000 group term life and AD&D (basic life) insurance benefit."
        )

    hours_additional_instruction = (
        "Only mark the claim as supported if the official Frisco ISD page (domain friscoisd.org) explicitly states "
        "the eligibility minimum in hours per week for the $10,000 basic life/group term life and AD&D benefit. "
        "Accept equivalent wording such as '20 or more hours', '≥20 hours', or 'half-time (20 hours)'. "
        "If no official Frisco ISD URL is provided, or the page does not clearly connect the hour threshold "
        "to the $10,000 basic life/group term life and AD&D eligibility, mark as not supported."
    )

    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=official_sources if official_sources else None,
        additional_instruction=hours_additional_instruction
    )

    # Leaf 2: Source_Reference (critical)
    source_leaf = evaluator.add_leaf(
        id="Source_Reference",
        desc="Includes a valid reference to Frisco ISD's official employment webpage or documentation",
        parent=main_node,
        critical=True
    )

    # Claim for official source presence and relevance
    source_claim = (
        "This URL is an official Frisco ISD webpage or official district document (domain friscoisd.org) and "
        "it discusses employee benefits relevant to group term life/basic life and/or AD&D."
    )
    source_additional_instruction = (
        "Confirm both: (1) the URL is clearly an official Frisco ISD resource (domain friscoisd.org), and "
        "(2) the page discusses employment benefits related to group term life/basic life and/or AD&D. "
        "If no such official page is present, mark as not supported."
    )

    # For this check, we specifically test official domains; pass only official sources.
    await evaluator.verify(
        claim=source_claim,
        node=source_leaf,
        sources=official_sources if official_sources else None,
        additional_instruction=source_additional_instruction
    )

    # Optional: record some helpful info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "extracted_min_weekly_hours": hours_str or None,
            "all_extracted_sources": all_sources,
            "official_fisd_sources_used": official_sources
        },
        info_type="debug",
        info_name="extraction_debug_info"
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
    Evaluate an answer for the Frisco ISD minimum weekly hours requirement for the $10,000 group term life and AD&D policy.
    """
    # Initialize evaluator with a parallel root strategy (single rubric group under root)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_min_hours(),
        template_class=FISDHoursExtraction,
        extraction_name="min_hours_extraction"
    )

    # Build verification tree and run checks per rubric
    await verify_minimum_hours_requirement(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()