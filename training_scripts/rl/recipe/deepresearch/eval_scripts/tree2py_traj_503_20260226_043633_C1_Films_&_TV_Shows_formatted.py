import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "avatar_fire_and_ash_info"
TASK_DESCRIPTION = "Provide the runtime and release date of Avatar: Fire and Ash"

EXPECTED_RUNTIME_MINUTES = 195
ALLOWED_RUNTIME_DEVIATION_MIN = 3
EXPECTED_RELEASE_DATE_TEXT = "December 19, 2025"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MovieInfoExtraction(BaseModel):
    """
    Extracted info from the agent's answer for the film Avatar: Fire and Ash.
    """
    # Runtime extracted from the answer (any textual form, e.g., "3h 15m", "195 minutes")
    runtime_text: Optional[str] = None
    # If the answer provides a minutes form, capture it as a string (keep string for flexibility)
    runtime_minutes: Optional[str] = None
    # URLs specifically cited for runtime
    runtime_sources: List[str] = Field(default_factory=list)

    # Release date extracted from the answer (e.g., "December 19, 2025")
    release_date_text: Optional[str] = None
    # If the answer provides an ISO-like form (YYYY-MM-DD), capture it
    release_date_iso: Optional[str] = None
    # URLs specifically cited for release date
    release_date_sources: List[str] = Field(default_factory=list)

    # All URLs mentioned anywhere in the answer (as a general fallback)
    global_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_movie_info() -> str:
    return """
    Extract the information for the film "Avatar: Fire and Ash" from the provided answer.

    You must extract:
    1) runtime_text: The runtime as stated in the answer (any form: e.g., "3h 15m", "195 minutes", "approx. 195 min"). If not mentioned, return null.
    2) runtime_minutes: The runtime in minutes if the answer explicitly provides it (e.g., "195 minutes"). If not provided as minutes, return null.
    3) runtime_sources: A list of all URLs cited in the answer that specifically support the runtime information. If no URLs are cited for runtime, return an empty list.

    4) release_date_text: The release date as stated in the answer (e.g., "December 19, 2025"). If not mentioned, return null.
    5) release_date_iso: The release date in ISO format (YYYY-MM-DD) if the answer provides something that can be parsed or is explicitly given. If not available, return null.
    6) release_date_sources: A list of all URLs cited in the answer that specifically support the release date information. If no URLs are cited for release date, return an empty list.

    7) global_sources: A list of every URL mentioned anywhere in the answer (including general references and sources sections). This should include all URLs present in the answer regardless of context. If none, return an empty list.

    Rules:
    - Do not invent any information. Only extract what is explicitly in the answer.
    - For URLs, include full URLs. Accept plain URLs or markdown links, but output the actual URL part.
    - If a field is not present, set it to null (for strings) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def choose_sources(preferred: List[str], fallback: List[str]) -> List[str]:
    """
    Choose sources for verification:
    - Use preferred if available (non-empty).
    - Otherwise use fallback.
    - Remove duplicates while preserving order.
    """
    combined = preferred if preferred else fallback
    seen = set()
    result = []
    for url in combined:
        if isinstance(url, str):
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_runtime_and_release(
    evaluator: Evaluator,
    root_node,
    extracted: MovieInfoExtraction,
) -> None:
    """
    Build leaf nodes per rubric and run verifications.
    """

    # Runtime leaf (Critical)
    runtime_node = evaluator.add_leaf(
        id="Runtime",
        desc="The runtime is approximately 3 hours and 15 minutes (195 minutes), with acceptable variation of ±3 minutes",
        parent=root_node,
        critical=True,
    )

    runtime_claim = (
        "For the film 'Avatar: Fire and Ash', the runtime is approximately 195 minutes "
        "(about 3 hours and 15 minutes). Deviations within ±3 minutes (i.e., 192–198 minutes) "
        "should be considered acceptable equivalents."
    )

    runtime_sources = choose_sources(extracted.runtime_sources, extracted.global_sources)

    await evaluator.verify(
        claim=runtime_claim,
        node=runtime_node,
        sources=runtime_sources if runtime_sources else None,
        additional_instruction=(
            "Verify that at least one provided webpage explicitly supports a runtime in the range 192–198 minutes "
            "for 'Avatar: Fire and Ash'. Treat '3h 15m' as equivalent to 195 minutes. "
            "If no URL supports this, judge as not supported."
        ),
    )

    # Release date leaf (Critical)
    release_node = evaluator.add_leaf(
        id="Release_Date",
        desc="The release date is December 19, 2025",
        parent=root_node,
        critical=True,
    )

    release_claim = "The film 'Avatar: Fire and Ash' has a release date of December 19, 2025."

    release_sources = choose_sources(extracted.release_date_sources, extracted.global_sources)

    await evaluator.verify(
        claim=release_claim,
        node=release_node,
        sources=release_sources if release_sources else None,
        additional_instruction=(
            "Confirm that a provided webpage explicitly states the release date as December 19, 2025 "
            "for 'Avatar: Fire and Ash'. Accept minor formatting variants like 'Dec 19, 2025'. "
            "If the URLs are irrelevant or do not confirm this date, judge as not supported."
        ),
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
    Evaluate an answer for the Avatar: Fire and Ash runtime and release date task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: runtime and release date checks are independent
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
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_movie_info(),
        template_class=MovieInfoExtraction,
        extraction_name="movie_info_extraction",
    )

    # Record ground truth expectations (for transparency)
    evaluator.add_ground_truth({
        "expected_runtime_minutes": f"{EXPECTED_RUNTIME_MINUTES} ± {ALLOWED_RUNTIME_DEVIATION_MIN} minutes",
        "expected_release_date": EXPECTED_RELEASE_DATE_TEXT,
    })

    # Add some custom info for debugging
    evaluator.add_custom_info(
        {
            "runtime_text": extracted_info.runtime_text,
            "runtime_minutes": extracted_info.runtime_minutes,
            "runtime_sources_count": len(extracted_info.runtime_sources),
            "release_date_text": extracted_info.release_date_text,
            "release_date_iso": extracted_info.release_date_iso,
            "release_date_sources_count": len(extracted_info.release_date_sources),
            "global_sources_count": len(extracted_info.global_sources),
        },
        info_type="extraction_debug",
    )

    # Build verification tree and run checks
    await verify_runtime_and_release(evaluator, root, extracted_info)

    # Return structured evaluation summary
    return evaluator.get_summary()