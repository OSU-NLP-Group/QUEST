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
TASK_ID = "project_hail_mary_details"
TASK_DESCRIPTION = (
    'What is the theatrical release date in the United States for the movie "Project Hail Mary," '
    'who are its directors, and what is the specific filming location in Portsmouth, England that was used for production?'
)

EXPECTED = {
    "us_theatrical_release_date": "March 20, 2026",
    "directors": ["Phil Lord", "Christopher Miller"],
    "portsmouth_filming_location": "South Parade Pier in Southsea (Portsmouth, England)",
    "runtime": "2 hours 36 minutes (156 minutes)",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectHailMaryExtraction(BaseModel):
    # Core facts stated in the answer (as written)
    movie_title: Optional[str] = None
    us_theatrical_release_date: Optional[str] = None
    directors: List[str] = Field(default_factory=list)
    portsmouth_filming_location: Optional[str] = None
    runtime: Optional[str] = None

    # Per-claim sources explicitly cited in the answer (URLs only)
    title_sources: List[str] = Field(default_factory=list)
    release_date_sources: List[str] = Field(default_factory=list)
    directors_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)
    runtime_sources: List[str] = Field(default_factory=list)

    # All URLs present in the answer (fallback source pool)
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project_hail_mary() -> str:
    return """
    Extract the requested information from the provided answer text regarding the film "Project Hail Mary".
    Return a JSON object with the following fields (use null where information is not stated in the answer):

    - movie_title: The movie title as explicitly stated in the answer (string or null).
    - us_theatrical_release_date: The US theatrical release date as stated in the answer (string or null). Keep the exact formatting from the answer (e.g., "March 20, 2026", "2026-03-20", "March 20th, 2026").
    - directors: An array of the director names as stated in the answer (e.g., ["Phil Lord", "Christopher Miller"]). If none mentioned, return an empty array.
    - portsmouth_filming_location: The specific filming location in Portsmouth, England that the answer claims was used for production (string or null). For example, "South Parade Pier in Southsea (Portsmouth, England)".
    - runtime: The runtime as stated in the answer (string or null). Accept formats like "2 hours 36 minutes", "156 minutes", "2h 36m", etc.

    Also extract URL sources cited in the answer for each field when available (URLs only; deduplicate; include full protocol):
    - title_sources: URLs that support the movie title identification (array).
    - release_date_sources: URLs that support the US theatrical release date (array).
    - directors_sources: URLs that support the directors attribution (array).
    - location_sources: URLs that support the Portsmouth filming location claim (array).
    - runtime_sources: URLs that support the runtime claim (array).

    Finally, also extract:
    - all_sources: An array containing every URL present anywhere in the answer (including any general "Sources" section).

    IMPORTANT:
    - Only extract what is explicitly present in the answer text. Do not invent or infer anything not stated.
    - For URL fields, extract only actual URLs (including those embedded in markdown links).
    - Deduplicate URLs while preserving their order of appearance.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls or []:
        if not u or not isinstance(u, str):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _pick_sources(extracted: ProjectHailMaryExtraction, primary: List[str]) -> List[str]:
    """Pick best sources: prefer primary list; otherwise fallback to all_sources. Always dedup."""
    candidates = primary if primary else extracted.all_sources
    return _dedup_preserve_order(candidates)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root,
    extracted: ProjectHailMaryExtraction,
) -> None:
    """
    Build the verification tree following the rubric and run verifications.
    Root rubric node is critical parallel; each child is critical (and often sequential) with
    separate checks:
    - what the answer states (presence/correctness in the answer text)
    - whether cited sources support that claim (URL-grounded)
    """

    # Top-level rubric node (critical parallel)
    rubric_node = evaluator.add_parallel(
        id="Project_Hail_Mary_Details",
        desc='Evaluate whether the answer satisfies all stated constraints for the movie "Project Hail Mary".',
        parent=root,
        critical=True,
    )

    # 1) Movie Title
    title_node = evaluator.add_sequential(
        id="Movie_Title",
        desc='Clearly identifies the movie as titled "Project Hail Mary".',
        parent=rubric_node,
        critical=True,
    )
    title_stated = evaluator.add_leaf(
        id="movie_title_stated",
        desc='The answer explicitly identifies the movie as titled "Project Hail Mary".',
        parent=title_node,
        critical=True,
    )
    await evaluator.verify(
        claim='The answer explicitly identifies the movie as titled "Project Hail Mary".',
        node=title_stated,
        additional_instruction="Check the answer text itself. Allow minor formatting like quotation marks or parenthetical '(film)'."
    )
    # (No source-grounding leaf for title; trivial identification is verified directly against answer.)

    # 2) US Theatrical Release Date
    release_node = evaluator.add_sequential(
        id="US_Theatrical_Release_Date",
        desc='States the US theatrical release date for "Project Hail Mary" as March 20, 2026.',
        parent=rubric_node,
        critical=True,
    )
    # 2.1 stated in answer (and correct)
    release_stated = evaluator.add_leaf(
        id="us_release_date_stated_correct",
        desc='The answer states the US theatrical release date as March 20, 2026.',
        parent=release_node,
        critical=True,
    )
    await evaluator.verify(
        claim='The answer states that the US theatrical release date for "Project Hail Mary" is March 20, 2026.',
        node=release_stated,
        additional_instruction="Accept equivalent date formats such as 'Mar 20, 2026', '2026-03-20', or 'March 20th, 2026'. Focus only on what the answer claims."
    )
    # 2.2 supported by sources
    release_sources = _pick_sources(extracted, extracted.release_date_sources)
    if release_sources:
        release_supported = evaluator.add_leaf(
            id="us_release_date_supported",
            desc="Cited sources support the US theatrical release date claim (March 20, 2026).",
            parent=release_node,
            critical=True,
        )
        await evaluator.verify(
            claim='The US theatrical release date of the film "Project Hail Mary" is March 20, 2026.',
            node=release_supported,
            sources=release_sources,
            additional_instruction=(
                "Verify the US (United States) theatrical release date specifically. "
                "If a source lists multiple country release dates, ensure the US date is March 20, 2026. "
                "If sources are irrelevant or do not clearly state the US release date, mark as not supported."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="us_release_date_supported",
            desc="Cited sources support the US theatrical release date claim (missing or insufficient sources).",
            parent=release_node,
            critical=True,
        )

    # 3) Directors
    directors_node = evaluator.add_sequential(
        id="Directors",
        desc='Names BOTH Phil Lord and Christopher Miller as the directors of "Project Hail Mary" (both must be credited).',
        parent=rubric_node,
        critical=True,
    )
    # 3.1 stated in answer (both credited)
    directors_stated = evaluator.add_leaf(
        id="directors_both_stated",
        desc='The answer names BOTH Phil Lord and Christopher Miller as directors.',
        parent=directors_node,
        critical=True,
    )
    await evaluator.verify(
        claim='The answer names both "Phil Lord" and "Christopher Miller" as directors of "Project Hail Mary".',
        node=directors_stated,
        additional_instruction="Check the answer text only. Both names must be present and credited as directors (co-directors)."
    )
    # 3.2 supported by sources
    directors_sources = _pick_sources(extracted, extracted.directors_sources)
    if directors_sources:
        directors_supported = evaluator.add_leaf(
            id="directors_supported",
            desc="Cited sources support that the directors are Phil Lord and Christopher Miller.",
            parent=directors_node,
            critical=True,
        )
        await evaluator.verify(
            claim='The directors of the film "Project Hail Mary" are Phil Lord and Christopher Miller (co-directors).',
            node=directors_supported,
            sources=directors_sources,
            additional_instruction=(
                "The pages must credit Phil Lord and Christopher Miller explicitly as directors (or co-directors). "
                "Mentions of them solely as producers or writers without directing credit are insufficient."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="directors_supported",
            desc="Cited sources support the directors claim (missing or insufficient sources).",
            parent=directors_node,
            critical=True,
        )

    # 4) Portsmouth Filming Location
    location_node = evaluator.add_sequential(
        id="Portsmouth_Filming_Location",
        desc='Identifies South Parade Pier in Southsea (Portsmouth, England) as a filming location used for production of "Project Hail Mary".',
        parent=rubric_node,
        critical=True,
    )
    # 4.1 stated in answer
    location_stated = evaluator.add_leaf(
        id="location_stated",
        desc='The answer identifies South Parade Pier in Southsea (Portsmouth, England) as a filming location used for production.',
        parent=location_node,
        critical=True,
    )
    await evaluator.verify(
        claim='The answer identifies "South Parade Pier" in "Southsea (Portsmouth, England)" as a filming location for "Project Hail Mary".',
        node=location_stated,
        additional_instruction=(
            "Check only the answer text. Allow phrasing variants like 'South Parade Pier, Southsea, Portsmouth' or "
            "'South Parade Pier in Southsea, Portsmouth'. It must clearly link this location to filming/production."
        )
    )
    # 4.2 supported by sources
    location_sources = _pick_sources(extracted, extracted.location_sources)
    if location_sources:
        location_supported = evaluator.add_leaf(
            id="location_supported",
            desc="Cited sources support that South Parade Pier (Southsea, Portsmouth) was used as a filming location.",
            parent=location_node,
            critical=True,
        )
        await evaluator.verify(
            claim='South Parade Pier in Southsea (Portsmouth, England) was used as a filming location for the film "Project Hail Mary".',
            node=location_supported,
            sources=location_sources,
            additional_instruction=(
                "The page must explicitly tie 'Project Hail Mary' filming/production to South Parade Pier in Southsea, Portsmouth. "
                "General mentions of the pier without linking it to this film are insufficient."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="location_supported",
            desc="Cited sources support the Portsmouth filming location (missing or insufficient sources).",
            parent=location_node,
            critical=True,
        )

    # 5) Runtime
    runtime_node = evaluator.add_sequential(
        id="Runtime",
        desc="States the movie runtime as 2 hours 36 minutes (156 minutes).",
        parent=rubric_node,
        critical=True,
    )
    # 5.1 stated in answer (and correct equivalence)
    runtime_stated = evaluator.add_leaf(
        id="runtime_stated_correct",
        desc="The answer states the runtime as 2 hours 36 minutes (156 minutes).",
        parent=runtime_node,
        critical=True,
    )
    await evaluator.verify(
        claim='The answer states that the runtime of "Project Hail Mary" is equivalent to 2 hours 36 minutes (156 minutes).',
        node=runtime_stated,
        additional_instruction=(
            "Check the answer text only. Accept equivalent expressions like '2h 36m', '156 min', '156 minutes', "
            "'2 hours and 36 minutes'. Ensure the number corresponds to 156 minutes."
        )
    )
    # 5.2 supported by sources
    runtime_sources = _pick_sources(extracted, extracted.runtime_sources)
    if runtime_sources:
        runtime_supported = evaluator.add_leaf(
            id="runtime_supported",
            desc="Cited sources support the runtime (2 hours 36 minutes / 156 minutes).",
            parent=runtime_node,
            critical=True,
        )
        await evaluator.verify(
            claim='The runtime of the film "Project Hail Mary" is 156 minutes (2 hours 36 minutes).',
            node=runtime_supported,
            sources=runtime_sources,
            additional_instruction=(
                "Verify that the page explicitly states the runtime as 156 minutes or an equivalent time expression (2h 36m). "
                "If runtime is not given or differs, mark as not supported."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="runtime_supported",
            desc="Cited sources support the runtime claim (missing or insufficient sources).",
            parent=runtime_node,
            critical=True,
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
    Evaluate an answer for the Project Hail Mary details task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall rubric node is parallel
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

    # Record ground truth expectations for context
    evaluator.add_ground_truth(
        {
            "expected_us_theatrical_release_date": EXPECTED["us_theatrical_release_date"],
            "expected_directors": EXPECTED["directors"],
            "expected_portsmouth_filming_location": EXPECTED["portsmouth_filming_location"],
            "expected_runtime": EXPECTED["runtime"],
        },
        gt_type="expected_values",
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_project_hail_mary(),
        template_class=ProjectHailMaryExtraction,
        extraction_name="project_hail_mary_extraction",
    )

    # Build and verify the rubric tree
    await build_and_verify_tree(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()