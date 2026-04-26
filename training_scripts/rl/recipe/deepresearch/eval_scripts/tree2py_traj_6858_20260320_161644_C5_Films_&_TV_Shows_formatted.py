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
TASK_ID = "netflix_final_season_8ep_q4_2025"
TASK_DESCRIPTION = """
Identify a Netflix Original series where the final season meets ALL of the following criteria:
(1) The final season consists of exactly 8 episodes;
(2) The final season was released in multiple separate volumes/parts, not all episodes at once;
(3) The finale episode was released separately as a single-episode drop, distinct from other volumes;
(4) The series filmed primarily in the United States;
(5) Filming for the final season was completed by December 2024;
(6) The final season premiered in the fourth quarter of 2025 (October through December).

Provide the series title and include official reference URLs that verify:
(a) its Netflix Original status,
(b) the complete episode structure and release dates for the final season, and
(c) the production filming timeline and completion date.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SeriesExtraction(BaseModel):
    """Structured info we expect from the agent's answer."""
    series_title: Optional[str] = None
    # URLs that prove Netflix Original status / platform distribution
    platform_urls: List[str] = Field(default_factory=list)
    # URLs that document the final season's episode count and release structure/dates
    structure_urls: List[str] = Field(default_factory=list)
    # URLs that confirm production/filming timeline, completion, premiere date, and filming location
    timeline_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    return """
    Extract the following information exactly as presented in the answer:

    1) series_title: The exact series title identified by the answer (string).
    2) platform_urls: All URLs explicitly cited that confirm the series is a Netflix Original and/or indicate exclusive streaming on Netflix.
       - Include official Netflix pages (e.g., netflix.com title pages, netflix.com/tudum/, about.netflix.com/press) or reputable entertainment news sites (e.g., Variety, Deadline, The Hollywood Reporter).
       - Return only valid URLs explicitly present in the answer text.
    3) structure_urls: All URLs explicitly cited that document the final season's total episode count and the multi-volume release structure with dates.
       - Include official Netflix or reputable entertainment news sources, show pages, or press releases that list episode counts and release dates.
    4) timeline_urls: All URLs explicitly cited that confirm the production timeline of the final season, including filming wrap/completion by December 2024, primary filming location (U.S.), and the premiere date (Q4 2025).
       - Include official Netflix or reputable entertainment news sources.

    Rules:
    - Extract only URLs explicitly mentioned in the answer. Do not invent any.
    - Return full URLs including protocol (http/https).
    - If any category has no URLs in the answer, return an empty list for that category.
    - Remove duplicates if any.

    Output JSON fields:
    - series_title: string | null
    - platform_urls: string[] (can be empty)
    - structure_urls: string[] (can be empty)
    - timeline_urls: string[] (can be empty)
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_platform_distribution_checks(
    evaluator: Evaluator,
    parent_node,
    series: SeriesExtraction,
) -> None:
    """
    Build and verify the 'Platform_Distribution' subtree:
      - Netflix_Original_Status
      - Exclusive_Distribution
      - Platform_Reference
      - Plus a critical existence check for platform URLs
    """
    platform_node = evaluator.add_parallel(
        id="Platform_Distribution",
        desc="Verify the series streaming platform and distribution rights",
        parent=parent_node,
        critical=True,
    )

    title = series.series_title or ""

    # Critical existence check for platform URLs
    platform_urls = _dedup_urls(series.platform_urls or [])
    evaluator.add_custom_node(
        result=len(platform_urls) > 0,
        id="platform_sources_provided",
        desc="At least one platform/distribution source URL is provided",
        parent=platform_node,
        critical=True,
    )

    # Leaf: Netflix Original status
    n_original_leaf = evaluator.add_leaf(
        id="Netflix_Original_Status",
        desc="The series must be labeled as a Netflix Original",
        parent=platform_node,
        critical=True,
    )

    claim_original = (
        f"The series titled '{title}' is a Netflix Original series (i.e., officially labeled "
        f"as a Netflix Original or described as 'Only on Netflix')."
    )
    await evaluator.verify(
        claim=claim_original,
        node=n_original_leaf,
        sources=platform_urls,
        additional_instruction=(
            "Use the provided URL(s). Accept explicit phrases like 'Netflix Original', 'Only on Netflix', "
            "'A Netflix series', or equivalent wording on official Netflix domains (e.g., netflix.com, "
            "netflix.com/tudum, about.netflix.com) or reputable entertainment news outlets (e.g., Variety, "
            "Deadline, The Hollywood Reporter)."
        ),
    )

    # Leaf: Exclusive distribution by Netflix
    exclusive_leaf = evaluator.add_leaf(
        id="Exclusive_Distribution",
        desc="Netflix must have exclusive streaming distribution rights for the series",
        parent=platform_node,
        critical=True,
    )

    claim_exclusive = (
        f"'{title}' streams exclusively on Netflix (Netflix holds exclusive streaming distribution rights), "
        f"indicated by phrases like 'Only on Netflix' or similar."
    )
    await evaluator.verify(
        claim=claim_exclusive,
        node=exclusive_leaf,
        sources=platform_urls,
        additional_instruction=(
            "Look for phrasing such as 'Only on Netflix', 'Streaming exclusively on Netflix', or similar "
            "language implying exclusive streaming rights. Consider the claim supported if exclusivity is "
            "explicitly stated by Netflix (preferred) or by a reputable outlet."
        ),
    )

    # Leaf: Platform reference validity
    platform_ref_leaf = evaluator.add_leaf(
        id="Platform_Reference",
        desc="Provide accessible official source URL from Netflix or reliable entertainment news confirming the series as Netflix Original",
        parent=platform_node,
        critical=True,
    )

    claim_platform_ref = (
        f"At least one of the provided platform/distribution URL(s) explicitly confirms that '{title}' is a "
        f"Netflix Original or indicates 'Only on Netflix'."
    )
    await evaluator.verify(
        claim=claim_platform_ref,
        node=platform_ref_leaf,
        sources=platform_urls,
        additional_instruction=(
            "The page should clearly state 'Netflix Original' or 'Only on Netflix' for the series. Favor "
            "official Netflix domains (netflix.com, netflix.com/tudum, about.netflix.com). Reputable outlets "
            "like variety.com, deadline.com, hollywoodreporter.com are also acceptable if explicit."
        ),
    )


async def build_episode_structure_checks(
    evaluator: Evaluator,
    parent_node,
    series: SeriesExtraction,
) -> None:
    """
    Build and verify the 'Episode_Structure' subtree:
      - Total_Episode_Count (8)
      - Multi_Volume_Release
      - Separate_Finale
      - Multi_Month_Span
      - Structure_Reference
      - Plus a critical existence check for structure URLs
    """
    structure_node = evaluator.add_parallel(
        id="Episode_Structure",
        desc="Verify the final season episode count and release structure",
        parent=parent_node,
        critical=True,
    )

    title = series.series_title or ""
    structure_urls = _dedup_urls(series.structure_urls or [])

    # Critical existence check for structure URLs
    evaluator.add_custom_node(
        result=len(structure_urls) > 0,
        id="structure_sources_provided",
        desc="At least one structure/release source URL is provided",
        parent=structure_node,
        critical=True,
    )

    # Total episode count = 8
    ep_count_leaf = evaluator.add_leaf(
        id="Total_Episode_Count",
        desc="The final season must consist of exactly 8 episodes",
        parent=structure_node,
        critical=True,
    )
    claim_ep_count = f"The final season of '{title}' consists of exactly 8 episodes."
    await evaluator.verify(
        claim=claim_ep_count,
        node=ep_count_leaf,
        sources=structure_urls,
        additional_instruction=(
            "Confirm the specific episode count for the final season equals 8. Prefer official Netflix or "
            "reputable sources listing the final season's episode tally."
        ),
    )

    # Multi-volume/parts release (not all at once)
    multi_volume_leaf = evaluator.add_leaf(
        id="Multi_Volume_Release",
        desc="The final season must be released in multiple separate volumes/parts, not all episodes at once",
        parent=structure_node,
        critical=True,
    )
    claim_multi_volume = (
        f"The final season of '{title}' was released in multiple separate volumes/parts (e.g., Volume 1, Volume 2, etc.), "
        f"not as a single all-at-once drop."
    )
    await evaluator.verify(
        claim=claim_multi_volume,
        node=multi_volume_leaf,
        sources=structure_urls,
        additional_instruction=(
            "Look for explicit references to 'Volume 1 and Volume 2', 'Part 1 and Part 2', or equivalent language. "
            "Avoid accepting a single-batch release."
        ),
    )

    # Separate finale as a single-episode drop
    separate_finale_leaf = evaluator.add_leaf(
        id="Separate_Finale",
        desc="The finale episode must be released separately as a single-episode drop, distinct from other volumes",
        parent=structure_node,
        critical=True,
    )
    claim_separate_finale = (
        f"The finale episode of '{title}' was released separately as a one-episode drop distinct from other volumes/parts."
    )
    await evaluator.verify(
        claim=claim_separate_finale,
        node=separate_finale_leaf,
        sources=structure_urls,
        additional_instruction=(
            "Verify that the final (last) episode launched on its own date as a single-episode release (e.g., a special or "
            "standalone finale), separate from previous multi-episode volumes."
        ),
    )

    # Release spans at least two different months
    multi_month_leaf = evaluator.add_leaf(
        id="Multi_Month_Span",
        desc="The release structure must span at least two different calendar months",
        parent=structure_node,
        critical=True,
    )
    claim_multi_month = (
        f"The release schedule for the final season of '{title}' spanned at least two distinct calendar months."
    )
    await evaluator.verify(
        claim=claim_multi_month,
        node=multi_month_leaf,
        sources=structure_urls,
        additional_instruction=(
            "Check the published release dates for the final season volumes and separate finale. They should cover at least "
            "two different months (e.g., October and November)."
        ),
    )

    # Structure reference validity
    structure_ref_leaf = evaluator.add_leaf(
        id="Structure_Reference",
        desc="Provide accessible official source URL documenting the complete release structure, episode count, and release dates for the final season",
        parent=structure_node,
        critical=True,
    )
    claim_structure_ref = (
        f"At least one provided structure/release URL for '{title}' explicitly documents the final season's total episode count, "
        f"the multi-volume (parts) structure, and the specific release dates."
    )
    await evaluator.verify(
        claim=claim_structure_ref,
        node=structure_ref_leaf,
        sources=structure_urls,
        additional_instruction=(
            "The page should include the final season’s episode count and the breakdown by volumes/parts with their exact release dates."
        ),
    )


async def build_production_timeline_checks(
    evaluator: Evaluator,
    parent_node,
    series: SeriesExtraction,
) -> None:
    """
    Build and verify the 'Production_Timeline' subtree:
      - US_Filming_Location
      - Filming_Completion (by Dec 2024)
      - Q4_2025_Premiere
      - Timeline_Reference
      - Plus a critical existence check for timeline URLs
    """
    timeline_node = evaluator.add_parallel(
        id="Production_Timeline",
        desc="Verify production filming timeline and premiere date",
        parent=parent_node,
        critical=True,
    )

    title = series.series_title or ""
    timeline_urls = _dedup_urls(series.timeline_urls or [])

    # Critical existence check for timeline URLs
    evaluator.add_custom_node(
        result=len(timeline_urls) > 0,
        id="timeline_sources_provided",
        desc="At least one production/timeline source URL is provided",
        parent=timeline_node,
        critical=True,
    )

    # Filmed primarily in the United States
    us_filming_leaf = evaluator.add_leaf(
        id="US_Filming_Location",
        desc="The series must have filmed primarily in the United States",
        parent=timeline_node,
        critical=True,
    )
    claim_us_filming = (
        f"The series '{title}' filmed primarily in the United States (i.e., the majority of principal photography "
        f"occurred in the U.S., including for the final season)."
    )
    await evaluator.verify(
        claim=claim_us_filming,
        node=us_filming_leaf,
        sources=timeline_urls,
        additional_instruction=(
            "Accept support such as 'principal photography in [U.S. city/state]' or credible reporting that primary filming "
            "locations were in the U.S. It's acceptable if some scenes were filmed elsewhere as long as most production was U.S.-based."
        ),
    )

    # Filming completed by December 2024
    wrap_leaf = evaluator.add_leaf(
        id="Filming_Completion",
        desc="Filming for the final season must have been completed by December 2024",
        parent=timeline_node,
        critical=True,
    )
    claim_wrap = (
        f"Filming for the final season of '{title}' was completed no later than December 31, 2024 "
        f"(e.g., reports of 'wrap', 'completed principal photography', or 'finished filming' by/before that date)."
    )
    await evaluator.verify(
        claim=claim_wrap,
        node=wrap_leaf,
        sources=timeline_urls,
        additional_instruction=(
            "Look for explicit language that filming 'wrapped', 'completed', or 'finished' for the final season by or before "
            "December 31, 2024. Earlier completion (e.g., October 2024) also satisfies this requirement."
        ),
    )

    # Premiered in Q4 2025
    q4_premiere_leaf = evaluator.add_leaf(
        id="Q4_2025_Premiere",
        desc="The final season must have premiered in the fourth quarter of 2025 (October-December)",
        parent=timeline_node,
        critical=True,
    )
    claim_q4_premiere = (
        f"The final season of '{title}' premiered in Q4 2025 (October, November, or December 2025). "
        f"This can be the release of the first volume/part."
    )
    await evaluator.verify(
        claim=claim_q4_premiere,
        node=q4_premiere_leaf,
        sources=timeline_urls,
        additional_instruction=(
            "Verify the initial release date for the final season (e.g., Volume 1) occurred during Oct–Dec 2025."
        ),
    )

    # Timeline reference validity
    timeline_ref_leaf = evaluator.add_leaf(
        id="Timeline_Reference",
        desc="Provide accessible official source URL confirming the production filming timeline, completion date, and premiere date",
        parent=timeline_node,
        critical=True,
    )
    claim_timeline_ref = (
        f"At least one provided production/timeline URL for '{title}' explicitly confirms the filming timeline of the final season, "
        f"including the completion (wrap) date and the premiere date."
    )
    await evaluator.verify(
        claim=claim_timeline_ref,
        node=timeline_ref_leaf,
        sources=timeline_urls,
        additional_instruction=(
            "The page should reference both the filming wrap/completion timing and the final season's premiere date (Q4 2025)."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Netflix final-season criteria task.
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

    # Extract structured information from the agent's answer
    series_info = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction",
    )

    # Record task constraints as "ground truth" context (not used for scoring)
    evaluator.add_ground_truth({
        "requirements": {
            "episodes_final_season": 8,
            "release_in_volumes": True,
            "separate_single_episode_finale": True,
            "primary_filming_location": "United States",
            "final_season_filming_completed_by": "2024-12-31",
            "final_season_premiere_window": "Q4 2025 (Oct–Dec)",
        }
    }, gt_type="constraints")

    # Build the top-level critical node mirroring the rubric root
    top_node = evaluator.add_parallel(
        id="Series_Identification",
        desc="Evaluate whether the identified series meets all specified criteria for a Netflix Original with specific episode structure and production timeline",
        parent=root,
        critical=True,
    )

    # Critical title existence check under the top-level node
    evaluator.add_custom_node(
        result=(series_info.series_title is not None and str(series_info.series_title).strip() != ""),
        id="Series_Title_Provided",
        desc="Series title is provided in the answer",
        parent=top_node,
        critical=True,
    )

    # Build subtrees according to rubric
    await build_platform_distribution_checks(evaluator, top_node, series_info)
    await build_episode_structure_checks(evaluator, top_node, series_info)
    await build_production_timeline_checks(evaluator, top_node, series_info)

    # Return structured summary
    return evaluator.get_summary()