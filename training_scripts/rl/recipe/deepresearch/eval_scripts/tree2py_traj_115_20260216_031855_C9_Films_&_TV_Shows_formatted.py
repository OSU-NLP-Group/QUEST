import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "emmy_streaming_originals_76th"
TASK_DESCRIPTION = """
Identify at least three streaming platform original series that won awards at the 76th Primetime Emmy Awards (held in September 2024). For each series, provide: 
(1) The series title and primary streaming platform; 
(2) At least one specific Emmy award category won (e.g., Outstanding Drama Series, Outstanding Lead Actor in a Drama Series, etc.); 
(3) The total number of Emmy awards won by the series at the 76th Primetime Emmy Awards (including both main ceremony and Creative Arts ceremonies); 
(4) The season or part number that was eligible for these awards; 
(5) At least one principal cast member's name and their character name in the series; 
(6) The original production company or studio behind the series; 
(7) The series' episode count for the eligible season; 
(8) Reference URLs to verify: the streaming platform, the specific Emmy win, the total Emmy count, the cast member and character, and the episode count. 
Note: 'Streaming platform original' refers to series primarily distributed by streaming services (Netflix, FX/Hulu, HBO/Max, Apple TV+, Prime Video, Peacock, etc.) rather than broadcast networks (ABC, CBS, NBC, Fox, The CW). 
The 76th Primetime Emmy Awards honored programming from the eligibility period of June 1, 2023 to May 31, 2024.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CastRole(BaseModel):
    actor_name: Optional[str] = None
    character_name: Optional[str] = None


class RefBundle(BaseModel):
    platform_urls: List[str] = Field(default_factory=list)
    emmy_category_urls: List[str] = Field(default_factory=list)
    total_emmy_count_urls: List[str] = Field(default_factory=list)
    season_urls: List[str] = Field(default_factory=list)
    cast_character_urls: List[str] = Field(default_factory=list)
    production_company_urls: List[str] = Field(default_factory=list)
    episode_count_urls: List[str] = Field(default_factory=list)


class SeriesInfo(BaseModel):
    title: Optional[str] = None
    primary_streaming_platform: Optional[str] = None
    categories_won: List[str] = Field(default_factory=list)
    total_wins_count_at_76th: Optional[str] = None
    eligible_season_or_part: Optional[str] = None
    principal_cast: List[CastRole] = Field(default_factory=list)
    production_company_or_studio: Optional[str] = None
    episode_count_for_eligible_season: Optional[str] = None
    refs: RefBundle = Field(default_factory=RefBundle)
    streaming_original_urls: List[str] = Field(default_factory=list)  # Evidence that it's a streaming original (often overlaps with platform_urls)


class SeriesExtraction(BaseModel):
    series: List[SeriesInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_entries() -> str:
    return """
    Extract up to five streaming-platform original series entries that the answer claims won awards at the 76th Primetime Emmy Awards (September 2024).
    For each series entry, return an object with these fields:

    1. title: The series title (string).
    2. primary_streaming_platform: The primary streaming platform (string), e.g., Netflix, Hulu, Max (HBO), Apple TV+, Prime Video, Peacock, FX/Hulu.
    3. categories_won: An array of at least one specific Emmy award category the series WON at the 76th Primetime Emmy Awards (can be from main ceremony or Creative Arts), e.g., ["Outstanding Drama Series", "Outstanding Lead Actor in a Drama Series"].
    4. total_wins_count_at_76th: The total number of Emmy awards that this series won at the 76th Primetime Emmy Awards (including Creative Arts) — return EXACTLY what the answer states, as a string (e.g., "6" or "six").
    5. eligible_season_or_part: The season number or part that was eligible (string; e.g., "Season 2", "Part 1").
    6. principal_cast: An array with at least one object containing:
       - actor_name: Name of a principal cast member (string).
       - character_name: The character they play (string).
    7. production_company_or_studio: The original production company/studio behind the series (string).
    8. episode_count_for_eligible_season: The episode count for the eligible season/part, as stated in the answer (string).

    9. refs: Reference URLs arrays specifically cited in the answer to verify each corresponding claim:
       - platform_urls: URLs supporting the stated primary streaming platform.
       - emmy_category_urls: URLs supporting the specific Emmy category win at the 76th.
       - total_emmy_count_urls: URLs supporting the total Emmy wins count at the 76th (main + Creative Arts).
       - season_urls: URLs supporting the eligible season/part information.
       - cast_character_urls: URLs supporting the actor-character pairing.
       - production_company_urls: URLs supporting the production company/studio.
       - episode_count_urls: URLs supporting the episode count for the eligible season/part.

    10. streaming_original_urls: URLs that support the claim that this series is a streaming-platform original (often the same as platform_urls or official platform pages/press; include any that the answer provides).

    IMPORTANT:
    - Only extract URLs that appear explicitly in the answer (plain URL or markdown link).
    - If the answer does not provide a given field, set it to null for single fields or [] for arrays.
    - Do not invent or infer any data beyond the answer; capture exactly as written (including numbers as strings).
    - Limit to the first five series found in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def first_non_empty(items: List[str]) -> str:
    for x in items:
        if x and x.strip():
            return x.strip()
    return ""


def get_first_category(series: SeriesInfo) -> str:
    return series.categories_won[0] if series.categories_won else ""


def get_first_cast_pair(series: SeriesInfo) -> CastRole:
    return series.principal_cast[0] if series.principal_cast else CastRole()


# --------------------------------------------------------------------------- #
# Verification for a single series                                            #
# --------------------------------------------------------------------------- #
async def verify_single_series(
    evaluator: Evaluator,
    parent_node,
    series: SeriesInfo,
    index: int
) -> "SeriesNode":
    """
    Build verification nodes for a single series candidate and run checks.
    Returns the created top node for this series (to be used in counting fully qualified).
    """
    sid = f"S{index}"
    title_safe = series.title or ""

    # Create the series node (parallel, non-critical to allow independent scoring)
    series_node = evaluator.add_parallel(
        id=f"Series_{index}",
        desc=f"Series candidate #{index} (if present).",
        parent=parent_node,
        critical=False
    )

    # ----- Critical existence checks (Provided) -----
    evaluator.add_custom_node(
        result=bool(series.title and series.title.strip()),
        id=f"{sid}_Title_Provided",
        desc="Provides the series title.",
        parent=series_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(series.primary_streaming_platform and series.primary_streaming_platform.strip()),
        id=f"{sid}_Primary_Streaming_Platform_Provided",
        desc="Provides the primary streaming platform for the series.",
        parent=series_node,
        critical=True
    )

    # Streaming original verification: needs sources present
    # Presence prerequisite for streaming-original verification
    streaming_original_sources = list(series.streaming_original_urls or []) + list(series.refs.platform_urls or [])
    prereq_streaming_original_sources = evaluator.add_custom_node(
        result=len(streaming_original_sources) > 0,
        id=f"{sid}_Streaming_Original_Source_Provided",
        desc="At least one URL is provided to support streaming-original status.",
        parent=series_node,
        critical=True
    )

    stream_orig_leaf = evaluator.add_leaf(
        id=f"{sid}_Streaming_Original_Not_Broadcast",
        desc="Series qualifies as a streaming-platform original (primarily distributed by a streaming service) and not a broadcast-network series, per the question definition.",
        parent=series_node,
        critical=True
    )
    claim_streaming_original = (
        f"The series '{title_safe}' is a streaming-platform original primarily distributed by {series.primary_streaming_platform}, "
        f"and it is not a broadcast-network series (e.g., ABC, CBS, NBC, Fox, The CW)."
    )
    await evaluator.verify(
        claim=claim_streaming_original,
        node=stream_orig_leaf,
        sources=streaming_original_sources,
        additional_instruction="Rely on platform pages, official press, or credible sources to confirm streaming-original status; "
                              "FX/Hulu, HBO/Max, Apple TV+, Prime Video, Netflix, and Peacock qualify as streaming-platform distribution. "
                              "If evidence suggests broadcast-network origin, mark incorrect.",
        extra_prerequisites=[prereq_streaming_original_sources]
    )

    evaluator.add_custom_node(
        result=bool(series.categories_won and len(series.categories_won) > 0),
        id=f"{sid}_Emmy_Win_Category_Identified_At_76th",
        desc="Identifies at least one specific Emmy award category that the series won at the 76th Primetime Emmy Awards (main ceremony or Creative Arts).",
        parent=series_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(series.total_wins_count_at_76th and series.total_wins_count_at_76th.strip()),
        id=f"{sid}_Total_Emmy_Wins_Count_Provided_At_76th",
        desc="Provides the total number of Emmy awards won by the series at the 76th Primetime Emmy Awards (main + Creative Arts).",
        parent=series_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(series.eligible_season_or_part and series.eligible_season_or_part.strip()),
        id=f"{sid}_Eligible_Season_Or_Part_Provided",
        desc="Specifies the season or part number that was eligible for the cited awards.",
        parent=series_node,
        critical=True
    )

    # Cast existence check: requires at least one pair with both actor and character
    first_cast = get_first_cast_pair(series)
    cast_provided = bool(first_cast.actor_name and first_cast.actor_name.strip() and first_cast.character_name and first_cast.character_name.strip())
    evaluator.add_custom_node(
        result=cast_provided,
        id=f"{sid}_Principal_Cast_And_Character_Provided",
        desc="Provides at least one principal cast member name and their character name in the series.",
        parent=series_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(series.production_company_or_studio and series.production_company_or_studio.strip()),
        id=f"{sid}_Production_Company_Or_Studio_Provided",
        desc="Identifies the original production company or studio behind the series.",
        parent=series_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(series.episode_count_for_eligible_season and series.episode_count_for_eligible_season.strip()),
        id=f"{sid}_Episode_Count_For_Eligible_Season_Provided",
        desc="Provides the episode count for the eligible season/part.",
        parent=series_node,
        critical=True
    )

    # ----- References Verification aggregator (parallel, critical) -----
    refs_node = evaluator.add_parallel(
        id=f"{sid}_References_Verify_All_Required_Claims",
        desc="Provides reference URLs that verify each required claim for this series (per the constraint that all information must be verifiable).",
        parent=series_node,
        critical=True
    )

    # 1) Platform verification
    platform_sources = series.refs.platform_urls or []
    prereq_platform_sources = evaluator.add_custom_node(
        result=len(platform_sources) > 0,
        id=f"{sid}_Ref_Platform_Source_Provided",
        desc="At least one provided URL exists to support the stated primary streaming platform.",
        parent=refs_node,
        critical=True
    )
    platform_leaf = evaluator.add_leaf(
        id=f"{sid}_Ref_Verifies_Platform",
        desc="At least one provided URL supports the stated primary streaming platform.",
        parent=refs_node,
        critical=True
    )
    claim_platform = f"The primary streaming platform for '{title_safe}' is {series.primary_streaming_platform}."
    await evaluator.verify(
        claim=claim_platform,
        node=platform_leaf,
        sources=platform_sources,
        additional_instruction="Use platform landing pages or credible sources to confirm the platform attribution; allow reasonable naming variants.",
        extra_prerequisites=[prereq_platform_sources]
    )

    # 2) Emmy category verification (use first category)
    category_str = get_first_category(series)
    emmy_cat_sources = series.refs.emmy_category_urls or []
    prereq_emmy_cat_sources = evaluator.add_custom_node(
        result=len(emmy_cat_sources) > 0,
        id=f"{sid}_Ref_Emmy_Category_Source_Provided",
        desc="At least one provided URL exists to support the cited Emmy category win at the 76th.",
        parent=refs_node,
        critical=True
    )
    emmy_category_leaf = evaluator.add_leaf(
        id=f"{sid}_Ref_Verifies_Emmy_Win_Category",
        desc="At least one provided URL supports that the series won the cited Emmy category at the 76th Primetime Emmy Awards (main or Creative Arts).",
        parent=refs_node,
        critical=True
    )
    claim_emmy_category = (
        f"At the 76th Primetime Emmy Awards (September 2024), the series '{title_safe}' won the category: {category_str}."
    )
    await evaluator.verify(
        claim=claim_emmy_category,
        node=emmy_category_leaf,
        sources=emmy_cat_sources,
        additional_instruction="Confirm the win specifically at the 76th Primetime Emmy Awards; Creative Arts wins count. "
                              "Use official Emmys site, trade press, or credible news coverage. Ignore nominations-only pages.",
        extra_prerequisites=[prereq_emmy_cat_sources]
    )

    # 3) Total Emmy wins count verification
    total_count_sources = series.refs.total_emmy_count_urls or []
    prereq_total_count_sources = evaluator.add_custom_node(
        result=len(total_count_sources) > 0,
        id=f"{sid}_Ref_Total_Count_Source_Provided",
        desc="At least one provided URL exists to support the total Emmy wins count at the 76th.",
        parent=refs_node,
        critical=True
    )
    total_count_leaf = evaluator.add_leaf(
        id=f"{sid}_Ref_Verifies_Total_Emmy_Win_Count",
        desc="At least one provided URL supports the stated total number of Emmy wins for the series at the 76th Primetime Emmy Awards.",
        parent=refs_node,
        critical=True
    )
    claim_total_count = (
        f"The series '{title_safe}' won a total of {series.total_wins_count_at_76th} Emmy awards at the 76th Primetime Emmy Awards, including Creative Arts."
    )
    await evaluator.verify(
        claim=claim_total_count,
        node=total_count_leaf,
        sources=total_count_sources,
        additional_instruction="Verify the total count across main and Creative Arts ceremonies; accept reasonable numeric formatting (e.g., '6' vs 'six').",
        extra_prerequisites=[prereq_total_count_sources]
    )

    # 4) Eligible season/part verification
    season_sources = series.refs.season_urls or []
    prereq_season_sources = evaluator.add_custom_node(
        result=len(season_sources) > 0,
        id=f"{sid}_Ref_Season_Source_Provided",
        desc="At least one provided URL exists to support the stated eligible season/part.",
        parent=refs_node,
        critical=True
    )
    season_leaf = evaluator.add_leaf(
        id=f"{sid}_Ref_Verifies_Eligible_Season_Or_Part",
        desc="At least one provided URL supports the stated eligible season/part information.",
        parent=refs_node,
        critical=True
    )
    claim_season = f"The eligible season/part for '{title_safe}' was {series.eligible_season_or_part}."
    await evaluator.verify(
        claim=claim_season,
        node=season_leaf,
        sources=season_sources,
        additional_instruction="Confirm the eligibility season/part cited for the awards period (June 1, 2023–May 31, 2024).",
        extra_prerequisites=[prereq_season_sources]
    )

    # 5) Cast and character verification
    cast_sources = series.refs.cast_character_urls or []
    prereq_cast_sources = evaluator.add_custom_node(
        result=len(cast_sources) > 0,
        id=f"{sid}_Ref_Cast_Source_Provided",
        desc="At least one provided URL exists to support the actor-character pairing.",
        parent=refs_node,
        critical=True
    )
    cast_leaf = evaluator.add_leaf(
        id=f"{sid}_Ref_Verifies_Cast_And_Character",
        desc="At least one provided URL supports the stated cast member and character pairing.",
        parent=refs_node,
        critical=True
    )
    cast_claim = (
        f"In '{title_safe}', {first_cast.actor_name} plays {first_cast.character_name}."
    )
    await evaluator.verify(
        claim=cast_claim,
        node=cast_leaf,
        sources=cast_sources,
        additional_instruction="Use official series pages, platform listings, or credible databases/news to verify the actor-character pairing.",
        extra_prerequisites=[prereq_cast_sources]
    )

    # 6) Production company/studio verification
    prod_sources = series.refs.production_company_urls or []
    prereq_prod_sources = evaluator.add_custom_node(
        result=len(prod_sources) > 0,
        id=f"{sid}_Ref_Production_Source_Provided",
        desc="At least one provided URL exists to support the production company/studio.",
        parent=refs_node,
        critical=True
    )
    prod_leaf = evaluator.add_leaf(
        id=f"{sid}_Ref_Verifies_Production_Company_Or_Studio",
        desc="At least one provided URL supports the stated production company/studio.",
        parent=refs_node,
        critical=True
    )
    prod_claim = f"The original production company/studio behind '{title_safe}' is {series.production_company_or_studio}."
    await evaluator.verify(
        claim=prod_claim,
        node=prod_leaf,
        sources=prod_sources,
        additional_instruction="Verify the original production company/studio as stated; accept co-production context if clearly indicated.",
        extra_prerequisites=[prereq_prod_sources]
    )

    # 7) Episode count verification
    ep_sources = series.refs.episode_count_urls or []
    prereq_ep_sources = evaluator.add_custom_node(
        result=len(ep_sources) > 0,
        id=f"{sid}_Ref_EpisodeCount_Source_Provided",
        desc="At least one provided URL exists to support the episode count for the eligible season/part.",
        parent=refs_node,
        critical=True
    )
    ep_leaf = evaluator.add_leaf(
        id=f"{sid}_Ref_Verifies_Episode_Count",
        desc="At least one provided URL supports the stated episode count for the eligible season/part.",
        parent=refs_node,
        critical=True
    )
    ep_claim = (
        f"The eligible season/part {series.eligible_season_or_part} of '{title_safe}' has {series.episode_count_for_eligible_season} episodes."
    )
    await evaluator.verify(
        claim=ep_claim,
        node=ep_leaf,
        sources=ep_sources,
        additional_instruction="Confirm the episode count for the specific season/part cited; allow minor formatting differences.",
        extra_prerequisites=[prereq_ep_sources]
    )

    return series_node


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
    Evaluate an answer for the 76th Primetime Emmys streaming originals task.
    """
    # Initialize evaluator (root is non-critical by design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Create Task Root (we keep this non-critical to satisfy framework constraints,
    # and use a critical leaf to enforce pass/fail at the task level)
    task_root = evaluator.add_sequential(
        id="Task_Root",
        desc="Identify at least three streaming-platform original series that won at least one Emmy at the 76th Primetime Emmy Awards, and for each provide all required attributes with verifiable reference URLs.",
        parent=root,
        critical=False
    )

    # Series candidates aggregator
    series_candidates = evaluator.add_parallel(
        id="Series_Candidates",
        desc="Evaluate up to five series entries supplied in the answer; each series entry is scored independently.",
        parent=task_root,
        critical=False
    )

    # Extract series entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_series_entries(),
        template_class=SeriesExtraction,
        extraction_name="series_entries"
    )

    # Normalize to up to five entries; pad with empty placeholders if fewer provided
    series_list: List[SeriesInfo] = list(extracted.series or [])
    series_list = series_list[:5]
    while len(series_list) < 5:
        series_list.append(SeriesInfo())

    # Verify each series candidate
    series_nodes = []
    for idx in range(1, 6):
        node = await verify_single_series(
            evaluator,
            series_candidates,
            series_list[idx - 1],
            idx
        )
        series_nodes.append(node)

    # Compute how many series fully satisfy their critical requirements
    qualified_count = 0
    for node in series_nodes:
        score = node.compute_score(mutate=True)
        # A fully qualified series must have passed all critical checks (score == 1.0 here)
        if score == 1.0:
            qualified_count += 1

    # Add critical task-level requirement: at least three fully qualified series
    evaluator.add_custom_node(
        result=(qualified_count >= 3),
        id="At_Least_Three_Series_Fully_Qualify",
        desc="At least three of the evaluated series candidates (Series_1–Series_5) satisfy all of their critical requirements.",
        parent=task_root,
        critical=True
    )

    # Record custom info summary
    evaluator.add_custom_info(
        info={"qualified_series_count": qualified_count, "evaluated_series": 5},
        info_type="metric",
        info_name="task_metrics"
    )

    # Return structured evaluation result
    return evaluator.get_summary()