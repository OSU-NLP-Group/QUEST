import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "emmys_2024_series_identification"
TASK_DESCRIPTION = (
    "What is the title of the 2024 limited series that meets all of the following criteria: "
    "(1) Won the Outstanding Limited or Anthology Series award at the 76th Primetime Emmy Awards (ceremony held September 15, 2024), "
    "(2) Won exactly 6 total Primetime Emmy Awards (including both the main ceremony and Creative Arts Emmy Awards), "
    "(3) The lead actor won both the Outstanding Lead Actor in a Limited or Anthology Series or Movie Emmy AND the Outstanding Writing for a Limited or Anthology Series or Movie Emmy at the 2024 ceremony, "
    "(4) A supporting actress from the series won the Outstanding Supporting Actress in a Limited or Anthology Series or Movie Emmy at the 2024 ceremony, "
    "(5) Was released on Netflix, "
    "(6) Had all episodes released simultaneously (not on a weekly schedule), "
    "(7) Was released in 2024, "
    "(8) Consists of exactly 7 episodes, "
    "(9) Features a lead character (played by the Emmy-winning lead actor) who works as a bartender, and "
    "(10) Features a supporting character (played by the Emmy-winning supporting actress) who has a criminal past?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SeriesExtraction(BaseModel):
    # Core identification
    series_title: Optional[str] = None
    lead_actor_name: Optional[str] = None
    supporting_actress_name: Optional[str] = None

    # Production & release
    platform: Optional[str] = None
    release_strategy: Optional[str] = None  # e.g., "all episodes released simultaneously"
    release_year: Optional[str] = None
    episode_count: Optional[str] = None

    # Source URLs per verification item
    outstanding_limited_series_urls: List[str] = Field(default_factory=list)
    total_emmy_count_urls: List[str] = Field(default_factory=list)
    lead_actor_emmy_urls: List[str] = Field(default_factory=list)
    writing_emmy_urls: List[str] = Field(default_factory=list)
    supporting_actress_emmy_urls: List[str] = Field(default_factory=list)
    streaming_platform_urls: List[str] = Field(default_factory=list)
    release_strategy_urls: List[str] = Field(default_factory=list)
    release_year_urls: List[str] = Field(default_factory=list)
    episode_count_urls: List[str] = Field(default_factory=list)
    lead_character_occupation_urls: List[str] = Field(default_factory=list)
    supporting_character_background_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_info() -> str:
    return """
    Extract the single limited series identified in the answer along with key fields and the specific URLs cited for each verification requirement. Only extract information explicitly present in the answer text.

    Required text fields (use null if not mentioned):
    - series_title: The title of the limited series.
    - lead_actor_name: Name of the series' lead actor (the one referenced as the Emmy-winning lead actor).
    - supporting_actress_name: Name of the supporting actress (the one referenced as the Emmy-winning supporting actress).
    - platform: The streaming platform (e.g., Netflix).
    - release_strategy: The release approach (e.g., "all episodes released simultaneously").
    - release_year: The year the series was released (expected 2024).
    - episode_count: The number of episodes (expected "7" or a phrase that clearly indicates 7).

    For each of the following categories, extract all URLs explicitly cited in the answer that support the claim. If none are cited, return an empty array. Do not invent URLs. Include only valid, complete URLs:
    - outstanding_limited_series_urls: URLs confirming the series won Outstanding Limited or Anthology Series at the 76th Primetime Emmy Awards (2024).
    - total_emmy_count_urls: URLs confirming the series won exactly 6 total Primetime Emmy Awards in 2024 (including Creative Arts Emmys).
    - lead_actor_emmy_urls: URLs confirming the series' lead actor won Outstanding Lead Actor in a Limited or Anthology Series or Movie at the 2024 Emmys.
    - writing_emmy_urls: URLs confirming the same lead actor also won Outstanding Writing for a Limited or Anthology Series or Movie at the 2024 Emmys.
    - supporting_actress_emmy_urls: URLs confirming a supporting actress from the series won Outstanding Supporting Actress in a Limited or Anthology Series or Movie at the 2024 Emmys.
    - streaming_platform_urls: URLs confirming the series was released on Netflix.
    - release_strategy_urls: URLs confirming all episodes were released simultaneously (not weekly).
    - release_year_urls: URLs confirming the series was released in 2024.
    - episode_count_urls: URLs confirming the series consists of exactly 7 episodes.
    - lead_character_occupation_urls: URLs confirming the lead character (played by the lead actor) works as a bartender.
    - supporting_character_background_urls: URLs confirming the supporting character (played by the supporting actress) has a criminal past.

    Output a single JSON object exactly following the SeriesExtraction schema.
    """


# --------------------------------------------------------------------------- #
# Verification tree builder and checks                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_series_tree(evaluator: Evaluator, extracted: SeriesExtraction):
    """
    Construct the verification tree reflecting the rubric and perform all checks.
    All nodes under the critical aggregators are set to critical=True to respect the framework constraint.
    """

    # Top-level critical series identification aggregator
    series_node = evaluator.add_parallel(
        id="SeriesIdentification",
        desc="Identify the limited series that meets all specified Emmy, production, episode, and character criteria from the 2024 Emmy ceremony",
        parent=evaluator.root,
        critical=True
    )

    # -------------------- EmmyAwardsVerification (critical, parallel) --------------------
    emmy_node = evaluator.add_parallel(
        id="EmmyAwardsVerification",
        desc="Verify all Emmy Award achievements from the 76th Primetime Emmy Awards (2024)",
        parent=series_node,
        critical=True
    )

    # Outstanding Limited Series win (critical)
    ols_parent = evaluator.add_parallel(
        id="OutstandingLimitedSeriesWin",
        desc="The series won the Outstanding Limited or Anthology Series award at the 76th Primetime Emmy Awards",
        parent=emmy_node,
        critical=True
    )
    ols_leaf = evaluator.add_leaf(
        id="OutstandingLimitedSeriesURL",
        desc="URL reference confirming the Outstanding Limited or Anthology Series win",
        parent=ols_parent,
        critical=True
    )
    claim_ols = f"The series '{extracted.series_title or ''}' won the Outstanding Limited or Anthology Series award at the 76th Primetime Emmy Awards (2024)."
    await evaluator.verify(
        claim=claim_ols,
        node=ols_leaf,
        sources=extracted.outstanding_limited_series_urls,
        additional_instruction="Confirm this is the 76th Primetime Emmy Awards winners list (2024). Minor title formatting variants are acceptable. The claim is about the series winning the category."
    )

    # Total Emmy count exactly 6 (critical)
    total_parent = evaluator.add_parallel(
        id="TotalEmmyCount",
        desc="The series won exactly 6 total Primetime Emmy Awards (including both main ceremony and Creative Arts Emmy Awards)",
        parent=emmy_node,
        critical=True
    )
    total_leaf = evaluator.add_leaf(
        id="TotalEmmyCountURL",
        desc="URL reference confirming the total of 6 Emmy wins",
        parent=total_parent,
        critical=True
    )
    claim_total = f"The series '{extracted.series_title or ''}' won exactly 6 total Primetime Emmy Awards at the 2024 Emmys, counting both the main telecast and the Creative Arts Emmys."
    await evaluator.verify(
        claim=claim_total,
        node=total_leaf,
        sources=extracted.total_emmy_count_urls,
        additional_instruction="Verify that the referenced page(s) explicitly indicate a total of 6 Emmy wins in 2024 for this series, including any Creative Arts awards."
    )

    # Lead actor achievements (critical, parallel)
    lead_ach_parent = evaluator.add_parallel(
        id="LeadActorAchievements",
        desc="The lead actor won both acting and writing Emmy awards",
        parent=emmy_node,
        critical=True
    )

    # Lead Actor Emmy win (critical)
    lead_actor_parent = evaluator.add_parallel(
        id="LeadActorEmmyWin",
        desc="The series' lead actor won the Outstanding Lead Actor in a Limited or Anthology Series or Movie Emmy at the 2024 ceremony",
        parent=lead_ach_parent,
        critical=True
    )
    lead_actor_leaf = evaluator.add_leaf(
        id="LeadActorEmmyURL",
        desc="URL reference confirming the lead actor's Emmy win",
        parent=lead_actor_parent,
        critical=True
    )
    claim_lead_actor = f"The lead actor '{extracted.lead_actor_name or ''}' won the 2024 Primetime Emmy for Outstanding Lead Actor in a Limited or Anthology Series or Movie for '{extracted.series_title or ''}'."
    await evaluator.verify(
        claim=claim_lead_actor,
        node=lead_actor_leaf,
        sources=extracted.lead_actor_emmy_urls,
        additional_instruction="Confirm the winner's name matches the lead actor specified and that the award is for the 2024 Emmys (76th) and for the identified series."
    )

    # Writing Emmy win by same lead actor (critical)
    writing_parent = evaluator.add_parallel(
        id="WritingEmmyWin",
        desc="The same lead actor won the Outstanding Writing for a Limited or Anthology Series or Movie Emmy at the 2024 ceremony",
        parent=lead_ach_parent,
        critical=True
    )
    writing_leaf = evaluator.add_leaf(
        id="WritingEmmyURL",
        desc="URL reference confirming the writing Emmy win by the lead actor",
        parent=writing_parent,
        critical=True
    )
    claim_writing = f"The same person '{extracted.lead_actor_name or ''}' also won the 2024 Primetime Emmy for Outstanding Writing for a Limited or Anthology Series or Movie for '{extracted.series_title or ''}'."
    await evaluator.verify(
        claim=claim_writing,
        node=writing_leaf,
        sources=extracted.writing_emmy_urls,
        additional_instruction="Confirm the writing award winner is exactly the same individual as the lead actor named, at the 2024 Emmys, for the same series."
    )

    # Supporting actress Emmy win (critical)
    supp_parent = evaluator.add_parallel(
        id="SupportingActressEmmyWin",
        desc="A supporting actress from the series won the Outstanding Supporting Actress in a Limited or Anthology Series or Movie Emmy at the 2024 ceremony",
        parent=emmy_node,
        critical=True
    )
    supp_leaf = evaluator.add_leaf(
        id="SupportingActressEmmyURL",
        desc="URL reference confirming the supporting actress Emmy win",
        parent=supp_parent,
        critical=True
    )
    claim_supporting = f"A supporting actress '{extracted.supporting_actress_name or ''}' from the series '{extracted.series_title or ''}' won the 2024 Primetime Emmy for Outstanding Supporting Actress in a Limited or Anthology Series or Movie."
    await evaluator.verify(
        claim=claim_supporting,
        node=supp_leaf,
        sources=extracted.supporting_actress_emmy_urls,
        additional_instruction="Verify that the winner is a supporting actress from the identified series and that the win is at the 2024 Emmys (76th)."
    )

    # -------------------- ProductionDetails (critical, parallel) --------------------
    prod_node = evaluator.add_parallel(
        id="ProductionDetails",
        desc="Verify the series' production and release specifications",
        parent=series_node,
        critical=True
    )

    # Streaming platform (Netflix) (critical)
    platform_parent = evaluator.add_parallel(
        id="StreamingPlatform",
        desc="The series was released on Netflix",
        parent=prod_node,
        critical=True
    )
    platform_leaf = evaluator.add_leaf(
        id="StreamingPlatformURL",
        desc="URL reference confirming Netflix as the streaming platform",
        parent=platform_parent,
        critical=True
    )
    claim_platform = f"The series '{extracted.series_title or ''}' was released on Netflix."
    await evaluator.verify(
        claim=claim_platform,
        node=platform_leaf,
        sources=extracted.streaming_platform_urls,
        additional_instruction="Accept official sources (Netflix page), reputable news, or reliable databases indicating Netflix as the release platform."
    )

    # Release strategy (all episodes simultaneous) (critical)
    strategy_parent = evaluator.add_parallel(
        id="ReleaseStrategy",
        desc="All episodes of the series were released simultaneously (not on a weekly schedule)",
        parent=prod_node,
        critical=True
    )
    strategy_leaf = evaluator.add_leaf(
        id="ReleaseStrategyURL",
        desc="URL reference confirming the simultaneous release of all episodes",
        parent=strategy_parent,
        critical=True
    )
    claim_strategy = f"All episodes of '{extracted.series_title or ''}' were released simultaneously on the same date (not weekly)."
    await evaluator.verify(
        claim=claim_strategy,
        node=strategy_leaf,
        sources=extracted.release_strategy_urls,
        additional_instruction="Look for phrasing like 'all episodes dropped at once' or a single release date for all episodes (typical for Netflix binges)."
    )

    # Release year is 2024 (critical)
    year_parent = evaluator.add_parallel(
        id="ReleaseYear",
        desc="The series was released in 2024",
        parent=prod_node,
        critical=True
    )
    year_leaf = evaluator.add_leaf(
        id="ReleaseYearURL",
        desc="URL reference confirming the 2024 release year",
        parent=year_parent,
        critical=True
    )
    claim_year = f"The series '{extracted.series_title or ''}' was released in 2024."
    await evaluator.verify(
        claim=claim_year,
        node=year_leaf,
        sources=extracted.release_year_urls,
        additional_instruction="Confirm the initial public release year is 2024 (e.g., Netflix release or premiere). Minor regional date variations are acceptable."
    )

    # -------------------- EpisodeCount (critical) --------------------
    ep_node = evaluator.add_parallel(
        id="EpisodeCount",
        desc="The series consists of exactly 7 episodes",
        parent=series_node,
        critical=True
    )
    ep_leaf = evaluator.add_leaf(
        id="EpisodeCountURL",
        desc="URL reference confirming the series has exactly 7 episodes",
        parent=ep_node,
        critical=True
    )
    claim_episodes = f"The series '{extracted.series_title or ''}' consists of exactly 7 episodes."
    await evaluator.verify(
        claim=claim_episodes,
        node=ep_leaf,
        sources=extracted.episode_count_urls,
        additional_instruction="Verify the total number of episodes equals 7 from a credible source (official page, press, reliable database)."
    )

    # -------------------- CharacterDetails (critical, parallel) --------------------
    char_node = evaluator.add_parallel(
        id="CharacterDetails",
        desc="Verify specific character occupation and background details",
        parent=series_node,
        critical=True
    )

    # Lead character occupation (bartender) (critical)
    lead_occ_parent = evaluator.add_parallel(
        id="LeadCharacterOccupation",
        desc="The lead character (played by the Emmy-winning lead actor) works as a bartender",
        parent=char_node,
        critical=True
    )
    lead_occ_leaf = evaluator.add_leaf(
        id="LeadCharacterOccupationURL",
        desc="URL reference confirming the lead character's occupation as a bartender",
        parent=lead_occ_parent,
        critical=True
    )
    claim_lead_occ = f"In the series '{extracted.series_title or ''}', the lead character played by '{extracted.lead_actor_name or ''}' works as a bartender."
    await evaluator.verify(
        claim=claim_lead_occ,
        node=lead_occ_leaf,
        sources=extracted.lead_character_occupation_urls,
        additional_instruction="Confirm by plot summaries, official descriptions, or credible reviews indicating the character's job is bartender."
    )

    # Supporting character background (criminal past) (critical)
    supp_bg_parent = evaluator.add_parallel(
        id="SupportingCharacterBackground",
        desc="The supporting character (played by the Emmy-winning supporting actress) has a criminal past",
        parent=char_node,
        critical=True
    )
    supp_bg_leaf = evaluator.add_leaf(
        id="SupportingCharacterBackgroundURL",
        desc="URL reference confirming the supporting character has a criminal past",
        parent=supp_bg_parent,
        critical=True
    )
    claim_supp_bg = f"In the series '{extracted.series_title or ''}', the supporting character played by '{extracted.supporting_actress_name or ''}' has a criminal past."
    await evaluator.verify(
        claim=claim_supp_bg,
        node=supp_bg_leaf,
        sources=extracted.supporting_character_background_urls,
        additional_instruction="Look for explicit mention that the character has a criminal history (e.g., prior offenses, ex-con, charges). Plot synopses or reputable reviews are acceptable."
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
    Evaluate an agent's answer to the 2024 Emmy-winning limited series identification task.
    """
    # Initialize evaluator (root is non-critical by default)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # High-level items are independent checks aggregated under a critical child
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_series_info(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction",
    )

    # Custom info: record quick summary
    evaluator.add_custom_info(
        {
            "series_title": extracted.series_title,
            "lead_actor_name": extracted.lead_actor_name,
            "supporting_actress_name": extracted.supporting_actress_name,
            "platform": extracted.platform,
            "release_strategy": extracted.release_strategy,
            "release_year": extracted.release_year,
            "episode_count": extracted.episode_count,
        },
        info_type="extracted_summary",
        info_name="extracted_series_summary",
    )

    # Build verification tree and verify all leaves
    await build_and_verify_series_tree(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()