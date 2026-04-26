import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "frank_darabont_st_s5_first_ep"
TASK_DESCRIPTION = (
    "Frank Darabont, known for directing The Shawshank Redemption, came out of retirement after 11 years to direct "
    "episodes for the fifth and final season of Netflix's Stranger Things. For his first directed episode in Season 5 "
    "(by episode number order), identify: (1) The episode number and complete title (in 'Chapter [Number]: [Title]' format), "
    "(2) The writer who wrote the episode, (3) The release date, and (4) Which volume (Volume 1, Volume 2, or Finale) "
    "the episode was released in. Provide a reference URL that verifies this information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FirstEpisodeInfo(BaseModel):
    """
    Information about the first episode (by episode number order) in Stranger Things Season 5
    that the answer claims Frank Darabont directed.
    """
    # Episode number as mentioned in the answer; keep as free-form string (e.g., "1", "Episode 1", "Chapter One")
    episode_number: Optional[str] = None

    # Full title as provided in the answer. Ideally "Chapter [Number]: [Title]".
    episode_title_full: Optional[str] = None

    # The credited writer (free-form string, can be a single name or multiple names as written in the answer)
    writer: Optional[str] = None

    # Release date (keep as raw string, e.g., "July 4, 2026" or "2026-07-04")
    release_date: Optional[str] = None

    # Volume identifier (e.g., "Volume 1", "Volume 2", "Finale")
    volume: Optional[str] = None

    # Reference URLs included in the answer to support the above details
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_first_episode_info() -> str:
    return """
    From the answer, extract the SINGLE Season 5 episode that is claimed to be the first (by episode number order)
    that Frank Darabont directed. Only consider Stranger Things Season 5.

    If the answer lists multiple episodes Frank Darabont directed, select the one with the lowest episode number.
    If episode numbers are given in different formats (e.g., "Chapter One", "Episode 1", "S05E01"), still choose
    the earliest by order.

    Extract the following fields for that episode (use exactly what the answer states; do not invent):
    - episode_number: the episode number as text (e.g., "1", "Episode 1", "Chapter One", "S05E01")
    - episode_title_full: the full title as written in the answer (ideally "Chapter [Number]: [Title]")
    - writer: the writer credited for the episode (as written in the answer)
    - release_date: the release date (as written in the answer)
    - volume: which volume the episode was released in (e.g., "Volume 1", "Volume 2", "Finale"), if mentioned
    - reference_urls: list of all URLs in the answer that support this episode’s details; include only valid URLs.
    
    If any piece of information is missing in the answer, set that field to null (or an empty array for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


def _safe_str(x: Optional[str]) -> str:
    return x.strip() if isinstance(x, str) else ""


# --------------------------------------------------------------------------- #
# Verification builder functions                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_episode_identification(
    evaluator: Evaluator,
    parent_node,
    info: FirstEpisodeInfo,
) -> None:
    """
    Step 1 (Sequential): Identify the first (by episode number) Season 5 episode Frank Darabont directed.
    Single critical leaf node that verifies the identification against provided sources.
    If required data (title, episode_number, and at least one reference URL) are missing, the leaf fails.
    """
    # Create the identification leaf node
    id_node = evaluator.add_leaf(
        id="Episode_Identification",
        desc="Identify the episode number of Frank Darabont's first episode in Season 5 (by episode number order)",
        parent=parent_node,
        critical=True,
        score=0.0,
        status="initialized",
    )

    # Basic required info check: title + episode_number + at least one URL
    ep_num = _safe_str(info.episode_number)
    ep_title = _safe_str(info.episode_title_full)
    urls = info.reference_urls

    if not ep_num or not ep_title or not _has_any_url(urls):
        # Directly fail the identification if insufficient info to verify
        id_node.score = 0.0
        id_node.status = "failed"
        return

    # Build the claim. Ask the verifier to check:
    # 1) Frank Darabont directed this episode.
    # 2) The episode is Season 5, and has the given number and title.
    # 3) Among Season 5 episodes he directs, this is the earliest by episode number (if the page lists multiple).
    claim = (
        f"According to the provided source(s), Frank Darabont directed the Stranger Things Season 5 episode "
        f"titled '{ep_title}', which is identified in Season 5 as episode '{ep_num}'. "
        f"If multiple Season 5 episodes directed by Frank Darabont are shown, confirm that '{ep_title}' has "
        f"the lowest episode number among them (i.e., it is his first by episode order)."
    )

    add_ins = (
        "Focus strictly on the webpage(s). Accept minor formatting variants for episode numbering (e.g., "
        "'Episode 1', 'Chapter One', 'S05E01'). If only one directed episode is shown for Season 5, consider it as his first. "
        "If the page is about a different show/season or does not support these details, mark as Incorrect."
    )

    await evaluator.verify(
        claim=claim,
        node=id_node,
        sources=urls,
        additional_instruction=add_ins,
    )


async def build_and_verify_episode_attributes(
    evaluator: Evaluator,
    parent_node,
    info: FirstEpisodeInfo,
) -> None:
    """
    Step 2 (Parallel & Critical): Verify each required attribute against the provided reference URL(s).
    Children:
      - Episode_Title (critical)
      - Episode_Writer (critical)
      - Episode_Release_Date (critical)
      - Episode_Volume (critical)
      - Reference_URL (critical existence)
    """
    attrs_node = evaluator.add_parallel(
        id="Episode_Attributes_Verification",
        desc="Provide and verify all required attributes of the identified episode",
        parent=parent_node,
        critical=True,
    )

    # 2.e Reference URL existence (critical)
    ref_exists = _has_any_url(info.reference_urls)
    ref_node = evaluator.add_custom_node(
        result=ref_exists,
        id="Reference_URL",
        desc="Provide a reference URL that verifies the episode number/title/writer/release date/volume information",
        parent=attrs_node,
        critical=True,
    )

    # Prepare common data
    ep_num = _safe_str(info.episode_number)
    ep_title = _safe_str(info.episode_title_full)
    writer = _safe_str(info.writer)
    rel_date = _safe_str(info.release_date)
    volume = _safe_str(info.volume)
    urls = info.reference_urls

    # 2.a Episode Title (critical)
    title_node = evaluator.add_leaf(
        id="Episode_Title",
        desc="Provide the complete episode title in the format 'Chapter [Number]: [Title]'",
        parent=attrs_node,
        critical=True,
    )
    title_claim = (
        f"The Season 5 episode identified by the answer is titled exactly '{ep_title}'. "
        f"If the page displays 'Chapter [Number]: [Title]' formatting, confirm it matches this text. "
        f"If the page uses a variation (e.g., 'Episode {ep_num} – [Title]'), accept it only if it clearly corresponds "
        f"to the same episode and full title."
    )
    title_instruction = (
        "Allow minor punctuation or typographic variants. If both the number and the title correspond to the same episode, "
        "consider it a match. Reject if the page shows a different episode title or season."
    )

    # 2.b Episode Writer (critical)
    writer_node = evaluator.add_leaf(
        id="Episode_Writer",
        desc="Identify the writer who wrote the episode",
        parent=attrs_node,
        critical=True,
    )
    writer_claim = (
        f"The credited writer(s) for the episode include '{writer}'. "
        f"Accept if the page lists this person among the writers. If multiple writers are listed, it is sufficient "
        f"that '{writer}' is one of them."
    )
    writer_instruction = (
        "Look for 'Written by' or writing credits on the page. Accept reasonable naming variants (middle initials, "
        "ampersands vs 'and'). Reject if the credited writers do not include the provided name."
    )

    # 2.c Release Date (critical)
    release_node = evaluator.add_leaf(
        id="Episode_Release_Date",
        desc="Provide the release date of the episode",
        parent=attrs_node,
        critical=True,
    )
    release_claim = (
        f"The release date for this episode is '{rel_date}'. "
        f"Accept common date-format variations that clearly correspond to the same calendar date."
    )
    release_instruction = (
        "Check the page for the episode's release or premiere date. Allow minor format differences such as "
        "'2026-07-04' vs 'July 4, 2026'. Reject if the date clearly does not match."
    )

    # 2.d Volume (critical)
    volume_node = evaluator.add_leaf(
        id="Episode_Volume",
        desc="Identify which volume (Volume 1, Volume 2, or Finale) the episode was released in",
        parent=attrs_node,
        critical=True,
    )
    volume_claim = (
        f"This episode was released as part of '{volume}' in Season 5 (i.e., Volume 1, Volume 2, or Finale)."
    )
    volume_instruction = (
        "Confirm that the page explicitly mentions the release grouping (Volume 1, Volume 2) or denotes it as the Finale. "
        "Accept closely equivalent phrasing such as 'final episode(s)' for Finale. Reject if volume information is missing or contradictory."
    )

    # If reference URL is missing, the following verifications should be skipped.
    # Because we executed the ref_node first and it is critical, verify() will auto-skip if ref_node failed.
    claims_to_verify = [
        (title_claim, urls, title_node, title_instruction),
        (writer_claim, urls, writer_node, writer_instruction),
        (release_claim, urls, release_node, release_instruction),
        (volume_claim, urls, volume_node, volume_instruction),
    ]
    await evaluator.batch_verify(claims_to_verify)


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
) -> Dict:
    """
    Evaluate an answer for the Frank Darabont Stranger Things S5 first-episode task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # overall we want Step1 then Step2
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

    # Extract the episode info claimed by the answer
    info = await evaluator.extract(
        prompt=prompt_extract_first_episode_info(),
        template_class=FirstEpisodeInfo,
        extraction_name="first_episode_info",
    )

    # Build a critical sequential node to mirror the rubric
    research_node = evaluator.add_sequential(
        id="Frank_Darabont_First_Episode_Research",
        desc="Research and identify details about Frank Darabont's first episode in Stranger Things Season 5",
        parent=root,
        critical=True,
    )

    # Step 1: Episode Identification (critical leaf)
    await build_and_verify_episode_identification(evaluator, research_node, info)

    # Step 2: Episode Attributes Verification (critical parallel children)
    await build_and_verify_episode_attributes(evaluator, research_node, info)

    # Return the evaluation summary
    return evaluator.get_summary()