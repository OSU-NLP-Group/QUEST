"""
Evaluation script for task **just_dance** — with explicit *version-mentioned* check
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "just_dance"
TASK_DESCRIPTION = """
Please identify the number of K-pop songs featured in each base version of the Just Dance video game series, from the earliest release up to and including this year's edition. Provide a complete list of the K-pop songs included in each version. Do not count songs available through Just Dance Unlimited, Just Dance+, or as regional exlusives.
"""

GROUND_TRUTH: Dict[str, List[str]] = {
    "Just Dance (2009, the original version)": [],
    "Just Dance 2 (2010)": [],
    "Just Dance 3 (2011)": [],
    "Just Dance 4 (2012)": [],
    "Just Dance 2014": ["Gangnam Style", "Gentleman"],
    "Just Dance 2015": ["Gangnam Style"],
    "Just Dance 2016": [],
    "Just Dance 2017": ["Daddy"],
    "Just Dance 2018": ["New Face", "Bubble Pop!"],
    "Just Dance 2019": ["Bang Bang Bang", "Ddu-Du Ddu-Du"],
    "Just Dance 2020": ["Fancy", "I Am the Best", "Kill This Love"],
    "Just Dance 2021": ["Feel Special", "Ice Cream", "Kick It"],
    "Just Dance 2022": ["Black Mamba", "Boombayah", "Jopping", "Pop/Stars"],
    "Just Dance 2023": ["Boys with Luv", "Dynamite", "More", "Psycho", "Wannabe"],
    "Just Dance 2024": ["Butter", "How You Like That", "Say My Name", "Seven"],
    "Just Dance 2025": ["Pink Venom"],
}

# --------------------------------------------------------------------------- #
# Extraction model & prompt helpers                                           #
# --------------------------------------------------------------------------- #
class VersionInfo(BaseModel):
    mentioned: bool
    songs: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


def _prompt_extract_version_info(version_name: str) -> str:
    special = ""
    if version_name.startswith("Just Dance 2020"):
        special = (
            "\nIMPORTANT: If the answer lists 'Bang Bang Bang' for this version, "
            "ignore it and DO NOT include it in the extracted songs."
        )

    return f"""
Extract data for **{version_name}** from the answer.

Return a JSON object with:
- **mentioned**: true if the answer explicitly mentioned {version_name} in the answer whole answer text. False if it didn't mention it specifically at all.
- **songs**: array of K-pop song titles claimed for this base game version
- **urls** : array of every URL cited for this version

Ignore tracks from Just Dance Unlimited / Just Dance+ / regional exclusives.{special}
""".strip()


# --------------------------------------------------------------------------- #
# Per-song verification                                                       #
# --------------------------------------------------------------------------- #
async def _verify_song_correctness(
    evaluator: Evaluator,
    song: str,
    version: str,
    song_node,
) -> None:
    leaf = evaluator.add_leaf(
        id=f"{song_node.id}_correct",
        desc=f"'{song}' is one of the ground-truth K-pop songs for {version}.",
        parent=song_node,
        critical=True,
    )

    gt_list = ", ".join(GROUND_TRUTH[version]) or "none"
    claim = f"The song '{song}' appears in the song list ({gt_list})."
    
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="Check if the song title matches any in the list, allowing for minor variations in capitalization or punctuation."
    )


async def _verify_song_provenance(
    evaluator: Evaluator,
    song: str,
    version: str,
    urls: List[str],
    song_node,
) -> None:
    leaf = evaluator.add_leaf(
        id=f"{song_node.id}_prov",
        desc=f"A cited URL supports that '{song}' is in {version}'s base track-list.",
        parent=song_node,
        critical=True,
    )

    prov_claim = f"'{song}' is included in the Just Dance {version} track-list (and the page is not for Just Dance Unlimited / + / regional exclusives, or it clearly indicate that the song is in the base track-list.)."

    await evaluator.verify(
        claim=prov_claim,
        node=leaf,
        sources=urls,
        additional_instruction="Verify the URL shows this song is in the base game track list, not in DLC or special editions."
    )


# --------------------------------------------------------------------------- #
# Per-version verification                                                    #
# --------------------------------------------------------------------------- #
async def _verify_version(
    evaluator: Evaluator,
    root,
    version: str,
    info: VersionInfo,
) -> None:
    v_node = evaluator.add_parallel(
        id=f"ver_{version.replace(' ', '_')}",
        desc=f"Verification for {version}.",
        parent=root,
    )

    # Critical check: version must be mentioned
    mention_node = evaluator.add_custom_node(
        result=info.mentioned,
        id=f"{v_node.id}_mentioned",
        desc=f"The answer explicitly mentions {version}.",
        parent=v_node,
        critical=True,
    )

    gt_songs = GROUND_TRUTH[version]
    urls = info.urls

    # No K-pop GT versions
    if not gt_songs:
        empty_node = evaluator.add_custom_node(
            result=len(info.songs) == 0,
            id=f"{v_node.id}_empty",
            desc=f"No K-pop songs should be listed for {version}.",
            parent=v_node,
            critical=True,
        )
        return

    # Versions with GT songs
    k = len(gt_songs)
    songs_to_check = info.songs[:k]  # ignore extras

    for idx in range(k):
        song_node = evaluator.add_parallel(
            id=f"{v_node.id}_song{idx+1}",
            desc=f"Song #{idx+1} verification for {version}.",
            parent=v_node,
        )

        if idx < len(songs_to_check):
            song = songs_to_check[idx]
            
            # Create existence check for song verification
            song_exists_node = evaluator.add_custom_node(
                result=bool(song),
                id=f"{song_node.id}_exists",
                desc=f"Song #{idx+1} was provided in the answer",
                parent=song_node,
                critical=True
            )
            
            await _verify_song_correctness(evaluator, song, version, song_node)
            await _verify_song_provenance(evaluator, song, version, urls, song_node)
        else:
            # Mark as skipped if no song provided
            evaluator.add_custom_node(
                result=False,
                id=f"{song_node.id}_missing",
                desc=f"Song #{idx+1} is missing from the answer",
                parent=song_node,
                critical=True
            )


# --------------------------------------------------------------------------- #
# Main evaluation entry-point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: openai.AsyncAzureOpenAI,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # Set up evaluator
    evaluator = Evaluator()
    
    # Initialize evaluator
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

    # 1. Concurrent extraction for all 16 versions
    extracted_list = await asyncio.gather(*[
        evaluator.extract(
            prompt=_prompt_extract_version_info(v),
            template_class=VersionInfo,
            extraction_name=f"version_{v.replace(' ', '_')}"
        )
        for v in GROUND_TRUTH
    ])
    extracted: Dict[str, VersionInfo] = dict(zip(GROUND_TRUTH, extracted_list))

    # Add extraction results to custom info for summary
    evaluator.add_custom_info(
        {v: info.dict() for v, info in extracted.items()},
        "extracted_info"
    )

    # 2. Build verification tree
    for version in GROUND_TRUTH:
        await _verify_version(evaluator, root, version, extracted[version])

    # 3. Return structured result
    return evaluator.get_summary()