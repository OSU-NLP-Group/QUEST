import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "artist_2024_triple_platform"
TASK_DESCRIPTION = """
In 2024, one artist achieved unprecedented success across three major music industry platforms. Identify the artist who accomplished all of the following in 2024:

1. Won Album of the Year at the Grammy Awards for the album 'Midnights', becoming the first artist to win this category four times
2. Was named Spotify's Global Top Artist with more than 26.6 billion streams globally, and had the most-streamed album on Spotify titled 'THE TORTURED POETS DEPARTMENT: THE ANTHOLOGY'
3. Won 10 awards at the Billboard Music Awards (including Top Artist and Top Billboard 200 Album for 'The Tortured Poets Department'), becoming the winningest artist in the show's history

Provide the artist's name and include reference URLs from reliable sources to verify each of these three achievements.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArtistAchievementsExtraction(BaseModel):
    """
    Core structured extraction from the agent's answer.
    - artist_name: The single artist claimed to have satisfied all achievements.
    - grammy_urls / spotify_urls / billboard_urls: All URLs explicitly cited in the answer
      to support the respective achievement.
    """
    artist_name: Optional[str] = None
    grammy_urls: List[str] = Field(default_factory=list)
    spotify_urls: List[str] = Field(default_factory=list)
    billboard_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    Extract the following fields from the answer:

    - artist_name: The single artist explicitly identified as achieving all listed 2024 accomplishments.
    - grammy_urls: All URLs provided as references supporting the Grammy achievement
                   (Album of the Year at the 2024 Grammys for 'Midnights', record 4th AOTY win).
    - spotify_urls: All URLs provided as references supporting the Spotify achievement
                    (2024 Global Top Artist with >26.6B streams; most-streamed album: 'THE TORTURED POETS DEPARTMENT: THE ANTHOLOGY').
    - billboard_urls: All URLs provided as references supporting the Billboard Music Awards achievement
                      (10 awards in 2024 BBMAs, winningest artist in show history, including Top Artist and Top Billboard 200 Album for 'The Tortured Poets Department').

    Important:
    - Only extract URLs explicitly present in the answer (including plain URLs or markdown links).
    - Do not infer or fabricate any URLs.
    - If a field is not present, return null (for artist_name) or an empty list (for URL arrays).
    - Include every distinct supporting URL the answer cites for each category.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_grammy_achievement(
    evaluator: Evaluator,
    parent_node,
    artist_name: str,
    grammy_urls: List[str]
) -> None:
    """
    Build and execute verification nodes for the Grammy achievement.
    """
    grammy_node = evaluator.add_parallel(
        id="grammy_achievement",
        desc="Grammy achievement constraints are satisfied for the identified artist and supported by evidence.",
        parent=parent_node,
        critical=True
    )

    # 1) Reference URL existence (critical)
    evaluator.add_custom_node(
        result=bool(grammy_urls),
        id="grammy_reference_url_provided",
        desc="Provides at least one reference URL from a reliable source that supports the Grammy claim (AOTY 2024, 'Midnights', 4th win/record).",
        parent=grammy_node,
        critical=True
    )

    # 2) Claim verification against provided URLs (critical)
    grammy_claim_node = evaluator.add_leaf(
        id="grammy_claim_correct",
        desc="States that the artist won Album of the Year at the 2024 (66th) Grammy Awards for 'Midnights' and that this was the artist's 4th Album of the Year win (record-setting).",
        parent=grammy_node,
        critical=True
    )

    grammy_claim = (
        f"The source supports that {artist_name} won Album of the Year at the 2024 Grammy Awards (the 66th Grammys) "
        f"for the album 'Midnights', and that this marked the artist's fourth Album of the Year win, setting the record "
        f"for the most wins in this category."
    )
    await evaluator.verify(
        claim=grammy_claim,
        node=grammy_claim_node,
        sources=grammy_urls,
        additional_instruction=(
            "Accept equivalent phrasings such as 'record fourth win', 'first artist to win four times', "
            "'4th AOTY', and wording that clearly implies the same fact. "
            "Minor naming variants for the ceremony (e.g., '2024 Grammys', '66th Grammy Awards') are acceptable."
        )
    )


async def verify_spotify_achievement(
    evaluator: Evaluator,
    parent_node,
    artist_name: str,
    spotify_urls: List[str]
) -> None:
    """
    Build and execute verification nodes for the Spotify achievement.
    """
    spotify_node = evaluator.add_parallel(
        id="spotify_achievement",
        desc="Spotify achievement constraints are satisfied for the identified artist and supported by evidence.",
        parent=parent_node,
        critical=True
    )

    # 1) Reference URL existence (critical)
    evaluator.add_custom_node(
        result=bool(spotify_urls),
        id="spotify_reference_url_provided",
        desc="Provides at least one reference URL from a reliable source that supports the Spotify claim (Global Top Artist, >26.6B streams, and the most-streamed album title).",
        parent=spotify_node,
        critical=True
    )

    # 2) Claim verification against provided URLs (critical)
    spotify_claim_node = evaluator.add_leaf(
        id="spotify_claim_correct",
        desc="States that the artist was Spotify's 2024 Global Top Artist with >26.6B streams and had the most-streamed album globally on Spotify in 2024 titled 'THE TORTURED POETS DEPARTMENT: THE ANTHOLOGY'.",
        parent=spotify_node,
        critical=True
    )

    spotify_claim = (
        f"The source supports that {artist_name} was Spotify's Global Top Artist in 2024 with more than 26.6 billion "
        f"global streams, and that the most-streamed album globally on Spotify in 2024 was "
        f"'THE TORTURED POETS DEPARTMENT: THE ANTHOLOGY'."
    )
    await evaluator.verify(
        claim=spotify_claim,
        node=spotify_claim_node,
        sources=spotify_urls,
        additional_instruction=(
            "Accept numeric expressions like '26.6B' meaning 26.6 billion and allow '> 26.6B' or 'more than 26.6B'. "
            "For the album title, allow minor formatting/case variations and acceptable variants like "
            "'The Tortured Poets Department: The Anthology', or where the subtitle is clearly indicated "
            "(e.g., '(The Anthology)'). Ensure the page clearly indicates this album was the most-streamed album globally in 2024."
        )
    )


async def verify_billboard_achievement(
    evaluator: Evaluator,
    parent_node,
    artist_name: str,
    billboard_urls: List[str]
) -> None:
    """
    Build and execute verification nodes for the Billboard Music Awards achievement.
    """
    billboard_node = evaluator.add_parallel(
        id="billboard_achievement",
        desc="Billboard Music Awards achievement constraints are satisfied for the identified artist and supported by evidence.",
        parent=parent_node,
        critical=True
    )

    # 1) Reference URL existence (critical)
    evaluator.add_custom_node(
        result=bool(billboard_urls),
        id="billboard_reference_url_provided",
        desc="Provides at least one reference URL from a reliable source that supports the Billboard claim (10 wins, winningest artist, Top Artist, and Top Billboard 200 Album for 'The Tortured Poets Department').",
        parent=billboard_node,
        critical=True
    )

    # 2) Claim verification against provided URLs (critical)
    billboard_claim_node = evaluator.add_leaf(
        id="billboard_claim_correct",
        desc="States that the artist won 10 awards at the 2024 Billboard Music Awards, became the winningest artist in BBMA history, and includes wins for Top Artist and Top Billboard 200 Album for 'The Tortured Poets Department'.",
        parent=billboard_node,
        critical=True
    )

    billboard_claim = (
        f"The source supports that {artist_name} won 10 awards at the 2024 Billboard Music Awards, including Top Artist "
        f"and Top Billboard 200 Album for 'The Tortured Poets Department', and that {artist_name} thereby became "
        f"the winningest (most-awarded) artist in the show's history."
    )
    await evaluator.verify(
        claim=billboard_claim,
        node=billboard_claim_node,
        sources=billboard_urls,
        additional_instruction=(
            "Accept equivalent phrasings for 'winningest artist' such as 'most-awarded' or 'most decorated'. "
            "Ensure the page clearly indicates 10 total wins in 2024, and specifically includes Top Artist and "
            "Top Billboard 200 Album for 'The Tortured Poets Department' among the awards."
        )
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
    Evaluate an answer for the 2024 triple-platform artist achievement task.
    """
    # 1) Initialize evaluator with a SEQUENTIAL root (acts as the Task_Completion node)
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
        default_model=model,
    )

    # 2) Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=ArtistAchievementsExtraction,
        extraction_name="core_extraction",
    )

    # 3) Optional ground truth info (for reporting/debugging; not used to judge)
    evaluator.add_ground_truth({
        "expected_achievements_summary": {
            "grammy": "AOTY 2024 (66th Grammys) for 'Midnights'; record 4th AOTY win",
            "spotify": "2024 Global Top Artist; >26.6B streams; most-streamed album: 'THE TORTURED POETS DEPARTMENT: THE ANTHOLOGY'",
            "billboard": "10 awards at 2024 BBMAs; winningest artist in show history; includes Top Artist and Top Billboard 200 Album for 'The Tortured Poets Department'"
        }
    })

    # 4) Artist name must be provided (Critical, first in sequential chain)
    artist_provided = bool(extraction.artist_name and extraction.artist_name.strip())
    evaluator.add_custom_node(
        result=artist_provided,
        id="artist_name_provided",
        desc="The response explicitly provides the artist's name.",
        parent=root,
        critical=True
    )

    # 5) Achievement verification block (Critical, evaluated only if artist name provided)
    achievement_block = evaluator.add_parallel(
        id="achievement_verification",
        desc="The response verifies (with supporting URLs) that the identified artist satisfies all Grammy, Spotify, and Billboard achievement constraints.",
        parent=root,
        critical=True
    )

    if artist_provided:
        artist_name = extraction.artist_name.strip()
    else:
        artist_name = ""  # Will be skipped due to sequential gating if missing

    # 6) Add and verify each achievement under the critical parallel block
    await verify_grammy_achievement(
        evaluator=evaluator,
        parent_node=achievement_block,
        artist_name=artist_name,
        grammy_urls=extraction.grammy_urls or []
    )

    await verify_spotify_achievement(
        evaluator=evaluator,
        parent_node=achievement_block,
        artist_name=artist_name,
        spotify_urls=extraction.spotify_urls or []
    )

    await verify_billboard_achievement(
        evaluator=evaluator,
        parent_node=achievement_block,
        artist_name=artist_name,
        billboard_urls=extraction.billboard_urls or []
    )

    # 7) Return structured evaluation summary
    return evaluator.get_summary()