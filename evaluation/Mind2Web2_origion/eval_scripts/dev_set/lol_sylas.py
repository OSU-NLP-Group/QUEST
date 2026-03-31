import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.llm_client.base_client import LLMClient
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lol_sylas"
TASK_DESCRIPTION = """
I recall watching one of the League of Legends World Championship finals, during an S-series tournament, where a professional esports player used Sylas near the mid-lane to steal Rakan's ultimate. This crucial play locked down the opposing team's double carries and secured the victory. Could you help me find out what that player's win rate with Sylas was throughout that entire S-series tournament?

(Note: League of Legends is a competitive multiplayer online game. Sylas and Rakan are playable characters, known as "champions," each with unique abilities.)
"""
PLAYER_GROUND_TRUTH = "the player is Faker from T1"
WIN_RATE_GROUND_TRUTH = "4 wins, 2 losses"
EVALUATION_INSTRUCTIONS = """
for the player, just compare the ground truth is faker. this is the first step to verify, and the answer should give sources where the sources actually indicate or explicitly mention faker used sylas to do something incredible in that game, where the game is the S-series final in that year. 
"""
JUDGE_MODEL = "o4-mini"


# --------------------------------------------------------------------------- #
# Data models for extracted info                                             #
# --------------------------------------------------------------------------- #
class PlayerInfo(BaseModel):
    """Information about the player who used Sylas in the World Championship finals."""
    player_name: Optional[str] = None
    team_name: Optional[str] = None
    win_rate: Optional[str] = None


class GameSourceURLs(BaseModel):
    """URLs specifically related to Faker using Sylas in an S-series final."""
    urls: List[str] = Field(default_factory=list)


class WinRateSourceURLs(BaseModel):
    """URLs specifically related to Faker's win rate with Sylas."""
    urls: List[str] = Field(default_factory=list)


# Extraction prompts
def prompt_extract_player_info() -> str:
    return """
    Extract the following information from the answer:

    1. The name of the player who used Sylas to steal Rakan's ultimate in the League of Legends World Championship finals.
    2. The team name of this player, if mentioned.
    3. The win rate of this player with Sylas throughout that S-series tournament, if mentioned. This could be expressed as a percentage, ratio, or in the format "X wins, Y losses".

    If any of this information is not explicitly mentioned in the answer, return null for that field.
    """


def prompt_extract_game_urls() -> str:
    return """
    Extract ONLY the URLs that provide evidence about Faker (or the identified player) using Sylas 
    in a League of Legends World Championship finals S-series match. Focus on sources that mention 
    the player making an impactful play with Sylas, particularly related to stealing Rakan's ultimate 
    or making a game-changing play.

    Only include URLs that directly substantiate this specific claim. Return these as a list of strings in the 'urls' field.

    Do not include URLs that only discuss the player's win rate without substantiating the specific play in the S-series finals.
    """


def prompt_extract_win_rate_urls() -> str:
    return """
    Extract ONLY the URLs that provide evidence about Faker's (or the identified player's) win rate 
    with Sylas throughout the S-series tournament. This includes:
    1. Sources directly stating the win rate
    2. Sources listing game results that would allow calculating the win rate
    3. Sources with match history showing when the player used Sylas and the game outcomes

    Only include URLs that directly substantiate the win rate claim. Return these as a list of strings in the 'urls' field.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                     #
# --------------------------------------------------------------------------- #
async def verify_player_identification(
        evaluator: Evaluator,
        parent_node,
        player_info: PlayerInfo,
) -> None:
    """
    Verify that the identified player is Faker.
    """
    # Create parent node for player verification
    player_parent_node = evaluator.add_parallel(
        id="player_verification",
        desc="Player Verification: Verifying Faker was identified",
        parent=parent_node,
        critical=True,
    )

    # Existence check
    player_exists_node = evaluator.add_custom_node(
        result=(player_info.player_name is not None and player_info.player_name.strip() != ""),
        id="player_info_exists",
        desc="Check if player name was provided in the answer",
        parent=player_parent_node,
        critical=True
    )

    # Verify player name
    player_verification_node = evaluator.add_leaf(
        id="player_name_verification",
        desc="Verification that the identified player is Faker",
        parent=player_parent_node,
        critical=True,
    )

    claim = f"The player who used Sylas to steal Rakan's ultimate in the World Championship finals is {player_info.player_name}."
    await evaluator.verify(
        claim=claim,
        node=player_verification_node,
        additional_instruction=f"Check if the player identified ({player_info.player_name}) is Faker. The ground truth is that the player is Faker from T1. The verification should pass only if the identified player is indeed Faker (case-insensitive comparison)."
    )


async def verify_game_evidence(
        evaluator: Evaluator,
        parent_node,
        player_info: PlayerInfo,
        game_sources: GameSourceURLs,
) -> None:
    """
    Verify that at least one source substantiates that Faker used Sylas effectively in a World Championship final.
    """
    # Create parent node for game evidence verification
    evidence_parent_node = evaluator.add_parallel(
        id="game_evidence_verification",
        desc="Game Evidence: Verifying sources support Faker's Sylas play",
        parent=parent_node,
        critical=True,
    )

    # Existence check
    evidence_exists_node = evaluator.add_custom_node(
        result=(player_info.player_name is not None and bool(game_sources.urls)),
        id="game_evidence_exists",
        desc="Check if player name and game evidence URLs were provided",
        parent=evidence_parent_node,
        critical=True
    )

    # Verify game evidence
    evidence_verification_node = evaluator.add_leaf(
        id="game_evidence_substantiation",
        desc="Verification that sources substantiate Faker using Sylas in S-series final",
        parent=evidence_parent_node,
        critical=True,
    )

    claim = "Faker used Sylas effectively in a League of Legends World Championship S-series final, specifically making an impactful play by stealing Rakan's ultimate or making another game-changing move that contributed to victory."
    await evaluator.verify(
        claim=claim,
        node=evidence_verification_node,
        sources=game_sources.urls,
        additional_instruction="""Verify that the source substantiates that Faker used Sylas effectively in a World Championship S-series final. The source should explicitly mention or indicate that Faker performed this play or a similarly important play using Sylas in an S-series tournament final.
         
         It's acceptable if the source roughly mention:
         1. faker steals Rakan's ultimate with Sylas' Hijack
         2. It's exactly S14 T1 VS BLG Game 4 as well as mentioned Faker used Sylas
         3. Faker made an impactful play with Sylas that contributed to victory in that game
         4. or something similar."""
    )


async def verify_win_rate(
        evaluator: Evaluator,
        parent_node,
        player_info: PlayerInfo,
        win_rate_sources: WinRateSourceURLs,
) -> None:
    """
    Verify both correctness and substantiation of the win rate.
    """
    # Existence check for win rate info
    win_rate_exists_node = evaluator.add_custom_node(
        result=(player_info.win_rate is not None and player_info.win_rate.strip() != ""),
        id="win_rate_info_exists",
        desc="Check if win rate was provided in the answer",
        parent=parent_node,
        critical=True
    )

    # Verify win rate correctness
    win_rate_correctness_node = evaluator.add_leaf(
        id="win_rate_correctness",
        desc=f"Verification that the provided win rate matches ground truth ({WIN_RATE_GROUND_TRUTH})",
        parent=parent_node,
        critical=True,
    )

    claim = f"Faker's win rate with Sylas throughout the S-series tournament was {player_info.win_rate}."
    await evaluator.verify(
        claim=claim,
        node=win_rate_correctness_node,
        additional_instruction=f"Check if the provided win rate ({player_info.win_rate}) matches the ground truth of {WIN_RATE_GROUND_TRUTH}. The verification should pass if the provided win rate is equivalent to 4 wins and 2 losses, regardless of the format (could be expressed as a ratio, percentage, fraction, or directly as 'X wins, Y losses'). Allow for minor variance of ±1% for percentage values."
    )

    # Combined existence check for win rate substantiation
    substantiation_exists_node = evaluator.add_custom_node(
        result=(
            player_info.win_rate is not None and 
            bool(win_rate_sources.urls) and 
            player_info.player_name is not None
        ),
        id="win_rate_substantiation_exists",
        desc="Check if all required information for win rate substantiation is present",
        parent=parent_node,
        critical=True
    )

    # Verify win rate substantiation
    win_rate_substantiation_node = evaluator.add_leaf(
        id="win_rate_substantiation",
        desc="Verification that the provided win rate is substantiated by sources",
        parent=parent_node,
        critical=True,
    )

    claim = f"Faker's win rate with Sylas throughout the S-series tournament was {player_info.win_rate}."
    await evaluator.verify(
        claim=claim,
        node=win_rate_substantiation_node,
        sources=win_rate_sources.urls,
        additional_instruction="""Verify that the source substantiates the claimed win rate for Faker with Sylas throughout the tournament. The source should provide specific statistics or data about Faker's performance with Sylas that supports the claimed win rate.

IMPORTANT: The data might be presented in a table format showing:
- Games played with different champions including Sylas
- Win/Loss records (W/L columns)
- Win rate percentages

Accept the claim if:
1. The source directly shows Faker played 6 games with Sylas with 4 wins and 2 losses
2. The source shows a win rate percentage for Faker with Sylas that matches the claim (allow ±1% variance, e.g., 66%, 66.7%, or 67% are all acceptable for 4W-2L)
3. The source provides match data where you can count that Faker used Sylas in 6 games with the correct win/loss record

Look carefully at any tables or statistics sections that show champion-specific performance data."""
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
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with sequential strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract structured info from the answer ---------------- #
    player_info = await evaluator.extract(
        prompt=prompt_extract_player_info(),
        template_class=PlayerInfo,
        extraction_name="player_info"
    )

    game_sources = await evaluator.extract(
        prompt=prompt_extract_game_urls(),
        template_class=GameSourceURLs,
        extraction_name="game_sources"
    )

    win_rate_sources = await evaluator.extract(
        prompt=prompt_extract_win_rate_urls(),
        template_class=WinRateSourceURLs,
        extraction_name="win_rate_sources"
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Step 1: Identify Faker and His Sylas Play
    step1_node = evaluator.add_parallel(
        id="identify_faker_step",
        desc="Verify the identification of Faker and his Sylas play in an S-series final",
        critical=False  # This is a multi-step question
    )

    await verify_player_identification(evaluator, step1_node, player_info)
    await verify_game_evidence(evaluator, step1_node, player_info, game_sources)

    # Step 2: Verify Win Rate Information
    step2_node = evaluator.add_parallel(
        id="win_rate_step",
        desc="Verify the win rate information provided for Faker with Sylas",
        critical=False  # Non-critical for overall evaluation
    )

    await verify_win_rate(evaluator, step2_node, player_info, win_rate_sources)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()