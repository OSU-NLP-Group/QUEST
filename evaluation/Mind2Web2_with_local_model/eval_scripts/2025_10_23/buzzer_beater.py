import asyncio
import logging
from typing import Optional, List, Dict
from datetime import datetime

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "buzzer_beater"
TASK_DESCRIPTION = """
Identify every game in the 2019 NBA Playoffs from the Conference Semifinals, Conference Finals, and NBA Finals in which a buzzer-beater occurred. By "buzzer-beater", I specifically mean a shot that was made within the final 3 seconds of the game, not including free throws. For each game, provide the game date, and the two teams involved.
"""

JUDGE_MODEL = "o4-mini"

# Ground truth games for reference
GROUND_TRUTH_GAMES = [
    {
        "teams": ["Houston Rockets", "Golden State Warriors"],
        "date": "April 30, 2019"
    },
    {
        "teams": ["Philadelphia 76ers", "Toronto Raptors"],
        "date": "May 12, 2019"
    }
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameInfo(BaseModel):
    """Information about a single game with buzzer-beater."""
    team1: Optional[str] = None
    team2: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class BuzzerBeaterGames(BaseModel):
    """All games with buzzer-beaters found in the answer."""
    games: List[GameInfo] = Field(default_factory=list)


class SourceLinks(BaseModel):
    """Source URLs mentioned in the answer."""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_games() -> str:
    return """
    Extract all games mentioned in the answer that had buzzer-beaters during the 2019 NBA Playoffs (Conference Semifinals, Conference Finals, and NBA Finals).

    For each game, extract:
    - team1: First team name (standardize to full team names like "Houston Rockets")
    - team2: Second team name (standardize to full team names like "Golden State Warriors") 
    - date: Game date in the format mentioned in the answer
    - description: Brief description of the buzzer-beater if provided
    - source_urls: Any URLs cited for this specific game information

    Only extract games that are explicitly mentioned as having buzzer-beaters within the final 3 seconds.
    Return empty list if no games are mentioned.
    """


def prompt_extract_source_urls() -> str:
    return """
    Extract all URLs mentioned anywhere in the answer text that could serve as sources for the buzzer-beater information.
    Include any URLs that appear to be citations or references.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
async def find_matching_game(
        evaluator: Evaluator,
        gt_game: Dict,
        extracted_games: List[GameInfo]
) -> Optional[GameInfo]:
    """Find an extracted game that matches the ground truth game using LLM verification."""
    
    for game in extracted_games:
        if not game.team1 or not game.team2 or not game.date:
            continue
        
        # Check if teams match using LLM
        teams_claim = f"The game between '{game.team1}' and '{game.team2}' is the same game as the one between '{gt_game['teams'][0]}' and '{gt_game['teams'][1]}' (team order doesn't matter)"
        teams_match = await evaluator.verify(
            claim=teams_claim,
            node=None,  # Don't assign to any node
            additional_instruction="Check if these refer to the same NBA game, considering team name variations (e.g., '76ers' vs 'Philadelphia 76ers')"
        )
        
        # Check if date matches using LLM
        date_claim = f"The date '{game.date}' refers to the same day as '{gt_game['date']}'"
        date_matches = await evaluator.verify(
            claim=date_claim,
            node=None,  # Don't assign to any node
            additional_instruction="Check if these dates refer to the same day, allowing for different date formats"
        )
        
        if teams_match and date_matches:
            return game
    
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_ground_truth_game(
        evaluator: Evaluator,
        parent_node,
        gt_index: int,
        gt_game: Dict,
        extracted_games: List[GameInfo],
        source_urls: List[str],
) -> None:
    """Verify a specific ground truth game."""
    
    # Create node for this ground truth game
    game_node = evaluator.add_parallel(
        id=f"game_{gt_index}_verification",
        desc=f"Game {gt_index + 1}: {gt_game['teams'][0]} vs {gt_game['teams'][1]} on {gt_game['date']}",
        parent=parent_node,
        critical=False,  # Allow partial credit
    )
    
    # Find matching extracted game using LLM verification
    matching_game = await find_matching_game(evaluator, gt_game, extracted_games)
    
    # Check if game was found in the answer
    game_found_node = evaluator.add_custom_node(
        result=matching_game is not None,
        id=f"game_{gt_index}_found",
        desc=f"Check if ground truth game {gt_index + 1} was identified in the answer",
        parent=game_node,
        critical=True
    )
    
    # If no matching game, use empty game for verification
    game_to_verify = matching_game if matching_game else GameInfo()
    
    # Verify buzzer-beater claim with sources
    await verify_buzzer_beater_claim(evaluator, game_node, game_to_verify, gt_index, source_urls)


async def verify_buzzer_beater_claim(
        evaluator: Evaluator,
        parent_node,
        game: GameInfo,
        gt_index: int,
        source_urls: List[str],
) -> None:
    """Verify the buzzer-beater claim is substantiated by source URLs."""
    
    # Check if source URLs are provided
    all_urls = list(set(game.source_urls + source_urls))
    
    source_exists_node = evaluator.add_custom_node(
        result=bool(all_urls),
        id=f"game_{gt_index}_source_exists",
        desc=f"Check if source URLs are provided for the buzzer-beater claim",
        parent=parent_node,
        critical=True
    )
    
    # Verify source substantiation
    provenance_node = evaluator.add_leaf(
        id=f"game_{gt_index}_buzzer_provenance",
        desc=f"Buzzer-beater claim is supported by provided source URLs",
        parent=parent_node,
        critical=True,
    )
    
    buzzer_claim = f"A buzzer-beater occurred in the game between {game.team1} and {game.team2} on {game.date}. By 'buzzer-beater', we specifically mean a shot that was made within the final 3 seconds of the game, not including free throws."
    
    await evaluator.verify(
        claim=buzzer_claim,
        node=provenance_node,
        sources=all_urls,
        additional_instruction="""Verify that the source URL(s) provide evidence of a buzzer-beater in this specific game. A buzzer-beater is defined as a shot (NOT a free throw) that was made within the final 3 seconds of the game.

For play-by-play tables (like basketball-reference.com):
- Look at the FINAL entries of the 4th quarter or any overtime period
- Check the "Time" column for entries showing 0:03 or less (0:02, 0:01, 0:00)
- Verify the action is a "makes 2-pt shot" or "makes 3-pt shot" (NOT "makes free throw")
- The successful shot should be one of the last actions before the game ends

For other sources:
- Look for specific mentions of time remaining (e.g., '2.5 seconds left', 'with 1 second remaining', 'at the buzzer')
- Confirm it was a field goal or three-pointer, not a free throw
- Verify this shot occurred at the very end of the game (not just end of a quarter)"""
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer for NBA buzzer-beater identification task.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with parallel strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract structured info from the answer ---------------- #

    # Extract games with buzzer-beaters
    games_info = await evaluator.extract(
        prompt=prompt_extract_games(),
        template_class=BuzzerBeaterGames,
        extraction_name="games_info"
    )

    # Extract source URLs
    source_links = await evaluator.extract(
        prompt=prompt_extract_source_urls(),
        template_class=SourceLinks,
        extraction_name="source_links"
    )

    # Add ground truth info
    evaluator.add_ground_truth({"ground_truth_games": GROUND_TRUTH_GAMES})

    # -------- 3. Build verification tree -------------------------------- #
    
    # Create exactly one node per ground truth game
    for i, gt_game in enumerate(GROUND_TRUTH_GAMES):
        await verify_ground_truth_game(
            evaluator,
            root,
            i,
            gt_game,
            games_info.games,
            source_links.urls
        )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()