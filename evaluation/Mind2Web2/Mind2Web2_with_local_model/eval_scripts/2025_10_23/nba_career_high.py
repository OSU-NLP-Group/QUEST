import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nba_career_high"
TASK_DESCRIPTION = """
Can you identify five NBA players in history whose regular season career highs in points, rebounds, assists, steals, and blocks are at least 60, 15, 15, 5, and 5, respectively?
Please include each player's specific career-high stats in these categories.
"""

MIN_REQUIREMENTS = {
    "points": 60,
    "rebounds": 15,
    "assists": 15,
    "steals": 5,
    "blocks": 5
}

REQUIRED_PLAYER_COUNT = 5

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PlayerStats(BaseModel):
    name: Optional[str] = None
    points: Optional[int] = None
    rebounds: Optional[int] = None
    assists: Optional[int] = None
    steals: Optional[int] = None
    blocks: Optional[int] = None
    
    @property
    def is_empty(self) -> bool:
        """Check if this is an empty player object."""
        return self.name is None or self.name.strip() == ""


class ExtractedPlayers(BaseModel):
    players: List[PlayerStats] = Field(default_factory=list)


class PlayerURLs(BaseModel):
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_players() -> str:
    return """
    Extract the list of NBA players mentioned in the answer along with their career-high stats.
    
    For each player, extract:
    1. The player's full name
    2. Career-high points
    3. Career-high rebounds
    4. Career-high assists
    5. Career-high steals
    6. Career-high blocks
    
    Extract these as integers (not as text like "60 points"). If any stat is missing, set it to null.
    Extract all players mentioned, even if there are more than 5.
    
    Do not make up or infer any values that are not explicitly mentioned in the answer.
    """


def prompt_extract_player_urls(player_name: str) -> str:
    return f"""
    Extract all URLs that are mentioned as sources for {player_name}'s career-high stats in the answer.
    
    Include any URLs that might contain information about {player_name}'s points, rebounds, assists, steals, or blocks career highs.
    
    Extract only valid URLs that are explicitly mentioned in the answer text.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_player(
    evaluator: Evaluator,
    parent_node,
    player: PlayerStats,
    player_index: int
) -> None:
    """
    Create a sequential node for a player and verify all requirements.
    """
    # Create sequential player node
    player_node = evaluator.add_parallel(
        id=f"player_{player_index}",
        desc=f"Verify player {player_index+1}: {player.name if player.name else 'Unknown'}",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit
    )

    # Extract URLs only for non-empty players
    unique_urls = []
    if not player.is_empty:
        # Extract URLs specifically for this player
        player_urls = await evaluator.extract(
            prompt=prompt_extract_player_urls(player.name),
            template_class=PlayerURLs,
            extraction_name=f"player_{player_index}_urls"
        )
        
        # Remove duplicates while preserving order
        for url in player_urls.urls:
            if url not in unique_urls:
                unique_urls.append(url)


    # Create a parallel node for player info verification
    player_info_node = evaluator.add_parallel(
        id=f"player_{player_index}_info",
        desc=f"Player {player_index+1} information verification",
        parent=player_node,
        critical=True
    )

    # Add existence check for player info
    info_exists = evaluator.add_custom_node(
        result=not player.is_empty and bool(unique_urls),
        id=f"player_{player_index}_info_exists",
        desc=f"Check if player {player_index+1} name and URLs are provided",
        parent=player_info_node,
        critical=True
    )

    # Verify this is an NBA player
    is_nba_player_node = evaluator.add_leaf(
        id=f"is_nba_player_{player_index}",
        desc=f"Verify that {player.name if player.name else 'Unknown'} is an NBA player",
        parent=player_info_node,
        critical=True
    )
    
    is_nba_claim = f"{player.name} is an NBA player (has played in the National Basketball Association)."
    await evaluator.verify(
        claim=is_nba_claim,
        node=is_nba_player_node,
        sources=unique_urls,
        additional_instruction=f"Check if {player.name} is confirmed to be an NBA player (current or former) from the provided sources. The player must have played in the NBA (not just drafted or signed)."
    )
    
    # Sequential verification of each stat
    stats_to_verify = [
        ("points", player.points, MIN_REQUIREMENTS["points"]),
        ("rebounds", player.rebounds, MIN_REQUIREMENTS["rebounds"]),
        ("assists", player.assists, MIN_REQUIREMENTS["assists"]),
        ("steals", player.steals, MIN_REQUIREMENTS["steals"]),
        ("blocks", player.blocks, MIN_REQUIREMENTS["blocks"])
    ]
    
    for stat_name, stat_value, min_value in stats_to_verify:
        # Create parallel node for stat verification
        stat_parent = evaluator.add_parallel(
            id=f"player_{player_index}_{stat_name}",
            desc=f"Player {player_index+1} {stat_name} verification",
            parent=player_node,
            critical=True
        )
        
        # Add existence and requirement check
        stat_valid = evaluator.add_custom_node(
            result=(stat_value is not None and stat_value >= min_value),
            id=f"player_{player_index}_{stat_name}_valid",
            desc=f"Check if {stat_name} value exists and meets minimum requirement of {min_value}",
            parent=stat_parent,
            critical=True
        )
        
        # Verify the stat
        stat_node = evaluator.add_leaf(
            id=f"{stat_name}_{player_index}_verification",
            desc=f"Verify {player.name if player.name else 'Unknown'}'s career-high {stat_name} is {stat_value}",
            parent=stat_parent,
            critical=True
        )
        
        claim = f"{player.name}'s career-high {stat_name} in regular season NBA games is {stat_value}, which is at least {min_value}."
        await evaluator.verify(
            claim=claim,
            node=stat_node,
            sources=unique_urls,
            additional_instruction=f"Specifically verify that {player.name} has a career-high of {stat_value} {stat_name} during regular season NBA games (not playoffs, not college, not international). The value must be explicitly supported by at least one of the provided sources."
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
    Evaluate a single answer and return a structured result dictionary.
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
        default_model=model
    )

    # -------- 2. Extract players and their stats from the answer --------- #
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_players(),
        template_class=ExtractedPlayers,
        extraction_name="extracted_players"
    )
    
    players = extracted_info.players
    
    # Pad with empty players if needed
    while len(players) < REQUIRED_PLAYER_COUNT:
        players.append(PlayerStats())
    
    # -------- 3. Build verification tree --------------------------------- #
    # Process exactly REQUIRED_PLAYER_COUNT players
    for i in range(REQUIRED_PLAYER_COUNT):
        await verify_player(
            evaluator=evaluator,
            parent_node=root,
            player=players[i],
            player_index=i
        )
    
    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()