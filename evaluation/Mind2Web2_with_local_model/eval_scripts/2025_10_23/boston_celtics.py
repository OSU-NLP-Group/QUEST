import asyncio
import logging
from typing import List, Dict, Optional

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "boston_celtics"
TASK_DESCRIPTION = """
Which players on the Boston Celtics' 2019-2020 playoff roster recorded a total field goal percentage of at least 50% in the playoffs, with a minimum of 10 field goal attempts? Provide their names along with a webpage displaying their per-game playoff statistics from the 2019-2020 season.
"""

# Ground truth players that match the criteria
GROUND_TRUTH_PLAYERS = [
    "Grant Williams",
    "Robert Williams", 
    "Enes Kanter Freedom",
    "Daniel Theis"
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PlayerNames(BaseModel):
    """List of player names extracted from the answer"""
    names: List[str] = Field(default_factory=list)

class PlayerInfo(BaseModel):
    """Detailed information about a player"""
    name: Optional[str] = None
    stats_urls: List[str] = Field(default_factory=list)

# Simple URL model for targeted extraction
class UrlModel(BaseModel):
    stats_urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_player_names() -> str:
    return """
    Extract only the names of all players mentioned in the answer who are identified as Boston Celtics players from the 2019-2020 playoff roster with a field goal percentage of at least 50% in the playoffs (minimum 10 field goal attempts).

    Format the extraction as a list of player names.

    If no players are mentioned, return an empty list.
    """

def prompt_extract_player_url(player_name: str) -> str:
    return f"""
    For the player '{player_name}', extract all URLs provided for their 2019-2020 playoff statistics. If the answer mentions a link containing statistics for multiple players, include that link for each relevant player.
    
    Return an empty list if no specific URL is provided for this player.
    """

# --------------------------------------------------------------------------- #
# Player verification functions                                               #
# --------------------------------------------------------------------------- #
async def verify_player_name(
    evaluator: Evaluator,
    player_info: PlayerInfo,
    player_index: int,
    parent_node
):
    """
    Verify that the player name matches one in the ground truth list.
    """
    claim = f"The player name '{player_info.name}' refers to one of these players: Grant Williams, Robert Williams, Enes Kanter Freedom, or Daniel Theis."
    
    # Create leaf node
    node = evaluator.add_leaf(
        id=f"player_{player_index+1}_name",
        desc=f"Player {player_index+1}'s name ({player_info.name}) matches one of the expected players in the ground truth (Grant Williams, Robert Williams, Enes Kanter Freedom, Daniel Theis).",
        parent=parent_node,
        critical=True
    )
    
    # Verify
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Compare the player name against the ground truth list. Consider variations in name format, such as shortened forms, cultural modifications, legal changes, alternate usages, full name versus initials and last name, inclusion of middle names."
    )

async def verify_player_stats_url(
    evaluator: Evaluator,
    player_info: PlayerInfo,
    player_index: int,
    parent_node
):
    """
    Verify that a valid URL is provided for the player's 2019-2020 playoff statistics.
    """
    # Create leaf node
    node = evaluator.add_leaf(
        id=f"player_{player_index+1}_stats_urls",
        desc=f"Player {player_index+1} ({player_info.name}) has a valid URL that contains their 2019-2020 Boston Celtics playoff statistics.",
        parent=parent_node,
        critical=True
    )
    
    # Verify URL contains the player's 2019-2020 playoff statistics
    claim = f"This webpage contains the 2019-2020 playoff statistics for {player_info.name} as a Boston Celtics player."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=player_info.stats_urls,
        additional_instruction="Check if the URL shows 2019-2020 playoff statistics for this player on the Boston Celtics. No need to verify specific field goal percentages or attempt numbers."
    )

async def verify_player(
    evaluator: Evaluator,
    player_info: PlayerInfo,
    player_index: int,
):
    """
    Verify both the player's name and their stats URL.
    """
    # Create player node
    player_node = evaluator.add_parallel(
        id=f"player_{player_index+1}",
        desc=f"Player {player_index+1}: {player_info.name}",
        critical=False
    )
    
    evaluator.add_custom_node(
        result=bool(player_info.name and player_info.stats_urls),
        id=f"player_{player_index+1}_existence",
        desc=f"Player {player_index+1} is provided with a name and url of statistics",
        parent=player_node,
        critical=True
    )

    # Verify player name
    await verify_player_name(evaluator, player_info, player_index, player_node)
    
    # Verify player stats URL
    await verify_player_stats_url(evaluator, player_info, player_index, player_node)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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
    
    # Add ground truth information
    evaluator.add_ground_truth(
        gt_info={"ground_truth_players": GROUND_TRUTH_PLAYERS},
        gt_type="ground_truth_players"
    )
    
    # Step 1: Extract just the player names first
    player_names_result = await evaluator.extract(
        prompt=prompt_extract_player_names(),
        template_class=PlayerNames,
        extraction_name="player_names"
    )
    
    # Step 2: For each player name, extract the URL information
    all_player_info = []
    for player_name in player_names_result.names:
        # Extract URL for this specific player
        url_data = await evaluator.extract(
            prompt=prompt_extract_player_url(player_name),
            template_class=UrlModel,
            extraction_name=f"url_for_{player_name.replace(' ', '_').lower()}"
        )
        
        # Create PlayerInfo with name and extracted URL
        all_player_info.append(PlayerInfo(
            name=player_name,
            stats_urls=url_data.stats_urls
        ))
    
    # Limit to the first 4 players as per evaluation instructions
    players_to_evaluate = all_player_info[:4]
    
    # Add extraction info
    evaluator.add_custom_info({
        "extracted_player_names": player_names_result.names,
        "extracted_players": [{"name": p.name, "stats_urls": p.stats_urls} for p in players_to_evaluate],
        "num_players_extracted": len(players_to_evaluate)
    }, "extraction_summary")
    
    # Verify each provided player
    for i, player_info in enumerate(players_to_evaluate):
        await verify_player(evaluator, player_info, i)
    
    # Create nodes for missing players if fewer than 4 were provided
    for i in range(len(players_to_evaluate), 4):
        await verify_player(evaluator, PlayerInfo(), i)
    
    # Return structured result using the new summary format
    return evaluator.get_summary()