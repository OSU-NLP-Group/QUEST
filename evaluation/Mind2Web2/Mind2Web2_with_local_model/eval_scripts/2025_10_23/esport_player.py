import asyncio
import logging
from typing import Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "esport_player"
TASK_DESCRIPTION = """
Find three professional esports players who have competed in both the Call of Duty League (CDL) and the Halo Championship Series (HCS). For each player, find their team history (including teams represented and years) and tournament results in both games.
"""


# --------------------------------------------------------------------------- #
# Extraction models for incremental extraction                               #
# --------------------------------------------------------------------------- #
class PlayerName(BaseModel):
    name: str = None


class PlayerNames(BaseModel):
    player_names: List[PlayerName] = Field(default_factory=list)


class TeamInfo(BaseModel):
    team_name: Optional[str] = None
    years: Optional[str] = None


class CDLInfo(BaseModel):
    teams: List[TeamInfo] = Field(default_factory=list)
    tournament_results: Optional[List[str]] = Field(default_factory=list)


class HCSInfo(BaseModel):
    teams: List[TeamInfo] = Field(default_factory=list)
    tournament_results: Optional[List[str]] = Field(default_factory=list)


class PlayerUrls(BaseModel):
    urls: List[str] = Field(default_factory=list)


class Player(BaseModel):
    name: Optional[str] = None
    cdl_info: Optional[CDLInfo] = None
    hcs_info: Optional[HCSInfo] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts for incremental extraction                              #
# --------------------------------------------------------------------------- #
def prompt_extract_player_names() -> str:
    """Extract just the names of players mentioned in the answer."""
    return """
    Extract only the names of professional esports players mentioned in the answer who have competed in both the Call of Duty League (CDL) and the Halo Championship Series (HCS).

    Return a list of player names, with each name in its own object.
    If the answer mentions more than three players, extract all of them.
    Only extract names that are explicitly mentioned in the answer.

    Do not invent or hallucinate any player names. Only extract what is stated in the answer.
    """


def prompt_extract_cdl_info(player_name: str) -> str:
    """Extract CDL information for a specific player."""
    return f"""
    Extract ONLY the Call of Duty League (CDL) information for player: {player_name}

    This should include:
    1. Teams they were part of (with years if provided)
    2. Tournament results in CDL (the result of one year or more if provided)

    Only extract information that is explicitly mentioned in the answer.
    If any information is missing, set the corresponding field to null.
    Do not invent or hallucinate any information not actually present in the answer.
    """


def prompt_extract_hcs_info(player_name: str) -> str:
    """Extract HCS information for a specific player."""
    return f"""
    Extract ONLY the Halo Championship Series (HCS) information for player: {player_name}

    This should include:
    1. Teams they were part of (with years if provided)
    2. Tournament results in HCS (the result of one year or more if provided)

    Only extract information that is explicitly mentioned in the answer.
    If any information is missing, set the corresponding field to null.
    Do not invent or hallucinate any information not actually present in the answer.
    """


def prompt_extract_player_urls(player_name: str) -> str:
    """Extract source URLs for a specific player."""
    return f"""
    Extract ONLY the source URLs mentioned in the answer that provide information about player: {player_name}

    Return a list of URLs that are explicitly linked to this player in the answer.
    Only extract URLs that are specifically related to {player_name}.

    Do not invent or hallucinate any URLs not actually present in the answer.
    If no URLs are provided for this player, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                     #
# --------------------------------------------------------------------------- #
async def verify_league_participation(
        evaluator: Evaluator,
        parent_node,
        player: Player,
        league: Literal["cdl", "hcs"]
):
    """
    Verify that the player has competed in the specified league.
    """
    league_map = {
        "cdl": {"full_name": "Call of Duty League (CDL)", "info": player.cdl_info},
        "hcs": {"full_name": "Halo Championship Series (HCS)", "info": player.hcs_info},
    }

    player_name = player.name if player.name else f"Missing player"

    node = evaluator.add_leaf(
        id=f"player_{player_name}_{league}_verification",
        desc=f"Verify that player {player_name} has competed in the {league_map[league]['full_name']}",
        parent=parent_node,
        critical=True,
    )

    claim = f"Player {player.name} has competed in the {league_map[league]['full_name']}. Minor variations in name spelling/capitalization are acceptable."
    
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=player.source_urls,
    )

    return node


async def verify_team_history(
        evaluator: Evaluator,
        parent_node,
        player: Player,
        league: Literal["cdl", "hcs"]
):
    """
    Verify the team history of the player in the specified league.
    Verifies each team entry separately, up to 5 entries maximum.
    """
    league_map = {
        "cdl": {"full_name": "Call of Duty League", "info": player.cdl_info},
        "hcs": {"full_name": "Halo Championship Series", "info": player.hcs_info},
    }

    player_name = player.name if player.name else f"Missing player"

    # Create parent node for team history
    team_history_node = evaluator.add_parallel(
        id=f"player_{player_name}_{league}_team_history",
        desc=f"Verify the team history of player {player_name} in {league.upper()}",
        parent=parent_node,
        critical=False,
    )

    # Get teams, limit to first 5
    teams = league_map[league]["info"].teams[:5]

    # Create individual verification nodes for each team entry
    for i, team in enumerate(teams):
        team_entry_node = evaluator.add_leaf(
            id=f"player_{player_name}_{league}_team_{i}",
            desc=f"Verify team entry: {team.team_name} ({team.years if team.years else 'years not specified'}) for player {player_name} in {league.upper()}",
            parent=team_history_node,
            critical=False,
        )

        if not team.team_name:
            team_entry_node.score = 0.0
            team_entry_node.status = "failed"
            continue

        # Build claim for this specific team entry
        team_claim = f"Player {player.name} played for team {team.team_name} in the {league_map[league]['full_name']}"
        if team.years:
            team_claim += f" during {team.years}"
        team_claim += ". Minor variations in team names, player names, and year formats are acceptable."

        await evaluator.verify(
            claim=team_claim,
            node=team_entry_node,
            sources=player.source_urls,
        )

    return team_history_node


async def verify_tournament_results(
        evaluator: Evaluator,
        parent_node,
        player: Player,
        league: Literal["cdl", "hcs"]
):
    """
    Verify the tournament results of the player in the specified league.
    If no results are provided, verify that the player indeed has no notable tournament results.
    """
    league_map = {
        "cdl": {"full_name": "Call of Duty", "info": player.cdl_info},
        "hcs": {"full_name": "Halo Championship Series", "info": player.hcs_info},
    }

    player_name = player.name if player.name else f"Missing player"

    # Create parent node for tournament results
    tournament_results_node = evaluator.add_parallel(
        id=f"player_{player_name}_{league}_tournament_results",
        desc=f"Verify the tournament results information for player {player_name} in {league.upper()}",
        parent=parent_node,
        critical=False,
    )

    # Check if tournament results are provided
    results = league_map[league]["info"].tournament_results

    if not results or len(results) == 0:
        # No results provided - verify that the player indeed has no notable tournament results
        no_results_node = evaluator.add_leaf(
            id=f"player_{player_name}_{league}_no_results",
            desc=f"Verify that player {player_name} has no notable tournament results in {league.upper()} (as stated in the answer)",
            parent=tournament_results_node,
            critical=False,
        )

        # Verify that the sources support the claim of no notable results
        no_results_claim = f"Player {player.name} has no notable tournament results or achievements in the {league_map[league]['full_name']}, or has not achieved significant tournament success in this league. Minor variations in player names are acceptable."

        await evaluator.verify(
            claim=no_results_claim,
            node=no_results_node,
            sources=player.source_urls,
        )

    else:
        # Results are provided - verify each one, limit to first 5
        results_to_verify = results[:5]

        for i, result in enumerate(results_to_verify):
            result_node = evaluator.add_leaf(
                id=f"player_{player_name}_{league}_result_{i}",
                desc=f"Verify tournament result: {result} for player {player_name} in {league.upper()}",
                parent=tournament_results_node,
                critical=False,
            )

            # Build claim for this specific result
            result_claim = f"Player {player.name} achieved the following tournament result in {league_map[league]['full_name']}: {result}."

            await evaluator.verify(
                claim=result_claim,
                node=result_node,
                sources=player.source_urls,
                additional_instruction="Minor variations in player names, tournament names, and result formats are acceptable. Verify only the tournament names; disregard opponent details and match results."
            )

    return tournament_results_node


async def verify_player(
        evaluator: Evaluator,
        player: Player,
        player_index: int,
):
    """
    Verify all information for a single player.
    This function handles both real players and placeholder players.
    """
    player_display_name = player.name if player.name else f"Missing player #{player_index + 1}"

    player_node = evaluator.add_parallel(
        id=f"player_{player_index}",
        desc=f"Player {player_index + 1}: {player_display_name} - Verify this player competed in both CDL and HCS with accurate team history and tournament results",
        critical=False,  # non-critical for partial scores
    )

    player_exists = bool(
        player.name and player.cdl_info and player.hcs_info and
        player.source_urls and len(player.source_urls) > 0 and
        player.cdl_info.teams and len(player.cdl_info.teams) > 0 and
        player.hcs_info.teams and len(player.hcs_info.teams) > 0
        # may indeed have no tournament results
    )

    evaluator.add_custom_node(
        result=player_exists,
        id=f"player_{player_index}_exists",
        desc=f"Verify player {player_display_name} has necessary information",
        parent=player_node,
        critical=True
    )

    # Create a node for dual league participation verification (critical)
    dual_league_node = evaluator.add_parallel(
        id=f"player_{player_index}_dual_league",
        desc=f"Verify player {player_display_name} has competed in both CDL and HCS",
        parent=player_node,
        critical=True,
    )

    # Verify participation in both leagues (both critical for dual league requirement)
    await verify_league_participation(evaluator, dual_league_node, player, "cdl")
    await verify_league_participation(evaluator, dual_league_node, player, "hcs")

    # Verify team history in both leagues (each critical)
    await verify_team_history(evaluator, player_node, player, "cdl")
    await verify_team_history(evaluator, player_node, player, "hcs")

    # Verify tournament results in both leagues (each critical)
    await verify_tournament_results(evaluator, player_node, player, "cdl")
    await verify_tournament_results(evaluator, player_node, player, "hcs")

    return player_node


# --------------------------------------------------------------------------- #
# Helper function to normalize player names                                   #
# --------------------------------------------------------------------------- #
def normalize_player_name(name: str) -> str:
    """
    Normalize player name by converting to lowercase and removing extra spaces.
    This helps prevent duplicate players with slightly different capitalization or spacing.
    """
    return " ".join(name.lower().split())


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: Any,
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
    # -------- 1. Initialize evaluator ----------------------------------- #
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
        default_model=model
    )

    # -------- 2. Extract player information in steps -------------------- #
    # Step 1: Extract just the player names
    player_names_result = await evaluator.extract(
        prompt=prompt_extract_player_names(),
        template_class=PlayerNames,
        extraction_name="player_names"
    )

    # Filter out duplicate player names (case-insensitive)
    unique_player_names = []
    seen_names = set()

    for player_name_obj in player_names_result.player_names:
        normalized_name = normalize_player_name(player_name_obj.name)
        if normalized_name not in seen_names:
            seen_names.add(normalized_name)
            unique_player_names.append(player_name_obj.name)

    # Create a list to hold complete player info
    complete_players = []

    # Step 2: For each unique player name, extract detailed information in separate steps
    for player_name in unique_player_names[:3]:  # Limit to first 3 unique players
        # Extract CDL information
        cdl_info = await evaluator.extract(
            prompt=prompt_extract_cdl_info(player_name),
            template_class=CDLInfo,
            extraction_name=f"{player_name}_cdl_info"
        )

        # Extract HCS information
        hcs_info = await evaluator.extract(
            prompt=prompt_extract_hcs_info(player_name),
            template_class=HCSInfo,
            extraction_name=f"{player_name}_hcs_info"
        )

        # Extract source URLs
        urls_info = await evaluator.extract(
            prompt=prompt_extract_player_urls(player_name),
            template_class=PlayerUrls,
            extraction_name=f"{player_name}_urls"
        )

        # Combine into a complete player record
        player = Player(
            name=player_name,
            cdl_info=cdl_info,
            hcs_info=hcs_info,
            source_urls=urls_info.urls
        )

        complete_players.append(player)

    # Log the extraction results
    logger.info(f"Extracted {len(complete_players)} unique players: {[p.name for p in complete_players]}")

    # -------- 3. Build verification tree -------------------------------- #

    # Prepare players to verify - ensuring we have exactly 3 slots
    players_to_verify = complete_players[:3]  # Limit to first 3 if more provided

    # Create empty Player objects for missing slots
    while len(players_to_verify) < 3:
        players_to_verify.append(Player())

    # Verify each player (including placeholders)
    for i, player in enumerate(players_to_verify):
        await verify_player(evaluator, player, i)

    # -------- 4. Add custom information ---------------------------------- #
    evaluator.add_custom_info(
        {
            "extracted_players": [player.dict() for player in complete_players],
            "unique_player_count": len(complete_players),
            "players_evaluated": 3,
            "missing_players": 3 - len(complete_players)
        },
        "evaluation_summary"
    )

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()