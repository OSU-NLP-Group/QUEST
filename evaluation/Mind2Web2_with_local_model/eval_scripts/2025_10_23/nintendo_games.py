import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nintendo_games"
TASK_DESCRIPTION = """
I recently purchased a Nintendo Switch and want to explore its best-selling games, and I don't like Pokémon games. Please find the 10 best-selling Nintendo Switch games based on the number of copies sold, according to Wikipedia, excluding Pokémon franchise titles. Please provide the link of your reference on Wikipedia. 

For each game, provide the following details:
1. The number of copies sold according to Wikipedia
2. Purchase link of the Nintendo Switch version on the official Nintendo website (in the US) and its price.
3. Metascore and User Score from Metacritic
"""

EVALUATION_NOTES = """
Note that Metacritic Metascore and User Score should refer to the Nintendo Switch version of the game
"""

# Ground truth Wikipedia URL for best-selling Nintendo Switch games
WIKIPEDIA_GROUND_TRUTH_URL = "https://en.wikipedia.org/wiki/List_of_best-selling_Nintendo_Switch_video_games"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WikipediaInfo(BaseModel):
    """Wikipedia reference and top games extracted from it."""
    url: Optional[str] = None
    top_games: List[str] = Field(default_factory=list)  # Names of top Nintendo Switch games (excluding Pokemon)


class WikipediaTopGames(BaseModel):
    """Top 10 best-selling Nintendo Switch games from Wikipedia."""
    games_list: List[str] = Field(default_factory=list)  # Names in order


class GameDetails(BaseModel):
    """Details for a single Nintendo Switch game."""
    name: Optional[str] = None
    copies_sold: Optional[str] = None
    nintendo_url: Optional[str] = None
    price: Optional[str] = None
    metascore: Optional[str] = None
    user_score: Optional[str] = None
    metacritic_urls: List[str] = Field(default_factory=list)  # Changed to list


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_wikipedia_info() -> str:
    return """
    Extract the following information from the answer:
    
    1. The Wikipedia URL that was referenced as the source for the list of best-selling Nintendo Switch games.
    
    2. The names of all Nintendo Switch games identified in the answer as the best-selling games (excluding Pokémon games).
    
    For the URL, return the complete Wikipedia link. If there's no explicit Wikipedia URL mentioned, return null.
    
    For the game names, extract them exactly as they appear in the answer, and include all games mentioned as best-selling Nintendo Switch games.
    """


def prompt_extract_wikipedia_top_10() -> str:
    return """
    Extract the top 10 best-selling Nintendo Switch games from this Wikipedia page, excluding any Pokémon games.
    
    Important instructions:
    1. Look for the main table listing best-selling Nintendo Switch games
    2. EXCLUDE any games with "Pokémon" in the title or that are clearly Pokémon franchise games
    3. Return exactly 10 games (or fewer if less than 10 non-Pokémon games are listed)
    4. Return only the game titles, nothing else
    """


def prompt_extract_game_details(game_name: str) -> str:
    return f"""
    Extract detailed information for the Nintendo Switch game "{game_name}" as presented in the answer. Extract:
    
    1. name: "{game_name}" (exactly as provided)
    2. copies_sold: The number of copies sold as stated in the answer (include units like "million")
    3. nintendo_url: The Nintendo store URL for purchasing the Nintendo Switch version on the official Nintendo website (in the US)
    4. price: The price of the game as stated in the answer (include currency symbol)
    5. metascore: The Metacritic Metascore as stated in the answer
    6. user_score: The Metacritic User Score as stated in the answer
    7. metacritic_urls: ALL Metacritic URLs provided for this game's Nintendo Switch version (return as a list)
    
    If any information is missing for the game, set the corresponding field to null.
    For metacritic_urls, if no URLs are provided, return an empty list.
    
    Note: Be precise and extract information exactly as presented in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_wikipedia_reference(
    evaluator: Evaluator,
    parent_node,
    wiki_info: WikipediaInfo,
) -> None:
    """
    Verify that a valid Wikipedia reference URL is provided that contains the list 
    of best-selling Nintendo Switch games and their sales numbers.
    """
    # Add existence check directly to parent (no wrapper node needed)
    wiki_exists = evaluator.add_custom_node(
        result=bool(wiki_info.url),
        id="wiki_url_exists",
        desc="Check if Wikipedia URL was provided",
        parent=parent_node,
        critical=True
    )

    # URL verification
    url_valid_node = evaluator.add_leaf(
        id="wiki_url_valid",
        desc="Verify Wikipedia page contains best-selling Nintendo Switch games list",
        parent=parent_node,
        critical=True,
    )

    await evaluator.verify(
        claim="This Wikipedia page contains a list of best-selling Nintendo Switch games with information on the number of copies sold for each game",
        node=url_valid_node,
        sources=wiki_info.url,
    )


async def verify_game(
    evaluator: Evaluator,
    parent_node,
    game: GameDetails,
    index: int,
    wiki_url: str,
    ground_truth_top_10: WikipediaTopGames,
) -> None:
    """
    Verify all required details for a single game.
    """
    game_node = evaluator.add_sequential(
        id=f"game_{index}",
        desc=f"Game #{index+1}: {game.name if game.name else 'Unknown game'} - All required details",
        parent=parent_node,
        critical=False,
    )

    # 1. Verify game is in Wikipedia's top games excluding Pokémon
    # Combined existence check and verification
    name_exists = evaluator.add_custom_node(
        result=bool(game.name),
        id=f"game_{index}_name_exists",
        desc="Check if game name was provided",
        parent=game_node,
        critical=True
    )

    top_game_node = evaluator.add_leaf(
        id=f"game_{index}_is_top_game",
        desc=f"Verify '{game.name if game.name else 'Unknown game'}' is in Wikipedia's top 10 list",
        parent=game_node,
        critical=True,
    )

    # Use the ground truth top 10 list for verification
    await evaluator.verify(
        claim=f"The game title '{game.name}' matches one of these titles: {', '.join(ground_truth_top_10.games_list)}",
        node=top_game_node,
        additional_instruction="Check if the game name matches any title in the provided list, allowing for minor variations in capitalization, punctuation, or abbreviations (e.g., 'Bros.' vs 'Brothers')."
    )

    # 2. Verify copies sold matches Wikipedia (use ground truth URL)
    copies_exists = evaluator.add_custom_node(
        result=bool(game.copies_sold),
        id=f"game_{index}_copies_exists",
        desc="Check if copies sold information was provided",
        parent=game_node,
        critical=True
    )

    copies_node = evaluator.add_leaf(
        id=f"game_{index}_copies_match",
        desc=f"Verify copies sold matches Wikipedia",
        parent=game_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"According to this Wikipedia page, '{game.name}' has sold approximately {game.copies_sold}",
        node=copies_node,
        sources=WIKIPEDIA_GROUND_TRUTH_URL,  # Use ground truth URL for verification
        additional_instruction="The numbers don't need to match exactly, but should be reasonably close (within ~10%).",
    )

    # 3. Verify Nintendo store link and price
    nintendo_exists = evaluator.add_custom_node(
        result=bool(game.nintendo_url and game.price),
        id=f"game_{index}_nintendo_exists",
        desc="Check if Nintendo URL and price were provided",
        parent=game_node,
        critical=True
    )

    nintendo_node = evaluator.add_leaf(
        id=f"game_{index}_nintendo_valid",
        desc=f"Verify Nintendo store link and price",
        parent=game_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"This is a valid Nintendo Store US website link for purchasing the '{game.name}', and the price of the Nintendo Switch version shown is {game.price}",
        node=nintendo_node,
        sources=game.nintendo_url,
        additional_instruction="The price should be acceptable if matching any listed price on the website, whether it's for the digital or physical version. But make sure it's for the Nintendo Switch version, not the Switch 2 version."
    )

    # 4. Verify Metacritic scores (split into two nodes)
    metacritic_exists = evaluator.add_custom_node(
        result=bool(game.metascore and game.user_score and game.metacritic_urls),
        id=f"game_{index}_metacritic_exists",
        desc="Check if Metacritic scores and URLs were provided",
        parent=game_node,
        critical=True
    )

    # Create a parent node for Metacritic verifications
    metacritic_parent = evaluator.add_parallel(
        id=f"game_{index}_metacritic_scores",
        desc=f"Verify Metacritic scores for '{game.name}'",
        parent=game_node,
        critical=True,
    )

    # 5a. Verify Metascore
    metascore_node = evaluator.add_leaf(
        id=f"game_{index}_metascore_valid",
        desc=f"Verify Metascore matches",
        parent=metacritic_parent,
        critical=True,
    )

    # Use all provided URLs for metascore verification
    await evaluator.verify(
        claim=f"For the Nintendo Switch version of '{game.name}', the Metacritic Metascore is {game.metascore}",
        node=metascore_node,
        sources=game.metacritic_urls,
        additional_instruction="Verify that the Metascore refers specifically to the Nintendo Switch version of the game, not other platforms.",
    )

    # 5b. Verify User Score
    user_score_node = evaluator.add_leaf(
        id=f"game_{index}_user_score_valid",
        desc=f"Verify User Score matches",
        parent=metacritic_parent,
        critical=True,
    )

    await evaluator.verify(
        claim=f"For the Nintendo Switch version of '{game.name}', the Metacritic User Score is {game.user_score}",
        node=user_score_node,
        sources=game.metacritic_urls,
        additional_instruction="Verify that the User Score refers specifically to the Nintendo Switch version of the game, not other platforms.",
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

    # First extract Wikipedia info and top games from the answer
    wiki_info = await evaluator.extract(
        prompt=prompt_extract_wikipedia_info(),
        template_class=WikipediaInfo,
        extraction_name="wikipedia_info"
    )

    # Extract the ground truth top 10 from the known Wikipedia URL
    ground_truth_top_10 = await evaluator.extract(
        prompt=prompt_extract_wikipedia_top_10(),
        template_class=WikipediaTopGames,
        extraction_name="ground_truth_top_10",
        source=WIKIPEDIA_GROUND_TRUTH_URL  # Always use the ground truth URL
    )

    # Add ground truth info to the evaluation summary
    evaluator.add_ground_truth({
        "wikipedia_url": WIKIPEDIA_GROUND_TRUTH_URL,
        "top_10_games": ground_truth_top_10.games_list
    }, "ground_truth_top_10")

    # Extract details for each identified game
    games_with_details = []
    for game_name in wiki_info.top_games:
        game_details = await evaluator.extract(
            prompt=prompt_extract_game_details(game_name),
            template_class=GameDetails,
            extraction_name=f"game_details_{game_name}",
            additional_instruction=EVALUATION_NOTES,
        )
        game_details.name = game_name  # Ensure the name is set
        games_with_details.append(game_details)

    # Pad to 10 games if needed (using empty GameDetails instances)
    expected_count = 10
    while len(games_with_details) < expected_count:
        games_with_details.append(GameDetails())

    # First verify Wikipedia reference - critical
    await verify_wikipedia_reference(
        evaluator=evaluator,
        parent_node=root,
        wiki_info=wiki_info,
    )

    # Then verify all games
    ten_games_node = evaluator.add_parallel(
        id="ten_games",
        desc="10 best-selling non-Pokémon Nintendo Switch games with all required details",
        parent=root,
        critical=False,
    )

    # Verify each game using unified logic for both real and empty games
    for i in range(expected_count):
        await verify_game(
            evaluator=evaluator,
            parent_node=ten_games_node,
            game=games_with_details[i],
            index=i,
            wiki_url=wiki_info.url if wiki_info.url else "",
            ground_truth_top_10=ground_truth_top_10,
        )

    # Return structured result
    return evaluator.get_summary()