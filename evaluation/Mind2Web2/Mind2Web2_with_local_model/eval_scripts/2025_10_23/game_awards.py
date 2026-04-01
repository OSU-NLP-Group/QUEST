import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

from datetime import datetime

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "game_awards"
TASK_DESCRIPTION = """
I want to gather information about award-winning games from the past year. Please find the links to the full lists of award-winning games from the following award events: The Game Awards (TGA), Golden Joystick Awards, and Steam Awards. Then, include the name of the Game of the Year (or "Ultimate Game of the Year," if applicable) winner from each event, along with a link to each game's official page (e.g., the official website, Steam page, or official vendor page). Finally, for each game, please find a review video (in English) on YouTube.
"""

current_year = datetime.now().year
LAST_YEAR = current_year - 1

# --------------------------------------------------------------------------- #
# Data models for extracted info                                               #
# --------------------------------------------------------------------------- #
class AwardLink(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None

class AwardLinks(BaseModel):
    links: List[AwardLink] = Field(default_factory=list)

class GameWinner(BaseModel):
    game_name: Optional[str] = None
    official_link: Optional[str] = None
    review_video_link: Optional[str] = None

class EventWinners(BaseModel):
    tga_winner: Optional[GameWinner] = None
    golden_joystick_winner: Optional[GameWinner] = None
    steam_winner: Optional[GameWinner] = None

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_award_links() -> str:
    return """
    Extract all links to the full lists of award-winning games from the answer. You need to extract links for the following award events:
    1. The Game Awards (TGA)
    2. Golden Joystick Awards
    3. Steam Awards
    
    For each award event, extract:
    - name: The exact name of the award event as mentioned in the answer
    - url: The URL link to the full list of award-winning games for that event
    
    Return null for any missing fields.
    """

def prompt_extract_tga_winner() -> str:
    return """
    Extract information about The Game Awards (TGA) Game of the Year winner from the answer. 
    Extract:
    - game_name: The name of the Game of the Year winner from The Game Awards
    - official_link: The URL to the game's official page (official website, Steam page, or vendor page)
    - review_video_link: The URL to a YouTube review video for this game
    
    Return null for any missing fields.
    """

def prompt_extract_golden_joystick_winner() -> str:
    return """
    Extract information about the Golden Joystick Awards Ultimate Game of the Year winner from the answer. 
    Extract:
    - game_name: The name of the Ultimate Game of the Year winner from Golden Joystick Awards
    - official_link: The URL to the game's official page (official website, Steam page, or vendor page)
    - review_video_link: The URL to a YouTube review video for this game
    
    Return null for any missing fields.
    """

def prompt_extract_steam_winner() -> str:
    return """
    Extract information about the Steam Awards Game of the Year winner from the answer. 
    Extract:
    - game_name: The name of the Game of the Year winner from Steam Awards
    - official_link: The URL to the game's official page (official website, Steam page, or vendor page)
    - review_video_link: The URL to a YouTube review video for this game
    
    Return null for any missing fields.
    """

# --------------------------------------------------------------------------- #
# Award Links Verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_award_links(
    evaluator: Evaluator,
    parent_node,
    award_links: AwardLinks,
) -> None:
    """
    Verify that links to award event lists are provided and valid.
    """
    # Expected award names
    expected_awards = ["The Game Awards", "Golden Joystick Awards", "Steam Awards"]
    
    # Ensure we have 3 award links (pad with empty if needed)
    award_links_map = {}
    for award_name in expected_awards:
        award_link = next((link for link in award_links.links if link.name and award_name.lower() in link.name.lower()), None)
        award_links_map[award_name] = award_link or AwardLink()
    
    # Create a node for each expected award
    for award_name in expected_awards:
        award_id = award_name.lower().replace(' ', '_')
        award_link = award_links_map[award_name]
        
        # Single parallel node for this award's link verification
        award_node = evaluator.add_parallel(
            id=f"{award_id}_link",
            desc=f"Verify link to {award_name} award list",
            parent=parent_node,
            critical=False,
        )
        
        # Check if link exists and is valid
        link_exists_node = evaluator.add_custom_node(
            result=bool(award_link.url),
            id=f"{award_id}_link_exists",
            desc=f"Link to {award_name} award list exists in the answer",
            parent=award_node,
            critical=True,
        )
        
        # Verify link validity
        link_valid_node = evaluator.add_leaf(
            id=f"{award_id}_link_valid",
            desc=f"Link to {award_name} award list is valid and points to the correct award event",
            parent=award_node,
            critical=True,
        )
        
        claim = f"This URL is a valid link to the {award_name} ({LAST_YEAR}) award list or results page."
        await evaluator.verify(
            claim=claim,
            node=link_valid_node,
            sources=award_link.url,
            additional_instruction=f"Check if the URL contains information about {award_name} and their award winners or nominees. The page should clearly be an official or reputable source for {award_name} results.",
        )

# --------------------------------------------------------------------------- #
# Game Winner Verification                                                   #
# --------------------------------------------------------------------------- #
async def verify_game_winner(
    evaluator: Evaluator,
    parent_node,
    winner: Optional[GameWinner],
    award_name: str,
    award_url: Optional[str],
) -> None:
    """
    Verify Game of the Year winner for a specific award event.
    """
    # Ensure we have a winner object (pad with empty if needed)
    if winner is None:
        winner = GameWinner()
    
    award_id = award_name.lower().replace(' ', '_')
    
    # Create sequential node for this award's winner (contains all verifications)
    award_winner_node = evaluator.add_sequential(
        id=f"{award_id}_winner",
        desc=f"Verify Game of the Year winner for {award_name}",
        parent=parent_node,
        critical=False,
    )
    
    # 1. Check if winner info exists (game name provided)
    winner_exists_node = evaluator.add_custom_node(
        result=bool(winner.game_name),
        id=f"{award_id}_winner_exists",
        desc=f"Game of the Year winner for {award_name} is provided",
        parent=award_winner_node,
        critical=True,
    )
    
    # 2. Verify game name is correct
    game_name_node = evaluator.add_leaf(
        id=f"{award_id}_game_name",
        desc=f"Game name for {award_name} winner is accurate",
        parent=award_winner_node,
        critical=True,
    )
    
    if award_url:
        claim = f"The game '{winner.game_name or 'N/A'}' was the Game of the Year (or Ultimate Game of the Year) winner at the {award_name} in the year {LAST_YEAR}."
        await evaluator.verify(
            claim=claim,
            node=game_name_node,
            sources=award_url,
            additional_instruction=f"Check if '{winner.game_name or 'N/A'}' is mentioned as the Game of the Year or Ultimate Game of the Year winner at {award_name}. The Game of the Year is typically the most prestigious award at these events.",
        )
    else:
        game_name_node.status = "failed"
        game_name_node.score = 0.0
    
    # 3. Official link verification (parallel node containing existence and validity)
    official_link_node = evaluator.add_parallel(
        id=f"{award_id}_official_link",
        desc=f"Official link for {award_name} winner",
        parent=award_winner_node,
        critical=False,
    )
    
    link_exists_node = evaluator.add_custom_node(
        result=bool(winner.official_link),
        id=f"{award_id}_official_link_exists",
        desc=f"Official link for {award_name} winner exists",
        parent=official_link_node,
        critical=True,
    )
    
    link_valid_node = evaluator.add_leaf(
        id=f"{award_id}_official_link_valid",
        desc=f"Official link for {award_name} winner is valid",
        parent=official_link_node,
        critical=True,
    )
    
    claim = f"This URL is a valid official page for the game '{winner.game_name or 'N/A'}'."
    await evaluator.verify(
        claim=claim,
        node=link_valid_node,
        sources=winner.official_link,
        additional_instruction=f"Check if the URL is an official website, Steam page, or vendor page for the game '{winner.game_name or 'N/A'}'. It should clearly be an official or authorized page for the game.",
    )
    
    # 4. Review video verification (parallel node containing existence and validity)
    review_link_node = evaluator.add_parallel(
        id=f"{award_id}_review_link",
        desc=f"Review video link for {award_name} winner",
        parent=award_winner_node,
        critical=False,
    )
    
    review_exists_node = evaluator.add_custom_node(
        result=bool(winner.review_video_link),
        id=f"{award_id}_review_exists",
        desc=f"Review video link for {award_name} winner exists",
        parent=review_link_node,
        critical=True,
    )
    
    # Check if it's YouTube in the additional instruction
    if winner.review_video_link and ("youtube.com" in winner.review_video_link or "youtu.be" in winner.review_video_link):
        is_youtube_instruction = f"Check if the URL is a review video in English for the game '{winner.game_name or 'N/A'}'. It should be a video that reviews, critiques, or analyzes the game."

        review_valid_node = evaluator.add_leaf(
            id=f"{award_id}_review_valid",
            desc=f"Review video is a valid YouTube review",
            parent=review_link_node,
            critical=True,
        )
        claim = f"This is a valid YouTube review video in English for the game '{winner.game_name or 'N/A'}'."
        await evaluator.verify(
            claim=claim,
            node=review_valid_node,
            sources=winner.review_video_link,
            additional_instruction=is_youtube_instruction,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{award_id}_review_valid",
            desc=f"Review video is a valid YouTube review",
            parent=review_link_node,
            critical=True
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
    # -------- 1. Set up evaluator ---------------------------------------- #
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

    # -------- 2. Extract structured info from the answer ---------------- #
    # Extract award links
    award_links = await evaluator.extract(
        prompt=prompt_extract_award_links(),
        template_class=AwardLinks,
        extraction_name="award_links",
    )
    
    # Extract individual game winners
    tga_winner = await evaluator.extract(
        prompt=prompt_extract_tga_winner(),
        template_class=GameWinner,
        extraction_name="tga_winner",
    )
    
    golden_joystick_winner = await evaluator.extract(
        prompt=prompt_extract_golden_joystick_winner(),
        template_class=GameWinner,
        extraction_name="golden_joystick_winner",
    )
    
    steam_winner = await evaluator.extract(
        prompt=prompt_extract_steam_winner(),
        template_class=GameWinner,
        extraction_name="steam_winner",
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Create parent node for award links
    award_links_node = evaluator.add_parallel(
        id="award_links",
        desc="Verify links to award event lists",
        parent=root,
        critical=False,
    )
    
    # Verify award links
    await verify_award_links(
        evaluator=evaluator,
        parent_node=award_links_node,
        award_links=award_links,
    )
    
    # Create game winners node
    winners_node = evaluator.add_parallel(
        id="game_winners",
        desc="Verify Game of the Year winners for each award event",
        parent=root,
        critical=False,
    )
    
    # Get URLs for each award event
    tga_url = next((link.url for link in award_links.links if link.name and "game awards" in link.name.lower()), None)
    golden_joystick_url = next((link.url for link in award_links.links if link.name and "golden joystick" in link.name.lower()), None)
    steam_url = next((link.url for link in award_links.links if link.name and "steam awards" in link.name.lower()), None)
    
    # Verify each winner
    await verify_game_winner(
        evaluator=evaluator,
        parent_node=winners_node,
        winner=tga_winner,
        award_name="The Game Awards",
        award_url=tga_url,
    )
    
    await verify_game_winner(
        evaluator=evaluator,
        parent_node=winners_node,
        winner=golden_joystick_winner,
        award_name="Golden Joystick Awards",
        award_url=golden_joystick_url,
    )
    
    await verify_game_winner(
        evaluator=evaluator,
        parent_node=winners_node,
        winner=steam_winner,
        award_name="Steam Awards",
        award_url=steam_url,
    )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()