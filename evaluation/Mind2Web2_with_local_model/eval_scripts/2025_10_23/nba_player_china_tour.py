import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "nba_player_china_tour"
TASK_DESCRIPTION = """
Identify 3 current NBA players who are now actively playing in the NBA, have participated in at least 5 NBA All-Star Games, are not Chinese citizens, and have never professionally played for a Chinese basketball team or held an official coaching or managerial role in China. However, each player should have conducted at least one basketball-related tour in China within the past 10 years. For each player, clearly provide the following: list any 5 NBA All-Star Games in which they participated, a link to their Wikipedia page, the year their China tour occurred, and a direct link to an article or news source confirming the China tour.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}


class PlayerNamesList(BaseModel):
    """List of NBA player names"""
    player_names: Optional[List[str]] = Field(default_factory=list, description="List of NBA player names")


class PlayerDetails(BaseModel):
    """Detailed information about a single NBA player"""
    name: Optional[str] = Field(default=None, description="Player's name")
    all_star_games: Optional[List[str]] = Field(default_factory=list, description="List of All-Star Game years")
    wikipedia_url: Optional[str] = Field(default=None, description="Wikipedia page URL")
    china_tour_year: Optional[str] = Field(default=None, description="Year of China tour")
    china_tour_source_url: Optional[str] = Field(default=None, description="News source URL confirming China tour")
    other_urls: Optional[List[str]] = Field(default_factory=list,
                                            description="Any other URLs mentioned for this player")


def prompt_extract_player_names() -> str:
    """Extraction prompt for player names only"""
    return """
    Extract ONLY the names of NBA players mentioned in the answer.

    Return a simple list of player names exactly as they appear in the text.
    Do not extract any other information at this stage.
    """


def prompt_extract_player_details(player_name: str) -> str:
    """Extraction prompt for a specific player's details"""
    return f"""
    Extract detailed information ONLY about the NBA player named "{player_name}".

    Look for and extract:
    - name: Confirm this is {player_name}
    - all_star_games: List of years they participated in NBA All-Star Games (as strings)
    - wikipedia_url: Their Wikipedia page URL
    - china_tour_year: The year they conducted a tour in China
    - china_tour_source_url: URL confirming their China tour
    - other_urls: Any other URLs mentioned in relation to this player

    Extract information exactly as it appears. If any field is not found, set it to null.
    """


async def verify_player(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        player: PlayerDetails,
        player_index: int,
        current_year: int,
) -> None:
    """Verify a single NBA player's information"""

    # Create player node
    player_node = evaluator.add_parallel(
        id=f"player_{player_index}",
        desc=f"Player {player_index + 1}: {player.name or 'Unknown'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Check all necessary information exists
    info_complete = bool(
        player.name and player.name.strip() and
        player.all_star_games and len(player.all_star_games) >= 5 and
        player.wikipedia_url and player.wikipedia_url.strip() and
        player.china_tour_year and player.china_tour_year.strip() and
        player.china_tour_source_url and player.china_tour_source_url.strip()
    )

    info_exists_node = evaluator.add_custom_node(
        result=info_complete,
        id=f"player_{player_index}_info_complete",
        desc=f"All required information is provided (name, 5+ All-Star games, Wikipedia URL, tour year, tour source)",
        parent=player_node,
        critical=True,
    )

    # Gather all URLs for this player
    all_urls = []
    if player.wikipedia_url:
        all_urls.append(player.wikipedia_url)
    if player.china_tour_source_url:
        all_urls.append(player.china_tour_source_url)
    if player.other_urls:
        all_urls.extend(player.other_urls)

    # Filter Wikipedia URLs
    wiki_urls = [url for url in all_urls if url and ('wikipedia' in url.lower() or 'wiki' in url.lower())]

    # Verify Wikipedia page corresponds to the player
    wiki_verification_node = evaluator.add_leaf(
        id=f"player_{player_index}_wikipedia_verification",
        desc=f"Wikipedia page correctly identifies {player.name}",
        parent=player_node,
        critical=True,
    )

    # if wiki_urls and player.name:

    if not wiki_urls:
        wiki_urls=[""]
    await evaluator.verify(
        claim=f"The Wikipedia page is about NBA player {player.name}",
        node=wiki_verification_node,
        sources=wiki_urls,  # Will use verify_by_urls
        additional_instruction="Confirm the Wikipedia page is about the correct NBA player"
    )

    # Verify all constraints using all available URLs

    # 1. Currently active in NBA
    currently_active_node = evaluator.add_leaf(
        id=f"player_{player_index}_currently_active",
        desc=f"Player is currently actively playing in the NBA",
        parent=player_node,
        critical=True,
    )

    # if all_urls and player.name:
    await evaluator.verify(
        claim=f"{player.name} is currently an active NBA player as indicated by the webpage",
        node=currently_active_node,
        sources=all_urls,
    )

    # 2. Verify All-Star Games participation
    allstar_verification_node = evaluator.add_leaf(
        id=f"player_{player_index}_allstar_verification",
        desc=f"Participated in at least 5 NBA All-Star Games as claimed",
        parent=player_node,
        critical=True,
    )

    # if all_urls and player.name and player.all_star_games and len(player.all_star_games) >= 5:
    allstar_years_str = ", ".join(player.all_star_games[:5])
    await evaluator.verify(
        claim=f"{player.name} participated in NBA All-Star Games in these years: {allstar_years_str}",
        node=allstar_verification_node,
        sources=all_urls,
        additional_instruction="Verify these All-Star Game participations are accurate"
    )

    # 3. Verify not Chinese citizen
    not_chinese_node = evaluator.add_leaf(
        id=f"player_{player_index}_not_chinese",
        desc=f"Player is not a Chinese citizen",
        parent=player_node,
        critical=True,
    )

    # if all_urls and player.name:
    await evaluator.verify(
        claim=f"{player.name} is not a Chinese citizen",
        node=not_chinese_node,
        sources=all_urls,
        additional_instruction="Check the player's nationality/citizenship to confirm they are not Chinese"
    )

    # 4. Verify no professional China connection
    no_china_pro_node = evaluator.add_leaf(
        id=f"player_{player_index}_no_china_pro",
        desc=f"Never professionally played for Chinese team or held coaching role in China",
        parent=player_node,
        critical=True,
    )

    # if all_urls and player.name:
    await evaluator.verify(
        claim=f"This page shows the professional history of an NBA player {player.name} or contains his bio or a summary of his bio. And according to the page, he has never professionally played for a Chinese basketball team and has never held an official coaching or managerial role in China",
        node=no_china_pro_node,
        sources=all_urls,
        additional_instruction="Check career history to ensure no professional Chinese team affiliations or coaching roles"
    )

    # 5. Verify China tour occurred
    china_tour_node = evaluator.add_leaf(
        id=f"player_{player_index}_china_tour_occurred",
        desc=f"Conducted basketball-related tour in China as claimed",
        parent=player_node,
        critical=True,
    )

    # if all_urls and player.name and player.china_tour_year:
    await evaluator.verify(
        claim=f"{player.name} conducted a basketball-related tour in China in {player.china_tour_year}",
        node=china_tour_node,
        sources=all_urls,
        additional_instruction="Verify a basketball-related tour (not just a visit) occurred in the specified year"
    )

    # 6. Verify tour was within past 10 years using simple_verify
    tour_recency_node = evaluator.add_leaf(
        id=f"player_{player_index}_tour_recency",
        desc=f"China tour occurred within past 10 years",
        parent=player_node,
        critical=True,
    )

    # if player.china_tour_year:
    await evaluator.verify(
        claim=f"The year {player.china_tour_year} is within the past 10 years (current year is {current_year})",
        node=tour_recency_node,
        sources=None,  # Will use simple_verify
        additional_instruction=f"In other words, verify that {player.china_tour_year} is between {current_year - 10} and {current_year}"
    )


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
    """Main evaluation function for NBA players China tour task"""

    # Initialize evaluator
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
        default_model=model,
    )

    # Get current year dynamically
    current_year = datetime.now().year

    # Step 1: Extract player names only
    player_names_info = await evaluator.extract(
        prompt=prompt_extract_player_names(),
        template_class=PlayerNamesList,
        extraction_name="player_names_extraction",
    )

    # Process exactly 3 players
    names_to_process = player_names_info.player_names[:3] if player_names_info.player_names else []

    # Step 2: Extract details for each player
    players_details = []
    for i, name in enumerate(names_to_process):
        player_detail = await evaluator.extract(
            prompt=prompt_extract_player_details(name),
            template_class=PlayerDetails,
            extraction_name=f"player_{i}_details",
        )
        players_details.append(player_detail)

    # Add placeholder players if fewer than 3
    while len(players_details) < 3:
        players_details.append(PlayerDetails())

    # Step 3: Verify each player
    for i, player in enumerate(players_details):
        await verify_player(evaluator, root, player, i, current_year)

    # Return evaluation results
    return evaluator.get_summary()