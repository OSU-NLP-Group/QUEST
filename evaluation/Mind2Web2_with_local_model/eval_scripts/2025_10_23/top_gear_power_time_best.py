import asyncio
import logging
from typing import Optional, Dict, List

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "top_gear_power_time_best"
TASK_DESCRIPTION = """
Top Gear is a British motoring television show. For many seasons, it was hosted by the iconic trio: Jeremy Clarkson, James May, and Richard Hammond. One of the show's key segments was the 'Top Gear Power Lap,' where cars were tested and their fastest lap times were ranked on a leaderboard from fastest to slowest, accumulated across all seasons.

Which car is at the top of the leaderboard in the final season hosted by the trio? Please include the specific car model and the season and episode in which the lap was recorded.
"""

# Ground truth information
GROUND_TRUTH = {
    "car_model": "Pagani Huayra",
    "season": "19",
    "episode": "1"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                 #
# --------------------------------------------------------------------------- #
class CarModelExtraction(BaseModel):
    """Extraction model for the top car on the Top Gear Power Lap leaderboard."""
    car_model: Optional[str] = None


class EpisodeInfoExtraction(BaseModel):
    """Extraction model for the season and episode information."""
    season: Optional[str] = None
    episode: Optional[str] = None


class ProvLink(BaseModel):
    """A single source URL with description."""
    url: str
    description: str = ""


class ProvLinks(BaseModel):
    """Collection of source URLs."""
    links: List[ProvLink] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_car_model() -> str:
    """
    Generate the prompt for extracting the car model from the answer.

    Returns:
        str: The extraction prompt for the car model.
    """
    return """
    Extract the car model that is stated to be at the top of the Top Gear Power Lap leaderboard 
    in the final season with the trio (Jeremy Clarkson, James May, and Richard Hammond).

    For the car model, extract the complete and specific model name (e.g., "Ferrari LaFerrari" not just "Ferrari").

    If this information is missing from the answer, return null for the car_model field.
    """


def prompt_extract_episode_info() -> str:
    """
    Generate the prompt for extracting the season and episode information from the answer.

    Returns:
        str: The extraction prompt for the season and episode information.
    """
    return """
    Extract the following information from the answer:
    1. The season number in which the top car's lap was recorded
    2. The episode number in which the top car's lap was recorded

    For the season and episode, extract just the numbers (e.g., "22" for season, "5" for episode).
    If any of this information is missing from the answer, return null for that field.
    """


def prompt_extract_car_urls() -> str:
    """
    Generate the prompt for extracting URLs related to the car model information.

    Returns:
        str: The extraction prompt for the car model URLs.
    """
    return """
    Extract all URLs that are provided in the answer to support the claim about which car 
    is at the top of the Top Gear Power Lap leaderboard in the final season with the trio.

    For each URL, briefly describe what information it is supposed to provide about the car model.
    Only extract URLs that are directly relevant to substantiating the car model claim.
    """


def prompt_extract_episode_urls() -> str:
    """
    Generate the prompt for extracting URLs related to the episode information.

    Returns:
        str: The extraction prompt for the episode information URLs.
    """
    return """
    Extract all URLs that are provided in the answer to support the claims about the season and episode 
    where the top car's lap was recorded on the Top Gear Power Lap leaderboard.

    For each URL, briefly describe what information it is supposed to provide about the season and episode.
    Only extract URLs that are directly relevant to substantiating the season and episode claims.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_car_info(
        evaluator: Evaluator,
        parent_node,
        car_model: Optional[str],
        car_urls: ProvLinks,
) -> None:
    """
    Verify the car model information, including both correctness and provenance.
    """
    # Create a node for car model verification (groups correctness and provenance)
    car_node = evaluator.add_parallel(
        id="car_model",
        desc="Verification of the car model at the top of the Top Gear Power Lap leaderboard",
        parent=parent_node,
        critical=False,
    )

    # ----- Existence check (critical) -----
    existence_check = evaluator.add_custom_node(
        result=bool(car_model and car_urls.links),
        id="car_model_exists",
        desc="Check if car model information is provided",
        parent=car_node,
        critical=True
    )

    # ----- Correctness check -----
    correctness_node = evaluator.add_leaf(
        id="car_model_correctness",
        desc=f"Verify if the car model '{car_model if car_model else 'Not provided'}' correctly matches the ground truth '{GROUND_TRUTH['car_model']}'",
        parent=car_node,
        critical=True,
    )

    # Always call verify regardless of data existence
    await evaluator.verify(
        claim=f"The car model '{car_model}' is the same as the expected car model '{GROUND_TRUTH['car_model']}'",
        node=correctness_node,
        sources=None,  # Simple verification against ground truth
        additional_instruction="Consider slight variations in naming or typography to be acceptable matches if they clearly refer to the same car model. For example, 'Pagani Huayra BC' would be considered a match for 'Pagani Huayra' as it's a variant of the same model."
    )

    # ----- Provenance check -----
    provenance_node = evaluator.add_leaf(
        id="car_model_provenance",
        desc="Verify if the car model appears on the Top Gear Power Lap leaderboard according to at least one valid source URL",
        parent=car_node,
        critical=True,
    )

    # Always call verify with the URLs we have
    urls = [link.url for link in car_urls.links]
    await evaluator.verify(
        claim=f"The car '{car_model}' appears on the Top Gear Power Lap leaderboard",
        node=provenance_node,
        sources=urls,  # Pass list directly, even if empty
        additional_instruction="Check if the source confirms that the specified car appears on the Top Gear Power Lap leaderboard. You don't need to verify that it was the top car or from a specific season."
    )


async def verify_episode_info(
        evaluator: Evaluator,
        parent_node,
        car_model: Optional[str],
        season: Optional[str],
        episode: Optional[str],
        episode_urls: ProvLinks,
) -> None:
    """
    Verify the season and episode information, including both correctness and provenance.
    """
    # Create a node for episode information verification
    episode_info_node = evaluator.add_parallel(
        id="episode_info",
        desc="Verification of the season and episode where the specific car's lap was recorded",
        parent=parent_node,
        critical=False,  # Episode info is supplementary to the main car model question
    )

    # ----- Existence check (critical) -----
    existence_check = evaluator.add_custom_node(
        result=bool(season) and bool(episode) and bool(episode_urls),
        id="episode_info_exists",
        desc="Check if both season and episode information are provided",
        parent=episode_info_node,
        critical=True
    )

    # ----- Combined season and episode correctness check -----
    episode_correctness_node = evaluator.add_leaf(
        id="episode_correctness",
        desc=f"Verify if the season '{season if season else 'Not provided'}' and episode '{episode if episode else 'Not provided'}' correctly match the ground truth season '{GROUND_TRUTH['season']}' and episode '{GROUND_TRUTH['episode']}'",
        parent=episode_info_node,
        critical=True,
    )

    # Always call verify
    await evaluator.verify(
        claim=f"The season '{season}' matches the expected season '{GROUND_TRUTH['season']}' and the episode '{episode}' matches the expected episode '{GROUND_TRUTH['episode']}'",
        node=episode_correctness_node,
        sources=None,
        additional_instruction="Consider variations like 'Season 19' and '19' as equivalent, same for episodes. The numerical values are what matter for comparison."
    )

    # ----- Provenance check -----
    provenance_node = evaluator.add_leaf(
        id="episode_info_provenance",
        desc=f"Verify if the season and episode for the '{car_model if car_model else 'specified car'}' are supported by at least one valid source URL",
        parent=episode_info_node,
        critical=True,
    )

    # Always call verify with the URLs we have
    urls = [link.url for link in episode_urls.links]
    await evaluator.verify(
        claim=f"The {car_model}'s lap on the Top Gear Power Lap leaderboard was recorded in Season {season if season else 'unknown'}, Episode {episode if episode else 'unknown'}",
        node=provenance_node,
        sources=urls,
        additional_instruction=f"Check if the source confirms that the {car_model}'s lap was recorded in the specific season and episode mentioned."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client,  # openai client
        answer: str,
        agent_name: str,
        answer_name: str,
        cache,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the answer for the Top Gear Power Lap task.

    This function performs a staged extraction and verification process:
    1. First extracts the car model
    2. Then extracts season and episode information
    3. Extracts URLs supporting each claim separately
    4. Verifies correctness and provenance for each piece of information

    Parameters:
        client: The OpenAI client for LLM calls.
        answer (str): The answer text to evaluate.
        agent_name (str): Identifier for the agent that produced the answer.
        answer_name (str): Identifier for the answer being evaluated.
        cache: Cache for storing webpage content.
        semaphore (asyncio.Semaphore): Semaphore for rate limiting.
        logger (logging.Logger): Logger for recording evaluation steps.
        model (str, optional): The LLM model to use. Defaults to "o4-mini".

    Returns:
        Dict: Evaluation results including score and verification breakdown.
    """
    # -------- 1. Initialize evaluator ----------------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        task_description=TASK_DESCRIPTION,
        client=client,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
        extract_model=model,
        verify_model=model
    )

    # -------- 2. Extract information in stages -------------------------- #
    # Stage 1: Extract car model first
    car_extraction = await evaluator.extract(
        prompt=prompt_extract_car_model(),
        template_class=CarModelExtraction,
        extraction_name="car_model_extraction",
    )

    # Stage 2: Extract season and episode information
    episode_extraction = await evaluator.extract(
        prompt=prompt_extract_episode_info(),
        template_class=EpisodeInfoExtraction,
        extraction_name="episode_info_extraction",
    )

    # Stage 3: Extract URLs for car model claim
    car_urls = await evaluator.extract(
        prompt=prompt_extract_car_urls(),
        template_class=ProvLinks,
        extraction_name="car_urls_extraction",
    )

    # Stage 4: Extract URLs for episode information claims
    episode_urls = await evaluator.extract(
        prompt=prompt_extract_episode_urls(),
        template_class=ProvLinks,
        extraction_name="episode_urls_extraction",
    )

    # Add ground truth info for reference
    evaluator.add_ground_truth(GROUND_TRUTH, "expected_values")

    # -------- 3. Verify information ------------------------------------- #
    # Verify car model information (both correctness and provenance)
    await verify_car_info(
        evaluator=evaluator,
        parent_node=root,
        car_model=car_extraction.car_model,
        car_urls=car_urls,
    )

    # Verify episode information (both correctness and provenance)
    await verify_episode_info(
        evaluator=evaluator,
        parent_node=root,
        car_model=car_extraction.car_model,
        season=episode_extraction.season,
        episode=episode_extraction.episode,
        episode_urls=episode_urls,
    )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()