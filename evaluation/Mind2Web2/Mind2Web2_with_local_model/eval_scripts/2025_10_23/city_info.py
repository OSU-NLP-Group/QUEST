import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import openai
from pydantic import BaseModel, Field

from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "city_info"
TASK_DESCRIPTION = """
I am conducting research on Seattle, Washington, and would like your help gathering some key information. Please start by providing the Wikipedia page for Seattle and, tell me the city's current population and its rank among U.S. cities by population by the information there. Additionally, find a webpage that ranks U.S. cities by rental prices and identify Seattle's position on that list. Lastly, I'd like to know more about the city's character—provide links to Wikipedia pages for three significant landmarks in Seattle as well as three notable historical events associated with the city.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SeattleWikipedia(BaseModel):
    """Information about the Wikipedia page for Seattle"""
    url: Optional[str] = None


class PopulationInfo(BaseModel):
    """Seattle's population information from Wikipedia"""
    population: Optional[str] = None
    population_rank: Optional[str] = None


class RentalRankInfo(BaseModel):
    """Information about Seattle's rental price ranking"""
    rental_rank: Optional[str] = None


class URLWithDescription(BaseModel):
    """URL with description"""
    url: Optional[str] = None
    description: Optional[str] = None


class RentalRankSource(BaseModel):
    """Source URLs for rental price rankings"""
    urls: List[URLWithDescription] = Field(default_factory=list)


class LandmarkName(BaseModel):
    """Name of a Seattle landmark"""
    name: Optional[str] = None


class LandmarkNames(BaseModel):
    """Collection of Seattle landmark names"""
    landmarks: List[LandmarkName] = Field(default_factory=list)


class LandmarkSource(BaseModel):
    """Wikipedia source for a specific landmark"""
    name: Optional[str] = None
    urls: List[URLWithDescription] = Field(default_factory=list)


class HistoricalEventName(BaseModel):
    """Name of a historical event in Seattle"""
    event: Optional[str] = None
    description: Optional[str] = None


class HistoricalEventNames(BaseModel):
    """Collection of Seattle historical event names"""
    events: List[HistoricalEventName] = Field(default_factory=list)


class HistoricalEventSource(BaseModel):
    """Source for a specific historical event"""
    event: Optional[str] = None
    urls: List[URLWithDescription] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_seattle_wikipedia() -> str:
    """
    Generate a prompt to extract the Wikipedia URL for Seattle.

    Returns:
        str: Extraction prompt for the Wikipedia URL.
    """
    return """
    Extract the URL for the Wikipedia page for Seattle that is provided in the answer.
    The URL should be a valid Wikipedia link for the city of Seattle, Washington.
    If no Wikipedia URL for Seattle is provided, return null.
    """


def prompt_extract_population_info() -> str:
    """
    Generate a prompt to extract Seattle's population information.

    Returns:
        str: Extraction prompt for population and rank information.
    """
    return """
    Extract the following information from the answer:
    1. Seattle's current population as mentioned in the answer
    2. Seattle's rank among U.S. cities by population as mentioned in the answer

    If any of these pieces of information are missing, return null for those fields.
    """


def prompt_extract_rental_rank_info() -> str:
    """
    Generate a prompt to extract Seattle's rental price ranking.

    Returns:
        str: Extraction prompt for rental rank information.
    """
    return """
    Extract Seattle's rank or position on a list of U.S. cities by rental prices as mentioned in the answer.
    The rental rank should include Seattle's position or rank number if provided.
    If this information is missing, return null.
    """


def prompt_extract_rental_rank_source() -> str:
    """
    Generate a prompt to extract the source URLs for rental price rankings.

    Returns:
        str: Extraction prompt for rental rank source URLs.
    """
    return """
    Extract all URL sources cited in the answer for information about Seattle's position on rental price rankings.
    For each URL, provide:
    1. The URL itself
    2. A brief description indicating this is a source for rental price rankings

    Return all URLs that appear to be sources for rental price information.
    """


def prompt_extract_landmark_names() -> str:
    """
    Generate a prompt to extract the names of Seattle landmarks.

    Returns:
        str: Extraction prompt for landmark names.
    """
    return """
    Extract the names of significant landmarks in Seattle mentioned in the answer.
    Return a list of all landmarks mentioned, with just their names.
    The task asks for three landmarks, but extract all that are provided in the answer.
    If no landmarks are mentioned, return an empty list.
    """


def prompt_extract_landmark_source(landmark_name: str) -> str:
    """
    Generate a prompt to extract the Wikipedia source URL for a specific landmark.

    Parameters:
        landmark_name (str): The name of the landmark to find sources for.

    Returns:
        str: Extraction prompt for landmark source URLs.
    """
    return f"""
    Extract all URL sources cited in the answer for the Seattle landmark named "{landmark_name}".
    For each URL, provide:
    1. The URL itself
    2. A brief description indicating this is a Wikipedia page for the landmark

    Return all URLs that appear to be sources for this specific landmark.
    If no URLs are provided for this landmark, return an empty list.
    """


def prompt_extract_historical_event_names() -> str:
    """
    Generate a prompt to extract the names of historical events in Seattle.

    Returns:
        str: Extraction prompt for historical event names and descriptions.
    """
    return """
    Extract information about the notable historical events associated with Seattle mentioned in the answer.
    For each historical event, extract:
    1. The name or title of the event
    2. Any description or details provided about the event

    Return a list of all historical events mentioned.
    The task asks for three historical events, but extract all that are provided in the answer.
    If no historical events are mentioned, return an empty list.
    """


def prompt_extract_historical_event_source(event_name: str) -> str:
    """
    Generate a prompt to extract source URLs for a specific historical event.

    Parameters:
        event_name (str): The name of the historical event to find sources for.

    Returns:
        str: Extraction prompt for historical event source URLs.
    """
    return f"""
    Extract all URL sources cited in the answer for the Seattle historical event "{event_name}".
    For each URL, provide:
    1. The URL itself
    2. A brief description indicating this is a source for information about the historical event

    Return all URLs that appear to be sources for this specific historical event.
    If no URLs are provided for this event, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
def is_valid_wikipedia_url(url: Optional[str]) -> bool:
    """
    Check if a URL is a valid Wikipedia page URL.

    Parameters:
        url (Optional[str]): The URL to check.

    Returns:
        bool: True if the URL is a valid Wikipedia URL, False otherwise.
    """
    if not url:
        return False

    parsed_url = urlparse(url)
    is_wikipedia = parsed_url.netloc.endswith("wikipedia.org")
    has_wiki_path = "/wiki/" in parsed_url.path

    return is_wikipedia and has_wiki_path


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_seattle_wikipedia_and_population(
        evaluator: Evaluator,
        wiki_info: SeattleWikipedia,
        pop_info: PopulationInfo,
) -> None:
    """
    Verify Seattle Wikipedia page and population information (combined task 1).
    This is an integrated task where all components are critical.
    """
    # Create main node - this task combines Wikipedia + population info
    wiki_pop_node = evaluator.add_parallel(
        "seattle_wiki_and_population",
        "The answer provides Seattle Wikipedia page and correct population information with proper sourcing.",
        critical=False,  # At root level this is non-critical for partial scoring
    )

    wiki_pop_exist = bool(
        wiki_info.url and is_valid_wikipedia_url(wiki_info.url)
        and pop_info.population
        and pop_info.population_rank
    )

    # 1. All basic info is provided
    wiki_pop_exist_node = evaluator.add_custom_node(
        result=wiki_pop_exist,
        id="wiki_population_exists",
        desc="The answer provides a valid Wikipedia page, information and URLs about Seattle's current population.",
        parent=wiki_pop_node,
        critical=True
    )

    # 2. Wikipedia verification
    wiki_node = evaluator.add_leaf(
        "seattle_wikipedia",
        "The answer provides a valid Wikipedia page URL for Seattle, Washington.",
        parent=wiki_pop_node,
        critical=True,
    )

    # Verify content is about Seattle using verifier
    claim = "This webpage is the Wikipedia article about Seattle, Washington, the city in the United States."
    await evaluator.verify(
        claim=claim,
        node=wiki_node,
        sources=wiki_info.url,
        additional_instruction="Verify that this is the main Wikipedia page for Seattle, Washington (the city in the United States), not a different Seattle or a related but different page."
    )


    # 3. Population accuracy (critical within this integrated task)
    pop_accuracy_node = evaluator.add_leaf(
        "population_accuracy",
        f"The provided population information ({pop_info.population}) is accurate according to the cited source.",
        parent=wiki_pop_node,
        critical=True,
    )

    pop_claim = f"According to this webpage, Seattle's population is {pop_info.population}."
    # Verify population against the wikipedia
    await evaluator.verify(
        claim=pop_claim,
        node=pop_accuracy_node,
        sources=wiki_info.url,
        additional_instruction="Verify that the population figure is accurately reported based on this source."
    )

    # 4. Rank accuracy
    rank_accuracy_node = evaluator.add_leaf(
        "population_rank_accuracy",
        f"The provided population rank information (Rank: {pop_info.population_rank}) is accurate according to the cited source.",
        parent=wiki_pop_node,
        critical=True,  # Within this independent task, accuracy is critical
    )

    # Create claim for rank verification
    rank_claim = f"According to this webpage, Seattle's rank among U.S. cities by population is {pop_info.population_rank}."
    # Verify rank against the source URL
    await evaluator.verify(
        claim=rank_claim,
        node=rank_accuracy_node,
        sources=wiki_info.url,
        additional_instruction="Verify that the population rank among U.S. cities is accurately reported based on this source."
    )


async def verify_rental_rank_info(
        evaluator: Evaluator,
        rental_info: RentalRankInfo,
        rental_sources: RentalRankSource,
) -> None:
    """
    Verify that the answer correctly provides Seattle's rental price ranking.
    """
    # Create main node for rental rank information
    rental_node = evaluator.add_parallel(
        "rental_rank_info",
        "The answer correctly provides Seattle's position on a list of U.S. cities by rental prices with proper sourcing.",
        critical=False,
    )

    # 1. Verify rental rank exists
    rank_exists_node = evaluator.add_custom_node(
        result=bool(rental_info.rental_rank and rental_sources.urls),
        id="rental_rank_exists",
        desc="The answer provides information and source URL(s) about Seattle's position on a list of U.S. cities by rental prices.",
        parent=rental_node,
        critical=True,
    )

    # 2. Verify rental rank accuracy
    rank_accuracy_node = evaluator.add_leaf(
        "rental_rank_accuracy",
        f"The provided rental rank information (Rank: {rental_info.rental_rank}) is accurate according to at least one of the cited sources.",
        parent=rental_node,
        critical=True,  # Within this task, accuracy is critical
    )

    # Create claim for verification
    claim = f"This webpage shows that Seattle's position or rank on a list of U.S. cities by rental prices is {rental_info.rental_rank}."

    # Extract just the URLs from the URLWithDescription objects
    urls = [link.url for link in rental_sources.urls if link.url]

    # Verify against any of the source URLs
    await evaluator.verify(
        claim=claim,
        node=rank_accuracy_node,
        sources=urls,
        additional_instruction="Verify that the webpage does rank U.S. cities by rental prices and that Seattle's position or rank on this list matches what was stated in the answer."
    )


async def verify_landmarks(
        evaluator: Evaluator,
        landmark_names: LandmarkNames,
        landmark_sources: List[LandmarkSource],
        logger: logging.Logger
) -> None:
    """
    Verify that the answer provides information about three significant landmarks in Seattle.
    """
    # Create main node for landmark information
    landmarks_node = evaluator.add_parallel(
        "landmarks_info",
        "The answer provides valid Wikipedia links to three significant landmarks in Seattle.",
        critical=False,
    )

    # Log information about number of landmarks provided
    total_landmarks = len(landmark_names.landmarks or [])
    if total_landmarks > 3:
        logger.info(f"{total_landmarks} landmarks provided, evaluating first 3 as requested.")
    elif total_landmarks < 3:
        logger.info(f"Only {total_landmarks} landmarks provided, creating placeholders for missing ones.")

    # Create a source lookup dictionary for easier access
    landmark_source_dict = {src.name: src for src in landmark_sources if src.name}

    # Take only the first 3 landmarks if more are provided
    landmarks_to_check = landmark_names.landmarks[:3]

    # Create placeholder nodes for missing landmarks
    while len(landmarks_to_check) < 3:
        landmarks_to_check.append(LandmarkName(name=None))

    # Verify each landmark
    for i, landmark in enumerate(landmarks_to_check):
        landmark_node = evaluator.add_parallel(
            f"landmark_{i + 1}",
            f"Landmark {i + 1}: {landmark.name if landmark.name else 'Missing'}",
            parent=landmarks_node,
            critical=False,
        )

        # Get the sources for this landmark
        sources = landmark_source_dict[landmark.name].urls if landmark.name in landmark_source_dict else []
        urls = [link.url for link in sources if link.url]
        wiki_urls = [url for url in urls if is_valid_wikipedia_url(url)]

        # Landmark info exists
        evaluator.add_custom_node(
            result=bool(landmark.name and len(wiki_urls)>0),
            id=f"landmark_{i + 1}_existence",
            desc=f"Landmark {i + 1} '{landmark.name}' is provided with name and wikipedia URL.",
            parent=landmark_node,
            critical=True
        )

        # Verify it's actually a Seattle landmark
        seattle_landmark_node = evaluator.add_leaf(
            f"landmark_{i + 1}_seattle_verification",
            f"The landmark '{landmark.name}' is a significant landmark located in Seattle.",
            parent=landmark_node,
            critical=True,
        )

        # Create claim for verification
        claim = f"This webpage confirms that '{landmark.name}' is a significant or notable landmark located in Seattle, Washington."

        # Verify against any of the Wikipedia URLs
        await evaluator.verify(
            claim=claim,
            node=seattle_landmark_node,
            sources=wiki_urls,
            additional_instruction="Verify that this is a significant or notable landmark and that it is located in Seattle, Washington. Both conditions must be met for this verification to pass."
        )


async def verify_historical_events(
        evaluator: Evaluator,
        event_names: HistoricalEventNames,
        event_sources: List[HistoricalEventSource],
        logger: logging.Logger
) -> None:
    """
    Verify that the answer provides information about three notable historical events in Seattle.
    """
    # Create main node for historical events information
    events_node = evaluator.add_parallel(
        "historical_events_info",
        "The answer provides information about three notable historical events associated with Seattle.",
        critical=False,
    )

    # Log information about number of events provided
    total_events = len(event_names.events) if event_names.events else 0
    if total_events > 3:
        logger.info(f"{total_events} historical events provided, evaluating first 3 as requested.")
    elif total_events < 3:
        logger.info(f"Only {total_events} historical events provided, creating placeholders for missing ones.")

    # Create a source lookup dictionary for easier access
    event_source_dict = {src.event: src for src in event_sources if src.event}

    # Take only the first 3 events if more are provided
    events_to_check = event_names.events[:3]

    # Create placeholder nodes for missing events
    while len(events_to_check) < 3:
        events_to_check.append(HistoricalEventName(event=None, description=None))

    # Verify each event
    for i, event in enumerate(events_to_check):
        event_node = evaluator.add_parallel(
            f"event_{i + 1}",
            f"Historical Event {i + 1}: {event.event if event.event else 'Missing'}",
            parent=events_node,
            critical=False,
        )

        # Get the sources for this event
        sources = event_source_dict[event.event].urls if event.event in event_source_dict else []
        urls = [link.url for link in sources if link.url]

        evaluator.add_custom_node(
            result=bool(event.event and len(urls)>0),
            id=f"event_{i + 1}_existence",
            desc=f"Historical Event {i + 1} '{event.event}' is provided with an event name and source URL(s).",
            parent=event_node,
            critical=True
        )


        # Verify it's a Seattle historical event using sources
        seattle_event_node = evaluator.add_leaf(
            f"event_{i + 1}_seattle_verification",
            f"The historical event '{event.event}' is a notable event associated with Seattle according to cited sources.",
            parent=event_node,
            critical=True,
        )

        # Create claim for verification
        claim = f"This webpage confirms that '{event.event}' is a notable historical event associated with Seattle, Washington."

        # Verify against any of the source URLs
        await evaluator.verify(
            claim=claim,
            node=seattle_event_node,
            sources=urls,
            additional_instruction="Verify that this is a notable historical event and that it is associated with Seattle, Washington. Both conditions must be met for this verification to pass."
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
    Evaluate a single answer to the Seattle information task and return a structured result dictionary.

    Parameters:
        client (openai.AsyncAzureOpenAI): The OpenAI client to use for LLM calls.
        answer (str): The answer text to evaluate.
        agent_name (str): Name of the agent that generated the answer.
        answer_name (str): Identifier for this specific answer.
        cache (CacheFileSys): Cache for web content to avoid repeated fetches.
        semaphore (asyncio.Semaphore): Semaphore to control concurrent API requests.
        logger (logging.Logger): Logger for recording evaluation process.
        model (str): The OpenAI model to use for evaluation, default is "o4-mini".

    Returns:
        Dict: Structured evaluation results including score and breakdown.
    """
    # Create evaluator instance
    evaluator = Evaluator()
    
    # Initialize with task information
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

    logger.info("Beginning evaluation of Seattle information task...")

    # Extract basic information first
    logger.info("Extracting Seattle Wikipedia URL...")
    seattle_wiki_info = await evaluator.extract(
        prompt=prompt_extract_seattle_wikipedia(),
        template_class=SeattleWikipedia,
        extraction_name="seattle_wiki_info"
    )

    logger.info("Extracting population information...")
    population_info = await evaluator.extract(
        prompt=prompt_extract_population_info(),
        template_class=PopulationInfo,
        extraction_name="population_info"
    )

    # Task 1: Verify Seattle Wikipedia + Population (integrated task)
    logger.info("Verifying Seattle Wikipedia page and population information...")
    await verify_seattle_wikipedia_and_population(evaluator, seattle_wiki_info, population_info)

    # Task 2: Extract and verify rental rank information (independent task)
    logger.info("Extracting rental rank information...")
    rental_rank_info = await evaluator.extract(
        prompt=prompt_extract_rental_rank_info(),
        template_class=RentalRankInfo,
        extraction_name="rental_rank_info"
    )

    logger.info("Extracting rental rank source URLs...")
    rental_rank_source = await evaluator.extract(
        prompt=prompt_extract_rental_rank_source(),
        template_class=RentalRankSource,
        extraction_name="rental_rank_source"
    )

    logger.info("Verifying rental rank information...")
    await verify_rental_rank_info(evaluator, rental_rank_info, rental_rank_source)

    # Task 3: Extract and verify landmarks information (3 independent subtasks)
    logger.info("Extracting landmark names...")
    landmark_names = await evaluator.extract(
        prompt=prompt_extract_landmark_names(),
        template_class=LandmarkNames,
        extraction_name="landmark_names"
    )

    # For each landmark, extract its sources
    landmark_sources = []
    for landmark in landmark_names.landmarks:
        if landmark.name:
            logger.info(f"Extracting sources for landmark: {landmark.name}...")
            landmark_source = await evaluator.extract(
                prompt=prompt_extract_landmark_source(landmark.name),
                template_class=LandmarkSource,
                extraction_name=f"landmark_source_{landmark.name}",
                additional_instruction=f"Focus specifically on finding URLs associated with the landmark named '{landmark.name}'. Only extract URLs that are clearly linked to this specific landmark."
            )
            landmark_source.name = landmark.name
            landmark_sources.append(landmark_source)

    logger.info("Verifying landmarks information...")
    await verify_landmarks(evaluator, landmark_names, landmark_sources, logger)

    # Task 4: Extract and verify historical events information (3 independent subtasks)
    logger.info("Extracting historical event names...")
    historical_event_names = await evaluator.extract(
        prompt=prompt_extract_historical_event_names(),
        template_class=HistoricalEventNames,
        extraction_name="historical_event_names"
    )

    # For each event, extract its sources
    event_sources = []
    for event in historical_event_names.events:
        if event.event:
            logger.info(f"Extracting sources for event: {event.event}...")
            event_source = await evaluator.extract(
                prompt=prompt_extract_historical_event_source(event.event),
                template_class=HistoricalEventSource,
                extraction_name=f"event_source_{event.event}",
                additional_instruction=f"Focus specifically on finding URLs associated with the historical event '{event.event}'. Only extract URLs that are clearly linked to this specific event."
            )
            event_source.event = event.event
            event_sources.append(event_source)

    logger.info("Verifying historical events information...")
    await verify_historical_events(evaluator, historical_event_names, event_sources, logger)

    # Get final score and summary
    logger.info("Calculating final score...")
    final_score = evaluator.score()

    logger.info(f"Evaluation complete. Final score: {final_score}")

    # Return structured result using the new get_summary method
    return evaluator.get_summary()