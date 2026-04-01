import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tokyo_filming_location"
TASK_DESCRIPTION = """
As a movie lover planning a trip to Tokyo, I want to explore the city through the lens of cinema. Please identify five filming locations in Tokyo, Japan, that have appeared in U.S.-directed films (i.e., films directed by a U.S. filmmaker). Each location must be a clearly identifiable place, such as a landmark, a distinct neighborhood or area (e.g., Shibuya Crossing or Kabukicho), a commercial venue (e.g., a hotel or restaurant), or a cultural/historical site (e.g., a shrine or museum).

For each location, provide the following details:
* A movie that was filmed there, including its title, a link to its Rotten Tomatoes page, the name of one U.S. director involved, and a link to that director's Wikipedia page.
* At least one verifiable source (e.g., IMDb, production notes, interviews) confirming the location's use in the film.

Each of the five locations must be associated with a different film. Please provide five unique film-location pairings.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmLocationPair(BaseModel):
    """Basic film-location pair information."""
    location_name: Optional[str] = None
    film_title: Optional[str] = None


class ExtractedPairs(BaseModel):
    """All film-location pairs extracted from the answer."""
    pairs: List[FilmLocationPair] = Field(default_factory=list)


class DetailedFilmingLocation(BaseModel):
    """Detailed information about a specific filming location and film."""
    location_name: Optional[str] = None
    film_title: Optional[str] = None
    rotten_tomatoes_url: Optional[str] = None
    director_name: Optional[str] = None
    director_wikipedia_url: Optional[str] = None
    verification_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_film_location_pairs() -> str:
    return """
    Extract the basic film-location pairs mentioned in the answer. For each pairing mentioned, extract:
    - location_name: The name of the Tokyo filming location
    - film_title: The title of the film that was shot there

    Extract all pairs mentioned, even if there are more or fewer than 5. Focus only on the core location and film names.
    """


def prompt_extract_detailed_info_for_pair(location_name: str, film_title: str) -> str:
    return f"""
    For the specific film-location pair: Location "{location_name}" and Film "{film_title}", extract the following detailed information:
    - location_name: The exact name of the Tokyo filming location (should match "{location_name}")
    - film_title: The exact title of the film (should match "{film_title}")
    - rotten_tomatoes_url: The Rotten Tomatoes URL for this film (if provided)
    - director_name: The name of the U.S. director mentioned for this film
    - director_wikipedia_url: The Wikipedia URL for this director (if provided)
    - verification_sources: List of URLs or sources provided to verify this specific filming location

    Focus only on information related to this specific film-location pair.
    """


# --------------------------------------------------------------------------- #
# Individual verification functions for each location                         #
# --------------------------------------------------------------------------- #
async def verify_single_location(
        evaluator: Evaluator,
        parent_node,
        pair: FilmLocationPair,
        location_idx: int,
) -> None:
    """Verify all aspects of a single filming location."""
    
    # Create parent node for this location
    location_node = evaluator.add_parallel(
        id=f"location_{location_idx}",
        desc=f"Location {location_idx + 1}: Film-location pairing '{pair.film_title or 'N/A'}' at '{pair.location_name or 'N/A'}' meets all requirements",
        parent=parent_node,
        critical=False  # Individual locations are non-critical to allow partial scoring
    )

    # Extract detailed information for this specific pair
    detailed_location = await evaluator.extract(
        prompt=prompt_extract_detailed_info_for_pair(pair.location_name or "", pair.film_title or ""),
        template_class=DetailedFilmingLocation,
        extraction_name=f"detailed_location_{location_idx}"
    )

    # Verify basic info completeness (this serves as the existence check)
    basic_info_complete = evaluator.add_custom_node(
        result=(
            bool(detailed_location.film_title and detailed_location.film_title.strip()) and
            bool(detailed_location.rotten_tomatoes_url and detailed_location.rotten_tomatoes_url.strip()) and
            bool(detailed_location.director_name and detailed_location.director_name.strip()) and
            bool(detailed_location.director_wikipedia_url and detailed_location.director_wikipedia_url.strip())
        ),
        id=f"location_{location_idx}_basic_info_complete",
        desc=f"Location {location_idx + 1} has all required information: film title, Rotten Tomatoes URL, director name, and director Wikipedia URL",
        parent=location_node,
        critical=True
    )

    # Verify Rotten Tomatoes correspondence
    rt_node = evaluator.add_leaf(
        id=f"location_{location_idx}_rt_verification",
        desc=f"The Rotten Tomatoes URL corresponds to the film '{detailed_location.film_title or 'N/A'}'",
        parent=location_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"This is a Rotten Tomatoes page, and it is for the film '{detailed_location.film_title or ''}'",
        node=rt_node,
        sources=detailed_location.rotten_tomatoes_url,
        additional_instruction="Check if the film title on this Rotten Tomatoes page matches the specified film title. Allow for reasonable variants in formatting, punctuation, subtitles, or minor spelling differences that represent the same film."
    )

    # Verify Wikipedia correspondence
    wiki_node = evaluator.add_leaf(
        id=f"location_{location_idx}_wiki_verification",
        desc=f"The Wikipedia URL corresponds to director '{detailed_location.director_name or 'N/A'}'",
        parent=location_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"This is a Wikipedia page, and it is about the director '{detailed_location.director_name or ''}'",
        node=wiki_node,
        sources=detailed_location.director_wikipedia_url,
        additional_instruction="Check if this Wikipedia page is about the specified director. Allow for reasonable variants in name formatting, including different orders of first/middle/last names, nicknames, full names vs. shortened versions, or minor spelling differences that refer to the same person."
    )

    # Verify U.S. director
    us_director_node = evaluator.add_leaf(
        id=f"location_{location_idx}_us_director",
        desc=f"Director '{detailed_location.director_name or 'N/A'}' is a U.S. director",
        parent=location_node,
        critical=True
    )
    
    # Prepare URLs to check
    urls_to_check = []
    if detailed_location.director_wikipedia_url:
        urls_to_check.append(detailed_location.director_wikipedia_url)
    if detailed_location.verification_sources:
        urls_to_check.extend(detailed_location.verification_sources)
    if detailed_location.rotten_tomatoes_url:
        urls_to_check.append(detailed_location.rotten_tomatoes_url)
    
    await evaluator.verify(
        claim=f"This page shows that the Director '{detailed_location.director_name or ''}' is a U.S. director (American filmmaker)",
        node=us_director_node,
        sources=urls_to_check,
        additional_instruction="Look for information about the director's nationality, birthplace, or career background that indicates they are a U.S. director or American filmmaker. Allow for reasonable variants in the director's name formatting, including different orders of first/middle/last names, nicknames, or minor spelling differences."
    )

    # Verify director-film participation
    director_film_node = evaluator.add_leaf(
        id=f"location_{location_idx}_director_film",
        desc=f"Director '{detailed_location.director_name or 'N/A'}' participated in filming the movie '{detailed_location.film_title or 'N/A'}'",
        parent=location_node,
        critical=True
    )
    
    # Prepare URLs for director-film verification
    director_film_urls = []
    if detailed_location.rotten_tomatoes_url:
        director_film_urls.append(detailed_location.rotten_tomatoes_url)
    if detailed_location.verification_sources:
        director_film_urls.extend(detailed_location.verification_sources)
    if detailed_location.director_wikipedia_url:
        director_film_urls.append(detailed_location.director_wikipedia_url)
    
    await evaluator.verify(
        claim=f"Director '{detailed_location.director_name or ''}' was involved in directing the film '{detailed_location.film_title or ''}'",
        node=director_film_node,
        sources=director_film_urls,
        additional_instruction="Look for director credits or information confirming that this director was involved in making this film. Allow reasonable variants in both the director's name (different name formatting, nicknames, etc.) and the film title (formatting differences, subtitles, etc.) as long as they refer to the same person and film. Don't be too strict on the full naming. For example, the full name of the director may not be presented, instead, there may be just the first name, middle name, last name or nick name presented, which should also be treated as correct."
    )

    # Verify Tokyo filming location
    tokyo_filming_node = evaluator.add_leaf(
        id=f"location_{location_idx}_tokyo_filming",
        desc=f"Film '{detailed_location.film_title or 'N/A'}' was shot at the identifiable Tokyo location '{detailed_location.location_name or 'N/A'}'",
        parent=location_node,
        critical=True
    )

    # Prepare verification sources
    verification_urls = detailed_location.verification_sources.copy()

    # Try primary sources first
    await evaluator.verify(
        claim=f"The filming of the film '{detailed_location.film_title or ''}' involved filming at '{detailed_location.location_name or ''}' in Tokyo, Japan, and this location is an identifiable place such as a landmark, neighborhood, commercial venue, or cultural/historical site",
        node=tokyo_filming_node,
        sources=verification_urls,
        additional_instruction="Look for evidence that: 1) this film was shot at this specific location in Tokyo, 2) this location is indeed in Tokyo, Japan (from your knowledge or evidence from the page), and 3) this location is clearly identifiable, instead of just rough descriptions that are not able to actually find (landmark, neighborhood, commercial venue, cultural site, exact locations, etc.). Allow reasonable variants in both the film title and location name formatting, as long as they refer to the same film and location."
    )

    # If not verified with primary sources, try additional source
    if tokyo_filming_node.score == 0.0 and detailed_location.location_name:
        from urllib.parse import quote
        extra_source = f'https://m.imdb.com/search/title/?locations={quote(detailed_location.location_name)}'
        
        # Try with the additional source
        result = await evaluator.verify(
            claim=f"The filming of the film '{detailed_location.film_title or ''}' involved filming at '{detailed_location.location_name or ''}' in Tokyo, Japan, and this location is an identifiable place such as a landmark, neighborhood, commercial venue, or cultural/historical site",
            node=None,  # Don't assign to node, just get the result
            sources=extra_source,
            additional_instruction="Look for evidence that: 1) this film was shot at this specific location in Tokyo, 2) this location is indeed in Tokyo, Japan (from your knowledge or evidence from the page), and 3) this location is clearly identifiable, instead of just rough descriptions that are not able to actually find (landmark, neighborhood, commercial venue, cultural site, exact locations, etc.). Allow reasonable variants in both the film title and location name formatting, as long as they refer to the same film and location."
        )
        
        # If fallback succeeded, update the node manually
        if result:
            tokyo_filming_node.score = 1.0
            tokyo_filming_node.status = "passed"


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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract basic film-location pairs first ------------------ #
    pairs_info = await evaluator.extract(
        prompt=prompt_extract_film_location_pairs(),
        template_class=ExtractedPairs,
        extraction_name="film_location_pairs"
    )

    # Limit to first 5 pairs if more are provided
    pairs = pairs_info.pairs[:5]

    # Pad with empty pairs if fewer than 5 are provided
    while len(pairs) < 5:
        pairs.append(FilmLocationPair())

    # -------- 3. Build verification tree ---------------------------------- #
    # Verify each location individually (non-critical to allow partial scoring)
    for i, pair in enumerate(pairs):
        await verify_single_location(evaluator, root, pair, i)

    # Verify uniqueness (critical - must pass)
    uniqueness_node = evaluator.add_leaf(
        id="uniqueness",
        desc="All film-location pairings mentioned in the answer are unique (no duplicate films or locations)",
        critical=True
    )
    
    await evaluator.verify(
        claim="All film-location pairings mentioned in this answer are unique - each film is different and each location is different, with no duplicates",
        node=uniqueness_node,
        additional_instruction="Check if there are any duplicate films or duplicate locations mentioned in the answer. If there are 1 or 0 film-location pairings, consider this automatically correct. Focus on whether each film title and each location name appears only once. But, remember, only check the locations and films mentioned as the final reported film-location pairings. It's okay if the answer contains more than 5 locations or films that are not mentioned as the film-location pairs, as long as the final pairings are unique. Allow for reasonable variants in names - if two entries refer to the same film or location with minor formatting differences, consider them duplicates."
    )

    # -------- 4. Return structured result --------------------------------- #
    return evaluator.get_summary()