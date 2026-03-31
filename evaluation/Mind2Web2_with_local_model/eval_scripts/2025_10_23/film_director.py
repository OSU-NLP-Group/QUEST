import asyncio
import logging
from typing import Dict, List, Optional

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "film_director"
TASK_DESCRIPTION = """
What is the name of the film released in 2024 that serves as a modern remake of an old classic and is starred by the actor who played the Green Goblin in the 2002 spiderman movie? I would like to learn more about the director of that film. Could you remind me who the director is and provide a link to their biography, along with the titles, release years, and IMDb pages of three prior films directed by the same director?
"""

# Ground truth information
FILM_NAME = "Nosferatu"
DIRECTOR_NAME = "Robert Eggers"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmDirectorInfo(BaseModel):
    film_name: Optional[str] = None
    director_name: Optional[str] = None
    director_film_urls: List[str] = Field(default_factory=list)


class DirectorBioInfo(BaseModel):
    director_bio_urls: List[str] = Field(default_factory=list)


class FilmInfo(BaseModel):
    title: Optional[str] = None
    release_year: Optional[str] = None
    imdb_url: Optional[str] = None


class DirectorFilms(BaseModel):
    films: List[FilmInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_film_director() -> str:
    return """
    Extract the following information from the answer:
    1. The name of the 2024 film remake mentioned in the answer
    2. The name of the director of this film
    3. All URLs that might provide evidence that this director directed this 2024 film

    If any information is missing, return null for that field.
    """


def prompt_extract_director_bio() -> str:
    return """
    Extract all URLs to the director's biography provided in the answer.
    If no biographical URLs are mentioned, return an empty list.
    """


def prompt_extract_director_films() -> str:
    return """
    Extract information about prior films directed by the same director mentioned in the answer. 
    For each film, extract:
    1. The title of the film
    2. The release year of the film
    3. The IMDb URL for the film

    Return the information for up to three films that have at least the title provided.
    If a film is missing any piece of information, still include the film but return null for the missing fields.
    If no prior films are mentioned, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_film_and_director_identification(
        evaluator: Evaluator,
        parent_node,
        info: FilmDirectorInfo,
) -> None:
    """
    Verify the 2024 film and its director identification.
    """
    # Film verification with existence check
    film_exists = evaluator.add_custom_node(
        result=(info.film_name is not None and info.film_name.strip() != ""),
        id="film_name_exists",
        desc="Check if film name was provided",
        parent=parent_node,
        critical=True
    )
    
    film_node = evaluator.add_leaf(
        id="film_correctness",
        desc=f"The 2024 film remake is correctly identified as {FILM_NAME}",
        parent=parent_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"{info.film_name} matches or is equivalent to the expected film {FILM_NAME}.",
        node=film_node,
        additional_instruction="Minor variations in the title are acceptable."
    )
    
    # Director verification with existence check
    director_exists = evaluator.add_custom_node(
        result=(info.director_name is not None and info.director_name.strip() != ""),
        id="director_name_exists",
        desc="Check if director name was provided",
        parent=parent_node,
        critical=True
    )
    
    director_node = evaluator.add_leaf(
        id="director_correctness",
        desc=f"The director is correctly identified as {DIRECTOR_NAME}",
        parent=parent_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"This director name {info.director_name} matches or is equivalent to the director {DIRECTOR_NAME}.",
        node=director_node,
        additional_instruction="Minor variations in the name are acceptable."
    )
    
    # Evidence verification with existence check
    evidence_exists = evaluator.add_custom_node(
        result=bool(info.director_film_urls),
        id="evidence_urls_exist",
        desc="Check if director-film evidence URLs were provided",
        parent=parent_node,
        critical=True
    )
    
    evidence_node = evaluator.add_leaf(
        id="director_film_evidence",
        desc=f"Provides verifiable URL evidence that {DIRECTOR_NAME} directed {FILM_NAME} (2024)",
        parent=parent_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"The webpage provides evidence that {DIRECTOR_NAME} directed {FILM_NAME} (2024)",
        node=evidence_node,
        sources=info.director_film_urls,
        additional_instruction=f"Verify that the URLs provide evidence that {DIRECTOR_NAME} directed {FILM_NAME} (2024) and that it is a remake."
    )


async def verify_director_bio(
        evaluator: Evaluator,
        parent_node,
        bio_info: DirectorBioInfo,
) -> None:
    """
    Verify the director's biography URLs.
    """
    # Bio verification with existence check
    bio_exists = evaluator.add_custom_node(
        result=bool(bio_info.director_bio_urls),
        id="bio_urls_exist",
        desc="Check if director biography URLs were provided",
        parent=parent_node,
        critical=True
    )
    
    bio_node = evaluator.add_leaf(
        id="director_bio",
        desc=f"Provides valid URLs to {DIRECTOR_NAME}'s biography that contain biographical information",
        parent=parent_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"This webpage shows biographical information about {DIRECTOR_NAME}.",
        node=bio_node,
        sources=bio_info.director_bio_urls,
        additional_instruction=f"Verify that this webpage contains biographical information about {DIRECTOR_NAME} (e.g., his life, career, or achievements)."
    )


async def verify_prior_films(
        evaluator: Evaluator,
        parent_node,
        films: DirectorFilms,
) -> None:
    """
    Verify information about three prior films directed by Robert Eggers.
    """
    prior_films_node = evaluator.add_parallel(
        id="prior_films",
        desc=f"Provides information about three prior films directed by {DIRECTOR_NAME}",
        parent=parent_node,
        critical=False,
    )
    
    # Ensure we have exactly 3 films to verify (pad with empty if needed)
    films_list = list(films.films)
    while len(films_list) < 3:
        films_list.append(FilmInfo())
    
    # Verify each film
    for i, film in enumerate(films_list[:3]):
        await verify_single_prior_film(
            evaluator=evaluator,
            parent_node=prior_films_node,
            film=film,
            film_index=i + 1,
        )


async def verify_single_prior_film(
        evaluator: Evaluator,
        parent_node,
        film: FilmInfo,
        film_index: int,
) -> None:
    """
    Verify information about a single prior film directed by Robert Eggers.
    """
    film_node = evaluator.add_parallel(
        id=f"prior_film_{film_index}",
        desc=f"Information about prior film #{film_index}: {film.title or 'Unknown'} ({film.release_year or 'Unknown'})",
        parent=parent_node,
        critical=False,
    )
    
    # Combined completeness check for all required fields
    film_complete = evaluator.add_custom_node(
        result=(
            film.title is not None and film.title.strip() != "" and
            film.release_year is not None and film.release_year.strip() != "" and
            film.imdb_url is not None and film.imdb_url.strip() != "" and
            ("imdb.com/title/" in film.imdb_url or "imdb.com/name/" in film.imdb_url)
        ),
        id=f"film_{film_index}_complete",
        desc=f"Check if all required information (title, year, valid IMDb URL) was provided for prior film #{film_index}",
        parent=film_node,
        critical=True
    )
    
    # Year validity check using LLM
    year_valid_node = evaluator.add_leaf(
        id=f"film_{film_index}_year_valid",
        desc=f"Release year for prior film #{film_index} is before 2024",
        parent=film_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The year {film.release_year} represents a year before 2024.",
        node=year_valid_node,
        additional_instruction="Check if this is a valid year before 2024. The year must be strictly less than 2024 (not equal to 2024). For ambiguous formats like 'early 2000s' or 'circa 2010', determine if the most likely year is before 2024."
    )
    
    # Verify film information with IMDb page
    verification_node = evaluator.add_leaf(
        id=f"film_{film_index}_verification",
        desc=f"The film title, year, and director information are supported by the IMDb page",
        parent=film_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"This IMDb page confirms that '{film.title}' was directed by {DIRECTOR_NAME} and was released in {film.release_year}.",
        node=verification_node,
        sources=film.imdb_url,
        additional_instruction=f"Verify that this IMDb page confirms '{film.title}' was directed by {DIRECTOR_NAME}. Also check if the release year matches {film.release_year}."
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
    Evaluate the answer to the film director task and return a structured result.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    
    # Extract structured info from the answer
    film_director_info = await evaluator.extract(
        prompt=prompt_extract_film_director(),
        template_class=FilmDirectorInfo,
        extraction_name="film_director_info"
    )
    
    bio_info = await evaluator.extract(
        prompt=prompt_extract_director_bio(),
        template_class=DirectorBioInfo,
        extraction_name="director_bio_info"
    )
    
    director_films = await evaluator.extract(
        prompt=prompt_extract_director_films(),
        template_class=DirectorFilms,
        extraction_name="director_films"
    )
    
    # First task: identify the film and director
    identify_director_node = evaluator.add_parallel(
        id="identify_director",
        desc="Correctly identify the 2024 film remake and its director",
        critical=False
    )
    
    await verify_film_and_director_identification(
        evaluator=evaluator,
        parent_node=identify_director_node,
        info=film_director_info
    )
    
    # Second task: director's films information
    director_films_node = evaluator.add_parallel(
        id="director_films_info",
        desc=f"Provides information about the director's biography and three prior films",
        critical=False
    )
    
    await verify_director_bio(
        evaluator=evaluator,
        parent_node=director_films_node,
        bio_info=bio_info
    )
    
    await verify_prior_films(
        evaluator=evaluator,
        parent_node=director_films_node,
        films=director_films
    )
    
    # Return structured result
    return evaluator.get_summary()