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
TASK_ID = "captain_america_starring"
TASK_DESCRIPTION = """
I'm interested in films released after Captain America: Civil War featuring at least two credited actors from its cast. First, provide the U.S. release date for Captain America: Civil War along with a cast page link (from any credible source clearly listing the cast). Then, identify 3 films released no later than 2020 featuring at least two actors who were credited in Captain America: Civil War. For each identified film, clearly state the film's title, the names of those actors, their character names in the identified film, the film's U.S. release date, a link to its Wikipedia page, and a link to a cast page clearly listing these actors.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CivilWarInfo(BaseModel):
    release_date: Optional[str] = None
    cast_page_url: Optional[str] = None

class ActorInfo(BaseModel):
    name: Optional[str] = None
    character: Optional[str] = None

class FilmInfo(BaseModel):
    title: Optional[str] = None
    actors: List[ActorInfo] = Field(default_factory=list)
    release_date: Optional[str] = None
    wikipedia_url: Optional[str] = None
    cast_page_url: Optional[str] = None

class FilmList(BaseModel):
    films: List[str] = Field(default_factory=list)

class ExtractedInfo(BaseModel):
    civil_war: Optional[CivilWarInfo] = None
    films: List[FilmInfo] = Field(default_factory=list)

class ProvLink(BaseModel):
    url: str
    description: Optional[str] = None

class ProvLinks(BaseModel):
    links: List[ProvLink] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_civil_war_info() -> str:
    return """
    Extract the following information about Captain America: Civil War from the answer:
    1. The U.S. release date of the film
    2. The URL of a cast page that lists the cast of the film

    If any information is missing, set the corresponding field to null.
    """

def prompt_extract_film_list() -> str:
    return """
    Extract the list of films mentioned in the answer that feature cast members from Captain America: Civil War.
    For each film, extract the title of the film.
    
    Return a JSON object with a single key "films" whose value is an array of film titles.
    
    If no films are mentioned, return an empty array. Extract ALL films mentioned in the answer, even if there are more than 3.
    """

def prompt_extract_film_details(film_title: str) -> str:
    return f"""
    Extract the following details about the film "{film_title}" from the answer:
    1. The full title of the film
    2. The actors from Captain America: Civil War who appear in this film and their character names in this film
    3. The U.S. release date of the film
    4. The URL to the film's Wikipedia page
    5. The URL to a cast page for the film
    
    For each actor, extract both their name and the character they play in this specific film.
    If any information is missing, set the corresponding field to null.
    """

def prompt_extract_urls_for_civil_war() -> str:
    return """
    Extract all URLs from the answer that might be related to information about Captain America: Civil War,
    particularly those that might contain information about its release date or cast.
    
    For each URL, provide a brief description of what information it likely contains based on context.
    """

# --------------------------------------------------------------------------- #
# Verification functions for Captain America: Civil War info                  #
# --------------------------------------------------------------------------- #
# Updated verify_civil_war_info function signature and implementation
async def verify_civil_war_info(
    evaluator: Evaluator,
    parent_node,
    civil_war_info: CivilWarInfo,
    civil_war_urls: ProvLinks,
) -> None:
    """Verify information about Captain America: Civil War."""
    
    civil_war_node = evaluator.add_parallel(
        id="civil_war_info",
        desc="Information about Captain America: Civil War is correct and substantiated",
        parent=parent_node,
        critical=False,
    )
    
    # Combined existence check for civil war info INCLUDING URLs
    info_exists_node = evaluator.add_custom_node(
        result=(civil_war_info.release_date is not None and 
                civil_war_info.cast_page_url is not None and
                bool(civil_war_urls.links)),
        id="civil_war_info_exists",
        desc="Check if Civil War release date, cast page URL, and verification URLs are provided",
        parent=civil_war_node,
        critical=True
    )
    
    # Verify release date
    release_date_node = evaluator.add_leaf(
        id="civil_war_release_date",
        desc="The U.S. release date for Captain America: Civil War is correct",
        parent=civil_war_node,
        critical=True,
    )
    
    # Verify release date against a ground truth
    claim = f"Captain America: Civil War was released in the U.S. on {civil_war_info.release_date}"
    
    # Extract just the URL strings from ProvLinks
    url_list = [link.url for link in civil_war_urls.links] if civil_war_urls.links else []
    
    await evaluator.verify(
        claim=claim,
        node=release_date_node,
        sources=url_list,
        additional_instruction="Verify that the release date mentioned matches the information on the webpage(s)."
    )
    
    # Verify cast page URL
    cast_page_node = evaluator.add_leaf(
        id="civil_war_cast_page",
        desc="A valid and substantive cast page URL for Captain America: Civil War is provided",
        parent=civil_war_node,
        critical=True,
    )
    
    cast_claim = "This webpage contains a cast list for the film Captain America: Civil War."
    await evaluator.verify(
        claim=cast_claim,
        node=cast_page_node,
        sources=civil_war_info.cast_page_url,
    )

# --------------------------------------------------------------------------- #
# Verification functions for subsequent films                                 #
# --------------------------------------------------------------------------- #
async def verify_film_info(
    evaluator: Evaluator,
    parent_node,
    film_info: FilmInfo,
    film_index: int,
    civil_war_cast_url: str,
) -> None:
    """Verify information about a film with Captain America: Civil War actors."""
    
    film_node = evaluator.add_parallel(
        id=f"film_{film_index}",
        desc=f"Film {film_index}: Information about '{film_info.title or 'Unnamed film'}' is correct and substantiated",
        parent=parent_node,
        critical=False,
    )
    
    # Pad actors list to ensure we have at least 2
    while len(film_info.actors) < 2:
        film_info.actors.append(ActorInfo())
    
    # Combined existence check for film info
    film_exists_node = evaluator.add_custom_node(
        result=(film_info.title is not None and 
                film_info.release_date is not None and
                film_info.wikipedia_url is not None and
                film_info.cast_page_url is not None and
                len([a for a in film_info.actors[:2] if a.name is not None]) >= 2),
        id=f"film_{film_index}_info_exists",
        desc=f"Check if film {film_index} has complete information (title, dates, URLs, and at least 2 actors)",
        parent=film_node,
        critical=True
    )
    
    # 1. Verify the film's release date (must be after Civil War and no later than 2020)
    # Create a parent node for the two-part release date verification
    release_date_parent = evaluator.add_parallel(
        id=f"film_{film_index}_release_date",
        desc=f"The film '{film_info.title}' was released after May 6, 2016 and no later than 2020",
        parent=film_node,
        critical=True,
    )
    
    # 1a. Verify the release date is correct for the film
    date_accuracy_node = evaluator.add_leaf(
        id=f"film_{film_index}_release_date_accuracy",
        desc=f"The film '{film_info.title}' was released on {film_info.release_date}",
        parent=release_date_parent,
        critical=True,
    )
    
    date_accuracy_claim = f"The film '{film_info.title}' was released in the U.S. on {film_info.release_date}"
    await evaluator.verify(
        claim=date_accuracy_claim,
        node=date_accuracy_node,
        sources=film_info.wikipedia_url,
        additional_instruction="Verify that the release date mentioned is correct for this film according to the Wikipedia page."
    )
    
    # 1b. Verify it's after Civil War and before/during 2020
    date_range_node = evaluator.add_leaf(
        id=f"film_{film_index}_release_date_range",
        desc=f"The date {film_info.release_date} is within the valid range",
        parent=release_date_parent,
        critical=True,
    )
    
    date_range_claim = f"The date {film_info.release_date} is after May 6, 2016 and no later than December 31, 2020"
    await evaluator.verify(
        claim=date_range_claim,
        node=date_range_node,
        sources=None,  # No source needed for date comparison
        additional_instruction="Perform a simple date comparison to verify the date falls within the specified range."
    )
    
    # 2. Verify Wikipedia URL
    wiki_node = evaluator.add_leaf(
        id=f"film_{film_index}_wikipedia",
        desc=f"A valid Wikipedia URL for the film '{film_info.title}' is provided",
        parent=film_node,
        critical=True,
    )
    
    wiki_claim = f"This webpage is a Wikipedia page about the film '{film_info.title}'"
    await evaluator.verify(
        claim=wiki_claim,
        node=wiki_node,
        sources=film_info.wikipedia_url,
    )
    
    # 3. Verify cast page URL
    cast_page_node = evaluator.add_leaf(
        id=f"film_{film_index}_cast_page",
        desc=f"A valid cast page URL for the film '{film_info.title}' is provided",
        parent=film_node,
        critical=True,
    )
    
    cast_claim = f"This webpage contains information about the cast of the film '{film_info.title}'"
    await evaluator.verify(
        claim=cast_claim,
        node=cast_page_node,
        sources=film_info.cast_page_url,
    )
    
    # 4. Verify that at least two actors from Civil War appear in this film
    actors_node = evaluator.add_parallel(
        id=f"film_{film_index}_actors",
        desc=f"At least two actors from Captain America: Civil War appear in '{film_info.title}'",
        parent=film_node,
        critical=True,
    )
    
    # Verify each of the first two actors
    for i, actor in enumerate(film_info.actors[:2]):
        await verify_actor_info(
            evaluator,
            actors_node,
            actor,
            i + 1,
            film_index,
            film_info.title,
            film_info.cast_page_url,
            civil_war_cast_url
        )

async def verify_actor_info(
    evaluator: Evaluator,
    parent_node,
    actor: ActorInfo,
    actor_index: int,
    film_index: int,
    film_title: str,
    film_cast_url: str,
    civil_war_cast_url: str,
) -> None:
    """Verify information about a single actor."""
    
    actor_node = evaluator.add_parallel(
        id=f"film_{film_index}_actor_{actor_index}",
        desc=f"Actor {actor_index}: {actor.name or 'Unnamed'} appeared in both Captain America: Civil War and '{film_title}'",
        parent=parent_node,
        critical=True,
    )
    
    # Combined existence check for actor info
    actor_exists_node = evaluator.add_custom_node(
        result=(actor.name is not None and actor.character is not None),
        id=f"film_{film_index}_actor_{actor_index}_exists",
        desc=f"Check if actor {actor_index} name and character are provided",
        parent=actor_node,
        critical=True
    )
    
    # Verify actor was in Civil War
    civil_war_actor_node = evaluator.add_leaf(
        id=f"film_{film_index}_actor_{actor_index}_in_civil_war",
        desc=f"Actor {actor.name or 'Unnamed'} appeared in Captain America: Civil War",
        parent=actor_node,
        critical=True,
    )
    
    civil_war_actor_claim = f"Actor {actor.name} appeared in Captain America: Civil War"
    await evaluator.verify(
        claim=civil_war_actor_claim,
        node=civil_war_actor_node,
        sources=civil_war_cast_url,
    )
    
    # Verify actor is in this film
    film_actor_node = evaluator.add_leaf(
        id=f"film_{film_index}_actor_{actor_index}_in_film",
        desc=f"Actor {actor.name or 'Unnamed'} appeared in '{film_title}'",
        parent=actor_node,
        critical=True,
    )
    
    film_actor_claim = f"Actor {actor.name} appeared in the film '{film_title}'"
    await evaluator.verify(
        claim=film_actor_claim,
        node=film_actor_node,
        sources=film_cast_url,
    )
    
    # Verify character name
    character_node = evaluator.add_leaf(
        id=f"film_{film_index}_actor_{actor_index}_character",
        desc=f"Actor {actor.name or 'Unnamed'} played the character '{actor.character or 'Unknown'}' in '{film_title}'",
        parent=actor_node,
        critical=True
    )
    
    character_claim = f"Actor {actor.name} played the character '{actor.character}' in the film '{film_title}'"
    await evaluator.verify(
        claim=character_claim,
        node=character_node,
        sources=film_cast_url,
    )

# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
# Updated evaluate_answer function
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
    Evaluate an answer to the Captain America film task and return structured results.
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
        default_model=model if model else JUDGE_MODEL
    )
    
    # Extract info about Captain America: Civil War
    civil_war_info = await evaluator.extract(
        prompt=prompt_extract_civil_war_info(),
        template_class=CivilWarInfo,
        extraction_name="civil_war_info"
    )
    
    # Extract URLs that might contain Civil War release date info
    civil_war_urls = await evaluator.extract(
        prompt=prompt_extract_urls_for_civil_war(),
        template_class=ProvLinks,
        extraction_name="civil_war_urls"
    )
    
    # Extract list of films mentioned in the answer
    film_list_result = await evaluator.extract(
        prompt=prompt_extract_film_list(),
        template_class=FilmList,
        extraction_name="film_list"
    )
    film_list = film_list_result.films
    
    # Extract detailed info for each film (up to 3)
    films_info = []
    for i, film_title in enumerate(film_list[:3]):
        film_details = await evaluator.extract(
            prompt=prompt_extract_film_details(film_title),
            template_class=FilmInfo,
            extraction_name=f"film_{i+1}_details"
        )
        films_info.append(film_details)
    
    # Pad to ensure we have 3 films
    while len(films_info) < 3:
        films_info.append(FilmInfo())
    
    # Create overall extracted info structure
    extracted_info = ExtractedInfo(
        civil_war=civil_war_info,
        films=films_info
    )
    
    # Add all extracted info to evaluator for summary
    evaluator.add_custom_info(
        {
            "extracted_info": extracted_info.dict(),
            "civil_war_urls": civil_war_urls.dict()  # Add the URLs to the output
        },
        "extracted_data"
    )
    
    # 1. Verify Captain America: Civil War info
    await verify_civil_war_info(evaluator, root, civil_war_info, civil_war_urls)
    
    # 2. Verify the films
    films_node = evaluator.add_parallel(
        id="films",
        desc="Verification of 3 films released after Captain America: Civil War featuring at least two of its cast members",
        parent=root,
        critical=False,
    )
    
    # Get Civil War cast URL for actor verification
    civil_war_cast_url = civil_war_info.cast_page_url if civil_war_info else None
    
    # Verify each film
    for i, film_info in enumerate(films_info):
        await verify_film_info(
            evaluator,
            films_node, 
            film_info, 
            i+1, 
            civil_war_cast_url,
        )
    
    # Return structured result
    return evaluator.get_summary()
