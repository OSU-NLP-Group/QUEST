import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "2024_film_premiere"
TASK_DESCRIPTION = """
Find 3 movies that premiered in 2024 and had matching theatrical release dates in both mainland China and the United States. For each movie, clearly provide the title, the shared theatrical release date, and credible source links (such as IMDb or official news articles) explicitly confirming these theatrical release dates. Consider only official theatrical release dates—exclude film festival premieres, limited screenings, digital releases, special events, or any other non-standard releases.
"""

JUDGE_MODEL = "o4-mini"
EXPECTED_MOVIE_COUNT = 3

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MovieTitles(BaseModel):
    """Container for movie titles extracted from the answer."""
    titles: List[str] = Field(default_factory=list)


class MovieDetails(BaseModel):
    """Details for a single movie."""
    release_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MovieEntry(BaseModel):
    """Combined movie information."""
    title: Optional[str] = None
    release_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_movie_titles() -> str:
    return """
    Extract the titles of the movies mentioned in the answer that were released in 2024 with matching theatrical release dates in both mainland China and the United States.

    Return just the titles of the movies in the order they appear in the answer.
    If no movies are mentioned, return an empty list.

    Include all titles mentioned in the answer, even if there are more than 3.
    """


def prompt_extract_movie_details(title: str) -> str:
    return f"""
    Extract the following details about the movie titled "{title}" from the answer:

    1. The shared theatrical release date mentioned for this movie (when it was released in both China and the US)
    2. All source URLs or links mentioned in relation to this movie

    If the release date is not mentioned, set it to null.
    If no URLs are mentioned for this movie, return an empty list for urls.
    """


# --------------------------------------------------------------------------- #
# Movie verification logic                                                    #
# --------------------------------------------------------------------------- #
async def verify_movie(
    evaluator: Evaluator,
    parent_node,
    movie_entry: MovieEntry,
    movie_index: int,
) -> None:
    """
    Verify a single movie's information according to the sequential rubric tree.
    """
    
    # Create a sequential node for this movie
    movie_node = evaluator.add_sequential(
        id=f"movie_{movie_index}",
        desc=f"Movie {movie_index}: {movie_entry.title if movie_entry.title else 'No title provided'}",
        parent=parent_node,
        critical=False
    )

    # 1. Verify movie title exists
    title_exists_node = evaluator.add_custom_node(
        result=bool(movie_entry.title and movie_entry.title.strip()),
        id=f"movie_{movie_index}_title",
        desc=f"Verify that movie {movie_index} has a provided title",
        parent=movie_node,
        critical=True
    )

    # 2. Verify shared release date exists
    date_exists_node = evaluator.add_custom_node(
        result=bool(movie_entry.release_date and movie_entry.release_date.strip()),
        id=f"movie_{movie_index}_shared_release_date",
        desc=f"Verify that '{movie_entry.title if movie_entry.title else 'movie'}' has a provided shared theatrical release date",
        parent=movie_node,
        critical=True
    )

    # 3. Verify source URLs exist
    urls_exist_node = evaluator.add_custom_node(
        result=bool(movie_entry.urls),
        id=f"movie_{movie_index}_urls_exist",
        desc=f"Verify that source URLs are provided for '{movie_entry.title if movie_entry.title else 'movie'}'",
        parent=movie_node,
        critical=True
    )

    # 4. Verify the provided release date is in 2024
    year_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_2024_release",
        desc=f"Verify that the provided release date '{movie_entry.release_date}' for '{movie_entry.title if movie_entry.title else 'movie'}' is in 2024",
        parent=movie_node,
        critical=True
    )

    # Safe check for 2024
    if movie_entry.release_date and "2024" in movie_entry.release_date:
        year_node.score = 1.0
        year_node.status = "passed"
    else:
        year_claim = f"The date '{movie_entry.release_date}' is a date in the year 2024."
        await evaluator.verify(
            claim=year_claim,
            node=year_node
        )

    # 5. Verify US theatrical release
    us_release_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_us_release",
        desc=f"Verify that '{movie_entry.title if movie_entry.title else 'movie'}' had an official theatrical release in the US",
        parent=movie_node,
        critical=True
    )


    # 6. Verify China theatrical release
    china_release_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_china_release",
        desc=f"Verify that '{movie_entry.title if movie_entry.title else 'movie'}' had an official theatrical release in mainland China",
        parent=movie_node,
        critical=True
    )

    # Safe claim construction
    china_claim = f"'{movie_entry.title if movie_entry.title else 'the movie'}' had an official theatrical release (not a film festival premiere, limited screening, digital release, special event, or any other non-standard release) in mainland China on {movie_entry.release_date}."
    await evaluator.verify(
        claim=china_claim,
        node=china_release_node,
        sources=movie_entry.urls,
        additional_instruction="Verify that this is an official theatrical release date, NOT a film festival premiere, limited screening, digital release, special event, or any other non-standard release."
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
    Evaluate an answer for the movie release dates task.
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

    # -------- 2. Extract structured info from the answer ---------------- #
    # Extract movie titles
    parsed_titles = await evaluator.extract(
        prompt=prompt_extract_movie_titles(),
        template_class=MovieTitles,
        extraction_name="movie_titles"
    )

    # -------- 3. Build movie entries with padding ----------------------- #
    movie_entries = []
    
    # Extract details for each title found
    for i, title in enumerate(parsed_titles.titles[:EXPECTED_MOVIE_COUNT]):
        movie_details = await evaluator.extract(
            prompt=prompt_extract_movie_details(title),
            template_class=MovieDetails,
            extraction_name=f"movie_{i+1}_details"
        )
        
        movie_entries.append(MovieEntry(
            title=title,
            release_date=movie_details.release_date,
            urls=movie_details.urls
        ))
    
    # Pad with empty entries if needed
    while len(movie_entries) < EXPECTED_MOVIE_COUNT:
        movie_entries.append(MovieEntry())

    # -------- 4. Verify all movies using unified logic ------------------ #
    for i, movie_entry in enumerate(movie_entries):
        await verify_movie(
            evaluator=evaluator,
            parent_node=root,
            movie_entry=movie_entry,
            movie_index=i + 1
        )

    # -------- 5. Add custom info for extracted details ----------------- #
    extracted_details = []
    for entry in movie_entries:
        if entry.title:
            extracted_details.append({
                "title": entry.title,
                "release_date": entry.release_date,
                "urls": entry.urls
            })
    
    evaluator.add_custom_info(
        {"extracted_movie_details": extracted_details},
        "movie_details",
        "extracted_movie_information"
    )

    # -------- 6. Return structured result ------------------------------- #
    return evaluator.get_summary()