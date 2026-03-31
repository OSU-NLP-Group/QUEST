import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.llm_client.base_client import LLMClient
from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "animation_movies"
TASK_DESCRIPTION = """
Find three animation movies with an average rating above 8.0 on IMDB or above 70% on Rotten Tomatoes Tomatometer. For each movie, list their directors, release dates, and main plot points. Additionally, for each movie, list their IMDB rating and Rotten Tomatoes Tomatometer, and include links to one positive and one negative review on IMDB.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MovieTitles(BaseModel):
    titles: List[str] = Field(default_factory=list)

class MovieDetails(BaseModel):
    director: Optional[str] = None
    release_date: Optional[str] = None
    plot_points: Optional[List[str]] = Field(default_factory=list)
    imdb_rating: Optional[str] = None
    rotten_tomatoes_rating: Optional[str] = None
    positive_review_url: Optional[str] = None
    negative_review_url: Optional[str] = None

class MovieSourceURLs(BaseModel):
    source_urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_movie_titles() -> str:
    return """
    Extract the titles of all animation movies mentioned in the answer. 
    Return them as a list of strings in the order they appear in the answer.
    If no movies are mentioned, return an empty list.
    """

def prompt_extract_movie_details(movie_title: str) -> str:
    return f"""
    Extract detailed information about the movie '{movie_title}' from the answer. Extract:
    
    1. The director(s) - extract as a single string even if multiple directors.
    2. The release date - extract exactly as mentioned.
    3. The main plot points - extract as a list of strings.
    4. The IMDB rating - extract exactly as mentioned.
    5. The Rotten Tomatoes rating - extract exactly as mentioned.
    6. URL for a positive IMDB review, if present.
    7. URL for a negative IMDB review, if present.
    
    For each field:
    - If the information is not explicitly mentioned in the answer for this specific movie, set it to null.
    - Do not invent or infer information not explicitly stated in the answer.
    """

def prompt_extract_movie_source_urls(movie_title: str) -> str:
    return f"""
    Extract all URLs (links) mentioned in the answer that are specifically related to the movie '{movie_title}'. 
    These could include:
    1. Links to IMDB pages for this movie
    2. Links to Rotten Tomatoes pages for this movie
    3. Links to reviews for this movie
    4. Links to any other information sites about this specific movie

    Return a list of all unique URLs found in the answer that are related to '{movie_title}'. 
    Do not include URLs that are for other movies or general movie information.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_movie(
    evaluator: Evaluator,
    parent_node,
    movie_title: str,
    movie_details: MovieDetails,
    source_urls: List[str],
    movie_index: int
) -> None:
    """
    Verify all aspects of a single movie with the new hierarchical structure.
    """
    # Create parent node for this movie
    movie_node = evaluator.add_sequential(
        id=f"movie_{movie_index}",
        desc=f"Movie #{movie_index}: '{movie_title}'",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )
    
    # 1. CRITICAL: Check essential ratings and source urls exist
    essential_info_provided = evaluator.add_custom_node(
        result=(
            movie_details.imdb_rating is not None and movie_details.imdb_rating.strip() != "" and
            movie_details.rotten_tomatoes_rating is not None and movie_details.rotten_tomatoes_rating.strip() != "" and
            source_urls
        ),
        id=f"movie_{movie_index}_essential_ratings_exist",
        desc=f"Check if both IMDB and Rotten Tomatoes ratings are provided for '{movie_title}'",
        parent=movie_node,
        critical=True
    )

    # 2. CRITICAL: Verify the movie is animation and has the required rating
    constraint_check = evaluator.add_parallel(
        id=f"movie_{movie_index}_constraint_check",
        desc=f"Check if the identified movie meet the constraints",
        parent=movie_node,
        critical=False
    )
    
    # 2a. CRITICAL: Verify the movie is animation
    animation_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_is_animation",
        desc=f"Verify that '{movie_title}' is an animation movie",
        parent=constraint_check,
        critical=True
    )
    
    claim = f"The movie '{movie_title}' is an animation movie."
    await evaluator.verify(
        claim=claim,
        node=animation_node,
        sources=source_urls if source_urls else None,
    )
    
    # 2b. CRITICAL: Verify rating threshold (IMDB > 8.0 OR RT > 70%)
    rating_threshold_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_rating_threshold_valid",
        desc=f"Verify '{movie_title}' has IMDB > 8.0 OR RT > 70%",
        parent=constraint_check,
        critical=True
    )
    
    claim = f"The movie '{movie_title}' has either an IMDB rating of {movie_details.imdb_rating} which is above 8.0, OR a Rotten Tomatoes rating of {movie_details.rotten_tomatoes_rating} which is above 70%."
    await evaluator.verify(
        claim=claim,
        node=rating_threshold_node,
        sources=source_urls if source_urls else None,
        additional_instruction="Verify that AT LEAST ONE of the following is true: (1) the IMDB rating shown on the webpage is above 8.0, OR (2) the Rotten Tomatoes rating shown on the webpage is above 70%. The movie passes if either condition is met."
    )
    
    # 3. NON-CRITICAL: additional information
    additional_check = evaluator.add_parallel(
        id=f"movie_{movie_index}_additional_info",
        desc=f"Check if all the required infomation is provided accurately",
        parent=movie_node,
        critical=False
    )

    # 3a. NON-CRITICAL: Both ratings accuracy
    ratings_section = evaluator.add_parallel(
        id=f"movie_{movie_index}_ratings_section",
        desc=f"Rating accuracy for '{movie_title}'",
        parent=additional_check,
        critical=False
    )
    
    # 3a-1. CRITICAL: Both ratings provided
    ratings_provided = evaluator.add_custom_node(
        result=(
            movie_details.imdb_rating is not None and movie_details.imdb_rating.strip() != "" and
            movie_details.rotten_tomatoes_rating is not None and movie_details.rotten_tomatoes_rating.strip() != ""
        ),
        id=f"movie_{movie_index}_ratings_provided",
        desc=f"Check if both IMDB and RT ratings are provided for '{movie_title}'",
        parent=ratings_section,
        critical=True
    )
    
    # 3a-2. CRITICAL: IMDB rating accuracy
    imdb_accuracy_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_imdb_rating_accuracy",
        desc=f"Verify IMDB rating accuracy for '{movie_title}'",
        parent=ratings_section,
        critical=True
    )
    
    claim = f"The movie '{movie_title}' has an IMDB rating of {movie_details.imdb_rating}."
    await evaluator.verify(
        claim=claim,
        node=imdb_accuracy_node,
        sources=source_urls if source_urls else None,
        additional_instruction="Verify that the claimed IMDB rating matches what is shown on the webpage."
    )
    
    # 3a-3. CRITICAL: RT rating accuracy
    rt_accuracy_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_rt_rating_accuracy",
        desc=f"Verify Rotten Tomatoes rating accuracy for '{movie_title}'",
        parent=ratings_section,
        critical=True
    )
    
    claim = f"The movie '{movie_title}' has a Rotten Tomatoes rating of {movie_details.rotten_tomatoes_rating}."
    await evaluator.verify(
        claim=claim,
        node=rt_accuracy_node,
        sources=source_urls if source_urls else None,
        additional_instruction="Verify that the claimed Rotten Tomatoes rating matches what is shown on the webpage."
    )

    # 3b. NON-CRITICAL: Director information
    director_parent_node = evaluator.add_parallel(
        id=f"movie_{movie_index}_director_section",
        desc=f"Director information for '{movie_title}'",
        parent=additional_check,
        critical=False
    )
    
    # 3b-1. CRITICAL: Director provided
    director_provided = evaluator.add_custom_node(
        result=movie_details.director is not None and movie_details.director.strip() != "",
        id=f"movie_{movie_index}_director_provided",
        desc=f"Check if director information is provided for '{movie_title}'",
        parent=director_parent_node,
        critical=True
    )
    
    # 3b-2. CRITICAL: Director accuracy
    director_accuracy_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_director_accuracy",
        desc=f"Verify director accuracy for '{movie_title}'",
        parent=director_parent_node,
        critical=True
    )
    
    claim = f"The movie '{movie_title}' was directed by {movie_details.director}."
    await evaluator.verify(
        claim=claim,
        node=director_accuracy_node,
        sources=source_urls if source_urls else None,
    )
    
    # 3c. NON-CRITICAL: Release date information
    release_parent_node = evaluator.add_parallel(
        id=f"movie_{movie_index}_release_section",
        desc=f"Release date information for '{movie_title}'",
        parent=additional_check,
        critical=False
    )
    
    # 3c-1. CRITICAL: Release date provided
    release_provided = evaluator.add_custom_node(
        result=movie_details.release_date is not None and movie_details.release_date.strip() != "",
        id=f"movie_{movie_index}_release_provided",
        desc=f"Check if release date is provided for '{movie_title}'",
        parent=release_parent_node,
        critical=True
    )
    
    # 3c-2. CRITICAL: Release date accuracy
    release_accuracy_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_release_accuracy",
        desc=f"Verify release date accuracy for '{movie_title}'",
        parent=release_parent_node,
        critical=True
    )
    
    claim = f"The movie '{movie_title}' was released on/in {movie_details.release_date}."
    await evaluator.verify(
        claim=claim,
        node=release_accuracy_node,
        sources=source_urls if source_urls else None,
    )
    
    # 3d. NON-CRITICAL: Plot points information
    plot_parent_node = evaluator.add_parallel(
        id=f"movie_{movie_index}_plot_section",
        desc=f"Plot points information for '{movie_title}'",
        parent=additional_check,
        critical=False
    )
    
    # 3d-1. CRITICAL: Plot points provided
    plot_provided = evaluator.add_custom_node(
        result=movie_details.plot_points is not None and len(movie_details.plot_points) > 0,
        id=f"movie_{movie_index}_plot_provided",
        desc=f"Check if plot points are provided for '{movie_title}'",
        parent=plot_parent_node,
        critical=True
    )
    
    # 3d-2. CRITICAL: Plot points accuracy
    plot_accuracy_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_plot_accuracy",
        desc=f"Verify plot points accuracy for '{movie_title}'",
        parent=plot_parent_node,
        critical=True
    )
    
    plot_points_text = " ".join([f"({i+1}) {point}" for i, point in enumerate(movie_details.plot_points)])
    claim = f"The main plot points of the movie '{movie_title}' include: {plot_points_text}"
    
    await evaluator.verify(
        claim=claim,
        node=plot_accuracy_node,
        sources=source_urls if source_urls else None,
    )
    
    # 3e. NON-CRITICAL: Positive IMDB review
    pos_review_parent_node = evaluator.add_parallel(
        id=f"movie_{movie_index}_pos_review_section",
        desc=f"Positive IMDB review for '{movie_title}'",
        parent=additional_check,
        critical=False
    )
    
    # 3e-1. CRITICAL: Positive review URL provided
    pos_review_provided = evaluator.add_custom_node(
        result=(
            movie_details.positive_review_url is not None and 
            movie_details.positive_review_url.strip() != "" and
            "imdb.com" in movie_details.positive_review_url.lower()
        ),
        id=f"movie_{movie_index}_pos_review_provided",
        desc=f"Check if positive IMDB review URL is provided for '{movie_title}'",
        parent=pos_review_parent_node,
        critical=True
    )
    
    # 3e-2. CRITICAL: Positive review validity
    pos_review_validity_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_pos_review_valid",
        desc=f"Verify positive IMDB review validity for '{movie_title}'",
        parent=pos_review_parent_node,
        critical=True
    )
    
    claim = f"The webpage contains a positive review for the movie '{movie_title}' on IMDB."
    await evaluator.verify(
        claim=claim,
        node=pos_review_validity_node,
        sources=movie_details.positive_review_url,
        additional_instruction="Check if the linked page actually contains a positive review for the specific movie. A positive review would generally rate the movie favorably or have positive comments about it."
    )
    
    # 3f. NON-CRITICAL: Negative IMDB review
    neg_review_parent_node = evaluator.add_parallel(
        id=f"movie_{movie_index}_neg_review_section",
        desc=f"Negative IMDB review for '{movie_title}'",
        parent=additional_check,
        critical=False
    )
    
    # 3f-1. CRITICAL: Negative review URL provided
    neg_review_provided = evaluator.add_custom_node(
        result=(
            movie_details.negative_review_url is not None and 
            movie_details.negative_review_url.strip() != "" and
            "imdb.com" in movie_details.negative_review_url.lower()
        ),
        id=f"movie_{movie_index}_neg_review_provided",
        desc=f"Check if negative IMDB review URL is provided for '{movie_title}'",
        parent=neg_review_parent_node,
        critical=True
    )
    
    # 3f-2. CRITICAL: Negative review validity
    neg_review_validity_node = evaluator.add_leaf(
        id=f"movie_{movie_index}_neg_review_valid",
        desc=f"Verify negative IMDB review validity for '{movie_title}'",
        parent=neg_review_parent_node,
        critical=True
    )
    
    claim = f"The webpage contains a negative review for the movie '{movie_title}' on IMDB."
    await evaluator.verify(
        claim=claim,
        node=neg_review_validity_node,
        sources=movie_details.negative_review_url,
        additional_instruction="Check if the linked page actually contains a negative review for the specific movie. A negative review would generally rate the movie unfavorably or have critical comments about it."
    )

# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
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

    # -------- 2. Extract movies ------------------------------------------ #
    # First extract all movie titles
    movie_titles = await evaluator.extract(
        prompt=prompt_extract_movie_titles(),
        template_class=MovieTitles,
        extraction_name="movie_titles"
    )
    
    # -------- 3. Extract details and verify movies ----------------------- #
    # Pad to ensure we have exactly 3 movies to verify
    titles_to_verify = movie_titles.titles[:3]
    while len(titles_to_verify) < 3:
        titles_to_verify.append(f"Missing Movie #{len(titles_to_verify) + 1}")
    
    # Process each movie
    for i, title in enumerate(titles_to_verify):
        if title.startswith("Missing Movie"):
            # Create empty details for missing movie
            details = MovieDetails()
            source_urls = []
        else:
            # Extract details for real movie
            details = await evaluator.extract(
                prompt=prompt_extract_movie_details(title),
                template_class=MovieDetails,
                extraction_name=f"movie_{i+1}_details"
            )
            
            # Extract source URLs specific to this movie
            movie_source_urls = await evaluator.extract(
                prompt=prompt_extract_movie_source_urls(title),
                template_class=MovieSourceURLs,
                extraction_name=f"movie_{i+1}_source_urls"
            )
            source_urls = movie_source_urls.source_urls
        
        # Verify the movie
        await verify_movie(
            evaluator,
            root,
            title,
            details,
            source_urls,
            i + 1
        )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()