import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "film_series_2024_quarters"
TASK_DESCRIPTION = (
    "I'm organizing a film appreciation series for a community college course studying contemporary American cinema. "
    "I need to select exactly 4 theatrical films released in the United States during 2024, ensuring genre and audience diversity. "
    "The selection must include:\n\n"
    "1. One Action film rated PG-13 that was released between January 1 and March 31, 2024, with a runtime between 90 and 170 minutes.\n\n"
    "2. One Drama film rated R that was released between April 1 and June 30, 2024, with a runtime between 90 and 140 minutes.\n\n"
    "3. One Comedy film rated PG-13 that was released between July 1 and September 30, 2024, with a runtime between 85 and 130 minutes.\n\n"
    "4. One Animation film rated PG that was released between October 1 and December 31, 2024, with a runtime between 80 and 120 minutes.\n\n"
    "For each film, please provide: the film's exact title, its theatrical release date in the United States, its MPAA rating, its primary genre classification, "
    "its runtime in minutes, and at least one reference URL from a reliable source (such as Box Office Mojo, IMDb, The Numbers, or Rotten Tomatoes) that confirms these details."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmItem(BaseModel):
    title: Optional[str] = None
    us_release_date: Optional[str] = None
    mpaa_rating: Optional[str] = None
    primary_genre: Optional[str] = None
    runtime_minutes: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class FilmSelectionExtraction(BaseModel):
    films: List[FilmItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_films_main() -> str:
    return """
    Extract all films presented in the answer. For each film entry, extract the following fields exactly as stated in the answer (do not invent values):
    - title: The exact film title string.
    - us_release_date: The film's theatrical release date in the United States as stated (if the answer mentions multiple dates, pick the one explicitly marked as U.S. theatrical release or Domestic release; otherwise provide whatever U.S. release date is stated).
    - mpaa_rating: The U.S. MPAA rating string (e.g., PG-13, R, PG).
    - primary_genre: The primary or main genre classification as stated (e.g., Action, Drama, Comedy, Animation).
    - runtime_minutes: The runtime expressed in minutes if available; if the answer provides a format like '2h 10m', extract exactly that text (do not convert).
    - reference_urls: A list of all URLs explicitly provided for this film that could confirm details. Include any URLs shown, especially from Box Office Mojo, IMDb, The Numbers, or Rotten Tomatoes. If no URLs are given for a film, return an empty list.

    Return a JSON object with a 'films' array. Preserve the order of films as they appear in the answer. If any field is missing in the answer for a film, set it to null (or empty list for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
ALLOWED_SOURCE_DOMAINS = {"boxofficemojo.com", "imdb.com", "the-numbers.com", "thenumbers.com", "rottentomatoes.com"}


def _domain_is_allowed(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        if not netloc:
            return False
        for d in ALLOWED_SOURCE_DOMAINS:
            if netloc.endswith(d):
                return True
        return False
    except Exception:
        return False


def _filter_allowed_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu in seen:
            continue
        seen.add(uu)
        out.append(uu)
    return out


def _quarter_constraints(idx: int) -> Dict[str, Any]:
    # idx is 1-based index for film_1 .. film_4
    if idx == 1:
        return {
            "label": "Action film rated PG-13 released in Q1 2024 (January 1 - March 31) with runtime 90-170 minutes",
            "required_genre": "Action",
            "required_rating": "PG-13",
            "date_range": ("January 1, 2024", "March 31, 2024"),
            "runtime_min": 90,
            "runtime_max": 170,
        }
    if idx == 2:
        return {
            "label": "Drama film rated R released in Q2 2024 (April 1 - June 30) with runtime 90-140 minutes",
            "required_genre": "Drama",
            "required_rating": "R",
            "date_range": ("April 1, 2024", "June 30, 2024"),
            "runtime_min": 90,
            "runtime_max": 140,
        }
    if idx == 3:
        return {
            "label": "Comedy film rated PG-13 released in Q3 2024 (July 1 - September 30) with runtime 85-130 minutes",
            "required_genre": "Comedy",
            "required_rating": "PG-13",
            "date_range": ("July 1, 2024", "September 30, 2024"),
            "runtime_min": 85,
            "runtime_max": 130,
        }
    # idx == 4
    return {
        "label": "Animation film rated PG released in Q4 2024 (October 1 - December 31) with runtime 80-120 minutes",
        "required_genre": "Animation",
        "required_rating": "PG",
        "date_range": ("October 1, 2024", "December 31, 2024"),
        "runtime_min": 80,
        "runtime_max": 120,
    }


def _film_node_description(idx: int) -> str:
    return _quarter_constraints(idx)["label"]


# --------------------------------------------------------------------------- #
# Verification for a single film                                              #
# --------------------------------------------------------------------------- #
async def verify_single_film(
    evaluator: Evaluator,
    parent_node,
    film: FilmItem,
    film_idx: int
) -> None:
    """
    Build verification subtree for a single film according to rubric.
    film_idx is 1-based and maps to Q1/Q2/Q3/Q4 constraint sets.
    """
    constraints = _quarter_constraints(film_idx)
    start_date, end_date = constraints["date_range"]
    req_rating = constraints["required_rating"]
    req_genre = constraints["required_genre"]
    rmin, rmax = constraints["runtime_min"], constraints["runtime_max"]

    # Parent node for this film (parallel aggregation, non-critical)
    film_node = evaluator.add_parallel(
        id=f"film_{film_idx}",
        desc=_film_node_description(film_idx),
        parent=parent_node,
        critical=False
    )

    # Normalize/collect allowed URLs from provided references (dedup, not filtering domain yet).
    refs_all = _filter_allowed_urls(film.reference_urls)
    refs_allowed = [u for u in refs_all if _domain_is_allowed(u)]

    # Leaf: film_X_title (critical) - existence of title
    evaluator.add_custom_node(
        result=bool(film.title and film.title.strip()),
        id=f"film_{film_idx}_title",
        desc="Provide the exact film title",
        parent=film_node,
        critical=True
    )

    # Leaf: film_X_reference (critical) - at least one allowed reference URL present
    evaluator.add_custom_node(
        result=(len(refs_allowed) > 0),
        id=f"film_{film_idx}_reference",
        desc="Provide at least one reference URL from a reliable source (Box Office Mojo, IMDb, The Numbers, or Rotten Tomatoes) confirming the film's details",
        parent=film_node,
        critical=True
    )

    # Leaf: film_X_release_date (critical) - verify release date constraints by URLs
    node_release = evaluator.add_leaf(
        id=f"film_{film_idx}_release_date",
        desc=f"Film was released theatrically in the United States between {start_date} and {end_date}",
        parent=film_node,
        critical=True
    )
    title_for_claim = film.title or "the film"
    release_claim = (
        f"The film titled '{title_for_claim}' had a theatrical release in the United States between {start_date} and {end_date} (inclusive) in 2024."
    )
    await evaluator.verify(
        claim=release_claim,
        node=node_release,
        sources=refs_allowed,
        additional_instruction=(
            "Only count a U.S. theatrical release (or 'Domestic release'). "
            "On Box Office Mojo, the 'Domestic Release' is the U.S. theatrical release. "
            "On IMDb, check 'Release date' for United States and ensure it's a theatrical release. "
            "If the URL is not from the allowed domains or the page does not support the claim, mark as not supported."
        )
    )

    # Leaf: film_X_rating (critical) - verify MPAA rating
    node_rating = evaluator.add_leaf(
        id=f"film_{film_idx}_rating",
        desc=f"Film has MPAA rating of {req_rating}",
        parent=film_node,
        critical=True
    )
    rating_claim = f"The MPAA rating (U.S. certificate) of the film titled '{title_for_claim}' is exactly '{req_rating}'."
    await evaluator.verify(
        claim=rating_claim,
        node=node_rating,
        sources=refs_allowed,
        additional_instruction=(
            "Verify the U.S. rating (MPAA/MPA), sometimes called 'Certificate' on IMDb. "
            "Accept minor variations in formatting (e.g., 'PG-13' with/without hyphen), but the rating must be equivalent to the requested value."
        )
    )

    # Leaf: film_X_genre (critical) - verify primary genre
    node_genre = evaluator.add_leaf(
        id=f"film_{film_idx}_genre",
        desc=f"Film's primary genre is {req_genre}",
        parent=film_node,
        critical=True
    )
    genre_claim = f"The film titled '{title_for_claim}' has primary (main) genre '{req_genre}'."
    await evaluator.verify(
        claim=genre_claim,
        node=node_genre,
        sources=refs_allowed,
        additional_instruction=(
            "If multiple genres are listed, treat the 'primary' genre as the one explicitly labeled as primary or, if unspecified, the first/main genre shown by the site. "
            "If the page only lists a set of genres without indicating primacy and the requested genre does not plausibly appear as the main classification, mark as not supported."
        )
    )

    # Leaf: film_X_runtime (critical) - verify runtime falls within range
    node_runtime = evaluator.add_leaf(
        id=f"film_{film_idx}_runtime",
        desc=f"Film's runtime is between {rmin} and {rmax} minutes",
        parent=film_node,
        critical=True
    )
    runtime_claim = (
        f"The film titled '{title_for_claim}' has a runtime between {rmin} and {rmax} minutes (inclusive)."
    )
    await evaluator.verify(
        claim=runtime_claim,
        node=node_runtime,
        sources=refs_allowed,
        additional_instruction=(
            "Check the runtime on the page. If runtime is presented as 'xh ym', convert conceptually to minutes for comparison. "
            "Accept reasonable rounding (e.g., 126 vs. 127 minutes). The runtime must fall within the specified inclusive range."
        )
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
    Evaluate an answer for the 2024 U.S. theatrical films selection by quarter with genre/rating/runtime constraints.
    """
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
        default_model=model
    )

    # Extract all films mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_films_main(),
        template_class=FilmSelectionExtraction,
        extraction_name="film_selection_extraction"
    )

    # Keep only the first 4 films; pad to 4 if fewer are provided
    films: List[FilmItem] = list(extracted.films[:4])
    while len(films) < 4:
        films.append(FilmItem())

    # Add ground-truth-like constraints info for transparency
    gt_info = {
        "required_selection": [
            {
                "slot": "film_1",
                "genre": "Action",
                "rating": "PG-13",
                "us_release_window": "Jan 1, 2024 – Mar 31, 2024",
                "runtime_range_minutes": [90, 170]
            },
            {
                "slot": "film_2",
                "genre": "Drama",
                "rating": "R",
                "us_release_window": "Apr 1, 2024 – Jun 30, 2024",
                "runtime_range_minutes": [90, 140]
            },
            {
                "slot": "film_3",
                "genre": "Comedy",
                "rating": "PG-13",
                "us_release_window": "Jul 1, 2024 – Sep 30, 2024",
                "runtime_range_minutes": [85, 130]
            },
            {
                "slot": "film_4",
                "genre": "Animation",
                "rating": "PG",
                "us_release_window": "Oct 1, 2024 – Dec 31, 2024",
                "runtime_range_minutes": [80, 120]
            }
        ],
        "allowed_sources": sorted(list(ALLOWED_SOURCE_DOMAINS))
    }
    evaluator.add_ground_truth(gt_info, gt_type="constraints_summary")

    # Build verification subtrees for each film index 1..4
    for idx in range(1, 5):
        await verify_single_film(
            evaluator=evaluator,
            parent_node=root,
            film=films[idx - 1],
            film_idx=idx
        )

    # Return structured evaluation summary
    return evaluator.get_summary()