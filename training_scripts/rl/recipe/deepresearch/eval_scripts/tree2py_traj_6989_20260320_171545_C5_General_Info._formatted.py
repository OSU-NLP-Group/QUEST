import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "feb_2025_wide_release_family_movies"
TASK_DESCRIPTION = """
Identify four movies that have wide theatrical releases in the United States during February 2025 (February 1-28, 2025) and meet all of the following criteria:

1. Wide Release: Each movie must be released in 600 or more theaters nationwide on its opening weekend, as defined by Nielsen EDI standards for wide releases.

2. Studio Distribution: Each movie must be distributed by one of the following major Hollywood studios: Disney (including Marvel Studios), Universal Pictures (including Focus Features), Paramount Pictures, Warner Bros., Sony Pictures, or Lionsgate.

3. Target Audience: Each movie must be aimed at general audiences or families. Exclude adult-oriented horror films or explicit content films. The movie should be suitable for mainstream theatrical audiences.

4. Genre Diversity: Collectively, the four movies must represent at least three different primary genres (e.g., action, adventure, comedy, thriller, drama, family, animation).

5. Official Announcement: The release date and distribution information must be officially announced and verifiable through industry sources such as Box Office Mojo, IMDb, or official studio websites.

For each of the four movies, provide:
- The complete official title of the movie
- The exact US wide theatrical release date in February 2025
- The distributing studio name
- The primary genre
- A brief description confirming the target audience is suitable for general audiences or families
- A reference URL from a reliable industry source (Box Office Mojo, IMDb, or official studio website) confirming the release information
"""

ALLOWED_SOURCE_DOMAINS = [
    "boxofficemojo.com",
    "imdb.com",
    # Disney and labels
    "disney.com",
    "marvel.com",
    "pixar.com",
    "20thcenturystudios.com",
    # Universal and labels
    "universalpictures.com",
    "focusfeatures.com",
    # Paramount
    "paramount.com",
    "paramountpictures.com",
    # Warner Bros.
    "warnerbros.com",
    "wb.com",
    # Sony
    "sonypictures.com",
    "columbiapictures.com",
    "tristarpictures.com",
    # Lionsgate
    "lionsgate.com",
]

ALLOWED_STUDIOS = [
    "Disney",
    "Walt Disney Studios Motion Pictures",
    "Marvel Studios",
    "Pixar",
    "20th Century Studios",
    "Universal Pictures",
    "Focus Features",
    "Paramount Pictures",
    "Warner Bros.",
    "Warner Bros. Pictures",
    "Sony Pictures",
    "Sony Pictures Releasing",
    "Columbia Pictures",
    "TriStar Pictures",
    "Lionsgate",
    "Lionsgate Films",
]

ALLOWED_STUDIOS_INSTRUCTION = (
    "Allowed distributors: Disney (Walt Disney Studios Motion Pictures, Marvel Studios, Pixar, 20th Century Studios), "
    "Universal Pictures (including Focus Features), Paramount Pictures, Warner Bros. (Warner Bros. Pictures), "
    "Sony Pictures (Sony Pictures Releasing, Columbia Pictures, TriStar Pictures), and Lionsgate."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class MovieItem(BaseModel):
    title: Optional[str] = None
    us_wide_release_date: Optional[str] = None
    distributing_studio: Optional[str] = None
    primary_genre: Optional[str] = None
    target_audience_note: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class MoviesExtraction(BaseModel):
    movies: List[MovieItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_movies() -> str:
    return """
    Extract up to the first four movies presented in the answer that purport to meet the task criteria. For each movie, extract:
    - title: The complete official movie title as stated in the answer.
    - us_wide_release_date: The exact U.S. wide theatrical release date (as presented in the answer). It should be a date in February 2025 if provided.
    - distributing_studio: The distributing studio as named in the answer.
    - primary_genre: The primary genre of the movie (single label preferred, e.g., 'Action', 'Comedy', 'Animation', 'Family', 'Drama', 'Adventure', 'Thriller').
    - target_audience_note: A brief description or note from the answer that indicates the movie is suitable for general audiences or families (or explicitly states it is not adult-oriented horror/explicit).
    - reference_urls: A list of one or more URLs cited in the answer that are intended to confirm release and distribution info; these should be to reliable sources such as Box Office Mojo, IMDb, or official studio websites.
    
    Important:
    - Only extract information that is explicitly present in the answer text.
    - For URLs, extract the actual URLs exactly as given (plain or markdown link targets).
    - If any field is missing, set it to null (or an empty list for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def has_allowed_reference(urls: List[str]) -> bool:
    if not urls:
        return False
    for u in urls:
        d = _domain_of(u)
        if any(d.endswith(allowed) for allowed in ALLOWED_SOURCE_DOMAINS):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_movie(
    evaluator: Evaluator,
    parent_node,
    movie: MovieItem,
    movie_index: int,
) -> Dict[str, Any]:
    """
    Build verification nodes for a single movie and run checks.
    Returns a dict with handles to some leaf nodes (e.g., genre node) for later dependencies.
    """
    idx1 = movie_index + 1

    # Aggregator for this movie
    movie_node = evaluator.add_parallel(
        id=f"movie_{movie_index}",
        desc=f"Movie #{idx1}: qualifying movie meeting all criteria.",
        parent=parent_node,
        critical=False,  # Allow partial credit across different movies
    )

    # 1) Reference URL presence and domain validity (critical)
    ref_ok = has_allowed_reference(movie.reference_urls)
    evaluator.add_custom_node(
        result=ref_ok,
        id=f"movie_{movie_index}_reference_url",
        desc=(
            f"Provide a reference URL from a reliable source (Box Office Mojo, IMDb, or official studio website). "
            f"Found: {movie.reference_urls[:3] if movie.reference_urls else []}"
        ),
        parent=movie_node,
        critical=True,
    )

    sources = movie.reference_urls if movie.reference_urls else []

    # 2) Title verification (critical)
    title_leaf = evaluator.add_leaf(
        id=f"movie_{movie_index}_title",
        desc="Provide the complete official title of the movie.",
        parent=movie_node,
        critical=True,
    )
    title_text = movie.title or ""
    await evaluator.verify(
        claim=f"The official title of the movie shown on the referenced page matches '{title_text}'.",
        node=title_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the on-page title equals or clearly matches the stated title (allow minor punctuation/casing differences). "
            "If multiple titles are shown, use the main/official title."
        ),
    )

    # 3) Release date verification (critical)
    rel_date_leaf = evaluator.add_leaf(
        id=f"movie_{movie_index}_release_date",
        desc="Provide the specific US wide theatrical release date between February 1-28, 2025.",
        parent=movie_node,
        critical=True,
    )
    date_text = movie.us_wide_release_date or ""
    await evaluator.verify(
        claim=(
            f"The U.S. (domestic) wide theatrical release date for this movie is {date_text}, "
            "and it falls between February 1 and February 28, 2025 (inclusive)."
        ),
        node=rel_date_leaf,
        sources=sources,
        additional_instruction=(
            "Use the page's U.S. or domestic release info. "
            "If multiple dates are listed (e.g., limited vs. wide), the date must correspond to the 'wide' U.S. theatrical release. "
            "Accept variations in date format (e.g., Feb 7, 2025 / 2025-02-07)."
        ),
    )

    # 4) Wide release (600+ theaters) verification (critical)
    wide_leaf = evaluator.add_leaf(
        id=f"movie_{movie_index}_wide_release",
        desc=(
            "Confirm the movie is released in 600 or more theaters nationwide on opening weekend, "
            "meeting the Nielsen EDI wide release standard."
        ),
        parent=movie_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "On its opening weekend in the United States, this film played in 600 or more theaters "
            "(i.e., 'wide release' under the Nielsen EDI definition)."
        ),
        node=wide_leaf,
        sources=sources,
        additional_instruction=(
            "Prefer explicit opening-weekend 'Theaters' counts (e.g., Box Office Mojo 'Opening: Theaters'). "
            "If the page clearly labels the U.S. opening as 'Wide', that is sufficient. "
            "If the page indicates fewer than 600 theaters or 'Limited', mark as incorrect."
        ),
    )

    # 5) Studio verification (critical)
    studio_leaf = evaluator.add_leaf(
        id=f"movie_{movie_index}_studio",
        desc=(
            "Provide the distributing studio, which must be Disney/Marvel, Universal/Focus Features, "
            "Paramount, Warner Bros., Sony, or Lionsgate."
        ),
        parent=movie_node,
        critical=True,
    )
    studio_text = movie.distributing_studio or ""
    await evaluator.verify(
        claim=(
            f"The movie's distributor is '{studio_text}', and it belongs to the approved list of major studios."
        ),
        node=studio_leaf,
        sources=sources,
        additional_instruction=(
            f"{ALLOWED_STUDIOS_INSTRUCTION} "
            "Confirm the distributor on the page matches the stated distributor and is in this list."
        ),
    )

    # 6) Genre verification (critical)
    genre_leaf = evaluator.add_leaf(
        id=f"movie_{movie_index}_genre",
        desc="Provide the primary genre of the movie.",
        parent=movie_node,
        critical=True,
    )
    genre_text = movie.primary_genre or ""
    await evaluator.verify(
        claim=f"The movie's primary genre is '{genre_text}'.",
        node=genre_leaf,
        sources=sources,
        additional_instruction=(
            "If multiple genres are listed, use the first/primary genre indicated. "
            "Allow reasonable synonyms (e.g., 'Action/Adventure' counted as primary 'Action' if listed first)."
        ),
    )

    # 7) Target audience verification (critical)
    audience_leaf = evaluator.add_leaf(
        id=f"movie_{movie_index}_target_audience",
        desc="Confirm the movie is aimed at general audiences or families, not adult-oriented horror or explicit content.",
        parent=movie_node,
        critical=True,
    )
    note_text = movie.target_audience_note or ""
    await evaluator.verify(
        claim=(
            "This movie is suitable for general audiences or families (i.e., not adult‑oriented horror or explicit content)."
        ),
        node=audience_leaf,
        sources=sources,
        additional_instruction=(
            "Use MPAA rating and page description. PG or PG‑13 is acceptable if content is mainstream; "
            "family, animation, or general‑audience descriptions also acceptable. "
            "If explicitly adult, explicit/sexual content, or strongly graphic horror emphasis, mark as incorrect. "
            f"Answer note (if any) provided: {note_text}"
        ),
    )

    return {
        "movie_node": movie_node,
        "genre_leaf": genre_leaf,
    }


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'four wide-release, family/general audience movies in Feb 2025' task.
    """
    # Initialize evaluator (root is non-critical parallel aggregator by default)
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

    # Extract movie candidates from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_movies(),
        template_class=MoviesExtraction,
        extraction_name="movies_extraction",
    )

    # Keep only the first four; pad with empty items if fewer
    movies = list(extracted.movies[:4])
    while len(movies) < 4:
        movies.append(MovieItem())

    # Build verification subtrees per movie (in parallel or sequentially)
    genre_leaves = []
    for i in range(4):
        result_handles = await verify_movie(evaluator, root, movies[i], i)
        genre_leaves.append(result_handles["genre_leaf"])

    # Genre diversity check (non-critical, to allow partial scoring)
    diversity_leaf = evaluator.add_leaf(
        id="genre_diversity_check",
        desc="Verify that the four movies collectively represent at least three different primary genres.",
        parent=root,
        critical=False,
    )

    # Prepare genres list for the claim
    genres_list = [m.primary_genre or "" for m in movies]
    diversity_claim = (
        f"The four movies' primary genres are: {genres_list}. There are at least three distinct primary genres represented."
    )

    # Make this check depend on successful genre identifications to avoid false positives
    await evaluator.verify(
        claim=diversity_claim,
        node=diversity_leaf,
        additional_instruction=(
            "Determine the count of distinct primary genres ignoring case and trivial punctuation. "
            "Treat close variants like 'Action/Adventure' as 'Action' if it appears first. "
            "At least three distinct primary genres across four movies are required for a pass."
        ),
        extra_prerequisites=genre_leaves,
    )

    return evaluator.get_summary()