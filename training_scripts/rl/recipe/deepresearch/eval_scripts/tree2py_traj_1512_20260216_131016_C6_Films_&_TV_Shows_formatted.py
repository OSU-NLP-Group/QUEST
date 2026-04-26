import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "linklater_lee_daniel_trio"
TASK_DESCRIPTION = (
    "Identify three feature films directed by Richard Linklater that were cinematographed by Lee Daniel, "
    "with one film from each of the following decades: the 1990s, the 2000s, and the 2010s. For each film, provide: "
    "(1) The film's title, (2) The exact release year, (3) A URL to the film's IMDb page that shows Lee Daniel's cinematography credit, "
    "(4) Whether the film is part of a trilogy or film series (and if so, which one), or if it is a standalone film, "
    "(5) Whether Lee Daniel received any cinematography award nominations or wins specifically for that film, and "
    "(6) The film's current Rotten Tomatoes Tomatometer score. "
    "Additionally, provide reference URLs to verify the trilogy/series status, award information, and Rotten Tomatoes score for each film."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmDetails(BaseModel):
    title: Optional[str] = None
    release_year: Optional[str] = None
    imdb_url: Optional[str] = None
    director_name: Optional[str] = None  # As written in the answer
    cinematographer_name: Optional[str] = None  # As written in the answer
    year_urls: List[str] = Field(default_factory=list)  # URLs verifying the release year
    series_status: Optional[str] = None  # e.g., "Before trilogy" or "standalone"
    series_urls: List[str] = Field(default_factory=list)  # URLs verifying series/trilogy membership
    award_statement: Optional[str] = None  # e.g., "no nominations/wins" or "won X"
    award_urls: List[str] = Field(default_factory=list)  # URLs verifying award status
    rt_score: Optional[str] = None  # e.g., "97%" or "92"
    rt_urls: List[str] = Field(default_factory=list)  # URLs to Rotten Tomatoes page(s)


class LinklaterFilmsExtraction(BaseModel):
    film_1990s: Optional[FilmDetails] = None
    film_2000s: Optional[FilmDetails] = None
    film_2010s: Optional[FilmDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_films() -> str:
    return (
        "From the provided answer, extract up to one film for each decade that matches ALL of the following:\n"
        "- Directed by Richard Linklater\n"
        "- Cinematography by Lee Daniel (also known as Director of Photography / DP)\n"
        "Decades required: 1990s, 2000s, 2010s. If multiple films per decade are mentioned, pick the first one.\n\n"
        "For each selected film, extract the following fields exactly as stated in the answer (return null if missing):\n"
        "1) title: The film title\n"
        "2) release_year: The exact release year stated in the answer\n"
        "3) imdb_url: A direct URL to the film's IMDb page that shows Lee Daniel's cinematography credit\n"
        "4) director_name: The director's name as cited (should be Richard Linklater)\n"
        "5) cinematographer_name: The cinematographer's name as cited (should be Lee Daniel)\n"
        "6) year_urls: All URLs mentioned that verify the release year (IMDb, Wikipedia, or other reliable sources)\n"
        "7) series_status: Either the trilogy/series name (e.g., 'Before trilogy') or 'standalone'\n"
        "8) series_urls: All URLs mentioned that verify the trilogy/series membership or standalone status\n"
        "9) award_statement: The statement about cinematography award nominations or wins for Lee Daniel specifically for this film; "
        "   if the answer says there were none, extract a clear 'no nominations/wins' statement\n"
        "10) award_urls: All URLs mentioned that verify the award status\n"
        "11) rt_score: The Rotten Tomatoes Tomatometer score as provided (e.g., '97%' or '92')\n"
        "12) rt_urls: All Rotten Tomatoes URLs for the film as mentioned\n\n"
        "Return the result as an object with fields film_1990s, film_2000s, film_2010s. "
        "Each field should be a FilmDetails object with the above keys, or null if the answer does not provide a valid film for that decade."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_year(year_text: Optional[str]) -> Optional[int]:
    if not year_text:
        return None
    m = re.search(r"(19|20)\d{2}", year_text)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def year_in_range(year: Optional[int], start: int, end: int) -> bool:
    return year is not None and start <= year <= end


def decade_name_for_year(year: Optional[int]) -> Optional[str]:
    if year is None:
        return None
    if 1990 <= year <= 1999:
        return "1990s"
    if 2000 <= year <= 2009:
        return "2000s"
    if 2010 <= year <= 2019:
        return "2010s"
    return None


def safe_sources(primary: Optional[List[str]], fallback: Optional[str]) -> Optional[List[str]]:
    """
    Return sources list to use for verification. If primary list is empty, fallback to single URL if available.
    If both are missing or empty, return None to indicate no sources.
    """
    primary = primary or []
    if len(primary) > 0:
        return primary
    if fallback and fallback.strip():
        return [fallback]
    return None


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Film verification                                                           #
# --------------------------------------------------------------------------- #
async def verify_one_film(
    evaluator: Evaluator,
    parent_node,
    film: Optional[FilmDetails],
    decade_label: str,
    decade_span: Tuple[int, int],
) -> None:
    """
    Build verification nodes and perform checks for a single film in a decade.
    """
    # Sequential node for this decade's film (non-critical to allow partial credit per film)
    film_node = evaluator.add_sequential(
        id=f"film_{decade_label}",
        desc=f"Identify and verify one Richard Linklater film from the {decade_label} with Lee Daniel as cinematographer",
        parent=parent_node,
        critical=False,
    )

    # Core Identification (critical, parallel)
    core_node = evaluator.add_parallel(
        id=f"film_{decade_label}_core",
        desc="Correctly identify and verify the basic film credentials",
        parent=film_node,
        critical=True,
    )

    # Creative Team (critical, parallel)
    creative_node = evaluator.add_parallel(
        id=f"film_{decade_label}_creative_team",
        desc="Verify the director and cinematographer credits",
        parent=core_node,
        critical=True,
    )

    # Director leaf
    director_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_director",
        desc="The film is directed by Richard Linklater",
        parent=creative_node,
        critical=True,
    )
    title = film.title if film else ""
    imdb_url = film.imdb_url if film else None
    director_claim = f"The film titled '{title}' is directed by Richard Linklater."
    await evaluator.verify(
        claim=director_claim,
        node=director_leaf,
        sources=imdb_url,
        additional_instruction=(
            "Verify on the IMDb page (or provided source) that the 'Director' credit is Richard Linklater. "
            "Allow minor variations (e.g., 'directed by' phrasing)."
        ),
    )

    # Cinematographer leaf
    cinematographer_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_cinematographer",
        desc="Lee Daniel is credited as cinematographer or director of photography",
        parent=creative_node,
        critical=True,
    )
    cinematographer_claim = (
        f"Lee Daniel is credited as 'Cinematography' or 'Director of Photography' for the film titled '{title}'."
    )
    await evaluator.verify(
        claim=cinematographer_claim,
        node=cinematographer_leaf,
        sources=imdb_url,
        additional_instruction=(
            "Look for 'Cinematography by' or 'Director of Photography' sections listing Lee Daniel on IMDb."
        ),
    )

    # IMDb URL leaf (credit page presence showing Lee Daniel)
    imdb_credit_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_imdb_url",
        desc="IMDb URL provided showing Lee Daniel's cinematography credit",
        parent=creative_node,
        critical=True,
    )
    imdb_credit_claim = (
        f"This IMDb page shows Lee Daniel credited for cinematography (or director of photography) for '{title}'."
    )
    await evaluator.verify(
        claim=imdb_credit_claim,
        node=imdb_credit_leaf,
        sources=imdb_url,
        additional_instruction=(
            "Confirm that the IMDb page explicitly lists Lee Daniel under Cinematography/Director of Photography."
        ),
    )

    # Temporal Verification (critical, parallel)
    temporal_node = evaluator.add_parallel(
        id=f"film_{decade_label}_temporal",
        desc="Verify the film's release timing",
        parent=core_node,
        critical=True,
    )

    # Decade Check leaf
    decade_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_decade_check",
        desc=f"Film was released between {decade_span[0]}-{decade_span[1]}",
        parent=temporal_node,
        critical=True,
    )
    release_year_text = film.release_year if film else None
    decade_claim = (
        f"The release year of '{title}' is {release_year_text} and falls within {decade_span[0]}-{decade_span[1]} inclusive."
    )
    decade_sources = safe_sources(film.year_urls if film else None, imdb_url)
    await evaluator.verify(
        claim=decade_claim,
        node=decade_leaf,
        sources=decade_sources,
        additional_instruction=(
            "Confirm the film's release year and that it is within the specified decade range using the provided source(s)."
        ),
    )

    # Exact Year leaf
    exact_year_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_exact_year",
        desc="Exact release year is provided and accurate",
        parent=temporal_node,
        critical=True,
    )
    exact_year_claim = f"The film '{title}' was released in {release_year_text}."
    await evaluator.verify(
        claim=exact_year_claim,
        node=exact_year_leaf,
        sources=decade_sources,
        additional_instruction=(
            "Use the provided source(s) to verify the exact release year. Accept minor regional differences "
            "if the provided source clearly supports the cited year."
        ),
    )

    # Year URL leaf (explicit verification via reference URL)
    year_url_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_year_url",
        desc="Reference URL verifying the release year",
        parent=temporal_node,
        critical=True,
    )
    year_url_claim = f"The provided URL(s) confirm the film '{title}' was released in {release_year_text}."
    await evaluator.verify(
        claim=year_url_claim,
        node=year_url_leaf,
        sources=decade_sources,
        additional_instruction="Verify that at least one provided URL explicitly confirms the stated release year.",
    )

    # Supplementary Details (non-critical, parallel)
    supplementary_node = evaluator.add_parallel(
        id=f"film_{decade_label}_supplementary",
        desc="Provide additional film details beyond basic identification",
        parent=film_node,
        critical=False,
    )

    # Series Context (non-critical, parallel)
    series_node = evaluator.add_parallel(
        id=f"film_{decade_label}_series_context",
        desc="Identify trilogy or series status",
        parent=supplementary_node,
        critical=False,
    )

    # Series Status leaf
    series_status_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_series_status",
        desc="Correct identification of trilogy/series membership or standalone status",
        parent=series_node,
        critical=True,
    )
    series_status_text = film.series_status if film else None
    if _nonempty(series_status_text) and series_status_text.lower().strip() != "standalone":
        series_claim = f"The film '{title}' is part of the '{series_status_text}' trilogy/series."
    else:
        series_claim = f"The film '{title}' is a standalone film and not part of a trilogy or series."
    series_sources = safe_sources(film.series_urls if film else None, imdb_url)
    await evaluator.verify(
        claim=series_claim,
        node=series_status_leaf,
        sources=series_sources,
        additional_instruction=(
            "Use Wikipedia/IMDb/official sources provided to confirm the film's trilogy/series membership or standalone status."
        ),
    )

    # Series URL leaf
    series_url_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_series_url",
        desc="URL verifying the series status",
        parent=series_node,
        critical=True,
    )
    series_url_claim = f"The provided URL(s) verify the series/standalone status for '{title}'."
    await evaluator.verify(
        claim=series_url_claim,
        node=series_url_leaf,
        sources=series_sources,
        additional_instruction="Confirm that at least one source validates the claimed series/standalone status.",
    )

    # Award Status (non-critical, parallel)
    award_node = evaluator.add_parallel(
        id=f"film_{decade_label}_award_status",
        desc="Identify cinematography award recognition",
        parent=supplementary_node,
        critical=False,
    )

    # Awards leaf
    awards_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_awards",
        desc="Accurate statement of award nominations or wins for Lee Daniel",
        parent=award_node,
        critical=True,
    )
    award_statement_text = film.award_statement if film else None
    awards_claim = (
        f"For the film '{title}', Lee Daniel's cinematography award status is: {award_statement_text}."
        if _nonempty(award_statement_text)
        else f"For the film '{title}', there are no known cinematography award nominations or wins for Lee Daniel."
    )
    awards_sources = safe_sources(film.award_urls if film else None, None)
    await evaluator.verify(
        claim=awards_claim,
        node=awards_leaf,
        sources=awards_sources,
        additional_instruction=(
            "Verify award nominations/wins specifically for Lee Daniel's cinematography for this film. "
            "If no awards are claimed and no sources are provided, treat as not supported."
        ),
    )

    # Award URL leaf
    award_url_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_award_url",
        desc="URL verifying award nomination/win status",
        parent=award_node,
        critical=True,
    )
    award_url_claim = f"The provided URL(s) verify the award nomination/win status for Lee Daniel for '{title}'."
    await evaluator.verify(
        claim=award_url_claim,
        node=award_url_leaf,
        sources=awards_sources,
        additional_instruction=(
            "Confirm that the URLs substantiate the stated award information (either presence or absence)."
        ),
    )

    # Critical Reception (non-critical, parallel)
    rt_node = evaluator.add_parallel(
        id=f"film_{decade_label}_critical_reception",
        desc="Provide Rotten Tomatoes score",
        parent=supplementary_node,
        critical=False,
    )

    # RT Score leaf
    rt_score_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_rt_score",
        desc="Rotten Tomatoes Tomatometer score is provided and accurate",
        parent=rt_node,
        critical=True,
    )
    rt_score_text = film.rt_score if film else None
    rt_sources = safe_sources(film.rt_urls if film else None, None)
    rt_score_claim = f"The current Rotten Tomatoes Tomatometer score for '{title}' is {rt_score_text}."
    await evaluator.verify(
        claim=rt_score_claim,
        node=rt_score_leaf,
        sources=rt_sources,
        additional_instruction=(
            "Check the Tomatometer score (not the Audience Score). Accept minor rounding differences. "
            "Use the provided Rotten Tomatoes URL(s)."
        ),
    )

    # RT URL leaf
    rt_url_leaf = evaluator.add_leaf(
        id=f"film_{decade_label}_rt_url",
        desc="Rotten Tomatoes URL for the film",
        parent=rt_node,
        critical=True,
    )
    rt_url_claim = f"The provided URL is the Rotten Tomatoes page for '{title}'."
    await evaluator.verify(
        claim=rt_url_claim,
        node=rt_url_leaf,
        sources=rt_sources,
        additional_instruction="Confirm the URL is a Rotten Tomatoes page corresponding to the specified film.",
    )


# --------------------------------------------------------------------------- #
# Cross-film verification                                                     #
# --------------------------------------------------------------------------- #
async def build_cross_verification(
    evaluator: Evaluator,
    parent_node,
    films: Dict[str, Optional[FilmDetails]],
) -> None:
    """
    Build and evaluate cross-film verification nodes covering distinctness, cinematographer consistency,
    decade coverage, and feature-film status.
    """
    cross_node = evaluator.add_parallel(
        id="cross_verification",
        desc="Verify overall task completion requirements across all three films",
        parent=parent_node,
        critical=True,
    )

    # Distinctness: All three films are distinct titles
    titles = [films.get("1990s").title if films.get("1990s") else None,
              films.get("2000s").title if films.get("2000s") else None,
              films.get("2010s").title if films.get("2010s") else None]
    nonempty_titles = [t for t in titles if _nonempty(t)]
    distinct = len(nonempty_titles) == 3 and len(set([t.strip().lower() for t in nonempty_titles])) == 3
    evaluator.add_custom_node(
        result=distinct,
        id="cross_distinctness",
        desc="All three films are distinct titles",
        parent=cross_node,
        critical=True,
    )

    # Cinematographer Consistency: All three films have Lee Daniel as cinematographer
    # Use earlier film-level cinematographer leaves as prerequisites
    def _leaf_passed(leaf_id: str) -> bool:
        node = evaluator.find_node(leaf_id)
        return bool(node and node.status == "passed")

    cine_ok = all([
        _leaf_passed("film_1990s_cinematographer"),
        _leaf_passed("film_2000s_cinematographer"),
        _leaf_passed("film_2010s_cinematographer"),
    ])
    evaluator.add_custom_node(
        result=cine_ok,
        id="cross_cinematographer_consistency",
        desc="All three films have Lee Daniel as cinematographer (no films with other cinematographers)",
        parent=cross_node,
        critical=True,
    )

    # Decade Coverage: Must cover 1990s, 2000s, 2010s uniquely
    years = [
        parse_year(films.get("1990s").release_year if films.get("1990s") else None),
        parse_year(films.get("2000s").release_year if films.get("2000s") else None),
        parse_year(films.get("2010s").release_year if films.get("2010s") else None),
    ]
    decades = [decade_name_for_year(y) for y in years]
    decade_ok = set(decades) == {"1990s", "2000s", "2010s"}
    evaluator.add_custom_node(
        result=decade_ok,
        id="cross_decade_coverage",
        desc="Films collectively cover all three decades (1990s, 2000s, 2010s) with one per decade",
        parent=cross_node,
        critical=True,
    )

    # Feature Film Status: Verify each is a feature-length film via IMDb (standalone verifications)
    async def _verify_feature_film(title: Optional[str], imdb_url: Optional[str]) -> bool:
        if not _nonempty(imdb_url):
            return False
        claim = (
            f"The IMDb page indicates that '{title}' is a feature-length film (not a short nor a TV episode). "
            f"Confirm by checking the title type classification."
        )
        return await evaluator.verify(
            claim=claim,
            node=None,  # standalone verification
            sources=imdb_url,
            additional_instruction=(
                "On IMDb, confirm the title type is 'Feature Film' or equivalent. If the page indicates 'TV Episode' or 'Short', it is not acceptable."
            ),
        )

    feature_checks = await asyncio.gather(
        _verify_feature_film(titles[0], films.get("1990s").imdb_url if films.get("1990s") else None),
        _verify_feature_film(titles[1], films.get("2000s").imdb_url if films.get("2000s") else None),
        _verify_feature_film(titles[2], films.get("2010s").imdb_url if films.get("2010s") else None),
        return_exceptions=True
    )
    feature_ok = all(isinstance(r, bool) and r for r in feature_checks)
    evaluator.add_custom_node(
        result=feature_ok,
        id="cross_feature_film_status",
        desc="All three are feature-length films (not shorts or TV episodes)",
        parent=cross_node,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Linklater/Lee Daniel film selection task.
    """
    # Initialize evaluator (root non-critical to allow partial credit aggregation)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates films & cross checks independently
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

    # Extract films information
    extracted: LinklaterFilmsExtraction = await evaluator.extract(
        prompt=prompt_extract_films(),
        template_class=LinklaterFilmsExtraction,
        extraction_name="linklater_lee_daniel_films",
    )

    # Build film verifications per decade (sequential nodes under root)
    # 1990s
    await verify_one_film(
        evaluator=evaluator,
        parent_node=root,
        film=extracted.film_1990s,
        decade_label="1990s",
        decade_span=(1990, 1999),
    )
    # 2000s
    await verify_one_film(
        evaluator=evaluator,
        parent_node=root,
        film=extracted.film_2000s,
        decade_label="2000s",
        decade_span=(2000, 2009),
    )
    # 2010s
    await verify_one_film(
        evaluator=evaluator,
        parent_node=root,
        film=extracted.film_2010s,
        decade_label="2010s",
        decade_span=(2010, 2019),
    )

    # Cross-film verification (critical)
    films_by_decade: Dict[str, Optional[FilmDetails]] = {
        "1990s": extracted.film_1990s,
        "2000s": extracted.film_2000s,
        "2010s": extracted.film_2010s,
    }
    await build_cross_verification(evaluator, root, films_by_decade)

    # Return structured result
    return evaluator.get_summary()