import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tiff2024_copros"
TASK_DESCRIPTION = (
    "Identify three feature-length films that had their world premiere at the 2024 Toronto International Film Festival "
    "(held September 5-15, 2024) and are international co-productions involving at least two countries. For each film, "
    "provide: (1) The film title, (2) The director's full name, (3) All co-production countries, (4) The cinematographer's "
    "full name, (5) At least one lead cast member's name, (6) At least one production company, (7) The theatrical distributor "
    "for at least one major market (United States, United Kingdom, Australia, or Canada), and (8) A reference URL confirming "
    "the film's TIFF 2024 premiere. Each film must have a runtime of 80 minutes or longer."
)

MAJOR_MARKET_ALIASES = {
    "united states": {"united states", "usa", "us"},
    "united kingdom": {"united kingdom", "uk"},
    "australia": {"australia"},
    "canada": {"canada"},
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistributionInfo(BaseModel):
    """Distribution info for a specific market."""
    market: Optional[str] = None  # e.g., "United States", "UK", "Australia", "Canada"
    distributor: Optional[str] = None
    url: Optional[str] = None


class FilmEntry(BaseModel):
    """Complete information for a single film extracted from the answer."""
    title: Optional[str] = None
    runtime_text: Optional[str] = None  # e.g., "102 minutes" or "1h 42m"
    director: Optional[str] = None
    countries: List[str] = Field(default_factory=list)
    cinematographer: Optional[str] = None
    lead_cast: List[str] = Field(default_factory=list)
    production_companies: List[str] = Field(default_factory=list)
    distributors: List[DistributionInfo] = Field(default_factory=list)
    tiff_premiere_url: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)


class FilmsExtraction(BaseModel):
    """Container for up to N films in the answer."""
    films: List[FilmEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_films() -> str:
    return """
    Extract up to five feature-length films (runtime >= 80 minutes) mentioned in the answer that claim to have had their world premiere at TIFF 2024 (held September 5–15, 2024).
    For each film, extract the following fields exactly as presented in the answer (do not invent):
    - title: The film title string.
    - runtime_text: The runtime as given in the answer (e.g., "102 minutes", "1h 42m"). If not explicitly stated in the answer, return null.
    - director: Director's full name as stated in the answer; if not present, return null.
    - countries: An array of all co-production countries listed in the answer text; return an empty array if none are listed.
    - cinematographer: Cinematographer's full name as stated in the answer; if not present, return null.
    - lead_cast: An array of at least one lead/principal cast member name as presented; return empty array if none are listed.
    - production_companies: An array of production company names as presented; return empty array if none are listed.
    - distributors: An array of distribution entries, each with:
        * market: The market name as provided (e.g., "United States", "USA", "US", "United Kingdom", "UK", "Australia", "Canada").
        * distributor: The theatrical distributor name.
        * url: A URL for distributor info if explicitly provided; otherwise null.
      If the answer does not provide distributor info, return an empty array.
    - tiff_premiere_url: A single URL provided in the answer that specifically confirms the film's world premiere at TIFF 2024. If multiple such URLs are present, choose the best/most direct one. If none is provided, return null.
    - other_urls: Any other film-related URLs mentioned in the answer for this film (official page, trade press articles, etc.). Return an empty array if none.

    IMPORTANT RULES:
    - Only extract URLs if they are explicitly present in the answer; include full protocol (http/https).
    - Do not infer or invent information; if any required field is missing from the answer, set it to null or an empty list accordingly.
    - Preserve the textual formatting of names as they appear in the answer.
    - If more than three films are present, extract them all; the evaluation will pick the first three.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(idx: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][idx] if idx < 5 else f"#{idx + 1}"


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for it in items:
        if not it:
            continue
        k = it.strip()
        if k and k not in seen:
            seen.add(k)
            result.append(k)
    return result


def collect_sources_for_film(film: FilmEntry) -> List[str]:
    combined: List[str] = []
    if film.tiff_premiere_url and film.tiff_premiere_url.strip():
        combined.append(film.tiff_premiere_url.strip())
    for d in film.distributors:
        if d.url and d.url.strip():
            combined.append(d.url.strip())
    for u in film.other_urls:
        if u and u.strip():
            combined.append(u.strip())
    return _unique_preserve_order(combined)


def choose_major_market_distribution(film: FilmEntry) -> Optional[DistributionInfo]:
    for d in film.distributors:
        if not d.market or not d.distributor:
            continue
        m = d.market.strip().lower()
        for canonical, aliases in MAJOR_MARKET_ALIASES.items():
            if m in aliases:
                return d
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_film(
    evaluator: Evaluator,
    parent_node,
    film: FilmEntry,
    film_index: int,
) -> None:
    """
    Build the verification subtree for one film and execute checks.
    """
    ord_name = _ordinal(film_index)
    film_node = evaluator.add_parallel(
        id=f"Film_{film_index + 1}",
        desc=f"{ord_name} qualifying film and its required details",
        parent=parent_node,
        critical=False,
    )

    sources = collect_sources_for_film(film)

    # Title existence (critical)
    title_exists_node = evaluator.add_custom_node(
        result=bool(film.title and film.title.strip()),
        id=f"Film_{film_index + 1}_Title",
        desc="Film title is provided",
        parent=film_node,
        critical=True,
    )

    # Runtime provided existence (critical)
    runtime_provided_node = evaluator.add_custom_node(
        result=bool(film.runtime_text and film.runtime_text.strip()),
        id=f"Film_{film_index + 1}_Runtime_Provided",
        desc="Film runtime is provided",
        parent=film_node,
        critical=True,
    )

    # Runtime >= 80 minutes verification (critical)
    runtime_verify_node = evaluator.add_leaf(
        id=f"Film_{film_index + 1}_Runtime",
        desc="Film runtime is provided and is 80 minutes or longer",
        parent=film_node,
        critical=True,
    )
    claim_runtime = f"The film '{film.title or 'the film'}' has a runtime of at least 80 minutes."
    await evaluator.verify(
        claim=claim_runtime,
        node=runtime_verify_node,
        sources=sources if sources else None,
        additional_instruction="Confirm the runtime on the referenced page(s) is >= 80 minutes. Accept formats like '80 min', '1h 20m', or any value equivalent or greater.",
        extra_prerequisites=[runtime_provided_node, title_exists_node],
    )

    # TIFF world premiere URL existence (critical)
    premiere_url_exists_node = evaluator.add_custom_node(
        result=bool(
            film.tiff_premiere_url
            and film.tiff_premiere_url.strip()
            and film.tiff_premiere_url.strip().startswith(("http://", "https://"))
        ),
        id=f"Film_{film_index + 1}_TIFF_2024_URL_Provided",
        desc="Premiere confirmation URL is provided",
        parent=film_node,
        critical=True,
    )

    # TIFF world premiere verification (critical)
    premiere_verify_node = evaluator.add_leaf(
        id=f"Film_{film_index + 1}_TIFF_2024_World_Premiere_With_URL",
        desc="A valid reference URL is provided that confirms the film had its world premiere at TIFF 2024 (Sept 5–15, 2024)",
        parent=film_node,
        critical=True,
    )
    claim_premiere = (
        f"The film '{film.title or ''}' had its world premiere at the 2024 Toronto International Film Festival "
        f"(held September 5–15, 2024)."
    )
    await evaluator.verify(
        claim=claim_premiere,
        node=premiere_verify_node,
        sources=film.tiff_premiere_url if film.tiff_premiere_url else None,
        additional_instruction=(
            "Confirm the page explicitly indicates TIFF 2024 and 'world premiere' for this film. "
            "Reject if it only states 'North American premiere', 'International premiere', or a different year."
        ),
        extra_prerequisites=[premiere_url_exists_node, title_exists_node],
    )

    # Director existence (critical)
    director_exists_node = evaluator.add_custom_node(
        result=bool(film.director and film.director.strip()),
        id=f"Film_{film_index + 1}_Director_Provided",
        desc="Director name is provided",
        parent=film_node,
        critical=True,
    )

    # Director correctness (critical)
    director_verify_node = evaluator.add_leaf(
        id=f"Film_{film_index + 1}_Director",
        desc="Director's full name is correctly provided",
        parent=film_node,
        critical=True,
    )
    claim_director = f"The director of '{film.title or ''}' is {film.director or ''}."
    await evaluator.verify(
        claim=claim_director,
        node=director_verify_node,
        sources=sources if sources else None,
        additional_instruction="Verify the director credit on TIFF or other referenced pages. Allow minor variants (middle names, diacritics, letter casing).",
        extra_prerequisites=[director_exists_node, title_exists_node],
    )

    # Countries existence (critical)
    countries_exists_node = evaluator.add_custom_node(
        result=bool(film.countries and len(film.countries) > 0),
        id=f"Film_{film_index + 1}_Coproduction_Countries_Provided",
        desc="Co-production countries are provided",
        parent=film_node,
        critical=True,
    )

    # Countries correctness (critical)
    countries_verify_node = evaluator.add_leaf(
        id=f"Film_{film_index + 1}_Coproduction_Countries_All",
        desc="All co-production countries are correctly identified",
        parent=film_node,
        critical=True,
    )
    countries_list_str = ", ".join(film.countries) if film.countries else ""
    claim_countries = (
        f"The film '{film.title or ''}' is a co-production involving the following countries: {countries_list_str}."
    )
    await evaluator.verify(
        claim=claim_countries,
        node=countries_verify_node,
        sources=sources if sources else None,
        additional_instruction="Confirm that the referenced page(s) list these production countries for the film.",
        extra_prerequisites=[countries_exists_node, title_exists_node],
    )

    # Countries minimum two (critical)
    min_two_result = bool(film.countries and len(film.countries) >= 2)
    evaluator.add_custom_node(
        result=min_two_result,
        id=f"Film_{film_index + 1}_Coproduction_Countries_Min_Two",
        desc="The film is an international co-production involving at least two countries (co-production countries count ≥ 2)",
        parent=film_node,
        critical=True,
    )

    # Cinematographer existence (critical)
    cinematographer_exists_node = evaluator.add_custom_node(
        result=bool(film.cinematographer and film.cinematographer.strip()),
        id=f"Film_{film_index + 1}_Cinematographer_Provided",
        desc="Cinematographer name is provided",
        parent=film_node,
        critical=True,
    )

    # Cinematographer correctness (critical)
    cinematographer_verify_node = evaluator.add_leaf(
        id=f"Film_{film_index + 1}_Cinematographer",
        desc="Cinematographer's full name is correctly provided",
        parent=film_node,
        critical=True,
    )
    claim_cinematographer = f"The cinematographer of '{film.title or ''}' is {film.cinematographer or ''}."
    await evaluator.verify(
        claim=claim_cinematographer,
        node=cinematographer_verify_node,
        sources=sources if sources else None,
        additional_instruction="Verify the credited cinematographer on TIFF or referenced pages. Allow minor naming variants.",
        extra_prerequisites=[cinematographer_exists_node, title_exists_node],
    )

    # Lead cast existence (critical)
    lead_cast_exists = bool(film.lead_cast and len(film.lead_cast) > 0 and film.lead_cast[0].strip())
    lead_cast_exists_node = evaluator.add_custom_node(
        result=lead_cast_exists,
        id=f"Film_{film_index + 1}_Lead_Cast_Provided",
        desc="At least one lead cast member is provided",
        parent=film_node,
        critical=True,
    )

    # Lead cast correctness (critical)
    lead_cast_member = film.lead_cast[0] if film.lead_cast else ""
    lead_cast_verify_node = evaluator.add_leaf(
        id=f"Film_{film_index + 1}_Lead_Cast",
        desc="At least one lead cast member is correctly identified",
        parent=film_node,
        critical=True,
    )
    claim_lead_cast = f"{lead_cast_member} is a lead or principal cast member in the film '{film.title or ''}'."
    await evaluator.verify(
        claim=claim_lead_cast,
        node=lead_cast_verify_node,
        sources=sources if sources else None,
        additional_instruction="Verify the named actor is listed among lead/principal cast on the referenced page(s). Allow minor name variants.",
        extra_prerequisites=[lead_cast_exists_node, title_exists_node],
    )

    # Production company existence (critical)
    production_exists = bool(
        film.production_companies and len(film.production_companies) > 0 and film.production_companies[0].strip()
    )
    production_exists_node = evaluator.add_custom_node(
        result=production_exists,
        id=f"Film_{film_index + 1}_Production_Company_Provided",
        desc="At least one production company is provided",
        parent=film_node,
        critical=True,
    )

    # Production company correctness (critical)
    production_company = film.production_companies[0] if film.production_companies else ""
    production_verify_node = evaluator.add_leaf(
        id=f"Film_{film_index + 1}_Production_Company",
        desc="At least one production company is correctly identified",
        parent=film_node,
        critical=True,
    )
    claim_production = f"The production company {production_company} is associated with the film '{film.title or ''}'."
    await evaluator.verify(
        claim=claim_production,
        node=production_verify_node,
        sources=sources if sources else None,
        additional_instruction="Confirm the company is credited as a production company/producer for the film on the referenced page(s).",
        extra_prerequisites=[production_exists_node, title_exists_node],
    )

    # Distributor for a major market existence (critical)
    chosen_dist = choose_major_market_distribution(film)
    dist_exists_node = evaluator.add_custom_node(
        result=bool(chosen_dist is not None),
        id=f"Film_{film_index + 1}_Distributor_Major_Market_Provided",
        desc="A theatrical distributor for a major market is provided",
        parent=film_node,
        critical=True,
    )

    # Distributor correctness (critical)
    distributor_verify_node = evaluator.add_leaf(
        id=f"Film_{film_index + 1}_Distributor_Major_Market",
        desc="The theatrical distributor for at least one major market (US, UK, Australia, or Canada) is correctly identified",
        parent=film_node,
        critical=True,
    )
    dist_sources = sources.copy()
    if chosen_dist and chosen_dist.url and chosen_dist.url.strip():
        dist_sources = [chosen_dist.url.strip()] + dist_sources
    claim_distributor = (
        f"The film '{film.title or ''}' has the theatrical distributor '{chosen_dist.distributor if chosen_dist else ''}' "
        f"in {chosen_dist.market if chosen_dist and chosen_dist.market else 'a major market'}."
    )
    await evaluator.verify(
        claim=claim_distributor,
        node=distributor_verify_node,
        sources=dist_sources if dist_sources else None,
        additional_instruction="Verify the distributor for a major market (United States/USA/US, United Kingdom/UK, Australia, Canada). Accept equivalent market names.",
        extra_prerequisites=[dist_exists_node, title_exists_node],
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
) -> Dict:
    """
    Evaluate an answer for TIFF 2024 international co-production films task.
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
        default_model=model,
    )

    # Extract film entries from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_films(),
        template_class=FilmsExtraction,
        extraction_name="films_extraction",
    )

    # Use the first three films; pad with empty entries if fewer provided
    films: List[FilmEntry] = list(extraction.films[:3])
    while len(films) < 3:
        films.append(FilmEntry())

    # Add a top-level parallel node to mirror rubric root (optional; root already serves this role)
    top_node = evaluator.add_parallel(
        id="Three_TIFF_2024_Films",
        desc="Provide three qualifying films and the required details for each",
        parent=root,
        critical=False,
    )

    # Build verification subtree for each of the three films
    for i, film in enumerate(films):
        await verify_single_film(evaluator, top_node, film, i)

    # Optional: record known festival window as custom info
    evaluator.add_custom_info(
        {"festival_window": "September 5–15, 2024", "festival": "Toronto International Film Festival (TIFF)"},
        info_type="context",
        info_name="festival_info",
    )

    # Return structured summary
    return evaluator.get_summary()