import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "films_2024_2025_cast"
TASK_DESCRIPTION = """Find four films released or scheduled for release between 2024 and 2025 that meet the following requirements:

1. Film 1: A film released in 2024 featuring Timothée Chalamet as a lead actor. Provide the director's name, release date (month and year), at least three main cast members with their character names (including Timothée Chalamet), and a link to the film's IMDb page.

2. Film 2: A film released or scheduled for release in 2025 featuring Timothée Chalamet as a lead actor. Provide the director's name, release date (month and year), at least three main cast members with their character names (including Timothée Chalamet), and a link to the film's IMDb page.

3. Film 3: A film released or scheduled for release in 2025 featuring Mia Goth in the cast. Provide the director's name, release date (month and year), at least three main cast members with their character names (including Mia Goth), and a link to the film's IMDb page.

4. Film 4: A film released or scheduled for release in 2025 featuring Hailee Steinfeld in the cast. Provide the director's name, release date (month and year), runtime (in hours and minutes), at least three main cast members with their character names (including Hailee Steinfeld), and a link to the film's IMDb page.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CastMember(BaseModel):
    actor_name: Optional[str] = None
    character_name: Optional[str] = None


class FilmInfo(BaseModel):
    title: Optional[str] = None
    director: Optional[str] = None
    release_date: Optional[str] = None  # Month and Year (string)
    runtime: Optional[str] = None       # e.g., "2h 22m" or "142 minutes"
    cast: List[CastMember] = Field(default_factory=list)
    imdb_url: Optional[str] = None


class FilmsExtraction(BaseModel):
    film1: Optional[FilmInfo] = None  # 2024 with Timothée Chalamet (lead)
    film2: Optional[FilmInfo] = None  # 2025 with Timothée Chalamet (lead)
    film3: Optional[FilmInfo] = None  # 2025 with Mia Goth (in cast)
    film4: Optional[FilmInfo] = None  # 2025 with Hailee Steinfeld (in cast) + runtime


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_films() -> str:
    return (
        "Extract four films mentioned in the answer, organized exactly as film1, film2, film3, film4 with the following mapping:\n"
        "- film1: A film released in 2024 featuring Timothée Chalamet as a lead actor.\n"
        "- film2: A film released or scheduled for release in 2025 featuring Timothée Chalamet as a lead actor.\n"
        "- film3: A film released or scheduled for release in 2025 featuring Mia Goth in the cast.\n"
        "- film4: A film released or scheduled for release in 2025 featuring Hailee Steinfeld in the cast.\n\n"
        "For each film, extract the following fields exactly as presented in the answer:\n"
        "1) title: The film's title\n"
        "2) director: The film's director name(s)\n"
        "3) release_date: The release month and year (string). If scheduled, provide the scheduled month and year as stated.\n"
        "4) runtime: The runtime in hours and minutes (e.g., '2h 22m' or '142 minutes'). This is required only for film4; for other films, include it only if present; otherwise return null.\n"
        "5) cast: At least three main cast members with their character names. Each entry must include actor_name and character_name. If fewer than three are provided in the answer, extract whatever is available.\n"
        "6) imdb_url: A link (URL) to the film's IMDb page. If no IMDb link is provided, return null.\n\n"
        "Important:\n"
        "- Only extract information explicitly present in the answer; do not invent missing parts.\n"
        "- For URLs, extract the actual URLs (including protocol). If multiple URLs are provided, choose the IMDb film title page URL.\n"
        "- If the answer mentions more than one candidate for a slot (e.g., multiple 2025 films), use the first one that fits the criteria based on the answer.\n"
        "- If any required field is missing, set it to null.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def format_cast_pairs(cast: List[CastMember], max_items: int = 6) -> str:
    """Format cast list as 'Actor as Character' pairs, limited to max_items."""
    pairs = []
    for cm in cast[:max_items]:
        actor = (cm.actor_name or "").strip()
        character = (cm.character_name or "").strip()
        if actor and character:
            pairs.append(f"{actor} as {character}")
        elif actor:
            pairs.append(f"{actor} (character name not provided)")
        elif character:
            pairs.append(f"(unknown actor) as {character}")
    return "; ".join(pairs)


# --------------------------------------------------------------------------- #
# Verification logic per film                                                 #
# --------------------------------------------------------------------------- #
async def verify_film(
    evaluator: Evaluator,
    parent_node,
    film_info: Optional[FilmInfo],
    film_id_prefix: str,
    film_parent_desc: str,
    required_year: Optional[int],
    required_actor: Optional[str],
    require_lead: bool,
    require_runtime: bool,
) -> None:
    """
    Build verification subtree for a single film and run checks.
    """
    # Create film parent node (parallel aggregation, non-critical to allow partial credit across films)
    film_node = evaluator.add_parallel(
        id=film_id_prefix,
        desc=film_parent_desc,
        parent=parent_node,
        critical=False,
    )

    # Basic existence check (title + IMDb URL are foundational)
    has_title = bool(film_info and film_info.title and film_info.title.strip())
    has_imdb = bool(film_info and film_info.imdb_url and film_info.imdb_url.strip())
    evaluator.add_custom_node(
        result=has_title and has_imdb,
        id=f"{film_id_prefix}_required_info",
        desc="Required info is provided: film title and IMDb URL",
        parent=film_node,
        critical=True
    )

    # IMDb URL validity & correspondence to the film title
    imdb_node = evaluator.add_leaf(
        id=f"{film_id_prefix}_imdb_url",
        desc="Provide a link to the film's IMDb page",
        parent=film_node,
        critical=True
    )
    imdb_claim_title = (film_info.title or "").strip() if film_info else ""
    imdb_url_source = (film_info.imdb_url or "").strip() if film_info else ""
    await evaluator.verify(
        claim=f"This URL is the IMDb page for the film titled '{imdb_claim_title}'.",
        node=imdb_node,
        sources=imdb_url_source,
        additional_instruction="Verify the URL is a valid IMDb title page for the stated film title. Allow minor punctuation/casing differences."
    )

    # Director verification
    director_node = evaluator.add_leaf(
        id=f"{film_id_prefix}_director",
        desc="Provide the film's director name",
        parent=film_node,
        critical=True
    )
    dir_claim = f"The director of the film '{imdb_claim_title}' is '{(film_info.director or '').strip()}'." if film_info else "The director is provided."
    await evaluator.verify(
        claim=dir_claim,
        node=director_node,
        sources=imdb_url_source,
        additional_instruction="Check the 'Director' or 'Directors' field on IMDb. Accept minor formatting variations (e.g., commas for multiple directors)."
    )

    # Release date (month and year) verification and year constraint
    release_node = evaluator.add_leaf(
        id=f"{film_id_prefix}_release_date",
        desc="Provide the film's release date with month and year",
        parent=film_node,
        critical=True
    )
    rel_text = (film_info.release_date or "").strip() if film_info else ""
    release_claim = f"The film '{imdb_claim_title}' was released or is scheduled for release in {rel_text}."
    add_instr_rel = "Verify the month and year against the IMDb page. "
    if required_year is not None:
        add_instr_rel += (
            f"Additionally, ensure the year shown on IMDb is {required_year}. "
            f"If multiple dates are shown, prefer the primary theatrical release or the announced schedule; "
            f"if the IMDb page indicates a scheduled release, that is acceptable."
        )
    await evaluator.verify(
        claim=release_claim,
        node=release_node,
        sources=imdb_url_source,
        additional_instruction=add_instr_rel
    )

    # Runtime verification (only required for film4)
    if require_runtime:
        runtime_node = evaluator.add_leaf(
            id=f"{film_id_prefix}_runtime",
            desc="Provide the film's runtime in hours and minutes",
            parent=film_node,
            critical=True
        )
        runtime_text = (film_info.runtime or "").strip() if film_info else ""
        runtime_claim = f"The runtime of the film '{imdb_claim_title}' is '{runtime_text}'."
        await evaluator.verify(
            claim=runtime_claim,
            node=runtime_node,
            sources=imdb_url_source,
            additional_instruction="Verify the runtime value from IMDb. Accept equivalence like '142 minutes' ≈ '2h 22m'. Minor formatting differences are acceptable."
        )

    # Cast verification: at least three main cast with character names and includes the required actor
    cast_node = evaluator.add_leaf(
        id=f"{film_id_prefix}_cast",
        desc=(
            "Provide at least three main cast members with their character names, "
            f"including {required_actor}" if required_actor else
            "Provide at least three main cast members with their character names"
        ),
        parent=film_node,
        critical=True
    )
    cast_pairs_text = format_cast_pairs(film_info.cast if film_info else [], max_items=8)
    include_text = f"This set includes {required_actor}." if required_actor else "This set includes all provided actors."
    cast_claim = (
        f"The film '{imdb_claim_title}' features the following main cast with character names: {cast_pairs_text}. {include_text}"
    )
    add_instr_cast = (
        "Verify that each 'Actor as Character' pair appears on IMDb for this film and that there are at least three valid pairs. "
        "Accept voice roles or variants. "
    )
    if required_actor:
        if require_lead:
            add_instr_cast += (
                f"Also verify that {required_actor} is a lead/main star (top-billed) for this film, not merely a minor role."
            )
        else:
            add_instr_cast += (
                f"Also verify that {required_actor} appears in the cast (role presence is sufficient)."
            )
    await evaluator.verify(
        claim=cast_claim,
        node=cast_node,
        sources=imdb_url_source,
        additional_instruction=add_instr_cast
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
    Evaluate an answer for the 2024-2025 films task with cast constraints.
    """
    # Initialize evaluator (root should be non-critical to allow partial scoring across films)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find four films released or scheduled for release between 2024-2025, meeting specific cast requirements, and provide detailed information for each",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured information for the four films
    films = await evaluator.extract(
        prompt=prompt_extract_films(),
        template_class=FilmsExtraction,
        extraction_name="films_extraction"
    )

    # Build and verify each film subtree according to rubric
    # Film 1: Timothée Chalamet lead, released in 2024
    await verify_film(
        evaluator=evaluator,
        parent_node=root,
        film_info=films.film1,
        film_id_prefix="film_1_timothee_2024",
        film_parent_desc="A film released in 2024 featuring Timothée Chalamet as a lead actor",
        required_year=2024,
        required_actor="Timothée Chalamet",
        require_lead=True,
        require_runtime=False
    )

    # Film 2: Timothée Chalamet lead, released/scheduled in 2025
    await verify_film(
        evaluator=evaluator,
        parent_node=root,
        film_info=films.film2,
        film_id_prefix="film_2_timothee_2025",
        film_parent_desc="A film released or scheduled for release in 2025 featuring Timothée Chalamet as a lead actor",
        required_year=2025,
        required_actor="Timothée Chalamet",
        require_lead=True,
        require_runtime=False
    )

    # Film 3: Mia Goth in cast, released/scheduled in 2025
    await verify_film(
        evaluator=evaluator,
        parent_node=root,
        film_info=films.film3,
        film_id_prefix="film_3_mia_goth",
        film_parent_desc="A film released or scheduled for release in 2025 featuring Mia Goth in the cast",
        required_year=2025,
        required_actor="Mia Goth",
        require_lead=False,
        require_runtime=False
    )

    # Film 4: Hailee Steinfeld in cast, released/scheduled in 2025, runtime required
    await verify_film(
        evaluator=evaluator,
        parent_node=root,
        film_info=films.film4,
        film_id_prefix="film_4_hailee_steinfeld",
        film_parent_desc="A film released or scheduled for release in 2025 featuring Hailee Steinfeld in the cast",
        required_year=2025,
        required_actor="Hailee Steinfeld",
        require_lead=False,
        require_runtime=True
    )

    # Return standardized evaluation summary
    return evaluator.get_summary()