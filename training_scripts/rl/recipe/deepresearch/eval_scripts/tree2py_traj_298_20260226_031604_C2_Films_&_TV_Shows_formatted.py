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
TASK_ID = "tom_holland_july_2026_films"
TASK_DESCRIPTION = (
    "Identify the two major theatrical films starring Tom Holland that are scheduled for release in July 2026. "
    "For each film, provide the following information: (1) The exact release date in July 2026; (2) The title of the film; "
    "(3) The name of the director; (4) The name of the character Tom Holland plays in the film. Additionally, for the film "
    "released earlier in July: (5) Specify whether the film was shot entirely with IMAX film cameras; (6) Identify the actor "
    "who plays the father of Tom Holland's character, and provide that character's name; (7) Name one actress who appears in "
    "both of Tom Holland's July 2026 film releases. Provide a direct link to a reliable source (such as IMDb, Wikipedia, or "
    "an official movie website) that confirms each piece of information."
)

# Ground truth expectations (used for simple equality/matching checks)
EXPECTED_EARLIER = {
    "title": "The Odyssey",
    "release_date": "July 17, 2026",
    "director": "Christopher Nolan",
    "tom_character": "Telemachus",
    "imax_entirely": "shot entirely with IMAX film cameras",
    "father_actor": "Matt Damon",
    "father_character": "Odysseus",
    "shared_actress": "Zendaya",
}

EXPECTED_LATER = {
    "title": "Spider-Man: Brand New Day",
    "release_date": "July 31, 2026",
    "director": "Destin Daniel Cretton",
    "tom_character": "Peter Parker/Spider-Man",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmInfo(BaseModel):
    film_title: Optional[str] = None
    release_date: Optional[str] = None
    director: Optional[str] = None
    tom_character: Optional[str] = None

    sources_title: List[str] = Field(default_factory=list)
    sources_release_date: List[str] = Field(default_factory=list)
    sources_director: List[str] = Field(default_factory=list)
    sources_tom_character: List[str] = Field(default_factory=list)

    # Additional fields for earlier film only
    imax_entirely: Optional[str] = None
    sources_imax: List[str] = Field(default_factory=list)

    father_actor: Optional[str] = None
    father_character: Optional[str] = None
    sources_father: List[str] = Field(default_factory=list)

    shared_actress: Optional[str] = None
    sources_shared_actress: List[str] = Field(default_factory=list)

    # General URLs mentioned for this film (IMDb, Wikipedia, official site, etc.)
    film_urls_general: List[str] = Field(default_factory=list)


class FilmsExtraction(BaseModel):
    """Extraction container for two Tom Holland July 2026 films."""
    earlier_film: Optional[FilmInfo] = None
    later_film: Optional[FilmInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_films() -> str:
    return """
    Extract structured information from the answer about exactly two major theatrical films starring Tom Holland that are scheduled for release in July 2026.
    Classify them as:
    - earlier_film: the one released earlier in July (e.g., July 17, 2026)
    - later_film: the one released later in July (e.g., July 31, 2026)

    For each film, extract:
    - film_title: The official title of the film.
    - release_date: The exact release date in July 2026 (string as in the answer; e.g., "July 17, 2026" or "July 17th, 2026").
    - director: The director's full name.
    - tom_character: The name of the character that Tom Holland plays in the film.
    - sources_title: All URLs in the answer that specifically support the film title.
    - sources_release_date: All URLs in the answer that specifically support the release date.
    - sources_director: All URLs in the answer that specifically support the director information.
    - sources_tom_character: All URLs in the answer that specifically support Tom Holland's character name.
    - film_urls_general: Any reliable film-related URLs mentioned (IMDb, Wikipedia, official sites) even if not tied to a specific field above.

    For the earlier_film in addition, also extract:
    - imax_entirely: A short phrase from the answer indicating whether the film was shot entirely with IMAX film cameras (e.g., "shot entirely with IMAX film cameras"); if absent, return null.
    - sources_imax: URLs that support the IMAX filming claim.
    - father_actor: The actor who plays the father of Tom Holland's character; if absent, return null.
    - father_character: The father's character name; if absent, return null.
    - sources_father: URLs that support the father actor and father character information.
    - shared_actress: The name of one actress who appears in both of Tom Holland's July 2026 films (e.g., "Zendaya"); if absent, return null.
    - sources_shared_actress: URLs that support the shared actress claim across both films.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer text. Do not invent URLs.
    - Accept URLs in plain form or markdown links. Normalize them as full URLs including protocol (prepend http:// if missing).
    - Prefer reliable sources such as IMDb, Wikipedia, or official movie websites if provided.
    - If no URL is provided for a specific field, return an empty list for that field.

    If any required field is missing, set it to null. If no URLs are provided for a field, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _union_sources(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                result.append(url)
    return result


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_earlier_film(
    evaluator: Evaluator,
    parent_node,
    earlier: Optional[FilmInfo],
    later: Optional[FilmInfo],
) -> None:
    # Create node for the earlier film
    earlier_node = evaluator.add_parallel(
        id="Earlier_Film_July_17",
        desc="The first film released on July 17, 2026",
        parent=parent_node,
        critical=True,
    )

    # Required info existence check (critical gate)
    required_info_ok = (
        earlier is not None
        and _is_nonempty(earlier.film_title)
        and _is_nonempty(earlier.release_date)
        and _is_nonempty(earlier.director)
        and _is_nonempty(earlier.tom_character)
    )
    evaluator.add_custom_node(
        result=required_info_ok,
        id="Earlier_Film_Required_Info",
        desc="Earlier film has core required information (title, release date, director, Tom's character)",
        parent=earlier_node,
        critical=True,
    )

    # Core Film Details (parallel, critical)
    core_node = evaluator.add_parallel(
        id="Earlier_Core_Film_Details",
        desc="Correct identification of the film title, release date, director, and Tom Holland's character name",
        parent=earlier_node,
        critical=True,
    )

    # Title: expected match (simple equality)
    title_expected_leaf = evaluator.add_leaf(
        id="Earlier_Title_Expected",
        desc=f"Expected film title is '{EXPECTED_EARLIER['title']}'",
        parent=core_node,
        critical=True,
    )
    extracted_title = earlier.film_title if earlier else ""
    await evaluator.verify(
        claim=f"The extracted earlier film title '{extracted_title}' matches the expected title '{EXPECTED_EARLIER['title']}'.",
        node=title_expected_leaf,
        additional_instruction="Allow minor or reasonable variants (e.g., casing or punctuation). The titles should refer to the same film.",
    )

    # Title: supported by sources
    title_source_leaf = evaluator.add_leaf(
        id="Earlier_Title_Supported",
        desc="Title is supported by reliable sources",
        parent=core_node,
        critical=True,
    )
    title_sources = _union_sources(earlier.sources_title if earlier else [], earlier.film_urls_general if earlier else [])
    await evaluator.verify(
        claim=f"The film's official title is '{EXPECTED_EARLIER['title']}'.",
        node=title_source_leaf,
        sources=title_sources,
        additional_instruction="Check the page to confirm the film title. Prefer IMDb, Wikipedia, or official sites.",
    )

    # Release date: expected match
    date_expected_leaf = evaluator.add_leaf(
        id="Earlier_ReleaseDate_Expected",
        desc=f"Expected release date is '{EXPECTED_EARLIER['release_date']}'",
        parent=core_node,
        critical=True,
    )
    extracted_date = earlier.release_date if earlier else ""
    await evaluator.verify(
        claim=f"The extracted earlier film release date '{extracted_date}' equals '{EXPECTED_EARLIER['release_date']}'.",
        node=date_expected_leaf,
        additional_instruction="Allow minor variants like 'July 17th, 2026'.",
    )

    # Release date: supported by sources
    date_source_leaf = evaluator.add_leaf(
        id="Earlier_ReleaseDate_Supported",
        desc="Release date is supported by reliable sources",
        parent=core_node,
        critical=True,
    )
    date_sources = _union_sources(earlier.sources_release_date if earlier else [], earlier.film_urls_general if earlier else [])
    await evaluator.verify(
        claim=f"The film is scheduled to be released on {EXPECTED_EARLIER['release_date']}.",
        node=date_source_leaf,
        sources=date_sources,
        additional_instruction="Confirm the date shown on IMDb/Wikipedia/official site.",
    )

    # Director: expected match
    director_expected_leaf = evaluator.add_leaf(
        id="Earlier_Director_Expected",
        desc=f"Expected director is '{EXPECTED_EARLIER['director']}'",
        parent=core_node,
        critical=True,
    )
    extracted_director = earlier.director if earlier else ""
    await evaluator.verify(
        claim=f"The extracted earlier film director '{extracted_director}' matches the expected director '{EXPECTED_EARLIER['director']}'.",
        node=director_expected_leaf,
        additional_instruction="Allow minor name variants (e.g., middle names/initials).",
    )

    # Director: supported by sources
    director_source_leaf = evaluator.add_leaf(
        id="Earlier_Director_Supported",
        desc="Director is supported by reliable sources",
        parent=core_node,
        critical=True,
    )
    director_sources = _union_sources(earlier.sources_director if earlier else [], earlier.film_urls_general if earlier else [])
    await evaluator.verify(
        claim=f"The film is directed by {EXPECTED_EARLIER['director']}.",
        node=director_source_leaf,
        sources=director_sources,
        additional_instruction="Verify the director credit on the page.",
    )

    # Tom's character: expected match
    character_expected_leaf = evaluator.add_leaf(
        id="Earlier_TomCharacter_Expected",
        desc=f"Expected Tom Holland character is '{EXPECTED_EARLIER['tom_character']}'",
        parent=core_node,
        critical=True,
    )
    extracted_char = earlier.tom_character if earlier else ""
    await evaluator.verify(
        claim=f"The extracted earlier film character '{extracted_char}' matches the expected character '{EXPECTED_EARLIER['tom_character']}'.",
        node=character_expected_leaf,
        additional_instruction="Allow small variants (e.g., accents or transliteration).",
    )

    # Tom's character: supported by sources
    character_source_leaf = evaluator.add_leaf(
        id="Earlier_TomCharacter_Supported",
        desc="Tom Holland's character is supported by reliable sources",
        parent=core_node,
        critical=True,
    )
    character_sources = _union_sources(earlier.sources_tom_character if earlier else [], earlier.film_urls_general if earlier else [])
    await evaluator.verify(
        claim=f"Tom Holland plays {EXPECTED_EARLIER['tom_character']} in this film.",
        node=character_source_leaf,
        sources=character_sources,
        additional_instruction="Confirm cast/role information for Tom Holland.",
    )

    # Special Production and Cast (parallel, critical)
    special_node = evaluator.add_parallel(
        id="Earlier_Special_Production_Cast",
        desc="Additional production details and key cast information specific to this film",
        parent=earlier_node,
        critical=True,
    )

    # IMAX Production: supported by sources
    imax_leaf = evaluator.add_leaf(
        id="Earlier_IMAX_Production",
        desc="The film is shot entirely with IMAX film cameras",
        parent=special_node,
        critical=True,
    )
    imax_sources = _union_sources(earlier.sources_imax if earlier else [], earlier.film_urls_general if earlier else [])
    await evaluator.verify(
        claim="The film was shot entirely with IMAX film cameras.",
        node=imax_leaf,
        sources=imax_sources,
        additional_instruction="Confirm that the production used IMAX film cameras for the entirety of principal photography.",
    )

    # Father actor expected match
    father_actor_expected_leaf = evaluator.add_leaf(
        id="Earlier_FatherActor_Expected",
        desc=f"Expected father actor is '{EXPECTED_EARLIER['father_actor']}'",
        parent=special_node,
        critical=True,
    )
    extracted_father_actor = (earlier.father_actor if earlier else "") or ""
    await evaluator.verify(
        claim=f"The extracted father actor '{extracted_father_actor}' matches the expected actor '{EXPECTED_EARLIER['father_actor']}'.",
        node=father_actor_expected_leaf,
        additional_instruction="Minor name variations acceptable.",
    )

    # Father character expected match
    father_char_expected_leaf = evaluator.add_leaf(
        id="Earlier_FatherCharacter_Expected",
        desc=f"Expected father character is '{EXPECTED_EARLIER['father_character']}'",
        parent=special_node,
        critical=True,
    )
    extracted_father_char = (earlier.father_character if earlier else "") or ""
    await evaluator.verify(
        claim=f"The extracted father character '{extracted_father_char}' matches the expected character '{EXPECTED_EARLIER['father_character']}'.",
        node=father_char_expected_leaf,
        additional_instruction="Minor variants acceptable.",
    )

    # Father actor plays father role: supported by sources (split into two checks)
    father_actor_source_leaf = evaluator.add_leaf(
        id="Earlier_FatherActor_Supported",
        desc="Matt Damon plays Odysseus in the film",
        parent=special_node,
        critical=True,
    )
    father_sources = _union_sources(earlier.sources_father if earlier else [], earlier.film_urls_general if earlier else [])
    await evaluator.verify(
        claim=f"{EXPECTED_EARLIER['father_actor']} plays {EXPECTED_EARLIER['father_character']} in this film.",
        node=father_actor_source_leaf,
        sources=father_sources,
        additional_instruction="Confirm the casting credit for Matt Damon as Odysseus.",
    )

    father_relation_source_leaf = evaluator.add_leaf(
        id="Earlier_FatherRelation_Supported",
        desc="Odysseus is Telemachus's father in the story",
        parent=special_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{EXPECTED_EARLIER['father_character']} is {EXPECTED_EARLIER['tom_character']}'s father in the story.",
        node=father_relation_source_leaf,
        sources=father_sources,
        additional_instruction="Confirm the relationship (Odysseus is Telemachus's father) based on the story synopsis.",
    )

    # Shared cast member (Zendaya) - confirm presence in both films
    shared_expected_leaf = evaluator.add_leaf(
        id="Earlier_SharedCast_Expected",
        desc=f"Expected shared actress is '{EXPECTED_EARLIER['shared_actress']}'",
        parent=special_node,
        critical=True,
    )
    extracted_shared = (earlier.shared_actress if earlier else "") or ""
    await evaluator.verify(
        claim=f"The extracted shared actress '{extracted_shared}' matches the expected actress '{EXPECTED_EARLIER['shared_actress']}'.",
        node=shared_expected_leaf,
        additional_instruction="Minor name variants acceptable.",
    )

    shared_in_earlier_leaf = evaluator.add_leaf(
        id="Earlier_SharedCast_AppearsInEarlier",
        desc="Zendaya appears in the earlier film",
        parent=special_node,
        critical=True,
    )
    shared_sources_earlier = _union_sources(earlier.sources_shared_actress if earlier else [], earlier.film_urls_general if earlier else [])
    await evaluator.verify(
        claim=f"{EXPECTED_EARLIER['shared_actress']} appears in the earlier film '{EXPECTED_EARLIER['title']}'.",
        node=shared_in_earlier_leaf,
        sources=shared_sources_earlier,
        additional_instruction="Confirm the cast list includes Zendaya for the earlier film.",
    )

    shared_in_later_leaf = evaluator.add_leaf(
        id="Earlier_SharedCast_AppearsInLater",
        desc="Zendaya appears in the later film",
        parent=special_node,
        critical=True,
    )
    # Use later film general URLs to confirm Zendaya in the later film
    later_general_sources = later.film_urls_general if later else []
    # Also use any shared actress sources provided
    shared_sources_later = _union_sources(earlier.sources_shared_actress if earlier else [], later_general_sources)
    await evaluator.verify(
        claim=f"{EXPECTED_EARLIER['shared_actress']} appears in the later film '{EXPECTED_LATER['title']}'.",
        node=shared_in_later_leaf,
        sources=shared_sources_later,
        additional_instruction="Confirm the cast list includes Zendaya for the later film.",
    )

    # URL References (critical): ensure at least one reliable source exists overall for this film
    urls_union = _union_sources(
        earlier.sources_title if earlier else [],
        earlier.sources_release_date if earlier else [],
        earlier.sources_director if earlier else [],
        earlier.sources_tom_character if earlier else [],
        earlier.sources_imax if earlier else [],
        earlier.sources_father if earlier else [],
        earlier.sources_shared_actress if earlier else [],
        earlier.film_urls_general if earlier else [],
    )
    evaluator.add_custom_node(
        result=len(urls_union) > 0,
        id="Earlier_URL_References",
        desc="Direct links to reliable sources exist for the earlier film",
        parent=earlier_node,
        critical=True,
    )


async def verify_later_film(
    evaluator: Evaluator,
    parent_node,
    later: Optional[FilmInfo],
) -> None:
    # Create node for the later film
    later_node = evaluator.add_parallel(
        id="Later_Film_July_31",
        desc="The second film released on July 31, 2026",
        parent=parent_node,
        critical=True,
    )

    # Required info existence check (critical gate)
    required_info_ok = (
        later is not None
        and _is_nonempty(later.film_title)
        and _is_nonempty(later.release_date)
        and _is_nonempty(later.director)
        and _is_nonempty(later.tom_character)
    )
    evaluator.add_custom_node(
        result=required_info_ok,
        id="Later_Film_Required_Info",
        desc="Later film has core required information (title, release date, director, Tom's character)",
        parent=later_node,
        critical=True,
    )

    # Core Film Details (parallel, critical)
    core_node = evaluator.add_parallel(
        id="Later_Core_Film_Details",
        desc="Correct identification of the film title, release date, director, and Tom Holland's character name",
        parent=later_node,
        critical=True,
    )

    # Title: expected match
    title_expected_leaf = evaluator.add_leaf(
        id="Later_Title_Expected",
        desc=f"Expected film title is '{EXPECTED_LATER['title']}'",
        parent=core_node,
        critical=True,
    )
    extracted_title = later.film_title if later else ""
    await evaluator.verify(
        claim=f"The extracted later film title '{extracted_title}' matches the expected title '{EXPECTED_LATER['title']}'.",
        node=title_expected_leaf,
        additional_instruction="Allow minor or reasonable variants.",
    )

    # Title: supported by sources
    title_source_leaf = evaluator.add_leaf(
        id="Later_Title_Supported",
        desc="Title is supported by reliable sources",
        parent=core_node,
        critical=True,
    )
    title_sources = _union_sources(later.sources_title if later else [], later.film_urls_general if later else [])
    await evaluator.verify(
        claim=f"The film's official title is '{EXPECTED_LATER['title']}'.",
        node=title_source_leaf,
        sources=title_sources,
        additional_instruction="Confirm title via IMDb/Wikipedia/official site.",
    )

    # Release date: expected match
    date_expected_leaf = evaluator.add_leaf(
        id="Later_ReleaseDate_Expected",
        desc=f"Expected release date is '{EXPECTED_LATER['release_date']}'",
        parent=core_node,
        critical=True,
    )
    extracted_date = later.release_date if later else ""
    await evaluator.verify(
        claim=f"The extracted later film release date '{extracted_date}' equals '{EXPECTED_LATER['release_date']}'.",
        node=date_expected_leaf,
        additional_instruction="Allow 'July 31st, 2026' variants.",
    )

    # Release date: supported by sources
    date_source_leaf = evaluator.add_leaf(
        id="Later_ReleaseDate_Supported",
        desc="Release date is supported by reliable sources",
        parent=core_node,
        critical=True,
    )
    date_sources = _union_sources(later.sources_release_date if later else [], later.film_urls_general if later else [])
    await evaluator.verify(
        claim=f"The film is scheduled to be released on {EXPECTED_LATER['release_date']}.",
        node=date_source_leaf,
        sources=date_sources,
        additional_instruction="Confirm on a reliable page.",
    )

    # Director: expected match
    director_expected_leaf = evaluator.add_leaf(
        id="Later_Director_Expected",
        desc=f"Expected director is '{EXPECTED_LATER['director']}'",
        parent=core_node,
        critical=True,
    )
    extracted_director = later.director if later else ""
    await evaluator.verify(
        claim=f"The extracted later film director '{extracted_director}' matches the expected director '{EXPECTED_LATER['director']}'.",
        node=director_expected_leaf,
        additional_instruction="Minor variants acceptable.",
    )

    # Director: supported by sources
    director_source_leaf = evaluator.add_leaf(
        id="Later_Director_Supported",
        desc="Director is supported by reliable sources",
        parent=core_node,
        critical=True,
    )
    director_sources = _union_sources(later.sources_director if later else [], later.film_urls_general if later else [])
    await evaluator.verify(
        claim=f"The film is directed by {EXPECTED_LATER['director']}.",
        node=director_source_leaf,
        sources=director_sources,
        additional_instruction="Confirm director credit.",
    )

    # Tom's character: expected match
    character_expected_leaf = evaluator.add_leaf(
        id="Later_TomCharacter_Expected",
        desc=f"Expected Tom Holland character is '{EXPECTED_LATER['tom_character']}'",
        parent=core_node,
        critical=True,
    )
    extracted_char = later.tom_character if later else ""
    await evaluator.verify(
        claim=f"The extracted later film character '{extracted_char}' matches the expected character '{EXPECTED_LATER['tom_character']}'.",
        node=character_expected_leaf,
        additional_instruction="Allow 'Peter Parker' vs 'Spider-Man' combined mentions.",
    )

    # Tom's character: supported by sources
    character_source_leaf = evaluator.add_leaf(
        id="Later_TomCharacter_Supported",
        desc="Tom Holland's character is supported by reliable sources",
        parent=core_node,
        critical=True,
    )
    character_sources = _union_sources(later.sources_tom_character if later else [], later.film_urls_general if later else [])
    await evaluator.verify(
        claim=f"Tom Holland plays {EXPECTED_LATER['tom_character']} in this film.",
        node=character_source_leaf,
        sources=character_sources,
        additional_instruction="Confirm cast/role information.",
    )

    # URL References (critical): ensure at least one reliable source exists overall for this film
    urls_union = _union_sources(
        later.sources_title if later else [],
        later.sources_release_date if later else [],
        later.sources_director if later else [],
        later.sources_tom_character if later else [],
        later.film_urls_general if later else [],
    )
    evaluator.add_custom_node(
        result=len(urls_union) > 0,
        id="Later_URL_References",
        desc="Direct links to reliable sources exist for the later film",
        parent=later_node,
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
) -> Dict:
    """
    Evaluate an answer for Tom Holland July 2026 films task.
    """
    # Initialize evaluator (root non-critical to allow child critical gating)
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

    # Extract films data
    films_data = await evaluator.extract(
        prompt=prompt_extract_films(),
        template_class=FilmsExtraction,
        extraction_name="films_extraction",
    )

    # Optional global check: both films in July 2026 (critical gate under root)
    earlier_date = films_data.earlier_film.release_date if films_data.earlier_film else None
    later_date = films_data.later_film.release_date if films_data.later_film else None
    both_july_2026 = (
        _is_nonempty(earlier_date) and "July" in earlier_date and "2026" in earlier_date
        and _is_nonempty(later_date) and "July" in later_date and "2026" in later_date
    )
    evaluator.add_custom_node(
        result=both_july_2026,
        id="Both_Films_July_2026",
        desc="Both films have release dates in July 2026",
        parent=root,
        critical=True,
    )

    # Build verification subtrees
    await verify_earlier_film(evaluator, root, films_data.earlier_film, films_data.later_film)
    await verify_later_film(evaluator, root, films_data.later_film)

    # Return summary
    return evaluator.get_summary()