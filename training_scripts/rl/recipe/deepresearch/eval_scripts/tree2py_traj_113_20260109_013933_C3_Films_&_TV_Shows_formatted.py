import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "film_palme_best_picture_2019_2020"
TASK_DESCRIPTION = (
    "Identify the film that won both the Palme d'Or at the 72nd Cannes Film Festival in 2019 and the "
    "Academy Award for Best Picture at the 92nd Academy Awards. For this film, provide the following "
    "information with supporting URL references from reliable sources: (1) the film title and director's name, "
    "(2) the theatrical runtime in minutes, (3) at least one primary production company, "
    "(4) the US theatrical distributor and the US theatrical release date, and (5) the primary language of the film's dialogue."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FilmInfoExtraction(BaseModel):
    # Identification
    film_title: Optional[str] = None
    film_title_urls: List[str] = Field(default_factory=list)

    director_name: Optional[str] = None
    director_urls: List[str] = Field(default_factory=list)

    # Awards proof
    palme_dor_urls: List[str] = Field(default_factory=list)
    best_picture_urls: List[str] = Field(default_factory=list)

    # Attributes
    runtime_minutes: Optional[str] = None
    runtime_urls: List[str] = Field(default_factory=list)

    production_companies: List[str] = Field(default_factory=list)
    production_company_urls: List[str] = Field(default_factory=list)

    us_distributor: Optional[str] = None
    distributor_urls: List[str] = Field(default_factory=list)

    us_release_date: Optional[str] = None
    release_date_urls: List[str] = Field(default_factory=list)

    primary_language: Optional[str] = None
    language_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_film_info() -> str:
    return """
    Extract information for the single film in the answer that is claimed to have won BOTH:
    • the Palme d'Or at the 72nd Cannes Film Festival (2019), and
    • the Academy Award for Best Picture at the 92nd Academy Awards (Oscars).
    
    Extract EXACTLY what is stated in the answer text. Do NOT invent anything. If something is missing, return null or an empty list as appropriate.

    Return a JSON object with the following fields:
    - film_title: string | null
    - film_title_urls: array of URL strings (all URLs in the answer that directly support the film title); can be empty
    - director_name: string | null
    - director_urls: array of URL strings supporting the director's name; can be empty

    - palme_dor_urls: array of URL strings that support the film's Palme d'Or win at the 72nd Cannes Film Festival (2019); can be empty
    - best_picture_urls: array of URL strings that support the film's Best Picture win at the 92nd Academy Awards; can be empty

    - runtime_minutes: string | null  (the theatrical runtime in minutes as written in the answer, e.g., "132")
    - runtime_urls: array of URL strings supporting the runtime; can be empty

    - production_companies: array of strings (primary production companies, list all mentioned; can be empty)
    - production_company_urls: array of URL strings supporting the production companies; can be empty

    - us_distributor: string | null  (the US theatrical distributor)
    - distributor_urls: array of URL strings supporting the distributor; can be empty

    - us_release_date: string | null  (US theatrical release date as written in the answer, any standard format)
    - release_date_urls: array of URL strings supporting the US release date; can be empty

    - primary_language: string | null  (primary dialogue language)
    - language_urls: array of URL strings supporting the primary language; can be empty

    SPECIAL URL RULES:
    - Extract actual URLs shown in the answer (plain or markdown link targets). Ignore domain mentions without URLs.
    - Include full URLs with protocol. If missing, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0 and any(u.strip() for u in urls))


def _first_company(companies: List[str]) -> Optional[str]:
    for c in companies:
        if _non_empty_str(c):
            return c.strip()
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identification_and_awards(
    evaluator: Evaluator,
    parent_node,
    info: FilmInfoExtraction,
) -> None:
    """
    Build 'Film_Identification_And_Awards' subtree with all critical checks.
    """
    ident_node = evaluator.add_parallel(
        id="Film_Identification_And_Awards",
        desc="Correctly identify the film and verify both awards with supporting reliable URL references (URLs may be separate).",
        parent=parent_node,
        critical=True,
    )

    # Film title existence checks
    evaluator.add_custom_node(
        result=_non_empty_str(info.film_title),
        id="Film_Title_Value_Exists",
        desc="Film title value is provided in the answer",
        parent=ident_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_sources(info.film_title_urls),
        id="Film_Title_Sources_Exist",
        desc="At least one URL is provided to support the film title",
        parent=ident_node,
        critical=True,
    )

    # Film title verification leaf
    title_leaf = evaluator.add_leaf(
        id="Film_Title_With_URL",
        desc="Provide the correct film title, supported by at least one reliable URL reference.",
        parent=ident_node,
        critical=True,
    )
    title_claim = f"The correct film title is '{info.film_title or ''}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=info.film_title_urls,
        additional_instruction=(
            "Verify that the provided URL(s) clearly identify the film and confirm the exact title spelling. "
            "Allow minor punctuation or capitalization variations."
        ),
    )

    # Director existence checks
    evaluator.add_custom_node(
        result=_non_empty_str(info.director_name),
        id="Director_Name_Value_Exists",
        desc="Director name value is provided in the answer",
        parent=ident_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_sources(info.director_urls),
        id="Director_Name_Sources_Exist",
        desc="At least one URL is provided to support the director name",
        parent=ident_node,
        critical=True,
    )

    # Director verification leaf
    director_leaf = evaluator.add_leaf(
        id="Director_Name_With_URL",
        desc="Provide the correct director's name, supported by at least one reliable URL reference.",
        parent=ident_node,
        critical=True,
    )
    director_claim = f"The director of the film '{info.film_title or ''}' is '{info.director_name or ''}'."
    await evaluator.verify(
        claim=director_claim,
        node=director_leaf,
        sources=info.director_urls,
        additional_instruction=(
            "Verify that the URL(s) explicitly state the film's director. "
            "Allow minor name variations (middle names/initials, diacritics, capitalization)."
        ),
    )

    # Palme d'Or existence check
    evaluator.add_custom_node(
        result=_has_sources(info.palme_dor_urls),
        id="Palme_dOr_Sources_Exist",
        desc="At least one URL confirms the film won the Palme d'Or (2019, 72nd Cannes)",
        parent=ident_node,
        critical=True,
    )

    # Palme d'Or verification leaf
    palme_leaf = evaluator.add_leaf(
        id="Palme_dOr_Win_With_URL",
        desc="Provide at least one reliable URL reference confirming the film won the Palme d'Or at the 72nd Cannes Film Festival (2019).",
        parent=ident_node,
        critical=True,
    )
    palme_claim = (
        f"The film '{info.film_title or ''}' won the Palme d'Or at the 72nd Cannes Film Festival in 2019."
    )
    await evaluator.verify(
        claim=palme_claim,
        node=palme_leaf,
        sources=info.palme_dor_urls,
        additional_instruction=(
            "Verify that the URL(s) explicitly confirm a Palme d'Or win at the 72nd Cannes Film Festival (2019) "
            "for the specified film."
        ),
    )

    # Best Picture existence check
    evaluator.add_custom_node(
        result=_has_sources(info.best_picture_urls),
        id="Best_Picture_Sources_Exist",
        desc="At least one URL confirms the film won Best Picture at the 92nd Academy Awards",
        parent=ident_node,
        critical=True,
    )

    # Best Picture verification leaf
    best_pic_leaf = evaluator.add_leaf(
        id="Best_Picture_Win_With_URL",
        desc="Provide at least one reliable URL reference confirming the film won the Academy Award for Best Picture at the 92nd Academy Awards.",
        parent=ident_node,
        critical=True,
    )
    best_pic_claim = (
        f"The film '{info.film_title or ''}' won the Academy Award for Best Picture at the 92nd Academy Awards."
    )
    await evaluator.verify(
        claim=best_pic_claim,
        node=best_pic_leaf,
        sources=info.best_picture_urls,
        additional_instruction=(
            "Verify that the URL(s) explicitly confirm a Best Picture win at the 92nd Academy Awards (Oscars) "
            "for the specified film."
        ),
    )


async def build_film_attributes(
    evaluator: Evaluator,
    parent_node,
    info: FilmInfoExtraction,
) -> None:
    """
    Build 'Film_Attributes' subtree with all critical checks.
    """
    attr_node = evaluator.add_parallel(
        id="Film_Attributes",
        desc="Provide the required film attributes, each supported by at least one reliable URL reference.",
        parent=parent_node,
        critical=True,
    )

    # Runtime
    evaluator.add_custom_node(
        result=_non_empty_str(info.runtime_minutes),
        id="Runtime_Value_Exists",
        desc="Runtime (minutes) value is provided in the answer",
        parent=attr_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_sources(info.runtime_urls),
        id="Runtime_Sources_Exist",
        desc="At least one URL supports the runtime value",
        parent=attr_node,
        critical=True,
    )

    runtime_leaf = evaluator.add_leaf(
        id="Runtime_Minutes_With_URL",
        desc="Provide the theatrical runtime in minutes, supported by at least one reliable URL reference.",
        parent=attr_node,
        critical=True,
    )
    runtime_claim = (
        f"The theatrical runtime of the film '{info.film_title or ''}' is {info.runtime_minutes or ''} minutes."
    )
    await evaluator.verify(
        claim=runtime_claim,
        node=runtime_leaf,
        sources=info.runtime_urls,
        additional_instruction=(
            "Confirm the film's theatrical runtime in minutes exactly or equivalently (e.g., '132 min', '2h 12m' "
            "is equivalent to 132). Prefer official or authoritative sources when available."
        ),
    )

    # Production company (verify at least one)
    first_company = _first_company(info.production_companies)
    evaluator.add_custom_node(
        result=_non_empty_str(first_company),
        id="Production_Company_Value_Exists",
        desc="At least one primary production company is provided in the answer",
        parent=attr_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_sources(info.production_company_urls),
        id="Production_Company_Sources_Exist",
        desc="At least one URL supports the production company information",
        parent=attr_node,
        critical=True,
    )

    prod_leaf = evaluator.add_leaf(
        id="Production_Company_With_URL",
        desc="Provide at least one primary production company, supported by at least one reliable URL reference.",
        parent=attr_node,
        critical=True,
    )
    prod_claim = (
        f"One of the primary production companies of the film '{info.film_title or ''}' is '{first_company or ''}'."
    )
    await evaluator.verify(
        claim=prod_claim,
        node=prod_leaf,
        sources=info.production_company_urls,
        additional_instruction=(
            "Verify that the URL(s) explicitly list the specified company as a production company for the film."
        ),
    )

    # US Theatrical Distributor
    evaluator.add_custom_node(
        result=_non_empty_str(info.us_distributor),
        id="US_Distributor_Value_Exists",
        desc="US theatrical distributor value is provided in the answer",
        parent=attr_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_sources(info.distributor_urls),
        id="US_Distributor_Sources_Exist",
        desc="At least one URL supports the US theatrical distributor",
        parent=attr_node,
        critical=True,
    )

    dist_leaf = evaluator.add_leaf(
        id="US_Theatrical_Distributor_With_URL",
        desc="Provide the US theatrical distributor, supported by at least one reliable URL reference.",
        parent=attr_node,
        critical=True,
    )
    dist_claim = (
        f"The US theatrical distributor of the film '{info.film_title or ''}' is '{info.us_distributor or ''}'."
    )
    await evaluator.verify(
        claim=dist_claim,
        node=dist_leaf,
        sources=info.distributor_urls,
        additional_instruction=(
            "Verify that the URL(s) explicitly identify the US theatrical distributor of the film."
        ),
    )

    # US Theatrical Release Date
    evaluator.add_custom_node(
        result=_non_empty_str(info.us_release_date),
        id="US_Release_Date_Value_Exists",
        desc="US theatrical release date value is provided in the answer",
        parent=attr_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_sources(info.release_date_urls),
        id="US_Release_Date_Sources_Exist",
        desc="At least one URL supports the US theatrical release date",
        parent=attr_node,
        critical=True,
    )

    rel_leaf = evaluator.add_leaf(
        id="US_Theatrical_Release_Date_With_URL",
        desc="Provide the US theatrical release date, supported by at least one reliable URL reference.",
        parent=attr_node,
        critical=True,
    )
    rel_claim = (
        f"The US theatrical release date of the film '{info.film_title or ''}' was '{info.us_release_date or ''}'."
    )
    await evaluator.verify(
        claim=rel_claim,
        node=rel_leaf,
        sources=info.release_date_urls,
        additional_instruction=(
            "Verify that the URL(s) give the US theatrical release date for the film. "
            "Allow minor formatting differences (e.g., 'Oct 11, 2019' vs '2019-10-11')."
        ),
    )

    # Primary Language
    evaluator.add_custom_node(
        result=_non_empty_str(info.primary_language),
        id="Primary_Language_Value_Exists",
        desc="Primary dialogue language value is provided in the answer",
        parent=attr_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_sources(info.language_urls),
        id="Primary_Language_Sources_Exist",
        desc="At least one URL supports the primary language claim",
        parent=attr_node,
        critical=True,
    )

    lang_leaf = evaluator.add_leaf(
        id="Primary_Language_With_URL",
        desc="Provide the primary language of the film's dialogue, supported by at least one reliable URL reference.",
        parent=attr_node,
        critical=True,
    )
    lang_claim = (
        f"The primary language of the film '{info.film_title or ''}' is '{info.primary_language or ''}'."
    )
    await evaluator.verify(
        claim=lang_claim,
        node=lang_leaf,
        sources=info.language_urls,
        additional_instruction=(
            "Verify that the URL(s) explicitly indicate the film's primary dialogue language."
        ),
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
    Evaluate an answer for the film identification and attribute verification task.
    """
    # Initialize evaluator (root is non-critical by design; we add a critical main node under it)
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
        default_model=model,
    )

    # Extract all necessary information from the answer
    film_info = await evaluator.extract(
        prompt=prompt_extract_film_info(),
        template_class=FilmInfoExtraction,
        extraction_name="film_info_extraction",
    )

    # Build the main critical sequential node representing the complete verification flow
    main_node = evaluator.add_sequential(
        id="Complete_Film_Verification",
        desc="Verify all required details of the film that won both the Palme d'Or at the 72nd Cannes Film Festival (2019) and the Academy Award for Best Picture at the 92nd Academy Awards, with reliable URL support.",
        parent=root,
        critical=True,
    )

    # Subtree 1: Identification and Awards
    await build_identification_and_awards(evaluator, main_node, film_info)

    # Subtree 2: Film Attributes
    await build_film_attributes(evaluator, main_node, film_info)

    # Optionally add reference ground truth (not used for scoring)
    evaluator.add_ground_truth({
        "hint_expected_film": "Parasite (2019)",
        "hint_director": "Bong Joon-ho",
        "hint_us_distributor": "Neon",
        "note": "Ground truth hints are informational only and not used for scoring."
    })

    # Return the summarized evaluation
    return evaluator.get_summary()