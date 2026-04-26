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
TASK_ID = "entertainment_milestones_1995_2022"
TASK_DESCRIPTION = (
    "Identify three distinct entertainment industry milestones from the period 1995-2022.\n\n"
    "Milestone 1 - Award Achievement: Identify a film director who won the Academy Award for Best Adapted Screenplay "
    "at the ceremony held in February 2016. Provide: (a) the director's name, (b) the exact date of the ceremony, "
    "(c) the title of the film for which the award was won, and (d) the release year of that film.\n\n"
    "Milestone 2 - Television Debut: Identify an actress who made her first appearance in a television series role as "
    "a replacement for another actress, with this debut occurring in an episode that aired in January 2020. Provide: "
    "(a) the actress's name, (b) the name of the television series, (c) the character name she portrayed, "
    "(d) the season number, (e) the episode number, (f) the episode title, and (g) the exact air date of the episode.\n\n"
    "Milestone 3 - Landmark Film Production: Identify a landmark animated film released theatrically in the United "
    "States in 1995 that holds the distinction of being the first fully computer-animated feature film. Provide: "
    "(a) the film title, (b) the production company, (c) the director's name, and (d) the exact U.S. theatrical release date.\n\n"
    "For each milestone, include supporting reference URLs that verify the provided information."
)

# Expected fixed constraints embedded in rubric
EXPECTED_OSCARS_CEREMONY_DATE = "February 28, 2016"
EXPECTED_OSCARS_CATEGORY = "Best Adapted Screenplay"
EXPECTED_FILM_RELEASE_YEAR_CASE1 = "2015"

EXPECTED_TV_SEASON = "3"
EXPECTED_TV_EPISODE = "9"
EXPECTED_TV_AIR_DATE = "January 17, 2020"

EXPECTED_LANDMARK_PRODUCTION_COMPANY = "Pixar Animation Studios"
EXPECTED_LANDMARK_DIRECTOR = "John Lasseter"
EXPECTED_LANDMARK_US_RELEASE_DATE = "November 22, 1995"
EXPECTED_LANDMARK_SIGNIFICANCE = "first fully computer-animated feature film"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Milestone1Award(BaseModel):
    director_name: Optional[str] = None
    award_category: Optional[str] = None  # e.g., "Best Adapted Screenplay"
    ceremony_date: Optional[str] = None   # e.g., "February 28, 2016"
    film_title: Optional[str] = None
    film_release_year: Optional[str] = None  # keep as string for robustness
    sources: List[str] = Field(default_factory=list)


class Milestone2TVDebut(BaseModel):
    actress_name: Optional[str] = None
    series_name: Optional[str] = None
    character_name: Optional[str] = None
    season_number: Optional[str] = None   # keep as string
    episode_number: Optional[str] = None  # keep as string
    episode_title: Optional[str] = None
    air_date: Optional[str] = None        # e.g., "January 17, 2020"
    replaced_actress_name: Optional[str] = None  # if specified
    replacement_description: Optional[str] = None  # brief text if present
    sources: List[str] = Field(default_factory=list)


class Milestone3LandmarkFilm(BaseModel):
    film_title: Optional[str] = None
    production_company: Optional[str] = None
    director: Optional[str] = None
    us_release_date: Optional[str] = None
    significance: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AllMilestonesExtraction(BaseModel):
    award: Optional[Milestone1Award] = None
    tv_debut: Optional[Milestone2TVDebut] = None
    landmark_film: Optional[Milestone3LandmarkFilm] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_milestones() -> str:
    return """
    Extract three milestones from the answer, one for each category below. Return null for any field not present in the answer. 
    IMPORTANT: Extract all reference URLs explicitly present in the answer text that support each milestone. 
    Do NOT invent any URLs. If multiple URLs are present, include all of them.

    1) Milestone 1 – Award Achievement (Academy Awards, February 2016):
       Fields:
       - director_name: the person's full name (they must be a film director).
       - award_category: the award category (e.g., "Best Adapted Screenplay").
       - ceremony_date: the exact date of the ceremony (e.g., "February 28, 2016").
       - film_title: the title of the winning film associated with this award.
       - film_release_year: the release year of that film (as 4-digit string, e.g., "2015").
       - sources: an array of all URLs cited for this milestone.

    2) Milestone 2 – Television Debut (replacement casting; aired in January 2020):
       Fields:
       - actress_name: full name of the actress who debuted in this role.
       - series_name: the television series name.
       - character_name: the character name portrayed in the debut.
       - season_number: season number (string, e.g., "3").
       - episode_number: episode number (string, e.g., "9").
       - episode_title: the episode title (exact text as given).
       - air_date: the exact air date of the debut episode (e.g., "January 17, 2020"; must be in January 2020).
       - replaced_actress_name: the name of the actress she replaced, if explicitly stated.
       - replacement_description: brief textual phrase about being a replacement, if present.
       - sources: an array of all URLs cited for this milestone.

    3) Milestone 3 – Landmark Film Production (1995; first fully computer-animated feature film; theatrical U.S. release):
       Fields:
       - film_title: the film title.
       - production_company: production company name.
       - director: director's name.
       - us_release_date: exact U.S. theatrical release date (e.g., "November 22, 1995").
       - significance: a phrase confirming it is the first fully computer-animated feature film.
       - sources: an array of all URLs cited for this milestone.

    Notes:
    - Always prefer strings. Do not coerce numbers; keep season and episode as strings.
    - For sources, return only valid URLs explicitly present in the answer; include all that apply.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_text(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _has_sources(urls: Optional[List[str]]) -> bool:
    return isinstance(urls, list) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_case_award(evaluator: Evaluator, parent_node, award: Optional[Milestone1Award]) -> None:
    # Case node (parallel)
    case_node = evaluator.add_parallel(
        id="Case_1_Award_Winner",
        desc="A film director who won an Academy Award for Adapted Screenplay at a ceremony held in February 2016.",
        parent=parent_node,
        critical=False
    )

    director_name = award.director_name if award else None
    award_category = award.award_category if award else None
    ceremony_date = award.ceremony_date if award else None
    film_title = award.film_title if award else None
    film_year = award.film_release_year if award else None
    sources = award.sources if award else []

    # Existence gate
    existence_ok = (
        award is not None and
        _has_nonempty_text(director_name) and
        _has_nonempty_text(film_title) and
        _has_sources(sources)
    )
    evaluator.add_custom_node(
        result=existence_ok,
        id="case1_existence_and_sources",
        desc="Milestone 1 has required fields (director and film) and at least one supporting URL.",
        parent=case_node,
        critical=True
    )

    # Director_Name
    node_director = evaluator.add_leaf(
        id="case1_director_name",
        desc="The director's name is provided with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    director_claim = f"{director_name} is credited as a film director and is relevant to the referenced award context."
    await evaluator.verify(
        claim=director_claim,
        node=node_director,
        sources=sources,
        additional_instruction="Check that this person is recognized as a film director in the provided sources."
    )

    # Award_Category (Best Adapted Screenplay)
    node_category = evaluator.add_leaf(
        id="case1_award_category",
        desc="The award category is confirmed as Best Adapted Screenplay with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    category_claim = (
        f"{director_name} won the Academy Award for {EXPECTED_OSCARS_CATEGORY}"
        + (f" for the film '{film_title}'." if _has_nonempty_text(film_title) else ".")
    )
    await evaluator.verify(
        claim=category_claim,
        node=node_category,
        sources=sources,
        additional_instruction="Confirm that the person is listed among the winner(s) for Best Adapted Screenplay at the cited ceremony; allow co-winner credit."
    )

    # Ceremony_Date
    node_ceremony = evaluator.add_leaf(
        id="case1_ceremony_date",
        desc=f"The ceremony date is confirmed as {EXPECTED_OSCARS_CEREMONY_DATE} with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    ceremony_claim = f"The Academy Awards ceremony in 2016 took place on {EXPECTED_OSCARS_CEREMONY_DATE}."
    await evaluator.verify(
        claim=ceremony_claim,
        node=node_ceremony,
        sources=sources,
        additional_instruction="Verify the official date for the Oscars ceremony referenced by the sources."
    )

    # Film_Title
    node_film_title = evaluator.add_leaf(
        id="case1_film_title",
        desc="The film title is provided and matches the award-winning work with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    film_title_claim = (
        f"The Best Adapted Screenplay award credited to {director_name} was for the film '{film_title}'."
        if _has_nonempty_text(film_title) else
        "The award-winning film title is clearly specified by the sources."
    )
    await evaluator.verify(
        claim=film_title_claim,
        node=node_film_title,
        sources=sources,
        additional_instruction="The sources should clearly associate the winner's name with the winning film title for Best Adapted Screenplay."
    )

    # Film_Release_Year (2015)
    node_film_year = evaluator.add_leaf(
        id="case1_film_release_year",
        desc=f"The film's release year is confirmed as {EXPECTED_FILM_RELEASE_YEAR_CASE1} with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    film_year_claim = f"The film '{film_title}' was released in {EXPECTED_FILM_RELEASE_YEAR_CASE1}."
    await evaluator.verify(
        claim=film_year_claim,
        node=node_film_year,
        sources=sources,
        additional_instruction="Confirm the film's initial release year as 2015 (e.g., U.S. release year) from the sources."
    )


async def verify_case_tv_debut(evaluator: Evaluator, parent_node, debut: Optional[Milestone2TVDebut]) -> None:
    case_node = evaluator.add_parallel(
        id="Case_2_Television_Debut",
        desc="An actress who made her first appearance in a specific television role in an episode that aired in January 2020.",
        parent=parent_node,
        critical=False
    )

    actress = debut.actress_name if debut else None
    series = debut.series_name if debut else None
    character = debut.character_name if debut else None
    season = debut.season_number if debut else None
    episode = debut.episode_number if debut else None
    episode_title = debut.episode_title if debut else None
    air_date = debut.air_date if debut else None
    replaced_actress = debut.replaced_actress_name if debut else None
    replacement_desc = debut.replacement_description if debut else None
    sources = debut.sources if debut else []

    existence_ok = (
        debut is not None and
        _has_nonempty_text(actress) and
        _has_nonempty_text(series) and
        _has_sources(sources)
    )
    evaluator.add_custom_node(
        result=existence_ok,
        id="case2_existence_and_sources",
        desc="Milestone 2 has required fields (actress and series) and at least one supporting URL.",
        parent=case_node,
        critical=True
    )

    # Actress_Name
    node_actress = evaluator.add_leaf(
        id="case2_actress_name",
        desc="The actress's name is provided with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    actress_claim = f"In the TV series context provided by the sources, the actress involved in the debut is {actress}."
    await evaluator.verify(
        claim=actress_claim,
        node=node_actress,
        sources=sources,
        additional_instruction="Confirm that the identified person is the actress who made the debut referenced by the sources."
    )

    # Television_Series
    node_series = evaluator.add_leaf(
        id="case2_television_series",
        desc="The TV series name is provided and confirmed with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    series_claim = f"The debut occurred in the television series '{series}'."
    await evaluator.verify(
        claim=series_claim,
        node=node_series,
        sources=sources,
        additional_instruction="Confirm the series title associated with the debut episode."
    )

    # Character_Name
    node_character = evaluator.add_leaf(
        id="case2_character_name",
        desc="The character name is provided and confirmed with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    character_claim = f"In '{series}', {actress} portrayed the character {character} in her debut episode."
    await evaluator.verify(
        claim=character_claim,
        node=node_character,
        sources=sources,
        additional_instruction="Confirm the character name tied to the actress's first appearance."
    )

    # Replacement_Casting
    node_replacement = evaluator.add_leaf(
        id="case2_replacement_casting",
        desc="The actress was cast to replace another actress in this role, confirmed with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    if _has_nonempty_text(replaced_actress):
        replacement_claim = f"{actress} was cast as a replacement for {replaced_actress} in the role of {character} in '{series}'."
    else:
        replacement_claim = f"{actress} was cast as a replacement for another actress in the role of {character} in '{series}'."
    await evaluator.verify(
        claim=replacement_claim,
        node=node_replacement,
        sources=sources,
        additional_instruction="Confirm that the casting was a replacement (recasting) for the specified role."
    )

    # Season_Number (Season 3)
    node_season = evaluator.add_leaf(
        id="case2_season_number",
        desc="The season number (Season 3) is provided and confirmed with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    season_claim = f"The debut appearance occurred in Season {EXPECTED_TV_SEASON} of '{series}'."
    await evaluator.verify(
        claim=season_claim,
        node=node_season,
        sources=sources,
        additional_instruction="Verify that the actress's first appearance episode belongs to Season 3."
    )

    # Episode_Number (Episode 9)
    node_episode = evaluator.add_leaf(
        id="case2_episode_number",
        desc="The episode number (Episode 9) is provided and confirmed with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    episode_claim = f"The debut appearance episode was Episode {EXPECTED_TV_EPISODE} of Season {EXPECTED_TV_SEASON}."
    await evaluator.verify(
        claim=episode_claim,
        node=node_episode,
        sources=sources,
        additional_instruction="Verify the exact episode number (9) for the debut."
    )

    # Episode_Title
    node_ep_title = evaluator.add_leaf(
        id="case2_episode_title",
        desc="The episode title is provided and confirmed with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    ep_title_claim = f"The title of the debut episode is '{episode_title}'."
    await evaluator.verify(
        claim=ep_title_claim,
        node=node_ep_title,
        sources=sources,
        additional_instruction="Confirm the exact episode title text from the provided sources."
    )

    # Air_Date (January 17, 2020)
    node_air_date = evaluator.add_leaf(
        id="case2_air_date",
        desc=f"The air date is confirmed as {EXPECTED_TV_AIR_DATE} with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    air_date_claim = f"The debut episode aired on {EXPECTED_TV_AIR_DATE}."
    await evaluator.verify(
        claim=air_date_claim,
        node=node_air_date,
        sources=sources,
        additional_instruction="Confirm the exact air date; it must match January 17, 2020."
    )


async def verify_case_landmark_film(evaluator: Evaluator, parent_node, film: Optional[Milestone3LandmarkFilm]) -> None:
    case_node = evaluator.add_parallel(
        id="Case_3_Landmark_Film",
        desc="A landmark animated film released in 1995 with specific production characteristics.",
        parent=parent_node,
        critical=False
    )

    title = film.film_title if film else None
    production_company = film.production_company if film else None
    director = film.director if film else None
    us_release_date = film.us_release_date if film else None
    significance = film.significance if film else None
    sources = film.sources if film else []

    existence_ok = (
        film is not None and
        _has_nonempty_text(title) and
        _has_sources(sources)
    )
    evaluator.add_custom_node(
        result=existence_ok,
        id="case3_existence_and_sources",
        desc="Milestone 3 has required fields (film title) and at least one supporting URL.",
        parent=case_node,
        critical=True
    )

    # Film_Title
    node_title = evaluator.add_leaf(
        id="case3_film_title",
        desc="The film title is provided and confirmed with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    title_claim = f"'{title}' is an animated feature film that had a U.S. theatrical release in 1995."
    await evaluator.verify(
        claim=title_claim,
        node=node_title,
        sources=sources,
        additional_instruction="Confirm that the film title refers to an animated feature released in U.S. theaters in 1995."
    )

    # Production_Company (Pixar Animation Studios)
    node_prod = evaluator.add_leaf(
        id="case3_production_company",
        desc="The production company is confirmed as Pixar Animation Studios with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    prod_claim = f"The film '{title}' was produced by {EXPECTED_LANDMARK_PRODUCTION_COMPANY}."
    await evaluator.verify(
        claim=prod_claim,
        node=node_prod,
        sources=sources,
        additional_instruction="Verify the production company for the film; it should be Pixar Animation Studios."
    )

    # Director (John Lasseter)
    node_director = evaluator.add_leaf(
        id="case3_director",
        desc="The director is confirmed as John Lasseter with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    director_claim = f"The film '{title}' was directed by {EXPECTED_LANDMARK_DIRECTOR}."
    await evaluator.verify(
        claim=director_claim,
        node=node_director,
        sources=sources,
        additional_instruction="Confirm that John Lasseter is credited as the director of the film."
    )

    # Theatrical_Release_Date (Nov 22, 1995)
    node_release = evaluator.add_leaf(
        id="case3_us_theatrical_release_date",
        desc=f"The U.S. theatrical release date is confirmed as {EXPECTED_LANDMARK_US_RELEASE_DATE} with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    release_claim = f"The U.S. theatrical release date of '{title}' was {EXPECTED_LANDMARK_US_RELEASE_DATE}."
    await evaluator.verify(
        claim=release_claim,
        node=node_release,
        sources=sources,
        additional_instruction="Confirm the exact U.S. theatrical release date."
    )

    # Historical_Significance (first fully computer-animated feature film)
    node_significance = evaluator.add_leaf(
        id="case3_historical_significance",
        desc="The film's historical significance as the first fully computer-animated feature film is confirmed with supporting URL reference.",
        parent=case_node,
        critical=True
    )
    significance_claim = f"'{title}' is widely recognized as the {EXPECTED_LANDMARK_SIGNIFICANCE}."
    await evaluator.verify(
        claim=significance_claim,
        node=node_significance,
        sources=sources,
        additional_instruction="Verify that the sources explicitly state the film is the first fully computer-animated feature film."
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
    Evaluate an answer for the entertainment milestones task and return the summary dict.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: parallel across three independent cases
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_milestones(),
        template_class=AllMilestonesExtraction,
        extraction_name="milestones_extraction"
    )

    # Add rubric-derived ground truth constraints for transparency
    evaluator.add_ground_truth({
        "case1_expected_category": EXPECTED_OSCARS_CATEGORY,
        "case1_expected_ceremony_date": EXPECTED_OSCARS_CEREMONY_DATE,
        "case1_expected_film_release_year": EXPECTED_FILM_RELEASE_YEAR_CASE1,
        "case2_expected_season": EXPECTED_TV_SEASON,
        "case2_expected_episode": EXPECTED_TV_EPISODE,
        "case2_expected_air_date": EXPECTED_TV_AIR_DATE,
        "case3_expected_production_company": EXPECTED_LANDMARK_PRODUCTION_COMPANY,
        "case3_expected_director": EXPECTED_LANDMARK_DIRECTOR,
        "case3_expected_us_release_date": EXPECTED_LANDMARK_US_RELEASE_DATE,
        "case3_expected_significance": EXPECTED_LANDMARK_SIGNIFICANCE
    }, gt_type="rubric_expectations")

    # Build verification tree per case
    await verify_case_award(evaluator, root, extracted.award)
    await verify_case_tv_debut(evaluator, root, extracted.tv_debut)
    await verify_case_landmark_film(evaluator, root, extracted.landmark_film)

    return evaluator.get_summary()