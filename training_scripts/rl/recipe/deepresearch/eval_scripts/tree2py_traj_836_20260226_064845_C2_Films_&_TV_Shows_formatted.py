import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bigelow_netflix_2025"
TASK_DESCRIPTION = (
    "Identify Kathryn Bigelow's feature film that was released on Netflix in 2025. "
    "Provide: (1) the film's title, (2) the exact date when it became available for global streaming on Netflix, "
    "(3) the runtime in minutes, and (4) Kathryn Bigelow's historic achievement related to the Academy Awards, "
    "including the specific film for which she received this recognition."
)

EXPECTED_GLOBAL_NETFLIX_DATE = "October 24, 2025"
EXPECTED_RUNTIME_MINUTES = "112"
EXPECTED_GENRE = "political thriller"
EXPECTED_LEAD_ACTOR = "Idris Elba"
EXPECTED_DIRECTOR = "Kathryn Bigelow"
EXPECTED_OSCAR_ACHIEVEMENT = "first woman to win the Academy Award for Best Director"
EXPECTED_OSCAR_FILM = "The Hurt Locker (2008)"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmRef(BaseModel):
    title: Optional[str] = None
    film_urls: List[str] = Field(default_factory=list)


class NetflixReleaseInfo(BaseModel):
    global_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    distributor_urls: List[str] = Field(default_factory=list)


class FilmAttributes(BaseModel):
    runtime_minutes: Optional[str] = None
    runtime_sources: List[str] = Field(default_factory=list)
    genre: Optional[str] = None
    genre_sources: List[str] = Field(default_factory=list)
    cast: List[str] = Field(default_factory=list)
    cast_sources: List[str] = Field(default_factory=list)


class FilmTypeDirection(BaseModel):
    director: Optional[str] = None
    director_sources: List[str] = Field(default_factory=list)
    is_feature_film_statement: Optional[str] = None
    feature_sources: List[str] = Field(default_factory=list)
    most_recent_statement: Optional[str] = None
    most_recent_sources: List[str] = Field(default_factory=list)


class AwardInfo(BaseModel):
    achievement_statement: Optional[str] = None
    achievement_sources: List[str] = Field(default_factory=list)
    best_director_film: Optional[str] = None
    best_director_film_year: Optional[str] = None
    best_director_sources: List[str] = Field(default_factory=list)


class BigelowFilmExtraction(BaseModel):
    film: Optional[FilmRef] = None
    netflix_release: Optional[NetflixReleaseInfo] = None
    attributes: Optional[FilmAttributes] = None
    type_direction: Optional[FilmTypeDirection] = None
    awards: Optional[AwardInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
    From the answer, extract structured information about Kathryn Bigelow's feature film that was released on Netflix in 2025.
    Return a JSON object with the following nested fields. Extract ONLY what is explicitly present in the answer text. Do not invent.

    film:
      - title: The film title as written in the answer.
      - film_urls: All URLs in the answer that directly reference the film (e.g., Netflix listing, official site, studio/press releases, trade articles, Wikipedia, IMDb).

    netflix_release:
      - global_date: The exact date string the answer gives for when the film became available for global streaming on Netflix. Use the exact phrasing (e.g., "October 24, 2025"). If not explicitly stated, null.
      - sources: All URLs cited in the answer that support the Netflix availability date (global streaming).
      - distributor_urls: All URLs cited that support that Netflix distributed the film for global streaming.

    attributes:
      - runtime_minutes: The runtime in minutes as written (e.g., "112", "112 minutes"). If not explicitly present, null.
      - runtime_sources: URLs cited for the runtime.
      - genre: The genre descriptor as written (e.g., "political thriller"). If multiple are provided, prefer the phrasing that includes "political". If not provided, null.
      - genre_sources: URLs cited for the genre.
      - cast: List of key cast names exactly as written in the answer (e.g., ["Idris Elba", "X"]). If none provided, return [].
      - cast_sources: URLs cited for cast information.

    type_direction:
      - director: The director's name as written for this film. If not stated, null.
      - director_sources: URLs cited for the director.
      - is_feature_film_statement: The text in the answer (if any) that indicates this is a feature film (not a series/short/episode). If none, null.
      - feature_sources: URLs cited for the feature-film format.
      - most_recent_statement: The answer's statement (if any) that this is Kathryn Bigelow's most recent feature film as of February 2026 (or similar wording). If none, null.
      - most_recent_sources: URLs cited that support this "most recent" characterization.

    awards:
      - achievement_statement: The statement in the answer about Kathryn Bigelow’s historic Academy Awards achievement (e.g., "first woman to win Best Director"). If none, null.
      - achievement_sources: URLs cited that support the achievement statement.
      - best_director_film: The film title for which Bigelow won Best Director (as written, e.g., "The Hurt Locker").
      - best_director_film_year: The film year if provided with the title (e.g., "2008"); otherwise null.
      - best_director_sources: URLs cited for the Best Director film info.

    IMPORTANT URL RULES:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - Include full URLs (with http:// or https://). If missing protocol, prepend http://.
    - If a field is missing, return null (or [] for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


def _safe_str(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, data: BigelowFilmExtraction) -> None:
    # Safely access nested structures
    film = data.film or FilmRef()
    release = data.netflix_release or NetflixReleaseInfo()
    attrs = data.attributes or FilmAttributes()
    td = data.type_direction or FilmTypeDirection()
    awards = data.awards or AwardInfo()

    title = _safe_str(film.title)

    # Root-level aggregators (all critical to match rubric)
    film_ident_node = evaluator.add_parallel(
        id="FilmIdentification",
        desc="Correctly identifies the film and satisfies director/format/recency constraints",
        parent=root_node,
        critical=True
    )

    netflix_constraints_node = evaluator.add_parallel(
        id="NetflixReleaseConstraints",
        desc="Satisfies Netflix distribution and timing constraints",
        parent=root_node,
        critical=True
    )

    film_attr_node = evaluator.add_parallel(
        id="FilmAttributeConstraints",
        desc="Satisfies runtime/genre/cast constraints and provides requested film attributes",
        parent=root_node,
        critical=True
    )

    awards_node = evaluator.add_parallel(
        id="DirectorOscarHistoricAchievement",
        desc="Provides Kathryn Bigelow’s Academy Awards historic achievement and the specific film associated with it (as constrained)",
        parent=root_node,
        critical=True
    )

    # --------------------- FilmIdentification children --------------------- #
    # FilmTitleProvided (existence check)
    evaluator.add_custom_node(
        result=bool(title.strip()),
        id="FilmTitleProvided",
        desc="Provides the film’s title",
        parent=film_ident_node,
        critical=True
    )

    # DirectedByKathrynBigelow
    node_directed = evaluator.add_leaf(
        id="DirectedByKathrynBigelow",
        desc="The identified film is directed by Kathryn Bigelow",
        parent=film_ident_node,
        critical=True
    )
    claim_directed = f"The film '{title}' is directed by Kathryn Bigelow."
    await evaluator.verify(
        claim=claim_directed,
        node=node_directed,
        sources=_combine_sources(td.director_sources, film.film_urls),
        additional_instruction="Verify the director credit for the identified film. Allow minor name formatting differences (e.g., middle initials)."
    )

    # IsFeatureFilm
    node_feature = evaluator.add_leaf(
        id="IsFeatureFilm",
        desc="The identified work is a feature film (not a short/episode/series)",
        parent=film_ident_node,
        critical=True
    )
    claim_feature = f"'{title}' is a feature film (a full-length motion picture), not a TV series, episode, or short."
    await evaluator.verify(
        claim=claim_feature,
        node=node_feature,
        sources=_combine_sources(td.feature_sources, film.film_urls),
        additional_instruction="Look for explicit descriptors like 'feature film', 'film', or runtime/format context that clearly indicates a standalone feature."
    )

    # MostRecentFeatureFilmAsOfFeb2026
    node_recent = evaluator.add_leaf(
        id="MostRecentFeatureFilmAsOfFeb2026",
        desc="The identified film is Kathryn Bigelow’s most recent feature film as of February 2026",
        parent=film_ident_node,
        critical=True
    )
    claim_recent = f"As of February 2026, '{title}' is Kathryn Bigelow’s most recent feature film."
    await evaluator.verify(
        claim=claim_recent,
        node=node_recent,
        sources=_combine_sources(td.most_recent_sources, film.film_urls),
        additional_instruction="Accept phrasing like 'latest film', 'newest feature', 'first feature in X years' that implies it is the most recent by Feb 2026. If any later feature is indicated, the claim is not supported."
    )

    # ----------------- NetflixReleaseConstraints children ------------------ #
    # DistributedByNetflixForGlobalStreaming
    node_distributed = evaluator.add_leaf(
        id="DistributedByNetflixForGlobalStreaming",
        desc="The film is distributed by Netflix for its global streaming release",
        parent=netflix_constraints_node,
        critical=True
    )
    claim_distributed = f"Netflix distributed '{title}' for its global streaming release."
    await evaluator.verify(
        claim=claim_distributed,
        node=node_distributed,
        sources=_combine_sources(release.distributor_urls, release.sources, film.film_urls),
        additional_instruction="Look for language like 'Netflix original film', 'a Netflix film', 'released globally on Netflix', or distributor credit indicating Netflix for worldwide streaming."
    )

    # GlobalNetflixDateProvidedAndIsExact (must be Oct 24, 2025)
    node_global_date = evaluator.add_leaf(
        id="GlobalNetflixDateProvidedAndIsExact",
        desc="Provides the exact global Netflix availability date, and it is October 24, 2025",
        parent=netflix_constraints_node,
        critical=True
    )
    claim_global_date = f"'{title}' became available for global streaming on Netflix on {EXPECTED_GLOBAL_NETFLIX_DATE}."
    await evaluator.verify(
        claim=claim_global_date,
        node=node_global_date,
        sources=_combine_sources(release.sources, film.film_urls),
        additional_instruction="Verify the exact global Netflix availability date; ensure it's the worldwide streaming launch date, not a theatrical or limited regional release date."
    )

    # ------------------- FilmAttributeConstraints children ----------------- #
    # RuntimeProvidedAndIs112Minutes
    node_runtime = evaluator.add_leaf(
        id="RuntimeProvidedAndIs112Minutes",
        desc="Provides the runtime in minutes, and it is 112 minutes",
        parent=film_attr_node,
        critical=True
    )
    claim_runtime = f"The runtime of '{title}' is {EXPECTED_RUNTIME_MINUTES} minutes."
    await evaluator.verify(
        claim=claim_runtime,
        node=node_runtime,
        sources=_combine_sources(attrs.runtime_sources, film.film_urls),
        additional_instruction="Match the numeric runtime. Minor formatting like '112 min' vs '112 minutes' is acceptable; the number must be 112."
    )

    # GenreIsPoliticalThriller
    node_genre = evaluator.add_leaf(
        id="GenreIsPoliticalThriller",
        desc="The film is a political thriller",
        parent=film_attr_node,
        critical=True
    )
    claim_genre = f"'{title}' is a political thriller."
    await evaluator.verify(
        claim=claim_genre,
        node=node_genre,
        sources=_combine_sources(attrs.genre_sources, film.film_urls),
        additional_instruction="Allow close variants like 'political action thriller' or 'political thriller drama' as long as 'political thriller' is clearly supported."
    )

    # StarsIdrisElbaLeadRole
    node_cast = evaluator.add_leaf(
        id="StarsIdrisElbaLeadRole",
        desc="The film stars Idris Elba in a lead role",
        parent=film_attr_node,
        critical=True
    )
    claim_cast = f"Idris Elba stars in a lead role in '{title}'."
    await evaluator.verify(
        claim=claim_cast,
        node=node_cast,
        sources=_combine_sources(attrs.cast_sources, film.film_urls),
        additional_instruction="Look for 'starring Idris Elba', 'Idris Elba leads', or top-billing that clearly indicates a lead role."
    )

    # --------------- DirectorOscarHistoricAchievement children ------------- #
    # FirstWomanBestDirector
    node_oscar_first = evaluator.add_leaf(
        id="FirstWomanBestDirector",
        desc="States that Kathryn Bigelow is the first woman to win the Academy Award for Best Director",
        parent=awards_node,
        critical=True
    )
    claim_oscar_first = "Kathryn Bigelow is the first woman to win the Academy Award for Best Director."
    await evaluator.verify(
        claim=claim_oscar_first,
        node=node_oscar_first,
        sources=_combine_sources(awards.achievement_sources),
        additional_instruction="Use reliable sources (Academy, major news, Wikipedia) that explicitly state this historic achievement."
    )

    # BestDirectorFilmIsTheHurtLocker2008
    node_oscar_film = evaluator.add_leaf(
        id="BestDirectorFilmIsTheHurtLocker2008",
        desc="Names The Hurt Locker (2008) as the film for which she won Best Director",
        parent=awards_node,
        critical=True
    )
    claim_oscar_film = "Kathryn Bigelow won the Academy Award for Best Director for The Hurt Locker (2008)."
    await evaluator.verify(
        claim=claim_oscar_film,
        node=node_oscar_film,
        sources=_combine_sources(awards.best_director_sources),
        additional_instruction="Verify that the Best Director Oscar was awarded for The Hurt Locker and the film year is 2008."
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=BigelowFilmExtraction,
        extraction_name="bigelow_film_extraction"
    )

    # Optional: add ground truth constraints to summary
    evaluator.add_ground_truth({
        "expected_global_netflix_date": EXPECTED_GLOBAL_NETFLIX_DATE,
        "expected_runtime_minutes": EXPECTED_RUNTIME_MINUTES,
        "expected_genre": EXPECTED_GENRE,
        "expected_lead_actor": EXPECTED_LEAD_ACTOR,
        "expected_director": EXPECTED_DIRECTOR,
        "expected_oscar_achievement": EXPECTED_OSCAR_ACHIEVEMENT,
        "expected_oscar_film": EXPECTED_OSCAR_FILM
    }, gt_type="expected_constraints")

    # Build verification tree per rubric
    await build_verification_tree(evaluator, root, extracted)

    # Return structured result
    return evaluator.get_summary()