import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "swedish_composer_awards_constraints"
TASK_DESCRIPTION = """
Identify the Swedish composer who meets all of the following criteria: (1) Won their first Academy Award for Best Original Score at the 91st Academy Awards (2019 ceremony) for a 2018 superhero film, (2) Won their second Academy Award for Best Original Score at the 96th Academy Awards (2024 ceremony) for a 2023 biographical film directed by Christopher Nolan, (3) Has a total of 5 Grammy Award wins as of the 68th Annual Grammy Awards (2026), (4) Won both Record of the Year and Song of the Year at the 61st Grammy Awards (2019) for production work on a 2018 single in collaboration with an artist who performs under a stage name and is also known for acting, (5) Won Best Score Soundtrack for Visual Media at the 68th Grammy Awards (2026), (6) Won Emmy Awards for Outstanding Music Composition for a science fiction series in two consecutive years (2020 and 2021), (7) Composed the score for a 2020 Christopher Nolan film, and (8) Composed scores for at least two films in the same superhero franchise. Provide the composer's full name and URL references that verify each of these achievements.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OscarFirst(BaseModel):
    award_ceremony: Optional[str] = None  # e.g., "91st Academy Awards (2019)"
    category: Optional[str] = None        # e.g., "Best Original Score"
    film_title: Optional[str] = None      # e.g., "Black Panther"
    film_year: Optional[str] = None       # e.g., "2018"
    film_type_hint: Optional[str] = None  # e.g., "superhero"
    sources: List[str] = Field(default_factory=list)


class OscarSecond(BaseModel):
    award_ceremony: Optional[str] = None  # e.g., "96th Academy Awards (2024)"
    category: Optional[str] = None        # e.g., "Best Original Score"
    film_title: Optional[str] = None      # e.g., "Oppenheimer"
    film_year: Optional[str] = None       # e.g., "2023"
    director: Optional[str] = None        # e.g., "Christopher Nolan"
    film_type_hint: Optional[str] = None  # e.g., "biographical"
    sources: List[str] = Field(default_factory=list)


class GrammyTotals(BaseModel):
    total_wins_as_of_2026: Optional[str] = None  # e.g., "5"
    sources: List[str] = Field(default_factory=list)


class Grammys2019(BaseModel):
    single_title_2018: Optional[str] = None              # e.g., "This Is America"
    collaborator_stage_name: Optional[str] = None        # e.g., "Childish Gambino"
    collaborator_legal_name: Optional[str] = None        # e.g., "Donald Glover"
    composer_role_on_single: Optional[str] = None        # e.g., "producer"
    sources: List[str] = Field(default_factory=list)


class CollaboratorFacts(BaseModel):
    stage_name: Optional[str] = None             # e.g., "Childish Gambino"
    legal_name: Optional[str] = None             # e.g., "Donald Glover"
    actor_status_note: Optional[str] = None      # e.g., "also an actor"
    sources: List[str] = Field(default_factory=list)


class Grammy2026Score(BaseModel):
    category: Optional[str] = None               # e.g., "Best Score Soundtrack for Visual Media"
    work_title: Optional[str] = None             # e.g., "Oppenheimer"
    sources: List[str] = Field(default_factory=list)


class EmmysInfo(BaseModel):
    series_name: Optional[str] = None            # e.g., "The Mandalorian"
    years: List[str] = Field(default_factory=list)  # e.g., ["2020", "2021"]
    category: Optional[str] = None               # e.g., "Outstanding Music Composition for a Series"
    sources: List[str] = Field(default_factory=list)


class Nolan2020Film(BaseModel):
    film_title: Optional[str] = None             # e.g., "Tenet"
    year: Optional[str] = None                   # e.g., "2020"
    sources: List[str] = Field(default_factory=list)


class SuperheroFranchiseWork(BaseModel):
    franchise_name: Optional[str] = None         # e.g., "Black Panther" or "Marvel Cinematic Universe"
    film_titles: List[str] = Field(default_factory=list)  # e.g., ["Black Panther", "Black Panther: Wakanda Forever"]
    sources: List[str] = Field(default_factory=list)


class ComposerExtraction(BaseModel):
    composer_full_name: Optional[str] = None

    nationality: Optional[str] = None
    nationality_sources: List[str] = Field(default_factory=list)

    oscar_first: Optional[OscarFirst] = None
    oscar_second: Optional[OscarSecond] = None

    grammy_totals: Optional[GrammyTotals] = None
    grammys_2019: Optional[Grammys2019] = None
    collaborator_meta: Optional[CollaboratorFacts] = None
    grammy_2026_score: Optional[Grammy2026Score] = None

    emmys: Optional[EmmysInfo] = None

    nolan_2020: Optional[Nolan2020Film] = None
    superhero_franchise: Optional[SuperheroFranchiseWork] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_composer() -> str:
    return """
    Extract the following structured information about the Swedish composer identified in the answer. Use exactly the information explicitly present in the answer. Include all URLs that the answer uses as references for each achievement. Do not invent URLs.

    Required fields:
    - composer_full_name: The full name of the composer.

    - nationality: The nationality string (e.g., "Swedish") as stated in the answer.
    - nationality_sources: Array of URL(s) explicitly provided in the answer that support nationality.

    - oscar_first:
        - award_ceremony: The ceremony label (e.g., "91st Academy Awards (2019)"), if stated.
        - category: The category (should be "Best Original Score" if stated).
        - film_title: The title of the 2018 superhero film, if stated.
        - film_year: The film's year if stated (ideally "2018").
        - film_type_hint: A short hint like "superhero" if present.
        - sources: Array of URL(s) that support this first Oscar claim.

    - oscar_second:
        - award_ceremony: The ceremony label (e.g., "96th Academy Awards (2024)"), if stated.
        - category: The category (should be "Best Original Score" if stated).
        - film_title: The title of the 2023 biographical film (directed by Christopher Nolan), if stated.
        - film_year: The film's year if stated (ideally "2023").
        - director: The director's name if stated (ideally "Christopher Nolan").
        - film_type_hint: A short hint like "biographical" if present.
        - sources: Array of URL(s) that support this second Oscar claim.

    - grammy_totals:
        - total_wins_as_of_2026: The total Grammy wins count as of the 68th Annual Grammy Awards (2026) as stated in the answer (e.g., "5").
        - sources: Array of URL(s) that support this count.

    - grammys_2019:
        - single_title_2018: The 2018 single title associated with the 61st Grammys (2019) Record of the Year and Song of the Year wins.
        - collaborator_stage_name: The collaborator’s stage name (e.g., "Childish Gambino").
        - collaborator_legal_name: The collaborator’s legal name if stated (e.g., "Donald Glover").
        - composer_role_on_single: The role such as "producer"/"production" if stated.
        - sources: Array of URL(s) that support these 2019 wins and the composer's production role.

    - collaborator_meta:
        - stage_name: The collaborator’s stage name (repeat if same as above).
        - legal_name: The legal name (repeat if available).
        - actor_status_note: A short note if the collaborator is known for acting (e.g., "also an actor").
        - sources: Array of URL(s) that explicitly support that the collaborator performs under a stage name and is also known for acting.

    - grammy_2026_score:
        - category: Should be "Best Score Soundtrack for Visual Media" if stated.
        - work_title: The work (film/series) name if stated.
        - sources: Array of URL(s) that support this 2026 Grammy win.

    - emmys:
        - series_name: The sci‑fi series title if stated (e.g., "The Mandalorian").
        - years: An array of years (should include 2020 and 2021) as stated.
        - category: The category string if present (e.g., "Outstanding Music Composition for a Series").
        - sources: Array of URL(s) supporting the Emmy wins for both years and that they are for the same series.

    - nolan_2020:
        - film_title: The 2020 Christopher Nolan film title (e.g., "Tenet") if stated.
        - year: The year if stated (ideally "2020").
        - sources: Array of URL(s) that support the composer’s scoring credit for this film.

    - superhero_franchise:
        - franchise_name: The franchise name if stated (e.g., "Black Panther" or "Marvel Cinematic Universe").
        - film_titles: Array of at least two film titles in the same superhero franchise that the composer scored.
        - sources: Array of URL(s) that support the composer’s scoring credits for at least two films in the same franchise.

    URL extraction rules:
    - Only return URLs that are explicitly present in the answer text (plain or markdown links). Do not infer or invent URLs.
    - Provide complete URLs. If a URL is missing a protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


def safe_join_urls(*url_lists: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for lst in url_lists:
        if lst:
            out.extend([u for u in lst if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def add_name_node(evaluator: Evaluator, parent, extracted: ComposerExtraction) -> None:
    evaluator.add_custom_node(
        result=bool(extracted.composer_full_name and extracted.composer_full_name.strip()),
        id="Composer_Full_Name",
        desc="Answer provides the composer's full name.",
        parent=parent,
        critical=True
    )


async def add_nationality_nodes(evaluator: Evaluator, parent, extracted: ComposerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Nationality",
        desc="Composer nationality requirement + supporting URL(s).",
        parent=parent,
        critical=True
    )

    # URL existence
    evaluator.add_custom_node(
        result=has_urls(extracted.nationality_sources),
        id="URL_Nationality_Provided",
        desc="At least one URL is provided that supports the Swedish nationality claim.",
        parent=node,
        critical=True
    )

    # Verify nationality
    leaf = evaluator.add_leaf(
        id="Swedish_Nationality_Verified",
        desc="Composer is Swedish by nationality.",
        parent=node,
        critical=True
    )
    name = extracted.composer_full_name or "the composer"
    nat = extracted.nationality or "Swedish"
    claim = f"{name} is {nat} by nationality or citizenship."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=extracted.nationality_sources,
        additional_instruction="Use the provided URL(s) (e.g., official bios, reputable encyclopedias) to confirm the person is Swedish. Accept wording like 'Swedish composer'."
    )


async def add_academy_awards_nodes(evaluator: Evaluator, parent, extracted: ComposerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Academy_Awards",
        desc="Two specified Academy Award wins (first and second) + supporting URL(s).",
        parent=parent,
        critical=True
    )
    name = extracted.composer_full_name or "the composer"

    # First Oscar: URL existence
    first_srcs = extracted.oscar_first.sources if extracted.oscar_first else []
    evaluator.add_custom_node(
        result=has_urls(first_srcs),
        id="URL_First_Oscar_Provided",
        desc="At least one URL is provided that supports the first Oscar requirement details.",
        parent=node,
        critical=True
    )

    # First Oscar: verify
    first_leaf = evaluator.add_leaf(
        id="First_Oscar_91st_2019_2018_Superhero",
        desc="Composer won their FIRST Academy Award for Best Original Score at the 91st Academy Awards (2019) for a 2018 superhero film.",
        parent=node,
        critical=True
    )
    first_film = (extracted.oscar_first.film_title if extracted.oscar_first else None) or "a 2018 superhero film"
    first_cer = (extracted.oscar_first.award_ceremony if extracted.oscar_first else None) or "91st Academy Awards (2019)"
    first_cat = (extracted.oscar_first.category if extracted.oscar_first else None) or "Best Original Score"
    claim_first = f"{name} won their first Academy Award for {first_cat} at the {first_cer} for {first_film}."
    await evaluator.verify(
        claim=claim_first,
        node=first_leaf,
        sources=first_srcs,
        additional_instruction="Confirm this was the person's first Academy Award and that the film is from 2018 and is a superhero film. Minor wording variations are acceptable."
    )

    # Second Oscar: URL existence
    second_srcs = extracted.oscar_second.sources if extracted.oscar_second else []
    evaluator.add_custom_node(
        result=has_urls(second_srcs),
        id="URL_Second_Oscar_Provided",
        desc="At least one URL is provided that supports the second Oscar requirement details.",
        parent=node,
        critical=True
    )

    # Second Oscar: verify
    second_leaf = evaluator.add_leaf(
        id="Second_Oscar_96th_2024_2023_Nolan_Biopic",
        desc="Composer won their SECOND Academy Award for Best Original Score at the 96th Academy Awards (2024) for a 2023 biographical film directed by Christopher Nolan.",
        parent=node,
        critical=True
    )
    second_film = (extracted.oscar_second.film_title if extracted.oscar_second else None) or "a 2023 biographical film"
    second_cer = (extracted.oscar_second.award_ceremony if extracted.oscar_second else None) or "96th Academy Awards (2024)"
    second_cat = (extracted.oscar_second.category if extracted.oscar_second else None) or "Best Original Score"
    claim_second = f"{name} won their second Academy Award for {second_cat} at the {second_cer} for {second_film}, which was directed by Christopher Nolan."
    await evaluator.verify(
        claim=claim_second,
        node=second_leaf,
        sources=second_srcs,
        additional_instruction="Confirm the film is a 2023 biographical film directed by Christopher Nolan and that this was the person's second Oscar."
    )


async def add_grammys_nodes(evaluator: Evaluator, parent, extracted: ComposerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Grammys",
        desc="All Grammy-related constraints + supporting URL(s).",
        parent=parent,
        critical=True
    )
    name = extracted.composer_full_name or "the composer"

    # Total wins as of 2026
    total_srcs = extracted.grammy_totals.sources if extracted.grammy_totals else []
    evaluator.add_custom_node(
        result=has_urls(total_srcs),
        id="URL_Total_Grammy_Wins_Provided",
        desc="At least one URL is provided that supports the total Grammy wins count as of 2026.",
        parent=node,
        critical=True
    )
    total_leaf = evaluator.add_leaf(
        id="Total_Grammy_Wins_AsOf_68th_2026",
        desc="Composer has a total of 5 Grammy Award wins as of the 68th Annual Grammy Awards (2026).",
        parent=node,
        critical=True
    )
    total_wins = (extracted.grammy_totals.total_wins_as_of_2026 if extracted.grammy_totals else None) or "5"
    claim_total = f"As of the 68th Annual Grammy Awards (2026), {name} has {total_wins} Grammy Award wins."
    await evaluator.verify(
        claim=claim_total,
        node=total_leaf,
        sources=total_srcs,
        additional_instruction="Verify the total Grammy wins count as of 2026. Prefer official Grammy pages or reputable sources."
    )

    # 2019 Record and Song of the Year for 2018 single + production role
    g2019_srcs = extracted.grammys_2019.sources if extracted.grammys_2019 else []
    evaluator.add_custom_node(
        result=has_urls(g2019_srcs),
        id="URL_Grammys_2019_Record_And_Song_Provided",
        desc="At least one URL (or set of URLs) is provided that supports the 2019 Record of the Year and Song of the Year wins and the composer's production role for the 2018 single.",
        parent=node,
        critical=True
    )
    g2019_leaf = evaluator.add_leaf(
        id="Grammys_61st_2019_Record_And_Song_For_2018_Single",
        desc="Composer won BOTH Record of the Year and Song of the Year at the 61st Grammy Awards (2019) for production work on a 2018 single.",
        parent=node,
        critical=True
    )
    single_title = (extracted.grammys_2019.single_title_2018 if extracted.grammys_2019 else None) or "a 2018 single"
    role = (extracted.grammys_2019.composer_role_on_single if extracted.grammys_2019 else None) or "production work"
    claim_g2019 = f"{name} won both Record of the Year and Song of the Year at the 61st Grammy Awards (2019) for {role} on '{single_title}'."
    await evaluator.verify(
        claim=claim_g2019,
        node=g2019_leaf,
        sources=g2019_srcs,
        additional_instruction="Confirm both categories (Record of the Year and Song of the Year) at the 61st Grammys (2019) and that the person is credited with production on the 2018 single."
    )

    # Collaborator stage name and acting
    coll_srcs = extracted.collaborator_meta.sources if extracted.collaborator_meta else []
    # If collaborator_meta lacks sources, try grammys_2019 sources (as fallback)
    if not has_urls(coll_srcs):
        coll_srcs = g2019_srcs
    evaluator.add_custom_node(
        result=has_urls(coll_srcs),
        id="URL_Collaborator_StageName_Actor_Provided",
        desc="At least one URL (or set of URLs) is provided that supports the collaborator being known by a stage name and also known for acting.",
        parent=node,
        critical=True
    )
    coll_leaf = evaluator.add_leaf(
        id="Collaborator_StageName_And_Actor",
        desc="The 2018 single collaboration involves an artist who performs under a stage name and is also known for acting.",
        parent=node,
        critical=True
    )
    stage_name = None
    legal_name = None
    if extracted.collaborator_meta:
        stage_name = extracted.collaborator_meta.stage_name or extracted.grammys_2019.collaborator_stage_name if extracted.grammys_2019 else extracted.collaborator_meta.stage_name
        legal_name = extracted.collaborator_meta.legal_name or (extracted.grammys_2019.collaborator_legal_name if extracted.grammys_2019 else None)
    else:
        if extracted.grammys_2019:
            stage_name = extracted.grammys_2019.collaborator_stage_name
            legal_name = extracted.grammys_2019.collaborator_legal_name

    coll_name_phrase = ""
    if stage_name and legal_name:
        coll_name_phrase = f"{stage_name} (legal name {legal_name})"
    elif stage_name:
        coll_name_phrase = f"{stage_name}"
    else:
        coll_name_phrase = "the collaborator"

    claim_coll = f"{coll_name_phrase} performs under a stage name and is also known for acting."
    await evaluator.verify(
        claim=claim_coll,
        node=coll_leaf,
        sources=coll_srcs,
        additional_instruction="Verify that the collaborator uses a stage name and is also an actor (or is known for acting). If both stage and legal names are given, confirm their equivalence."
    )

    # 2026 Grammy Best Score Soundtrack for Visual Media
    g2026_srcs = extracted.grammy_2026_score.sources if extracted.grammy_2026_score else []
    evaluator.add_custom_node(
        result=has_urls(g2026_srcs),
        id="URL_Grammy_2026_Best_Score_Provided",
        desc="At least one URL is provided that supports the 2026 Best Score Soundtrack for Visual Media win.",
        parent=node,
        critical=True
    )
    g2026_leaf = evaluator.add_leaf(
        id="Grammy_68th_2026_Best_Score_Soundtrack",
        desc="Composer won Best Score Soundtrack for Visual Media at the 68th Grammy Awards (2026).",
        parent=node,
        critical=True
    )
    work_title = (extracted.grammy_2026_score.work_title if extracted.grammy_2026_score else None) or ""
    work_phrase = f" for '{work_title}'" if work_title else ""
    claim_g2026 = f"{name} won Best Score Soundtrack for Visual Media at the 68th Grammy Awards (2026){work_phrase}."
    await evaluator.verify(
        claim=claim_g2026,
        node=g2026_leaf,
        sources=g2026_srcs,
        additional_instruction="Confirm the category and year (68th Grammys, 2026). If a work title is provided, ensure it matches."
    )


async def add_emmys_nodes(evaluator: Evaluator, parent, extracted: ComposerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Emmys",
        desc="Emmy constraint (same sci-fi series, consecutive years 2020 and 2021) + supporting URL(s).",
        parent=parent,
        critical=True
    )
    name = extracted.composer_full_name or "the composer"

    emmy_srcs = extracted.emmys.sources if extracted.emmys else []
    evaluator.add_custom_node(
        result=has_urls(emmy_srcs),
        id="URL_Emmys_2020_2021_Provided",
        desc="At least one URL (or set of URLs) is provided that supports the 2020 and 2021 Emmy wins being consecutive and for the same science fiction series.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Emmy_2020_And_2021_Same_SciFi_Series",
        desc="Composer won Emmy Awards for Outstanding Music Composition for a science fiction series in two consecutive years (2020 and 2021) for the same series.",
        parent=node,
        critical=True
    )
    series = (extracted.emmys.series_name if extracted.emmys else None) or "a science fiction series"
    claim = f"{name} won Emmy Awards for Outstanding Music Composition for a Series in consecutive years 2020 and 2021 for the same series, {series}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=emmy_srcs,
        additional_instruction="Confirm the category (or equivalent phrasing) and that both 2020 and 2021 wins are for the same sci‑fi series."
    )


async def add_film_score_nodes(evaluator: Evaluator, parent, extracted: ComposerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Film_Score_Work",
        desc="Film scoring requirements + supporting URL(s).",
        parent=parent,
        critical=True
    )
    name = extracted.composer_full_name or "the composer"

    # Nolan 2020 film
    nolan_srcs = extracted.nolan_2020.sources if extracted.nolan_2020 else []
    evaluator.add_custom_node(
        result=has_urls(nolan_srcs),
        id="URL_Nolan_Film_2020_Provided",
        desc="At least one URL is provided that supports the 2020 Nolan-directed film score credit.",
        parent=node,
        critical=True
    )
    nolan_leaf = evaluator.add_leaf(
        id="Nolan_Film_2020_Score",
        desc="Composer composed the score for a 2020 film directed by Christopher Nolan.",
        parent=node,
        critical=True
    )
    nolan_title = (extracted.nolan_2020.film_title if extracted.nolan_2020 else None) or "a 2020 film"
    claim_nolan = f"{name} composed the score for {nolan_title} directed by Christopher Nolan in 2020."
    await evaluator.verify(
        claim=claim_nolan,
        node=nolan_leaf,
        sources=nolan_srcs,
        additional_instruction="Confirm that the composer is credited as the score composer for the 2020 Christopher Nolan film."
    )

    # Superhero franchise (at least two films)
    fran_srcs = extracted.superhero_franchise.sources if extracted.superhero_franchise else []
    evaluator.add_custom_node(
        result=has_urls(fran_srcs),
        id="URL_Superhero_Franchise_Films_Provided",
        desc="At least one URL (or set of URLs) is provided that supports the two-or-more same-franchise superhero film score credits.",
        parent=node,
        critical=True
    )
    franchise_leaf = evaluator.add_leaf(
        id="Two_Films_Same_Superhero_Franchise",
        desc="Composer composed scores for at least two films in the same superhero franchise.",
        parent=node,
        critical=True
    )
    films = (extracted.superhero_franchise.film_titles if extracted.superhero_franchise else []) or []
    franchise = (extracted.superhero_franchise.franchise_name if extracted.superhero_franchise else None) or "the same superhero franchise"
    film_list_phrase = ", ".join(f"'{t}'" for t in films[:3]) if films else "at least two films"
    claim_franchise = f"{name} composed scores for {film_list_phrase} within {franchise}."
    await evaluator.verify(
        claim=claim_franchise,
        node=franchise_leaf,
        sources=fran_srcs,
        additional_instruction="Verify that at least two listed films belong to the same superhero franchise and that the composer scored both."
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
    Evaluate an answer for the Swedish composer constraints task.
    """
    # Initialize evaluator (root is a general non-critical container)
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

    # Extraction
    extracted: ComposerExtraction = await evaluator.extract(
        prompt=prompt_extract_composer(),
        template_class=ComposerExtraction,
        extraction_name="composer_extraction"
    )

    # Build critical Composer Identification node
    composer_node = evaluator.add_parallel(
        id="Composer_Identification",
        desc="Identify the Swedish composer who satisfies all listed constraints and provide URL references supporting each constraint.",
        parent=root,
        critical=True
    )

    # Leaf: Full name exists
    await add_name_node(evaluator, composer_node, extracted)

    # Parallel sub-groups (all critical under the critical parent)
    await add_nationality_nodes(evaluator, composer_node, extracted)
    await add_academy_awards_nodes(evaluator, composer_node, extracted)
    await add_grammys_nodes(evaluator, composer_node, extracted)
    await add_emmys_nodes(evaluator, composer_node, extracted)
    await add_film_score_nodes(evaluator, composer_node, extracted)

    # Return summary
    return evaluator.get_summary()