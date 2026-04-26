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
TASK_ID = "sundance_2021_top_awards_investigation"
TASK_DESCRIPTION = (
    "A film premiered at the 2021 Sundance Film Festival and made history by winning all four top awards in the "
    "U.S. Dramatic category: the Grand Jury Prize, the Audience Award, the Directing Award, and the Special Jury "
    "Ensemble Cast Award. Identify this film and provide the following information: (1) The film's title, "
    "(2) The country from which the film's director's father fled as a 16-year-old refugee during a revolution, "
    "(3) The year of the revolution that caused this refugee crisis, "
    "(4) The professional role (job title) held by a deaf supporting actor's father in the actor's hometown, and "
    "(5) The name of the city where this deaf supporting actor was born and where their father served in that "
    "professional role. All information should be verifiable through reliable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FilmSection(BaseModel):
    title: Optional[str] = None
    # optional free-text description of premiere/awards claims (extracted verbatim from answer if present)
    premiere_awards_summary: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DirectorSection(BaseModel):
    name: Optional[str] = None
    birth_place: Optional[str] = None  # e.g., "Cambridge, Massachusetts"
    education_bfa_cmu_drama: Optional[str] = None  # verbatim proof text from answer if present
    father_nationality: Optional[str] = None  # e.g., "Hungarian"
    father_profession: Optional[str] = None  # e.g., "public artist"
    father_country_fled: Optional[str] = None  # e.g., "Hungary"
    father_refugee_age: Optional[str] = None  # e.g., "16"
    father_revolution_year: Optional[str] = None  # e.g., "1956"
    mother_nationality: Optional[str] = None  # e.g., "Welsh"
    mother_profession: Optional[str] = None  # e.g., "public artist"
    sources: List[str] = Field(default_factory=list)


class ActorSection(BaseModel):
    name: Optional[str] = None
    deaf_status: Optional[str] = None  # e.g., "deaf"
    birth_city: Optional[str] = None  # e.g., "Mesa"
    birth_state: Optional[str] = None  # e.g., "Arizona"
    birth_date: Optional[str] = None  # e.g., "July 24, 1968"
    father_name: Optional[str] = None  # e.g., "Leonard 'Len' Kotsur"
    father_role: Optional[str] = None  # e.g., "Chief of Police"
    father_city: Optional[str] = None  # e.g., "Mesa"
    father_incident_date: Optional[str] = None  # e.g., "March 28, 1987"
    father_incident_cause: Optional[str] = None  # e.g., "car accident caused by a drunk driver"
    father_death_date: Optional[str] = None  # e.g., "July 5, 2001"
    father_death_cause: Optional[str] = None  # e.g., "kidney failure and pneumonia"
    education_gallaudet_years_program: Optional[str] = None  # e.g., "Gallaudet University 1987–1989, studying theater, television, and film"
    oscar_2022_best_supporting_actor: Optional[str] = None  # free text confirmation if present
    first_deaf_male_academy_award: Optional[str] = None  # free text confirmation if present
    sources: List[str] = Field(default_factory=list)


class InvestigationExtraction(BaseModel):
    film: Optional[FilmSection] = None
    director: Optional[DirectorSection] = None
    actor: Optional[ActorSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_investigation() -> str:
    return """
    Extract the structured information explicitly stated in the answer for this Sundance investigation task.

    You must only extract what is explicitly present in the answer text. Do not infer or invent. If something is not present, return null (for single fields) or an empty list (for sources).

    Required sections and fields:

    film:
      - title: The film's title.
      - premiere_awards_summary: If the answer provides a summary stating premiere at 2021 Sundance and winning all four U.S. Dramatic awards (Grand Jury Prize, Audience Award, Directing Award, Special Jury Ensemble Cast Award), extract the sentence(s) verbatim; otherwise null.
      - sources: All URLs cited for the film's premiere/awards/history. Use only valid URLs explicitly included in the answer.

    director:
      - name: The director's name (if stated).
      - birth_place: Place of birth (e.g., "Cambridge, Massachusetts").
      - education_bfa_cmu_drama: Verbatim statement (if present) confirming graduation from Carnegie Mellon University School of Drama with a BFA.
      - father_nationality: e.g., "Hungarian".
      - father_profession: e.g., "public artist".
      - father_country_fled: e.g., "Hungary".
      - father_refugee_age: e.g., "16".
      - father_revolution_year: e.g., "1956".
      - mother_nationality: e.g., "Welsh".
      - mother_profession: e.g., "public artist".
      - sources: All URLs cited for director background and father/mother details.

    actor:
      - name: The supporting actor's name (e.g., Troy Kotsur), if stated.
      - deaf_status: Extract the word/phrase indicating the actor is deaf (e.g., "deaf"), if stated.
      - birth_city: e.g., "Mesa".
      - birth_state: e.g., "Arizona".
      - birth_date: e.g., "July 24, 1968".
      - father_name: e.g., 'Leonard "Len" Kotsur'.
      - father_role: e.g., "Chief of Police".
      - father_city: e.g., "Mesa".
      - father_incident_date: e.g., "March 28, 1987".
      - father_incident_cause: e.g., "car accident caused by a drunk driver".
      - father_death_date: e.g., "July 5, 2001".
      - father_death_cause: e.g., "kidney failure and pneumonia".
      - education_gallaudet_years_program: Verbatim statement (e.g., "attended Gallaudet University from 1987 to 1989, studying theater, television, and film").
      - oscar_2022_best_supporting_actor: Verbatim statement or mention indicating the actor won the 2022 Oscar for Best Supporting Actor.
      - first_deaf_male_academy_award: Verbatim statement or mention indicating the actor is the first deaf male actor to win an Academy Award.
      - sources: All URLs cited for the actor background and father details.

    SPECIAL RULES FOR URL EXTRACTION:
    - Only extract valid URLs explicitly present in the answer (plain URL or markdown link). Do not infer URLs.
    - If a URL lacks a protocol, prepend http://
    - If no URLs are present for a section, return an empty list.

    Output a single JSON object matching the schema described.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_film_section(
    evaluator: Evaluator,
    parent: Any,
    film: Optional[FilmSection],
) -> None:
    """
    Build and verify the Film_Identification branch.
    """
    film_node = evaluator.add_parallel(
        id="Film_Identification",
        desc="Correctly identify the film and ensure it satisfies the Sundance premiere/awards/history constraints.",
        parent=parent,
        critical=True,
    )

    title_str = (film.title or "").strip()
    film_sources = film.sources if film and film.sources else []

    # Leaf: Film_Title_Is_Correct
    title_leaf = evaluator.add_leaf(
        id="Film_Title_Is_Correct",
        desc="Answer states the film title and it is the same film that satisfies the Sundance constraints below.",
        parent=film_node,
        critical=True,
    )
    # Verify that at least one cited film source page is about the film titled {title_str}
    title_claim = f"This cited source page is about the film titled \"{title_str}\"."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=film_sources,
        additional_instruction="Allow minor title formatting differences. The page should clearly indicate the film title."
    )

    # Leaf: Film_Sundance_Premiere_Awards_History
    history_leaf = evaluator.add_leaf(
        id="Film_Sundance_Premiere_Awards_History",
        desc="The film premiered at the 2021 Sundance Film Festival and is the first in Sundance history to win all four top awards in the U.S. Dramatic category: Grand Jury Prize, Audience Award, Directing Award, and Special Jury Ensemble Cast Award.",
        parent=film_node,
        critical=True,
    )
    history_claim = (
        f"The film \"{title_str}\" premiered at the 2021 Sundance Film Festival and is the first in Sundance history "
        "to win all four top U.S. Dramatic awards: the Grand Jury Prize, the Audience Award, the Directing Award, and the Special Jury Award for Ensemble Cast."
    )
    await evaluator.verify(
        claim=history_claim,
        node=history_leaf,
        sources=film_sources,
        additional_instruction="Verify both the premiere (Sundance 2021) and the historic sweep of all four top U.S. Dramatic awards."
    )


async def verify_director_section(
    evaluator: Evaluator,
    parent: Any,
    director: Optional[DirectorSection],
) -> None:
    """
    Build and verify the Director_Constraints_And_Requested_Details branch.
    """
    dir_node = evaluator.add_parallel(
        id="Director_Constraints_And_Requested_Details",
        desc="Director-related constraints are satisfied, including the requested father-refugee country and revolution-year details.",
        parent=parent,
        critical=True,
    )

    dir_name = (director.name or "the film's director").strip()
    dir_sources = director.sources if director and director.sources else []

    # Director_Born_Cambridge_MA
    born_leaf = evaluator.add_leaf(
        id="Director_Born_Cambridge_MA",
        desc="The film's director was born in Cambridge, Massachusetts.",
        parent=dir_node,
        critical=True,
    )
    born_claim = f"{dir_name} was born in Cambridge, Massachusetts (MA)."
    await evaluator.verify(
        claim=born_claim,
        node=born_leaf,
        sources=dir_sources,
        additional_instruction="Allow equivalent phrasing like 'Cambridge, MA' or 'Cambridge, Massachusetts, USA'."
    )

    # Director_BFA_CMU_Drama
    bfa_leaf = evaluator.add_leaf(
        id="Director_BFA_CMU_Drama",
        desc="The film's director graduated from Carnegie Mellon University School of Drama with a BFA.",
        parent=dir_node,
        critical=True,
    )
    bfa_claim = f"{dir_name} graduated from Carnegie Mellon University School of Drama with a BFA (Bachelor of Fine Arts)."
    await evaluator.verify(
        claim=bfa_claim,
        node=bfa_leaf,
        sources=dir_sources,
        additional_instruction="The source should explicitly indicate Carnegie Mellon University School of Drama and a BFA degree."
    )

    # Director_Father_Hungarian_Public_Artist
    father_nat_prof_leaf = evaluator.add_leaf(
        id="Director_Father_Hungarian_Public_Artist",
        desc="The film's director's father is Hungarian and works as a public artist.",
        parent=dir_node,
        critical=True,
    )
    father_nat = (director.father_nationality or "Hungarian").strip()
    father_prof = (director.father_profession or "public artist").strip()
    father_nat_prof_claim = f"The director's father is {father_nat} and works as a {father_prof}."
    await evaluator.verify(
        claim=father_nat_prof_claim,
        node=father_nat_prof_leaf,
        sources=dir_sources,
        additional_instruction="The source should indicate both nationality (Hungarian) and profession (public artist)."
    )

    # Director_Father_Refugee_Details_Country_Age_RevolutionYear
    father_refugee_leaf = evaluator.add_leaf(
        id="Director_Father_Refugee_Details_Country_Age_RevolutionYear",
        desc="The director's father fled Hungary as a 16-year-old refugee during the 1956 Hungarian Revolution, and the answer explicitly provides (a) the country fled from and (b) the revolution year.",
        parent=dir_node,
        critical=True,
    )
    country_fled = (director.father_country_fled or "Hungary").strip()
    refugee_age = (director.father_refugee_age or "16").strip()
    revolution_year = (director.father_revolution_year or "1956").strip()
    father_refugee_claim = (
        f"The director's father fled {country_fled} as a {refugee_age}-year-old refugee during the {revolution_year} Hungarian Revolution."
    )
    await evaluator.verify(
        claim=father_refugee_claim,
        node=father_refugee_leaf,
        sources=dir_sources,
        additional_instruction="Confirm both the country (Hungary), the age (16), and the revolution year (1956)."
    )

    # Director_Mother_Welsh_Public_Artist
    mother_leaf = evaluator.add_leaf(
        id="Director_Mother_Welsh_Public_Artist",
        desc="The film's director's mother is Welsh and works as a public artist.",
        parent=dir_node,
        critical=True,
    )
    mother_nat = (director.mother_nationality or "Welsh").strip()
    mother_prof = (director.mother_profession or "public artist").strip()
    mother_claim = f"The director's mother is {mother_nat} and works as a {mother_prof}."
    await evaluator.verify(
        claim=mother_claim,
        node=mother_leaf,
        sources=dir_sources,
        additional_instruction="The source should indicate both nationality (Welsh) and profession (public artist)."
    )


async def verify_actor_section(
    evaluator: Evaluator,
    parent: Any,
    actor: Optional[ActorSection],
) -> None:
    """
    Build and verify the Supporting_Actor_Constraints_And_Requested_Details branch.
    """
    actor_node = evaluator.add_parallel(
        id="Supporting_Actor_Constraints_And_Requested_Details",
        desc="Supporting-actor-related constraints are satisfied, including the requested father job-title and city details.",
        parent=parent,
        critical=True,
    )

    actor_name = (actor.name or "the supporting actor").strip()
    actor_sources = actor.sources if actor and actor.sources else []

    # Supporting_Actor_Deaf
    deaf_leaf = evaluator.add_leaf(
        id="Supporting_Actor_Deaf",
        desc="The supporting actor is deaf.",
        parent=actor_node,
        critical=True,
    )
    deaf_claim = f"{actor_name} is deaf."
    await evaluator.verify(
        claim=deaf_claim,
        node=deaf_leaf,
        sources=actor_sources,
        additional_instruction="Explicit phrasing confirming the actor is deaf should be present in the source."
    )

    # Supporting_Actor_Born_Mesa_AZ_July24_1968
    born_city = (actor.birth_city or "Mesa").strip()
    born_state = (actor.birth_state or "Arizona").strip()
    born_date = (actor.birth_date or "July 24, 1968").strip()
    born_leaf = evaluator.add_leaf(
        id="Supporting_Actor_Born_Mesa_AZ_July24_1968",
        desc="The supporting actor was born in Mesa, Arizona on July 24, 1968.",
        parent=actor_node,
        critical=True,
    )
    born_claim = f"{actor_name} was born in {born_city}, {born_state} on {born_date}."
    await evaluator.verify(
        claim=born_claim,
        node=born_leaf,
        sources=actor_sources,
        additional_instruction="Allow equivalent formats for the date and the state (e.g., 'AZ' for Arizona)."
    )

    # Actor_Father_Is_Leonard_Len_Kotsur
    father_name = (actor.father_name or "Leonard \"Len\" Kotsur").strip()
    father_name_leaf = evaluator.add_leaf(
        id="Actor_Father_Is_Leonard_Len_Kotsur",
        desc="The actor's father was Leonard \"Len\" Kotsur.",
        parent=actor_node,
        critical=True,
    )
    father_name_claim = f"{actor_name}'s father was {father_name}."
    await evaluator.verify(
        claim=father_name_claim,
        node=father_name_leaf,
        sources=actor_sources,
        additional_instruction="The source should clearly state the father's name."
    )

    # Actor_Father_Role_And_City_Provided_And_Correct
    father_role = (actor.father_role or "Chief of Police").strip()
    father_city = (actor.father_city or "Mesa").strip()
    role_city_leaf = evaluator.add_leaf(
        id="Actor_Father_Role_And_City_Provided_And_Correct",
        desc="The answer provides the father’s professional role and the city, and they match the constraint: father served as Chief of Police in Mesa, Arizona (and Mesa is the actor’s birth city).",
        parent=actor_node,
        critical=True,
    )
    role_city_claim = (
        f"{actor_name}'s father served as {father_role} in {father_city}, Arizona, which is also {actor_name}'s birth city."
    )
    await evaluator.verify(
        claim=role_city_claim,
        node=role_city_leaf,
        sources=actor_sources,
        additional_instruction="Confirm both the specific role (Chief of Police) and the city (Mesa, Arizona), and that it matches the actor's birth city."
    )

    # Actor_Father_Paralyzed_Incident_March28_1987
    incident_date = (actor.father_incident_date or "March 28, 1987").strip()
    incident_cause = (actor.father_incident_cause or "a car accident caused by a drunk driver").strip()
    paralyzed_leaf = evaluator.add_leaf(
        id="Actor_Father_Paralyzed_Incident_March28_1987",
        desc="The actor's father was paralyzed in a car accident caused by a drunk driver on March 28, 1987.",
        parent=actor_node,
        critical=True,
    )
    paralyzed_claim = f"{actor_name}'s father was paralyzed in {incident_cause} on {incident_date}."
    await evaluator.verify(
        claim=paralyzed_claim,
        node=paralyzed_leaf,
        sources=actor_sources,
        additional_instruction="The source should clearly indicate paralysis, the cause (drunk driver car accident), and the date."
    )

    # Actor_Father_Death_July5_2001_KidneyFailure_Pneumonia
    death_date = (actor.father_death_date or "July 5, 2001").strip()
    death_cause = (actor.father_death_cause or "complications of kidney failure and pneumonia").strip()
    death_leaf = evaluator.add_leaf(
        id="Actor_Father_Death_July5_2001_KidneyFailure_Pneumonia",
        desc="The actor's father died on July 5, 2001, from complications of kidney failure and pneumonia.",
        parent=actor_node,
        critical=True,
    )
    death_claim = f"{actor_name}'s father died on {death_date} from {death_cause}."
    await evaluator.verify(
        claim=death_claim,
        node=death_leaf,
        sources=actor_sources,
        additional_instruction="The source should specify both the date and cause of death."
    )

    # Actor_Gallaudet_1987_1989_Theater_TV_Film
    gallaudet_leaf = evaluator.add_leaf(
        id="Actor_Gallaudet_1987_1989_Theater_TV_Film",
        desc="The actor attended Gallaudet University from 1987 to 1989, studying theater, television, and film.",
        parent=actor_node,
        critical=True,
    )
    gallaudet_claim = f"{actor_name} attended Gallaudet University from 1987 to 1989, studying theater, television, and film."
    await evaluator.verify(
        claim=gallaudet_claim,
        node=gallaudet_leaf,
        sources=actor_sources,
        additional_instruction="Equivalent phrasing is acceptable as long as years and fields of study are clear."
    )

    # Actor_Won_2022_Oscar_Best_Supporting_Actor
    oscar_leaf = evaluator.add_leaf(
        id="Actor_Won_2022_Oscar_Best_Supporting_Actor",
        desc="The actor won the 2022 Oscar for Best Supporting Actor.",
        parent=actor_node,
        critical=True,
    )
    oscar_claim = f"{actor_name} won the 2022 Oscar for Best Supporting Actor."
    await evaluator.verify(
        claim=oscar_claim,
        node=oscar_leaf,
        sources=actor_sources,
        additional_instruction="The source should mention the Academy Awards/Oscars and the specific category/year."
    )

    # Actor_First_Deaf_Male_Academy_Award
    first_deaf_male_leaf = evaluator.add_leaf(
        id="Actor_First_Deaf_Male_Academy_Award",
        desc="The actor is the first deaf male actor to win an Academy Award.",
        parent=actor_node,
        critical=True,
    )
    first_deaf_male_claim = f"{actor_name} is the first deaf male actor to win an Academy Award."
    await evaluator.verify(
        claim=first_deaf_male_claim,
        node=first_deaf_male_leaf,
        sources=actor_sources,
        additional_instruction="The source should clearly indicate 'first deaf male actor' and an Academy Award (Oscar)."
    )


async def verify_source_verifiability(
    evaluator: Evaluator,
    parent: Any,
    film: Optional[FilmSection],
    director: Optional[DirectorSection],
    actor: Optional[ActorSection],
) -> None:
    """
    Build and verify the Source_Verifiability branch (coverage by citations).
    """
    src_node = evaluator.add_parallel(
        id="Source_Verifiability",
        desc="All required claims are supported by reliable, verifiable sources sufficient to substantiate each constraint and each requested output.",
        parent=parent,
        critical=True,
    )

    coverage_leaf = evaluator.add_leaf(
        id="Citations_Cover_Film_Director_Actor_Claims",
        desc="Provides citations/URLs that collectively verify: (1) film premiere/awards/history, (2) director constraints and father-refugee details, and (3) supporting actor + father constraints and requested role/city outputs.",
        parent=src_node,
        critical=True,
    )

    coverage_claim = (
        "The answer provides citations/URLs that collectively verify: "
        "(1) the film’s 2021 Sundance premiere and historic sweep of the four top U.S. Dramatic awards, "
        "(2) the director’s constraints and father refugee details including the specific country fled and the revolution year, and "
        "(3) the supporting actor’s constraints including deaf status, birth details, father’s name, role (Chief of Police), city (Mesa), incident/death details, "
        "Gallaudet attendance, 2022 Oscar, and status as the first deaf male actor to win an Academy Award."
    )
    # Here we rely on the judge to read the answer and confirm that citations are present and cover all categories.
    await evaluator.verify(
        claim=coverage_claim,
        node=coverage_leaf,
        additional_instruction="Judge based on whether the answer includes sufficient citations/URLs that cover all listed categories; consider earlier verifications reference these citations."
    )


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
    Evaluate an answer for the Sundance 2021 U.S. Dramatic awards investigation task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # We'll create an explicit critical sequential root below
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_investigation(),
        template_class=InvestigationExtraction,
        extraction_name="structured_extraction"
    )

    # Build the explicit critical sequential root to match rubric
    inv_root = evaluator.add_sequential(
        id="Investigation_Root",
        desc="Identify the Sundance film and provide the requested facts while satisfying all listed constraints, with reliable sourcing.",
        parent=root,
        critical=True,
    )

    # Film identification branch
    await verify_film_section(evaluator, inv_root, extraction.film)

    # Director constraints and requested details branch
    await verify_director_section(evaluator, inv_root, extraction.director)

    # Supporting actor constraints and requested details branch
    await verify_actor_section(evaluator, inv_root, extraction.actor)

    # Source verifiability branch
    await verify_source_verifiability(evaluator, inv_root, extraction.film, extraction.director, extraction.actor)

    # Return standardized summary
    return evaluator.get_summary()