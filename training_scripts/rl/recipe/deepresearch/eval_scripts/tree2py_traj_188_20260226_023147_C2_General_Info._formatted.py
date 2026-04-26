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
TASK_ID = "power_rangers_director_2017"
TASK_DESCRIPTION = """
Who directed the 2017 superhero film reboot in which the actor born in 1994 who played Billy Hargrove in Stranger Things portrayed Jason Scott, the Red Ranger?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmDirectorQueryExtraction(BaseModel):
    """
    Structured extraction of the answer for the Power Rangers (2017) director query.
    """
    # Target film identification
    film_title: Optional[str] = None
    film_year: Optional[str] = None

    # Actor clues
    actor_name: Optional[str] = None
    actor_birth_year: Optional[str] = None
    actor_role_stranger_things: Optional[str] = None
    actor_role_power_rangers_character: Optional[str] = None

    # Director
    director_name: Optional[str] = None

    # Source URLs explicitly cited in the answer
    film_sources: List[str] = Field(default_factory=list)
    actor_sources: List[str] = Field(default_factory=list)
    director_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_main_fields() -> str:
    return """
    Extract the following structured information from the provided answer text related to the question:
    1) film_title: The film title explicitly identified in the answer (e.g., "Power Rangers")
    2) film_year: The year of the film as stated in the answer (e.g., "2017")
    3) actor_name: The actor referenced in the answer (e.g., "Dacre Montgomery")
    4) actor_birth_year: The actor's birth year as stated in the answer (e.g., "1994")
    5) actor_role_stranger_things: The role the actor played in Stranger Things as stated (e.g., "Billy Hargrove")
    6) actor_role_power_rangers_character: The character the actor portrayed in the Power Rangers film (e.g., "Jason Scott (Red Ranger)")
    7) director_name: The director's name for the identified film as stated in the answer (e.g., "Dean Israelite")

    Also extract the URL sources explicitly cited in the answer to support each of the following aspects:
    - film_sources: URLs that support the identification of the film and its year or reboot context.
    - actor_sources: URLs that support the actor clues (birth year 1994, Stranger Things role "Billy Hargrove", and portrayal of Jason Scott/Red Ranger in Power Rangers (2017)).
    - director_sources: URLs that support or verify the director of the identified film. Prefer official sources such as studio/distributor websites, official press releases, or official credits listings.

    SPECIAL RULES FOR URL EXTRACTION:
    - Only include URLs explicitly present in the answer text (plain URLs or markdown links). Do not invent URLs.
    - Ensure URLs are valid. If a URL is missing a protocol, prepend "http://".
    - Deduplicate URLs.
    - If no URLs are present for a category, return an empty array for that field.

    If any field is missing from the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_and_dedup_urls(*url_lists: List[str]) -> List[str]:
    """Combine multiple URL lists and deduplicate while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for url in lst:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extraction: FilmDirectorQueryExtraction
) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """

    # TaskCompletion (critical sequential)
    task_node = evaluator.add_sequential(
        id="TaskCompletion",
        desc="Identify who directed the specified 2017 superhero film reboot described in the question/constraints, with verification from official sources.",
        parent=root_node,
        critical=True
    )

    # 1) VerifyTargetFilmAndClues (critical parallel)
    verify_target_node = evaluator.add_parallel(
        id="VerifyTargetFilmAndClues",
        desc="Verify the target film and that it matches the actor/role clues given in the question/constraints.",
        parent=task_node,
        critical=True
    )

    # 1.a) FilmIsPowerRangers2017 (critical leaf)
    film_leaf = evaluator.add_leaf(
        id="FilmIsPowerRangers2017",
        desc="The film is the 2017 superhero film reboot titled 'Power Rangers'.",
        parent=verify_target_node,
        critical=True
    )
    # Build claim using question constraints; use any film/actor sources provided
    # This claim verifies title, year 2017, and reboot nature.
    film_claim = "The film in question is 'Power Rangers' (2017), a superhero film reboot."
    film_sources = combine_and_dedup_urls(extraction.film_sources, extraction.actor_sources)

    await evaluator.verify(
        claim=film_claim,
        node=film_leaf,
        sources=film_sources,
        additional_instruction=(
            "Confirm that the provided webpage(s) explicitly indicate the film title 'Power Rangers', the year 2017, "
            "and that it is a superhero film reboot. Allow reasonable wording variations (e.g., 'reimagining' or 'reboot'). "
            "If none of the provided URLs support this, mark as not supported."
        )
    )

    # 1.b) ActorCluesMatch (critical leaf)
    actor_leaf = evaluator.add_leaf(
        id="ActorCluesMatch",
        desc="The identifying actor matches all stated clues: born in 1994; played Billy Hargrove in Stranger Things; and portrayed Jason Scott (the Red Ranger) in Power Rangers (2017).",
        parent=verify_target_node,
        critical=True
    )
    # Compose the claim based on extracted data; tolerate minor variations
    actor_name = extraction.actor_name or "the actor"
    actor_claim = (
        f"{actor_name} was born in 1994, played Billy Hargrove in Stranger Things, "
        f"and portrayed Jason Scott (the Red Ranger) in Power Rangers (2017)."
    )
    await evaluator.verify(
        claim=actor_claim,
        node=actor_leaf,
        sources=extraction.actor_sources,
        additional_instruction=(
            "Verify all three parts together from the provided URLs: birth year 1994, Stranger Things role 'Billy Hargrove', "
            "and portrayal of Jason Scott (Red Ranger) in Power Rangers (2017). "
            "Minor formatting differences (e.g., middle names, casing) are acceptable. "
            "Fail if any component is not supported."
        )
    )

    # 2) ProvideAndVerifyDirector (critical parallel)
    director_node = evaluator.add_parallel(
        id="ProvideAndVerifyDirector",
        desc="Provide the director of Power Rangers (2017) and verify using official sources about the film.",
        parent=task_node,
        critical=True
    )

    # 2.a) DirectorNameProvided (critical existence check)
    director_name_provided = evaluator.add_custom_node(
        result=bool(extraction.director_name and extraction.director_name.strip()),
        id="DirectorNameProvided",
        desc="The response clearly states the director’s name for Power Rangers (2017).",
        parent=director_node,
        critical=True
    )

    # 2.b) DirectorVerifiedFromOfficialSources (critical leaf)
    director_verify_leaf = evaluator.add_leaf(
        id="DirectorVerifiedFromOfficialSources",
        desc="The response verifies the director using at least one official source about the 2017 Power Rangers film (e.g., studio/distributor official site, official press materials, official credits listing).",
        parent=director_node,
        critical=True
    )

    director_name = extraction.director_name or ""
    director_claim = f"The director of the film 'Power Rangers' (2017) is {director_name}."
    await evaluator.verify(
        claim=director_claim,
        node=director_verify_leaf,
        sources=extraction.director_sources,
        additional_instruction=(
            "Confirm the director using at least one official source about the 2017 Power Rangers film. "
            "Official sources include studio/distributor websites (e.g., Lionsgate or Saban), official press releases, or "
            "official credits listings. If none of the provided URLs are official or they do not support the claim, mark as not supported."
        )
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
    Evaluate an answer for the Power Rangers (2017) director identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Follow sequential structure for overall task
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_main_fields(),
        template_class=FilmDirectorQueryExtraction,
        extraction_name="extracted_fields",
    )

    # Build tree and run verifications
    await build_verification_tree(evaluator, root, extraction)

    # Return standardized summary
    return evaluator.get_summary()