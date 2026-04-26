import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "director_cinematographer_collab"
TASK_DESCRIPTION = (
    "Identify the film director who meets ALL of the following criteria: "
    "(1) Born in the 1960s (between 1960 and 1969, inclusive); "
    "(2) Released their feature-length directorial debut film in the 1990s; "
    "(3) Established an ongoing collaboration with a single cinematographer starting from one of their films released in the 2000s or later; "
    "(4) This collaboration includes at least five feature films together; "
    "(5) That cinematographer has received at least three Academy Award nominations for Best Cinematography specifically for films directed by this director; "
    "(6) That same cinematographer has won the Academy Award for Best Cinematography at least twice in their career (across all work). "
    "Provide the director's full name and the full name of their long-term cinematographer collaborator."
)


class PairExtraction(BaseModel):
    director_name: Optional[str] = None
    director_birth_year: Optional[str] = None
    feature_directorial_debut_title: Optional[str] = None
    feature_directorial_debut_year: Optional[str] = None
    cinematographer_name: Optional[str] = None

    collaboration_start_film_title: Optional[str] = None
    collaboration_start_year: Optional[str] = None

    collaboration_feature_films: List[str] = Field(default_factory=list)
    collaboration_feature_years: List[str] = Field(default_factory=list)

    nominations_with_director_count: Optional[str] = None
    nominations_with_director_films: List[str] = Field(default_factory=list)

    cinematographer_career_wins_count: Optional[str] = None

    sources: List[str] = Field(default_factory=list)


def prompt_extract_pair() -> str:
    return (
        "Extract the following information exactly as stated in the answer (do not invent data):\n"
        "1. director_name: The full name of the director identified.\n"
        "2. director_birth_year: The director's year of birth (4-digit).\n"
        "3. feature_directorial_debut_title: The title of the director's feature-length directorial debut film.\n"
        "4. feature_directorial_debut_year: The year that debut film was released (4-digit).\n"
        "5. cinematographer_name: The full name of the long-term cinematographer collaborator.\n"
        "6. collaboration_start_film_title: The title of the film where this ongoing collaboration began (should be a film from the 2000s or later). If not specified, return null.\n"
        "7. collaboration_start_year: The release year of that start film (4-digit). If not specified, return null.\n"
        "8. collaboration_feature_films: A list of feature-length film titles the director and this cinematographer worked on together (as many as are listed in the answer).\n"
        "9. collaboration_feature_years: The corresponding release years (4-digit) for the films in collaboration_feature_films (same order; if unknown for a film, use null for that position).\n"
        "10. nominations_with_director_count: The count of Academy Award nominations for Best Cinematography that this cinematographer received for films directed by this director (as stated in the answer). If unspecified, return null.\n"
        "11. nominations_with_director_films: List the specific film titles (directed by this director) for which the cinematographer received those nominations, as stated in the answer (if listed).\n"
        "12. cinematographer_career_wins_count: The total number of Academy Award wins for Best Cinematography this cinematographer has across their career (as stated in the answer). If unspecified, return null.\n"
        "13. sources: Extract ALL URLs explicitly cited in the answer that support any of the above claims. Include Wikipedia, Academy Awards (oscars.org), film pages, interviews, or any other links mentioned. Only include valid URLs actually present in the answer."
    )


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        v = u.strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


async def build_verification_tree(evaluator: Evaluator, root, data: PairExtraction) -> None:
    all_sources = _dedup_urls(data.sources or [])

    task_node = evaluator.add_parallel(
        id="Task",
        desc="Identify the director and long-term cinematographer collaborator, verifying all specified criteria",
        parent=root,
        critical=True,
    )

    # Director Basic Criteria (parallel, critical)
    dir_basic = evaluator.add_parallel(
        id="Director_Basic_Criteria",
        desc="Verify the identified director meets the birth decade and debut film release criteria",
        parent=task_node,
        critical=True,
    )

    # Birth Decade Verification
    birth_leaf = evaluator.add_leaf(
        id="Birth_Decade_Verification",
        desc="The director was born between 1960 and 1969 (inclusive)",
        parent=dir_basic,
        critical=True,
    )
    birth_claim = (
        f"The director {data.director_name or '[unknown director]'} was born in {data.director_birth_year or '[unknown year]'}, "
        "which is between 1960 and 1969 inclusive."
    )
    await evaluator.verify(
        claim=birth_claim,
        node=birth_leaf,
        sources=all_sources,
        additional_instruction=(
            "Verify the birth year from authoritative sources. The check passes only if the director's birth year is between 1960 and 1969 inclusive."
        ),
    )

    # Debut Film Era Verification
    debut_leaf = evaluator.add_leaf(
        id="Debut_Film_Era_Verification",
        desc="The director's feature-length directorial debut film was released between 1990 and 1999 (inclusive)",
        parent=dir_basic,
        critical=True,
    )
    debut_claim = (
        f"The director's feature-length directorial debut film is '{data.feature_directorial_debut_title or '[unknown title]'}' "
        f"released in {data.feature_directorial_debut_year or '[unknown year]'}, which falls between 1990 and 1999 inclusive."
    )
    await evaluator.verify(
        claim=debut_claim,
        node=debut_leaf,
        sources=all_sources,
        additional_instruction=(
            "Focus on confirming the film is the director's first feature-length directorial release and the release year is in 1990–1999 inclusive."
        ),
    )

    # Cinematographer Collaboration and Achievements (sequential, critical)
    coll_seq = evaluator.add_sequential(
        id="Cinematographer_Collaboration_and_Achievements",
        desc="Verify the director's ongoing collaboration with a single cinematographer and that cinematographer's Academy Award record",
        parent=task_node,
        critical=True,
    )

    # Collaboration Definition (parallel, critical)
    coll_def = evaluator.add_parallel(
        id="Collaboration_Definition",
        desc="Verify the collaboration is with a single cinematographer and began from a film released in the 2000s or later",
        parent=coll_seq,
        critical=True,
    )

    # Single Cinematographer Verification
    single_dp_leaf = evaluator.add_leaf(
        id="Single_Cinematographer_Verification",
        desc="The director established an ongoing collaboration with a single (one) cinematographer",
        parent=coll_def,
        critical=True,
    )
    single_claim = (
        f"The director {data.director_name or '[unknown director]'} established an ongoing collaboration with a single cinematographer, "
        f"{data.cinematographer_name or '[unknown cinematographer]'}."
    )
    await evaluator.verify(
        claim=single_claim,
        node=single_dp_leaf,
        sources=all_sources,
        additional_instruction=(
            "Confirm that sources identify one primary long-term cinematographer collaborator (a single person) for this director, "
            "not multiple equally primary collaborators."
        ),
    )

    # Collaboration Start Timing Verification
    start_leaf = evaluator.add_leaf(
        id="Collaboration_Start_Timing_Verification",
        desc="The collaboration started from one of the director's films released in the 2000s or later",
        parent=coll_def,
        critical=True,
    )
    start_claim = (
        f"The ongoing collaboration between {data.director_name or '[unknown director]'} and "
        f"{data.cinematographer_name or '[unknown cinematographer]'} started with the film "
        f"'{data.collaboration_start_film_title or '[unknown film]'}' released in {data.collaboration_start_year or '[unknown year]'}, "
        "which is in 2000 or later."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=all_sources,
        additional_instruction=(
            "Verify that the collaboration began from a film released in the 2000s or later (year >= 2000). "
            "If the pair worked together earlier, the 'ongoing collaboration' must be established starting with a 2000s-or-later film."
        ),
    )

    # Collaboration Film Count Verification (critical leaf after definition)
    count_leaf = evaluator.add_leaf(
        id="Collaboration_Film_Count_Verification",
        desc="The director and that cinematographer have collaborated on at least five feature films",
        parent=coll_seq,
        critical=True,
    )
    films_list = ", ".join(data.collaboration_feature_films) if data.collaboration_feature_films else "[no films listed]"
    count_claim = (
        f"The director {data.director_name or '[unknown director]'} and cinematographer "
        f"{data.cinematographer_name or '[unknown cinematographer]'} have collaborated on at least five feature-length films together. "
        f"Examples include: {films_list}."
    )
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=all_sources,
        additional_instruction=(
            "Count only feature-length films (exclude shorts, TV episodes, or commercials). "
            "The check passes if there are five or more collaborative feature films."
        ),
    )

    # Cinematographer Achievements (parallel, critical)
    ach_node = evaluator.add_parallel(
        id="Cinematographer_Achievements",
        desc="Verify the collaborating cinematographer's Academy Award nominations/wins criteria",
        parent=coll_seq,
        critical=True,
    )

    # Oscar Nominations With Director
    noms_leaf = evaluator.add_leaf(
        id="Oscar_Nominations_With_Director",
        desc="The cinematographer received at least three Academy Award nominations for Best Cinematography specifically for films directed by this director",
        parent=ach_node,
        critical=True,
    )
    noms_list = ", ".join(data.nominations_with_director_films) if data.nominations_with_director_films else "[no films listed]"
    noms_claim = (
        f"The cinematographer {data.cinematographer_name or '[unknown cinematographer]'} has received at least three "
        f"Academy Award nominations for Best Cinematography specifically for films directed by {data.director_name or '[unknown director]'}. "
        f"Nominated films mentioned include: {noms_list}."
    )
    await evaluator.verify(
        claim=noms_claim,
        node=noms_leaf,
        sources=all_sources,
        additional_instruction=(
            "Confirm Academy Award nominations in the Best Cinematography category tied to films directed by the specified director. "
            "Count only those nominations linked to this director's films; the threshold is ≥ 3."
        ),
    )

    # Career Oscar Wins
    wins_leaf = evaluator.add_leaf(
        id="Career_Oscar_Wins",
        desc="The cinematographer has won the Academy Award for Best Cinematography at least twice in their career (across all work)",
        parent=ach_node,
        critical=True,
    )
    wins_claim = (
        f"The cinematographer {data.cinematographer_name or '[unknown cinematographer]'} has won the Academy Award "
        "for Best Cinematography at least two times in their career (across all their work)."
    )
    await evaluator.verify(
        claim=wins_claim,
        node=wins_leaf,
        sources=all_sources,
        additional_instruction=(
            "Verify total career wins in the Best Cinematography category for this cinematographer (across any films). "
            "The check passes if the number of wins is ≥ 2."
        ),
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_pair(),
        template_class=PairExtraction,
        extraction_name="pair_extraction",
    )

    await build_verification_tree(evaluator, root, extracted)

    return evaluator.get_summary()