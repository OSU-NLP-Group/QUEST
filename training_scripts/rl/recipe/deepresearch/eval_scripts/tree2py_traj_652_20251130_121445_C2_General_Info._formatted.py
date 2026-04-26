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
TASK_ID = "jimmy_olsen_actor_animated_sequel"
TASK_DESCRIPTION = (
    "An actor played the character Jimmy Olsen in a Superman film that was released in 2025. "
    "This same actor is also cast in an upcoming animated film sequel scheduled for release in 2027, "
    "where he will voice one of the main characters' sons. What is the full name of this actor, "
    "what is the title of the animated sequel he is cast in, what specific character type does he voice in that film, "
    "and what is the exact theatrical release date of that animated sequel?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ActorFilmExtraction(BaseModel):
    """
    Structure extracted from the agent's answer.

    actor_full_name: Full name of the actor identified by the answer.
    superman_2025_film_title: Title of the 2025 Superman film (if provided).
    superman_film_urls: All URLs cited in the answer that support the Jimmy Olsen role claim.
    animated_sequel_title: Title of the animated sequel scheduled for 2027.
    animated_sequel_role_desc: The answer's description of the role (e.g., 'voices the son of [main character]').
    animated_sequel_release_date: Exact theatrical release date as stated in the answer (e.g., 'July 15, 2027').
    animated_sequel_urls: All URLs cited in the answer that support the animated sequel casting, role, and release date.
    """
    actor_full_name: Optional[str] = None

    superman_2025_film_title: Optional[str] = None
    superman_film_urls: List[str] = Field(default_factory=list)

    animated_sequel_title: Optional[str] = None
    animated_sequel_role_desc: Optional[str] = None
    animated_sequel_release_date: Optional[str] = None
    animated_sequel_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_actor_and_film_details() -> str:
    return """
    From the provided answer, extract the following structured information:

    1) actor_full_name:
       - The full name of the actor the answer identifies as having played Jimmy Olsen.
       - Return null if not explicitly provided.

    2) superman_2025_film_title:
       - The title of the Superman film that the answer states was released in 2025.
       - If the title is not explicitly mentioned in the answer, return null.
    
    3) superman_film_urls:
       - All URLs cited in the answer that directly support the claim that this actor played 'Jimmy Olsen' in a Superman film released in 2025.
       - Return an array of URLs. If none are provided, return an empty array.

    4) animated_sequel_title:
       - The official title of the animated film sequel (scheduled for release in 2027) the actor is cast in, as stated in the answer.
       - If the title is not explicitly provided, return null.

    5) animated_sequel_role_desc:
       - The specific role description the answer gives for the actor in the animated sequel.
       - We are particularly interested in whether the answer claims he 'voices one of the main characters' sons'.
       - Return the phrasing used in the answer. If not provided, return null.

    6) animated_sequel_release_date:
       - The exact theatrical release date of the animated sequel as stated in the answer (e.g., 'July 15, 2027').
       - Return null if not explicitly provided.

    7) animated_sequel_urls:
       - All URLs cited in the answer that support the casting in the animated sequel, the role description (son of a main character), and the release date.
       - Return an array of URLs. If none are provided, return an empty array.

    IMPORTANT:
    - Only extract information explicitly mentioned in the answer. Do not infer or invent.
    - For URLs, only return valid URLs explicitly present in the answer (including markdown links). If a URL is missing protocol, prepend http://.
    - If multiple films are mentioned, select the one that fits the constraints (Superman released in 2025; animated sequel scheduled for 2027).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(val: Optional[str]) -> str:
    return val.strip() if isinstance(val, str) else ""


def _build_superman_claim(details: ActorFilmExtraction) -> str:
    actor = _safe_str(details.actor_full_name) or "[UNKNOWN ACTOR]"
    title = _safe_str(details.superman_2025_film_title)
    if title:
        return f"{actor} played 'Jimmy Olsen' in the Superman film '{title}', which was released in 2025."
    else:
        return f"{actor} played 'Jimmy Olsen' in a Superman film that was released in 2025."


def _build_film_identification_claim(details: ActorFilmExtraction) -> str:
    actor = _safe_str(details.actor_full_name) or "[UNKNOWN ACTOR]"
    film = _safe_str(details.animated_sequel_title)
    return f"{actor} is cast in the animated film sequel titled '{film}', which is scheduled for release in 2027."


def _build_character_role_claim(details: ActorFilmExtraction) -> str:
    actor = _safe_str(details.actor_full_name) or "[UNKNOWN ACTOR]"
    film = _safe_str(details.animated_sequel_title)
    return f"In '{film}', {actor} voices a character who is one of the main characters' sons."


def _build_release_date_claim(details: ActorFilmExtraction) -> str:
    film = _safe_str(details.animated_sequel_title)
    date = _safe_str(details.animated_sequel_release_date)
    return f"The exact theatrical release date of '{film}' is {date}."


# --------------------------------------------------------------------------- #
# Verification sub-tree builders                                              #
# --------------------------------------------------------------------------- #
async def build_actor_identification(
    evaluator: Evaluator,
    parent_node,
    details: ActorFilmExtraction,
) -> None:
    """
    Build and verify the 'actor_identification' parallel node:
    - actor_name_provided (custom existence check)
    - jimmy_olsen_role_verification (evidence-backed verification)
    """
    actor_node = evaluator.add_parallel(
        id="actor_identification",
        desc="Correctly identify the actor who played Jimmy Olsen in a Superman film released in 2025",
        parent=parent_node,
        critical=True,
    )

    # Leaf: The answer provides the actor's full name (existence check)
    name_provided = bool(_safe_str(details.actor_full_name))
    evaluator.add_custom_node(
        result=name_provided,
        id="actor_name_provided",
        desc="The answer provides the actor's full name",
        parent=actor_node,
        critical=True,
    )

    # Leaf: The identified actor played Jimmy Olsen in a 2025 Superman film (verify by URLs if provided)
    olsen_leaf = evaluator.add_leaf(
        id="jimmy_olsen_role_verification",
        desc="The identified actor played Jimmy Olsen in a Superman film released in 2025",
        parent=actor_node,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_superman_claim(details),
        node=olsen_leaf,
        sources=details.superman_film_urls,
        additional_instruction=(
            "Use the cited URLs to confirm that the named actor played 'Jimmy Olsen' in a Superman film released in 2025. "
            "Accept variants of the film title (e.g., 'Superman (2025)', 'DCU Superman') as long as the release year is 2025 and the role is Jimmy Olsen."
        ),
    )


async def build_animated_film_details(
    evaluator: Evaluator,
    parent_node,
    details: ActorFilmExtraction,
) -> None:
    """
    Build and verify the 'animated_film_details' parallel node:
    - film_identification
    - character_role
    - release_date
    """
    film_node = evaluator.add_parallel(
        id="animated_film_details",
        desc="Correctly identify the animated film sequel the actor is cast in and provide the requested role and release details",
        parent=parent_node,
        critical=True,
    )

    # Leaf: The answer identifies the animated film sequel the actor is cast in
    film_id_leaf = evaluator.add_leaf(
        id="film_identification",
        desc="The answer identifies the animated film sequel the actor is cast in",
        parent=film_node,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_film_identification_claim(details),
        node=film_id_leaf,
        sources=details.animated_sequel_urls,
        additional_instruction=(
            "Confirm that the actor is in the cast for the specified animated sequel and that the film is a sequel scheduled for release in 2027. "
            "If the film title is missing or sources do not confirm the casting, judge as not supported."
        ),
    )

    # Leaf: The answer specifies the actor voices one of the main characters' sons
    role_leaf = evaluator.add_leaf(
        id="character_role",
        desc="The answer specifies the actor voices one of the main characters' sons in that animated film",
        parent=film_node,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_character_role_claim(details),
        node=role_leaf,
        sources=details.animated_sequel_urls,
        additional_instruction=(
            "Verify that the role is explicitly a 'son' of a main character in the franchise (e.g., 'the son of the lead protagonist'). "
            "If sources only say 'child' without confirming 'son' or male context, do not accept."
        ),
    )

    # Leaf: The answer provides the theatrical release date of the animated sequel
    release_leaf = evaluator.add_leaf(
        id="release_date",
        desc="The answer provides the theatrical release date of the animated sequel",
        parent=film_node,
        critical=True,
    )
    await evaluator.verify(
        claim=_build_release_date_claim(details),
        node=release_leaf,
        sources=details.animated_sequel_urls,
        additional_instruction=(
            "Confirm the exact theatrical release date (Month Day, Year) for the animated sequel from reliable sources "
            "(e.g., studio press releases, major trade publications). If only a year is provided or sources contradict, judge as not supported."
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
    Evaluate an answer for the Jimmy Olsen/animated sequel task.
    """
    # Initialize evaluator with sequential aggregation at the top level
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

    # Create a critical sequential node to mirror rubric root (since evaluator root is non-critical by design)
    main_seq = evaluator.add_sequential(
        id="task_main",
        desc="Identify the actor who played Jimmy Olsen in a 2025 Superman film and provide details about the 2027 animated sequel role and release date",
        parent=root,
        critical=True,
    )

    # Extract structured information from the answer
    extracted_details = await evaluator.extract(
        prompt=prompt_extract_actor_and_film_details(),
        template_class=ActorFilmExtraction,
        extraction_name="actor_and_film_details",
    )

    # Build and verify sub-trees according to the rubric
    await build_actor_identification(evaluator, main_seq, extracted_details)
    await build_animated_film_details(evaluator, main_seq, extracted_details)

    # Return standardized summary with the verification tree and scores
    return evaluator.get_summary()