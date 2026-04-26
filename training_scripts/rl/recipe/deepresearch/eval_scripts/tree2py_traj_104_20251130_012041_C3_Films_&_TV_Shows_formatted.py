import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "st_s5_director_king_adaptation"
TASK_DESCRIPTION = (
    "For Stranger Things Season 5, identify the director who came out of retirement to direct episodes appearing in both "
    "Volume 1 and Volume 2. Then, research this director's most acclaimed film adaptation of a Stephen King work from the 1990s. "
    "Provide: (1) the film's release year, (2) the names of the two lead actors, and (3) the character names each actor portrayed."
)

# Ground truth expectations (used for claim construction and transparency)
EXPECTED_DIRECTOR = "Frank Darabont"
EXPECTED_FILM_TITLE = "The Shawshank Redemption"
EXPECTED_RELEASE_YEAR = "1994"
EXPECTED_LEAD_1_ACTOR = "Tim Robbins"
EXPECTED_LEAD_1_CHARACTER = "Andy Dufresne"
EXPECTED_LEAD_2_ACTOR = "Morgan Freeman"
EXPECTED_LEAD_2_CHARACTER = "Ellis Boyd 'Red' Redding"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ActorCharacter(BaseModel):
    actor: Optional[str] = None
    character: Optional[str] = None


class AnswerExtraction(BaseModel):
    # Director info
    director_name: Optional[str] = None
    director_sources: List[str] = Field(default_factory=list)

    # Film identification
    film_title: Optional[str] = None
    film_sources: List[str] = Field(default_factory=list)

    # Film details
    release_year: Optional[str] = None
    leads: List[ActorCharacter] = Field(default_factory=list)

    # Any additional links the answer cites for details
    details_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the following information strictly from the provided answer text.

    1) director_name:
       - The name of the director identified as having come out of retirement to direct episodes appearing in BOTH Volume 1 and Volume 2 of Stranger Things Season 5.
       - If not explicitly stated in the answer, return null.

    2) director_sources:
       - All URLs explicitly cited in the answer that support the identification of this director or their involvement with Stranger Things Season 5.
       - Return an array of valid URLs. If none are provided, return an empty array.

    3) film_title:
       - The film that the answer claims is this director's most acclaimed adaptation of a Stephen King work from the 1990s.
       - If not explicitly stated in the answer, return null.

    4) film_sources:
       - All URLs explicitly cited in the answer that pertain to the identified film (e.g., Wikipedia, IMDb, studio page).
       - Return an array of valid URLs. If none provided, return an empty array.

    5) release_year:
       - The single release year of the identified film as written in the answer (prefer a 4-digit year like "1994").
       - If multiple dates are present, extract the primary theatrical release year. If not present, return null.

    6) leads:
       - Extract exactly two lead actor-character pairs for the identified film as given in the answer.
       - Each element is an object with "actor" and "character".
       - If the answer lists more than two, choose the two most prominently mentioned as leads. If fewer are present, include as many as available (the rest can be omitted).

    7) details_sources:
       - Any additional URLs explicitly cited in the answer that support the film's release year and/or the lead actor-character details.
       - Return an array of valid URLs. If none provided, return an empty array.

    Notes:
    - Only extract information explicitly present in the answer. Do not invent or infer missing values.
    - For URLs, accept plain links or markdown links. Return fully qualified URLs including protocol.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_and_verify_director_section(evaluator: Evaluator, parent_node, ex: AnswerExtraction) -> None:
    """
    Build the 'DirectorIdentification' subtree and run verifications.
    Critical node: If fails, subsequent steps are skipped due to root sequential gating.
    """
    director_node = evaluator.add_parallel(
        id="DirectorIdentification",
        desc="Identify the director who came out of retirement to direct episodes appearing in both Volume 1 and Volume 2 of Stranger Things Season 5 (Frank Darabont)",
        parent=parent_node,
        critical=True,
    )

    # Existence check: director name provided
    director_exists = evaluator.add_custom_node(
        result=(ex.director_name is not None and str(ex.director_name).strip() != ""),
        id="director_exists",
        desc="Director name is provided in the answer",
        parent=director_node,
        critical=True
    )

    # Match check: the answer's identified director is Frank Darabont
    director_match = evaluator.add_leaf(
        id="director_match",
        desc="The identified director is Frank Darabont",
        parent=director_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The identified director is {EXPECTED_DIRECTOR}.",
        node=director_match,
        sources=_dedupe_urls(ex.director_sources) if ex.director_sources else None,
        additional_instruction="Judge based on the answer and provided sources. Minor name formatting differences should be treated as the same person."
    )


async def build_and_verify_film_identification(evaluator: Evaluator, parent_node, ex: AnswerExtraction) -> None:
    """
    Build the 'FilmIdentification' subtree and run verifications.
    Critical node: If fails, subsequent steps are skipped due to root sequential gating.
    """
    film_node = evaluator.add_parallel(
        id="FilmIdentification",
        desc="Identify the director's most acclaimed film adaptation of a Stephen King work from the 1990s (The Shawshank Redemption)",
        parent=parent_node,
        critical=True,
    )

    # Existence check: film title provided
    film_exists = evaluator.add_custom_node(
        result=(ex.film_title is not None and str(ex.film_title).strip() != ""),
        id="film_title_provided",
        desc="Film title is provided in the answer",
        parent=film_node,
        critical=True
    )

    # Match check: film matches "The Shawshank Redemption"
    film_match = evaluator.add_leaf(
        id="film_title_match",
        desc="The identified film is The Shawshank Redemption",
        parent=film_node,
        critical=True
    )

    film_all_sources = _dedupe_urls((ex.film_sources or []) + (ex.details_sources or []))
    await evaluator.verify(
        claim=f"The director's most acclaimed 1990s Stephen King film adaptation is '{EXPECTED_FILM_TITLE}'.",
        node=film_match,
        sources=film_all_sources if film_all_sources else None,
        additional_instruction="Allow reasonable short forms like 'Shawshank Redemption' vs 'The Shawshank Redemption' as equivalent."
    )


async def build_and_verify_film_details(evaluator: Evaluator, parent_node, ex: AnswerExtraction) -> None:
    """
    Build the 'FilmDetails' subtree and run verifications (parallel).
    All leaves are critical to satisfy the rubric's requirements.
    """
    details_node = evaluator.add_parallel(
        id="FilmDetails",
        desc="Provide the film's release year and the two lead actors with their character names",
        parent=parent_node,
        critical=True
    )

    film_all_sources = _dedupe_urls((ex.film_sources or []) + (ex.details_sources or []))

    # Release year check (expects 1994)
    release_year_node = evaluator.add_leaf(
        id="release_year_correct",
        desc="Provide the correct release year of the film (1994)",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The film '{EXPECTED_FILM_TITLE}' was released in {EXPECTED_RELEASE_YEAR}.",
        node=release_year_node,
        sources=film_all_sources if film_all_sources else None,
        additional_instruction="If multiple dates exist, judge by the primary theatrical release year. Minor regional variations are acceptable as long as 1994 is correct."
    )

    # Lead actor 1 with character
    lead1_node = evaluator.add_leaf(
        id="lead_actor_1_with_character",
        desc="Provide one lead actor and the character portrayed (Tim Robbins as Andy Dufresne)",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{EXPECTED_LEAD_1_ACTOR} portrays {EXPECTED_LEAD_1_CHARACTER} in '{EXPECTED_FILM_TITLE}'.",
        node=lead1_node,
        sources=film_all_sources if film_all_sources else None,
        additional_instruction="Allow minor variations in the character name (e.g., 'Andrew' vs 'Andy') and punctuation."
    )

    # Lead actor 2 with character
    lead2_node = evaluator.add_leaf(
        id="lead_actor_2_with_character",
        desc="Provide the other lead actor and the character portrayed (Morgan Freeman as Ellis Boyd 'Red' Redding)",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{EXPECTED_LEAD_2_ACTOR} portrays {EXPECTED_LEAD_2_CHARACTER} in '{EXPECTED_FILM_TITLE}'.",
        node=lead2_node,
        sources=film_all_sources if film_all_sources else None,
        additional_instruction="Treat 'Red' as a nickname for Ellis Boyd Redding and allow minor punctuation/casing variations."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Stranger Things S5 director and Stephen King adaptation task.
    Returns a structured evaluation summary suitable for downstream consumption.
    """
    # Initialize evaluator with a sequential root to enforce step-wise dependency
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
        default_model=model
    )

    # Extract all required info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction"
    )

    # Record ground truth for transparency
    evaluator.add_ground_truth({
        "expected_director": EXPECTED_DIRECTOR,
        "expected_film_title": EXPECTED_FILM_TITLE,
        "expected_release_year": EXPECTED_RELEASE_YEAR,
        "expected_leads": [
            {"actor": EXPECTED_LEAD_1_ACTOR, "character": EXPECTED_LEAD_1_CHARACTER},
            {"actor": EXPECTED_LEAD_2_ACTOR, "character": EXPECTED_LEAD_2_CHARACTER},
        ]
    }, gt_type="ground_truth")

    # Build and verify subtrees in order
    await build_and_verify_director_section(evaluator, root, extraction)
    await build_and_verify_film_identification(evaluator, root, extraction)
    await build_and_verify_film_details(evaluator, root, extraction)

    # Return the structured summary
    return evaluator.get_summary()