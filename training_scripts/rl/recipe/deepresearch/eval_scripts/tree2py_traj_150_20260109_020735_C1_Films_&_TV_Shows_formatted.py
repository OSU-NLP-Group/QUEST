import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "almodovar_cinematographer_venice2024"
TASK_DESCRIPTION = (
    "What is the name of the cinematographer who shot Pedro Almodóvar's film that won the Golden Lion at the 2024 Venice International Film Festival?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CineAnswerExtraction(BaseModel):
    """
    Structured extraction from the agent's answer:
    - film_title: The film title mentioned by the agent (if any)
    - cinematographer: The cinematographer name(s) stated by the agent
    - sources: Any URLs the agent cites to support the film identification or the cinematographer credit
    """
    film_title: Optional[str] = None
    cinematographer: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cine_fields() -> str:
    return """
    You must extract structured information from the answer for the following question:
    "What is the name of the cinematographer who shot Pedro Almodóvar's film that won the Golden Lion at the 2024 Venice International Film Festival?"

    Extract the following fields from the answer text exactly as written:
    1) film_title: The title of the Pedro Almodóvar film referenced (if explicitly named in the answer). If not provided, return null.
    2) cinematographer: The cinematographer's name (or names) that the answer claims shot that film. If multiple names are provided, include them all in a single string as they appear (e.g., separated by commas or 'and'). If not provided, return null.
    3) sources: A list of all URLs explicitly cited in the answer that could support either:
       - that the referenced film is Pedro Almodóvar’s and won the Golden Lion at the 2024 Venice International Film Festival, or
       - the cinematographer (director of photography) credit for that film.
       Include URLs presented in plain form or as markdown links. Do not fabricate URLs.
       If no URLs are present, return an empty list.

    Return strictly these fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions for claim construction                                     #
# --------------------------------------------------------------------------- #
def build_film_reference_claim(extracted: CineAnswerExtraction) -> str:
    """
    Construct a claim for verifying that the answer unambiguously refers
    to Pedro Almodóvar’s film that won the Golden Lion at the 2024 Venice International Film Festival.
    """
    # Do not force a film title; presence of a title is optional per rubric.
    if extracted.film_title and extracted.film_title.strip():
        return (
            f"The response unambiguously refers to the Pedro Almodóvar film that won the Golden Lion "
            f"at the 2024 Venice International Film Festival; the film title mentioned is "
            f"'{extracted.film_title}'. The response does not contradict these constraints."
        )
    else:
        return (
            "The response unambiguously refers to the Pedro Almodóvar film that won the Golden Lion "
            "at the 2024 Venice International Film Festival and does not contradict these constraints, "
            "even if it does not explicitly name the film title."
        )


def build_cinematographer_claim(extracted: CineAnswerExtraction) -> str:
    """
    Construct a claim for verifying that the stated cinematographer matches authoritative sources
    for that specific film's cinematography credit.
    """
    film_part = (
        f"('{extracted.film_title}')" if extracted.film_title and extracted.film_title.strip() else
        "(the Pedro Almodóvar film that won the Golden Lion at the 2024 Venice International Film Festival)"
    )
    cinematographer_name = extracted.cinematographer if extracted.cinematographer else "UNKNOWN"
    return (
        f"The cinematographer (director of photography) of Pedro Almodóvar’s film {film_part} is "
        f"'{cinematographer_name}'."
    )


# --------------------------------------------------------------------------- #
# Verification routine that builds the rubric tree                            #
# --------------------------------------------------------------------------- #
async def _build_and_verify(evaluator: Evaluator, extracted: CineAnswerExtraction) -> None:
    """
    Build the verification tree according to the rubric and run the checks.
    """
    # Create the main sequential node (critical) under the root to model the rubric.
    main_node = evaluator.add_sequential(
        id="answer_about_cinematographer",
        desc="Evaluate whether the response correctly identifies the cinematographer of Pedro Almodóvar’s Golden Lion-winning film at Venice 2024.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Identify the correct referenced film (critical leaf)
    film_id_node = evaluator.add_leaf(
        id="identify_correct_film_venice2024",
        desc="The response unambiguously refers to Pedro Almodóvar’s film that won the Golden Lion at the 2024 Venice International Film Festival (may name the film; must not contradict).",
        parent=main_node,
        critical=True
    )

    film_claim = build_film_reference_claim(extracted)
    # This is primarily about the answer's internal reference consistency, so use simple verification (no sources).
    await evaluator.verify(
        claim=film_claim,
        node=film_id_node,
        sources=None,
        additional_instruction=(
            "Judge only whether the answer itself refers to the correct film described by the question "
            "(Almodóvar + Golden Lion + Venice 2024) without contradiction. The film title is optional. "
            "If the response clearly indicates the described film (even implicitly) and does not point to a different film or event/year, mark Correct. "
            "If it's ambiguous or contradicts (e.g., wrong year, festival, or director), mark Incorrect."
        )
    )

    # 2) Provide the correct cinematographer name for that film (critical leaf)
    cine_leaf = evaluator.add_leaf(
        id="provide_correct_cinematographer",
        desc="The response states the correct cinematographer for that specific film, matching authoritative sources.",
        parent=main_node,
        critical=True
    )

    cine_claim = build_cinematographer_claim(extracted)
    sources_to_use: Optional[List[str]] = extracted.sources if extracted.sources else None
    await evaluator.verify(
        claim=cine_claim,
        node=cine_leaf,
        sources=sources_to_use,
        additional_instruction=(
            "Verify using the provided URLs (if any) whether the named person is credited as the film’s cinematographer "
            "(a.k.a. director of photography). Accept reasonable variants (e.g., accents, middle names, short forms). "
            "Look for credit sections such as 'Cinematography' or 'Director of Photography'. "
            "If multiple names are stated for cinematography in the answer, treat it as correct if at least one matches the authoritative credit. "
            "If no URLs are provided, judge based on the answer text and task context; however, if the claim is unsupported or contradicts known credits on the provided page(s), mark Incorrect."
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
    Entrypoint for evaluating an answer to the cinematographer question.
    Returns the structured evaluation summary produced by obj_task_eval.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Enforce ordered checks
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

    # Extract structured fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_cine_fields(),
        template_class=CineAnswerExtraction,
        extraction_name="cine_answer_extraction"
    )

    # Build the tree and perform verifications
    await _build_and_verify(evaluator, extracted)

    # Return the final summary including the verification tree and scores
    return evaluator.get_summary()