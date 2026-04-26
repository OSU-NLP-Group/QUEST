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
TASK_ID = "sundance_2024_film_info"
TASK_DESCRIPTION = (
    "Identify the film that won the U.S. Dramatic Grand Jury Prize at the 2024 Sundance Film Festival and was also "
    "the director's feature directorial debut. Provide the following information: (1) The title of the film, "
    "(2) Confirmation that it won the U.S. Dramatic Grand Jury Prize at Sundance 2024, "
    "(3) Confirmation that it was the director's feature directorial debut, "
    "(4) The cinematographer's full name, "
    "(5) The exact U.S. theatrical release date (month, day, and year), and "
    "(6) A reference URL from an official or reputable source that confirms the film's award and debut status."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmIdentity(BaseModel):
    film_title: Optional[str] = None
    # URLs the answer cites to support BOTH the award and debut claims (can include multiple if provided)
    award_debut_reference_urls: List[str] = Field(default_factory=list)


class FilmDetails(BaseModel):
    cinematographer_name: Optional[str] = None
    cinematographer_reference_urls: List[str] = Field(default_factory=list)

    us_theatrical_release_date: Optional[str] = None
    release_date_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_film_identity() -> str:
    return """
    Extract the identity and eligibility information for the film identified in the answer.

    You must extract:
    - film_title: The film title as written in the answer.
    - award_debut_reference_urls: A list of URL(s) that the answer presents as evidence confirming both of the following:
        (a) the film won the U.S. Dramatic Grand Jury Prize at the 2024 Sundance Film Festival, and
        (b) it was the director’s feature directorial debut.
      If the answer provides separate URLs for the award and the debut confirmation, include all relevant URLs here.

    Rules:
    - Extract only URLs explicitly present in the answer (plain URLs or markdown links).
    - If no such URL is provided, return an empty list.
    - Do not invent or infer any URLs.
    """


def prompt_extract_film_details(context_title: Optional[str]) -> str:
    title_hint = context_title or "the film"
    return f"""
    Extract production and release details for {title_hint} as given in the answer.

    You must extract:
    - cinematographer_name: The full name of the cinematographer (a.k.a. director of photography).
    - cinematographer_reference_urls: URL(s) the answer cites to support the cinematographer information.
    - us_theatrical_release_date: The exact U.S. theatrical release date (Month Day, Year), as stated in the answer.
    - release_date_reference_urls: URL(s) the answer cites to support the U.S. theatrical release date.

    Rules:
    - Extract only what is explicitly present in the answer text.
    - For each set of URLs, include all that are associated with the specific fact (cinematographer or release date).
    - If a field is missing in the answer, set it to null (or an empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_string(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_identity_and_eligibility(
    evaluator: Evaluator,
    parent_node,
    identity: FilmIdentity,
) -> None:
    """
    Build and verify the 'Film_Identity_and_Eligibility' subtree:
    - Film_Title (existence)
    - Award_and_Debut_Reference_URL (existence of at least one URL)
    - Sundance_Grand_Jury_Prize (claim verified by the provided URL(s))
    - Directorial_Debut (claim verified by the provided URL(s))
    """
    node = evaluator.add_parallel(
        id="Film_Identity_and_Eligibility",
        desc="Identify the film and verify it meets the award and debut criteria with an appropriate reference",
        parent=parent_node,
        critical=True,
    )

    title_exists = _non_empty_string(identity.film_title)
    urls = _valid_urls(identity.award_debut_reference_urls)

    # Film_Title (existence)
    evaluator.add_custom_node(
        result=title_exists,
        id="Film_Title",
        desc="Provide the title of the film",
        parent=node,
        critical=True,
    )

    # Award_and_Debut_Reference_URL (existence of URL(s))
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Award_and_Debut_Reference_URL",
        desc="Provide a reference URL from an official or reputable source confirming the film's award and debut status",
        parent=node,
        critical=True,
    )

    # Sundance_Grand_Jury_Prize (verification by URLs)
    prize_leaf = evaluator.add_leaf(
        id="Sundance_Grand_Jury_Prize",
        desc="Confirm the film won the U.S. Dramatic Grand Jury Prize at the 2024 Sundance Film Festival",
        parent=node,
        critical=True,
    )

    film_title_for_claim = identity.film_title or ""
    prize_claim = (
        f"The film '{film_title_for_claim}' won the U.S. Dramatic Grand Jury Prize at the 2024 Sundance Film Festival."
    )
    await evaluator.verify(
        claim=prize_claim,
        node=prize_leaf,
        sources=urls,  # must be from reputable/official sources; otherwise, the verification should fail
        additional_instruction=(
            "Use only the provided URL(s). Confirm explicitly that the page states the film won the "
            "U.S. Dramatic Grand Jury Prize at the 2024 Sundance Film Festival. Prefer official/reputable sources "
            "such as sundance.org (official festival), distributor/production official pages, or established trades "
            "like Variety, THR, Deadline, IndieWire, or major newspapers. If the URL(s) are missing, irrelevant, "
            "or do not explicitly support this claim, mark as not supported."
        ),
    )

    # Directorial_Debut (verification by URLs)
    debut_leaf = evaluator.add_leaf(
        id="Directorial_Debut",
        desc="Confirm the film was the director's feature directorial debut",
        parent=node,
        critical=True,
    )

    debut_claim = (
        f"The film '{film_title_for_claim}' was the director's feature directorial debut (the director's first feature film)."
    )
    await evaluator.verify(
        claim=debut_claim,
        node=debut_leaf,
        sources=urls,
        additional_instruction=(
            "Use only the provided URL(s). Confirm explicitly that the page states the film marks the director’s feature "
            "directorial debut or first feature. Prefer official/reputable sources as described. If the URL(s) are missing, "
            "irrelevant, or do not explicitly support this claim, mark as not supported."
        ),
    )


async def build_details_and_references(
    evaluator: Evaluator,
    parent_node,
    identity: FilmIdentity,
    details: FilmDetails,
) -> None:
    """
    Build and verify the 'Film_Details_and_References' subtree:
    - Cinematographer (existence)
    - Cinematographer_Reference_URL (verify cinematographer via URL(s))
    - US_Theatrical_Release_Date (existence)
    - Release_Date_Reference_URL (verify release date via URL(s))
    """
    node = evaluator.add_parallel(
        id="Film_Details_and_References",
        desc="Provide requested production/release details and verifiable references for them",
        parent=parent_node,
        critical=True,
    )

    film_title_for_claim = identity.film_title or "the film"

    # Cinematographer name provided
    has_cinematographer = _non_empty_string(details.cinematographer_name)
    evaluator.add_custom_node(
        result=has_cinematographer,
        id="Cinematographer",
        desc="Provide the cinematographer's full name",
        parent=node,
        critical=True,
    )

    # Cinematographer verification by URL(s)
    cinematographer_urls = _valid_urls(details.cinematographer_reference_urls)
    cinematographer_leaf = evaluator.add_leaf(
        id="Cinematographer_Reference_URL",
        desc="Provide a verifiable reference URL supporting the cinematographer information",
        parent=node,
        critical=True,
    )
    cinematographer_claim = (
        f"The cinematographer (director of photography) of '{film_title_for_claim}' is {details.cinematographer_name}."
    )
    await evaluator.verify(
        claim=cinematographer_claim,
        node=cinematographer_leaf,
        sources=cinematographer_urls if cinematographer_urls else None,
        additional_instruction=(
            "Verify this claim using the provided URL(s). The page should explicitly credit the named individual as "
            "cinematographer or 'director of photography'. If multiple cinematographers are listed on the page, the "
            "provided name must be included among them. If no valid URL is provided, mark as not supported."
        ),
    )

    # U.S. theatrical release date provided
    has_release_date = _non_empty_string(details.us_theatrical_release_date)
    evaluator.add_custom_node(
        result=has_release_date,
        id="US_Theatrical_Release_Date",
        desc="Provide the exact U.S. theatrical release date (month, day, and year)",
        parent=node,
        critical=True,
    )

    # Release date verification by URL(s)
    release_urls = _valid_urls(details.release_date_reference_urls)
    release_leaf = evaluator.add_leaf(
        id="Release_Date_Reference_URL",
        desc="Provide a verifiable reference URL supporting the U.S. theatrical release date",
        parent=node,
        critical=True,
    )
    release_claim = (
        f"The U.S. theatrical release date of '{film_title_for_claim}' was {details.us_theatrical_release_date}."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=release_urls if release_urls else None,
        additional_instruction=(
            "Verify this claim using the provided URL(s). The page should explicitly list the U.S. theatrical release "
            "date matching the stated date (Month Day, Year). Do NOT confuse film festival premieres or limited event "
            "screenings with a general U.S. theatrical release. If no valid URL is provided or the page does not "
            "clearly state the U.S. theatrical release date, mark as not supported."
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
    Evaluate an answer for the Sundance 2024 film information task.
    """
    # Initialize evaluator with a neutral root; construct rubric under a critical main node
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

    # Extract identity and details from the answer
    identity = await evaluator.extract(
        prompt=prompt_extract_film_identity(),
        template_class=FilmIdentity,
        extraction_name="film_identity",
    )
    details = await evaluator.extract(
        prompt=prompt_extract_film_details(identity.film_title),
        template_class=FilmDetails,
        extraction_name="film_details",
    )

    # Build main critical sequential node as per rubric
    main_node = evaluator.add_sequential(
        id="Sundance_2024_Film_Information",
        desc="Provide complete and verified information about the film that won the U.S. Dramatic Grand Jury Prize at the 2024 Sundance Film Festival and was the director's feature directorial debut",
        parent=root,
        critical=True,
    )

    # 1) Film Identity and Eligibility (parallel, critical)
    await build_identity_and_eligibility(evaluator, main_node, identity)

    # 2) Film Details and References (parallel, critical)
    await build_details_and_references(evaluator, main_node, identity, details)

    # Return the final structured evaluation summary
    return evaluator.get_summary()