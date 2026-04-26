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
TASK_ID = "cannes_oscars_2024_film_verification"
TASK_DESCRIPTION = (
    "Identify the 2024 film that won the Palme d'Or at the 77th Cannes Film Festival in May 2024, "
    "subsequently won Best Picture, Best Director, and Best Original Screenplay at the 97th Academy Awards, "
    "was produced with a budget of $10 million or less, and grossed at least $50 million worldwide. "
    "Provide the film's title and the director's full name."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmAnswerExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer.

    - film_title: The film’s title as stated in the answer.
    - director_name: The director’s full name as stated in the answer.
    - source_urls: All URLs explicitly cited in the answer (including markdown link targets).
    """
    film_title: Optional[str] = None
    director_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_film_info() -> str:
    return """
    Extract the core information about the identified film from the answer.

    Required fields:
    1) film_title: The film's title exactly as presented in the answer text.
    2) director_name: The director's full name exactly as presented in the answer text.
    3) source_urls: An array of all URLs explicitly cited in the answer that could support any of the claims
       (e.g., official Cannes page, Oscars page, Wikipedia, Box Office Mojo, The Numbers, Variety/Hollywood Reporter articles, etc.).
       Include plain URLs and targets of markdown links. Only include valid URLs mentioned in the answer.
       If there are no URLs provided, return an empty list.

    Notes:
    - Do not invent or infer URLs; extract only those explicitly present in the answer.
    - If any required field is not present, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def verify_film_criteria(
    evaluator: Evaluator,
    parent_node,
    info: FilmAnswerExtraction,
) -> None:
    """
    Build the verification tree under the 'film_criteria_verification' node
    and run all leaf verifications according to the rubric.
    """
    # Create the critical parallel node aggregating all criteria
    film_node = evaluator.add_parallel(
        id="film_criteria_verification",
        desc="Verify that the identified film satisfies all specified criteria from 2024 and that both the film title and director name are provided",
        parent=parent_node,
        critical=True,
    )

    # Prepare sources (None if empty, allowing the verifier to route to simple verification)
    sources: Optional[List[str]] = info.source_urls if info.source_urls else None
    film_title = _norm(info.film_title)
    director_name = _norm(info.director_name)

    # 1) Film title identification (presence/explicit identification in the answer)
    title_node = evaluator.add_leaf(
        id="film_title_identification",
        desc="The film's title must be correctly identified and provided",
        parent=film_node,
        critical=True,
    )
    title_claim = f"The answer identifies a specific film titled '{film_title}' (non-empty, explicit)."
    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        sources=None,  # Check presence in the answer text itself
        additional_instruction=(
            "Verify within the provided answer text that the film title is explicitly given and non-empty. "
            "If the title is missing, null, placeholder, or ambiguous, mark this as incorrect."
        ),
    )

    # 2) Director identification (verify director for the film via sources)
    director_node = evaluator.add_leaf(
        id="director_identification",
        desc="The director's full name must be correctly identified and provided",
        parent=film_node,
        critical=True,
    )
    director_claim = f"The director of the film '{film_title}' is '{director_name}'."
    await evaluator.verify(
        claim=director_claim,
        node=director_node,
        sources=sources,
        additional_instruction=(
            "Use the provided URLs to confirm the director of the specified film. "
            "Allow minor variations (diacritics, middle names/initials, casing). "
            "If sources do not explicitly support this, mark as not supported."
        ),
    )

    # 3) Release year: 2024 theatrical release
    release_node = evaluator.add_leaf(
        id="release_year_2024",
        desc="The film must have been released theatrically in 2024",
        parent=film_node,
        critical=True,
    )
    release_claim = f"The film '{film_title}' had a theatrical release (in at least one territory) in 2024."
    await evaluator.verify(
        claim=release_claim,
        node=release_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the film had a theatrical release (not just a festival premiere) in calendar year 2024 "
            "in at least one territory. If the page lists 'Release date: 2024' or similar for theatrical release, "
            "mark supported. If unclear or only festival premiere is mentioned without theatrical release, mark not supported."
        ),
    )

    # 4) Palme d'Or winner at the 77th Cannes Film Festival (May 2024)
    cannes_node = evaluator.add_leaf(
        id="palme_dor_winner",
        desc="The film must have won the Palme d'Or at the 77th Cannes Film Festival in May 2024",
        parent=film_node,
        critical=True,
    )
    cannes_claim = f"The film '{film_title}' won the Palme d'Or at the 77th Cannes Film Festival (May 2024)."
    await evaluator.verify(
        claim=cannes_claim,
        node=cannes_node,
        sources=sources,
        additional_instruction=(
            "Prefer official festival-cannes.com pages or reliable listings (e.g., Wikipedia/major outlets) that explicitly "
            "state the film won the Palme d'Or at the 77th Cannes Film Festival in May 2024."
        ),
    )

    # 5) Best Picture at the 97th Academy Awards
    bp_node = evaluator.add_leaf(
        id="best_picture_winner",
        desc="The film must have won Best Picture at the 97th Academy Awards",
        parent=film_node,
        critical=True,
    )
    bp_claim = f"At the 97th Academy Awards, Best Picture was awarded to '{film_title}'."
    await evaluator.verify(
        claim=bp_claim,
        node=bp_node,
        sources=sources,
        additional_instruction=(
            "Use reliable sources (e.g., oscars.org or major outlets) to confirm that the 97th Academy Awards "
            "Best Picture winner was the specified film."
        ),
    )

    # 6) Best Director at the 97th Academy Awards
    bd_node = evaluator.add_leaf(
        id="best_director_winner",
        desc="The film must have won Best Director at the 97th Academy Awards",
        parent=film_node,
        critical=True,
    )
    bd_claim = f"At the 97th Academy Awards, Best Director was awarded to {director_name} for '{film_title}'."
    await evaluator.verify(
        claim=bd_claim,
        node=bd_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the named director won Best Director at the 97th Academy Awards specifically for the identified film."
        ),
    )

    # 7) Best Original Screenplay at the 97th Academy Awards
    bos_node = evaluator.add_leaf(
        id="best_screenplay_winner",
        desc="The film must have won Best Original Screenplay at the 97th Academy Awards",
        parent=film_node,
        critical=True,
    )
    bos_claim = f"At the 97th Academy Awards, Best Original Screenplay was awarded to '{film_title}' (i.e., it won Original Screenplay)."
    await evaluator.verify(
        claim=bos_claim,
        node=bos_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the film won Best Original Screenplay (not Adapted) at the 97th Academy Awards."
        ),
    )

    # 8) Budget constraint: $10 million or less
    budget_node = evaluator.add_leaf(
        id="budget_constraint",
        desc="The film's production budget must be $10 million or less",
        parent=film_node,
        critical=True,
    )
    budget_claim = f"The production budget of '{film_title}' was at most $10,000,000 (USD or equivalent)."
    await evaluator.verify(
        claim=budget_claim,
        node=budget_node,
        sources=sources,
        additional_instruction=(
            "Check credible sources (trade publications, Wikipedia with citations, studio/press materials) for the production budget. "
            "Allow phrasing like '$X million' or approximate ranges. If conflicting reports exist or the budget is clearly above $10M, "
            "mark not supported."
        ),
    )

    # 9) Worldwide box office threshold: at least $50 million
    box_node = evaluator.add_leaf(
        id="box_office_threshold",
        desc="The film's worldwide box office gross must be at least $50 million",
        parent=film_node,
        critical=True,
    )
    box_claim = f"The worldwide box office gross of '{film_title}' was at least $50,000,000."
    await evaluator.verify(
        claim=box_claim,
        node=box_node,
        sources=sources,
        additional_instruction=(
            "Prefer boxofficemojo.com or The Numbers pages, or reliable sources summarizing worldwide grosses. "
            "Confirm that global cumulative gross meets or exceeds $50M."
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
    Evaluate an agent's answer for the Cannes + Oscars + budget + box office 2024 film criteria task.
    Returns a structured summary with the verification tree and final score.
    """
    # 1) Initialize evaluator
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

    # 2) Extract film information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_film_info(),
        template_class=FilmAnswerExtraction,
        extraction_name="film_answer_extraction",
    )

    # Optional: add a concise record of criteria for readability in summary
    evaluator.add_ground_truth({
        "required_criteria": [
            "Film title provided",
            "Director's full name provided",
            "Theatrical release in 2024",
            "Won Palme d'Or (77th Cannes, May 2024)",
            "Won Best Picture (97th Academy Awards)",
            "Won Best Director (97th Academy Awards)",
            "Won Best Original Screenplay (97th Academy Awards)",
            "Budget ≤ $10 million",
            "Worldwide gross ≥ $50 million",
        ]
    }, gt_type="criteria")

    # 3) Build verification nodes and run checks
    await verify_film_criteria(evaluator, root, extraction)

    # 4) Return summary
    return evaluator.get_summary()