import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "broadway_debut_and_proctor_actor_2015_emmy_context"
TASK_DESCRIPTION = (
    "In 2015, a choreographer won his second Emmy Award for Outstanding Choreography for a routine titled "
    "\"Elastic Heart\" on Dancing With The Stars. Fourteen years prior to this Emmy win, an actress made her Broadway "
    "debut playing Becky Thatcher in a musical that opened and closed within the same calendar month. "
    "What was the exact opening date (formatted as Month DD, YYYY) of this Broadway musical debut? Additionally, "
    "in her second Broadway production which opened in March 2002, she appeared in a revival of a classic American drama. "
    "Provide the full name of the actor who portrayed John Proctor in this 2002 revival."
)


class DebutInfo(BaseModel):
    show_title: Optional[str] = None
    role_name: Optional[str] = None
    opening_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProctorRevivalInfo(BaseModel):
    show_title: Optional[str] = None
    actor_name: Optional[str] = None
    revival_open_month_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


def prompt_extract_debut_info() -> str:
    return (
        "From the answer, identify the Broadway musical in which the actress made her Broadway debut playing Becky "
        "Thatcher (this debut occurred in 2001, fourteen years before the 2015 Emmy mentioned). Extract:\n"
        "1) show_title: the exact title of the Broadway musical debut\n"
        "2) role_name: the role name the actress played in that production (should be Becky Thatcher)\n"
        "3) opening_date: the Broadway opening date as stated in the answer (do not invent; keep the original format)\n"
        "4) sources: all URLs cited in the answer that specifically support the show's Broadway opening date and debut context. "
        "Include Wikipedia, IBDB, Playbill, or other credible Broadway references if provided; return only URLs.\n"
        "If any field is missing in the answer, set it to null (or empty array for sources)."
    )


def prompt_extract_proctor_info() -> str:
    return (
        "From the answer, identify the actress's second Broadway production that opened in March 2002. "
        "This was a revival of a classic American drama where John Proctor is a character. Extract:\n"
        "1) show_title: the exact title of the 2002 Broadway revival (e.g., the drama's title)\n"
        "2) actor_name: the full name (first and last name) of the actor who portrayed John Proctor in this 2002 revival\n"
        "3) revival_open_month_year: the opening month and year as stated in the answer (e.g., 'March 2002')\n"
        "4) sources: all URLs cited in the answer that support the actor portrayal and the specific 2002 Broadway revival context. "
        "Include Wikipedia, IBDB, Playbill, or other credible references if provided; return only URLs.\n"
        "If any field is missing in the answer, set it to null (or empty array for sources)."
    )


async def verify_broadway_debut(
    evaluator: Evaluator,
    parent_node,
    debut: DebutInfo,
) -> None:
    node = evaluator.add_sequential(
        id="broadway_debut_opening_date",
        desc="Provide the exact opening date of the Broadway musical debut (the show where the actress played Becky Thatcher in 2001)",
        parent=parent_node,
        critical=False,
    )

    date_str = debut.opening_date or ""
    show_title = debut.show_title or ""
    role_name = debut.role_name or ""
    sources = debut.sources or []

    # 1) Date format check (Month DD, YYYY) - simple logical verification
    date_format_leaf = evaluator.add_leaf(
        id="date_format_check",
        desc="Verify the date is provided in the correct format (Month DD, YYYY)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The string '{date_str}' is formatted as 'Month DD, YYYY' (for example, 'April 26, 2001').",
        node=date_format_leaf,
        additional_instruction=(
            "Judge correctness purely by format: spelled-out month, a space, a two-digit day (allow 1 or 2 digits), a comma, a space, and a four-digit year. "
            "Minor acceptable variants: single-digit day is acceptable. If the date is missing or clearly not in that format, mark incorrect."
        ),
    )

    # 2) Date accuracy check - verify opening date and correct show context using sources (prefer URLs)
    date_accuracy_leaf = evaluator.add_leaf(
        id="date_accuracy_check",
        desc="Verify the opening date corresponds to the correct Broadway musical (the one with Becky Thatcher in 2001)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The Broadway musical '{show_title}', in which the actress played '{role_name}', opened on {date_str} on Broadway."
        ),
        node=date_accuracy_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the Broadway opening date and correct production context (the show featured Becky Thatcher and occurred in 2001). "
            "Rely on credible references in the provided URLs (e.g., IBDB, Playbill, Wikipedia). "
            "If multiple productions exist (regional, previews, etc.), ensure it is the Broadway opening date. "
            "If the provided URLs do not support this exact opening date for the Broadway production, mark incorrect."
        ),
    )

    # 3) Reference URL check - ensure at least one URL confirms the opening date
    ref_url_leaf = evaluator.add_leaf(
        id="reference_url_debut_date",
        desc="Provide at least one reference URL that confirms the opening date of the Broadway debut show",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The provided sources confirm that the Broadway opening date of '{show_title}' is {date_str}."
        ),
        node=ref_url_leaf,
        sources=sources,
        additional_instruction=(
            "Judge as incorrect if there are no URLs provided in the answer supporting this opening date. "
            "If URLs are provided, confirm that at least one clearly supports the exact opening date for the Broadway production."
        ),
    )


async def verify_proctor_actor(
    evaluator: Evaluator,
    parent_node,
    proctor: ProctorRevivalInfo,
) -> None:
    node = evaluator.add_sequential(
        id="john_proctor_actor",
        desc="Provide the full name of the actor who portrayed John Proctor in the 2002 Broadway revival",
        parent=parent_node,
        critical=False,
    )

    actor_name = proctor.actor_name or ""
    show_title = proctor.show_title or ""
    month_year = proctor.revival_open_month_year or ""
    sources = proctor.sources or []

    # 1) Full name provided (first and last)
    name_provided_node = evaluator.add_custom_node(
        result=(bool(actor_name) and (" " in actor_name.strip())),
        id="actor_name_provided",
        desc="Verify that a full name (first and last name) is provided for the actor",
        parent=node,
        critical=True,
    )

    # 2) Actor accuracy check - confirm correct 2002 Broadway revival that opened in March 2002
    actor_accuracy_leaf = evaluator.add_leaf(
        id="actor_accuracy_check",
        desc="Verify the actor name corresponds to the correct 2002 Broadway revival that opened in March 2002",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"In the 2002 Broadway revival of '{show_title}' that opened in March 2002 (the production described as '{month_year}'), "
            f"the role of John Proctor was portrayed by '{actor_name}'."
        ),
        node=actor_accuracy_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm via the provided sources that the named actor played John Proctor in the 2002 Broadway revival that opened in March 2002. "
            "Use credible pages (IBDB, Playbill, Wikipedia). Allow minor name variations (middle initials). "
            "If the sources do not support this specific revival or the actor-role association, mark incorrect."
        ),
    )

    # 3) Reference URL check - ensure at least one URL confirms the actor-role in the 2002 revival
    ref_url_actor_leaf = evaluator.add_leaf(
        id="reference_url_actor",
        desc="Provide at least one reference URL that confirms the actor's role as John Proctor in the 2002 revival",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The provided sources confirm that '{actor_name}' portrayed John Proctor in the 2002 Broadway revival of '{show_title}'."
        ),
        node=ref_url_actor_leaf,
        sources=sources,
        additional_instruction=(
            "Judge as incorrect if the answer provides no URLs. If URLs are provided, confirm that at least one explicitly shows the actor played John Proctor "
            "in the 2002 Broadway revival that opened in March 2002."
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

    debut_info, proctor_info = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_debut_info(),
            template_class=DebutInfo,
            extraction_name="broadway_debut_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_proctor_info(),
            template_class=ProctorRevivalInfo,
            extraction_name="proctor_revival_info",
        ),
    )

    await asyncio.gather(
        verify_broadway_debut(evaluator, root, debut_info),
        verify_proctor_actor(evaluator, root, proctor_info),
    )

    return evaluator.get_summary()