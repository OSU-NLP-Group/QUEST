import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pittsburgh_arena_forbes_2020s"
TASK_DESCRIPTION = (
    "Identify the college basketball arena located in Pittsburgh, Pennsylvania that meets ALL of the following criteria: "
    "(1) has a seating capacity between 3,000 and 4,000 for basketball games, "
    "(2) is named after a Naismith Memorial Basketball Hall of Fame inductee, "
    "(3) is located on Forbes Avenue, and "
    "(4) opened or underwent major renovation in the 2020s. "
    "Provide the complete name of the arena, the university it serves, its exact seating capacity, "
    "the Hall of Fame inductee after whom it is named, and a reference URL supporting your answer."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArenaExtraction(BaseModel):
    """Structured fields expected from the agent's answer."""
    arena_name: Optional[str] = None
    university_name: Optional[str] = None
    exact_capacity: Optional[str] = None
    hall_of_famer_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    opened_year: Optional[str] = None
    renovation_year: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_arena_info() -> str:
    return (
        "Extract the specific information about the arena mentioned in the answer. "
        "Return a JSON object with the following fields:\n"
        "1) arena_name: The complete name of the arena as written.\n"
        "2) university_name: The university that uses the arena for men's or women's college basketball home games.\n"
        "3) exact_capacity: The exact seating capacity for basketball games (extract exactly as provided; if a range is given, return it verbatim).\n"
        "4) hall_of_famer_name: The name of the Naismith Memorial Basketball Hall of Fame inductee for whom the arena is named.\n"
        "5) reference_urls: An array of all URLs explicitly cited in the answer that are relevant to supporting the claims about the arena (only actual URLs; include full protocols; ignore non-URL mentions).\n"
        "6) opened_year: The year the arena opened, if the answer provides it (otherwise null).\n"
        "7) renovation_year: The year the arena underwent major renovation, if provided (otherwise null).\n"
        "Notes:\n"
        "- Extract only what is explicitly stated in the answer; do not invent or infer new details.\n"
        "- For reference_urls, include every URL that appears associated with this arena in the answer. "
        "URLs can be plain or in markdown [text](url) format; always extract the actual URL string.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_sources(urls: List[str]) -> List[str]:
    """Return non-empty, stripped URLs."""
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_tree_and_verify(
    evaluator: Evaluator,
    root: VerificationNode,
    extracted: ArenaExtraction
) -> None:
    """
    Build the verification tree according to rubric and run verifications.
    """

    # Parent node: The identified arena meets all specified criteria (critical, parallel)
    criteria_node = evaluator.add_parallel(
        id="arena_criteria_compliance",
        desc="The identified arena meets all specified criteria",
        parent=root,
        critical=True
    )

    # Parent node: The response includes all required fields (critical, parallel)
    fields_node = evaluator.add_parallel(
        id="required_response_fields",
        desc="The response includes all required fields",
        parent=root,
        critical=True
    )

    # -------------------- Required fields: existence checks -------------------- #
    arena_name_exists = evaluator.add_custom_node(
        result=bool(extracted.arena_name and extracted.arena_name.strip()),
        id="arena_name",
        desc="The complete name of the arena is provided",
        parent=fields_node,
        critical=True
    )

    university_name_exists = evaluator.add_custom_node(
        result=bool(extracted.university_name and extracted.university_name.strip()),
        id="university_name",
        desc="The university the arena serves is provided",
        parent=fields_node,
        critical=True
    )

    # Accept any non-empty capacity string; exactness will be verified via sources in constraints
    capacity_exists = evaluator.add_custom_node(
        result=bool(extracted.exact_capacity and extracted.exact_capacity.strip()),
        id="exact_capacity",
        desc="The exact seating capacity is provided",
        parent=fields_node,
        critical=True
    )

    hof_name_exists = evaluator.add_custom_node(
        result=bool(extracted.hall_of_famer_name and extracted.hall_of_famer_name.strip()),
        id="hall_of_famer_name",
        desc="The name of the Hall of Fame inductee after whom the arena is named is provided",
        parent=fields_node,
        critical=True
    )

    sources_list = _clean_sources(extracted.reference_urls)

    # Supporting reference: at least one provided URL supports key arena claims (single leaf verification)
    supporting_ref_leaf = evaluator.add_leaf(
        id="supporting_reference",
        desc="At least one valid reference URL is provided that supports the key claims in the answer",
        parent=fields_node,
        critical=True
    )

    support_claim = (
        f"The provided reference page(s) explicitly support that the arena named '{extracted.arena_name}' "
        f"serves '{extracted.university_name}' for college basketball home games."
    )
    await evaluator.verify(
        claim=support_claim,
        node=supporting_ref_leaf,
        sources=sources_list or None,  # If empty, this will route to simple_verify and likely fail
        additional_instruction=(
            "Look for evidence that the arena is the home court, primary venue, or hosts home games for the specified university. "
            "A single credible page confirming this is sufficient. Accept official university/athletics pages or other credible sources."
        )
    )

    # Prepare prerequisites for criteria checks: ensure we don't proceed if key fields or supporting reference fail
    extra_prereqs = [arena_name_exists, university_name_exists, capacity_exists, hof_name_exists, supporting_ref_leaf]

    # -------------------- Criteria verifications (all critical leaves) -------------------- #
    # 1) Location: Pittsburgh, Pennsylvania
    loc_pitt_leaf = evaluator.add_leaf(
        id="location_pittsburgh",
        desc="The arena is located in Pittsburgh, Pennsylvania",
        parent=criteria_node,
        critical=True
    )
    loc_pitt_claim = f"The arena '{extracted.arena_name}' is located in Pittsburgh, Pennsylvania."
    await evaluator.verify(
        claim=loc_pitt_claim,
        node=loc_pitt_leaf,
        sources=sources_list or None,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Verify that the arena's address or location explicitly indicates Pittsburgh, PA (Pennsylvania). "
            "Allow 'Pittsburgh, PA' or equivalent phrasing."
        )
    )

    # 2) Capacity range: 3,000 to 4,000 for basketball
    capacity_range_leaf = evaluator.add_leaf(
        id="capacity_range",
        desc="The arena has a seating capacity between 3,000 and 4,000 for basketball games",
        parent=criteria_node,
        critical=True
    )
    capacity_range_claim = (
        f"The arena '{extracted.arena_name}' has a seating capacity between 3,000 and 4,000 for basketball games."
    )
    await evaluator.verify(
        claim=capacity_range_claim,
        node=capacity_range_leaf,
        sources=sources_list or None,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Check the capacity listed for basketball games. "
            "If multiple capacities are listed (e.g., different configurations), use the standard basketball capacity. "
            "Allow minor wording differences such as 'approximately' within this range."
        )
    )

    # 3) Named after a Naismith Hall of Fame inductee
    hof_named_leaf = evaluator.add_leaf(
        id="named_after_hall_of_famer",
        desc="The arena is named after a Naismith Memorial Basketball Hall of Fame inductee",
        parent=criteria_node,
        critical=True
    )
    hof_named_claim = (
        f"The arena '{extracted.arena_name}' is named after '{extracted.hall_of_famer_name}', "
        "who is a Naismith Memorial Basketball Hall of Fame inductee."
    )
    await evaluator.verify(
        claim=hof_named_claim,
        node=hof_named_leaf,
        sources=sources_list or None,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Confirm both parts: (a) the arena is named after the specified person, and (b) that person is a Naismith Hall of Fame inductee. "
            "Evidence can be found on the arena page, the university/athletics page, press releases, or the Hall of Fame website."
        )
    )

    # 4) Located on Forbes Avenue
    forbes_leaf = evaluator.add_leaf(
        id="forbes_avenue_location",
        desc="The arena is located on Forbes Avenue",
        parent=criteria_node,
        critical=True
    )
    forbes_claim = f"The arena '{extracted.arena_name}' is located on Forbes Avenue."
    await evaluator.verify(
        claim=forbes_claim,
        node=forbes_leaf,
        sources=sources_list or None,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Look for the street address or descriptive text indicating Forbes Avenue. "
            "Treat 'Forbes Ave' and 'Forbes Avenue' as equivalent."
        )
    )

    # 5) Opened or major renovation in the 2020s
    renovated_leaf = evaluator.add_leaf(
        id="opened_renovated_2020s",
        desc="The arena opened or underwent major renovation in the 2020s (2020-2029)",
        parent=criteria_node,
        critical=True
    )
    renovated_claim = (
        f"The arena '{extracted.arena_name}' opened or underwent major renovation in the 2020s (between 2020 and 2029)."
    )
    await evaluator.verify(
        claim=renovated_claim,
        node=renovated_leaf,
        sources=sources_list or None,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Confirm either an original opening or a major renovation (substantial overhaul) occurred in the 2020s. "
            "Terms such as 'renovated', 'reopened', 'major renovation' or equivalent are acceptable."
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
    Evaluate an answer for the Pittsburgh arena identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluation of college basketball arena identification task",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_arena_info(),
        template_class=ArenaExtraction,
        extraction_name="arena_extraction"
    )

    # Build verification tree and run checks
    await build_tree_and_verify(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()