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
TASK_ID = "nasa_astronaut_jhu_se_2026"
TASK_DESCRIPTION = """
Which NASA astronaut scheduled for a mission in 2026 holds a Master of Science degree in Systems Engineering from Johns Hopkins University?
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AstronautExtraction(BaseModel):
    """
    Structured extraction of the identified astronaut and cited sources from the answer.
    """
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_astronaut() -> str:
    return """
    Extract from the answer:
    1) name: The full name of the single identified astronaut who matches the description
       (NASA astronaut scheduled for a 2026 mission and who holds a Master of Science in Systems Engineering from Johns Hopkins University).
       - If multiple names are mentioned, choose the first one the answer presents as the correct match.
       - Return exactly the name string as it appears in the answer (do not add honorifics or titles).
    2) sources: A list of all URLs explicitly mentioned in the answer that are used as citations or evidence (including markdown links).
       - Include each URL exactly once.
       - Only include valid URLs that appear in the answer.
       - If no URLs are present, return an empty list.
    """.strip()


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for identifying the NASA astronaut with the specified educational background and 2026 mission assignment.
    """
    # Initialize evaluator with a parallel root (default)
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
        default_model=model
    )

    # Extract the astronaut name and all cited URLs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_astronaut(),
        template_class=AstronautExtraction,
        extraction_name="astronaut_extraction"
    )

    astronaut_name = extraction.name or ""
    sources_list: List[str] = extraction.sources or []

    # Build the rubric tree: Top-level critical node with three critical verification leaves
    astro_node = evaluator.add_parallel(
        id="Astronaut_Identification",
        desc="Correctly identifies the NASA astronaut with the specified educational background scheduled for a 2026 mission",
        parent=root,
        critical=True  # Critical parent: all children must pass
    )

    # Leaf 1: NASA Astronaut status
    leaf_status = evaluator.add_leaf(
        id="NASA_Astronaut_Status",
        desc="The identified individual is a NASA astronaut",
        parent=astro_node,
        critical=True
    )

    # Leaf 2: 2026 mission assignment
    leaf_mission = evaluator.add_leaf(
        id="2026_Mission_Assignment",
        desc="The identified individual is scheduled for a NASA mission in 2026",
        parent=astro_node,
        critical=True
    )

    # Leaf 3: Educational credential
    leaf_edu = evaluator.add_leaf(
        id="Educational_Credential",
        desc="The identified individual holds a Master of Science degree in Systems Engineering from Johns Hopkins University",
        parent=astro_node,
        critical=True
    )

    # Prepare claims and tailored verification instructions
    claim_status = f"The person named '{astronaut_name}' is a NASA astronaut."
    ins_status = (
        "Prefer NASA official pages (e.g., astronaut biography on nasa.gov). "
        "Mark as supported only if the provided webpage(s) explicitly indicate the person is "
        "a NASA astronaut or astronaut candidate (current or former). "
        "If the provided name is missing or empty, or if no relevant evidence appears in the URLs, mark as not supported."
    )

    claim_mission = f"'{astronaut_name}' is scheduled for a NASA mission in 2026."
    ins_mission = (
        "Verify that the referenced page(s) explicitly state that this person is assigned to or scheduled for a NASA mission "
        "in the year 2026 (e.g., a mission assignment press release or biography indicating a 2026 mission). "
        "Accept phrases like 'scheduled for 2026', 'targeted for 2026', or a specific 2026 date. "
        "If the date is different or unclear, mark as not supported. "
        "If the provided name is missing or empty, or if no relevant evidence appears in the URLs, mark as not supported."
    )

    claim_edu = (
        f"'{astronaut_name}' holds a Master of Science (MS/M.S./Master's) in Systems Engineering from Johns Hopkins University."
    )
    ins_edu = (
        "Verify the educational credential is explicitly stated on the page(s), allowing minor phrasing variations "
        "such as 'M.S.', 'MS', 'Master of Science', and references to Johns Hopkins University or its engineering school/affiliates "
        "(e.g., Whiting School of Engineering, Engineering for Professionals). "
        "The field must be Systems Engineering. "
        "If the provided name is missing or empty, or if no relevant evidence appears in the URLs, mark as not supported."
    )

    # Run three verifications in parallel for better concurrency and to avoid cross-sibling gating
    await evaluator.batch_verify(
        [
            (claim_status, sources_list, leaf_status, ins_status),
            (claim_mission, sources_list, leaf_mission, ins_mission),
            (claim_edu,    sources_list, leaf_edu,    ins_edu),
        ]
    )

    # Return the evaluation summary
    return evaluator.get_summary()