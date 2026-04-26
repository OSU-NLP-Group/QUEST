import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "austin_deputy_cm_verification"
TASK_DESCRIPTION = (
    "Identify the individual who currently serves as Deputy City Manager for the City of Austin, Texas, "
    "and verify that this person meets the following criteria: "
    "(1) previously served in a senior management position (Deputy City Manager or Assistant City Manager level) with the City of Dallas, Texas; "
    "(2) completed the Senior Executives in State and Local Government program at Harvard University; "
    "(3) is a credentialed manager with the International City/County Management Association (ICMA); and "
    "(4) currently oversees at least one major utility department (such as Austin Energy or Austin Water) as part of their permanent portfolio."
)


class DeputyCMExtraction(BaseModel):
    person_name: Optional[str] = None
    current_title: Optional[str] = None
    current_agency: Optional[str] = None
    position_sources: List[str] = Field(default_factory=list)

    dallas_title: Optional[str] = None
    dallas_level: Optional[str] = None  # e.g., "Assistant City Manager" or "Deputy City Manager"
    dallas_agency: Optional[str] = None
    dallas_sources: List[str] = Field(default_factory=list)

    harvard_program_name: Optional[str] = None
    harvard_sources: List[str] = Field(default_factory=list)

    icma_credential_status: Optional[str] = None
    icma_sources: List[str] = Field(default_factory=list)

    oversight_departments: List[str] = Field(default_factory=list)
    oversight_sources: List[str] = Field(default_factory=list)


def prompt_extract_deputy_cm() -> str:
    return """
    Extract structured information from the answer regarding Austin's Deputy City Manager and the four specified criteria.
    Return a JSON object with these fields:

    1) person_name: The full name of the individual the answer claims is currently Deputy City Manager of Austin.
    2) current_title: The current title attributed to this individual (e.g., "Deputy City Manager").
    3) current_agency: The current agency/org (e.g., "City of Austin").
    4) position_sources: All URLs explicitly cited that support the current position claim.

    5) dallas_title: The title held in the City of Dallas (if claimed).
    6) dallas_level: If the answer indicates senior management level, specify "Assistant City Manager" or "Deputy City Manager" when applicable.
    7) dallas_agency: The agency/org for the Dallas role (should be "City of Dallas" or equivalent).
    8) dallas_sources: All URLs explicitly cited that support the Dallas senior management experience claim.

    9) harvard_program_name: The exact program name if mentioned (e.g., "Senior Executives in State and Local Government").
    10) harvard_sources: All URLs explicitly cited that support completion of the Harvard program (often Harvard Kennedy School).

    11) icma_credential_status: A phrase indicating ICMA credential (e.g., "ICMA Credentialed Manager", "ICMA-CM"), if claimed.
    12) icma_sources: All URLs explicitly cited that support the ICMA credential status.

    13) oversight_departments: A list of department names the answer claims this person currently oversees as part of their permanent portfolio (e.g., "Austin Energy", "Austin Water").
    14) oversight_sources: All URLs explicitly cited that support the current portfolio oversight claim.

    IMPORTANT:
    - Extract only what is explicitly stated in the answer. Do not invent information.
    - For URL fields, include only valid URLs that appear in the answer (including markdown links).
    - If a field is missing in the answer, set it to null (for strings) or an empty list (for arrays).
    - Do not deduplicate across categories; place each URL in the corresponding *_sources list.
    """


async def verify_position_identification(
    evaluator: Evaluator,
    parent_node,
    info: DeputyCMExtraction,
) -> None:
    """
    Create and verify the Position_Identification leaf under the critical sequential parent.
    """
    pos_leaf = evaluator.add_leaf(
        id="Position_Identification",
        desc="Correctly identified the individual currently serving as Deputy City Manager of Austin, Texas",
        parent=parent_node,
        critical=True,
    )

    person = info.person_name or ""
    claim = (
        f"The answer identifies '{person}' as the current Deputy City Manager of the City of Austin, Texas, "
        f"and this identification is correct."
    )

    await evaluator.verify(
        claim=claim,
        node=pos_leaf,
        sources=info.position_sources if info.position_sources else None,
        additional_instruction=(
            "Verify from the cited sources (if any) whether this person currently holds the title 'Deputy City Manager' "
            "at the City of Austin. If no sources are provided, rely on the answer's content to judge correctness. "
            "Focus on current status, not past roles."
        ),
    )


async def verify_criteria(
    evaluator: Evaluator,
    parent_node,
    info: DeputyCMExtraction,
) -> None:
    """
    Build the parallel critical node 'Criteria_Verification' and add four critical leaves.
    """
    criteria_node = evaluator.add_parallel(
        id="Criteria_Verification",
        desc="Verification that the identified individual meets all four required criteria specified in the question",
        parent=parent_node,
        critical=True,
    )

    # 1) Prior Dallas Experience
    dallas_leaf = evaluator.add_leaf(
        id="Prior_Dallas_Experience",
        desc="Verified that the individual previously held a senior management position (Deputy City Manager or Assistant City Manager level) with the City of Dallas, Texas",
        parent=criteria_node,
        critical=True,
    )

    dallas_title = info.dallas_title or ""
    dallas_level = info.dallas_level or ""
    dallas_agency = info.dallas_agency or "City of Dallas"
    dallas_claim = (
        f"This person previously served in a senior management position (Assistant City Manager or Deputy City Manager) "
        f"with {dallas_agency}. Claimed title: '{dallas_title or dallas_level}'."
    )
    await evaluator.verify(
        claim=dallas_claim,
        node=dallas_leaf,
        sources=info.dallas_sources if info.dallas_sources else None,
        additional_instruction=(
            "Senior management is defined here as either 'Assistant City Manager' or 'Deputy City Manager'. "
            "Confirm that the role was specifically at the City of Dallas."
        ),
    )

    # 2) Harvard Executive Program
    harvard_leaf = evaluator.add_leaf(
        id="Harvard_Executive_Program",
        desc="Confirmed completion of the Senior Executives in State and Local Government program at Harvard University",
        parent=criteria_node,
        critical=True,
    )

    program_name = info.harvard_program_name or "Senior Executives in State and Local Government"
    harvard_claim = (
        f"This person completed the '{program_name}' program at Harvard University (commonly offered by Harvard Kennedy School)."
    )
    await evaluator.verify(
        claim=harvard_claim,
        node=harvard_leaf,
        sources=info.harvard_sources if info.harvard_sources else None,
        additional_instruction=(
            "Look for explicit statements of program completion at Harvard (especially Harvard Kennedy School). "
            "Allow reasonable naming variations of the program."
        ),
    )

    # 3) ICMA Credential
    icma_leaf = evaluator.add_leaf(
        id="ICMA_Credential",
        desc="Verified status as a credentialed manager with the International City/County Management Association (ICMA)",
        parent=criteria_node,
        critical=True,
    )

    icma_status = info.icma_credential_status or "ICMA Credentialed Manager (ICMA-CM)"
    icma_claim = (
        f"This person is recognized as an ICMA Credentialed Manager (e.g., '{icma_status}')."
    )
    await evaluator.verify(
        claim=icma_claim,
        node=icma_leaf,
        sources=info.icma_sources if info.icma_sources else None,
        additional_instruction=(
            "Confirm credentialed status specifically with ICMA (International City/County Management Association). "
            "Accept synonyms like 'ICMA-CM' or 'Credentialed Manager'."
        ),
    )

    # 4) Current Portfolio Oversight
    oversight_leaf = evaluator.add_leaf(
        id="Current_Portfolio_Oversight",
        desc="Verified that the individual currently oversees at least one major utility department (such as Austin Energy or Austin Water) as part of their permanent portfolio",
        parent=criteria_node,
        critical=True,
    )

    depts = info.oversight_departments or []
    depts_str = ", ".join(depts) if depts else "None provided"
    oversight_claim = (
        f"This person currently oversees at least one major utility department as part of their permanent portfolio "
        f"(e.g., Austin Energy or Austin Water). Claimed departments: {depts_str}."
    )
    await evaluator.verify(
        claim=oversight_claim,
        node=oversight_leaf,
        sources=info.oversight_sources if info.oversight_sources else None,
        additional_instruction=(
            "Verify current, ongoing portfolio oversight (not temporary assignments). "
            "Treat Austin Energy and Austin Water as prime examples of 'major utility departments'. "
            "If sources indicate oversight of either, the criterion is satisfied."
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
    """
    Entry point for evaluating the Deputy City Manager of Austin verification task.
    """
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

    # Extract structured information from the answer
    info: DeputyCMExtraction = await evaluator.extract(
        prompt=prompt_extract_deputy_cm(),
        template_class=DeputyCMExtraction,
        extraction_name="deputy_cm_extraction",
    )

    # Build the critical sequential verification block as described in the rubric
    top_node = evaluator.add_sequential(
        id="Deputy_City_Manager_Verification",
        desc="Complete verification of Austin's Deputy City Manager position holder and their qualifications",
        parent=root,
        critical=True,
    )

    # 1) Position Identification (leaf)
    await verify_position_identification(evaluator, top_node, info)

    # 2) Criteria Verification (parallel node with 4 critical leaves)
    await verify_criteria(evaluator, top_node, info)

    return evaluator.get_summary()