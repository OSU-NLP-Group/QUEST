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
TASK_ID = "quantum_four_nines_2025"
TASK_DESCRIPTION = (
    "Which quantum computing company announced in October 2025 that it achieved a new world record "
    "for two-qubit gate fidelity exceeding 99.99%, becoming the first company to cross the 'four-nines' benchmark?"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class QuantumAchievementExtraction(BaseModel):
    """
    Information explicitly stated in the answer, as extracted for checking rubric constraints.
    """
    company_name: Optional[str] = None
    mentions_october_2025: Optional[bool] = None
    mentions_exceed_99_99: Optional[bool] = None
    claims_world_record: Optional[bool] = None
    claims_first_to_cross_four_nines: Optional[bool] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_quantum_achievement() -> str:
    return """
    Extract the specific details the answer provides regarding the asked quantum computing announcement.

    Return a JSON object with the following fields:
    - company_name: The exact company name explicitly identified in the answer as the one that made the announcement. If not clearly named, return null.
    - mentions_october_2025: true if the answer explicitly states that the announcement occurred in October 2025 (allow variants like "Oct 2025" or "Oct. 2025"); otherwise false.
    - mentions_exceed_99_99: true if the answer explicitly states the two-qubit gate fidelity exceeded 99.99% (e.g., “exceeded 99.99%”, “> 99.99%”, “crossed four nines (99.99%)”); otherwise false.
    - claims_world_record: true if the answer explicitly states that this was a new world record for two-qubit gate fidelity (allow variants like “world-record”); otherwise false.
    - claims_first_to_cross_four_nines: true if the answer explicitly states the company was the first to cross the “four-nines” (99.99%) benchmark (allow phrasing like “first to reach >99.99%”); otherwise false.
    - source_urls: an array of all URLs explicitly present in the answer that the answer cites as sources for this claim (e.g., press releases, official blogs, news coverage). If none, return an empty array.

    Important:
    - Only extract information that is explicitly stated in the answer text.
    - Do not infer any missing information.
    - For URLs, include only full valid URLs that appear in the answer (including those inside markdown links).
    """


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_quantum_achievement(
    evaluator: Evaluator,
    parent_node,
    extracted: QuantumAchievementExtraction,
) -> None:
    """
    Build the verification tree to match the rubric and run the corresponding checks.
    This rubric focuses on what the answer explicitly states, so we primarily use simple verification
    against the answer text for each constraint.
    """

    # Critical parent node: all children must pass
    qca_node = evaluator.add_parallel(
        id="Quantum_Computing_Achievement_Identification",
        desc="Identify the quantum computing company described and ensure the answer satisfies all stated constraints about the October 2025 two-qubit gate fidelity announcement.",
        parent=parent_node,
        critical=True,
    )

    # 1) Company_Identification
    node_company = evaluator.add_leaf(
        id="Company_Identification",
        desc="The answer names a specific quantum computing company as the one that made the announcement (i.e., provides a concrete company identity, not just a description).",
        parent=qca_node,
        critical=True,
    )
    company_claim = (
        "The answer names a specific, concrete quantum computing company that made the announcement, "
        "not just a vague description like 'a company' or 'the team'."
    )
    await evaluator.verify(
        claim=company_claim,
        node=node_company,
        additional_instruction=(
            "Pass only if the answer clearly identifies a company by name (e.g., 'IBM', 'Google Quantum AI', "
            "'Quantinuum', 'IonQ', 'Rigetti', 'Atom Computing', 'PsiQuantum', etc.). "
            "If the answer is vague (e.g., 'a company', 'the researchers', 'the team') or does not specify the company, fail."
        ),
    )

    # 2) Announcement_Timing
    node_timing = evaluator.add_leaf(
        id="Announcement_Timing",
        desc="The answer states that the company announced the achievement in October 2025.",
        parent=qca_node,
        critical=True,
    )
    timing_claim = (
        "The answer explicitly states that the announcement occurred in October 2025."
    )
    await evaluator.verify(
        claim=timing_claim,
        node=node_timing,
        additional_instruction=(
            "Accept variants like 'October 2025', 'Oct 2025', or 'Oct. 2025'. "
            "The timing must be clearly tied to the announcement."
        ),
    )

    # 3) Two_Qubit_Gate_Fidelity_Threshold
    node_threshold = evaluator.add_leaf(
        id="Two_Qubit_Gate_Fidelity_Threshold",
        desc="The answer states that the reported two-qubit gate fidelity exceeded 99.99% (i.e., crossed the 'four-nines' benchmark).",
        parent=qca_node,
        critical=True,
    )
    threshold_claim = (
        "The answer explicitly states that the two-qubit gate fidelity exceeded 99.99%, i.e., it crossed the 'four-nines' threshold."
    )
    await evaluator.verify(
        claim=threshold_claim,
        node=node_threshold,
        additional_instruction=(
            "Accept equivalent phrasings such as 'exceeded 99.99%', '> 99.99%', 'greater than 99.99%', "
            "or 'crossed four nines (99.99%)'. If the answer merely says '99.99%' without implying 'exceeded', fail."
        ),
    )

    # 4) World_Record_Claim
    node_world_record = evaluator.add_leaf(
        id="World_Record_Claim",
        desc="The answer states that the achievement was a new world record for two-qubit gate fidelity.",
        parent=qca_node,
        critical=True,
    )
    world_record_claim = (
        "The answer explicitly states that this was a new world record for two-qubit gate fidelity."
    )
    await evaluator.verify(
        claim=world_record_claim,
        node=node_world_record,
        additional_instruction=(
            "Look for clear wording like 'new world record' or 'world-record'. "
            "If the answer only says 'record' without global context or is ambiguous, fail."
        ),
    )

    # 5) First_To_Cross_Four_Nines
    node_first = evaluator.add_leaf(
        id="First_To_Cross_Four_Nines",
        desc="The answer states that the company was the first to cross the 'four-nines' (99.99%) benchmark in this context.",
        parent=qca_node,
        critical=True,
    )
    first_claim = (
        "The answer explicitly states that the company was the first to cross the 'four-nines' (99.99%) benchmark for two-qubit gate fidelity."
    )
    await evaluator.verify(
        claim=first_claim,
        node=node_first,
        additional_instruction=(
            "Accept equivalent phrasing such as 'first to exceed 99.99%' or 'first to reach >99.99%'. "
            "If the answer does not indicate 'first', fail."
        ),
    )

    # Optionally record a concise extraction summary for debugging/inspection
    evaluator.add_custom_info(
        info={
            "extracted_company": extracted.company_name,
            "answer_mentions": {
                "october_2025": extracted.mentions_october_2025,
                "exceed_99_99": extracted.mentions_exceed_99_99,
                "world_record": extracted.claims_world_record,
                "first_to_cross_four_nines": extracted.claims_first_to_cross_four_nines,
            },
            "source_urls": extracted.source_urls,
        },
        info_type="extraction_summary"
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
    Evaluate an answer for the 'quantum_four_nines_2025' task.

    The rubric requires that the answer itself explicitly states:
    - which company made the announcement,
    - that the announcement occurred in October 2025,
    - that two-qubit gate fidelity exceeded 99.99%,
    - that it was a new world record,
    - and that the company was the first to cross the four-nines benchmark.
    """
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_quantum_achievement(),
        template_class=QuantumAchievementExtraction,
        extraction_name="quantum_achievement_extraction",
    )

    # Build tree and run checks per rubric
    await build_and_verify_quantum_achievement(
        evaluator=evaluator,
        parent_node=root,
        extracted=extracted,
    )

    # Return final evaluation summary
    return evaluator.get_summary()