import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

TASK_ID = "uneswa_phd_econ_admission"
TASK_DESCRIPTION = "What is the name of the doctoral program in the field of economics offered at the University of Eswatini, and what is the general educational degree requirement for admission to this program?"

EXPECTED_PROGRAM_NAME = "PhD in Agricultural and Applied Economics"
EXPECTED_REQUIREMENT_SUMMARY = "Applicants must hold a relevant Master's degree from the University of Eswatini or its equivalent from another recognized university."


class AnswerExtraction(BaseModel):
    program_name: Optional[str] = None
    program_sources: List[str] = Field(default_factory=list)
    admission_requirement: Optional[str] = None
    requirement_sources: List[str] = Field(default_factory=list)


def prompt_extract_answer_info() -> str:
    return """
    Extract from the answer the following items:
    1) program_name: The explicit name given for the doctoral program in the field of economics at the University of Eswatini (UNESWA). Capture the program name exactly as stated (e.g., 'PhD in Agricultural and Applied Economics'). If not present, return null.
    2) program_sources: All URLs in the answer that are cited as sources for the program identification. Return an array of URLs. If none, return an empty array.
    3) admission_requirement: The general educational degree requirement that the answer states for admission to this program. Focus on whether it states a relevant Master's degree is required (e.g., 'a relevant Master's degree from UNESWA or equivalent from a recognized university'). If not present, return null.
    4) requirement_sources: All URLs in the answer that are cited as sources for the admission requirement. Return an array of URLs. If none, return an empty array.

    Rules:
    - Extract only what is explicitly present in the provided answer text.
    - For any missing item, return null (or empty array for sources).
    - For URLs, include only valid URLs (plain or markdown), and ensure they are full URLs (prepend http:// if missing).
    """


async def build_verification_tree_and_verify(
    evaluator: Evaluator,
    extraction: AnswerExtraction,
) -> None:
    # Create a critical parallel aggregator to mirror the rubric root
    overall_node = evaluator.add_parallel(
        id="overall_evaluation",
        desc="Evaluate the complete answer about the PhD program and admission requirement",
        parent=evaluator.root,
        critical=True,
    )

    # Leaf 1: Program identification check (critical)
    program_leaf = evaluator.add_leaf(
        id="program_identification",
        desc="The answer correctly identifies that the PhD in Agricultural and Applied Economics is the economics-focused doctoral program at University of Eswatini",
        parent=overall_node,
        critical=True,
    )

    # Claim focuses on whether the answer correctly names the program.
    # We use simple verification against the provided answer text.
    # If URLs are present, we still keep the claim about the program name itself (not "the answer states ..."),
    # because URL verification expects a fact supported by the page, not meta-statements about the answer.
    program_claim = (
        "The University of Eswatini offers a doctoral program in the field of economics called "
        "'PhD in Agricultural and Applied Economics'."
    )

    # Prefer simple verification to check answer content; provide permissive matching instructions.
    await evaluator.verify(
        claim=program_claim,
        node=program_leaf,
        sources=None,
        additional_instruction=(
            "Judge based on the answer text whether it clearly names the doctoral program as "
            "'PhD in Agricultural and Applied Economics'. Accept minor variations like 'Ph.D.' or "
            "the use of '&' in 'Agricultural & Applied Economics', and accept 'UNESWA' as equivalent "
            "to 'University of Eswatini'. If the answer names a different program, treat this as incorrect."
        ),
    )

    # Leaf 2: Admission requirement check (critical)
    admission_leaf = evaluator.add_leaf(
        id="admission_requirement",
        desc="The answer states that applicants must hold a relevant Master's degree from the University of Eswatini or its equivalent from another recognized university",
        parent=overall_node,
        critical=True,
    )

    # Claim focuses on the degree requirement; again verify against the answer content.
    admission_claim = (
        "Admission to the PhD in Agricultural and Applied Economics requires applicants to hold "
        "a relevant Master's degree from the University of Eswatini or an equivalent Master's degree "
        "from another recognized university."
    )

    await evaluator.verify(
        claim=admission_claim,
        node=admission_leaf,
        sources=None,
        additional_instruction=(
            "Judge based on the answer text whether it states a relevant Master's degree requirement for admission, "
            "with allowance for synonyms such as 'Master’s degree', 'equivalent qualification', or 'recognized university'. "
            "If the answer omits this requirement or states a different requirement, treat this as incorrect."
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
    evaluator.initialize(
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

    extraction = await evaluator.extract(
        prompt=prompt_extract_answer_info(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction",
    )

    evaluator.add_ground_truth(
        {
            "expected_program_name": EXPECTED_PROGRAM_NAME,
            "expected_requirement_summary": EXPECTED_REQUIREMENT_SUMMARY,
        },
        gt_type="ground_truth",
    )

    await build_verification_tree_and_verify(evaluator, extraction)

    return evaluator.get_summary()