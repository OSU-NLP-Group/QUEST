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
TASK_ID = "uiuc_imba_tuition_tests"
TASK_DESCRIPTION = """
What is the total tuition cost for the University of Illinois online MBA program (iMBA) offered through Coursera, and does this program require GMAT or GRE scores for admission?
"""

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class IMBAExtraction(BaseModel):
    """
    Structured extraction from the agent's answer for the UIUC iMBA task.
    """
    # Optional text snippet indicating the program scope as stated in the answer
    program_scope_text: Optional[str] = None

    # URLs explicitly cited in the answer that identify or describe the program
    program_sources: List[str] = Field(default_factory=list)

    # The total tuition amount for the entire iMBA program as stated in the answer (keep exact formatting, e.g., "$24,000", "about $24k USD")
    tuition_total_usd: Optional[str] = None

    # URLs explicitly cited in the answer to support the tuition figure
    tuition_sources: List[str] = Field(default_factory=list)

    # The GMAT/GRE requirement status as stated in the answer (e.g., "not required", "required", "waived", "test optional")
    tests_requirement: Optional[str] = None

    # URLs explicitly cited in the answer to support the GMAT/GRE requirement
    tests_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_imba_info() -> str:
    return """
    Extract the information the answer provides about the University of Illinois iMBA (online MBA) offered through Coursera.

    You must return the following fields:
    1) program_scope_text: The exact phrase(s) in the answer that identify the program (e.g., "University of Illinois iMBA", "Gies online MBA", "iMBA on Coursera"). If the answer mentions a different program (e.g., iMSA, iMSM, on-campus MBA), return that text here. If no program is clearly identified, return null.
    2) program_sources: All URLs explicitly cited in the answer that identify or describe the program (e.g., the Coursera iMBA page, Gies College of Business iMBA page). Return only actual URLs mentioned. If none are provided, return an empty list.

    3) tuition_total_usd: The total tuition cost for the entire iMBA program as stated in the answer, preserving the exact formatting and currency (e.g., "$24,000", "around $24k USD", "USD 24,000"). If the answer does not state a total tuition amount in USD terms, return null.
    4) tuition_sources: All URLs explicitly cited in the answer that support the tuition amount. Return only actual URLs mentioned. If none are provided, return an empty list.

    5) tests_requirement: The GMAT/GRE requirement status as stated in the answer (e.g., "GMAT/GRE are not required", "GMAT required", "test optional", "waivers available"). Preserve the wording from the answer. If it's not stated, return null.
    6) tests_sources: All URLs explicitly cited in the answer that support the GMAT/GRE requirement status. Return only actual URLs mentioned. If none are provided, return an empty list.

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not infer or invent any information.
    - For URLs, include only valid URLs that appear in the answer (plain URLs or markdown links).
    - If a field is missing, return null (for single-value fields) or an empty list (for list fields).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine multiple URL lists into a unique, ordered list."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_program_scope(
    evaluator: Evaluator,
    parent_node,
    extraction: IMBAExtraction
) -> None:
    """
    Verify that the answer is explicitly about the University of Illinois iMBA (online MBA) offered through Coursera,
    not a different Illinois MBA variant.
    """
    node = evaluator.add_leaf(
        id="Correct_Program_Scope",
        desc="The answer is explicitly about the University of Illinois iMBA (online MBA) offered through Coursera (not a different Illinois MBA variant).",
        parent=parent_node,
        critical=True,
    )

    claim = (
        "The answer is explicitly about the University of Illinois iMBA (the Gies College of Business online MBA) offered through Coursera, "
        "and not a different Illinois MBA variant (e.g., on-campus MBA, Professional MBA, iMSA, iMSM, executive MBA, or programs from UIC)."
    )

    # Use any program-identifying URLs if the answer cited them; otherwise the verifier will fall back to simple verification.
    sources = extraction.program_sources if extraction.program_sources else None

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=(
            "Confirm that the described program is the iMBA offered by Gies College of Business (University of Illinois Urbana-Champaign) via Coursera. "
            "Accept reasonable phrasing variants (e.g., 'UIUC iMBA', 'Gies iMBA on Coursera'). "
            "If the answer is about a different Illinois program (e.g., iMSA/iMSM or on-campus MBA) or a different institution, judge this claim as incorrect."
        ),
    )


async def verify_tuition_cost(
    evaluator: Evaluator,
    parent_node,
    extraction: IMBAExtraction
) -> None:
    """
    Verify that the answer states the total tuition cost in USD and that the amount matches official published information.
    """
    node = evaluator.add_leaf(
        id="Tuition_Cost_Accuracy_USD",
        desc="The answer states the total tuition cost for the iMBA program in USD and the stated amount matches the official published tuition information.",
        parent=parent_node,
        critical=True,
    )

    # If the answer provides a figure, verify that exact figure against official sources;
    # otherwise verify the existence claim and instruct the judge to mark incorrect if not present.
    if extraction.tuition_total_usd and extraction.tuition_total_usd.strip():
        value_text = extraction.tuition_total_usd.strip()
        claim = (
            f"The total tuition cost for the University of Illinois iMBA program is {value_text} USD (for the entire program)."
        )
        sources = _combine_sources(extraction.tuition_sources, extraction.program_sources)
        add_ins = (
            "Verify the total-program tuition figure (not per-course or per-credit). "
            "Allow reasonable rounding (e.g., '$24k' vs '$24,000'). "
            "Only accept USD amounts. Use official pages (Gies iMBA tuition/admissions or Coursera iMBA program pages) to confirm."
        )
    else:
        claim = (
            "The answer includes a total tuition amount in USD for the University of Illinois iMBA program and that amount matches official published information."
        )
        sources = None
        add_ins = (
            "If the answer does not include an explicit USD total tuition figure for the entire program, judge this claim as incorrect."
        )

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins,
    )


async def verify_tests_requirement(
    evaluator: Evaluator,
    parent_node,
    extraction: IMBAExtraction
) -> None:
    """
    Verify that the answer correctly states whether GMAT/GRE scores are required or not required for admission.
    """
    node = evaluator.add_leaf(
        id="GMAT_GRE_Requirement_Status",
        desc="The answer correctly states whether GMAT/GRE scores are required or not required for admission, consistent with the official admission policy.",
        parent=parent_node,
        critical=True,
    )

    if extraction.tests_requirement and extraction.tests_requirement.strip():
        status_text = extraction.tests_requirement.strip()
        claim = (
            f"GMAT/GRE scores are {status_text} for admission to the University of Illinois iMBA program."
        )
        sources = _combine_sources(extraction.tests_sources, extraction.program_sources)
        add_ins = (
            "Interpret common wording: 'not required', 'no GMAT/GRE', or 'test optional' should be treated as not required; "
            "'required' means mandatory. Verify against official admissions/policy pages for the iMBA."
        )
    else:
        claim = (
            "The answer clearly states whether GMAT/GRE scores are required or not required for admission to the University of Illinois iMBA program, consistent with official policy."
        )
        sources = None
        add_ins = (
            "If the answer does not clearly state the GMAT/GRE requirement status, judge this claim as incorrect."
        )

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins,
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
    Evaluate an agent's answer to the UIUC iMBA tuition and GMAT/GRE requirement task.
    """
    # Initialize evaluator: Root is a parallel aggregator for this task
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

    # Extract relevant structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_imba_info(),
        template_class=IMBAExtraction,
        extraction_name="imba_answer_extraction",
    )

    # Build a critical aggregation node that mirrors the rubric root
    rubric_root = evaluator.add_parallel(
        id="Program_Information_Verification",
        desc="Verify that the answer correctly addresses the University of Illinois iMBA (online MBA) offered through Coursera and provides the required tuition and admission-test information.",
        parent=root,
        critical=True,
    )

    # Add the three critical verification leaves
    await verify_program_scope(evaluator, rubric_root, extraction)
    await verify_tuition_cost(evaluator, rubric_root, extraction)
    await verify_tests_requirement(evaluator, rubric_root, extraction)

    # Return standardized summary
    return evaluator.get_summary()