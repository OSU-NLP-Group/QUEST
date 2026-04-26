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
TASK_ID = "nsf_bio_postdoc_program"
TASK_DESCRIPTION = (
    "What is the name of the NSF-funded postdoctoral fellowship program in biological sciences that provides an "
    "annual stipend of $70,000 (paid directly to fellows at $5,833.33 per month), requires applicants to be U.S. "
    "citizens, nationals, or permanent residents who hold a doctoral degree before the fellowship begins, has a duration "
    "of two or three years, and provides a research and training allowance in addition to the stipend?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    """
    Extracted program information from the answer.
    """
    program_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
    Identify the single postdoctoral fellowship program that the answer claims satisfies all of the constraints in the task.
    Extract the following:
    - program_name: The exact name of the fellowship program as written in the answer. Do not invent or normalize; copy the name from the answer text.
    - source_urls: A list of all URLs explicitly included in the answer (including markdown links). Only include valid URLs actually present in the answer text.
    If the answer mentions multiple programs, choose the one that best matches the constraints described in the task, and still extract all URLs mentioned in the answer.
    If program_name is not stated explicitly, return null for program_name.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _prog_ref(name: Optional[str]) -> str:
    """
    Create a human-readable reference to the program in claims.
    """
    return name if (name and name.strip()) else "the identified fellowship program"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_fellowship_criteria(
    evaluator: Evaluator,
    parent_node,
    extracted: ProgramExtraction,
) -> None:
    """
    Build the verification subtree and run checks for the eight critical criteria.
    """
    # Parent critical node aggregating all criteria
    criteria_node = evaluator.add_parallel(
        id="correct_fellowship_program_identification",
        desc="The identified postdoctoral fellowship program matches all specified criteria in the question/constraints.",
        parent=parent_node,
        critical=True,
    )

    program_name = extracted.program_name or None
    sources = extracted.source_urls  # can be empty; framework will fallback to simple verify

    # 1) NSF Funding Source
    nsf_node = evaluator.add_leaf(
        id="nsf_funding_source",
        desc="The fellowship program is funded by the National Science Foundation (NSF).",
        parent=criteria_node,
        critical=True,
    )
    claim_nsf = f"{_prog_ref(program_name)} is funded by the National Science Foundation (NSF)."
    await evaluator.verify(
        claim=claim_nsf,
        node=nsf_node,
        sources=sources,
        additional_instruction=(
            "Support may include an explicit NSF logo, hosting on an nsf.gov domain, or text stating NSF sponsorship/"
            "funding. Consider it supported if the page clearly indicates NSF is the funding agency."
        ),
    )

    # 2) Discipline Focus: Postdoctoral fellowship in biological sciences
    discipline_node = evaluator.add_leaf(
        id="discipline_focus",
        desc="The fellowship is specifically for postdoctoral research in biological sciences.",
        parent=criteria_node,
        critical=True,
    )
    claim_discipline = (
        f"{_prog_ref(program_name)} is specifically a postdoctoral fellowship in the biological sciences (biology)."
    )
    await evaluator.verify(
        claim=claim_discipline,
        node=discipline_node,
        sources=sources,
        additional_instruction=(
            "Look for language like 'Postdoctoral Research Fellowships in Biology' or equivalent. Accept reasonable "
            "synonyms (e.g., 'biology', 'BIO division') indicating the program is for postdoctoral researchers in the "
            "biological sciences."
        ),
    )

    # 3) Annual Stipend Amount: $70,000
    stipend_node = evaluator.add_leaf(
        id="annual_stipend_amount",
        desc="The fellowship provides an annual stipend of $70,000.",
        parent=criteria_node,
        critical=True,
    )
    claim_stipend = f"{_prog_ref(program_name)} provides an annual stipend of $70,000."
    await evaluator.verify(
        claim=claim_stipend,
        node=stipend_node,
        sources=sources,
        additional_instruction=(
            "Verify the stipend amount is $70,000 per year. If the page lists a different yearly amount, mark as not supported."
        ),
    )

    # 4) Monthly Payment Structure: Paid directly to fellows at $5,833.33 per month
    monthly_node = evaluator.add_leaf(
        id="monthly_payment_structure",
        desc="The stipend is paid directly to fellows at $5,833.33 per month.",
        parent=criteria_node,
        critical=True,
    )
    claim_monthly = (
        f"The stipend for {_prog_ref(program_name)} is paid directly to fellows monthly at $5,833.33."
    )
    await evaluator.verify(
        claim=claim_monthly,
        node=monthly_node,
        sources=sources,
        additional_instruction=(
            "Consider this supported if BOTH of the following are satisfied by the source(s): "
            "(1) The stipend is paid directly to the fellow (not to the host institution), and "
            "(2) The stipend is paid monthly at a rate equivalent to $70,000/12 ≈ $5,833.33. "
            "If the exact $5,833.33 number is not explicitly printed but the page clearly states a $70,000 annual stipend "
            "paid monthly, allow arithmetic inference for the monthly amount."
        ),
    )

    # 5) Citizenship Eligibility
    citizenship_node = evaluator.add_leaf(
        id="citizenship_eligibility",
        desc="Applicants must be U.S. citizens, U.S. nationals, or U.S. permanent residents.",
        parent=criteria_node,
        critical=True,
    )
    claim_citizenship = (
        f"Eligible applicants for {_prog_ref(program_name)} must be U.S. citizens, U.S. nationals, or U.S. permanent residents."
    )
    await evaluator.verify(
        claim=claim_citizenship,
        node=citizenship_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit eligibility criteria language listing U.S. citizens, U.S. nationals, or permanent residents. "
            "All three categories must be acceptable for support."
        ),
    )

    # 6) Doctoral Degree Requirement
    degree_node = evaluator.add_leaf(
        id="doctoral_degree_requirement",
        desc="Applicants must hold a doctoral degree before the fellowship begins.",
        parent=criteria_node,
        critical=True,
    )
    claim_degree = (
        f"Applicants to {_prog_ref(program_name)} must hold a doctoral degree (e.g., PhD) before the fellowship start date."
    )
    await evaluator.verify(
        claim=claim_degree,
        node=degree_node,
        sources=sources,
        additional_instruction=(
            "Support requires the source to state that the doctoral degree must be earned prior to the fellowship start/begin date."
        ),
    )

    # 7) Fellowship Duration: two or three years
    duration_node = evaluator.add_leaf(
        id="fellowship_duration",
        desc="The fellowship duration is two or three years.",
        parent=criteria_node,
        critical=True,
    )
    claim_duration = (
        f"The duration of {_prog_ref(program_name)} is two or three years."
    )
    await evaluator.verify(
        claim=claim_duration,
        node=duration_node,
        sources=sources,
        additional_instruction=(
            "Accept if the program allows 2-year or 3-year awards, or states a standard duration of two or three years. "
            "If only one fixed duration (e.g., only two years) is stated with no option for three, mark as not supported."
        ),
    )

    # 8) Research and Training Allowance
    allowance_node = evaluator.add_leaf(
        id="research_training_allowance",
        desc="The fellowship provides a research and training allowance in addition to the stipend.",
        parent=criteria_node,
        critical=True,
    )
    claim_allowance = (
        f"{_prog_ref(program_name)} provides a separate research and training allowance in addition to the stipend."
    )
    await evaluator.verify(
        claim=claim_allowance,
        node=allowance_node,
        sources=sources,
        additional_instruction=(
            "Look for a line item or section describing 'research and training allowance', 'budget for research/training', "
            "or equivalent funds provided separately from the stipend."
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
    Evaluate an answer for the NSF biology postdoctoral fellowship program identification task.
    """
    # Initialize evaluator (root is non-critical; child node will be critical)
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

    # Extract the program name and any cited source URLs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_program_name": extraction.program_name,
            "extracted_source_urls": extraction.source_urls,
        },
        info_type="extraction_summary",
        info_name="extracted_program_info",
    )

    # Build and run verification subtree
    await verify_fellowship_criteria(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()