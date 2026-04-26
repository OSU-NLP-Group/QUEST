import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ssa_fra_1960_plus_2026"
TASK_DESCRIPTION = "According to the Social Security Administration, what is the full retirement age for individuals born in 1960 and later, as of 2026?"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SSAFRAExtraction(BaseModel):
    """
    Extract key statements and sources from the answer related to the SSA full retirement age
    for individuals born in 1960 and later, in a 2026 context.
    """
    full_retirement_age_text: Optional[str] = None
    birth_year_qualification_text: Optional[str] = None
    context_2026_text: Optional[str] = None
    ssa_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ssa_fra() -> str:
    return """
    Extract the following items exactly as stated in the answer text:

    1) full_retirement_age_text:
       - The exact phrase where the answer states the SSA full retirement age (FRA) for the relevant group.
       - Prefer the phrasing that most clearly states the specific age (e.g., "full retirement age is 67").
       - If not stated, return null.

    2) birth_year_qualification_text:
       - The exact phrase that ties the FRA to individuals "born in 1960 and later" (or equivalent wording such as
         "born in 1960 or later", "born in or after 1960", "1960 onward(s)").
       - If the answer does not specify this birth-year qualification, return null.

    3) context_2026_text:
       - The exact words that tie the statement to the 2026 context (e.g., "as of 2026", "in 2026", "2026 context").
       - If the answer does not clearly reference 2026, return null.

    4) ssa_urls:
       - List all URLs in the answer that point to official Social Security Administration resources.
         These typically have domains like "ssa.gov" (including subdomains) or official SSA publication links.
       - Only include URLs explicitly present in the answer.

    5) other_urls:
       - List all other URLs mentioned in the answer that are NOT from "ssa.gov".
       - Only include URLs explicitly present in the answer.

    Notes:
    - Do not invent content. Extract verbatim text when asked for "text".
    - For URLs, include the full link. If a URL is present without protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: SSAFRAExtraction
) -> None:
    """
    Build the verification tree according to the rubric:
    - Social_Security_Full_Retirement_Age (critical, parallel)
      - Retirement_Age_Value (critical leaf)
      - Birth_Year_Qualification (critical leaf)
      - As_of_2026_Context (critical leaf)
      - SSA_Source_Reference (critical leaf)
    """
    # Create the main critical node under the root
    main_node = evaluator.add_parallel(
        id="Social_Security_Full_Retirement_Age",
        desc="Verify the full retirement age for people born in 1960 and later according to the Social Security Administration, as framed for 2026.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Retirement_Age_Value
    node_ret_age = evaluator.add_leaf(
        id="Retirement_Age_Value",
        desc="States the full retirement age as 67 years old.",
        parent=main_node,
        critical=True
    )
    # Use simple verification against the answer text
    # We do not force source evidence here because the rubric only requires that the answer states "67"
    await evaluator.verify(
        claim="The answer explicitly states that the full retirement age is 67 years old.",
        node=node_ret_age,
        additional_instruction="Check the answer text for language like 'full retirement age is 67', 'FRA is 67', or equivalent phrasing. Focus on the answer text only."
    )

    # 2) Birth_Year_Qualification
    node_birth_year = evaluator.add_leaf(
        id="Birth_Year_Qualification",
        desc="Specifies that this full retirement age applies to individuals born in 1960 and later.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly specifies that this full retirement age applies to individuals born in 1960 and later.",
        node=node_birth_year,
        additional_instruction="Accept equivalent phrasing such as 'born 1960 or later', 'born in or after 1960', or '1960 onward(s)'. Focus on the answer text only."
    )

    # 3) As_of_2026_Context
    node_2026 = evaluator.add_leaf(
        id="As_of_2026_Context",
        desc="Explicitly ties the statement to the 2026 context (e.g., mentions 'as of 2026' or an equivalent framing clearly referencing 2026).",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly ties the statement to the 2026 context (e.g., 'as of 2026' or an equivalent).",
        node=node_2026,
        additional_instruction="Look for a clear mention of 2026 that frames the fact in that time context."
    )

    # 4) SSA_Source_Reference
    node_ssa_source = evaluator.add_leaf(
        id="SSA_Source_Reference",
        desc="Cites an official Social Security Administration website (e.g., ssa.gov) or an official SSA publication as the source.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer cites an official Social Security Administration website (e.g., a link to ssa.gov) or an official SSA publication as the source.",
        node=node_ssa_source,
        additional_instruction="It's sufficient if the answer includes a link to ssa.gov or clearly references an official SSA publication. A plain-text citation without a URL can count if it unambiguously identifies an official SSA source."
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
    Evaluate an answer for the SSA full retirement age (1960+ as of 2026) task using the Mind2Web2 evaluation framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates all checks in parallel
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

    # Extraction (recorded for transparency; verification primarily relies on answer text per rubric)
    extraction = await evaluator.extract(
        prompt=prompt_extract_ssa_fra(),
        template_class=SSAFRAExtraction,
        extraction_name="ssa_fra_extraction"
    )

    # Build and run verification according to rubric
    await build_verification_tree(evaluator, extraction)

    # Return structured result
    return evaluator.get_summary()