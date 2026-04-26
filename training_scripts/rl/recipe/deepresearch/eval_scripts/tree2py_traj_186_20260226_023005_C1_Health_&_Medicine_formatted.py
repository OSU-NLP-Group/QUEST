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
TASK_ID = "wegovy_pill_timing"
TASK_DESCRIPTION = "According to the official Wegovy pill administration guidelines, what are the specific timing requirements that must be followed when taking the Wegovy pill each day?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RequirementEvidence(BaseModel):
    """
    Represents one requirement statement and its cited sources from the agent's answer.
    """
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WegovyTimingExtraction(BaseModel):
    """
    Two key timing requirements to extract from the answer:
    - morning_empty_stomach: morning + empty stomach + water limit (up to 4 ounces)
    - waiting_period: wait at least 30 minutes before eating/drinking after pill
    """
    morning_empty_stomach: Optional[RequirementEvidence] = None
    waiting_period: Optional[RequirementEvidence] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_timing_requirements() -> str:
    return """
    Extract the specific daily timing requirements for taking the Wegovy pill as stated in the answer. There are two requirements of interest:

    1) morning_empty_stomach:
       - statement: Copy the exact sentence(s) from the answer that state the pill must be taken in the morning on an empty stomach AND that it should be taken with water only, with a volume limit of no more than 4 ounces (≈120 mL). If the answer uses equivalent phrasing (e.g., "first thing in the morning", "empty stomach", "plain water", "up to 4 oz", "no more than 4 ounces", "approximately 120 mL"), include those exact sentences as the statement.
       - sources: A list of all URLs cited in the answer that support this morning/empty-stomach/water-limit instruction.

    2) waiting_period:
       - statement: Copy the exact sentence(s) from the answer that state a person must wait at least 30 minutes after taking the pill before eating food or drinking beverages (including coffee). Equivalent phrasing such as "≥30 minutes", "at least half an hour", "wait 30 min before breakfast or liquids" is acceptable; copy the exact text from the answer.
       - sources: A list of all URLs cited in the answer that support the ≥30-minute waiting requirement.

    Rules:
    - Do NOT invent or infer anything; only extract what is explicitly present in the answer.
    - If a required statement is not present in the answer, set the corresponding 'statement' to null.
    - If no supporting URLs are provided in the answer for a requirement, return an empty list for 'sources'.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_morning_requirement(
    evaluator: Evaluator,
    parent_node,
    extracted: WegovyTimingExtraction,
) -> None:
    """
    Build and verify the subtree for the morning/empty-stomach/water-limit requirement.
    """
    info = extracted.morning_empty_stomach or RequirementEvidence()

    # Create a critical sequential node under the critical root
    morning_main = evaluator.add_sequential(
        id="Morning_Empty_Stomach_Main",
        desc="Morning/empty stomach/water-limit requirement verification pipeline",
        parent=parent_node,
        critical=True,
    )

    # 1) Existence check: statement provided in the answer
    has_statement = bool(info.statement and info.statement.strip())
    evaluator.add_custom_node(
        result=has_statement,
        id="Morning_Empty_Stomach_Text_Provided",
        desc="The answer provides a statement about morning intake on an empty stomach with water limit",
        parent=morning_main,
        critical=True,
    )

    # 2) Content correctness: does the provided statement actually include all elements?
    stated_leaf = evaluator.add_leaf(
        id="Morning_Empty_Stomach_Requirement",
        desc="The answer states that the Wegovy pill must be taken on an empty stomach in the morning with water (up to 4 ounces)",
        parent=morning_main,
        critical=True,
    )

    claim_stmt = f"""
    Judge whether the following excerpt from the answer clearly and explicitly conveys ALL of the following:
    - It must be taken in the morning (e.g., "first thing in the morning").
    - It must be taken on an empty stomach.
    - It must be taken with water only, and the water volume must be no more than 4 ounces (≈120 mL).
    Excerpt:
    {info.statement or ""}
    """
    await evaluator.verify(
        claim=claim_stmt,
        node=stated_leaf,
        additional_instruction=(
            "Allow minor phrasing variants (e.g., 'first thing in the morning', 'plain water', "
            "'up to 4 oz', 'no more than 4 ounces', 'approximately 120 mL'). "
            "If any one of the three required elements is missing in the excerpt, mark as Incorrect."
        ),
    )

    # 3) Sources existence check (critical for source-grounding)
    has_sources = bool(info.sources and len(info.sources) > 0)
    evaluator.add_custom_node(
        result=has_sources,
        id="Morning_Empty_Stomach_Sources_Provided",
        desc="Supporting URL sources for the morning/empty-stomach/water-limit requirement are provided",
        parent=morning_main,
        critical=True,
    )

    # 4) Official guideline support by cited sources
    supported_leaf = evaluator.add_leaf(
        id="Morning_Empty_Stomach_Guideline_Supported",
        desc="The cited sources support the morning/empty-stomach with ≤4 oz water requirement",
        parent=morning_main,
        critical=True,
    )

    support_claim = (
        "The webpage(s) explicitly state that the pill must be taken first thing in the morning "
        "on an empty stomach, with plain water only, and the water volume must be no more than 4 ounces (≈120 mL)."
    )
    await evaluator.verify(
        claim=support_claim,
        node=supported_leaf,
        sources=info.sources,
        additional_instruction=(
            "Confirm the page content clearly lists ALL of the following in one place or across sentences: "
            "morning intake, empty stomach, and water-only with ≤4 ounces (≈120 mL). "
            "Accept official patient guides or prescribing information pages. "
            "If the provided webpage does not clearly state ALL these elements, mark as Not Supported."
        ),
    )


async def verify_waiting_requirement(
    evaluator: Evaluator,
    parent_node,
    extracted: WegovyTimingExtraction,
) -> None:
    """
    Build and verify the subtree for the ≥30-minute waiting requirement after taking the pill.
    """
    info = extracted.waiting_period or RequirementEvidence()

    # Create a critical sequential node under the critical root
    waiting_main = evaluator.add_sequential(
        id="Waiting_Period_Main",
        desc="Waiting period (≥30 minutes before eating/drinking) verification pipeline",
        parent=parent_node,
        critical=True,
    )

    # 1) Existence check: statement provided in the answer
    has_statement = bool(info.statement and info.statement.strip())
    evaluator.add_custom_node(
        result=has_statement,
        id="Waiting_Period_Text_Provided",
        desc="The answer provides a statement about waiting at least 30 minutes after taking the pill before eating/drinking",
        parent=waiting_main,
        critical=True,
    )

    # 2) Content correctness: does the provided statement include ≥30 minutes wait before food/drinks?
    stated_leaf = evaluator.add_leaf(
        id="Waiting_Period_Requirement",
        desc="The answer states that a person must wait at least 30 minutes after taking the pill before eating food or drinking beverages",
        parent=waiting_main,
        critical=True,
    )

    claim_stmt = f"""
    Judge whether the following excerpt from the answer clearly and explicitly conveys that
    the person must wait at least 30 minutes after taking the pill before eating food or drinking beverages:
    Excerpt:
    {info.statement or ""}
    """
    await evaluator.verify(
        claim=claim_stmt,
        node=stated_leaf,
        additional_instruction=(
            "Allow minor variants such as '≥30 minutes', 'at least half an hour', or 'wait 30 min before breakfast or liquids'. "
            "If the minimum 30-minute wait is not clearly stated, mark as Incorrect."
        ),
    )

    # 3) Sources existence check (critical for source-grounding)
    has_sources = bool(info.sources and len(info.sources) > 0)
    evaluator.add_custom_node(
        result=has_sources,
        id="Waiting_Period_Sources_Provided",
        desc="Supporting URL sources for the ≥30-minute waiting requirement are provided",
        parent=waiting_main,
        critical=True,
    )

    # 4) Official guideline support by cited sources
    supported_leaf = evaluator.add_leaf(
        id="Waiting_Period_Guideline_Supported",
        desc="The cited sources support waiting ≥30 minutes after the pill before eating/drinking",
        parent=waiting_main,
        critical=True,
    )

    support_claim = (
        "The webpage(s) explicitly state that after taking the pill, you must wait at least 30 minutes before "
        "eating food or drinking beverages."
    )
    await evaluator.verify(
        claim=support_claim,
        node=supported_leaf,
        sources=info.sources,
        additional_instruction=(
            "Confirm the page clearly indicates a minimum 30-minute waiting interval before food or beverages. "
            "Accept official patient guides or prescribing information pages. "
            "If the provided webpage does not clearly state this, mark as Not Supported."
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
    Evaluate an answer for the Wegovy pill daily timing requirements.
    """
    # Initialize evaluator
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

    # Add critical root node reflecting the rubric root
    timing_root = evaluator.add_parallel(
        id="Wegovy_Pill_Administration_Timing",
        desc="The answer correctly identifies the timing requirements for taking the Wegovy pill",
        parent=root,
        critical=True,
    )

    # Extract timing requirement statements and their sources from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_timing_requirements(),
        template_class=WegovyTimingExtraction,
        extraction_name="wegovy_timing_requirements",
    )

    # Build verification subtrees
    await verify_morning_requirement(evaluator, timing_root, extraction)
    await verify_waiting_requirement(evaluator, timing_root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()