import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "real_estate_salesperson_max_hours"
TASK_DESCRIPTION = """
Which U.S. state requires the most pre-licensing education hours for obtaining a real estate salesperson license?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RealEstateRequirementExtraction(BaseModel):
    """
    Structured extraction from the agent's answer.
    """
    state: Optional[str] = None  # The U.S. state named as having the highest pre-licensing requirement (salesperson).
    hours: Optional[str] = None  # The total hours claimed (string as written, e.g., "180" or "180 hours").
    license_scope: Optional[str] = None  # Exact wording describing the license scope/type (e.g., "salesperson", "sales agent", "broker").
    education_type: Optional[str] = None  # Exact wording describing the education type (e.g., "pre-licensing", "qualifying education", "continuing education").
    claim_sentence: Optional[str] = None  # The sentence that claims which state requires the most hours.
    sources: List[str] = Field(default_factory=list)  # All URLs cited in the answer.


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the key information the answer presents regarding the U.S. state that requires the most pre-licensing education hours for a real estate salesperson (sales agent) license.

    Return a JSON object with these fields:
    - state: The U.S. state named as having the highest pre-licensing education-hour requirement (for salesperson/sales agent).
    - hours: The total hours the answer claims are required for that state (as plain text, do not perform arithmetic or normalization; e.g., "180" or "180 hours").
    - license_scope: The exact wording in the answer describing the relevant license type/scope (e.g., "salesperson", "sales agent", "broker", "associate broker").
    - education_type: The exact wording in the answer describing the type of education (e.g., "pre-licensing", "pre-license", "qualifying education", "post-licensing", "continuing education").
    - claim_sentence: The sentence in the answer (verbatim) that explicitly claims which state has the most hours for a real estate salesperson pre-licensing requirement. If none is explicit, return null.
    - sources: An array of all URLs cited in the answer that are meant to support the claim. Extract explicit URLs only (including those in markdown links).

    Important:
    - Only extract information explicitly present in the answer.
    - If any item is missing, set it to null or an empty list for sources.
    - Do not add or infer any URLs or values that are not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _format_hours_for_claim(hours_text: Optional[str]) -> str:
    """
    Normalize the hours text for embedding into a claim sentence in a readable way.
    """
    if not hours_text:
        return "the stated number of"
    ht = hours_text.strip()
    lower = ht.lower()
    # If already contains 'hour'
    if "hour" in lower:
        return ht
    # If is numeric-like, append 'hours'
    return f"{ht} hours"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extracted: RealEstateRequirementExtraction) -> None:
    """
    Build the verification tree according to the rubric and execute verification.
    """
    # Add critical parallel node "correct_answer"
    correct_node = evaluator.add_parallel(
        id="correct_answer",
        desc="Identifies the U.S. state with the highest pre-licensing education-hour requirement for a real estate salesperson (not broker) license, and provides the required hours with official/current verification.",
        parent=evaluator.root,
        critical=True
    )

    # Prepare common variables
    state = (extracted.state or "").strip()
    hours_text = (extracted.hours or "").strip()
    hours_for_claim = _format_hours_for_claim(hours_text)
    urls: List[str] = extracted.sources or []

    # Create leaf nodes (all critical under a critical parent)
    node_state = evaluator.add_leaf(
        id="correct_state_identified",
        desc="Names the correct U.S. state that has the highest pre-licensing education-hour requirement for a real estate salesperson license.",
        parent=correct_node,
        critical=True
    )
    node_hours = evaluator.add_leaf(
        id="correct_hour_total",
        desc="States the correct total number of required pre-licensing education hours for that state's real estate salesperson license.",
        parent=correct_node,
        critical=True
    )
    node_scope = evaluator.add_leaf(
        id="salesperson_not_broker_scope",
        desc="Makes clear the requirement cited applies to a real estate salesperson/sales agent license (not a broker license).",
        parent=correct_node,
        critical=True
    )
    node_prelicense = evaluator.add_leaf(
        id="prelicensing_not_other_education",
        desc="Makes clear the hours cited are for pre-licensing education (not continuing education or post-licensing).",
        parent=correct_node,
        critical=True
    )
    node_official = evaluator.add_leaf(
        id="official_current_verifiable_source",
        desc="Provides citation(s) to an official state real estate regulatory authority source (and/or similarly authoritative official source) that are current/active and allow verification of the stated pre-licensing salesperson education-hour requirement (e.g., URL and page/title).",
        parent=correct_node,
        critical=True
    )

    # Build claims and additional instructions
    claim_state = (
        f"Among U.S. states that license real estate salespersons/sales agents, {state} requires the most "
        f"pre-licensing education hours for a salesperson/sales agent license."
        if state else
        "The answer identifies a specific U.S. state as requiring the most pre-licensing education hours for a salesperson license."
    )
    add_ins_state = (
        "Use the provided URLs to confirm that the named state indeed has the highest pre-licensing hour requirement "
        "for a real estate salesperson/sales agent license. Pass if at least one credible page explicitly states this "
        "(e.g., 'most hours' or 'highest requirement') or presents a nationwide comparison list clearly showing the named "
        "state's number equals or exceeds every other state's. If the sources only discuss broker licenses or do not "
        "establish 'highest', fail."
    )

    claim_hours = (
        f"In {state}, the total required pre-licensing education for a real estate salesperson/sales agent is {hours_for_claim}."
        if state or hours_text else
        "The answer states a specific total number of pre-licensing education hours required for a real estate salesperson/sales agent."
    )
    add_ins_hours = (
        "Verify the total hour requirement as stated on the provided pages. Prefer official state regulatory pages if present. "
        "If the program is split into multiple courses (e.g., six courses totaling 180 hours), accept the correctly summed total. "
        "Do not accept broker-only requirements."
    )

    claim_scope = (
        f"The cited requirement applies specifically to the real estate salesperson/sales agent license in {state}, not the broker license."
        if state else
        "The cited requirement applies specifically to the real estate salesperson/sales agent license, not the broker license."
    )
    add_ins_scope = (
        "Pass only if the page(s) indicate 'salesperson' or 'sales agent' (or an equivalent entry-level salesperson license) "
        "for the requirement. If the state licenses only brokers and the requirement is broker-only, fail. "
        "Treat 'sales agent' as equivalent to 'salesperson' (e.g., Texas 'Sales Agent')."
    )

    claim_prelicense = (
        f"The stated {hours_for_claim} refer specifically to pre-licensing (qualifying) education for the salesperson/sales agent license in {state}, not post-licensing or continuing education."
        if state or hours_text else
        "The hours cited refer specifically to pre-licensing (qualifying) education for the salesperson/sales agent license, not post-licensing or continuing education."
    )
    add_ins_prelicense = (
        "Pass only if the page(s) clearly indicate 'pre-licensing', 'pre-license', 'qualifying education', or equivalent wording "
        "that unambiguously means education required before initial licensure. If the hours are for post-licensing or continuing education, fail."
    )

    claim_official = (
        f"This webpage is an official state real estate regulatory authority page for {state} that clearly states the "
        f"pre-licensing education hour requirement for a real estate salesperson/sales agent license."
        if state else
        "This webpage is an official state real estate regulatory authority page that clearly states the pre-licensing "
        "education hour requirement for a real estate salesperson/sales agent license."
    )
    add_ins_official = (
        "Pass if at least one provided URL is an official state regulatory source (e.g., .gov domain, state real estate commission/department) "
        "that is active/reachable and explicitly states the pre-licensing salesperson/sales agent hour requirement. "
        "Do not pass if all sources are only vendor/education provider/aggregator pages or dead links."
    )

    # Run verifications (in parallel)
    await evaluator.batch_verify([
        (claim_state, urls, node_state, add_ins_state),
        (claim_hours, urls, node_hours, add_ins_hours),
        (claim_scope, urls, node_scope, add_ins_scope),
        (claim_prelicense, urls, node_prelicense, add_ins_prelicense),
        (claim_official, urls, node_official, add_ins_official),
    ])


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
    Evaluate an answer for the 'most pre-licensing hours (salesperson)' question.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RealEstateRequirementExtraction,
        extraction_name="answer_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()