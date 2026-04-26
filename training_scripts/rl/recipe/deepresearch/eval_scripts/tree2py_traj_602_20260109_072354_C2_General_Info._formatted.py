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
TASK_ID = "ohio_veto_override"
TASK_DESCRIPTION = """
According to the Ohio Constitution, what is the minimum number of votes required in the Ohio House of Representatives and the Ohio Senate, respectively, to override a governor's veto?
"""

# Ground truth context (for summary only; not used to bias verification)
GROUND_TRUTH_CONTEXT = {
    "constitutional_rule": "Three-fifths (3/5) of the members elected to each house are required to override a governor's veto.",
    "ohio_house_size": 99,
    "ohio_senate_size": 33,
    "expected_house_min_votes": 60,  # ceil(3/5 * 99) = 60
    "expected_senate_min_votes": 20  # ceil(3/5 * 33) = 20
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VetoOverrideExtraction(BaseModel):
    """
    Extracted information from the agent's answer regarding Ohio veto override requirements.
    """
    house_min_votes: Optional[str] = None
    senate_min_votes: Optional[str] = None
    constitutional_rule_text: Optional[str] = None  # e.g., "three-fifths of members elected"
    house_size: Optional[str] = None                # if the answer mentions House chamber size
    senate_size: Optional[str] = None               # if the answer mentions Senate chamber size
    references: List[str] = Field(default_factory=list)  # URLs cited to support the claims


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_veto_override() -> str:
    return """
    Extract the specific information stated in the answer about Ohio's veto override thresholds.

    You must return a JSON object with these fields:
    - house_min_votes: The minimum number of votes stated for the Ohio House of Representatives to override a governor's veto (as it appears in the answer; capture just the number if possible, e.g., "60").
    - senate_min_votes: The minimum number of votes stated for the Ohio Senate to override a governor's veto (as it appears in the answer; capture just the number if possible, e.g., "20").
    - constitutional_rule_text: The rule text the answer references (e.g., "three-fifths of the members elected to each house") if present; otherwise null.
    - house_size: The Ohio House chamber size mentioned in the answer (e.g., "99") if present; otherwise null.
    - senate_size: The Ohio Senate chamber size mentioned in the answer (e.g., "33") if present; otherwise null.
    - references: An array of all URLs explicitly included in the answer as supporting citations for these veto-override requirements. Include each URL exactly once. If no URLs are present, return an empty array.

    IMPORTANT:
    - Only extract what is explicitly stated in the answer. Do not infer or invent numbers or URLs.
    - For references, include only actual URLs (plain or Markdown links). If the answer mentions a source by name without a URL, exclude it from the references array.
    - If any field is not present in the answer, set it to null (or empty array for references).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty_text(v: Optional[str]) -> bool:
    return bool(v) and bool(v.strip())


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    extracted: VetoOverrideExtraction
) -> None:
    """
    Build the verification tree per rubric and run the verifications.
    """
    # Create the critical parallel parent node per rubric
    req_node = evaluator.add_parallel(
        id="ohio_veto_override_requirements",
        desc="Provides the minimum number of votes required in both the Ohio House of Representatives and the Ohio Senate to override a governor's veto, with supporting reference(s).",
        parent=root_node,
        critical=True
    )

    # Existence checks (critical gating)
    evaluator.add_custom_node(
        result=_is_nonempty_text(extracted.house_min_votes),
        id="house_count_provided",
        desc="House minimum vote count is provided in the answer",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_text(extracted.senate_min_votes),
        id="senate_count_provided",
        desc="Senate minimum vote count is provided in the answer",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(extracted.references is not None and len(extracted.references) > 0),
        id="references_provided",
        desc="At least one supporting reference URL is provided in the answer",
        parent=req_node,
        critical=True
    )

    # Leaf: House vote count correctness
    house_leaf = evaluator.add_leaf(
        id="house_vote_count",
        desc="States the correct minimum number of votes required in the Ohio House of Representatives to override a governor's veto (per the Ohio Constitution rule and chamber size).",
        parent=req_node,
        critical=True
    )
    house_claim = f"The minimum number of votes required in the Ohio House of Representatives to override a governor's veto is {extracted.house_min_votes}."
    house_instruction = (
        "Judge whether this number is correct under Ohio's constitutional standard for veto overrides. "
        "The Ohio Constitution requires at least three-fifths (3/5) of the members elected to each house to override a gubernatorial veto. "
        "The Ohio House has 99 members; thus the minimum is ceil(3/5 * 99) = 60. "
        "Accept minor formatting (e.g., '60 votes' vs '60'). If the stated number matches 60, mark correct; otherwise incorrect."
    )
    # Use simple verification (logical check), gated by existence
    await evaluator.verify(
        claim=house_claim,
        node=house_leaf,
        sources=None,
        additional_instruction=house_instruction
    )

    # Leaf: Senate vote count correctness
    senate_leaf = evaluator.add_leaf(
        id="senate_vote_count",
        desc="States the correct minimum number of votes required in the Ohio Senate to override a governor's veto (per the Ohio Constitution rule and chamber size).",
        parent=req_node,
        critical=True
    )
    senate_claim = f"The minimum number of votes required in the Ohio Senate to override a governor's veto is {extracted.senate_min_votes}."
    senate_instruction = (
        "Judge whether this number is correct under Ohio's constitutional standard for veto overrides. "
        "The Ohio Constitution requires at least three-fifths (3/5) of the members elected to each house to override a gubernatorial veto. "
        "The Ohio Senate has 33 members; thus the minimum is ceil(3/5 * 33) = 20. "
        "Accept minor formatting (e.g., '20 votes' vs '20'). If the stated number matches 20, mark correct; otherwise incorrect."
    )
    # Use simple verification (logical check), gated by existence
    await evaluator.verify(
        claim=senate_claim,
        node=senate_leaf,
        sources=None,
        additional_instruction=senate_instruction
    )

    # Leaf: Supporting references confirm constitutional rule
    refs_leaf = evaluator.add_leaf(
        id="supporting_references",
        desc="Provides reference(s) (e.g., citation or URL) to the Ohio Constitution or an official legislative source that support the stated veto-override vote requirements for both chambers.",
        parent=req_node,
        critical=True
    )
    refs_claim = (
        "The provided reference(s) include an official Ohio government or legislative source that explicitly states the veto-override requirement: "
        "at least three-fifths of the members elected to each house must vote to override the governor's veto."
    )
    refs_instruction = (
        "Verify that at least one of the provided URLs is an official legal or legislative source that clearly states the 3/5 requirement applying to each chamber. "
        "Accept pages such as the Ohio Constitution (e.g., Article II, Section 16) or official legislative sites (e.g., legislature.ohio.gov, codes.ohio.gov). "
        "Mark supported only if the text on the page explicitly states the three-fifths requirement for veto overrides."
    )
    await evaluator.verify(
        claim=refs_claim,
        node=refs_leaf,
        sources=extracted.references,
        additional_instruction=refs_instruction
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
    Evaluate an answer for the Ohio veto override question.
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_veto_override(),
        template_class=VetoOverrideExtraction,
        extraction_name="veto_override_extraction",
    )

    # Add ground truth context (for summary only)
    evaluator.add_ground_truth({
        "constitutional_rule": GROUND_TRUTH_CONTEXT["constitutional_rule"],
        "ohio_house_size": GROUND_TRUTH_CONTEXT["ohio_house_size"],
        "ohio_senate_size": GROUND_TRUTH_CONTEXT["ohio_senate_size"],
        "expected_house_min_votes": GROUND_TRUTH_CONTEXT["expected_house_min_votes"],
        "expected_senate_min_votes": GROUND_TRUTH_CONTEXT["expected_senate_min_votes"]
    })

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()