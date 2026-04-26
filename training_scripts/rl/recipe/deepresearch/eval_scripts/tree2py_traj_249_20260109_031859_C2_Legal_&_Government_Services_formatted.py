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
TASK_ID = "arkansas_veto_override_requirements"
TASK_DESCRIPTION = (
    "In the Arkansas General Assembly, what is the minimum number of votes required in each chamber "
    "(House of Representatives and Senate) to successfully override a gubernatorial veto? Provide the specific "
    "vote count for each chamber along with reference URLs supporting your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GlobalOverrideConstraints(BaseModel):
    """Global constraints as stated in the answer and related constitutional references."""
    both_chambers_must_override: Optional[bool] = None
    majority_of_elected_members_standard: Optional[bool] = None
    constitutional_basis_stated: Optional[bool] = None
    constitution_reference_urls: List[str] = Field(default_factory=list)


class ChamberRequirement(BaseModel):
    """Per-chamber details from the answer including counts and supporting URLs."""
    member_count: Optional[str] = None
    vote_count_required: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ArkansasVetoOverrideExtraction(BaseModel):
    """Complete extraction for Arkansas veto override requirements."""
    global_constraints: Optional[GlobalOverrideConstraints] = None
    house: Optional[ChamberRequirement] = None
    senate: Optional[ChamberRequirement] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_override_info() -> str:
    return """
    Extract structured information about Arkansas gubernatorial veto override requirements from the provided answer.

    Return a JSON object with the following structure:
    {
      "global_constraints": {
        "both_chambers_must_override": boolean | null,
        "majority_of_elected_members_standard": boolean | null,
        "constitutional_basis_stated": boolean | null,
        "constitution_reference_urls": string[]  // Only URLs explicitly present in the answer that support the constitutional basis for veto override rules
      },
      "house": {
        "member_count": string | null,           // Prefer digits or a clear phrase like "100" or "100 members"
        "vote_count_required": string | null,    // Prefer digits like "51" or a clear phrase like "51 votes"
        "reference_urls": string[]               // Only URLs explicitly present in the answer that support the House veto override threshold
      },
      "senate": {
        "member_count": string | null,           // Prefer digits or a clear phrase like "35" or "35 members"
        "vote_count_required": string | null,    // Prefer digits like "18" or a clear phrase like "18 votes"
        "reference_urls": string[]               // Only URLs explicitly present in the answer that support the Senate veto override threshold
      }
    }

    Notes and rules:
    - The boolean fields under global_constraints should be true if the answer explicitly states the respective facts; false if the answer explicitly contradicts them; null if the answer does not address them.
    - For counts and vote thresholds, extract the exact values or phrases as they appear in the answer; do not infer or calculate new values.
    - For URL fields, extract only valid URLs explicitly present in the answer. Include full URLs; if protocol is missing, prepend http://. Do not invent URLs.
    - If any data item is missing from the answer, set its value to null (or an empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_urls(*lists: Optional[List[str]]) -> List[str]:
    """Combine multiple URL lists into a unique, clean list."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url or not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: ArkansasVetoOverrideExtraction
) -> None:
    """
    Build verification tree according to the rubric and run verifications.
    """
    # Create critical root node for this task under evaluator root
    root_main = evaluator.add_parallel(
        id="Arkansas_Veto_Override_Requirements",
        desc="Identifies the minimum votes needed in each Arkansas General Assembly chamber to override a gubernatorial veto, with supporting references, while satisfying all stated constraints",
        parent=evaluator.root,
        critical=True
    )

    # Prepare convenience variables
    global_info = extraction.global_constraints or GlobalOverrideConstraints()
    house_info = extraction.house or ChamberRequirement()
    senate_info = extraction.senate or ChamberRequirement()

    # ---------------------- Global Override Constraints ---------------------- #
    global_node = evaluator.add_parallel(
        id="Global_Override_Constraints",
        desc="Satisfies global constraints about how veto overrides work in Arkansas",
        parent=root_main,
        critical=True
    )

    # Both chambers must override (answer should state it)
    both_chambers_leaf = evaluator.add_leaf(
        id="Both_Chambers_Must_Override",
        desc="States that both the House and the Senate must vote to override for the override to be successful",
        parent=global_node,
        critical=True
    )
    both_claim = (
        "The answer explicitly states that both the House and the Senate must vote to override for the override to be successful in Arkansas."
    )
    await evaluator.verify(
        claim=both_claim,
        node=both_chambers_leaf,
        additional_instruction="Judge this against the answer text only. Pass only if the answer clearly asserts the requirement that both chambers must vote to override."
    )

    # Majority of elected members standard (answer should state it)
    majority_std_leaf = evaluator.add_leaf(
        id="Majority_of_Elected_Members_Standard",
        desc="States that the override threshold is a majority of elected members (not a supermajority and not merely a majority of those present)",
        parent=global_node,
        critical=True
    )
    majority_claim = (
        "The answer explicitly states that Arkansas's veto override threshold is a majority of the members elected to each chamber, "
        "not merely a majority of those present and not a supermajority."
    )
    await evaluator.verify(
        claim=majority_claim,
        node=majority_std_leaf,
        additional_instruction="Judge against the answer text only. Look for a clear statement regarding 'majority of elected members' as the threshold."
    )

    # Constitutional basis stated (answer should state it)
    const_basis_leaf = evaluator.add_leaf(
        id="Constitutional_Basis_Stated",
        desc="States that the veto override power/standard is specified in the Arkansas state constitution",
        parent=global_node,
        critical=True
    )
    const_basis_claim = "The answer explicitly states that the veto override power or standard is specified in the Arkansas state constitution."
    await evaluator.verify(
        claim=const_basis_claim,
        node=const_basis_leaf,
        additional_instruction="Judge against the answer text only. Pass only if the answer clearly mentions the Arkansas state constitution as the source of the override rule."
    )

    # Constitution reference URL (web verification)
    const_ref_leaf = evaluator.add_leaf(
        id="Constitution_Reference_URL",
        desc="Provides at least one valid reference URL that supports the constitutional basis for the veto override rule",
        parent=global_node,
        critical=True
    )
    const_ref_claim = (
        "These referenced webpages explicitly support that the Arkansas Constitution sets the veto override standard as a majority of the members elected to each chamber."
    )
    await evaluator.verify(
        claim=const_ref_claim,
        node=const_ref_leaf,
        sources=global_info.constitution_reference_urls,
        additional_instruction=(
            "Pass only if at least one provided URL clearly supports the constitutional basis for the veto override rule "
            "(e.g., text from the Arkansas Constitution or an official Arkansas constitutional reference page)."
        )
    )

    # ---------------------- House Requirements ---------------------- #
    house_node = evaluator.add_parallel(
        id="House_Override_Requirement",
        desc="Provides the House-specific details required by the question/constraints",
        parent=root_main,
        critical=True
    )

    # House member count stated (answer should state "100 members")
    house_members_leaf = evaluator.add_leaf(
        id="House_Member_Count",
        desc="States that the Arkansas House of Representatives has 100 members",
        parent=house_node,
        critical=True
    )
    house_members_claim = "The answer explicitly states that the Arkansas House of Representatives has 100 members."
    await evaluator.verify(
        claim=house_members_claim,
        node=house_members_leaf,
        additional_instruction="Judge against the answer text only. Look for an explicit statement of '100 members' for the Arkansas House."
    )

    # House vote count minimum (answer should state "51 votes")
    house_votes_leaf = evaluator.add_leaf(
        id="House_Vote_Count",
        desc="States the minimum House vote count required to override a veto is at least 51 votes",
        parent=house_node,
        critical=True
    )
    house_votes_claim = "The answer explicitly states that the minimum House votes required to override a gubernatorial veto in Arkansas is 51 votes."
    await evaluator.verify(
        claim=house_votes_claim,
        node=house_votes_leaf,
        additional_instruction=(
            "Judge against the answer text only. Pass only if the answer clearly provides the specific threshold '51 votes' "
            "for the Arkansas House veto override."
        ),
        extra_prerequisites=[house_members_leaf, majority_std_leaf]
    )

    # House reference URL(s) support the threshold
    house_urls_leaf = evaluator.add_leaf(
        id="House_Reference_URL",
        desc="Provides a valid reference URL supporting the House vote requirement (may be the same as the constitutional reference if it explicitly supports the House threshold)",
        parent=house_node,
        critical=True
    )
    house_support_urls = combine_urls(house_info.reference_urls, global_info.constitution_reference_urls)
    house_urls_claim = (
        "These referenced webpages support that the House veto override threshold is 51 votes (i.e., a majority of the 100 members elected)."
    )
    await evaluator.verify(
        claim=house_urls_claim,
        node=house_urls_leaf,
        sources=house_support_urls,
        additional_instruction=(
            "Pass only if at least one provided URL clearly supports the House threshold. "
            "Explicit evidence should indicate either the majority-of-elected-members rule along with the House member count (100), "
            "or directly state the 51-vote threshold."
        )
    )

    # ---------------------- Senate Requirements ---------------------- #
    senate_node = evaluator.add_parallel(
        id="Senate_Override_Requirement",
        desc="Provides the Senate-specific details required by the question/constraints",
        parent=root_main,
        critical=True
    )

    # Senate member count stated (answer should state "35 members")
    senate_members_leaf = evaluator.add_leaf(
        id="Senate_Member_Count",
        desc="States that the Arkansas Senate has 35 members",
        parent=senate_node,
        critical=True
    )
    senate_members_claim = "The answer explicitly states that the Arkansas Senate has 35 members."
    await evaluator.verify(
        claim=senate_members_claim,
        node=senate_members_leaf,
        additional_instruction="Judge against the answer text only. Look for an explicit statement of '35 members' for the Arkansas Senate."
    )

    # Senate vote count minimum (answer should state "18 votes")
    senate_votes_leaf = evaluator.add_leaf(
        id="Senate_Vote_Count",
        desc="States the minimum Senate vote count required to override a veto is at least 18 votes",
        parent=senate_node,
        critical=True
    )
    senate_votes_claim = "The answer explicitly states that the minimum Senate votes required to override a gubernatorial veto in Arkansas is 18 votes."
    await evaluator.verify(
        claim=senate_votes_claim,
        node=senate_votes_leaf,
        additional_instruction=(
            "Judge against the answer text only. Pass only if the answer clearly provides the specific threshold '18 votes' "
            "for the Arkansas Senate veto override."
        ),
        extra_prerequisites=[senate_members_leaf, majority_std_leaf]
    )

    # Senate reference URL(s) support the threshold
    senate_urls_leaf = evaluator.add_leaf(
        id="Senate_Reference_URL",
        desc="Provides a valid reference URL supporting the Senate vote requirement (may be the same as the constitutional reference if it explicitly supports the Senate threshold)",
        parent=senate_node,
        critical=True
    )
    senate_support_urls = combine_urls(senate_info.reference_urls, global_info.constitution_reference_urls)
    senate_urls_claim = (
        "These referenced webpages support that the Senate veto override threshold is 18 votes (i.e., a majority of the 35 members elected)."
    )
    await evaluator.verify(
        claim=senate_urls_claim,
        node=senate_urls_leaf,
        sources=senate_support_urls,
        additional_instruction=(
            "Pass only if at least one provided URL clearly supports the Senate threshold. "
            "Explicit evidence should indicate either the majority-of-elected-members rule along with the Senate member count (35), "
            "or directly state the 18-vote threshold."
        )
    )

    # Record some custom info for debugging/visibility
    evaluator.add_custom_info(
        info={
            "global_constraints_extracted": (global_info.dict() if hasattr(global_info, "dict") else {}),
            "house_extracted": (house_info.dict() if hasattr(house_info, "dict") else {}),
            "senate_extracted": (senate_info.dict() if hasattr(senate_info, "dict") else {}),
        },
        info_type="extraction_snapshot",
        info_name="extraction_snapshot"
    )

    evaluator.add_ground_truth({
        "expected_house_members": "100",
        "expected_house_votes_min": "51",
        "expected_senate_members": "35",
        "expected_senate_votes_min": "18",
        "expected_threshold_definition": "Majority of members elected to each chamber; both chambers must override; basis in Arkansas Constitution."
    }, gt_type="expected_values")


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for Arkansas veto override requirements.
    """
    # Initialize evaluator with a parallel root
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
        default_model=model
    )

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_override_info(),
        template_class=ArkansasVetoOverrideExtraction,
        extraction_name="arkansas_veto_override_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return summary
    return evaluator.get_summary()