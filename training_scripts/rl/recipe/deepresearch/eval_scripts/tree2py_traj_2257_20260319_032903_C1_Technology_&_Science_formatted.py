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
TASK_ID = "verizon_outage_2026"
TASK_DESCRIPTION = (
    "On January 14, 2026, Verizon experienced a major nationwide network outage. "
    "According to official investigations, what specific technical network system failed, "
    "and in which U.S. state were the suspected failed servers located?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageExtraction(BaseModel):
    """
    Structured extraction from the agent's answer.
    """
    technical_system: Optional[str] = None
    location_state: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_facts() -> str:
    return """
    Extract the following fields from the answer:

    1) technical_system: The specific technical network system that failed, as stated in the answer text. 
       - Prefer the most specific term referring to the subsystem (e.g., "5G Standalone (5G SA) core network", "5G SA core", etc.).
       - Return exactly the phrasing used in the answer when possible. If multiple variants appear, select the clearest/most specific one.
       - If not explicitly provided, return null.

    2) location_state: The U.S. state where officials or investigations suspected the failed servers were located.
       - Return the state name in long form (e.g., "New Jersey"), even if the answer uses abbreviations like "NJ" or "N.J.".
       - If not explicitly provided, return null.

    3) source_urls: Extract all explicit URLs cited in the answer that support these facts (the technical system and/or the state location).
       - Include all valid URLs present anywhere in the answer (plain links or markdown links).
       - Do not invent any URLs. If no URLs are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions for building verification subtrees                         #
# --------------------------------------------------------------------------- #
async def build_technical_system_checks(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageExtraction,
) -> None:
    """
    Build and run verification nodes for the 'Technical_System' requirement.
    This subtree is sequential so that missing prerequisites skip downstream checks.
    """
    tech_node = evaluator.add_sequential(
        id="Technical_System",
        desc="Correctly identify that the failure occurred in Verizon's 5G Standalone (5G SA) core network",
        parent=parent_node,
        critical=True,  # Per rubric intent: this is an essential fact
    )

    # Step 1: Existence of the extracted technical system
    tech_present = evaluator.add_custom_node(
        result=bool(extraction.technical_system and extraction.technical_system.strip()),
        id="tech_system_provided",
        desc="The answer provides a specific technical system for the failure",
        parent=tech_node,
        critical=True,
    )

    # Step 2: Value match to expected concept (5G Standalone core network)
    tech_match_leaf = evaluator.add_leaf(
        id="tech_system_value_match",
        desc="The identified technical system matches '5G Standalone (5G SA) core network' (or an equivalent phrase)",
        parent=tech_node,
        critical=True,
    )
    extracted_tech = extraction.technical_system or ""
    claim_match = (
        f"The extracted technical system text is: '{extracted_tech}'. "
        "This refers to Verizon's 5G Standalone (5G SA) core network, or is an equivalent phrasing "
        "(e.g., '5G SA core', '5G standalone core network', 'standalone 5G core')."
    )
    await evaluator.verify(
        claim=claim_match,
        node=tech_match_leaf,
        additional_instruction=(
            "Judge whether the extracted phrase denotes the same system as '5G Standalone (5G SA) core network'. "
            "Allow minor wording variations and synonyms (e.g., '5G SA core', 'SA 5G core', 'standalone 5G core'). "
            "Be lenient on formatting, capitalization, or hyphenation."
        ),
    )

    # Step 3: Presence of sources (so that the support check can be grounded)
    tech_sources_present = evaluator.add_custom_node(
        result=bool(extraction.source_urls and len(extraction.source_urls) > 0),
        id="tech_system_sources_present",
        desc="URLs are provided to support the technical system claim",
        parent=tech_node,
        critical=True,
    )

    # Step 4: Verify the claim is supported by cited sources
    tech_supported_leaf = evaluator.add_leaf(
        id="tech_system_supported_by_sources",
        desc="Sources support that the outage cause was a failure in the 5G Standalone (5G SA) core network",
        parent=tech_node,
        critical=True,
    )
    claim_supported = (
        "According to officials or formal investigations into the January 14, 2026 Verizon outage, "
        "the cause was a failure in Verizon's 5G Standalone (5G SA) core network "
        "(allow equivalent phrasings such as '5G SA core', '5G standalone core network')."
    )
    await evaluator.verify(
        claim=claim_supported,
        node=tech_supported_leaf,
        sources=extraction.source_urls,
        additional_instruction=(
            "Confirm that at least one cited page explicitly attributes the Jan. 14, 2026 Verizon outage "
            "to a failure in the 5G Standalone (5G SA) core network. Accept equivalent phrasing such as "
            "'5G SA core', 'standalone 5G core network', or similar. "
            "It's acceptable if the page is a credible report quoting officials or an official statement."
        ),
    )


async def build_location_checks(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageExtraction,
) -> None:
    """
    Build and run verification nodes for the 'Geographic_Location' requirement.
    This subtree is sequential so that missing prerequisites skip downstream checks.
    """
    loc_node = evaluator.add_sequential(
        id="Geographic_Location",
        desc="Correctly identify that officials pointed to a suspected server failure in New Jersey",
        parent=parent_node,
        critical=True,  # Per rubric intent: this is an essential fact
    )

    # Step 1: Existence of the extracted state
    loc_present = evaluator.add_custom_node(
        result=bool(extraction.location_state and extraction.location_state.strip()),
        id="location_state_provided",
        desc="The answer provides a U.S. state for the suspected failed servers",
        parent=loc_node,
        critical=True,
    )

    # Step 2: Value match to 'New Jersey' (allow 'NJ', 'N.J.')
    loc_match_leaf = evaluator.add_leaf(
        id="location_state_value_match",
        desc="The identified state matches 'New Jersey' (allow 'NJ' / 'N.J.')",
        parent=loc_node,
        critical=True,
    )
    extracted_state = extraction.location_state or ""
    claim_state_match = (
        f"The extracted state is '{extracted_state}'. "
        "This is the same U.S. state as 'New Jersey' (accept 'NJ' or 'N.J.' as equivalent)."
    )
    await evaluator.verify(
        claim=claim_state_match,
        node=loc_match_leaf,
        additional_instruction=(
            "Judge whether the extracted state denotes 'New Jersey'. "
            "Accept common abbreviations like 'NJ' or 'N.J.' as equivalent."
        ),
    )

    # Step 3: Presence of sources (so that the support check can be grounded)
    loc_sources_present = evaluator.add_custom_node(
        result=bool(extraction.source_urls and len(extraction.source_urls) > 0),
        id="location_state_sources_present",
        desc="URLs are provided to support the location claim",
        parent=loc_node,
        critical=True,
    )

    # Step 4: Verify the claim is supported by cited sources
    loc_supported_leaf = evaluator.add_leaf(
        id="location_state_supported_by_sources",
        desc="Sources support that officials suspected the failed servers were in New Jersey",
        parent=loc_node,
        critical=True,
    )
    claim_loc_supported = (
        "Officials or formal investigations into the January 14, 2026 Verizon outage indicated or suspected "
        "that the failed servers were located in New Jersey (accept 'NJ' / 'N.J.')."
    )
    await evaluator.verify(
        claim=claim_loc_supported,
        node=loc_supported_leaf,
        sources=extraction.source_urls,
        additional_instruction=(
            "Confirm that at least one cited page clearly states that officials or investigators suspected "
            "the failed servers were in New Jersey. Accept 'NJ' or 'N.J.' as equivalent to 'New Jersey'."
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
    Evaluate an answer for the Verizon outage facts task.

    Returns a structured summary with the verification tree and final score.
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_outage_facts(),
        template_class=OutageExtraction,
        extraction_name="outage_extraction",
    )

    # Add Ground Truth (for context in the final report; not used directly for scoring)
    evaluator.add_ground_truth(
        {
            "expected_technical_system": "5G Standalone (5G SA) core network",
            "expected_location_state": "New Jersey",
            "event_date": "January 14, 2026",
        },
        gt_type="ground_truth",
    )

    # Build the rubric root node (parallel aggregation, with two critical child groups)
    rubric_root = evaluator.add_parallel(
        id="Verizon_Outage_Facts",
        desc=(
            "Correctly identify the technical system that failed and the U.S. state location of the suspected "
            "server failure during Verizon's January 14, 2026 network outage"
        ),
        parent=root,
        critical=False,  # Keep this non-critical; make its children critical to require both facts
    )

    # Build and verify the two core fact groups (both marked critical under the rubric root)
    await build_technical_system_checks(evaluator, rubric_root, extraction)
    await build_location_checks(evaluator, rubric_root, extraction)

    # Return the final structured summary
    return evaluator.get_summary()