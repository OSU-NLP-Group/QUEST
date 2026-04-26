import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "alaska_pfd_2026_info"
TASK_DESCRIPTION = (
    "What are the application period dates (start and end) for the 2026 Alaska Permanent Fund Dividend, "
    "what was the 2025 PFD payment amount, on what date will applications in 'Eligible-Not Paid' status as of March 11, 2026 "
    "receive their distribution, and what are the two key residency-related eligibility requirements that applicants must meet?"
)

# Ground truth claims derived from the rubric (used for logging/summary context)
GROUND_TRUTH = {
    "Application_Start_Date": "The PFD application period opens on January 1, 2026",
    "Application_End_Date": "The PFD application period closes on March 31, 2026",
    "Payment_Amount_2025": "The 2025 PFD payment amount is $1,000",
    "March_Payment_Distribution_Date": "Applications in 'Eligible-Not Paid' status on March 11, 2026 will be distributed on March 19, 2026",
    "Residency_Duration_Requirement": "Applicants must be residents of Alaska during all of the previous calendar year (January 1 - December 31)",
    "Intent_Requirement": "Applicants must intend to remain Alaska residents indefinitely at least on the date of application",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AlaskaPFDExtraction(BaseModel):
    application_start_date: Optional[str] = None
    application_end_date: Optional[str] = None
    payment_amount_2025: Optional[str] = None
    march_distribution_date: Optional[str] = None
    residency_duration_requirement: Optional[str] = None
    intent_requirement: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pfd_info() -> str:
    return (
        "Extract the following fields exactly as stated in the answer text. Return null for any missing field.\n"
        "Fields to extract:\n"
        "1. application_start_date: The opening date of the 2026 PFD application period (e.g., 'January 1, 2026').\n"
        "2. application_end_date: The closing date of the 2026 PFD application period (e.g., 'March 31, 2026').\n"
        "3. payment_amount_2025: The 2025 PFD payment amount (preserve formatting, e.g., '$1,000').\n"
        "4. march_distribution_date: The distribution date for applications in 'Eligible-Not Paid' status as of March 11, 2026.\n"
        "5. residency_duration_requirement: The residency duration requirement text (e.g., being an Alaska resident during the entire previous calendar year).\n"
        "6. intent_requirement: The intent-to-remain requirement text (e.g., intend to remain an Alaska resident indefinitely at least on the date of application).\n"
        "7. source_urls: Extract all URLs explicitly presented in the answer (including markdown links). Only include valid URLs. If none are present, return an empty array.\n"
        "Do not infer or invent data; extract only what is present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper: build verification claims                                           #
# --------------------------------------------------------------------------- #
def build_claim_texts() -> Dict[str, str]:
    # Use the rubric-provided claims directly
    return {
        "Application_Start_Date": GROUND_TRUTH["Application_Start_Date"],
        "Application_End_Date": GROUND_TRUTH["Application_End_Date"],
        "Payment_Amount_2025": GROUND_TRUTH["Payment_Amount_2025"],
        "March_Payment_Distribution_Date": GROUND_TRUTH["March_Payment_Distribution_Date"],
        "Residency_Duration_Requirement": GROUND_TRUTH["Residency_Duration_Requirement"],
        "Intent_Requirement": GROUND_TRUTH["Intent_Requirement"],
    }


def build_additional_instructions() -> Dict[str, str]:
    # Tailored guidance for each verification
    return {
        "Application_Start_Date": (
            "Verify on the cited source(s) that the Alaska Permanent Fund Dividend application period opens on January 1, 2026. "
            "Allow minor phrasing variations (e.g., 'opens Jan 1, 2026')."
        ),
        "Application_End_Date": (
            "Verify on the cited source(s) that the Alaska Permanent Fund Dividend application period closes on March 31, 2026. "
            "Allow equivalent phrasing."
        ),
        "Payment_Amount_2025": (
            "Verify that the official 2025 Alaska PFD payment amount is $1,000 as stated. "
            "Accept formatting variants such as '1000', 'USD 1,000', or '1,000 dollars' if the meaning is the same."
        ),
        "March_Payment_Distribution_Date": (
            "Verify that applications in 'Eligible-Not Paid' status on March 11, 2026 will receive distribution on March 19, 2026. "
            "Look for payment schedule, distribution calendar, or specific announcements."
        ),
        "Residency_Duration_Requirement": (
            "Verify that the residency requirement includes being an Alaska resident for the entire previous calendar year (January 1 through December 31). "
            "Accept equivalent language describing the full prior calendar year residency requirement."
        ),
        "Intent_Requirement": (
            "Verify that applicants must intend to remain Alaska residents indefinitely at least on the date of application. "
            "Equivalent phrasing like 'intend to remain in Alaska indefinitely' should be accepted."
        ),
    }


# --------------------------------------------------------------------------- #
# Verification flow                                                           #
# --------------------------------------------------------------------------- #
async def verify_pfd_info(
    evaluator: Evaluator,
    parent_node,
    extraction: AlaskaPFDExtraction,
    sources_prereq_node: Optional[Any],
) -> None:
    """
    Construct the verification nodes under the critical parent and run evidence-based checks
    using the answer's cited sources. If sources_prereq_node is provided, set it as a prerequisite
    so leaf checks are skipped when no sources are available.
    """
    # Create critical parallel aggregator for all key items
    critical_parent = evaluator.add_parallel(
        id="Alaska_PFD_2026_Information",
        desc="Verify all key information about Alaska Permanent Fund Dividend 2026 application and payment details",
        parent=parent_node,
        critical=True,
    )

    # Build claims and instructions
    claims = build_claim_texts()
    add_ins = build_additional_instructions()

    # Collect sources from the extraction
    sources = extraction.source_urls if extraction and extraction.source_urls else []

    # Optional extra prerequisites gating (sources presence)
    extra_prereqs = [sources_prereq_node] if sources_prereq_node is not None else None

    # Define leaf nodes and schedule verifications
    leaf_nodes_and_specs: List[Dict[str, Any]] = []

    # 1. Application Start Date
    node_start = evaluator.add_leaf(
        id="Application_Start_Date",
        desc=claims["Application_Start_Date"],
        parent=critical_parent,
        critical=True,
    )
    leaf_nodes_and_specs.append({
        "claim": claims["Application_Start_Date"],
        "node": node_start,
        "additional_instruction": add_ins["Application_Start_Date"],
    })

    # 2. Application End Date
    node_end = evaluator.add_leaf(
        id="Application_End_Date",
        desc=claims["Application_End_Date"],
        parent=critical_parent,
        critical=True,
    )
    leaf_nodes_and_specs.append({
        "claim": claims["Application_End_Date"],
        "node": node_end,
        "additional_instruction": add_ins["Application_End_Date"],
    })

    # 3. 2025 Payment Amount
    node_amt = evaluator.add_leaf(
        id="Payment_Amount_2025",
        desc=claims["Payment_Amount_2025"],
        parent=critical_parent,
        critical=True,
    )
    leaf_nodes_and_specs.append({
        "claim": claims["Payment_Amount_2025"],
        "node": node_amt,
        "additional_instruction": add_ins["Payment_Amount_2025"],
    })

    # 4. March Payment Distribution Date
    node_march = evaluator.add_leaf(
        id="March_Payment_Distribution_Date",
        desc=claims["March_Payment_Distribution_Date"],
        parent=critical_parent,
        critical=True,
    )
    leaf_nodes_and_specs.append({
        "claim": claims["March_Payment_Distribution_Date"],
        "node": node_march,
        "additional_instruction": add_ins["March_Payment_Distribution_Date"],
    })

    # 5. Residency Duration Requirement
    node_res = evaluator.add_leaf(
        id="Residency_Duration_Requirement",
        desc=claims["Residency_Duration_Requirement"],
        parent=critical_parent,
        critical=True,
    )
    leaf_nodes_and_specs.append({
        "claim": claims["Residency_Duration_Requirement"],
        "node": node_res,
        "additional_instruction": add_ins["Residency_Duration_Requirement"],
    })

    # 6. Intent Requirement
    node_intent = evaluator.add_leaf(
        id="Intent_Requirement",
        desc=claims["Intent_Requirement"],
        parent=critical_parent,
        critical=True,
    )
    leaf_nodes_and_specs.append({
        "claim": claims["Intent_Requirement"],
        "node": node_intent,
        "additional_instruction": add_ins["Intent_Requirement"],
    })

    # Prepare batch tasks to verify by URLs when available; otherwise fall back to simple verification.
    # We pass sources (possibly empty) and prerequisites to skip when no sources are present.
    batch_items = []
    for spec in leaf_nodes_and_specs:
        batch_items.append((
            spec["claim"],
            sources,  # list of URLs from the answer; can be empty which routes to simple_verify
            spec["node"],
            spec["additional_instruction"],
        ))

    # Execute verifications in parallel
    # Pass extra_prerequisites through **kwargs so Evaluator.verify can apply gating logic
    await evaluator.batch_verify(
        batch_items,
        extra_prerequisites=extra_prereqs
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the agent's answer for Alaska PFD 2026 key information using the Mind2Web2 framework.
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_pfd_info(),
        template_class=AlaskaPFDExtraction,
        extraction_name="alaska_pfd_extraction",
    )

    # Add ground truth information to summary
    evaluator.add_ground_truth({
        "expected_claims": GROUND_TRUTH,
        "note": "Verification checks use the agent-provided source URLs when available."
    }, gt_type="ground_truth_expected")

    # Record a custom info entry with source stats
    num_sources = len(extracted.source_urls) if extracted and extracted.source_urls else 0
    evaluator.add_custom_info(
        info={
            "total_extracted_sources": num_sources,
            "sources_preview": extracted.source_urls[:5] if extracted and extracted.source_urls else [],
        },
        info_type="source_statistics",
        info_name="answer_source_stats"
    )

    # Optional precondition: require at least one source URL from the answer.
    # This is non-critical at the root level but will be used to skip leaf verifications when absent,
    # causing the critical child aggregator to fail due to skipped children (score 0).
    sources_available_node = evaluator.add_custom_node(
        result=(num_sources > 0),
        id="sources_available",
        desc="Answer provides at least one source URL for verification",
        parent=root,
        critical=False
    )

    # Build and verify the critical subtree using the extracted data and sources gating
    await verify_pfd_info(
        evaluator=evaluator,
        parent_node=root,
        extraction=extracted,
        sources_prereq_node=sources_available_node
    )

    # Return the standard summary
    return evaluator.get_summary()