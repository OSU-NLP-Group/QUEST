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
TASK_ID = "verizon_outage_jan_2026"
TASK_DESCRIPTION = (
    "In January 2026, Verizon experienced a major nationwide network outage that affected millions of customers. "
    "Provide comprehensive details about this outage, including: (1) the exact date it occurred, "
    "(2) the exact time when Verizon officially declared it resolved, (3) the total duration of the outage, "
    "(4) the peak number of customers affected, (5) the root cause stated by Verizon, "
    "(6) the amount of account credit offered as compensation, (7) the method for affected customers to redeem this credit, "
    "and (8) what this credit amount covers according to Verizon's statement."
)

# Ground truth expectations to record in summary
GROUND_TRUTH_EXPECTATIONS = {
    "OutageDate": "January 14, 2026",
    "ResolutionTime": "10:15 PM ET on January 14, 2026",
    "OutageDuration": "More than 10 hours",
    "CustomersAffected": "Over 1.5 million at peak (per Downdetector)",
    "RootCause": "Software issue/problem (as stated by Verizon)",
    "CompensationAmount": "$20 account credit",
    "RedemptionMethod": "Redeem via the myVerizon app",
    "CreditCoverage": "Covers multiple days of service on average (per Verizon)"
}

# --------------------------------------------------------------------------- #
# Data model for extracted information                                        #
# --------------------------------------------------------------------------- #
class VerizonOutageExtraction(BaseModel):
    """
    Extracted details from the agent's answer about the January 2026 Verizon outage,
    plus per-item source URLs explicitly cited in the answer (if any).
    """
    outage_date: Optional[str] = None  # e.g., "January 14, 2026"
    resolution_time: Optional[str] = None  # e.g., "10:15 PM ET on January 14, 2026"
    outage_duration: Optional[str] = None  # e.g., "more than 10 hours"
    customers_affected: Optional[str] = None  # e.g., "over 1.5 million"
    root_cause: Optional[str] = None  # e.g., "software issue/problem"
    compensation_amount: Optional[str] = None  # e.g., "$20"
    redemption_method: Optional[str] = None  # e.g., "myVerizon app"
    credit_coverage: Optional[str] = None  # e.g., "covers multiple days of service"

    # Per-field sources extracted from the answer text (explicit URLs only)
    date_sources: List[str] = Field(default_factory=list)
    resolution_time_sources: List[str] = Field(default_factory=list)
    duration_sources: List[str] = Field(default_factory=list)
    affected_sources: List[str] = Field(default_factory=list)
    root_cause_sources: List[str] = Field(default_factory=list)
    compensation_sources: List[str] = Field(default_factory=list)
    redemption_sources: List[str] = Field(default_factory=list)
    coverage_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_verizon_outage() -> str:
    return (
        "Extract the eight requested outage details from the provided answer text for Verizon's January 2026 outage. "
        "Return a JSON object with the following fields:\n"
        "1) outage_date: The specific date when the outage occurred (string; return null if missing)\n"
        "2) resolution_time: The exact time when Verizon officially declared the outage resolved (include timezone and date if present; string; null if missing)\n"
        "3) outage_duration: The total duration of the outage (string; null if missing)\n"
        "4) customers_affected: The peak number of customers affected (string; null if missing)\n"
        "5) root_cause: The stated cause of the outage per Verizon (string; null if missing)\n"
        "6) compensation_amount: The amount of account credit offered (string; null if missing)\n"
        "7) redemption_method: How affected customers can claim/redeem the credit (string; null if missing)\n"
        "8) credit_coverage: What this credit amount covers according to Verizon's statement (string; null if missing)\n\n"
        "Additionally, extract explicit source URLs cited in the answer for each field (if any). "
        "Include these arrays in the JSON:\n"
        "• date_sources\n"
        "• resolution_time_sources\n"
        "• duration_sources\n"
        "• affected_sources\n"
        "• root_cause_sources\n"
        "• compensation_sources\n"
        "• redemption_sources\n"
        "• coverage_sources\n\n"
        "Extraction rules for URLs:\n"
        "- Only include explicit URLs present in the answer text (including markdown links); do not invent any URLs.\n"
        "- If a URL is missing a protocol, prepend http://\n"
        "- If no explicit URLs are provided for a field, return an empty array for that field's sources.\n"
    )


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
async def build_and_verify_outage_nodes(
    evaluator: Evaluator,
    parent_node,
    info: VerizonOutageExtraction
) -> None:
    """
    Construct leaf nodes per rubric and run verification for each factual claim,
    grounded in the sources cited in the agent's answer (when present).
    """
    # OutageDate
    node_date = evaluator.add_leaf(
        id="OutageDate",
        desc="The specific date when the Verizon outage occurred (January 14, 2026)",
        parent=parent_node,
        critical=True
    )
    claim_date = "Verizon’s nationwide network outage occurred on January 14, 2026."
    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        sources=info.date_sources,
        additional_instruction=(
            "Confirm that the outage date is January 14, 2026. Prefer official Verizon statements or credible coverage."
        ),
    )

    # ResolutionTime
    node_resolution = evaluator.add_leaf(
        id="ResolutionTime",
        desc="The exact time when Verizon declared the outage resolved (10:15 PM ET on January 14, 2026)",
        parent=parent_node,
        critical=True
    )
    claim_resolution = "Verizon officially declared the outage resolved at 10:15 PM ET on January 14, 2026."
    await evaluator.verify(
        claim=claim_resolution,
        node=node_resolution,
        sources=info.resolution_time_sources,
        additional_instruction=(
            "Allow minor formatting variations (e.g., 10:15 pm ET). The key is the official declaration time from Verizon."
        ),
    )

    # OutageDuration
    node_duration = evaluator.add_leaf(
        id="OutageDuration",
        desc="The total duration of the outage (more than 10 hours)",
        parent=parent_node,
        critical=True
    )
    claim_duration = "The January 2026 Verizon outage lasted more than 10 hours."
    await evaluator.verify(
        claim=claim_duration,
        node=node_duration,
        sources=info.duration_sources,
        additional_instruction=(
            "Confirm that the total duration exceeds 10 hours. Accept phrasing like 'over 10 hours' or 'more than 10 hours'."
        ),
    )

    # CustomersAffected
    node_customers = evaluator.add_leaf(
        id="CustomersAffected",
        desc="The number of customers affected at peak (over 1.5 million according to Downdetector reports)",
        parent=parent_node,
        critical=True
    )
    claim_customers = (
        "At peak, over 1.5 million customers were reported affected by the Verizon outage, according to Downdetector."
    )
    await evaluator.verify(
        claim=claim_customers,
        node=node_customers,
        sources=info.affected_sources,
        additional_instruction=(
            "Prefer a Downdetector source or credible coverage that cites Downdetector. "
            "Allow reasonable approximations (e.g., 'about 1.5 million+')."
        ),
    )

    # RootCause
    node_rootcause = evaluator.add_leaf(
        id="RootCause",
        desc="The stated cause of the outage (software issue or software problem)",
        parent=parent_node,
        critical=True
    )
    claim_rootcause = "Verizon stated the outage was caused by a software issue (software problem)."
    await evaluator.verify(
        claim=claim_rootcause,
        node=node_rootcause,
        sources=info.root_cause_sources,
        additional_instruction=(
            "Verify that Verizon (the company) attributed the outage to a software issue/problem. "
            "Allow synonyms like 'software change' or 'software-related error' if explicitly stated by Verizon."
        ),
    )

    # CompensationAmount
    node_compensation = evaluator.add_leaf(
        id="CompensationAmount",
        desc="The amount of account credit offered to affected customers ($20)",
        parent=parent_node,
        critical=True
    )
    claim_compensation = "Verizon offered $20 in account credit to affected customers."
    await evaluator.verify(
        claim=claim_compensation,
        node=node_compensation,
        sources=info.compensation_sources,
        additional_instruction=(
            "Confirm the compensation amount is $20 as announced by Verizon. "
            "If multiple credits are mentioned, confirm the $20 account credit applies to affected outage customers."
        ),
    )

    # RedemptionMethod
    node_redemption = evaluator.add_leaf(
        id="RedemptionMethod",
        desc="How affected customers can claim the credit (through the myVerizon app)",
        parent=parent_node,
        critical=True
    )
    claim_redemption = "Affected customers can redeem the credit through the myVerizon app."
    await evaluator.verify(
        claim=claim_redemption,
        node=node_redemption,
        sources=info.redemption_sources,
        additional_instruction=(
            "Verify that the redemption path is via the myVerizon app (not website or phone only), per Verizon's communication."
        ),
    )

    # CreditCoverage
    node_coverage = evaluator.add_leaf(
        id="CreditCoverage",
        desc="What the $20 credit covers according to Verizon (multiple days of service on average)",
        parent=parent_node,
        critical=True
    )
    claim_coverage = "According to Verizon, the $20 credit covers multiple days of service on average."
    await evaluator.verify(
        claim=claim_coverage,
        node=node_coverage,
        sources=info.coverage_sources,
        additional_instruction=(
            "Confirm that Verizon stated the $20 credit covers 'multiple days of service on average' (or equivalent phrasing)."
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an agent's answer for the January 2026 Verizon outage details.
    Returns a structured summary including the verification tree and final score.
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
        default_model=model
    )

    # Add a top-level critical node representing the rubric root
    task_node = evaluator.add_parallel(
        id="VerizonOutageInformation",
        desc="Comprehensive information about the January 2026 Verizon network outage",
        parent=root,
        critical=True
    )

    # Extract structured information and cited sources from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_verizon_outage(),
        template_class=VerizonOutageExtraction,
        extraction_name="verizon_outage_extraction"
    )

    # Record expected ground truth for transparency
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH_EXPECTATIONS,
        "note": "Expected facts per Verizon and widely reported coverage for January 2026 outage."
    })

    # Build leaf nodes and run verification for each criterion
    await build_and_verify_outage_nodes(evaluator, task_node, extraction)

    # Return evaluator summary
    return evaluator.get_summary()