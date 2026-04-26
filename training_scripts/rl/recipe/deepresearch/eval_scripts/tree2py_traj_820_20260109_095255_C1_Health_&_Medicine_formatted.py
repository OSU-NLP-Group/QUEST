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
TASK_ID = "first_non_opioid_analgesic_2025"
TASK_DESCRIPTION = """
What is the brand name of the first non-opioid analgesic approved by the FDA in 2025 for the treatment of moderate to severe acute pain in adults that also received breakthrough therapy designation?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DrugExtraction(BaseModel):
    """Structured extraction of the identified drug entity and supporting context from the answer."""
    drug_brand_name: Optional[str] = None
    drug_generic_name: Optional[str] = None
    manufacturer: Optional[str] = None
    fda_approval_date_text: Optional[str] = None  # e.g., "January 15, 2025"
    indication_text: Optional[str] = None         # e.g., "moderate to severe acute pain in adults"
    classification_text: Optional[str] = None     # e.g., "non-opioid analgesic"
    breakthrough_designation_text: Optional[str] = None  # e.g., "received FDA Breakthrough Therapy designation"
    first_qualifying_2025_text: Optional[str] = None     # e.g., "first such approval in 2025"
    source_urls: List[str] = Field(default_factory=list) # URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_drug_info() -> str:
    return """
    Extract details about the single drug entity the answer claims satisfies all of the following constraints:
    - Approved by the FDA in 2025
    - Indicated for moderate to severe acute pain in adults
    - Classified as a non-opioid analgesic
    - Received FDA Breakthrough Therapy designation
    - Claimed to be the first such approval in 2025

    Return a JSON object with the following fields:
    - drug_brand_name: The brand name of the drug given in the answer. If not present, null.
    - drug_generic_name: The generic/active ingredient name. If not present, null.
    - manufacturer: The company/manufacturer, if provided. If not present, null.
    - fda_approval_date_text: The FDA approval date mentioned in the answer (any text format; e.g., "January 3, 2025", "2025"). If not present, null.
    - indication_text: The indication described for this drug as stated in the answer; make sure to capture any mention of "moderate to severe acute pain in adults" if present. If not present, null.
    - classification_text: The classification description from the answer regarding non-opioid status (e.g., "non-opioid analgesic"). If not present, null.
    - breakthrough_designation_text: The text in the answer indicating FDA Breakthrough Therapy designation (e.g., "received Breakthrough Therapy designation"). If not present, null.
    - first_qualifying_2025_text: Any text in the answer that asserts it is the first qualifying approval in 2025. If not present, null.
    - source_urls: An array of all URLs explicitly cited in the answer that support any of the claims for this drug. Extract actual URLs only (including those in markdown). If none are present, return an empty array.

    Special rules for URLs:
    - Extract only valid and complete URLs explicitly present in the answer.
    - Include full URLs with protocol; if missing, prepend "http://".
    - Do not invent or infer URLs; if not provided, leave the array empty.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _display_drug_name(dx: DrugExtraction) -> str:
    """Return a readable drug identifier from extracted info."""
    if dx.drug_brand_name and dx.drug_generic_name:
        return f"{dx.drug_brand_name} ({dx.drug_generic_name})"
    if dx.drug_brand_name:
        return dx.drug_brand_name
    if dx.drug_generic_name:
        return dx.drug_generic_name
    return "the identified drug"


def _dedup_urls(urls: List[str]) -> List[str]:
    """Return a de-duplicated list of URLs while preserving order."""
    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_drug_checks(
    evaluator: Evaluator,
    parent_seq_node,
    dx: DrugExtraction,
) -> None:
    """
    Build 'Identify_Correct_Drug' critical parallel node and verify each required constraint using the extracted info.
    """
    # Create the critical parallel node for identifying the correct drug
    identify_node = evaluator.add_parallel(
        id="Identify_Correct_Drug",
        desc="Identifies the correct drug entity that satisfies all stated constraints, including being the first qualifying approval in 2025",
        parent=parent_seq_node,
        critical=True  # Parent is critical; all children must be critical
    )

    # Prepare shared values
    drug_display = _display_drug_name(dx)
    sources_list = _dedup_urls(dx.source_urls)

    # 1) FDA_Approval_2025
    fda_approval_leaf = evaluator.add_leaf(
        id="FDA_Approval_2025",
        desc="The drug was approved by the FDA in 2025",
        parent=identify_node,
        critical=True
    )
    if dx.fda_approval_date_text:
        claim = f"The drug {drug_display} was approved by the U.S. FDA in 2025 (approval date mentioned as '{dx.fda_approval_date_text}')."
    else:
        claim = f"The drug {drug_display} was approved by the U.S. FDA in 2025."
    await evaluator.verify(
        claim=claim,
        node=fda_approval_leaf,
        sources=sources_list,
        additional_instruction="Verify the year of FDA approval is 2025. Focus on confirming the approval year; minor differences in month/day are acceptable."
    )

    # 2) Pain_Indication: moderate to severe acute pain in adults
    pain_ind_leaf = evaluator.add_leaf(
        id="Pain_Indication",
        desc="The drug is indicated for the treatment of moderate to severe acute pain in adults",
        parent=identify_node,
        critical=True
    )
    if dx.indication_text:
        claim = f"The drug {drug_display} is indicated for the treatment of moderate to severe acute pain in adults (as described: '{dx.indication_text}')."
    else:
        claim = f"The drug {drug_display} is indicated for the treatment of moderate to severe acute pain in adults."
    await evaluator.verify(
        claim=claim,
        node=pain_ind_leaf,
        sources=sources_list,
        additional_instruction="Check the product labeling or official sources to confirm the indication explicitly includes 'moderate to severe acute pain in adults'. Allow reasonable phrasing variants ('moderate-to-severe', 'adult patients')."
    )

    # 3) Non_Opioid_Classification
    non_op_leaf = evaluator.add_leaf(
        id="Non_Opioid_Classification",
        desc="The drug is classified as a non-opioid analgesic",
        parent=identify_node,
        critical=True
    )
    if dx.classification_text:
        claim = f"The drug {drug_display} is a non-opioid analgesic (classification noted as '{dx.classification_text}')."
    else:
        claim = f"The drug {drug_display} is a non-opioid analgesic."
    await evaluator.verify(
        claim=claim,
        node=non_op_leaf,
        sources=sources_list,
        additional_instruction="Confirm the drug is not an opioid and is properly characterized as a non-opioid analgesic. Consider mechanism class descriptions as supporting evidence."
    )

    # 4) Breakthrough_Therapy_Designation
    btd_leaf = evaluator.add_leaf(
        id="Breakthrough_Therapy_Designation",
        desc="The drug received FDA breakthrough therapy designation",
        parent=identify_node,
        critical=True
    )
    if dx.breakthrough_designation_text:
        claim = f"The drug {drug_display} received FDA Breakthrough Therapy designation (noted as '{dx.breakthrough_designation_text}')."
    else:
        claim = f"The drug {drug_display} received FDA Breakthrough Therapy designation."
    await evaluator.verify(
        claim=claim,
        node=btd_leaf,
        sources=sources_list,
        additional_instruction="Confirm that the FDA granted Breakthrough Therapy designation for the drug; official FDA communications or reputable sources should indicate this."
    )

    # 5) First_Qualifying_Approval_In_2025
    first_leaf = evaluator.add_leaf(
        id="First_Qualifying_Approval_In_2025",
        desc="Among drugs meeting the above criteria, this drug is the first (earliest) such FDA approval in 2025",
        parent=identify_node,
        critical=True
    )
    if dx.first_qualifying_2025_text and dx.fda_approval_date_text:
        claim = (
            f"Among FDA approvals in 2025 that meet the non-opioid analgesic, indication, and BTD criteria, "
            f"{drug_display} was the first such approval (the answer asserts '{dx.first_qualifying_2025_text}' with approval date '{dx.fda_approval_date_text}')."
        )
    elif dx.first_qualifying_2025_text:
        claim = (
            f"Among FDA approvals in 2025 that meet these criteria, {drug_display} was the first such approval "
            f"(the answer asserts: '{dx.first_qualifying_2025_text}')."
        )
    else:
        claim = (
            f"Among FDA approvals in 2025 that meet the non-opioid analgesic, indication, and BTD criteria, "
            f"{drug_display} was the first such approval."
        )
    await evaluator.verify(
        claim=claim,
        node=first_leaf,
        sources=sources_list,
        additional_instruction=(
            "Determine whether the claim that this was the first qualifying FDA approval in 2025 is explicitly supported "
            "by the provided sources (e.g., official announcements or timeline comparisons). "
            "If sources are absent or do not explicitly support 'first', assess consistency with the answer; "
            "however, prefer explicit evidence for 'first'."
        )
    )


async def verify_brand_name_provided(
    evaluator: Evaluator,
    parent_seq_node,
    dx: DrugExtraction,
) -> None:
    """
    Build and verify the final brand name provision as a critical leaf under the sequential 'Answer' node.
    """
    brand_leaf = evaluator.add_leaf(
        id="Brand_Name_Provided",
        desc="Provides the brand name of the identified drug as the final answer (not only the generic name or manufacturer)",
        parent=parent_seq_node,
        critical=True
    )

    brand = dx.drug_brand_name or ""
    generic = dx.drug_generic_name or ""
    claim = (
        f"The answer provides the brand name of the identified drug, not only the generic name or manufacturer. "
        f"If present, the brand name given is '{brand}'."
    )
    await evaluator.verify(
        claim=claim,
        node=brand_leaf,
        additional_instruction=(
            "Examine the full answer: confirm that a brand name is explicitly provided as the final answer. "
            "Providing only a generic name or manufacturer is insufficient."
        )
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer to determine whether it correctly provides the brand name of the first non-opioid analgesic
    approved by the FDA in 2025 for moderate to severe acute pain in adults that received breakthrough therapy designation.
    """
    # Initialize evaluator (root is non-critical by framework design; we will create a critical child node)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential: identify correct drug, then brand name provision
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

    # Extract structured drug info from the answer
    dx = await evaluator.extract(
        prompt=prompt_extract_drug_info(),
        template_class=DrugExtraction,
        extraction_name="drug_extraction"
    )

    # Create the top-level critical sequential "Answer" node to mirror rubric root
    answer_node = evaluator.add_sequential(
        id="Answer",
        desc="Provides the brand name of the first FDA-approved-in-2025 non-opioid analgesic for moderate to severe acute pain in adults that received breakthrough therapy designation",
        parent=root,
        critical=True
    )

    # Child 1: Identify Correct Drug (critical parallel group)
    await build_and_verify_drug_checks(evaluator, answer_node, dx)

    # Child 2: Brand Name Provided (critical leaf)
    await verify_brand_name_provided(evaluator, answer_node, dx)

    # Return the standardized evaluation summary
    return evaluator.get_summary()