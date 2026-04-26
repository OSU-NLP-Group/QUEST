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
TASK_ID = "ocala_retail_min_lot_size"
TASK_DESCRIPTION = """
According to Ocala, Florida's zoning regulations, what is the minimum lot size required to develop a single retail store? Provide the minimum acreage requirement and cite the specific regulation or code section that establishes this requirement.
"""

EXPECTED_MIN_ACRES = "5 acres"
EXPECTED_CITATION_TEXT = "Division 29, Section 122-905 of Ocala's Code of Ordinances"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RetailLotSizeCitation(BaseModel):
    """
    Extracted information about the minimum lot size and the regulatory citation,
    as stated in the agent's answer.
    """
    minimum_acres: Optional[str] = None
    citation_text: Optional[str] = None
    citation_section: Optional[str] = None
    citation_division: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_retail_lot_size_and_citation() -> str:
    return """
    Extract the minimum lot size requirement and the specific regulatory citation for developing a single retail store in Ocala, Florida, as stated in the answer.

    Return a JSON object with these fields:
    - minimum_acres: The minimum lot size value as presented in the answer (e.g., "5 acres", "5-acre minimum", "5 ac"). Use a string exactly as stated in the answer. If not specified, return null.
    - citation_text: The regulatory citation text exactly as presented in the answer (e.g., "Division 29, Section 122-905", "Sec. 122-905 (Division 29)"). If not specified, return null.
    - citation_section: The section number if mentioned (e.g., "122-905"), otherwise null.
    - citation_division: The division number if mentioned (e.g., "29"), otherwise null.
    - source_urls: A list of all explicit URLs included in the answer that are relevant to this requirement or citation. Extract actual URLs only (including markdown links). If none are provided, return an empty list.

    Notes:
    - Only extract what is explicitly present in the answer. Do not infer or add information.
    - If the answer uses equivalent formats or abbreviations (e.g., "Sec. 122-905", "§122-905", "Div. 29"), capture them faithfully.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: RetailLotSizeCitation,
) -> None:
    """
    Build the verification sub-tree and execute verifications.
    """
    # Create the primary critical node (parallel aggregation)
    main_node = evaluator.add_parallel(
        id="Minimum_Lot_Size_Requirement_Identification",
        desc="Identify the minimum lot size requirement for developing a single retail store in Ocala, Florida, including both the numeric acreage value and the regulatory source",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Minimum acreage value (critical)
    acreage_leaf = evaluator.add_leaf(
        id="Minimum_Acreage_Value",
        desc="Provide the specific minimum acreage requirement (5 acres)",
        parent=main_node,
        critical=True,
    )

    # Verify that the answer explicitly states the minimum is 5 acres
    acreage_claim = (
        "The answer states that the minimum lot size required to develop a single retail store "
        "in Ocala, Florida is 5 acres."
    )
    await evaluator.verify(
        claim=acreage_claim,
        node=acreage_leaf,
        additional_instruction=(
            "Verify within the answer text that the minimum lot size is explicitly given as 5 acres. "
            "Accept reasonable variations such as '5-acre minimum', 'minimum of 5 acres', or '5 ac' "
            "only when clearly indicating a minimum requirement for a single retail store. "
            "If the number differs, is ambiguous, or is not presented as a minimum for a single retail store, mark as Incorrect."
        ),
    )

    # Leaf 2: Regulatory citation (critical)
    citation_leaf = evaluator.add_leaf(
        id="Regulatory_Citation",
        desc="Cite the specific regulation or code section that establishes this requirement (Division 29, Section 122-905 of Ocala's Code of Ordinances)",
        parent=main_node,
        critical=True,
    )

    citation_claim = (
        "The answer cites Division 29, Section 122-905 of Ocala's Code of Ordinances as the regulation "
        "establishing the minimum lot size requirement for a single retail store."
    )
    await evaluator.verify(
        claim=citation_claim,
        node=citation_leaf,
        additional_instruction=(
            "Check the answer text to confirm that it explicitly cites the specific code section. "
            "Allow equivalent forms such as 'Sec. 122-905', '§122-905', 'Section 122-905', and references to 'Division 29'. "
            "Both the section number (122-905) and the division (29) should be present (or clearly implied) for this to pass."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Ocala retail store minimum lot size requirement task.
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_retail_lot_size_and_citation(),
        template_class=RetailLotSizeCitation,
        extraction_name="retail_lot_size_and_citation",
    )

    # Add expected ground truth info for transparency (not used to drive verification directly)
    evaluator.add_ground_truth({
        "expected_minimum_acres": EXPECTED_MIN_ACRES,
        "expected_regulatory_citation": EXPECTED_CITATION_TEXT,
    }, gt_type="expected_values")

    # Build and verify according to rubric
    await build_and_verify_nodes(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()