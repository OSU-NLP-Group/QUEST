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
TASK_ID = "sd_zoo_parking_info"
TASK_DESCRIPTION = """
I am planning to visit the San Diego Zoo and need to know the parking situation. Please provide the following information: 
(1) The complete physical address of the San Diego Zoo, and 
(2) The current parking fee per vehicle for non-members.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ZooInfoExtraction(BaseModel):
    # Physical address content and its cited sources (URLs)
    address: Optional[str] = None
    address_sources: List[str] = Field(default_factory=list)

    # Parking fee content and its cited sources (URLs)
    parking_fee: Optional[str] = None
    parking_fee_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_zoo_info() -> str:
    return """
    Extract from the answer the following fields about the San Diego Zoo:
    1) address: The complete physical/mailing address for the San Diego Zoo as written in the answer.
    2) address_sources: A list of all URLs cited in the answer that support the address.
    3) parking_fee: The stated current parking fee per vehicle for non-members at the San Diego Zoo as written in the answer. 
       If the answer states parking is free, extract the value exactly as written (e.g., "free", "$0", "no charge").
    4) parking_fee_sources: A list of all URLs cited in the answer that support the parking fee information.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer text.
    - Do not infer or add any URLs not present.
    - For each list of sources, include every relevant URL mentioned for that item. If no URL is provided in the answer for an item, return an empty list for that item's sources.
    - If a field is not present in the answer, set it to null (for strings) or [] (for lists).
    """.strip()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_physical_address(evaluator: Evaluator, parent_node, extracted: ZooInfoExtraction) -> None:
    """
    Build and verify the 'Physical Address' subtree:
    - Check presence + at least one URL source
    - Verify the stated address is supported by the cited URLs
    """
    # Parent node for address (critical, sequential: presence gate before verification)
    addr_node = evaluator.add_sequential(
        id="Physical_Address",
        desc="The answer provides the complete physical address of the San Diego Zoo",
        parent=parent_node,
        critical=True
    )

    address_val = (extracted.address or "").strip()
    address_srcs = extracted.address_sources or []

    # Existence + source presence check (critical)
    evaluator.add_custom_node(
        result=(len(address_val) > 0 and len(address_srcs) > 0),
        id="address_provided_with_sources",
        desc="A non-empty address is provided and at least one supporting URL is cited for the address",
        parent=addr_node,
        critical=True
    )

    # Verify the address content is supported by the provided URLs (critical)
    addr_supported_leaf = evaluator.add_leaf(
        id="address_supported_by_sources",
        desc="The stated complete physical address is supported by the cited sources",
        parent=addr_node,
        critical=True
    )

    addr_claim = f"The complete physical address of the San Diego Zoo is '{address_val}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_supported_leaf,
        sources=address_srcs,
        additional_instruction=(
            "Verify that at least one provided URL explicitly lists the official address for the San Diego Zoo "
            "(the one located in Balboa Park, not the San Diego Zoo Safari Park). "
            "To be considered 'complete', the address should include street number and name, city, state (CA), and ZIP code. "
            "Allow minor formatting differences (e.g., 'Drive' vs 'Dr', presence/absence of 'USA')."
        ),
    )


async def verify_parking_fee(evaluator: Evaluator, parent_node, extracted: ZooInfoExtraction) -> None:
    """
    Build and verify the 'Parking Fee' subtree:
    - Check presence + at least one URL source
    - Verify the stated fee is supported by the cited URLs and applies to non-members at the San Diego Zoo
    """
    # Parent node for parking fee (critical, sequential: presence gate before verification)
    fee_node = evaluator.add_sequential(
        id="Parking_Fee",
        desc="The answer provides the current parking fee at the San Diego Zoo for non-members",
        parent=parent_node,
        critical=True
    )

    fee_val = (extracted.parking_fee or "").strip()
    fee_srcs = extracted.parking_fee_sources or []

    # Existence + source presence check (critical)
    evaluator.add_custom_node(
        result=(len(fee_val) > 0 and len(fee_srcs) > 0),
        id="parking_fee_provided_with_sources",
        desc="A non-empty parking fee is provided and at least one supporting URL is cited for the parking fee",
        parent=fee_node,
        critical=True
    )

    # Verify the parking fee content is supported by the provided URLs (critical)
    fee_supported_leaf = evaluator.add_leaf(
        id="parking_fee_supported_by_sources",
        desc="The stated current parking fee per vehicle for non-members is supported by the cited sources",
        parent=fee_node,
        critical=True
    )

    fee_claim = f"The current parking fee per vehicle for non-members at the San Diego Zoo is '{fee_val}'."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_supported_leaf,
        sources=fee_srcs,
        additional_instruction=(
            "Verify that the provided page(s) specifically refer to the San Diego Zoo in Balboa Park (not the Safari Park) "
            "and state the current parking fee per vehicle for non-members. "
            "If the page states that parking is free, consider that equivalent to '$0'. "
            "Allow minor wording differences (e.g., 'free parking', 'no charge')."
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
) -> Dict:
    """
    Evaluate an answer for the San Diego Zoo parking/address task and return a structured summary.
    """
    # Initialize evaluator with a parallel root to mirror rubric
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

    # Create the critical top-level rubric node (since initialize() root is non-critical by design)
    top_node = evaluator.add_parallel(
        id="San_Diego_Zoo_Information",
        desc="Verify that the answer provides both the physical address and current parking fee for the San Diego Zoo",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_zoo_info(),
        template_class=ZooInfoExtraction,
        extraction_name="zoo_info_extraction"
    )

    # Build and verify subtrees
    await verify_physical_address(evaluator, top_node, extracted)
    await verify_parking_fee(evaluator, top_node, extracted)

    # Return final structured results
    return evaluator.get_summary()