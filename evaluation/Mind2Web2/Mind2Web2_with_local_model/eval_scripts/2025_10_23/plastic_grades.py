import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "plastic_grades"
TASK_DESCRIPTION = """
There are seven commonly used grades of plastic, each with different implications for human health. Please identify these seven types, and for each one, find an example of a bottle or box on Amazon that explicitly states it is made from that specific type of plastic in the item title or description, and return the link. Seeing actual examples would really help me understand the differences between these plastics.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}

# The 7 standard plastic types with RIC, abbreviation, and full name
PLASTICS = [
    ("1", "PET", "Polyethylene Terephthalate"),
    ("2", "HDPE", "High-Density Polyethylene"),
    ("3", "PVC", "Polyvinyl Chloride"),
    ("4", "LDPE", "Low-Density Polyethylene"),
    ("5", "PP", "Polypropylene"),
    ("6", "PS", "Polystyrene"),
    ("7", "OTHER", "Various (e.g., Polycarbonate, PLA, Nylon, Acrylic)")
]


class PlasticTypeInfo(BaseModel):
    """Information extracted for a specific plastic type"""
    ric: str = Field(description="Recycling Identification Code (1-7)")
    abbreviation: str = Field(description="Common abbreviation (PET, HDPE, etc.)")
    full_name: str = Field(description="Full chemical name")
    amazon_urls: List[str] = Field(default_factory=list, description="Amazon URLs for products made from this plastic")


def create_extraction_prompt(ric: str, abbrev: str, full_name: str) -> str:
    """Create extraction prompt for a specific plastic type"""
    return f"""
    Extract all Amazon product URLs from the answer that are specifically for plastic type:
    - RIC (Recycling Code): {ric}
    - Common abbreviation: {abbrev}
    - Full name: {full_name}

    Look for any mention of:
    - The number {ric} (as recycling code, #1, type 1, etc.)
    - The abbreviation {abbrev}
    - The full name or variations of {full_name}

    Extract ALL Amazon URLs associated with this plastic type.
    If this plastic type is not mentioned in the answer, return empty list for amazon_urls.
    """


async def verify_plastic_type(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        plastic_info: PlasticTypeInfo,
) -> None:
    """Verify Amazon products for a specific plastic type"""

    # Create node for this plastic type
    plastic_node = evaluator.add_parallel(
        id=f"plastic_{plastic_info.ric}_{plastic_info.abbreviation}",
        desc=f"Plastic Type {plastic_info.ric} ({plastic_info.abbreviation}): {plastic_info.full_name}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Check if any Amazon URLs were found for this plastic type
    has_urls_node = evaluator.add_custom_node(
        result=len(plastic_info.amazon_urls) > 0,
        id=f"plastic_{plastic_info.ric}_has_urls",
        desc=f"Amazon URLs found for {plastic_info.abbreviation}",
        parent=plastic_node,
        critical=True,  # If no URLs, can't verify anything
    )

    # if plastic_info.amazon_urls:
        # Verify that at least one URL contains a product made from this plastic
    verify_node = evaluator.add_leaf(
        id=f"plastic_{plastic_info.ric}_product_verified",
        desc=f"Amazon product explicitly states it's made from {plastic_info.abbreviation}",
        parent=plastic_node,
        critical=True,
    )

    # Create claim with all possible variations
    variations = [
        f"recycling code {plastic_info.ric}",
        f"#{plastic_info.ric}",
        f"type {plastic_info.ric}",
        plastic_info.abbreviation,
        plastic_info.full_name
    ]

    if plastic_info.ric == "7":
        variations.extend(["polycarbonate", "PC", "PLA", "nylon", "acrylic"])

    claim = f"The Amazon product page explicitly confirms that the item is a bottle or box made from plastic type {plastic_info.ric} ({plastic_info.abbreviation} - {plastic_info.full_name}). Notice, it should be a bottle or box or reasonably similar things, not just any product."

    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=plastic_info.amazon_urls,  # Will check all URLs until one verifies
        additional_instruction=f" Accept variations like: {', '.join(variations)}, or sub-types of this plastic type.",
    )


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
    """Main evaluation function for plastic grades task"""

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

    # Extract information for each of the 7 plastic types
    all_plastic_info = []

    for ric, abbrev, full_name in PLASTICS:
        # Create PlasticTypeInfo with the standard information
        plastic_info_data = {
            "ric": ric,
            "abbreviation": abbrev,
            "full_name": full_name,
            "amazon_urls": []
        }

        # Extract Amazon URLs for this specific plastic type
        extracted = await evaluator.extract(
            prompt=create_extraction_prompt(ric, abbrev, full_name),
            template_class=PlasticTypeInfo,
            extraction_name=f"plastic_{ric}_{abbrev}_extraction",
        )

        # Update with extracted URLs
        plastic_info_data["amazon_urls"] = extracted.amazon_urls
        plastic_info = PlasticTypeInfo(**plastic_info_data)

        all_plastic_info.append(plastic_info)

        # Verify this plastic type
        await verify_plastic_type(evaluator, root, plastic_info)

    # Add summary statistics
    evaluator.add_custom_info({
        "plastics_with_urls": sum(1 for p in all_plastic_info if p.amazon_urls),
        "total_urls_found": sum(len(p.amazon_urls) for p in all_plastic_info),
    }, "extraction_summary")

    return evaluator.get_summary()