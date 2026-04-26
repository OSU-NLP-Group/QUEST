import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "united_center_concert_capacity"
TASK_DESCRIPTION = "What is the seating capacity for concerts at the United Center in Chicago?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CapacityExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer regarding
    the United Center concert seating capacity.
    """
    capacity_value: Optional[str] = None          # e.g., "23,500", "approximately 23,500"
    capacity_context: Optional[str] = None        # e.g., "for concerts", "concert configuration"
    supporting_urls: List[str] = Field(default_factory=list)  # URLs cited in the answer to support the capacity claim


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_capacity_info() -> str:
    return """
    Extract the seating capacity information for CONCERTS at the United Center in Chicago from the provided answer text.

    Return the following fields:
    1) capacity_value: The numeric capacity value the answer claims for concerts at the United Center, exactly as written (keep commas or modifiers like "about", "up to", etc.). If multiple values are mentioned, pick the one explicitly tied to concerts. If none is stated, return null.
    2) capacity_context: The short phrase or sentence fragment from the answer that makes it clear the capacity is specifically for concert configuration (e.g., "for concerts", "concert configuration", "end-stage concert", "center-stage concert"). If not clearly specified, return null.
    3) supporting_urls: All URLs explicitly cited in the answer that are intended to support the capacity information (e.g., official venue pages, event guides, PDFs). Include only actual URLs found in the answer. If none are provided, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_numeric(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(ch.isdigit() for ch in s)


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_capacity_info(
    evaluator: Evaluator,
    parent_node,
    extracted: CapacityExtraction,
) -> None:
    """
    Build the rubric tree specified by the JSON and run the corresponding checks.

    Rubric nodes:
    - United_Center_Concert_Capacity_Information (parallel, non-critical)
        - Capacity_Value_Provided (critical)
        - Concert_Configuration_Specified (critical)
        - Official_Source_Referenced (critical)
    """

    # Parent node for this rubric section
    section_node = evaluator.add_parallel(
        id="United_Center_Concert_Capacity_Information",
        desc="Verify that the seating capacity for concerts at the United Center in Chicago is correctly identified and properly sourced.",
        parent=parent_node,
        critical=False
    )

    # 1) Capacity_Value_Provided (Critical)
    cap_value_provided = evaluator.add_custom_node(
        result=_has_numeric(extracted.capacity_value),
        id="Capacity_Value_Provided",
        desc="A specific numerical seating capacity value for concert events at the United Center is provided in the answer.",
        parent=section_node,
        critical=True
    )

    # 2) Concert_Configuration_Specified (Critical)
    concert_config_leaf = evaluator.add_leaf(
        id="Concert_Configuration_Specified",
        desc="The capacity value provided is explicitly stated to be for concert configuration (not for other event types such as basketball or hockey).",
        parent=section_node,
        critical=True
    )

    # Claim checks the answer text itself; no external URLs required
    if extracted.capacity_value and _has_numeric(extracted.capacity_value):
        config_claim = (
            f"The answer explicitly states that the seating capacity {extracted.capacity_value} applies to concerts at the United Center "
            f"(i.e., concert configuration), not to basketball, hockey, or generic seating."
        )
    else:
        config_claim = (
            "The answer explicitly states that the seating capacity applies to concerts at the United Center "
            "(i.e., concert configuration), not to basketball, hockey, or generic seating."
        )
    await evaluator.verify(
        claim=config_claim,
        node=concert_config_leaf,
        additional_instruction=(
            "Examine ONLY the provided answer text. Look for clear signals like 'concert', 'concert configuration', "
            "'end‑stage concert', 'center‑stage concert', or similar. The statement must make it explicit that the capacity "
            "is for concerts, not just a general or sports-specific capacity. Minor wording variations are acceptable."
        ),
    )

    # 3) Official_Source_Referenced (Critical)
    official_src_leaf = evaluator.add_leaf(
        id="Official_Source_Referenced",
        desc="The capacity information is supported by a reference to an official United Center source or authoritative venue documentation.",
        parent=section_node,
        critical=True
    )

    # If there are no URLs in the answer at all, this check must fail immediately
    if not extracted.supporting_urls:
        official_src_leaf.score = 0.0
        official_src_leaf.status = "failed"
    else:
        # Build the source-grounded claim. If we have a value, check the specific value (allowing small variations).
        if extracted.capacity_value and _has_numeric(extracted.capacity_value):
            src_claim = (
                f"At least one of the cited webpages is an official United Center source or clearly authoritative venue documentation, "
                f"and it explicitly states that the United Center's CONCERT seating capacity is approximately {extracted.capacity_value}."
            )
        else:
            src_claim = (
                "At least one of the cited webpages is an official United Center source or clearly authoritative venue documentation, "
                "and it explicitly states the United Center's seating capacity for concerts."
            )

        await evaluator.verify(
            claim=src_claim,
            node=official_src_leaf,
            sources=extracted.supporting_urls,
            additional_instruction=(
                "Only mark this as supported if at least one provided URL is an OFFICIAL venue source or an equally authoritative "
                "venue document. Preferred: unitedcenter.com domain (including its asset/CDN subdomains) or an official venue PDF/guide. "
                "Also acceptable: a government/municipal document or a clearly official event/venue manual that explicitly provides "
                "the concert seating capacity for the United Center. The page must explicitly mention concert capacity (allowing phrases "
                "like 'for concerts', 'concert capacity', 'up to X for concerts', 'end‑stage concert', 'center‑stage concert'). "
                "Allow minor rounding/formatting differences (commas, 'about', 'approximately', 'up to'). If the page is unrelated, "
                "generic, or clearly unofficial (e.g., wiki-style aggregators without official sourcing), do not mark as supported."
            )
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
    Entry point for evaluating the agent's answer for the United Center concert capacity task.
    """
    # Initialize evaluator with a parallel root (single sub-task)
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

    # Extract capacity information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_capacity_info(),
        template_class=CapacityExtraction,
        extraction_name="capacity_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify_capacity_info(
        evaluator=evaluator,
        parent_node=root,
        extracted=extracted
    )

    # Return standard evaluation summary
    return evaluator.get_summary()