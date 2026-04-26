import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "iclr_2025_venue"
TASK_DESCRIPTION = "What is the name of the venue facility hosting the ICLR 2025 conference?"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract the venue facility name and any cited reference URLs from the answer.

    Required fields:
    - venue_name: The proper name of the physical venue facility (e.g., "Vancouver Convention Centre", "Suntec Singapore Convention & Exhibition Centre") where ICLR 2025 is held. 
      Do NOT return only the city/country or a generic location; return the facility/building name if provided.
    - reference_urls: A list of all URLs cited in the answer that are intended to confirm the venue.
      Include all URLs explicitly present in the answer text (including markdown links), even if non-official.
      Do not invent URLs. If no URLs are provided, return an empty list.

    If multiple venue names appear, choose the one that is explicitly presented as the official venue facility hosting ICLR 2025.
    If the answer does not provide a venue facility name, set venue_name to null.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: VenueExtraction) -> None:
    """
    Build the verification tree and run checks for the ICLR 2025 venue task.
    """
    root = evaluator.root

    # Parent node matching rubric: critical parallel aggregation
    iclr_node = evaluator.add_parallel(
        id="ICLR_2025_Venue",
        desc="Identifies the venue facility name for ICLR 2025 conference",
        parent=root,
        critical=True
    )

    # 1) Venue name correctness check (critical)
    if extraction.venue_name and extraction.venue_name.strip():
        venue_leaf = evaluator.add_leaf(
            id="Venue_Name",
            desc="Provides the correct venue facility name where ICLR 2025 is held",
            parent=iclr_node,
            critical=True
        )

        venue_claim = f"The venue facility hosting the ICLR 2025 conference is '{extraction.venue_name}'."
        await evaluator.verify(
            claim=venue_claim,
            node=venue_leaf,
            sources=extraction.reference_urls if extraction.reference_urls else None,
            additional_instruction=(
                "Verify against the provided page(s) that the stated facility name is exactly the venue building hosting ICLR 2025. "
                "Allow minor naming variants (e.g., hyphenation, abbreviations, or local language versions) but ensure it is the facility name, "
                "not merely the city or country."
            )
        )
    else:
        # No venue name provided -> fail this critical leaf
        evaluator.add_custom_node(
            result=False,
            id="Venue_Name",
            desc="Provides the correct venue facility name where ICLR 2025 is held",
            parent=iclr_node,
            critical=True
        )

    # 2) Reference URL from official ICLR sources confirming the venue (critical)
    # If there are no URLs at all, this must fail (since the rubric requires a reference URL).
    if extraction.reference_urls:
        ref_leaf = evaluator.add_leaf(
            id="Reference_URL",
            desc="Provides a reference URL from official ICLR 2025 sources confirming the venue",
            parent=iclr_node,
            critical=True
        )

        # Craft additional instruction to enforce "official" requirement
        venue_hint = (
            f"The answer-provided venue name is '{extraction.venue_name}'. "
            "If a venue name is provided, ensure the page explicitly mentions this same facility (allow minor naming variants). "
            if extraction.venue_name else
            "Confirm that the page explicitly states the venue facility hosting ICLR 2025 (not just the city/country). "
        )
        add_ins = (
            "Treat a source as 'official ICLR 2025' only if the URL is on the 'iclr.cc' domain or its subdomains and the page clearly pertains to the 2025 ICLR conference. "
            + venue_hint +
            "Fail if the URL is not on iclr.cc (or its subdomains), or if the page does not explicitly confirm the venue facility."
        )

        # Multi-URL verification: pass if any provided URL satisfies the claim and constraints
        ref_claim = (
            "This page is an official ICLR 2025 source (hosted on iclr.cc or its subdomains) and it explicitly confirms the venue facility hosting ICLR 2025."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=extraction.reference_urls,
            additional_instruction=add_ins
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Reference_URL",
            desc="Provides a reference URL from official ICLR 2025 sources confirming the venue",
            parent=iclr_node,
            critical=True
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
    Evaluate an answer for the task: Identify the venue facility hosting ICLR 2025.
    """
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
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Build verification checks based on extraction
    await build_verification_tree(evaluator, extraction)

    # Return final summary
    return evaluator.get_summary()