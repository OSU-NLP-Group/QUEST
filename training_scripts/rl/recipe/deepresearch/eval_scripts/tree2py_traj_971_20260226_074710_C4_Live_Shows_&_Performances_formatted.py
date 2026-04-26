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
TASK_ID = "il_largest_indoor_concert_venue"
TASK_DESCRIPTION = """
Identify the largest indoor concert venue in Illinois by concert capacity. Provide the following information:
(1) The venue's official name,
(2) The city where the venue is located,
(3) The venue's concert capacity (seating number),
(4) A reference URL that verifies this information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured extraction for the venue task from the agent's answer."""
    venue_name: Optional[str] = None
    city: Optional[str] = None
    concert_capacity: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract the requested details about the largest indoor concert venue in Illinois as presented in the answer.

    Required fields:
    1) venue_name: The official name of the venue that the answer claims is the largest indoor concert venue in Illinois by concert capacity.
    2) city: The city where the venue is located (the city string exactly as written in the answer; it may include the state or country—do not modify).
    3) concert_capacity: The concert capacity number (or phrase) exactly as written in the answer. If multiple numbers are given in the answer (e.g., basketball/hockey vs. concerts), select the value the answer cites for concerts or the maximum capacity used for concerts. Return it as a string exactly as appears in the answer (e.g., "23,500", "about 23,500", "up to 23,500 for concerts").
    4) reference_urls: All URLs the answer cites to support this venue and its details (capacity and/or location). Include every URL you can find in the answer. The URLs can be in plain form or markdown links; extract the actual link URLs. If none are present, return an empty list.

    If any field is missing in the answer, set it to null (or empty list for reference_urls).
    Do not invent or infer any values that are not explicitly stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: List[str]) -> List[str]:
    """Basic normalization of URL list: strip, drop empties, de-duplicate (preserve order)."""
    seen = set()
    cleaned: List[str] = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            cleaned.append(uu)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_venue(evaluator: Evaluator, extracted: VenueExtraction) -> None:
    """
    Build the verification tree according to the rubric and run all checks.
    The rubric's top node is CRITICAL and aggregates four CRITICAL leaves in parallel.
    """
    # Create the rubric's main node (critical, parallel)
    main_node = evaluator.add_parallel(
        id="Identify_Illinois_Largest_Indoor_Concert_Venue",
        desc="Correctly identify the largest indoor concert venue in Illinois by concert capacity, and provide all required supporting information",
        parent=evaluator.root,
        critical=True
    )

    # Normalize URLs once
    urls = _normalize_urls(extracted.reference_urls)

    # ----------------------------- Venue Name ----------------------------- #
    venue_name_leaf = evaluator.add_leaf(
        id="Venue_Name",
        desc="Provide the correct name of the largest indoor concert venue in Illinois by concert capacity",
        parent=main_node,
        critical=True
    )

    venue_name = extracted.venue_name or ""
    claim_name = (
        f"The largest indoor concert venue in Illinois by concert capacity is '{venue_name}'."
    )
    add_ins_name = (
        "Judge this claim strictly using only the provided URL(s). "
        "Accept the claim if at least one cited page explicitly states that this venue is the largest indoor concert venue in Illinois by concert capacity, "
        "or if the page presents a list/table of indoor venues in Illinois with capacities that clearly indicates this venue has the highest concert capacity among them. "
        "Allow minor variations in the venue name (e.g., sponsor naming or punctuation). "
        "If there are no provided URLs, or if no provided page supports the 'largest indoor concert venue in Illinois by concert capacity' claim, return Incorrect."
    )
    # If no URLs, the verify() will route to simple verification; force a fail via instruction.
    await evaluator.verify(
        claim=claim_name,
        node=venue_name_leaf,
        sources=urls if urls else None,
        additional_instruction=add_ins_name
    )

    # ----------------------------- Venue Location ----------------------------- #
    location_leaf = evaluator.add_leaf(
        id="Venue_Location",
        desc="Provide the correct city location where the venue is situated",
        parent=main_node,
        critical=True
    )

    city_text = extracted.city or ""
    claim_location = f"The venue '{venue_name}' is located in {city_text}."
    add_ins_location = (
        "Use only the cited URL(s). The page should clearly indicate the venue's location including the city name (e.g., 'Chicago' or 'Chicago, Illinois, United States'). "
        "A page passes if the city string in the claim is evidently mentioned as the venue's location on that page (reasonable variants acceptable). "
        "If no URLs are provided, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_location,
        node=location_leaf,
        sources=urls if urls else None,
        additional_instruction=add_ins_location
    )

    # ----------------------------- Concert Capacity ----------------------------- #
    capacity_leaf = evaluator.add_leaf(
        id="Concert_Capacity",
        desc="Provide the accurate concert capacity (seating number) of the identified venue",
        parent=main_node,
        critical=True
    )

    capacity_text = extracted.concert_capacity or ""
    claim_capacity = f"The concert capacity of the venue '{venue_name}' is {capacity_text}."
    add_ins_capacity = (
        "Verify using only the cited URL(s). Prefer a capacity number explicitly labeled as 'concert capacity', 'for concerts', or 'maximum capacity suitable for concerts'. "
        "If multiple capacities are listed for different configurations (e.g., basketball, hockey, concerts), accept the claim only if the concerts figure matches the claimed value (allow minor rounding). "
        "If the page provides a single 'seating capacity' but also indicates it serves as the concert capacity, that is acceptable. "
        "If no URLs are provided, or the pages do not support the concerts capacity figure, return Incorrect."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=urls if urls else None,
        additional_instruction=add_ins_capacity
    )

    # ----------------------------- Reference URL ----------------------------- #
    reference_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provide a valid and accessible reference URL that verifies the venue's capacity and location information",
        parent=main_node,
        critical=True
    )

    claim_reference = (
        f"At least one of the provided URLs contains both the venue's location (city) and the concert capacity (or an explicitly labeled concerts capacity) for '{venue_name}'."
    )
    add_ins_reference = (
        "Pass this verification if a single page among the provided URLs clearly states the city where the venue is located and also shows a concerts capacity (or an equivalent maximum capacity used for concerts). "
        "The page must be accessible and relevant to the venue. "
        "If no URLs are provided or no single page contains both the city/location and the concerts capacity, return Incorrect."
    )
    await evaluator.verify(
        claim=claim_reference,
        node=reference_leaf,
        sources=urls if urls else None,
        additional_instruction=add_ins_reference
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
    Evaluate an answer for the Illinois largest indoor concert venue task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator; rubric's main node added as child
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
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build verification tree and run checks
    await verify_venue(evaluator, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()