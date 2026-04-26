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
TASK_ID = "tx_music_venue_3k_5k"
TASK_DESCRIPTION = (
    "Identify a music venue located in Texas that has a seating/standing capacity between 3,000 and 5,000 people "
    "(inclusive) and is designed for hosting live music concerts. Provide the venue's official name, the specific city "
    "where it is located in Texas, its exact capacity, whether it is an indoor or outdoor venue, its address or location "
    "details, and a reference URL that verifies this information. The venue must be currently operational and available "
    "for booking concert events."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured extraction of a single Texas music venue from the agent's answer."""
    name: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    capacity_text: Optional[str] = None
    capacity_number: Optional[int] = None
    venue_function_text: Optional[str] = None
    address: Optional[str] = None
    venue_type: Optional[str] = None  # e.g., "indoor", "outdoor", "mixed", "unknown"
    operational_status_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return (
        "Extract a single venue described in the answer that matches ALL of the following requirements:\n"
        "- Located in the U.S. state of Texas\n"
        "- Seating/standing capacity between 3,000 and 5,000 inclusive\n"
        "- Designed for or regularly hosts live music concerts\n"
        "- Currently operational and available for booking concert events\n\n"
        "Return the following fields:\n"
        "1) name: The official venue name as stated in the answer.\n"
        "2) state: The state where the venue is located (should be 'Texas' or 'TX' if applicable).\n"
        "3) city: The specific Texas city mentioned in the answer.\n"
        "4) capacity_text: The capacity string exactly as mentioned (e.g., '4,500', 'approx. 3,200', 'up to 5,000').\n"
        "5) capacity_number: If the answer provides an exact single number for capacity, extract it as an integer; "
        "   otherwise return null. If a range is provided, choose the main/typical capacity if clearly indicated; "
        "   otherwise return null.\n"
        "6) venue_function_text: A brief phrase from the answer indicating the venue hosts live music concerts.\n"
        "7) address: The street address or location details if provided in the answer.\n"
        "8) venue_type: 'indoor', 'outdoor', or 'mixed' (amphitheater/pavilion typically 'outdoor'). "
        "   If unclear, use 'unknown'.\n"
        "9) operational_status_text: A short phrase showing the venue is currently operational and accepts bookings "
        "   or has upcoming concerts/events.\n"
        "10) reference_urls: All publicly accessible URLs mentioned in the answer that can verify the above details. "
        "    Include official venue site pages, ticketing pages, event calendars, Wikipedia pages, city/municipal pages, "
        "    or reputable listings. Extract actual URLs only.\n\n"
        "If multiple venues are listed, choose the first that appears to meet the requirements. If any field is not "
        "present in the answer, return null for that field (or empty list for reference_urls)."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def add_venue_identification_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: VenueExtraction,
) -> None:
    """
    Add verification checks under the 'venue_identification' node.
    This node is critical and uses parallel aggregation of its children.
    """
    ident_node = evaluator.add_parallel(
        id="venue_identification",
        desc="Identify a qualifying venue meeting geographic, capacity, and concert-venue requirements",
        parent=parent_node,
        critical=True,
    )

    # Optional early gate: Ensure at least one reference URL is provided before attempting URL-based verifications.
    has_sources = bool(extracted.reference_urls)
    evaluator.add_custom_node(
        result=has_sources,
        id="sources_provided",
        desc="At least one reference URL is provided in the answer to verify venue information",
        parent=ident_node,
        critical=True,
    )

    # 1) Venue name
    name_node = evaluator.add_leaf(
        id="venue_name",
        desc="Provide the official name of the venue",
        parent=ident_node,
        critical=True,
    )
    name_claim = f"The official name of the venue is '{(extracted.name or '').strip()}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Verify the official venue name using the provided URL(s). Allow minor formatting differences, punctuation, "
            "and casing variations. If the claim lacks a clear name in the answer or the URL(s) do not corroborate it, "
            "mark as incorrect."
        ),
    )

    # 2) Geographic location - state must be Texas
    state_node = evaluator.add_leaf(
        id="geographic_location_state",
        desc="The venue must be located in Texas",
        parent=ident_node,
        critical=True,
    )
    state_val = (extracted.state or "").strip()
    state_claim = (
        "The venue is located in the U.S. state of Texas."
        if state_val
        else "The venue is located in the U.S. state of Texas."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_node,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm via the URL(s) that the venue is in Texas (TX). Terms like 'TX' or 'Texas' should be treated as "
            "equivalent for the state. If the page indicates a different state or does not indicate Texas, mark as incorrect."
        ),
    )

    # 3) Geographic location - specific city in Texas
    city_node = evaluator.add_leaf(
        id="geographic_location_city",
        desc="Provide the specific city in Texas where the venue is located",
        parent=ident_node,
        critical=True,
    )
    city_val = (extracted.city or "").strip()
    city_claim = f"The venue is located in {city_val}, Texas."
    await evaluator.verify(
        claim=city_claim,
        node=city_node,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm the exact Texas city using the provided URL(s). Allow minor variations (e.g., 'Ft. Worth' vs "
            "'Fort Worth'). If the city is not clearly indicated or is outside Texas, mark as incorrect."
        ),
    )

    # 4) Capacity requirement (exact capacity and ensure within 3,000–5,000 inclusive)
    capacity_node = evaluator.add_leaf(
        id="capacity_requirement",
        desc="Provide the venue's exact capacity and ensure it is between 3,000 and 5,000 people (inclusive)",
        parent=ident_node,
        critical=True,
    )
    cap_text = (extracted.capacity_text or "").strip()
    cap_num = extracted.capacity_number
    if cap_num is not None:
        capacity_claim = (
            f"The venue's capacity is {cap_num} and it lies between 3,000 and 5,000 inclusive."
        )
    else:
        capacity_claim = (
            f"The venue's capacity is '{cap_text}' and it lies between 3,000 and 5,000 inclusive."
        )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Check the page(s) for a stated capacity number (seating, standing, or total capacity). If a range or "
            "approximation is listed, judge whether the typical capacity falls within 3,000–5,000 inclusive. If the "
            "page contradicts the range or does not provide capacity, mark as incorrect."
        ),
    )

    # 5) Venue function (concert venue)
    function_node = evaluator.add_leaf(
        id="venue_function",
        desc="The venue must be designed for or regularly host live music concerts and performances",
        parent=ident_node,
        critical=True,
    )
    venue_func_claim = (
        "The venue is designed for or regularly hosts live music concerts and performances."
    )
    await evaluator.verify(
        claim=venue_func_claim,
        node=function_node,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm via the URL(s) that the venue regularly hosts live music concerts or performances (e.g., event "
            "listings, show calendars, concert descriptions). If the venue is primarily non-music or the pages do not "
            "indicate music concerts, mark as incorrect."
        ),
    )


async def add_venue_detail_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: VenueExtraction,
) -> None:
    """
    Add verification checks under the 'venue_details' node.
    This node is critical and uses parallel aggregation of its children.
    """
    details_node = evaluator.add_parallel(
        id="venue_details",
        desc="Provide required venue details (type, address/location, and operational/booking status)",
        parent=parent_node,
        critical=True,
    )

    # Address / location details
    address_node = evaluator.add_leaf(
        id="venue_address",
        desc="Provide the street address or location details of the venue",
        parent=details_node,
        critical=True,
    )
    addr_val = (extracted.address or "").strip()
    address_claim = f"The venue's address or location details are: {addr_val}."
    await evaluator.verify(
        claim=address_claim,
        node=address_node,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Verify the address or clear location details from the provided URL(s). Accept either a full street address "
            "or an unambiguous location description on the venue's page."
        ),
    )

    # Venue type (indoor / outdoor)
    type_node = evaluator.add_leaf(
        id="venue_type",
        desc="Specify whether the venue is indoor or outdoor",
        parent=details_node,
        critical=True,
    )
    type_val = (extracted.venue_type or "").strip().lower()
    if type_val not in ("indoor", "outdoor", "mixed", "unknown"):
        type_val = "unknown"
    type_claim = f"The venue is {type_val}."
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Determine indoor/outdoor from the page(s). Amphitheaters, open-air pavilions typically count as 'outdoor'. "
            "If both configurations exist or unclear, 'mixed' or 'unknown' may be acceptable only if the page supports it."
        ),
    )

    # Operational status and booking availability
    operational_node = evaluator.add_leaf(
        id="operational_status",
        desc="Confirm the venue is currently operational and available for booking/hosting concert events",
        parent=details_node,
        critical=True,
    )
    operational_claim = (
        "The venue is currently operational and available for booking concert events."
    )
    await evaluator.verify(
        claim=operational_claim,
        node=operational_node,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Look for evidence such as upcoming events/calendar, a 'Book Now' or 'Rent the venue' page, or explicit "
            "statements of ongoing operations. If the page indicates closures or no booking capability, mark as incorrect."
        ),
    )


async def add_reference_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: VenueExtraction,
) -> None:
    """
    Add the final reference documentation verification.
    The JSON rubric treats this as a single critical leaf.
    """
    ref_node = evaluator.add_leaf(
        id="reference_documentation",
        desc="Provide at least one publicly accessible reference URL that verifies the provided venue information",
        parent=parent_node,
        critical=True,
    )

    # General verification claim leveraging multi-URL support
    # This validates that at least one provided URL supports core details (name, location in Texas, city, capacity range).
    core_name = (extracted.name or "").strip()
    core_city = (extracted.city or "").strip()
    core_cap_text = (extracted.capacity_text or "").strip()
    core_cap_num = extracted.capacity_number

    cap_frag = core_cap_text if core_cap_num is None else str(core_cap_num)
    general_claim = (
        f"The provided URL(s) publicly verify that the venue named '{core_name}' is located in {core_city}, Texas, "
        f"and has a capacity '{cap_frag}' that lies between 3,000 and 5,000 inclusive."
    )

    await evaluator.verify(
        claim=general_claim,
        node=ref_node,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Evaluate whether at least one of the provided URLs explicitly supports the venue name, Texas location "
            "(including the specific city), and a capacity that falls within 3,000–5,000 inclusive."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an agent's answer for the Texas music venue capacity task.
    Builds a hierarchical verification tree and returns a structured summary.
    """
    # Initialize the evaluator and root container
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Top-level critical sequential node that mirrors the rubric root
    task_main = evaluator.add_sequential(
        id="task_main",
        desc="Identify and verify a music venue in Texas with capacity between 3,000 and 5,000 people that hosts live music concerts, and provide the required venue details with a verifying URL.",
        parent=root,
        critical=True,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build verification tree according to rubric order
    await add_venue_identification_checks(evaluator, task_main, extracted)
    await add_venue_detail_checks(evaluator, task_main, extracted)
    await add_reference_checks(evaluator, task_main, extracted)

    # Return structured result with verification tree and scores
    return evaluator.get_summary()