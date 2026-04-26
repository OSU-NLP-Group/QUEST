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
TASK_ID = "wa_wilderness_permit_centers_2025"
TASK_DESCRIPTION = """
I am planning wilderness camping trips to Olympic National Park and North Cascades National Park in Washington State during summer 2025. I need to obtain wilderness permits for Olympic and backcountry permits for North Cascades. Identify the visitor center at each park where these permits are issued. For each of the two centers, provide: (1) the official name of the visitor center or facility, (2) the complete physical address including street address and city, and (3) the direct phone number for that facility.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityContact(BaseModel):
    """Contact info for a single permit-issuing facility."""
    name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None  # Include city, state, ZIP if present (e.g., "Port Angeles, WA 98362")
    phone: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    """Structured extraction for both parks."""
    olympic: Optional[FacilityContact] = None
    north_cascades: Optional[FacilityContact] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract the permit-issuing visitor center or facility information for the two parks as stated in the answer.
    You must extract exactly two objects:
    - olympic: The facility that issues wilderness camping permits for Olympic National Park.
    - north_cascades: The facility that issues backcountry permits for North Cascades National Park.

    For each object, extract the following fields strictly from the answer text:
    1) name: The official facility name (e.g., "Wilderness Information Center" or a specific visitor center name).
    2) street_address: The street address line (e.g., "3002 Mount Angeles Road").
    3) city: The city (and if available, state and ZIP) portion (e.g., "Port Angeles, WA 98362").
       Note: The 'city' field should include the city, and can also include state and ZIP if the answer provides them.
    4) phone: The direct phone number for that facility (e.g., "360-565-3100").
    5) sources: A list of all URLs explicitly cited in the answer that support this facility identification and/or its contact information.
       Include only actual URLs present in the answer (plain URLs or markdown links). Do not invent URLs.

    If any field is missing in the answer for a park, return null for that field.
    If no sources are cited for a park, return an empty list for sources.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def build_address_string(street_address: Optional[str], city: Optional[str]) -> str:
    """Combine street address and city/state/ZIP into a single address string."""
    parts: List[str] = []
    if street_address and street_address.strip():
        parts.append(street_address.strip())
    if city and city.strip():
        parts.append(city.strip())
    return ", ".join(parts).strip()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_facility(
    evaluator: Evaluator,
    parent_node,
    facility: FacilityContact,
    prefix: str,
    park_display_name: str,
    permit_term: str,
) -> None:
    """
    Build and run verification for one park's permit-issuing facility.

    Args:
        evaluator: Evaluator instance.
        parent_node: The sequential parent node under root for this park.
        facility: Extracted FacilityContact data for this park.
        prefix: ID prefix ("olympic" or "noca").
        park_display_name: Human-readable park name ("Olympic National Park" or "North Cascades National Park").
        permit_term: "wilderness camping permits" (Olympic) or "backcountry permits" (North Cascades).
    """
    # Identification (leaf, critical)
    ident_leaf = evaluator.add_leaf(
        id=f"{prefix}_identification",
        desc=f"The correct facility name that issues {permit_term} at {park_display_name} is provided",
        parent=parent_node,
        critical=True,
    )
    facility_name = (facility.name or "").strip()
    ident_claim = (
        f"The permit-issuing facility for {park_display_name} is '{facility_name}', "
        f"and it issues {permit_term}."
    )
    await evaluator.verify(
        claim=ident_claim,
        node=ident_leaf,
        sources=facility.sources if facility and facility.sources else None,
        additional_instruction=(
            "Verify both the facility's official name and that it is the designated office/visitor center "
            f"for obtaining {permit_term} (walk-in or in-person). Accept minor naming variations or abbreviations."
        ),
    )

    # Contact info (parallel, critical)
    contact_node = evaluator.add_parallel(
        id=f"{prefix}_contact_info",
        desc=f"Complete and accurate contact information for {park_display_name}'s permit-issuing facility is provided",
        parent=parent_node,
        critical=True,
    )

    # Address leaf (critical)
    address_leaf = evaluator.add_leaf(
        id=f"{prefix}_address",
        desc="Complete physical address with street address and city is provided and matches official records",
        parent=contact_node,
        critical=True,
    )
    combined_address = build_address_string(facility.street_address if facility else None,
                                            facility.city if facility else None)
    address_claim = (
        f"The physical address of '{facility_name}' is '{combined_address}'."
    )
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=facility.sources if facility and facility.sources else None,
        additional_instruction=(
            "Check the official NPS facility page or contact page. Allow minor formatting differences "
            "(e.g., 'Road' vs 'Rd', punctuation, capitalization). The city/state/ZIP can appear within the city field. "
            "Consider equivalent abbreviations as matches."
        ),
    )

    # Phone leaf (critical)
    phone_leaf = evaluator.add_leaf(
        id=f"{prefix}_phone",
        desc="Direct phone number is provided and matches official records",
        parent=contact_node,
        critical=True,
    )
    phone_number = (facility.phone or "").strip()
    phone_claim = (
        f"The direct phone number for '{facility_name}' is '{phone_number}'."
    )
    await evaluator.verify(
        claim=phone_claim,
        node=phone_leaf,
        sources=facility.sources if facility and facility.sources else None,
        additional_instruction=(
            "Match the digits of the phone number from official sources. Allow formatting variants like "
            "spaces, hyphens, or parentheses (e.g., 360-565-3100 vs (360) 565-3100)."
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
    Evaluate an answer for identifying permit-issuing visitor centers and their contact details.
    """
    # Initialize evaluator with parallel root per rubric
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify two wilderness/backcountry permit-issuing visitor centers (one at Olympic NP, one at North Cascades NP) with complete contact information",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=ParksExtraction,
        extraction_name="permit_facilities",
    )

    # Build park nodes (sequential per rubric)
    olympic_node = evaluator.add_sequential(
        id="olympic_permit_center",
        desc="Olympic National Park wilderness permit-issuing facility correctly identified with complete contact information",
        parent=root,
        critical=False,
    )
    noca_node = evaluator.add_sequential(
        id="north_cascades_permit_center",
        desc="North Cascades National Park backcountry permit-issuing facility correctly identified with complete contact information",
        parent=root,
        critical=False,
    )

    # Verify Olympic National Park facility
    await verify_facility(
        evaluator=evaluator,
        parent_node=olympic_node,
        facility=extraction.olympic or FacilityContact(),
        prefix="olympic",
        park_display_name="Olympic National Park",
        permit_term="wilderness camping permits",
    )

    # Verify North Cascades National Park facility
    await verify_facility(
        evaluator=evaluator,
        parent_node=noca_node,
        facility=extraction.north_cascades or FacilityContact(),
        prefix="noca",
        park_display_name="North Cascades National Park",
        permit_term="backcountry permits",
    )

    # Return final summary with verification tree
    return evaluator.get_summary()