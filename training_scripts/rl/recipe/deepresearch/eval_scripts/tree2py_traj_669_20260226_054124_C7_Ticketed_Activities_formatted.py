import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_indoor_arena_ada"
TASK_DESCRIPTION = (
    "Identify a major indoor arena venue located in New York City that has a seating capacity of at least 15,000 "
    "for concert events. For the venue you identify, provide the following information: (1) The venue's official name, "
    "(2) The exact seating capacity for concerts, (3) The number of wheelchair-accessible seats required by ADA standards "
    "based on the venue's capacity, (4) The venue's minimum age admission policy for events, (5) The ticket verification "
    "method(s) used at the venue, (6) Whether companion seating adjacent to wheelchair-accessible seats is available, "
    "(7) Whether tickets are available at multiple price levels, (8) Information about the venue's box office "
    "(location or availability), (9) Confirmation that accessible routes are provided for wheelchair users, and (10) "
    "A reference URL from the venue's official website or a reliable source for verification. Ensure all information "
    "provided is accurate and verifiable."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    venue_classification: Optional[str] = None  # e.g., "indoor arena", "indoor multipurpose arena"
    location_text: Optional[str] = None         # e.g., "New York, NY", "Manhattan, NYC", "Brooklyn, NY"
    concert_capacity: Optional[str] = None      # exact seating capacity for concerts, as written in answer
    ada_wheelchair_required_seats: Optional[str] = None  # number provided by the answer
    age_admission_policy: Optional[str] = None
    ticket_verification_methods: List[str] = Field(default_factory=list)  # e.g., ["QR code scanning", "RFID"]
    companion_seating_available: Optional[str] = None     # e.g., "Yes", "No", "Available", "Not available"
    multiple_price_levels: Optional[str] = None          # e.g., "Yes" / "No" / "Multiple price tiers"
    box_office_information: Optional[str] = None         # free text
    accessible_routes_available: Optional[str] = None     # free text or "Yes"/"No"
    reference_urls: List[str] = Field(default_factory=list)  # official or reliable sources


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract the venue information presented in the answer. If multiple venues are mentioned, focus on the primary one that fits the task (major indoor arena in NYC with ~15,000+ concert capacity). Extract exactly what the answer claims, without inventing any new information.

    Required fields to extract:
    1. venue_name: The official venue name, exactly as given.
    2. venue_classification: The venue type/classification as stated (e.g., "indoor arena", "indoor multipurpose arena"). If not explicitly stated, return null.
    3. location_text: The location as stated (e.g., "Manhattan, NYC", "New York, NY", "Brooklyn, NY").
    4. concert_capacity: The exact concert seating capacity number provided in the answer (the number associated with concerts), including commas if present.
    5. ada_wheelchair_required_seats: The number of wheelchair-accessible seats the answer claims are required by ADA standards based on the venue's capacity. If the answer gives a number, extract it; otherwise return null.
    6. age_admission_policy: The minimum age policy as provided (e.g., "All ages unless otherwise noted", "Varies by event", "18+").
    7. ticket_verification_methods: A list of ticket verification method(s) used (e.g., "QR code scanning", "barcodes", "RFID", "mobile tickets only"). Extract as a list of strings. If a single method is stated, return a single-element list.
    8. companion_seating_available: Whether companion seating adjacent to wheelchair-accessible seats is available (e.g., "Yes", "No", "Available", "Provided").
    9. multiple_price_levels: Whether tickets are available at multiple price levels (e.g., "Yes", "Multiple tiers", "No", "Not stated").
    10. box_office_information: Box office details (location, hours, or availability) if provided.
    11. accessible_routes_available: Confirmation that accessible routes are provided (e.g., "Yes", "Accessible routes are provided", or a short phrase).
    12. reference_urls: A list of all URLs included in the answer as sources or references. Include official venue pages or other reliable sites; return an empty list if no URLs provided.

    Rules:
    - Do not infer information not explicitly in the answer text.
    - Keep strings exactly as they appear in the answer (preserve letter case and punctuation).
    - For any missing item, return null (or [] for lists).
    """


# --------------------------------------------------------------------------- #
# Utilities: parsing & ADA calculation                                        #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\d[\d,]*", text)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except Exception:
        return None


def compute_ada_wheelchair_spaces(total_seats: int) -> int:
    """
    Implements the 2010 ADA Standards for Accessible Design, Section 221.2.1.1 (Assembly Areas — Wheelchair Spaces).
    - Up to 25: 1
    - 26–50: 2
    - 51–150: 4
    - 151–300: 5
    - 301–500: 6
    - 501–5000: 6 + 1 for each 150 seats or fraction thereof above 500
    - 5001 and over: 36 + 1 for each 200 seats or fraction thereof above 5000
    """
    if total_seats <= 25:
        return 1
    if total_seats <= 50:
        return 2
    if total_seats <= 150:
        return 4
    if total_seats <= 300:
        return 5
    if total_seats <= 500:
        return 6
    if total_seats <= 5000:
        # 6 + ceil((N - 500) / 150)
        extra = (total_seats - 500 + 150 - 1) // 150
        return 6 + extra
    # >= 5001
    extra = (total_seats - 5000 + 200 - 1) // 200
    return 36 + extra


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: VenueExtraction,
) -> None:

    # Create a critical parent node (parallel) under the evaluator root
    req_node = evaluator.add_parallel(
        id="Venue_Identification_and_Requirements",
        desc="Identifies a major indoor arena venue in New York City with at least 15,000 seating capacity for concerts and provides all required information",
        parent=evaluator.root,
        critical=True
    )

    # Convenience variables
    name = extracted.venue_name or "the venue"
    sources = extracted.reference_urls or []

    # 1) Venue_Name_Provided (critical existence)
    evaluator.add_custom_node(
        result=bool(extracted.venue_name and extracted.venue_name.strip()),
        id="Venue_Name_Provided",
        desc="The response provides the specific name of the venue",
        parent=req_node,
        critical=True
    )

    # 2) Reference_URL (critical existence)
    ref_node = evaluator.add_custom_node(
        result=bool(extracted.reference_urls and len(extracted.reference_urls) > 0),
        id="Reference_URL",
        desc="The response includes a URL reference from the venue's official website or reliable source for verification",
        parent=req_node,
        critical=True
    )

    # 3) NYC_Location
    nyc_node = evaluator.add_leaf(
        id="NYC_Location",
        desc="The venue is located in New York City",
        parent=req_node,
        critical=True
    )
    nyc_claim = (
        f"{name} is located in New York City (NYC). NYC includes the five boroughs: Manhattan, Brooklyn, Queens, "
        f"The Bronx, and Staten Island."
    )
    await evaluator.verify(
        claim=nyc_claim,
        node=nyc_node,
        sources=sources,
        additional_instruction=(
            "Verify that the venue's location is within New York City. Accept locations within any of the five boroughs "
            "(Manhattan, Brooklyn (Kings County), Queens, The Bronx, Staten Island)."
        )
    )

    # 4) Indoor_Arena_Type
    indoor_node = evaluator.add_leaf(
        id="Indoor_Arena_Type",
        desc="The venue is classified as an indoor arena",
        parent=req_node,
        critical=True
    )
    indoor_claim = f"{name} is an indoor arena (an indoor multipurpose or sports arena)."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=sources,
        additional_instruction=(
            "Determine from the provided URL(s) whether the venue is an indoor arena. "
            "Accept equivalent phrasing such as 'indoor multipurpose arena' or 'indoor stadium/arena'. "
            "Do not count outdoor stadiums as indoor arenas."
        )
    )

    # 5) Exact_Capacity_Provided (critical existence of exact concert capacity)
    capacity_provided_node = evaluator.add_custom_node(
        result=bool(extracted.concert_capacity and re.search(r"\d", extracted.concert_capacity or "")),
        id="Exact_Capacity_Provided",
        desc="The response provides the exact seating capacity number for concerts",
        parent=req_node,
        critical=True
    )

    # 6) Minimum_Capacity (>= 15,000 for concerts)
    min_capacity_node = evaluator.add_leaf(
        id="Minimum_Capacity",
        desc="The venue has seating capacity of at least 15,000 for concerts",
        parent=req_node,
        critical=True
    )
    capacity_text = extracted.concert_capacity or ""
    min_capacity_claim = (
        f"The concert seating capacity of {name} is {capacity_text}, and this is at least 15,000."
    )
    await evaluator.verify(
        claim=min_capacity_claim,
        node=min_capacity_node,
        sources=sources,
        additional_instruction=(
            "Verify from the provided URL(s) that the concert seating capacity is the stated number, and that it is "
            ">= 15,000. If multiple capacities are listed (e.g., basketball vs concert), use the concert capacity."
        )
    )

    # 7) ADA_Wheelchair_Seats (correct calculation based on capacity)
    ada_calc_node = evaluator.add_leaf(
        id="ADA_Wheelchair_Seats",
        desc="The response calculates and provides the correct number of wheelchair-accessible seats required by ADA standards for the venue's capacity",
        parent=req_node,
        critical=True
    )
    capacity_int = parse_first_int(extracted.concert_capacity)
    provided_ada_int = parse_first_int(extracted.ada_wheelchair_required_seats)
    computed_ada = compute_ada_wheelchair_spaces(capacity_int) if capacity_int else None

    if capacity_int is not None and provided_ada_int is not None and computed_ada is not None:
        ada_claim = (
            f"Given a total concert seating capacity of {capacity_int}, the number of wheelchair-accessible seating "
            f"locations required by the 2010 ADA Standards for Accessible Design (Assembly Areas, Section 221.2.1.1) "
            f"is {computed_ada}, and the answer's stated number {provided_ada_int} is correct (matches the standard)."
        )
    elif capacity_int is not None and computed_ada is not None:
        ada_claim = (
            f"Given a total concert seating capacity of {capacity_int}, the number of wheelchair-accessible seating "
            f"locations required by the 2010 ADA Standards for Accessible Design (Assembly Areas, Section 221.2.1.1) "
            f"is {computed_ada}."
        )
    else:
        ada_claim = (
            "The number of wheelchair-accessible seating locations required by ADA standards is correctly calculated "
            "based on the provided concert seating capacity."
        )

    await evaluator.verify(
        claim=ada_claim,
        node=ada_calc_node,
        sources=None,  # This is a logical check using a known standard/formula; verify correctness of calculation itself.
        additional_instruction=(
            "Use the following 2010 ADA Standards for Accessible Design, Section 221.2.1.1 (Assembly Areas — Wheelchair Spaces):\n"
            "- Up to 25: 1\n"
            "- 26–50: 2\n"
            "- 51–150: 4\n"
            "- 151–300: 5\n"
            "- 301–500: 6\n"
            "- 501–5000: 6 + 1 for each 150 seats or fraction thereof above 500\n"
            "- 5001 and over: 36 + 1 for each 200 seats or fraction thereof above 5000\n"
            "Compute the required number from the capacity and judge whether the answer's provided number (if given) matches this computed result."
        )
    )

    # 8) Age_Admission_Policy
    age_node = evaluator.add_leaf(
        id="Age_Admission_Policy",
        desc="The response provides the venue's minimum age admission policy",
        parent=req_node,
        critical=True
    )
    age_text = extracted.age_admission_policy or "Not stated"
    age_claim = f"The venue's minimum age admission policy is: {age_text}."
    await evaluator.verify(
        claim=age_claim,
        node=age_node,
        sources=sources,
        additional_instruction=(
            "Look for age-related policies on the provided URL(s). Accept statements such as 'All ages unless specified', "
            "'Varies by event', '18+ only', or similar explicit policy phrasing."
        )
    )

    # 9) Ticket_Verification_Method
    ticket_node = evaluator.add_leaf(
        id="Ticket_Verification_Method",
        desc="The response describes the ticket verification method used at the venue (e.g., barcode scanning, QR codes, RFID)",
        parent=req_node,
        critical=True
    )
    ticket_methods_text = ", ".join(extracted.ticket_verification_methods) if extracted.ticket_verification_methods else "Not stated"
    ticket_claim = f"The venue uses the following ticket verification method(s): {ticket_methods_text}."
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_node,
        sources=sources,
        additional_instruction=(
            "Verify from the provided URL(s) whether the venue verifies tickets via QR codes, barcodes, mobile tickets, "
            "RFID/NFC, or other explicit verification methods. Accept equivalent terminology (e.g., 'mobile-only tickets', "
            "'digital ticket scanning')."
        )
    )

    # 10) Companion_Seating_Availability
    companion_node = evaluator.add_leaf(
        id="Companion_Seating_Availability",
        desc="The response confirms whether companion seating adjacent to wheelchair-accessible seats is available",
        parent=req_node,
        critical=True
    )
    companion_text = extracted.companion_seating_available or "Not stated"
    companion_claim = f"Companion seating adjacent to wheelchair-accessible seating is available: {companion_text}."
    await evaluator.verify(
        claim=companion_claim,
        node=companion_node,
        sources=sources,
        additional_instruction=(
            "Check accessibility/ADA pages on the provided URL(s) for confirmation of companion seating adjacent to "
            "wheelchair spaces; look for phrases like 'companion seats available' or 'adjacent companion seating'."
        )
    )

    # 11) Multiple_Price_Levels
    price_levels_node = evaluator.add_leaf(
        id="Multiple_Price_Levels",
        desc="The response indicates whether tickets are available at multiple price levels",
        parent=req_node,
        critical=True
    )
    price_levels_text = extracted.multiple_price_levels or "Not stated"
    price_claim = f"Tickets for events at the venue are available at multiple price levels: {price_levels_text}."
    await evaluator.verify(
        claim=price_claim,
        node=price_levels_node,
        sources=sources,
        additional_instruction=(
            "Look for evidence of multiple price tiers/levels/categories in ticketing or event pages (e.g., different "
            "sections with different prices, 'price levels', or 'pricing tiers')."
        )
    )

    # 12) Box_Office_Information
    box_office_node = evaluator.add_leaf(
        id="Box_Office_Information",
        desc="The response provides information about the venue's box office location or availability",
        parent=req_node,
        critical=True
    )
    box_text = extracted.box_office_information or "Not stated"
    box_claim = f"Box office information for the venue: {box_text}."
    await evaluator.verify(
        claim=box_claim,
        node=box_office_node,
        sources=sources,
        additional_instruction=(
            "Verify information about the venue's box office from the provided URL(s): location, hours, or whether a "
            "physical box office exists or not (e.g., mobile-only ticketing)."
        )
    )

    # 13) Accessible_Routes
    accessible_routes_node = evaluator.add_leaf(
        id="Accessible_Routes",
        desc="The response confirms the provision of accessible routes for wheelchair users throughout the venue",
        parent=req_node,
        critical=True
    )
    accessible_text = extracted.accessible_routes_available or "Not stated"
    accessible_claim = f"Accessible routes for wheelchair users are provided throughout the venue: {accessible_text}."
    await evaluator.verify(
        claim=accessible_claim,
        node=accessible_routes_node,
        sources=sources,
        additional_instruction=(
            "Check the venue's ADA/accessibility pages for statements about accessible routes, elevators, ramps, or "
            "step-free access connecting entrances, seating, restrooms, and concourses."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the NYC indoor arena ADA-capacity task and return a structured summary.
    """
    # Initialize evaluator
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

    # Extract structured data from the answer
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Compute ADA spaces (for logging/debug; verification uses LLM)
    capacity_int = parse_first_int(extracted.concert_capacity)
    computed_ada = compute_ada_wheelchair_spaces(capacity_int) if capacity_int else None
    evaluator.add_custom_info(
        info={
            "parsed_concert_capacity": capacity_int,
            "computed_ada_wheelchair_spaces": computed_ada,
            "provided_ada_wheelchair_spaces_in_answer": parse_first_int(extracted.ada_wheelchair_required_seats),
            "reference_urls_count": len(extracted.reference_urls or [])
        },
        info_type="diagnostics",
        info_name="processing_notes"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()