import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "avelo_hvn_florida_gulf_coast"
TASK_DESCRIPTION = (
    "Identify the Florida Gulf Coast beach destination that is served by nonstop Avelo Airlines flights "
    "from Tweed New Haven Airport (HVN) in Connecticut. Provide the following information: "
    "(1) the destination airport's three-letter IATA code, (2) confirmation that this is a nonstop flight service, "
    "(3) confirmation that Avelo Airlines operates this route, (4) confirmation of the departure airport, "
    "(5) Avelo Airlines' free personal item maximum dimensions, and (6) Avelo Airlines' carry-on bag fee range based on booking timing."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RouteBaggageExtraction(BaseModel):
    # Destination info (exactly as stated in the answer)
    destination_name: Optional[str] = None
    destination_airport_name: Optional[str] = None
    destination_iata: Optional[str] = None

    # Departure info (as stated)
    departure_airport_name: Optional[str] = None
    departure_iata: Optional[str] = None

    # Route confirmations (as stated in the answer, not verified yet)
    nonstop_text: Optional[str] = None
    operated_by_avelo_text: Optional[str] = None
    operational_as_of_nov_2025_text: Optional[str] = None

    # Baggage policy details (as stated)
    personal_item_dimensions: Optional[str] = None
    carry_on_fee_range: Optional[str] = None

    # Source URLs mentioned in the answer (grouped by purpose)
    route_sources: List[str] = Field(default_factory=list)
    baggage_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_route_baggage_info() -> str:
    return (
        "Extract the Florida Gulf Coast destination and Avelo route details as presented in the answer, "
        "along with Avelo baggage policy details. Return the following fields:\n"
        "1) destination_name: The city or destination name (e.g., Fort Myers, Sarasota).\n"
        "2) destination_airport_name: The destination airport name (e.g., Southwest Florida International Airport).\n"
        "3) destination_iata: The destination airport IATA code (three letters, e.g., RSW, SRQ, TPA).\n"
        "4) departure_airport_name: The departure airport name (should be Tweed New Haven Airport).\n"
        "5) departure_iata: The departure airport IATA code (should be HVN).\n"
        "6) nonstop_text: Any wording in the answer confirming it is a nonstop/direct flight.\n"
        "7) operated_by_avelo_text: Any wording confirming Avelo Airlines operates the route.\n"
        "8) operational_as_of_nov_2025_text: Any wording confirming the route is operational as of November 2025.\n"
        "9) personal_item_dimensions: Avelo's free personal item maximum dimensions as stated (e.g., '17 in (L) x 13 in (H) x 9 in (W)').\n"
        "10) carry_on_fee_range: Avelo's carry-on bag fee range by booking timing as stated (e.g., '$45 to $77 per bag').\n"
        "11) route_sources: All URLs in the answer that specifically support the route details (origin/destination, airline, nonstop, operational status).\n"
        "12) baggage_sources: All URLs in the answer that specifically support Avelo baggage policy details (personal item dimensions, carry-on fees).\n"
        "If any piece of information is not mentioned, set it to null. For sources, extract only actual URLs present in the answer (including markdown links), return an empty list if none."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_three_letter_iata(code: Optional[str]) -> bool:
    if not code:
        return False
    return re.fullmatch(r"[A-Za-z]{3}", code.strip()) is not None


def label_destination(ex: RouteBaggageExtraction) -> str:
    # Build a readable destination label from extracted fields
    parts = []
    if ex.destination_name:
        parts.append(ex.destination_name.strip())
    if ex.destination_airport_name:
        parts.append(ex.destination_airport_name.strip())
    if ex.destination_iata:
        parts.append(ex.destination_iata.strip().upper())
    return " / ".join(parts) if parts else "the destination"


def route_urls(ex: RouteBaggageExtraction) -> List[str]:
    return ex.route_sources or []


def baggage_urls(ex: RouteBaggageExtraction) -> List[str]:
    return ex.baggage_sources or []


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_route_group(evaluator: Evaluator, parent_node, ex: RouteBaggageExtraction) -> None:
    # Create a critical sequential node for route checks (gates downstream verifications)
    route_main = evaluator.add_sequential(
        id="Route_Main",
        desc="Route verification group (HVN → Florida Gulf Coast destination)",
        parent=parent_node,
        critical=True
    )

    # Existence gate: Destination IATA code provided (critical as per rubric)
    iata_provided = evaluator.add_custom_node(
        result=is_three_letter_iata(ex.destination_iata),
        id="Destination_Airport_IATA_Code_Provided",
        desc="Provides the destination airport's three-letter IATA code.",
        parent=route_main,
        critical=True
    )

    # Destination qualifies as Florida Gulf Coast beach destination
    dest_is_gulf_leaf = evaluator.add_leaf(
        id="Destination_Is_Florida_Gulf_Coast_Beach_Destination",
        desc="The identified destination qualifies as a Florida Gulf Coast beach destination.",
        parent=route_main,
        critical=True
    )
    dest_claim = (
        f"{label_destination(ex)} is located on Florida's Gulf Coast and is a beach destination."
    )
    await evaluator.verify(
        claim=dest_claim,
        node=dest_is_gulf_leaf,
        sources=route_urls(ex),
        additional_instruction=(
            "Confirm the destination lies on Florida's Gulf Coast (west coast on the Gulf of Mexico) and is a beach destination. "
            "Allow reasonable naming variations (e.g., Fort Myers/RSW, Sarasota/SRQ, Tampa/TPA, St. Pete/PIE). "
            "If sources do not explicitly state 'Gulf Coast', rely on clear geographic cues from authoritative pages."
        )
    )

    # Departure airport is HVN (Tweed New Haven Airport in Connecticut)
    dep_hvn_leaf = evaluator.add_leaf(
        id="Departure_Airport_Is_HVN",
        desc="Confirms the departure airport is Tweed New Haven Airport (HVN) in Connecticut.",
        parent=route_main,
        critical=True
    )
    dep_claim = "The departure airport is Tweed New Haven Airport (HVN) in Connecticut."
    await evaluator.verify(
        claim=dep_claim,
        node=dep_hvn_leaf,
        sources=route_urls(ex),
        additional_instruction=(
            "Verify that the origin shown for the route is HVN (Tweed New Haven). "
            "Accept minor naming variations such as 'New Haven (HVN)'."
        )
    )

    # Operated by Avelo Airlines
    avelo_op_leaf = evaluator.add_leaf(
        id="Operated_By_Avelo",
        desc="Confirms Avelo Airlines operates the route.",
        parent=route_main,
        critical=True
    )
    op_claim = (
        f"Avelo Airlines operates the route from HVN to {ex.destination_iata or 'the destination'}."
    )
    await evaluator.verify(
        claim=op_claim,
        node=avelo_op_leaf,
        sources=route_urls(ex),
        additional_instruction=(
            "Confirm the operating carrier is Avelo Airlines on this origin-destination pair. "
            "Prefer official Avelo route/schedule pages if available."
        )
    )

    # Route is nonstop/direct
    nonstop_leaf = evaluator.add_leaf(
        id="Route_Is_Nonstop",
        desc="Confirms the route is nonstop/direct with no connections required.",
        parent=route_main,
        critical=True
    )
    nonstop_claim = (
        f"The route from HVN to {ex.destination_iata or 'the destination'} is nonstop/direct (no connections)."
    )
    await evaluator.verify(
        claim=nonstop_claim,
        node=nonstop_leaf,
        sources=route_urls(ex),
        additional_instruction=(
            "Check whether the service is a direct point-to-point flight. "
            "Accept 'nonstop' or 'direct' indications; if the airline route page lists the city pair as a single flight, treat as nonstop."
        )
    )

    # Route currently operational as of Nov 2025
    operational_leaf = evaluator.add_leaf(
        id="Route_Currently_Operational_As_Of_Nov_2025",
        desc="Confirms the route is currently operational as of November 2025.",
        parent=route_main,
        critical=True
    )
    operational_claim = (
        "This HVN → Florida Gulf Coast destination route is operational (bookable or actively scheduled) as of November 2025."
    )
    await evaluator.verify(
        claim=operational_claim,
        node=operational_leaf,
        sources=route_urls(ex),
        additional_instruction=(
            "Use airline schedule/booking pages or recent announcements referenced in the answer. "
            "If the page indicates flights are available around November 2025, consider it operational. "
            "If sources are outdated or clearly archived, do not support."
        )
    )


async def verify_baggage_group(evaluator: Evaluator, parent_node, ex: RouteBaggageExtraction) -> None:
    # Create a critical sequential node for baggage checks (gates downstream verifications)
    baggage_main = evaluator.add_sequential(
        id="Baggage_Main",
        desc="Avelo baggage policy verification group",
        parent=parent_node,
        critical=True
    )

    # Existence gate: baggage details provided
    baggage_exists = evaluator.add_custom_node(
        result=bool(ex.personal_item_dimensions) and bool(ex.carry_on_fee_range),
        id="Baggage_Info_Provided",
        desc="Answer provides personal item dimensions and carry-on fee range information.",
        parent=baggage_main,
        critical=True
    )

    # Personal item maximum dimensions
    personal_item_leaf = evaluator.add_leaf(
        id="Personal_Item_Max_Dimensions",
        desc="States Avelo Airlines' free personal item maximum dimensions: 17 in (L) x 13 in (H) x 9 in (W).",
        parent=baggage_main,
        critical=True
    )
    personal_item_claim = (
        "Avelo's free personal item maximum dimensions are 17 in (L) x 13 in (H) x 9 in (W)."
    )
    await evaluator.verify(
        claim=personal_item_claim,
        node=personal_item_leaf,
        sources=baggage_urls(ex),
        additional_instruction=(
            "Check Avelo's official baggage/personal item policy page. "
            "Allow minor formatting variations such as '17\" x 13\" x 9\"' or '17 x 13 x 9 inches'."
        )
    )

    # Carry-on bag fee range by booking timing
    carry_on_leaf = evaluator.add_leaf(
        id="Carry_On_Fee_Range_By_Booking_Timing",
        desc="States Avelo Airlines' carry-on bag fee range by booking timing: $45 to $77 per bag.",
        parent=baggage_main,
        critical=True
    )
    carry_on_claim = (
        "Avelo's carry-on bag fee ranges from $45 to $77 per bag depending on when it is purchased (e.g., during booking vs later/at the airport)."
    )
    await evaluator.verify(
        claim=carry_on_claim,
        node=carry_on_leaf,
        sources=baggage_urls(ex),
        additional_instruction=(
            "Check Avelo's official baggage fees page for carry-on pricing by timing. "
            "Confirm the fee range lower bound is $45 and upper bound is $77 per bag. "
            "Allow minor variations in currency formatting."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict:
    # Initialize evaluator with a parallel root
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_route_baggage_info(),
        template_class=RouteBaggageExtraction,
        extraction_name="route_baggage_extraction"
    )

    # Add ground truth expectations (for transparency in the summary)
    evaluator.add_ground_truth({
        "expected_personal_item_max_dimensions": "17 in (L) x 13 in (H) x 9 in (W)",
        "expected_carry_on_fee_range": "$45 to $77 per bag (depending on booking timing)"
    })

    # Build planning node (critical, as per rubric)
    planning_node = evaluator.add_parallel(
        id="Florida_Gulf_Coast_Destination_Planning",
        desc="Identify the Florida Gulf Coast destination served by Avelo nonstop from HVN and verify route and baggage policy details.",
        parent=root,
        critical=True
    )

    # Verify route group (critical subtree)
    await verify_route_group(evaluator, planning_node, extracted)

    # Verify baggage policy group (critical subtree)
    await verify_baggage_group(evaluator, planning_node, extracted)

    # Return structured summary
    return evaluator.get_summary()