import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fl_state_park_precruise_everglades_destiny"
TASK_DESCRIPTION = (
    "A family of four is embarking on a Disney Destiny cruise from Port Everglades in Fort Lauderdale on November 25, 2025. "
    "They will arrive on the evening of November 23 and have the full day of November 24 free before their cruise. "
    "They want to spend the morning (8:00 AM to 12:00 PM) at a Florida state park that offers outdoor recreation activities suitable for children aged 8 and 12. "
    "Identify ONE Florida state park that meets ALL of the following requirements: "
    "(1) Located within 5 miles of Port Everglades Cruise Terminal #4 (1800 SE 20th Street, Fort Lauderdale, FL 33316), "
    "(2) Open by 8:00 AM on November 24, 2025, "
    "(3) Charges a vehicle entrance fee of $6 or less for a vehicle with 2-8 people, "
    "(4) Offers hiking trails or nature walking paths, "
    "(5) Provides beach access for swimming, "
    "(6) Has kayaking or canoeing facilities (rentals or launch sites), "
    "(7) Has parking facilities for visitors, and "
    "(8) Has restroom facilities. "
    "Provide the park's official current name, complete street address, approximate distance from the cruise terminal, "
    "exact vehicle entrance fee amount for 2-8 people, and confirmation that each of the required facilities and activities is available, "
    "with at least one reference URL to verify the information."
)

CRUISE_TERMINAL_NAME = "Port Everglades Cruise Terminal #4"
CRUISE_TERMINAL_ADDR = "1800 SE 20th Street, Fort Lauderdale, FL 33316"
VISIT_DATE = "November 24, 2025"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkExtraction(BaseModel):
    """Structured info for the selected Florida state park as extracted from the answer."""
    name: Optional[str] = None  # Official current name
    address: Optional[str] = None  # Complete street address
    distance_miles: Optional[str] = None  # Approx distance from cruise terminal (e.g., "3.2 miles")
    opening_time_text: Optional[str] = None  # Stated opening time or hours text from the answer
    entrance_fee_vehicle_2_8: Optional[str] = None  # Exact vehicle fee for 2–8 people (e.g., "$6")
    reference_urls: List[str] = Field(default_factory=list)  # All URLs cited to support claims


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_park_info() -> str:
    return (
        "Extract the SINGLE Florida state park proposed in the answer along with core fields needed to verify the requirements. "
        "Return the following fields:\n"
        "1. name: The park's current official name (must be a Florida State Park).\n"
        "2. address: The complete street address of the park (include street, city, state, ZIP if provided).\n"
        "3. distance_miles: The approximate distance in miles from Port Everglades Cruise Terminal #4 to the park, "
        "   as stated in the answer (e.g., '3.1 miles'); if not given, return null.\n"
        "4. opening_time_text: The opening time or hours text relevant to the morning visit; for example, "
        "   'Open 8:00 AM' or '8 AM to sunset'; if not provided, set to null.\n"
        "5. entrance_fee_vehicle_2_8: The exact vehicle entrance fee amount for 2–8 people as stated in the answer "
        "   (e.g., '$6' or '$4 per vehicle'); if not provided, set to null.\n"
        "6. reference_urls: All reference URLs cited in the answer to support any of the park details, including official websites, "
        "   Florida State Parks pages, Google Maps, or other credible sources. Extract actual URLs only.\n"
        "If any field is not present in the answer, return null for that field (or an empty array for reference_urls). "
        "Extract only what is explicitly included in the answer; do not invent or infer values."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(val: Optional[str]) -> str:
    return val.strip() if isinstance(val, str) else ""

def _get_sources(park: ParkExtraction) -> List[str]:
    # Filter obvious empties
    return [u for u in (park.reference_urls or []) if isinstance(u, str) and u.strip()]

# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, root, park: ParkExtraction) -> None:
    """
    Build the verification tree according to the rubric and run all checks.
    Root is critical/parallel; every child group and leaf is critical.
    """
    sources = _get_sources(park)

    # 0) Reference URL presence – create first to gate other verifications
    ref_node = evaluator.add_custom_node(
        result=(len(sources) >= 1),
        id="reference_url",
        desc="At least one reference URL is provided to verify the park information",
        parent=root,
        critical=True
    )

    # 1) Park identity and location requirements
    pil_node = evaluator.add_parallel(
        id="park_identity_and_location",
        desc="Park identity and location requirements",
        parent=root,
        critical=True
    )

    # 1.1 Valid park name (must be a Florida State Park, current official name)
    valid_name_leaf = evaluator.add_leaf(
        id="valid_park_name",
        desc="A valid current official Florida state park name is provided (must be a state park, not a county park, national park, or other type)",
        parent=pil_node,
        critical=True
    )
    name_val = _safe_str(park.name)
    await evaluator.verify(
        claim=f"'{name_val}' is an official Florida State Park (current official name) under the Florida State Parks system.",
        node=valid_name_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the named park belongs to the Florida State Parks system (Florida Department of Environmental Protection). "
            "Do not accept county/city parks, national parks, or other types. Allow minor name formatting variations; "
            "prefer confirmation from floridastateparks.org or other authoritative sources."
        ),
    )

    # 1.2 Complete address
    complete_addr_leaf = evaluator.add_leaf(
        id="complete_address",
        desc="The complete street address of the park is provided",
        parent=pil_node,
        critical=True
    )
    addr_val = _safe_str(park.address)
    await evaluator.verify(
        claim=f"The official street address of {name_val} is '{addr_val}'.",
        node=complete_addr_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the park's address as listed on official or authoritative sources (ideally the official Florida State Parks page). "
            "The address should include street number/name, city, state, and ZIP where available. "
            "Minor formatting differences are acceptable if they refer to the same place."
        ),
    )

    # 1.3 Distance requirement: within 5 miles of Port Everglades Cruise Terminal #4
    distance_leaf = evaluator.add_leaf(
        id="distance_requirement",
        desc=f"The park is located within 5 miles of {CRUISE_TERMINAL_NAME} ({CRUISE_TERMINAL_ADDR})",
        parent=pil_node,
        critical=True
    )
    dist_text = _safe_str(park.distance_miles)
    # Construct claim that either includes stated distance or generic <=5 miles assertion
    if dist_text:
        distance_claim = (
            f"The distance from {CRUISE_TERMINAL_NAME} ({CRUISE_TERMINAL_ADDR}) to {name_val} "
            f"is approximately {dist_text} and is within 5 miles."
        )
    else:
        distance_claim = (
            f"{name_val} at '{addr_val}' is located within 5 miles of {CRUISE_TERMINAL_NAME} ({CRUISE_TERMINAL_ADDR})."
        )
    await evaluator.verify(
        claim=distance_claim,
        node=distance_leaf,
        sources=sources,
        additional_instruction=(
            "Use map or directions pages (e.g., Google Maps) if available; rely on page text and/or screenshots. "
            "Pass if the provided evidence clearly indicates the distance is ≤ 5 miles. "
            "If only address is given without distance evidence, do not pass unless the screenshot/URL explicitly shows ≤ 5 miles."
        ),
    )

    # 2) Operating parameters: opening time and entrance fee
    op_node = evaluator.add_parallel(
        id="operating_parameters",
        desc="Operating hours and fee requirements",
        parent=root,
        critical=True
    )

    # 2.1 Opening time: by 8:00 AM on Nov 24, 2025
    opening_leaf = evaluator.add_leaf(
        id="opening_time",
        desc="The park opens by 8:00 AM (to accommodate the morning visit on November 24, 2025)",
        parent=op_node,
        critical=True
    )
    hours_text = _safe_str(park.opening_time_text)
    await evaluator.verify(
        claim=(
            f"On {VISIT_DATE}, {name_val} opens at or before 8:00 AM. "
            f"The stated hours indicate opening at 8:00 AM (or earlier)."
        ),
        node=opening_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm from official or authoritative sources that the park typically opens at 8:00 AM (or earlier). "
            "Phrasings like 'Open 8:00 a.m. until sunset' or '8 AM daily' are acceptable evidence for the given date. "
            "If a specific exception/closure is stated for that date, do not pass."
        ),
    )

    # 2.2 Entrance fee: $6 or less for a vehicle with 2–8 people
    fee_leaf = evaluator.add_leaf(
        id="entrance_fee",
        desc="The vehicle entrance fee is $6 or less for a vehicle with 2-8 people",
        parent=op_node,
        critical=True
    )
    fee_text = _safe_str(park.entrance_fee_vehicle_2_8)
    await evaluator.verify(
        claim=(
            f"The vehicle entrance fee at {name_val} for 2–8 people is '{fee_text}', "
            "which is $6 or less."
        ),
        node=fee_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the exact vehicle admission fee (2–8 people) from an official fee schedule (ideally Florida State Parks site). "
            "Pass only if the stated fee is ≤ $6. If the page shows a higher fee, fail."
        ),
    )

    # 3) Required outdoor activities: hiking, beach access for swimming, kayaking/canoeing
    act_node = evaluator.add_parallel(
        id="required_outdoor_activities",
        desc="Required outdoor recreation activities",
        parent=root,
        critical=True
    )

    hiking_leaf = evaluator.add_leaf(
        id="hiking_trails",
        desc="The park offers hiking trails or nature walking paths",
        parent=act_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} offers hiking trails or nature walking paths.",
        node=hiking_leaf,
        sources=sources,
        additional_instruction=(
            "Look for mentions of 'hiking', 'nature trail', 'walking paths', 'boardwalk', or similar on the park page."
        ),
    )

    beach_leaf = evaluator.add_leaf(
        id="beach_access",
        desc="The park provides beach access for swimming",
        parent=act_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} provides beach access where swimming is permitted.",
        node=beach_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that swimming is allowed (seasonal or general) at the park's beach. "
            "Mentions like 'swimming', 'beach for swimming', or similar qualify; "
            "if explicitly prohibited, fail."
        ),
    )

    kayak_leaf = evaluator.add_leaf(
        id="kayaking_canoeing",
        desc="The park has kayaking or canoeing facilities (rentals or launch sites)",
        parent=act_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} offers kayaking or canoeing facilities (rentals or launch/launch sites).",
        node=kayak_leaf,
        sources=sources,
        additional_instruction=(
            "Evidence can include kayak/canoe rentals, launch sites, paddling trails, or boat ramps suitable for paddlecraft."
        ),
    )

    # 4) Required amenities: parking and restrooms
    amen_node = evaluator.add_parallel(
        id="required_amenities",
        desc="Required visitor amenities",
        parent=root,
        critical=True
    )

    parking_leaf = evaluator.add_leaf(
        id="parking_facilities",
        desc="The park has parking facilities available for visitors",
        parent=amen_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} has parking facilities for visitors.",
        node=parking_leaf,
        sources=sources,
        additional_instruction=(
            "Accept mentions of parking lots, vehicle parking, or similar. "
            "If parking is unavailable or restricted such that typical visitor parking is not provided, fail."
        ),
    )

    restroom_leaf = evaluator.add_leaf(
        id="restroom_facilities",
        desc="The park has restroom facilities on-site",
        parent=amen_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} provides restroom facilities on-site.",
        node=restroom_leaf,
        sources=sources,
        additional_instruction=(
            "Look for mentions of restrooms, bathrooms, facilities, or comfort stations."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Florida State Park pre-cruise morning plan task.
    Constructs a critical parallel rubric tree; any failed critical child fails the root.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # As specified by rubric root
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

    # Record custom context info
    evaluator.add_custom_info(
        {"cruise_terminal_name": CRUISE_TERMINAL_NAME, "cruise_terminal_address": CRUISE_TERMINAL_ADDR,
         "visit_date": VISIT_DATE, "time_window": "8:00 AM to 12:00 PM"},
        info_type="context",
        info_name="visit_context"
    )

    # Extract park info
    park_info = await evaluator.extract(
        prompt=prompt_extract_park_info(),
        template_class=ParkExtraction,
        extraction_name="park_extraction"
    )

    # Build tree and verify according to rubric
    await build_and_verify(evaluator, root, park_info)

    # Return evaluation summary
    return evaluator.get_summary()