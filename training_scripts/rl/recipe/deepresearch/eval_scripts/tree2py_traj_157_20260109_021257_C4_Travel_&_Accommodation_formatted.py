import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dfw_hotel_criteria"
TASK_DESCRIPTION = (
    "I am planning a business trip to the Dallas/Fort Worth area and need to find a hotel near Dallas/Fort Worth "
    "International Airport (DFW) that meets all of the following requirements:\n\n"
    "1. Provides free shuttle service to and from DFW Airport\n"
    "2. Has on-site meeting room facilities that can accommodate at least 50 people\n"
    "3. Offers suite accommodations with separate bedroom and living room areas (not junior suites or studio-style rooms)\n"
    "4. Provides on-site parking for guests\n"
    "5. Includes complimentary breakfast with the room rate\n"
    "6. Has an on-site fitness center\n"
    "7. Maintains a 24-hour front desk\n"
    "8. Allows pets (dogs and/or cats)\n\n"
    "Please identify at least one hotel that satisfies all eight of these criteria. For the hotel you identify, provide:\n"
    "- The hotel's name\n"
    "- A link to the hotel's official website\n"
    "- Confirmation that each of the eight requirements is met, with supporting reference URLs"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelEvidence(BaseModel):
    """Structured extraction of the selected hotel's identification and evidence URLs per constraint."""
    hotel_name: Optional[str] = None
    official_website_url: Optional[str] = None

    # Evidence URL sets for each constraint
    near_dfw_evidence_urls: List[str] = Field(default_factory=list)
    shuttle_dfw_evidence_urls: List[str] = Field(default_factory=list)
    meeting_room_50_evidence_urls: List[str] = Field(default_factory=list)
    true_suite_evidence_urls: List[str] = Field(default_factory=list)
    parking_evidence_urls: List[str] = Field(default_factory=list)
    breakfast_evidence_urls: List[str] = Field(default_factory=list)
    fitness_evidence_urls: List[str] = Field(default_factory=list)
    front_desk_evidence_urls: List[str] = Field(default_factory=list)
    pets_evidence_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_evidence() -> str:
    return """
    From the provided answer, extract the single hotel that the answer is proposing as satisfying ALL 8 constraints.
    If the answer mentions multiple hotels, select the primary one the answer recommends (or the first one listed) and
    extract its identification info and the specific evidence URLs cited for each constraint.

    Extract the following fields exactly as they appear in the answer:
    1. hotel_name: The hotel's name.
    2. official_website_url: A link to the hotel's official website (prefer property/brand official site, not aggregators).
    3. near_dfw_evidence_urls: All URLs cited that support the hotel being near DFW airport (e.g., address page, location page).
    4. shuttle_dfw_evidence_urls: All URLs cited that support the hotel provides FREE shuttle service to/from DFW airport.
    5. meeting_room_50_evidence_urls: All URLs cited that support on-site meeting rooms accommodating at least 50 people.
    6. true_suite_evidence_urls: All URLs cited that support suites with separate bedroom AND living room areas (not studio/junior).
    7. parking_evidence_urls: All URLs cited that support on-site guest parking (free or paid).
    8. breakfast_evidence_urls: All URLs cited that support complimentary breakfast included with the room rate.
    9. fitness_evidence_urls: All URLs cited that support an on-site fitness center.
    10. front_desk_evidence_urls: All URLs cited that support a 24-hour front desk.
    11. pets_evidence_urls: All URLs cited that support pets are allowed (dogs and/or cats).

    IMPORTANT:
    - Only include actual URLs explicitly present in the answer. If none are provided for a field, return an empty list.
    - URLs may be plain or in markdown links; extract the real URL.
    - Do not invent or infer URLs.
    - If a URL lacks protocol, prepend http://.
    - Deduplicate URLs within each list, but keep all unique URLs mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _hotel_ref(name: Optional[str]) -> str:
    return f"the hotel '{name}'" if _has_text(name) else "the hotel"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_near_dfw_and_shuttle(
    evaluator: Evaluator,
    parent_node,
    ev: HotelEvidence,
) -> None:
    """
    Build verification for the combined requirement:
    - Near DFW
    - Free shuttle to/from DFW
    Separate into two concrete leaf checks with their own evidence existence checks.
    """
    group_node = evaluator.add_parallel(
        id="Near_DFW_And_Free_Shuttle_With_Evidence",
        desc="Hotel is located near DFW and provides free shuttle service to/from DFW, with supporting reference URL(s)",
        parent=parent_node,
        critical=True,
    )

    # Existence checks for evidence URLs
    near_urls_provided = evaluator.add_custom_node(
        result=_has_urls(ev.near_dfw_evidence_urls),
        id="Near_DFW_Evidence_Provided",
        desc="At least one supporting reference URL is provided for 'near DFW' claim",
        parent=group_node,
        critical=True,
    )
    shuttle_urls_provided = evaluator.add_custom_node(
        result=_has_urls(ev.shuttle_dfw_evidence_urls),
        id="Free_Shuttle_DFW_Evidence_Provided",
        desc="At least one supporting reference URL is provided for 'free shuttle to/from DFW' claim",
        parent=group_node,
        critical=True,
    )

    # Verify "Near DFW"
    near_leaf = evaluator.add_leaf(
        id="Near_DFW_Verified",
        desc="Hotel is near DFW",
        parent=group_node,
        critical=True,
    )
    near_claim = (
        f"{_hotel_ref(ev.hotel_name)} is located near Dallas/Fort Worth International Airport (DFW)."
    )
    await evaluator.verify(
        claim=near_claim,
        node=near_leaf,
        sources=ev.near_dfw_evidence_urls,
        additional_instruction=(
            "Accept if the page explicitly states the hotel is at/inside/adjacent to/near DFW airport, "
            "or within a short distance (e.g., 'minutes from DFW'). Do not rely on external knowledge; "
            "base the decision solely on the provided page content/screenshots."
        ),
    )

    # Verify "Free shuttle to/from DFW"
    shuttle_leaf = evaluator.add_leaf(
        id="Free_Shuttle_DFW_Verified",
        desc="Hotel provides free shuttle service to/from DFW",
        parent=group_node,
        critical=True,
    )
    shuttle_claim = (
        f"{_hotel_ref(ev.hotel_name)} provides FREE shuttle service to and from Dallas/Fort Worth International Airport (DFW)."
    )
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_leaf,
        sources=ev.shuttle_dfw_evidence_urls,
        additional_instruction=(
            "The page must indicate 'free' or 'complimentary' airport shuttle specifically to/from DFW. "
            "If it only says 'airport shuttle' without indicating free/complimentary, consider it NOT meeting the requirement."
        ),
    )


async def _add_constraint_group(
    evaluator: Evaluator,
    parent_node,
    group_id: str,
    group_desc: str,
    evidence_urls: List[str],
    claim: str,
    add_ins: str,
) -> None:
    """
    Generic builder for a single-claim constraint node:
    - Adds a critical parallel node representing the constraint
    - Adds a critical evidence existence leaf
    - Adds a critical verification leaf referencing the provided URLs
    """
    node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=True,
    )

    evidence_leaf = evaluator.add_custom_node(
        result=_has_urls(evidence_urls),
        id=f"{group_id}_Evidence_Provided",
        desc="At least one supporting reference URL is provided",
        parent=node,
        critical=True,
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{group_id}_Supported",
        desc="Constraint is supported by cited references",
        parent=node,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=evidence_urls,
        additional_instruction=add_ins,
    )


async def build_verification_tree(evaluator: Evaluator, ev: HotelEvidence) -> None:
    """
    Construct the verification tree according to the rubric, using the extracted evidence.
    """
    # Top-level critical node for the whole task (under evaluator root)
    top = evaluator.add_parallel(
        id="Find_Qualifying_Hotel",
        desc="Identify at least one hotel that satisfies all stated constraints and provide required identifying info and supporting references",
        parent=evaluator.root,
        critical=True,
    )

    # Hotel identification (critical)
    ident = evaluator.add_parallel(
        id="Hotel_Identification",
        desc="Provide the required identifying information for the selected hotel",
        parent=top,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_text(ev.hotel_name),
        id="Hotel_Name_Provided",
        desc="Provide the hotel's name",
        parent=ident,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_text(ev.official_website_url),
        id="Official_Website_Link_Provided",
        desc="Provide a link to the hotel's official website",
        parent=ident,
        critical=True,
    )

    # Constraints & evidence checks (critical)
    constraints = evaluator.add_parallel(
        id="Constraint_And_Evidence_Checks",
        desc="Confirm each constraint is met and provide supporting reference URL(s) for each",
        parent=top,
        critical=True,
    )

    # Combined Near DFW + Free Shuttle
    await _verify_near_dfw_and_shuttle(evaluator, constraints, ev)

    # Meeting room capacity ≥ 50
    meeting_claim = (
        f"{_hotel_ref(ev.hotel_name)} has on-site meeting room facilities capable of accommodating at least 50 people "
        f"in a room or configuration (e.g., theater/classroom/banquet)."
    )
    await _add_constraint_group(
        evaluator,
        constraints,
        "Meeting_Room_50_With_Evidence",
        "Hotel has on-site meeting room facilities capable of accommodating at least 50 people, with supporting reference URL(s)",
        ev.meeting_room_50_evidence_urls,
        meeting_claim,
        add_ins=(
            "Confirm capacity ≥ 50 in any room or combined space per the page's capacity charts or descriptions. "
            "Accept any layout (theater, classroom, banquet) as long as max capacity ≥ 50."
        ),
    )

    # True suite with separate bedroom and living room
    suite_claim = (
        f"{_hotel_ref(ev.hotel_name)} offers suite accommodations that have a separate bedroom and a separate living room area "
        f"(not a studio/junior suite layout)."
    )
    await _add_constraint_group(
        evaluator,
        constraints,
        "True_Suite_Separate_Rooms_With_Evidence",
        "Hotel offers suite accommodations with separate bedroom and living room areas (not junior suites or studio layouts), with supporting reference URL(s)",
        ev.true_suite_evidence_urls,
        suite_claim,
        add_ins=(
            "The page should clearly indicate separate rooms (e.g., 'one-bedroom suite' with a distinct bedroom and living room; "
            "prefer mention of a door between rooms). Do NOT accept studio/junior suites or 'open-plan' layouts."
        ),
    )

    # On-site parking
    parking_claim = f"{_hotel_ref(ev.hotel_name)} provides on-site parking for guests."
    await _add_constraint_group(
        evaluator,
        constraints,
        "On_Site_Parking_With_Evidence",
        "Hotel provides on-site parking for guests (free or paid acceptable), with supporting reference URL(s)",
        ev.parking_evidence_urls,
        parking_claim,
        add_ins=(
            "Accept both free and paid parking as long as it is on-site and available to guests. "
            "Reject only if the page indicates no on-site parking or off-site only."
        ),
    )

    # Complimentary breakfast
    breakfast_claim = f"{_hotel_ref(ev.hotel_name)} includes complimentary breakfast with the room rate."
    await _add_constraint_group(
        evaluator,
        constraints,
        "Complimentary_Breakfast_With_Evidence",
        "Hotel includes complimentary breakfast with the room rate (continental or full acceptable), with supporting reference URL(s)",
        ev.breakfast_evidence_urls,
        breakfast_claim,
        add_ins=(
            "The page must indicate breakfast is complimentary/free/included in rate. "
            "Continental or full breakfast are both acceptable as long as included."
        ),
    )

    # Fitness center on-site
    fitness_claim = f"{_hotel_ref(ev.hotel_name)} has an on-site fitness center available to guests."
    await _add_constraint_group(
        evaluator,
        constraints,
        "Fitness_Center_With_Evidence",
        "Hotel has an on-site fitness center available to guests, with supporting reference URL(s)",
        ev.fitness_evidence_urls,
        fitness_claim,
        add_ins=(
            "Accept gym/fitness center references. It must be on-site and available to guests (hours may vary)."
        ),
    )

    # 24-hour front desk
    front_desk_claim = f"{_hotel_ref(ev.hotel_name)} maintains a 24-hour front desk."
    await _add_constraint_group(
        evaluator,
        constraints,
        "24_Hour_Front_Desk_With_Evidence",
        "Hotel maintains a 24-hour front desk, with supporting reference URL(s)",
        ev.front_desk_evidence_urls,
        front_desk_claim,
        add_ins=(
            "Confirm that the front desk is staffed 24 hours. "
            "If the page shows limited hours only, consider it NOT meeting the requirement."
        ),
    )

    # Pets allowed
    pets_claim = f"{_hotel_ref(ev.hotel_name)} allows pets (dogs and/or cats)."
    await _add_constraint_group(
        evaluator,
        constraints,
        "Pets_Allowed_With_Evidence",
        "Hotel allows pets (dogs and/or cats), with supporting reference URL(s)",
        ev.pets_evidence_urls,
        pets_claim,
        add_ins=(
            "Accept if the page indicates pets allowed (e.g., 'pet-friendly', 'dogs allowed'). "
            "Cats allowed is optional; at least one of dogs or cats must be permitted. "
            "Fees/restrictions are acceptable if pets are permitted."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for the DFW hotel constraints task and return a structured summary.
    """
    # Initialize evaluator with a parallel root; we'll add a critical top-level node underneath.
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
        default_model=model,
    )

    # Extract structured hotel evidence from the answer
    ev: HotelEvidence = await evaluator.extract(
        prompt=prompt_extract_hotel_evidence(),
        template_class=HotelEvidence,
        extraction_name="hotel_evidence",
    )

    # Build verification tree and run all checks
    await build_verification_tree(evaluator, ev)

    # Return final summary
    return evaluator.get_summary()