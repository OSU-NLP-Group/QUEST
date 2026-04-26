import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "md_state_park_swim_reserve_pets"
TASK_DESCRIPTION = (
    "Which Maryland state park requires advance day-use reservations on weekends and holidays from Memorial Day "
    "through Labor Day (with reservations available up to 7 days beforehand), features a designated swimming beach "
    "with lifeguard services on duty Thursday through Sunday from 11:00 AM to 6:00 PM during that same Memorial Day "
    "through Labor Day period, and allows pets in day-use areas with a mandatory leash requirement?"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ParkExtraction(BaseModel):
    """Minimal structured data needed from the answer."""
    park_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_info() -> str:
    return """
    From the answer, extract the single Maryland state park the answer identifies as meeting all the criteria.
    Return:
    - park_name: The exact, proper name of the specific Maryland state park identified in the answer. If multiple park names appear, choose the one the answer finally recommends/identifies; if ambiguous, pick the first clearly asserted park.
    - sources: An array of all URLs explicitly provided in the answer that support the park identification and/or any of the requirements (swimming beach, lifeguard schedule and hours, reservation system requirements, pet policy). Include all URLs mentioned anywhere in the answer (plain links, markdown links, or within a 'Sources' section). Deduplicate and ensure full URLs with protocol.
    
    If the answer does not clearly identify a specific park, set park_name to null.
    If the answer contains no URLs, return an empty array for sources.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_park_name(extracted: ParkExtraction) -> str:
    return (extracted.park_name or "").strip()


def have_sources(extracted: ParkExtraction) -> bool:
    return bool(extracted.sources and len([u for u in extracted.sources if isinstance(u, str) and u.strip()]) > 0)


def add_leaf_and_task(
    evaluator: Evaluator,
    tasks: List,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    add_ins: str
):
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True
    )
    tasks.append((claim, sources, node, add_ins))
    return node


# --------------------------------------------------------------------------- #
# Main evaluation logic                                                       #
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
    Evaluate an answer for the Maryland state park identification task with strict, source-grounded verification.
    """

    # 1) Initialize evaluator/root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level can aggregate groups in parallel
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

    # 2) Extract park name + all cited URLs from the answer
    extracted: ParkExtraction = await evaluator.extract(
        prompt=prompt_extract_park_info(),
        template_class=ParkExtraction,
        extraction_name="park_extraction"
    )

    # 3) Add "ground truth style" criteria metadata for transparency
    evaluator.add_ground_truth({
        "required_criteria": {
            "location": "Park is located in Maryland",
            "swimming": {
                "swim_beach": "Designated swimming beach or swimming area exists",
                "lifeguard_days": "Lifeguards on duty Thursday through Sunday (at minimum)",
                "lifeguard_hours": "Lifeguards on duty 11:00 AM to 6:00 PM",
                "lifeguard_period": "Lifeguard services Memorial Day weekend through Labor Day weekend"
            },
            "day_use_reservations": {
                "weekend_holiday_required": "Advance day-use reservations required on weekends and holidays during Memorial Day–Labor Day",
                "advance_window": "Reservations available up to 7 days in advance",
                "official_system": "Reservations use Maryland's official reservation system"
            },
            "pets": {
                "allowed_day_use": "Pets allowed in day-use areas",
                "leash_required": "Leash required at all times"
            }
        }
    })

    park_name = safe_park_name(extracted)
    sources_list = [u.strip() for u in (extracted.sources or []) if isinstance(u, str) and u.strip()]

    # 4) Build the verification tree following the rubric
    # Main critical node: all children must pass
    main_node = evaluator.add_parallel(
        id="Maryland_State_Park_Identification",
        desc="Identify a specific Maryland state park that meets all specified swimming facility, lifeguard service, reservation system, and pet policy criteria",
        parent=root,
        critical=True
    )

    # Gating existence check: park identified AND at least one source provided
    evaluator.add_custom_node(
        result=(bool(park_name) and have_sources(extracted)),
        id="Park_Name_and_Sources_Provided",
        desc="The answer names a specific park and provides at least one source URL",
        parent=main_node,
        critical=True
    )

    # 4.1 Location verification (leaf)
    location_tasks: List = []
    add_leaf_and_task(
        evaluator,
        location_tasks,
        parent=main_node,
        node_id="Location_Verification",
        desc="The identified park must be located in the state of Maryland",
        claim=f"{park_name} is a Maryland state park located in the state of Maryland.",
        sources=sources_list,
        add_ins="Accept if the page clearly indicates the park is in Maryland (MD) or is part of the Maryland Park Service/DNR."
    )
    await evaluator.batch_verify(location_tasks)

    # 4.2 Swimming facility + lifeguard services group (parallel, critical)
    swim_group = evaluator.add_parallel(
        id="Swimming_Facility_Requirements",
        desc="The park must provide swimming facilities with specific lifeguard services",
        parent=main_node,
        critical=True
    )
    swim_tasks: List = []

    add_leaf_and_task(
        evaluator,
        swim_tasks,
        parent=swim_group,
        node_id="Swimming_Beach_Area_Present",
        desc="The park must have a designated swimming beach or swimming area",
        claim=f"{park_name} has a designated swimming beach or a clearly marked swimming area.",
        sources=sources_list,
        add_ins="Look for explicit mentions of a 'swim beach', 'designated swimming area', 'swimming beach', or similar."
    )

    add_leaf_and_task(
        evaluator,
        swim_tasks,
        parent=swim_group,
        node_id="Lifeguard_Days_Compliance",
        desc="Lifeguards must be on duty at minimum Thursday through Sunday",
        claim=f"At {park_name}, lifeguards are on duty at least Thursday through Sunday (Thu–Sun).",
        sources=sources_list,
        add_ins="Pass if the schedule explicitly includes Thursday through Sunday. If the page states lifeguards are on duty daily or more days than Thu–Sun, that also satisfies this requirement."
    )

    add_leaf_and_task(
        evaluator,
        swim_tasks,
        parent=swim_group,
        node_id="Lifeguard_Hours_Compliance",
        desc="Lifeguard duty hours must cover the period from 11:00 AM to 6:00 PM",
        claim=f"At {park_name}, lifeguards are on duty from 11:00 AM to 6:00 PM.",
        sources=sources_list,
        add_ins="Accept equivalent expressions like '11am–6pm', '11 to 6', or ranges that clearly match 11:00 AM to 6:00 PM."
    )

    add_leaf_and_task(
        evaluator,
        swim_tasks,
        parent=swim_group,
        node_id="Lifeguard_Service_Period",
        desc="Lifeguard services must operate from Memorial Day weekend through Labor Day weekend",
        claim=f"At {park_name}, lifeguard services operate from Memorial Day weekend through Labor Day weekend.",
        sources=sources_list,
        add_ins="Accept equivalent phrasing such as 'between Memorial Day and Labor Day', 'Memorial Day to Labor Day', or 'MDW through LDW'."
    )

    await evaluator.batch_verify(swim_tasks)

    # 4.3 Day-use reservation system group (parallel, critical)
    res_group = evaluator.add_parallel(
        id="Day_Use_Reservation_System",
        desc="The park must implement Maryland's day-use reservation system with specific requirements",
        parent=main_node,
        critical=True
    )
    res_tasks: List = []

    add_leaf_and_task(
        evaluator,
        res_tasks,
        parent=res_group,
        node_id="Weekend_Holiday_Reservation_Requirement",
        desc="The park must require advance day-use reservations on weekends and holidays during the Memorial Day through Labor Day period",
        claim=f"{park_name} requires advance day-use reservations on weekends and holidays during the Memorial Day through Labor Day period.",
        sources=sources_list,
        add_ins="Accept equivalent phrasing (e.g., 'peak season weekends and holidays', 'summer weekends/holidays Memorial Day–Labor Day')."
    )

    add_leaf_and_task(
        evaluator,
        res_tasks,
        parent=res_group,
        node_id="Seven_Day_Advance_Booking",
        desc="Day-use reservations must be available for booking up to 7 days in advance",
        claim=f"Day-use reservations for {park_name} are available up to 7 days in advance.",
        sources=sources_list,
        add_ins="Accept synonyms for the window like 'one week in advance' or explicit '(7) days prior'."
    )

    add_leaf_and_task(
        evaluator,
        res_tasks,
        parent=res_group,
        node_id="Maryland_Official_System",
        desc="The park must use the official Maryland state park reservation system",
        claim=f"Day-use reservations for {park_name} are made through Maryland's official reservation system (e.g., parkreservations.maryland.gov or a Maryland.gov DNR portal).",
        sources=sources_list,
        add_ins="Pass if the source indicates 'parkreservations.maryland.gov' or an official Maryland DNR reservations portal/process for day-use."
    )

    await evaluator.batch_verify(res_tasks)

    # 4.4 Pet policy group (parallel, critical)
    pet_group = evaluator.add_parallel(
        id="Pet_Policy_Requirements",
        desc="The park must have a pet-friendly policy with specific leash requirements",
        parent=main_node,
        critical=True
    )
    pet_tasks: List = []

    add_leaf_and_task(
        evaluator,
        pet_tasks,
        parent=pet_group,
        node_id="Pets_Allowed_Day_Use",
        desc="Pets must be permitted in day-use areas of the park",
        claim=f"Pets are permitted in day-use areas at {park_name}.",
        sources=sources_list,
        add_ins="Pass if the policy states pets are allowed in the general day-use area(s). It's acceptable if certain sub-areas (e.g., sandy beach) restrict pets seasonally."
    )

    add_leaf_and_task(
        evaluator,
        pet_tasks,
        parent=pet_group,
        node_id="Mandatory_Leash_Requirement",
        desc="The park must require all pets to be leashed at all times",
        claim=f"At {park_name}, pets must be kept on a leash at all times (typically a 6-foot leash) in day-use areas.",
        sources=sources_list,
        add_ins="Accept equivalent wording like 'leash required', 'must be leashed', or 'under control on a leash (e.g., 6 ft)'."
    )

    await evaluator.batch_verify(pet_tasks)

    # 5) Return summary
    return evaluator.get_summary()