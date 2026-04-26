import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wyco_rv_campgrounds"
TASK_DESCRIPTION = (
    "I'm planning an RV camping trip through Wyoming and Colorado with my family of 6 and our dog. "
    "I need to find four campgrounds that meet all of the following requirements: "
    "(1) Located in a National Forest or National Park in Wyoming and/or Colorado, "
    "(2) Reservable through Recreation.gov (not first-come, first-served only), "
    "(3) Accommodates RVs of at least 30 feet in length, "
    "(4) Provides basic amenities including picnic tables and fire rings at campsites, "
    "(5) Has potable water available, "
    "(6) Has restroom facilities (either flush toilets or vault toilets), "
    "(7) Allows pets, and "
    "(8) Campsites can accommodate at least 6 people. "
    "For each of the four campgrounds, please provide: official campground name, "
    "location (specific National Forest or National Park name), elevation, maximum RV length accepted, "
    "and direct link to the campground's reservation page on Recreation.gov."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Campground(BaseModel):
    official_name: Optional[str] = None
    location_nf_np_name: Optional[str] = None  # Specific National Forest or National Park name
    elevation: Optional[str] = None
    max_rv_length_accepted: Optional[str] = None
    recreation_gov_url: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)  # e.g., ["Picnic table", "Fire ring"]
    potable_water_info: Optional[str] = None           # e.g., "Drinking water available"
    restroom_info: Optional[str] = None               # e.g., "Flush toilets", "Vault toilets"
    pets_policy: Optional[str] = None                 # e.g., "Pets allowed"
    capacity_info: Optional[str] = None               # e.g., "Occupancy: 8 people"
    source_urls: List[str] = Field(default_factory=list)  # Any additional URLs cited in the answer


class CampgroundsExtraction(BaseModel):
    campgrounds: List[Campground] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract up to FOUR distinct campgrounds listed in the answer. For each campground, return the following fields:
    - official_name: The official campground name as stated.
    - location_nf_np_name: The specific National Forest or National Park name (e.g., "Shoshone National Forest", "Rocky Mountain National Park").
    - elevation: The elevation value if provided (any format, e.g., "8,500 ft").
    - max_rv_length_accepted: The maximum RV length accepted (as stated, e.g., "32 ft", "30 feet").
    - recreation_gov_url: A direct URL to the reservation page on Recreation.gov (full URL).
    - amenities: A list of amenities mentioned (e.g., ["Picnic table", "Fire ring"]).
    - potable_water_info: Text describing if potable/drinking water is available (e.g., "Drinking water available").
    - restroom_info: Text describing restroom facilities (e.g., "Flush toilets", "Vault toilets", "Pit toilet").
    - pets_policy: Text describing pet policy (e.g., "Pets allowed").
    - capacity_info: Text describing occupancy per campsite (e.g., "Max occupancy 8").
    - source_urls: Any additional URLs explicitly cited in the answer specific to this campground (exclude duplicates of recreation.gov link).

    Rules:
    - Only extract information explicitly present in the answer text.
    - If a field is not mentioned for a campground, set it to null (or an empty list for list fields).
    - Only include URLs that are explicitly present in the answer text. If the Recreation.gov URL is missing or incomplete, still return it as provided; do not invent.
    - Return exactly up to four campgrounds in the 'campgrounds' array, in the same order as they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_four(camps: List[Campground]) -> List[Campground]:
    if not camps:
        return []
    return camps[:4]


def _sources_for_campground(cg: Campground) -> List[str]:
    srcs: List[str] = []
    if cg.recreation_gov_url and cg.recreation_gov_url.strip():
        srcs.append(cg.recreation_gov_url.strip())
    if cg.source_urls:
        srcs.extend([u for u in cg.source_urls if isinstance(u, str) and u.strip()])
    return srcs


# --------------------------------------------------------------------------- #
# Verification for one campground                                             #
# --------------------------------------------------------------------------- #
async def verify_one_campground(
    evaluator: Evaluator,
    parent_node,
    cg: Campground,
    index: int,
) -> None:
    """
    Build verification subtree for a single campground, including:
    - Output field presence checks (critical).
    - Constraint checks (critical), verified against sources (prefer Recreation.gov).
    """
    idx = index + 1
    camp_node = evaluator.add_parallel(
        id=f"Campground_{idx}",
        desc=f"Evaluate campground #{idx} against all constraints and required output fields.",
        parent=parent_node,
        critical=False,
    )

    # Output fields node (critical parent -> all children must be critical)
    out_node = evaluator.add_parallel(
        id=f"C{idx}_Output_Fields",
        desc=f"All required informational fields are provided for campground #{idx}.",
        parent=camp_node,
        critical=True,
    )

    # Output field presence checks (custom nodes -> binary)
    evaluator.add_custom_node(
        result=bool(cg.official_name and cg.official_name.strip()),
        id=f"C{idx}_Provide_Official_Name",
        desc=f"Provides the official campground name for campground #{idx}.",
        parent=out_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(cg.location_nf_np_name and cg.location_nf_np_name.strip()),
        id=f"C{idx}_Provide_Location_NF_or_NP_Name",
        desc=f"Provides the specific National Forest or National Park name for campground #{idx}.",
        parent=out_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(cg.elevation and cg.elevation.strip()),
        id=f"C{idx}_Provide_Elevation",
        desc=f"Provides the elevation for campground #{idx}.",
        parent=out_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(cg.max_rv_length_accepted and cg.max_rv_length_accepted.strip()),
        id=f"C{idx}_Provide_Max_RV_Length_Accepted",
        desc=f"Provides the maximum RV length accepted for campground #{idx}.",
        parent=out_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(cg.recreation_gov_url and ("recreation.gov" in (cg.recreation_gov_url or ""))),
        id=f"C{idx}_Provide_RecreationGov_Reservation_Link",
        desc=f"Provides a direct link to campground #{idx}'s reservation page on Recreation.gov.",
        parent=out_node,
        critical=True,
    )

    # Constraints: create leaves and verify by sources (prefer Recreation.gov)
    sources = _sources_for_campground(cg)

    # 1) Location in NF or NP in WY or CO
    loc_node = evaluator.add_leaf(
        id=f"C{idx}_Location_NF_or_NP_in_WY_or_CO",
        desc=f"Campground #{idx} is located in a National Forest or National Park in Wyoming and/or Colorado.",
        parent=camp_node,
        critical=True,
    )
    loc_claim = (
        f"The campground {cg.official_name or 'this campground'} is located within a National Forest or a National Park, "
        f"and it is in either Wyoming (WY) or Colorado (CO)."
    )

    # 2) Reservable via Recreation.gov (not FCFS-only)
    res_node = evaluator.add_leaf(
        id=f"C{idx}_Reservable_on_RecreationGov",
        desc=f"Campground #{idx} is reservable via Recreation.gov (not first-come, first-served only).",
        parent=camp_node,
        critical=True,
    )
    res_claim = (
        "This campground accepts advance reservations on Recreation.gov; it is not exclusively first-come-first-served only."
    )

    # 3) RV length >= 30 feet
    rv_node = evaluator.add_leaf(
        id=f"C{idx}_RV_Length_AtLeast_30ft",
        desc=f"Campground #{idx} accommodates RVs of at least 30 feet in length.",
        parent=camp_node,
        critical=True,
    )
    rv_claim = "The campground accommodates RVs with a maximum accepted length of at least 30 feet."

    # 4) Amenities: picnic tables AND fire rings
    am_node = evaluator.add_leaf(
        id=f"C{idx}_Amenities_PicnicTable_and_FireRing",
        desc=f"Campground #{idx} provides picnic tables and fire rings at campsites.",
        parent=camp_node,
        critical=True,
    )
    am_claim = (
        "Campsites at this campground include both picnic tables and fire rings (or fire pits/grills)."
    )

    # 5) Potable water available
    water_node = evaluator.add_leaf(
        id=f"C{idx}_Potable_Water_Available",
        desc=f"Campground #{idx} has potable water available.",
        parent=camp_node,
        critical=True,
    )
    water_claim = "Potable drinking water is available at the campground."

    # 6) Restrooms: flush or vault toilets
    rr_node = evaluator.add_leaf(
        id=f"C{idx}_Restrooms_Flush_or_Vault",
        desc=f"Campground #{idx} has restroom facilities (flush toilets or vault toilets).",
        parent=camp_node,
        critical=True,
    )
    rr_claim = "Restroom facilities are available at the campground: either flush toilets or vault/pit toilets."

    # 7) Pets allowed
    pets_node = evaluator.add_leaf(
        id=f"C{idx}_Pets_Allowed",
        desc=f"Campground #{idx} allows pets.",
        parent=camp_node,
        critical=True,
    )
    pets_claim = "Pets (e.g., dogs) are allowed at the campground."

    # 8) Capacity: sites accommodate at least 6 people
    cap_node = evaluator.add_leaf(
        id=f"C{idx}_Capacity_AtLeast_6_People",
        desc=f"Campground #{idx} campsites can accommodate at least 6 people.",
        parent=camp_node,
        critical=True,
    )
    cap_claim = (
        "A standard family campsite at this campground has an occupancy capacity of at least 6 people."
    )

    # Batch verify all constraints for this campground
    await evaluator.batch_verify(
        [
            (
                loc_claim,
                sources,
                loc_node,
                "Confirm the page identifies the site as inside a named National Forest or National Park and the state is either Wyoming or Colorado. "
                "Accept minor naming variants and abbreviations.",
            ),
            (
                res_claim,
                sources,
                res_node,
                "On the Recreation.gov page, look for indicators of reservability such as 'Reserve', 'Book Now', 'Available to reserve', "
                "or explicit mention of advance reservations. If it states 'first-come, first-served only', the claim is not supported.",
            ),
            (
                rv_claim,
                sources,
                rv_node,
                "Check 'Max RV length', 'Maximum vehicle/trailer length', or similar fields. The accepted length must be >= 30 feet.",
            ),
            (
                am_claim,
                sources,
                am_node,
                "Verify amenities list or campsite details indicate BOTH 'Picnic table' AND 'Fire ring' (or equivalent like 'Fire pit'/'Grill').",
            ),
            (
                water_claim,
                sources,
                water_node,
                "Look for 'Drinking water', 'Potable water', or equivalent. If specifically 'No water', the claim is not supported.",
            ),
            (
                rr_claim,
                sources,
                rr_node,
                "Look for 'Flush toilets' or 'Vault toilets'. 'Pit toilet' counts as vault. Either type is acceptable.",
            ),
            (
                pets_claim,
                sources,
                pets_node,
                "Check for 'Pets allowed', 'Dogs permitted', or equivalent policy statements.",
            ),
            (
                cap_claim,
                sources,
                cap_node,
                "Check 'Occupancy', 'Max people per site', or similar. Standard family sites should accommodate >= 6 people. "
                "If occupancy varies per site, it is acceptable if typical/standard sites support >= 6.",
            ),
        ]
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
    Evaluate an answer for the WY/CO RV campground task and return an evaluation summary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Task_Completion uses parallel aggregation
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

    # Root node representing Task_Completion
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Provide four campgrounds meeting all stated constraints and include all required fields for each campground.",
        parent=root,
        critical=False,
    )

    # Extract campgrounds from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction",
    )

    camps = _first_four(extracted.campgrounds)

    # Pad to exactly 4 entries to build a consistent tree
    while len(camps) < 4:
        camps.append(Campground())

    # Build verification subtrees for each campground
    for idx, cg in enumerate(camps):
        await verify_one_campground(evaluator, task_node, cg, idx)

    return evaluator.get_summary()