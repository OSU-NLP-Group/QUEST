import asyncio
import logging
from datetime import datetime
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "desolation_wilderness_trip_planning_2026"
TASK_DESCRIPTION = (
    "You are planning a 3-day, 2-night backpacking trip to Desolation Wilderness (located in the Lake Tahoe area of California) "
    "for a group of 8 adults. Your planned entry date is Saturday, July 11, 2026. Provide comprehensive trip planning information including: "
    "(1) Permit Reservation Details: When is the earliest date you can make a reservation for this trip? What is the total cost for permits for your group of 8 people for 2 nights? "
    "Through which platform should you make the reservation? "
    "(2) Required Equipment: What type of bear-resistant food storage container is required? Are campfires allowed, and what permit (if any) is needed for camp stoves? "
    "(3) Camping Regulations: What is the maximum group size allowed? What is the minimum distance you must camp from lakes, streams, and trails? "
    "(4) Additional Requirements: Identify any other permits or passes that would be helpful for accessing the wilderness area."
)

# Entry date (fixed per task) and derived earliest reservation date (6 months prior)
ENTRY_DATE = datetime(2026, 7, 11)
EARLIEST_RESERVATION_DATE = datetime(2026, 1, 11)  # 6 months before July 11, 2026
EARLIEST_RESERVATION_DATE_TEXT = EARLIEST_RESERVATION_DATE.strftime("%B %d, %Y")  # "January 11, 2026"

# Group size and fee schedule expectations
GROUP_SIZE = 8
RESERVATION_FEE = 6  # USD
PER_PERSON_FEE = 10  # USD for 2–14 nights
EXPECTED_TOTAL_COST = RESERVATION_FEE + GROUP_SIZE * PER_PERSON_FEE  # 6 + 8*10 = 86 USD

# Ground truth helpful info (for logging in summary)
GROUND_TRUTH_INFO = {
    "entry_date": ENTRY_DATE.strftime("%B %d, %Y"),
    "earliest_reservation_date": EARLIEST_RESERVATION_DATE_TEXT,
    "fee_schedule": {
        "reservation_fee_usd": RESERVATION_FEE,
        "per_person_fee_usd": PER_PERSON_FEE,
        "nights_range": "2–14"
    },
    "expected_total_permit_cost_usd_for_group_8_2_nights": EXPECTED_TOTAL_COST,
    "expected_platform": "Recreation.gov",
    "camping_regulations": {
        "maximum_group_size": "12 people",
        "minimum_camping_distance": "at least 100 feet from lakes, streams, and trails"
    },
    "equipment_fire_rules": {
        "bear_container": "Hard-sided bear canister required",
        "campfires": "Prohibited; only camp stoves permitted",
        "stove_permit": "California campfire permit (free) required to operate stoves on federal lands"
    },
    "designated_site_additional_rule": "Use designated sites when available at certain lakes; if all designated sites are occupied, camp more than 500 feet from lakeshore."
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PermitDetails(BaseModel):
    earliest_reservation_date: Optional[str] = None
    reservation_platform: Optional[str] = None
    total_permit_cost: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EquipmentRules(BaseModel):
    bear_resistant_food_storage: Optional[str] = None
    campfire_rules: Optional[str] = None
    camp_stove_permit: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CampingRegulations(BaseModel):
    maximum_group_size: Optional[str] = None
    minimum_camping_distance: Optional[str] = None
    designated_site_rules_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AdditionalPermits(BaseModel):
    permits_or_passes: List[str] = Field(default_factory=list)
    explicit_none: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class TripPlanningExtraction(BaseModel):
    permit_details: Optional[PermitDetails] = None
    equipment_rules: Optional[EquipmentRules] = None
    camping_regulations: Optional[CampingRegulations] = None
    additional_permits: Optional[AdditionalPermits] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_planning_info() -> str:
    return (
        "Extract structured information from the answer for the Desolation Wilderness trip planning task.\n"
        "Return a JSON object with the following nested sections and fields. Extract values exactly as stated in the answer text. "
        "If a field is missing, set it to null; if URLs are requested, extract only valid URLs explicitly present in the answer.\n\n"
        "Sections and fields:\n"
        "1) permit_details:\n"
        "   - earliest_reservation_date: The earliest date to make a permit reservation (as stated in the answer).\n"
        "   - reservation_platform: The platform used to make reservations (e.g., 'Recreation.gov').\n"
        "   - total_permit_cost: The total permit cost for 8 people for 2 nights (as stated in the answer; keep formatting from answer).\n"
        "   - sources: An array of URLs specifically cited in the answer for permit reservation details (official pages, Recreation.gov, etc.).\n"
        "2) equipment_rules:\n"
        "   - bear_resistant_food_storage: The required type of bear-resistant container (e.g., 'hard-sided bear canister required').\n"
        "   - campfire_rules: The rule regarding campfires and stoves (e.g., 'campfires prohibited; only stoves permitted').\n"
        "   - camp_stove_permit: Whether a California campfire permit is required for camp stoves.\n"
        "   - sources: An array of URLs cited in the answer supporting equipment/fire/stove rules.\n"
        "3) camping_regulations:\n"
        "   - maximum_group_size: The maximum group size allowed (e.g., '12').\n"
        "   - minimum_camping_distance: The minimum distance to camp from lakes, streams, and trails (e.g., '100 feet').\n"
        "   - designated_site_rules_text: Any statement in the answer about designated sites and 500-foot lakeshore rules (exact text from answer).\n"
        "   - sources: An array of URLs cited in the answer supporting camping regulations.\n"
        "4) additional_permits:\n"
        "   - permits_or_passes: An array listing any additional permits/passes mentioned (e.g., parking, access passes).\n"
        "   - explicit_none: Set to true if the answer explicitly states no additional permits/passes are needed beyond the wilderness permit; otherwise false or null.\n"
        "   - sources: An array of URLs cited in the answer relevant to additional permits/passes.\n\n"
        "Follow the SPECIAL RULES FOR URL EXTRACTION: extract actual URLs (including protocol), and do not invent any URL."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_permit_reservation_details(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanningExtraction,
) -> None:
    """
    Build and verify the 'permit_reservation_details' subtree with critical leaves:
    - earliest_reservation_date
    - reservation_platform
    - total_permit_cost
    """
    pd = extracted.permit_details or PermitDetails()
    section_node = evaluator.add_parallel(
        id="permit_reservation_details",
        desc="Permit reservation timing, cost, and reservation platform are correct",
        parent=parent_node,
        critical=True  # Critical group: all children must be critical by framework constraint
    )

    # Leaf: earliest reservation date (critical)
    earliest_node = evaluator.add_leaf(
        id="earliest_reservation_date",
        desc="States the earliest reservation date based on the rule that reservations can be made up to 6 months before the July 11, 2026 entry date (i.e., Jan 11, 2026).",
        parent=section_node,
        critical=True
    )
    claim_earliest = (
        f"The answer correctly states that the earliest reservation date for a July 11, 2026 entry, "
        f"given a 6-month advance reservation rule, is {EARLIEST_RESERVATION_DATE_TEXT}."
    )
    await evaluator.verify(
        claim=claim_earliest,
        node=earliest_node,
        # Use simple verification to check alignment with the answer; allow textual date variants
        additional_instruction="Accept equivalent date formats such as 'Jan 11, 2026' or '01/11/2026' when judging correctness."
    )

    # Leaf: reservation platform (critical)
    platform_node = evaluator.add_leaf(
        id="reservation_platform",
        desc="Identifies Recreation.gov as the platform for Desolation Wilderness permit reservations.",
        parent=section_node,
        critical=True
    )
    claim_platform = "The answer identifies Recreation.gov as the platform to make Desolation Wilderness overnight permit reservations."
    await evaluator.verify(
        claim=claim_platform,
        node=platform_node,
        # If the answer cited official URLs for permitting, use them; otherwise simple verify
        sources=pd.sources if pd.sources else None,
        additional_instruction="Focus on whether the answer itself names 'Recreation.gov'; allow minor variations in capitalization."
    )

    # Leaf: total permit cost (critical)
    total_cost_node = evaluator.add_leaf(
        id="total_permit_cost",
        desc="Gives the correct total permit cost for 8 people for 2 nights using the stated fee schedule ($6 reservation fee + $10 per person for 2–14 nights).",
        parent=section_node,
        critical=True
    )
    claim_total = (
        f"For a group of 8 people on a 2-night trip, using a $6 reservation fee plus $10 per person (for 2–14 nights), "
        f"the correct total permit cost is ${EXPECTED_TOTAL_COST}, and the answer's stated total matches ${EXPECTED_TOTAL_COST}."
    )
    await evaluator.verify(
        claim=claim_total,
        node=total_cost_node,
        # Use simple verification to judge if the stated total in the answer matches the correct computed total.
        additional_instruction=(
            "Judge correctness using the provided fee schedule ($6 reservation fee + $10 per person for 2–14 nights). "
            "Accept equivalent currency formatting (e.g., '86 USD', '$86 total')."
        )
    )


async def verify_required_equipment_and_fire_rules(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanningExtraction,
) -> None:
    """
    Build and verify the 'required_equipment_and_fire_rules' subtree with critical leaves:
    - bear_resistant_food_storage
    - campfire_rules
    - camp_stove_permit
    """
    eq = extracted.equipment_rules or EquipmentRules()
    section_node = evaluator.add_parallel(
        id="required_equipment_and_fire_rules",
        desc="Required food storage and fire/stove rules are correct",
        parent=parent_node,
        critical=True
    )

    # Leaf: bear-resistant food storage (critical)
    bear_node = evaluator.add_leaf(
        id="bear_resistant_food_storage",
        desc="States that hard-sided bear canisters are mandatory for all overnight trips.",
        parent=section_node,
        critical=True
    )
    claim_bear = "Hard-sided bear canisters are mandatory for all overnight trips in Desolation Wilderness."
    await evaluator.verify(
        claim=claim_bear,
        node=bear_node,
        sources=eq.sources if eq.sources else None,
        additional_instruction="Verify against official USFS/Desolation Wilderness guidance if URLs are provided; otherwise judge based on the answer."
    )

    # Leaf: campfire rules (critical)
    campfire_node = evaluator.add_leaf(
        id="campfire_rules",
        desc="States that campfires are prohibited and only camp stoves are permitted.",
        parent=section_node,
        critical=True
    )
    claim_campfire = "Campfires are prohibited in Desolation Wilderness; only camp stoves are permitted."
    await evaluator.verify(
        claim=claim_campfire,
        node=campfire_node,
        sources=eq.sources if eq.sources else None,
        additional_instruction="If URLs are provided, check official restrictions; allow equivalent phrasing indicating prohibition of campfires."
    )

    # Leaf: camp stove permit (critical)
    stove_permit_node = evaluator.add_leaf(
        id="camp_stove_permit",
        desc="States that a (free) California campfire permit is required for operating camp stoves on federal lands.",
        parent=section_node,
        critical=True
    )
    claim_stove_permit = "A free California campfire permit is required to operate camp stoves on federal lands, including Desolation Wilderness."
    await evaluator.verify(
        claim=claim_stove_permit,
        node=stove_permit_node,
        sources=eq.sources if eq.sources else None,
        additional_instruction="Accept equivalent phrasing such as 'CA campfire permit required for stoves'."
    )


async def verify_camping_regulations(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanningExtraction,
) -> None:
    """
    Build and verify the critical 'camping_regulations' subtree:
    - maximum_group_size
    - minimum_camping_distance

    And a separate non-critical sibling node to avoid framework constraint:
    - designated_site_rules_no_contradiction (non-critical)
    """
    cr = extracted.camping_regulations or CampingRegulations()

    # Critical regulations node
    regs_node = evaluator.add_parallel(
        id="camping_regulations",
        desc="Camping regulations (group size and campsite distance rules) are correct",
        parent=parent_node,
        critical=True
    )

    # Leaf: maximum group size (critical)
    max_group_node = evaluator.add_leaf(
        id="maximum_group_size",
        desc="States the maximum group size is 12 people.",
        parent=regs_node,
        critical=True
    )
    claim_max_group = "The maximum group size allowed in Desolation Wilderness is 12 people."
    await evaluator.verify(
        claim=claim_max_group,
        node=max_group_node,
        sources=cr.sources if cr.sources else None,
        additional_instruction="Allow numeric formatting variants (e.g., '12', 'twelve')."
    )

    # Leaf: minimum camping distance (critical)
    min_distance_node = evaluator.add_leaf(
        id="minimum_camping_distance",
        desc="States the minimum camping distance is at least 100 feet from lakes, streams, and trails.",
        parent=regs_node,
        critical=True
    )
    claim_min_distance = "Camping must be at least 100 feet from lakes, streams, and trails in Desolation Wilderness."
    await evaluator.verify(
        claim=claim_min_distance,
        node=min_distance_node,
        sources=cr.sources if cr.sources else None,
        additional_instruction="Accept equivalent phrasing indicating a minimum of 100 feet from water bodies and trails."
    )

    # Non-critical designated site consistency check as separate sibling (to satisfy framework rule)
    designated_sibling = evaluator.add_parallel(
        id="designated_site_rules_consistency",
        desc="Designated-site/500-foot rules consistency check (non-critical)",
        parent=parent_node,
        critical=False
    )
    designated_leaf = evaluator.add_leaf(
        id="designated_site_rules_no_contradiction",
        desc="Does not contradict the designated-site/500-foot rules in the constraints; if the answer mentions designated sites, it must match: use designated sites when available at certain lakes, and if all designated sites are occupied then camp >500 feet from the lakeshore.",
        parent=designated_sibling,
        critical=False
    )
    claim_designated = (
        "The answer does not contradict the designated-site rules: use designated sites when available at certain lakes; "
        "if all designated sites are occupied, camp more than 500 feet from the lakeshore."
    )
    await evaluator.verify(
        claim=claim_designated,
        node=designated_leaf,
        additional_instruction=(
            "If the answer mentions designated sites, ensure it matches the rule summarized above. "
            "If designated sites are not mentioned, this check should still pass as non-contradicting."
        )
    )


async def verify_additional_permits_or_passes(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanningExtraction,
) -> None:
    """
    Build and verify the 'additional_permits_or_passes' subtree (non-critical):
    - mentions_additional_access_permit_or_explicit_none
    """
    ap = extracted.additional_permits or AdditionalPermits()
    section_node = evaluator.add_parallel(
        id="additional_permits_or_passes",
        desc="Addresses the question’s request for other permits/passes helpful for access",
        parent=parent_node,
        critical=False
    )

    # Leaf implemented via custom check: either they listed at least one permit/pass or explicitly 'none'
    mentions = bool(ap.permits_or_passes) if ap.permits_or_passes is not None else False
    explicit_none = bool(ap.explicit_none) if ap.explicit_none is not None else False
    result_ok = mentions or explicit_none

    evaluator.add_custom_node(
        result=result_ok,
        id="mentions_additional_access_permit_or_explicit_none",
        desc="Either (a) mentions at least one additional permit/pass that may be helpful for access/parking/entry, or (b) explicitly states that no additional permits/passes are needed beyond the wilderness permit (without contradicting the provided constraints).",
        parent=section_node,
        critical=False
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the agent's answer for the Desolation Wilderness trip planning task using the Mind2Web2 framework.
    """
    # Initialize evaluator and root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel per rubric
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

    # Record ground truth helpful info for transparency
    evaluator.add_ground_truth(GROUND_TRUTH_INFO, gt_type="ground_truth")

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_planning_info(),
        template_class=TripPlanningExtraction,
        extraction_name="trip_planning_extraction"
    )

    # Build and verify subtrees based on rubric
    await verify_permit_reservation_details(evaluator, root, extracted)
    await verify_required_equipment_and_fire_rules(evaluator, root, extracted)
    await verify_camping_regulations(evaluator, root, extracted)
    await verify_additional_permits_or_passes(evaluator, root, extracted)

    # Return standardized evaluation summary
    return evaluator.get_summary()