import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travel_planning_verification"
TASK_DESCRIPTION = (
    "A traveler is planning a winter trip and needs to verify specific travel details. Provide the following information: "
    "(1) Which road in Yellowstone National Park remains open to regular private vehicles year-round during winter (December-January)? "
    "(2) What is the standard maximum weight limit (in pounds) for a checked bag on Frontier Airlines? "
    "(3) How many days before travel to Aruba must travelers complete the mandatory ED Card? "
    "(4) Is the Aruba ED Card mandatory for all air travelers entering Aruba? "
    "(5) What is the entrance fee for a private vehicle at Yellowstone National Park, and how many consecutive days is this pass valid?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RoadInfo(BaseModel):
    road_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FrontierCheckedBagPolicy(BaseModel):
    checked_bag_weight_lb: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArubaEDCardDeadline(BaseModel):
    days_before_travel: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArubaEDCardMandatory(BaseModel):
    is_mandatory_for_all_air_travelers: Optional[str] = None  # Expect values like "yes", "no", "true", "false"
    sources: List[str] = Field(default_factory=list)


class YellowstoneVehiclePass(BaseModel):
    entrance_fee: Optional[str] = None
    validity_days: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TravelDetailsExtraction(BaseModel):
    yellowstone_winter_road: Optional[RoadInfo] = None
    frontier_checked_bag_weight: Optional[FrontierCheckedBagPolicy] = None
    aruba_ed_card_deadline: Optional[ArubaEDCardDeadline] = None
    aruba_ed_card_mandatory: Optional[ArubaEDCardMandatory] = None
    yellowstone_vehicle_pass: Optional[YellowstoneVehiclePass] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_details() -> str:
    return """
Extract from the answer the exact values the assistant provided for each of the following, along with the specific source URLs (as they appear in the answer) that support each item. Do not invent or infer URLs.

Fields to extract:
1) yellowstone_winter_road:
   - road_name: The road/road segment described as remaining open to regular private (wheeled) vehicles during winter (December–January).
   - sources: Array of URLs cited for this statement.

2) frontier_checked_bag_weight:
   - checked_bag_weight_lb: The standard maximum weight limit for a checked bag on Frontier Airlines (as stated), e.g., "40 lb" or "50 pounds" or just "40".
   - sources: Array of URLs cited for this statement.

3) aruba_ed_card_deadline:
   - days_before_travel: How many days before travel to Aruba the ED Card must be completed (as stated), e.g., "3 days", "within 3 days", "up to 7 days".
   - sources: Array of URLs cited for this statement.

4) aruba_ed_card_mandatory:
   - is_mandatory_for_all_air_travelers: Whether the Aruba ED Card is mandatory for all air travelers entering Aruba, in simple form like "yes"/"no" (or "true"/"false") exactly as implied by the answer.
   - sources: Array of URLs cited for this statement.

5) yellowstone_vehicle_pass:
   - entrance_fee: The entrance fee amount for a private vehicle at Yellowstone National Park (include currency if present, e.g., "$35").
   - validity_days: How many consecutive days this private-vehicle entrance pass is valid (e.g., "7 days" or "7").
   - sources: Array of URLs cited for these fee/validity details.

General instructions:
- Extract exactly what is explicitly stated in the answer text.
- For any field not stated in the answer, set it to null (or empty list for sources).
- Source extraction: only include URLs explicitly present in the answer (plain URLs or markdown links). Do not infer or fabricate links.
"""


# --------------------------------------------------------------------------- #
# Helper: normalize yes/no                                                    #
# --------------------------------------------------------------------------- #
def _is_affirmative(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    t = text.strip().lower()
    if t in {"yes", "true", "y", "mandatory", "required"}:
        return True
    if t in {"no", "false", "n", "not mandatory", "optional"}:
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_yellowstone_winter_road(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelDetailsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="yellowstone_winter_road_access_group",
        desc="Identify which Yellowstone road remains open to regular private vehicles during winter (Dec–Jan).",
        parent=parent_node,
        critical=True,
    )

    info = extracted.yellowstone_winter_road or RoadInfo()
    exists = bool(info.road_name and info.road_name.strip()) and bool(info.sources)

    evaluator.add_custom_node(
        result=exists,
        id="yellowstone_winter_road_access_exists",
        desc="Answer provides a road name and at least one source URL for Yellowstone winter private-vehicle access.",
        parent=node,
        critical=True,
    )

    verify_leaf = evaluator.add_leaf(
        id="yellowstone_winter_road_access",
        desc="Yellowstone winter road access statement is supported by cited sources.",
        parent=node,
        critical=True,
    )

    claim = (
        f"During winter (December–January), a road that remains open to regular private (wheeled) vehicles in Yellowstone National Park is: '{info.road_name}'."
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=info.sources,
        additional_instruction=(
            "Verify that the provided webpage(s) explicitly state that this road segment remains open "
            "to regular private (wheeled) vehicles during winter months (including December–January). "
            "Do not count snowmobile/snowcoach-only access as 'open to regular private vehicles'. "
            "Accept equivalent naming of the same road/segment."
        ),
    )


async def verify_frontier_checked_bag_weight(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelDetailsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="frontier_checked_baggage_weight_group",
        desc="State Frontier Airlines' standard maximum checked-bag weight (pounds).",
        parent=parent_node,
        critical=True,
    )

    info = extracted.frontier_checked_bag_weight or FrontierCheckedBagPolicy()
    exists = bool(info.checked_bag_weight_lb and info.checked_bag_weight_lb.strip()) and bool(info.sources)

    evaluator.add_custom_node(
        result=exists,
        id="frontier_checked_baggage_weight_exists",
        desc="Answer provides a weight value and at least one source URL for Frontier's checked-bag maximum weight.",
        parent=node,
        critical=True,
    )

    verify_leaf = evaluator.add_leaf(
        id="frontier_checked_baggage_weight",
        desc="Frontier checked-bag maximum weight statement is supported by cited sources.",
        parent=node,
        critical=True,
    )

    claim = (
        f"The standard maximum weight for a checked bag on Frontier Airlines is {info.checked_bag_weight_lb} pounds."
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=info.sources,
        additional_instruction=(
            "Confirm the standard (non-exception) maximum weight limit for a checked bag before overweight surcharges apply. "
            "Rely on the cited policy pages. Accept minor formatting differences (e.g., '40 lb' vs '40 pounds')."
        ),
    )


async def verify_aruba_ed_card_deadline(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelDetailsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="aruba_ed_card_deadline_group",
        desc="State how many days before travel to Aruba the ED Card must be completed.",
        parent=parent_node,
        critical=True,
    )

    info = extracted.aruba_ed_card_deadline or ArubaEDCardDeadline()
    exists = bool(info.days_before_travel and info.days_before_travel.strip()) and bool(info.sources)

    evaluator.add_custom_node(
        result=exists,
        id="aruba_ed_card_deadline_exists",
        desc="Answer provides a days-before-travel value and at least one source URL for the Aruba ED Card deadline.",
        parent=node,
        critical=True,
    )

    verify_leaf = evaluator.add_leaf(
        id="aruba_ed_card_deadline",
        desc="Aruba ED Card deadline statement is supported by cited sources.",
        parent=node,
        critical=True,
    )

    claim = (
        f"Travelers must complete Aruba's ED Card {info.days_before_travel} before travel."
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=info.sources,
        additional_instruction=(
            "Verify the timing requirement for completing Aruba's online ED Card prior to travel. "
            "Treat phrasing like 'within X days before travel' or 'up to X days prior' as equivalent to 'X days before travel' if they match in number. "
            "Use only the cited webpages."
        ),
    )


async def verify_aruba_ed_card_mandatory(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelDetailsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="aruba_ed_card_mandatory_group",
        desc="Confirm whether the Aruba ED Card is mandatory for all air travelers entering Aruba.",
        parent=parent_node,
        critical=True,
    )

    info = extracted.aruba_ed_card_mandatory or ArubaEDCardMandatory()
    exists = (info.is_mandatory_for_all_air_travelers is not None and str(info.is_mandatory_for_all_air_travelers).strip() != "") and bool(info.sources)

    evaluator.add_custom_node(
        result=exists,
        id="aruba_ed_card_mandatory_exists",
        desc="Answer indicates 'mandatory or not' and provides at least one source URL regarding the Aruba ED Card requirement.",
        parent=node,
        critical=True,
    )

    verify_leaf = evaluator.add_leaf(
        id="aruba_ed_card_mandatory",
        desc="Aruba ED Card mandate statement is supported by cited sources.",
        parent=node,
        critical=True,
    )

    affirmative = _is_affirmative(info.is_mandatory_for_all_air_travelers)
    if affirmative is True:
        claim = "The Aruba ED Card is mandatory for all air travelers entering Aruba."
    elif affirmative is False:
        claim = "The Aruba ED Card is not mandatory for all air travelers entering Aruba."
    else:
        # If unclear, preserve the raw text in the claim
        claim = f"The Aruba ED Card mandatory status for all air travelers entering Aruba is: '{info.is_mandatory_for_all_air_travelers}'."

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=info.sources,
        additional_instruction=(
            "Determine from the cited webpages whether all air travelers entering Aruba must complete the ED Card. "
            "A statement like 'all travelers must complete the ED card' should be treated as 'mandatory'."
        ),
    )


async def verify_yellowstone_vehicle_pass_details(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelDetailsExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="yellowstone_vehicle_pass_details",
        desc="Provide Yellowstone private-vehicle entrance fee and pass validity duration.",
        parent=parent_node,
        critical=True,
    )

    info = extracted.yellowstone_vehicle_pass or YellowstoneVehiclePass()
    exists = (
        bool(info.entrance_fee and info.entrance_fee.strip())
        and bool(info.validity_days and info.validity_days.strip())
        and bool(info.sources)
    )

    evaluator.add_custom_node(
        result=exists,
        id="yellowstone_vehicle_pass_details_exists",
        desc="Answer provides a fee amount, a validity duration, and at least one source URL for Yellowstone private-vehicle entry.",
        parent=group,
        critical=True,
    )

    fee_leaf = evaluator.add_leaf(
        id="yellowstone_vehicle_entrance_fee_amount",
        desc="Yellowstone private-vehicle entrance fee amount is supported by cited sources.",
        parent=group,
        critical=True,
    )
    fee_claim = f"The entrance fee for a private vehicle at Yellowstone National Park is {info.entrance_fee}."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=info.sources,
        additional_instruction=(
            "Verify the stated private-vehicle entrance fee for Yellowstone from the cited webpages. "
            "Accept currency formatting variations (e.g., '$35', '35 USD')."
        ),
    )

    validity_leaf = evaluator.add_leaf(
        id="yellowstone_vehicle_pass_validity_days",
        desc="Yellowstone private-vehicle pass validity duration (consecutive days) is supported by cited sources.",
        parent=group,
        critical=True,
    )
    validity_claim = f"The Yellowstone private-vehicle entrance pass is valid for {info.validity_days} consecutive days."
    await evaluator.verify(
        claim=validity_claim,
        node=validity_leaf,
        sources=info.sources,
        additional_instruction=(
            "Verify the number of consecutive days for which the private-vehicle entrance pass is valid. "
            "Accept phrasing variations like 'valid for 7 days' or 'good for seven consecutive days'."
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
    Evaluate a single answer for the winter travel planning details task.
    """
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

    # Extraction
    extracted: TravelDetailsExtraction = await evaluator.extract(
        prompt=prompt_extract_travel_details(),
        template_class=TravelDetailsExtraction,
        extraction_name="travel_details_extraction",
    )

    # Build rubric root (critical, parallel)
    rubric_root = evaluator.add_parallel(
        id="travel_planning_verification",
        desc="Provide all requested travel details (Yellowstone winter road access, Frontier checked-bag weight, Aruba ED Card timing and mandate, and Yellowstone vehicle entrance fee/pass validity).",
        parent=root,
        critical=True,
    )

    # Verification subtrees
    await verify_yellowstone_winter_road(evaluator, rubric_root, extracted)
    await verify_frontier_checked_bag_weight(evaluator, rubric_root, extracted)
    await verify_aruba_ed_card_deadline(evaluator, rubric_root, extracted)
    await verify_aruba_ed_card_mandatory(evaluator, rubric_root, extracted)
    await verify_yellowstone_vehicle_pass_details(evaluator, rubric_root, extracted)

    return evaluator.get_summary()