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
TASK_ID = "nc_outdoor_facility_multi_constraints"
TASK_DESCRIPTION = (
    "Identify an outdoor recreation facility in the United States that meets all of the following criteria: "
    "(1) Located in the state of North Carolina, "
    "(2) Accessible from a major commercial airport within 15 miles, "
    "(3) Offers whitewater rafting as an activity, "
    "(4) Charges less than $75 for a single-day whitewater rafting activity pass, "
    "(5) Offers rock climbing as an activity, "
    "(6) Charges less than $35 for a single-day climbing activity pass, "
    "(7) Offers mountain biking with dedicated trails, "
    "(8) Charges less than $45 for a single-day mountain biking activity pass, "
    "(9) Has at least 40 miles of trails, "
    "(10) Offers flatwater kayaking or stand-up paddleboarding (SUP), "
    "(11) Features a manmade whitewater river, "
    "(12) Charges a daily parking fee, "
    "(13) Offers more than 25 different recreational activities, "
    "(14) Is located on at least 1,000 acres of land. "
    "Provide the facility's name, complete address, and a reference URL that confirms these details."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    # Core identity
    facility_name: Optional[str] = None
    address_full: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    # Airport proximity
    airport_name: Optional[str] = None
    airport_code: Optional[str] = None
    airport_distance_miles: Optional[str] = None  # keep as string to be flexible (e.g., "10", "10 miles", "~13")

    # Activities and prices
    whitewater_rafting_offered: Optional[str] = None
    rafting_day_pass_price: Optional[str] = None

    rock_climbing_offered: Optional[str] = None
    climbing_day_pass_price: Optional[str] = None

    mountain_biking_offered: Optional[str] = None
    mountain_biking_has_dedicated_trails: Optional[str] = None
    mountain_biking_day_pass_price: Optional[str] = None

    # Other features and metrics
    trail_miles: Optional[str] = None
    flatwater_kayak_or_sup_offered: Optional[str] = None
    manmade_whitewater_river: Optional[str] = None
    daily_parking_fee: Optional[str] = None
    activities_count: Optional[str] = None
    acreage: Optional[str] = None

    # Evidence
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_info() -> str:
    return """
    From the answer, extract the details of the single outdoor recreation facility the user proposes.
    Only extract information explicitly present in the answer text; do not infer or invent.

    Required fields (use strings for all numeric values; if not present, return null):
    - facility_name: Full name of the facility.
    - address_full: Complete address text as presented.
    - city: City name if present.
    - state: State name or abbreviation (e.g., North Carolina or NC) if present.
    - zip_code: ZIP/postal code if present.

    - airport_name: The nearby major commercial airport named in the answer (e.g., Charlotte Douglas International Airport), if mentioned.
    - airport_code: IATA code if mentioned (e.g., CLT).
    - airport_distance_miles: The distance in miles to the airport as stated, if mentioned (keep as the original text, such as "10", "10 miles", "~12", etc.).

    Activities and prices (use "yes"/"no" when the answer clearly states availability; otherwise null):
    - whitewater_rafting_offered
    - rafting_day_pass_price  (the stated single-day price for whitewater rafting if provided, e.g., "$69", "USD 69", "68-74", etc.)
    - rock_climbing_offered
    - climbing_day_pass_price
    - mountain_biking_offered
    - mountain_biking_has_dedicated_trails  ("yes"/"no" if explicitly stated or strongly implied; else null)
    - mountain_biking_day_pass_price
    - trail_miles                    (string like "50", "50+", "over 40", etc.)
    - flatwater_kayak_or_sup_offered ("yes"/"no")
    - manmade_whitewater_river       ("yes"/"no")
    - daily_parking_fee              (string like "$6/day", "$8", etc., if provided)
    - activities_count               (string like "30+", "over 25", "30", etc.)
    - acreage                        (string like "1,300 acres", "1000+", etc.)

    Evidence:
    - reference_urls: extract ALL URLs shown in the answer that are used to support/confirm the details
      (include the facility’s official site pages and any other cited sources). If no URL is present, return an empty list.

    Return a single JSON object with these fields. Do not add extra fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(val: Optional[str]) -> str:
    return val.strip() if isinstance(val, str) else ""


def _sources(extracted: FacilityExtraction) -> List[str]:
    return extracted.reference_urls if extracted.reference_urls else []


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_facility(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    """
    Build the verification tree for the facility and run all checks
    following the rubric. All children under this node are critical.
    """
    facility_node = evaluator.add_parallel(
        id="Outdoor_Adventure_Facility",
        desc="An outdoor recreation facility in the United States that meets all specified criteria for activities, accessibility, pricing, and features",
        parent=parent_node,
        critical=True
    )

    # Existence gates (critical)
    name_and_address_provided = evaluator.add_custom_node(
        result=bool(_safe(extracted.facility_name)) and bool(_safe(extracted.address_full)),
        id="facility_name_address_provided",
        desc="Facility name and complete address are provided in the answer",
        parent=facility_node,
        critical=True
    )
    refs_provided = evaluator.add_custom_node(
        result=len(_sources(extracted)) > 0,
        id="references_provided",
        desc="At least one reference URL is provided in the answer",
        parent=facility_node,
        critical=True
    )

    sources_list = _sources(extracted)
    facility_name = _safe(extracted.facility_name) or "the facility"

    # 1) North Carolina location
    nc_node = evaluator.add_leaf(
        id="North_Carolina_Location",
        desc="The facility is located in the state of North Carolina",
        parent=facility_node,
        critical=True
    )
    nc_claim = f"{facility_name} is located in the state of North Carolina (NC)."
    await evaluator.verify(
        claim=nc_claim,
        node=nc_node,
        sources=sources_list,
        additional_instruction="Check the address or location statement on the cited page(s). Accept 'NC' as North Carolina."
    )

    # 2) Airport accessibility within 15 miles of a major commercial airport
    airport_node = evaluator.add_leaf(
        id="Airport_Accessibility",
        desc="The facility is accessible from a major commercial airport within 15 miles",
        parent=facility_node,
        critical=True
    )
    airport_name = _safe(extracted.airport_name)
    airport_code = _safe(extracted.airport_code)
    airport_distance = _safe(extracted.airport_distance_miles)
    airport_hint = f" (e.g., {airport_name} {airport_code})" if airport_name or airport_code else ""
    airport_claim = f"{facility_name} is within 15 miles of a major commercial airport{airport_hint}."
    await evaluator.verify(
        claim=airport_claim,
        node=airport_node,
        sources=sources_list,
        additional_instruction=(
            "Verify that a major commercial airport is named (e.g., CLT, RDU) and that the distance is stated as 15 miles or less. "
            "If an explicit distance is shown as ≤ 15 miles, accept. If the page explicitly states a nearby international airport "
            "within 15 miles, accept. Otherwise, do not infer; fail the check."
        )
    )

    # 3) Whitewater rafting available
    ww_available_node = evaluator.add_leaf(
        id="Whitewater_Rafting_Available",
        desc="The facility offers whitewater rafting as an activity option",
        parent=facility_node,
        critical=True
    )
    ww_available_claim = f"{facility_name} offers whitewater rafting."
    await evaluator.verify(
        claim=ww_available_claim,
        node=ww_available_node,
        sources=sources_list,
        additional_instruction="Look for activity lists or pages describing whitewater rafting being available at the facility."
    )

    # 4) Rafting day pass < $75
    ww_price_node = evaluator.add_leaf(
        id="Rafting_Price_Under_75",
        desc="The single-day whitewater rafting activity pass costs less than $75",
        parent=facility_node,
        critical=True
    )
    ww_price_claim = f"{facility_name} sells a single-day whitewater rafting activity pass for less than $75 before taxes/fees."
    await evaluator.verify(
        claim=ww_price_claim,
        node=ww_price_node,
        sources=sources_list,
        additional_instruction=(
            "Confirm that a single-day pass specific to whitewater rafting (or a single-activity day pass that includes rafting) "
            "is priced under $75 for a standard adult ticket. If multiple prices exist, it's sufficient that at least one normal single-day "
            "option is < $75. Exclude equipment rental add-ons and taxes/fees."
        )
    )

    # 5) Rock climbing available
    climb_available_node = evaluator.add_leaf(
        id="Rock_Climbing_Available",
        desc="The facility offers rock climbing as an activity option",
        parent=facility_node,
        critical=True
    )
    climb_available_claim = f"{facility_name} offers rock climbing."
    await evaluator.verify(
        claim=climb_available_claim,
        node=climb_available_node,
        sources=sources_list,
        additional_instruction="Check activities or tickets pages for 'rock climbing', 'climbing', 'top rope', 'bouldering', or similar."
    )

    # 6) Climbing day pass < $35
    climb_price_node = evaluator.add_leaf(
        id="Climbing_Price_Under_35",
        desc="The single-day climbing activity pass costs less than $35",
        parent=facility_node,
        critical=True
    )
    climb_price_claim = f"{facility_name} sells a single-day rock climbing activity pass for less than $35 before taxes/fees."
    await evaluator.verify(
        claim=climb_price_claim,
        node=climb_price_node,
        sources=sources_list,
        additional_instruction=(
            "Confirm a single-day pass specific to climbing (or a single-activity day pass that grants climbing access) costs < $35 "
            "for a standard adult ticket. Exclude rentals and taxes/fees."
        )
    )

    # 7) Mountain biking available with dedicated trails
    mb_available_node = evaluator.add_leaf(
        id="Mountain_Biking_Available",
        desc="The facility offers mountain biking with dedicated trails",
        parent=facility_node,
        critical=True
    )
    mb_available_claim = f"{facility_name} offers mountain biking with dedicated trails."
    await evaluator.verify(
        claim=mb_available_claim,
        node=mb_available_node,
        sources=sources_list,
        additional_instruction="Confirm the presence of mountain biking and a dedicated trail network (not just rentals on roads)."
    )

    # 8) Mountain biking day pass < $45
    mb_price_node = evaluator.add_leaf(
        id="Biking_Price_Under_45",
        desc="The single-day mountain biking activity pass costs less than $45",
        parent=facility_node,
        critical=True
    )
    mb_price_claim = f"{facility_name} sells a single-day mountain biking activity pass for less than $45 before taxes/fees."
    await evaluator.verify(
        claim=mb_price_claim,
        node=mb_price_node,
        sources=sources_list,
        additional_instruction=(
            "Confirm a single-day pass for mountain biking access costs < $45 for a standard adult. "
            "If multiple tiers exist, at least one normal single-day option must be < $45. Exclude rentals and taxes/fees."
        )
    )

    # 9) At least 40 miles of trails
    trails_node = evaluator.add_leaf(
        id="Trail_Mileage_40_Plus",
        desc="The facility has at least 40 miles of trails available",
        parent=facility_node,
        critical=True
    )
    trails_claim = f"{facility_name} has at least 40 miles of trails."
    await evaluator.verify(
        claim=trails_claim,
        node=trails_node,
        sources=sources_list,
        additional_instruction="Accept phrases like '40+ miles', 'over 40 miles', or any explicit total >= 40."
    )

    # 10) Flatwater kayaking or SUP available
    flatwater_node = evaluator.add_leaf(
        id="Flatwater_Paddling_Available",
        desc="The facility offers flatwater kayaking or stand-up paddleboarding (SUP)",
        parent=facility_node,
        critical=True
    )
    flatwater_claim = f"{facility_name} offers flatwater kayaking or stand-up paddleboarding (SUP)."
    await evaluator.verify(
        claim=flatwater_claim,
        node=flatwater_node,
        sources=sources_list,
        additional_instruction="Look for 'flatwater kayaking', 'flatwater', 'SUP', 'stand-up paddleboarding' as listed activities."
    )

    # 11) Manmade whitewater river
    manmade_node = evaluator.add_leaf(
        id="Manmade_Whitewater_River",
        desc="The facility features a manmade whitewater river",
        parent=facility_node,
        critical=True
    )
    manmade_claim = f"{facility_name} features a manmade (artificial) whitewater river."
    await evaluator.verify(
        claim=manmade_claim,
        node=manmade_node,
        sources=sources_list,
        additional_instruction="Confirm that the whitewater course is artificial/manmade (not a natural river)."
    )

    # 12) Daily parking fee
    parking_node = evaluator.add_leaf(
        id="Daily_Parking_Fee",
        desc="The facility charges a daily parking fee for vehicles",
        parent=facility_node,
        critical=True
    )
    parking_claim = f"{facility_name} charges a daily parking fee."
    await evaluator.verify(
        claim=parking_claim,
        node=parking_node,
        sources=sources_list,
        additional_instruction="Check for information such as 'parking is $X/day', 'daily parking fee', or similar."
    )

    # 13) Activity count > 25
    activity_count_node = evaluator.add_leaf(
        id="Activity_Count_25_Plus",
        desc="The facility offers more than 25 different recreational activities",
        parent=facility_node,
        critical=True
    )
    activity_count_claim = f"{facility_name} offers more than 25 different recreational activities."
    await evaluator.verify(
        claim=activity_count_claim,
        node=activity_count_node,
        sources=sources_list,
        additional_instruction="Accept 'over 25 activities', '30+ activities', or any explicit assertion > 25."
    )

    # 14) Land area >= 1,000 acres
    acreage_node = evaluator.add_leaf(
        id="Land_Area_1000_Plus_Acres",
        desc="The facility is located on at least 1,000 acres of land",
        parent=facility_node,
        critical=True
    )
    acreage_claim = f"{facility_name} is located on at least 1,000 acres of land."
    await evaluator.verify(
        claim=acreage_claim,
        node=acreage_node,
        sources=sources_list,
        additional_instruction="Accept '1,000+ acres', 'over 1,000 acres', 'approximately 1,300 acres', etc., as ≥ 1000."
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
    Evaluate an answer for the NC outdoor recreation facility multi-constraint task.
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
        default_model=model
    )

    # Extract structured facility info from the answer text
    extracted = await evaluator.extract(
        prompt=prompt_extract_facility_info(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction"
    )

    # Build verification tree and run verifications
    await build_and_verify_facility(evaluator, root, extracted)

    return evaluator.get_summary()