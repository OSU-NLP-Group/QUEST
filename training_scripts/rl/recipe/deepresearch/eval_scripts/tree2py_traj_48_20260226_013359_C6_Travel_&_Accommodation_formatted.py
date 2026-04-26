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
TASK_ID = "white_lotus_koh_samui_trip"
TASK_DESCRIPTION = (
    "You are planning a trip to visit the resort where White Lotus Season 3 was primarily filmed in Thailand. "
    "For your travel planning, you need to:\n"
    "1) Identify the specific Four Seasons resort used for filming on Koh Samui and verify how many total accommodations (rooms/villas) it has.\n"
    "2) Determine the complete flight routing from the United States to reach this resort, specifically using Turkish Airlines to Bangkok (which airport), "
    "and identify the domestic carrier to Koh Samui and why it has a monopoly; provide USM airport code.\n"
    "3) Research American Express Platinum Card benefits for this trip: lounge access at BKK and via which network, "
    "the annual spend threshold to unlock complimentary guest access (up to 2 guests) at Centurion Lounges, "
    "and the current annual fee.\n"
    "For each piece of information, provide supporting reference URLs."
)

# Ground truth expectations (for context in summary)
GROUND_TRUTH_INFO = {
    "expected_resort": "Four Seasons Resort Koh Samui",
    "expected_location": "Koh Samui, Thailand",
    "expected_total_accommodations": "71",
    "expected_breakdown": "60 pool villas and 11 private residences",
    "expected_bangkok_airport": "BKK (Suvarnabhumi Airport)",
    "expected_domestic_carrier": "Bangkok Airways",
    "expected_usm_code": "USM",
    "expected_centurion_spend_threshold": "$75,000",
    "expected_centurion_guest_count": "2",
    "expected_amex_platinum_annual_fee_2026": "$895",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortInfo(BaseModel):
    brand_name: Optional[str] = None
    location: Optional[str] = None
    total_accommodations: Optional[str] = None
    breakdown: Optional[str] = None
    resort_urls: List[str] = Field(default_factory=list)


class TurkishRouteInfo(BaseModel):
    bangkok_service: Optional[str] = None
    bangkok_airport_code: Optional[str] = None
    koh_samui_direct: Optional[str] = None
    turkish_urls: List[str] = Field(default_factory=list)


class DomesticConnectionInfo(BaseModel):
    carrier_name: Optional[str] = None
    monopoly_reason: Optional[str] = None
    usm_code: Optional[str] = None
    domestic_urls: List[str] = Field(default_factory=list)


class LoungeAccessInfo(BaseModel):
    bkk_access: Optional[str] = None
    network_name: Optional[str] = None
    lounge_urls: List[str] = Field(default_factory=list)


class GuestAccessInfo(BaseModel):
    annual_spend: Optional[str] = None
    guest_count: Optional[str] = None
    guest_access_urls: List[str] = Field(default_factory=list)


class AnnualFeeInfo(BaseModel):
    annual_fee_amount: Optional[str] = None
    fee_urls: List[str] = Field(default_factory=list)


class TravelPlanExtraction(BaseModel):
    resort: Optional[ResortInfo] = None
    turkish: Optional[TurkishRouteInfo] = None
    domestic: Optional[DomesticConnectionInfo] = None
    lounge: Optional[LoungeAccessInfo] = None
    guest_access: Optional[GuestAccessInfo] = None
    annual_fee: Optional[AnnualFeeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_plan() -> str:
    return (
        "Extract the following structured information from the answer. Use exactly what the answer states; do not invent any values.\n\n"
        "1) Resort identification and capacity (filming location on Koh Samui):\n"
        "- resort.brand_name: The resort name/brand (e.g., 'Four Seasons Resort Koh Samui').\n"
        "- resort.location: The geographic location (e.g., 'Koh Samui, Thailand').\n"
        "- resort.total_accommodations: The total number of accommodations (e.g., '71').\n"
        "- resort.breakdown: The composition (e.g., '60 pool villas and 11 private residences').\n"
        "- resort.resort_urls: All URLs cited that support the resort identification and capacity.\n\n"
        "2) International routing via Turkish Airlines to Bangkok:\n"
        "- turkish.bangkok_service: Whether the answer says Turkish Airlines operates flights to Bangkok (use 'yes'/'no' or a short phrase).\n"
        "- turkish.bangkok_airport_code: The Bangkok airport code Turkish Airlines serves (e.g., 'BKK').\n"
        "- turkish.koh_samui_direct: Whether Turkish Airlines flies directly to Koh Samui (e.g., 'no', 'does not', or 'not direct').\n"
        "- turkish.turkish_urls: All URLs cited for Turkish Airlines routing info.\n\n"
        "3) Domestic connection Bangkok → Koh Samui:\n"
        "- domestic.carrier_name: The airline operating Bangkok–Koh Samui (e.g., 'Bangkok Airways').\n"
        "- domestic.monopoly_reason: The reason for monopoly (e.g., 'Bangkok Airways owns Koh Samui Airport').\n"
        "- domestic.usm_code: Koh Samui (Samui) Airport code (e.g., 'USM').\n"
        "- domestic.domestic_urls: All URLs cited for domestic flight info.\n\n"
        "4) Amex Platinum benefits:\n"
        "- lounge.bkk_access: Whether Amex Platinum provides lounge access at Bangkok Suvarnabhumi (BKK) (e.g., 'yes').\n"
        "- lounge.network_name: The lounge network used (e.g., 'Global Lounge Collection' or 'Priority Pass').\n"
        "- lounge.lounge_urls: All URLs cited for BKK lounge access.\n"
        "- guest_access.annual_spend: Annual spending threshold to unlock complimentary Centurion Lounge guest access (e.g., '$75,000').\n"
        "- guest_access.guest_count: Complimentary guest count after meeting threshold (e.g., '2').\n"
        "- guest_access.guest_access_urls: All URLs cited for guest access requirements.\n"
        "- annual_fee.annual_fee_amount: The current annual fee for Amex Platinum (e.g., '$895').\n"
        "- annual_fee.fee_urls: All URLs cited for the annual fee.\n\n"
        "Rules:\n"
        "- Always return full valid URLs in the *_urls fields; if none are given, return an empty list.\n"
        "- If any field is not explicitly stated in the answer, return null for that field.\n"
        "- Do not infer values; extract only what is present.\n"
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_filming_location(
    evaluator: Evaluator,
    parent_node,
    data: TravelPlanExtraction,
) -> None:
    # Top-level section (critical, parallel)
    filming_node = evaluator.add_parallel(
        id="Filming_Location_Research",
        desc="Identify and verify the White Lotus Season 3 filming resort and its details",
        parent=parent_node,
        critical=True,
    )

    resort = data.resort or ResortInfo()

    # Resort Basic Details (critical, parallel)
    basic_node = evaluator.add_parallel(
        id="Resort_Basic_Details",
        desc="Identify the resort name, brand, and geographic location",
        parent=filming_node,
        critical=True,
    )

    # Brand and Name (leaf, critical)
    brand_leaf = evaluator.add_leaf(
        id="Brand_and_Name",
        desc="Identify the resort as Four Seasons Resort Koh Samui",
        parent=basic_node,
        critical=True,
    )
    brand_claim = "The resort used for filming on Koh Samui is Four Seasons Resort Koh Samui."
    await evaluator.verify(
        claim=brand_claim,
        node=brand_leaf,
        sources=resort.resort_urls,
        additional_instruction="Verify that reputable sources confirm White Lotus Season 3 primarily filmed at Four Seasons Resort Koh Samui.",
    )

    # Geographic Location (leaf, critical)
    geo_leaf = evaluator.add_leaf(
        id="Geographic_Location",
        desc="Verify the resort is located on Koh Samui island in Thailand",
        parent=basic_node,
        critical=True,
    )
    geo_claim = "Four Seasons Resort Koh Samui is located on Koh Samui island in Thailand."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=resort.resort_urls,
        additional_instruction="Use the resort's official site or other authoritative sources to confirm the location (Koh Samui, Thailand).",
    )

    # Accommodation Capacity (critical, parallel)
    capacity_node = evaluator.add_parallel(
        id="Accommodation_Capacity",
        desc="Verify the total number of accommodations at the resort",
        parent=filming_node,
        critical=True,
    )

    # Total Count (leaf, critical)
    total_leaf = evaluator.add_leaf(
        id="Total_Count",
        desc="State the total of 71 accommodations",
        parent=capacity_node,
        critical=True,
    )
    total_claim = "Four Seasons Resort Koh Samui has a total of 71 accommodations."
    await evaluator.verify(
        claim=total_claim,
        node=total_leaf,
        sources=resort.resort_urls,
        additional_instruction="Confirm the total number of accommodations is 71; allow minor phrasing variations like 'keys' or 'units'.",
    )

    # Breakdown (leaf, critical – adjusted to satisfy framework constraint)
    breakdown_leaf = evaluator.add_leaf(
        id="Breakdown",
        desc="Provide the composition (60 pool villas and 11 private residences)",
        parent=capacity_node,
        critical=True,
    )
    breakdown_claim = "The accommodations consist of 60 pool villas and 11 private residences."
    await evaluator.verify(
        claim=breakdown_claim,
        node=breakdown_leaf,
        sources=resort.resort_urls,
        additional_instruction="Check the resort's accommodation breakdown: 60 pool villas and 11 private residences.",
    )

    # Resort References (leaf, critical)
    refs_leaf = evaluator.add_leaf(
        id="Resort_References",
        desc="Provide URL references supporting the resort identification",
        parent=filming_node,
        critical=True,
    )
    refs_claim = "Reliable sources explicitly state that White Lotus Season 3 was primarily filmed at Four Seasons Resort Koh Samui."
    await evaluator.verify(
        claim=refs_claim,
        node=refs_leaf,
        sources=resort.resort_urls,
        additional_instruction="Links should include official announcements or credible media confirming the filming location.",
    )


async def verify_flight_routing(
    evaluator: Evaluator,
    parent_node,
    data: TravelPlanExtraction,
) -> None:
    # Flight Routing Research (critical, sequential)
    flight_node = evaluator.add_sequential(
        id="Flight_Routing_Research",
        desc="Determine the complete flight routing from the US to Koh Samui",
        parent=parent_node,
        critical=True,
    )

    turkish = data.turkish or TurkishRouteInfo()
    domestic = data.domestic or DomesticConnectionInfo()

    # International Flight via Turkish Airlines (critical, parallel)
    intl_node = evaluator.add_parallel(
        id="International_Flight",
        desc="Research Turkish Airlines service to Thailand",
        parent=flight_node,
        critical=True,
    )

    # Bangkok Route Details (critical, parallel)
    bkk_details_node = evaluator.add_parallel(
        id="Bangkok_Route_Details",
        desc="Verify Turkish Airlines route details to Bangkok",
        parent=intl_node,
        critical=True,
    )

    # Bangkok Service (leaf, critical)
    tk_bkk_service_leaf = evaluator.add_leaf(
        id="Bangkok_Service",
        desc="Confirm Turkish Airlines operates flights to Bangkok",
        parent=bkk_details_node,
        critical=True,
    )
    tk_bkk_service_claim = "Turkish Airlines operates flights to Bangkok, Thailand."
    await evaluator.verify(
        claim=tk_bkk_service_claim,
        node=tk_bkk_service_leaf,
        sources=turkish.turkish_urls,
        additional_instruction="Use official route maps, schedules, or reputable sources to confirm TK service to Bangkok.",
    )

    # Bangkok Airport Code (leaf, critical)
    tk_bkk_airport_leaf = evaluator.add_leaf(
        id="Bangkok_Airport_Code",
        desc="Identify Bangkok Suvarnabhumi Airport (BKK) as the served airport",
        parent=bkk_details_node,
        critical=True,
    )
    tk_bkk_airport_claim = "Turkish Airlines serves Bangkok Suvarnabhumi Airport (BKK)."
    await evaluator.verify(
        claim=tk_bkk_airport_claim,
        node=tk_bkk_airport_leaf,
        sources=turkish.turkish_urls,
        additional_instruction="Verify the served Bangkok airport is BKK (Suvarnabhumi), not DMK.",
    )

    # Koh Samui Direct (leaf, critical)
    tk_usm_direct_leaf = evaluator.add_leaf(
        id="Koh_Samui_Direct",
        desc="Verify Turkish Airlines does not fly directly to Koh Samui",
        parent=bkk_details_node,
        critical=True,
    )
    tk_usm_direct_claim = "Turkish Airlines does not operate direct flights to Koh Samui (USM)."
    await evaluator.verify(
        claim=tk_usm_direct_claim,
        node=tk_usm_direct_leaf,
        sources=turkish.turkish_urls,
        additional_instruction="Check TK's destination list or schedules to confirm USM is not served directly.",
    )

    # Turkish Airlines References (critical) – ensure URLs exist and are relevant by verification
    tk_refs_leaf = evaluator.add_leaf(
        id="Turkish_Airlines_References",
        desc="Provide URL references for Turkish Airlines routing",
        parent=intl_node,
        critical=True,
    )
    tk_refs_claim = "These sources accurately document Turkish Airlines routing to Bangkok and lack of direct service to Koh Samui (USM)."
    await evaluator.verify(
        claim=tk_refs_claim,
        node=tk_refs_leaf,
        sources=turkish.turkish_urls,
        additional_instruction="Evaluate whether the provided links substantiate the stated Turkish Airlines routing facts.",
    )

    # Domestic Connection (critical, parallel)
    dom_node = evaluator.add_parallel(
        id="Domestic_Connection",
        desc="Identify the domestic flight connection from Bangkok to Koh Samui",
        parent=flight_node,
        critical=True,
    )

    # Domestic Airline Details (critical, parallel)
    dom_details_node = evaluator.add_parallel(
        id="Domestic_Airline_Details",
        desc="Identify the airline and explain the monopoly situation",
        parent=dom_node,
        critical=True,
    )

    # Carrier Name (leaf, critical)
    carrier_leaf = evaluator.add_leaf(
        id="Carrier_Name",
        desc="Identify Bangkok Airways as the operating carrier",
        parent=dom_details_node,
        critical=True,
    )
    carrier_claim = "Bangkok Airways operates the domestic flights between Bangkok and Koh Samui Airport (USM)."
    await evaluator.verify(
        claim=carrier_claim,
        node=carrier_leaf,
        sources=domestic.domestic_urls,
        additional_instruction="Confirm that Bangkok Airways runs BKK/DMK ↔ USM routes.",
    )

    # Monopoly Reason (leaf, critical)
    monopoly_leaf = evaluator.add_leaf(
        id="Monopoly_Reason",
        desc="Explain monopoly exists because Bangkok Airways owns Koh Samui Airport",
        parent=dom_details_node,
        critical=True,
    )
    monopoly_claim = "Bangkok Airways owns Koh Samui Airport (USM), which gives it a monopoly on regular scheduled flights to USM."
    await evaluator.verify(
        claim=monopoly_claim,
        node=monopoly_leaf,
        sources=domestic.domestic_urls,
        additional_instruction="Verify ownership and resulting exclusivity regarding scheduled flights into USM.",
    )

    # Koh Samui Airport Code (leaf, critical)
    usm_code_leaf = evaluator.add_leaf(
        id="Koh_Samui_Airport_Code",
        desc="Provide Koh Samui Airport code (USM)",
        parent=dom_details_node,
        critical=True,
    )
    usm_code_claim = "The IATA code for Koh Samui Airport is USM."
    await evaluator.verify(
        claim=usm_code_claim,
        node=usm_code_leaf,
        sources=domestic.domestic_urls,
        additional_instruction="Confirm that 'USM' is the correct IATA code for Samui Airport.",
    )

    # Domestic Flight References (leaf, critical)
    dom_refs_leaf = evaluator.add_leaf(
        id="Domestic_Flight_References",
        desc="Provide URL references for domestic flight information",
        parent=dom_node,
        critical=True,
    )
    dom_refs_claim = "The provided sources substantiate the Bangkok → Koh Samui carrier details, monopoly reason, and USM airport code."
    await evaluator.verify(
        claim=dom_refs_claim,
        node=dom_refs_leaf,
        sources=domestic.domestic_urls,
        additional_instruction="Ensure the links are relevant and support the domestic flight facts stated.",
    )


async def verify_card_benefits(
    evaluator: Evaluator,
    parent_node,
    data: TravelPlanExtraction,
) -> None:
    # Credit Card Benefits (critical, parallel)
    cc_node = evaluator.add_parallel(
        id="Credit_Card_Benefits",
        desc="Research American Express Platinum Card benefits for the trip",
        parent=parent_node,
        critical=True,
    )

    lounge = data.lounge or LoungeAccessInfo()
    guest = data.guest_access or GuestAccessInfo()
    fee = data.annual_fee or AnnualFeeInfo()

    # Bangkok Lounge Access (critical, parallel)
    lounge_node = evaluator.add_parallel(
        id="Bangkok_Lounge_Access",
        desc="Verify lounge access at Bangkok Suvarnabhumi Airport",
        parent=cc_node,
        critical=True,
    )

    # BKK Access Details (critical, parallel)
    bkk_access_details_node = evaluator.add_parallel(
        id="BKK_Access_Details",
        desc="Confirm lounge access and identify the network",
        parent=lounge_node,
        critical=True,
    )

    # BKK Access (leaf, critical)
    bkk_access_leaf = evaluator.add_leaf(
        id="BKK_Access",
        desc="Confirm Amex Platinum provides lounge access at Bangkok Suvarnabhumi (BKK)",
        parent=bkk_access_details_node,
        critical=True,
    )
    bkk_access_claim = "The American Express Platinum Card provides lounge access at Bangkok Suvarnabhumi Airport (BKK)."
    await evaluator.verify(
        claim=bkk_access_claim,
        node=bkk_access_leaf,
        sources=lounge.lounge_urls,
        additional_instruction="Confirm via Amex or partner lounge network documentation that Platinum cardholders have lounge access at BKK.",
    )

    # Network Name (leaf, critical – adjusted to satisfy framework constraint)
    network_leaf = evaluator.add_leaf(
        id="Network_Name",
        desc="Identify the lounge network (Global Lounge Collection/Priority Pass)",
        parent=bkk_access_details_node,
        critical=True,
    )
    network_claim = "This access is provided via the Amex Global Lounge Collection, which includes partner networks like Priority Pass."
    await evaluator.verify(
        claim=network_claim,
        node=network_leaf,
        sources=lounge.lounge_urls,
        additional_instruction="Verify that Amex Platinum benefits include access through the Global Lounge Collection and/or Priority Pass at BKK.",
    )

    # Lounge References (leaf, critical)
    lounge_refs_leaf = evaluator.add_leaf(
        id="Lounge_References",
        desc="Provide URL references for Bangkok lounge access",
        parent=lounge_node,
        critical=True,
    )
    lounge_refs_claim = "The provided sources validate Amex Platinum lounge access at BKK and the applicable lounge network."
    await evaluator.verify(
        claim=lounge_refs_claim,
        node=lounge_refs_leaf,
        sources=lounge.lounge_urls,
        additional_instruction="Ensure sources are authoritative (Amex, Priority Pass, or airport lounge operator pages) and support the claims.",
    )

    # Guest Access Requirements (critical, parallel)
    guest_node = evaluator.add_parallel(
        id="Guest_Access_Requirements",
        desc="Determine Centurion Lounge guest access requirements",
        parent=cc_node,
        critical=True,
    )

    # Spending Threshold Details (critical, parallel)
    spend_details_node = evaluator.add_parallel(
        id="Spending_Threshold_Details",
        desc="Identify annual spending threshold and guest allowance",
        parent=guest_node,
        critical=True,
    )

    # Annual Spend (leaf, critical)
    spend_leaf = evaluator.add_leaf(
        id="Annual_Spend",
        desc="State the $75,000 annual spending requirement",
        parent=spend_details_node,
        critical=True,
    )
    spend_claim = "The Amex Platinum card requires $75,000 in annual spending to unlock complimentary guest access at Centurion Lounges."
    await evaluator.verify(
        claim=spend_claim,
        node=spend_leaf,
        sources=guest.guest_access_urls,
        additional_instruction="Confirm the exact spend threshold ($75,000) and that it unlocks complimentary guest access privileges.",
    )

    # Guest Count (leaf, critical)
    guest_count_leaf = evaluator.add_leaf(
        id="Guest_Count",
        desc="Specify up to 2 complimentary guests after meeting threshold",
        parent=spend_details_node,
        critical=True,
    )
    guest_count_claim = "After meeting the threshold, up to 2 guests can enter Centurion Lounges complimentary with the cardmember."
    await evaluator.verify(
        claim=guest_count_claim,
        node=guest_count_leaf,
        sources=guest.guest_access_urls,
        additional_instruction="Confirm the complimentary guest allowance is up to 2 guests once the spend requirement is met.",
    )

    # Guest Access References (leaf, critical)
    guest_refs_leaf = evaluator.add_leaf(
        id="Guest_Access_References",
        desc="Provide URL references for guest access requirements",
        parent=guest_node,
        critical=True,
    )
    guest_refs_claim = "The provided sources substantiate the Centurion Lounge guest access spend threshold and guest count."
    await evaluator.verify(
        claim=guest_refs_claim,
        node=guest_refs_leaf,
        sources=guest.guest_access_urls,
        additional_instruction="Ensure the links clearly state both the spend threshold and the complimentary guest policy.",
    )

    # Annual Fee (critical, parallel)
    annual_fee_node = evaluator.add_parallel(
        id="Annual_Fee",
        desc="Provide American Express Platinum Card annual fee",
        parent=cc_node,
        critical=True,
    )

    # Annual Fee Amount (leaf, critical)
    annual_fee_leaf = evaluator.add_leaf(
        id="Annual_Fee_Amount",
        desc="State the $895 annual fee as of 2026",
        parent=annual_fee_node,
        critical=True,
    )
    annual_fee_claim = "The annual fee for the American Express Platinum Card is $895."
    await evaluator.verify(
        claim=annual_fee_claim,
        node=annual_fee_leaf,
        sources=fee.fee_urls,
        additional_instruction="Verify current official fee amount as stated in authoritative sources (Amex official site or equivalent).",
    )

    # Fee References (leaf, critical)
    fee_refs_leaf = evaluator.add_leaf(
        id="Fee_References",
        desc="Provide URL references for annual fee information",
        parent=annual_fee_node,
        critical=True,
    )
    fee_refs_claim = "The provided sources confirm the current annual fee for the American Express Platinum Card."
    await evaluator.verify(
        claim=fee_refs_claim,
        node=fee_refs_leaf,
        sources=fee.fee_urls,
        additional_instruction="Ensure links are authoritative and explicitly state the annual fee amount.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the White Lotus Koh Samui travel planning task.
    """
    # Initialize evaluator and root
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

    # Create top-level critical node under root to represent the rubric root
    main_node = evaluator.add_parallel(
        id="Travel_Planning_Research",
        desc="Complete comprehensive travel planning research for a trip to the White Lotus Season 3 filming location",
        parent=root,
        critical=True,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_travel_plan(),
        template_class=TravelPlanExtraction,
        extraction_name="travel_plan_extraction",
    )

    # Add ground truth info to summary
    evaluator.add_ground_truth(GROUND_TRUTH_INFO, gt_type="expected_values")

    # Build and verify subtrees
    await verify_filming_location(evaluator, main_node, extracted)
    await verify_flight_routing(evaluator, main_node, extracted)
    await verify_card_benefits(evaluator, main_node, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()