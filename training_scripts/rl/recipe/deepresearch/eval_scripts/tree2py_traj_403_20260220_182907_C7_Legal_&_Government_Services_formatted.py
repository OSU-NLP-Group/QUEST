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
TASK_ID = "cascm_visitor_info"
TASK_DESCRIPTION = (
    "Provide comprehensive visitor service information for the California State Capitol Museum in Sacramento, California. "
    "Your response must include: (1) the complete street address, (2) operating hours for public visitors, (3) admission cost, "
    "(4) the schedule for public guided tours, (5) maximum capacity per tour, (6) group reservation requirements, "
    "(7) the tour office phone number, (8) locations of wheelchair-accessible entrances, (9) wheelchair checkout service details, "
    "(10) availability of assistive listening devices, (11) locations of wheelchair-accessible restrooms, "
    "(12) elevator access information, (13) security screening procedures, (14) maximum allowed bag dimensions, "
    "(15) the service animal policy, (16) parking information including nearby alternatives, and "
    "(17) the languages in which self-guided tour materials are available."
)

# Ground truth / expected values derived from rubric descriptions
EXPECTED_INFO: Dict[str, str] = {
    "address": "1315 10th Street, Sacramento, CA 95814",
    "operating_hours": "Weekdays from 9am to 5pm; closed on weekends and most holidays",
    "admission_cost": "Free and open to the public",
    "public_tour_schedule": "Public tours run on the hour from 10am to 4pm on weekdays",
    "tour_capacity_limit": "Maximum 35 individuals per guided tour; first-come, first-served",
    "group_reservation_requirement": "Groups of 10 or more must make advance reservations by calling Reserve California at 1-866-240-4655",
    "tour_office_phone": "916-324-0333",
    "accessible_entrances": "Ramps at North (L Street) and South (N Street) entrances; accessible sidewalks",
    "wheelchair_checkout_service": "Wheelchairs available for checkout at the first-floor rotunda information desk; driver's license held as collateral",
    "assistive_listening_devices": "Assistive listening devices available for guided tours at the first-floor rotunda information desk",
    "accessible_restrooms": "Wheelchair-accessible restrooms on the first floor on either side of the rotunda",
    "elevator_access": "Elevators available on either side of the rotunda",
    "security_screening": "All visitors must pass through metal detectors; bags subject to X-ray and visual examination",
    "maximum_bag_dimensions": "Maximum bag size 14 inches wide × 13 inches high × 4 inches deep",
    "service_animal_policy": "Only trained service animals are allowed inside the Capitol building",
    "parking_information": "No public parking at the facility; use nearby metered parking or Capitol Garage at 10th and L Streets",
    "self_guided_tour_languages": "English, Chinese, Spanish, Dutch, German, and French",
}

# Mapping from rubric node names to internal keys (to keep leaf IDs aligned with rubric)
RUBRIC_NODE_ID_MAP: Dict[str, str] = {
    "Complete_Street_Address": "address",
    "Operating_Hours": "operating_hours",
    "Admission_Cost": "admission_cost",
    "Public_Tour_Schedule": "public_tour_schedule",
    "Tour_Capacity_Limit": "tour_capacity_limit",
    "Group_Reservation_Requirement": "group_reservation_requirement",
    "Tour_Office_Contact": "tour_office_phone",
    "Wheelchair_Accessible_Entrances": "accessible_entrances",
    "Wheelchair_Checkout_Service": "wheelchair_checkout_service",
    "Assistive_Listening_Devices": "assistive_listening_devices",
    "Accessible_Restrooms": "accessible_restrooms",
    "Elevator_Access": "elevator_access",
    "Security_Screening": "security_screening",
    "Maximum_Bag_Dimensions": "maximum_bag_dimensions",
    "Service_Animal_Policy": "service_animal_policy",
    "Parking_Information": "parking_information",
    "Self_Guided_Tour_Languages": "self_guided_tour_languages",
}

RUBRIC_DESCRIPTIONS: Dict[str, str] = {
    "Complete_Street_Address": "The complete street address of the California State Capitol Museum is 1315 10th Street, Sacramento, CA 95814",
    "Operating_Hours": "The facility operates on weekdays from 9am to 5pm and is closed on weekends and most holidays",
    "Admission_Cost": "Admission to the facility is free of charge and open to the public",
    "Public_Tour_Schedule": "Public tours run on the hour from 10am to 4pm on weekdays",
    "Tour_Capacity_Limit": "Each guided tour has a maximum capacity of 35 individuals on a first-come, first-served basis",
    "Group_Reservation_Requirement": "Groups of 10 or more people must make advance reservations by calling Reserve California at 1-866-240-4655",
    "Tour_Office_Contact": "The Capitol Tour Office phone number (916-324-0333) is provided for visitor inquiries",
    "Wheelchair_Accessible_Entrances": "Ramps at North (L Street) and South (N Street) entrances provide wheelchair access with accessible sidewalks",
    "Wheelchair_Checkout_Service": "Wheelchairs are available for checkout at the first-floor rotunda information desk with a driver's license held as collateral",
    "Assistive_Listening_Devices": "Assistive listening devices are available for guided tours at the first-floor rotunda information desk",
    "Accessible_Restrooms": "Wheelchair-accessible restrooms are located on the first floor on either side of the rotunda",
    "Elevator_Access": "Elevators are available on either side of the rotunda for multi-floor access",
    "Security_Screening": "All visitors must pass through metal detectors and have bags subject to X-ray and visual examination",
    "Maximum_Bag_Dimensions": "Bags must not exceed 14 inches wide × 13 inches high × 4 inches deep",
    "Service_Animal_Policy": "Only trained service animals are allowed inside the Capitol building",
    "Parking_Information": "The facility does not have public parking; visitors must use nearby metered parking or the Capitol Garage at 10th and L Streets",
    "Self_Guided_Tour_Languages": "Self-guided tour brochures are available in six languages: English, Chinese, Spanish, Dutch, German, and French",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ItemField(BaseModel):
    text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MuseumVisitorInfoExtraction(BaseModel):
    address: Optional[ItemField] = None
    operating_hours: Optional[ItemField] = None
    admission_cost: Optional[ItemField] = None
    public_tour_schedule: Optional[ItemField] = None
    tour_capacity_limit: Optional[ItemField] = None
    group_reservation_requirement: Optional[ItemField] = None
    tour_office_phone: Optional[ItemField] = None
    accessible_entrances: Optional[ItemField] = None
    wheelchair_checkout_service: Optional[ItemField] = None
    assistive_listening_devices: Optional[ItemField] = None
    accessible_restrooms: Optional[ItemField] = None
    elevator_access: Optional[ItemField] = None
    security_screening: Optional[ItemField] = None
    maximum_bag_dimensions: Optional[ItemField] = None
    service_animal_policy: Optional[ItemField] = None
    parking_information: Optional[ItemField] = None
    self_guided_tour_languages: Optional[ItemField] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_museum_info() -> str:
    return (
        "Extract the visitor service information for the California State Capitol Museum exactly as presented in the answer. "
        "For each of the following items, return an object with 'text' (the content verbatim or closely paraphrased as written in the answer) "
        "and 'sources' (all URLs explicitly cited in the answer that support this specific item). "
        "If an item is not mentioned, set 'text' to null and 'sources' to an empty array. "
        "If the answer mentions a source but not as a URL, do not include it.\n\n"
        "Items to extract:\n"
        "- address: Full street address\n"
        "- operating_hours: Hours for public visitors (incl. weekend/holiday notes)\n"
        "- admission_cost: Admission policy/cost\n"
        "- public_tour_schedule: Schedule for public guided tours\n"
        "- tour_capacity_limit: Maximum capacity per tour and any first-come policy\n"
        "- group_reservation_requirement: Requirements and phone number for groups\n"
        "- tour_office_phone: Capitol Tour Office phone number\n"
        "- accessible_entrances: Wheelchair-accessible entrance locations and features\n"
        "- wheelchair_checkout_service: Wheelchair checkout location and collateral policy\n"
        "- assistive_listening_devices: Availability and location for ALDs\n"
        "- accessible_restrooms: Locations of wheelchair-accessible restrooms\n"
        "- elevator_access: Elevator access locations\n"
        "- security_screening: Screening procedures (metal detectors, bag checks)\n"
        "- maximum_bag_dimensions: Maximum allowed bag size\n"
        "- service_animal_policy: Service animal policy\n"
        "- parking_information: Parking availability and nearby alternatives\n"
        "- self_guided_tour_languages: Languages of self-guided tour materials (comma-separated if multiple)\n\n"
        "Return the JSON in the following schema with exactly these keys."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_field(extraction: MuseumVisitorInfoExtraction, key: str) -> ItemField:
    value = getattr(extraction, key, None)
    return value or ItemField(text=None, sources=[])


def make_supported_claim(item_key: str, text: Optional[str]) -> str:
    museum_name = "California State Capitol Museum"
    t = text or ""
    if item_key == "address":
        return f"The complete street address of the {museum_name} is '{t}'."
    if item_key == "operating_hours":
        return f"The operating hours for public visitors at the {museum_name} are: '{t}'."
    if item_key == "admission_cost":
        return f"The admission policy for the {museum_name} is: '{t}'."
    if item_key == "public_tour_schedule":
        return f"The public guided tour schedule at the {museum_name} is: '{t}'."
    if item_key == "tour_capacity_limit":
        return f"The guided tour capacity policy at the {museum_name} is: '{t}'."
    if item_key == "group_reservation_requirement":
        return f"The group reservation requirement at the {museum_name} is: '{t}'."
    if item_key == "tour_office_phone":
        return f"The Capitol Tour Office phone number for the {museum_name} is '{t}'."
    if item_key == "accessible_entrances":
        return f"Wheelchair-accessible entrances for the {museum_name} are described as: '{t}'."
    if item_key == "wheelchair_checkout_service":
        return f"Wheelchair checkout service details at the {museum_name} are: '{t}'."
    if item_key == "assistive_listening_devices":
        return f"Assistive listening devices availability at the {museum_name} is: '{t}'."
    if item_key == "accessible_restrooms":
        return f"Wheelchair-accessible restroom locations at the {museum_name} are: '{t}'."
    if item_key == "elevator_access":
        return f"Elevator access information at the {museum_name} is: '{t}'."
    if item_key == "security_screening":
        return f"Security screening procedures at the {museum_name} are: '{t}'."
    if item_key == "maximum_bag_dimensions":
        return f"The maximum allowed bag dimensions at the {museum_name} are: '{t}'."
    if item_key == "service_animal_policy":
        return f"The service animal policy at the {museum_name} is: '{t}'."
    if item_key == "parking_information":
        return f"Parking information and nearby alternatives for the {museum_name} are: '{t}'."
    if item_key == "self_guided_tour_languages":
        return f"Self-guided tour materials at the {museum_name} are available in: '{t}'."
    return f"The information for '{item_key}' is: '{t}'."


def match_expected_additional_instruction(item_key: str) -> str:
    common = "Allow minor formatting variations (e.g., punctuation, capitalization, am/pm formatting, use of symbols like × vs x, or inclusion/exclusion of the word 'approximately'). Focus on whether they mean the same factual requirement."
    if item_key in {"maximum_bag_dimensions"}:
        return common + " For bag sizes, consider 14 inches wide by 13 inches high by 4 inches deep as equivalent to similar formatting (e.g., 14\" x 13\" x 4\")."
    if item_key in {"operating_hours", "public_tour_schedule"}:
        return common + " Accept 'on the hour' phrased as 'every hour' and weekday phrasing variations (e.g., Monday–Friday)."
    if item_key in {"tour_office_phone", "group_reservation_requirement"}:
        return common + " Phone number formatting may vary with spaces or dashes; treat equivalent formats as matching."
    if item_key in {"accessible_entrances"}:
        return common + " Treat 'North (L Street)' and 'South (N Street)' as equivalent even if order varies."
    return common


async def build_item_verification(
    evaluator: Evaluator,
    parent_node,
    rubric_node_name: str,
    item_key: str,
    item_desc: str,
    extracted: ItemField,
    expected_text: str,
) -> None:
    """
    Build a sequential critical sub-tree for one rubric item:
      1) sources existence check
      2) content supported by cited sources
      3) matches expected ground truth statement
    """
    # Container node (critical, sequential to gate later checks)
    container = evaluator.add_sequential(
        id=f"{item_key}_main",
        desc=f"Verification for {rubric_node_name}: {item_desc}",
        parent=parent_node,
        critical=True
    )

    # 1) Existence check: answer provided text and at least one source URL
    has_text = (extracted.text is not None and str(extracted.text).strip() != "")
    has_sources = bool(extracted.sources)
    evaluator.add_custom_node(
        result=has_text and has_sources,
        id=f"{item_key}_sources_provided",
        desc=f"{rubric_node_name} sources are provided and information is present in the answer",
        parent=container,
        critical=True
    )

    # 2) Supported by sources
    supported_leaf = evaluator.add_leaf(
        id=f"{item_key}_supported_by_sources",
        desc=f"{rubric_node_name} information is supported by cited sources",
        parent=container,
        critical=True
    )
    supported_claim = make_supported_claim(item_key, extracted.text)
    await evaluator.verify(
        claim=supported_claim,
        node=supported_leaf,
        sources=extracted.sources,
        additional_instruction="Verify that the provided webpages explicitly support this claim about the California State Capitol Museum. "
                               "Allow minor phrasing differences, but the substance must match the claim."
    )

    # 3) Matches expected ground truth (rubric statement)
    match_leaf = evaluator.add_leaf(
        id=rubric_node_name,  # Keep rubric ID for the key verification leaf
        desc=item_desc,
        parent=container,
        critical=True
    )
    match_claim = (
        f"The answer's value for {rubric_node_name} ('{extracted.text or ''}') matches the expected statement: '{expected_text}'."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_leaf,
        additional_instruction=match_expected_additional_instruction(item_key)
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the California State Capitol Museum visitor information task.
    """
    # Initialize evaluator with parallel root (matches rubric root)
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

    # Extract structured visitor info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_museum_info(),
        template_class=MuseumVisitorInfoExtraction,
        extraction_name="museum_visitor_info"
    )

    # Record ground truth information for transparency
    evaluator.add_ground_truth({
        "expected_info": EXPECTED_INFO,
        "rubric_descriptions": RUBRIC_DESCRIPTIONS
    }, gt_type="expected_info")

    # Build verification subtrees for all 17 rubric items
    items_order = [
        "Complete_Street_Address",
        "Operating_Hours",
        "Admission_Cost",
        "Public_Tour_Schedule",
        "Tour_Capacity_Limit",
        "Group_Reservation_Requirement",
        "Tour_Office_Contact",
        "Wheelchair_Accessible_Entrances",
        "Wheelchair_Checkout_Service",
        "Assistive_Listening_Devices",
        "Accessible_Restrooms",
        "Elevator_Access",
        "Security_Screening",
        "Maximum_Bag_Dimensions",
        "Service_Animal_Policy",
        "Parking_Information",
        "Self_Guided_Tour_Languages",
    ]

    for rubric_node in items_order:
        item_key = RUBRIC_NODE_ID_MAP[rubric_node]
        item_desc = RUBRIC_DESCRIPTIONS[rubric_node]
        expected_text = EXPECTED_INFO[item_key]
        extracted_field = get_field(extraction, item_key)

        await build_item_verification(
            evaluator=evaluator,
            parent_node=root,
            rubric_node_name=rubric_node,
            item_key=item_key,
            item_desc=item_desc,
            extracted=extracted_field,
            expected_text=expected_text
        )

    # Return structured summary
    return evaluator.get_summary()