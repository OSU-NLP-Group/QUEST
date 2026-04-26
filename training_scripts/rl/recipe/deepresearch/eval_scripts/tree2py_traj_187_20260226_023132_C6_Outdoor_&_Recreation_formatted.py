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
TASK_ID = "fl_state_park_accessible_beach_camping"
TASK_DESCRIPTION = (
    "Identify a state park in Florida that offers camping facilities with wheelchair-accessible campsites, "
    "provides beach access with wheelchair accessibility features or equipment, accepts advance reservations "
    "through Florida's state park reservation system, has a visitor center or park office providing visitor services, "
    "and offers at least two additional recreational activities such as fishing, hiking trails, or boating."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkExtraction(BaseModel):
    # Core identity
    park_name: Optional[str] = None
    state_or_region: Optional[str] = None

    # Camping details
    camping_available_text: Optional[str] = None
    accessible_campsites_features: List[str] = Field(default_factory=list)

    # Reservation details
    reservation_platform_url: Optional[str] = None
    reservation_policy_url: Optional[str] = None
    reservation_policy_text: Optional[str] = None

    # Beach details
    beach_available_text: Optional[str] = None
    beach_access_features: List[str] = Field(default_factory=list)

    # Visitor services
    visitor_services_text: Optional[str] = None

    # Activities
    activity_1: Optional[str] = None
    activity_2: Optional[str] = None
    extra_activities: List[str] = Field(default_factory=list)

    # URL sources grouped by topic
    location_urls: List[str] = Field(default_factory=list)
    camping_urls: List[str] = Field(default_factory=list)
    reservation_urls: List[str] = Field(default_factory=list)
    beach_urls: List[str] = Field(default_factory=list)
    visitor_urls: List[str] = Field(default_factory=list)
    activity_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_info() -> str:
    return """
    From the answer text, extract information for exactly ONE Florida state park that matches the task.
    If the answer mentions multiple parks, choose the first one that is a Florida state park and appears to meet the requirements.

    Return a JSON object with the following fields (set any missing field to null or empty array as appropriate):
    - park_name: The exact park name identified in the answer (e.g., "Grayton Beach State Park")
    - state_or_region: The state or region mentioned (should be "Florida" if provided)
    - camping_available_text: Text snippet indicating that camping is available at the park
    - accessible_campsites_features: Array of wheelchair-accessible campsite features mentioned (e.g., ["paved sites", "accessible picnic tables", "concrete pads", "accessible grills"])
    - reservation_platform_url: The URL to the Florida state park reservation platform for this park (if provided)
    - reservation_policy_url: The URL that describes the advance booking policy for Florida state parks (if provided)
    - reservation_policy_text: Text snippet explaining the policy (e.g., "Florida residents can book 11 months ahead, non-residents 10 months")
    - beach_available_text: Text snippet indicating that beach access is available at the park
    - beach_access_features: Array of wheelchair beach accessibility features or equipment (e.g., ["beach wheelchairs", "accessible boardwalks", "beach mats", "paved pathways"])
    - visitor_services_text: Text snippet indicating the presence of a visitor center, ranger station, or park office
    - activity_1: Name of the first additional recreational activity (e.g., "fishing", "hiking trails", "boating", "kayaking")
    - activity_2: Name of the second distinct activity (different from activity_1)
    - extra_activities: Array of any additional recreational activities beyond the required two

    Also extract source URLs grouped by topic (arrays of URLs):
    - location_urls: URLs confirming the park is in Florida and is a Florida state park
    - camping_urls: URLs confirming camping facilities and accessibility features at the park
    - reservation_urls: URLs confirming reservation system details or linking to the reservation portal
    - beach_urls: URLs confirming beach access and wheelchair accessibility features
    - visitor_urls: URLs confirming visitor center/ranger station/park office services
    - activity_urls: URLs confirming recreational activities (first, second, or others)

    RULES:
    - Extract only URLs explicitly shown or linked in the answer (plain URLs or markdown links). Do not invent URLs.
    - If a category has no explicit URL in the answer, return an empty array for that category.
    - Do NOT rely on your own knowledge; only extract what is present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for urls in url_lists:
        for u in urls:
            if not u:
                continue
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                result.append(uu)
    return result


def features_str(features: List[str]) -> str:
    return ", ".join([f.strip() for f in features if f and f.strip()]) if features else "accessible features"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_location(evaluator: Evaluator, parent_node, ex: ParkExtraction) -> None:
    loc_node = evaluator.add_sequential(
        id="Location_Verification",
        desc="The park is located in Florida",
        parent=parent_node,
        critical=True
    )

    # Existence of location sources (critical prerequisite)
    loc_sources_provided = evaluator.add_custom_node(
        result=bool(ex.location_urls),
        id="Location_Source_Provided",
        desc="At least one URL is provided for location verification",
        parent=loc_node,
        critical=True
    )

    # Leaf: Confirm Florida state park
    florida_leaf = evaluator.add_leaf(
        id="Florida_State",
        desc="The park is confirmed to be a Florida state park",
        parent=loc_node,
        critical=True
    )
    claim_florida = f"The park named '{ex.park_name}' is a Florida state park located in Florida."
    await evaluator.verify(
        claim=claim_florida,
        node=florida_leaf,
        sources=ex.location_urls,
        additional_instruction="Use the provided webpage(s) to confirm the park is in Florida and part of the Florida State Parks system."
    )

    # Leaf: Location reference confirms Florida
    loc_ref_leaf = evaluator.add_leaf(
        id="Location_Reference",
        desc="URL reference confirming park location in Florida",
        parent=loc_node,
        critical=True
    )
    claim_loc_ref = f"This page confirms that '{ex.park_name}' is located in Florida."
    await evaluator.verify(
        claim=claim_loc_ref,
        node=loc_ref_leaf,
        sources=ex.location_urls,
        additional_instruction="Judge 'supported' only if the page clearly indicates the park is in Florida."
    )


async def verify_camping(evaluator: Evaluator, parent_node, ex: ParkExtraction) -> None:
    camp_node = evaluator.add_parallel(
        id="Camping_Facilities",
        desc="The park offers camping with accessibility features and reservation capabilities",
        parent=parent_node,
        critical=True
    )

    # Existence of camping sources
    camp_sources_node = evaluator.add_custom_node(
        result=bool(ex.camping_urls),
        id="Camping_Source_Provided",
        desc="At least one URL confirms camping facilities or accessibility features",
        parent=camp_node,
        critical=True
    )

    # Camping available
    camp_avail_leaf = evaluator.add_leaf(
        id="Camping_Available",
        desc="The park offers camping facilities (RV sites and/or tent camping)",
        parent=camp_node,
        critical=True
    )
    claim_camp = f"'{ex.park_name}' offers camping facilities (RV and/or tent camping)."
    await evaluator.verify(
        claim=claim_camp,
        node=camp_avail_leaf,
        sources=ex.camping_urls,
        additional_instruction="Verify that the page explicitly mentions camping at the park."
    )

    # Accessible campsites
    accessible_leaf = evaluator.add_leaf(
        id="Accessible_Campsites",
        desc=("The park has wheelchair-accessible campsites with features such as paved sites, concrete pads, "
              "accessible picnic tables, or accessible grills"),
        parent=camp_node,
        critical=True
    )
    claim_accessible = (
        f"'{ex.park_name}' has wheelchair-accessible campsites with features like {features_str(ex.accessible_campsites_features)}."
    )
    await evaluator.verify(
        claim=claim_accessible,
        node=accessible_leaf,
        sources=ex.camping_urls,
        additional_instruction=("Confirm that the page documents wheelchair-accessible campsite features (e.g., paved sites, "
                                "concrete pads, accessible picnic tables, accessible grills).")
    )

    # Reservation system (critical sub-sequential)
    res_node = evaluator.add_sequential(
        id="Reservation_System",
        desc="The park accepts reservations through Florida's state park reservation system",
        parent=camp_node,
        critical=True
    )

    res_sources_node = evaluator.add_custom_node(
        result=bool(ex.reservation_platform_url) or bool(ex.reservation_urls),
        id="Reservation_Source_Provided",
        desc="At least one URL confirms reservation system details",
        parent=res_node,
        critical=True
    )

    uses_state_leaf = evaluator.add_leaf(
        id="Uses_State_System",
        desc="Reservations are made through reserve.floridastateparks.org or the official Florida state parks reservation platform",
        parent=res_node,
        critical=True
    )
    res_sources = unique_urls(
        [ex.reservation_platform_url] if ex.reservation_platform_url else [],
        ex.reservation_urls
    )
    claim_uses_state = (
        f"Reservations for '{ex.park_name}' are made through Florida's official state park reservation platform "
        f"(e.g., reserve.floridastateparks.org)."
    )
    await evaluator.verify(
        claim=claim_uses_state,
        node=uses_state_leaf,
        sources=res_sources,
        additional_instruction="The page should show a reservation link or portal associated with Florida State Parks reservations."
    )

    res_ref_leaf = evaluator.add_leaf(
        id="Reservation_Reference",
        desc="URL reference confirming reservation system details",
        parent=res_node,
        critical=True
    )
    claim_res_ref = "This page confirms the reservation system details for the park."
    await evaluator.verify(
        claim=claim_res_ref,
        node=res_ref_leaf,
        sources=res_sources,
        additional_instruction="Confirm that this page provides reservation details (portal link, booking process, etc.)."
    )

    # Camping reference page confirms both camping and accessibility
    camp_ref_leaf = evaluator.add_leaf(
        id="Camping_Reference",
        desc="URL reference confirming camping facilities and accessibility features",
        parent=camp_node,
        critical=True
    )
    claim_camp_ref = (
        f"This page confirms camping facilities at '{ex.park_name}' and mentions accessibility features for campsites."
    )
    await evaluator.verify(
        claim=claim_camp_ref,
        node=camp_ref_leaf,
        sources=ex.camping_urls,
        additional_instruction="Look for explicit mentions of camping and accessible campsite features."
    )


async def verify_beach(evaluator: Evaluator, parent_node, ex: ParkExtraction) -> None:
    beach_node = evaluator.add_parallel(
        id="Beach_Access_Features",
        desc="The park provides beach access with wheelchair accessibility features",
        parent=parent_node,
        critical=True
    )

    beach_sources_node = evaluator.add_custom_node(
        result=bool(ex.beach_urls),
        id="Beach_Source_Provided",
        desc="At least one URL confirms beach access and/or wheelchair beach accessibility features",
        parent=beach_node,
        critical=True
    )

    beach_avail_leaf = evaluator.add_leaf(
        id="Beach_Available",
        desc="The park has beach access for visitors",
        parent=beach_node,
        critical=True
    )
    claim_beach_avail = f"'{ex.park_name}' has beach access for visitors."
    await evaluator.verify(
        claim=claim_beach_avail,
        node=beach_avail_leaf,
        sources=ex.beach_urls,
        additional_instruction="Confirm that the page mentions beach access at the park."
    )

    wchair_node = evaluator.add_sequential(
        id="Wheelchair_Beach_Access",
        desc="The park provides wheelchair accessibility features or equipment for beach access",
        parent=beach_node,
        critical=True
    )

    wchair_feat_leaf = evaluator.add_leaf(
        id="Accessibility_Features_Present",
        desc=("The park offers at least one wheelchair accessibility feature or equipment for beach access, "
              "such as beach wheelchairs, accessible boardwalks, beach mats, or paved pathways to the beach"),
        parent=wchair_node,
        critical=True
    )
    claim_wchair_feat = (
        f"'{ex.park_name}' provides wheelchair beach accessibility features or equipment such as "
        f"{features_str(ex.beach_access_features)}."
    )
    await evaluator.verify(
        claim=claim_wchair_feat,
        node=wchair_feat_leaf,
        sources=ex.beach_urls,
        additional_instruction="The page should explicitly mention wheelchair-accessible beach features (e.g., beach wheelchairs, mats, boardwalks)."
    )

    wchair_ref_leaf = evaluator.add_leaf(
        id="Beach_Accessibility_Reference",
        desc="URL reference confirming wheelchair beach accessibility features or equipment",
        parent=wchair_node,
        critical=True
    )
    claim_wchair_ref = "This page documents wheelchair beach accessibility features or equipment at the park."
    await evaluator.verify(
        claim=claim_wchair_ref,
        node=wchair_ref_leaf,
        sources=ex.beach_urls,
        additional_instruction="Confirm that wheelchair beach accessibility is explicitly described."
    )

    beach_ref_leaf = evaluator.add_leaf(
        id="Beach_Reference",
        desc="URL reference confirming beach access availability",
        parent=beach_node,
        critical=True
    )
    claim_beach_ref = "This page confirms beach access is available at the park."
    await evaluator.verify(
        claim=claim_beach_ref,
        node=beach_ref_leaf,
        sources=ex.beach_urls,
        additional_instruction="Page should state beach access is available."
    )


async def verify_visitor(evaluator: Evaluator, parent_node, ex: ParkExtraction) -> None:
    visitor_node = evaluator.add_sequential(
        id="Visitor_Services",
        desc="The park has a visitor center, ranger station, or park office providing visitor services",
        parent=parent_node,
        critical=True
    )

    visitor_sources_node = evaluator.add_custom_node(
        result=bool(ex.visitor_urls),
        id="Visitor_Source_Provided",
        desc="At least one URL confirms visitor services availability",
        parent=visitor_node,
        critical=True
    )

    service_leaf = evaluator.add_leaf(
        id="Service_Facility",
        desc=("The park has a visitor center, ranger station, park office, or equivalent facility staffed to provide "
              "visitor information and services"),
        parent=visitor_node,
        critical=True
    )
    claim_service = (
        f"'{ex.park_name}' has a visitor center, ranger station, or park office that provides visitor services."
    )
    await evaluator.verify(
        claim=claim_service,
        node=service_leaf,
        sources=ex.visitor_urls,
        additional_instruction="Confirm the presence of a staffed visitor services facility."
    )

    visitor_ref_leaf = evaluator.add_leaf(
        id="Visitor_Services_Reference",
        desc="URL reference confirming visitor services availability",
        parent=visitor_node,
        critical=True
    )
    claim_visitor_ref = "This page confirms visitor services are available at the park."
    await evaluator.verify(
        claim=claim_visitor_ref,
        node=visitor_ref_leaf,
        sources=ex.visitor_urls,
        additional_instruction="Page should explicitly mention visitor services, visitor center, ranger station, or park office."
    )


async def verify_recreation(evaluator: Evaluator, parent_node, ex: ParkExtraction) -> None:
    rec_node = evaluator.add_parallel(
        id="Recreational_Amenities",
        desc="The park offers at least two additional recreational activities beyond camping and beach access",
        parent=parent_node,
        critical=True
    )

    # First activity
    first_node = evaluator.add_sequential(
        id="First_Recreational_Activity",
        desc="The park offers a first recreational activity (fishing, hiking trails, boating, kayaking, or similar outdoor activity)",
        parent=rec_node,
        critical=True
    )

    first_src_node = evaluator.add_custom_node(
        result=bool(ex.activity_urls),
        id="First_Activity_Source_Provided",
        desc="At least one URL reference for the first recreational activity",
        parent=first_node,
        critical=True
    )

    first_avail_leaf = evaluator.add_leaf(
        id="Activity_Available",
        desc="A specific recreational activity is available at the park",
        parent=first_node,
        critical=True
    )
    claim_first_avail = f"'{ex.park_name}' offers the recreational activity: {ex.activity_1 or 'an activity such as fishing/hiking/boating'}."
    await evaluator.verify(
        claim=claim_first_avail,
        node=first_avail_leaf,
        sources=ex.activity_urls,
        additional_instruction="Confirm the specific activity is offered at the park."
    )

    first_ref_leaf = evaluator.add_leaf(
        id="First_Activity_Reference",
        desc="URL reference confirming the first recreational activity",
        parent=first_node,
        critical=True
    )
    claim_first_ref = "This page confirms the first recreational activity offered at the park."
    await evaluator.verify(
        claim=claim_first_ref,
        node=first_ref_leaf,
        sources=ex.activity_urls,
        additional_instruction="Look for explicit mention of the activity and its availability at the park."
    )

    # Second activity
    second_node = evaluator.add_sequential(
        id="Second_Recreational_Activity",
        desc="The park offers a second distinct recreational activity different from the first",
        parent=rec_node,
        critical=True
    )

    second_src_node = evaluator.add_custom_node(
        result=bool(ex.activity_urls),
        id="Second_Activity_Source_Provided",
        desc="At least one URL reference for the second recreational activity",
        parent=second_node,
        critical=True
    )

    second_avail_leaf = evaluator.add_leaf(
        id="Second_Activity_Available",
        desc="A second distinct recreational activity is available at the park",
        parent=second_node,
        critical=True
    )
    claim_second_avail = f"'{ex.park_name}' offers a second distinct activity: {ex.activity_2 or 'another activity such as fishing/hiking/boating'}."
    await evaluator.verify(
        claim=claim_second_avail,
        node=second_avail_leaf,
        sources=ex.activity_urls,
        additional_instruction="Confirm a second activity that is distinct from the first is offered at the park."
    )

    second_ref_leaf = evaluator.add_leaf(
        id="Second_Activity_Reference",
        desc="URL reference confirming the second recreational activity",
        parent=second_node,
        critical=True
    )
    claim_second_ref = "This page confirms the second recreational activity offered at the park."
    await evaluator.verify(
        claim=claim_second_ref,
        node=second_ref_leaf,
        sources=ex.activity_urls,
        additional_instruction="Look for explicit mention of the second activity."
    )


async def verify_optional(evaluator: Evaluator, parent_node, ex: ParkExtraction) -> None:
    # Advance booking policy (optional check)
    policy_node = evaluator.add_sequential(
        id="Advance_Booking_Policy",
        desc="Advance booking policy verification (optional, non-critical)",
        parent=parent_node,
        critical=False
    )

    policy_src_node = evaluator.add_custom_node(
        result=bool(ex.reservation_policy_url) or bool(ex.reservation_urls),
        id="Policy_Source_Provided",
        desc="At least one URL reference for advance booking policy",
        parent=policy_node,
        critical=True  # Critical here to ensure we skip verification when there's no source
    )

    adv_booking_leaf = evaluator.add_leaf(
        id="Advance_Booking",
        desc="The reservation system allows advance booking with Florida residents booking 11 months ahead and non-residents 10 months ahead",
        parent=policy_node,
        critical=False
    )
    sources = unique_urls(
        [ex.reservation_policy_url] if ex.reservation_policy_url else [],
        ex.reservation_urls
    )
    claim_adv = (
        "Florida State Parks reservations allow Florida residents to book 11 months in advance and non-residents 10 months in advance."
    )
    await evaluator.verify(
        claim=claim_adv,
        node=adv_booking_leaf,
        sources=sources,
        additional_instruction="Verify the advance booking window policy on the official reservation policy page."
    )

    # Additional activities (optional, non-critical)
    add_node = evaluator.add_sequential(
        id="Additional_Activities",
        desc="The park offers additional recreational activities beyond the required two (optional)",
        parent=parent_node,
        critical=False
    )

    add_src_node = evaluator.add_custom_node(
        result=bool(ex.activity_urls),
        id="Additional_Activities_Source_Provided",
        desc="At least one URL reference for additional activities",
        parent=add_node,
        critical=True
    )

    add_acts_leaf = evaluator.add_leaf(
        id="Third_Or_More_Activities",
        desc="Additional recreational activities are documented at the park",
        parent=add_node,
        critical=False
    )
    claim_additional = (
        f"'{ex.park_name}' offers additional recreational activities beyond the two required (e.g., {features_str(ex.extra_activities)})."
    )
    await evaluator.verify(
        claim=claim_additional,
        node=add_acts_leaf,
        sources=ex.activity_urls,
        additional_instruction="Confirm at least one more activity beyond the two required."
    )

    add_ref_leaf = evaluator.add_leaf(
        id="Additional_Activities_Reference",
        desc="URL reference for additional activities",
        parent=add_node,
        critical=True
    )
    claim_add_ref = "This page confirms additional recreational activities at the park."
    await evaluator.verify(
        claim=claim_add_ref,
        node=add_ref_leaf,
        sources=ex.activity_urls,
        additional_instruction="Look for explicit mention of additional activities."
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
    Evaluate an answer for the Florida accessible beach camping state park task.
    """
    # Initialize evaluator
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

    # Extract structured park info from the answer
    ex: ParkExtraction = await evaluator.extract(
        prompt=prompt_extract_park_info(),
        template_class=ParkExtraction,
        extraction_name="park_extraction"
    )

    # Add high-level ground truth requirement info for context (not strict GT data)
    evaluator.add_ground_truth({
        "requirements": [
            "Florida state park",
            "Camping facilities available",
            "Wheelchair-accessible campsites",
            "Beach access with wheelchair accessibility features/equipment",
            "Reservations via Florida state park reservation system",
            "Visitor services (visitor center/ranger station/park office)",
            "At least two additional recreational activities"
        ]
    }, gt_type="task_requirements")

    # Build main critical aggregation node
    main_node = evaluator.add_parallel(
        id="State_Park_Identification",
        desc="The identified state park meets all specified criteria for accessible beach camping with recreational amenities",
        parent=root,
        critical=True
    )

    # Park name must be provided (critical gate)
    evaluator.add_custom_node(
        result=bool(ex.park_name) and bool(ex.park_name.strip()),
        id="Park_Name_Provided",
        desc="A specific Florida state park name is provided in the answer",
        parent=main_node,
        critical=True
    )

    # Verify each essential criterion
    await verify_location(evaluator, main_node, ex)
    await verify_camping(evaluator, main_node, ex)
    await verify_beach(evaluator, main_node, ex)
    await verify_visitor(evaluator, main_node, ex)
    await verify_recreation(evaluator, main_node, ex)

    # Optional / non-critical checks node
    optional_node = evaluator.add_parallel(
        id="Optional_Checks",
        desc="Optional or non-critical checks (advance booking policy, additional activities)",
        parent=root,
        critical=False
    )
    await verify_optional(evaluator, optional_node, ex)

    # Return structured evaluation summary
    return evaluator.get_summary()