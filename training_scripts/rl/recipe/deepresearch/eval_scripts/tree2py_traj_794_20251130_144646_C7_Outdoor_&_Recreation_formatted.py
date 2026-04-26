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
TASK_ID = "nps_west_visitor_center"
TASK_DESCRIPTION = (
    "Identify a national park in the western United States (California, Utah, Arizona, Colorado, Washington, Wyoming, "
    "Montana, Oregon, Nevada, Idaho, or New Mexico) that has a main visitor center meeting all of the following "
    "requirements: (1) The visitor center has a bookstore or gift shop, (2) The visitor center has restroom facilities, "
    "(3) The visitor center is wheelchair accessible, (4) The visitor center features educational exhibits or museum "
    "displays, (5) The visitor center has a ranger-staffed information desk, and (6) The visitor center has parking "
    "facilities available. For your identified park, provide the following information: the national park name, the "
    "state where it is located, the current entrance fee for private vehicles (or indicate if it is fee-free), the "
    "visitor center's operating hours or seasonal schedule, and a direct URL to the park's official NPS.gov visitor "
    "center information page. All information must be verifiable from official National Park Service sources (nps.gov domain)."
)

ALLOWED_STATES = {
    "California", "Utah", "Arizona", "Colorado", "Washington", "Wyoming", "Montana",
    "Oregon", "Nevada", "Idaho", "New Mexico"
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkVisitorCenterExtraction(BaseModel):
    park_name: Optional[str] = None
    state: Optional[str] = None
    visitor_center_name: Optional[str] = None
    visitor_center_url: Optional[str] = None

    bookstore_gift_shop: Optional[bool] = None
    restrooms: Optional[bool] = None
    wheelchair_accessible: Optional[bool] = None
    exhibits: Optional[bool] = None
    ranger_info_desk: Optional[bool] = None
    parking: Optional[bool] = None

    entrance_fee_private_vehicle: Optional[str] = None
    visitor_center_hours: Optional[str] = None

    fees_url: Optional[str] = None
    hours_url: Optional[str] = None
    additional_nps_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_visitor_center_info() -> str:
    return (
        "Extract from the answer the following structured information about a single identified national park and its "
        "main visitor center:\n"
        "1) park_name: The full name of the national park.\n"
        "2) state: The U.S. state where the park is located (full state name, not abbreviation).\n"
        "3) visitor_center_name: The name of the main visitor center being evaluated.\n"
        "4) visitor_center_url: A direct NPS.gov URL that specifically provides official visitor center information for this park.\n"
        "5) bookstore_gift_shop: Does the visitor center have a bookstore or gift shop? Return true/false (boolean).\n"
        "6) restrooms: Does the visitor center have restroom facilities? Return true/false (boolean).\n"
        "7) wheelchair_accessible: Is the visitor center wheelchair/ADA accessible? Return true/false (boolean).\n"
        "8) exhibits: Does the visitor center feature educational exhibits or museum displays? Return true/false (boolean).\n"
        "9) ranger_info_desk: Does the visitor center have a ranger-staffed information desk? Return true/false (boolean).\n"
        "10) parking: Does the visitor center have parking facilities available? Return true/false (boolean).\n"
        "11) entrance_fee_private_vehicle: The current entrance fee for private vehicles, or indicate 'fee-free' if applicable, exactly as stated in the answer.\n"
        "12) visitor_center_hours: The operating hours or seasonal schedule for the visitor center, as stated in the answer.\n"
        "13) fees_url: If the answer cites an NPS.gov page for fees (e.g., Fees & Passes), extract that URL. Otherwise null.\n"
        "14) hours_url: If the answer cites an NPS.gov page that lists visitor center hours/schedule, extract that URL. Otherwise null.\n"
        "15) additional_nps_urls: Extract any other NPS.gov URLs explicitly mentioned in the answer that support the claims above.\n\n"
        "Important:\n"
        "- Only extract URLs explicitly present in the answer; do not invent URLs.\n"
        "- Prefer NPS.gov URLs. If a URL is not from nps.gov, still include it in the structured output, but it may fail verification later.\n"
        "- Return boolean values for the yes/no facility/accessibility items.\n"
        "- If any field is not provided in the answer, return null (for strings) or false (for booleans if clearly negated); if not mentioned at all, use null for strings and null for booleans.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_nps_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return "nps.gov" in u


def collect_all_nps_sources(info: ParkVisitorCenterExtraction) -> List[str]:
    urls: List[str] = []
    for u in [info.visitor_center_url, info.fees_url, info.hours_url]:
        if u:
            urls.append(u)
    urls.extend([u for u in (info.additional_nps_urls or []) if u])
    # De-duplicate
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def feature_sources(info: ParkVisitorCenterExtraction) -> List[str]:
    """
    Prefer the visitor center page; include hours/fees pages and additional NPS URLs if present.
    """
    srcs = collect_all_nps_sources(info)
    # If no sources, return empty list to let the verifier conclude not supported
    return srcs if srcs else []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root: Any,
    info: ParkVisitorCenterExtraction
) -> None:
    """
    Build the verification tree following the rubric and run verifications.
    """
    # Create a critical parent node under evaluator.root to mirror the rubric root
    main_node = evaluator.add_parallel(
        id="National_Park_Visitor_Center_Meeting_All_Criteria",
        desc="A single national park in the western United States whose main visitor center meets all specified facility, accessibility, and information requirements, with all claims verifiable from official NPS sources.",
        parent=root,
        critical=True
    )

    # Western_US_Location (leaf verification)
    western_leaf = evaluator.add_leaf(
        id="Western_US_Location",
        desc="The national park is located in a western U.S. state (California, Utah, Arizona, Colorado, Washington, Wyoming, Montana, Oregon, Nevada, Idaho, or New Mexico).",
        parent=main_node,
        critical=True
    )
    state_val = info.state or ""
    allowed_list_str = ", ".join(sorted(ALLOWED_STATES))
    western_claim = f"The state '{state_val}' is one of the following western U.S. states: {allowed_list_str}."
    await evaluator.verify(
        claim=western_claim,
        node=western_leaf,
        additional_instruction="This is a simple membership check. Consider the claim correct if the state exactly matches any item in the list."
    )

    # Park_Name_Identified (existence)
    evaluator.add_custom_node(
        result=bool(info.park_name and info.park_name.strip()),
        id="Park_Name_Identified",
        desc="The specific national park name is clearly identified.",
        parent=main_node,
        critical=True
    )

    # State_Location_Specified (existence)
    evaluator.add_custom_node(
        result=bool(info.state and info.state.strip()),
        id="State_Location_Specified",
        desc="The specific state where the park is located is clearly specified.",
        parent=main_node,
        critical=True
    )

    # Main_Visitor_Center_Identified (existence)
    evaluator.add_custom_node(
        result=bool(info.visitor_center_name and info.visitor_center_name.strip()),
        id="Main_Visitor_Center_Identified",
        desc="The park's main visitor center being evaluated is identified.",
        parent=main_node,
        critical=True
    )

    # Bookstore_Gift_Shop_Present (leaf by URLs)
    bookstore_leaf = evaluator.add_leaf(
        id="Bookstore_Gift_Shop_Present",
        desc="The visitor center has a bookstore or gift shop.",
        parent=main_node,
        critical=True
    )
    bookstore_claim = (
        f"The visitor center '{info.visitor_center_name or ''}' at {info.park_name or 'the park'} "
        f"has a bookstore or gift shop."
    )
    await evaluator.verify(
        claim=bookstore_claim,
        node=bookstore_leaf,
        sources=feature_sources(info),
        additional_instruction="On the official NPS visitor center or park pages, look for mentions of 'Bookstore', 'Gift Shop', 'Park Store', or similar."
    )

    # Restroom_Facilities (leaf by URLs)
    restrooms_leaf = evaluator.add_leaf(
        id="Restroom_Facilities",
        desc="The visitor center has restroom facilities.",
        parent=main_node,
        critical=True
    )
    restrooms_claim = (
        f"The visitor center '{info.visitor_center_name or ''}' at {info.park_name or 'the park'} "
        f"has public restroom facilities."
    )
    await evaluator.verify(
        claim=restrooms_claim,
        node=restrooms_leaf,
        sources=feature_sources(info),
        additional_instruction="On NPS pages, verify presence of 'Restrooms' or 'Toilets' listed as an amenity at the visitor center."
    )

    # Wheelchair_Accessible (leaf by URLs)
    wheelchair_leaf = evaluator.add_leaf(
        id="Wheelchair_Accessible",
        desc="The visitor center is wheelchair accessible / ADA accessible.",
        parent=main_node,
        critical=True
    )
    wheelchair_claim = (
        f"The visitor center '{info.visitor_center_name or ''}' at {info.park_name or 'the park'} "
        f"is wheelchair accessible (ADA accessible)."
    )
    await evaluator.verify(
        claim=wheelchair_claim,
        node=wheelchair_leaf,
        sources=feature_sources(info),
        additional_instruction="Check for 'Accessible', 'Wheelchair accessible', or ADA accessibility statements on the NPS page for the visitor center or park."
    )

    # Educational_Exhibits (leaf by URLs)
    exhibits_leaf = evaluator.add_leaf(
        id="Educational_Exhibits",
        desc="The visitor center features educational exhibits or museum displays.",
        parent=main_node,
        critical=True
    )
    exhibits_claim = (
        f"The visitor center '{info.visitor_center_name or ''}' at {info.park_name or 'the park'} "
        f"features educational exhibits or museum displays."
    )
    await evaluator.verify(
        claim=exhibits_claim,
        node=exhibits_leaf,
        sources=feature_sources(info),
        additional_instruction="Look for 'Exhibits', 'Displays', 'Interpretive exhibits', or similar language on the official NPS page."
    )

    # Ranger_Information_Desk (leaf by URLs)
    ranger_desk_leaf = evaluator.add_leaf(
        id="Ranger_Information_Desk",
        desc="The visitor center has a ranger-staffed information desk.",
        parent=main_node,
        critical=True
    )
    ranger_desk_claim = (
        f"The visitor center '{info.visitor_center_name or ''}' at {info.park_name or 'the park'} "
        f"has a ranger-staffed information desk."
    )
    await evaluator.verify(
        claim=ranger_desk_claim,
        node=ranger_desk_leaf,
        sources=feature_sources(info),
        additional_instruction="Look for phrases like 'Information desk staffed by rangers', 'Ranger information desk', or 'Rangers available to answer questions'."
    )

    # Parking_Available (leaf by URLs)
    parking_leaf = evaluator.add_leaf(
        id="Parking_Available",
        desc="The visitor center has parking facilities available.",
        parent=main_node,
        critical=True
    )
    parking_claim = (
        f"The visitor center '{info.visitor_center_name or ''}' at {info.park_name or 'the park'} "
        f"has parking facilities available."
    )
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=feature_sources(info),
        additional_instruction="Check for 'Parking', 'Parking lot', or 'Visitor center parking' on the official NPS page."
    )

    # Entrance_Fee_Information (leaf by URLs)
    fee_leaf = evaluator.add_leaf(
        id="Entrance_Fee_Information",
        desc="Current entrance fee information for private vehicles is provided (or explicitly indicated as fee-free).",
        parent=main_node,
        critical=True
    )
    fee_text = (info.entrance_fee_private_vehicle or "").strip()
    if fee_text and ("fee-free" in fee_text.lower() or "free" in fee_text.lower()):
        fee_claim = "The park is fee-free for private vehicles."
    else:
        fee_claim = f"The entrance fee for private vehicles is {fee_text}."
    fee_sources = [u for u in [info.fees_url, info.visitor_center_url] if u] + (info.additional_nps_urls or [])
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=fee_sources if fee_sources else None,
        additional_instruction="Verify this against the official NPS Fees & Passes page or other NPS pages listing entrance fees for private vehicles."
    )

    # Operating_Hours_Provided (leaf by URLs)
    hours_leaf = evaluator.add_leaf(
        id="Operating_Hours_Provided",
        desc="The visitor center's operating hours or seasonal schedule is provided.",
        parent=main_node,
        critical=True
    )
    hours_text = (info.visitor_center_hours or "").strip()
    hours_claim = (
        f"The operating hours or seasonal schedule for the visitor center '{info.visitor_center_name or ''}' is: {hours_text}"
    )
    hours_sources = [u for u in [info.hours_url, info.visitor_center_url] if u] + (info.additional_nps_urls or [])
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=hours_sources if hours_sources else None,
        additional_instruction="Verify the hours/schedule on the official NPS visitor center page or Hours/Schedule pages for the park."
    )

    # NPS_Official_Visitor_Center_URL (leaf by URL)
    nps_vc_url_leaf = evaluator.add_leaf(
        id="NPS_Official_Visitor_Center_URL",
        desc="A direct URL to the park's official NPS.gov visitor center information page is provided.",
        parent=main_node,
        critical=True
    )
    nps_vc_claim = (
        f"This page is an official NPS visitor center information page for the visitor center '{info.visitor_center_name or ''}' "
        f"at {info.park_name or 'the park'}."
    )
    await evaluator.verify(
        claim=nps_vc_claim,
        node=nps_vc_url_leaf,
        sources=info.visitor_center_url,
        additional_instruction="Confirm the page is on the nps.gov domain and specifically provides visitor center information (e.g., amenities, hours, exhibits)."
    )

    # All_Info_Verifiable_From_NPS_Only (custom domain check)
    used_sources = collect_all_nps_sources(info)
    all_nps = (len(used_sources) > 0) and all(is_nps_url(u) for u in used_sources)
    evaluator.add_custom_node(
        result=all_nps,
        id="All_Info_Verifiable_From_NPS_Only",
        desc="All required claims and provided details are verifiable from official NPS sources on the nps.gov domain (no third-party sources used as verification).",
        parent=main_node,
        critical=True
    )

    # Record custom info for debugging
    evaluator.add_custom_info(
        info={"used_sources": used_sources, "visitor_center_url": info.visitor_center_url},
        info_type="url_sources",
        info_name="verification_sources"
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
    Evaluate an answer for the western NPS visitor center task.
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

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_park_visitor_center_info(),
        template_class=ParkVisitorCenterExtraction,
        extraction_name="park_visitor_center_info"
    )

    # Add ground truth context (allowed states list)
    evaluator.add_ground_truth({
        "allowed_states": sorted(list(ALLOWED_STATES)),
        "requirement": "Park must be in allowed western U.S. states; all claims must be supported by NPS.gov pages."
    })

    # Build tree and verify
    await build_and_verify_tree(evaluator, root, extracted_info)

    # Return summary
    return evaluator.get_summary()