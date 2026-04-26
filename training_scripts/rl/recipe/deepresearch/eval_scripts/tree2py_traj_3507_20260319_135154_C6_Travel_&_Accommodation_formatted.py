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
TASK_ID = "epic_universe_travel_plan"
TASK_DESCRIPTION = """
A family of 4 (2 adults and 2 children) with a service dog is planning to visit Universal Epic Universe in Orlando. They need to book appropriate accommodations, flights, and ensure their service dog documentation is in order for their broader travel plans that include a Celebrity Cruise.

For their Epic Universe visit, they want to stay at the Universal Helios Grand Hotel in a Theme Park View room. They need to fly via Breeze Airways and want to ensure they select the most cost-effective fare bundle that allows each family member to check at least one bag. Their service dog must meet all Celebrity Cruises documentation requirements.

Provide a comprehensive travel plan that includes:

1. The specific room type at Universal Helios Grand Hotel that accommodates the family of 4 with a Theme Park View of Epic Universe, confirming it provides the dedicated entrance benefit to the park

2. The earliest possible visit date to Epic Universe based on the park's opening date

3. The minimum Breeze Airways fare bundle (Nice, Nicer, or Nicest) that provides adequate checked baggage allowance for the family of 4, where each person needs to check at least one bag

4. Confirmation that all required service dog documentation is prepared per Celebrity Cruises requirements, including current vaccination records and health certificate

For each component, provide specific details with supporting reference URLs from official sources.
"""

EPIC_UNIVERSE_OPENING_ISO = "2025-05-22"
EPIC_UNIVERSE_OPENING_HUMAN = "May 22, 2025"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    room_type: Optional[str] = None
    view: Optional[str] = None  # e.g., "Theme Park View"
    max_occupancy: Optional[str] = None  # keep as free text to be robust
    room_specs_urls: List[str] = Field(default_factory=list)
    dedicated_entrance: Optional[str] = None  # free text yes/no claim in answer
    dedicated_entrance_urls: List[str] = Field(default_factory=list)


class ParkVisitInfo(BaseModel):
    proposed_visit_date: Optional[str] = None  # any format as in the answer
    opening_date_claim: Optional[str] = None   # e.g., "May 22, 2025"
    opening_date_urls: List[str] = Field(default_factory=list)


class BreezeInfo(BaseModel):
    minimum_bundle: Optional[str] = None  # "Nice", "Nicer", or "Nicest"
    nice_checked_bag_included: Optional[str] = None  # e.g., "0", "none", "no"
    nice_urls: List[str] = Field(default_factory=list)
    nicer_checked_bag_included: Optional[str] = None  # e.g., "1", "one per person"
    nicer_urls: List[str] = Field(default_factory=list)
    bundle_urls: List[str] = Field(default_factory=list)  # any general Breeze bundle URL(s)


class ServiceDogInfo(BaseModel):
    vaccinations_current_claim: Optional[str] = None  # answer explicitly states current incl. Rabies
    vaccination_urls: List[str] = Field(default_factory=list)  # Celebrity policy/supporting URLs
    health_certificate_obtained_claim: Optional[str] = None
    health_certificate_urls: List[str] = Field(default_factory=list)
    service_dog_policy_urls: List[str] = Field(default_factory=list)  # general service dog policy URLs


class TravelPlanExtraction(BaseModel):
    hotel: Optional[HotelInfo] = None
    park_visit: Optional[ParkVisitInfo] = None
    breeze: Optional[BreezeInfo] = None
    service_dog: Optional[ServiceDogInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_plan() -> str:
    return """
Extract structured details from the provided answer for a travel plan to Universal Epic Universe with a family of 4 and a service dog. Return data according to the following schema:

hotel:
- room_type: The specific named room type at Universal Helios Grand Hotel as stated in the answer (e.g., "Theme Park View 2 Queen Room"). If not provided, null.
- view: The view designation as stated (e.g., "Theme Park View"). If absent, null.
- max_occupancy: The stated maximum occupancy or capacity text (e.g., "sleeps up to 4"). If absent, null.
- room_specs_urls: All URLs in the answer that support the room specifications (official Universal/Hotel pages preferred). Only include actual URLs mentioned.
- dedicated_entrance: The answer's explicit statement about a dedicated entrance benefit (e.g., "yes, dedicated entrance" or any similar phrasing). If absent, null.
- dedicated_entrance_urls: All URLs in the answer supporting the dedicated entrance benefit (official Universal site preferred).

park_visit:
- proposed_visit_date: The proposed visit date for Epic Universe given by the answer. If multiple, extract the earliest one mentioned. Preserve the exact format from the answer.
- opening_date_claim: The opening date text if the answer states Epic Universe's opening date. If absent, null.
- opening_date_urls: All URLs in the answer supporting the opening date (prefer official Universal announcements or authoritative sources).

breeze:
- minimum_bundle: The minimum Breeze Airways bundle the answer recommends or selects to meet "at least one checked bag per person" (Nice, Nicer, or Nicest). If not stated, null.
- nice_checked_bag_included: The answer's explicit statement for Nice bundle checked bag inclusion (e.g., "0", "none", "no"). If absent, null.
- nice_urls: All URLs for the Nice bundle specifications mentioned in the answer (official Breeze pages preferred).
- nicer_checked_bag_included: The answer's explicit statement for Nicer bundle checked bag inclusion (e.g., "1", "one per person"). If absent, null.
- nicer_urls: All URLs for the Nicer bundle specifications mentioned in the answer (official Breeze pages preferred).
- bundle_urls: Any general Breeze bundle/fare comparison URLs mentioned.

service_dog:
- vaccinations_current_claim: The answer's explicit statement confirming current vaccinations, including Rabies, for the service dog. If absent, null.
- vaccination_urls: All URLs for Celebrity Cruises vaccination/service dog requirements cited in the answer (official Celebrity pages preferred).
- health_certificate_obtained_claim: The answer's explicit statement that a USDA or International Health Certificate has been or will be obtained. If absent, null.
- health_certificate_urls: All URLs supporting health certificate requirements cited (official Celebrity pages preferred).
- service_dog_policy_urls: Any URLs in the answer pointing to Celebrity Cruises service dog policy pages.

Rules:
- Extract only what is explicitly present in the answer. Do not invent or infer.
- For URLs, extract the actual URLs (plain or in markdown). Do not paraphrase.
- If an item is missing, set it to null (for string fields) or an empty list (for URL arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def union_urls(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst:
            nu = (u or "").strip()
            if nu and nu not in seen:
                seen.add(nu)
                out.append(nu)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_hotel_verification(evaluator: Evaluator, parent) -> None:
    """
    Build and run verification nodes for Hotel_Verification.
    """
    extracted: TravelPlanExtraction = next(
        (info.get("travel_plan") for info in evaluator.get_summary()["eval_breakdown"][0]["info"]
         if "travel_plan" in info),
        None
    )  # This is not available until we call get_summary; instead we should use stored extraction result.

    # Better: we need direct access. We'll pass via closure argument when calling this function.
    raise RuntimeError("Internal misuse: build_hotel_verification requires extracted object passed in.")


async def verify_hotel(evaluator: Evaluator, parent, hotel: Optional[HotelInfo]) -> None:
    hotel_node = evaluator.add_parallel(
        id="Hotel_Verification",
        desc="Verify Helios Grand Hotel accommodation adequacy",
        parent=parent,
        critical=True
    )

    # Room_Adequacy
    room_node = evaluator.add_parallel(
        id="Room_Adequacy",
        desc="Verify room specifications meet requirements",
        parent=hotel_node,
        critical=True
    )

    # Occupancy_And_View
    occ_view_node = evaluator.add_parallel(
        id="Occupancy_And_View",
        desc="Confirm room has 4+ occupancy AND Theme Park View designation",
        parent=room_node,
        critical=True
    )

    room_urls = non_empty_urls(hotel.room_specs_urls if hotel else [])

    # Occupancy_4Plus
    occ_leaf = evaluator.add_leaf(
        id="Occupancy_4Plus",
        desc="Verify maximum occupancy is 4 or more guests",
        parent=occ_view_node,
        critical=True
    )
    room_name_for_claim = hotel.room_type if (hotel and hotel.room_type) else "the Theme Park View room at Universal Helios Grand Hotel"
    occ_claim = (
        f"The specified room '{room_name_for_claim}' at Universal Helios Grand Hotel accommodates at least 4 guests "
        f"(e.g., 'sleeps up to 4' or similar capacity)."
    )
    await evaluator.verify(
        claim=occ_claim,
        node=occ_leaf,
        sources=room_urls,
        additional_instruction="Look for capacity text like 'sleeps up to 4' or occupancy for the exact room type. "
                              "If no URL is provided or the page does not clearly show 4+ capacity, mark as not supported."
    )

    # Theme_Park_View
    view_leaf = evaluator.add_leaf(
        id="Theme_Park_View",
        desc="Verify room is designated Theme Park View",
        parent=occ_view_node,
        critical=True
    )
    # Be flexible with view wording
    view_target = hotel.view if (hotel and hotel.view) else "Theme Park View"
    view_claim = (
        f"The specified room '{room_name_for_claim}' is explicitly designated as '{view_target}' "
        f"(i.e., a Theme Park View of Epic Universe or equivalent wording)."
    )
    await evaluator.verify(
        claim=view_claim,
        node=view_leaf,
        sources=room_urls,
        additional_instruction="Accept reasonable variants like 'Theme Park View', 'Park View', or explicit mention of view of Epic Universe. "
                              "The evidence must be on the hotel/Universal official room page."
    )

    # Room_Specs_URL (URL presence)
    room_url_presence = evaluator.add_custom_node(
        result=len(room_urls) > 0,
        id="Room_Specs_URL",
        desc="Provide URL confirming room specifications",
        parent=occ_view_node,
        critical=True
    )

    # Dedicated_Entrance
    entrance_node = evaluator.add_parallel(
        id="Dedicated_Entrance",
        desc="Confirm hotel provides dedicated entrance to Epic Universe",
        parent=room_node,
        critical=True
    )
    entrance_urls = non_empty_urls(hotel.dedicated_entrance_urls if hotel else [])

    entrance_leaf = evaluator.add_leaf(
        id="Entrance_Benefit",
        desc="Verify dedicated entrance benefit exists",
        parent=entrance_node,
        critical=True
    )
    entrance_claim = "Universal Helios Grand Hotel offers a dedicated entrance to Epic Universe for its hotel guests."
    await evaluator.verify(
        claim=entrance_claim,
        node=entrance_leaf,
        sources=entrance_urls,
        additional_instruction="The supporting page should explicitly mention a dedicated/private/park entrance benefit to Epic Universe."
    )

    entrance_url_presence = evaluator.add_custom_node(
        result=len(entrance_urls) > 0,
        id="Entrance_URL",
        desc="Provide URL confirming entrance benefit",
        parent=entrance_node,
        critical=True
    )

    # Visit_Date_Compliance
    visit_seq = evaluator.add_sequential(
        id="Visit_Date_Compliance",
        desc="Verify visit date is valid relative to park opening",
        parent=hotel_node,
        critical=True
    )

    # Opening date container
    opening_parallel = evaluator.add_parallel(
        id="Opening_Date_May_22_2025",
        desc="Confirm Epic Universe opened May 22, 2025",
        parent=visit_seq,
        critical=True
    )

    # Opening date fact
    opening_urls = non_empty_urls(hotel.room_specs_urls if False else [])  # placeholder to ensure variable defined
    # Actually, opening URLs are in park_visit.opening_date_urls; this function doesn't have that. We'll handle in visit section.
    # For correctness, this function only builds hotel nodes. The visit nodes are built in a separate function.
    # So nothing more here for visit in this function.


async def verify_visit(evaluator: Evaluator, parent, visit: Optional[ParkVisitInfo]) -> None:
    # Build Visit_Date_Compliance under a "Hotel_Verification" sibling per rubric tree structure, but the rubric
    # places Visit_Date_Compliance under Hotel_Verification. We'll follow rubric: we'll add under Hotel_Verification
    # in the calling hierarchy. To keep it modular, this function will return a node we can attach under Hotel_Verification.
    # However, to match the rubric exactly, we will assume parent passed here is the Hotel_Verification node.

    visit_seq = evaluator.add_sequential(
        id="Visit_Date_Compliance",
        desc="Verify visit date is valid relative to park opening",
        parent=parent,
        critical=True
    )

    # Opening_Date_May_22_2025
    opening_node = evaluator.add_parallel(
        id="Opening_Date_May_22_2025",
        desc="Confirm Epic Universe opened May 22, 2025",
        parent=visit_seq,
        critical=True
    )

    opening_urls = non_empty_urls(visit.opening_date_urls if visit else [])

    opening_fact_leaf = evaluator.add_leaf(
        id="Opening_Date_Fact",
        desc="Verify opening date is May 22, 2025",
        parent=opening_node,
        critical=True
    )
    opening_claim = f"Epic Universe opens/opened on {EPIC_UNIVERSE_OPENING_HUMAN}."
    await evaluator.verify(
        claim=opening_claim,
        node=opening_fact_leaf,
        sources=opening_urls,
        additional_instruction="Verify using official Universal Orlando or authoritative announcements. "
                              "If the URL is irrelevant/invalid, mark as not supported."
    )

    opening_url_presence = evaluator.add_custom_node(
        result=len(opening_urls) > 0,
        id="Opening_Date_URL",
        desc="Provide URL confirming opening date",
        parent=opening_node,
        critical=True
    )

    # Date_After_Opening
    after_leaf = evaluator.add_leaf(
        id="Date_After_Opening",
        desc="Confirm proposed visit date is on or after May 22, 2025",
        parent=visit_seq,
        critical=True
    )
    proposed_date = visit.proposed_visit_date if visit and visit.proposed_visit_date else "MISSING"
    after_claim = f"The proposed visit date '{proposed_date}' is on or after {EPIC_UNIVERSE_OPENING_ISO}."
    await evaluator.verify(
        claim=after_claim,
        node=after_leaf,
        additional_instruction=(
            "Use the answer text as the only source for the proposed date. "
            "Accept common formats like 'May 23, 2025', '2025-05-23', '05/23/2025'. "
            "If the answer provides no proposed visit date (e.g., 'MISSING'), judge the claim incorrect."
        )
    )


async def verify_flight_baggage(evaluator: Evaluator, parent, breeze: Optional[BreezeInfo]) -> None:
    flight_seq = evaluator.add_sequential(
        id="Flight_Baggage_Verification",
        desc="Verify Breeze Airways fare provides adequate checked baggage",
        parent=parent,
        critical=True
    )

    # Baggage_Requirement (logical)
    bag_req_leaf = evaluator.add_leaf(
        id="Baggage_Requirement",
        desc="Determine family requires minimum 4 checked bags (1 per person)",
        parent=flight_seq,
        critical=True
    )
    bag_req_claim = "A family of 4, where each person needs at least one checked bag, requires a minimum of 4 checked bags total."
    await evaluator.verify(
        claim=bag_req_claim,
        node=bag_req_leaf,
        additional_instruction="This is a simple logical check based on the task description."
    )

    # Fare_Bundle_Identification
    bundle_node = evaluator.add_parallel(
        id="Fare_Bundle_Identification",
        desc="Identify Breeze Airways fare bundle meeting baggage requirement",
        parent=flight_seq,
        critical=True
    )

    # Nice_Bundle_Analysis
    nice_node = evaluator.add_parallel(
        id="Nice_Bundle_Analysis",
        desc="Analyze Nice bundle baggage allowance",
        parent=bundle_node,
        critical=True
    )

    nice_urls = non_empty_urls(breeze.nice_urls if breeze else [])
    nice_no_bag_leaf = evaluator.add_leaf(
        id="Nice_Includes_No_Checked_Bag",
        desc="Confirm Nice includes only personal item and carry-on with no checked bag",
        parent=nice_node,
        critical=True
    )
    nice_claim = "Breeze Airways 'Nice' bundle does not include any checked bag (only a personal item and/or carry-on)."
    await evaluator.verify(
        claim=nice_claim,
        node=nice_no_bag_leaf,
        sources=nice_urls,
        additional_instruction="Verify on Breeze official fare/bundle page(s) and baggage policy. "
                              "If 'Nice' includes zero checked bags by default, pass."
    )

    nice_url_presence = evaluator.add_custom_node(
        result=len(nice_urls) > 0,
        id="Nice_Bundle_URL",
        desc="Provide URL for Nice bundle specifications",
        parent=nice_node,
        critical=True
    )

    # Nicer_Bundle_Analysis
    nicer_node = evaluator.add_parallel(
        id="Nicer_Bundle_Analysis",
        desc="Analyze Nicer bundle baggage allowance",
        parent=bundle_node,
        critical=True
    )

    nicer_urls = non_empty_urls(breeze.nicer_urls if breeze else [])
    nicer_1_bag_leaf = evaluator.add_leaf(
        id="Nicer_Includes_1_Bag",
        desc="Confirm Nicer includes 1 checked bag per person",
        parent=nicer_node,
        critical=True
    )
    nicer_claim = "Breeze Airways 'Nicer' bundle includes 1 checked bag per passenger."
    await evaluator.verify(
        claim=nicer_claim,
        node=nicer_1_bag_leaf,
        sources=nicer_urls,
        additional_instruction="Verify on Breeze official fare/bundle comparison page. "
                              "Allow wording variants like 'includes one checked bag' or an icon indicating 1 checked bag."
    )

    nicer_url_presence = evaluator.add_custom_node(
        result=len(nicer_urls) > 0,
        id="Nicer_Bundle_URL",
        desc="Provide URL for Nicer bundle specifications",
        parent=nicer_node,
        critical=True
    )

    # Minimum_Bundle_Selection
    min_bundle_leaf = evaluator.add_leaf(
        id="Minimum_Bundle_Selection",
        desc="Identify Nicer as minimum bundle providing 4 checked bags total",
        parent=bundle_node,
        critical=True
    )
    all_bundle_urls = union_urls(nice_urls, nicer_urls, non_empty_urls(breeze.bundle_urls if breeze else []))
    min_bundle_claim = "For Breeze Airways, 'Nicer' is the minimum bundle that includes at least 1 checked bag per person, whereas 'Nice' does not include checked bags."
    await evaluator.verify(
        claim=min_bundle_claim,
        node=min_bundle_leaf,
        sources=all_bundle_urls,
        additional_instruction="Use Breeze official fare comparison. The conclusion should be that 'Nicer' is the lowest-cost bundle that includes a checked bag for each traveler."
    )


async def verify_service_dog(evaluator: Evaluator, parent, svc: Optional[ServiceDogInfo]) -> None:
    svc_node = evaluator.add_parallel(
        id="Service_Dog_Documentation_Verification",
        desc="Verify service dog documentation meets Celebrity Cruises requirements",
        parent=parent,
        critical=True
    )

    # Vaccination_Records
    vacc_node = evaluator.add_parallel(
        id="Vaccination_Records",
        desc="Verify current vaccination records including Rabies",
        parent=svc_node,
        critical=True
    )

    vacc_current_leaf = evaluator.add_leaf(
        id="All_Vaccinations_Current",
        desc="Confirm all vaccinations including Rabies are current",
        parent=vacc_node,
        critical=True
    )
    vacc_current_claim = (
        "The travel plan explicitly confirms that the service dog has current vaccinations, including Rabies."
    )
    await evaluator.verify(
        claim=vacc_current_claim,
        node=vacc_current_leaf,
        additional_instruction="Judge based on the answer text alone. If the answer does not explicitly confirm current vaccinations including Rabies, mark as incorrect."
    )

    vacc_urls = non_empty_urls(svc.vaccination_urls if svc else [])
    vacc_url_presence = evaluator.add_custom_node(
        result=len(vacc_urls) > 0,
        id="Vaccination_Requirements_URL",
        desc="Provide URL for Celebrity Cruises vaccination requirements",
        parent=vacc_node,
        critical=True
    )

    # Health_Certificate
    cert_node = evaluator.add_parallel(
        id="Health_Certificate",
        desc="Verify USDA or International Health Certificate obtained",
        parent=svc_node,
        critical=True
    )

    cert_obtained_leaf = evaluator.add_leaf(
        id="Certificate_Obtained",
        desc="Confirm USDA or International Health Certificate is available",
        parent=cert_node,
        critical=True
    )
    cert_claim = "The travel plan confirms that a USDA or International Health Certificate for the service dog has been or will be obtained."
    await evaluator.verify(
        claim=cert_claim,
        node=cert_obtained_leaf,
        additional_instruction="Judge based on the answer text alone. If not explicitly stated, mark as incorrect."
    )

    cert_urls = non_empty_urls(svc.health_certificate_urls if svc else [])
    cert_url_presence = evaluator.add_custom_node(
        result=len(cert_urls) > 0,
        id="Certificate_Requirements_URL",
        desc="Provide URL for health certificate requirements",
        parent=cert_node,
        critical=True
    )

    # Acceptance_Policy
    accept_node = evaluator.add_parallel(
        id="Acceptance_Policy",
        desc="Verify Celebrity Cruises accepts service dogs",
        parent=svc_node,
        critical=True
    )

    policy_urls = non_empty_urls(svc.service_dog_policy_urls if svc else [])
    accepted_leaf = evaluator.add_leaf(
        id="Service_Dogs_Accepted",
        desc="Confirm Celebrity Cruises accepts service dogs on applicable ships",
        parent=accept_node,
        critical=True
    )
    accepted_claim = "Celebrity Cruises accepts service dogs (service animals) onboard, while emotional support animals are not accepted."
    await evaluator.verify(
        claim=accepted_claim,
        node=accepted_leaf,
        sources=policy_urls,
        additional_instruction="Verify against Celebrity Cruises official policy page(s)."
    )

    policy_url_presence = evaluator.add_custom_node(
        result=len(policy_urls) > 0,
        id="Service_Dog_Policy_URL",
        desc="Provide URL for Celebrity Cruises service dog policy",
        parent=accept_node,
        critical=True
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
    Evaluate the comprehensive travel plan for Universal Epic Universe (Helios Grand Hotel room, visit date, Breeze Airways bundle for checked bags, and Celebrity Cruises service dog documentation).
    """
    # Initialize evaluator (root is a non-critical aggregator by framework design)
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

    # Extract structured info
    extracted: TravelPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_travel_plan(),
        template_class=TravelPlanExtraction,
        extraction_name="travel_plan"
    )

    # Add ground truth info for reference
    evaluator.add_ground_truth({
        "epic_universe_opening_date": EPIC_UNIVERSE_OPENING_HUMAN,
        "required_checked_bags_minimum": "4 (1 per person for a family of 4)",
        "expected_minimum_breeze_bundle": "Nicer (includes 1 checked bag per passenger)"
    }, gt_type="ground_truth")

    # Build Travel_Plan_Validation node as top-level under root
    plan_node = evaluator.add_parallel(
        id="Travel_Plan_Validation",
        desc=("Verify family of 4 with service dog travel plan for Epic Universe includes: Helios Grand Hotel "
              "Theme Park View room with 4+ capacity and dedicated entrance, visit date on/after May 22, 2025, "
              "Breeze Airways fare providing 4+ checked bags total, and service dog documentation meeting "
              "Celebrity Cruises requirements"),
        parent=root,
        critical=True
    )

    # Hotel_Verification (and nested Visit_Date_Compliance per rubric)
    hotel_parent = evaluator.add_parallel(
        id="Hotel_Verification",
        desc="Verify Helios Grand Hotel accommodation adequacy",
        parent=plan_node,
        critical=True
    )

    # Room adequacy + entrance
    await verify_hotel(evaluator, hotel_parent, extracted.hotel)

    # Visit date compliance under Hotel_Verification (per rubric structure)
    await verify_visit(evaluator, hotel_parent, extracted.park_visit)

    # Flight/Baggage verification
    await verify_flight_baggage(evaluator, plan_node, extracted.breeze)

    # Service dog documentation
    await verify_service_dog(evaluator, plan_node, extracted.service_dog)

    # Return structured summary
    return evaluator.get_summary()