import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mlk_resort_2026"
TASK_DESCRIPTION = (
    "I'm planning a family vacation for MLK Day weekend 2026 (January 16-19, 2026) and need to find a suitable "
    "resort in the United States. The resort must meet the following requirements:\n\n"
    "Essential Requirements:\n"
    "- Located within the United States\n"
    "- Available for booking during MLK Day weekend 2026 (January 16-19, 2026)\n"
    "- Offers rooms that can accommodate at least 4 adult guests plus one infant under age 3 in a crib\n"
    "- Has ADA-compliant accessible rooms available\n"
    "- Minimum check-in age is 18 years old (allowing 18-year-olds to check in independently)\n\n"
    "Additional Desired Features:\n"
    "- Pet-friendly policy that allows dogs (with information about maximum number of dogs permitted per room)\n"
    "- On-site parking services with pricing information\n"
    "- Supervised kids club or children's activity program with specified age ranges\n"
    "- At least one swimming pool on property\n"
    "- At least one on-site restaurant (with information about dining variety)\n"
    "- Spa services or treatments available on property\n"
    "- Fitness center or gym facility\n"
    "- Cancellation policy allowing free cancellation at least 24 hours before check-in\n"
    "- Hotel loyalty program with membership benefits\n\n"
    "Please identify one resort that meets all essential requirements and provide information about as many of the "
    "additional desired features as possible, along with reference URLs for verification."
)


# --------------------------------------------------------------------------- #
# Data Models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortExtraction(BaseModel):
    # Core identification
    resort_name: Optional[str] = None
    location_text: Optional[str] = None  # e.g., "Scottsdale, Arizona, USA"
    primary_url: Optional[str] = None

    # Essential requirements (values as they appear in the answer)
    mlk_available: Optional[bool] = None
    mlk_dates_text: Optional[str] = None  # e.g., "Jan 16–19, 2026"

    min_four_adults_supported: Optional[bool] = None
    room_min_adults_text: Optional[str] = None  # e.g., "Sleeps 4 adults"

    crib_allowed: Optional[bool] = None
    crib_policy_text: Optional[str] = None

    ada_accessible_rooms: Optional[bool] = None
    ada_text: Optional[str] = None

    checkin_age_18_ok: Optional[bool] = None
    checkin_age_text: Optional[str] = None

    # Desired features
    dogs_allowed: Optional[bool] = None
    pet_max_dogs_per_room: Optional[str] = None  # e.g., "2 dogs per room"
    pet_policy_text: Optional[str] = None

    parking_available: Optional[bool] = None
    parking_pricing_info_provided: Optional[bool] = None
    parking_pricing_text: Optional[str] = None

    kids_club_available: Optional[bool] = None
    kids_club_age_range: Optional[str] = None
    kids_club_text: Optional[str] = None

    pool_available: Optional[bool] = None

    dining_available: Optional[bool] = None
    dining_options_details: Optional[str] = None

    spa_available: Optional[bool] = None
    fitness_center_available: Optional[bool] = None

    cancellation_free_24h: Optional[bool] = None
    cancellation_policy_text: Optional[str] = None

    loyalty_program_present: Optional[bool] = None
    loyalty_program_text: Optional[str] = None

    # URL sources (extracted from the answer; use only explicitly provided URLs)
    location_urls: List[str] = Field(default_factory=list)
    availability_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    crib_urls: List[str] = Field(default_factory=list)
    ada_urls: List[str] = Field(default_factory=list)
    checkin_age_urls: List[str] = Field(default_factory=list)
    pet_policy_urls: List[str] = Field(default_factory=list)
    parking_urls: List[str] = Field(default_factory=list)
    kids_club_urls: List[str] = Field(default_factory=list)
    pool_urls: List[str] = Field(default_factory=list)
    dining_urls: List[str] = Field(default_factory=list)
    spa_urls: List[str] = Field(default_factory=list)
    fitness_urls: List[str] = Field(default_factory=list)
    cancellation_urls: List[str] = Field(default_factory=list)
    loyalty_urls: List[str] = Field(default_factory=list)
    other_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resort() -> str:
    return """
    Extract information about ONE resort recommended in the answer (if multiple are mentioned, choose the FIRST one).
    Return fields exactly as stated in the answer text. If a field is missing, set it to null (or an empty list for URL lists).

    Core identification:
    - resort_name: name of the resort or hotel
    - location_text: the location string given (e.g., city/state/country)
    - primary_url: the main official webpage URL for this resort, if present

    Essential requirements:
    - mlk_available: true/false if the answer explicitly asserts availability for Jan 16–19, 2026; null if unstated
    - mlk_dates_text: the exact dates or phrasing the answer used for MLK weekend availability (e.g., "Jan 16–19, 2026")
    - min_four_adults_supported: true/false if the answer explicitly claims at least one room accommodates ≥4 adults
    - room_min_adults_text: any occupancy phrasing copied from the answer (e.g., "Sleeps 4 adults")
    - crib_allowed: true/false if an infant under age 3 in a crib is allowed; null if unstated
    - crib_policy_text: any crib/pack-and-play phrasing
    - ada_accessible_rooms: true/false if ADA-compliant accessible rooms exist; null if unstated
    - ada_text: copied accessible-room phrasing
    - checkin_age_18_ok: true/false if 18-year-olds can check in; null if unstated
    - checkin_age_text: copied phrasing about minimum check-in age

    Desired features:
    - dogs_allowed: true/false; null if unstated
    - pet_max_dogs_per_room: e.g., "2 dogs per room" (string as written), or null
    - pet_policy_text: copied pet policy phrasing
    - parking_available: true/false; null if unstated
    - parking_pricing_info_provided: true/false; null if unstated
    - parking_pricing_text: the pricing details (string)
    - kids_club_available: true/false; null if unstated
    - kids_club_age_range: string (e.g., "ages 4–12"), or null
    - kids_club_text: copied kids program phrasing
    - pool_available: true/false; null if unstated
    - dining_available: true/false; null if unstated
    - dining_options_details: string about variety/types/cuisines, or null
    - spa_available: true/false; null if unstated
    - fitness_center_available: true/false; null if unstated
    - cancellation_free_24h: true/false; null if unstated
    - cancellation_policy_text: copied cancellation phrasing
    - loyalty_program_present: true/false; null if unstated
    - loyalty_program_text: copied loyalty/membership benefits text

    URL sources (extract only actual URLs explicitly present in the answer; do not invent):
    - location_urls: URLs that help confirm the address/location
    - availability_urls: URLs that show availability for Jan 16–19, 2026 (booking search result pages if provided)
    - capacity_urls: URLs that show room occupancy/maximum adults
    - crib_urls: URLs that show crib/infant policy
    - ada_urls: URLs that show accessible/ADA rooms
    - checkin_age_urls: URLs that show minimum check-in age policy
    - pet_policy_urls: URLs that show pet policy
    - parking_urls: URLs that show parking info and price
    - kids_club_urls: URLs that show kids club/children’s program
    - pool_urls: URLs that show pool info
    - dining_urls: URLs that show onsite dining/restaurants and cuisines
    - spa_urls: URLs that show spa info
    - fitness_urls: URLs that show fitness/gym info
    - cancellation_urls: URLs that show cancellation policy and timing
    - loyalty_urls: URLs that show loyalty program and member benefits
    - other_sources: any other URLs included in the answer

    If a URL is missing a protocol, prepend http:// as instructed in the general rules.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def gather_sources(extracted: ResortExtraction, keys: List[str]) -> List[str]:
    """Collect and deduplicate URLs from the specified extraction fields."""
    seen = set()
    urls: List[str] = []
    for k in keys:
        val = getattr(extracted, k, None)
        if not val:
            continue
        if isinstance(val, list):
            for u in val:
                if isinstance(u, str) and u.strip() and u not in seen:
                    seen.add(u)
                    urls.append(u)
        elif isinstance(val, str) and val.strip():
            if val not in seen:
                seen.add(val)
                urls.append(val)
    return urls


async def verify_with_sources(
    evaluator: Evaluator,
    *,
    claim: str,
    node,
    sources: List[str],
    additional_instruction: str,
    require_url: bool = True
) -> bool:
    """
    Wrapper around evaluator.verify():
    - If require_url is True and no sources provided, instruct the judge to mark as incorrect due to lack of evidence.
    - Otherwise, proceed with provided sources and instruction.
    """
    if require_url and (not sources):
        add_ins = (
            additional_instruction
            + "\nIMPORTANT: No URL evidence was provided with this verification request. "
              "Per the evaluation policy, consider the claim unsupported and mark it as Incorrect."
        )
        return await evaluator.verify(claim=claim, node=node, sources=None, additional_instruction=add_ins)
    else:
        return await evaluator.verify(claim=claim, node=node, sources=sources, additional_instruction=additional_instruction)


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, root, extracted: ResortExtraction):
    resort_name = extracted.resort_name or "the resort"
    # 1) Resort location within the United States (Critical)
    loc_node = evaluator.add_leaf(
        id="Resort_Location",
        desc="The resort must be located within the United States",
        parent=root,
        critical=True
    )
    loc_sources = gather_sources(extracted, ["location_urls", "primary_url", "other_sources"])
    loc_claim = f"The resort named '{resort_name}' is located within the United States."
    await verify_with_sources(
        evaluator,
        claim=loc_claim,
        node=loc_node,
        sources=loc_sources,
        additional_instruction=(
            "Verify by checking the property's address or location on the cited page(s). "
            "Accept US states/territories or 'United States' in the address. "
            "If the address is outside the US or cannot be determined, mark as unsupported."
        ),
    )

    # 2) Availability for Jan 16–19, 2026 (Critical)
    avail_node = evaluator.add_leaf(
        id="MLK_Weekend_Availability",
        desc="The resort must have availability for booking during MLK Day weekend 2026 (January 16-19, 2026)",
        parent=root,
        critical=True
    )
    avail_sources = gather_sources(extracted, ["availability_urls"])
    avail_dates_text = extracted.mlk_dates_text or "January 16–19, 2026"
    avail_claim = f"There is availability for at least one room for a 3-night stay from January 16, 2026 to January 19, 2026."
    await verify_with_sources(
        evaluator,
        claim=avail_claim,
        node=avail_node,
        sources=avail_sources,
        additional_instruction=(
            "Look for booking search results or calendars explicitly showing availability for check-in on Jan 16, 2026 "
            "and check-out on Jan 19, 2026 (3 nights). If the pages do not show those dates or show 'sold out', mark unsupported. "
            "If the evidence is ambiguous or not date-specific, mark unsupported."
        ),
    )

    # 3) Room capacity requirements (Critical group)
    capacity_group = evaluator.add_parallel(
        id="Room_Capacity_Requirements",
        desc="Room accommodation specifications for the family",
        parent=root,
        critical=True
    )

    # 3a) Minimum 4 adults (Critical)
    cap_adults_node = evaluator.add_leaf(
        id="Minimum_Guest_Capacity",
        desc="The room must accommodate at least 4 adult guests",
        parent=capacity_group,
        critical=True
    )
    cap_sources = gather_sources(extracted, ["capacity_urls"])
    cap_claim = "At least one room type at this resort can accommodate 4 or more adults (e.g., 'sleeps 4 adults')."
    await verify_with_sources(
        evaluator,
        claim=cap_claim,
        node=cap_adults_node,
        sources=cap_sources,
        additional_instruction=(
            "Verify occupancy information (e.g., 'Sleeps 4', 'Max 4 adults'). "
            "Do not count infants/children towards the adult count. "
            "If occupancy is unclear or not shown, mark unsupported."
        ),
    )

    # 3b) Infant in a crib under age 3 (Critical)
    crib_node = evaluator.add_leaf(
        id="Infant_Accommodation",
        desc="The room must allow for a child under age 3 in a crib",
        parent=capacity_group,
        critical=True
    )
    crib_sources = gather_sources(extracted, ["crib_urls", "capacity_urls"])
    crib_claim = "The resort allows an infant (under age 3) in a crib/pack-and-play in the room."
    await verify_with_sources(
        evaluator,
        claim=crib_claim,
        node=crib_node,
        sources=crib_sources,
        additional_instruction=(
            "Look for explicit crib/pack-and-play availability or infant policy allowing a crib for children under 3 (or under 2). "
            "If only rollaway beds are mentioned without crib policy, or if nothing about infants is provided, mark unsupported."
        ),
    )

    # 4) ADA accessible rooms (Critical)
    ada_node = evaluator.add_leaf(
        id="ADA_Accessible_Rooms",
        desc="The resort must offer ADA-compliant accessible rooms",
        parent=root,
        critical=True
    )
    ada_sources = gather_sources(extracted, ["ada_urls"])
    ada_claim = "The resort offers ADA-compliant accessible rooms available to book."
    await verify_with_sources(
        evaluator,
        claim=ada_claim,
        node=ada_node,
        sources=ada_sources,
        additional_instruction=(
            "Look for 'accessible rooms', 'ADA accessible', or specific accessibility room categories/features. "
            "General ADA compliance statements without rooms do not satisfy; the page must indicate accessible rooms."
        ),
    )

    # 5) Check-in age 18+ (Critical)
    checkin_node = evaluator.add_leaf(
        id="Check_In_Age_Requirement",
        desc="The check-in age requirement must be 18 years or older (allowing 18-year-olds to check in independently)",
        parent=root,
        critical=True
    )
    checkin_sources = gather_sources(extracted, ["checkin_age_urls"])
    checkin_claim = "The minimum check-in age at this resort is 18 years old."
    await verify_with_sources(
        evaluator,
        claim=checkin_claim,
        node=checkin_node,
        sources=checkin_sources,
        additional_instruction=(
            "Verify policy language stating minimum check-in age. "
            "If it says 21+ or higher, this fails. If not stated, mark unsupported."
        ),
    )

    # 6) Pet-friendly policy (Non-critical group)
    pet_group = evaluator.add_parallel(
        id="Pet_Friendly_Policy",
        desc="The resort allows dogs",
        parent=root,
        critical=False
    )

    # 6a) Dogs allowed
    dogs_node = evaluator.add_leaf(
        id="Dogs_Allowed",
        desc="The resort explicitly allows dogs",
        parent=pet_group,
        critical=False
    )
    pet_sources = gather_sources(extracted, ["pet_policy_urls"])
    dogs_claim = "Dogs (pets) are allowed at this resort per the pet policy (excluding 'service animals only')."
    await verify_with_sources(
        evaluator,
        claim=dogs_claim,
        node=dogs_node,
        sources=pet_sources,
        additional_instruction=(
            "Confirm the pet policy allows dogs. Do NOT count 'service animals only' as pet-friendly. "
            "If the policy excludes pets or is missing, mark unsupported."
        ),
    )

    # 6b) Max dogs info
    max_dogs_node = evaluator.add_leaf(
        id="Maximum_Dogs_Information",
        desc="Information provided about maximum number of dogs allowed per room",
        parent=pet_group,
        critical=False
    )
    max_dogs_text = extracted.pet_max_dogs_per_room
    if max_dogs_text and max_dogs_text.strip():
        max_dogs_claim = f"The pet policy specifies the maximum number of dogs per room as: {max_dogs_text}."
    else:
        max_dogs_claim = "The pet policy specifies a maximum number of dogs allowed per room."
    await verify_with_sources(
        evaluator,
        claim=max_dogs_claim,
        node=max_dogs_node,
        sources=pet_sources,
        additional_instruction=(
            "Look for explicit 'maximum number of dogs' (e.g., '2 dogs per room'). "
            "If unspecified or policy absent, mark unsupported."
        ),
    )

    # 7) Parking services (Non-critical group)
    parking_group = evaluator.add_parallel(
        id="Parking_Services",
        desc="The resort offers on-site parking",
        parent=root,
        critical=False
    )

    parking_available_node = evaluator.add_leaf(
        id="Parking_Available",
        desc="On-site parking is available",
        parent=parking_group,
        critical=False
    )
    parking_sources = gather_sources(extracted, ["parking_urls"])
    parking_available_claim = "On-site parking (self or valet) is available at the resort."
    await verify_with_sources(
        evaluator,
        claim=parking_available_claim,
        node=parking_available_node,
        sources=parking_sources,
        additional_instruction="Verify the presence of on-site self-parking and/or valet parking. If only offsite parking is mentioned, mark unsupported.",
    )

    parking_pricing_node = evaluator.add_leaf(
        id="Parking_Pricing",
        desc="Pricing information for parking services is provided",
        parent=parking_group,
        critical=False
    )
    parking_price_claim = "The page provides pricing for on-site parking (self or valet)."
    await verify_with_sources(
        evaluator,
        claim=parking_price_claim,
        node=parking_pricing_node,
        sources=parking_sources,
        additional_instruction=(
            "Look for dollar amounts or pricing language tied to parking (e.g., '$XX per night'). "
            "If no pricing is shown, mark unsupported."
        ),
    )

    # 8) Kids club (Non-critical group)
    kids_group = evaluator.add_parallel(
        id="Kids_Club",
        desc="The resort has a supervised kids club or children's activity program",
        parent=root,
        critical=False
    )

    kids_available_node = evaluator.add_leaf(
        id="Kids_Club_Available",
        desc="A kids club or supervised children's program is available",
        parent=kids_group,
        critical=False
    )
    kids_sources = gather_sources(extracted, ["kids_club_urls"])
    kids_available_claim = "The resort offers a supervised kids' club or children's activity program on property."
    await verify_with_sources(
        evaluator,
        claim=kids_available_claim,
        node=kids_available_node,
        sources=kids_sources,
        additional_instruction="Look for terms like 'Kids Club', 'children's program', 'Camp', with supervised activities.",
    )

    kids_age_node = evaluator.add_leaf(
        id="Kids_Club_Age_Range",
        desc="The age range for kids club participants is specified",
        parent=kids_group,
        critical=False
    )
    kids_age_text = extracted.kids_club_age_range
    if kids_age_text and kids_age_text.strip():
        kids_age_claim = f"The kids' club specifies an age range for participants: {kids_age_text}."
    else:
        kids_age_claim = "The kids' club page specifies an age range for participants."
    await verify_with_sources(
        evaluator,
        claim=kids_age_claim,
        node=kids_age_node,
        sources=kids_sources,
        additional_instruction="Verify that the program lists eligible ages (e.g., 4–12). If not provided, mark unsupported.",
    )

    # 9) Pool facilities (Non-critical)
    pool_node = evaluator.add_leaf(
        id="Pool_Facilities",
        desc="The resort has at least one swimming pool on property",
        parent=root,
        critical=False
    )
    pool_sources = gather_sources(extracted, ["pool_urls"])
    pool_claim = "There is at least one swimming pool on the resort property (indoor or outdoor)."
    await verify_with_sources(
        evaluator,
        claim=pool_claim,
        node=pool_node,
        sources=pool_sources,
        additional_instruction="Look for 'pool' facilities. If only nearby/partner pools are referenced (off-property), mark unsupported.",
    )

    # 10) On-site dining (Non-critical group)
    dining_group = evaluator.add_parallel(
        id="On_Site_Dining",
        desc="The resort has on-site restaurant options",
        parent=root,
        critical=False
    )

    dining_available_node = evaluator.add_leaf(
        id="Restaurant_Available",
        desc="At least one on-site restaurant is available",
        parent=dining_group,
        critical=False
    )
    dining_sources = gather_sources(extracted, ["dining_urls"])
    dining_available_claim = "At least one on-site restaurant or dining venue is available at the resort."
    await verify_with_sources(
        evaluator,
        claim=dining_available_claim,
        node=dining_available_node,
        sources=dining_sources,
        additional_instruction="Look for 'on-site restaurant', 'dining venue', or similar language.",
    )

    dining_details_node = evaluator.add_leaf(
        id="Dining_Options_Details",
        desc="Information about types of dining options or cuisine styles is provided",
        parent=dining_group,
        critical=False
    )
    if extracted.dining_options_details and extracted.dining_options_details.strip():
        dining_details_claim = f"The page provides information about dining/cuisine options, such as: {extracted.dining_options_details}."
    else:
        dining_details_claim = "The page provides information about types of dining options or cuisine styles available on-site."
    await verify_with_sources(
        evaluator,
        claim=dining_details_claim,
        node=dining_details_node,
        sources=dining_sources,
        additional_instruction="Confirm the presence of cuisine types, dining variety, or specific venue styles.",
    )

    # 11) Spa services (Non-critical)
    spa_node = evaluator.add_leaf(
        id="Spa_Services",
        desc="The resort offers spa services or treatments on property",
        parent=root,
        critical=False
    )
    spa_sources = gather_sources(extracted, ["spa_urls"])
    spa_claim = "Spa services or treatments are available on the resort property."
    await verify_with_sources(
        evaluator,
        claim=spa_claim,
        node=spa_node,
        sources=spa_sources,
        additional_instruction="Verify on-property spa or treatment offerings (massage, facials, etc.).",
    )

    # 12) Fitness center (Non-critical)
    fitness_node = evaluator.add_leaf(
        id="Fitness_Center",
        desc="The resort has a fitness center or gym facility",
        parent=root,
        critical=False
    )
    fitness_sources = gather_sources(extracted, ["fitness_urls"])
    fitness_claim = "A fitness center or gym facility is available on the resort property."
    await verify_with_sources(
        evaluator,
        claim=fitness_claim,
        node=fitness_node,
        sources=fitness_sources,
        additional_instruction="Look for 'fitness center', 'gym', or similar terms.",
    )

    # 13) Cancellation policy (Non-critical)
    cancel_node = evaluator.add_leaf(
        id="Cancellation_Policy",
        desc="The resort's cancellation policy allows free cancellation at least 24 hours before check-in",
        parent=root,
        critical=False
    )
    cancel_sources = gather_sources(extracted, ["cancellation_urls"])
    cancel_claim = "There is a flexible rate or standard policy allowing free cancellation at least 24 hours before check-in."
    await verify_with_sources(
        evaluator,
        claim=cancel_claim,
        node=cancel_node,
        sources=cancel_sources,
        additional_instruction=(
            "Policy should allow free cancellation ≥24 hours before check-in (e.g., 24h, 48h, etc.). "
            "If only non-refundable rates are shown or the policy is stricter than 24h without alternatives, mark unsupported. "
            "Rate-specific exceptions are acceptable if a qualifying flexible rate exists."
        ),
    )

    # 14) Loyalty program (Non-critical)
    loyalty_node = evaluator.add_leaf(
        id="Loyalty_Program",
        desc="Information about hotel loyalty program membership benefits is provided",
        parent=root,
        critical=False
    )
    loyalty_sources = gather_sources(extracted, ["loyalty_urls"])
    if extracted.loyalty_program_text and extracted.loyalty_program_text.strip():
        loyalty_claim = f"The resort participates in a hotel loyalty program with member benefits (e.g., {extracted.loyalty_program_text})."
    else:
        loyalty_claim = "The resort participates in a hotel loyalty program that provides member benefits."
    await verify_with_sources(
        evaluator,
        claim=loyalty_claim,
        node=loyalty_node,
        sources=loyalty_sources,
        additional_instruction="Look for brand programs like Marriott Bonvoy, Hilton Honors, World of Hyatt, IHG One, etc., and member benefits.",
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
    Evaluate an answer for the MLK 2026 family resort task and return a structured result dictionary.
    """
    # Initialize evaluator with root as a parallel node (non-critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find a family-friendly resort in the United States suitable for a family trip during MLK Day weekend 2026",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Record task meta as custom info
    evaluator.add_custom_info(
        info={"mlk_weekend_dates": "Jan 16–19, 2026", "essential_requirements": 5, "desired_features": 9},
        info_type="task_meta",
        info_name="mlk_task_meta"
    )

    # Extraction
    extracted: ResortExtraction = await evaluator.extract(
        prompt=prompt_extract_resort(),
        template_class=ResortExtraction,
        extraction_name="resort_extraction"
    )

    # Verification tree and checks
    await build_and_verify(evaluator, root, extracted)

    # Return standard summary
    return evaluator.get_summary()