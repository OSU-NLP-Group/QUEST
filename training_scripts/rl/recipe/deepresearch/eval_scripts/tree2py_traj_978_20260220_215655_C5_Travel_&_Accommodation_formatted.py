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
TASK_ID = "whistler_mlk_2026_hotels"
TASK_DESCRIPTION = (
    "I am planning a family ski vacation to Whistler, British Columbia, for the Martin Luther King Jr. Day long "
    "weekend in 2026 (Saturday, January 17 to Monday, January 19 - staying 3 nights). My family consists of 2 adults "
    "and 2 children, and we will be bringing our dog.\n\n"
    "Please identify three different hotels in Whistler Village that meet all of the following requirements:\n\n"
    "1. The hotel must be located in Whistler Village with convenient ski access (either within walking distance to ski lifts or offering ski-in/ski-out access).\n"
    "2. The hotel must be pet-friendly and allow dogs, with a pet fee structure where the total pet fee for our 3-night stay does not exceed $150 CAD.\n"
    "3. The hotel must offer room configurations suitable for our family of 4 (such as connecting rooms, family suites, or rooms that can accommodate 2 adults and 2 children).\n"
    "4. The hotel must include complimentary breakfast for all guests.\n"
    "5. The hotel must provide on-site parking (either free or paid).\n"
    "6. The hotel must offer a package deal that bundles accommodation with Whistler Blackcomb lift tickets.\n\n"
    "For each of the three hotels, please provide:\n"
    "- The hotel name\n"
    "- The specific pet fee policy and total cost for 3 nights\n"
    "- The type of room configuration available for a family of 4\n"
    "- Confirmation that breakfast is included\n"
    "- Parking details (free or paid, and cost if applicable)\n"
    "- Information about the ski lift ticket package availability\n"
    "- A reference URL to the hotel's official website or booking page"
)

MLK_2026_NIGHTS = 3
PET_FEE_LIMIT_CAD = 150

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    reference_url: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)

    location_desc: Optional[str] = None
    ski_access_desc: Optional[str] = None

    pet_policy_desc: Optional[str] = None
    pet_fee_total_cad: Optional[str] = None  # Keep as string for flexibility

    room_configuration_desc: Optional[str] = None

    breakfast_details: Optional[str] = None
    parking_details: Optional[str] = None

    ski_package_details: Optional[str] = None


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return (
        "Extract all hotels mentioned in the answer (we will later consider only the first three). For each hotel, "
        "return a JSON object with the following fields:\n"
        "1. name: The hotel name as written in the answer.\n"
        "2. reference_url: A URL to the hotel's official website or booking page (explicitly provided in the answer). If missing, set to null.\n"
        "3. support_urls: An array of any additional URLs cited in the answer that contain relevant policy or package details "
        "(e.g., pet policy page, breakfast policy page, parking info page, ski/lift package page, whistler.com package page). "
        "Include only URLs explicitly present in the answer.\n"
        "4. location_desc: Extract the description supporting that the hotel is in Whistler Village (e.g., \"in Whistler Village\"). If missing, set to null.\n"
        "5. ski_access_desc: Extract the description of ski access (e.g., \"ski-in/ski-out\" or \"walking distance to lifts\"). If missing, set to null.\n"
        "6. pet_policy_desc: Extract the pet policy text as provided (e.g., fee per night/per stay, dog allowed). If missing, set to null.\n"
        "7. pet_fee_total_cad: If the answer provides enough information to compute the total pet fees for a 3-night stay for ONE dog, "
        "compute and return the total (e.g., \"$150 CAD\" or \"150\"). If computation is not provided or unclear, set to null.\n"
        "8. room_configuration_desc: Extract the room configuration suitable for a family of 4 (e.g., \"family suite\", \"connecting rooms\", \"2 queen beds, sleeps 4\"). If missing, set to null.\n"
        "9. breakfast_details: Extract confirmation/details that breakfast is included/complimentary. If missing, set to null.\n"
        "10. parking_details: Extract details that on-site parking is available (free or paid; include cost if provided). If missing, set to null.\n"
        "11. ski_package_details: Extract information that the hotel offers a package bundling accommodation with Whistler Blackcomb lift tickets "
        "(e.g., \"ski & stay\", \"lodging + lift tickets\"), including any package name. If missing, set to null.\n\n"
        "Return a JSON with a single field 'hotels' which is an array of such hotel objects. "
        "If any field is not present in the answer, use null or empty array according to the field type. "
        "Do NOT invent any information; only extract what is explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_sources_for_hotel(h: HotelItem) -> List[str]:
    """Gather unique sources for verification: reference_url plus support_urls."""
    urls: List[str] = []
    if h.reference_url:
        urls.append(h.reference_url)
    for u in h.support_urls:
        if u and u not in urls:
            urls.append(u)
    return urls


def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n + 1}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_one_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    idx: int,
) -> None:
    """
    Build verification nodes and run checks for one hotel.
    Each leaf node represents a single verification step as specified in the rubric.
    """
    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{idx + 1}",
        desc=f"{ordinal(idx)} qualifying hotel meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # Existence: Name provided (critical)
    name_provided = evaluator.add_custom_node(
        result=bool(hotel.name and hotel.name.strip()),
        id=f"Hotel_{idx + 1}_Name_Provided",
        desc=f"The name of Hotel {idx + 1} is explicitly provided",
        parent=hotel_node,
        critical=True,
    )

    # Existence: Reference URL provided (critical)
    ref_url_provided = evaluator.add_custom_node(
        result=bool(hotel.reference_url and hotel.reference_url.strip()),
        id=f"Hotel_{idx + 1}_Reference_URL",
        desc=f"A reference URL to Hotel {idx + 1}'s official website or booking page is provided",
        parent=hotel_node,
        critical=True,
    )

    # Location & Ski Access (critical)
    loc_ski_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx + 1}_Location_And_Ski_Access",
        desc=f"Hotel {idx + 1} is located in Whistler Village and provides either walking distance access to ski lifts or ski-in/ski-out access",
        parent=hotel_node,
        critical=True,
    )
    loc_ski_claim = (
        f"Hotel '{hotel.name or ''}' is located in Whistler Village and provides convenient ski access "
        f"(either walking distance to ski lifts or ski-in/ski-out)."
    )
    await evaluator.verify(
        claim=loc_ski_claim,
        node=loc_ski_leaf,
        sources=get_sources_for_hotel(hotel) or None,
        additional_instruction=(
            "Confirm the hotel is in Whistler Village (not Creekside or Upper Village unless explicitly within Whistler Village) "
            "and that it provides convenient ski access: either explicitly 'ski-in/ski-out' or described as within walking distance "
            "to Whistler or Blackcomb lifts/gondolas. Use any provided official or booking pages."
        ),
    )

    # Pet Policy within CAD 150 for 3 nights (critical)
    pet_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx + 1}_Pet_Policy",
        desc=f"Hotel {idx + 1} is pet-friendly, allows dogs, and has a pet fee structure where the total cost for 3 nights does not exceed $150 CAD, with specific pet fee policy and total cost provided",
        parent=hotel_node,
        critical=True,
    )
    pet_claim = (
        f"Hotel '{hotel.name or ''}' allows dogs and the total pet fees for ONE dog over a 3-night stay "
        f"do not exceed $150 CAD. Policy details: {hotel.pet_policy_desc or 'N/A'}; "
        f"Computed/quoted total for 3 nights: {hotel.pet_fee_total_cad or 'N/A'}."
    )
    await evaluator.verify(
        claim=pet_claim,
        node=pet_leaf,
        sources=get_sources_for_hotel(hotel) or None,
        additional_instruction=(
            "Verify the hotel is pet-friendly (dogs allowed). If the policy lists a per-night or per-stay fee, compute the total for ONE dog over 3 nights. "
            "Accept totals <= 150 CAD; if the fee appears per pet and the user has one dog, use one pet in the computation. "
            "If fees are in USD or another currency, consider whether the page also provides CAD; otherwise, treat as not clearly within CAD 150."
        ),
    )

    # Family accommodation (critical)
    family_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx + 1}_Family_Accommodation",
        desc=f"Hotel {idx + 1} offers room configurations (connecting rooms, family suites, or rooms) that can accommodate 4 people (2 adults and 2 children), with the specific room type provided",
        parent=hotel_node,
        critical=True,
    )
    family_claim = (
        f"Hotel '{hotel.name or ''}' offers room configurations suitable for a family of 4 "
        f"(2 adults and 2 children), such as connecting rooms, family suites, or rooms with capacity 4. Example: {hotel.room_configuration_desc or 'N/A'}."
    )
    await evaluator.verify(
        claim=family_claim,
        node=family_leaf,
        sources=get_sources_for_hotel(hotel) or None,
        additional_instruction=(
            "Check occupancy/capacity details or explicit mentions of 'family suite' or 'connecting rooms'. "
            "A room with two queen beds and occupancy 4 qualifies. The evidence must indicate the configuration can accommodate 4 guests."
        ),
    )

    # Breakfast inclusion (critical)
    breakfast_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx + 1}_Breakfast_Inclusion",
        desc=f"Hotel {idx + 1} includes complimentary breakfast for all guests, with confirmation provided",
        parent=hotel_node,
        critical=True,
    )
    breakfast_claim = (
        f"Hotel '{hotel.name or ''}' includes complimentary breakfast for all guests. Details: {hotel.breakfast_details or 'N/A'}."
    )
    await evaluator.verify(
        claim=breakfast_claim,
        node=breakfast_leaf,
        sources=get_sources_for_hotel(hotel) or None,
        additional_instruction=(
            "Confirm that breakfast is included/complimentary in the room/package for guests, not merely available for purchase. "
            "Look for phrases such as 'free breakfast', 'complimentary breakfast', or inclusion in a package rate."
        ),
    )

    # Parking (critical)
    parking_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx + 1}_Parking",
        desc=f"Hotel {idx + 1} offers on-site parking (either free or paid), with parking details provided",
        parent=hotel_node,
        critical=True,
    )
    parking_claim = (
        f"Hotel '{hotel.name or ''}' provides on-site parking (free or paid). Details: {hotel.parking_details or 'N/A'}."
    )
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=get_sources_for_hotel(hotel) or None,
        additional_instruction=(
            "Verify that parking is on-site at the hotel (either complimentary or paid). Include confirmation if costs are listed, "
            "but the primary requirement is that on-site parking is available."
        ),
    )

    # Ski package (critical)
    package_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx + 1}_Ski_Package",
        desc=f"Hotel {idx + 1} offers a package that bundles accommodation with Whistler Blackcomb lift tickets, with package information provided",
        parent=hotel_node,
        critical=True,
    )
    package_claim = (
        f"Hotel '{hotel.name or ''}' offers a package bundling accommodation with Whistler Blackcomb lift tickets "
        f"(e.g., stay-and-ski, lodging + lift tickets). Details: {hotel.ski_package_details or 'N/A'}."
    )
    await evaluator.verify(
        claim=package_claim,
        node=package_leaf,
        sources=get_sources_for_hotel(hotel) or None,
        additional_instruction=(
            "Accept official hotel pages or Whistler.com/official booking pages that clearly bundle lodging with Whistler Blackcomb lift tickets. "
            "The page should explicitly state lift tickets included with accommodation or a similar 'ski & stay' offer."
        ),
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
    Evaluate the answer for the Whistler MLK 2026 hotels task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall hotels are independent
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

    # Add a top-level aggregation node representing the overall requirement
    top_node = evaluator.add_parallel(
        id="Find_Three_Qualifying_Whistler_Hotels",
        desc="Identify three distinct hotels in Whistler Village that meet all specified criteria for a family vacation with a pet during MLK weekend 2026 (January 17-19, 2026)",
        parent=root,
        critical=False  # Set non-critical to allow partial credit; critical parent cannot have non-critical children
    )

    # Extract structured hotel info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Prepare the first three hotels (pad if fewer)
    hotels: List[HotelItem] = list(extracted.hotels[:3])
    while len(hotels) < 3:
        hotels.append(HotelItem())

    # Verify each of the three hotels
    for i in range(3):
        await verify_one_hotel(evaluator, top_node, hotels[i], i)

    # Add custom info for context (optional)
    evaluator.add_custom_info(
        {
            "weekend_nights": MLK_2026_NIGHTS,
            "pet_fee_limit_cad": PET_FEE_LIMIT_CAD,
            "notes": "Assumed ONE dog for pet fee calculation based on user statement."
        },
        info_type="task_parameters"
    )

    return evaluator.get_summary()