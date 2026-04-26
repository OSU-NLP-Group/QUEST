import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "duluth_hotel_requirements"
TASK_DESCRIPTION = (
    "Find a hotel in Duluth, Minnesota that meets ALL of the following requirements: "
    "(1) offers free WiFi throughout the property, "
    "(2) provides free on-site parking for guests, "
    "(3) includes complimentary hot breakfast, "
    "(4) has an indoor swimming pool, "
    "(5) features a fitness center, "
    "(6) has a hot tub or whirlpool, "
    "(7) allows pets with a stated pet policy, "
    "(8) is entirely non-smoking, "
    "(9) has a business center, "
    "(10) offers meeting room facilities, "
    "(11) provides connecting rooms for families or groups, "
    "(12) supports digital key or mobile check-in technology. "
    "Provide the hotel's name, address, and a reference URL to the hotel's official page or listing."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class HotelOverview(BaseModel):
    """Basic hotel info extracted from the answer."""
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_overview() -> str:
    return (
        "Extract the basic information for the first hotel mentioned in the answer. "
        "Only consider the first hotel if multiple are mentioned.\n"
        "Return a JSON object with the following fields:\n"
        "1. name: The hotel's name exactly as written in the answer text.\n"
        "2. address: The hotel's full street address (if provided). If not provided, return null.\n"
        "3. city: The city of the hotel (if provided). If not provided, return null.\n"
        "4. state: The state of the hotel (if provided). If not provided, return null.\n"
        "5. reference_urls: An array of all URLs explicitly included in the answer that correspond "
        "   to the hotel's official website or major booking platform listings. Extract actual URLs "
        "   (including those from markdown links) and provide full URLs with protocol. If none provided, return [].\n"
        "Do not invent any information. Only extract what the answer explicitly presents."
    )


# --------------------------------------------------------------------------- #
# Helper: additional instructions per feature                                 #
# --------------------------------------------------------------------------- #
def feature_additional_instructions() -> Dict[str, str]:
    base_no_source_penalty = (
        "Important: If the answer did not provide any source URLs, treat this claim as unsupported/incorrect.\n"
    )
    return {
        "hotel_location": base_no_source_penalty +
            "Verify that the hotel's page clearly indicates an address in Duluth, Minnesota (accept 'Duluth, MN'). "
            "Use the address or location details on the page to confirm the city/state.",
        "free_wifi": base_no_source_penalty +
            "Verify that the page explicitly mentions 'Free WiFi', 'Complimentary Wi-Fi', or equivalent. "
            "Treat as meeting 'throughout the property' if it indicates in-room and public areas or is generally property-wide. "
            "If restricted to lobby only, do not count.",
        "free_parking": base_no_source_penalty +
            "Verify 'Free parking' or 'Complimentary parking' explicitly, and that it is on-site for guests.",
        "free_breakfast": base_no_source_penalty +
            "Verify 'Complimentary hot breakfast' (accept 'free hot breakfast', 'hot breakfast included'). "
            "A cold continental breakfast alone does not satisfy.",
        "indoor_pool": base_no_source_penalty +
            "Verify that the hotel has an 'indoor pool' (explicitly 'indoor'). An outdoor-only pool does not satisfy.",
        "fitness_center": base_no_source_penalty +
            "Verify a 'fitness center' or 'gym' mentioned as an amenity.",
        "hot_tub": base_no_source_penalty +
            "Verify a 'hot tub', 'whirlpool', 'spa tub', or 'jacuzzi' presence.",
        "pet_friendly": base_no_source_penalty +
            "Verify that pets are allowed AND a written pet policy is present including fees and weight limits "
            "(accept 'no weight limit' as satisfying the weight limit condition). "
            "A generic 'pet-friendly' with no details does not satisfy.",
        "non_smoking": base_no_source_penalty +
            "Verify that the property is entirely non-smoking (accept '100% smoke-free').",
        "business_center": base_no_source_penalty +
            "Verify a 'business center' amenity (accept equivalent phrasing 'business services' if clearly a dedicated facility).",
        "meeting_rooms": base_no_source_penalty +
            "Verify that 'meeting rooms' or 'event space' are available.",
        "connecting_rooms": base_no_source_penalty +
            "Verify 'connecting rooms' or 'adjoining rooms' availability for families or groups.",
        "digital_key": base_no_source_penalty +
            "Verify support for 'Digital Key', 'Mobile Key', 'mobile check-in', 'keyless entry', or app-based check-in technology.",
        "reference_url": (
            "Judge whether at least one provided URL is an official hotel website or a listing on a major booking platform. "
            "Major booking platforms include (examples, not exhaustive): booking.com, expedia.com, hotels.com, tripadvisor.com, "
            "agoda.com, travelocity.com, priceline.com. Official brand sites include (examples): hilton.com, marriott.com, ihg.com, "
            "hyatt.com, choicehotels.com, bestwestern.com, wyndhamhotels.com, radissonhotels.com. "
            "If no URL is provided in the answer, deem this incorrect."
        ),
    }


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
def build_feature_claims(hotel_name: Optional[str]) -> List[Tuple[str, str, str]]:
    """
    Build (leaf_id, leaf_desc, claim) tuples for feature verification leaves.
    hotel_name may be used for clarity in claim but is optional.
    """
    ref_name = f" for the hotel '{hotel_name}'" if hotel_name else ""
    return [
        ("hotel_location", "The hotel is confirmed to be located in Duluth, Minnesota",
         f"The hotel{ref_name} is located in Duluth, Minnesota."),
        ("free_wifi", "Hotel offers free WiFi throughout the property",
         f"The hotel{ref_name} offers free WiFi throughout the property."),
        ("free_parking", "Hotel provides free on-site parking for guests",
         f"The hotel{ref_name} provides free on-site parking for guests."),
        ("free_breakfast", "Hotel includes complimentary hot breakfast",
         f"The hotel{ref_name} includes complimentary hot breakfast."),
        ("indoor_pool", "Hotel has an indoor swimming pool",
         f"The hotel{ref_name} has an indoor swimming pool."),
        ("fitness_center", "Hotel features a fitness center",
         f"The hotel{ref_name} features a fitness center."),
        ("hot_tub", "Hotel has a hot tub or whirlpool",
         f"The hotel{ref_name} has a hot tub or whirlpool."),
        ("pet_friendly", "Hotel allows pets with a stated pet policy including fees and weight limits",
         f"The hotel{ref_name} allows pets and provides a written pet policy including fees and weight limits."),
        ("non_smoking", "Hotel is entirely non-smoking",
         f"The hotel{ref_name} is entirely non-smoking."),
        ("business_center", "Hotel has a business center",
         f"The hotel{ref_name} has a business center."),
        ("meeting_rooms", "Hotel offers meeting room facilities",
         f"The hotel{ref_name} offers meeting room facilities."),
        ("connecting_rooms", "Hotel provides connecting rooms for families or groups",
         f"The hotel{ref_name} provides connecting/adjoining rooms."),
        ("digital_key", "Hotel supports digital key or mobile check-in technology",
         f"The hotel{ref_name} supports digital key or mobile check-in technology."),
        ("reference_url", "A valid reference URL from the hotel's official website or major booking platform is provided",
         "At least one provided URL is either the hotel's official website or a major booking platform listing for this hotel."),
    ]


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
    Evaluate the agent's answer for the Duluth hotel requirements task.
    """
    # 1) Initialize evaluator with a parallel root
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

    # 2) Extract hotel overview and reference URLs from the answer
    hotel: HotelOverview = await evaluator.extract(
        prompt=prompt_extract_hotel_overview(),
        template_class=HotelOverview,
        extraction_name="hotel_overview",
    )

    # 3) Build a critical parent node that aggregates all checks
    #    Since all requirements are mandatory, we keep this node critical.
    critical_parent = evaluator.add_parallel(
        id="hotel_verification",
        desc="A hotel in Duluth, Minnesota that meets all specified requirements",
        parent=root,
        critical=True,
    )

    # 4) Leaf checks: name/address presence (answer completeness)
    evaluator.add_custom_node(
        result=bool(hotel.name and hotel.name.strip()),
        id="hotel_name",
        desc="The hotel's name is provided",
        parent=critical_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(hotel.address and hotel.address.strip()),
        id="hotel_address",
        desc="The hotel's address is provided",
        parent=critical_parent,
        critical=True,
    )

    # 5) Prepare feature verification leaves that require URL grounding
    #    Use batch verification to avoid order-dependent precondition skipping.
    instructions = feature_additional_instructions()
    claims = build_feature_claims(hotel_name=hotel.name)

    # Create leaf nodes for all URL-grounded checks (including location and reference_url)
    leaf_nodes: Dict[str, Any] = {}
    for leaf_id, leaf_desc, _ in claims:
        leaf_nodes[leaf_id] = evaluator.add_leaf(
            id=leaf_id,
            desc=leaf_desc,
            parent=critical_parent,
            critical=True,
        )

    # 6) Compose claims and sources for batch verification
    # Limit number of sources if too many; keep first few to reduce load.
    sources: List[str] = hotel.reference_urls[:6] if hotel.reference_urls else []
    if not sources:
        # For URL-grounded checks, when no source is available, the instruction tells the judge to mark as incorrect.
        pass

    claims_and_sources: List[Tuple[str, List[str] | None, Any, Optional[str]]] = []
    for leaf_id, _, claim in claims:
        add_ins = instructions.get(leaf_id, "None")
        # Use list of URLs if present; otherwise None falls back to simple verification with instruction penalizing lack of sources.
        srcs = sources if sources else None
        node = leaf_nodes[leaf_id]
        claims_and_sources.append((claim, srcs, node, add_ins))

    # 7) Run batch verification for all URL-grounded leaves
    await evaluator.batch_verify(claims_and_sources)

    # 8) Optionally, verify that the name/address match the page (not required by rubric; we keep them as provided-only checks)
    #    The rubric explicitly asks only that these are provided, so we do not add extra verification nodes here.

    # 9) Return standardized evaluation summary
    return evaluator.get_summary()