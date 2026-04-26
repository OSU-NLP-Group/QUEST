import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travel_planning_universal_breeze_yellowstone"
TASK_DESCRIPTION = (
    "You are planning a multi-destination vacation that includes visits to Universal Orlando Resort in Florida and Yellowstone National Park. "
    "You will be flying via Breeze Airways.\n\n"
    "For your Universal Orlando portion, you need to stay at an on-site hotel that provides both of the following benefits to guests:\n"
    "1. Early Park Admission (allowing entry to theme parks up to 1 hour before regular opening)\n"
    "2. Universal Express Unlimited passes\n\n"
    "Please provide the following information:\n\n"
    "A) Name one Universal Orlando on-site hotel that offers both Early Park Admission and Universal Express Unlimited benefits to its guests.\n\n"
    "B) Confirm whether Breeze Airways operates nonstop flights between Orlando, Florida (MCO) and Hartford, Connecticut (BDL).\n\n"
    "C) What is the standard check-in time at Yellowstone National Park lodges?"
)

# Canonical list of Universal Orlando hotels that include Express Unlimited (aka Premier hotels)
SIGNATURE_HOTELS = [
    "Loews Portofino Bay Hotel",
    "Hard Rock Hotel",
    "Loews Royal Pacific Resort",
]

EXPECTED_YELLOWSTONE_CHECKIN = "4:00 PM"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversalHotel(BaseModel):
    hotel_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class BreezeRouteInfo(BaseModel):
    # Normalize to 'yes' or 'no' or 'unknown' for nonstop status mentioned in the answer
    nonstop_status: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class YellowstoneInfo(BaseModel):
    checkin_time: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TripExtraction(BaseModel):
    universal_hotel: Optional[UniversalHotel] = None
    breeze_route: Optional[BreezeRouteInfo] = None
    yellowstone: Optional[YellowstoneInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_info() -> str:
    return (
        "Extract structured information from the answer for three parts of the trip.\n\n"
        "1) Universal Orlando hotel selection:\n"
        "- hotel_name: The single on-site hotel named in the answer that is claimed to offer BOTH Early Park Admission and Universal Express Unlimited benefits. "
        "If multiple hotels are named, select the first one that meets both benefits as stated by the answer; if the answer does not explicitly say a hotel offers both, choose the first on-site hotel named.\n"
        "- urls: All URLs in the answer that relate to Universal Orlando hotels or benefits (e.g., hotel pages, Universal Orlando benefits pages). Return only valid URLs.\n\n"
        "2) Breeze Airways route confirmation:\n"
        "- nonstop_status: Return 'yes' if the answer explicitly claims Breeze operates nonstop flights between Orlando (MCO) and Hartford (BDL), 'no' if the answer explicitly claims they do not operate nonstop, otherwise 'unknown'.\n"
        "- urls: All URLs in the answer that relate to Breeze Airways schedules, route maps, booking pages, or announcements relevant to MCO–BDL.\n\n"
        "3) Yellowstone National Park lodges check-in time:\n"
        "- checkin_time: The standard check-in time stated in the answer for Yellowstone National Park lodges (e.g., '4:00 PM'). If not stated, return null.\n"
        "- urls: All URLs in the answer that relate to Yellowstone National Park lodges or official booking/check-in info.\n\n"
        "Rules:\n"
        "- Extract only what appears in the answer.\n"
        "- For any missing field, return null or an empty list accordingly.\n"
        "- For URLs, include full URLs; ignore malformed links."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.strip().lower()


def _is_signature_hotel(name: Optional[str]) -> bool:
    """
    Determine whether the extracted hotel appears to be one of the three signature/premier hotels
    that include Universal Express Unlimited: Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort.
    Uses case-insensitive substring match to allow minor variants like 'Hard Rock Hotel at Universal Orlando'.
    """
    n = _normalize_text(name)
    if not n:
        return False
    canonical = [h.lower() for h in SIGNATURE_HOTELS]
    return any(c in n for c in canonical)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_universal_hotel_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: TripExtraction,
) -> None:
    """
    Build verification nodes for Universal Orlando hotel selection and benefits.
    """
    # Create critical parallel node for Universal Orlando hotel selection
    uni_node = evaluator.add_parallel(
        id="Universal_Orlando_Hotel_Selection",
        desc="Identify a Universal Orlando on-site hotel that provides both Early Park Admission and Universal Express Unlimited benefits",
        parent=parent_node,
        critical=True,
    )

    hotel_name = extracted.universal_hotel.hotel_name if extracted.universal_hotel else None
    hotel_urls = extracted.universal_hotel.urls if extracted.universal_hotel else []

    # Leaf 1: Must be one of the signature collection (premier) hotels
    evaluator.add_custom_node(
        result=_is_signature_hotel(hotel_name),
        id="Signature_Collection_Hotel",
        desc="Hotel must be one of the three Signature Collection hotels: Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort",
        parent=uni_node,
        critical=True,
    )

    # Leaf 2: Hotel provides Universal Express Unlimited passes
    express_leaf = evaluator.add_leaf(
        id="Express_Unlimited_Benefit",
        desc="Hotel provides Universal Express Unlimited passes to guests",
        parent=uni_node,
        critical=True,
    )
    express_claim = (
        f"The hotel '{hotel_name or 'the selected hotel'}' provides Universal Express Unlimited ride access/passes to its registered guests."
    )
    await evaluator.verify(
        claim=express_claim,
        node=express_leaf,
        sources=hotel_urls if hotel_urls else None,
        additional_instruction=(
            "Verify on the hotel's official page or Universal Orlando website whether Universal Express Unlimited is included for hotel guests. "
            "Accept synonymous phrasing such as 'Free Universal Express Unlimited' or 'Unlimited Express pass included'."
        ),
    )

    # Leaf 3: Hotel provides Early Park Admission (up to 1 hour before)
    early_leaf = evaluator.add_leaf(
        id="Early_Park_Admission_Benefit",
        desc="Hotel provides Early Park Admission allowing entry up to 1 hour before regular park opening",
        parent=uni_node,
        critical=True,
    )
    early_claim = (
        f"The hotel '{hotel_name or 'the selected hotel'}' offers Early Park Admission to Universal Orlando parks (allowing entry up to 1 hour before regular opening)."
    )
    await evaluator.verify(
        claim=early_claim,
        node=early_leaf,
        sources=hotel_urls if hotel_urls else None,
        additional_instruction=(
            "Check the hotel's benefits or Universal Orlando official benefits pages for 'Early Park Admission'. "
            "Minor variations like 'early admission' or 'early entry' are acceptable if clearly equivalent."
        ),
    )


async def build_breeze_route_check(
    evaluator: Evaluator,
    parent_node,
    extracted: TripExtraction,
) -> None:
    """
    Build verification node for Breeze Airways route nonstop confirmation between MCO and BDL.
    """
    breeze_leaf = evaluator.add_leaf(
        id="Breeze_Airways_Route",
        desc="Confirm whether Breeze Airways offers nonstop service between Orlando (MCO) and Hartford, Connecticut (BDL) (must confirm nonstop service per constraints)",
        parent=parent_node,
        critical=True,
    )

    status = _normalize_text(extracted.breeze_route.nonstop_status if extracted.breeze_route else None)
    route_urls = extracted.breeze_route.urls if extracted.breeze_route else []

    if status == "no":
        claim = "Breeze Airways does not operate nonstop flights between Orlando (MCO) and Hartford (BDL)."
    else:
        # Default to positive claim if 'yes' or unknown; the verification will check support via provided sources
        claim = "Breeze Airways operates nonstop flights between Orlando (MCO) and Hartford (BDL)."

    await evaluator.verify(
        claim=claim,
        node=breeze_leaf,
        sources=route_urls if route_urls else None,
        additional_instruction=(
            "Use Breeze Airways official route map, booking pages, schedules, or credible sources to determine if there is nonstop service. "
            "Interpret 'nonstop' as direct service without intermediate stops."
        ),
    )


async def build_yellowstone_checkin_check(
    evaluator: Evaluator,
    parent_node,
    extracted: TripExtraction,
) -> None:
    """
    Build verification node for Yellowstone National Park lodges standard check-in time.
    """
    yellow_leaf = evaluator.add_leaf(
        id="Yellowstone_Check_In_Time",
        desc="Provide the standard check-in time at Yellowstone National Park lodges (4:00 PM per constraints)",
        parent=parent_node,
        critical=True,
    )

    yz_urls = extracted.yellowstone.urls if extracted.yellowstone else []
    # We verify the ground-truth statement with available sources; minor format variants are acceptable (e.g., 4 PM)
    claim = "The standard check-in time at Yellowstone National Park lodges is 4:00 PM."
    await evaluator.verify(
        claim=claim,
        node=yellow_leaf,
        sources=yz_urls if yz_urls else None,
        additional_instruction=(
            "Check official Yellowstone National Park Lodges (Xanterra) information pages for standard check-in time; allow minor format variants like '4 PM'."
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
    Evaluate the travel planning answer for Universal Orlando hotel benefits, Breeze Airways MCO–BDL nonstop confirmation,
    and Yellowstone lodges standard check-in time.
    """
    # Initialize evaluator with a parallel root by default
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_info(),
        template_class=TripExtraction,
        extraction_name="trip_extraction",
    )

    # Create critical parent node for the overall travel planning task
    travel_node = evaluator.add_parallel(
        id="Travel_Planning_Task",
        desc="Complete travel planning information for a multi-destination trip",
        parent=root,
        critical=True,
    )

    # Add ground truth information for reference
    evaluator.add_ground_truth({
        "signature_hotels": SIGNATURE_HOTELS,
        "expected_yellowstone_checkin": EXPECTED_YELLOWSTONE_CHECKIN,
        "route_pair": "MCO-BDL",
    }, gt_type="ground_truth")

    # Build and run Universal Orlando hotel selection checks
    await build_universal_hotel_checks(evaluator, travel_node, extracted)

    # Breeze Airways route check
    await build_breeze_route_check(evaluator, travel_node, extracted)

    # Yellowstone check-in time check
    await build_yellowstone_checkin_check(evaluator, travel_node, extracted)

    # Return the final structured summary
    return evaluator.get_summary()