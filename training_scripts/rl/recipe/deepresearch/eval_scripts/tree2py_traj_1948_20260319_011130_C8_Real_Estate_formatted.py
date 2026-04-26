import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "avelo_hotels_2026"
TASK_DESCRIPTION = """
For each of Avelo Airlines' four current operating bases as of March 2026 (Tweed-New Haven Airport in Connecticut, Wilmington Airport in Delaware, Lakeland International Airport in Florida, and Concord-Padgett Regional Airport in North Carolina), identify one existing hotel property that meets the following requirements: (1) located within 5 miles of the airport, (2) has at least 80 guest rooms, (3) provides on-site parking for guests. For each hotel, provide the hotel name, the number of guest rooms, and a reference URL for verification.
"""

AIRPORTS = {
    "hvn": {
        "full_name": "Tweed-New Haven Airport",
        "code": "HVN",
        "node_id": "Hotel_Near_Tweed_New_Haven_Airport",
        "node_desc": "Hotel property near Tweed-New Haven Airport (HVN) in Connecticut",
        "leaves": {
            "ref_id": "HVN_Reference_URL",
            "ref_desc": "A valid reference URL is provided to verify hotel information",
            "loc_id": "HVN_Location_Within_5_Miles",
            "loc_desc": "Hotel is located within 5 miles of Tweed-New Haven Airport",
            "rooms_id": "HVN_Minimum_80_Rooms",
            "rooms_desc": "Hotel has at least 80 guest rooms",
            "park_id": "HVN_On_Site_Parking",
            "park_desc": "Hotel provides on-site parking for guests",
        },
    },
    "ilg": {
        "full_name": "Wilmington Airport",
        "code": "ILG",
        "node_id": "Hotel_Near_Wilmington_Airport",
        "node_desc": "Hotel property near Wilmington Airport (ILG) in Delaware",
        "leaves": {
            "ref_id": "ILG_Reference_URL",
            "ref_desc": "A valid reference URL is provided to verify hotel information",
            "loc_id": "ILG_Location_Within_5_Miles",
            "loc_desc": "Hotel is located within 5 miles of Wilmington Airport",
            "rooms_id": "ILG_Minimum_80_Rooms",
            "rooms_desc": "Hotel has at least 80 guest rooms",
            "park_id": "ILG_On_Site_Parking",
            "park_desc": "Hotel provides on-site parking for guests",
        },
    },
    "lal": {
        "full_name": "Lakeland International Airport",
        "code": "LAL",
        "node_id": "Hotel_Near_Lakeland_International_Airport",
        "node_desc": "Hotel property near Lakeland International Airport (LAL) in Florida",
        "leaves": {
            "ref_id": "LAL_Reference_URL",
            "ref_desc": "A valid reference URL is provided to verify hotel information",
            "loc_id": "LAL_Location_Within_5_Miles",
            "loc_desc": "Hotel is located within 5 miles of Lakeland International Airport",
            "rooms_id": "LAL_Minimum_80_Rooms",
            "rooms_desc": "Hotel has at least 80 guest rooms",
            "park_id": "LAL_On_Site_Parking",
            "park_desc": "Hotel provides on-site parking for guests",
        },
    },
    "usa": {
        "full_name": "Concord-Padgett Regional Airport",
        "code": "USA",
        "node_id": "Hotel_Near_Concord_Padgett_Regional_Airport",
        "node_desc": "Hotel property near Concord-Padgett Regional Airport (USA) in North Carolina",
        "leaves": {
            "ref_id": "USA_Reference_URL",
            "ref_desc": "A valid reference URL is provided to verify hotel information",
            "loc_id": "USA_Location_Within_5_Miles",
            "loc_desc": "Hotel is located within 5 miles of Concord-Padgett Regional Airport",
            "rooms_id": "USA_Minimum_80_Rooms",
            "rooms_desc": "Hotel has at least 80 guest rooms",
            "park_id": "USA_On_Site_Parking",
            "park_desc": "Hotel provides on-site parking for guests",
        },
    },
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelPick(BaseModel):
    hotel_name: Optional[str] = None
    rooms_claim: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hvn: Optional[HotelPick] = None
    ilg: Optional[HotelPick] = None
    lal: Optional[HotelPick] = None
    usa: Optional[HotelPick] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract the hotel selections the answer provides for each of the four Avelo Airlines operating bases.
    You must strictly extract only what is explicitly present in the answer text; do not invent details.

    For each base, pick the first hotel mentioned for that base (if any) and extract:
    - hotel_name: The hotel's name exactly as written in the answer.
    - rooms_claim: The number of guest rooms as written in the answer (e.g., "120 rooms", "128 guestrooms", or a numeric string). If not stated, return null.
    - reference_urls: All URLs that the answer cites for that specific hotel (official site, brand page, booking site, or news page). Extract actual URLs from plain text or from markdown links.

    The bases are:
    - hvn: Tweed-New Haven Airport (HVN) in Connecticut
    - ilg: Wilmington Airport (ILG) in Delaware
    - lal: Lakeland International Airport (LAL) in Florida
    - usa: Concord-Padgett Regional Airport (USA) in North Carolina

    If the answer does not provide a hotel for a base, return null for that base.
    If the answer provides multiple hotels for a base, choose the first one mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def any_valid_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(is_valid_url(u) for u in urls)


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_base_hotel(
    evaluator: Evaluator,
    root,
    base_key: str,
    base_info: Dict[str, Any],
    hotel_pick: Optional[HotelPick],
) -> None:
    """
    Build the verification subtree for one base and run verifications according to the rubric.
    """
    airport_full = base_info["full_name"]
    code = base_info["code"]
    node_meta = base_info["leaves"]

    # Add base node (parallel, non-critical) under root
    base_node = evaluator.add_parallel(
        id=base_info["node_id"],
        desc=base_info["node_desc"],
        parent=root,
        critical=False,
    )

    # Prepare extracted data
    hotel_name = (hotel_pick.hotel_name if hotel_pick else "") or ""
    ref_urls = (hotel_pick.reference_urls if hotel_pick else []) or []

    # 1) Reference URL presence (critical)
    ref_node = evaluator.add_custom_node(
        result=any_valid_url(ref_urls),
        id=node_meta["ref_id"],
        desc=node_meta["ref_desc"],
        parent=base_node,
        critical=True,
    )

    # 2) Location within 5 miles (critical)
    loc_node = evaluator.add_leaf(
        id=node_meta["loc_id"],
        desc=node_meta["loc_desc"],
        parent=base_node,
        critical=True,
    )
    loc_claim = (
        f"The hotel '{hotel_name}' is located within 5 miles of {airport_full} ({code})."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=ref_urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Verify on the cited page(s) that the hotel is within 5 miles of the specified airport. "
            "Accept explicit statements like 'X miles from [airport]' with X ≤ 5, or a clearly labeled 'Airport distance: X mi/km' with X ≤ 5. "
            "If the page only shows an address without an explicit airport distance, or gives distances to other landmarks (e.g., downtown), "
            "consider the claim not supported. Prefer explicit distance mentions tied to the named airport."
        ),
    )

    # 3) Minimum 80 rooms (critical)
    rooms_node = evaluator.add_leaf(
        id=node_meta["rooms_id"],
        desc=node_meta["rooms_desc"],
        parent=base_node,
        critical=True,
    )
    rooms_claim = f"The hotel '{hotel_name}' has at least 80 guest rooms."
    await evaluator.verify(
        claim=rooms_claim,
        node=rooms_node,
        sources=ref_urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Confirm the total number of guestrooms (rooms and/or suites) is ≥ 80 on the cited page(s). "
            "Accept phrasings such as 'X rooms', 'X guestrooms', 'X rooms and Y suites' (use the total), or facts like 'an 120-room hotel'. "
            "If the total room count is unclear or not stated, consider the claim not supported."
        ),
    )

    # 4) On-site parking (critical)
    park_node = evaluator.add_leaf(
        id=node_meta["park_id"],
        desc=node_meta["park_desc"],
        parent=base_node,
        critical=True,
    )
    park_claim = f"The hotel '{hotel_name}' provides on-site parking for guests."
    await evaluator.verify(
        claim=park_claim,
        node=park_node,
        sources=ref_urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Look for explicit mentions of parking offered on the property: 'on-site parking', 'free parking', "
            "'self-parking', or 'valet parking on site'. These count as on-site parking. "
            "Do NOT accept 'street parking', 'nearby public parking', or vague 'parking nearby' as on site."
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
    Evaluate an answer for the Avelo bases hotel task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Each base evaluated independently
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

    # Extract hotel selections from the answer
    extracted_hotels = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_near_avelo_bases",
    )

    # Add ground truth context (airports and codes)
    evaluator.add_ground_truth(
        {
            "bases": [
                {"airport": AIRPORTS["hvn"]["full_name"], "code": AIRPORTS["hvn"]["code"]},
                {"airport": AIRPORTS["ilg"]["full_name"], "code": AIRPORTS["ilg"]["code"]},
                {"airport": AIRPORTS["lal"]["full_name"], "code": AIRPORTS["lal"]["code"]},
                {"airport": AIRPORTS["usa"]["full_name"], "code": AIRPORTS["usa"]["code"]},
            ],
            "requirement_summary": [
                "Hotel within 5 miles of airport",
                "At least 80 guest rooms",
                "On-site parking provided",
                "Provide a verification URL"
            ],
        },
        gt_type="airports_and_requirements",
    )

    # Build verification tree for each base
    await verify_base_hotel(
        evaluator=evaluator,
        root=root,
        base_key="hvn",
        base_info=AIRPORTS["hvn"],
        hotel_pick=extracted_hotels.hvn if extracted_hotels else None,
    )
    await verify_base_hotel(
        evaluator=evaluator,
        root=root,
        base_key="ilg",
        base_info=AIRPORTS["ilg"],
        hotel_pick=extracted_hotels.ilg if extracted_hotels else None,
    )
    await verify_base_hotel(
        evaluator=evaluator,
        root=root,
        base_key="lal",
        base_info=AIRPORTS["lal"],
        hotel_pick=extracted_hotels.lal if extracted_hotels else None,
    )
    await verify_base_hotel(
        evaluator=evaluator,
        root=root,
        base_key="usa",
        base_info=AIRPORTS["usa"],
        hotel_pick=extracted_hotels.usa if extracted_hotels else None,
    )

    # Return structured summary
    return evaluator.get_summary()