import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "disney_springs_travel_planning"
TASK_DESCRIPTION = (
    "I'm planning a trip to Orlando, Florida, and want to stay near Disney Springs. "
    "Please identify three hotels from the official Disney Springs Resort Area hotels list and provide a reference URL for each. "
    "Additionally, I need information about the most affordable public transportation option from Orlando International Airport (MCO) to Disney Springs, "
    "including the service name/route number, one-way fare, daily operating hours, and how frequently the service runs. "
    "Please include a reference URL for the transportation information as well."
)

OFFICIAL_LIST_INDEX_URL = "https://disneyspringshotels.com/hotels/"  # Official Disney Springs Resort Area Hotels list index


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    reference_url: Optional[str] = None


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


class TransportationExtraction(BaseModel):
    service_name: Optional[str] = None
    route_number: Optional[str] = None
    one_way_fare: Optional[str] = None
    operating_hours: Optional[str] = None
    frequency: Optional[str] = None
    reference_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract up to three hotel entries mentioned in the answer that the user proposes near Disney Springs.
    For each hotel, return:
    - name: The complete official hotel name exactly as shown in the answer
    - reference_url: A single URL provided in the answer that references this hotel (any supporting page the answer cites; if multiple URLs are given, pick the most relevant one)
    Rules:
    - If the answer lists more than three hotels, extract them all, we will consider only the first three later.
    - If a field is missing, set it to null.
    - Do NOT invent URLs; only use URLs explicitly present in the answer text (including markdown links).
    """


def prompt_extract_transportation() -> str:
    return """
    Extract the most affordable public transportation option from Orlando International Airport (MCO) to the Disney Springs area as presented in the answer.
    Return the following fields:
    - service_name: The service name (e.g., LYNX Bus)
    - route_number: The route number/code if provided (e.g., 111), or null if not specified
    - one_way_fare: The one-way fare cost exactly as stated (keep currency symbols or words as in the answer)
    - operating_hours: The daily operating hours (start time to end time) as a single string (e.g., "5:00 AM – 11:00 PM")
    - frequency: How often the service runs (e.g., "every 30 minutes")
    - reference_url: The URL cited in the answer that supports these transportation details
    Rules:
    - Extract only what is explicitly present in the answer; if a field is missing, set it to null.
    - Do NOT invent URLs; only use URLs explicitly present in the answer text (including markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper verification builders                                                 #
# --------------------------------------------------------------------------- #
async def verify_single_hotel(
    evaluator: Evaluator,
    hotels_parent_node,
    hotel: HotelItem,
    index: int,
) -> None:
    """
    Build verification nodes for one hotel and execute checks.
    All children of Hotel_Selection are critical to comply with rubric and framework consistency.
    """
    idx_to_name = {0: "First_Hotel", 1: "Second_Hotel", 2: "Third_Hotel"}
    idx_to_desc = {
        0: "First hotel selection with verification and details",
        1: "Second hotel selection with verification and details",
        2: "Third hotel selection with verification and details",
    }
    hotel_node = evaluator.add_parallel(
        id=idx_to_name.get(index, f"Hotel_{index+1}"),
        desc=idx_to_desc.get(index, f"Hotel #{index+1} selection with verification and details"),
        parent=hotels_parent_node,
        critical=True,  # Parent is critical; children must be critical
    )

    # 1) Hotel_Name_Provided_i (existence/format check)
    name_exists = hotel.name is not None and bool(str(hotel.name).strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"Hotel_Name_Provided_{index+1}",
        desc="Complete official hotel name is provided",
        parent=hotel_node,
        critical=True,
    )

    # 2) Reference_URL_Provided_i (existence/validity check)
    url_val = hotel.reference_url or ""
    url_valid = url_val.startswith("http://") or url_val.startswith("https://")
    evaluator.add_custom_node(
        result=url_valid,
        id=f"Reference_URL_Provided_{index+1}",
        desc="Valid reference URL supporting the hotel information is provided",
        parent=hotel_node,
        critical=True,
    )

    # 3) Official_List_Membership_i (grounded verification against official list)
    membership_leaf = evaluator.add_leaf(
        id=f"Official_List_Membership_{index+1}",
        desc="Hotel is one of the 7 official Disney Springs Resort Area hotels listed on disneyspringshotels.com",
        parent=hotel_node,
        critical=True,
    )
    claim = (
        f"The hotel named '{hotel.name or ''}' is listed among the official Disney Springs Resort Area hotels "
        f"on disneyspringshotels.com."
    )
    # Build sources: use provided hotel URL (if any) plus the official list index page
    sources: List[str] = []
    if url_valid:
        sources.append(url_val)
    # Always include the official index list as authoritative evidence for membership
    sources.append(OFFICIAL_LIST_INDEX_URL)

    await evaluator.verify(
        claim=claim,
        node=membership_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm on disneyspringshotels.com whether the hotel's name appears among the official Disney Springs Resort Area hotels. "
            "It may appear on the main list page or on the hotel's dedicated page on that domain. "
            "Allow minor naming variations (e.g., punctuation, 'Orlando' vs. 'Lake Buena Vista')."
        ),
    )


async def verify_transportation(
    evaluator: Evaluator,
    root_parent_node,
    trans: TransportationExtraction,
) -> None:
    """
    Build verification nodes for the transportation info and execute checks.
    All leaves critical per rubric.
    """
    trans_node = evaluator.add_parallel(
        id="Public_Transportation_Information",
        desc="Complete information about the most affordable public transportation option from Orlando International Airport to Disney Springs",
        parent=root_parent_node,
        critical=True,  # Parent is critical; leaves must be critical
    )

    # Reference URL existence/validity gate
    t_url = trans.reference_url or ""
    t_url_valid = t_url.startswith("http://") or t_url.startswith("https://")
    evaluator.add_custom_node(
        result=t_url_valid,
        id="Transportation_Reference_URL",
        desc="Valid reference URL supporting the transportation information is provided",
        parent=trans_node,
        critical=True,
    )

    # Service Identification
    svc_leaf = evaluator.add_leaf(
        id="Service_Identification",
        desc="The public transportation service name/route number is correctly identified",
        parent=trans_node,
        critical=True,
    )
    id_parts = []
    if trans.service_name:
        id_parts.append(str(trans.service_name).strip())
    if trans.route_number:
        id_parts.append(f"Route {str(trans.route_number).strip()}")
    identified_str = " ".join(id_parts).strip()
    svc_claim = (
        f"The identified public transportation from MCO to the Disney Springs area is '{identified_str}'."
        if identified_str else
        "The identified public transportation from MCO to the Disney Springs area is provided."
    )
    await evaluator.verify(
        claim=svc_claim,
        node=svc_leaf,
        sources=t_url if t_url_valid else None,
        additional_instruction=(
            "Verify on the provided page that the service (e.g., LYNX bus) and/or its route number explicitly references travel "
            "from Orlando International Airport (MCO) to Disney Springs or the Walt Disney World/Disney Springs vicinity."
        ),
    )

    # Fare Information
    fare_leaf = evaluator.add_leaf(
        id="Fare_Information",
        desc="The one-way fare cost is provided",
        parent=trans_node,
        critical=True,
    )
    fare_val = trans.one_way_fare or ""
    fare_claim = f"The one-way fare for this service is '{fare_val}'."
    await evaluator.verify(
        claim=fare_claim,
        node=fare_leaf,
        sources=t_url if t_url_valid else None,
        additional_instruction=(
            "Confirm the one-way fare amount on the referenced page. If multiple fare types exist, match the answer's stated fare."
        ),
    )

    # Operating Hours
    hours_leaf = evaluator.add_leaf(
        id="Operating_Hours",
        desc="The daily operating hours (start time and end time) are provided",
        parent=trans_node,
        critical=True,
    )
    hours_val = trans.operating_hours or ""
    hours_claim = f"The daily operating hours are '{hours_val}'."
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=t_url if t_url_valid else None,
        additional_instruction=(
            "Check the referenced page for the daily operating hours window (start and end times). "
            "Allow reasonable formatting variations (e.g., '5:00 AM – 11:00 PM' vs '05:00-23:00')."
        ),
    )

    # Service Frequency
    freq_leaf = evaluator.add_leaf(
        id="Service_Frequency",
        desc="The service frequency (how often the service runs) is provided",
        parent=trans_node,
        critical=True,
    )
    freq_val = trans.frequency or ""
    freq_claim = f"The service frequency is '{freq_val}'."
    await evaluator.verify(
        claim=freq_claim,
        node=freq_leaf,
        sources=t_url if t_url_valid else None,
        additional_instruction=(
            "Confirm on the referenced page how often the service runs (e.g., every 30 minutes, hourly, etc.). "
            "Accept minor wording variations that convey the same frequency."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate the travel planning task answer using the Mind2Web2 framework and return a structured summary.
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
        default_model=model,
    )

    # Root child: Travel Planning Task (parallel, non-critical – root already embodies description)
    # We will treat root as the overall aggregator; add two critical subdomains under it.

    # 1) Hotels extraction
    hotels_extraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )
    # Normalize hotels list to exactly three items
    hotels_list = list(hotels_extraction.hotels)
    if len(hotels_list) < 3:
        hotels_list.extend([HotelItem() for _ in range(3 - len(hotels_list))])
    else:
        hotels_list = hotels_list[:3]

    # Hotel_Selection node (critical, parallel)
    hotels_parent_node = evaluator.add_parallel(
        id="Hotel_Selection",
        desc="Three hotels from the official Disney Springs Resort Area hotels list with complete information",
        parent=root,
        critical=True,
    )

    # Verify each of the three hotels (all critical children under Hotel_Selection)
    for idx in range(3):
        await verify_single_hotel(evaluator, hotels_parent_node, hotels_list[idx], idx)

    # 2) Transportation extraction
    trans_extraction = await evaluator.extract(
        prompt=prompt_extract_transportation(),
        template_class=TransportationExtraction,
        extraction_name="transportation_extraction",
    )

    # Public_Transportation_Information node (critical, parallel)
    await verify_transportation(evaluator, root, trans_extraction)

    # Optional: record auxiliary info
    evaluator.add_custom_info(
        info={"official_list_index_url": OFFICIAL_LIST_INDEX_URL},
        info_type="reference",
        info_name="official_list_index",
    )

    # Return result
    return evaluator.get_summary()