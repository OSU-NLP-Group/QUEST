import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lga_hotels"
TASK_DESCRIPTION = """
I have an extended layover at LaGuardia Airport in New York and would like to find hotels where I can rest and relax. Please identify at least four hotels in the Queens area (near LaGuardia Airport) that meet all of the following requirements:

1. The hotel must offer complimentary (free) airport shuttle service to and from LaGuardia Airport
2. The hotel must have an indoor pool
3. The hotel must provide 24-hour front desk service

For each hotel, please provide:
- The hotel name
- Complete physical address
- Direct contact phone number
- A link to the hotel's official website or a reputable booking page
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    website_urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract hotel entries from the answer. We need up to 8 hotels mentioned in the answer text.
    For each hotel, extract the following fields exactly as they appear:
    - name: Hotel name
    - address: Complete physical address as written (include street, city, state, and ZIP/postal if present)
    - phone: Direct contact phone number as presented
    - website_urls: A list of all URLs explicitly associated with this hotel in the answer.
      These can include the official hotel website and reputable booking pages (e.g., Marriott/Hilton/Hyatt brand pages, Booking.com, Expedia, Hotels.com).
      Include only valid URLs mentioned in the answer. If a URL is missing a protocol, prepend "http://".
    Return a JSON object:
      {"hotels": [ { "name": ..., "address": ..., "phone": ..., "website_urls": [...] }, ... ]}
    If any field is missing for a hotel, set it to null (or [] for website_urls).
    Do not invent any information or URLs that are not explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"Hotel #{n}")


def _normalize_and_dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls or []:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if not (s.startswith("http://") or s.startswith("https://")):
            s = "http://" + s
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Verification for a single hotel                                             #
# --------------------------------------------------------------------------- #
async def verify_one_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    index: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single hotel.
    """
    hotel_idx = index  # 1-based
    hotel_desc = f"{_ordinal(hotel_idx)} hotel meeting all specified criteria"

    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{hotel_idx}",
        desc=hotel_desc,
        parent=parent_node,
        critical=False  # Allow partial scoring across hotels
    )

    # Prepare URLs (evidence)
    all_urls = _normalize_and_dedupe_urls(hotel.website_urls)

    # Existence nodes (non-critical content provided checks as per rubric)
    evaluator.add_custom_node(
        result=(hotel.address is not None and hotel.address.strip() != ""),
        id=f"Physical_Address_{hotel_idx}",
        desc=f"Complete physical address is provided",
        parent=hotel_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=(hotel.phone is not None and hotel.phone.strip() != ""),
        id=f"Phone_Number_{hotel_idx}",
        desc=f"Direct contact phone number is provided",
        parent=hotel_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=(len(all_urls) > 0),
        id=f"Website_Link_{hotel_idx}",
        desc=f"Link to hotel's official website or booking page is provided",
        parent=hotel_node,
        critical=False
    )

    # Additional gating: require at least one source URL for factual verification (critical sibling)
    # This ensures factual leaves won't pass without web evidence.
    evaluator.add_custom_node(
        result=(len(all_urls) > 0),
        id=f"Sources_Present_{hotel_idx}",
        desc=f"At least one valid website or reputable booking URL is provided as evidence for this hotel",
        parent=hotel_node,
        critical=True
    )

    # Prepare leaves for factual verification (critical per rubric)
    # 1) Queens location near LGA
    queens_node = evaluator.add_leaf(
        id=f"Queens_Location_{hotel_idx}",
        desc="Hotel is located in Queens, New York (near LaGuardia Airport)",
        parent=hotel_node,
        critical=True
    )
    address_for_hint = hotel.address or ""
    name_for_hint = hotel.name or f"Hotel #{hotel_idx}"
    queens_claim = (
        "This hotel is located in Queens, New York City and is near LaGuardia Airport (LGA)."
    )
    queens_instruction = (
        f"Use only the provided webpage(s) to judge. If the page lists an address in neighborhoods such as "
        f"East Elmhurst, Flushing, Astoria, Long Island City, Jackson Heights, Corona, Jamaica, Woodside, "
        f"Elmhurst, College Point, Rego Park, Forest Hills, Kew Gardens, etc., consider that as Queens. "
        f"Mentions like 'near LaGuardia' or 'minutes from LGA' also support proximity. "
        f"If the pages indicate another borough (Manhattan, Brooklyn, Bronx, Staten Island) or do not support Queens/LGA proximity, mark as not supported. "
        f"Hotel name hint: {name_for_hint}. Address provided in the answer (for context only): {address_for_hint}."
    )

    # 2) Free (complimentary) shuttle to/from LGA
    shuttle_free_node = evaluator.add_leaf(
        id=f"Free_Shuttle_Service_{hotel_idx}",
        desc="Hotel provides complimentary airport shuttle service to LaGuardia Airport",
        parent=hotel_node,
        critical=True
    )
    shuttle_free_claim = (
        "The hotel provides a complimentary (free) airport shuttle service to and/or from LaGuardia Airport (LGA)."
    )
    shuttle_free_instruction = (
        "Look for explicit terms like 'complimentary', 'free', 'no charge', or equivalent. "
        "If the page only states 'shuttle available' without indicating it's free/complimentary, treat as not supported. "
        "Mentions of a fee or paid shuttle should be marked not supported."
    )

    # 3) Shuttle hours stated
    shuttle_hours_node = evaluator.add_leaf(
        id=f"Shuttle_Hours_Stated_{hotel_idx}",
        desc="Shuttle service operating hours are clearly stated",
        parent=hotel_node,
        critical=True
    )
    shuttle_hours_claim = (
        "The webpage(s) clearly state the operating hours or schedule of the hotel's airport shuttle "
        "(for example, '24/7', '24 hours', '6 AM–11 PM', or a frequency tied to times)."
    )
    shuttle_hours_instruction = (
        "Accept explicit hours windows (e.g., '5:00 AM to 11:00 PM'), phrases like '24/7' or '24-hour shuttle', "
        "or a schedule with times. Vague statements like 'daily shuttle' without any hours are insufficient."
    )

    # 4) Indoor pool
    indoor_pool_node = evaluator.add_leaf(
        id=f"Indoor_Pool_{hotel_idx}",
        desc="Hotel has an indoor pool",
        parent=hotel_node,
        critical=True
    )
    indoor_pool_claim = "The hotel has an indoor swimming pool."
    indoor_pool_instruction = (
        "The page must explicitly indicate 'indoor pool' (or equivalent like 'heated indoor pool'). "
        "If it only says 'pool' or 'outdoor pool', this should NOT be considered supported."
    )

    # 5) 24-hour front desk
    front_desk_node = evaluator.add_leaf(
        id=f"24Hour_Service_{hotel_idx}",
        desc="Hotel provides 24-hour front desk service",
        parent=hotel_node,
        critical=True
    )
    front_desk_claim = "The hotel provides 24-hour front desk service."
    front_desk_instruction = (
        "Accept synonyms such as '24-hour reception', '24/7 front desk', or equivalent. "
        "If hours are limited (e.g., 7 AM–11 PM), mark as not supported."
    )

    # Batch verify all factual claims (will be auto-skipped if a critical sibling like Sources_Present fails)
    await evaluator.batch_verify(
        [
            (queens_claim, all_urls, queens_node, queens_instruction),
            (shuttle_free_claim, all_urls, shuttle_free_node, shuttle_free_instruction),
            (shuttle_hours_claim, all_urls, shuttle_hours_node, shuttle_hours_instruction),
            (indoor_pool_claim, all_urls, indoor_pool_node, indoor_pool_instruction),
            (front_desk_claim, all_urls, front_desk_node, front_desk_instruction),
        ]
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
    Evaluate an answer for the LaGuardia Airport hotels task.

    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator (root set to non-critical to allow partial credit across hotels)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Root node for evaluating the hotel search task: finding hotels near LaGuardia Airport with specific amenities",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract hotels from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction"
    )

    hotels = list(extracted.hotels or [])
    # Ensure we evaluate exactly 4 hotels: take first 4 or pad with empty placeholders
    if len(hotels) < 4:
        hotels.extend([HotelItem() for _ in range(4 - len(hotels))])
    else:
        hotels = hotels[:4]

    # Build and verify each hotel subtree
    for idx, hotel in enumerate(hotels, start=1):
        await verify_one_hotel(evaluator, root, hotel, idx)

    # Return evaluation summary
    return evaluator.get_summary()