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
TASK_ID = "airport_hotels_requirements"
TASK_DESCRIPTION = """
Identify one hotel near each of the following four major US airports: Nashville International Airport (BNA), Miami International Airport (MIA), Denver International Airport (DEN), and Los Angeles International Airport (LAX). For each hotel, provide the hotel name, the exact distance from the airport terminal (in miles), and a reference URL. Each hotel must meet all of the following requirements:

1. Be located within 2 miles of the airport terminal
2. Provide complimentary (free) airport shuttle service for guests
3. Offer free parking for guests during their stay
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    name: Optional[str] = None
    # Keep distance as a string to maximize compatibility with varied formats like "1.2 mi", "1 mile", "0.8 miles"
    distance_miles: Optional[str] = None
    # Collect all URLs (markdown links resolved to URLs) cited for this hotel's info
    sources: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    bna: Optional[HotelInfo] = None
    mia: Optional[HotelInfo] = None
    den: Optional[HotelInfo] = None
    lax: Optional[HotelInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract exactly one hotel for each of the following airports from the answer text:
    - bna: Nashville International Airport (BNA)
    - mia: Miami International Airport (MIA)
    - den: Denver International Airport (DEN)
    - lax: Los Angeles International Airport (LAX)

    For each airport, extract an object with:
    1) name: The hotel name exactly as written in the answer (string, or null if missing).
    2) distance_miles: The exact distance from the airport terminal in miles as presented in the answer (string, e.g., "1.5 miles", "0.8 mi", "2 miles"; return null if missing).
    3) sources: An array of all URLs cited in the answer specifically for this hotel's information. Include URLs in any reasonable format (plain, markdown links). If none are provided, return an empty array.

    If the answer lists multiple hotels for an airport, select the first one that appears. If none are provided for an airport, set that airport field to null.

    Return a JSON object with keys: bna, mia, den, lax, each mapping to the corresponding object described above (or null).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_number(text: Optional[str]) -> bool:
    if not text:
        return False
    return any(ch.isdigit() for ch in text)


def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Allow both http/https; if protocol missing, leave as-is (Extractor may add); still treat as provided
        cleaned.append(u)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification per-airport                                                    #
# --------------------------------------------------------------------------- #
async def verify_airport_hotel(
    evaluator: Evaluator,
    parent_node,
    *,
    airport_code: str,
    airport_full_name: str,
    aggregator_id: str,
    leaf_prefix: str,
    extracted: Optional[HotelInfo],
) -> None:
    """
    Build verification sub-tree for a single airport.

    Parameters:
    - airport_code: e.g., "BNA"
    - airport_full_name: e.g., "Nashville International Airport (BNA)"
    - aggregator_id: e.g., "Nashville_BNA_Hotel"
    - leaf_prefix: e.g., "Nashville" | "Miami" | "Denver" | "LAX"
    - extracted: extracted HotelInfo (may be None)
    """
    # Critical aggregator for this airport (because the top task node is critical)
    airport_node = evaluator.add_parallel(
        id=aggregator_id,
        desc=f"Requirements for the hotel near {airport_full_name}",
        parent=parent_node,
        critical=True,
    )

    name = (extracted.name.strip() if extracted and extracted.name else None)
    dist_str = (extracted.distance_miles.strip() if extracted and extracted.distance_miles else None)
    urls = _non_empty_urls(extracted.sources if extracted else [])

    # 1) Name provided (critical existence)
    evaluator.add_custom_node(
        result=bool(name),
        id=f"{leaf_prefix}_Hotel_Name_Provided",
        desc=f"A valid hotel name near {airport_full_name} is provided",
        parent=airport_node,
        critical=True,
    )

    # 2) Distance value provided (critical existence)
    evaluator.add_custom_node(
        result=bool(dist_str) and _has_number(dist_str),
        id=f"{leaf_prefix}_Hotel_Distance_Value_Provided",
        desc=f"The exact distance in miles from {airport_full_name} to the hotel is provided",
        parent=airport_node,
        critical=True,
    )

    # 3) Reference URL provided (critical existence)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{leaf_prefix}_Hotel_Reference_URL",
        desc=f"A reference URL is provided to verify the {airport_full_name.split(' ')[0]} hotel's information",
        parent=airport_node,
        critical=True,
    )

    # 4) Distance within 2 miles (critical; source-grounded)
    within_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Hotel_Distance_Within_2_Miles",
        desc=f"The {airport_full_name.split(' ')[0]} hotel is located within 2 miles of {airport_full_name} terminal",
        parent=airport_node,
        critical=True,
    )
    hotel_label = name if name else "the identified hotel"
    distance_claim = (
        f"{hotel_label} is located within 2 miles of the {airport_full_name} terminal."
    )
    await evaluator.verify(
        claim=distance_claim,
        node=within_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the webpage explicitly supports that the hotel is within 2 miles of the stated airport terminal. "
            "Accept phrases like 'within 2 miles', '1.9 miles', '1 mile', '0.8 mi', etc. Minor rounding differences are acceptable. "
            "If the page indicates a distance greater than 2 miles, or only states shuttle availability without distance, mark as not supported."
        ),
    )

    # 5) Complimentary (free) airport shuttle (critical; source-grounded)
    shuttle_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Hotel_Free_Shuttle",
        desc=f"The {airport_full_name.split(' ')[0]} hotel provides complimentary airport shuttle service",
        parent=airport_node,
        critical=True,
    )
    shuttle_claim = (
        f"{hotel_label} provides complimentary (free) airport shuttle service for guests to/from {airport_full_name}."
    )
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm that the page explicitly mentions free or complimentary airport shuttle service. "
            "Accept synonyms like 'complimentary', 'no charge'. "
            "If it only says 'shuttle available' or indicates fees (e.g., 'paid shuttle', 'surcharge'), treat as not supported."
        ),
    )

    # 6) Free parking for guests during their stay (critical; source-grounded)
    parking_leaf = evaluator.add_leaf(
        id=f"{leaf_prefix}_Hotel_Free_Parking",
        desc=f"The {airport_full_name.split(' ')[0]} hotel offers free parking for guests during their stay",
        parent=airport_node,
        critical=True,
    )
    parking_claim = (
        f"{hotel_label} offers free parking for guests during their stay (complimentary self-parking)."
    )
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm the webpage states free or complimentary parking for registered/overnight hotel guests. "
            "Accept phrases like 'complimentary self-parking' for hotel guests. "
            "Do NOT accept paid parking, valet-only paid parking, or park-and-fly packages that are not free during the guest's stay."
        ),
    )


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
    Evaluate an answer against the airport hotel requirements rubric.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Create a top-level critical node to mirror the rubric's critical root
    top_critical = evaluator.add_parallel(
        id="Airport_Hotel_Requirements",
        desc="Verify that one hotel near each of four specified US airports meets all required criteria for distance, shuttle service, parking, and information provision",
        parent=root,
        critical=True,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="airport_hotels_extraction",
    )

    # Airport configurations (IDs and labels follow the rubric JSON)
    airports_cfg = [
        {
            "code": "BNA",
            "full": "Nashville International Airport (BNA)",
            "agg_id": "Nashville_BNA_Hotel",
            "prefix": "Nashville",
            "info": extracted.bna if extracted else None,
        },
        {
            "code": "MIA",
            "full": "Miami International Airport (MIA)",
            "agg_id": "Miami_MIA_Hotel",
            "prefix": "Miami",
            "info": extracted.mia if extracted else None,
        },
        {
            "code": "DEN",
            "full": "Denver International Airport (DEN)",
            "agg_id": "Denver_DEN_Hotel",
            "prefix": "Denver",
            "info": extracted.den if extracted else None,
        },
        {
            "code": "LAX",
            "full": "Los Angeles International Airport (LAX)",
            "agg_id": "Los_Angeles_LAX_Hotel",
            "prefix": "LAX",
            "info": extracted.lax if extracted else None,
        },
    ]

    # Build verification subtrees for each airport
    for cfg in airports_cfg:
        await verify_airport_hotel(
            evaluator,
            top_critical,
            airport_code=cfg["code"],
            airport_full_name=cfg["full"],
            aggregator_id=cfg["agg_id"],
            leaf_prefix=cfg["prefix"],
            extracted=cfg["info"],
        )

    # Return summary
    return evaluator.get_summary()