import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "oregon_coast_pet_friendly_yurts"
TASK_DESCRIPTION = """
I'm planning a camping trip on the Oregon coast with my dog and prefer staying in a yurt. Identify one Oregon State Park on the coast that has pet-friendly yurt rentals, with at least 15 total yurts available. For your answer, provide: the park name, its location on the Oregon coast, the number of pet-friendly yurts available, the official Oregon State Parks website URL for this park, confirmation of online reservation availability, and a description of the basic amenities included in the yurts.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkInfo(BaseModel):
    park_name: Optional[str] = None
    location: Optional[str] = None  # Free text location on the Oregon coast (e.g., "near Newport on the central coast")
    total_yurts: Optional[str] = None  # Keep as string to allow ranges or qualifiers
    pet_friendly_yurts: Optional[str] = None  # Specific number or textual number as provided
    official_url: Optional[str] = None  # Prefer stateparks.oregon.gov or oregonstateparks.org
    reservation_url: Optional[str] = None  # Prefer official reservation portal link if provided
    reservation_online_available: Optional[str] = None  # "yes"/"no" or textual confirmation
    amenities: List[str] = Field(default_factory=list)  # Short phrases for basic amenities (e.g., "heat", "electricity")
    additional_urls: List[str] = Field(default_factory=list)  # Any other URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_info() -> str:
    return """
    Extract information about exactly one Oregon State Park on the coast featuring pet-friendly yurt rentals from the answer text. 
    If multiple parks are mentioned, select the first one mentioned and extract its details only.

    Return a JSON object with the following fields:
    - park_name: The park's name as written in the answer (string or null)
    - location: The described location on the Oregon coast (string or null)
    - total_yurts: The total number of yurts at the park as stated in the answer (string; can include numbers, ranges, or qualifiers; null if missing)
    - pet_friendly_yurts: The specific number of pet-friendly yurts as stated in the answer (string; null if not explicitly provided)
    - official_url: The official Oregon State Parks page URL for this park if provided. Prefer domains:
        • stateparks.oregon.gov
        • oregonstateparks.org
      (Use the exact URL from the answer, or null if none.)
    - reservation_url: The online reservation URL for booking yurts (e.g., a ReserveAmerica or Oregon State Parks reservation link), if provided in the answer; otherwise null.
    - reservation_online_available: "yes" or "no" (string) if the answer explicitly confirms whether online reservations are available; otherwise null.
    - amenities: A list of short phrases for basic yurt amenities explicitly stated in the answer (e.g., ["heat", "electricity", "bunk beds", "table and chairs"]). Return an empty list if none listed.
    - additional_urls: Any other URLs cited in the answer related to this park or the yurts (array of strings; empty array if none).

    IMPORTANT:
    - Do not invent any information; only extract what is explicitly present in the answer.
    - For URLs, extract the actual link targets (resolve markdown links to the underlying URL).
    - If the answer cites multiple URLs, put the main official park page in official_url if present; everything else goes into additional_urls.
    - If some information is missing, use null (or empty list where appropriate).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def gather_sources(park: ParkInfo) -> List[str]:
    urls = []
    if park.official_url:
        urls.append(park.official_url)
    if park.reservation_url:
        urls.append(park.reservation_url)
    for u in park.additional_urls or []:
        if u:
            urls.append(u)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, park: ParkInfo) -> None:
    # Create a critical parallel aggregator node (reflecting the rubric root)
    main_node = evaluator.add_parallel(
        id="Oregon_Coastal_State_Park_with_Pet_Friendly_Yurts",
        desc="Evaluation of whether the identified park meets all specified criteria for pet-friendly yurt camping on the Oregon coast",
        parent=root_node,
        critical=True
    )

    park_name = park.park_name or "the identified park"
    all_urls = gather_sources(park)
    primary_source = park.official_url or (all_urls[0] if all_urls else None)

    # 1) Oregon_State_Park_System
    node1 = evaluator.add_leaf(
        id="Oregon_State_Park_System",
        desc="The park is confirmed to be part of the Oregon State Parks system",
        parent=main_node,
        critical=True
    )
    claim1 = f"{park_name} is a park managed by the Oregon State Parks (Oregon Parks and Recreation Department)."
    await evaluator.verify(
        claim=claim1,
        node=node1,
        sources=primary_source or all_urls,
        additional_instruction="Confirm the page is for an Oregon State Parks park. Evidence includes branding, site header/footer, or explicit text indicating it is a State Parks site."
    )

    # 2) Coastal_Location
    node2 = evaluator.add_leaf(
        id="Coastal_Location",
        desc="The park is located directly on the Oregon coast (not inland)",
        parent=main_node,
        critical=True
    )
    claim2 = f"{park_name} is located on the Oregon coast along the Pacific Ocean (i.e., a coastal state park, not inland)."
    await evaluator.verify(
        claim=claim2,
        node=node2,
        sources=primary_source or all_urls,
        additional_instruction="Look for wording such as 'coast', 'beach', 'ocean', 'Pacific Ocean', or explicit coastal region references (north/central/south coast). If the park is inland, mark as not supported."
    )

    # 3) Yurt_Availability
    node3 = evaluator.add_leaf(
        id="Yurt_Availability",
        desc="The park offers yurt camping facilities for rent",
        parent=main_node,
        critical=True
    )
    claim3 = f"{park_name} offers yurt camping that can be rented."
    await evaluator.verify(
        claim=claim3,
        node=node3,
        sources=[u for u in [park.official_url, park.reservation_url] if u] or all_urls,
        additional_instruction="Check for explicit mentions of 'Yurt(s)' under lodging/accommodations or camping. Phrases like 'yurt rentals' or 'yurt camping' should be present."
    )

    # 4) Pet_Friendly_Policy
    node4 = evaluator.add_leaf(
        id="Pet_Friendly_Policy",
        desc="The park has yurts designated as pet-friendly available for rent",
        parent=main_node,
        critical=True
    )
    claim4 = f"At {park_name}, some yurts are specifically designated pet-friendly and available to rent (dogs allowed in designated yurts)."
    await evaluator.verify(
        claim=claim4,
        node=node4,
        sources=[u for u in [park.official_url, park.reservation_url] if u] or all_urls,
        additional_instruction="Look for 'pet-friendly yurt(s)', 'pets allowed in yurts', or iconography/notes indicating pet-friendly lodging. If pets are prohibited in yurts, mark as not supported."
    )

    # 5) Minimum_Total_Yurts (>= 15)
    node5 = evaluator.add_leaf(
        id="Minimum_Total_Yurts",
        desc="The park has at least 15 total yurts available",
        parent=main_node,
        critical=True
    )
    claim5 = f"{park_name} has at least 15 total yurts (counting standard and deluxe yurts combined)."
    await evaluator.verify(
        claim=claim5,
        node=node5,
        sources=primary_source or all_urls,
        additional_instruction="Verify the total number of yurts is 15 or more. If the page lists separate counts for standard and deluxe yurts, sum them. Do not count cabins or other lodging types."
    )

    # 6) Pet_Friendly_Yurt_Count (specific number provided and accurate)
    node6 = evaluator.add_leaf(
        id="Pet_Friendly_Yurt_Count",
        desc="The specific number of pet-friendly yurts is provided",
        parent=main_node,
        critical=True
    )
    pet_count_text = park.pet_friendly_yurts or ""
    claim6 = f"The number of pet-friendly yurts at {park_name} is {pet_count_text}."
    await evaluator.verify(
        claim=claim6,
        node=node6,
        sources=[u for u in [park.reservation_url, park.official_url] if u] or all_urls,
        additional_instruction="Check whether the cited number of pet-friendly yurts matches the official information. Allow minor wording differences, but the numeric value must be supported."
    )

    # 7) Online_Reservation_System
    node7 = evaluator.add_leaf(
        id="Online_Reservation_System",
        desc="The park has an accessible online reservation system for booking yurts",
        parent=main_node,
        critical=True
    )
    claim7 = f"Yurts at {park_name} can be reserved online using the official Oregon State Parks reservation system."
    await evaluator.verify(
        claim=claim7,
        node=node7,
        sources=[u for u in [park.reservation_url, park.official_url] if u] or all_urls,
        additional_instruction="Confirm there is an online reservation pathway (e.g., 'Make a reservation' link/button or a link to the official reservation portal such as oregonstateparks.reserveamerica.com)."
    )

    # 8) Official_Reference_URL (ensure provided and official OSP page)
    node8 = evaluator.add_leaf(
        id="Official_Reference_URL",
        desc="An official Oregon State Parks website URL for the park is provided",
        parent=main_node,
        critical=True
    )
    if nonempty(park.official_url):
        claim8 = f"The provided URL {park.official_url} is an official Oregon State Parks webpage for {park_name}."
    else:
        claim8 = f"An official Oregon State Parks URL for {park_name} is provided in the answer."
    await evaluator.verify(
        claim=claim8,
        node=node8,
        sources=park.official_url or None,
        additional_instruction="Treat as 'official' if the URL domain is stateparks.oregon.gov or the legacy oregonstateparks.org, and the page is specifically for the identified park."
    )

    # 9) Yurt_Amenities
    node9 = evaluator.add_leaf(
        id="Yurt_Amenities",
        desc="Basic amenities included in the yurts are described (e.g., heating, furnishings, utilities)",
        parent=main_node,
        critical=True
    )
    amenities_text = ", ".join(park.amenities) if park.amenities else ""
    claim9 = f"Basic yurt amenities at {park_name} include: {amenities_text}."
    await evaluator.verify(
        claim=claim9,
        node=node9,
        sources=[u for u in [park.official_url, park.reservation_url] if u] or all_urls,
        additional_instruction="Verify that the listed amenities are actually stated on the official page(s). Focus on basics such as heat, electricity, beds/bunks, futon, table/chairs, lighting. Minor phrasing differences are acceptable."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    # Initialize unified evaluator (root is non-critical container)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall rubric root is parallel aggregation
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

    # Extract structured info from the answer
    park_info = await evaluator.extract(
        prompt=prompt_extract_park_info(),
        template_class=ParkInfo,
        extraction_name="park_info"
    )

    # Optionally record extracted summary for debugging/visibility
    evaluator.add_custom_info(
        info={
            "park_name": park_info.park_name,
            "location": park_info.location,
            "total_yurts_mentioned": park_info.total_yurts,
            "pet_friendly_yurts_mentioned": park_info.pet_friendly_yurts,
            "official_url": park_info.official_url,
            "reservation_url": park_info.reservation_url,
            "reservation_online_available": park_info.reservation_online_available,
            "amenities_list": park_info.amenities,
            "additional_urls": park_info.additional_urls,
        },
        info_type="extraction_debug",
        info_name="extracted_park_fields"
    )

    # Build and execute verification tree according to rubric
    await build_verification_tree(evaluator, root, park_info)

    # Return final structured summary
    return evaluator.get_summary()