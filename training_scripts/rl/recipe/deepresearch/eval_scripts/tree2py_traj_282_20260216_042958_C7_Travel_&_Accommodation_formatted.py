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
TASK_ID = "disney_destiny_trip_requirements"
TASK_DESCRIPTION = """
A family living in Nashville, Tennessee is planning to take a Disney Destiny cruise departing from Fort Lauderdale, Florida in February 2026, with Aruba as one of the ports of call. They plan to fly to Fort Lauderdale the day before the cruise and stay at a hotel near the cruise terminal. What are all the essential travel requirements, documentation needs, airline baggage policies, accommodation considerations, and cruise booking details they must address for this trip?
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ClaimBlock(BaseModel):
    claim_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class CategoriesBlock(BaseModel):
    categories: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class DatesBlock(BaseModel):
    dates: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class TripRequirementsExtraction(BaseModel):
    # Cruise logistics
    departure_port: Optional[ClaimBlock] = None
    stateroom_categories: Optional[CategoriesBlock] = None
    february_departures: Optional[DatesBlock] = None

    # Documentation requirements
    passport_validity: Optional[ClaimBlock] = None
    ed_card_requirement: Optional[ClaimBlock] = None
    ed_card_timing: Optional[ClaimBlock] = None

    # Airline and baggage policy (Avelo)
    airline_from_nashville: Optional[ClaimBlock] = None
    free_personal_item: Optional[ClaimBlock] = None
    personal_item_size: Optional[ClaimBlock] = None
    baggage_fees: Optional[ClaimBlock] = None
    checked_bag_weight: Optional[ClaimBlock] = None

    # Accommodation considerations near the port
    hotel_proximity: Optional[ClaimBlock] = None
    hotel_shuttle: Optional[ClaimBlock] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_requirements() -> str:
    return """
    Extract from the answer all specific claims and their cited URLs relevant to the family's Disney Destiny cruise trip.

    For each field below, return either the exact phrasing used in the answer (as claim_text) and a list of URLs explicitly cited in the answer (as sources), or return null/empty list when missing.

    Fields to extract:
    1) departure_port: claim_text that Disney Destiny cruises depart from Fort Lauderdale (Port Everglades), and sources (URLs).
    2) stateroom_categories: the list of stateroom category names mentioned (e.g., Inside, Oceanview, Verandah, Concierge), and sources (URLs).
    3) february_departures: a list of the specific February 2026 departure dates from Fort Lauderdale for Disney Destiny that the answer mentions, and sources (URLs).
    4) passport_validity: claim_text about passport validity requirements (e.g., valid for 6 months beyond departure from Aruba), and sources (URLs).
    5) ed_card_requirement: claim_text that Aruba requires all travelers to complete an online ED card, and sources (URLs).
    6) ed_card_timing: claim_text about when the Aruba ED card can be completed (e.g., within 7 days before travel), and sources (URLs).
    7) airline_from_nashville: claim_text identifying that Avelo Airlines operates flights from Nashville (BNA) to Florida destinations, and sources (URLs).
    8) free_personal_item: claim_text stating that Avelo Airlines allows one free personal item, and sources (URLs).
    9) personal_item_size: claim_text providing the maximum size for Avelo's free personal item (e.g., 17\" L x 13\" H x 9\" W), and sources (URLs).
    10) baggage_fees: claim_text mentioning that Avelo charges fees for carry‑on and checked bags (e.g., $40–$60 depending on when purchased), and sources (URLs).
    11) checked_bag_weight: claim_text stating that Avelo's checked bag weight limit is 50 lbs (22 kg), and sources (URLs).
    12) hotel_proximity: claim_text that hotels near Port Everglades cruise terminal are available within a few miles of the port, and sources (URLs).
    13) hotel_shuttle: claim_text that some hotels near the cruise port offer shuttle service to Port Everglades, and sources (URLs).

    IMPORTANT:
    - Only extract URLs that appear in the answer (including markdown links). Do not invent URLs.
    - If a field is not mentioned, set its claim_text to null and its sources to [].
    - For stateroom_categories, populate the categories array with the categories explicitly listed in the answer; for february_departures, populate dates with the date strings explicitly listed.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _get_sources(block: Optional[ClaimBlock | CategoriesBlock | DatesBlock]) -> List[str]:
    if block is None:
        return []
    return getattr(block, "sources", []) or []

def _has_sources(block: Optional[ClaimBlock | CategoriesBlock | DatesBlock]) -> bool:
    return len(_get_sources(block)) > 0

def _text(block: Optional[ClaimBlock]) -> str:
    return (block.claim_text or "").strip() if block else ""

def _first_n(items: List[str], n: int = 5) -> List[str]:
    return items[:n]


# --------------------------------------------------------------------------- #
# Verification builder functions                                              #
# --------------------------------------------------------------------------- #
async def _verify_claim_group(
    evaluator: Evaluator,
    parent,
    group_id: str,
    group_desc: str,
    claim_id: str,
    claim_desc: str,
    claim_text: str,
    sources: List[str],
    critical_group: bool,
    extra_prereq_leaves: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Create a group for a single rubric item:
      - Existence of sources (critical leaf)
      - Evidence-supported verification (critical leaf)
    """
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=critical_group
    )

    # Existence check for sources (critical leaf)
    exist_node = evaluator.add_custom_node(
        result=bool(sources),
        id=f"{group_id}_sources_exist",
        desc=f"{group_desc}: sources are provided in the answer",
        parent=group_node,
        critical=True
    )

    # Verification leaf
    verify_leaf = evaluator.add_leaf(
        id=claim_id,
        desc=claim_desc,
        parent=group_node,
        critical=True
    )

    # Route verification through sources
    await evaluator.verify(
        claim=claim_text if claim_text else claim_desc,
        node=verify_leaf,
        sources=sources,
        additional_instruction="Verify that the cited source(s) explicitly support the claim. Allow reasonable naming variations and synonyms.",
        extra_prerequisites=[exist_node] + (extra_prereq_leaves or [])
    )

    return {"group": group_node, "exist_node": exist_node, "verify_leaf": verify_leaf}


async def _verify_simple_answer_match(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    leaf_desc: str,
    claim_text: str,
    critical: bool = True,
    extra_prereq_leaves: Optional[List] = None,
):
    """
    Simple verification against the answer text (no web sources).
    """
    leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim_text if claim_text else leaf_desc,
        node=leaf,
        sources=None,
        additional_instruction="Check the answer text to confirm this information is explicitly listed.",
        extra_prerequisites=extra_prereq_leaves or []
    )
    return leaf


# --------------------------------------------------------------------------- #
# Main verification orchestration                                             #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, extracted: TripRequirementsExtraction) -> None:
    # 1) Cruise departure port (critical)
    await _verify_claim_group(
        evaluator,
        root,
        group_id="cruise_departure_port_main",
        group_desc="Disney Destiny departure port verification",
        claim_id="cruise_departure_port",
        claim_desc="Correctly identifies that Disney Destiny cruises depart from Fort Lauderdale (Port Everglades)",
        claim_text="Disney Destiny cruises depart from Fort Lauderdale (Port Everglades).",
        sources=_get_sources(extracted.departure_port),
        critical_group=True,
    )

    # 2) Stateroom categories (critical)
    # 2a) Evidence-supported categories statement
    await _verify_claim_group(
        evaluator,
        root,
        group_id="stateroom_categories_main",
        group_desc="Disney Destiny stateroom categories verification",
        claim_id="stateroom_categories_supported",
        claim_desc="Lists the four main stateroom categories available on Disney Destiny: Inside, Oceanview, Verandah, and Concierge",
        claim_text="The four main stateroom categories available on Disney Destiny are Inside, Oceanview, Verandah, and Concierge.",
        sources=_get_sources(extracted.stateroom_categories),
        critical_group=True,
    )
    # 2b) Also check the answer text lists all four (simple verification)
    await _verify_simple_answer_match(
        evaluator,
        parent=evaluator.find_node("stateroom_categories_main"),
        leaf_id="stateroom_categories_listed",
        leaf_desc="Answer lists the four categories: Inside, Oceanview, Verandah, Concierge",
        claim_text="The answer explicitly lists Inside, Oceanview, Verandah, and Concierge as the stateroom categories.",
        critical=True
    )

    # 3) Passport validity (critical)
    await _verify_claim_group(
        evaluator,
        root,
        group_id="passport_validity_main",
        group_desc="Passport validity requirement verification",
        claim_id="passport_validity",
        claim_desc="States that passports must be valid for at least 6 months beyond the departure date from Aruba",
        claim_text="Passports must be valid for at least 6 months beyond the departure date from Aruba.",
        sources=_get_sources(extracted.passport_validity),
        critical_group=True,
    )

    # 4) Aruba ED card requirement (critical)
    ed_req = await _verify_claim_group(
        evaluator,
        root,
        group_id="ed_card_requirement_main",
        group_desc="Aruba ED card requirement verification",
        claim_id="ed_card_requirement",
        claim_desc="Mentions that Aruba requires all travelers to complete an online ED card (Embarkation/Disembarkation card)",
        claim_text="Aruba requires all travelers to complete an online ED card (Embarkation/Disembarkation card).",
        sources=_get_sources(extracted.ed_card_requirement),
        critical_group=True,
    )

    # 5) Aruba ED card timing (critical) – depends on requirement
    await _verify_claim_group(
        evaluator,
        root,
        group_id="ed_card_timing_main",
        group_desc="Aruba ED card timing verification",
        claim_id="ed_card_timing",
        claim_desc="Specifies that the Aruba ED card can only be completed within 7 days prior to travel to Aruba",
        claim_text="The Aruba ED card can only be completed within 7 days prior to travel to Aruba.",
        sources=_get_sources(extracted.ed_card_timing),
        critical_group=True,
        extra_prereq_leaves=[ed_req["verify_leaf"]]
    )

    # 6) Airline from Nashville (Avelo) (non-critical)
    avelo_airline = await _verify_claim_group(
        evaluator,
        root,
        group_id="airline_from_nashville_main",
        group_desc="Avelo Airlines operations from Nashville verification",
        claim_id="airline_from_nashville",
        claim_desc="Identifies that Avelo Airlines operates flights from Nashville (BNA) to Florida destinations",
        claim_text="Avelo Airlines operates flights from Nashville (BNA) to Florida destinations.",
        sources=_get_sources(extracted.airline_from_nashville),
        critical_group=False,
    )

    # 7) Free personal item (non-critical, depends on Avelo airline identification)
    await _verify_claim_group(
        evaluator,
        root,
        group_id="free_personal_item_main",
        group_desc="Avelo free personal item policy verification",
        claim_id="free_personal_item",
        claim_desc="States that Avelo Airlines allows one free personal item",
        claim_text="Avelo Airlines allows one free personal item.",
        sources=_get_sources(extracted.free_personal_item),
        critical_group=False,
        extra_prereq_leaves=[avelo_airline["verify_leaf"]]
    )

    # 8) Personal item size (non-critical, depends on Avelo airline identification)
    await _verify_claim_group(
        evaluator,
        root,
        group_id="personal_item_size_main",
        group_desc="Avelo personal item size limit verification",
        claim_id="personal_item_size",
        claim_desc="Provides the maximum size for Avelo's free personal item: 17 inches L x 13 inches H x 9 inches W",
        claim_text="Avelo's free personal item maximum size is 17 inches (L) x 13 inches (H) x 9 inches (W).",
        sources=_get_sources(extracted.personal_item_size),
        critical_group=False,
        extra_prereq_leaves=[avelo_airline["verify_leaf"]]
    )

    # 9) Baggage fees (non-critical, depends on Avelo airline identification)
    await _verify_claim_group(
        evaluator,
        root,
        group_id="baggage_fees_main",
        group_desc="Avelo baggage fee policy verification",
        claim_id="baggage_fees",
        claim_desc="Mentions that Avelo Airlines charges fees for carry-on bags and checked bags (ranging from $40-60 depending on when purchased)",
        claim_text="Avelo Airlines charges fees for carry-on and checked bags, typically ranging from about $40 to $60 depending on when purchased.",
        sources=_get_sources(extracted.baggage_fees),
        critical_group=False,
        extra_prereq_leaves=[avelo_airline["verify_leaf"]]
    )

    # 10) Checked bag weight limit (non-critical, depends on Avelo airline identification)
    await _verify_claim_group(
        evaluator,
        root,
        group_id="checked_bag_weight_main",
        group_desc="Avelo checked bag weight limit verification",
        claim_id="checked_bag_weight",
        claim_desc="States that Avelo's checked bag weight limit is 50 lbs (22 kg)",
        claim_text="Avelo's checked bag weight limit is 50 lbs (22 kg).",
        sources=_get_sources(extracted.checked_bag_weight),
        critical_group=False,
        extra_prereq_leaves=[avelo_airline["verify_leaf"]]
    )

    # 11) Hotel proximity (non-critical)
    await _verify_claim_group(
        evaluator,
        root,
        group_id="hotel_proximity_main",
        group_desc="Hotel proximity to Port Everglades verification",
        claim_id="hotel_proximity",
        claim_desc="Mentions that hotels near Port Everglades cruise terminal are available within a few miles of the port",
        claim_text="Hotels near the Port Everglades cruise terminal are available within a few miles of the port.",
        sources=_get_sources(extracted.hotel_proximity),
        critical_group=False,
    )

    # 12) Hotel shuttle (non-critical)
    await _verify_claim_group(
        evaluator,
        root,
        group_id="hotel_shuttle_main",
        group_desc="Hotel shuttle service to Port Everglades verification",
        claim_id="hotel_shuttle",
        claim_desc="Notes that some hotels near the cruise port offer shuttle service to Port Everglades",
        claim_text="Some hotels near the cruise port offer shuttle service to Port Everglades.",
        sources=_get_sources(extracted.hotel_shuttle),
        critical_group=False,
    )

    # 13) February 2026 Disney Destiny departure dates from Fort Lauderdale (critical)
    feb_sources = _get_sources(extracted.february_departures)
    feb_dates = extracted.february_departures.dates if extracted.february_departures else []
    readable_dates = ", ".join(_first_n(feb_dates, 6)) if feb_dates else "at least one date in February 2026"
    await _verify_claim_group(
        evaluator,
        root,
        group_id="february_departure_main",
        group_desc="Disney Destiny February 2026 departures verification",
        claim_id="february_departure",
        claim_desc="Provides or references specific Disney Destiny departure dates available in February 2026 from Fort Lauderdale",
        claim_text=f"The sources show Disney Destiny departure date(s) in February 2026 from Fort Lauderdale (Port Everglades): {readable_dates}.",
        sources=feb_sources,
        critical_group=True,
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
    Build an evaluation tree for the Disney Destiny trip requirements task and return a structured summary.
    Note: The JSON root was marked critical, but since it contains both critical and non‑critical children,
    we initialize a non‑critical root to comply with tree constraints while still enforcing criticality at item level.
    """
    # Initialize evaluator (root as parallel, non-critical to allow mixed-critical children)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_requirements(),
        template_class=TripRequirementsExtraction,
        extraction_name="trip_requirements_extraction"
    )

    # Build verification tree according to the rubric
    await build_verification_tree(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()