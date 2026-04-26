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
TASK_ID = "yellowstone_winter_lodge"
TASK_DESCRIPTION = (
    "Yellowstone National Park operates only two lodges during the winter season. "
    "Identify which of these winter lodges is accessible by personal vehicle year-round through the north entrance. "
    "For this lodge, provide: (1) its winter season 2025-2026 opening date and closing date, "
    "(2) information about the winter-only shuttle service that operates from Bozeman-Yellowstone International Airport to this lodge, "
    "and (3) the URL to the official Yellowstone National Park Lodges winter lodging page that contains this information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WinterLodgeExtraction(BaseModel):
    # Identification of the correct lodge
    vehicle_accessible_lodge: Optional[str] = None

    # Winter season dates for 2025-2026
    opening_date_2025_2026: Optional[str] = None
    closing_date_2025_2026: Optional[str] = None

    # Official page(s) cited in the answer
    official_winter_lodging_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)

    # Shuttle info as stated in the answer
    shuttle_origin: Optional[str] = None
    shuttle_destination: Optional[str] = None
    shuttle_frequency: Optional[str] = None
    shuttle_seasonality: Optional[str] = None
    shuttle_description: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_winter_lodge_info() -> str:
    return """
    Extract from the answer the specific information requested about Yellowstone's winter lodges.

    Required fields (return all fields even if null):
    1) vehicle_accessible_lodge: The name of the winter lodge that is accessible by personal vehicle year-round via the North Entrance (e.g., "Mammoth Hot Springs Hotel").
    2) opening_date_2025_2026: The winter 2025–2026 opening date for the identified lodge, as written in the answer (keep formatting as-is).
    3) closing_date_2025_2026: The winter 2025–2026 closing date for the identified lodge, as written in the answer (keep formatting as-is).
    4) official_winter_lodging_url: The URL to the official Yellowstone National Park Lodges winter lodging page that the answer cites as the source of these details.
    5) additional_urls: Any other official URLs cited in the answer that provide relevant winter lodging or airport shuttle information.
    6) shuttle_origin: The origin location stated in the answer for the airport shuttle (e.g., "Bozeman-Yellowstone International Airport", "Bozeman", or similar).
    7) shuttle_destination: The destination for the shuttle as stated in the answer (ideally the lodge identified in #1).
    8) shuttle_frequency: The shuttle frequency as stated in the answer (e.g., "daily").
    9) shuttle_seasonality: The seasonality as stated in the answer (e.g., "winter-only", "winter season only").
    10) shuttle_description: A brief textual description of the shuttle service as stated in the answer.

    IMPORTANT:
    - Extract ONLY what is explicitly present in the answer text.
    - For URLs, apply the SPECIAL RULES FOR URL SOURCES EXTRACTION:
      * The URL(s) must be explicitly present in the answer. If absent, return null or an empty array as appropriate.
      * Accept plain URLs or markdown links but extract the actual URL.
      * If a URL is missing a protocol, prepend http:// as needed.
    - If any field is not present in the answer, set it to null (or an empty list for additional_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _lower(s: Optional[str]) -> str:
    return _norm(s).lower()


def _non_empty(s: Optional[str]) -> bool:
    return bool(_norm(s))


def _gather_sources(extracted: WinterLodgeExtraction) -> List[str]:
    urls: List[str] = []
    if _non_empty(extracted.official_winter_lodging_url):
        urls.append(_norm(extracted.official_winter_lodging_url))
    if extracted.additional_urls:
        for u in extracted.additional_urls:
            if _non_empty(u):
                urls.append(_norm(u))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _shuttle_required_elements_present(extracted: WinterLodgeExtraction) -> Dict[str, bool]:
    """
    Check if the answer provided the required shuttle elements:
    - origin mentions Bozeman (ideally Bozeman-Yellowstone International Airport)
    - destination matches or contains the lodge name
    - frequency mentions 'daily' (or obvious synonyms)
    - seasonality mentions 'winter'
    """
    origin_l = _lower(extracted.shuttle_origin)
    dest_l = _lower(extracted.shuttle_destination)
    freq_l = _lower(extracted.shuttle_frequency)
    seas_l = _lower(extracted.shuttle_seasonality)
    lodge_l = _lower(extracted.vehicle_accessible_lodge)

    origin_ok = ("bozeman" in origin_l) or ("bozeman-yellowstone international airport" in origin_l) or ("bzn" in origin_l)
    dest_ok = False
    if lodge_l:
        dest_ok = (lodge_l in dest_l) or (dest_l in lodge_l) or ("mammoth" in dest_l and "mammoth" in lodge_l)
    else:
        dest_ok = _non_empty(extracted.shuttle_destination)

    # Consider common ways to express daily
    freq_ok = ("daily" in freq_l) or ("every day" in freq_l) or ("7 days" in freq_l)
    # Seasonality mentions winter
    season_ok = ("winter" in seas_l)

    return {
        "origin_ok": origin_ok,
        "destination_ok": dest_ok,
        "frequency_ok": freq_ok,
        "seasonality_ok": season_ok,
    }


# --------------------------------------------------------------------------- #
# Verification tree building                                                  #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: WinterLodgeExtraction) -> None:
    """
    Build the verification tree based on the rubric. This function creates:
    - Root child: complete_lodge_information (parallel, critical)
      - vehicle_accessible_lodge_identification (sequential, critical)
          - existence
          - support by official URL(s)
      - winter_season_dates (sequential, critical)
          - existence
          - support by official URL(s)
      - bozeman_shuttle_service (sequential, critical)
          - completeness existence (daily + winter-only + origin + destination)
          - support by official URL(s)
      - official_source_url (sequential, critical)
          - existence (URL provided)
          - verification that it is the official winter lodging page containing info
    """
    # Create the top-level critical node
    complete_node = evaluator.add_parallel(
        id="complete_lodge_information",
        desc="Complete information about the vehicle-accessible Yellowstone winter lodge including identification, dates, shuttle service, and official source",
        parent=evaluator.root,
        critical=True
    )

    # Collect sources (official URL + any additional official URLs the answer provided)
    sources_list = _gather_sources(extracted)

    # 1) Vehicle-accessible lodge identification
    vehicle_node = evaluator.add_sequential(
        id="vehicle_accessible_lodge_identification",
        desc="Correctly identified which of the two winter lodges is accessible by personal vehicle year-round via the north entrance",
        parent=complete_node,
        critical=True
    )

    # Existence check: the answer must explicitly identify the lodge
    evaluator.add_custom_node(
        result=_non_empty(extracted.vehicle_accessible_lodge),
        id="vehicle_accessible_lodge_identification_exists",
        desc="Answer identifies the winter lodge accessible by personal vehicle via the North Entrance",
        parent=vehicle_node,
        critical=True
    )

    # Verification leaf against official source(s)
    lodge_identification_leaf = evaluator.add_leaf(
        id="vehicle_accessible_lodge_identification_supported",
        desc="The identified lodge is indeed accessible by personal vehicle year-round via the North Entrance",
        parent=vehicle_node,
        critical=True
    )

    lodge_name = _norm(extracted.vehicle_accessible_lodge)
    lodge_identification_claim = (
        f"{lodge_name} is the Yellowstone winter lodge that is accessible by personal vehicle year-round via the North Entrance."
    )
    await evaluator.verify(
        claim=lodge_identification_claim,
        node=lodge_identification_leaf,
        sources=sources_list,
        additional_instruction=(
            "Confirm on the official Yellowstone National Park Lodges site that among the two winter lodges, "
            "this specific lodge is accessible by personal vehicle year-round via the North Entrance."
        ),
    )

    # 2) Winter season dates (2025-2026)
    dates_node = evaluator.add_sequential(
        id="winter_season_dates",
        desc="Provided correct winter season 2025-2026 opening date and closing date for the vehicle-accessible lodge",
        parent=complete_node,
        critical=True
    )

    # Existence check: both opening and closing dates must be present in the answer
    evaluator.add_custom_node(
        result=_non_empty(extracted.opening_date_2025_2026) and _non_empty(extracted.closing_date_2025_2026),
        id="winter_season_dates_provided",
        desc="Answer provides both winter 2025–2026 opening and closing dates for the identified lodge",
        parent=dates_node,
        critical=True
    )

    # Verification leaf for dates
    dates_leaf = evaluator.add_leaf(
        id="winter_season_dates_supported",
        desc="The winter 2025–2026 opening and closing dates match the official information",
        parent=dates_node,
        critical=True
    )
    opening_str = _norm(extracted.opening_date_2025_2026)
    closing_str = _norm(extracted.closing_date_2025_2026)
    dates_claim = (
        f"For {lodge_name}, the winter season 2025–2026 opening date is '{opening_str}' and the closing date is '{closing_str}'."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_leaf,
        sources=sources_list,
        additional_instruction=(
            "Verify the season dates for winter 2025–2026 for the specified lodge on the official winter lodging page. "
            "Allow minor formatting variations (e.g., 'Dec' vs 'December'). Both opening and closing dates must match."
        ),
    )

    # 3) Bozeman shuttle service (winter-only daily shuttle from BZN to the lodge)
    shuttle_node = evaluator.add_sequential(
        id="bozeman_shuttle_service",
        desc="Provided information about the winter-only daily shuttle service that operates from Bozeman-Yellowstone International Airport to the vehicle-accessible lodge",
        parent=complete_node,
        critical=True
    )

    # Existence/completeness check for shuttle details in the answer
    shuttle_flags = _shuttle_required_elements_present(extracted)
    shuttle_existence_result = all(shuttle_flags.values())
    evaluator.add_custom_node(
        result=shuttle_existence_result,
        id="bozeman_shuttle_service_provided",
        desc="Answer includes shuttle origin (BZN/Bozeman), destination (this lodge), frequency (daily), and seasonality (winter)",
        parent=shuttle_node,
        critical=True
    )
    # Optional: record these flags for debugging
    evaluator.add_custom_info({"shuttle_checks": shuttle_flags}, info_type="debug", info_name="shuttle_extraction_checks")

    # Verification leaf for shuttle details
    shuttle_leaf = evaluator.add_leaf(
        id="bozeman_shuttle_service_supported",
        desc="The winter-only daily shuttle from Bozeman-Yellowstone International Airport to the identified lodge is supported by the official source",
        parent=shuttle_node,
        critical=True
    )

    shuttle_claim = (
        f"There is a winter-only daily shuttle service that operates from Bozeman-Yellowstone International Airport to {lodge_name}."
    )
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_leaf,
        sources=sources_list,
        additional_instruction=(
            "On the official site, confirm that an airport shuttle is offered during the winter season, operates daily, "
            "and runs between Bozeman-Yellowstone International Airport (BZN) and the named lodge."
        ),
    )

    # 4) Official source URL
    official_node = evaluator.add_sequential(
        id="official_source_url",
        desc="Provided URL to the official Yellowstone National Park Lodges winter lodging page",
        parent=complete_node,
        critical=True
    )

    # Existence check: the answer must include an official winter lodging page URL
    official_url = _norm(extracted.official_winter_lodging_url)
    evaluator.add_custom_node(
        result=_non_empty(official_url),
        id="official_source_url_provided",
        desc="Answer includes a URL to the official Yellowstone National Park Lodges winter lodging page",
        parent=official_node,
        critical=True
    )

    # Verification leaf: ensure the provided page is official and contains the requested info
    official_leaf = evaluator.add_leaf(
        id="official_source_url_supported",
        desc="The provided URL is an official winter lodging page and contains the dates and shuttle information for the lodge",
        parent=official_node,
        critical=True
    )

    official_claim = (
        f"This webpage is on the official Yellowstone National Park Lodges website and provides winter lodging information "
        f"for {lodge_name}, including winter 2025–2026 season dates and airport shuttle details."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_leaf,
        sources=official_url if _non_empty(official_url) else None,
        additional_instruction=(
            "Verify that the URL is part of the official Yellowstone National Park Lodges domain and that it includes winter lodging information for the lodge, "
            "including both the winter 2025–2026 open/close dates and the airport shuttle details."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Yellowstone winter lodges task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container
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
        prompt=prompt_extract_winter_lodge_info(),
        template_class=WinterLodgeExtraction,
        extraction_name="winter_lodge_extraction",
    )

    # Build and execute verification tree according to rubric
    await build_verification_tree(evaluator, extracted)

    # Return the structured summary with verification tree and scores
    return evaluator.get_summary()