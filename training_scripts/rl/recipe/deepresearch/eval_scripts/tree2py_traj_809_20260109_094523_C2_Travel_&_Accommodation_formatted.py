import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "airport_hotel_leed_highest"
TASK_DESCRIPTION = (
    "Which airport hotel at a major U.S. airport holds the highest LEED certification rating? "
    "Please provide the hotel's name and airport location, confirm its LEED certification level, "
    "verify that it is the highest-rated among major U.S. airport hotels, and describe its direct rail transit "
    "connectivity to downtown and the size of its conference/meeting space."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ClaimSources(BaseModel):
    """Categorized source URLs explicitly mentioned in the answer."""
    hotel_urls: List[str] = Field(default_factory=list)
    airport_urls: List[str] = Field(default_factory=list)
    leed_urls: List[str] = Field(default_factory=list)
    ranking_urls: List[str] = Field(default_factory=list)
    transit_urls: List[str] = Field(default_factory=list)
    conference_urls: List[str] = Field(default_factory=list)


class AirportHotelExtraction(BaseModel):
    """Core structured information extracted from the answer."""
    hotel_name: Optional[str] = None
    airport_name: Optional[str] = None
    airport_code: Optional[str] = None
    city: Optional[str] = None

    leed_level: Optional[str] = None

    transit_description: Optional[str] = None
    transit_mode_name: Optional[str] = None   # e.g., "MARTA", "CTA Blue Line", "BART"
    transit_directness: Optional[str] = None  # e.g., "direct one-seat ride", "via people mover to rail"

    conference_space: Optional[str] = None    # e.g., "32,000 sq ft", "3,000 sqm"
    conference_units: Optional[str] = None    # e.g., "sq ft", "sqm"

    sources: ClaimSources = Field(default_factory=ClaimSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airport_hotel() -> str:
    return """
    Extract the key facts provided in the answer about the airport hotel with the highest LEED rating.

    Return a JSON object with the following fields:
    - hotel_name: The hotel's name as stated in the answer.
    - airport_name: The full airport name as stated in the answer (e.g., "Hartsfield-Jackson Atlanta International Airport").
    - airport_code: The airport IATA code if provided (e.g., "ATL", "SFO"); otherwise null.
    - city: The city associated with the airport/hotel if mentioned (e.g., "Atlanta", "San Francisco"); otherwise null.
    - leed_level: The hotel's LEED certification level as explicitly stated in the answer (e.g., "LEED Platinum", "Platinum"); otherwise null.

    Transit:
    - transit_description: The description in the answer about direct rail transit from the airport/hotel to downtown (free text).
    - transit_mode_name: The name of the rail system or line if provided (e.g., "MARTA", "CTA Blue Line", "BART"); otherwise null.
    - transit_directness: A short phrase summarizing directness if stated (e.g., "direct one-seat ride", "via people mover to rail"); otherwise null.

    Conference/Meeting Space:
    - conference_space: The total conference/meeting space size including the numeric value and units as stated in the answer (e.g., "32,000 sq ft", "3,000 sqm"); otherwise null.
    - conference_units: The units corresponding to the stated size (e.g., "sq ft", "square feet", "sqm", "m²"); otherwise null.

    Sources:
    - sources: An object containing arrays of URLs explicitly cited in the answer.
        * hotel_urls: URLs about the hotel (official site, press releases, fact sheets)
        * airport_urls: URLs about the airport (official site, Wikipedia, etc.)
        * leed_urls: URLs specifically supporting the LEED certification level (USGBC directory pages or credible articles)
        * ranking_urls: URLs supporting the claim that this hotel is the highest LEED-rated among major U.S. airport hotels
        * transit_urls: URLs supporting direct rail transit connectivity to downtown
        * conference_urls: URLs supporting the conference/meeting space size

    RULES:
    - Extract only what is explicitly present in the answer. If a field is not present, return null.
    - For URLs, return only valid, complete URLs explicitly mentioned in the answer (plain or markdown).
    - Do not invent or infer data or URLs beyond the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip()


def combine_sources(*source_groups: List[str]) -> List[str]:
    combined: List[str] = []
    for group in source_groups:
        for url in group:
            u = (url or "").strip()
            if u and u not in combined:
                combined.append(u)
    return combined


def parse_area_to_sqft(area_text: Optional[str], units_hint: Optional[str]) -> Optional[float]:
    """
    Attempt to parse an area expression into square feet.
    Supports common units: sq ft, ft², square feet; sqm, m², square meters.
    Returns None if parsing fails.
    """
    if not area_text:
        return None

    text = area_text.lower()
    units = (units_hint or "").lower()

    # Find first number (allow commas)
    m = re.search(r"([\d,.]+)", text)
    if not m:
        return None

    try:
        num = float(m.group(1).replace(",", ""))
    except Exception:
        return None

    # Detect units
    is_sqft = any(u in text for u in ["sq ft", "sqft", "ft²", "square feet"]) or "sq ft" in units or "square feet" in units or "ft²" in units
    is_sqm = any(u in text for u in ["sqm", "m²", "square meter", "square meters"]) or "sqm" in units or "m²" in units

    if is_sqft and not is_sqm:
        return num
    if is_sqm and not is_sqft:
        return num * 10.7639
    # If units unclear, try units_hint
    if units:
        if "sq ft" in units or "square feet" in units or "ft²" in units:
            return num
        if "sqm" in units or "m²" in units or "square meter" in units:
            return num * 10.7639

    # Fallback: If text contains "ft" assume feet; if contains "m" assume meters
    if "ft" in text:
        return num
    if "m" in text:
        return num * 10.7639

    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_identify_hotel_and_airport(
    evaluator: Evaluator,
    parent_node,
    ex: AirportHotelExtraction
) -> None:
    """
    Build and verify the 'Identify_Hotel_and_Airport' parallel critical node.
    """
    node = evaluator.add_parallel(
        id="Identify_Hotel_and_Airport",
        desc="The hotel and its airport location are clearly identified.",
        parent=parent_node,
        critical=True
    )

    # Hotel name provided (existence check)
    hotel_name_ok = bool(_normalize_text(ex.hotel_name))
    evaluator.add_custom_node(
        result=hotel_name_ok,
        id="Hotel_Name_Provided",
        desc="Provides the hotel's name.",
        parent=node,
        critical=True
    )

    # Airport location provided (existence check: name or code)
    airport_ok = bool(_normalize_text(ex.airport_name) or _normalize_text(ex.airport_code))
    evaluator.add_custom_node(
        result=airport_ok,
        id="Airport_Location_Provided",
        desc="Provides the airport location (airport name and/or code) where the hotel is located.",
        parent=node,
        critical=True
    )

    # Major U.S. airport confirmed (verification)
    airport_display = ""
    if _normalize_text(ex.airport_name) and _normalize_text(ex.airport_code):
        airport_display = f"{ex.airport_name} ({ex.airport_code})"
    elif _normalize_text(ex.airport_name):
        airport_display = ex.airport_name
    elif _normalize_text(ex.airport_code):
        airport_display = ex.airport_code

    major_airport_leaf = evaluator.add_leaf(
        id="Major_US_Airport_Confirmed",
        desc="The hotel is located at a major U.S. airport (as required by the constraints).",
        parent=node,
        critical=True
    )

    claim_major = f"{airport_display} is a major U.S. airport."
    await evaluator.verify(
        claim=claim_major,
        node=major_airport_leaf,
        sources=combine_sources(ex.sources.airport_urls, ex.sources.hotel_urls),
        additional_instruction=(
            "Consider an airport 'major' if it is the primary airport for a major U.S. city, "
            "has substantial passenger volumes, or is widely recognized as a major/international hub. "
            "Look for language such as 'primary airport', 'international airport', 'busiest', or similar."
        )
    )


async def add_verify_leed_and_ranking(
    evaluator: Evaluator,
    parent_node,
    ex: AirportHotelExtraction
) -> None:
    """
    Build and verify the 'Verify_LEED_And_Ranking' parallel critical node.
    """
    node = evaluator.add_parallel(
        id="Verify_LEED_And_Ranking",
        desc="Verifies the hotel's LEED certification level and that it is the highest-rated among major U.S. airport hotels.",
        parent=parent_node,
        critical=True
    )

    # LEED Platinum certification
    leed_leaf = evaluator.add_leaf(
        id="LEED_Platinum_Certification",
        desc="Confirms the hotel holds LEED Platinum certification.",
        parent=node,
        critical=True
    )

    hotel_display = _normalize_text(ex.hotel_name) or "the hotel"
    claim_leed = f"{hotel_display} holds LEED Platinum certification (Platinum is the highest LEED level)."
    await evaluator.verify(
        claim=claim_leed,
        node=leed_leaf,
        sources=combine_sources(ex.sources.leed_urls, ex.sources.hotel_urls),
        additional_instruction=(
            "Verify explicit statements that the hotel achieved LEED Platinum. "
            "Accept synonyms like 'LEED Platinum', 'Platinum-level LEED certification', including LEED v4/v4.1 where applicable. "
            "Do not accept statements indicating Gold, Silver, or Certified instead of Platinum."
        )
    )

    # Highest-rated among major U.S. airport hotels
    highest_leaf = evaluator.add_leaf(
        id="Highest_Rated_Among_Major_US_Airport_Hotels",
        desc="Confirms the hotel is the highest LEED-rated among airport hotels at major U.S. airports.",
        parent=node,
        critical=True
    )

    claim_highest = (
        f"Among airport hotels at major U.S. airports, {hotel_display} has the highest LEED certification rating."
    )
    await evaluator.verify(
        claim=claim_highest,
        node=highest_leaf,
        sources=combine_sources(ex.sources.ranking_urls, ex.sources.leed_urls, ex.sources.hotel_urls),
        additional_instruction=(
            "LEED levels are, from lowest to highest: Certified, Silver, Gold, Platinum. "
            "Confirm that this hotel is the highest-rated among airport hotels at major U.S. airports. "
            "Accept credible sources explicitly stating it is 'the only' or 'the first' LEED Platinum airport hotel in the U.S., "
            "or otherwise supporting that no other major U.S. airport hotel has a higher or equal LEED rating."
        )
    )


async def add_transit_and_conference_space(
    evaluator: Evaluator,
    parent_node,
    ex: AirportHotelExtraction
) -> None:
    """
    Build and verify the 'Transit_And_Conference_Space' parallel critical node.
    """
    node = evaluator.add_parallel(
        id="Transit_And_Conference_Space",
        desc="Describes required transit connectivity and conference/meeting space size.",
        parent=parent_node,
        critical=True
    )

    # Direct rail transit to downtown
    transit_leaf = evaluator.add_leaf(
        id="Direct_Rail_Transit_To_Downtown",
        desc="Describes direct rail transit connectivity from the hotel/airport location to the city's downtown area.",
        parent=node,
        critical=True
    )

    airport_display = ""
    if _normalize_text(ex.airport_name) and _normalize_text(ex.airport_code):
        airport_display = f"{ex.airport_name} ({ex.airport_code})"
    elif _normalize_text(ex.airport_name):
        airport_display = ex.airport_name
    elif _normalize_text(ex.airport_code):
        airport_display = ex.airport_code

    city_display = _normalize_text(ex.city) or "the city's downtown"
    transit_mode = _normalize_text(ex.transit_mode_name) or "the city's rail system"
    directness = _normalize_text(ex.transit_directness) or "a direct rail connection"

    claim_transit = (
        f"There is {directness} from {airport_display} to {city_display} via {transit_mode}, "
        f"as described for {hotel_display}."
    )
    await evaluator.verify(
        claim=claim_transit,
        node=transit_leaf,
        sources=combine_sources(ex.sources.transit_urls, ex.sources.airport_urls, ex.sources.hotel_urls),
        additional_instruction=(
            "Confirm that a rail transit line directly connects the airport to downtown (e.g., one-seat ride on a metro/rail line), "
            "or that the hotel is directly linked to the airport rail station by walkway/people mover enabling a direct rail trip to downtown. "
            "Prefer official transit authority, airport pages, or the hotel's official site."
        )
    )

    # Conference space size and threshold (>30,000 sq ft)
    conference_leaf = evaluator.add_leaf(
        id="Conference_Space_Size_And_Threshold",
        desc="States the conference/meeting space size (with units) and it exceeds 30,000 square feet.",
        parent=node,
        critical=True
    )

    # Build claim using extracted value if present; additionally add numeric interpretation in instruction
    conf_val_text = _normalize_text(ex.conference_space)
    conf_units = _normalize_text(ex.conference_units)
    sqft_val = parse_area_to_sqft(conf_val_text, conf_units)

    if conf_val_text:
        claim_conference = (
            f"The hotel's total conference/meeting space is {conf_val_text} as stated, and it exceeds 30,000 square feet."
        )
    else:
        claim_conference = (
            "The hotel's total conference/meeting space exceeds 30,000 square feet."
        )

    add_ins = (
        "Confirm the hotel's total conference/meeting space size on official sources (hotel site, fact sheet) or credible references. "
        "If the value is provided in square meters, convert to square feet (1 sqm ≈ 10.7639 sq ft). "
        "The requirement is that the total exceeds 30,000 sq ft."
    )
    # Record parsed numeric info for transparency (optional)
    evaluator.add_custom_info(
        info={
            "extracted_conference_space_text": conf_val_text or "null",
            "extracted_conference_units": conf_units or "null",
            "parsed_square_feet": sqft_val if sqft_val is not None else "unparsed",
            "threshold_sqft": 30000
        },
        info_type="parsed_conference_space",
        info_name="conference_space_parsing"
    )

    await evaluator.verify(
        claim=claim_conference,
        node=conference_leaf,
        sources=combine_sources(ex.sources.conference_urls, ex.sources.hotel_urls),
        additional_instruction=add_ins
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
    Evaluate an answer to the airport hotel LEED highest-rating task.
    Constructs a sequential critical tree:
      1) Identify hotel and airport
      2) Verify LEED level and ranking
      3) Transit connectivity and conference space
    """
    # Initialize evaluator (root is created as non-critical internally; we will mark child groups as critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    extraction: AirportHotelExtraction = await evaluator.extract(
        prompt=prompt_extract_airport_hotel(),
        template_class=AirportHotelExtraction,
        extraction_name="airport_hotel_highest_leed_extraction"
    )

    # Build verification tree: three critical sequential steps
    # Step 1: Identify hotel and airport
    await add_identify_hotel_and_Airport := add_identify_hotel_and_airport(evaluator, root, extraction)

    # Step 2: Verify LEED and ranking
    await add_verify_leed_and_ranking(evaluator, root, extraction)

    # Step 3: Transit connectivity and conference space
    await add_transit_and_conference_space(evaluator, root, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()