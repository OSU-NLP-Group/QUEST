import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "copper_canyon_itinerary"
TASK_DESCRIPTION = """
Plan a 2-day adventure itinerary in the Copper Canyon region of Chihuahua, Mexico, starting from the town of Creel. The itinerary must include the following requirements:

Day 1 Requirements:
- Visit the Copper Canyon Adventure Park (Parque de Aventura Barrancas del Cobre)
- Experience the ZipRider, which is one of the longest ziplines in the world
- Provide the ZipRider's length specification and cost per person

Accommodation Requirement:
- Select a hotel to stay for night 1 that is located within 2 miles of the Copper Canyon Adventure Park
- Specify the hotel name and its distance from the Adventure Park

Day 2 Requirements:
- Visit at least 2 different natural attractions located near Creel
- For each attraction, provide: the attraction name, distance from Creel, and entrance fee (if applicable)

Budget Requirement:
- Calculate the total cost in Mexican pesos for one person covering all entrance fees and activity costs for the 2-day itinerary, including:
  - Copper Canyon Adventure Park entrance fee
  - ZipRider activity cost
  - Entrance fees for the 2 natural attractions visited on Day 2

For each component of the itinerary (Adventure Park, hotel, natural attractions), provide reference URLs that confirm the details.
"""

ALLOWED_ATTRACTIONS = [
    "Cusarare Falls",
    "Lake Arareko",
    "Valle de los Hongos",
    "Valley of the Mushrooms",
]
ALLOWED_HOTELS = [
    "Hotel El Mirador by Balderrama Hotel Collection",
    "Cabañas Darely",
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Day1Park(BaseModel):
    included: Optional[bool] = None
    urls: List[str] = Field(default_factory=list)


class ZipRiderInfo(BaseModel):
    included: Optional[bool] = None
    length_text: Optional[str] = None
    cost_text: Optional[str] = None
    longest_claim_stated: Optional[bool] = None
    urls: List[str] = Field(default_factory=list)


class HotelInfo(BaseModel):
    name: Optional[str] = None
    distance_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AttractionInfo(BaseModel):
    name: Optional[str] = None
    distance_text: Optional[str] = None
    entrance_fee_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class BudgetInfo(BaseModel):
    total_cost_mxn_text: Optional[str] = None
    includes_required_components_explicit: Optional[bool] = None
    line_items: List[str] = Field(default_factory=list)


class ItineraryExtraction(BaseModel):
    has_day_1_section: Optional[bool] = None
    has_day_2_section: Optional[bool] = None
    start_location_text: Optional[str] = None
    park: Optional[Day1Park] = None
    ziprider: Optional[ZipRiderInfo] = None
    hotel: Optional[HotelInfo] = None
    attractions: List[AttractionInfo] = Field(default_factory=list)
    budget: Optional[BudgetInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_itinerary() -> str:
    return """
    Extract structured information from the itinerary answer exactly as it appears. Do not invent or infer. If a field is missing, return null or an empty list as appropriate.

    1) Structure:
       - has_day_1_section: boolean indicating whether the answer uses an explicit "Day 1" (or equivalent, e.g., "Day One") section label.
       - has_day_2_section: boolean indicating whether the answer uses an explicit "Day 2" (or equivalent, e.g., "Day Two") section label.
       - start_location_text: the explicit text snippet that indicates the itinerary starts from the town of Creel.

    2) Day 1 – Adventure Park:
       - park.included: boolean indicating if visiting Copper Canyon Adventure Park (Parque de Aventura Barrancas del Cobre) is explicitly included.
       - park.urls: list of all URLs provided in the answer for the Adventure Park component (extract only actual URLs mentioned).

    3) Day 1 – ZipRider:
       - ziprider.included: boolean indicating if riding the ZipRider is explicitly included.
       - ziprider.length_text: the ZipRider length as stated in the answer (e.g., "2,545 meters" or "2.5 km").
       - ziprider.cost_text: the ZipRider cost per person as stated in the answer (e.g., "1,000 MXN").
       - ziprider.longest_claim_stated: boolean indicating if the answer states the ZipRider is one of the longest ziplines in the world.
       - ziprider.urls: list of all URLs provided specifically for ZipRider details.

    4) Accommodation (Night 1):
       - hotel.name: the chosen hotel name.
       - hotel.distance_text: the stated distance from the Adventure Park (as text, with units if present).
       - hotel.urls: list of all URLs provided to support the hotel's proximity/distance to the Adventure Park.

    5) Day 2 – Natural Attractions near Creel:
       Extract up to the first two natural attractions described in the answer (if more than two are present, take the first two).
       For each attraction:
       - attractions[i].name: the attraction name as stated.
       - attractions[i].distance_text: the stated distance from Creel (text with any units).
       - attractions[i].entrance_fee_text: the entrance fee text (numeric with currency, or "free"/"not applicable" if stated).
       - attractions[i].urls: list of URLs provided for that attraction.

    6) Budget:
       - budget.total_cost_mxn_text: the single total cost for one person in Mexican pesos (MXN) as stated (e.g., "Total: 2,300 MXN").
       - budget.includes_required_components_explicit: boolean indicating whether the answer explicitly states that the total includes all required components (Adventure Park entrance fee, ZipRider cost, and entrance fees for the two Day 2 attractions).
       - budget.line_items: a list of the line-item costs as they appear in the answer (each entry is a textual line, e.g., "ZipRider: 1,000 MXN").

    Return a JSON object following the ItineraryExtraction schema exactly.
    """


# --------------------------------------------------------------------------- #
# Helper functions for existence/format checks                                #
# --------------------------------------------------------------------------- #
def _has_numeric_with_units(text: Optional[str]) -> bool:
    if not text:
        return False
    pattern = r"(?i)\b(\d+(?:[.,]\d+)?)\s*(km|kilometers?|kms?|mi|miles?)\b"
    return re.search(pattern, text) is not None


def _has_fee_or_free(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.lower()
    if any(word in s for word in ["free", "no fee", "sin costo", "gratuito", "sin cargo", "n/a", "not applicable"]):
        return True
    money_pattern = r"(?i)\b(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d+)?)\s*(mxn|mx\$|mxn\$|pesos?)\b"
    return re.search(money_pattern, text) is not None


def _has_mxn_amount(text: Optional[str]) -> bool:
    if not text:
        return False
    pattern = r"(?i)\b(\d{1,3}(?:[,\s]?\d{3})*(?:\.\d+)?)\s*(mxn|mx\$|mxn\$|mexican\s*pesos|pesos)\b"
    return re.search(pattern, text) is not None


def _list_present(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_structure_and_start(evaluator: Evaluator, parent_node) -> None:
    node = evaluator.add_parallel(
        id="itinerary_structure_and_start",
        desc="Response is organized into Day 1 and Day 2 and explicitly states the start location is Creel.",
        parent=parent_node,
        critical=True,
    )
    # Leaf: Day sections present
    leaf_day_sections = evaluator.add_leaf(
        id="has_day_1_and_day_2_sections",
        desc="Includes distinct Day 1 and Day 2 sections (or equivalent explicit labeling).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes distinct sections labeled 'Day 1' and 'Day 2' or equivalent explicit labeling (e.g., 'Day One', 'Day Two', 'Día 1', 'Día 2').",
        node=leaf_day_sections,
        additional_instruction="Accept reasonable equivalent labels. Base judgment strictly on the provided answer text.",
    )

    # Leaf: Starts from Creel
    leaf_start_creel = evaluator.add_leaf(
        id="starts_from_creel",
        desc="Explicitly indicates the itinerary starts from the town of Creel.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The itinerary explicitly states that it starts from the town of Creel.",
        node=leaf_start_creel,
        additional_instruction="Look for phrasing like 'Starting from Creel', 'Departing from Creel', or equivalent.",
    )


async def verify_day1(evaluator: Evaluator, parent_node, info: ItineraryExtraction) -> None:
    node = evaluator.add_parallel(
        id="day_1_adventure_park_and_ziprider",
        desc="Day 1 includes Copper Canyon Adventure Park and the ZipRider, with required ZipRider length and price, plus reference URLs.",
        parent=parent_node,
        critical=True,
    )
    # Adventure Park sub-node
    park_node = evaluator.add_parallel(
        id="adventure_park_included_with_reference",
        desc="Day 1 includes visiting Copper Canyon Adventure Park (Parque de Aventura Barrancas del Cobre) and provides at least one reference URL for the park component.",
        parent=node,
        critical=True,
    )
    # Leaf: Park visit included
    leaf_park_included = evaluator.add_leaf(
        id="park_visit_included",
        desc="Day 1 explicitly includes visiting Copper Canyon Adventure Park.",
        parent=park_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Day 1 explicitly includes visiting Copper Canyon Adventure Park (Parque de Aventura Barrancas del Cobre).",
        node=leaf_park_included,
        additional_instruction="Allow Spanish naming; synonyms accepted if clearly referencing the park.",
    )

    # Leaf: Park reference URL presence (existence check)
    park_urls = _safe_list(info.park.urls if info.park else [])
    evaluator.add_custom_node(
        result=_list_present(park_urls),
        id="park_reference_url_present",
        desc="Provides ≥1 reference URL for the Adventure Park component.",
        parent=park_node,
        critical=True,
    )

    # ZipRider sub-node
    zr_node = evaluator.add_parallel(
        id="ziprider_details_with_reference",
        desc="Day 1 includes the ZipRider and provides its length and cost per person, plus at least one reference URL for ZipRider details.",
        parent=node,
        critical=True,
    )

    # Reference presence first (to gate other checks)
    zip_urls = _safe_list(info.ziprider.urls if info.ziprider else [])
    evaluator.add_custom_node(
        result=_list_present(zip_urls),
        id="ziprider_reference_url_present",
        desc="Provides ≥1 reference URL supporting the ZipRider length and pricing.",
        parent=zr_node,
        critical=True,
    )

    # Leaf: ZipRider included
    leaf_zr_included = evaluator.add_leaf(
        id="ziprider_included",
        desc="Day 1 explicitly includes doing the ZipRider.",
        parent=zr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Day 1 explicitly includes riding the ZipRider at the Adventure Park.",
        node=leaf_zr_included,
        additional_instruction="Accept equivalent phrasing like 'Zipline ZipRider' or 'ZipRider experience'.",
    )

    # Leaf: ZipRider length constraint (verify with URLs)
    leaf_zr_length = evaluator.add_leaf(
        id="ziprider_length_matches_constraint",
        desc="States ZipRider length as 2,545 meters (allowing equivalent unit conversion).",
        parent=zr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The ZipRider length is approximately 2,545 meters (about 2.5 km).",
        node=leaf_zr_length,
        sources=zip_urls,
        additional_instruction="Allow unit conversions and minor rounding. Confirm the length figure via the provided source(s).",
    )

    # Leaf: ZipRider cost constraint (verify with URLs)
    leaf_zr_cost = evaluator.add_leaf(
        id="ziprider_cost_matches_constraint",
        desc="States ZipRider cost per person as 1,000 MXN pesos.",
        parent=zr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The ZipRider costs 1,000 MXN per person.",
        node=leaf_zr_cost,
        sources=zip_urls,
        additional_instruction="Confirm the per-person price via the provided source(s). Accept formats like 'MXN 1000', 'MX$ 1000'.",
    )

    # Leaf: ZipRider 'one of the longest' claim (verify with URLs)
    leaf_zr_longest = evaluator.add_leaf(
        id="ziprider_longest_claim_stated",
        desc="States that ZipRider is one of the longest ziplines in the world.",
        parent=zr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The ZipRider in Copper Canyon is described as one of the longest ziplines in the world.",
        node=leaf_zr_longest,
        sources=zip_urls,
        additional_instruction="Check the phrasing on the source(s) that supports this claim.",
    )


async def verify_accommodation(evaluator: Evaluator, parent_node, info: ItineraryExtraction) -> None:
    node = evaluator.add_parallel(
        id="accommodation_night_1",
        desc="Selects a night-1 hotel within 2 miles of the Adventure Park, provides the hotel name, distance, and a reference URL.",
        parent=parent_node,
        critical=True,
    )
    hotel_name = (info.hotel.name if info.hotel else "") or ""
    hotel_urls = _safe_list(info.hotel.urls if info.hotel else [])

    # Leaf: Hotel name is one of allowed constraint hotels
    leaf_hotel_allowed = evaluator.add_leaf(
        id="hotel_is_one_of_constraint_hotels",
        desc="Hotel selected is either (a) Hotel El Mirador by Balderrama Hotel Collection or (b) Cabañas Darely.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The selected hotel '{hotel_name}' is either 'Hotel El Mirador by Balderrama Hotel Collection' or 'Cabañas Darely'.",
        node=leaf_hotel_allowed,
        additional_instruction="Allow minor naming variations or accents. Base strictly on the hotel explicitly chosen in the answer.",
    )

    # Reference presence (existence check)
    evaluator.add_custom_node(
        result=_list_present(hotel_urls),
        id="hotel_reference_url_present",
        desc="Provides ≥1 reference URL supporting the hotel’s proximity/distance to the Adventure Park.",
        parent=node,
        critical=True,
    )

    # Leaf: Hotel distance within 2 miles (verify with URLs)
    leaf_hotel_distance = evaluator.add_leaf(
        id="hotel_distance_within_2_miles",
        desc="States the hotel distance from the Adventure Park and the stated distance is ≤ 2 miles.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The selected hotel is within 2 miles (approximately ≤ 3.2 km) of Copper Canyon Adventure Park.",
        node=leaf_hotel_distance,
        sources=hotel_urls,
        additional_instruction="Confirm via the provided source(s). Allow kilometers and minor rounding; judge proximity within the threshold.",
    )


async def _verify_attraction(
    evaluator: Evaluator,
    parent_node,
    attr: AttractionInfo,
    idx: int,
    other_attr_name: Optional[str] = None,
) -> None:
    node = evaluator.add_parallel(
        id=f"attraction_{idx}",
        desc=("First natural attraction (from the constraint-defined set) with required details and reference(s)." if idx == 1
              else "Second natural attraction (from the constraint-defined set) with required details and reference(s), and distinct from attraction 1."),
        parent=parent_node,
        critical=True,
    )
    name = (attr.name or "").strip()
    dist_text = (attr.distance_text or "").strip()
    fee_text = (attr.entrance_fee_text or "").strip()
    urls = _safe_list(attr.urls)

    # Name from allowed set (+ distinctness for #2)
    if idx == 1:
        leaf_name_allowed = evaluator.add_leaf(
            id="attraction_1_name_from_allowed_set",
            desc="Attraction 1 is one of: Cusarare Falls, Lake Arareko, Valle de los Hongos (Valley of the Mushrooms).",
            parent=node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The first attraction '{name}' is one of: Cusarare Falls, Lake Arareko, Valle de los Hongos (Valley of the Mushrooms).",
            node=leaf_name_allowed,
            additional_instruction="Accept Spanish/English variants and minor spelling/diacritic variations.",
        )
    else:
        leaf_name_allowed_distinct = evaluator.add_leaf(
            id="attraction_2_name_from_allowed_set_and_distinct",
            desc="Attraction 2 is one of: Cusarare Falls, Lake Arareko, Valle de los Hongos; and it is different from attraction 1.",
            parent=node,
            critical=True,
        )
        other_name = (other_attr_name or "").strip()
        await evaluator.verify(
            claim=f"The second attraction '{name}' is one of: Cusarare Falls, Lake Arareko, Valle de los Hongos (Valley of the Mushrooms), and it is different from the first attraction '{other_name}'.",
            node=leaf_name_allowed_distinct,
            additional_instruction="Accept Spanish/English variants and minor spelling differences; ensure the two attractions are not the same.",
        )

    # Distance provided (existence/format check)
    evaluator.add_custom_node(
        result=_has_numeric_with_units(dist_text),
        id=f"attraction_{idx}_distance_provided",
        desc=f"Provides a numeric distance from Creel to attraction {idx} with units.",
        parent=node,
        critical=True,
    )

    # Entrance fee provided (existence/format check)
    evaluator.add_custom_node(
        result=_has_fee_or_free(fee_text),
        id=f"attraction_{idx}_entrance_fee_provided",
        desc=f"Provides entrance fee info for attraction {idx} (numeric fee, or explicitly states free/not applicable).",
        parent=node,
        critical=True,
    )

    # Reference URL presence (existence check)
    evaluator.add_custom_node(
        result=_list_present(urls),
        id=f"attraction_{idx}_reference_url_present",
        desc=f"Provides ≥1 reference URL supporting attraction {idx} distance/location context and fee info (if any).",
        parent=node,
        critical=True,
    )


async def verify_day2(evaluator: Evaluator, parent_node, info: ItineraryExtraction) -> None:
    node = evaluator.add_parallel(
        id="day_2_two_natural_attractions_near_creel",
        desc="Day 2 includes at least two different natural attractions near Creel; for each provides name, distance from Creel, entrance fee (or explicitly states none), and reference URL(s).",
        parent=parent_node,
        critical=True,
    )
    # Prepare two attractions (pad if fewer)
    a1 = info.attractions[0] if len(info.attractions) >= 1 else AttractionInfo()
    a2 = info.attractions[1] if len(info.attractions) >= 2 else AttractionInfo()

    await _verify_attraction(evaluator, node, a1, idx=1, other_attr_name=None)
    await _verify_attraction(evaluator, node, a2, idx=2, other_attr_name=a1.name or "")


async def verify_budget(evaluator: Evaluator, parent_node, info: ItineraryExtraction) -> None:
    node = evaluator.add_parallel(
        id="budget_total_cost_mxn",
        desc="Calculates the total cost in MXN for one person, including park entrance, ZipRider, and the two Day 2 attraction entrance fees (if applicable).",
        parent=parent_node,
        critical=True,
    )
    total_text = (info.budget.total_cost_mxn_text if info.budget else "") or ""

    # Existence: single total MXN value provided
    evaluator.add_custom_node(
        result=_has_mxn_amount(total_text),
        id="total_cost_value_in_mxn_provided",
        desc="Provides a single total cost value in Mexican pesos (MXN) for one person.",
        parent=node,
        critical=True,
    )

    # Total includes required components (simple verify against answer)
    leaf_includes_components = evaluator.add_leaf(
        id="budget_includes_required_components",
        desc="Total explicitly includes: Adventure Park entrance fee, ZipRider cost, and entrance fees for both Day 2 attractions (if applicable).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The stated total explicitly includes the Adventure Park entrance fee, the ZipRider cost, and the entrance fees for the two Day 2 attractions (if applicable).",
        node=leaf_includes_components,
        additional_instruction="Base your judgment on explicit statements in the answer; synonyms and clear equivalence are acceptable.",
    )

    # Arithmetic consistency (simple verify against answer)
    leaf_arithmetic = evaluator.add_leaf(
        id="budget_arithmetic_consistent_with_stated_line_items",
        desc="The stated total equals the sum of the stated line-item costs included in the total.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The stated total equals (or is consistent with) the sum of the stated line-item costs included in the total.",
        node=leaf_arithmetic,
        additional_instruction="Check all listed line items (park entrance, ZipRider, two attractions) add up to the stated total; allow minor rounding differences.",
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
    Evaluate the Copper Canyon 2-day itinerary answer using the Mind2Web2 framework.
    Returns a structured summary including the verification tree and final score.
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

    # Extract structured info from the answer
    info: ItineraryExtraction = await evaluator.extract(
        prompt=prompt_extract_itinerary(),
        template_class=ItineraryExtraction,
        extraction_name="itinerary_extraction",
    )

    # Build verification tree according to rubric (all critical under critical parents)
    await verify_structure_and_start(evaluator, root)
    await verify_day1(evaluator, root, info)
    await verify_accommodation(evaluator, root, info)
    await verify_day2(evaluator, root, info)
    await verify_budget(evaluator, root, info)

    # Add custom info to summary (allowed sets for reference)
    evaluator.add_custom_info(
        {"allowed_hotels": ALLOWED_HOTELS, "allowed_attractions": ALLOWED_ATTRACTIONS},
        info_type="constraints",
        info_name="allowed_items",
    )

    return evaluator.get_summary()