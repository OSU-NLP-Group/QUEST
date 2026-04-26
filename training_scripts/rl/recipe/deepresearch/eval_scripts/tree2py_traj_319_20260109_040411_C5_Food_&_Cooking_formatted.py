import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fine_dining_ca_2023"
TASK_DESCRIPTION = (
    "Identify a fine dining restaurant that meets ALL of the following criteria: "
    "(1) The restaurant opened in 2023, (2) The restaurant is located in California, "
    "(3) The restaurant has a seating capacity between 30 and 40 seats (inclusive), "
    "(4) The chef or owner previously worked at a Michelin-starred restaurant, "
    "(5) The restaurant serves a tasting menu format, "
    "(6) The restaurant emphasizes specific culinary techniques such as curing, drying, fermentation, or pickling. "
    "Provide the restaurant's name, complete street address, specific opening date (month and day), current seating capacity, "
    "the chef's name, and the name of at least one Michelin-starred restaurant where the chef previously worked. "
    "Include verifiable URL references for all key information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RestaurantExtraction(BaseModel):
    # Core details (prefer strings for robustness)
    name: Optional[str] = None
    classification: Optional[str] = None  # e.g., "fine dining", "high-end"
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    opening_date_text: Optional[str] = None  # e.g., "March 15, 2023" or "2023-03-15"
    opening_month: Optional[str] = None      # "March" or "03"
    opening_day: Optional[str] = None        # "15"
    opening_year: Optional[str] = None       # "2023"

    seating_capacity: Optional[str] = None   # e.g., "35", "35 seats", "around 35"
    chef_name: Optional[str] = None
    michelin_restaurants: List[str] = Field(default_factory=list)  # restaurants where chef worked
    tasting_menu_format: Optional[str] = None  # e.g., "tasting menu", "multi-course tasting"
    techniques_emphasis: List[str] = Field(default_factory=list)   # e.g., ["fermentation", "curing"]

    # Source URLs explicitly mentioned in the answer (per-field + general)
    name_sources: List[str] = Field(default_factory=list)
    classification_sources: List[str] = Field(default_factory=list)
    address_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)
    opening_date_sources: List[str] = Field(default_factory=list)
    capacity_sources: List[str] = Field(default_factory=list)
    chef_sources: List[str] = Field(default_factory=list)
    michelin_sources: List[str] = Field(default_factory=list)
    tasting_menu_sources: List[str] = Field(default_factory=list)
    techniques_sources: List[str] = Field(default_factory=list)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurant() -> str:
    return """
    Extract details for ONE restaurant described in the answer that is intended to meet the specified criteria.
    Return a single JSON object with the following fields. If a field is missing in the answer, set it to null (for strings) or an empty list (for arrays).

    Core details:
    - name: The restaurant's name (string).
    - classification: Any explicit classification or description indicating the restaurant is fine dining or high-end (string).
    - address: A complete street address if provided (string).
    - city: City name (string).
    - state: State name or abbreviation (string).
    - zip_code: ZIP/postal code (string).

    Opening date:
    - opening_date_text: The opening date as mentioned (string, include month and day if present).
    - opening_month: The opening month (string, e.g., "March" or "03").
    - opening_day: The opening day number (string, e.g., "15").
    - opening_year: The opening year (string, expect "2023").

    Seating capacity:
    - seating_capacity: The stated current seating capacity number as text (string, e.g., "35", "35 seats").

    Chef / Michelin background:
    - chef_name: The chef's name (string).
    - michelin_restaurants: An array of names of Michelin-starred restaurants where the chef or owner previously worked (array of strings).

    Service format & techniques:
    - tasting_menu_format: A phrase indicating tasting menu format (string).
    - techniques_emphasis: An array listing any of the following techniques if emphasized: curing, drying, fermentation, pickling. Only include techniques explicitly mentioned in the answer (array of strings).

    URL sources (extract ONLY URLs explicitly present in the answer text; include full URLs):
    - name_sources: URLs that support the restaurant's name (array).
    - classification_sources: URLs supporting fine-dining/high-end classification (array).
    - address_sources: URLs supporting the street address (array).
    - location_sources: URLs supporting the CA location (array).
    - opening_date_sources: URLs supporting the opening date (array).
    - capacity_sources: URLs supporting seating capacity (array).
    - chef_sources: URLs supporting the chef identity (array).
    - michelin_sources: URLs supporting the Michelin background AND the specific Michelin restaurant(s) named (array).
    - tasting_menu_sources: URLs supporting tasting menu format (array).
    - techniques_sources: URLs supporting techniques emphasis (array).
    - general_sources: Any additional URLs cited for this restaurant that may support multiple facts (array).

    IMPORTANT:
    - Extract only URLs explicitly present in the answer (including markdown links). Do not invent any URLs.
    - Prefer strings for numbers (e.g., "35") to avoid strict formatting issues.
    - If multiple restaurants are mentioned, extract the first one that appears to satisfy the criteria.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    try:
        return int(nums[0])
    except Exception:
        return None


def unique_merge(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                result.append(url)
    return result


def sources_present(*lists: List[str]) -> bool:
    return len(unique_merge(*lists)) > 0


def month_day_year_2023_valid(ex: RestaurantExtraction) -> bool:
    # Must have month and day, and year is 2023
    has_month_day = bool(ex.opening_month and ex.opening_day)
    is_2023 = (ex.opening_year or "").strip() == "2023"
    return has_month_day and is_2023


def complete_address_present(ex: RestaurantExtraction) -> bool:
    # Heuristic check for "complete street address": non-empty address, includes a number and state info
    addr = (ex.address or "").strip()
    if not addr:
        return False
    has_number = bool(re.search(r"\d+", addr))
    has_state = (" CA" in addr) or ("California" in addr) or ((ex.state or "").strip().lower() in {"ca", "california"})
    has_city = bool(ex.city)
    return has_number and has_state and has_city


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: RestaurantExtraction, parent_node) -> None:
    # Create the main critical node for the restaurant identification
    main_node = evaluator.add_parallel(
        id="Restaurant_Identification",
        desc="Identify ONE fine dining restaurant meeting all criteria and provide all required details with verifiable URLs",
        parent=parent_node,
        critical=True
    )

    # 1. Restaurant Name Provided (existence check)
    evaluator.add_custom_node(
        result=bool((extraction.name or "").strip()),
        id="Restaurant_Name_Provided",
        desc="Restaurant name is provided",
        parent=main_node,
        critical=True
    )

    # Precondition nodes: ensure sources exist for key source-based verifications
    fd_src_present = evaluator.add_custom_node(
        result=sources_present(extraction.classification_sources, extraction.general_sources),
        id="fine_dining_sources_present",
        desc="Sources present for fine-dining classification",
        parent=main_node,
        critical=True
    )

    loc_src_present = evaluator.add_custom_node(
        result=sources_present(extraction.location_sources, extraction.address_sources, extraction.general_sources),
        id="location_sources_present",
        desc="Sources present for California location",
        parent=main_node,
        critical=True
    )

    open_src_present = evaluator.add_custom_node(
        result=sources_present(extraction.opening_date_sources, extraction.general_sources),
        id="opening_date_sources_present",
        desc="Sources present for opening date",
        parent=main_node,
        critical=True
    )

    cap_src_present = evaluator.add_custom_node(
        result=sources_present(extraction.capacity_sources, extraction.general_sources),
        id="capacity_sources_present",
        desc="Sources present for seating capacity",
        parent=main_node,
        critical=True
    )

    chef_src_present = evaluator.add_custom_node(
        result=sources_present(extraction.chef_sources, extraction.general_sources),
        id="chef_sources_present",
        desc="Sources present for chef identity",
        parent=main_node,
        critical=True
    )

    mic_src_present = evaluator.add_custom_node(
        result=sources_present(extraction.michelin_sources, extraction.general_sources),
        id="michelin_sources_present",
        desc="Sources present for Michelin background",
        parent=main_node,
        critical=True
    )

    taste_src_present = evaluator.add_custom_node(
        result=sources_present(extraction.tasting_menu_sources, extraction.general_sources),
        id="tasting_menu_sources_present",
        desc="Sources present for tasting menu format",
        parent=main_node,
        critical=True
    )

    tech_src_present = evaluator.add_custom_node(
        result=sources_present(extraction.techniques_sources, extraction.general_sources),
        id="techniques_sources_present",
        desc="Sources present for techniques emphasis",
        parent=main_node,
        critical=True
    )

    # 2. Fine Dining Classification Supported (verify via URLs)
    fd_leaf = evaluator.add_leaf(
        id="Fine_Dining_Classification_Supported",
        desc="At least one provided source characterizes the restaurant as fine dining (or an equivalent high-end/fine-dining designation)",
        parent=main_node,
        critical=True
    )
    fd_sources = unique_merge(extraction.classification_sources, extraction.general_sources)
    fd_claim = (
        f"At least one provided source explicitly characterizes the restaurant '{extraction.name or 'the restaurant'}' "
        f"as fine dining or an equivalent high-end designation (e.g., fine-dining, upscale tasting-menu restaurant)."
    )
    await evaluator.verify(
        claim=fd_claim,
        node=fd_leaf,
        sources=fd_sources,
        extra_prerequisites=[fd_src_present],
        additional_instruction="Allow synonymous terms like 'fine dining', 'upscale', 'high-end', 'Michelin-style', or 'tasting menu restaurant'."
    )

    # 3. California Location (verify via URLs)
    ca_leaf = evaluator.add_leaf(
        id="California_Location",
        desc="Restaurant is located in California",
        parent=main_node,
        critical=True
    )
    ca_sources = unique_merge(extraction.location_sources, extraction.address_sources, extraction.general_sources)
    ca_claim = f"The restaurant '{extraction.name or 'the restaurant'}' is located in California."
    await evaluator.verify(
        claim=ca_claim,
        node=ca_leaf,
        sources=ca_sources,
        extra_prerequisites=[loc_src_present],
        additional_instruction="Accept either 'California' or 'CA' in the address or location; city must be in California."
    )

    # 4. Complete Street Address Provided (existence/format check)
    evaluator.add_custom_node(
        result=complete_address_present(extraction),
        id="Complete_Street_Address_Provided",
        desc="A complete street address is provided",
        parent=main_node,
        critical=True
    )

    # 5. Opening Date in 2023 with Month and Day
    # Existence: month/day present and year 2023
    opening_ok_node = evaluator.add_custom_node(
        result=month_day_year_2023_valid(extraction),
        id="Opening_Date_In_2023_components_valid",
        desc="Opening date has month and day and year is 2023",
        parent=main_node,
        critical=True
    )
    # Verification via URLs
    opening_leaf = evaluator.add_leaf(
        id="Opening_Date_In_2023_With_Month_And_Day",
        desc="A specific opening date including month and day is provided, and the opening year is 2023",
        parent=main_node,
        critical=True
    )
    open_sources = unique_merge(extraction.opening_date_sources, extraction.general_sources)
    date_text = extraction.opening_date_text or (
        f"{extraction.opening_month or ''} {extraction.opening_day or ''}, {extraction.opening_year or ''}".strip()
    )
    open_claim = (
        f"The restaurant '{extraction.name or 'the restaurant'}' opened on {date_text}, and the opening year is 2023."
    )
    await evaluator.verify(
        claim=open_claim,
        node=opening_leaf,
        sources=open_sources,
        extra_prerequisites=[opening_ok_node, open_src_present],
        additional_instruction="Verify the stated opening date includes both month and day and that the year is 2023; soft openings in 2023 count."
    )

    # 6. Seating Capacity Number and Range (30–40 inclusive)
    capacity_num = parse_first_int(extraction.seating_capacity)
    capacity_in_range = capacity_num is not None and 30 <= capacity_num <= 40
    cap_exist_range_node = evaluator.add_custom_node(
        result=capacity_in_range,
        id="Seating_Capacity_Number_And_Range",
        desc="A specific seating capacity number is provided, and it is between 30 and 40 seats inclusive",
        parent=main_node,
        critical=True
    )
    # Additional verification via URLs to support the number claimed
    cap_leaf = evaluator.add_leaf(
        id="Seating_Capacity_Supported_By_Sources",
        desc="Seating capacity is supported by sources",
        parent=main_node,
        critical=True
    )
    cap_sources = unique_merge(extraction.capacity_sources, extraction.general_sources)
    cap_claim = (
        f"The restaurant '{extraction.name or 'the restaurant'}' has a seating capacity of {capacity_num} seats."
        if capacity_num is not None else
        f"The restaurant '{extraction.name or 'the restaurant'}' has a seating capacity between 30 and 40 seats."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=cap_sources,
        extra_prerequisites=[cap_exist_range_node, cap_src_present],
        additional_instruction="Verify the stated seating capacity number; allow minor rounding but ensure it falls within 30 to 40 seats."
    )

    # 7. Chef Name Provided (existence check)
    chef_exist_node = evaluator.add_custom_node(
        result=bool((extraction.chef_name or "").strip()),
        id="Chef_Name_Provided",
        desc="Chef's name is provided",
        parent=main_node,
        critical=True
    )

    # 8. Michelin Background With Example (verify via URLs)
    # Ensure we have at least one Michelin restaurant name
    michelin_name_provided_node = evaluator.add_custom_node(
        result=bool(extraction.michelin_restaurants),
        id="Michelin_Restaurant_Name_Provided",
        desc="At least one Michelin restaurant name is provided for chef background",
        parent=main_node,
        critical=True
    )
    michelin_leaf = evaluator.add_leaf(
        id="Michelin_Background_With_Example",
        desc="Chef or owner previously worked at a Michelin-starred restaurant, and the name of at least one such Michelin-starred restaurant is provided",
        parent=main_node,
        critical=True
    )
    mic_sources = unique_merge(extraction.michelin_sources, extraction.general_sources)
    mic_rest_name = extraction.michelin_restaurants[0] if extraction.michelin_restaurants else "a Michelin-starred restaurant"
    mic_claim = (
        f"The chef '{extraction.chef_name or 'the chef'}' previously worked at the Michelin-starred restaurant '{mic_rest_name}'."
    )
    await evaluator.verify(
        claim=mic_claim,
        node=michelin_leaf,
        sources=mic_sources,
        extra_prerequisites=[chef_exist_node, michelin_name_provided_node, mic_src_present],
        additional_instruction="Verify both that the chef worked at the named restaurant and that the restaurant is Michelin-starred."
    )

    # 9. Tasting Menu Format (verify via URLs)
    tasting_leaf = evaluator.add_leaf(
        id="Tasting_Menu_Format",
        desc="Restaurant serves a tasting menu format",
        parent=main_node,
        critical=True
    )
    taste_sources = unique_merge(extraction.tasting_menu_sources, extraction.general_sources)
    taste_claim = f"The restaurant '{extraction.name or 'the restaurant'}' serves a tasting menu format."
    await evaluator.verify(
        claim=taste_claim,
        node=tasting_leaf,
        sources=taste_sources,
        extra_prerequisites=[taste_src_present],
        additional_instruction="Look for terms like 'tasting menu', 'multi-course tasting', or prix-fixe tasting format."
    )

    # 10. Culinary Techniques Emphasis (verify via URLs)
    techniques_leaf = evaluator.add_leaf(
        id="Culinary_Techniques_Emphasis",
        desc="Restaurant emphasizes at least one of the specified techniques (curing, drying, fermentation, or pickling)",
        parent=main_node,
        critical=True
    )
    tech_sources = unique_merge(extraction.techniques_sources, extraction.general_sources)
    emphasized = extraction.techniques_emphasis or []
    if emphasized:
        tech_list_text = ", ".join(emphasized)
        tech_claim = (
            f"The restaurant '{extraction.name or 'the restaurant'}' emphasizes at least one of the specified techniques, "
            f"specifically: {tech_list_text}."
        )
    else:
        tech_claim = (
            f"The restaurant '{extraction.name or 'the restaurant'}' emphasizes at least one of the specified techniques "
            f"(curing, drying, fermentation, or pickling)."
        )
    await evaluator.verify(
        claim=tech_claim,
        node=techniques_leaf,
        sources=tech_sources,
        extra_prerequisites=[tech_src_present],
        additional_instruction="Check for explicit mentions of curing, drying, fermentation, or pickling in menus, concept pages, or press materials."
    )

    # 11. Verifiable URL references for all key information (existence check)
    # Check that at least one URL is provided for each key fact
    urls_all_keys_present = (
        sources_present(extraction.address_sources, extraction.location_sources, extraction.general_sources) and
        sources_present(extraction.opening_date_sources, extraction.general_sources) and
        sources_present(extraction.capacity_sources, extraction.general_sources) and
        sources_present(extraction.chef_sources, extraction.general_sources) and
        sources_present(extraction.michelin_sources, extraction.general_sources) and
        sources_present(extraction.tasting_menu_sources, extraction.general_sources) and
        sources_present(extraction.techniques_sources, extraction.general_sources)
    )
    evaluator.add_custom_node(
        result=urls_all_keys_present,
        id="Verifiable_URL_References_For_All_Key_Information",
        desc="Verifiable URL references are provided that support all key required facts (name, address/CA location, opening date, seating capacity, chef identity, Michelin background/restaurant name, tasting menu format, and techniques emphasis)",
        parent=main_node,
        critical=True
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
    """
    Evaluate an answer for the fine dining California 2023 criteria task.

    Returns a structured summary containing the verification tree and overall score.
    """
    # Initialize evaluator (root is non-critical, then we add our critical parent node)
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
    extraction: RestaurantExtraction = await evaluator.extract(
        prompt=prompt_extract_restaurant(),
        template_class=RestaurantExtraction,
        extraction_name="restaurant_extraction"
    )

    # Record constraints in summary for transparency
    evaluator.add_custom_info(
        info={
            "required_year": 2023,
            "required_state": "California",
            "capacity_range_inclusive": [30, 40],
            "techniques": ["curing", "drying", "fermentation", "pickling"],
            "service_format": "tasting menu",
            "must_be_fine_dining": True
        },
        info_type="constraints",
        info_name="evaluation_constraints"
    )

    # Build and run verification checks
    await build_verification_tree(evaluator, extraction, parent_node=root)

    # Return standardized summary
    return evaluator.get_summary()