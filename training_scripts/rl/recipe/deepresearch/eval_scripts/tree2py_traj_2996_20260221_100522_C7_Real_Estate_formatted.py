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
TASK_ID = "harris_teeter_dev_specs_jax"
TASK_DESCRIPTION = (
    "Identify the comprehensive property development specifications for the new Harris Teeter grocery store at "
    "11901 Atlantic Boulevard (Atlantic North Shopping Center) in Jacksonville, Florida. Include details on building size, "
    "parking requirements, ADA accessibility compliance, operational features (fuel center, pharmacy, coffee shop), "
    "building setbacks, fire safety systems, loading facilities, and employment projections."
)
PROJECT_CONTEXT = "for the Harris Teeter grocery store at 11901 Atlantic Boulevard (Atlantic North Shopping Center) in Jacksonville, Florida"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProjectSpecs(BaseModel):
    # Core building and parking
    building_size: Optional[str] = None
    building_size_sources: List[str] = Field(default_factory=list)

    parking_minimum_spaces: Optional[str] = None
    parking_sources: List[str] = Field(default_factory=list)

    # ADA accessibility
    ada_entrance_width: Optional[str] = None
    ada_entrance_sources: List[str] = Field(default_factory=list)

    ada_parking_space_width: Optional[str] = None
    ada_parking_sources: List[str] = Field(default_factory=list)

    ada_access_aisle_width: Optional[str] = None
    ada_access_aisle_sources: List[str] = Field(default_factory=list)

    # Loading and fire safety
    loading_dock_height: Optional[str] = None
    loading_dock_sources: List[str] = Field(default_factory=list)

    fire_sprinkler_system: Optional[str] = None
    fire_sprinkler_sources: List[str] = Field(default_factory=list)

    # Operational features
    fuel_center_included: Optional[str] = None
    fuel_center_hours: Optional[str] = None
    fuel_center_sources: List[str] = Field(default_factory=list)

    pharmacy_drive_through: Optional[str] = None
    pharmacy_sources: List[str] = Field(default_factory=list)

    coffee_shop: Optional[str] = None
    coffee_shop_sources: List[str] = Field(default_factory=list)

    # Setbacks
    front_setback: Optional[str] = None
    front_setback_sources: List[str] = Field(default_factory=list)

    side_setback: Optional[str] = None
    side_setback_sources: List[str] = Field(default_factory=list)

    # Employment
    employment_projection: Optional[str] = None
    employment_sources: List[str] = Field(default_factory=list)

    # Fallback/global sources mentioned in the answer
    global_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project_specs() -> str:
    return """
    Extract the project development specifications for a proposed Harris Teeter grocery store project in Jacksonville, FL (Atlantic North Shopping Center at 11901 Atlantic Boulevard).
    For each item below, return both the value mentioned in the answer and the list of source URLs that support that specific item. Extract values EXACTLY as they appear in the answer (do not invent). If an item is not mentioned, set its value to null and the sources to an empty list.
    IMPORTANT: For all 'sources' fields, only include URLs explicitly present in the answer text that support the corresponding item. If the answer provides a single consolidated sources section for all claims, include those URLs in 'global_sources' and also try to assign relevant ones per field when possible.

    Return a single JSON object with these fields:

    - building_size: string | null
    - building_size_sources: string[]  // URLs supporting the building size

    - parking_minimum_spaces: string | null
    - parking_sources: string[]        // URLs supporting the minimum required parking count

    - ada_entrance_width: string | null
    - ada_entrance_sources: string[]   // URLs that support the ADA entrance clear width requirement (≥32")

    - ada_parking_space_width: string | null
    - ada_parking_sources: string[]    // URLs that support accessible parking width (≥96")

    - ada_access_aisle_width: string | null
    - ada_access_aisle_sources: string[] // URLs that support access aisle width (≥60")

    - loading_dock_height: string | null
    - loading_dock_sources: string[]   // URLs that support the specified dock height

    - fire_sprinkler_system: string | null
    - fire_sprinkler_sources: string[] // URLs that support inclusion of an automatic fire sprinkler system

    - fuel_center_included: string | null   // e.g., "yes", "no", "not included", or a descriptive phrase
    - fuel_center_hours: string | null      // only if a fuel center is included and hours are specified
    - fuel_center_sources: string[]         // URLs that support fuel center and/or hours

    - pharmacy_drive_through: string | null // e.g., "yes", "drive-through pharmacy", or "no"
    - pharmacy_sources: string[]            // URLs that support pharmacy drive-through

    - coffee_shop: string | null            // e.g., "Starbucks", "in-store coffee", or "no"
    - coffee_shop_sources: string[]         // URLs that support coffee shop amenity

    - front_setback: string | null
    - front_setback_sources: string[]       // URLs that support front setback distance

    - side_setback: string | null
    - side_setback_sources: string[]        // URLs that support side setback distance

    - employment_projection: string | null
    - employment_sources: string[]          // URLs that support estimated number of jobs to be created

    - global_sources: string[]              // Any sources listed in the answer that apply broadly to the project

    SPECIAL RULES:
    - Only extract URLs explicitly present in the answer. Do not infer or create new URLs.
    - If a URL is missing a protocol (http/https), prepend http://
    - Keep numbers and units as provided (e.g., "60,000 SF", "300 spaces", "36 inches").
    - If the fuel center is not included, set fuel_center_included to something like "no" and fuel_center_hours to null.
    - Do not attempt to calculate or normalize values; preserve exactly how they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_content(val: Optional[str]) -> bool:
    return bool(val and str(val).strip())


def pick_sources(specific: List[str], global_sources: List[str]) -> List[str]:
    # Prefer specific URLs; fallback to global if specific is empty
    if specific and len(specific) > 0:
        return specific
    return global_sources or []


def is_fuel_center_included(specs: ProjectSpecs) -> bool:
    # Heuristic: hours provided implies inclusion; otherwise parse string indicator
    if has_content(specs.fuel_center_hours):
        return True
    indicator = (specs.fuel_center_included or "").strip().lower()
    if not indicator:
        return False
    # simple affirmative detection
    affirmative_tokens = ["yes", "true", "include", "included", "with", "has", "fuel", "gas"]
    return any(tok in indicator for tok in affirmative_tokens)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_value_verification(
    evaluator: Evaluator,
    parent_node,
    *,
    node_id: str,
    node_desc: str,
    value: Optional[str],
    sources: List[str],
    claim: str,
    additional_instruction: str = ""
) -> None:
    """
    Create a critical spec node with:
      - existence custom node (value present + sources provided)
      - source-supported verification leaf
    """
    spec_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )

    # Existence: require both a stated value and at least one source URL
    exists_result = has_content(value) and bool(sources)
    evaluator.add_custom_node(
        result=exists_result,
        id=f"{node_id}_exists",
        desc=f"{node_desc} is provided in the answer with at least one source URL",
        parent=spec_node,
        critical=True
    )

    # Supported by sources (auto-skipped if existence fails due to critical sibling precondition)
    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{node_desc} is supported by cited sources",
        parent=spec_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=sources,
        additional_instruction=additional_instruction
    )


async def add_optional_fuel_center_verification(
    evaluator: Evaluator,
    parent_node,
    specs: ProjectSpecs
) -> None:
    """
    Conditional handling for fuel center hours:
      - If a fuel center is included, require hours value + sources and verify.
      - If not included or unclear, mark as not applicable but pass the criterion.
    """
    node_desc = "Fuel center operating hours (if applicable)"
    node_id = "Fuel_Center_Operating_Hours"

    fuel_node = evaluator.add_parallel(
        id=node_id,
        desc="If fuel center is included, the operating hours are specified (typically 6 AM - 10 PM with 24-hour pump access)",
        parent=parent_node,
        critical=True
    )

    included = is_fuel_center_included(specs)
    if not included:
        # Not applicable: pass as satisfied for this project (hours not required)
        evaluator.add_custom_node(
            result=True,
            id=f"{node_id}_not_applicable",
            desc="Fuel center not included in this project; operating hours not applicable",
            parent=fuel_node,
            critical=True
        )
        return

    # If included, verify hours and sources
    hours_val = specs.fuel_center_hours
    sources = pick_sources(specs.fuel_center_sources, specs.global_sources)

    exists_result = has_content(hours_val) and bool(sources)
    evaluator.add_custom_node(
        result=exists_result,
        id=f"{node_id}_exists",
        desc="Fuel center hours are provided with at least one supporting source URL",
        parent=fuel_node,
        critical=True
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc="Fuel center hours are supported by cited sources",
        parent=fuel_node,
        critical=True
    )
    # Claim for hours
    hours_text = hours_val or ""
    claim = f"The fuel center operating hours {hours_text} are specified {PROJECT_CONTEXT}."
    add_ins = (
        "Verify that the cited page(s) mention or clearly imply the operating hours for the fuel center; "
        "if separate kiosk vs. pump hours are listed (e.g., kiosk 6 AM–10 PM with 24-hour pump access), accept that as matching."
    )
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=sources,
        additional_instruction=add_ins
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Harris Teeter development specifications in Jacksonville, FL.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured project specs from the answer
    specs: ProjectSpecs = await evaluator.extract(
        prompt=prompt_extract_project_specs(),
        template_class=ProjectSpecs,
        extraction_name="project_specs"
    )

    # Main critical parent node (as per rubric)
    main_node = evaluator.add_parallel(
        id="Grocery_Store_Property_Development_Specifications",
        desc="Comprehensive evaluation of all required specifications and compliance features for a new grocery store development in Jacksonville, Florida",
        parent=root,
        critical=True
    )

    # Building Size
    await add_value_verification(
        evaluator,
        main_node,
        node_id="Building_Size_Specification",
        node_desc="The total building size in square feet is specified",
        value=specs.building_size,
        sources=pick_sources(specs.building_size_sources, specs.global_sources),
        claim=f"The total building size is {specs.building_size or ''} {PROJECT_CONTEXT}.",
        additional_instruction="Accept format variations such as 'SF', 'sq ft', or comma-separated numbers."
    )

    # Parking Space Count (minimum required)
    await add_value_verification(
        evaluator,
        main_node,
        node_id="Parking_Space_Count",
        node_desc="The minimum required number of parking spaces is specified",
        value=specs.parking_minimum_spaces,
        sources=pick_sources(specs.parking_sources, specs.global_sources),
        claim=f"The minimum required number of parking spaces is {specs.parking_minimum_spaces or ''} {PROJECT_CONTEXT}.",
        additional_instruction="If both 'required' and 'provided' counts are listed, ensure the claim refers to the 'required' minimum."
    )

    # ADA Entrance Doorway Width (≥32 inches)
    await add_value_verification(
        evaluator,
        main_node,
        node_id="ADA_Entrance_Doorway_Width",
        node_desc="Entrance doorways meet the minimum 32-inch clear width requirement for ADA accessibility",
        value=specs.ada_entrance_width,
        sources=pick_sources(specs.ada_entrance_sources, specs.global_sources),
        claim=f"The project specifies entrance doorways with a clear width of at least 32 inches (ADA compliant) {PROJECT_CONTEXT}.",
        additional_instruction="If a larger width (e.g., 36 inches) is stated, that satisfies 'at least 32 inches'."
    )

    # ADA Accessible Parking Space Width (≥96 inches)
    await add_value_verification(
        evaluator,
        main_node,
        node_id="ADA_Parking_Space_Width",
        node_desc="Accessible parking spaces are at least 96 inches (8 feet) wide as required by ADA standards",
        value=specs.ada_parking_space_width,
        sources=pick_sources(specs.ada_parking_sources, specs.global_sources),
        claim=f"The project specifies accessible parking spaces at a minimum width of 96 inches (8 feet) {PROJECT_CONTEXT}.",
        additional_instruction="Allow synonyms or explicit larger widths (e.g., 9 ft, 132 inches for van-accessible) to indicate compliance."
    )

    # ADA Access Aisle Width (≥60 inches)
    await add_value_verification(
        evaluator,
        main_node,
        node_id="ADA_Access_Aisle_Width",
        node_desc="Access aisle adjacent to accessible parking is at least 60 inches (5 feet) wide as required by ADA standards",
        value=specs.ada_access_aisle_width,
        sources=pick_sources(specs.ada_access_aisle_sources, specs.global_sources),
        claim=f"The project specifies that the access aisle adjacent to accessible parking is at least 60 inches (5 feet) wide {PROJECT_CONTEXT}.",
        additional_instruction="Accept explicit larger aisle widths as satisfying the minimum."
    )

    # Loading Dock Platform Height
    await add_value_verification(
        evaluator,
        main_node,
        node_id="Loading_Dock_Height",
        node_desc="Loading dock platform height is specified",
        value=specs.loading_dock_height,
        sources=pick_sources(specs.loading_dock_sources, specs.global_sources),
        claim=f"The loading dock platform height is specified as {specs.loading_dock_height or ''} {PROJECT_CONTEXT}.",
        additional_instruction="Accept synonyms like 'dock height' or 'platform elevation'. Do not require confirmation of typical ranges; just confirm the stated height is present."
    )

    # Fire Sprinkler System
    await add_value_verification(
        evaluator,
        main_node,
        node_id="Fire_Sprinkler_System",
        node_desc="Automatic fire sprinkler system is specified",
        value=specs.fire_sprinkler_system,
        sources=pick_sources(specs.fire_sprinkler_sources, specs.global_sources),
        claim=f"The project includes an automatic fire sprinkler system {PROJECT_CONTEXT}.",
        additional_instruction="Accept phrases such as 'automatic sprinkler system', 'NFPA-compliant sprinklers', or 'fire suppression sprinkler system'."
    )

    # Fuel Center Operating Hours (conditional)
    await add_optional_fuel_center_verification(evaluator, main_node, specs)

    # Pharmacy with Drive-Through
    await add_value_verification(
        evaluator,
        main_node,
        node_id="Pharmacy_Drive_Through",
        node_desc="Pharmacy with drive-through capability is specified as a store feature",
        value=specs.pharmacy_drive_through,
        sources=pick_sources(specs.pharmacy_sources, specs.global_sources),
        claim=f"The store includes a pharmacy with drive-through capability {PROJECT_CONTEXT}.",
        additional_instruction="Accept synonyms like 'drive-thru pharmacy'."
    )

    # In-Store Coffee Shop
    await add_value_verification(
        evaluator,
        main_node,
        node_id="In_Store_Coffee_Shop",
        node_desc="In-store coffee shop or Starbucks is specified as a store amenity",
        value=specs.coffee_shop,
        sources=pick_sources(specs.coffee_shop_sources, specs.global_sources),
        claim=f"The store includes an in-store coffee shop (e.g., Starbucks) {PROJECT_CONTEXT}.",
        additional_instruction="Accept 'Starbucks kiosk', 'coffee bar', or similar phrasings as indicating an in-store coffee offering."
    )

    # Front Building Setback
    await add_value_verification(
        evaluator,
        main_node,
        node_id="Front_Building_Setback",
        node_desc="Front building setback distance from property line is specified",
        value=specs.front_setback,
        sources=pick_sources(specs.front_setback_sources, specs.global_sources),
        claim=f"The front building setback from the property line is {specs.front_setback or ''} {PROJECT_CONTEXT}.",
        additional_instruction="Accept equivalent terms such as 'front yard setback' or 'setback from right-of-way'."
    )

    # Side Building Setback
    await add_value_verification(
        evaluator,
        main_node,
        node_id="Side_Building_Setback",
        node_desc="Side building setback distance from property line is specified",
        value=specs.side_setback,
        sources=pick_sources(specs.side_setback_sources, specs.global_sources),
        claim=f"The side building setback from the property line is {specs.side_setback or ''} {PROJECT_CONTEXT}.",
        additional_instruction="Accept equivalent terms such as 'side yard setback'."
    )

    # Employment Projection
    await add_value_verification(
        evaluator,
        main_node,
        node_id="Employment_Projection",
        node_desc="The estimated number of jobs to be created by the new store is specified",
        value=specs.employment_projection,
        sources=pick_sources(specs.employment_sources, specs.global_sources),
        claim=f"The estimated number of jobs created by the project is {specs.employment_projection or ''} {PROJECT_CONTEXT}.",
        additional_instruction="Accept phrasing like 'jobs', 'positions', 'employees', or 'employment opportunities'."
    )

    # Optional: add custom info about project context
    evaluator.add_custom_info(
        {"project_address": "11901 Atlantic Boulevard, Jacksonville, FL (Atlantic North Shopping Center)"},
        info_type="project_context",
        info_name="project_context"
    )

    # Return evaluation summary
    return evaluator.get_summary()