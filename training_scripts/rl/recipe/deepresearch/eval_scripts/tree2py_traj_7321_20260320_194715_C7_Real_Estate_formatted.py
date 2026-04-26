import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "atl_industrial_property"
TASK_DESCRIPTION = """
A logistics company is seeking to establish a new distribution center and needs to identify an industrial warehouse property near Atlanta's Hartsfield-Jackson International Airport. The property must meet the following specifications:

- Located in the Aerotropolis Atlanta region or designated airport vicinity submarkets (such as Airport South, South Atlanta, or similar airport-adjacent industrial areas)
- At least 100,000 square feet of warehouse space
- Minimum clear height of 30 feet
- At least 20 dock-high doors
- Currently available for lease or sale (as of 2025-2026)
- Building depth of at least 400 feet
- Within 10 miles of Hartsfield-Jackson Atlanta International Airport
- Includes dedicated office space
- Has adequate truck parking spaces
- Equipped with three-phase electrical power
- Features modern LED warehouse lighting systems
- Has an ESFR (Early Suppression Fast Response) sprinkler system
- Column spacing of at least 40 feet
- Configured as either rear-load or cross-dock

Provide the specific property address, property name (if applicable), and square footage of one property that meets all these requirements.
""".strip()

YEARS_ALLOWED = [2025, 2026]
AIRPORT_NAME = "Hartsfield-Jackson Atlanta International Airport"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class IndustrialProperty(BaseModel):
    property_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None

    submarket_or_region: Optional[str] = None  # e.g., Aerotropolis, Airport South, South Atlanta, etc.

    square_footage: Optional[str] = None  # Keep as string to handle ranges or formatted values
    clear_height: Optional[str] = None
    dock_high_doors: Optional[str] = None
    availability: Optional[str] = None  # e.g., "For Lease", "For Sale", "Available now", include any dates if present
    building_depth: Optional[str] = None
    distance_to_airport_miles: Optional[str] = None  # if stated

    office_space: Optional[str] = None  # textual phrase if present
    truck_parking: Optional[str] = None  # textual phrase if present (e.g., "trailer parking", "X stalls")
    electrical_power: Optional[str] = None  # textual (e.g., "277/480V 3-phase")
    led_lighting: Optional[str] = None  # textual (e.g., "LED lighting")
    fire_suppression: Optional[str] = None  # textual (e.g., "ESFR sprinklers")
    column_spacing: Optional[str] = None
    loading_configuration: Optional[str] = None  # e.g., "rear-load", "cross-dock", "front-load"

    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property() -> str:
    return """
Extract from the answer exactly one industrial warehouse property that the answer claims satisfies the requirements. If multiple properties are mentioned, extract the first one that is presented as meeting the criteria.

Return the following fields from the answer text (do not invent; use null if not present):
- property_name: name of the property or park (if any)
- address: full street address (if present)
- city
- state
- postal_code
- submarket_or_region: the described submarket/region (e.g., "Aerotropolis Atlanta", "Airport South", "South Atlanta", "South Fulton", etc.)
- square_footage: the building square footage as written (e.g., "120,000 SF")
- clear_height: as written (e.g., "36' clear")
- dock_high_doors: as written (e.g., "24 dock doors")
- availability: as written (e.g., "For Lease", "For Sale", "Available Q1 2026", include any dates shown)
- building_depth: as written (e.g., "520' building depth")
- distance_to_airport_miles: as written if explicit (e.g., "5 miles to ATL")
- office_space: text indicating dedicated office space (e.g., "2,500 SF office", "2% office"), else null
- truck_parking: text indicating dedicated truck/trailer parking or trailer stalls, else null
- electrical_power: text indicating three-phase or equivalent power (e.g., "277/480V 3-phase"), else null
- led_lighting: text indicating LED lighting, else null
- fire_suppression: text indicating ESFR sprinklers, else null
- column_spacing: as written (e.g., "50' x 54' column spacing", "50' OC"), else null
- loading_configuration: as written (e.g., "rear-load", "cross-dock", "front-load"), else null
- source_urls: list of all URLs in the answer that directly correspond to the property listing, brochure, flyer, or official page

Rules:
- Extract only what is explicitly present in the answer.
- For URLs, include only valid, complete URLs; ignore malformed ones.
- If a field is missing, set it to null (or empty list for source_urls).
""".strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _label_for_property(prop: IndustrialProperty) -> str:
    parts = []
    if prop.property_name:
        parts.append(prop.property_name)
    if prop.address:
        parts.append(prop.address)
    if not parts:
        return "the property"
    return " / ".join(parts)


def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_property(
    evaluator: Evaluator,
    parent_node,
    prop: IndustrialProperty,
) -> None:
    """
    Build the verification subtree under the critical IndustrialPropertyIdentification node.
    """
    # Create the critical parent node per rubric
    industrial_node = evaluator.add_parallel(
        id="IndustrialPropertyIdentification",
        desc="Identify one industrial warehouse property near Atlanta's Hartsfield-Jackson International Airport that meets all specified requirements for a logistics distribution center operation",
        parent=parent_node,
        critical=True,
    )

    # Quick existence checks (critical)
    has_required_identity = bool(prop.address) and bool(prop.square_footage)
    evaluator.add_custom_node(
        result=has_required_identity,
        id="PropertySpecified",
        desc="Property address and square footage are provided in the answer",
        parent=industrial_node,
        critical=True,
    )

    sources = _normalize_urls(prop.source_urls)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="SourcesProvided",
        desc="At least one valid source URL is provided for the property",
        parent=industrial_node,
        critical=True,
    )

    label = _label_for_property(prop)

    # 1) Geographic Location: Aerotropolis / airport-adjacent submarket
    node_geo = evaluator.add_leaf(
        id="GeographicLocation",
        desc="The property is located in the Aerotropolis Atlanta region or designated airport vicinity submarkets (such as Airport South, South Atlanta, or similar airport-adjacent industrial areas)",
        parent=industrial_node,
        critical=True,
    )
    claim_geo = (
        f"The property {label} is located in the Aerotropolis Atlanta region or an airport-adjacent industrial "
        f"submarket such as Airport South, South Atlanta, South Fulton, College Park, Hapeville, East Point, "
        f"Forest Park, Union City, or similar airport vicinity designations."
    )
    await evaluator.verify(
        claim=claim_geo,
        node=node_geo,
        sources=sources,
        additional_instruction="Accept if the listing explicitly states 'Aerotropolis', 'Airport South', 'South Atlanta', or indicates a recognized airport-adjacent submarket or municipality immediately around ATL (College Park, Hapeville, East Point, Forest Park, Union City, Clayton County near the airport).",
    )

    # 2) Minimum square footage >= 100,000 SF
    node_sf = evaluator.add_leaf(
        id="MinimumSquareFootage",
        desc="The industrial warehouse property has at least 100,000 square feet of warehouse space",
        parent=industrial_node,
        critical=True,
    )
    claim_sf = f"The property {label} has at least 100,000 square feet of warehouse space."
    await evaluator.verify(
        claim=claim_sf,
        node=node_sf,
        sources=sources,
        additional_instruction="Check the building size/SF on the listing or brochure. If a range is given, confirm the maximum or typical rentable area is >= 100,000 SF.",
    )

    # 3) Clear height >= 30'
    node_clear = evaluator.add_leaf(
        id="ClearHeightRequirement",
        desc="The property has a minimum clear height of 30 feet",
        parent=industrial_node,
        critical=True,
    )
    claim_clear = f"The property {label} has a clear height of at least 30 feet."
    await evaluator.verify(
        claim=claim_clear,
        node=node_clear,
        sources=sources,
        additional_instruction="Look for 'clear height' or 'clearance' specs (e.g., 32' clear, 36' clear).",
    )

    # 4) Dock-high doors >= 20
    node_doors = evaluator.add_leaf(
        id="LoadingDockDoors",
        desc="The property has at least 20 dock-high doors",
        parent=industrial_node,
        critical=True,
    )
    claim_doors = f"The property {label} has at least 20 dock-high doors."
    await evaluator.verify(
        claim=claim_doors,
        node=node_doors,
        sources=sources,
        additional_instruction="Verify dock door count; accept equivalents like 'dock positions' or 'dock doors', excluding drive-in doors unless explicitly counted as dock-high.",
    )

    # 5) Availability in 2025-2026
    node_avail = evaluator.add_leaf(
        id="PropertyAvailability",
        desc="The property is currently available for lease or sale (as of 2025-2026)",
        parent=industrial_node,
        critical=True,
    )
    years_str = " or ".join(str(y) for y in YEARS_ALLOWED)
    claim_avail = f"As of {years_str}, the property {label} is available for lease or sale."
    await evaluator.verify(
        claim=claim_avail,
        node=node_avail,
        sources=sources,
        additional_instruction="Accept if the page explicitly indicates For Lease/For Sale and the listing page shows a last-updated or brochure date in 2025 or 2026, or states availability in 2025/2026 (e.g., 'Available now' with a 2025/2026 document). Reject if clearly marked leased or only historic (pre-2025).",
    )

    # 6) Building depth >= 400'
    node_depth = evaluator.add_leaf(
        id="BuildingDepth",
        desc="The warehouse has a building depth of at least 400 feet",
        parent=industrial_node,
        critical=True,
    )
    claim_depth = f"The warehouse for {label} has a building depth of at least 400 feet."
    await evaluator.verify(
        claim=claim_depth,
        node=node_depth,
        sources=sources,
        additional_instruction="Look for 'building depth' (e.g., 400'+, 520'). If multiple depths are shown, accept if any principal building depth (not just bay depth) is >= 400'.",
    )

    # 7) Within 10 miles of ATL airport
    node_prox = evaluator.add_leaf(
        id="AirportProximity",
        desc="The property is located within 10 miles of Hartsfield-Jackson Atlanta International Airport",
        parent=industrial_node,
        critical=True,
    )
    claim_prox = f"The property {label} is within 10 miles of {AIRPORT_NAME}."
    await evaluator.verify(
        claim=claim_prox,
        node=node_prox,
        sources=sources,
        additional_instruction="Prefer explicit statements like 'X miles to ATL' where X ≤ 10. Otherwise, accept if the listing indicates immediate proximity to the airport with a site map showing a location clearly within ~10 miles. Mentions like 'minutes to ATL' may be acceptable if likely ≤ 10 miles.",
    )

    # 8) Dedicated office space
    node_office = evaluator.add_leaf(
        id="OfficeSpace",
        desc="The property includes dedicated office space within the facility",
        parent=industrial_node,
        critical=True,
    )
    claim_office = f"The property {label} includes dedicated office space."
    await evaluator.verify(
        claim=claim_office,
        node=node_office,
        sources=sources,
        additional_instruction="Accept language such as 'X SF office', 'office buildout', or 'office percentage'.",
    )

    # 9) Adequate truck parking
    node_truck = evaluator.add_leaf(
        id="TruckParking",
        desc="The property has adequate truck parking spaces for staging and operations",
        parent=industrial_node,
        critical=True,
    )
    claim_truck = f"The property {label} provides dedicated truck or trailer parking suitable for operations."
    await evaluator.verify(
        claim=claim_truck,
        node=node_truck,
        sources=sources,
        additional_instruction="Look for 'trailer parking', 'trailer stalls', 'truck/trailer storage', or clearly designated truck parking areas. A large truck court alone without stated trailer parking may be insufficient unless it explicitly allows staging/parking.",
    )

    # 10) Three-phase electrical power
    node_power = evaluator.add_leaf(
        id="ElectricalPower",
        desc="The property has three-phase electrical power to support industrial operations",
        parent=industrial_node,
        critical=True,
    )
    claim_power = f"The property {label} is equipped with three-phase electrical power."
    await evaluator.verify(
        claim=claim_power,
        node=node_power,
        sources=sources,
        additional_instruction="Accept '3-phase', 'three-phase', or specs like '277/480V 3-phase'.",
    )

    # 11) LED lighting
    node_led = evaluator.add_leaf(
        id="LEDLighting",
        desc="The warehouse features modern LED warehouse lighting systems",
        parent=industrial_node,
        critical=True,
    )
    claim_led = f"The warehouse for {label} features LED lighting."
    await evaluator.verify(
        claim=claim_led,
        node=node_led,
        sources=sources,
        additional_instruction="Look for 'LED', 'LED high-bay', or similar phrasing. Do not accept only fluorescent or metal halide.",
    )

    # 12) ESFR sprinklers
    node_esfr = evaluator.add_leaf(
        id="FireSuppression",
        desc="The property has an ESFR (Early Suppression Fast Response) sprinkler system installed",
        parent=industrial_node,
        critical=True,
    )
    claim_esfr = f"The property {label} has an ESFR sprinkler system."
    await evaluator.verify(
        claim=claim_esfr,
        node=node_esfr,
        sources=sources,
        additional_instruction="Accept 'ESFR' explicitly. Do not accept only 'wet system' or 'ordinary hazard' unless ESFR is specified.",
    )

    # 13) Column spacing >= 40'
    node_cols = evaluator.add_leaf(
        id="ColumnSpacing",
        desc="The warehouse has column spacing of at least 40 feet",
        parent=industrial_node,
        critical=True,
    )
    claim_cols = f"The warehouse for {label} has column spacing of at least 40 feet."
    await evaluator.verify(
        claim=claim_cols,
        node=node_cols,
        sources=sources,
        additional_instruction="Look for 'column spacing' values; typical modern spacing is 50' x 54' or similar. Accept if any principal spacing dimension is >= 40'.",
    )

    # 14) Rear-load or cross-dock configuration
    node_load = evaluator.add_leaf(
        id="LoadingConfiguration",
        desc="The property has either a rear-load or cross-dock configuration",
        parent=industrial_node,
        critical=True,
    )
    claim_load = f"The building {label} is configured as either rear-load or cross-dock."
    await evaluator.verify(
        claim=claim_load,
        node=node_load,
        sources=sources,
        additional_instruction="Accept if the page explicitly states 'rear-load', 'rear load', 'cross-dock', or 'cross dock'. Do not accept 'front-load' unless the page also confirms rear-load or cross-dock areas.",
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

    # Extract structured property information
    extracted_prop = await evaluator.extract(
        prompt=prompt_extract_property(),
        template_class=IndustrialProperty,
        extraction_name="industrial_property",
    )

    # Add ground-truth thresholds (contextual info)
    evaluator.add_ground_truth({
        "airport": AIRPORT_NAME,
        "min_square_feet": "100,000 SF",
        "min_clear_height_ft": "30",
        "min_dock_doors": "20",
        "min_building_depth_ft": "400",
        "max_distance_to_airport_miles": "10",
        "min_column_spacing_ft": "40",
        "allowed_loading_configurations": ["rear-load", "cross-dock"],
        "availability_years": YEARS_ALLOWED,
    }, gt_type="requirements")

    # Build and verify
    await build_and_verify_property(evaluator, root, extracted_prop)

    return evaluator.get_summary()