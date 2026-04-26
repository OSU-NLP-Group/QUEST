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
TASK_ID = "warehouse_facilities_2026"
TASK_DESCRIPTION = """A logistics company is planning to expand its distribution network in 2026 and needs to evaluate suitable warehouse facilities in three major U.S. markets. Identify one warehouse facility in each of the following cities that meets the company's operational requirements:

Cities: Atlanta, Georgia; Dallas, Texas; Charlotte, North Carolina

Minimum Requirements for Each Facility:
- At least 150,000 square feet of warehouse space
- Minimum clear height of 28 feet
- At least 8 loading dock doors
- Zoned for industrial or warehouse use
- Currently available for lease or sale, OR scheduled for delivery in 2026
- Must be an actual warehouse or distribution center facility

Required Information for Each Facility:
For each of the three facilities, provide:
1. Facility name and complete street address
2. Square footage specification
3. Clear height specification
4. Number of loading dock doors
5. Zoning classification
6. Availability status
7. Reference URL(s) verifying the specifications

Additional Information (if available):
- Truck court depth
- Column spacing
- Power supply specifications

Please provide verified, current information with supporting URLs for each facility.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Facility(BaseModel):
    facility_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    square_footage: Optional[str] = None
    clear_height: Optional[str] = None
    loading_dock_doors: Optional[str] = None
    zoning: Optional[str] = None
    availability_status: Optional[str] = None
    building_type: Optional[str] = None
    truck_court_depth: Optional[str] = None
    column_spacing: Optional[str] = None
    power_supply: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    atlanta: Optional[Facility] = None
    dallas: Optional[Facility] = None
    charlotte: Optional[Facility] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
Extract one warehouse facility for each of the following cities from the provided answer:
- Atlanta, Georgia
- Dallas, Texas (Dallas-Fort Worth / DFW metro area is acceptable)
- Charlotte, North Carolina

If the answer mentions multiple facilities for a city, choose the first one that appears in the answer for that city. If information is missing, set the field to null. Extract only what is explicitly present in the answer.

For each city, produce an object with the following fields:
- facility_name: The identifiable facility name (e.g., a branded park/building name) if given
- street_address: Complete street address if provided
- city: City name (try to extract it if present)
- state: State name or abbreviation (try to extract it if present)
- square_footage: Building size information as text (e.g., "220,000 SF" or "200k sq ft")
- clear_height: Clear height as text (e.g., "32' clear" or "30 ft clear")
- loading_dock_doors: Number of dock doors as text (e.g., "36 dock-high doors")
- zoning: Zoning classification as text (e.g., "I-2 Industrial", "M-1", "Industrial")
- availability_status: Availability as text (e.g., "For lease", "Available Q4 2025", "Delivering 2026")
- building_type: Facility type text indicating warehouse/distribution/industrial
- truck_court_depth: Truck court depth if mentioned (text)
- column_spacing: Column spacing if mentioned (text)
- power_supply: Power/electrical service if mentioned (text)
- reference_urls: An array of all explicit URLs in the answer that pertain to this facility (include brochure links, listing pages, or official property pages). 
  Extract actual URLs (including those embedded in markdown). Include only valid http/https URLs.

Return a JSON object with exactly three top-level fields: 'atlanta', 'dallas', and 'charlotte', each mapping to the facility object for that city. If a city has no facility mentioned, set that city to null.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_non_empty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def valid_http_url(u: Optional[str]) -> bool:
    if not isinstance(u, str):
        return False
    u = u.strip()
    return u.lower().startswith("http://") or u.lower().startswith("https://")


def normalize_urls(urls: Optional[List[str]]) -> List[str]:
    urls = urls or []
    clean = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if valid_http_url(uu) and uu not in seen:
            clean.append(uu)
            seen.add(uu)
    return clean


def facility_label(f: Facility) -> str:
    # Helpful label for claims to disambiguate the facility on the source page
    if is_non_empty(f.facility_name) and is_non_empty(f.street_address):
        return f"facility named '{f.facility_name}' at '{f.street_address}'"
    if is_non_empty(f.facility_name):
        return f"facility named '{f.facility_name}'"
    if is_non_empty(f.street_address):
        return f"facility at '{f.street_address}'"
    return "facility"


# --------------------------------------------------------------------------- #
# Verification per-city                                                       #
# --------------------------------------------------------------------------- #
async def verify_city_facility(
    evaluator: Evaluator,
    parent_node,
    node_id_prefix: str,
    node_desc: str,
    facility: Optional[Facility],
    city_name: str,
    state_name: str,
    allow_metro_for_city: Optional[str] = None,
) -> None:
    """
    Build and verify the tree for a single city's facility.
    """
    fac = facility or Facility()
    urls = normalize_urls(fac.reference_urls)

    # Parent parallel node per city
    city_node = evaluator.add_parallel(
        id=node_id_prefix,
        desc=node_desc,
        parent=parent_node,
        critical=False
    )

    # 1) City verification (critical)
    city_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_')[1]}_city_verification",
        desc=f"Facility is located in {city_name}" if not allow_metro_for_city
             else f"Facility is located in {city_name} or {allow_metro_for_city} metro area",
        parent=city_node,
        critical=True,
    )
    if allow_metro_for_city:
        claim_city = f"The {facility_label(fac)} is located in {city_name}, {state_name} or within the {allow_metro_for_city} metropolitan area."
        add_ins_city = f"Accept suburbs or municipalities clearly within the {allow_metro_for_city} metro area (e.g., DFW submarkets)."
    else:
        claim_city = f"The {facility_label(fac)} is located in {city_name}, {state_name}."
        add_ins_city = f"If the page shows a suburb that is clearly part of the {city_name} city limits or commonly considered the city, it is acceptable; otherwise it must be in {city_name}."
    await evaluator.verify(
        claim=claim_city,
        node=city_leaf,
        sources=urls,
        additional_instruction=add_ins_city
    )

    # 2) State verification (critical)
    state_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_')[1]}_state_verification",
        desc=f"Facility is located in {state_name}",
        parent=city_node,
        critical=True,
    )
    claim_state = f"The {facility_label(fac)} is located in the state of {state_name}."
    await evaluator.verify(
        claim=claim_state,
        node=state_leaf,
        sources=urls,
        additional_instruction=f"Accept either full state name or common postal abbreviation (e.g., GA for Georgia, TX for Texas, NC for North Carolina)."
    )

    # 3) Facility name presence (critical, presence check)
    evaluator.add_custom_node(
        result=is_non_empty(fac.facility_name),
        id=f"{node_id_prefix.split('_')[1]}_facility_name",
        desc="Provides identifiable facility name",
        parent=city_node,
        critical=True
    )

    # 4) Street address presence (critical, presence check)
    evaluator.add_custom_node(
        result=is_non_empty(fac.street_address),
        id=f"{node_id_prefix.split('_')[1]}_street_address",
        desc="Provides complete street address",
        parent=city_node,
        critical=True
    )

    # 5) Square footage >= 150,000 (critical, verify with URLs)
    sf_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_')[1]}_square_footage",
        desc="Facility has minimum 150,000 square feet of warehouse space",
        parent=city_node,
        critical=True
    )
    claim_sf = f"The {facility_label(fac)} has at least 150,000 square feet of warehouse or building space."
    await evaluator.verify(
        claim=claim_sf,
        node=sf_leaf,
        sources=urls,
        additional_instruction="Accept if the page shows total building area ≥ 150,000 SF (e.g., '150,000 SF', '200,000 SF', '0.2M SF'). Consider variations like 'SF', 'sq ft', 'square feet'."
    )

    # 6) Clear height >= 28 ft (critical, verify with URLs)
    ch_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_')[1]}_clear_height",
        desc="Facility has minimum 28 feet clear height",
        parent=city_node,
        critical=True
    )
    claim_ch = f"The {facility_label(fac)} has a clear height of at least 28 feet."
    await evaluator.verify(
        claim=claim_ch,
        node=ch_leaf,
        sources=urls,
        additional_instruction="Accept indications like '28 ft clear', '28'–'36' clear', '28′ clear'. Consider feet symbols (′ or ')."
    )

    # 7) Loading docks >= 8 (critical, verify with URLs)
    docks_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_')[1]}_loading_docks",
        desc="Facility has minimum 8 loading dock doors",
        parent=city_node,
        critical=True
    )
    claim_docks = f"The {facility_label(fac)} has at least 8 dock or loading doors."
    await evaluator.verify(
        claim=claim_docks,
        node=docks_leaf,
        sources=urls,
        additional_instruction="Accept language like 'dock-high doors', 'dock doors', 'loading doors'. Count should be ≥ 8."
    )

    # 8) Zoning industrial/warehouse (critical, verify with URLs)
    zoning_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_')[1]}_zoning",
        desc="Property is zoned for industrial/warehouse use",
        parent=city_node,
        critical=True
    )
    claim_zoning = f"The {facility_label(fac)} is zoned for industrial or warehouse use."
    await evaluator.verify(
        claim=claim_zoning,
        node=zoning_leaf,
        sources=urls,
        additional_instruction=(
            "Accept explicit industrial/warehouse zoning, or standard codes that imply industrial use "
            "(e.g., I-1, I-2, IN, IR, LI, HI, M-1, M-2, IND). If the page clearly describes the property as an "
            "industrial/warehouse use with an applicable zoning code, that suffices."
        )
    )

    # 9) Availability or 2026 delivery (critical, verify with URLs)
    avail_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_')[1]}_availability",
        desc="Facility is currently available or scheduled for 2026 delivery",
        parent=city_node,
        critical=True
    )
    claim_avail = (
        f"The {facility_label(fac)} is currently available for lease or sale, OR is scheduled for delivery in 2026."
    )
    await evaluator.verify(
        claim=claim_avail,
        node=avail_leaf,
        sources=urls,
        additional_instruction="Accept 'For lease', 'For sale', 'Available now', 'Now leasing', or 'Under construction/delivering in 2026' (any quarter of 2026)."
    )

    # 10) Building type (critical, verify with URLs)
    type_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_')[1]}_building_type",
        desc="Confirmed as warehouse or distribution center facility type",
        parent=city_node,
        critical=True
    )
    claim_type = f"The {facility_label(fac)} is a warehouse or distribution center facility."
    await evaluator.verify(
        claim=claim_type,
        node=type_leaf,
        sources=urls,
        additional_instruction="Accept synonyms like 'industrial warehouse', 'distribution facility', 'logistics center', 'fulfillment center'."
    )

    # 11) Optional info presence checks (non-critical)
    evaluator.add_custom_node(
        result=is_non_empty(fac.truck_court_depth),
        id=f"{node_id_prefix.split('_')[1]}_truck_court",
        desc="Provides truck court depth information",
        parent=city_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=is_non_empty(fac.column_spacing),
        id=f"{node_id_prefix.split('_')[1]}_column_spacing",
        desc="Provides column spacing information",
        parent=city_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=is_non_empty(fac.power_supply),
        id=f"{node_id_prefix.split('_')[1]}_power_supply",
        desc="Provides power supply specifications",
        parent=city_node,
        critical=False
    )

    # 12) Reference URL presence (critical as per rubric)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{node_id_prefix.split('_')[1]}_reference_url",
        desc="Provides valid reference URL supporting facility specifications",
        parent=city_node,
        critical=True
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
    Evaluate an answer against the warehouse facility rubric for Atlanta, Dallas, and Charlotte.
    """
    # Initialize evaluator
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

    # Extract structured facilities info
    facilities = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Create top-level node
    top_node = evaluator.add_parallel(
        id="warehouse_facility_evaluation",
        desc="Evaluate three warehouse facilities across different U.S. cities for distribution network expansion",
        parent=root,
        critical=False
    )

    # Atlanta
    await verify_city_facility(
        evaluator=evaluator,
        parent_node=top_node,
        node_id_prefix="facility_atlanta",
        node_desc="Warehouse facility in Atlanta, Georgia meeting operational requirements",
        facility=facilities.atlanta if facilities else None,
        city_name="Atlanta",
        state_name="Georgia",
        allow_metro_for_city=None
    )

    # Dallas (DFW metro acceptable)
    await verify_city_facility(
        evaluator=evaluator,
        parent_node=top_node,
        node_id_prefix="facility_dallas",
        node_desc="Warehouse facility in Dallas, Texas meeting operational requirements",
        facility=facilities.dallas if facilities else None,
        city_name="Dallas",
        state_name="Texas",
        allow_metro_for_city="Dallas–Fort Worth"
    )

    # Charlotte
    await verify_city_facility(
        evaluator=evaluator,
        parent_node=top_node,
        node_id_prefix="facility_charlotte",
        node_desc="Warehouse facility in Charlotte, North Carolina meeting operational requirements",
        facility=facilities.charlotte if facilities else None,
        city_name="Charlotte",
        state_name="North Carolina",
        allow_metro_for_city=None
    )

    # Return summary
    return evaluator.get_summary()