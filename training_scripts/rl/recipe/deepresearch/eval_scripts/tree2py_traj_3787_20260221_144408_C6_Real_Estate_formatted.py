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
TASK_ID = "dfw_class_a_warehouse"
TASK_DESCRIPTION = """
Identify four Class A industrial warehouse properties in the Dallas-Fort Worth metropolitan area that meet modern bulk distribution facility standards. For each property, provide:
1) Property Name and Location (confirm DFW metro);
2) Building Specifications (size ≥ 50,000 SF; clear height ≥ 32');
3) Loading Infrastructure (adequate docks ~1 per 10,000 SF; dock height 48–52");
4) Property Quality and Availability (Class A; available now or delivering in 2026/2027);
5) Reference URLs supporting each of the above items.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PropertyInfo(BaseModel):
    """Structured info for one warehouse property."""
    name: Optional[str] = None
    location_text: Optional[str] = None  # e.g., city/submarket/address as written in the answer
    building_size_sqft: Optional[str] = None  # Extract as string (e.g., "650,000 SF")
    clear_height_ft: Optional[str] = None     # Extract as string (e.g., "36 ft", "36'")
    loading_docks: Optional[str] = None       # Extract as string (e.g., "50 dock doors")
    dock_height_inches: Optional[str] = None  # Extract as string (e.g., "48 inches", "48\"")
    property_class: Optional[str] = None      # e.g., "Class A"
    availability_timeline: Optional[str] = None  # e.g., "Now leasing", "Delivering Q2 2027"

    # URL references by category
    location_urls: List[str] = Field(default_factory=list)
    specs_urls: List[str] = Field(default_factory=list)
    loading_urls: List[str] = Field(default_factory=list)
    quality_urls: List[str] = Field(default_factory=list)


class PropertiesExtraction(BaseModel):
    """Top-level extraction result."""
    properties: List[PropertyInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
    Extract up to FOUR industrial warehouse properties mentioned in the answer that are intended to be within the Dallas–Fort Worth (DFW) market and meet modern bulk distribution standards.

    For EACH property, extract the following fields EXACTLY as they appear in the answer (use strings, do not normalize numbers):
    - name: The property or building name (or project name). If unnamed, use a short descriptor from the answer.
    - location_text: The city/submarket/address or any location text given.
    - building_size_sqft: Warehouse/building size (e.g., "650,000 SF", "52,000 square feet").
    - clear_height_ft: Clear height (e.g., "32 ft", "36'", "35 feet").
    - loading_docks: Number of loading docks/doors (e.g., "52 dock doors", "40 docks").
    - dock_height_inches: Dock height value if present (e.g., "48 inches", "50\"", "4 ft ~ 48 in").
    - property_class: Class designation if present (e.g., "Class A").
    - availability_timeline: Availability timing (e.g., "Now leasing", "Delivering Q4 2026", "Available 2027").

    Also extract URL references as arrays (extract ALL valid URLs explicitly present in the answer text):
    - location_urls: URLs supporting that the property is in the DFW metro (or confirming the location).
    - specs_urls: URLs supporting building size and/or clear height.
    - loading_urls: URLs supporting dock count and/or dock height.
    - quality_urls: URLs supporting Class A designation and/or availability timeline.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer (plain URLs or markdown links). Do not invent any URL.
    - If a field is not mentioned, set it to null (or empty array for URLs).
    - Return a JSON object: { "properties": [ ... up to 4 items ... ] }.
    """


# --------------------------------------------------------------------------- #
# Helper functions and verification instructions                              #
# --------------------------------------------------------------------------- #
def ordinal_label(i: int) -> str:
    return ["First", "Second", "Third", "Fourth"][i] if 0 <= i < 4 else f"Property #{i+1}"


def combine_sources(prop: PropertyInfo) -> List[str]:
    """Return a unique, combined list of all URLs for a property."""
    all_urls = []
    for lst in [prop.location_urls, prop.specs_urls, prop.loading_urls, prop.quality_urls]:
        for u in lst or []:
            if u and u not in all_urls:
                all_urls.append(u)
    return all_urls


def prefer_sources(primary: List[str], prop: PropertyInfo) -> List[str]:
    """Prefer category URLs; if empty, fall back to any available property URLs; may return empty."""
    return primary if (primary and len(primary) > 0) else combine_sources(prop)


DFW_LOCATION_INSTRUCTION = (
    "Determine if the property is within the Dallas–Fort Worth metropolitan area "
    "(Dallas–Fort Worth–Arlington MSA). Accept municipalities commonly recognized in the DFW metroplex "
    "such as Dallas, Fort Worth, Arlington, Irving, Grand Prairie, Coppell, Carrollton, Richardson, Plano, "
    "Frisco, McKinney, Allen, Garland, Mesquite, DeSoto, Lancaster, Wilmer, Hutchins, Mansfield, Northlake, "
    "Grapevine, Lewisville, etc. Use the webpage(s) to confirm the location text. If the page does not clearly "
    "place the property within DFW, mark as not supported."
)

SPEC_INSTRUCTION = (
    "Verify BOTH: building size (square feet) and clear height (feet) from the referenced page(s). "
    "Accept approximate/rounded values and common notation (SF, sq ft; clear height 32′/32 ft). "
    "For building size: verify that it is at least 50,000 SF. For clear height: verify that it is at least 32 ft."
)

DOCK_ADEQUACY_INSTRUCTION = (
    "Verify the number of loading docks (doors) and the building size from the page(s), then determine whether "
    "the dock density is adequate for regional bulk distribution (approximately ≥ 1 dock per 10,000 SF). "
    "Allow minor deviation (≈ ±20%). If either the dock count or building size cannot be found on the page(s), "
    "conclude not supported."
)

DOCK_HEIGHT_INSTRUCTION = (
    "Verify that the loading dock height is within the standard range of 48–52 inches above grade. "
    "Accept equivalent expressions like '48\"', '4 ft (≈48\")', '50 inches'. If dock height cannot be located on the page(s), "
    "conclude not supported."
)

CLASS_A_INSTRUCTION = (
    "Verify that the property is Class A industrial per the page(s): explicit 'Class A' labeling or clear language "
    "typical of Class A industrial product (modern construction, institutional quality). If this is not clearly stated, "
    "conclude not supported."
)

AVAILABILITY_INSTRUCTION = (
    "Verify that the property is either currently available (e.g., 'Now leasing', 'Available') or has stated completion/availability "
    "in 2026 or 2027 (e.g., 'Delivers Q4 2026', 'Available 2027'). Use the page(s). If timing cannot be confirmed, conclude not supported."
)


# --------------------------------------------------------------------------- #
# Verification builder for a single property                                  #
# --------------------------------------------------------------------------- #
async def verify_property(
    evaluator: Evaluator,
    root_parent,
    prop: PropertyInfo,
    index: int,
) -> None:
    """
    Build the verification subtree for one property and issue verifications.
    """
    # Property node (non-critical, parallel aggregation)
    property_desc = f"{ordinal_label(index)} qualifying industrial warehouse property in Dallas-Fort Worth"
    property_node = evaluator.add_parallel(
        id=f"Property_{index+1}",
        desc=property_desc,
        parent=root_parent,
        critical=False
    )

    # ---------------------- Location Verification (Critical) ----------------------
    location_node = evaluator.add_parallel(
        id=f"property_{index+1}_location_verification",
        desc="Verify property is located in Dallas-Fort Worth metropolitan area with supporting documentation",
        parent=property_node,
        critical=True
    )

    dfw_leaf = evaluator.add_leaf(
        id=f"property_{index+1}_dfw_market",
        desc="Property is located within the Dallas-Fort Worth metropolitan area",
        parent=location_node,
        critical=True
    )

    dfw_claim = (
        f"The property '{prop.name or f'Property #{index+1}'}' is located within the Dallas–Fort Worth metropolitan area. "
        f"Documented location: {prop.location_text or 'not specified in the answer'}."
    )
    await evaluator.verify(
        claim=dfw_claim,
        node=dfw_leaf,
        sources=prefer_sources(prop.location_urls, prop),
        additional_instruction=DFW_LOCATION_INSTRUCTION
    )

    evaluator.add_custom_node(
        result=(bool(prop.location_urls) and len(prop.location_urls) > 0),
        id=f"property_{index+1}_location_ref_url",
        desc="Provide URL reference confirming the property location in Dallas-Fort Worth",
        parent=location_node,
        critical=True
    )

    # ---------------------- Building Specifications (Critical) -------------------
    specs_node = evaluator.add_parallel(
        id=f"property_{index+1}_building_specifications",
        desc="Verify building meets size and structural specifications for modern warehouse operations",
        parent=property_node,
        critical=True
    )

    size_leaf = evaluator.add_leaf(
        id=f"property_{index+1}_building_size_minimum",
        desc="Property has minimum 50,000 square feet of warehouse space (bulk distribution facility threshold)",
        parent=specs_node,
        critical=True
    )
    size_claim = (
        f"The building size is {prop.building_size_sqft or 'not specified'}, and it meets or exceeds 50,000 square feet."
    )
    await evaluator.verify(
        claim=size_claim,
        node=size_leaf,
        sources=prefer_sources(prop.specs_urls, prop),
        additional_instruction=SPEC_INSTRUCTION
    )

    clear_leaf = evaluator.add_leaf(
        id=f"property_{index+1}_clear_height_standard",
        desc="Property has minimum 32 feet clear height (modern warehouse standard)",
        parent=specs_node,
        critical=True
    )
    clear_claim = (
        f"The clear height is {prop.clear_height_ft or 'not specified'}, and it meets or exceeds 32 feet."
    )
    await evaluator.verify(
        claim=clear_claim,
        node=clear_leaf,
        sources=prefer_sources(prop.specs_urls, prop),
        additional_instruction=SPEC_INSTRUCTION
    )

    evaluator.add_custom_node(
        result=(bool(prop.specs_urls) and len(prop.specs_urls) > 0),
        id=f"property_{index+1}_specs_reference_url",
        desc="Provide URL reference confirming building size and clear height specifications",
        parent=specs_node,
        critical=True
    )

    # ---------------------- Loading Infrastructure (Critical) --------------------
    loading_node = evaluator.add_parallel(
        id=f"property_{index+1}_loading_infrastructure",
        desc="Verify property has adequate loading dock infrastructure for warehouse operations",
        parent=property_node,
        critical=True
    )

    dock_config_leaf = evaluator.add_leaf(
        id=f"property_{index+1}_dock_configuration_adequate",
        desc="Property has adequate number of loading docks (approximately 1 dock per 10,000 square feet for regional warehouses)",
        parent=loading_node,
        critical=True
    )
    dock_config_claim = (
        f"Given a building size of {prop.building_size_sqft or 'unknown'} and {prop.loading_docks or 'unknown'} loading dock doors, "
        f"the property provides approximately 1 dock per 10,000 square feet, which is adequate for regional bulk distribution."
    )
    # For dock adequacy, utilize both loading and specs URLs (ratio needs both)
    combined_loading_specs = list({*prefer_sources(prop.loading_urls, prop), *prefer_sources(prop.specs_urls, prop)})
    await evaluator.verify(
        claim=dock_config_claim,
        node=dock_config_leaf,
        sources=combined_loading_specs,
        additional_instruction=DOCK_ADEQUACY_INSTRUCTION
    )

    dock_height_leaf = evaluator.add_leaf(
        id=f"property_{index+1}_dock_height_standard",
        desc="Loading docks are within standard height range of 48-52 inches above ground level",
        parent=loading_node,
        critical=True
    )
    dock_height_claim = (
        f"The loading dock height is {prop.dock_height_inches or 'not specified'}, within the standard range of 48–52 inches above grade."
    )
    await evaluator.verify(
        claim=dock_height_claim,
        node=dock_height_leaf,
        sources=prefer_sources(prop.loading_urls, prop),
        additional_instruction=DOCK_HEIGHT_INSTRUCTION
    )

    evaluator.add_custom_node(
        result=(bool(prop.loading_urls) and len(prop.loading_urls) > 0),
        id=f"property_{index+1}_loading_reference_url",
        desc="Provide URL reference confirming loading dock count and specifications",
        parent=loading_node,
        critical=True
    )

    # ---------------------- Property Quality & Availability (Critical) ----------
    qa_node = evaluator.add_parallel(
        id=f"property_{index+1}_quality_availability",
        desc="Verify property meets Class A quality standards and is available within specified timeline",
        parent=property_node,
        critical=True
    )

    class_a_leaf = evaluator.add_leaf(
        id=f"property_{index+1}_class_a_quality",
        desc="Property meets Class A industrial standards (highest quality, modern construction, well-located)",
        parent=qa_node,
        critical=True
    )
    class_a_claim = (
        f"The property is Class A industrial. Claimed class: {prop.property_class or 'not specified'}."
    )
    await evaluator.verify(
        claim=class_a_claim,
        node=class_a_leaf,
        sources=prefer_sources(prop.quality_urls, prop),
        additional_instruction=CLASS_A_INSTRUCTION
    )

    availability_leaf = evaluator.add_leaf(
        id=f"property_{index+1}_availability_timeline_2026_2027",
        desc="Property is either currently available or expected to be completed and available in 2026 or 2027",
        parent=qa_node,
        critical=True
    )
    availability_claim = (
        f"The property is currently available or will be completed/available in 2026 or 2027. "
        f"Availability noted: {prop.availability_timeline or 'not specified'}."
    )
    await evaluator.verify(
        claim=availability_claim,
        node=availability_leaf,
        sources=prefer_sources(prop.quality_urls, prop),
        additional_instruction=AVAILABILITY_INSTRUCTION
    )

    evaluator.add_custom_node(
        result=(bool(prop.quality_urls) and len(prop.quality_urls) > 0),
        id=f"property_{index+1}_quality_timeline_reference_url",
        desc="Provide URL reference confirming property class and availability timeline",
        parent=qa_node,
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
    Evaluate an answer for the DFW Class A Industrial Warehouse properties task.
    """
    # Initialize evaluator with PARALLEL aggregation at root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find 4 industrial warehouse properties in the Dallas-Fort Worth market that meet modern bulk distribution facility standards",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured properties from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="properties_extraction"
    )

    # Normalize property list to exactly 4 items (pad if fewer; take first 4 if more)
    props: List[PropertyInfo] = list(extracted.properties or [])
    if len(props) < 4:
        props = props + [PropertyInfo() for _ in range(4 - len(props))]
    else:
        props = props[:4]

    # Build verification subtrees for each of the four properties
    for i, prop in enumerate(props):
        await verify_property(evaluator, root, prop, i)

    # Return structured summary
    return evaluator.get_summary()