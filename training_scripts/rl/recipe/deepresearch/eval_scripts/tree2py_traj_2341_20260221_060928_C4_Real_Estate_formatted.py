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
TASK_ID = "hartford_class_a_office_120"
TASK_DESCRIPTION = (
    "Identify a Class A office building in Hartford, Connecticut that has available office space suitable for a technology company planning to accommodate 120 employees. "
    "The space must meet the following requirements:\n\n"
    "1. Classified as Class A office space with modern construction and premium amenities\n"
    "2. Sufficient available space to provide 150-200 square feet per employee (18,000-24,000 total square feet)\n"
    "3. Parking availability of at least 4 spaces per 1,000 square feet of office space\n"
    "4. Full ADA compliance including accessible entrances, pathways, and elevators\n"
    "5. Located in Hartford, Connecticut (ranked as the #1 hottest real estate market for 2026)\n"
    "6. LEED certification at Silver level or higher\n"
    "7. Available under a gross lease or modified gross lease structure (not triple net)\n\n"
    "Provide the name and address of one specific building that meets all these criteria, along with supporting documentation for each requirement."
)

# Ground truth (constraints) for reference/context
GROUND_TRUTH_CONSTRAINTS = {
    "employee_count": 120,
    "sf_per_employee_min": 150,
    "sf_per_employee_max": 200,
    "total_sf_min": 18000,
    "total_sf_max": 24000,
    "min_parking_ratio_per_1000_sf": 4.0,
    "required_leed_level_min": "Silver",
    "allowed_lease_types": ["gross", "modified gross", "full service", "full-service", "full service gross"],
    "disallowed_lease_types": ["triple net", "nnn"]
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BuildingPick(BaseModel):
    # Identification
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    # Criteria claims (as stated in the answer)
    class_a: Optional[str] = None                  # e.g., "Class A", "Class A office"
    available_sqft: Optional[str] = None           # e.g., "20,000 SF", "18,500-22,000 SF"
    parking_ratio: Optional[str] = None            # e.g., "4/1,000 SF", "5 per 1,000 SF"
    ada_compliance: Optional[str] = None           # e.g., "ADA compliant with elevator"
    leed_level: Optional[str] = None               # e.g., "LEED Silver", "LEED Gold"
    lease_type: Optional[str] = None               # e.g., "Modified Gross", "Full Service", "Gross"

    # Source URLs per criterion (as cited in the answer)
    sources_building: List[str] = Field(default_factory=list)        # name/address/overview page
    sources_class: List[str] = Field(default_factory=list)           # Class A classification
    sources_space: List[str] = Field(default_factory=list)           # space availability
    sources_parking: List[str] = Field(default_factory=list)         # parking ratio
    sources_ada: List[str] = Field(default_factory=list)             # ADA access
    sources_location: List[str] = Field(default_factory=list)        # geographic location
    sources_leed: List[str] = Field(default_factory=list)            # LEED certification
    sources_lease: List[str] = Field(default_factory=list)           # lease structure


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_building_pick() -> str:
    return """
    Extract exactly one building (the primary building recommended in the answer) and the supporting information cited for each criterion.
    If multiple buildings are mentioned, pick the first one that the answer presents as the recommended/selected building.

    Return a JSON object with these fields:
    - name: The building name (string)
    - address: The street address (string)
    - city: City (string)
    - state: State (string, e.g., CT)
    - zip_code: ZIP code (string, if present)
    - class_a: The building classification text if provided (e.g., "Class A office")
    - available_sqft: The available space claimed for the building (e.g., "20,000 SF", "approx 18,500–22,000 SF"). If multiple sizes are given, extract the one that is claimed to be available for this tenant need.
    - parking_ratio: The parking ratio statement (e.g., "4 per 1,000 SF", "5/1,000")
    - ada_compliance: The ADA compliance statement (e.g., "ADA compliant", "accessible entrances and elevators")
    - leed_level: The LEED certification text (e.g., "LEED Silver", "LEED Gold")
    - lease_type: The lease structure (e.g., "Gross", "Modified Gross", "Full Service", "NNN")
    - sources_building: Array of URLs that support name/address/overview of this building
    - sources_class: Array of URLs that specifically support the Class A classification (if provided)
    - sources_space: Array of URLs that specifically support the available square footage
    - sources_parking: Array of URLs that specifically support the parking ratio
    - sources_ada: Array of URLs that specifically support ADA accessibility
    - sources_location: Array of URLs that specifically confirm the building is in Hartford, CT
    - sources_leed: Array of URLs that specifically support the claimed LEED certification
    - sources_lease: Array of URLs that specifically support the lease structure

    SPECIAL RULES:
    - Only extract URLs that are explicitly present in the answer (including markdown links).
    - If a specific criterion doesn't have its own dedicated sources, but general building URLs are provided, do NOT infer; just leave the criterion’s sources list empty.
    - If a field is not mentioned, set it to null; if a sources list is not present, return an empty array for that list.

    IMPORTANT:
    - Do not invent any information. Reflect exactly what is claimed in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    merged.append(u)
    return merged


def _format_full_address(b: BuildingPick) -> str:
    parts = [b.address or "", b.city or "", b.state or "", b.zip_code or ""]
    # Keep empty parts out to avoid awkward punctuation; join with commas where present.
    return ", ".join([p for p in parts if p and p.strip() != ""])


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_building_identification(evaluator: Evaluator, parent, b: BuildingPick) -> None:
    node = evaluator.add_parallel(
        id="Building_Identification",
        desc="The solution must provide the name and address of one specific building that meets all criteria, along with supporting documentation (reference URLs or sources) for verification",
        parent=parent,
        critical=True
    )

    # Existence of basic identification
    has_name_addr = bool(b and b.name and b.name.strip() and b.address and b.address.strip())
    evaluator.add_custom_node(
        result=has_name_addr,
        id="building_identification_exists",
        desc="Building name and street address are provided in the answer",
        parent=node,
        critical=True
    )

    # Existence of at least one source for identification
    has_id_sources = len(b.sources_building) > 0 or len(b.sources_location) > 0
    evaluator.add_custom_node(
        result=has_id_sources,
        id="building_identification_sources_provided",
        desc="Supporting source URL(s) for building identification are provided",
        parent=node,
        critical=True
    )

    # Verify that sources support name and address
    verify_node = evaluator.add_leaf(
        id="building_identification_supported",
        desc="Cited sources support the building name and address",
        parent=node,
        critical=True
    )
    full_addr = _format_full_address(b)
    claim = f"The cited sources show a building named '{b.name}' located at '{full_addr}'."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_merge_sources(b.sources_building, b.sources_location),
        additional_instruction="Allow minor variations in formatting (abbreviations, punctuation, or ZIP code presence). Focus on clear support that the building name and street address match."
    )


async def verify_building_classification(evaluator: Evaluator, parent, b: BuildingPick) -> None:
    node = evaluator.add_parallel(
        id="Building_Classification",
        desc="The identified building must be classified as Class A office space (highest quality with modern construction, premium amenities, prime location, and professional management)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(b.class_a and b.class_a.strip()),
        id="building_classification_claim_provided",
        desc="Answer explicitly states building classification information",
        parent=node,
        critical=True
    )

    # Verify Class A classification
    leaf = evaluator.add_leaf(
        id="building_classification_supported",
        desc="Cited sources support the claim that the building is Class A office space",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This building is classified as Class A office space.",
        node=leaf,
        sources=_merge_sources(b.sources_class, b.sources_building),
        additional_instruction="Confirm that the webpage explicitly states 'Class A' for this building or otherwise clearly indicates Class A classification."
    )


async def verify_office_space_capacity(evaluator: Evaluator, parent, b: BuildingPick) -> None:
    node = evaluator.add_parallel(
        id="Office_Space_Capacity",
        desc="The building must have sufficient available office space to accommodate 120 employees at the industry standard of 150-200 square feet per employee (total 18,000-24,000 square feet)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(b.available_sqft and b.available_sqft.strip()),
        id="office_space_claim_provided",
        desc="Answer provides an available square footage figure or statement",
        parent=node,
        critical=True
    )

    # Verify the available space is within 18,000–24,000 SF from sources
    leaf = evaluator.add_leaf(
        id="office_space_in_range_supported",
        desc="Cited sources support that the available office space is between 18,000 and 24,000 square feet",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The available office space for lease is between 18,000 and 24,000 square feet.",
        node=leaf,
        sources=_merge_sources(b.sources_space, b.sources_building),
        additional_instruction="Verify that the page indicates available contiguous or total leasable office space within 18,000–24,000 SF. If multiple figures are shown, look for the specific availability referenced for this tenant need."
    )


async def verify_parking_availability(evaluator: Evaluator, parent, b: BuildingPick) -> None:
    node = evaluator.add_parallel(
        id="Parking_Availability",
        desc="The building must provide parking at a ratio of at least 4 spaces per 1,000 square feet of leased office space",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(b.parking_ratio and b.parking_ratio.strip()),
        id="parking_claim_provided",
        desc="Answer provides a parking ratio statement",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="parking_ratio_supported",
        desc="Cited sources support parking at least 4 spaces per 1,000 SF",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The building offers parking at a ratio of at least 4 spaces per 1,000 square feet.",
        node=leaf,
        sources=_merge_sources(b.sources_parking, b.sources_building),
        additional_instruction="Confirm the stated parking ratio is ≥ 4 per 1,000 SF. Accept equivalent phrasing (e.g., 4/1,000 SF)."
    )


async def verify_ada_compliance(evaluator: Evaluator, parent, b: BuildingPick) -> None:
    node = evaluator.add_parallel(
        id="ADA_Compliance",
        desc="The building must meet ADA accessibility requirements including accessible entrances, pathways at least 36 inches wide, and elevator access (for multi-story buildings)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(b.ada_compliance and b.ada_compliance.strip()),
        id="ada_claim_provided",
        desc="Answer provides an ADA compliance statement",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="ada_supported",
        desc="Cited sources support ADA compliance (accessible entrance, accessible routes, and elevators for multi-story buildings)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The building is ADA compliant, including accessible entrance, accessible routes, and elevator access (if multi-story).",
        node=leaf,
        sources=_merge_sources(b.sources_ada, b.sources_building),
        additional_instruction="Look for explicit mentions of ADA compliance, accessibility features, accessible entry, routes/hallways, and elevators. Do not infer without explicit wording."
    )


async def verify_geographic_location(evaluator: Evaluator, parent, b: BuildingPick) -> None:
    node = evaluator.add_parallel(
        id="Geographic_Location",
        desc="The building must be located in Hartford, Connecticut (identified as the #1 hottest real estate market for 2026)",
        parent=parent,
        critical=True
    )

    # We treat the core requirement as confirming Hartford, CT location.
    has_city_state = (b.city and b.city.strip()) and (b.state and b.state.strip())
    evaluator.add_custom_node(
        result=bool(has_city_state),
        id="location_claim_provided",
        desc="Answer provides city and state information for the building",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="location_supported",
        desc="Cited sources support that the building is located in Hartford, Connecticut",
        parent=node,
        critical=True
    )
    full_addr = _format_full_address(b)
    claim = f"The building at '{full_addr}' is located in Hartford, Connecticut."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_merge_sources(b.sources_location, b.sources_building),
        additional_instruction="Confirm that the building is in Hartford, CT. Allow minor address formatting differences."
    )


async def verify_sustainability_certification(evaluator: Evaluator, parent, b: BuildingPick) -> None:
    node = evaluator.add_parallel(
        id="Sustainability_Certification",
        desc="The building must have LEED certification at Silver level or higher (minimum 50 points under LEED rating system)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(b.leed_level and b.leed_level.strip()),
        id="leed_claim_provided",
        desc="Answer provides a LEED certification level for the building",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="leed_supported",
        desc="Cited sources support that the building has LEED certification at Silver level or higher",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The building has LEED certification at Silver level or higher.",
        node=leaf,
        sources=_merge_sources(b.sources_leed, b.sources_building),
        additional_instruction="Verify that the page explicitly states the building’s LEED level (Silver, Gold, or Platinum)."
    )


async def verify_lease_terms(evaluator: Evaluator, parent, b: BuildingPick) -> None:
    node = evaluator.add_parallel(
        id="Lease_Terms",
        desc="Available space must be offered under a gross lease or modified gross lease structure (not triple net lease)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(b.lease_type and b.lease_type.strip()),
        id="lease_claim_provided",
        desc="Answer provides a lease structure (e.g., Gross, Modified Gross, Full Service, or NNN)",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="lease_supported",
        desc="Cited sources support that the space is offered as Gross or Modified Gross (not NNN)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The available space is offered under a gross or modified gross lease (i.e., not triple net).",
        node=leaf,
        sources=_merge_sources(b.sources_lease, b.sources_building),
        additional_instruction="Confirm that the lease type is described as Gross, Modified Gross, or Full Service. If the page states NNN/triple net instead, the claim should be considered unsupported."
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
    Evaluate an answer for the Hartford Class A office requirement task.
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
        default_model=model
    )

    # Extract structured info about the single selected building
    extracted = await evaluator.extract(
        prompt=prompt_extract_building_pick(),
        template_class=BuildingPick,
        extraction_name="selected_building"
    )

    # Add constraints as ground truth/context
    evaluator.add_ground_truth({
        "constraints": GROUND_TRUTH_CONSTRAINTS,
        "notes": "All criteria are mandatory. Verification requires cited web sources supporting each factual claim."
    }, gt_type="constraints")

    # Create the top critical node mirroring the rubric root
    top = evaluator.add_parallel(
        id="Suitable_Class_A_Office_Building",
        desc="A Class A office building in Hartford, Connecticut that meets all specified requirements for a company expansion to accommodate 120 employees",
        parent=root,
        critical=True
    )

    # Build and verify each critical criterion subtree
    await verify_building_identification(evaluator, top, extracted)
    await verify_building_classification(evaluator, top, extracted)
    await verify_office_space_capacity(evaluator, top, extracted)
    await verify_parking_availability(evaluator, top, extracted)
    await verify_ada_compliance(evaluator, top, extracted)
    await verify_geographic_location(evaluator, top, extracted)
    await verify_sustainability_certification(evaluator, top, extracted)
    await verify_lease_terms(evaluator, top, extracted)

    return evaluator.get_summary()