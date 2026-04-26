import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "atl_office_buildings_classA_relocation"
TASK_DESCRIPTION = """
I am evaluating potential office locations in Atlanta for a corporate relocation. Please identify three Class A office buildings located in either Downtown Atlanta or Midtown Atlanta that meet ALL of the following requirements:

1. The building must be classified as Class A office space
2. The building must have achieved LEED Gold or Platinum certification
3. The building must contain at least 200,000 gross square feet of office space
4. The building must provide a parking ratio of at least 4 parking spaces per 1,000 square feet of office space
5. The building must be located within 0.5 miles (approximately 0.8 kilometers) of a MARTA station
6. The building must have ENERGY STAR certification with a score of 75 or higher
7. The building must have fiber-optic internet connectivity infrastructure
8. The building must be fully ADA-compliant with accessible entrances, elevators, and common areas
9. The building must have been either constructed or undergone major renovation after 2010
10. The building must include at least two of the following on-site amenities: fitness center, conference center, food service/café, or outdoor terrace/plaza space

For each of the three buildings, provide:
- The building name
- The complete street address
- A reference URL that supports the building information and confirms it meets the criteria
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BuildingItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    class_a_status: Optional[str] = None               # e.g., "Class A", or description
    leed_level: Optional[str] = None                   # e.g., "LEED Gold", "LEED Platinum"
    gross_sqft: Optional[str] = None                   # keep as string to allow ranges/notes
    parking_ratio: Optional[str] = None                # e.g., "4/1,000", "4 per 1,000"
    marta_station: Optional[str] = None                # nearest MARTA station name if given
    marta_distance_miles: Optional[str] = None         # e.g., "0.3 miles", keep as string
    energy_star_score: Optional[str] = None            # e.g., "75", "85"
    fiber_connectivity: Optional[str] = None           # description; e.g., "fiber-ready", "AT&T fiber"
    ada_compliance: Optional[str] = None               # description; e.g., "ADA-compliant"
    year_built_or_renovated: Optional[str] = None      # e.g., "Built 2013", "Renovated 2018"
    amenities: List[str] = Field(default_factory=list) # list of amenities mentioned


class BuildingsExtraction(BaseModel):
    buildings: List[BuildingItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_buildings() -> str:
    return """
Extract up to 5 candidate Class A office buildings in Atlanta listed in the answer. For each building, extract:

1) name: Building name (string)
2) address: Complete street address (string)
3) reference_urls: All URLs explicitly cited in the answer that directly describe or substantiate the building information (array of URLs). Only include actual URLs mentioned.
4) class_a_status: The portion of the answer that indicates the building is Class A (string; e.g., "Class A", "Trophy/Class A", or a sentence stating Class A)
5) leed_level: LEED certification level text if given (string; e.g., "LEED Gold", "LEED Platinum")
6) gross_sqft: Total or office gross square footage (string; keep units like "SF", "sq ft")
7) parking_ratio: Parking ratio text (string; e.g., "4/1,000", "4 per 1,000", "4:1000")
8) marta_station: Name of nearest MARTA rail station if mentioned (string, else null)
9) marta_distance_miles: Proximity distance if given (string; e.g., "0.2 miles", "2 blocks", else null)
10) energy_star_score: ENERGY STAR score value if given (string; e.g., "75", "85", else null)
11) fiber_connectivity: Text indicating fiber presence/connectivity (string; e.g., "fiber-ready", "AT&T fiber", else null)
12) ada_compliance: Text indicating ADA compliance/accessibility (string; else null)
13) year_built_or_renovated: Build year or major renovation year(s) (string; e.g., "Built 2012", "Renovated 2019")
14) amenities: List of amenities explicitly mentioned for the building (array of strings). Include terms like "fitness center", "conference center", "food service", "café", "outdoor terrace", "plaza", "roof deck", "outdoor patio".

Rules:
- Return fields exactly as stated in the answer; do not invent.
- If a field is missing in the answer, set it to null (or empty array for lists).
- Only include valid URLs that are explicitly present in the answer text (plain or markdown).
- Do not normalize numbers; keep strings (e.g., "200,000 SF", "4/1,000").
- The "amenities" list should only include amenities explicitly mentioned in the answer.
Output as: { "buildings": [ ... ] }.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def building_label(idx: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][idx] if idx < 5 else f"Building_{idx+1}"


def building_node_id(idx: int) -> str:
    return f"{building_label(idx)}_Building"


def get_sources(b: Optional[BuildingItem]) -> List[str]:
    if not b:
        return []
    if not b.reference_urls:
        return []
    return [u for u in b.reference_urls if isinstance(u, str) and len(u.strip()) > 0]


# --------------------------------------------------------------------------- #
# Verification for one building                                               #
# --------------------------------------------------------------------------- #
async def verify_single_building(
    evaluator: Evaluator,
    parent_node,
    b: BuildingItem,
    idx: int,
) -> None:
    label = building_label(idx)
    node_id = building_node_id(idx)

    # Create parent node for this building (parallel; non-critical so root can average across buildings)
    building_node = evaluator.add_parallel(
        id=node_id,
        desc=f"{label} qualifying Class A office building meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # 1) Identification exists (name + address)
    identification_exists = (
        (b is not None)
        and (b.name is not None and b.name.strip() != "")
        and (b.address is not None and b.address.strip() != "")
    )
    evaluator.add_custom_node(
        result=identification_exists,
        id=f"{label}_Building_Identification",
        desc="Provides the building name and complete street address",
        parent=building_node,
        critical=True,
    )

    # 2) URL Reference exists (at least one URL provided in the answer)
    urls = get_sources(b)
    url_exists = len(urls) > 0
    evaluator.add_custom_node(
        result=url_exists,
        id=f"{label}_Building_URL_Reference",
        desc="Provides a reference URL supporting the building information",
        parent=building_node,
        critical=True,
    )

    # Below: All are factual checks grounded by the provided URL(s)

    # 3) Class A status
    node = evaluator.add_leaf(
        id=f"{label}_Building_Class_A_Status",
        desc="Verifies the building is classified as Class A office space",
        parent=building_node,
        critical=True,
    )
    claim = f"The building named '{b.name or 'the building'}' is classified as Class A office space."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Look for explicit mentions such as 'Class A', 'Trophy/Class A', or marketing text clearly indicating Class A classification on the provided page(s). Allow minor text variations.",
    )

    # 4) LEED certification Gold or Platinum
    node = evaluator.add_leaf(
        id=f"{label}_Building_LEED_Certification",
        desc="Verifies the building has LEED Gold or Platinum certification",
        parent=building_node,
        critical=True,
    )
    claim = f"The building '{b.name or 'the building'}' has achieved LEED Gold or LEED Platinum certification."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Accept proof such as 'LEED Gold', 'LEED Platinum', or official LEED citations. Do not accept Silver/Certified only.",
    )

    # 5) Size >= 200,000 GSF (office)
    node = evaluator.add_leaf(
        id=f"{label}_Building_Size",
        desc="Verifies the building contains at least 200,000 gross square feet of office space",
        parent=building_node,
        critical=True,
    )
    claim = f"The building '{b.name or 'the building'}' contains at least 200,000 gross square feet of office space."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Confirm from page text that total or office gross square footage is ≥ 200,000 SF. Accept minor formatting (e.g., 'SF', 'sq ft'). If multiple figures are given, prefer total/office GSF.",
    )

    # 6) Parking ratio >= 4 / 1,000 SF
    node = evaluator.add_leaf(
        id=f"{label}_Building_Parking_Ratio",
        desc="Verifies the building provides at least 4 parking spaces per 1,000 square feet",
        parent=building_node,
        critical=True,
    )
    claim = f"The building '{b.name or 'the building'}' provides a parking ratio of at least 4 spaces per 1,000 square feet of office space."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Look for '4/1,000', '4:1000', '4 per 1,000', or clear equivalent. If multiple ratios are given, use the office ratio.",
    )

    # 7) Within 0.5 miles of a MARTA station
    node = evaluator.add_leaf(
        id=f"{label}_Building_MARTA_Proximity",
        desc="Verifies the building is within 0.5 miles of a MARTA station",
        parent=building_node,
        critical=True,
    )
    claim = f"The building '{b.name or 'the building'}' is within 0.5 miles (approximately 0.8 km) of a MARTA rail station."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Accept explicit distances ≤ 0.5 miles or clear phrases like 'adjacent to', 'connected to', or 'within a few blocks of' a named MARTA station (e.g., Midtown, Arts Center, North Avenue, Civic Center, Peachtree Center, Five Points).",
    )

    # 8) ENERGY STAR score >= 75
    node = evaluator.add_leaf(
        id=f"{label}_Building_Energy_Star",
        desc="Verifies the building has ENERGY STAR certification with a score of 75 or higher",
        parent=building_node,
        critical=True,
    )
    claim = f"The building '{b.name or 'the building'}' has ENERGY STAR certification with a score of at least 75."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Look for explicit ENERGY STAR certification and a score value ≥ 75. If certification is mentioned without a score, consider the 'score ≥75' requirement not satisfied.",
    )

    # 9) Fiber-optic connectivity
    node = evaluator.add_leaf(
        id=f"{label}_Building_Fiber_Connectivity",
        desc="Verifies the building has fiber-optic internet connectivity",
        parent=building_node,
        critical=True,
    )
    claim = f"The building '{b.name or 'the building'}' has fiber-optic internet connectivity infrastructure."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Accept terms like 'fiber', 'fiber-ready', 'fiber backbone', 'gigabit fiber', 'AT&T fiber', 'Google Fiber', or equivalent indicating fiber connectivity in the building.",
    )

    # 10) ADA compliance
    node = evaluator.add_leaf(
        id=f"{label}_Building_ADA_Compliance",
        desc="Verifies the building is fully ADA-compliant",
        parent=building_node,
        critical=True,
    )
    claim = f"The building '{b.name or 'the building'}' is fully ADA-compliant with accessible entrances, elevators, and common areas."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Accept explicit statements like 'ADA compliant', or equivalent evidence of accessibility features across entrances, elevators, and common areas.",
    )

    # 11) Submarket location (Downtown or Midtown Atlanta)
    node = evaluator.add_leaf(
        id=f"{label}_Building_Submarket_Location",
        desc="Verifies the building is located in Downtown Atlanta or Midtown Atlanta",
        parent=building_node,
        critical=True,
    )
    claim = f"The building '{b.name or 'the building'}' is located in either Downtown Atlanta or Midtown Atlanta."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Look for explicit mentions of 'Downtown' or 'Midtown' Atlanta on the page, or credible references tying the building to these submarkets (e.g., 'Peachtree Center in Downtown', 'Midtown core', etc.).",
    )

    # 12) Built or major renovation after 2010
    node = evaluator.add_leaf(
        id=f"{label}_Building_Age_or_Renovation",
        desc="Verifies the building was constructed or underwent major renovation after 2010",
        parent=building_node,
        critical=True,
    )
    claim = f"The building '{b.name or 'the building'}' was constructed or underwent a major renovation after 2010."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Accept explicit build years ≥ 2011, or major renovation years ≥ 2011 (e.g., 'Renovated 2015'). Marketing references to 'recently renovated' must be tied to a concrete year ≥ 2011.",
    )

    # 13) Amenities: at least two of [fitness center, conference center, food service/café, outdoor terrace/plaza]
    node = evaluator.add_leaf(
        id=f"{label}_Building_Amenities",
        desc="Verifies the building includes at least two of the following: fitness center, conference center, food service/café, or outdoor terrace/plaza",
        parent=building_node,
        critical=True,
    )
    claim = (
        f"The building '{b.name or 'the building'}' has at least two of these on-site amenities: "
        f"fitness center, conference center, food service/café, outdoor terrace/plaza."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction="Confirm that at least two of the specified amenities are explicitly present. Accept close variants: 'fitness facility/gym', 'conference center/meeting center', 'food service/café/restaurant/coffee bar', 'outdoor terrace/plaza/patio/roof deck'.",
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
    Entry point for evaluating the agent's answer for the Atlanta office building criteria task.
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

    # 1) Extract buildings proposed in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_buildings(),
        template_class=BuildingsExtraction,
        extraction_name="buildings_extraction",
    )

    # 2) Normalize to exactly three buildings (pad with empty if fewer)
    buildings: List[BuildingItem] = list(extracted.buildings[:3])
    while len(buildings) < 3:
        buildings.append(BuildingItem())

    # 3) Build verification subtrees (one per building)
    for i in range(3):
        await verify_single_building(evaluator, root, buildings[i], i)

    # 4) Return structured result
    return evaluator.get_summary()