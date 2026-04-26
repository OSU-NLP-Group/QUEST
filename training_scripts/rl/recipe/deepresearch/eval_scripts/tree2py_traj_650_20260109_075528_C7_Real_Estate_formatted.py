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
TASK_ID = "sf_class_a_office_building"
TASK_DESCRIPTION = (
    "A commercial real estate brokerage firm is compiling a portfolio of premium office buildings in downtown San Francisco "
    "for a major corporate client. They need to identify one qualifying Class A office tower that meets the following specifications:\n\n"
    "Required Specifications:\n"
    "- Class A office space designation\n"
    "- Located in downtown San Francisco\n"
    "- LEED Gold or Platinum certification\n"
    "- Minimum 400,000 rentable square feet total building size\n"
    "- Floor plates of at least 14,000 square feet\n"
    "- On-site parking availability\n"
    "- ADA-compliant elevator systems\n"
    "- Floor-to-ceiling heights of at least 9 feet\n"
    "- 24/7 security services\n"
    "- High-speed internet connectivity\n"
    "- Modern HVAC system with after-hours capability\n\n"
    "Preferred Features:\n"
    "- On-site fitness center\n"
    "- On-site cafe or food service\n"
    "- ENERGY STAR certification\n\n"
    "Identify one office building in downtown San Francisco that meets all the required specifications. Provide the building's name, "
    "address, total square footage, LEED certification level, and confirm which required and preferred features it has."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BuildingInfo(BaseModel):
    """Structured representation of a single identified building and its features, extracted from the answer."""
    name: Optional[str] = None
    address: Optional[str] = None

    class_designation: Optional[str] = None               # e.g., "Class A"
    leed_level: Optional[str] = None                      # e.g., "LEED Gold", "LEED Platinum"
    total_square_footage: Optional[str] = None            # total rentable SF (string as written in the answer)
    floor_plate_size: Optional[str] = None                # typical/average or minimum floor plate SF

    parking: Optional[str] = None                         # mention of on-site parking
    ada_elevators: Optional[str] = None                   # ADA compliance of elevators
    ceiling_height: Optional[str] = None                  # floor-to-ceiling height (string)
    security: Optional[str] = None                        # mention of 24/7 security
    internet: Optional[str] = None                        # high-speed internet/fiber connectivity
    hvac: Optional[str] = None                            # modern HVAC with after-hours capability

    # Preferred features
    fitness_center: Optional[str] = None                  # on-site gym/fitness center
    food_service: Optional[str] = None                    # on-site cafe/food service
    energy_star: Optional[str] = None                     # ENERGY STAR certification mention

    # Source URLs explicitly included in the answer
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_building_info() -> str:
    return """
    Extract exactly one building proposed in the answer that is intended to meet the required specifications for a Class A office tower
    in downtown San Francisco. If multiple buildings are mentioned, choose the primary building the answer is recommending (or the first one listed).
    Return a JSON object with the following fields exactly as they appear in the answer:

    Required identifying details:
    - name: Building name
    - address: Street address, city, state, and zip if provided

    Required specification-related details (strings as written):
    - class_designation: Building class designation (e.g., "Class A")
    - leed_level: LEED certification level (e.g., "LEED Gold" or "LEED Platinum")
    - total_square_footage: The total rentable square feet of the building, written as in the answer (e.g., "560,000 RSF")
    - floor_plate_size: The typical/minimum floor plate size, written as in the answer (e.g., "18,000 SF plates")
    - parking: Any statement about on-site parking
    - ada_elevators: Any statement about ADA-compliant elevator systems
    - ceiling_height: Floor-to-ceiling height (e.g., "10 ft")
    - security: Any statement about 24/7 security services
    - internet: Any statement about high-speed internet connectivity (e.g., "fiber")
    - hvac: Any statement about modern HVAC and after-hours capability

    Preferred features (strings as written):
    - fitness_center: Any statement about an on-site fitness center or gym
    - food_service: Any statement about an on-site cafe or food service
    - energy_star: Any statement indicating ENERGY STAR certification

    Sources:
    - source_urls: An array of all URLs explicitly provided in the answer that support the information for this building.
      Include only actual URLs. Do not invent or infer URLs.

    If a field is not present in the answer, set it to null. For source_urls, only include URLs that appear in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(value: Optional[str]) -> bool:
    return bool(value and str(value).strip())


def _first_n_urls(urls: List[str], n: int = 6) -> List[str]:
    """Limit the number of URLs passed to verification to avoid overly long multi-URL checks."""
    return urls[:n] if urls else []


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_qualifying_building_verifications(
    evaluator: Evaluator,
    parent_node,
    b: BuildingInfo
) -> None:
    """
    Build the verification tree for the qualifying building and execute verifications.
    We keep the 'qualifying_building' node non-critical so preferred features can be soft (non-critical),
    while all required specs are added as critical children under it.
    """
    # Create main node for qualifying building (set non-critical to allow preferred features as non-critical children)
    qb_node = evaluator.add_parallel(
        id="qualifying_building",
        desc="The identified building meets all required specifications for a Class A office tower and provides the requested identifying details.",
        parent=parent_node,
        critical=False
    )

    # Common sources for verification
    sources = _first_n_urls(b.source_urls, 8)

    # ---------------------- Basic identifying details (critical) ----------------------
    evaluator.add_custom_node(
        result=_has_text(b.name),
        id="building_name",
        desc="The solution provides the name of the building.",
        parent=qb_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(b.address),
        id="building_address",
        desc="The solution provides the address of the building.",
        parent=qb_node,
        critical=True
    )

    # ---------------------- Class designation (critical) -----------------------------
    class_node = evaluator.add_sequential(
        id="building_class_main",
        desc="Class A office space designation is provided and supported.",
        parent=qb_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(b.class_designation),
        id="building_class_provided",
        desc="Class designation is provided in the solution.",
        parent=class_node,
        critical=True
    )

    class_leaf = evaluator.add_leaf(
        id="building_class_supported",
        desc="The building is designated as Class A office space.",
        parent=class_node,
        critical=True
    )
    class_claim = f"The building '{b.name or 'the building'}' at '{b.address or 'its stated address'}' is designated as Class A office space."
    await evaluator.verify(
        claim=class_claim,
        node=class_leaf,
        sources=sources,
        additional_instruction="Confirm that the property is explicitly described as 'Class A' on the cited webpages."
    )

    # ---------------------- Downtown San Francisco location (critical) --------------
    location_leaf = evaluator.add_leaf(
        id="location_downtown",
        desc="The building is located in downtown San Francisco.",
        parent=qb_node,
        critical=True
    )
    location_claim = f"The building '{b.name or 'the building'}' is located in downtown San Francisco."
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the building is in downtown San Francisco. Accept mentions of central downtown neighborhoods such as "
            "Financial District (including South Financial), Embarcadero, Union Square, Mid-Market, parts of SoMa adjacent to downtown, or zip codes like 94104, 94105, 94111, 94108. "
            "If the source clearly indicates the building is within the downtown core, consider this supported."
        )
    )

    # ---------------------- Total square footage (critical) -------------------------
    sqft_node = evaluator.add_sequential(
        id="total_square_footage_main",
        desc="Total rentable square footage is provided and meets the minimum.",
        parent=qb_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(b.total_square_footage),
        id="total_square_footage_provided",
        desc="The solution provides the building’s total rentable square footage.",
        parent=sqft_node,
        critical=True
    )

    sqft_leaf = evaluator.add_leaf(
        id="total_square_footage_meets_minimum",
        desc="The building has at least 400,000 rentable square feet.",
        parent=sqft_node,
        critical=True
    )
    sqft_claim = "The building has at least 400,000 rentable square feet (RSF)."
    await evaluator.verify(
        claim=sqft_claim,
        node=sqft_leaf,
        sources=sources,
        additional_instruction=(
            "Look for 'total building size' or 'rentable square feet (RSF)' numbers. If the RSF is ≥ 400,000, this is supported. "
            "If only gross SF is available but clearly indicates ≥ 400,000 and context suggests rentable area is comparable, consider it acceptable."
        )
    )

    # ---------------------- LEED certification (critical) ---------------------------
    leed_node = evaluator.add_sequential(
        id="leed_certification_main",
        desc="LEED certification level is provided and valid.",
        parent=qb_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(b.leed_level),
        id="leed_certification_level_provided",
        desc="The solution provides the building’s LEED certification level.",
        parent=leed_node,
        critical=True
    )

    leed_leaf = evaluator.add_leaf(
        id="leed_certification_gold_or_platinum",
        desc="The building’s LEED certification is Gold or Platinum.",
        parent=leed_node,
        critical=True
    )
    leed_claim = "The building is certified LEED Gold or LEED Platinum."
    await evaluator.verify(
        claim=leed_claim,
        node=leed_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the building's LEED level is Gold or Platinum (any version, e.g., LEED EB:O&M Gold). "
            "If sources explicitly state Gold or Platinum certification, mark as supported."
        )
    )

    # ---------------------- Floor plates (critical) ---------------------------------
    fp_node = evaluator.add_sequential(
        id="floor_plate_size_main",
        desc="Floor plate size is provided and meets the minimum.",
        parent=qb_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(b.floor_plate_size),
        id="floor_plate_size_provided",
        desc="Floor plate size information is provided.",
        parent=fp_node,
        critical=True
    )

    fp_leaf = evaluator.add_leaf(
        id="floor_plate_size_meets_minimum",
        desc="Floor plates are at least 14,000 square feet.",
        parent=fp_node,
        critical=True
    )
    fp_claim = "Floor plates are at least 14,000 square feet."
    await evaluator.verify(
        claim=fp_claim,
        node=fp_leaf,
        sources=sources,
        additional_instruction=(
            "Check for typical or average floor plate size. Accept synonyms like 'typical floor size', 'floor plate', etc. "
            "If any typical/average/minimum floor plate size is ≥ 14,000 SF, consider this supported."
        )
    )

    # ---------------------- Parking (critical) --------------------------------------
    parking_node = evaluator.add_sequential(
        id="parking_main",
        desc="Parking availability is provided and supported.",
        parent=qb_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(b.parking),
        id="parking_provided",
        desc="On-site parking information is provided.",
        parent=parking_node,
        critical=True
    )
    parking_leaf = evaluator.add_leaf(
        id="parking_available",
        desc="On-site parking is available in the building.",
        parent=parking_node,
        critical=True
    )
    parking_claim = "On-site parking is available at the building."
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=sources,
        additional_instruction="Look for mentions of 'on-site parking', 'parking garage', or similar facility within the building."
    )

    # ---------------------- ADA-compliant elevators (critical) ----------------------
    ada_node = evaluator.add_sequential(
        id="ada_elevators_main",
        desc="ADA elevator compliance is provided and supported.",
        parent=qb_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(b.ada_elevators),
        id="ada_elevators_provided",
        desc="ADA-compliant elevator information is provided.",
        parent=ada_node,
        critical=True
    )
    ada_leaf = evaluator.add_leaf(
        id="ada_elevators",
        desc="The building has ADA-compliant elevator systems.",
        parent=ada_node,
        critical=True
    )
    ada_claim = "The building has ADA-compliant elevator systems."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=sources,
        additional_instruction="Confirm ADA compliance of elevator systems or accessibility features that imply elevator ADA compliance."
    )

    # ---------------------- Ceiling height (critical) --------------------------------
    ch_node = evaluator.add_sequential(
        id="ceiling_height_main",
        desc="Ceiling height is provided and meets the minimum.",
        parent=qb_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(b.ceiling_height),
        id="ceiling_height_provided",
        desc="Floor-to-ceiling height information is provided.",
        parent=ch_node,
        critical=True
    )
    ch_leaf = evaluator.add_leaf(
        id="ceiling_height_meets_minimum",
        desc="Floor-to-ceiling heights are at least 9 feet.",
        parent=ch_node,
        critical=True
    )
    ch_claim = "Floor-to-ceiling heights are at least 9 feet."
    await evaluator.verify(
        claim=ch_claim,
        node=ch_leaf,
        sources=sources,
        additional_instruction="Accept mentions of 'floor-to-ceiling height', 'clear height', or similar indicating ≥ 9 ft."
    )

    # ---------------------- 24/7 security (critical) --------------------------------
    sec_node = evaluator.add_sequential(
        id="security_system_main",
        desc="Security information is provided and supported.",
        parent=qb_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(b.security),
        id="security_system_provided",
        desc="Security service information is provided.",
        parent=sec_node,
        critical=True
    )
    sec_leaf = evaluator.add_leaf(
        id="security_system",
        desc="The building provides 24/7 security services.",
        parent=sec_node,
        critical=True
    )
    sec_claim = "The building provides 24/7 security services."
    await evaluator.verify(
        claim=sec_claim,
        node=sec_leaf,
        sources=sources,
        additional_instruction="Look for '24/7 security', '24-hour security', or equivalent phrasing."
    )

    # ---------------------- High-speed internet (critical) ---------------------------
    net_node = evaluator.add_sequential(
        id="high_speed_internet_main",
        desc="Internet connectivity information is provided and supported.",
        parent=qb_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(b.internet),
        id="high_speed_internet_provided",
        desc="High-speed internet connectivity is mentioned.",
        parent=net_node,
        critical=True
    )
    net_leaf = evaluator.add_leaf(
        id="high_speed_internet",
        desc="High-speed internet connectivity is available.",
        parent=net_node,
        critical=True
    )
    net_claim = "High-speed internet connectivity is available at the building."
    await evaluator.verify(
        claim=net_claim,
        node=net_leaf,
        sources=sources,
        additional_instruction="Accept mentions of 'fiber connectivity', 'high-speed internet', or similar language indicating robust connectivity."
    )

    # ---------------------- Modern HVAC with after-hours (critical) ------------------
    hvac_node = evaluator.add_sequential(
        id="hvac_system_main",
        desc="HVAC information is provided and supported.",
        parent=qb_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(b.hvac),
        id="hvac_system_provided",
        desc="HVAC system information is provided.",
        parent=hvac_node,
        critical=True
    )
    hvac_leaf = evaluator.add_leaf(
        id="hvac_system",
        desc="A modern HVAC system with after-hours capability is available.",
        parent=hvac_node,
        critical=True
    )
    hvac_claim = "A modern HVAC system with after-hours capability is available."
    await evaluator.verify(
        claim=hvac_claim,
        node=hvac_leaf,
        sources=sources,
        additional_instruction="Look for 'after-hours HVAC', 'OTAC', 'after-hours air', or equivalent capability in building specs."
    )

    # ---------------------- Preferred features (non-critical) ------------------------
    fit_node = evaluator.add_sequential(
        id="fitness_center_main",
        desc="Preferred feature: On-site fitness center is provided and supported (non-critical).",
        parent=qb_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_has_text(b.fitness_center),
        id="fitness_center_provided",
        desc="Preferred feature: On-site fitness center is mentioned.",
        parent=fit_node,
        critical=False
    )
    fit_leaf = evaluator.add_leaf(
        id="fitness_center",
        desc="Preferred feature: An on-site fitness center is available.",
        parent=fit_node,
        critical=False
    )
    fit_claim = "An on-site fitness center is available at the building."
    await evaluator.verify(
        claim=fit_claim,
        node=fit_leaf,
        sources=sources,
        additional_instruction="Accept mentions of 'fitness center', 'gym', or similar on-site amenity."
    )

    food_node = evaluator.add_sequential(
        id="food_service_main",
        desc="Preferred feature: On-site cafe or food service is provided and supported (non-critical).",
        parent=qb_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_has_text(b.food_service),
        id="food_service_provided",
        desc="Preferred feature: On-site cafe/food service is mentioned.",
        parent=food_node,
        critical=False
    )
    food_leaf = evaluator.add_leaf(
        id="food_service",
        desc="Preferred feature: An on-site cafe or food service is available.",
        parent=food_node,
        critical=False
    )
    food_claim = "An on-site cafe or food service is available at the building."
    await evaluator.verify(
        claim=food_claim,
        node=food_leaf,
        sources=sources,
        additional_instruction="Accept mentions of on-site cafe, coffee shop, food hall, or food service."
    )

    es_node = evaluator.add_sequential(
        id="energy_star_main",
        desc="Preferred feature: ENERGY STAR certification is provided and supported (non-critical).",
        parent=qb_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_has_text(b.energy_star),
        id="energy_star_provided",
        desc="Preferred feature: ENERGY STAR certification is mentioned.",
        parent=es_node,
        critical=False
    )
    es_leaf = evaluator.add_leaf(
        id="energy_star",
        desc="Preferred feature: The building has ENERGY STAR certification.",
        parent=es_node,
        critical=False
    )
    es_claim = "The building has ENERGY STAR certification."
    await evaluator.verify(
        claim=es_claim,
        node=es_leaf,
        sources=sources,
        additional_instruction="Confirm explicit mention of ENERGY STAR certification in the cited sources."
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
    Evaluate an answer for the San Francisco Class A office building task.
    """
    # Initialize evaluator (root node is non-critical by default; we use PARALLEL aggregation)
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

    # Extract the proposed building information
    building_info = await evaluator.extract(
        prompt=prompt_extract_building_info(),
        template_class=BuildingInfo,
        extraction_name="building_info"
    )

    # Build verification tree and run checks
    await build_qualifying_building_verifications(
        evaluator=evaluator,
        parent_node=root,
        b=building_info
    )

    # Return structured summary
    return evaluator.get_summary()