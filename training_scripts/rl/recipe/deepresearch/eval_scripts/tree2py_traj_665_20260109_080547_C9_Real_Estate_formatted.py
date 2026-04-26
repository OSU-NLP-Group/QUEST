import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "class_a_multifamily_4_cities"
TASK_DESCRIPTION = (
    "Identify one Class A multifamily apartment property in each of the following four major US cities: "
    "Phoenix (Arizona), Atlanta (Georgia), Dallas (Texas), and Chicago (Illinois). For each property, provide: "
    "1) The property name and full address, 2) Verification that the property meets ALL of the following investment criteria: "
    "Is classified as Class A (built within the last 10-15 years OR substantially renovated to Class A standards, in top condition with no significant deferred maintenance), "
    "has a minimum of 100 units, maintains a physical occupancy rate of at least 85%, has a unit mix following industry standards with approximately 2 two-bedroom units for every 1 one-bedroom unit "
    "(roughly 65-67% two-bedrooms, 33-35% one-bedrooms), includes a state-of-the-art fitness center with a minimum size of 500 square feet, includes a swimming pool with proper safety barriers "
    "(minimum 60-inch height), provides parking at a ratio of at least 1 space per unit, is well-located within its metro area, is professionally managed, and demands competitive or above-average rents "
    "for its market. For each property, provide URL references that document the property information and verify the stated criteria."
)


# =========================
# Data Models for Extraction
# =========================

class PropertyInfo(BaseModel):
    city: Optional[str] = None
    name: Optional[str] = None
    address: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

    year_built: Optional[str] = None
    renovation_desc: Optional[str] = None
    class_rating: Optional[str] = None  # e.g., "Class A", "A", "Luxury"
    units: Optional[str] = None  # Keep as string to accommodate ranges or phrasing
    occupancy: Optional[str] = None  # e.g., "92%", "high-80s"
    unit_mix: Optional[str] = None  # Free-form description of 1BR/2BR mix
    fitness_desc: Optional[str] = None
    fitness_size: Optional[str] = None  # e.g., "700 sf"
    pool_desc: Optional[str] = None
    pool_barrier_height: Optional[str] = None  # e.g., "60-inch fence"
    parking_ratio: Optional[str] = None  # e.g., "1.2 per unit"
    location_desc: Optional[str] = None  # e.g., "prime location in Midtown"
    management: Optional[str] = None  # e.g., "Managed by XYZ Management"
    rent_positioning: Optional[str] = None  # e.g., "premium/market-leading rents"


class PropertiesExtraction(BaseModel):
    phoenix: Optional[PropertyInfo] = None
    atlanta: Optional[PropertyInfo] = None
    dallas: Optional[PropertyInfo] = None
    chicago: Optional[PropertyInfo] = None


# =====================
# Extraction Prompt
# =====================

def prompt_extract_properties() -> str:
    return """
Extract exactly one multifamily apartment property per city (Phoenix AZ, Atlanta GA, Dallas TX, Chicago IL) as presented in the answer. If more than one is mentioned for a city, pick the first one mentioned for that city. If a city is missing, return null for that city.

For each city, extract the following fields strictly as they appear in the answer:
- city: The city name (Phoenix, Atlanta, Dallas, or Chicago)
- name: Property name
- address: Full street address including city and state
- urls: An array of all URLs that the answer cites for that specific property (include any property website, listing, brochure, management page, or article; only URLs explicitly present in the answer)

Additionally, extract any directly stated or implied details (as strings exactly as the answer states them):
- year_built: Year (or phrase) indicating when built
- renovation_desc: Any statement describing substantial renovation or upgrade
- class_rating: Any label like "Class A", "luxury", "top-tier"
- units: Total unit count description (e.g., "220 units")
- occupancy: Physical occupancy rate statement (e.g., "92% occupied")
- unit_mix: Description of the 1BR and 2BR mix (e.g., "60% two-bedrooms, 30% one-bedrooms...")
- fitness_desc: Description of the fitness center
- fitness_size: Size of the fitness center if present (e.g., "700 square feet")
- pool_desc: Description of pool
- pool_barrier_height: Stated fence or barrier height if mentioned (e.g., "60-inch fence")
- parking_ratio: Parking ratio if given (e.g., "1.2 spaces per unit")
- location_desc: Any phrase suggesting strong location within the metro (e.g., "prime location", "walkable to major employers")
- management: Management company or statement of professional management
- rent_positioning: Any phrase suggesting competitive or above-average/premium rents

Return a JSON object with four top-level fields: phoenix, atlanta, dallas, chicago.
Each of those fields should be an object with the fields above, or null if not present in the answer.
Only include URLs explicitly present in the answer.
"""


# =====================
# Verification Utilities
# =====================

def _safe_name(prop: Optional[PropertyInfo]) -> str:
    return prop.name if (prop and prop.name) else "the property"

def _safe_city(prop: Optional[PropertyInfo]) -> str:
    return prop.city if (prop and prop.city) else "the city"

def _urls(prop: Optional[PropertyInfo]) -> List[str]:
    return prop.urls if (prop and prop.urls) else []


# =====================
# City Property Verification
# =====================

async def verify_city_property(
    evaluator: Evaluator,
    root_node,
    city_key: str,
    city_title: str,
    prop: Optional[PropertyInfo],
) -> None:
    """
    Build the sequential verification subtree for a given city and run verifications.
    """
    # Create the city node
    city_node = evaluator.add_sequential(
        id=f"{city_key}_Property",
        desc=f"{city_title}: one qualifying Class A multifamily property with required info, verification, and URLs",
        parent=root_node,
        critical=False
    )

    # 1) Identification exists (Critical)
    name_ok = bool(prop and prop.name and prop.name.strip())
    addr_ok = bool(prop and prop.address and prop.address.strip())
    id_node = evaluator.add_custom_node(
        result=(name_ok and addr_ok),
        id=f"{city_key}_Identification",
        desc=f"Provide property name and full address in {city_title.split(',')[0]}, {city_title.split(', ')[1]}",
        parent=city_node,
        critical=True
    )

    # 2) URLs exist (Critical)
    urls_ok = bool(_urls(prop))
    urls_node = evaluator.add_custom_node(
        result=urls_ok,
        id=f"{city_key}_URLs",
        desc="Provide URL references that document the property information and substantiate the stated criteria",
        parent=city_node,
        critical=True
    )

    # 3) Criteria (Critical parent, all children critical)
    criteria_node = evaluator.add_parallel(
        id=f"{city_key}_Criteria",
        desc=f"Verify the {city_title.split(',')[0]} property meets ALL investment criteria",
        parent=city_node,
        critical=True
    )

    # Prepare leaf nodes for each criterion
    # a) Class A + built last 10–15 years OR substantially renovated
    classa_node = evaluator.add_leaf(
        id=f"{city_key}_ClassA_BuildOrRenovation",
        desc="Verify Class A qualification: built within the last 10–15 years OR substantially renovated to Class A standards",
        parent=criteria_node,
        critical=True
    )

    # b) Top condition, no significant deferred maintenance
    condition_node = evaluator.add_leaf(
        id=f"{city_key}_TopCondition_NoDeferredMaintenance",
        desc="Verify the property is in top condition with no significant deferred maintenance",
        parent=criteria_node,
        critical=True
    )

    # c) Minimum 100 units
    units_node = evaluator.add_leaf(
        id=f"{city_key}_MinUnits",
        desc="Verify the property has at least 100 units",
        parent=criteria_node,
        critical=True
    )

    # d) Occupancy >= 85%
    occ_node = evaluator.add_leaf(
        id=f"{city_key}_Occupancy",
        desc="Verify physical occupancy rate is at least 85%",
        parent=criteria_node,
        critical=True
    )

    # e) Unit mix approximately 2:1 (2BR:1BR)
    mix_node = evaluator.add_leaf(
        id=f"{city_key}_UnitMix",
        desc="Verify unit mix is approximately 2 two-bedrooms per 1 one-bedroom (≈65–67% 2BR, 33–35% 1BR)",
        parent=criteria_node,
        critical=True
    )

    # f) Fitness center >= 500 sf
    fitness_node = evaluator.add_leaf(
        id=f"{city_key}_FitnessCenter",
        desc="Verify state-of-the-art fitness center is included and is at least 500 square feet",
        parent=criteria_node,
        critical=True
    )

    # g) Pool with safety barriers >= 60 inches
    pool_node = evaluator.add_leaf(
        id=f"{city_key}_PoolSafety",
        desc="Verify swimming pool is included and has proper safety barriers with minimum 60-inch height",
        parent=criteria_node,
        critical=True
    )

    # h) Parking ratio >= 1 per unit
    parking_node = evaluator.add_leaf(
        id=f"{city_key}_ParkingRatio",
        desc="Verify parking is provided at a ratio of at least 1 space per unit",
        parent=criteria_node,
        critical=True
    )

    # i) Well-located within metro
    location_node = evaluator.add_leaf(
        id=f"{city_key}_WellLocated",
        desc="Verify the property is well-located within its metro area",
        parent=criteria_node,
        critical=True
    )

    # j) Professionally managed
    mgmt_node = evaluator.add_leaf(
        id=f"{city_key}_ProfessionallyManaged",
        desc="Verify the property is professionally managed",
        parent=criteria_node,
        critical=True
    )

    # k) Competitive or above-average rents
    rents_node = evaluator.add_leaf(
        id=f"{city_key}_CompetitiveRents",
        desc="Verify the property demands competitive or above-average rents for its market",
        parent=criteria_node,
        critical=True
    )

    # Build claims and run verifications (batch) for criteria
    name = _safe_name(prop)
    cityname = _safe_city(prop)
    urls = _urls(prop)

    claims_and_sources = [
        (
            f"The property {name} in {cityname} is a Class A multifamily asset and either (a) was built in 2011 or later (within the last 15 years as of 2026) or (b) has been substantially renovated to Class A standards.",
            urls,
            classa_node,
            "Treat 'Class A', 'luxury', or equivalent language as Class A when clearly referring to the asset class. "
            "If the source provides a build year, 2011 or later qualifies for the 15-year window (as of 2026). "
            "If an older build is present, a clear, substantial renovation bringing it to Class A also qualifies."
        ),
        (
            f"The property {name} is in top condition with no significant deferred maintenance.",
            urls,
            condition_node,
            "Look for explicit or strong implied evidence: 'well-maintained', 'excellent condition', 'like-new', "
            "'recently renovated', or similar. If newly built or fully renovated and marketed as high-end Class A with no "
            "mentions of major issues, consider this criterion satisfied."
        ),
        (
            f"The property {name} has at least 100 units.",
            urls,
            units_node,
            "Check total unit count. If the site lists unit totals or a clear number >= 100, pass. If unknown or <100, fail."
        ),
        (
            f"The property {name} maintains a physical occupancy rate of at least 85%.",
            urls,
            occ_node,
            "Verify explicit occupancy disclosures. Accept >= 85% (allow rounding). If no occupancy info, fail."
        ),
        (
            f"The unit mix for {name} is approximately two two-bedroom units for every one one-bedroom unit (about 65–67% 2BR and 33–35% 1BR) when considering only 1BR and 2BR.",
            urls,
            mix_node,
            "Use floorplan counts or unit distribution if provided. Focus only on 1BR vs 2BR proportions; ignore studios/3BR when computing ratio. "
            "Accept approximate match: 2BR share between 60–70% or a 2BR:1BR ratio roughly between 1.8:1 and 2.2:1."
        ),
        (
            f"The property {name} includes a state-of-the-art fitness center of at least 500 square feet.",
            urls,
            fitness_node,
            "Confirm a dedicated fitness center/gym exists. Check for size; pass if an explicit size >= 500 sf is shown or clearly implied to exceed 500 sf."
        ),
        (
            f"The property {name} includes a swimming pool with safety barriers that are at least 60 inches in height.",
            urls,
            pool_node,
            "Look for explicit mention of a fence or safety barrier height of 60 inches (or 5 feet) or higher. "
            "If barrier heights are not stated, fail this criterion."
        ),
        (
            f"The property {name} provides parking at a ratio of at least 1 space per unit.",
            urls,
            parking_node,
            "Look for explicit parking ratio (e.g., '1.2 spaces per unit'). If not specified, fail."
        ),
        (
            f"The property {name} is well-located within its metro area.",
            urls,
            location_node,
            "Look for phrases indicating a strong location (e.g., 'prime location', 'desirable neighborhood', close proximity to jobs/transit/amenities). "
            "If the page clearly markets superior location within the metro, pass."
        ),
        (
            f"The property {name} is professionally managed.",
            urls,
            mgmt_node,
            "Look for the name of a professional property management company or explicit statements of professional management."
        ),
        (
            f"The property {name} commands competitive or above-average rents for its market.",
            urls,
            rents_node,
            "Look for phrasing such as 'premium rents', 'market-leading rents', 'top-of-market', or rent tables implying above-average pricing. "
            "If no comparative rent positioning is stated, fail."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


# =====================
# Main Evaluation Entry
# =====================

async def evaluate_answer(
    client: LLMClient,
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
        strategy=AggregationStrategy.PARALLEL,  # Parallel across cities
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

    # Extract structured info for all four cities from the answer
    extracted: PropertiesExtraction = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="properties_extraction"
    )

    # Build and verify each city subtree (sequential per city as per rubric)
    await verify_city_property(evaluator, root, "Phoenix", "Phoenix, AZ", extracted.phoenix)
    await verify_city_property(evaluator, root, "Atlanta", "Atlanta, GA", extracted.atlanta)
    await verify_city_property(evaluator, root, "Dallas", "Dallas, TX", extracted.dallas)
    await verify_city_property(evaluator, root, "Chicago", "Chicago, IL", extracted.chicago)

    return evaluator.get_summary()