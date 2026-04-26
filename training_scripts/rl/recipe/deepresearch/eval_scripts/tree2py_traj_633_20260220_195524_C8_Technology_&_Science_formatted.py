import asyncio
import logging
from typing import Any, List, Optional, Dict, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_dc_location_2026"
TASK_DESCRIPTION = (
    "Identify three suitable US metropolitan areas for establishing new large-scale AI data center facilities in 2026. "
    "For each location, provide a detailed analysis demonstrating that it meets the following requirements: "
    "(1) Power Infrastructure - The location must have access to at least 100 megawatts (MW) of available power capacity to support large-scale AI operations, "
    "with infrastructure capable of supporting AI-optimized rack densities of 60+ kilowatts per rack. "
    "(2) Established Market Presence - The location must be part of an established data center market with more than 1,000 MW of existing data center capacity, indicating proven infrastructure and ecosystem. "
    "(3) Water Resources - The location must have water availability capable of supporting approximately 200 million gallons annually for data center cooling operations. "
    "(4) Network Infrastructure - The location must have network infrastructure capable of supporting less than 5 milliseconds latency for inter-data-center communications. "
    "(5) Renewable Energy Access - The location must have documented access to renewable energy sources to support sustainability goals for data center operations. "
    "(6) Grid Reliability - The location must have a reliable power grid infrastructure with established utility providers capable of delivering data center-scale power requirements. "
    "(7) Geographic Positioning - The location must be recognized as one of the major US data center hubs. "
    "For each of the three metropolitan areas identified, provide: the name of the metropolitan area, verification that it meets each of the above requirements, and at least one reference URL from a reliable source supporting the suitability of this location for AI data center development. "
    "Your analysis should be based on current infrastructure capabilities and documented market data as of early 2026."
)

EARLY_2026_INSTRUCTION = (
    "All judgments must be grounded in documented data current as of early 2026 (late 2025/early 2026 acceptable). "
    "Consider the claim SUPPORTED only if the provided URL(s) explicitly support it with credible evidence (e.g., official utility/market reports, hyperscaler/colocation announcements, reputable industry analyses). "
    "If no URLs are provided for this verification, or the pages are irrelevant, conclude NOT SUPPORTED."
)

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class LocationItem(BaseModel):
    metro_area: Optional[str] = None

    # Requirement-specific sources (URLs explicitly mentioned in the answer)
    power_sources: List[str] = Field(default_factory=list)
    market_sources: List[str] = Field(default_factory=list)
    water_sources: List[str] = Field(default_factory=list)
    network_sources: List[str] = Field(default_factory=list)
    renewable_sources: List[str] = Field(default_factory=list)
    grid_sources: List[str] = Field(default_factory=list)
    hub_sources: List[str] = Field(default_factory=list)
    construction_sources: List[str] = Field(default_factory=list)
    growth_sources: List[str] = Field(default_factory=list)

    # General/reference URLs supporting overall suitability for AI data center development
    general_sources: List[str] = Field(default_factory=list)


class LocationsExtraction(BaseModel):
    locations: List[LocationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_locations() -> str:
    return (
        "Extract up to THREE distinct US metropolitan areas proposed in the answer as candidates for new large-scale AI data center facilities in 2026. "
        "For each location, return the following fields (use EXACTLY these field names):\n"
        "- metro_area: the name of the US metropolitan area.\n"
        "- general_sources: array of all general/reference URLs explicitly cited that support overall suitability for AI data center development.\n"
        "- power_sources: array of URLs explicitly supporting access to ≥100 MW available power capacity.\n"
        "- market_sources: array of URLs explicitly supporting >1,000 MW existing data center capacity in the market.\n"
        "- water_sources: array of URLs explicitly supporting ~200 million gallons/year water availability for cooling.\n"
        "- network_sources: array of URLs explicitly supporting <5 ms inter-data-center latency capability.\n"
        "- renewable_sources: array of URLs explicitly supporting documented access to renewable energy.\n"
        "- grid_sources: array of URLs explicitly supporting grid reliability and established utilities.\n"
        "- hub_sources: array of URLs explicitly supporting the location being a recognized major US data center hub.\n"
        "- construction_sources: array of URLs explicitly supporting 18–24 month construction feasibility.\n"
        "- growth_sources: array of URLs explicitly supporting capacity for future expansion.\n\n"
        "GENERAL RULES:\n"
        "1) Extract ONLY URLs explicitly present in the answer (including markdown links). Do NOT invent or infer any URLs.\n"
        "2) If a field is not mentioned for a location, set that field to null (for metro_area) or [] (for URL arrays).\n"
        "3) If the answer provides more than three locations, include only the first three in the returned JSON.\n"
        "4) Ensure that metro_area values are US metropolitan areas as named in the answer.\n"
        "Return a JSON object: {\"locations\": [ ... up to 3 LocationItem objects ... ]}."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(name: Optional[str]) -> str:
    return (name or "").strip()


def distinct_names(names: List[str]) -> Set[str]:
    return {n.lower().strip() for n in names if n and n.strip()}


def union_sources(item: LocationItem) -> List[str]:
    s: Set[str] = set()
    for arr in [
        item.general_sources,
        item.power_sources,
        item.market_sources,
        item.water_sources,
        item.network_sources,
        item.renewable_sources,
        item.grid_sources,
        item.hub_sources,
        item.construction_sources,
        item.growth_sources,
    ]:
        for u in arr:
            if isinstance(u, str) and u.strip():
                s.add(u.strip())
    return list(s)


def pick_sources(item: LocationItem, key: str) -> List[str]:
    arr = getattr(item, f"{key}_sources", [])
    if arr:
        return arr
    if item.general_sources:
        return item.general_sources
    return union_sources(item)


def additional_instruction_for(req_desc: str) -> str:
    return (
        f"{EARLY_2026_INSTRUCTION}\n"
        f"Specific requirement: {req_desc}\n"
        "Allow reasonable phrasing variations (e.g., 'about', '~', 'greater than'), but the source must clearly support the threshold or property."
    )


# --------------------------------------------------------------------------- #
# Per-location verification logic                                             #
# --------------------------------------------------------------------------- #
async def verify_location(
    evaluator: Evaluator,
    parent_node,
    item: LocationItem,
    idx: int,
) -> None:
    """
    Build verification sub-tree for a single location.
    """
    loc_id = f"Location_{idx + 1}"
    loc_name = normalize_name(item.metro_area)

    # Create the location node (non-critical to allow partial credit across locations)
    loc_node = evaluator.add_parallel(
        id=loc_id,
        desc=f"{['First','Second','Third'][idx]} metropolitan area candidate requirements check",
        parent=parent_node,
        critical=False,
    )

    # Name provided (critical existence check)
    name_exists = bool(loc_name)
    evaluator.add_custom_node(
        result=name_exists,
        id=f"L{idx + 1}_Metro_Area_Name",
        desc="Provides the name of the metropolitan area.",
        parent=loc_node,
        critical=True,
    )

    # Reference URL provided (critical existence check: at least one URL anywhere)
    ref_urls = item.general_sources if item.general_sources else union_sources(item)
    evaluator.add_custom_node(
        result=(len(ref_urls) > 0),
        id=f"L{idx + 1}_Reference_URL",
        desc="Provides at least one reference URL from a reliable source supporting suitability for AI data center development.",
        parent=loc_node,
        critical=True,
    )

    # Power Infrastructure (split into two critical leaves under a critical parallel aggregator)
    power_parent = evaluator.add_parallel(
        id=f"L{idx + 1}_Power_Infrastructure",
        desc="Meets power infrastructure requirement: access to ≥100 MW available power capacity AND infrastructure capable of supporting AI-optimized rack densities of 60+ kW per rack.",
        parent=loc_node,
        critical=True,
    )

    # 100 MW available capacity
    power_cap_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Power_Available_100MW",
        desc="Access to ≥100 MW available power capacity.",
        parent=power_parent,
        critical=True,
    )
    claim_power_cap = (
        f"As of early 2026, the {loc_name} metropolitan area has access to at least 100 MW of available power "
        f"capacity for new data center loads."
    )
    await evaluator.verify(
        claim=claim_power_cap,
        node=power_cap_leaf,
        sources=pick_sources(item, "power"),
        additional_instruction=additional_instruction_for(
            "Access to ≥100 MW available power capacity (new data center loads)."
        ),
    )

    # 60+ kW per rack density capability
    rack_density_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Rack_Density_60kW",
        desc="Infrastructure supports AI-optimized rack densities of 60+ kW per rack.",
        parent=power_parent,
        critical=True,
    )
    claim_rack = (
        f"As of early 2026, data center infrastructure in {loc_name} supports high-density deployments "
        f"with rack power of at least 60 kW per rack (e.g., appropriate cooling and power distribution)."
    )
    await evaluator.verify(
        claim=claim_rack,
        node=rack_density_leaf,
        sources=pick_sources(item, "power"),
        additional_instruction=additional_instruction_for(
            "Infrastructure supports AI-optimized rack densities of ≥60 kW per rack."
        ),
    )

    # Established Market Presence (>1,000 MW existing capacity)
    market_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Established_Market_Presence",
        desc="Is part of an established data center market with >1,000 MW of existing data center capacity.",
        parent=loc_node,
        critical=True,
    )
    claim_market = (
        f"As of early 2026, {loc_name} is part of an established US data center market "
        f"with more than 1,000 MW of existing data center capacity."
    )
    await evaluator.verify(
        claim=claim_market,
        node=market_leaf,
        sources=pick_sources(item, "market"),
        additional_instruction=additional_instruction_for(
            "Established data center market with >1,000 MW existing capacity."
        ),
    )

    # Water Resources (~200 million gallons annually)
    water_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Water_Resources",
        desc="Has water availability capable of supporting approximately 200 million gallons annually for data center cooling operations.",
        parent=loc_node,
        critical=True,
    )
    claim_water = (
        f"As of early 2026, the {loc_name} area has water availability (e.g., municipal/reclaimed/industrial) "
        f"capable of supporting around 200 million gallons per year for data center cooling operations."
    )
    await evaluator.verify(
        claim=claim_water,
        node=water_leaf,
        sources=pick_sources(item, "water"),
        additional_instruction=additional_instruction_for(
            "Water availability of ~200 million gallons/year for cooling, via credible utility/program sources."
        ),
    )

    # Network Infrastructure (<5 ms inter-DC latency)
    network_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Network_Infrastructure",
        desc="Has network infrastructure capable of supporting <5 milliseconds latency for inter-data-center communications.",
        parent=loc_node,
        critical=True,
    )
    claim_net = (
        f"As of early 2026, {loc_name} has fiber/network infrastructure enabling less than 5 milliseconds "
        f"latency for inter-data-center communications (e.g., metro/region round-trip latency figures)."
    )
    await evaluator.verify(
        claim=claim_net,
        node=network_leaf,
        sources=pick_sources(item, "network"),
        additional_instruction=additional_instruction_for(
            "<5 ms inter-data-center latency capability; accept credible vendor/network operator metrics."
        ),
    )

    # Renewable Energy Access (documented)
    renewable_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Renewable_Energy_Access",
        desc="Has documented access to renewable energy sources for data center operations.",
        parent=loc_node,
        critical=True,
    )
    claim_renew = (
        f"As of early 2026, {loc_name} has documented access to renewable energy sources "
        f"(e.g., PPAs, renewables programs, RPS-backed utility offerings) for data center operations."
    )
    await evaluator.verify(
        claim=claim_renew,
        node=renewable_leaf,
        sources=pick_sources(item, "renewable"),
        additional_instruction=additional_instruction_for(
            "Documented access to renewable energy (utility programs, PPAs, market offerings)."
        ),
    )

    # Grid Reliability & Utilities (established utility providers)
    grid_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Grid_Reliability_and_Utilities",
        desc="Has a reliable power grid infrastructure with established utility providers capable of delivering data center-scale power requirements.",
        parent=loc_node,
        critical=True,
    )
    claim_grid = (
        f"As of early 2026, {loc_name} has a reliable power grid with established utility providers "
        f"capable of delivering data center-scale power requirements."
    )
    await evaluator.verify(
        claim=claim_grid,
        node=grid_leaf,
        sources=pick_sources(item, "grid"),
        additional_instruction=additional_instruction_for(
            "Reliable grid and established utilities delivering data center-scale power."
        ),
    )

    # Geographic Positioning (recognized major US data center hub)
    hub_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Geographic_Positioning",
        desc="Is recognized as one of the major US data center hubs.",
        parent=loc_node,
        critical=True,
    )
    claim_hub = (
        f"As of early 2026, {loc_name} is recognized as a major US data center hub by reputable industry sources."
    )
    await evaluator.verify(
        claim=claim_hub,
        node=hub_leaf,
        sources=pick_sources(item, "hub"),
        additional_instruction=additional_instruction_for(
            "Recognition as major US data center hub (industry reports/analyses)."
        ),
    )

    # Construction Feasibility (18–24 month timeline)
    construction_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Construction_Feasibility",
        desc="Location supports an 18–24 month construction timeline for a large-scale data center facility.",
        parent=loc_node,
        critical=True,
    )
    claim_construction = (
        f"As of early 2026, data center development in {loc_name} can be executed on an approximately 18–24 month "
        f"construction timeline (e.g., shovel-ready sites, streamlined permitting/zoning, established contractors)."
    )
    await evaluator.verify(
        claim=claim_construction,
        node=construction_leaf,
        sources=pick_sources(item, "construction"),
        additional_instruction=additional_instruction_for(
            "Feasibility of 18–24 month DC construction timeline; accept credible local/industry evidence."
        ),
    )

    # Growth Capacity (future expansion)
    growth_leaf = evaluator.add_leaf(
        id=f"L{idx + 1}_Growth_Capacity",
        desc="Location has capacity for future expansion (e.g., land/power/network scalability as documented).",
        parent=loc_node,
        critical=True,
    )
    claim_growth = (
        f"As of early 2026, {loc_name} has capacity for future data center expansion (e.g., "
        f"available land, scalable power/network plans, documented expansion pipelines)."
    )
    await evaluator.verify(
        claim=claim_growth,
        node=growth_leaf,
        sources=pick_sources(item, "growth"),
        additional_instruction=additional_instruction_for(
            "Documented capacity for future expansion: land, power, network scalability."
        ),
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
    Evaluate an answer for the AI data center location task (2026).
    """
    # Initialize evaluator with parallel aggregation at root
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

    # Extract locations and URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_locations(),
        template_class=LocationsExtraction,
        extraction_name="locations_extraction",
    )

    # Filter to exactly three locations (pad with empty items if fewer)
    locations: List[LocationItem] = list(extracted.locations[:3])
    while len(locations) < 3:
        locations.append(LocationItem())

    # Global critical check: provides three distinct US metro areas
    names = [normalize_name(loc.metro_area) for loc in locations]
    distinct = distinct_names(names)
    evaluator.add_custom_node(
        result=(len(distinct) >= 3),
        id="Three_Distinct_Metro_Areas_Provided",
        desc="Provides three (and not fewer) distinct US metropolitan area candidates.",
        parent=root,
        critical=True,
    )

    # Global critical check: uses documented data as of early 2026 (answer-level, simple verification)
    date_leaf = evaluator.add_leaf(
        id="Uses_Documented_Data_As_Of_Early_2026",
        desc="Analysis is based on current infrastructure capabilities and documented market data as of early 2026 (i.e., claims are framed/dated accordingly, not purely speculative).",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The analysis frames its claims using documented market/infrastructure data current as of early 2026 "
            "(explicit cues such as '2026', 'early 2026', or late-2025/2026 data references are present; "
            "claims are not purely speculative)."
        ),
        node=date_leaf,
        additional_instruction=(
            "Examine the provided answer text. If it references 2026 (or early 2026), or late-2025/2026 data, "
            "and clearly grounds claims in documented market/infrastructure facts, mark as CORRECT; "
            "otherwise mark as INCORRECT."
        ),
    )

    # Build and verify each location subtree
    for i, item in enumerate(locations):
        await verify_location(evaluator, root, item, i)

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "evaluated_locations": names,
            "distinct_count": len(distinct),
            "note": "Only the first three locations in the answer were evaluated; source-grounding required for factual checks."
        },
        info_type="evaluation_meta",
        info_name="meta"
    )

    # Return structured summary
    return evaluator.get_summary()