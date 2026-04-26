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
TASK_ID = "regional_grocery_chain_multicriteria_2026"
TASK_DESCRIPTION = """
Identify a major regional grocery store chain that operates stores in both North Carolina and Virginia, maintains at least 250 store locations across a minimum of 7 U.S. states, and operates its own fuel centers or gas stations (not just fuel rewards partnerships with third-party providers). Provide the chain's name along with documentation for: (1) its geographic presence across at least 7 states including North Carolina and Virginia, (2) its total number of store locations as of 2025-2026, (3) its fuel center or gas station operations, (4) its pharmacy services, (5) its online grocery ordering capabilities (delivery or pickup), (6) its customer loyalty or rewards program, and (7) its ownership structure.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChainExtraction(BaseModel):
    # Core identification
    chain_name: Optional[str] = None
    general_urls: List[str] = Field(default_factory=list)

    # Geographic coverage
    states_listed: List[str] = Field(default_factory=list)
    geography_urls: List[str] = Field(default_factory=list)

    # Store count
    store_count_text: Optional[str] = None  # Keep as free text to allow "over 250", "~300", etc.
    store_count_urls: List[str] = Field(default_factory=list)

    # Fuel centers / gas stations
    fuel_ops_desc: Optional[str] = None
    fuel_urls: List[str] = Field(default_factory=list)

    # Pharmacy
    pharmacy_desc: Optional[str] = None
    pharmacy_urls: List[str] = Field(default_factory=list)

    # Online ordering
    online_services_desc: Optional[str] = None
    online_urls: List[str] = Field(default_factory=list)

    # Loyalty / rewards
    loyalty_program_name: Optional[str] = None
    loyalty_urls: List[str] = Field(default_factory=list)

    # Ownership
    ownership_desc: Optional[str] = None
    ownership_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chain_info() -> str:
    return """
    Extract the structured information about the single regional grocery store chain that the answer identifies and evaluates.

    Return a JSON object with the following fields:
    - chain_name: The name of the grocery chain identified in the answer (string).
    - general_urls: URLs cited that generally describe the chain (e.g., official About page, corporate fact sheet, Wikipedia).
    - states_listed: A list of U.S. states explicitly claimed in the answer where the chain operates stores. Use full state names (e.g., "North Carolina", "Virginia", "Georgia", etc.). If none are listed, return an empty list.
    - geography_urls: URLs cited that document the chain's geographic footprint or store locator showing states of operation.
    - store_count_text: The stated total store count as mentioned in the answer (free text; examples: "over 250", "about 300", "2,700+"), ideally for 2025–2026.
    - store_count_urls: URLs cited that document the store count (prefer official/corporate fact pages, 10-Ks, press releases from 2025–2026).
    - fuel_ops_desc: A short description from the answer about the chain's own fuel centers or gas stations (not just rewards partnerships). If unspecified, null.
    - fuel_urls: URLs cited that document the chain's fuel center/gas station operations (e.g., "Fuel Center", "Gas Station" pages).
    - pharmacy_desc: Short description of pharmacy services, if provided.
    - pharmacy_urls: URLs cited that document pharmacy services (official pharmacy page, service pages).
    - online_services_desc: Short description of online grocery ordering (delivery and/or pickup) from the answer.
    - online_urls: URLs cited that document the online ordering services (official ordering page, delivery/pickup info, or recognized partner).
    - loyalty_program_name: The name of the customer loyalty or rewards program if provided (e.g., "Kroger Plus", "MVP", "Club Card"). If not stated, null.
    - loyalty_urls: URLs cited that document the loyalty/rewards program (official program page).
    - ownership_desc: The ownership structure statement as provided in the answer (e.g., "publicly traded", "subsidiary of X", "employee-owned"). If not stated, null.
    - ownership_urls: URLs cited that document ownership structure (e.g., investor relations, corporate overview, Wikipedia with citations).

    IMPORTANT:
    - Extract ONLY what is explicitly mentioned and cited in the answer.
    - For all URL arrays, include only valid, complete URLs that are actually present in the answer text (including markdown links).
    - Do not invent URLs. If the answer does not provide any URLs for a category, return an empty list for that category.
    - Normalize U.S. state names to their full names (e.g., use "North Carolina" not "NC").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*url_lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists while preserving order and removing duplicates and empties."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                s = u.strip()
                if s and s not in seen:
                    seen.add(s)
                    merged.append(s)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_chain_identification(evaluator: Evaluator, parent_node, data: ChainExtraction):
    """
    Chain Identification (critical): ensure a concrete chain is named and is a grocery store chain.
    """
    node = evaluator.add_sequential(
        id="Chain_Identification",
        desc="Correctly identify a regional grocery store chain that meets all the specified criteria",
        parent=parent_node,
        critical=True,
    )

    # 1) Chain name provided (existence)
    name_exists = evaluator.add_custom_node(
        result=bool(data.chain_name and data.chain_name.strip()),
        id="chain_name_provided",
        desc="Chain name is provided in the answer",
        parent=node,
        critical=True
    )

    # 2) Chain is a grocery store chain (web-supported)
    id_leaf = evaluator.add_leaf(
        id="chain_is_grocery_store_chain",
        desc="The identified company is a grocery/supermarket retail chain in the United States",
        parent=node,
        critical=True
    )
    id_sources = _merge_sources(data.general_urls, data.geography_urls, data.store_count_urls)
    chain_name = data.chain_name or ""
    id_claim = f"{chain_name} is a grocery (supermarket) retail chain operating in the United States."
    await evaluator.verify(
        claim=id_claim,
        node=id_leaf,
        sources=id_sources,
        additional_instruction="Accept synonyms like 'supermarket chain' or 'grocery retailer'. Prefer official sites, investor pages, or well-cited profiles."
    )


async def build_geographic_coverage(evaluator: Evaluator, parent_node, data: ChainExtraction):
    """
    Geographic Coverage (critical, sequential): reference URL, at least 7 states, NC presence, VA presence.
    """
    node = evaluator.add_sequential(
        id="Geographic_Coverage",
        desc="Verify the chain's geographic presence meets all location requirements",
        parent=parent_node,
        critical=True,
    )

    # 0) Reference URL existence first to gate subsequent checks
    geo_ref = evaluator.add_custom_node(
        result=bool(data.geography_urls and len(data.geography_urls) > 0),
        id="geo_reference_url",
        desc="Provide a URL reference documenting the chain's geographic presence",
        parent=node,
        critical=True
    )

    # 1) Minimum seven states
    min_states_leaf = evaluator.add_leaf(
        id="Minimum_Seven_States",
        desc="Confirm the chain operates stores in at least 7 U.S. states",
        parent=node,
        critical=True
    )
    min_states_sources = _merge_sources(data.geography_urls)
    chain_name = data.chain_name or "The chain"
    min_states_claim = f"{chain_name} operates stores in at least seven (7) U.S. states."
    await evaluator.verify(
        claim=min_states_claim,
        node=min_states_leaf,
        sources=min_states_sources,
        additional_instruction="Prefer evidence that enumerates states or explicitly states 'X states'. Accept 'over 7 states' or 'at least 7 states'. Store locator pages listing states count also qualify."
    )

    # 2) North Carolina presence
    nc_leaf = evaluator.add_leaf(
        id="North_Carolina_Presence",
        desc="Confirm the chain operates stores in North Carolina",
        parent=node,
        critical=True
    )
    nc_claim = f"{chain_name} operates stores in North Carolina."
    await evaluator.verify(
        claim=nc_claim,
        node=nc_leaf,
        sources=min_states_sources,
        additional_instruction="Look for 'North Carolina' presence on store locator/state lists or corporate fact pages."
    )

    # 3) Virginia presence
    va_leaf = evaluator.add_leaf(
        id="Virginia_Presence",
        desc="Confirm the chain operates stores in Virginia",
        parent=node,
        critical=True
    )
    va_claim = f"{chain_name} operates stores in Virginia."
    await evaluator.verify(
        claim=va_claim,
        node=va_leaf,
        sources=min_states_sources,
        additional_instruction="Look for 'Virginia' presence on store locator/state lists or corporate fact pages."
    )


async def build_store_count(evaluator: Evaluator, parent_node, data: ChainExtraction):
    """
    Store Count (critical, sequential): reference URL first, then >= 250 as of 2025–2026.
    """
    node = evaluator.add_sequential(
        id="Store_Count",
        desc="Verify the chain operates at least 250 store locations",
        parent=parent_node,
        critical=True,
    )

    # 0) Reference URL existence
    count_ref = evaluator.add_custom_node(
        result=bool(data.store_count_urls and len(data.store_count_urls) > 0),
        id="store_count_reference_url",
        desc="Provide a URL reference documenting the store count",
        parent=node,
        critical=True
    )

    # 1) Minimum 250 locations (as of 2025–2026)
    min_count_leaf = evaluator.add_leaf(
        id="Minimum_250_Locations",
        desc="Confirm the chain has at least 250 store locations as of 2025-2026",
        parent=node,
        critical=True
    )
    count_sources = _merge_sources(data.store_count_urls)
    chain_name = data.chain_name or "The chain"
    stated = (data.store_count_text or "").strip()
    # Include the stated text to help the verifier locate phrasing like "over 250"
    min_count_claim = (
        f"As of 2025 or 2026, {chain_name} operates at least 250 store locations."
        + (f" The answer cites: '{stated}'." if stated else "")
    )
    await evaluator.verify(
        claim=min_count_claim,
        node=min_count_leaf,
        sources=count_sources,
        additional_instruction="Prefer sources dated 2025–2026 (e.g., corporate fact sheet, investor facts, press releases). Accept phrasings like 'over 250', '250+', 'approximately 300'. If only older counts are present, judge as not supported."
    )


async def build_fuel_centers(evaluator: Evaluator, parent_node, data: ChainExtraction):
    """
    Fuel Centers (critical, sequential): reference URL first, then own/operate fuel centers (not just rewards).
    """
    node = evaluator.add_sequential(
        id="Fuel_Centers",
        desc="Verify the chain operates its own fuel centers or gas stations",
        parent=parent_node,
        critical=True,
    )

    # 0) Reference URL existence
    fuel_ref = evaluator.add_custom_node(
        result=bool(data.fuel_urls and len(data.fuel_urls) > 0),
        id="fuel_reference_url",
        desc="Provide a URL reference documenting the chain's fuel center operations",
        parent=node,
        critical=True
    )

    # 1) Owns/operates fuel centers (not just rewards)
    fuel_leaf = evaluator.add_leaf(
        id="Owns_Fuel_Operations",
        desc="Confirm the chain operates its own fuel centers or gas stations, not merely fuel rewards partnerships",
        parent=node,
        critical=True
    )
    fuel_sources = _merge_sources(data.fuel_urls)
    chain_name = data.chain_name or "The chain"
    fuel_desc = (data.fuel_ops_desc or "").strip()
    fuel_claim = (
        f"{chain_name} owns and operates its own branded fuel centers or gas stations (not just a third‑party fuel rewards partnership)."
        + (f" The answer describes: '{fuel_desc}'." if fuel_desc else "")
    )
    await evaluator.verify(
        claim=fuel_claim,
        node=fuel_leaf,
        sources=fuel_sources,
        additional_instruction="Evidence should mention the chain's own 'Fuel Center(s)' or 'gas station(s)' it operates. Pages only about fuel rewards with third parties are insufficient."
    )


async def build_pharmacy(evaluator: Evaluator, parent_node, data: ChainExtraction):
    """
    Pharmacy Services (critical, sequential): reference URL first, then pharmacy presence.
    """
    node = evaluator.add_sequential(
        id="Pharmacy_Services",
        desc="Verify the chain provides pharmacy services at store locations",
        parent=parent_node,
        critical=True,
    )

    # 0) Reference URL existence
    pharm_ref = evaluator.add_custom_node(
        result=bool(data.pharmacy_urls and len(data.pharmacy_urls) > 0),
        id="pharmacy_reference_url",
        desc="Provide a URL reference documenting pharmacy services",
        parent=node,
        critical=True
    )

    # 1) Offers pharmacy
    pharm_leaf = evaluator.add_leaf(
        id="Offers_Pharmacy",
        desc="Confirm the chain has pharmacy services available at store locations",
        parent=node,
        critical=True
    )
    pharm_sources = _merge_sources(data.pharmacy_urls)
    chain_name = data.chain_name or "The chain"
    pharm_desc = (data.pharmacy_desc or "").strip()
    pharm_claim = (
        f"{chain_name} provides pharmacy services at its store locations."
        + (f" The answer notes: '{pharm_desc}'." if pharm_desc else "")
    )
    await evaluator.verify(
        claim=pharm_claim,
        node=pharm_leaf,
        sources=pharm_sources,
        additional_instruction="Look for official pharmacy pages, service descriptions, or store service listings that clearly indicate in-store pharmacies."
    )


async def build_online_ordering(evaluator: Evaluator, parent_node, data: ChainExtraction):
    """
    Online Ordering (critical, sequential): reference URL first, then delivery/pickup capability.
    """
    node = evaluator.add_sequential(
        id="Online_Ordering",
        desc="Verify the chain provides online grocery ordering with delivery or pickup",
        parent=parent_node,
        critical=True,
    )

    # 0) Reference URL existence
    online_ref = evaluator.add_custom_node(
        result=bool(data.online_urls and len(data.online_urls) > 0),
        id="online_reference_url",
        desc="Provide a URL reference documenting online ordering services",
        parent=node,
        critical=True
    )

    # 1) Offers online ordering with delivery or pickup
    online_leaf = evaluator.add_leaf(
        id="Offers_Online_Services",
        desc="Confirm the chain provides online grocery ordering with either delivery or curbside pickup options",
        parent=node,
        critical=True
    )
    online_sources = _merge_sources(data.online_urls)
    chain_name = data.chain_name or "The chain"
    online_desc = (data.online_services_desc or "").strip()
    online_claim = (
        f"{chain_name} offers online grocery ordering with delivery and/or curbside pickup."
        + (f" The answer mentions: '{online_desc}'." if online_desc else "")
    )
    await evaluator.verify(
        claim=online_claim,
        node=online_leaf,
        sources=online_sources,
        additional_instruction="Evidence can include official online ordering pages, 'Delivery' or 'Pickup' info, or recognized partners (e.g., Instacart) explicitly supporting this chain."
    )


async def build_loyalty_program(evaluator: Evaluator, parent_node, data: ChainExtraction):
    """
    Loyalty Program (critical, sequential): reference URL first, then program existence (optionally name).
    """
    node = evaluator.add_sequential(
        id="Loyalty_Program",
        desc="Verify the chain has a customer loyalty or rewards program",
        parent=parent_node,
        critical=True,
    )

    # 0) Reference URL existence
    loyalty_ref = evaluator.add_custom_node(
        result=bool(data.loyalty_urls and len(data.loyalty_urls) > 0),
        id="loyalty_reference_url",
        desc="Provide a URL reference documenting the loyalty program",
        parent=node,
        critical=True
    )

    # 1) Has program
    loyalty_leaf = evaluator.add_leaf(
        id="Has_Program",
        desc="Confirm the chain offers a customer loyalty or rewards program",
        parent=node,
        critical=True
    )
    loyalty_sources = _merge_sources(data.loyalty_urls)
    chain_name = data.chain_name or "The chain"
    program_name = (data.loyalty_program_name or "").strip()
    if program_name:
        loyalty_claim = f"{chain_name} offers a customer loyalty or rewards program named '{program_name}'."
    else:
        loyalty_claim = f"{chain_name} offers a customer loyalty or rewards program."
    await evaluator.verify(
        claim=loyalty_claim,
        node=loyalty_leaf,
        sources=loyalty_sources,
        additional_instruction="Look for official loyalty/rewards program pages or detailed descriptions."
    )


async def build_ownership(evaluator: Evaluator, parent_node, data: ChainExtraction):
    """
    Ownership Structure (critical, sequential): reference URL first, then documented ownership statement.
    """
    node = evaluator.add_sequential(
        id="Ownership_Structure",
        desc="Verify and document the chain's ownership structure",
        parent=parent_node,
        critical=True,
    )

    # 0) Reference URL existence
    own_ref = evaluator.add_custom_node(
        result=bool(data.ownership_urls and len(data.ownership_urls) > 0),
        id="ownership_reference_url",
        desc="Provide a URL reference documenting the ownership structure",
        parent=node,
        critical=True
    )

    # 1) Documented ownership
    own_leaf = evaluator.add_leaf(
        id="Documented_Ownership",
        desc="Provide clear documentation of the chain's ownership structure (e.g., employee-owned, family-owned, or subsidiary)",
        parent=node,
        critical=True
    )
    own_sources = _merge_sources(data.ownership_urls)
    chain_name = data.chain_name or "The chain"
    ownership_desc = (data.ownership_desc or "").strip()
    if ownership_desc:
        own_claim = f"The ownership structure of {chain_name} is accurately described as: {ownership_desc}."
    else:
        own_claim = f"The ownership structure of {chain_name} is clearly documented by the provided sources."
    await evaluator.verify(
        claim=own_claim,
        node=own_leaf,
        sources=own_sources,
        additional_instruction="Accept well-supported statements like 'publicly traded', 'employee-owned', 'subsidiary of X', etc. Prefer official investor or corporate sources; Wikipedia acceptable if well-cited."
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
    Evaluate an answer for the regional grocery chain multi-criteria task.
    """
    # Initialize evaluator (root is a non-critical container; we create a critical task root under it)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Top-level process is sequential
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

    # Create a critical task root to reflect rubric's critical root requirement
    task_root = evaluator.add_sequential(
        id="Task_Root",
        desc="Complete task by identifying a qualifying grocery chain and providing all required information",
        parent=root,
        critical=True
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_chain_info(),
        template_class=ChainExtraction,
        extraction_name="chain_info",
    )

    # Build verification subtrees (all critical, sequential as per rubric)
    await build_chain_identification(evaluator, task_root, extracted)
    await build_geographic_coverage(evaluator, task_root, extracted)
    await build_store_count(evaluator, task_root, extracted)
    await build_fuel_centers(evaluator, task_root, extracted)
    await build_pharmacy(evaluator, task_root, extracted)
    await build_online_ordering(evaluator, task_root, extracted)
    await build_loyalty_program(evaluator, task_root, extracted)
    await build_ownership(evaluator, task_root, extracted)

    # Return standardized summary
    return evaluator.get_summary()