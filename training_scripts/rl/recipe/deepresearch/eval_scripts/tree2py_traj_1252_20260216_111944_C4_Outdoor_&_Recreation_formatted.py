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
TASK_ID = "bna_budget_outdoor_2026"
TASK_DESCRIPTION = """
A travel planning company is creating a guide for budget-conscious outdoor enthusiasts departing from Nashville International Airport (BNA) in 2026. They need comprehensive information about low-cost airline options for reaching outdoor recreation destinations via nonstop flights. Please provide the following information: (1) Identify at least one budget airline that operates nonstop flights from BNA to a city providing access to Rocky Mountain National Park or Colorado outdoor recreation areas, (2) Identify at least one budget airline that operates nonstop flights from BNA to Mexican destinations known for outdoor recreation or beaches, (3) Specify the exact month and year when BNA's first outdoor terrace opened in the Concourse D extension, (4) Identify at least one budget airline that specifically markets national park access as part of its route network from BNA, (5) Identify at least one budget airline that operates nonstop flights from BNA to Pacific Northwest cities with outdoor recreation access, (6) Identify at least one budget airline that operates nonstop flights from BNA to U.S. beach or coastal destinations, (7) Specify which concourse at BNA houses the outdoor terrace amenity. For each airline identified, provide the airline name and at least one specific destination city it serves from BNA. All routes must be nonstop flights.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RouteItem(BaseModel):
    """Represents a single airline route item with evidence."""
    airline: Optional[str] = None
    destination_city: Optional[str] = None
    # URLs that support the nonstop route claim (e.g., airline route page, news, schedule)
    route_urls: List[str] = Field(default_factory=list)
    # URLs that support the category qualification (e.g., Rocky Mtn access, Mexico/PNW/beach classification)
    category_urls: List[str] = Field(default_factory=list)


class NationalParkMarketingItem(BaseModel):
    """Represents a budget airline marketing national park access from BNA."""
    airline: Optional[str] = None
    destination_city: Optional[str] = None
    marketed_park: Optional[str] = None
    # URLs showing marketing that mentions national parks and (ideally) Nashville/BNA
    marketing_urls: List[str] = Field(default_factory=list)
    # URLs proving the airline operates the nonstop route from BNA to that destination
    route_urls: List[str] = Field(default_factory=list)


class TerraceOpening(BaseModel):
    """Opening date for BNA's first outdoor terrace in Concourse D extension."""
    month: Optional[str] = None
    year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TerraceLocation(BaseModel):
    """Concourse location of BNA's outdoor terrace amenity."""
    concourse: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class BNAOutdoorExtraction(BaseModel):
    """Top-level extraction model for all required categories."""
    rocky: Optional[RouteItem] = None
    mexico: Optional[RouteItem] = None
    pnw: Optional[RouteItem] = None
    beach: Optional[RouteItem] = None
    park_marketing: Optional[NationalParkMarketingItem] = None
    terrace_opening: Optional[TerraceOpening] = None
    terrace_location: Optional[TerraceLocation] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bna_outdoor() -> str:
    return """
Extract the following structured information exactly as presented in the answer text. Do not invent any values. If an item is missing, return null for that object or an empty array for URL lists.

You must extract:

1) rocky: The first example (if multiple are provided) of a budget airline that operates nonstop flights from BNA to a city providing access to Rocky Mountain National Park or Colorado outdoor recreation areas.
   Fields:
   - airline: airline name (e.g., Frontier)
   - destination_city: destination city name as stated (e.g., Denver)
   - route_urls: array of URLs cited that specifically support the nonstop route from BNA to this destination
   - category_urls: array of URLs cited that support that the destination provides access to Rocky Mountain National Park or Colorado outdoor recreation

2) mexico: The first example of a budget airline that operates nonstop flights from BNA to a Mexican destination known for outdoor recreation or beaches.
   Fields (same as rocky):
   - airline
   - destination_city
   - route_urls
   - category_urls

3) pnw: The first example of a budget airline that operates nonstop flights from BNA to a Pacific Northwest (PNW) city with outdoor recreation access.
   Fields (same as rocky):
   - airline
   - destination_city
   - route_urls
   - category_urls

4) beach: The first example of a budget airline that operates nonstop flights from BNA to a U.S. beach/coastal destination.
   Fields (same as rocky):
   - airline
   - destination_city
   - route_urls
   - category_urls

5) park_marketing: The first example of a budget airline that specifically markets or promotes national park access as part of its route network from BNA.
   Fields:
   - airline
   - destination_city: at least one destination city the airline serves nonstop from BNA that is relevant to the national park positioning
   - marketed_park: the national park name if mentioned
   - marketing_urls: array of URLs cited that show the airline marketing national park access (ideally mentioning BNA/Nashville)
   - route_urls: array of URLs cited that support the nonstop route from BNA to the destination_city

6) terrace_opening: The exact opening month and year of BNA's first outdoor terrace in the Concourse D extension.
   Fields:
   - month: month name or abbreviation as written (e.g., April)
   - year: four-digit year (e.g., 2023)
   - urls: array of URLs cited that support this date

7) terrace_location: The specific concourse at BNA that houses the outdoor terrace amenity.
   Fields:
   - concourse: e.g., Concourse D
   - urls: array of URLs cited that support this location

IMPORTANT NOTES:
- All routes must be nonstop flights. Only extract entries that explicitly describe nonstop or direct service.
- Only use URLs that are explicitly present in the answer text (including markdown links).
- Limit to the first example for each category if the answer lists multiple.
- If the answer does not cite any URLs for a given item, return an empty array for URLs.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_route_category(
    evaluator: Evaluator,
    parent_node,
    category_id: str,
    category_desc: str,
    item: Optional[RouteItem],
    region_kind: str
) -> None:
    """
    Generic verifier for a route-based category.
    region_kind: one of {'rocky', 'mexico', 'pnw', 'beach'} to customize the category verification claim.
    """
    node = evaluator.add_parallel(
        id=category_id,
        desc=category_desc,
        parent=parent_node,
        critical=False
    )

    # Existence (critical) – require airline, destination, and at least one route URL
    exists = (
        item is not None and
        item.airline is not None and item.airline.strip() != "" and
        item.destination_city is not None and item.destination_city.strip() != "" and
        isinstance(item.route_urls, list) and len(item.route_urls) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id=f"{category_id}_exists",
        desc="Required fields provided (airline, destination city, and at least one route source URL)",
        parent=node,
        critical=True
    )

    airline = item.airline if item and item.airline else ""
    city = item.destination_city if item and item.destination_city else ""
    route_urls = item.route_urls if item else []
    cat_urls = item.category_urls if item else []
    cat_sources = cat_urls if cat_urls else route_urls

    # Leaf: Nonstop route verification (critical)
    nonstop_leaf = evaluator.add_leaf(
        id=f"{category_id}_nonstop_route",
        desc=f"Nonstop route supported: {airline} operates nonstop BNA → {city}",
        parent=node,
        critical=True
    )
    nonstop_claim = f"{airline} operates nonstop flights from Nashville (BNA) to {city}."
    await evaluator.verify(
        claim=nonstop_claim,
        node=nonstop_leaf,
        sources=route_urls,
        additional_instruction=(
            "Verify the page(s) explicitly indicate a nonstop/direct flight between Nashville (BNA) and the given city. "
            "Accept 'nonstop' or 'direct' wording. If the route is not clearly nonstop, mark as not supported."
        )
    )

    # Leaf: Category qualification (critical)
    cat_leaf = evaluator.add_leaf(
        id=f"{category_id}_category_support",
        desc="Destination qualifies for the specified outdoor category",
        parent=node,
        critical=True
    )

    if region_kind == "rocky":
        cat_claim = (
            f"{city} is in Colorado or is a city that provides access to Rocky Mountain National Park or Colorado outdoor recreation areas."
        )
        add_ins = (
            "Support can come from tourism pages, park pages, or authoritative descriptions. "
            "If the evidence indicates the city is in Colorado or is a gateway to Rocky Mountain National Park, mark supported."
        )
    elif region_kind == "mexico":
        cat_claim = f"{city} is in Mexico and is known for outdoor recreation or beaches."
        add_ins = (
            "Look for sources indicating the city is in Mexico and is recognized for outdoor recreation or beach tourism. "
            "If the route source itself shows the country as Mexico, that can suffice."
        )
    elif region_kind == "pnw":
        cat_claim = (
            f"{city} is a Pacific Northwest city (e.g., in Washington or Oregon) and provides access to outdoor recreation."
        )
        add_ins = (
            "Accept evidence that the city is in Washington or Oregon and is associated with outdoor recreation access. "
            "If sources clearly identify the city as PNW or show it is in WA/OR with outdoor access, mark supported."
        )
    elif region_kind == "beach":
        cat_claim = f"{city} is a U.S. beach or coastal destination."
        add_ins = (
            "Look for sources indicating a coastal or beach location in the United States. "
            "Tourism bureau pages or route pages referencing a beach/coast city name suffice."
        )
    else:
        cat_claim = f"{city} fits the required outdoor category."
        add_ins = "Use the provided sources to confirm the category."

    await evaluator.verify(
        claim=cat_claim,
        node=cat_leaf,
        sources=cat_sources,
        additional_instruction=add_ins
    )


async def verify_park_marketing(
    evaluator: Evaluator,
    parent_node,
    item: Optional[NationalParkMarketingItem]
) -> None:
    """
    Verify the airline specifically markets national park access as part of its route network from BNA,
    and verify at least one relevant nonstop route.
    """
    node = evaluator.add_parallel(
        id="National_Park_Focused_Airline",
        desc="Identifies a budget airline that markets national park access from BNA and provides a relevant nonstop route",
        parent=parent_node,
        critical=False
    )

    exists = (
        item is not None and
        item.airline is not None and item.airline.strip() != "" and
        isinstance(item.marketing_urls, list) and len(item.marketing_urls) > 0 and
        item.destination_city is not None and item.destination_city.strip() != "" and
        isinstance(item.route_urls, list) and len(item.route_urls) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="np_marketing_exists",
        desc="Required fields provided (airline, marketing URLs, destination city, and route URLs)",
        parent=node,
        critical=True
    )

    airline = item.airline if item and item.airline else ""
    city = item.destination_city if item and item.destination_city else ""
    marketed_park = item.marketed_park if item and item.marketed_park else ""
    mkt_urls = item.marketing_urls if item else []
    rte_urls = item.route_urls if item else []

    # Leaf: Marketing support (critical)
    marketing_leaf = evaluator.add_leaf(
        id="np_marketing_supported",
        desc=f"{airline} markets national park access tied to its BNA route network",
        parent=node,
        critical=True
    )
    marketing_claim = (
        f"The provided marketing source(s) show that {airline} markets national park access in connection with routes from Nashville (BNA)."
    )
    await evaluator.verify(
        claim=marketing_claim,
        node=marketing_leaf,
        sources=mkt_urls,
        additional_instruction=(
            "Look for mentions of 'national park(s)' and Nashville or BNA on the page. "
            "If the page explicitly connects national parks to the airline's Nashville routes or mentions Nashville as a gateway to a park, mark supported."
        )
    )

    # Leaf: Nonstop route support (critical)
    route_leaf = evaluator.add_leaf(
        id="np_nonstop_route",
        desc=f"Nonstop route supported: {airline} operates nonstop BNA → {city}",
        parent=node,
        critical=True
    )
    route_claim = f"{airline} operates nonstop flights from Nashville (BNA) to {city}."
    await evaluator.verify(
        claim=route_claim,
        node=route_leaf,
        sources=rte_urls,
        additional_instruction=(
            "Verify that the page(s) indicate nonstop/direct service between Nashville (BNA) and the specified city."
        )
    )

    # Leaf: Park accessibility from destination (non-critical, optional but useful)
    if marketed_park:
        park_access_leaf = evaluator.add_leaf(
            id="np_park_access_from_city",
            desc=f"Marketed park accessibility supported: {marketed_park} is accessible from {city}",
            parent=node,
            critical=False
        )
        park_access_claim = f"{marketed_park} is accessible from {city}."
        # Use both marketing and route evidence if available
        all_urls = (mkt_urls or []) + (rte_urls or [])
        await evaluator.verify(
            claim=park_access_claim,
            node=park_access_leaf,
            sources=all_urls,
            additional_instruction=(
                "If the marketing page or other provided sources indicate the city is a gateway to the named national park, mark supported."
            )
        )
    else:
        # If park not provided, add a skipped leaf (not critical)
        park_access_leaf = evaluator.add_leaf(
            id="np_park_access_from_city",
            desc="Marketed park accessibility supported (park name missing in answer)",
            parent=node,
            critical=False,
            score=0.0,
            status="skipped"
        )


async def verify_terrace_opening(
    evaluator: Evaluator,
    parent_node,
    item: Optional[TerraceOpening]
) -> None:
    node = evaluator.add_parallel(
        id="BNA_Outdoor_Terrace_Opening",
        desc="Opening date (month and year) of BNA's first outdoor terrace in the Concourse D extension is provided and supported",
        parent=parent_node,
        critical=False
    )

    exists = (
        item is not None and
        item.month is not None and item.month.strip() != "" and
        item.year is not None and item.year.strip() != "" and
        isinstance(item.urls, list) and len(item.urls) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="terrace_opening_exists",
        desc="Opening month/year and at least one source URL provided",
        parent=node,
        critical=True
    )

    month = item.month if item and item.month else ""
    year = item.year if item and item.year else ""
    urls = item.urls if item else []

    leaf = evaluator.add_leaf(
        id="terrace_opening_supported",
        desc=f"Opening date supported: first outdoor terrace opened in {month} {year}",
        parent=node,
        critical=True
    )
    claim = f"Nashville International Airport's first outdoor terrace in the Concourse D extension opened in {month} {year}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Check that the source explicitly states the opening month and year for the first outdoor terrace in the Concourse D extension."
        )
    )


async def verify_terrace_location(
    evaluator: Evaluator,
    parent_node,
    item: Optional[TerraceLocation]
) -> None:
    node = evaluator.add_parallel(
        id="BNA_Outdoor_Amenity_Location",
        desc="Concourse location of the outdoor terrace amenity at BNA is provided and supported",
        parent=parent_node,
        critical=False
    )

    exists = (
        item is not None and
        item.concourse is not None and item.concourse.strip() != "" and
        isinstance(item.urls, list) and len(item.urls) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="terrace_location_exists",
        desc="Concourse and at least one source URL provided",
        parent=node,
        critical=True
    )

    concourse = item.concourse if item and item.concourse else ""
    urls = item.urls if item else []

    leaf = evaluator.add_leaf(
        id="terrace_location_supported",
        desc=f"Concourse location supported: outdoor terrace is in {concourse}",
        parent=node,
        critical=True
    )
    claim = f"The outdoor terrace amenity at Nashville International Airport is located in {concourse}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="The source should clearly identify the concourse where the outdoor terrace is located."
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
    Evaluate an answer for low-cost airline outdoor destination options from BNA (2026 guide).
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_bna_outdoor(),
        template_class=BNAOutdoorExtraction,
        extraction_name="bna_outdoor_extraction"
    )

    # Verify Rocky Mountain access airline
    await verify_route_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Rocky_Mountain_Access_Airline",
        category_desc="Budget airline: nonstop BNA → Rocky Mountain NP/Colorado access city",
        item=extracted.rocky,
        region_kind="rocky"
    )

    # Verify Mexico outdoor/beach destination airline
    await verify_route_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Mexican_Outdoor_Destination_Airline",
        category_desc="Budget airline: nonstop BNA → Mexican outdoor/beach destination",
        item=extracted.mexico,
        region_kind="mexico"
    )

    # Verify Pacific Northwest access airline
    await verify_route_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Pacific_Northwest_Access_Airline",
        category_desc="Budget airline: nonstop BNA → Pacific Northwest outdoor access city",
        item=extracted.pnw,
        region_kind="pnw"
    )

    # Verify U.S. beach/coastal destination airline
    await verify_route_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="Beach_Destination_Airline",
        category_desc="Budget airline: nonstop BNA → U.S. beach/coastal destination",
        item=extracted.beach,
        region_kind="beach"
    )

    # Verify national park-focused marketing airline
    await verify_park_marketing(
        evaluator=evaluator,
        parent_node=root,
        item=extracted.park_marketing
    )

    # Verify terrace opening date
    await verify_terrace_opening(
        evaluator=evaluator,
        parent_node=root,
        item=extracted.terrace_opening
    )

    # Verify terrace location (concourse)
    await verify_terrace_location(
        evaluator=evaluator,
        parent_node=root,
        item=extracted.terrace_location
    )

    return evaluator.get_summary()