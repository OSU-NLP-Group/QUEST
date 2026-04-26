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
TASK_ID = "finger_lakes_parks_2026"
TASK_DESCRIPTION = (
    "I am planning a family camping trip to the Finger Lakes region of New York for summer 2026, and I need to "
    "identify three New York State Parks that meet specific requirements. Each park must be located directly on one of "
    "the Finger Lakes (Seneca, Cayuga, Keuka, Canandaigua, Owasco, Skaneateles, Conesus, Hemlock, Canadice, Honeoye, or "
    "Otisco Lake) and must have comprehensive camping and recreational facilities.\n\n"
    "For each of the three parks, provide the following information:\n\n"
    "Basic Information:\n"
    "- Official park name\n"
    "- Which specific Finger Lake the park is located on\n"
    "- Link to the park's official page on parks.ny.gov\n\n"
    "Camping Requirements:\n"
    "- The park must have at least 40 campsites with electric hookups\n"
    "- The park must have at least 100 total campsites (including both electric and non-electric sites, but excluding cabins and pavilions)\n"
    "- The park's 2026 camping season must include the period from June 1 through September 30, 2026\n"
    "- The park must use the official New York State Parks reservation system (accessible through newyorkstateparks.reserveamerica.com or parks.ny.gov)\n"
    "- Provide the specific number of electric campsites and total campsites for each park\n\n"
    "Boating and Water Recreation:\n"
    "- The park must have a boat launch facility\n"
    "- The park must offer at least one additional form of water recreation beyond the boat launch (such as: marina with boat slips, boat rentals, kayak rentals, canoe rentals, paddleboard rentals, or boat dockage)\n"
    "- Specify what type of additional water recreation is offered\n\n"
    "Other Amenities:\n"
    "- The park must have a designated swimming beach\n"
    "- The park must have hiking trails\n\n"
    "For each park, provide reference URLs from official sources (parks.ny.gov or newyorkstateparks.reserveamerica.com) that document each of the key requirements: the number of electric campsites, total campsite count, 2026 camping season dates, boat launch availability, the additional water recreation option, swimming beach, and hiking trails."
)

ALLOWED_FINGER_LAKES = [
    "Seneca Lake", "Cayuga Lake", "Keuka Lake", "Canandaigua Lake", "Owasco Lake",
    "Skaneateles Lake", "Conesus Lake", "Hemlock Lake", "Canadice Lake", "Honeoye Lake", "Otisco Lake"
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkInfo(BaseModel):
    """Structured info for a single park, extracted from the agent's answer."""
    name: Optional[str] = None
    lake: Optional[str] = None
    official_url: Optional[str] = None

    # Camping: counts + sources
    electric_count: Optional[str] = None
    electric_source_urls: List[str] = Field(default_factory=list)

    total_count: Optional[str] = None
    total_source_urls: List[str] = Field(default_factory=list)

    # Season and reservations
    season_start_2026: Optional[str] = None
    season_end_2026: Optional[str] = None
    season_source_urls: List[str] = Field(default_factory=list)

    reservation_url: Optional[str] = None
    reservation_source_urls: List[str] = Field(default_factory=list)

    # Boating & water recreation
    boat_launch_source_urls: List[str] = Field(default_factory=list)
    water_recreation_type: Optional[str] = None
    water_recreation_source_urls: List[str] = Field(default_factory=list)

    # Other amenities
    beach_source_urls: List[str] = Field(default_factory=list)
    trails_source_urls: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    """Model capturing up to three parks with all required fields."""
    parks: List[ParkInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return (
        "Extract up to the first three New York State Parks described in the answer that the agent proposes for the "
        "Finger Lakes trip. For each park, extract the following fields exactly as stated in the answer, and extract "
        "all official reference URLs (only parks.ny.gov or newyorkstateparks.reserveamerica.com) cited for each requirement:\n\n"
        "For each park, return an object with these fields:\n"
        "- name: official park name\n"
        "- lake: which specific Finger Lake the park is located on (use the exact lake name stated; if missing, null)\n"
        "- official_url: URL to the park's official page on parks.ny.gov (if none given, null)\n"
        "- electric_count: specific number of campsites with electric hookups as stated (string; if missing, null)\n"
        "- electric_source_urls: array of official URLs that document the number of electric campsites (filter only parks.ny.gov or newyorkstateparks.reserveamerica.com)\n"
        "- total_count: specific total number of campsites as stated (string; exclude cabins/pavilions; if missing, null)\n"
        "- total_source_urls: array of official URLs that document the total campsite count (filter domains as above)\n"
        "- season_start_2026: start date of the 2026 camping season if explicitly provided (string; else null)\n"
        "- season_end_2026: end date of the 2026 camping season if explicitly provided (string; else null)\n"
        "- season_source_urls: array of official URLs that document the 2026 camping season dates and/or show reservation availability for those dates (filter domains)\n"
        "- reservation_url: the URL used to make reservations (prefer a direct newyorkstateparks.reserveamerica.com page; else parks.ny.gov reservation link; if none given, null)\n"
        "- reservation_source_urls: array of official URLs that show or link to the reservation system (filter domains)\n"
        "- boat_launch_source_urls: array of official URLs that document boat launch facility availability (filter domains)\n"
        "- water_recreation_type: the specific additional water recreation offered (e.g., marina/boat slips, boat rentals, kayak/canoe/Paddleboard rentals, dockage). If multiple types are listed, choose the most prominent or the first mentioned; if none, null.\n"
        "- water_recreation_source_urls: array of official URLs that document the additional water recreation option (filter domains)\n"
        "- beach_source_urls: array of official URLs that document a designated swimming beach (filter domains)\n"
        "- trails_source_urls: array of official URLs that document hiking trails (filter domains)\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly present in the answer. If anything is missing, set the field to null or empty array.\n"
        "2) For URLs, include only official sources: parks.ny.gov or newyorkstateparks.reserveamerica.com. If the answer mentions other sources, ignore them.\n"
        "3) Keep numbers as strings (e.g., '124', 'about 110'). Do not convert to numeric types.\n"
        "4) Return a JSON object with a 'parks' array containing up to 3 park objects in the order they appear in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third"][n] if 0 <= n <= 2 else f"#{n + 1}"


def _filter_official(urls: List[str]) -> List[str]:
    valid = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        lu = u.lower().strip()
        if "parks.ny.gov" in lu or "newyorkstateparks.reserveamerica.com" in lu:
            valid.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for u in valid:
        if u not in seen:
            dedup.append(u)
            seen.add(u)
    return dedup


def _merge_sources(*lists: List[str], include: Optional[List[str]] = None) -> List[str]:
    merged = []
    for lst in lists:
        merged.extend(lst or [])
    if include:
        merged.extend(include)
    # Filter to official only
    merged = _filter_official(merged)
    # Deduplicate preserving order
    seen = set()
    out = []
    for u in merged:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkInfo,
    idx: int,
) -> None:
    """Build the verification subtree for one park and perform all checks."""
    park_node = evaluator.add_parallel(
        id=f"park_{idx + 1}",
        desc=f"{ordinal(idx)} qualifying state park with all required amenities",
        parent=parent_node,
        critical=False
    )

    # ---------------- Basic Information ----------------
    basic_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_basic_information",
        desc="Basic park identification and location information",
        parent=park_node,
        critical=True
    )

    # Official park name
    name_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_park_name",
        desc="Official name of the New York State Park",
        parent=basic_node,
        critical=True
    )
    name_claim = f"The official park name is '{park.name or ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=park.official_url,
        additional_instruction="Verify on the parks.ny.gov official page that the page header or prominent title matches the stated park name (minor casing or punctuation variations are acceptable)."
    )

    # Lake location (must be on one of the Finger Lakes directly)
    lake_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_lake_location",
        desc="Located directly on one of the Finger Lakes (Seneca, Cayuga, Keuka, Canandaigua, Owasco, Skaneateles, Conesus, Hemlock, Canadice, Honeoye, or Otisco Lake)",
        parent=basic_node,
        critical=True
    )
    lake_value = park.lake or ""
    lake_claim = f"This park is located directly on the shore of {lake_value}, which is one of the Finger Lakes."
    await evaluator.verify(
        claim=lake_claim,
        node=lake_leaf,
        sources=park.official_url,
        additional_instruction=(
            "Confirm the park is directly lakeside on the specified Finger Lake (not merely in the region). "
            f"Allowed lakes: {', '.join(ALLOWED_FINGER_LAKES)}. Use the official page text/screenshot to confirm."
        )
    )

    # Official website verification
    website_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_official_website",
        desc="Link to the park's official page on parks.ny.gov",
        parent=basic_node,
        critical=True
    )
    website_claim = f"This URL is the official parks.ny.gov page for '{park.name or ''}'."
    await evaluator.verify(
        claim=website_claim,
        node=website_leaf,
        sources=park.official_url,
        additional_instruction="Pass only if the URL domain is parks.ny.gov and the page content corresponds to the named park."
    )

    # ---------------- Camping Facilities ----------------
    camping_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_camping_facilities",
        desc="Camping infrastructure and capacity requirements",
        parent=park_node,
        critical=True
    )

    # Electric campsites
    electric_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_electric_campsites",
        desc="Number of campsites with electric hookups",
        parent=camping_node,
        critical=True
    )
    electric_sources_union = _merge_sources(park.electric_source_urls, include=[park.official_url] if park.official_url else None)

    electric_count_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_electric_count",
        desc="Park has at least 40 electric campsites available",
        parent=electric_node,
        critical=True
    )
    electric_count_claim = "The park has at least 40 campsites with electric hookups."
    await evaluator.verify(
        claim=electric_count_claim,
        node=electric_count_leaf,
        sources=electric_sources_union,
        additional_instruction="Use official pages (ReserveAmerica or parks.ny.gov) that indicate the number of electric sites; consider multiple loops and sum if explicit; ignore cabins/pavilions."
    )

    electric_source_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_electric_source",
        desc="URL reference documenting the number of electric campsites",
        parent=electric_node,
        critical=True
    )
    electric_exact_claim = (
        f"At least one provided official source explicitly states the number of electric campsites "
        f"(e.g., '{park.electric_count or ''}') for this park."
    )
    await evaluator.verify(
        claim=electric_exact_claim,
        node=electric_source_leaf,
        sources=_filter_official(park.electric_source_urls),
        additional_instruction="On ReserveAmerica, look for 'sites with electric' or similar counts; on parks.ny.gov, accept explicit numeric statements of electric sites."
    )

    # Total capacity
    capacity_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_total_capacity",
        desc="Total campsite capacity including electric and non-electric sites",
        parent=camping_node,
        critical=True
    )
    capacity_sources_union = _merge_sources(park.total_source_urls, include=[park.official_url] if park.official_url else None)

    capacity_count_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_capacity_count",
        desc="Park has at least 100 total campsites (excluding cabins and pavilions)",
        parent=capacity_node,
        critical=True
    )
    capacity_count_claim = "The park has at least 100 total campsites (excluding cabins and pavilions)."
    await evaluator.verify(
        claim=capacity_count_claim,
        node=capacity_count_leaf,
        sources=capacity_sources_union,
        additional_instruction="Confirm total campsite count from official sources; exclude cabins/pavilions and count only campsites."
    )

    capacity_source_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_capacity_source",
        desc="URL reference documenting total campsite count",
        parent=capacity_node,
        critical=True
    )
    capacity_exact_claim = (
        f"At least one provided official source explicitly states the total number of campsites "
        f"(e.g., '{park.total_count or ''}') for this park, excluding cabins/pavilions."
    )
    await evaluator.verify(
        claim=capacity_exact_claim,
        node=capacity_source_leaf,
        sources=_filter_official(park.total_source_urls),
        additional_instruction="Look for explicit counts on ReserveAmerica or parks.ny.gov pages; do not treat cabins/pavilions as campsites."
    )

    # Camping season
    season_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_camping_season",
        desc="2026 camping season dates and availability",
        parent=camping_node,
        critical=True
    )
    season_coverage_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_season_coverage",
        desc="Camping season includes at least June 1 through September 30, 2026",
        parent=season_node,
        critical=True
    )
    season_coverage_claim = "The 2026 camping season includes at least the period from June 1 through September 30, 2026."
    await evaluator.verify(
        claim=season_coverage_claim,
        node=season_coverage_leaf,
        sources=_filter_official(park.season_source_urls),
        additional_instruction="Accept if official pages list season dates covering that range or show reservation availability across that period in 2026."
    )

    season_source_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_season_source",
        desc="URL reference documenting 2026 camping season dates",
        parent=season_node,
        critical=True
    )
    season_source_claim = (
        f"At least one provided official source explicitly lists 2026 season dates "
        f"(e.g., '{(park.season_start_2026 or '')} to {(park.season_end_2026 or '')}') or clearly shows "
        "reservation availability covering June 1–September 30, 2026."
    )
    await evaluator.verify(
        claim=season_source_claim,
        node=season_source_leaf,
        sources=_filter_official(park.season_source_urls),
        additional_instruction="Check for 'Season' lines or calendars on ReserveAmerica; the content should directly support the claimed date coverage for 2026."
    )

    # Reservation system
    reservation_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_reservation_system",
        desc="Uses New York State Parks reservation system (newyorkstateparks.reserveamerica.com or parks.ny.gov reservation system)",
        parent=camping_node,
        critical=True
    )
    system_confirmed_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_system_confirmed",
        desc="Reservation system is confirmed to be the official NY State Parks system",
        parent=reservation_node,
        critical=True
    )
    system_confirmed_claim = (
        "This park uses the official New York State Parks reservation system (either via newyorkstateparks.reserveamerica.com "
        "or the parks.ny.gov reservation interface)."
    )
    reservation_sources_union = _merge_sources(park.reservation_source_urls, include=[park.reservation_url] if park.reservation_url else [park.official_url] if park.official_url else None)
    await evaluator.verify(
        claim=system_confirmed_claim,
        node=system_confirmed_leaf,
        sources=reservation_sources_union,
        additional_instruction="Pass only if the reservation link or page is under newyorkstateparks.reserveamerica.com or clearly the parks.ny.gov reservation interface for this park."
    )

    system_source_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_system_source",
        desc="URL reference to reservation system information",
        parent=reservation_node,
        critical=True
    )
    system_source_claim = "The provided reservation URL is under newyorkstateparks.reserveamerica.com or parks.ny.gov and corresponds to this park."
    await evaluator.verify(
        claim=system_source_claim,
        node=system_source_leaf,
        sources=park.reservation_url,
        additional_instruction="Confirm domain and that the page is the reservation portal for the specific park."
    )

    # ---------------- Boating Facilities ----------------
    boating_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_boating_facilities",
        desc="Boating and water access infrastructure",
        parent=park_node,
        critical=True
    )

    # Boat launch
    launch_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_boat_launch",
        desc="Boat launch facility availability",
        parent=boating_node,
        critical=True
    )
    launch_available_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_launch_available",
        desc="Park has a boat launch facility",
        parent=launch_node,
        critical=True
    )
    launch_sources_union = _merge_sources(park.boat_launch_source_urls, include=[park.official_url] if park.official_url else None)
    launch_claim = "The park has a boat launch facility."
    await evaluator.verify(
        claim=launch_claim,
        node=launch_available_leaf,
        sources=launch_sources_union,
        additional_instruction="Confirm on official sources; e.g., amenities list showing 'Boat Launch'."
    )

    launch_source_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_launch_source",
        desc="URL reference documenting boat launch facility",
        parent=launch_node,
        critical=True
    )
    launch_source_claim = "At least one provided official source explicitly documents the presence of a boat launch at this park."
    await evaluator.verify(
        claim=launch_source_claim,
        node=launch_source_leaf,
        sources=_filter_official(park.boat_launch_source_urls),
        additional_instruction="Look for explicit mention of 'Boat launch' on amenities or facilities sections."
    )

    # Additional water recreation
    waterrec_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_additional_water_recreation",
        desc="At least one additional water recreation option beyond boat launch (marina with slips, boat rentals, kayak rentals, canoe rentals, paddleboard rentals, or dockage)",
        parent=boating_node,
        critical=True
    )
    recreation_type_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_recreation_type",
        desc="Specific type of additional water recreation offered",
        parent=waterrec_node,
        critical=True
    )
    recreation_type_claim = (
        f"The park offers {park.water_recreation_type or ''} as an additional water recreation option beyond the boat launch."
    )
    recreation_sources_union = _merge_sources(park.water_recreation_source_urls, include=[park.official_url] if park.official_url else None)
    await evaluator.verify(
        claim=recreation_type_claim,
        node=recreation_type_leaf,
        sources=recreation_sources_union,
        additional_instruction="Examples include marina/boat slips, boat rentals, kayak/canoe/paddleboard rentals, or dockage; confirm explicitly on official sources."
    )

    recreation_source_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_recreation_source",
        desc="URL reference documenting the additional water recreation option",
        parent=waterrec_node,
        critical=True
    )
    recreation_source_claim = (
        f"At least one provided official source explicitly documents that {park.water_recreation_type or 'an additional water recreation option'} is offered at this park."
    )
    await evaluator.verify(
        claim=recreation_source_claim,
        node=recreation_source_leaf,
        sources=_filter_official(park.water_recreation_source_urls),
        additional_instruction="The source should clearly indicate the specific additional water recreation offering."
    )

    # ---------------- Other Amenities ----------------
    other_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_other_amenities",
        desc="Additional recreational facilities required for family camping",
        parent=park_node,
        critical=True
    )

    # Swimming beach
    beach_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_swimming_beach",
        desc="Designated swimming beach availability",
        parent=other_node,
        critical=True
    )
    beach_available_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_beach_available",
        desc="Park has a designated swimming beach",
        parent=beach_node,
        critical=True
    )
    beach_sources_union = _merge_sources(park.beach_source_urls, include=[park.official_url] if park.official_url else None)
    beach_claim = "The park has a designated swimming beach."
    await evaluator.verify(
        claim=beach_claim,
        node=beach_available_leaf,
        sources=beach_sources_union,
        additional_instruction="Look for 'Swimming' or 'Beach' amenity on official sources; accept seasonal beach statements."
    )

    beach_source_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_beach_source",
        desc="URL reference documenting swimming beach",
        parent=beach_node,
        critical=True
    )
    beach_source_claim = "At least one provided official source explicitly documents a designated swimming beach at this park."
    await evaluator.verify(
        claim=beach_source_claim,
        node=beach_source_leaf,
        sources=_filter_official(park.beach_source_urls),
        additional_instruction="The page should clearly mention a swimming beach; accept seasonal availability indications."
    )

    # Hiking trails
    trails_node = evaluator.add_parallel(
        id=f"park_{idx + 1}_hiking_trails",
        desc="Hiking trail availability",
        parent=other_node,
        critical=True
    )
    trails_available_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_trails_available",
        desc="Park has hiking trails",
        parent=trails_node,
        critical=True
    )
    trails_sources_union = _merge_sources(park.trails_source_urls, include=[park.official_url] if park.official_url else None)
    trails_claim = "The park has hiking trails."
    await evaluator.verify(
        claim=trails_claim,
        node=trails_available_leaf,
        sources=trails_sources_union,
        additional_instruction="Confirm that hiking trails are available at the park via official sources."
    )

    trails_source_leaf = evaluator.add_leaf(
        id=f"park_{idx + 1}_trails_source",
        desc="URL reference documenting hiking trails",
        parent=trails_node,
        critical=True
    )
    trails_source_claim = "At least one provided official source explicitly documents hiking trails at this park."
    await evaluator.verify(
        claim=trails_source_claim,
        node=trails_source_leaf,
        sources=_filter_official(park.trails_source_urls),
        additional_instruction="The source should clearly indicate the presence of hiking trails."
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
    Evaluate an answer for the Finger Lakes parks 2026 task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    # Note: Root set as non-critical parallel to allow partial credit across parks and to avoid
    # critical-child consistency constraint (critical parents must have all critical children).
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_selection"
    )

    # Record ground truth-like constraints for context
    evaluator.add_custom_info(
        info={"allowed_finger_lakes": ALLOWED_FINGER_LAKES},
        info_type="constraints",
        info_name="allowed_lakes"
    )

    # Prepare up to three parks, pad with empty ParkInfo if fewer than 3
    parks: List[ParkInfo] = list(extracted.parks[:3])
    while len(parks) < 3:
        parks.append(ParkInfo())

    # Build verification tree and run checks per park
    for i in range(3):
        await verify_park(evaluator, root, parks[i], i)

    # Return structured summary
    return evaluator.get_summary()