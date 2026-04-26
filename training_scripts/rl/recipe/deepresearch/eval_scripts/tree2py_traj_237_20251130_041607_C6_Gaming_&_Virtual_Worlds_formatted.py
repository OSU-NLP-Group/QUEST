import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "esports_lounge_benchmark"
TASK_DESCRIPTION = (
    "I'm planning to open a small esports gaming lounge in Texas and want to benchmark against successful venues. "
    "Find three established gaming lounges, each in a different US city (at least one in Florida and at least one in Illinois or another Midwest state). "
    "For each lounge, provide: (1) complete physical address, (2) contact information (phone or email), (3) number of gaming stations/PCs if publicly available, "
    "(4) at least two different pricing options (hourly rates or packages), (5) at least three amenities they offer, and (6) a description of their typical weekly operating schedule. "
    "Additionally, based on publicly available industry standards for competitive gaming, provide: (A) minimum recommended PC specifications including CPU (minimum cores and recommended processor family like Intel Core i7/i9 or AMD Ryzen 7/9), "
    "GPU (recommended series like NVIDIA RTX or AMD Radeon and minimum VRAM), RAM capacity, and monitor refresh rate, and (B) the estimated total startup cost range for opening a venue with 15-20 gaming stations."
)

# --------------------------------------------------------------------------- #
# State/Region Utilities                                                      #
# --------------------------------------------------------------------------- #
STATE_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC", "dc": "DC"
}
MIDWEST_STATES = {"IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"}


def normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    key = s.lower()
    return STATE_ABBR.get(key, None)


def parse_city_state_from_address(address: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not address:
        return None, None
    # Try patterns like "City, ST" or "City, State"
    m = re.search(r"([A-Za-z\.\-\'\s]+),\s*([A-Z]{2})\b", address)
    if m:
        city = m.group(1).strip()
        state = normalize_state(m.group(2))
        return city, state
    # Try "City, StateName"
    m2 = re.search(r"([A-Za-z\.\-\'\s]+),\s*([A-Za-z\s]+)$", address)
    if m2:
        city = m2.group(1).strip()
        state = normalize_state(m2.group(2))
        return city, state
    return None, None


def valid_url(u: str) -> bool:
    if not u or not isinstance(u, str):
        return False
    u = u.strip()
    return u.startswith("http://") or u.startswith("https://")


def all_valid_urls(urls: List[str]) -> bool:
    if not urls:
        return False
    return all(valid_url(u) for u in urls)


def dedupe_and_select_venues(venues: List["VenueItem"], k: int = 3) -> List["VenueItem"]:
    seen = set()
    selected: List[VenueItem] = []
    for v in venues:
        # Fill missing city/state from address if possible
        city = v.city
        state = v.state
        if not city or not state:
            pc, ps = parse_city_state_from_address(v.address)
            city = city or pc
            state = state or ps
        state = normalize_state(state) if state else None
        key = (v.name.strip().lower() if v.name else "", (city or "").strip().lower(), (state or "").strip().upper())
        if key in seen:
            continue
        seen.add(key)
        # Propagate parsed city/state back into the object for downstream checks
        if not v.city:
            v.city = city
        if not v.state:
            v.state = state
        selected.append(v)
        if len(selected) >= k:
            break
    # Pad if fewer than k
    while len(selected) < k:
        selected.append(VenueItem())
    return selected


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    address: Optional[str] = None
    address_sources: List[str] = Field(default_factory=list)

    contact: Optional[str] = None
    contact_sources: List[str] = Field(default_factory=list)

    stations: Optional[str] = None
    stations_sources: List[str] = Field(default_factory=list)

    pricing_options: List[str] = Field(default_factory=list)
    pricing_sources: List[str] = Field(default_factory=list)

    amenities: List[str] = Field(default_factory=list)
    amenities_sources: List[str] = Field(default_factory=list)

    weekly_hours: Optional[str] = None
    hours_sources: List[str] = Field(default_factory=list)

    homepage_url: Optional[str] = None
    general_sources: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


class PCSpecs(BaseModel):
    cpu: Optional[str] = None
    cpu_sources: List[str] = Field(default_factory=list)

    gpu: Optional[str] = None
    gpu_sources: List[str] = Field(default_factory=list)

    ram: Optional[str] = None
    ram_sources: List[str] = Field(default_factory=list)

    monitor_refresh_rate: Optional[str] = None
    monitor_sources: List[str] = Field(default_factory=list)


class StartupCost(BaseModel):
    cost_range: Optional[str] = None
    cost_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return (
        "Extract all gaming lounges mentioned in the answer, keeping their original order. For each venue, extract:\n"
        "- name: The venue or business name.\n"
        "- city: The US city the venue is located in (infer from the address text if not explicitly listed).\n"
        "- state: The US state (as full name or 2-letter abbreviation; infer from address if necessary).\n"
        "- address: The complete physical address as presented (street + city + state, include ZIP if shown).\n"
        "- address_sources: URL(s) explicitly provided that support the address.\n"
        "- contact: A phone number or email address, if provided.\n"
        "- contact_sources: URL(s) explicitly provided that support the contact.\n"
        "- stations: The number of gaming stations/PCs (if the answer provides it).\n"
        "- stations_sources: URL(s) explicitly provided that support the stations count.\n"
        "- pricing_options: At least two pricing options if provided in the answer (e.g., hourly/package/member pricing); include every option shown.\n"
        "- pricing_sources: URL(s) explicitly provided that support the pricing information.\n"
        "- amenities: At least three amenities if provided in the answer (e.g., consoles, VR, tournaments, snacks, streaming setups).\n"
        "- amenities_sources: URL(s) explicitly provided that support the amenities.\n"
        "- weekly_hours: A description of the typical weekly operating schedule.\n"
        "- hours_sources: URL(s) explicitly provided that support the weekly hours.\n"
        "- homepage_url: A general homepage URL for the venue if provided.\n"
        "- general_sources: Any additional relevant URLs provided for the venue.\n"
        "Rules:\n"
        "1) Only extract URLs explicitly present in the answer text (including markdown links). Do not invent URLs.\n"
        "2) If an item is not present in the answer, set the value to null or an empty list as appropriate.\n"
        "3) If the answer mentions more than three venues, still extract all of them in order."
    )


def prompt_extract_pc_specs() -> str:
    return (
        "From the answer, extract the minimum recommended PC specifications for competitive gaming. Extract:\n"
        "- cpu: A string that clearly includes a minimum core count and a recommended processor family (e.g., Intel Core i7/i9 or AMD Ryzen 7/9).\n"
        "- cpu_sources: URL(s) explicitly provided supporting the CPU spec.\n"
        "- gpu: A string that clearly includes a recommended series (e.g., NVIDIA RTX or AMD Radeon) and a minimum VRAM.\n"
        "- gpu_sources: URL(s) explicitly provided supporting the GPU spec.\n"
        "- ram: The minimum recommended RAM capacity string (e.g., '16 GB minimum').\n"
        "- ram_sources: URL(s) explicitly provided supporting the RAM recommendation.\n"
        "- monitor_refresh_rate: The minimum recommended monitor refresh rate (e.g., '144 Hz minimum').\n"
        "- monitor_sources: URL(s) explicitly provided supporting the monitor recommendation.\n"
        "Only include URLs actually present in the answer. If a field is not present, set it to null or an empty list."
    )


def prompt_extract_startup_cost() -> str:
    return (
        "From the answer, extract the estimated total startup cost range for opening a venue with 15–20 gaming stations. Extract:\n"
        "- cost_range: The quoted cost range string exactly as given (e.g., '$200k–$350k').\n"
        "- cost_sources: URL(s) explicitly provided supporting the cost estimate.\n"
        "Only include URLs actually present in the answer. If the range is not present, set it to null."
    )


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def combine_sources(*source_lists: List[str], fallback: Optional[str] = None) -> List[str]:
    combined = []
    for sl in source_lists:
        if sl:
            combined.extend([u for u in sl if valid_url(u)])
    if fallback and valid_url(fallback):
        combined.append(fallback)
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for u in combined:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def check_citations_presence(venues: List[VenueItem], specs: PCSpecs, cost: StartupCost) -> bool:
    def has_sources_if_value(value_present: bool, urls: List[str]) -> bool:
        return (not value_present) or all_valid_urls(urls)

    # Venue-level checks: Sources required only when the corresponding value(s) are provided
    for v in venues:
        if v.address and not all_valid_urls(v.address_sources):
            return False
        if v.contact and not all_valid_urls(v.contact_sources):
            return False
        if v.stations and not all_valid_urls(v.stations_sources):
            return False
        if v.pricing_options and len(v.pricing_options) > 0 and not all_valid_urls(v.pricing_sources):
            return False
        if v.amenities and len(v.amenities) > 0 and not all_valid_urls(v.amenities_sources):
            return False
        if v.weekly_hours and not all_valid_urls(v.hours_sources):
            return False

    # PC spec checks
    if specs.cpu and not all_valid_urls(specs.cpu_sources):
        return False
    if specs.gpu and not all_valid_urls(specs.gpu_sources):
        return False
    if specs.ram and not all_valid_urls(specs.ram_sources):
        return False
    if specs.monitor_refresh_rate and not all_valid_urls(specs.monitor_sources):
        return False

    # Startup cost check
    if cost.cost_range and not all_valid_urls(cost.cost_sources):
        return False

    return True


def compute_geo_constraints(venues3: List[VenueItem], all_venues_count: int) -> Dict[str, bool]:
    # Exactly three in the answer
    three_total = (all_venues_count == 3)

    # Normalize city/state and compute unique cities among selected 3
    cities = []
    has_fl = False
    has_midwest = False
    for v in venues3:
        city = (v.city or "").strip()
        state = normalize_state(v.state) if v.state else None
        if (not city or not state) and v.address:
            pc, ps = parse_city_state_from_address(v.address)
            city = city or (pc or "")
            state = state or (ps or None)

        cities.append(city.lower() if city else "")
        st = state or ""
        if st == "FL":
            has_fl = True
        if st in MIDWEST_STATES:
            has_midwest = True

    three_different_cities = len({c for c in cities if c}) == 3

    return {
        "Three_Venues_Total": three_total,
        "Three_Different_US_Cities": three_different_cities,
        "At_Least_One_Florida": has_fl,
        "At_Least_One_Illinois_or_Midwest": has_midwest
    }


# --------------------------------------------------------------------------- #
# Venue verification                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(evaluator: Evaluator, parent_node, venue: VenueItem, index: int) -> None:
    """
    Build verification leaves for one venue under the provided parent node.
    """
    vid = index + 1
    v_node = evaluator.add_parallel(
        id=f"Venue_{vid}",
        desc=f"Gaming lounge #{vid} details.",
        parent=parent_node,
        critical=False
    )

    # Address
    addr_node = evaluator.add_leaf(
        id=f"Venue_{vid}_Address",
        desc=f"Provide complete physical address for venue #{vid}.",
        parent=v_node,
        critical=True
    )
    if venue.address:
        addr_sources = combine_sources(venue.address_sources, venue.general_sources, fallback=venue.homepage_url)
        claim = f"The complete physical address of {venue.name or f'venue #{vid}'} is: {venue.address}."
        await evaluator.verify(
            claim=claim,
            node=addr_node,
            sources=addr_sources,
            additional_instruction="Verify the exact street address (including city and state) is shown on the provided page(s). Allow minor formatting differences."
        )
    else:
        addr_node.score = 0.0
        addr_node.status = "failed"

    # Contact
    contact_node = evaluator.add_leaf(
        id=f"Venue_{vid}_Contact",
        desc=f"Provide contact information (phone or email) for venue #{vid}.",
        parent=v_node,
        critical=True
    )
    if venue.contact:
        contact_sources = combine_sources(venue.contact_sources, venue.general_sources, fallback=venue.homepage_url)
        claim = f"The contact (phone or email) for {venue.name or f'venue #{vid}'} is: {venue.contact}."
        await evaluator.verify(
            claim=claim,
            node=contact_node,
            sources=contact_sources,
            additional_instruction="Verify that the provided phone number or email address appears on the source page(s). Minor formatting differences are acceptable."
        )
    else:
        contact_node.score = 0.0
        contact_node.status = "failed"

    # Stations (non-critical)
    stations_node = evaluator.add_leaf(
        id=f"Venue_{vid}_Stations",
        desc=f"Provide number of gaming stations/PCs for venue #{vid} if publicly available.",
        parent=v_node,
        critical=False
    )
    if venue.stations:
        stations_sources = combine_sources(venue.stations_sources, venue.general_sources, fallback=venue.homepage_url)
        claim = f"The number of gaming stations (PCs) at {venue.name or f'venue #{vid}'} is {venue.stations}."
        await evaluator.verify(
            claim=claim,
            node=stations_node,
            sources=stations_sources,
            additional_instruction="Verify the station/PC count is stated on the source page(s). Accept 'at least' or '+' counts if clearly indicated."
        )
    else:
        # Not provided or not publicly available; treat as failed but non-critical
        stations_node.score = 0.0
        stations_node.status = "failed"

    # Pricing (critical)
    pricing_node = evaluator.add_leaf(
        id=f"Venue_{vid}_Pricing",
        desc=f"Provide at least two different pricing options (hourly rates and/or packages) for venue #{vid}.",
        parent=v_node,
        critical=True
    )
    if venue.pricing_options and len(venue.pricing_options) >= 2:
        pricing_sources = combine_sources(venue.pricing_sources, venue.general_sources, fallback=venue.homepage_url)
        listed = "; ".join(venue.pricing_options)
        claim = f"{venue.name or f'venue #{vid}'} offers at least two different pricing options: {listed}."
        await evaluator.verify(
            claim=claim,
            node=pricing_node,
            sources=pricing_sources,
            additional_instruction="Confirm that the page(s) list at least two distinct rate options (e.g., hourly, day passes, memberships, packages)."
        )
    else:
        pricing_node.score = 0.0
        pricing_node.status = "failed"

    # Amenities (critical)
    amenities_node = evaluator.add_leaf(
        id=f"Venue_{vid}_Amenities",
        desc=f"Provide at least three amenities offered by venue #{vid}.",
        parent=v_node,
        critical=True
    )
    if venue.amenities and len(venue.amenities) >= 3:
        amenities_sources = combine_sources(venue.amenities_sources, venue.general_sources, fallback=venue.homepage_url)
        listed = "; ".join(venue.amenities)
        claim = f"{venue.name or f'venue #{vid}'} offers at least three amenities, such as: {listed}."
        await evaluator.verify(
            claim=claim,
            node=amenities_node,
            sources=amenities_sources,
            additional_instruction="Verify that at least three amenities/features are listed on the provided page(s). Examples: consoles, VR, tournament hosting, snacks/drinks, streaming setups."
        )
    else:
        amenities_node.score = 0.0
        amenities_node.status = "failed"

    # Weekly Hours (critical)
    hours_node = evaluator.add_leaf(
        id=f"Venue_{vid}_Weekly_Hours",
        desc=f"Provide a description of typical weekly operating schedule for venue #{vid}.",
        parent=v_node,
        critical=True
    )
    if venue.weekly_hours:
        hours_sources = combine_sources(venue.hours_sources, venue.general_sources, fallback=venue.homepage_url)
        claim = f"The typical weekly operating schedule for {venue.name or f'venue #{vid}'} is: {venue.weekly_hours}."
        await evaluator.verify(
            claim=claim,
            node=hours_node,
            sources=hours_sources,
            additional_instruction="Check the business hours shown on the page(s). Allow for typical schedule descriptions; minor formatting differences are acceptable."
        )
    else:
        hours_node.score = 0.0
        hours_node.status = "failed"


# --------------------------------------------------------------------------- #
# PC specs and cost verification                                              #
# --------------------------------------------------------------------------- #
async def verify_pc_specs_and_cost(evaluator: Evaluator, parent_node, specs: PCSpecs, cost: StartupCost) -> None:
    ind_node = evaluator.add_parallel(
        id="Industry_Standards_Research",
        desc="Provide competitive-gaming PC standards and a startup cost range for a 15–20 station venue.",
        parent=parent_node,
        critical=False  # Note: Parent is non-critical to satisfy framework's critical-child constraint
    )

    # PC Specifications
    pc_node = evaluator.add_parallel(
        id="PC_Specifications",
        desc="Minimum recommended PC specifications for competitive gaming.",
        parent=ind_node,
        critical=True
    )

    # CPU spec
    cpu_node = evaluator.add_leaf(
        id="CPU_Spec",
        desc="CPU spec includes minimum core count and recommended processor family (Intel Core i7/i9 or AMD Ryzen 7/9).",
        parent=pc_node,
        critical=True
    )
    if specs.cpu:
        cpu_sources = combine_sources(specs.cpu_sources)
        claim = f"The minimum recommended CPU specification for competitive gaming is: {specs.cpu}."
        await evaluator.verify(
            claim=claim,
            node=cpu_node,
            sources=cpu_sources,
            additional_instruction="Verify that the CPU recommendation clearly includes BOTH a minimum core count and a recommended family such as Intel Core i7/i9 or AMD Ryzen 7/9."
        )
    else:
        cpu_node.score = 0.0
        cpu_node.status = "failed"

    # GPU spec
    gpu_node = evaluator.add_leaf(
        id="GPU_Spec",
        desc="GPU spec includes recommended series (NVIDIA RTX or AMD Radeon) and minimum VRAM.",
        parent=pc_node,
        critical=True
    )
    if specs.gpu:
        gpu_sources = combine_sources(specs.gpu_sources)
        claim = f"The minimum recommended GPU specification for competitive gaming is: {specs.gpu}."
        await evaluator.verify(
            claim=claim,
            node=gpu_node,
            sources=gpu_sources,
            additional_instruction="Verify that the GPU recommendation includes BOTH a recommended series (e.g., NVIDIA RTX or AMD Radeon) and a minimum VRAM amount."
        )
    else:
        gpu_node.score = 0.0
        gpu_node.status = "failed"

    # RAM minimum
    ram_node = evaluator.add_leaf(
        id="RAM_Minimum",
        desc="Provide minimum recommended RAM capacity for competitive gaming.",
        parent=pc_node,
        critical=True
    )
    if specs.ram:
        ram_sources = combine_sources(specs.ram_sources)
        claim = f"The minimum recommended RAM capacity for competitive gaming is: {specs.ram}."
        await evaluator.verify(
            claim=claim,
            node=ram_node,
            sources=ram_sources,
            additional_instruction="Verify that the stated minimum RAM capacity is clearly recommended for competitive gaming."
        )
    else:
        ram_node.score = 0.0
        ram_node.status = "failed"

    # Monitor refresh rate
    mon_node = evaluator.add_leaf(
        id="Monitor_Refresh_Rate_Minimum",
        desc="Provide minimum recommended monitor refresh rate for competitive gaming.",
        parent=pc_node,
        critical=True
    )
    if specs.monitor_refresh_rate:
        mon_sources = combine_sources(specs.monitor_sources)
        claim = f"The minimum recommended monitor refresh rate for competitive gaming is: {specs.monitor_refresh_rate}."
        await evaluator.verify(
            claim=claim,
            node=mon_node,
            sources=mon_sources,
            additional_instruction="Verify that the monitor recommendation clearly states a minimum refresh rate suitable for competitive gaming (e.g., 120–144 Hz+)."
        )
    else:
        mon_node.score = 0.0
        mon_node.status = "failed"

    # Startup cost estimate
    cost_node_parent = evaluator.add_parallel(
        id="Startup_Cost_Estimate",
        desc="Estimated total startup cost range for opening a venue with 15–20 gaming stations.",
        parent=ind_node,
        critical=True
    )

    cost_node = evaluator.add_leaf(
        id="Startup_Cost_Range",
        desc="Provide an estimated total startup cost range specifically for a 15–20 station venue.",
        parent=cost_node_parent,
        critical=True
    )
    if cost.cost_range:
        cost_sources = combine_sources(cost.cost_sources)
        claim = f"The estimated total startup cost range for a 15–20 station esports gaming lounge is: {cost.cost_range}."
        await evaluator.verify(
            claim=claim,
            node=cost_node,
            sources=cost_sources,
            additional_instruction="Confirm that the cost range explicitly refers to a venue with about 15–20 gaming stations. Allow equivalent units (e.g., $200k–$350k)."
        )
    else:
        cost_node.score = 0.0
        cost_node.status = "failed"


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
    Evaluate an answer for the esports gaming lounge benchmarking and standards task.
    Note on criticality adjustments:
    To satisfy framework constraints (critical parent cannot have non-critical children), high-level containers are set non-critical, while specific requirement leaves are marked critical where appropriate.
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

    # Extract in parallel
    venues_task = evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues"
    )
    specs_task = evaluator.extract(
        prompt=prompt_extract_pc_specs(),
        template_class=PCSpecs,
        extraction_name="pc_specs"
    )
    cost_task = evaluator.extract(
        prompt=prompt_extract_startup_cost(),
        template_class=StartupCost,
        extraction_name="startup_cost"
    )

    venues_extract, specs_extract, cost_extract = await asyncio.gather(venues_task, specs_task, cost_task)

    # Build top-level research task node (non-critical container)
    research_node = evaluator.add_parallel(
        id="Research_Task",
        desc="Benchmark required gaming lounges and provide PC standards + startup cost estimate.",
        parent=root,
        critical=False
    )

    # Citations node: custom check that all provided factual claims have URL(s)
    all_citations_ok = check_citations_presence(venues_extract.venues, specs_extract, cost_extract)
    evaluator.add_custom_node(
        result=all_citations_ok,
        id="Citations",
        desc="All provided factual claims include publicly accessible supporting URL reference(s).",
        parent=research_node,
        critical=True
    )

    # Gaming lounges research
    lounges_node = evaluator.add_parallel(
        id="Gaming_Lounges_Research",
        desc="Provide venue details for the required set of gaming lounges and satisfy geographic constraints.",
        parent=research_node,
        critical=False
    )

    # Select first 3 unique venues (preserving order); pad if fewer than 3
    selected_venues = dedupe_and_select_venues(venues_extract.venues, k=3)

    # Venue #1-#3 verification
    for i, v in enumerate(selected_venues[:3]):
        await verify_single_venue(evaluator, lounges_node, v, i)

    # Geographic and count constraints
    geo_node = evaluator.add_parallel(
        id="Geographic_And_Count_Constraints",
        desc="Cross-venue count and geographic distribution requirements.",
        parent=lounges_node,
        critical=True
    )
    geo_results = compute_geo_constraints(selected_venues[:3], all_venues_count=len(venues_extract.venues))

    evaluator.add_custom_node(
        result=geo_results["Three_Venues_Total"],
        id="Three_Venues_Total",
        desc="Answer includes exactly three gaming lounges.",
        parent=geo_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=geo_results["Three_Different_US_Cities"],
        id="Three_Different_US_Cities",
        desc="The three venues are each in a different US city.",
        parent=geo_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=geo_results["At_Least_One_Florida"],
        id="At_Least_One_Florida",
        desc="At least one venue is located in Florida.",
        parent=geo_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=geo_results["At_Least_One_Illinois_or_Midwest"],
        id="At_Least_One_Illinois_or_Midwest",
        desc="At least one venue is located in Illinois or another Midwest state.",
        parent=geo_node,
        critical=True
    )

    # Industry standards and costs
    await verify_pc_specs_and_cost(evaluator, research_node, specs_extract, cost_extract)

    # Add custom info context for debugging/traceability
    evaluator.add_custom_info(
        info={
            "selected_venues": [
                {
                    "name": v.name,
                    "city": v.city,
                    "state": v.state,
                    "address": v.address,
                    "contact": v.contact,
                    "stations": v.stations,
                    "pricing_options": v.pricing_options,
                    "amenities": v.amenities,
                    "weekly_hours": v.weekly_hours
                }
                for v in selected_venues[:3]
            ],
            "geo_results": geo_results,
            "midwest_states_used": sorted(list(MIDWEST_STATES)),
            "citations_ok": all_citations_ok
        },
        info_type="debug_info",
        info_name="evaluation_debug_info"
    )

    return evaluator.get_summary()