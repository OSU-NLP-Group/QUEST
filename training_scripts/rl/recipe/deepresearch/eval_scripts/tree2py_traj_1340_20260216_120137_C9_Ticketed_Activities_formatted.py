import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "venues_by_region_accessibility_pricing_ticketing"
TASK_DESCRIPTION = """Identify 4 concert venues across 4 different US geographic regions (Northeast, South, Midwest, and West Coast) where each venue must meet ALL of the following requirements:

1. Location: One venue must be in a Northeast state (ME, NH, VT, MA, RI, CT, NY, NJ, PA); one in a Southern state (TX, OK, AR, LA, MS, AL, TN, KY, FL, GA, SC, NC, VA, WV, MD, DE, DC); one in a Midwest state (OH, MI, IN, IL, WI, MN, IA, MO, ND, SD, NE, KS); and one in a Western state (WA, OR, CA, NV, AZ, UT, ID, MT, WY, CO, NM).

2. Capacity: Each venue must have a seating capacity between 15,000 and 25,000.

3. Venue Type: Each venue must be classified as either an arena or an amphitheater.

4. Accessibility: Each venue must offer ADA-compliant accessible seating for mobility-impaired guests, and these accessible seats must be sold through the same ticketing channels and at the same time as regular tickets.

5. Pricing Structure: Each venue must offer at least 3 distinct ticket price tiers for standard events and must provide VIP or premium ticket packages.

6. Ticketing Platform: Each venue must sell tickets through one of the following major platforms: Ticketmaster, Live Nation, AXS, or SeatGeek.

For each venue, provide:
- The venue name
- The city and state location
- The seating capacity
- The venue type (arena or amphitheater)
- Confirmation of accessible seating availability and policy
- Evidence of multi-tier pricing and VIP packages
- The primary ticketing platform used
- Reference URLs for each verified piece of information
"""


# --------------------------------------------------------------------------- #
# Region and platform constants                                               #
# --------------------------------------------------------------------------- #
NORTHEAST_STATES = {"ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"}
SOUTH_STATES = {"TX", "OK", "AR", "LA", "MS", "AL", "TN", "KY", "FL", "GA", "SC", "NC", "VA", "WV", "MD", "DE", "DC"}
MIDWEST_STATES = {"OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"}
WEST_STATES = {"WA", "OR", "CA", "NV", "AZ", "UT", "ID", "MT", "WY", "CO", "NM"}

ALLOWED_PLATFORMS = {"Ticketmaster", "Live Nation", "AXS", "SeatGeek"}

REGION_CONFIG = {
    "Northeast": {
        "states": NORTHEAST_STATES,
        "region_node_id": "Venue_1_Northeast",
        "prefix": "Northeast",
        "desc": "Identify a concert venue in the Northeast US region meeting all specified criteria",
    },
    "South": {
        "states": SOUTH_STATES,
        "region_node_id": "Venue_2_South",
        "prefix": "South",
        "desc": "Identify a concert venue in the South US region meeting all specified criteria",
    },
    "Midwest": {
        "states": MIDWEST_STATES,
        "region_node_id": "Venue_3_Midwest",
        "prefix": "Midwest",
        "desc": "Identify a concert venue in the Midwest US region meeting all specified criteria",
    },
    "West": {
        "states": WEST_STATES,
        "region_node_id": "Venue_4_West",
        "prefix": "West",
        "desc": "Identify a concert venue in the West Coast US region meeting all specified criteria",
    },
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Use two-letter postal abbreviation if possible
    capacity_text: Optional[str] = None  # Keep as free text (e.g., "18,200", "approx. 20,000", "18,000-20,000")
    venue_type: Optional[str] = None  # e.g., "arena", "amphitheater" (allow variants/spellings)
    ticketing_platform: Optional[str] = None  # e.g., "Ticketmaster", "Live Nation", "AXS", "SeatGeek"

    # Source URLs per aspect
    location_sources: List[str] = Field(default_factory=list)
    capacity_sources: List[str] = Field(default_factory=list)
    type_sources: List[str] = Field(default_factory=list)

    accessibility_sources: List[str] = Field(default_factory=list)  # general/accessibility info
    accessibility_policy_sources: List[str] = Field(default_factory=list)  # policy about same channel/time

    pricing_sources: List[str] = Field(default_factory=list)  # for multi-tier standard pricing
    vip_sources: List[str] = Field(default_factory=list)  # for VIP/premium packages

    platform_sources: List[str] = Field(default_factory=list)  # for ticketing platform confirmation


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract all concert venues mentioned in the answer. For each venue, return the following fields:

- name: venue name as written in the answer
- city: city name
- state: two-letter US state code if provided (e.g., NY, CA). If not provided, try to derive from the answer; else null.
- capacity_text: the seating capacity text as written (e.g., "18,200", "about 20,000", "18,000–20,000")
- venue_type: the classification as written (e.g., "arena", "amphitheater"; allow variants like "amphitheatre", "pavilion (amphitheater)")
- ticketing_platform: the primary ticketing platform name if provided (Ticketmaster, Live Nation, AXS, or SeatGeek)

Also extract URL sources associated with each of the following aspects (ONLY URLs explicitly present in the answer):
- location_sources: URLs that support the city/state location
- capacity_sources: URLs that state or imply seating capacity
- type_sources: URLs that describe the venue type (arena/amphitheater)
- accessibility_sources: URLs that confirm accessible/ADA seating is available
- accessibility_policy_sources: URLs that state that accessible seats are sold via the same channels and at the same time as regular tickets
- pricing_sources: URLs that show 3+ distinct standard ticket price tiers for events
- vip_sources: URLs that show VIP or premium ticket packages
- platform_sources: URLs that confirm the ticketing platform used

Return a JSON object with a 'venues' array. If a field or URL list is not present in the answer, set it to null (for strings) or an empty list (for URL arrays). Do not fabricate any URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if s and (s.startswith("http://") or s.startswith("https://")):
            cleaned.append(s)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state or not isinstance(state, str):
        return None
    return state.strip().upper()


def normalize_platform(platform: Optional[str]) -> Optional[str]:
    if not platform:
        return None
    p = platform.strip()
    # Normalize some common variants
    if p.lower().replace(" ", "") in {"livenation", "live-nation"}:
        return "Live Nation"
    if p.lower() == "tm":
        return "Ticketmaster"
    if p.lower() == "axs":
        return "AXS"
    if p.lower().replace(" ", "") in {"seatgeek", "seat-geek"}:
        return "SeatGeek"
    # Title-case typical names
    for std in ALLOWED_PLATFORMS:
        if p.lower() == std.lower():
            return std
    return p


def classify_region_by_state(state: Optional[str]) -> Optional[str]:
    st = normalize_state(state)
    if not st:
        return None
    if st in NORTHEAST_STATES:
        return "Northeast"
    if st in SOUTH_STATES:
        return "South"
    if st in MIDWEST_STATES:
        return "Midwest"
    if st in WEST_STATES:
        return "West"
    return None


def pick_venues_by_region(venues: List[VenueItem]) -> Dict[str, VenueItem]:
    picked: Dict[str, VenueItem] = {}
    for v in venues:
        # Normalize fields
        v.state = normalize_state(v.state)
        v.ticketing_platform = normalize_platform(v.ticketing_platform)
        v.location_sources = clean_urls(v.location_sources)
        v.capacity_sources = clean_urls(v.capacity_sources)
        v.type_sources = clean_urls(v.type_sources)
        v.accessibility_sources = clean_urls(v.accessibility_sources)
        v.accessibility_policy_sources = clean_urls(v.accessibility_policy_sources)
        v.pricing_sources = clean_urls(v.pricing_sources)
        v.vip_sources = clean_urls(v.vip_sources)
        v.platform_sources = clean_urls(v.platform_sources)

        region = classify_region_by_state(v.state)
        if region and region not in picked:
            picked[region] = v
        if len(picked) == 4:
            break
    # For missing regions, create placeholders
    for region in ["Northeast", "South", "Midwest", "West"]:
        if region not in picked:
            picked[region] = VenueItem()
    return picked


def safe_name(venue: VenueItem) -> str:
    return venue.name or "the venue"


def safe_city_state(venue: VenueItem) -> Tuple[str, str]:
    return (venue.city or "", venue.state or "")


def get_policy_sources(venue: VenueItem) -> List[str]:
    # Prefer dedicated policy sources; if not available, fall back to accessibility sources
    sources = venue.accessibility_policy_sources or []
    if not sources:
        sources = venue.accessibility_sources or []
    return clean_urls(sources)


def get_vip_sources(venue: VenueItem) -> List[str]:
    # Prefer dedicated VIP sources; if not available, fall back to pricing sources
    sources = venue.vip_sources or []
    if not sources:
        sources = venue.pricing_sources or []
    return clean_urls(sources)


# --------------------------------------------------------------------------- #
# Verification per region                                                     #
# --------------------------------------------------------------------------- #
async def verify_region_venue(
    evaluator: Evaluator,
    root_node,
    region_label: str,
    config: Dict[str, Any],
    venue: VenueItem
) -> None:
    """
    Build the verification subtree for a given region and venue using IDs from the rubric.
    """
    prefix = config["prefix"]
    region_node_id = config["region_node_id"]
    allowed_states = config["states"]

    # Region parent node (parallel, non-critical)
    region_node = evaluator.add_parallel(
        id=region_node_id,
        desc=config["desc"],
        parent=root_node,
        critical=False
    )

    # --------------------------- Identification --------------------------- #
    ident_node = evaluator.add_parallel(
        id=f"{prefix}_Venue_Identification",
        desc="Venue must be properly identified with location, capacity, and type",
        parent=region_node,
        critical=True
    )

    # Location reference presence (critical)
    loc_sources = venue.location_sources
    loc_ref_node = evaluator.add_custom_node(
        result=len(loc_sources) > 0,
        id=f"{prefix}_Location_Reference",
        desc="Provide URL reference verifying the venue's location",
        parent=ident_node,
        critical=True
    )

    # Location verification (critical) - verify city/state via sources and check region membership by simple logic
    loc_city, loc_state = safe_city_state(venue)
    loc_claim = (
        f"The venue {safe_name(venue)} is located in {loc_city}, {loc_state}. "
        f"Also check that the state code '{loc_state}' is among the {region_label} set: "
        f"{', '.join(sorted(allowed_states))}."
    )
    loc_node = evaluator.add_leaf(
        id=f"{prefix}_Location",
        desc=f"Venue must be located in a {region_label} US state ({', '.join(sorted(allowed_states))})",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=loc_sources,
        additional_instruction=(
            "First, use the provided URLs to confirm the venue's city/state. "
            "Then, treat the check that the state code is in the provided allowed list as a simple logical check; "
            "the webpage does not need to mention the region name."
        ),
        extra_prerequisites=[loc_ref_node]
    )

    # Capacity reference presence (critical)
    cap_sources = venue.capacity_sources
    cap_ref_node = evaluator.add_custom_node(
        result=len(cap_sources) > 0,
        id=f"{prefix}_Capacity_Reference",
        desc="Provide URL reference verifying the venue's seating capacity",
        parent=ident_node,
        critical=True
    )

    # Capacity verification (critical)
    cap_node = evaluator.add_leaf(
        id=f"{prefix}_Capacity",
        desc="Venue must have a seating capacity between 15,000 and 25,000",
        parent=ident_node,
        critical=True
    )
    cap_claim = (
        f"The seating capacity of {safe_name(venue)} is between 15,000 and 25,000 inclusive."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=cap_sources,
        additional_instruction=(
            "Use the provided URLs to find the venue's seating capacity. "
            "Accept reasonable representations (single number, approximate like 'about 20,000', or a range like '18,000–20,000'). "
            "Consider the requirement satisfied if the figure or reasonable typical capacity falls within 15,000–25,000."
        ),
        extra_prerequisites=[cap_ref_node]
    )

    # Type reference presence (critical)
    type_sources = venue.type_sources
    type_ref_node = evaluator.add_custom_node(
        result=len(type_sources) > 0,
        id=f"{prefix}_Type_Reference",
        desc="Provide URL reference verifying the venue type",
        parent=ident_node,
        critical=True
    )

    # Type verification (critical)
    venue_type_text = (venue.venue_type or "").strip()
    type_node = evaluator.add_leaf(
        id=f"{prefix}_Type",
        desc="Venue must be classified as an arena or amphitheater",
        parent=ident_node,
        critical=True
    )
    type_claim = (
        f"The venue {safe_name(venue)} is an '{venue_type_text}' and this classification corresponds to either "
        f"'arena' or 'amphitheater' (allow minor spelling variants like 'amphitheatre' and synonyms like 'pavilion' for amphitheater)."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=type_sources,
        additional_instruction=(
            "Verify that the referenced page/classification clearly supports the venue being an arena or an amphitheater. "
            "Allow reasonable synonyms/variants (e.g., 'amphitheatre', 'pavilion' for amphitheater)."
        ),
        extra_prerequisites=[type_ref_node]
    )

    # --------------------------- Accessibility --------------------------- #
    acc_node = evaluator.add_parallel(
        id=f"{prefix}_Accessibility",
        desc="Venue must offer ADA-compliant accessible seating options",
        parent=region_node,
        critical=True
    )

    # Accessible availability reference presence
    acc_sources = venue.accessibility_sources
    acc_ref_node = evaluator.add_custom_node(
        result=len(acc_sources) > 0,
        id=f"{prefix}_Accessible_Reference",
        desc="Provide URL reference confirming accessible seating availability",
        parent=acc_node,
        critical=True
    )

    # Accessible availability verification
    acc_avail_node = evaluator.add_leaf(
        id=f"{prefix}_Accessible_Available",
        desc="Venue must explicitly offer accessible seating for mobility-impaired guests",
        parent=acc_node,
        critical=True
    )
    acc_avail_claim = (
        f"{safe_name(venue)} offers ADA-compliant accessible seating for mobility-impaired guests."
    )
    await evaluator.verify(
        claim=acc_avail_claim,
        node=acc_avail_node,
        sources=acc_sources,
        additional_instruction=(
            "Look for mentions such as 'accessible seating', 'ADA seating', or instructions for mobility-impaired guests."
        ),
        extra_prerequisites=[acc_ref_node]
    )

    # Accessible policy reference presence
    policy_sources = get_policy_sources(venue)
    policy_ref_node = evaluator.add_custom_node(
        result=len(policy_sources) > 0,
        id=f"{prefix}_Accessible_Policy_Reference",
        desc="Provide URL reference for accessible ticketing policy",
        parent=acc_node,
        critical=True
    )

    # Accessible policy verification
    acc_policy_node = evaluator.add_leaf(
        id=f"{prefix}_Accessible_Policy",
        desc="Venue must sell accessible seats through the same channels and at the same time as regular tickets",
        parent=acc_node,
        critical=True
    )
    acc_policy_claim = (
        f"Accessible seats at {safe_name(venue)} are sold through the same ticketing channels and at the same time "
        f"as regular tickets."
    )
    await evaluator.verify(
        claim=acc_policy_claim,
        node=acc_policy_node,
        sources=policy_sources,
        additional_instruction=(
            "Confirm language like 'Accessible seating is available through the same ticketing platform/box office at the same on-sale time as general tickets.' "
            "Allow phrasing that clearly implies parity of sales channel and timing."
        ),
        extra_prerequisites=[policy_ref_node]
    )

    # --------------------------- Pricing --------------------------------- #
    pricing_node = evaluator.add_parallel(
        id=f"{prefix}_Pricing",
        desc="Venue must offer multi-tier ticket pricing for events",
        parent=region_node,
        critical=True
    )

    # Pricing tiers reference presence
    pricing_sources = venue.pricing_sources
    pricing_ref_node = evaluator.add_custom_node(
        result=len(pricing_sources) > 0,
        id=f"{prefix}_Pricing_Reference",
        desc="Provide URL reference showing ticket pricing tier structure",
        parent=pricing_node,
        critical=True
    )

    # Multiple tiers verification
    tiers_node = evaluator.add_leaf(
        id=f"{prefix}_Multiple_Tiers",
        desc="Venue must offer at least 3 distinct ticket price tiers for standard events",
        parent=pricing_node,
        critical=True
    )
    tiers_claim = (
        f"{safe_name(venue)} offers at least three distinct ticket price tiers for standard events."
    )
    await evaluator.verify(
        claim=tiers_claim,
        node=tiers_node,
        sources=pricing_sources,
        additional_instruction=(
            "Look for at least three different listed prices or categories (e.g., floor, lower bowl, upper bowl; "
            "or GA, reserved, premium). Dynamic pricing is acceptable if clearly showing at least three distinct prices."
        ),
        extra_prerequisites=[pricing_ref_node]
    )

    # VIP reference presence
    vip_sources = get_vip_sources(venue)
    vip_ref_node = evaluator.add_custom_node(
        result=len(vip_sources) > 0,
        id=f"{prefix}_VIP_Reference",
        desc="Provide URL reference for VIP package availability",
        parent=pricing_node,
        critical=True
    )

    # VIP availability verification
    vip_node = evaluator.add_leaf(
        id=f"{prefix}_VIP_Available",
        desc="Venue must offer VIP or premium ticket packages",
        parent=pricing_node,
        critical=True
    )
    vip_claim = (
        f"{safe_name(venue)} offers VIP or premium ticket packages (e.g., VIP seats, suites, boxes, or club packages)."
    )
    await evaluator.verify(
        claim=vip_claim,
        node=vip_node,
        sources=vip_sources,
        additional_instruction=(
            "Accept terms like 'VIP packages', 'premium seating', 'club packages', 'suites', or similar premium options."
        ),
        extra_prerequisites=[vip_ref_node]
    )

    # --------------------------- Ticketing -------------------------------- #
    ticketing_node = evaluator.add_parallel(
        id=f"{prefix}_Ticketing",
        desc="Venue must use a major verified ticketing platform",
        parent=region_node,
        critical=True
    )

    # Ticketing platform reference presence
    platform_sources = venue.platform_sources
    platform_ref_node = evaluator.add_custom_node(
        result=len(platform_sources) > 0,
        id=f"{prefix}_Platform_Reference",
        desc="Provide URL reference showing the venue's primary ticketing platform",
        parent=ticketing_node,
        critical=True
    )

    # Platform verification
    platform_node = evaluator.add_leaf(
        id=f"{prefix}_Platform",
        desc="Venue must sell tickets through Ticketmaster, Live Nation, AXS, or SeatGeek",
        parent=ticketing_node,
        critical=True
    )
    platform_text = normalize_platform(venue.ticketing_platform) or ""
    platform_claim = (
        f"{safe_name(venue)} sells tickets through '{platform_text}', and this is one of the following major platforms: "
        f"{', '.join(sorted(ALLOWED_PLATFORMS))}."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_node,
        sources=platform_sources,
        additional_instruction=(
            "Confirm that the referenced page shows tickets are sold via the named platform. "
            "Treat the membership check (whether it is among the allowed list) as simple logic."
        ),
        extra_prerequisites=[platform_ref_node]
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
    Evaluate an answer for the multi-constraint venue identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Four regions are independent
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
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Pick one venue per region (by state-based classification)
    picked_by_region = pick_venues_by_region(extracted.venues)

    # Ground truth / configuration info for transparency
    evaluator.add_ground_truth({
        "regions": {
            "Northeast": sorted(list(NORTHEAST_STATES)),
            "South": sorted(list(SOUTH_STATES)),
            "Midwest": sorted(list(MIDWEST_STATES)),
            "West": sorted(list(WEST_STATES)),
        },
        "allowed_ticketing_platforms": sorted(list(ALLOWED_PLATFORMS))
    }, gt_type="constraints")

    # Build and verify each region subtree
    tasks = []
    for region_name, cfg in REGION_CONFIG.items():
        venue = picked_by_region.get(region_name, VenueItem())
        tasks.append(
            verify_region_venue(evaluator, root, region_name, cfg, venue)
        )

    # Run verifications sequentially to keep logs ordered (or use asyncio.gather for parallel)
    for t in tasks:
        await t

    return evaluator.get_summary()