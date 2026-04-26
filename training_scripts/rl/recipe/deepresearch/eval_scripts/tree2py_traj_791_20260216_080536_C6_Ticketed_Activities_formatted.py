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
TASK_ID = "us_venues_by_region_capacity"
TASK_DESCRIPTION = (
    "I am researching comedy and theater venues across the United States for potential tour planning. "
    "I need to identify four venues (one per region: Northeast, Midwest, South, West) that meet distinct seating "
    "capacity requirements and host comedy shows or live entertainment. For each venue, provide name, city/state, "
    "address, official capacity, website, capacity source URL, events page URL, and at least one official ticketing URL."
)

# Region state lists (full names)
NORTHEAST_STATES = [
    "Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island",
    "Vermont", "New York", "New Jersey", "Pennsylvania"
]
MIDWEST_STATES = [
    "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota", "Missouri",
    "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"
]
SOUTH_STATES = [
    "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Kentucky", "Louisiana",
    "Maryland", "Mississippi", "North Carolina", "Oklahoma", "South Carolina",
    "Tennessee", "Texas", "Virginia", "West Virginia"
]
WEST_STATES = [
    "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho", "Montana",
    "Nevada", "New Mexico", "Oregon", "Utah", "Washington", "Wyoming"
]

# Capacity ranges per region
CAPACITY_RANGES = {
    "northeast": (700, 1000),
    "midwest": (1200, 1600),
    "south": (1800, 2200),
    "west": (400, 600),
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    address: Optional[str] = None
    capacity: Optional[str] = None
    website_url: Optional[str] = None
    capacity_source_url: Optional[str] = None
    events_url: Optional[str] = None
    ticketing_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    northeast: Optional[VenueItem] = None
    midwest: Optional[VenueItem] = None
    south: Optional[VenueItem] = None
    west: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract exactly four venues from the answer, one for each U.S. region below, and fill the requested fields.

Regions and capacity ranges:
- northeast: states ∈ {Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont, New York, New Jersey, Pennsylvania}; capacity 700–1000 seats (inclusive)
- midwest: states ∈ {Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, Wisconsin}; capacity 1200–1600 seats (inclusive)
- south: states ∈ {Alabama, Arkansas, Delaware, Florida, Georgia, Kentucky, Louisiana, Maryland, Mississippi, North Carolina, Oklahoma, South Carolina, Tennessee, Texas, Virginia, West Virginia}; capacity 1800–2200 seats (inclusive)
- west: states ∈ {Alaska, Arizona, California, Colorado, Hawaii, Idaho, Montana, Nevada, New Mexico, Oregon, Utah, Washington, Wyoming}; capacity 400–600 seats (inclusive)

For each region, extract the first matching venue as presented in the answer. If the answer includes more than one candidate per region, select the first one for that region. If a region's venue is missing, set the entire object for that region to null.

For each region object (northeast, midwest, south, west), extract:
- name: Complete official venue name as stated in the answer.
- city: City of the venue.
- state: State of the venue (can be full name or 2-letter abbreviation as written).
- address: Complete street address (as provided).
- capacity: The official seating capacity string from the answer (keep the exact text, e.g., "about 950", "1,500", "approximately 2,000").
- website_url: The venue's official website URL.
- capacity_source_url: A URL (venue website or authoritative source) confirming capacity.
- events_url: A URL to the venue's events/schedule page demonstrating comedy or live entertainment.
- ticketing_urls: Array of 1 or more URLs to official ticket purchase options (e.g., venue box office page, Ticketmaster, official ticketing platform). If none are given, return an empty array.

Important rules:
- Extract only what appears in the answer. Do not invent fields or URLs.
- Normalize URLs (include http:// or https://). If malformed, set to null.
- If any field is missing for a given region's venue, set it to null (or empty array for ticketing_urls).
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def non_empty_list(items: List[Optional[str]]) -> List[str]:
    """Return unique, trimmed, non-empty items."""
    seen = set()
    out: List[str] = []
    for it in items:
        if it is None:
            continue
        val = it.strip()
        if not val:
            continue
        if val not in seen:
            seen.add(val)
            out.append(val)
    return out


def region_configurations() -> List[Dict[str, Any]]:
    return [
        {
            "key": "northeast",
            "node_id": "Venue_1",
            "description": "Identify a venue in the Northeastern United States (CT, ME, MA, NH, RI, VT, NY, NJ, PA) with capacity 700–1,000 that hosts comedy or live entertainment.",
            "allowed_states": NORTHEAST_STATES,
            "cap_range": CAPACITY_RANGES["northeast"],
        },
        {
            "key": "midwest",
            "node_id": "Venue_2",
            "description": "Identify a venue in the Midwestern United States (IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI) with capacity 1,200–1,600 that hosts comedy or live entertainment.",
            "allowed_states": MIDWEST_STATES,
            "cap_range": CAPACITY_RANGES["midwest"],
        },
        {
            "key": "south",
            "node_id": "Venue_3",
            "description": "Identify a venue in the Southern United States (AL, AR, DE, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV) with capacity 1,800–2,200 that hosts comedy or live entertainment.",
            "allowed_states": SOUTH_STATES,
            "cap_range": CAPACITY_RANGES["south"],
        },
        {
            "key": "west",
            "node_id": "Venue_4",
            "description": "Identify a venue in the Western United States (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY) with capacity 400–600 that hosts comedy or live entertainment.",
            "allowed_states": WEST_STATES,
            "cap_range": CAPACITY_RANGES["west"],
        },
    ]


# --------------------------------------------------------------------------- #
# Verification builder per venue/region                                       #
# --------------------------------------------------------------------------- #
async def verify_region_venue(
    evaluator: Evaluator,
    parent_node,
    region_key: str,
    node_id: str,
    region_desc: str,
    allowed_states: List[str],
    cap_range: tuple,
    venue: Optional[VenueItem],
):
    """
    Build and verify all rubric nodes for a region's venue.
    The structure mirrors the rubric tree provided.
    """
    # Add region node (non-critical to allow partial credit across regions)
    region_node = evaluator.add_parallel(
        id=node_id,
        desc=region_desc,
        parent=parent_node,
        critical=False,
    )

    # Safe accessors
    name = venue.name if venue else None
    city = venue.city if venue else None
    state = venue.state if venue else None
    address = venue.address if venue else None
    capacity_txt = venue.capacity if venue else None
    website_url = venue.website_url if venue else None
    capacity_url = venue.capacity_source_url if venue else None
    events_url = venue.events_url if venue else None
    ticket_urls = venue.ticketing_urls if (venue and venue.ticketing_urls) else []

    # ---------------- Identification ----------------
    ident_node = evaluator.add_parallel(
        id=f"{node_id}_Identification",
        desc=f"Provide the official name and confirm location in the specified {region_key} states.",
        parent=region_node,
        critical=True
    )

    # Name leaf
    name_leaf = evaluator.add_leaf(
        id=f"{node_id}_Name",
        desc="Provide the complete official name of the venue.",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official venue name is '{name}'. Allow minor formatting variations (e.g., 'Theatre' vs 'Theater', punctuation, or branding).",
        node=name_leaf,
        sources=website_url or capacity_url or events_url,
        additional_instruction="Verify the page shows the venue's official name or a very close variant."
    )

    # State leaf (membership in allowed set)
    states_list_str = ", ".join(allowed_states)
    state_leaf = evaluator.add_leaf(
        id=f"{node_id}_State",
        desc=f"Confirm the venue is located in one of the specified states for this region.",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided state '{state}' is one of the allowed states for the {region_key} region: [{states_list_str}].",
        node=state_leaf,
        additional_instruction="Treat the check as a logical membership test. Accept standard 2-letter abbreviations or full state names. Case-insensitive."
    )

    # City leaf (verify via website or official page)
    city_leaf = evaluator.add_leaf(
        id=f"{node_id}_City",
        desc="Provide the city where the venue is located.",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue is located in {city}, {state}.",
        node=city_leaf,
        sources=website_url or capacity_url or events_url,
        additional_instruction="Verify that the page shows the venue's city and state (e.g., on Contact, Visit, or footer section). Minor formatting variations acceptable."
    )

    # ---------------- Capacity Verification ----------------
    cap_node = evaluator.add_parallel(
        id=f"{node_id}_Capacity_Verification",
        desc=f"Verify the venue's seating capacity falls within the required range for {region_key}.",
        parent=region_node,
        critical=True
    )

    # Capacity value (grounded to capacity source)
    cap_value_leaf = evaluator.add_leaf(
        id=f"{node_id}_Capacity_Value",
        desc="Provide the official seating capacity of the venue.",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official seating capacity of '{name}' is '{capacity_txt}'.",
        node=cap_value_leaf,
        sources=capacity_url or website_url,
        additional_instruction="Verify that the page explicitly states (or clearly implies) the seating capacity for the venue. Allow minor wording variation and approximate phrasing (e.g., 'about', '~')."
    )

    # Capacity range check (logical check)
    cap_min, cap_max = cap_range
    cap_range_leaf = evaluator.add_leaf(
        id=f"{node_id}_Capacity_Range_Check",
        desc=f"Confirm the capacity is between {cap_min} and {cap_max} seats (inclusive).",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The seating capacity described as '{capacity_txt}' is between {cap_min} and {cap_max} (inclusive).",
        node=cap_range_leaf,
        additional_instruction="Consider reasonable interpretations (e.g., 'about 950' is 950). If a range is given (e.g., 900–1000), it's acceptable if any reasonable single capacity for the venue lies within bounds."
    )

    # Capacity source present & supportive
    cap_source_leaf = evaluator.add_leaf(
        id=f"{node_id}_Capacity_Source",
        desc="Provide a URL to the official/authoritative source confirming seating capacity.",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page confirms the seating capacity for '{name}'.",
        node=cap_source_leaf,
        sources=capacity_url or website_url,
        additional_instruction="Accept the venue's official website or an authoritative source (e.g., municipal/civic page, venue technical specs PDF). The page should include or strongly imply a capacity figure."
    )

    # ---------------- Entertainment Type ----------------
    ent_node = evaluator.add_parallel(
        id=f"{node_id}_Entertainment_Type",
        desc="Confirm the venue hosts comedy shows or live entertainment.",
        parent=region_node,
        critical=True
    )

    ent_type_leaf = evaluator.add_leaf(
        id=f"{node_id}_Event_Type_Confirmation",
        desc="Verify the venue regularly hosts comedy shows, theatrical performances, or live entertainment.",
        parent=ent_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows that the venue hosts comedy shows and/or live entertainment (comedy, theatre, concerts, or similar).",
        node=ent_type_leaf,
        sources=events_url or website_url,
        additional_instruction="Look for evidence such as 'Events', 'Calendar', 'Comedy', 'Shows', 'Performances', 'Live entertainment'."
    )

    ent_src_leaf = evaluator.add_leaf(
        id=f"{node_id}_Event_Source",
        desc="Provide a URL to the venue's official events page or schedule showing comedy or live entertainment.",
        parent=ent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This URL is the venue's official events/calendar page for '{name}'.",
        node=ent_src_leaf,
        sources=events_url,
        additional_instruction="The page should be on the venue's official domain or a dedicated official subdomain and clearly present upcoming/past events or a schedule."
    )

    # ---------------- Official Information ----------------
    info_node = evaluator.add_parallel(
        id=f"{node_id}_Official_Information",
        desc="Provide official website, complete address, and at least one official ticketing method.",
        parent=region_node,
        critical=True
    )

    # Website
    website_leaf = evaluator.add_leaf(
        id=f"{node_id}_Website",
        desc="Provide the URL of the venue's official website.",
        parent=info_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This is the official website for '{name}'.",
        node=website_leaf,
        sources=website_url,
        additional_instruction="Verify that the domain and page branding indicate it is the official site for the venue (not a third-party listing or ticketing marketplace)."
    )

    # Address
    address_leaf = evaluator.add_leaf(
        id=f"{node_id}_Address",
        desc="Provide the complete street address of the venue.",
        parent=info_node,
        critical=True
    )
    addr_sources = non_empty_list([website_url, capacity_url, events_url])
    await evaluator.verify(
        claim=f"The complete address of '{name}' is '{address}'.",
        node=address_leaf,
        sources=addr_sources,
        additional_instruction="Verify the address appears on the page. Minor differences in abbreviations (e.g., St. vs Street), punctuation, or formatting are acceptable."
    )

    # Ticketing
    ticket_leaf = evaluator.add_leaf(
        id=f"{node_id}_Ticketing",
        desc="Identify at least one official ticket purchase method (e.g., box office website, Ticketmaster, official vendor).",
        parent=info_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page provides an official way to purchase tickets for events at '{name}'.",
        node=ticket_leaf,
        sources=ticket_urls,
        additional_instruction="Accept official venue ticketing pages, Ticketmaster/AXS/Etix/Eventbrite venue-specific pages, or clearly official ticket purchase pages. Reject resale/secondary marketplaces without official endorsement."
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
    Evaluate an answer for the 4-region U.S. venue identification and verification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    # Note: Root is set to non-critical to allow partial credit across regions.
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Identify four venues across four U.S. regions with specified capacity ranges and official info "
            "(name, location, address, capacity with source, official website, events page, ticketing)."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract region-specific venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Record ground truth constraints for transparency
    evaluator.add_ground_truth({
        "region_constraints": {
            "northeast": {"states": NORTHEAST_STATES, "capacity_range": CAPACITY_RANGES["northeast"]},
            "midwest": {"states": MIDWEST_STATES, "capacity_range": CAPACITY_RANGES["midwest"]},
            "south": {"states": SOUTH_STATES, "capacity_range": CAPACITY_RANGES["south"]},
            "west": {"states": WEST_STATES, "capacity_range": CAPACITY_RANGES["west"]},
        }
    })

    # Build and verify for each region
    cfgs = region_configurations()
    for cfg in cfgs:
        region_key = cfg["key"]
        node_id = cfg["node_id"]
        region_desc = cfg["description"]
        allowed_states = cfg["allowed_states"]
        cap_range = cfg["cap_range"]

        venue_obj: Optional[VenueItem] = getattr(extracted, region_key)

        await verify_region_venue(
            evaluator=evaluator,
            parent_node=root,
            region_key=region_key,
            node_id=node_id,
            region_desc=region_desc,
            allowed_states=allowed_states,
            cap_range=cap_range,
            venue=venue_obj
        )

    # Return result summary
    return evaluator.get_summary()