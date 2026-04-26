import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "chicago_venues_apr2026"
TASK_DESCRIPTION = """
I'm planning a concert-going trip to Chicago in April 2026 and need to find accessible venues that accommodate my accessibility needs. Please identify 3 concert venues in Chicago that meet ALL of the following requirements:

1. Capacity: The venue must have a seating capacity between 1,000 and 5,000 seats (medium-sized venue)
2. Accessibility: The venue must be ADA-compliant and provide wheelchair-accessible seating, with documented accessibility features such as accessible entrances, restrooms, and parking
3. April 2026 Concert: The venue must have at least one confirmed concert or musical performance scheduled in April 2026

For each of the 3 venues, please provide:
- Venue name and complete street address in Chicago
- Official website or reliable source URL for verification
- Total seating capacity (with source URL to verify)
- Confirmation of wheelchair-accessible seating availability (including number of accessible seats if available)
- Documentation of ADA-compliant features (with source URL for verification)
- Details of the April 2026 concert including specific date(s), performing artist/band name, and source URL for verification
"""


# ---------------------------- Data Models ---------------------------- #

class VenueBasic(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    website_url: Optional[str] = None


class VenueCapacity(BaseModel):
    capacity_text: Optional[str] = None  # e.g., "3,500", "approx. 2,000"
    capacity_source_urls: List[str] = Field(default_factory=list)


class VenueAccessibility(BaseModel):
    wheelchair_accessibility: Optional[str] = None  # e.g., "Wheelchair accessible seating available"
    wheelchair_seats: Optional[str] = None  # e.g., "12 wheelchair seats"
    ada_features: List[str] = Field(default_factory=list)  # e.g., ["accessible entrances", "accessible restrooms", "accessible parking"]
    accessibility_source_urls: List[str] = Field(default_factory=list)


class VenueEvent(BaseModel):
    event_dates: List[str] = Field(default_factory=list)  # date strings in April 2026
    event_artist: Optional[str] = None
    event_source_urls: List[str] = Field(default_factory=list)


class VenueItem(BaseModel):
    basic: VenueBasic = Field(default_factory=VenueBasic)
    capacity: VenueCapacity = Field(default_factory=VenueCapacity)
    accessibility: VenueAccessibility = Field(default_factory=VenueAccessibility)
    event: VenueEvent = Field(default_factory=VenueEvent)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# ------------------------- Extraction Prompt ------------------------- #

def prompt_extract_venues() -> str:
    return """
    Extract up to 3 concert venues in Chicago from the answer. Only include the first 3 if more are listed.
    For each venue, return a JSON object with the following fields grouped as shown:
    {
      "basic": {
        "name": string | null,
        "address": string | null,
        "website_url": string | null
      },
      "capacity": {
        "capacity_text": string | null,  // the stated total seating capacity exactly as written in the answer; keep units or commas if present
        "capacity_source_urls": string[] // URLs that specifically support/verify the capacity (can include official site pages, venue spec sheets, Wikipedia, etc.)
      },
      "accessibility": {
        "wheelchair_accessibility": string | null,  // explicit statement about wheelchair accessible seating availability, exactly as written
        "wheelchair_seats": string | null,         // number of accessible seats or a specific detail string if provided (null if not provided)
        "ada_features": string[],                   // list of documented features such as 'accessible entrances', 'accessible restrooms', 'accessible parking'
        "accessibility_source_urls": string[]       // URLs that document ADA/accessibility info (prioritize official sources)
      },
      "event": {
        "event_dates": string[],       // specific date(s) in April 2026 (e.g., "April 12, 2026"); return an empty array if none were provided
        "event_artist": string | null, // the performing artist or band for the April 2026 concert
        "event_source_urls": string[]  // URLs that verify the April 2026 concert (venue calendar page, ticketing, artist site, etc.)
      }
    }
    RULES:
    - Extract ONLY what appears in the answer text. Do not invent any information.
    - URLs can be plain, markdown links, or embedded; extract the actual URL strings.
    - If any field is missing in the answer, set it to null (for strings) or an empty array (for lists).
    - Ensure addresses are complete street addresses and belong to Chicago (as stated in the answer).
    - Keep capacity as a string; do not convert to a number.
    - Return the object as { "venues": [...] }.
    """


# ----------------------------- Helpers ------------------------------ #

def _coalesce_urls(*url_lists: List[str], fallback: Optional[str] = None) -> List[str]:
    """Merge multiple URL lists, deduplicate while preserving order. Optionally append fallback."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    if fallback and fallback not in seen and fallback:
        merged.append(fallback)
    return merged


def _fmt_dates(dates: List[str]) -> str:
    if not dates:
        return ""
    return "; ".join(dates)


# ------------------------ Verification Builder ----------------------- #

async def verify_one_venue(
    evaluator: Evaluator,
    root_node,
    venue: VenueItem,
    idx: int,
) -> None:
    """Build tree and run verifications for a single venue."""
    vnum = idx + 1
    v_basic = venue.basic
    v_cap = venue.capacity
    v_acc = venue.accessibility
    v_evt = venue.event

    # Venue container (parallel, non-critical as per rubric)
    venue_node = evaluator.add_parallel(
        id=f"Venue_{vnum}",
        desc=f"Venue #{vnum} meeting all requirements",
        parent=root_node,
        critical=False
    )

    # 1) Identification & Capacity (sequential, critical)
    idcap_node = evaluator.add_sequential(
        id=f"Venue_{vnum}_Identification_And_Capacity",
        desc="Venue identification and capacity verification",
        parent=venue_node,
        critical=True
    )

    # 1.1) Basic Info (parallel, critical)
    basic_info_node = evaluator.add_parallel(
        id=f"Venue_{vnum}_Basic_Info",
        desc="Essential venue information provided",
        parent=idcap_node,
        critical=True
    )

    # Name & Location existence (critical leaf via custom)
    evaluator.add_custom_node(
        result=bool(v_basic.name and v_basic.name.strip()) and bool(v_basic.address and v_basic.address.strip()),
        id=f"Venue_{vnum}_Name_Location",
        desc="Venue name and complete street address in Chicago provided",
        parent=basic_info_node,
        critical=True
    )

    # Website existence (critical leaf via custom)
    evaluator.add_custom_node(
        result=bool(v_basic.website_url and v_basic.website_url.strip()),
        id=f"Venue_{vnum}_Website",
        desc="Official website or source URL for verification provided",
        parent=basic_info_node,
        critical=True
    )

    # 1.2) Capacity requirements (parallel, critical)
    capacity_node = evaluator.add_parallel(
        id=f"Venue_{vnum}_Capacity_Requirements",
        desc="Capacity meets specified range",
        parent=idcap_node,
        critical=True
    )

    # Capacity stated (critical existence)
    evaluator.add_custom_node(
        result=bool(v_cap.capacity_text and v_cap.capacity_text.strip()),
        id=f"Venue_{vnum}_Capacity_Stated",
        desc="Total seating capacity number is stated",
        parent=capacity_node,
        critical=True
    )

    # Capacity in range (critical leaf, simple logical verification)
    cap_range_leaf = evaluator.add_leaf(
        id=f"Venue_{vnum}_Capacity_In_Range",
        desc="Capacity is between 1,000 and 5,000 seats",
        parent=capacity_node,
        critical=True
    )
    cap_range_claim = f"The stated capacity '{v_cap.capacity_text or ''}' indicates the venue capacity is between 1,000 and 5,000 seats."
    await evaluator.verify(
        claim=cap_range_claim,
        node=cap_range_leaf,
        additional_instruction="Interpret common forms like 'approx.', ranges, or commas in numbers. If the number or range clearly lies within [1000, 5000], consider it correct."
    )

    # Capacity verification via source (critical leaf, URL-grounded)
    cap_verify_leaf = evaluator.add_leaf(
        id=f"Venue_{vnum}_Capacity_Verification",
        desc="Source URL provided to verify capacity",
        parent=capacity_node,
        critical=True
    )
    cap_sources = _coalesce_urls(v_cap.capacity_source_urls, fallback=v_basic.website_url)
    cap_verify_claim = f"The total seating capacity of {v_basic.name or 'the venue'} is '{v_cap.capacity_text or ''}' as supported by the cited source(s)."
    await evaluator.verify(
        claim=cap_verify_claim,
        node=cap_verify_leaf,
        sources=cap_sources,
        additional_instruction="Check the source page explicitly for venue capacity or seating capacity. Accept minor variations (e.g., different formatting or rounding)."
    )

    # 2) Accessibility (parallel, critical)
    access_node = evaluator.add_parallel(
        id=f"Venue_{vnum}_Accessibility",
        desc="ADA-compliant accessibility features documented",
        parent=venue_node,
        critical=True
    )

    # 2.1) Wheelchair access (parallel, critical)
    wheelchair_node = evaluator.add_parallel(
        id=f"Venue_{vnum}_Wheelchair_Access",
        desc="Wheelchair seating availability and details",
        parent=access_node,
        critical=True
    )

    # Wheelchair confirmed (critical leaf, URL-grounded)
    wc_confirm_leaf = evaluator.add_leaf(
        id=f"Venue_{vnum}_Wheelchair_Confirmed",
        desc="Wheelchair-accessible seating is confirmed available",
        parent=wheelchair_node,
        critical=True
    )
    acc_sources = _coalesce_urls(v_acc.accessibility_source_urls, fallback=v_basic.website_url)
    wc_confirm_claim = f"Wheelchair-accessible seating is available at {v_basic.name or 'the venue'}."
    await evaluator.verify(
        claim=wc_confirm_claim,
        node=wc_confirm_leaf,
        sources=acc_sources,
        additional_instruction="Look for phrases like 'accessible seating', 'wheelchair seating', 'ADA seating', or similar policy statements on official venue pages or reliable sources."
    )

    # Wheelchair quantity/details (critical per engine constraints; presence/custom)
    evaluator.add_custom_node(
        result=bool(v_acc.wheelchair_seats and v_acc.wheelchair_seats.strip()) or bool(v_acc.wheelchair_accessibility and v_acc.wheelchair_accessibility.strip()),
        id=f"Venue_{vnum}_Wheelchair_Quantity",
        desc="Number of wheelchair seats or accessibility details provided",
        parent=wheelchair_node,
        critical=True
    )

    # 2.2) ADA features (parallel, critical)
    ada_node = evaluator.add_parallel(
        id=f"Venue_{vnum}_ADA_Features",
        desc="Additional ADA compliance documentation",
        parent=access_node,
        critical=True
    )

    # ADA documented features (critical leaf, URL-grounded)
    ada_doc_leaf = evaluator.add_leaf(
        id=f"Venue_{vnum}_ADA_Documented",
        desc="Specific ADA features documented (accessible entrances, restrooms, parking)",
        parent=ada_node,
        critical=True
    )
    ada_claim = f"{v_basic.name or 'The venue'} provides accessible entrances, accessible restrooms, and accessible parking."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_doc_leaf,
        sources=acc_sources,
        additional_instruction="Confirm the presence of all three: accessible entrances, accessible restrooms, and accessible parking. Accept reasonable synonyms (e.g., ADA-compliant parking, accessible lavatories)."
    )

    # ADA source provided (critical existence)
    evaluator.add_custom_node(
        result=len(v_acc.accessibility_source_urls) > 0,
        id=f"Venue_{vnum}_ADA_Source",
        desc="Source URL provided to verify accessibility information",
        parent=ada_node,
        critical=True
    )

    # 3) April 2026 event (sequential, critical)
    event_node = evaluator.add_sequential(
        id=f"Venue_{vnum}_April_2026_Event",
        desc="Scheduled concert in April 2026",
        parent=venue_node,
        critical=True
    )

    # Event exists (critical leaf, URL-grounded)
    evt_exists_leaf = evaluator.add_leaf(
        id=f"Venue_{vnum}_Event_Exists",
        desc="Concert event in April 2026 confirmed",
        parent=event_node,
        critical=True
    )
    evt_sources = _coalesce_urls(v_evt.event_source_urls, fallback=v_basic.website_url)
    evt_exists_claim = f"There is at least one confirmed concert or musical performance scheduled at {v_basic.name or 'the venue'} in April 2026."
    await evaluator.verify(
        claim=evt_exists_claim,
        node=evt_exists_leaf,
        sources=evt_sources,
        additional_instruction="Verify the venue's calendar or ticketing page indicates an event in April 2026. Any concert or musical performance qualifies."
    )

    # Event information (parallel, critical)
    evt_info_node = evaluator.add_parallel(
        id=f"Venue_{vnum}_Event_Information",
        desc="Complete event details provided",
        parent=event_node,
        critical=True
    )

    # Event date(s) (critical leaf, URL-grounded)
    evt_date_leaf = evaluator.add_leaf(
        id=f"Venue_{vnum}_Event_Date",
        desc="Specific date(s) in April 2026 provided",
        parent=evt_info_node,
        critical=True
    )
    date_str = _fmt_dates(v_evt.event_dates)
    evt_date_claim = f"The event date(s) are {date_str} and they take place in April 2026."
    await evaluator.verify(
        claim=evt_date_claim,
        node=evt_date_leaf,
        sources=evt_sources,
        additional_instruction="Check that the listed dates fall within April 2026, allowing for varying date formats."
    )

    # Event artist (critical leaf, URL-grounded)
    evt_artist_leaf = evaluator.add_leaf(
        id=f"Venue_{vnum}_Event_Artist",
        desc="Performing artist or band identified",
        parent=evt_info_node,
        critical=True
    )
    evt_artist_claim = f"The performing artist or band for the April 2026 concert is '{v_evt.event_artist or ''}'."
    await evaluator.verify(
        claim=evt_artist_claim,
        node=evt_artist_leaf,
        sources=evt_sources,
        additional_instruction="Confirm the artist/band name for the April 2026 event from the event listing, ticketing page, or artist site."
    )

    # Event source existence (critical existence)
    evaluator.add_custom_node(
        result=len(v_evt.event_source_urls) > 0,
        id=f"Venue_{vnum}_Event_Source",
        desc="Source URL to verify April 2026 concert",
        parent=evt_info_node,
        critical=True
    )


# -------------------------- Main Evaluation -------------------------- #

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
    Evaluate an answer for the Chicago April 2026 venue task and return a structured summary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # venues evaluated independently
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

    # Extract venues
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Take first 3 venues (pad with empty ones if fewer)
    venues: List[VenueItem] = list(extraction.venues[:3])
    while len(venues) < 3:
        venues.append(VenueItem())

    # Build verification tree for each venue
    root_node = evaluator.add_parallel(
        id="Root",
        desc="Find and verify 3 concert venues in Chicago that meet all specified requirements: medium capacity (1,000-5,000 seats), ADA-compliant accessibility features, and scheduled concerts in April 2026",
        parent=evaluator.root,
        critical=False
    )

    for i, v in enumerate(venues):
        await verify_one_venue(evaluator, root_node, v, i)

    return evaluator.get_summary()