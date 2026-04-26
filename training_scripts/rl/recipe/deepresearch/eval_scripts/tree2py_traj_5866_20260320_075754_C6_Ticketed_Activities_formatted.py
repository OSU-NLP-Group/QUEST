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
TASK_ID = "la_accessible_venues"
TASK_DESCRIPTION = """
Identify three distinct concert or comedy venues in Los Angeles, California that meet the following criteria:

1. Each venue must have a seating capacity between 1,000 and 20,000.
2. Each venue must regularly host live music concerts or comedy shows.
3. Each venue must provide wheelchair-accessible seating with comparable lines of sight to other spectators (ADA-compliant).
4. Each venue must allow the purchase of up to 3 companion seats adjacent to wheelchair spaces per ADA requirements.
5. Each venue must have accessible parking spaces meeting ADA standards.
6. Each venue must provide accessible restroom facilities.
7. At least one of the three venues must be accessible via public transportation (Metro or bus service).

For each venue, provide:
- Full official venue name
- Complete street address
- Seating capacity (specific number)
- Official website URL
- Contact information for accessibility services (phone number, email, or direct accessibility services webpage URL)
- Reference URLs confirming: location, capacity, event type, wheelchair seating, companion seat policy, accessible parking, accessible restrooms, and accessibility contact information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    # Core info
    name: Optional[str] = None
    address: Optional[str] = None
    capacity_text: Optional[str] = None  # e.g., "18,000 seats"
    capacity_number: Optional[int] = None  # e.g., 18000 if explicitly provided
    website: Optional[str] = None

    # Accessibility contacts (any of these qualifies as "provided")
    accessibility_contact_phone: Optional[str] = None
    accessibility_contact_email: Optional[str] = None
    accessibility_page_url: Optional[str] = None

    # Reference URLs by aspect (must be explicitly cited in the answer)
    location_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    event_urls: List[str] = Field(default_factory=list)
    wheelchair_urls: List[str] = Field(default_factory=list)
    companion_urls: List[str] = Field(default_factory=list)
    parking_urls: List[str] = Field(default_factory=list)
    restroom_urls: List[str] = Field(default_factory=list)
    contact_urls: List[str] = Field(default_factory=list)
    transit_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to the first three distinct Los Angeles concert or comedy venues described in the answer. For each venue, return a JSON object with these fields:

    Core fields:
    - name: Full official venue name (string or null)
    - address: Complete street address including city and state (string or null)
    - capacity_text: The seating capacity as it appears in the answer (string or null), e.g., "18,000 seats"
    - capacity_number: A numeric integer seating capacity if explicitly provided (integer or null). If not given exactly as a number in the answer, use null.
    - website: The official venue website URL (string URL or null)

    Accessibility contacts (any subset may be present):
    - accessibility_contact_phone: A phone number for accessibility/ADA services if provided (string or null)
    - accessibility_contact_email: An email address for accessibility/ADA services if provided (string or null)
    - accessibility_page_url: A direct URL to the venue's accessibility/ADA services page if provided (string URL or null)

    Reference URL arrays (extract only URLs explicitly cited in the answer for each aspect):
    - location_urls: URLs confirming the venue is in Los Angeles, CA
    - capacity_urls: URLs with official/semi-official capacity information
    - event_urls: URLs showing the venue regularly hosts live music concerts or comedy shows (e.g., events calendar)
    - wheelchair_urls: URLs confirming wheelchair-accessible seating and, ideally, comparable lines of sight
    - companion_urls: URLs confirming that up to three companion seats adjacent to wheelchair spaces may be purchased (allow policies phrased as “one to three,” “up to 3,” or similar)
    - parking_urls: URLs confirming ADA-accessible parking availability
    - restroom_urls: URLs confirming accessible/ADA restrooms are available
    - contact_urls: URLs pointing to pages that display/accessibility contact information (could overlap with accessibility_page_url)
    - transit_urls: URLs confirming public transportation (LA Metro rail or bus) access to the venue

    Rules:
    - Return exactly the fields listed. If a field is missing in the answer, set it to null (for scalars) or [] (for arrays).
    - Only include URLs that are explicitly present in the answer (plain links or markdown links). Do not invent or infer URLs.
    - Normalize obviously malformed links (prepend http:// if no scheme).
    - Put the results in an object with a top-level key "venues": a list of up to 3 venue objects in the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(*url_lists: List[str]) -> List[str]:
    """Combine and deduplicate URL lists while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            u_stripped = u.strip()
            if not u_stripped:
                continue
            if u_stripped not in seen:
                seen.add(u_stripped)
                combined.append(u_stripped)
    return combined


def _has_any(values: List[Optional[str]]) -> bool:
    return any(v is not None and str(v).strip() != "" for v in values)


# --------------------------------------------------------------------------- #
# Venue verification                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int,
) -> None:
    """
    Build and verify the subtree for a single venue.
    """
    vid = f"V{idx + 1}"

    # Top-level node for this venue (non-critical to allow partial credit across venues)
    venue_node = evaluator.add_parallel(
        id=f"Venue_{idx + 1}",
        desc=f"{['First','Second','Third'][idx] if idx < 3 else f'Venue {idx+1}'} qualifying venue meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # -------------------- Venue Qualification (critical cluster) -------------------- #
    qual_node = evaluator.add_parallel(
        id=f"{vid}_Venue_Qualification",
        desc="Venue meets location, capacity, and event type requirements",
        parent=venue_node,
        critical=True
    )

    # Location check
    loc_parent = evaluator.add_parallel(
        id=f"{vid}_Location",
        desc="Venue is located in Los Angeles, California",
        parent=qual_node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id=f"{vid}_Location_Check",
        desc="Los Angeles location is supported by sources",
        parent=loc_parent,
        critical=True
    )
    loc_sources = _dedup_urls(venue.location_urls, [venue.website] if venue.website else [])
    loc_addr = venue.address or "the venue's listed address"
    await evaluator.verify(
        claim=f"The venue at address '{loc_addr}' is located in Los Angeles, California (Los Angeles, CA).",
        node=loc_leaf,
        sources=loc_sources,
        additional_instruction="Treat 'Los Angeles, CA', 'LA, CA', or neighborhoods within the City of Los Angeles (e.g., Hollywood, Downtown LA, Koreatown, etc.) as valid. The page should clearly indicate Los Angeles, California."
    )
    evaluator.add_custom_node(
        result=len(venue.location_urls) > 0,
        id=f"{vid}_Location_Reference",
        desc="Provide URL confirming Los Angeles location",
        parent=loc_parent,
        critical=True
    )

    # Capacity check
    cap_parent = evaluator.add_parallel(
        id=f"{vid}_Capacity",
        desc="Venue seating capacity is between 1,000 and 20,000",
        parent=qual_node,
        critical=True
    )
    cap_leaf = evaluator.add_leaf(
        id=f"{vid}_Capacity_Range",
        desc="Capacity is within required range and supported by sources",
        parent=cap_parent,
        critical=True
    )
    cap_sources = _dedup_urls(venue.capacity_urls, [venue.website] if venue.website else [])
    if venue.capacity_number is not None:
        cap_claim = f"The venue has a seating capacity of approximately {venue.capacity_number}, which lies between 1,000 and 20,000."
    elif venue.capacity_text:
        cap_claim = f"The venue's seating capacity is '{venue.capacity_text}', and this indicates a capacity between 1,000 and 20,000."
    else:
        cap_claim = "The venue's seating capacity is between 1,000 and 20,000 people."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=cap_sources,
        additional_instruction="Verify that the capacity figure or description on the provided page(s) falls in the inclusive range 1,000–20,000. Allow minor phrasing like 'about' or 'approximately'."
    )
    evaluator.add_custom_node(
        result=len(venue.capacity_urls) > 0,
        id=f"{vid}_Capacity_Reference",
        desc="Provide URL with official capacity information",
        parent=cap_parent,
        critical=True
    )

    # Event type check
    evt_parent = evaluator.add_parallel(
        id=f"{vid}_Event_Type",
        desc="Venue regularly hosts live music concerts or comedy shows",
        parent=qual_node,
        critical=True
    )
    evt_leaf = evaluator.add_leaf(
        id=f"{vid}_Event_Type_Check",
        desc="Event type is supported by sources",
        parent=evt_parent,
        critical=True
    )
    evt_sources = _dedup_urls(venue.event_urls, [venue.website] if venue.website else [])
    await evaluator.verify(
        claim="This venue regularly hosts live music concerts or comedy shows (e.g., recurring events, tours, or a consistent calendar of such events).",
        node=evt_leaf,
        sources=evt_sources,
        additional_instruction="Look for an events calendar, schedule, or past events history indicating concerts or comedy shows are a regular/recurring offering, not a one-off."
    )
    evaluator.add_custom_node(
        result=len(venue.event_urls) > 0,
        id=f"{vid}_Event_Type_Reference",
        desc="Provide URL showing concert/comedy event schedule",
        parent=evt_parent,
        critical=True
    )

    # -------------------- Accessibility Features (critical cluster) -------------------- #
    acc_node = evaluator.add_parallel(
        id=f"{vid}_Accessibility_Features",
        desc="Venue provides comprehensive accessibility features",
        parent=venue_node,
        critical=True
    )

    # Wheelchair seating with comparable sight lines
    wh_parent = evaluator.add_parallel(
        id=f"{vid}_Wheelchair_Seating",
        desc="Venue offers wheelchair-accessible seating with comparable lines of sight",
        parent=acc_node,
        critical=True
    )
    wh_leaf = evaluator.add_leaf(
        id=f"{vid}_Wheelchair_Seating_Check",
        desc="Wheelchair seating and comparable sight lines supported by sources",
        parent=wh_parent,
        critical=True
    )
    wh_sources = _dedup_urls(venue.wheelchair_urls, [venue.website] if venue.website else [])
    await evaluator.verify(
        claim="The venue provides wheelchair-accessible seating with comparable lines of sight to those of other spectators.",
        node=wh_leaf,
        sources=wh_sources,
        additional_instruction="Accept language indicating accessible seating is integrated with non-ADA seating and provides comparable or similar sight lines, consistent with ADA guidance."
    )
    evaluator.add_custom_node(
        result=len(venue.wheelchair_urls) > 0,
        id=f"{vid}_Wheelchair_Reference",
        desc="Provide URL confirming wheelchair seating availability",
        parent=wh_parent,
        critical=True
    )

    # Companion seats (up to 3 adjacent)
    comp_parent = evaluator.add_parallel(
        id=f"{vid}_Companion_Seats",
        desc="Venue allows purchase of up to 3 companion seats with accessible seating",
        parent=acc_node,
        critical=True
    )
    comp_leaf = evaluator.add_leaf(
        id=f"{vid}_Companion_Seats_Check",
        desc="Companion seats policy (up to 3 adjacent) supported by sources",
        parent=comp_parent,
        critical=True
    )
    comp_sources = _dedup_urls(venue.companion_urls, [venue.website] if venue.website else [])
    await evaluator.verify(
        claim="The venue allows purchasing up to three companion seats adjacent to wheelchair-accessible seating (subject to availability).",
        node=comp_leaf,
        sources=comp_sources,
        additional_instruction="Accept phrasing like 'one to three', 'up to 3', 'at least one companion with up to two additional seats', provided adjacency to accessible seating is stated or clearly implied."
    )
    evaluator.add_custom_node(
        result=len(venue.companion_urls) > 0,
        id=f"{vid}_Companion_Reference",
        desc="Provide URL confirming companion seat policy",
        parent=comp_parent,
        critical=True
    )

    # Accessible parking
    park_parent = evaluator.add_parallel(
        id=f"{vid}_Accessible_Parking",
        desc="Venue provides ADA-compliant accessible parking",
        parent=acc_node,
        critical=True
    )
    park_leaf = evaluator.add_leaf(
        id=f"{vid}_Accessible_Parking_Check",
        desc="Accessible parking supported by sources",
        parent=park_parent,
        critical=True
    )
    park_sources = _dedup_urls(venue.parking_urls, [venue.website] if venue.website else [])
    await evaluator.verify(
        claim="ADA-compliant accessible parking is available at or near the venue.",
        node=park_leaf,
        sources=park_sources,
        additional_instruction="Look for explicit mention of accessible/ADA parking spaces or instructions for ADA parking at/near the venue."
    )
    evaluator.add_custom_node(
        result=len(venue.parking_urls) > 0,
        id=f"{vid}_Parking_Reference",
        desc="Provide URL confirming accessible parking availability",
        parent=park_parent,
        critical=True
    )

    # Accessible restrooms
    rest_parent = evaluator.add_parallel(
        id=f"{vid}_Accessible_Restrooms",
        desc="Venue provides accessible restroom facilities",
        parent=acc_node,
        critical=True
    )
    rest_leaf = evaluator.add_leaf(
        id=f"{vid}_Accessible_Restrooms_Check",
        desc="Accessible restrooms supported by sources",
        parent=rest_parent,
        critical=True
    )
    rest_sources = _dedup_urls(venue.restroom_urls, [venue.website] if venue.website else [])
    await evaluator.verify(
        claim="Accessible (ADA-compliant) restroom facilities are available at the venue.",
        node=rest_leaf,
        sources=rest_sources,
        additional_instruction="Accept language such as 'accessible restrooms', 'ADA restrooms', 'restrooms with accessible stalls/amenities'."
    )
    evaluator.add_custom_node(
        result=len(venue.restroom_urls) > 0,
        id=f"{vid}_Restroom_Reference",
        desc="Provide URL confirming accessible restroom availability",
        parent=rest_parent,
        critical=True
    )

    # -------------------- Venue Information (critical cluster) -------------------- #
    info_node = evaluator.add_parallel(
        id=f"{vid}_Venue_Information",
        desc="Provide complete venue information including contact details",
        parent=venue_node,
        critical=True
    )

    # Name provided
    evaluator.add_custom_node(
        result=(venue.name is not None and str(venue.name).strip() != ""),
        id=f"{vid}_Name",
        desc="Provide full official venue name",
        parent=info_node,
        critical=True
    )

    # Address provided
    evaluator.add_custom_node(
        result=(venue.address is not None and str(venue.address).strip() != ""),
        id=f"{vid}_Address",
        desc="Provide complete street address",
        parent=info_node,
        critical=True
    )

    # Website provided
    evaluator.add_custom_node(
        result=(venue.website is not None and str(venue.website).strip() != ""),
        id=f"{vid}_Website",
        desc="Provide official venue website URL",
        parent=info_node,
        critical=True
    )

    # Accessibility contact: existence + reference verification
    ac_parent = evaluator.add_parallel(
        id=f"{vid}_Accessibility_Contact",
        desc="Provide accessibility services contact information or webpage",
        parent=info_node,
        critical=True
    )

    # Existence of at least one contact modality
    evaluator.add_custom_node(
        result=_has_any([
            venue.accessibility_contact_phone,
            venue.accessibility_contact_email,
            venue.accessibility_page_url
        ]),
        id=f"{vid}_Accessibility_Contact_Provided",
        desc="Accessibility services contact information is provided (phone/email/page URL)",
        parent=ac_parent,
        critical=True
    )

    # Reference verification for accessibility contact info/page
    contact_leaf = evaluator.add_leaf(
        id=f"{vid}_Contact_Reference",
        desc="Provide URL to accessibility services page or contact details",
        parent=ac_parent,
        critical=True
    )

    # Build claim and sources for contact verification
    contact_sources = _dedup_urls(
        venue.contact_urls,
        [venue.accessibility_page_url] if venue.accessibility_page_url else [],
        [venue.website] if venue.website else []
    )

    if venue.accessibility_page_url:
        contact_claim = "This page is the venue's official accessibility or ADA services page."
    elif venue.accessibility_contact_phone:
        contact_claim = f"The accessibility services contact phone number for the venue is '{venue.accessibility_contact_phone}'."
    elif venue.accessibility_contact_email:
        contact_claim = f"The accessibility services contact email for the venue is '{venue.accessibility_contact_email}'."
    else:
        contact_claim = "This page provides official accessibility services contact information for the venue (phone or email)."

    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=contact_sources,
        additional_instruction="Verify that the page explicitly provides ADA/accessibility contact details or is clearly labeled as an accessibility/ADA page of the venue."
    )


# --------------------------------------------------------------------------- #
# Public transportation requirement (global critical)                         #
# --------------------------------------------------------------------------- #
async def verify_public_transportation_requirement(
    evaluator: Evaluator,
    parent_node,
    venues: List[VenueItem]
) -> None:
    """
    At least one venue must be accessible via public transportation (Metro or bus).
    """
    transit_node = evaluator.add_parallel(
        id="Public_Transportation_Requirement",
        desc="At least one venue must be accessible via public transportation (Metro or bus)",
        parent=parent_node,
        critical=True
    )

    # Combine all transit URLs from the three venues
    all_transit_urls: List[str] = []
    for v in venues:
        all_transit_urls = _dedup_urls(all_transit_urls, v.transit_urls)

    # Verification leaf: any URL that proves transit access suffices
    transit_leaf = evaluator.add_leaf(
        id="Transit_Access_Verified",
        desc="Transit access to at least one venue is supported by sources",
        parent=transit_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows that the venue described is accessible via public transportation such as LA Metro rail or bus (with a nearby station or stop).",
        node=transit_leaf,
        sources=all_transit_urls,
        additional_instruction="Look for explicit mentions of LA Metro (rail or bus) or other public transit lines/stops serving the venue. Any one valid page across the provided URLs is sufficient."
    )

    # Existence of at least one transit reference URL across venues
    evaluator.add_custom_node(
        result=len(all_transit_urls) > 0,
        id="Transportation_Reference",
        desc="Provide URL confirming public transit access to at least one venue",
        parent=transit_node,
        critical=True
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
    Evaluate an answer for the Los Angeles accessible venues task.
    """
    # Initialize evaluator (root is non-critical to allow partial across venues;
    # global constraints like transit are added as critical child nodes)
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

    # Extract structured venues info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Ensure exactly 3 entries for evaluation (pad with empty if fewer)
    venues: List[VenueItem] = list(extracted.venues[:3])
    while len(venues) < 3:
        venues.append(VenueItem())

    # Build the top-level node mirroring the rubric root (non-critical to allow partial credit)
    task_root = evaluator.add_parallel(
        id="Find_Three_Accessible_Concert_Venues",
        desc="Identify three distinct concert or comedy venues in Los Angeles, CA that meet all specified accessibility and operational criteria",
        parent=root,
        critical=False
    )

    # Add the three venue subtrees
    for i in range(3):
        await verify_single_venue(evaluator, task_root, venues[i], i)

    # Add the public transportation requirement (critical) under the task root
    await verify_public_transportation_requirement(evaluator, task_root, venues)

    # Return summary
    return evaluator.get_summary()