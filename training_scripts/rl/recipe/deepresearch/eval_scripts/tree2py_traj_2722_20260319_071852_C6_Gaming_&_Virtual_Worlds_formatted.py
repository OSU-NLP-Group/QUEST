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
TASK_ID = "us_esports_venues_3"
TASK_DESCRIPTION = """
Identify three dedicated esports venues in the United States that meet all of the following requirements:

1. Each venue must be located in the United States and have a complete, verifiable physical address (including street address, city, state, and ZIP code).

2. Each venue must have a minimum spectator capacity of at least 500 people for seated or standing attendance.

3. Each venue must have at least 7,000 square feet of total facility space.

4. Each venue must feature dedicated gaming infrastructure, including gaming PCs or console stations available for competitive play.

5. Each venue must have professional-grade visual display systems, such as LED walls, video walls, or large projection screens.

6. Each venue must provide official contact information (phone number, email, or website) for booking or inquiries.

7. Each venue must be confirmed as operational or have operated as an esports facility (not merely proposed or under construction).

8. All specifications (capacity, square footage, address, gaming infrastructure, and visual display systems) must be verifiable through official sources, such as the venue's official website, university pages, or established facility directories.

For each of the three venues, provide:
- The venue name
- Complete physical address (street address, city, state, ZIP code)
- Spectator capacity
- Total facility square footage
- Description of gaming infrastructure (number and type of gaming stations)
- Description of visual display systems
- Official contact information
- URL references to official sources that verify each specification
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Address(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None


class VenueSources(BaseModel):
    address_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    size_urls: List[str] = Field(default_factory=list)
    infrastructure_urls: List[str] = Field(default_factory=list)
    display_urls: List[str] = Field(default_factory=list)
    contact_urls: List[str] = Field(default_factory=list)
    operational_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class VenueItem(BaseModel):
    name: Optional[str] = None
    address: Address = Field(default_factory=Address)
    capacity: Optional[str] = None
    square_footage: Optional[str] = None
    gaming_infrastructure: Optional[str] = None
    display_systems: Optional[str] = None
    contact_info: Optional[str] = None
    sources: VenueSources = Field(default_factory=VenueSources)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to FIVE dedicated esports venues mentioned in the answer (we will evaluate only the first three if more are present).
    For each venue, extract the following fields exactly as presented in the answer text:

    - name: Venue name
    - address: 
        - street: street address (include suite if given)
        - city: city
        - state: two-letter or full state name as written
        - zip_code: ZIP code (5-digit or ZIP+4 if given)
    - capacity: spectator capacity as a textual value (e.g., "600", "500+", "1,200 standing")
    - square_footage: total facility size as a textual value (e.g., "10,000 sq ft", "7000 square feet")
    - gaming_infrastructure: brief description of gaming PCs or console stations available for competitive play (e.g., "50 high-end PCs and 10 console bays")
    - display_systems: brief description of LED walls, video walls, or large projection screens
    - contact_info: official contact information (phone/email/booking page/website) as text

    Additionally, categorize all URLs cited in the answer by what they verify. For each venue, provide:
    - sources:
        - address_urls: URLs that show the physical address
        - capacity_urls: URLs that show the spectator capacity
        - size_urls: URLs that show square footage or total space
        - infrastructure_urls: URLs that show gaming PCs/console stations suitable for competition
        - display_urls: URLs that show LED walls/video walls/large projection
        - contact_urls: URLs that provide official contact info or booking
        - operational_urls: URLs that indicate the venue is/was operational (e.g., events, schedules, "open now")
        - other_urls: any additional relevant official or reputable directory URLs not covered above

    IMPORTANT RULES:
    - Extract only what is explicitly present in the answer text. Do not invent or infer missing information.
    - A URL must be explicitly present in the answer; if a source is mentioned without a URL, do not include it.
    - Keep textual fields as strings exactly as shown; do not normalize numbers.
    - If any field is missing, set it to null (for strings) or [] (for URL lists).

    Return JSON with a single top-level key 'venues' containing an array of venue objects as defined by the schema.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def full_address_str(addr: Address) -> str:
    parts = []
    if addr and addr.street:
        parts.append(addr.street.strip())
    city_state_zip = " ".join(
        [p for p in [addr.city or "", (addr.state or ""), (addr.zip_code or "")] if p]
    ).strip()
    if city_state_zip:
        if addr.city and addr.state:
            # Prefer "City, ST ZIP"
            cs = f"{addr.city.strip()}, {addr.state.strip()}"
            if addr.zip_code:
                cs = f"{cs} {addr.zip_code.strip()}"
            parts.append(cs)
        else:
            parts.append(city_state_zip)
    return ", ".join([p for p in parts if p]).strip()


def aggregate_all_urls(v: VenueItem) -> List[str]:
    all_lists = [
        v.sources.address_urls,
        v.sources.capacity_urls,
        v.sources.size_urls,
        v.sources.infrastructure_urls,
        v.sources.display_urls,
        v.sources.contact_urls,
        v.sources.operational_urls,
        v.sources.other_urls,
    ]
    seen = set()
    out: List[str] = []
    for lst in all_lists:
        for u in lst:
            if u and isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    out.append(uu)
    return out


def pick_sources(v: VenueItem, primary: List[str]) -> List[str]:
    if primary and len(primary) > 0:
        return primary
    return aggregate_all_urls(v)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_venue(evaluator: Evaluator, parent_node, v: VenueItem, idx: int) -> None:
    vid = idx + 1
    venue_node = evaluator.add_parallel(
        id=f"Venue_{vid}",
        desc=f"{['First','Second','Third'][idx]} qualifying esports venue with complete information",
        parent=parent_node,
        critical=False  # allow partial credit across venues
    )

    # ---------------- Location Information (sequential, critical) ----------------
    loc_node = evaluator.add_sequential(
        id=f"V{vid}_Location_Information",
        desc=f"Complete and verifiable location information for Venue {vid}",
        parent=venue_node,
        critical=True
    )

    # V*_US_Location (leaf, critical) - verify by URLs
    us_loc_leaf = evaluator.add_leaf(
        id=f"V{vid}_US_Location",
        desc="Venue is located in the United States",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is located in the United States.",
        node=us_loc_leaf,
        sources=pick_sources(v, v.sources.address_urls),
        additional_instruction="Use the address or page context to confirm the venue is in the United States."
    )

    # V*_Full_Address (parallel, critical)
    full_addr_node = evaluator.add_parallel(
        id=f"V{vid}_Full_Address",
        desc="Venue has a complete physical address including street address, city, state, and ZIP code",
        parent=loc_node,
        critical=True
    )

    # Street address present (existence)
    street_exists = evaluator.add_custom_node(
        result=bool(v.address and v.address.street and v.address.street.strip()),
        id=f"V{vid}_Street_Address",
        desc="Street address is provided",
        parent=full_addr_node,
        critical=True
    )

    # City/state/ZIP present (existence)
    city_state_zip_ok = evaluator.add_custom_node(
        result=bool(v.address and v.address.city and v.address.state and v.address.zip_code
                    and v.address.city.strip() and v.address.state.strip() and v.address.zip_code.strip()),
        id=f"V{vid}_City_State_ZIP",
        desc="City, state, and ZIP code are provided",
        parent=full_addr_node,
        critical=True
    )

    # Address reference (verify full address by URLs)
    addr_ref_leaf = evaluator.add_leaf(
        id=f"V{vid}_Address_Reference",
        desc="URL reference confirming the venue's address",
        parent=full_addr_node,
        critical=True
    )
    full_addr = full_address_str(v.address)
    await evaluator.verify(
        claim=f"The venue's address is '{full_addr}'.",
        node=addr_ref_leaf,
        sources=pick_sources(v, v.sources.address_urls),
        additional_instruction="Confirm the full street address (allow minor formatting differences and common abbreviations like 'St.' vs 'Street')."
    )

    # ---------------- Capacity Requirements (sequential, critical) ----------------
    cap_node = evaluator.add_sequential(
        id=f"V{vid}_Capacity_Requirements",
        desc="Venue meets minimum capacity requirements",
        parent=venue_node,
        critical=True
    )

    # Minimum capacity >= 500 (verify by URLs)
    min_cap_leaf = evaluator.add_leaf(
        id=f"V{vid}_Minimum_Capacity",
        desc="Venue has minimum spectator capacity of at least 500 people",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue's spectator capacity is at least 500.",
        node=min_cap_leaf,
        sources=pick_sources(v, v.sources.capacity_urls),
        additional_instruction="Check seating or standing capacity; synonyms like 'seats', 'standing room', 'attendance capacity' are acceptable."
    )

    # Capacity verification block (parallel, critical)
    cap_verify_node = evaluator.add_parallel(
        id=f"V{vid}_Capacity_Verification",
        desc="Capacity specification is verifiable from official source",
        parent=cap_node,
        critical=True
    )

    # Capacity value exists (existence)
    cap_value_exists = evaluator.add_custom_node(
        result=bool(v.capacity and v.capacity.strip()),
        id=f"V{vid}_Capacity_Value",
        desc="Specific capacity number is provided",
        parent=cap_verify_node,
        critical=True
    )

    # Capacity reference exact value (verify by URLs)
    cap_ref_leaf = evaluator.add_leaf(
        id=f"V{vid}_Capacity_Reference",
        desc="URL reference confirming the venue's capacity",
        parent=cap_verify_node,
        critical=True
    )
    cap_text = v.capacity or ""
    await evaluator.verify(
        claim=f"The spectator capacity is '{cap_text}'.",
        node=cap_ref_leaf,
        sources=pick_sources(v, v.sources.capacity_urls),
        additional_instruction="Match the specific capacity value (allow minor formatting differences like commas or '+' signs)."
    )

    # ---------------- Facility Specifications (parallel, critical) ----------------
    fac_node = evaluator.add_parallel(
        id=f"V{vid}_Facility_Specifications",
        desc="Venue meets facility size and infrastructure requirements",
        parent=venue_node,
        critical=True
    )

    # Square footage (sequential, critical)
    size_node = evaluator.add_sequential(
        id=f"V{vid}_Square_Footage",
        desc="Venue has at least 7,000 square feet of total facility space",
        parent=fac_node,
        critical=True
    )

    size_spec_exists = evaluator.add_custom_node(
        result=bool(v.square_footage and v.square_footage.strip()),
        id=f"V{vid}_Size_Specification",
        desc="Square footage specification is provided",
        parent=size_node,
        critical=True
    )

    size_ref_leaf = evaluator.add_leaf(
        id=f"V{vid}_Size_Reference",
        desc="URL reference confirming the venue's size",
        parent=size_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has at least 7,000 square feet of total facility space.",
        node=size_ref_leaf,
        sources=pick_sources(v, v.sources.size_urls),
        additional_instruction="Verify total facility size (accept synonyms like 'sq ft', 'square feet', 'sf', 'ft²'). If multiple spaces are listed, total area should meet or exceed 7,000 sq ft."
    )

    # Gaming infrastructure (sequential, critical)
    infra_node = evaluator.add_sequential(
        id=f"V{vid}_Gaming_Infrastructure",
        desc="Venue features dedicated gaming infrastructure",
        parent=fac_node,
        critical=True
    )

    gaming_leaf = evaluator.add_leaf(
        id=f"V{vid}_Gaming_Stations",
        desc="Venue has gaming PCs or console stations available",
        parent=infra_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue features dedicated gaming infrastructure with gaming PCs or console stations available for competitive play.",
        node=gaming_leaf,
        sources=pick_sources(v, v.sources.infrastructure_urls),
        additional_instruction="Look for mentions of PC bays, console stations, competitive setups, or similar infrastructure intended for esports play."
    )

    infra_ref_exists = evaluator.add_custom_node(
        result=bool(v.sources.infrastructure_urls and len(v.sources.infrastructure_urls) > 0),
        id=f"V{vid}_Infrastructure_Reference",
        desc="URL reference confirming gaming infrastructure",
        parent=infra_node,
        critical=True
    )

    # Visual display systems (sequential, critical)
    display_node = evaluator.add_sequential(
        id=f"V{vid}_Visual_Display_Systems",
        desc="Venue has professional-grade visual display systems",
        parent=fac_node,
        critical=True
    )

    display_leaf = evaluator.add_leaf(
        id=f"V{vid}_Display_Specification",
        desc="LED walls, video walls, or large projection screens are specified",
        parent=display_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has professional-grade visual display systems (LED walls, video walls, or large projection screens).",
        node=display_leaf,
        sources=pick_sources(v, v.sources.display_urls),
        additional_instruction="Look for LED wall specifications, video wall systems, projector screens, or similarly capable display systems used for live esports viewing."
    )

    display_ref_exists = evaluator.add_custom_node(
        result=bool(v.sources.display_urls and len(v.sources.display_urls) > 0),
        id=f"V{vid}_Display_Reference",
        desc="URL reference confirming visual display systems",
        parent=display_node,
        critical=True
    )

    # ---------------- Contact & Operational (parallel, critical) ----------------
    contact_node = evaluator.add_parallel(
        id=f"V{vid}_Contact_Verification",
        desc="Venue provides official contact information and is operational",
        parent=venue_node,
        critical=True
    )

    contact_exists = evaluator.add_custom_node(
        result=bool(v.contact_info and v.contact_info.strip()),
        id=f"V{vid}_Contact_Information",
        desc="Official contact information is provided (phone, email, or website)",
        parent=contact_node,
        critical=True
    )

    operational_leaf = evaluator.add_leaf(
        id=f"V{vid}_Operational_Status",
        desc="Venue is confirmed as operational or has operated as an esports facility",
        parent=contact_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is operational or has operated as an esports facility (not merely proposed or under construction).",
        node=operational_leaf,
        sources=pick_sources(v, v.sources.operational_urls),
        additional_instruction="Look for evidence like event listings, past tournaments, schedules, 'open now', booking availability, or prior operations as an esports venue."
    )

    official_source_leaf = evaluator.add_leaf(
        id=f"V{vid}_Official_Source",
        desc="Information is verifiable through official venue website, university page, or established facility directory",
        parent=contact_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official venue website, a university page, or a recognized facility directory listing for the venue.",
        node=official_source_leaf,
        sources=aggregate_all_urls(v),
        additional_instruction="Judge whether the page is official (e.g., venue's own domain, .edu university page) or a well-established directory (credible industry or university-backed directory)."
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
    Evaluate an answer for the 'three US esports venues' task.
    """
    # 1) Initialize evaluator (root as parallel, non-critical to allow partial credit across venues)
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

    # 2) Extract venues from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # 3) Select first three venues (pad with empty if fewer)
    venues: List[VenueItem] = list(extracted.venues or [])
    while len(venues) < 3:
        venues.append(VenueItem())

    # 4) Build verification tree per venue
    # Root-level node (non-critical, parallel aggregation for the three venues)
    venues_root = evaluator.add_parallel(
        id="Three_US_Esports_Venues",
        desc="Identify three dedicated esports venues in the United States that meet all specified requirements",
        parent=root,
        critical=False  # allow partial scoring if fewer venues fully qualify
    )

    # Process only first three venues
    for i in range(3):
        await verify_venue(evaluator, venues_root, venues[i], i)

    # 5) Return evaluation summary
    return evaluator.get_summary()