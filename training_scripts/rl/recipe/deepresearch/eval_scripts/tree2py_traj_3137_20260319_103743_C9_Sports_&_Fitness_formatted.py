import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_multipurpose_venues_4"
TASK_DESCRIPTION = """Identify 4 distinct multi-purpose sports venues in the United States that meet ALL of the following requirements:

Capacity Requirements:
- Seating capacity of at least 60,000 for football/soccer configuration
- Capability to accommodate crowds exceeding 70,000 for major events

Event Hosting Requirements:
- Must have hosted or be scheduled to host at least one major championship event, including: NCAA Final Four, College Football Playoff game, FIFA World Cup 2026 match, Super Bowl, or equivalent major championship
- The event must have occurred within the past 5 years (since March 2021) OR be scheduled within the next 3 years (through March 2029)

Accessibility Requirements:
- Must comply with ADA requirements for dispersed wheelchair seating locations (required for all venues with 300+ seats)
- Must provide wheelchair accessible seating equal to at least 1% of total capacity
- Each wheelchair space must have adjacent companion seating

Infrastructure Requirements:
- Must have a roof configuration that is either retractable, fixed, or open-air suitable for all-weather events
- Must feature luxury suites, club seating, and premium hospitality areas
- Must have been built after 1990 OR undergone major renovation within the past 15 years (since 2011)

Multi-Sport Capability Requirements:
- Must be capable of hosting both American football and international soccer matches
- Must have infrastructure that supports field conversion between different sports within 72 hours

For each of the 4 venues you identify, provide:
1. Official venue name
2. City and state location
3. Seating capacity with evidence it meets the 60,000+ requirement
4. At least one specific major championship event hosted (with dates) that falls within the specified timeframe
5. Evidence of ADA compliance including wheelchair seating provisions
6. Roof configuration type
7. Evidence of premium amenities (luxury suites, club seating, etc.)
8. Construction date or most recent major renovation date
9. Evidence of multi-sport hosting capability
10. Reference URLs supporting each category of information
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    # Basic info
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    basic_urls: List[str] = Field(default_factory=list)

    # Capacity info
    capacity: Optional[str] = None  # as provided in the answer, any format
    can_exceed_70k: Optional[str] = None  # textual evidence/statement if any
    capacity_urls: List[str] = Field(default_factory=list)

    # Event hosting (at least one)
    event_name: Optional[str] = None
    event_type: Optional[str] = None  # e.g., NCAA Final Four, CFP, World Cup 2026, Super Bowl
    event_date: Optional[str] = None  # any human-readable date or just year
    event_urls: List[str] = Field(default_factory=list)

    # Accessibility
    wheelchair_dispersal: Optional[str] = None
    wheelchair_percentage: Optional[str] = None
    companion_seating: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=list)

    # Infrastructure
    roof_type: Optional[str] = None  # retractable / fixed / open-air / dome etc.
    premium_amenities: List[str] = Field(default_factory=list)  # suites, clubs, lounges, etc.
    construction_or_renovation: Optional[str] = None  # e.g., "Opened 2017", "Renovated 2019"
    infrastructure_urls: List[str] = Field(default_factory=list)

    # Multi-sport
    football_soccer_capability: Optional[str] = None
    conversion_time: Optional[str] = None  # e.g., "within 48 hours", "72 hours", etc.
    multisport_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to 4 distinct U.S. multi-purpose sports venues from the answer. For each venue, return the following fields:

    REQUIRED FIELDS FOR EACH VENUE:
    - name: Official current venue name (string).
    - city: City where the venue is located (string).
    - state: State where the venue is located; use the two-letter postal abbreviation if available (string).
    - basic_urls: Array of 1–3 URLs that the answer cites to support the basic info (name/location/capacity). If none cited, return an empty list.

    CAPACITY:
    - capacity: Seating capacity for football or soccer configuration as stated in the answer (string; keep format as written).
    - can_exceed_70k: A short phrase from the answer (if any) indicating the venue can exceed 70,000 for major events (e.g., expandable capacity, standing room, attendance records). If not stated, set null.
    - capacity_urls: Array of 0–3 URLs specifically cited for capacity/attendance/expandable capacity. If none, return an empty list.

    EVENT HOSTING:
    - event_name: Name of at least one major championship event hosted or scheduled (e.g., "Super Bowl LVII", "NCAA Men's Final Four", "CFP Semifinal", "FIFA World Cup 2026"). If multiple are mentioned, provide one representative name.
    - event_type: The category of that event from this set if possible: ["NCAA Final Four","College Football Playoff","FIFA World Cup 2026","Super Bowl","equivalent major championship"]. If unclear, keep as provided in the answer.
    - event_date: The date or year for that event as provided in the answer (string).
    - event_urls: Array of 1–3 URLs cited that support the event hosting/scheduling claim.

    ACCESSIBILITY (ADA):
    - wheelchair_dispersal: Phrase from the answer (if any) that indicates dispersed wheelchair seating locations. If not provided, set null.
    - wheelchair_percentage: Phrase from the answer (if any) indicating wheelchair seating equals at least 1% of capacity; could be a number or policy statement. If not provided, set null.
    - companion_seating: Phrase from the answer (if any) indicating each wheelchair space has adjacent companion seating. If not provided, set null.
    - accessibility_urls: Array of 1–3 URLs cited that support ADA accessibility provisions.

    INFRASTRUCTURE:
    - roof_type: One of ["retractable","fixed","open-air","dome","partially retractable","convertible"], or as described in the answer (string).
    - premium_amenities: Array of strings for amenities (e.g., "luxury suites","club seating","club lounges","hospitality clubs","loges","premium boxes").
    - construction_or_renovation: Year or phrase indicating opening year or most recent major renovation (string).
    - infrastructure_urls: Array of 1–3 URLs that support roof type, amenities, and/or construction/renovation details.

    MULTI-SPORT:
    - football_soccer_capability: Phrase from the answer indicating the venue hosts both American football and international soccer (string if available, else null).
    - conversion_time: Phrase or number indicating the field can be converted within 72 hours (string if available, else null).
    - multisport_urls: Array of 1–3 URLs that support football/soccer and conversion capability.

    RULES:
    - Extract only what is explicitly present in the answer text.
    - For all URL arrays, include only URLs that actually appear in the answer; if none appear, return an empty list.
    - If a field is not mentioned, set it to null or empty list as appropriate.
    - Return a JSON object with a single top-level key "venues" which is an array of up to 4 VenueItem objects in the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_year_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(20\\d{2})", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def is_within_timeframe(event_date: Optional[str]) -> bool:
    """
    Check if the event date (by year) falls within:
    - Past 5 years since March 2021, OR
    - Next 3 years through March 2029
    Approximation by year boundaries: accept any year 2021–2029 inclusive.
    """
    y = _first_year_from_text(event_date)
    if y is None:
        return False
    return 2021 <= y <= 2029


def uniq_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            u = u.strip()
            if not u:
                continue
            if not (u.startswith("http://") or u.startswith("https://")):
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def basic_sources_for(venue: VenueItem) -> List[str]:
    # Prefer basic/infrastructure/capacity/event as fallbacks
    return uniq_urls(venue.basic_urls, venue.infrastructure_urls, venue.capacity_urls, venue.event_urls)


def capacity_sources_for(venue: VenueItem) -> List[str]:
    return uniq_urls(venue.capacity_urls, venue.basic_urls, venue.event_urls)


def infra_sources_for(venue: VenueItem) -> List[str]:
    return uniq_urls(venue.infrastructure_urls, venue.basic_urls)


def multisport_sources_for(venue: VenueItem) -> List[str]:
    return uniq_urls(venue.multisport_urls, venue.event_urls, venue.basic_urls)


# --------------------------------------------------------------------------- #
# Verification logic per venue                                                #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int,
) -> None:
    # Venue-level node (non-critical to allow partial across venues)
    venue_node = evaluator.add_parallel(
        id=f"Venue_{idx}",
        desc=f"{['First','Second','Third','Fourth'][idx-1]} qualifying venue meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # -------------------- Basic Info (critical group) -------------------- #
    basic_node = evaluator.add_parallel(
        id=f"V{idx}_Basic_Info",
        desc="Basic venue information including name, location, and capacity",
        parent=venue_node,
        critical=True,
    )
    # Name verification
    vname = venue.name or ""
    name_node = evaluator.add_leaf(
        id=f"V{idx}_Name",
        desc="Provide the official venue name",
        parent=basic_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official name of the venue is '{vname}'.",
        node=name_node,
        sources=basic_sources_for(venue),
        additional_instruction="Allow minor naming variations and sponsor naming changes if clearly referring to the same venue."
    )

    # Location verification (US)
    vcity = venue.city or ""
    vstate = venue.state or ""
    location_node = evaluator.add_leaf(
        id=f"V{idx}_Location",
        desc="Provide the city and state location, verifying the venue is within the United States",
        parent=basic_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue is located in {vcity}, {vstate}, United States.",
        node=location_node,
        sources=basic_sources_for(venue),
        additional_instruction="Accept state abbreviations (e.g., AZ for Arizona). The page should clearly indicate the city and US state."
    )

    # Capacity split into two binary checks
    capacity60_node = evaluator.add_leaf(
        id=f"V{idx}_Capacity_60k",
        desc="Verify venue capacity is at least 60,000 seats (football/soccer configuration)",
        parent=basic_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The stadium's seating capacity in football or soccer configuration is at least 60,000.",
        node=capacity60_node,
        sources=capacity_sources_for(venue),
        additional_instruction="Use the page's stated seating capacity; rounding and reasonable approximations are acceptable."
    )

    capacity70_node = evaluator.add_leaf(
        id=f"V{idx}_Capacity_70kplus",
        desc="Verify venue can accommodate crowds exceeding 70,000 for major events",
        parent=basic_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue can accommodate more than 70,000 spectators for major events (e.g., via expandable seating, standing room, or historically documented attendance).",
        node=capacity70_node,
        sources=capacity_sources_for(venue),
        additional_instruction="Pass if any provided source indicates expandable capacity, standing-room, or event attendance over 70,000."
    )

    # Basic URL presence (critical)
    basic_url_exists = evaluator.add_custom_node(
        result=len(basic_sources_for(venue)) > 0,
        id=f"V{idx}_Basic_URL",
        desc="Provide reference URL verifying basic venue information",
        parent=basic_node,
        critical=True
    )

    # -------------------- Event History (critical + sequential) ---------- #
    event_node = evaluator.add_sequential(
        id=f"V{idx}_Event_History",
        desc="Verification of major championship event hosting within the specified timeframe",
        parent=venue_node,
        critical=True,
    )

    event_type_leaf = evaluator.add_leaf(
        id=f"V{idx}_Event_Type",
        desc="Identify at least one major championship event hosted or scheduled (NCAA Final Four, CFP, FIFA World Cup 2026, Super Bowl, or equivalent)",
        parent=event_node,
        critical=True,
    )
    event_claim = (
        f"The cited source shows that {vname or 'this venue'} has hosted or is scheduled to host a major championship event "
        f"such as NCAA Final Four, a College Football Playoff game, a FIFA World Cup 2026 match, a Super Bowl, "
        f"or an equivalent major championship. Example mentioned: {venue.event_name or 'unspecified'} ({venue.event_date or 'unspecified date'})."
    )
    await evaluator.verify(
        claim=event_claim,
        node=event_type_leaf,
        sources=venue.event_urls,
        additional_instruction="Focus on explicit statements that the venue hosted or will host one of the listed categories. Synonyms and official branding variations are acceptable."
    )

    # Timeframe check (critical) — logical check based on extracted date
    timeframe_ok = is_within_timeframe(venue.event_date)
    timeframe_leaf = evaluator.add_custom_node(
        result=timeframe_ok,
        id=f"V{idx}_Event_Timeframe",
        desc="Verify the event occurred within past 5 years or is scheduled within next 3 years",
        parent=event_node,
        critical=True
    )

    # Event URL presence (critical)
    event_url_leaf = evaluator.add_custom_node(
        result=len(venue.event_urls) > 0,
        id=f"V{idx}_Event_URL",
        desc="Provide reference URL verifying event hosting information",
        parent=event_node,
        critical=True
    )

    # -------------------- Accessibility (critical + parallel) ------------- #
    access_node = evaluator.add_parallel(
        id=f"V{idx}_Accessibility",
        desc="ADA compliance verification for wheelchair seating requirements",
        parent=venue_node,
        critical=True,
    )

    wc_disp_leaf = evaluator.add_leaf(
        id=f"V{idx}_Wheelchair_Dispersal",
        desc="Verify venue provides dispersed wheelchair seating locations (required for 300+ seat venues)",
        parent=access_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue provides dispersed wheelchair seating locations distributed across various seating areas or price levels.",
        node=wc_disp_leaf,
        sources=venue.accessibility_urls,
        additional_instruction="Look for explicit ADA seating policies indicating dispersed wheelchair seating across sections/levels."
    )

    wc_pct_leaf = evaluator.add_leaf(
        id=f"V{idx}_Wheelchair_Percentage",
        desc="Verify wheelchair accessible seating equals at least 1% of total capacity",
        parent=access_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue provides wheelchair accessible seating equal to at least 1% of total capacity.",
        node=wc_pct_leaf,
        sources=venue.accessibility_urls,
        additional_instruction="Pass if the source explicitly states ≥1% or provides counts that clearly meet/exceed 1% of stated capacity."
    )

    companion_leaf = evaluator.add_leaf(
        id=f"V{idx}_Companion_Seats",
        desc="Verify each wheelchair space has adjacent companion seating",
        parent=access_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Each wheelchair space at the venue includes adjacent companion seating.",
        node=companion_leaf,
        sources=venue.accessibility_urls,
        additional_instruction="Look for phrasing like 'companion seats' adjacent to wheelchair locations per ADA policy."
    )

    access_url_exists = evaluator.add_custom_node(
        result=len(venue.accessibility_urls) > 0,
        id=f"V{idx}_Accessibility_URL",
        desc="Provide reference URL verifying accessibility compliance",
        parent=access_node,
        critical=True
    )

    # -------------------- Infrastructure (critical + parallel) ------------ #
    infra_node = evaluator.add_parallel(
        id=f"V{idx}_Infrastructure",
        desc="Infrastructure and facility modernization requirements",
        parent=venue_node,
        critical=True,
    )

    roof_leaf = evaluator.add_leaf(
        id=f"V{idx}_Roof_Config",
        desc="Identify roof configuration (retractable, fixed, or open-air)",
        parent=infra_node,
        critical=True,
    )
    roof_type_txt = venue.roof_type or "retractable, fixed, or open-air"
    await evaluator.verify(
        claim=f"The venue has a roof configuration that is {roof_type_txt} (or an equivalent term).",
        node=roof_leaf,
        sources=infra_sources_for(venue),
        additional_instruction="Accept synonyms like 'dome' for fixed roof, or 'convertible/retractable' variants. Focus on explicit roof type description."
    )

    premium_leaf = evaluator.add_leaf(
        id=f"V{idx}_Premium_Amenities",
        desc="Verify presence of luxury suites, club seating, and premium hospitality areas",
        parent=infra_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue features premium amenities including luxury suites, club seating, and hospitality club/lounge areas.",
        node=premium_leaf,
        sources=infra_sources_for(venue),
        additional_instruction="Pass if the page shows any combination of suites + club seating + premium/hospitality areas, even if naming varies (e.g., lounges, clubs, loges)."
    )

    constr_leaf = evaluator.add_leaf(
        id=f"V{idx}_Construction_Date",
        desc="Verify venue was built after 1990 or underwent major renovation within past 15 years",
        parent=infra_node,
        critical=True,
    )
    year_txt = venue.construction_or_renovation or "unspecified year"
    await evaluator.verify(
        claim=f"The venue meets modernization timing: it was built after 1990 or had a major renovation in 2011 or later (evidence includes: {year_txt}).",
        node=constr_leaf,
        sources=infra_sources_for(venue),
        additional_instruction="Pass if any cited source shows opening year ≥ 1991, or a major renovation year ≥ 2011."
    )

    infra_url_exists = evaluator.add_custom_node(
        result=len(infra_sources_for(venue)) > 0,
        id=f"V{idx}_Infrastructure_URL",
        desc="Provide reference URL verifying infrastructure information",
        parent=infra_node,
        critical=True
    )

    # -------------------- Multi-sport capability (critical + parallel) ---- #
    multi_node = evaluator.add_parallel(
        id=f"V{idx}_Multi_Sport",
        desc="Multi-sport capability verification",
        parent=venue_node,
        critical=True,
    )

    fb_soccer_leaf = evaluator.add_leaf(
        id=f"V{idx}_Football_Soccer",
        desc="Verify capability to host both American football and international soccer",
        parent=multi_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue is capable of hosting both American football and international soccer matches.",
        node=fb_soccer_leaf,
        sources=multisport_sources_for(venue),
        additional_instruction="Pass if sources show football games and soccer matches (friendlies, MLS, World Cup, etc.) hosted or scheduled at the venue."
    )

    conv72_leaf = evaluator.add_leaf(
        id=f"V{idx}_Field_Conversion",
        desc="Verify infrastructure supports field conversion between sports within 72 hours",
        parent=multi_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue's infrastructure supports conversion between football and soccer field configurations within 72 hours.",
        node=conv72_leaf,
        sources=multisport_sources_for(venue),
        additional_instruction="Look for explicit conversion time claims, operations scheduling, or venue/operator statements indicating ≤72 hours."
    )

    multi_url_exists = evaluator.add_custom_node(
        result=len(multisport_sources_for(venue)) > 0,
        id=f"V{idx}_Multi_Sport_URL",
        desc="Provide reference URL verifying multi-sport capability",
        parent=multi_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the '4 U.S. multi-purpose venues' task.
    """
    # Initialize evaluator (Root node is non-critical by default to allow partial credit across venues)
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

    # Record timeframe boundaries for reference
    evaluator.add_custom_info(
        info={"timeframe_years_inclusive": [2021, 2029], "note": "Approximate by year; Mar 2021 through Mar 2029."},
        info_type="timeframe_context",
        info_name="timeframe_context"
    )

    # Extract structured venues info
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    venues = list(extracted.venues or [])
    # Filter to exactly 4 entries: take first 4, pad with empty placeholders if fewer
    venues = venues[:4]
    while len(venues) < 4:
        venues.append(VenueItem())

    # Build verification subtrees for 4 venues
    for i, v in enumerate(venues, start=1):
        await verify_single_venue(evaluator, root, v, i)

    # Return the standard summary
    return evaluator.get_summary()