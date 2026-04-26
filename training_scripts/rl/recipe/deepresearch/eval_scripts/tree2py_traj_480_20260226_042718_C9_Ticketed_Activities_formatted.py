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
TASK_ID = "events_feb2026_four_categories"
TASK_DESCRIPTION = (
    "I need to identify four distinct ticketed entertainment events scheduled to perform in February 2026 in the United States, "
    "with each event belonging to a different category and meeting specific requirements. "
    "The four categories are:\n\n"
    "Category 1 - Broadway Production: Identify one Broadway show that begins preview performances in February or March 2026. "
    "The show must be performed at a Broadway theater (defined as having 500+ seats and located in Manhattan's Theater District). "
    "Provide the show title, venue name and full address, theater seating capacity, preview start date, official opening night date, "
    "name of at least one lead performer, and a link to the official ticketing page.\n\n"
    "Category 2 - National Touring Show: Identify one touring production of a television-based live entertainment show that has a scheduled "
    "performance in February 2026. Provide the tour name, specific performance date in February 2026, venue name and full address where the February "
    "performance occurs, venue seating capacity, names of at least two special guests (co-hosts or guest performers) who will appear on the tour, "
    "and a link to the tour's official website or ticketing page.\n\n"
    "Category 3 - Comedy Tour Event: Identify one comedy tour that has a scheduled performance in February 2026. The tour must feature multiple comedians "
    "(not a solo comedian tour). Provide the tour name, specific performance date in February 2026, venue name and city where the February performance occurs, "
    "names of at least three comedians featured on the tour, and a link to an official ticketing or tour information page.\n\n"
    "Category 4 - Live Entertainment Event: Identify one additional ticketed live entertainment event (concert, festival, sporting event, or other live performance) "
    "scheduled in February 2026. This event must be from a different category than the previous three. Provide the event name, specific date(s) in February 2026, "
    "venue name and location, event type/description, and a link to an official event or ticketing page.\n\n"
    "Additional Requirements:\n"
    "- All events must have publicly available ticket purchasing information\n"
    "- All venue locations must include complete address or at minimum city and state\n"
    "- All dates must be specifically in February 2026 (or early spring 2026 for Category 1)\n"
    "- Each event must be a distinct, separately ticketed production (not multiple performances of the same show)\n"
    "- All information must be verifiable through official sources"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BroadwayInfo(BaseModel):
    show_title: Optional[str] = None
    show_title_sources: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    venue_location_sources: List[str] = Field(default_factory=list)

    theatre_district_sources: List[str] = Field(default_factory=list)

    seating_capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)

    preview_start_date: Optional[str] = None
    preview_date_sources: List[str] = Field(default_factory=list)

    opening_night_date: Optional[str] = None
    opening_date_sources: List[str] = Field(default_factory=list)

    lead_performer: Optional[str] = None
    cast_sources: List[str] = Field(default_factory=list)

    ticketing_link: Optional[str] = None


class TouringInfo(BaseModel):
    tour_name: Optional[str] = None
    tour_name_sources: List[str] = Field(default_factory=list)

    tv_basis_text: Optional[str] = None
    tv_basis_sources: List[str] = Field(default_factory=list)

    february_date: Optional[str] = None
    date_sources: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    venue_city_state: Optional[str] = None
    venue_sources: List[str] = Field(default_factory=list)

    venue_capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)

    special_guests: List[str] = Field(default_factory=list)
    special_guests_sources: List[str] = Field(default_factory=list)

    official_link: Optional[str] = None


class ComedyInfo(BaseModel):
    tour_name: Optional[str] = None
    tour_name_sources: List[str] = Field(default_factory=list)

    february_date: Optional[str] = None
    date_sources: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_sources: List[str] = Field(default_factory=list)

    comedians: List[str] = Field(default_factory=list)
    comedians_sources: List[str] = Field(default_factory=list)

    ticketing_link: Optional[str] = None


class LiveEventInfo(BaseModel):
    event_name: Optional[str] = None
    event_name_sources: List[str] = Field(default_factory=list)

    event_type_description: Optional[str] = None
    event_type_sources: List[str] = Field(default_factory=list)

    february_dates: List[str] = Field(default_factory=list)
    event_dates_sources: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None
    venue_location: Optional[str] = None
    venue_sources: List[str] = Field(default_factory=list)

    official_link: Optional[str] = None


class EventsExtraction(BaseModel):
    category1_broadway: Optional[BroadwayInfo] = None
    category2_touring: Optional[TouringInfo] = None
    category3_comedy: Optional[ComedyInfo] = None
    category4_event: Optional[LiveEventInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract structured information for four distinct events across specified categories from the answer. 
    Return JSON strictly matching the following schema. Use null for any missing field. 
    For sources, extract all URLs explicitly mentioned; do not invent URLs.

    {
      "category1_broadway": {
        "show_title": string|null,
        "show_title_sources": [url,...],
        "venue_name": string|null,
        "venue_address": string|null,
        "venue_location_sources": [url,...],
        "theatre_district_sources": [url,...],
        "seating_capacity": string|null,
        "capacity_sources": [url,...],
        "preview_start_date": string|null,
        "preview_date_sources": [url,...],
        "opening_night_date": string|null,
        "opening_date_sources": [url,...],
        "lead_performer": string|null,
        "cast_sources": [url,...],
        "ticketing_link": url|null
      },
      "category2_touring": {
        "tour_name": string|null,
        "tour_name_sources": [url,...],
        "tv_basis_text": string|null,
        "tv_basis_sources": [url,...],
        "february_date": string|null,
        "date_sources": [url,...],
        "venue_name": string|null,
        "venue_address": string|null,
        "venue_city_state": string|null,
        "venue_sources": [url,...],
        "venue_capacity": string|null,
        "capacity_sources": [url,...],
        "special_guests": [string,...],
        "special_guests_sources": [url,...],
        "official_link": url|null
      },
      "category3_comedy": {
        "tour_name": string|null,
        "tour_name_sources": [url,...],
        "february_date": string|null,
        "date_sources": [url,...],
        "venue_name": string|null,
        "venue_city": string|null,
        "venue_sources": [url,...],
        "comedians": [string,...],
        "comedians_sources": [url,...],
        "ticketing_link": url|null
      },
      "category4_event": {
        "event_name": string|null,
        "event_name_sources": [url,...],
        "event_type_description": string|null,
        "event_type_sources": [url,...],
        "february_dates": [string,...],
        "event_dates_sources": [url,...],
        "venue_name": string|null,
        "venue_location": string|null,
        "venue_sources": [url,...],
        "official_link": url|null
      }
    }

    Notes:
    - Dates must be explicit strings as written in the answer (e.g., "February 12, 2026", "2026-02-12").
    - For addresses, include full address where possible; otherwise provide city and state.
    - For TV-based touring show, include any phrase indicating it is based on a television program in "tv_basis_text".
    - Ensure each URL is extracted as a full URL (prepend http:// if protocol missing).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _to_list(urls: Optional[List[str] | str]) -> List[str]:
    if urls is None:
        return []
    if isinstance(urls, list):
        return [u for u in urls if isinstance(u, str) and u.strip()]
    if isinstance(urls, str) and urls.strip():
        return [urls.strip()]
    return []


def _parse_int(s: Optional[str]) -> Optional[int]:
    if not s or not isinstance(s, str):
        return None
    digits = re.sub(r"[^\d]", "", s)
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _is_feb_2026(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    s = date_str.strip().lower()
    if "2026" not in s:
        return False
    # Month name checks
    if "february" in s or "feb" in s:
        return True
    # Numeric formats: 2026-02-.. or 02/.. /2026 or 2/.../2026 etc.
    patterns = [
        r"\b2026[-/\.]02\b",
        r"\b02[-/\.]\d{1,2}[-/\.]2026\b",
        r"\b2[-/\.]\d{1,2}[-/\.]2026\b",
        r"\bfeb[-/\.]?\s*\d{1,2},?\s*2026\b",
    ]
    return any(re.search(p, s) for p in patterns)


def _is_feb_or_mar_2026(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    s = date_str.strip().lower()
    if "2026" not in s:
        return False
    if any(m in s for m in ["february", "feb", "march", "mar"]):
        return True
    patterns = [
        r"\b2026[-/\.](02|03)\b",
        r"\b(02|03)[-/\.]\d{1,2}[-/\.]2026\b",
        r"\b(2|3)[-/\.]\d{1,2}[-/\.]2026\b",
    ]
    return any(re.search(p, s) for p in patterns)


def _names_str(names: List[str]) -> str:
    return ", ".join([n for n in names if n and n.strip()])


# --------------------------------------------------------------------------- #
# Category 1: Broadway production verification                                #
# --------------------------------------------------------------------------- #
async def verify_category1_broadway(evaluator: Evaluator, root_node, b: Optional[BroadwayInfo]) -> None:
    cat = evaluator.add_parallel(
        id="category_1_broadway_production",
        desc="Category 1: Broadway production with preview performances in February or March 2026",
        parent=root_node,
        critical=False,
    )

    # Fallback empty object to avoid None checks everywhere
    b = b or BroadwayInfo()

    # Subgroup: identification
    ident = evaluator.add_parallel(
        id="broadway_identification",
        desc="Show identification and basic information",
        parent=cat,
        critical=True,
    )
    show_group = evaluator.add_parallel(
        id="broadway_show_title",
        desc="Show title is provided",
        parent=ident,
        critical=True,
    )
    # Existence check
    evaluator.add_custom_node(
        result=bool(b.show_title and b.show_title.strip()),
        id="broadway_show_title_provided",
        desc="Show title string is present",
        parent=show_group,
        critical=True,
    )
    # Reference verification
    title_ref = evaluator.add_leaf(
        id="show_title_reference",
        desc="Reference URL provided confirming the show title",
        parent=show_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The show is titled '{b.show_title or ''}'.",
        node=title_ref,
        sources=_to_list(b.show_title_sources),
        additional_instruction="Confirm the show's official title on the cited official sources.",
    )

    # Subgroup: venue details
    venue = evaluator.add_parallel(
        id="broadway_venue_details",
        desc="Venue meets Broadway theater requirements and location information is complete",
        parent=cat,
        critical=True,
    )
    # Venue name and address
    venue_info = evaluator.add_parallel(
        id="venue_name_and_address",
        desc="Venue name and full address provided",
        parent=venue,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(b.venue_name and b.venue_name.strip() and b.venue_address and b.venue_address.strip()),
        id="venue_name_address_provided",
        desc="Venue name and full address strings are present",
        parent=venue_info,
        critical=True,
    )
    venue_ref = evaluator.add_leaf(
        id="venue_location_reference",
        desc="Reference URL confirming venue name and address",
        parent=venue_info,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue is '{b.venue_name or ''}', located at '{b.venue_address or ''}'.",
        node=venue_ref,
        sources=_to_list(b.venue_location_sources),
        additional_instruction="Verify the venue name and full address match the official venue or show page.",
    )

    # Theater District location
    district = evaluator.add_parallel(
        id="theater_district_location",
        desc="Venue is located in Manhattan's Theater District (between 41st and 54th Streets, between 6th and 8th Avenues)",
        parent=venue,
        critical=True,
    )
    district_ref = evaluator.add_leaf(
        id="location_verification_reference",
        desc="Reference URL confirming Theater District location",
        parent=district,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue '{b.venue_name or ''}' is located in Manhattan's Theater District.",
        node=district_ref,
        sources=_to_list(b.theatre_district_sources),
        additional_instruction="Confirm the venue is in Manhattan's Theater District—roughly 41st to 54th Streets, and 6th to 8th Avenues.",
    )

    # Seating capacity >= 500
    capacity = evaluator.add_parallel(
        id="seating_capacity",
        desc="Theater seating capacity is provided and meets the 500+ seats requirement for Broadway theaters",
        parent=venue,
        critical=True,
    )
    cap_int = _parse_int(b.seating_capacity)
    evaluator.add_custom_node(
        result=bool(cap_int is not None and cap_int >= 500),
        id="capacity_meets_requirement",
        desc=f"Capacity parsed as {cap_int if cap_int is not None else 'N/A'} is >= 500",
        parent=capacity,
        critical=True,
    )
    cap_ref = evaluator.add_leaf(
        id="capacity_reference",
        desc="Reference URL confirming seating capacity",
        parent=capacity,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The seating capacity of '{b.venue_name or ''}' is {b.seating_capacity or ''}.",
        node=cap_ref,
        sources=_to_list(b.capacity_sources),
        additional_instruction="Confirm the capacity number on official venue or Broadway sources; capacity must be at least 500.",
    )

    # Performance dates
    dates = evaluator.add_parallel(
        id="broadway_performance_dates",
        desc="Preview start date and official opening night date are provided and verified",
        parent=cat,
        critical=True,
    )
    # Preview date
    preview = evaluator.add_parallel(
        id="preview_start_date",
        desc="Preview start date is provided and occurs in February or March 2026",
        parent=dates,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_is_feb_or_mar_2026(b.preview_start_date),
        id="preview_month_in_feb_or_mar_2026",
        desc=f"Preview date '{b.preview_start_date or ''}' is in February or March 2026",
        parent=preview,
        critical=True,
    )
    preview_ref = evaluator.add_leaf(
        id="preview_date_reference",
        desc="Reference URL confirming preview start date",
        parent=preview,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Preview performances begin on {b.preview_start_date or ''}.",
        node=preview_ref,
        sources=_to_list(b.preview_date_sources),
        additional_instruction="Verify the preview start date is correctly stated and corresponds to Feb or Mar 2026.",
    )

    # Opening night date
    opening = evaluator.add_parallel(
        id="opening_night_date",
        desc="Official opening night date is provided",
        parent=dates,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(b.opening_night_date and b.opening_night_date.strip()),
        id="opening_date_provided",
        desc=f"Opening night date '{b.opening_night_date or ''}' is present",
        parent=opening,
        critical=True,
    )
    opening_ref = evaluator.add_leaf(
        id="opening_date_reference",
        desc="Reference URL confirming opening night date",
        parent=opening,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official opening night is {b.opening_night_date or ''}.",
        node=opening_ref,
        sources=_to_list(b.opening_date_sources),
        additional_instruction="Confirm opening night date on official sources.",
    )

    # Cast information (lead performer)
    cast = evaluator.add_parallel(
        id="broadway_cast_information",
        desc="Cast information provided",
        parent=cat,
        critical=True,
    )
    lead = evaluator.add_parallel(
        id="lead_performer",
        desc="Name of at least one lead performer is provided",
        parent=cast,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(b.lead_performer and b.lead_performer.strip()),
        id="lead_performer_provided",
        desc="At least one lead performer is present",
        parent=lead,
        critical=True,
    )
    cast_ref = evaluator.add_leaf(
        id="cast_reference",
        desc="Reference URL confirming lead performer information",
        parent=lead,
        critical=True,
    )
    await evaluator.verify(
        claim=f"One of the lead performers is {b.lead_performer or ''}.",
        node=cast_ref,
        sources=_to_list(b.cast_sources),
        additional_instruction="Confirm the named performer is listed as lead/principal on official sources.",
    )

    # Ticketing link
    ticketing = evaluator.add_parallel(
        id="broadway_ticketing",
        desc="Ticketing information provided",
        parent=cat,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(b.ticketing_link and b.ticketing_link.strip()),
        id="broadway_ticketing_link_provided",
        desc="Official ticketing link is present",
        parent=ticketing,
        critical=True,
    )
    tix_leaf = evaluator.add_leaf(
        id="broadway_ticketing_link",
        desc="Link to official ticketing page is provided",
        parent=ticketing,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page is the official ticketing page for '{b.show_title or ''}'.",
        node=tix_leaf,
        sources=b.ticketing_link or None,
        additional_instruction="Verify the page sells tickets or directs to official ticket purchase for the show.",
    )


# --------------------------------------------------------------------------- #
# Category 2: National touring show verification                              #
# --------------------------------------------------------------------------- #
async def verify_category2_touring(evaluator: Evaluator, root_node, t: Optional[TouringInfo]) -> None:
    cat = evaluator.add_parallel(
        id="category_2_national_touring_show",
        desc="Category 2: National touring show with February 2026 performance",
        parent=root_node,
        critical=False,
    )
    t = t or TouringInfo()

    # Identification & TV basis
    ident = evaluator.add_parallel(
        id="tour_identification",
        desc="Tour identification and classification",
        parent=cat,
        critical=True,
    )
    tour_name_group = evaluator.add_parallel(
        id="tour_name",
        desc="Tour name is provided and identified as a television-based live entertainment show",
        parent=ident,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(t.tour_name and t.tour_name.strip()),
        id="tour_name_provided",
        desc="Tour name string is present",
        parent=tour_name_group,
        critical=True,
    )
    tour_name_ref = evaluator.add_leaf(
        id="tour_name_reference",
        desc="Reference URL confirming tour name and television basis",
        parent=tour_name_group,
        critical=True,
    )
    # Combine sources for name and tv basis
    name_tv_sources = _to_list(t.tour_name_sources) + _to_list(t.tv_basis_sources)
    await evaluator.verify(
        claim=f"The tour '{t.tour_name or ''}' is a television-based live entertainment show.",
        node=tour_name_ref,
        sources=name_tv_sources if name_tv_sources else None,
        additional_instruction="Confirm the tour is derived from or based on a TV program or TV franchise and validate its official name.",
    )

    # Performance date
    perf_date = evaluator.add_parallel(
        id="tour_performance_date",
        desc="Performance date verification",
        parent=cat,
        critical=True,
    )
    feb_date_group = evaluator.add_parallel(
        id="tour_february_date",
        desc="Specific performance date in February 2026 is provided",
        parent=perf_date,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_is_feb_2026(t.february_date),
        id="tour_date_is_feb_2026",
        desc=f"Date '{t.february_date or ''}' is in February 2026",
        parent=feb_date_group,
        critical=True,
    )
    feb_date_ref = evaluator.add_leaf(
        id="february_date_reference",
        desc="Reference URL confirming February 2026 performance date",
        parent=feb_date_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The tour performs on {t.february_date or ''}.",
        node=feb_date_ref,
        sources=_to_list(t.date_sources),
        additional_instruction="Confirm the performance date is in February 2026.",
    )

    # Venue details
    venue = evaluator.add_parallel(
        id="tour_venue_details",
        desc="Venue information for February performance",
        parent=cat,
        critical=True,
    )
    venue_info = evaluator.add_parallel(
        id="venue_name_address",
        desc="Venue name and complete address (or minimum city and state) provided",
        parent=venue,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(t.venue_name and t.venue_name.strip() and ((t.venue_address and t.venue_address.strip()) or (t.venue_city_state and t.venue_city_state.strip()))),
        id="tour_venue_info_provided",
        desc="Venue name and address or city/state are present",
        parent=venue_info,
        critical=True,
    )
    venue_ref = evaluator.add_leaf(
        id="venue_info_reference",
        desc="Reference URL confirming venue name and address",
        parent=venue_info,
        critical=True,
    )
    venue_loc_text = t.venue_address or t.venue_city_state or ""
    await evaluator.verify(
        claim=f"The February performance takes place at '{t.venue_name or ''}', located at '{venue_loc_text}'.",
        node=venue_ref,
        sources=_to_list(t.venue_sources),
        additional_instruction="Confirm the venue name and location (full address preferred; city/state acceptable) for the February performance.",
    )

    # Venue capacity
    cap_group = evaluator.add_parallel(
        id="venue_capacity_info",
        desc="Venue seating capacity information provided",
        parent=venue,
        critical=True,
    )
    venue_cap_int = _parse_int(t.venue_capacity)
    evaluator.add_custom_node(
        result=bool(t.venue_capacity and t.venue_capacity.strip()),
        id="tour_venue_capacity_provided",
        desc=f"Venue capacity string '{t.venue_capacity or ''}' is present",
        parent=cap_group,
        critical=True,
    )
    cap_ref = evaluator.add_leaf(
        id="capacity_info_reference",
        desc="Reference URL supporting venue capacity",
        parent=cap_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The seating capacity of '{t.venue_name or ''}' is {t.venue_capacity or ''}.",
        node=cap_ref,
        sources=_to_list(t.capacity_sources),
        additional_instruction="Confirm the capacity on official venue sources.",
    )

    # Personnel (special guests)
    personnel = evaluator.add_parallel(
        id="tour_personnel",
        desc="Special guests and performers information",
        parent=cat,
        critical=True,
    )
    guests_group = evaluator.add_parallel(
        id="tour_special_guests",
        desc="Names of at least two special guests (co-hosts or guest performers) provided",
        parent=personnel,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(len([g for g in t.special_guests if g and g.strip()]) >= 2),
        id="tour_special_guests_count",
        desc=f"At least two special guests provided ({len([g for g in t.special_guests if g and g.strip()])})",
        parent=guests_group,
        critical=True,
    )
    guests_ref = evaluator.add_leaf(
        id="special_guests_reference",
        desc="Reference URL confirming special guest information",
        parent=guests_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Special guests on the tour include: {_names_str(t.special_guests)}.",
        node=guests_ref,
        sources=_to_list(t.special_guests_sources),
        additional_instruction="Confirm that at least two named guests are officially listed for the tour.",
    )

    # Ticketing / official link
    ticketing = evaluator.add_parallel(
        id="tour_ticketing",
        desc="Ticketing information provided",
        parent=cat,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(t.official_link and t.official_link.strip()),
        id="tour_official_link_provided",
        desc="Official tour website or ticketing link is present",
        parent=ticketing,
        critical=True,
    )
    link_leaf = evaluator.add_leaf(
        id="tour_official_link",
        desc="Link to tour's official website or ticketing page provided",
        parent=ticketing,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page is the official website or ticketing page for the tour '{t.tour_name or ''}'.",
        node=link_leaf,
        sources=t.official_link or None,
        additional_instruction="Verify that the page is official and provides ticket purchasing or official tour info.",
    )


# --------------------------------------------------------------------------- #
# Category 3: Comedy tour verification                                        #
# --------------------------------------------------------------------------- #
async def verify_category3_comedy(evaluator: Evaluator, root_node, c: Optional[ComedyInfo]) -> None:
    cat = evaluator.add_parallel(
        id="category_3_comedy_tour_event",
        desc="Category 3: Multi-comedian tour with February 2026 performance",
        parent=root_node,
        critical=False,
    )
    c = c or ComedyInfo()

    # Identification
    ident = evaluator.add_parallel(
        id="comedy_identification",
        desc="Comedy tour identification",
        parent=cat,
        critical=True,
    )
    name_group = evaluator.add_parallel(
        id="comedy_tour_name",
        desc="Comedy tour name is provided",
        parent=ident,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(c.tour_name and c.tour_name.strip()),
        id="comedy_tour_name_provided",
        desc="Comedy tour name string is present",
        parent=name_group,
        critical=True,
    )
    name_ref = evaluator.add_leaf(
        id="comedy_tour_reference",
        desc="Reference URL confirming comedy tour name",
        parent=name_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The comedy tour is named '{c.tour_name or ''}'.",
        node=name_ref,
        sources=_to_list(c.tour_name_sources),
        additional_instruction="Confirm the official tour name on the cited source.",
    )

    # Performance date verification
    perf_date = evaluator.add_parallel(
        id="comedy_performance_date",
        desc="Performance date verification",
        parent=cat,
        critical=True,
    )
    feb_date_group = evaluator.add_parallel(
        id="comedy_february_date",
        desc="Specific performance date in February 2026 is provided",
        parent=perf_date,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_is_feb_2026(c.february_date),
        id="comedy_date_is_feb_2026",
        desc=f"Date '{c.february_date or ''}' is in February 2026",
        parent=feb_date_group,
        critical=True,
    )
    date_ref = evaluator.add_leaf(
        id="comedy_date_reference",
        desc="Reference URL confirming February 2026 performance date",
        parent=feb_date_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The comedy tour performs on {c.february_date or ''}.",
        node=date_ref,
        sources=_to_list(c.date_sources),
        additional_instruction="Confirm the performance date is in February 2026.",
    )

    # Venue information
    venue = evaluator.add_parallel(
        id="comedy_venue_details",
        desc="Venue information for February performance",
        parent=cat,
        critical=True,
    )
    venue_loc = evaluator.add_parallel(
        id="comedy_venue_location",
        desc="Venue name and city where February performance occurs are provided",
        parent=venue,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(c.venue_name and c.venue_name.strip() and c.venue_city and c.venue_city.strip()),
        id="comedy_venue_info_provided",
        desc="Venue name and city are present",
        parent=venue_loc,
        critical=True,
    )
    venue_ref = evaluator.add_leaf(
        id="comedy_venue_reference",
        desc="Reference URL confirming venue name and location",
        parent=venue_loc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The performance takes place at '{c.venue_name or ''}' in {c.venue_city or ''}.",
        node=venue_ref,
        sources=_to_list(c.venue_sources),
        additional_instruction="Confirm venue name and city on an official source or ticketing page.",
    )

    # Comedian lineup
    lineup = evaluator.add_parallel(
        id="comedy_performers",
        desc="Comedian lineup information",
        parent=cat,
        critical=True,
    )
    comedians_group = evaluator.add_parallel(
        id="comedy_tour_comedians",
        desc="Names of at least three comedians featured on the tour are provided, confirming it is a multi-comedian tour (not a solo comedian tour)",
        parent=lineup,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(len([n for n in c.comedians if n and n.strip()]) >= 3),
        id="comedy_three_or_more_comedians",
        desc=f"At least three comedians provided ({len([n for n in c.comedians if n and n.strip()])})",
        parent=comedians_group,
        critical=True,
    )
    comedians_ref = evaluator.add_leaf(
        id="comedians_reference",
        desc="Reference URL confirming comedian lineup",
        parent=comedians_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The comedian lineup includes: {_names_str(c.comedians)}.",
        node=comedians_ref,
        sources=_to_list(c.comedians_sources),
        additional_instruction="Confirm that three or more named comedians are listed for the tour.",
    )

    # Ticketing link
    ticketing = evaluator.add_parallel(
        id="comedy_ticketing",
        desc="Ticketing information provided",
        parent=cat,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(c.ticketing_link and c.ticketing_link.strip()),
        id="comedy_tour_link_provided",
        desc="Official ticketing or tour info link is present",
        parent=ticketing,
        critical=True,
    )
    link_leaf = evaluator.add_leaf(
        id="comedy_tour_link",
        desc="Link to official ticketing or tour information page provided",
        parent=ticketing,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page provides official ticketing or tour information for '{c.tour_name or ''}'.",
        node=link_leaf,
        sources=c.ticketing_link or None,
        additional_instruction="Verify the page is official and includes ticket purchase or official tour details.",
    )


# --------------------------------------------------------------------------- #
# Category 4: Additional live entertainment event verification                #
# --------------------------------------------------------------------------- #
async def verify_category4_event(evaluator: Evaluator, root_node, e: Optional[LiveEventInfo]) -> None:
    cat = evaluator.add_parallel(
        id="category_4_live_entertainment_event",
        desc="Category 4: Additional live entertainment event in February 2026 from a different category",
        parent=root_node,
        critical=False,
    )
    e = e or LiveEventInfo()

    # Event identification and type
    ident = evaluator.add_parallel(
        id="event_identification",
        desc="Event identification and classification",
        parent=cat,
        critical=True,
    )
    name_group = evaluator.add_parallel(
        id="event_name",
        desc="Event name is provided",
        parent=ident,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(e.event_name and e.event_name.strip()),
        id="event_name_provided",
        desc="Event name string is present",
        parent=name_group,
        critical=True,
    )
    name_ref = evaluator.add_leaf(
        id="event_name_reference",
        desc="Reference URL confirming event name",
        parent=name_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The event is named '{e.event_name or ''}'.",
        node=name_ref,
        sources=_to_list(e.event_name_sources),
        additional_instruction="Confirm the official event name.",
    )

    type_group = evaluator.add_parallel(
        id="event_type_description",
        desc="Event type/description provided and distinct from previous three categories (not Broadway, not TV-based touring show, not comedy tour)",
        parent=ident,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(e.event_type_description and e.event_type_description.strip()),
        id="event_type_provided",
        desc="Event type/description is present",
        parent=type_group,
        critical=True,
    )
    type_ref = evaluator.add_leaf(
        id="event_type_reference",
        desc="Reference URL supporting event type/description",
        parent=type_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The event type is '{e.event_type_description or ''}', which is distinct from Broadway shows, TV-based touring shows, and comedy tours.",
        node=type_ref,
        sources=_to_list(e.event_type_sources),
        additional_instruction="Confirm the event type/description and ensure it is NOT a Broadway production, NOT a TV-based live show, and NOT a comedy tour.",
    )

    # Performance date(s)
    perf = evaluator.add_parallel(
        id="event_performance_date",
        desc="Performance date verification",
        parent=cat,
        critical=True,
    )
    perf_dates_group = evaluator.add_parallel(
        id="event_february_dates",
        desc="Specific date(s) in February 2026 provided",
        parent=perf,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(e.february_dates) and all(_is_feb_2026(d) for d in e.february_dates),
        id="event_dates_are_feb_2026",
        desc=f"All event dates are in February 2026: {', '.join(e.february_dates) if e.february_dates else 'none'}",
        parent=perf_dates_group,
        critical=True,
    )
    dates_ref = evaluator.add_leaf(
        id="event_dates_reference",
        desc="Reference URL confirming February 2026 date(s)",
        parent=perf_dates_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The event occurs on the following date(s): {', '.join(e.february_dates) if e.february_dates else ''}.",
        node=dates_ref,
        sources=_to_list(e.event_dates_sources),
        additional_instruction="Confirm each listed date is in February 2026.",
    )

    # Venue details
    venue = evaluator.add_parallel(
        id="event_venue_details",
        desc="Venue information",
        parent=cat,
        critical=True,
    )
    venue_loc = evaluator.add_parallel(
        id="event_venue_location",
        desc="Venue name and location provided",
        parent=venue,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(e.venue_name and e.venue_name.strip() and e.venue_location and e.venue_location.strip()),
        id="event_venue_info_provided",
        desc="Event venue name and location are present",
        parent=venue_loc,
        critical=True,
    )
    venue_ref = evaluator.add_leaf(
        id="event_location_reference",
        desc="Reference URL confirming venue and location",
        parent=venue_loc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The event takes place at '{e.venue_name or ''}' in {e.venue_location or ''}.",
        node=venue_ref,
        sources=_to_list(e.venue_sources),
        additional_instruction="Confirm venue name and location (address or city/state) from official sources.",
    )

    # Ticketing link
    ticketing = evaluator.add_parallel(
        id="event_ticketing",
        desc="Ticketing information provided",
        parent=cat,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(e.official_link and e.official_link.strip()),
        id="event_official_link_provided",
        desc="Official event or ticketing page link is present",
        parent=ticketing,
        critical=True,
    )
    link_leaf = evaluator.add_leaf(
        id="event_official_link",
        desc="Link to official event or ticketing page provided",
        parent=ticketing,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page is the official event or ticketing page for '{e.event_name or ''}'.",
        node=link_leaf,
        sources=e.official_link or None,
        additional_instruction="Verify the page is official and allows ticket purchase or provides official event info.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the four-category February 2026 events task.
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
        default_model=model,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Build verification subtrees
    await verify_category1_broadway(evaluator, root, extracted.category1_broadway)
    await verify_category2_touring(evaluator, root, extracted.category2_touring)
    await verify_category3_comedy(evaluator, root, extracted.category3_comedy)
    await verify_category4_event(evaluator, root, extracted.category4_event)

    # Return structured result
    return evaluator.get_summary()