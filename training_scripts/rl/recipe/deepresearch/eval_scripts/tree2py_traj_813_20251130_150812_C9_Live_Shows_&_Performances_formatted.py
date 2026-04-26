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
TASK_ID = "nyc_live_performances_dec2025_jan2026"
TASK_DESCRIPTION = (
    "Find 4 upcoming live performances in New York City scheduled between December 2025 and January 2026, "
    "where each performance meets the specified category constraints and includes all required fields and verifying reference URLs."
)

DATE_RANGE_HUMAN = "between December 1, 2025 and January 31, 2026 inclusive"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PerformanceItem(BaseModel):
    category: Optional[str] = None  # e.g., "broadway_musical", "arena_concert", "midsized_theater", "standup_comedy"
    venue_name: Optional[str] = None
    full_address: Optional[str] = None
    capacity: Optional[str] = None  # keep as string for flexibility
    headline: Optional[str] = None  # show title / artist / performer / comedian
    date_time: Optional[str] = None
    ticket_price_range: Optional[str] = None
    accessibility_feature: Optional[str] = None
    box_office_contact: Optional[str] = None
    reference_venue_urls: List[str] = Field(default_factory=list)
    reference_show_urls: List[str] = Field(default_factory=list)
    reference_ticket_urls: List[str] = Field(default_factory=list)


class PerformancesExtraction(BaseModel):
    performance1: Optional[PerformanceItem] = None
    performance2: Optional[PerformanceItem] = None
    performance3: Optional[PerformanceItem] = None
    performance4: Optional[PerformanceItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_performances() -> str:
    return (
        "Extract structured details for four NYC performances scheduled between December 2025 and January 2026. "
        "Return a JSON object with four fields: performance1, performance2, performance3, performance4. "
        "Each should include:\n"
        "- category: one of ['broadway_musical','arena_concert','midsized_theater','standup_comedy'] matching the required type for that performance index\n"
        "- venue_name\n- full_address\n- capacity (exact value from the answer; keep as text; e.g., '1,600' or 'around 16,000')\n"
        "- headline (show title / artist / performer / comedian name)\n- date_time (specific performance date and time)\n"
        "- ticket_price_range (e.g., '$49–$129' or '$50-$200')\n- accessibility_feature (at least one feature, e.g., 'wheelchair accessible')\n"
        "- box_office_contact (phone/email or official box-office contact method)\n"
        "- reference_venue_urls: array of URL(s) verifying venue details (address/capacity/location)\n"
        "- reference_show_urls: array of URL(s) verifying show details (title/date/time/type)\n"
        "- reference_ticket_urls: array of URL(s) verifying ticketing info (prices/contact or official ticketing)\n\n"
        "Assign the categories for each performance exactly as:\n"
        "• performance1: 'broadway_musical'\n"
        "• performance2: 'arena_concert'\n"
        "• performance3: 'midsized_theater'\n"
        "• performance4: 'standup_comedy'\n\n"
        "Special rules:\n"
        "1) Extract only information explicitly present in the answer. If a field is missing, set it to null (or empty array for URL lists).\n"
        "2) URLs must be actual explicit links; include full protocol (http:// or https://). If URLs are not provided, return an empty list.\n"
        "3) Do not invent or infer numbers or details. Preserve text exactly as stated in the answer.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return urls or []


def _capacity_threshold_claim_text(min_seats: int) -> str:
    return f"The venue has a seating capacity of at least {min_seats} seats."


def _capacity_range_claim_text(min_seats: int, max_seats: int) -> str:
    return f"The venue has a seating capacity between {min_seats} and {max_seats} seats (inclusive)."


def _nyc_claim_text(address: Optional[str]) -> str:
    if _nonempty(address):
        return f"The venue address '{address}' is in New York City (NYC)."
    return "The venue is in New York City (NYC)."


def _date_in_range_claim(date_time: Optional[str]) -> str:
    if _nonempty(date_time):
        return f"The performance date '{date_time}' falls {DATE_RANGE_HUMAN}."
    return f"The performance date falls {DATE_RANGE_HUMAN}."


def _show_type_claim(category_key: str, headline: Optional[str]) -> str:
    name = headline or "the event"
    if category_key == "broadway_musical":
        return f"{name} is a Broadway musical production."
    if category_key == "arena_concert":
        return f"{name} is a concert or live music event."
    if category_key == "midsized_theater":
        return f"{name} is a theatrical or music performance."
    if category_key == "standup_comedy":
        return f"{name} is a stand-up comedy performance."
    return f"{name} matches the required event type."


def _venue_reference_claim(item: PerformanceItem) -> str:
    parts = []
    if _nonempty(item.venue_name):
        parts.append(f"venue named '{item.venue_name}'")
    if _nonempty(item.full_address):
        parts.append(f"located at '{item.full_address}'")
    base = "The referenced venue page(s) confirm the " + ", ".join(parts) if parts else "The referenced venue page(s) confirm the venue details"
    if _nonempty(item.capacity):
        base += f" and indicate a seating capacity '{item.capacity}'."
    else:
        base += "."
    return base


def _show_reference_claim(item: PerformanceItem, category_key: str) -> str:
    name = item.headline or "the event"
    venue = item.venue_name or "the venue"
    dt = item.date_time or "the specified date and time"
    type_desc = _show_type_claim(category_key, item.headline)
    return f"The referenced show page(s) confirm that '{name}' at '{venue}' is scheduled on {dt}, and that {type_desc.lower()}."


def _ticket_reference_claim(item: PerformanceItem) -> str:
    price = item.ticket_price_range or "the stated ticket prices"
    contact = item.box_office_contact or "box office contact or official ticketing details"
    return f"The referenced ticketing page(s) show prices consistent with '{price}' and include {contact}."


# --------------------------------------------------------------------------- #
# Verification functions per performance                                      #
# --------------------------------------------------------------------------- #
async def verify_performance_1(evaluator: Evaluator, parent_node, item: PerformanceItem) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_1_broadway",
        desc="Performance 1 (Broadway musical; venue capacity ≥ 1,500 seats) with all required details and references",
        parent=parent_node,
        critical=False
    )

    # Category constraints
    cat_node = evaluator.add_parallel(
        id="p1_category_constraints",
        desc="Meets Performance 1 category constraints",
        parent=perf_node,
        critical=True
    )

    leaf_broadway = evaluator.add_leaf(
        id="p1_is_broadway_musical",
        desc="Event is a Broadway musical production",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=_show_type_claim("broadway_musical", item.headline),
        node=leaf_broadway,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction="Confirm that the show is a Broadway musical; allow reasonable phrasing variants like 'Broadway production'."
    )

    leaf_capacity_thresh = evaluator.add_leaf(
        id="p1_capacity_threshold",
        desc="Venue seating capacity is ≥ 1,500 seats",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=_capacity_threshold_claim_text(1500),
        node=leaf_capacity_thresh,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Use venue sources to confirm that capacity meets or exceeds 1,500 seats; approximate or 'about' is acceptable if clearly ≥1500."
    )

    # Required fields
    req_node = evaluator.add_parallel(
        id="p1_required_fields",
        desc="Includes all required fields for Performance 1",
        parent=perf_node,
        critical=True
    )
    evaluator.add_custom_node(_nonempty(item.venue_name), "p1_venue_name", "Venue name is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.full_address), "p1_full_address", "Full venue address is provided", parent=req_node, critical=True)
    leaf_in_nyc = evaluator.add_leaf(id="p1_in_nyc", desc="Venue location is in New York City (NYC)", parent=req_node, critical=True)
    await evaluator.verify(
        claim=_nyc_claim_text(item.full_address),
        node=leaf_in_nyc,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Treat 'New York, NY', 'Manhattan, NY', 'Brooklyn, NY', etc., as NYC. Verify from venue references."
    )
    evaluator.add_custom_node(_nonempty(item.capacity), "p1_exact_capacity", "Exact venue seating capacity value is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.headline), "p1_show_title", "Show title is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.date_time), "p1_date_time", "Specific performance date and time are provided", parent=req_node, critical=True)
    leaf_date_range = evaluator.add_leaf(id="p1_date_in_range", desc="Performance date is between December 2025 and January 2026", parent=req_node, critical=True)
    await evaluator.verify(
        claim=_date_in_range_claim(item.date_time),
        node=leaf_date_range,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction=f"Verify the stated date on the show page and confirm it falls {DATE_RANGE_HUMAN}. Minor time formatting differences are fine."
    )
    evaluator.add_custom_node(_nonempty(item.ticket_price_range), "p1_ticket_price_range", "Ticket price range is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.accessibility_feature), "p1_accessibility_feature", "At least one accessibility feature is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.box_office_contact), "p1_box_office_contact", "Box office contact information is provided", parent=req_node, critical=True)

    # References verification
    ref_node = evaluator.add_parallel(
        id="p1_references",
        desc="Provides reference URL(s) verifying venue info, show details, and ticketing info for Performance 1",
        parent=perf_node,
        critical=True
    )
    leaf_venue_ref = evaluator.add_leaf(id="p1_venue_reference", desc="Reference URL(s) provided for venue information (address/capacity/location)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_venue_reference_claim(item),
        node=leaf_venue_ref,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Check that venue references confirm name, address (NYC), and capacity. Approximate capacity wordings are acceptable if consistent."
    )
    leaf_show_ref = evaluator.add_leaf(id="p1_show_reference", desc="Reference URL(s) provided for show details (title/date/time/type)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_show_reference_claim(item, "broadway_musical"),
        node=leaf_show_ref,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction="Confirm title, date/time, venue, and that it's a Broadway musical."
    )
    leaf_ticket_ref = evaluator.add_leaf(id="p1_ticketing_reference", desc="Reference URL(s) provided for ticketing information (price range/box office contact or official ticketing)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_ticket_reference_claim(item),
        node=leaf_ticket_ref,
        sources=_urls_or_empty(item.reference_ticket_urls),
        additional_instruction="Ticket sources should show prices consistent with the stated range and include official sale/box office contact."
    )


async def verify_performance_2(evaluator: Evaluator, parent_node, item: PerformanceItem) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_2_arena",
        desc="Performance 2 (concert/live music; venue capacity ≥ 15,000) with all required details and references",
        parent=parent_node,
        critical=False
    )

    # Category constraints
    cat_node = evaluator.add_parallel(
        id="p2_category_constraints",
        desc="Meets Performance 2 category constraints",
        parent=perf_node,
        critical=True
    )

    leaf_concert = evaluator.add_leaf(
        id="p2_is_concert",
        desc="Event is a concert or live music event",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=_show_type_claim("arena_concert", item.headline),
        node=leaf_concert,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction="Confirm that this is a concert/live music event; terms like 'tour', 'live performance' are acceptable if clearly music."
    )

    leaf_capacity_thresh = evaluator.add_leaf(
        id="p2_capacity_threshold",
        desc="Venue capacity is ≥ 15,000 people",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=_capacity_threshold_claim_text(15000),
        node=leaf_capacity_thresh,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Use venue references to confirm capacity ≥15,000; if multiple configurations, the main capacity should meet/exceed."
    )

    # Required fields
    req_node = evaluator.add_parallel(
        id="p2_required_fields",
        desc="Includes all required fields for Performance 2",
        parent=perf_node,
        critical=True
    )
    evaluator.add_custom_node(_nonempty(item.venue_name), "p2_venue_name", "Venue name is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.full_address), "p2_full_address", "Full venue address is provided", parent=req_node, critical=True)
    leaf_in_nyc = evaluator.add_leaf(id="p2_in_nyc", desc="Venue location is in New York City (NYC)", parent=req_node, critical=True)
    await evaluator.verify(
        claim=_nyc_claim_text(item.full_address),
        node=leaf_in_nyc,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Treat boroughs within NYC as NYC (Manhattan, Brooklyn, Queens, Bronx, Staten Island). Verify from venue references."
    )
    evaluator.add_custom_node(_nonempty(item.capacity), "p2_exact_capacity", "Exact venue capacity value is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.headline), "p2_artist_name", "Artist/performer name is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.date_time), "p2_date_time", "Specific performance date and time are provided", parent=req_node, critical=True)
    leaf_date_range = evaluator.add_leaf(id="p2_date_in_range", desc="Performance date is between December 2025 and January 2026", parent=req_node, critical=True)
    await evaluator.verify(
        claim=_date_in_range_claim(item.date_time),
        node=leaf_date_range,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction=f"Confirm the stated date falls {DATE_RANGE_HUMAN}; minor time formatting differences are acceptable."
    )
    evaluator.add_custom_node(_nonempty(item.ticket_price_range), "p2_ticket_price_range", "Ticket price range is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.accessibility_feature), "p2_accessibility_feature", "At least one accessibility feature is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.box_office_contact), "p2_box_office_contact", "Box office contact information is provided", parent=req_node, critical=True)

    # References verification
    ref_node = evaluator.add_parallel(
        id="p2_references",
        desc="Provides reference URL(s) verifying venue info, show details, and ticketing info for Performance 2",
        parent=perf_node,
        critical=True
    )
    leaf_venue_ref = evaluator.add_leaf(id="p2_venue_reference", desc="Reference URL(s) provided for venue information (address/capacity/location)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_venue_reference_claim(item),
        node=leaf_venue_ref,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Venue references should confirm name, address (NYC), and capacity; arena capacities can be stated as 'up to' or 'approximate'."
    )
    leaf_show_ref = evaluator.add_leaf(id="p2_show_reference", desc="Reference URL(s) provided for show details (artist/date/time/type)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_show_reference_claim(item, "arena_concert"),
        node=leaf_show_ref,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction="Confirm artist/performer, date/time, venue, and that it's a concert/live music event."
    )
    leaf_ticket_ref = evaluator.add_leaf(id="p2_ticketing_reference", desc="Reference URL(s) provided for ticketing information (price range/box office contact or official ticketing)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_ticket_reference_claim(item),
        node=leaf_ticket_ref,
        sources=_urls_or_empty(item.reference_ticket_urls),
        additional_instruction="Ticket sources should show prices consistent with the stated range and include official sale or box office contact details."
    )


async def verify_performance_3(evaluator: Evaluator, parent_node, item: PerformanceItem) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_3_midsized",
        desc="Performance 3 (theatrical or music; venue seating capacity 2,500–6,000) with all required details and references",
        parent=parent_node,
        critical=False
    )

    # Category constraints
    cat_node = evaluator.add_parallel(
        id="p3_category_constraints",
        desc="Meets Performance 3 category constraints",
        parent=perf_node,
        critical=True
    )

    leaf_type = evaluator.add_leaf(
        id="p3_is_theatrical_or_music",
        desc="Event is a theatrical or music performance",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=_show_type_claim("midsized_theater", item.headline),
        node=leaf_type,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction="Confirm that the event is a theatrical or music performance; allow phrasing like 'live show', 'concert', 'theatre performance'."
    )

    leaf_capacity_range = evaluator.add_leaf(
        id="p3_capacity_range",
        desc="Venue seating capacity is between 2,500 and 6,000 seats (inclusive)",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=_capacity_range_claim_text(2500, 6000),
        node=leaf_capacity_range,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Use venue references to confirm capacity within 2,500–6,000 seats; approximate statements are acceptable if clearly within range."
    )

    # Required fields
    req_node = evaluator.add_parallel(
        id="p3_required_fields",
        desc="Includes all required fields for Performance 3",
        parent=perf_node,
        critical=True
    )
    evaluator.add_custom_node(_nonempty(item.venue_name), "p3_venue_name", "Venue name is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.full_address), "p3_full_address", "Full venue address is provided", parent=req_node, critical=True)
    leaf_in_nyc = evaluator.add_leaf(id="p3_in_nyc", desc="Venue location is in New York City (NYC)", parent=req_node, critical=True)
    await evaluator.verify(
        claim=_nyc_claim_text(item.full_address),
        node=leaf_in_nyc,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Verify the venue is in NYC using the venue references; borough addresses count as NYC."
    )
    evaluator.add_custom_node(_nonempty(item.capacity), "p3_exact_capacity", "Exact venue seating capacity value is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.headline), "p3_show_or_performer_name", "Show/performer name is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.date_time), "p3_date_time", "Specific performance date and time are provided", parent=req_node, critical=True)
    leaf_date_range = evaluator.add_leaf(id="p3_date_in_range", desc="Performance date is between December 2025 and January 2026", parent=req_node, critical=True)
    await evaluator.verify(
        claim=_date_in_range_claim(item.date_time),
        node=leaf_date_range,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction=f"Confirm the stated date falls {DATE_RANGE_HUMAN}."
    )
    evaluator.add_custom_node(_nonempty(item.ticket_price_range), "p3_ticket_price_range", "Ticket price range is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.accessibility_feature), "p3_accessibility_feature", "At least one accessibility feature is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.box_office_contact), "p3_box_office_contact", "Box office contact information is provided", parent=req_node, critical=True)

    # References verification
    ref_node = evaluator.add_parallel(
        id="p3_references",
        desc="Provides reference URL(s) verifying venue info, show details, and ticketing info for Performance 3",
        parent=perf_node,
        critical=True
    )
    leaf_venue_ref = evaluator.add_leaf(id="p3_venue_reference", desc="Reference URL(s) provided for venue information (address/capacity/location)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_venue_reference_claim(item),
        node=leaf_venue_ref,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Venue references should confirm name, address (NYC), and capacity."
    )
    leaf_show_ref = evaluator.add_leaf(id="p3_show_reference", desc="Reference URL(s) provided for show details (name/date/time/type)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_show_reference_claim(item, "midsized_theater"),
        node=leaf_show_ref,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction="Confirm show/performer, date/time, venue, and that it's a theatrical or music performance."
    )
    leaf_ticket_ref = evaluator.add_leaf(id="p3_ticketing_reference", desc="Reference URL(s) provided for ticketing information (price range/box office contact or official ticketing)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_ticket_reference_claim(item),
        node=leaf_ticket_ref,
        sources=_urls_or_empty(item.reference_ticket_urls),
        additional_instruction="Ticket sources should show prices consistent with the stated range and include official sale or box office contact details."
    )


async def verify_performance_4(evaluator: Evaluator, parent_node, item: PerformanceItem) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_4_comedy",
        desc="Performance 4 (stand-up comedy; any NYC venue) with all required details and references",
        parent=parent_node,
        critical=False
    )

    # Category constraints
    cat_node = evaluator.add_parallel(
        id="p4_category_constraints",
        desc="Meets Performance 4 category constraints",
        parent=perf_node,
        critical=True
    )

    leaf_type = evaluator.add_leaf(
        id="p4_is_standup_comedy",
        desc="Event is a stand-up comedy performance",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=_show_type_claim("standup_comedy", item.headline),
        node=leaf_type,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction="Confirm it is stand-up comedy; terms like 'comedy show', 'stand-up', 'comedian performance' are acceptable."
    )

    # Required fields
    req_node = evaluator.add_parallel(
        id="p4_required_fields",
        desc="Includes all required fields for Performance 4",
        parent=perf_node,
        critical=True
    )
    evaluator.add_custom_node(_nonempty(item.venue_name), "p4_venue_name", "Venue name is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.full_address), "p4_full_address", "Full venue address is provided", parent=req_node, critical=True)
    leaf_in_nyc = evaluator.add_leaf(id="p4_in_nyc", desc="Venue location is in New York City (NYC)", parent=req_node, critical=True)
    await evaluator.verify(
        claim=_nyc_claim_text(item.full_address),
        node=leaf_in_nyc,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Verify from venue references that the venue is in NYC."
    )
    evaluator.add_custom_node(_nonempty(item.capacity), "p4_exact_capacity", "Exact venue capacity value is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.headline), "p4_comedian_name", "Comedian name is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.date_time), "p4_date_time", "Specific performance date and time are provided", parent=req_node, critical=True)
    leaf_date_range = evaluator.add_leaf(id="p4_date_in_range", desc="Performance date is between December 2025 and January 2026", parent=req_node, critical=True)
    await evaluator.verify(
        claim=_date_in_range_claim(item.date_time),
        node=leaf_date_range,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction=f"Confirm the stated date falls {DATE_RANGE_HUMAN}."
    )
    evaluator.add_custom_node(_nonempty(item.ticket_price_range), "p4_ticket_price_range", "Ticket price range is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.accessibility_feature), "p4_accessibility_feature", "At least one accessibility feature is provided", parent=req_node, critical=True)
    evaluator.add_custom_node(_nonempty(item.box_office_contact), "p4_box_office_contact", "Box office contact information is provided", parent=req_node, critical=True)

    # References verification
    ref_node = evaluator.add_parallel(
        id="p4_references",
        desc="Provides reference URL(s) verifying venue info, show details, and ticketing info for Performance 4",
        parent=perf_node,
        critical=True
    )
    leaf_venue_ref = evaluator.add_leaf(id="p4_venue_reference", desc="Reference URL(s) provided for venue information (address/capacity/location)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_venue_reference_claim(item),
        node=leaf_venue_ref,
        sources=_urls_or_empty(item.reference_venue_urls),
        additional_instruction="Venue references should confirm name, address (NYC), and capacity."
    )
    leaf_show_ref = evaluator.add_leaf(id="p4_show_reference", desc="Reference URL(s) provided for show details (comedian/date/time/type)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_show_reference_claim(item, "standup_comedy"),
        node=leaf_show_ref,
        sources=_urls_or_empty(item.reference_show_urls),
        additional_instruction="Confirm comedian, date/time, venue, and that it's stand-up comedy."
    )
    leaf_ticket_ref = evaluator.add_leaf(id="p4_ticketing_reference", desc="Reference URL(s) provided for ticketing information (price range/box office contact or official ticketing)", parent=ref_node, critical=True)
    await evaluator.verify(
        claim=_ticket_reference_claim(item),
        node=leaf_ticket_ref,
        sources=_urls_or_empty(item.reference_ticket_urls),
        additional_instruction="Ticket sources should show prices consistent with the stated range and include official sale or box office contact details."
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
    Evaluate the answer for the NYC performances task and return the standard summary dict.
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

    # Extract performances
    extraction = await evaluator.extract(
        prompt=prompt_extract_performances(),
        template_class=PerformancesExtraction,
        extraction_name="performances_extraction"
    )

    # Build tree and verify each performance
    # Performance 1 - Broadway Musical
    await verify_performance_1(evaluator, root, extraction.performance1 or PerformanceItem(category="broadway_musical"))

    # Performance 2 - Major Arena Concert
    await verify_performance_2(evaluator, root, extraction.performance2 or PerformanceItem(category="arena_concert"))

    # Performance 3 - Mid-Sized Theater Performance
    await verify_performance_3(evaluator, root, extraction.performance3 or PerformanceItem(category="midsized_theater"))

    # Performance 4 - Comedy Show
    await verify_performance_4(evaluator, root, extraction.performance4 or PerformanceItem(category="standup_comedy"))

    return evaluator.get_summary()