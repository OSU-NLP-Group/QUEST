import asyncio
import logging
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ent_events_2024_2026"
TASK_DESCRIPTION = """
Identify 4 distinct entertainment industry figures or events from the period between January 2024 and March 2026 that experienced major life events, career milestones, or significant developments. For each of the 4 items, provide the following information:

1. Event Details: The specific date (month, day, and year) of the major event and a clear description of what the event was (e.g., death, career announcement, project premiere, deal completion).

2. Location Information: The specific geographic location (city, venue, or country) where the event occurred or where production/activity was based.

3. Impact Description: An explanation of why this event was significant for the person's career or the entertainment industry.

4. Additional Verifiable Detail: At least one additional factual detail about the figure or event that can be verified through reliable sources.

5. Reference URLs: For each piece of information provided, include valid reference URLs from reliable sources that support your claims.

Requirements:
- The 4 items must cover diverse aspects of the entertainment industry (e.g., television, film, streaming, sports entertainment, reality TV)
- All dates must fall between January 1, 2024, and March 19, 2026
- All information must be verifiable through the provided reference URLs
- Each item must be a distinct figure or event (no duplicates or overlapping subjects)
"""

DATE_RANGE_START = date(2024, 1, 1)
DATE_RANGE_END = date(2026, 3, 19)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    """
    One entertainment figure or event item with required fields and URLs.
    All string fields should be extracted exactly as in the answer.
    """
    subject: Optional[str] = None  # The figure or event name/title (e.g., "Matthew Perry", "2025 Oscars")
    category: Optional[str] = None  # e.g., film, television, streaming, sports entertainment, reality TV, music, awards, festival, theater, gaming
    event_date: Optional[str] = None  # e.g., "March 5, 2025"
    event_description: Optional[str] = None  # e.g., "announced retirement", "won Best Actor", "series premiere"
    event_urls: List[str] = Field(default_factory=list)  # URLs supporting event details/date/type

    location: Optional[str] = None  # city/venue/country or production base
    location_urls: List[str] = Field(default_factory=list)  # URLs supporting location info

    impact: Optional[str] = None  # why it was significant/notable
    impact_urls: List[str] = Field(default_factory=list)  # URLs supporting impact significance

    additional_detail: Optional[str] = None  # any additional factual detail
    additional_detail_urls: List[str] = Field(default_factory=list)  # URLs supporting additional detail


class EventsExtraction(BaseModel):
    """Collection of extracted items."""
    items: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to 6 entertainment industry figures or events mentioned in the answer that occurred between January 2024 and March 2026.
    For each item, extract the following fields exactly as stated in the answer:

    - subject: The person's name or event title (e.g., "Taylor Swift", "2025 Oscars", "Dune: Part Two").
    - category: The sector/category (e.g., film, television, streaming, sports entertainment, reality TV, music, awards, festival, theater, gaming). Choose a concise single label if possible.
    - event_date: The specific date of the major event, including month, day, and year (e.g., "March 5, 2025"). If the day is omitted in the answer, still extract what is present.
    - event_description: A clear description of what the event was (e.g., death, announcement, premiere, deal signing, award win).
    - event_urls: All URLs that support the event details and/or date/type.

    - location: The specific city/venue/country or production location relevant to the event.
    - location_urls: All URLs that support the location information.

    - impact: Why the event was significant for the person's career or the entertainment industry.
    - impact_urls: All URLs that support the impact/significance claim.

    - additional_detail: At least one additional verifiable factual detail about the figure or event (e.g., box office number, role name, network, contract value, lineup details).
    - additional_detail_urls: All URLs that support the additional detail.

    IMPORTANT:
    - Only include URLs that actually appear in the answer. Do not invent URLs.
    - Include full valid URLs starting with http:// or https://. Ignore obviously invalid or malformed URLs.
    - If any field is missing in the answer, set it to null (for strings) or [] (for URL arrays).
    - Keep all text exactly as written in the answer; do not rewrite or normalize.
    - Return a JSON object with a single key "items" as an array of objects with the fields defined.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
DATE_PATTERNS = [
    "%B %d, %Y",   # March 5, 2025
    "%b %d, %Y",   # Mar 5, 2025
    "%Y-%m-%d",    # 2025-03-05
    "%m/%d/%Y",    # 03/05/2025
    "%d %B %Y",    # 5 March 2025
    "%d %b %Y",    # 5 Mar 2025
]


def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return bool(re.match(r"^https?://", url.strip()))


def sanitize_urls(urls: List[str]) -> List[str]:
    return [u.strip() for u in urls if is_valid_url(u)]


def try_parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip()
    for fmt in DATE_PATTERNS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date()
        except Exception:
            continue
    # Try to loosely detect "Month Day, Year" even with extra whitespace or ordinal (e.g., "March 5th, 2025")
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,\s*(\d{4})", s)
    if m:
        try:
            mon_name = m.group(1)
            day = int(m.group(2))
            year = int(m.group(3))
            dt = datetime.strptime(f"{mon_name} {day} {year}", "%B %d %Y")
            return dt.date()
        except Exception:
            pass
    # ISO-like with missing zeros: 2025-3-5
    m2 = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m2:
        try:
            y, mth, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            return date(y, mth, d)
        except Exception:
            pass
    return None


def has_complete_month_day_year(date_str: Optional[str]) -> bool:
    """Check that the date string likely contains month, day, and year."""
    if not date_str or not isinstance(date_str, str):
        return False
    parsed = try_parse_date(date_str)
    return parsed is not None  # Parsed formats all include day explicitly


def within_range(parsed: Optional[date], start: date, end: date) -> bool:
    if parsed is None:
        return False
    return start <= parsed <= end


def norm_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def normalize_category(cat: Optional[str]) -> str:
    c = norm_text(cat)
    # Basic normalization map
    map_pairs = {
        "tv": "television",
        "television": "television",
        "film": "film",
        "movie": "film",
        "movies": "film",
        "cinema": "film",
        "streaming": "streaming",
        "sports entertainment": "sports entertainment",
        "wwe": "sports entertainment",
        "ufc": "sports entertainment",
        "reality": "reality tv",
        "reality tv": "reality tv",
        "music": "music",
        "awards": "awards",
        "festival": "festival",
        "festivals": "festival",
        "gaming": "gaming",
        "theater": "theater",
        "theatre": "theater",
        "live events": "live events",
    }
    return map_pairs.get(c, c)


# --------------------------------------------------------------------------- #
# Verification logic per item                                                 #
# --------------------------------------------------------------------------- #
async def verify_item(
    evaluator: Evaluator,
    parent_node,
    item: EventItem,
    idx: int,
) -> None:
    """
    Build verification sub-tree for one item.
    """
    # Item node (parallel, non-critical)
    item_node = evaluator.add_parallel(
        id=f"Item_{idx+1}",
        desc=f"Item {idx+1}: Entertainment figure or event within 2024-2026",
        parent=parent_node,
        critical=False,
    )

    # Normalize URLs now (avoid malformed)
    event_urls = sanitize_urls(item.event_urls or [])
    location_urls = sanitize_urls(item.location_urls or [])
    impact_urls = sanitize_urls(item.impact_urls or [])
    detail_urls = sanitize_urls(item.additional_detail_urls or [])

    # -------------------- Event Details -------------------- #
    event_details_node = evaluator.add_parallel(
        id=f"Item_{idx+1}_Event_Details",
        desc="Specific event type, date, and nature of the major development",
        parent=item_node,
        critical=True,
    )

    # Event Date Validation (parallel, critical)
    date_validation_node = evaluator.add_parallel(
        id=f"Item_{idx+1}_Event_Date_Validation",
        desc="Validation of the event date",
        parent=event_details_node,
        critical=True,
    )

    # Date completeness (custom boolean)
    evaluator.add_custom_node(
        result=has_complete_month_day_year(item.event_date),
        id=f"Item_{idx+1}_Date_Completeness",
        desc="Date includes month, day, and year",
        parent=date_validation_node,
        critical=True,
    )

    # Date range validity (custom boolean)
    parsed_date = try_parse_date(item.event_date)
    evaluator.add_custom_node(
        result=within_range(parsed_date, DATE_RANGE_START, DATE_RANGE_END),
        id=f"Item_{idx+1}_Date_Range_Validity",
        desc="Date falls between January 1, 2024, and March 19, 2026",
        parent=date_validation_node,
        critical=True,
    )

    # Event type/description presence
    evaluator.add_custom_node(
        result=bool(item.event_description and item.event_description.strip()),
        id=f"Item_{idx+1}_Event_Type_Description",
        desc="Clear description of what the major event was (e.g., death, career milestone, announcement)",
        parent=event_details_node,
        critical=True,
    )

    # Event URLs presence (acts as gate for URL-grounded verification)
    evaluator.add_custom_node(
        result=len(event_urls) > 0,
        id=f"Item_{idx+1}_Event_URLs_Present",
        desc="At least one valid reference URL provided for event details",
        parent=event_details_node,
        critical=True,
    )

    # Event Reference URL verification (URL grounded)
    event_ref_leaf = evaluator.add_leaf(
        id=f"Item_{idx+1}_Event_Reference_URL",
        desc="Valid reference URL supporting the event details",
        parent=event_details_node,
        critical=True,
    )
    subj = item.subject or "the subject"
    date_str = item.event_date or "the stated date"
    ev_desc = item.event_description or "a major development"
    event_claim = f"On {date_str}, {subj} {ev_desc}."
    await evaluator.verify(
        claim=event_claim,
        node=event_ref_leaf,
        sources=event_urls,
        additional_instruction="Verify that at least one cited source explicitly supports the described event and (approximately) the same date. Allow reasonable timezone/date reporting differences (±1 day).",
    )

    # -------------------- Location Information -------------------- #
    location_node = evaluator.add_parallel(
        id=f"Item_{idx+1}_Location_Information",
        desc="Geographic location or production location details relevant to the event",
        parent=item_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(item.location and item.location.strip()),
        id=f"Item_{idx+1}_Location_Specificity",
        desc="Specific location (city, venue, or country) where event occurred or was based",
        parent=location_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(location_urls) > 0,
        id=f"Item_{idx+1}_Location_URLs_Present",
        desc="At least one valid reference URL provided for the location information",
        parent=location_node,
        critical=True,
    )

    loc_ref_leaf = evaluator.add_leaf(
        id=f"Item_{idx+1}_Location_Reference_URL",
        desc="Valid reference URL supporting location information",
        parent=location_node,
        critical=True,
    )
    loc_claim = f"The event involving {subj} took place in/at {item.location}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_ref_leaf,
        sources=location_urls if location_urls else event_urls,
        additional_instruction="Confirm that a cited source explicitly mentions the same location (city/venue/country or production base) for this event.",
    )

    # -------------------- Impact Details -------------------- #
    impact_node = evaluator.add_parallel(
        id=f"Item_{idx+1}_Impact_Details",
        desc="Description of the significance or impact of the event on the person's career or life",
        parent=item_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(item.impact and item.impact.strip()),
        id=f"Item_{idx+1}_Impact_Description",
        desc="Clear explanation of why this event was significant",
        parent=impact_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(impact_urls) > 0 or len(event_urls) > 0,  # allow fallback to event URLs if impact URLs missing
        id=f"Item_{idx+1}_Impact_URLs_Present",
        desc="At least one valid reference URL provided for the impact/significance (or event URL that discusses impact)",
        parent=impact_node,
        critical=True,
    )

    impact_ref_leaf = evaluator.add_leaf(
        id=f"Item_{idx+1}_Impact_Reference_URL",
        desc="Valid reference URL supporting impact information",
        parent=impact_node,
        critical=True,
    )
    impact_claim = f"This event was significant for {subj} or the entertainment industry because: {item.impact or 'the stated reason'}."
    await evaluator.verify(
        claim=impact_claim,
        node=impact_ref_leaf,
        sources=impact_urls if impact_urls else event_urls,
        additional_instruction="Check that the cited source(s) explicitly support or substantiate the stated significance/impact (e.g., career milestone, industry shift, major deal).",
    )

    # -------------------- Additional Verifiable Detail -------------------- #
    add_detail_node = evaluator.add_parallel(
        id=f"Item_{idx+1}_Additional_Verifiable_Detail",
        desc="At least one additional verifiable fact about the figure or event",
        parent=item_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(detail_urls) > 0,
        id=f"Item_{idx+1}_Detail_URLs_Present",
        desc="At least one valid reference URL provided for the additional detail",
        parent=add_detail_node,
        critical=True,
    )

    detail_leaf = evaluator.add_leaf(
        id=f"Item_{idx+1}_Detail_Accuracy",
        desc="Accuracy of the additional detail provided",
        parent=add_detail_node,
        critical=True,
    )
    add_detail_claim = f"Additional factual detail for {subj}: {item.additional_detail or 'the stated detail'}."
    await evaluator.verify(
        claim=add_detail_claim,
        node=detail_leaf,
        sources=detail_urls,
        additional_instruction="Verify that at least one cited source explicitly supports the additional detail exactly or near-exactly.",
    )


# --------------------------------------------------------------------------- #
# Cross-item validation                                                       #
# --------------------------------------------------------------------------- #
def diversity_check(items: List[EventItem]) -> bool:
    """
    Heuristic: at least 3 unique normalized categories across the 4 items.
    """
    cats = set()
    for it in items:
        c = normalize_category(it.category)
        if c:
            cats.add(c)
    return len(cats) >= 3


def distinctness_check(items: List[EventItem]) -> bool:
    """
    Check subjects are distinct (ignoring empty).
    """
    names = [norm_text(it.subject) for it in items if norm_text(it.subject)]
    return len(names) == len(set(names))


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
    Evaluate an answer for the 'Entertainment_Industry_Events_2024_2026' task.
    """
    # Initialize evaluator (root parallel per rubric)
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

    # Add bounds info as ground truth-style metadata
    evaluator.add_ground_truth({
        "date_bounds": {
            "start": DATE_RANGE_START.isoformat(),
            "end": DATE_RANGE_END.isoformat()
        },
        "requirements": [
            "4 distinct items",
            "Diverse categories across items",
            "Each item includes event date, description, location, impact, additional detail, and supporting URLs"
        ]
    }, gt_type="task_requirements")

    # Extract structured items
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Keep only the first 4 items, pad if fewer
    items = list(extracted.items[:4])
    while len(items) < 4:
        items.append(EventItem())

    # Record some custom info
    evaluator.add_custom_info({
        "detected_items": len(extracted.items),
        "used_items": 4,
        "categories_used": [normalize_category(it.category) for it in items],
        "subjects_used": [it.subject for it in items],
    }, info_type="extraction_summary")

    # -------------------- Cross-item validation (critical) -------------------- #
    cross_node = evaluator.add_parallel(
        id="Cross_Item_Validation",
        desc="Validation of requirements that apply across all 4 items collectively",
        parent=root,
        critical=True,
    )

    # Diversity check (leaf). Use a custom node for strict binary judgment.
    evaluator.add_custom_node(
        result=diversity_check(items),
        id="Diversity_Check",
        desc="Verify that the 4 items collectively cover diverse aspects of the entertainment industry (e.g., television, film, streaming, sports entertainment, reality TV)",
        parent=cross_node,
        critical=True,
    )

    # Distinctness check (leaf). Use a custom node for strict binary judgment.
    evaluator.add_custom_node(
        result=distinctness_check(items),
        id="Distinctness_Check",
        desc="Verify that each item represents a distinct figure or event with no duplicates or overlapping subjects",
        parent=cross_node,
        critical=True,
    )

    # -------------------- Per-item verification -------------------- #
    for i in range(4):
        await verify_item(evaluator, root, items[i], i)

    # Return summary
    return evaluator.get_summary()