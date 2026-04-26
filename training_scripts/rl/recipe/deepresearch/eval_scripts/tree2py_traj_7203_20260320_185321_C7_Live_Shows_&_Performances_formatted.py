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
TASK_ID = "nyc_live_performances_mar_2026"
TASK_DESCRIPTION = """
Find three different types of live performances taking place in New York City between March 20-31, 2026: one Broadway show, one stand-up comedy show, and one music concert. For each performance, provide the following information:
(1) The name of the show/performer,
(2) The venue name where the performance will take place,
(3) The complete street address of the venue,
(4) The exact date and start time of the performance,
(5) A URL where tickets can be purchased,
(6) Ticket pricing information (starting price, price range, or specific price points available).
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PerformanceItem(BaseModel):
    # General fields for any performance
    category: Optional[str] = None  # e.g., "broadway", "comedy", "concert"
    name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None  # full street address
    date: Optional[str] = None           # e.g., "March 24, 2026" or "2026-03-24"
    start_time: Optional[str] = None     # e.g., "7:00 PM", "19:00"
    datetime: Optional[str] = None       # optional combined "2026-03-24 19:00"
    ticket_url: Optional[str] = None     # primary ticket purchase URL
    other_urls: List[str] = Field(default_factory=list)  # any additional relevant URLs cited
    ticket_price: Optional[str] = None   # e.g., "from $59", "$45–$120", or "$75"


class PerformancesExtraction(BaseModel):
    broadway: Optional[PerformanceItem] = None
    comedy: Optional[PerformanceItem] = None
    concert: Optional[PerformanceItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_performances() -> str:
    return """
    Your task is to extract exactly one item for each of the following three categories from the provided answer:
    - Broadway show (category: "broadway")
    - Stand-up comedy show (category: "comedy")
    - Music concert (category: "concert")

    For each of the three categories, extract a JSON object with these fields:
    - category: one of "broadway", "comedy", or "concert". If unclear, infer from the context of the answer.
    - name: the show/performer/concert name exactly as written in the answer.
    - venue_name: the specific venue or theater name.
    - venue_address: the complete street address of the venue as stated in the answer.
    - date: the performance date as a human-readable string (e.g., "March 24, 2026" or "2026-03-24").
    - start_time: the start time as a string (e.g., "7:00 PM", "20:00").
    - datetime: if the answer already provides a combined date-time string, include it here; otherwise set to null.
    - ticket_url: a URL explicitly presented in the answer where tickets can be purchased for the performance.
      If multiple URLs are given, choose the main ticket-purchase page for this performance.
    - other_urls: an array of any additional URLs mentioned in the answer that are relevant to the performance (e.g., event page, venue page).
      Do not invent URLs; include only URLs that actually appear in the answer text.
    - ticket_price: the ticket pricing information as text (e.g., "from $59", "$45–$120", "$75", or similar).

    IMPORTANT:
    - Extract only information explicitly present in the answer text.
    - If multiple candidate items are listed for a category, choose the first one mentioned in the answer.
    - If a field is missing in the answer, set it to null (or [] for other_urls).
    - Do not invent any information.
    - For URL fields, extract valid full URLs that actually appear in the answer. If a URL lacks a protocol, prepend "http://".
    - Return a JSON object with three top-level fields: "broadway", "comedy", and "concert", each holding the corresponding PerformanceItem or null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def combine_sources(item: Optional[PerformanceItem]) -> List[str]:
    if not item:
        return []
    urls: List[str] = []
    if item.ticket_url and item.ticket_url.strip():
        urls.append(item.ticket_url.strip())
    urls.extend(item.other_urls or [])
    return _dedup_preserve_order(urls)


def format_dt_snippet(item: Optional[PerformanceItem], label: str) -> str:
    if not item:
        return f"{label}: (missing)"
    if item.datetime and item.datetime.strip():
        return f"{label}: {item.datetime.strip()}"
    # Fallback to date + start_time
    date = (item.date or "").strip()
    time = (item.start_time or "").strip()
    if date and time:
        return f"{label}: {date} at {time}"
    elif date:
        return f"{label}: {date} (time missing)"
    elif time:
        return f"{label}: (date missing) at {time}"
    else:
        return f"{label}: (date/time missing)"


# --------------------------------------------------------------------------- #
# Verification logic per category                                             #
# --------------------------------------------------------------------------- #
async def verify_performance(
    evaluator: Evaluator,
    parent_node,
    item: PerformanceItem,
    category_id: str,
    category_human: str,
) -> Dict[str, Any]:
    """
    Build verification subtree for a single performance category and run URL-grounded checks.
    Returns a dict of created important leaf nodes for potential cross-check prerequisites.
    """
    # Parent category node (parallel)
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=f"Complete information for the {category_human} performance",
        parent=parent_node,
        critical=False
    )

    sources_all = combine_sources(item)

    # 1) Ticket URL (critical)
    ticket_url_leaf = evaluator.add_leaf(
        id=f"{category_id}_Ticket_URL",
        desc=f"Provides a valid URL where tickets for the {category_human.lower()} can be purchased",
        parent=cat_node,
        critical=True,
    )
    ticket_claim = (
        f"This URL is a valid page where tickets for the {category_human.lower()} can be purchased."
    )
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_url_leaf,
        sources=item.ticket_url or None,
        additional_instruction=(
            "Assess whether the page is a ticket purchasing page for the specified performance "
            "or venue/date, or a direct ticket vendor listing (e.g., Ticketmaster, Telecharge, SeatGeek, venue's official ticketing). "
            "Look for signals like 'Buy Tickets', 'Find Tickets', 'Get Tickets', seat selection, or cart/checkout. "
            "If the URL is missing or invalid, this should fail."
        ),
    )

    # 2) Performance name (critical)
    name_leaf = evaluator.add_leaf(
        id=f"{category_id}_Broadway_Show_Performance_Name" if category_id == "Broadway_Show"
           else (f"{category_id}_Comedy_Show_Performance_Name" if category_id == "Comedy_Show"
                 else f"{category_id}_Concert_Performance_Name"),
        desc=f"Provides the name of the {category_human.lower()}",
        parent=cat_node,
        critical=True,
    )
    name_claim = (
        f"On the provided page(s), the performance/show name matches '{(item.name or '').strip()}'. "
        "Allow minor formatting, case, or punctuation variations."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=sources_all if sources_all else None,
        extra_prerequisites=[ticket_url_leaf],
        additional_instruction=(
            "Verify that the main show or performer name on the page matches the provided name. "
            "Accept minor variants (case, punctuation, middle initials)."
        ),
    )

    # 3) Venue name (critical)
    venue_name_leaf = evaluator.add_leaf(
        id=f"{category_id}_Broadway_Show_Venue_Name" if category_id == "Broadway_Show"
           else (f"{category_id}_Comedy_Show_Venue_Name" if category_id == "Comedy_Show"
                 else f"{category_id}_Concert_Venue_Name"),
        desc=f"Provides the specific venue name where the {category_human.lower()} will be performed",
        parent=cat_node,
        critical=True,
    )
    venue_name_claim = (
        f"On the provided page(s), the venue for this {category_human.lower()} is '{(item.venue_name or '').strip()}'. "
        "Allow minor naming variants but it should clearly match the same venue."
    )
    await evaluator.verify(
        claim=venue_name_claim,
        node=venue_name_leaf,
        sources=sources_all if sources_all else None,
        extra_prerequisites=[ticket_url_leaf],
        additional_instruction="Confirm the venue name as shown on the ticket/event page.",
    )

    # 4) Venue address (critical)
    venue_addr_leaf = evaluator.add_leaf(
        id=f"{category_id}_Broadway_Show_Venue_Address" if category_id == "Broadway_Show"
           else (f"{category_id}_Comedy_Show_Venue_Address" if category_id == "Comedy_Show"
                 else f"{category_id}_Concert_Venue_Address"),
        desc=f"Provides the complete street address of the {category_human.lower()} venue",
        parent=cat_node,
        critical=True,
    )
    venue_addr_claim = (
        f"The full street address for the venue on the provided page(s) is '{(item.venue_address or '').strip()}'. "
        "Small stylistic differences (e.g., 'St' vs 'Street') are acceptable."
    )
    await evaluator.verify(
        claim=venue_addr_claim,
        node=venue_addr_leaf,
        sources=sources_all if sources_all else None,
        extra_prerequisites=[ticket_url_leaf],
        additional_instruction="Match the venue street address as listed on the ticketing or official event page.",
    )

    # 5) Date and time (critical)
    dt_leaf = evaluator.add_leaf(
        id=f"{category_id}_Broadway_Show_Date_and_Time" if category_id == "Broadway_Show"
           else (f"{category_id}_Comedy_Show_Date_and_Time" if category_id == "Comedy_Show"
                 else f"{category_id}_Concert_Date_and_Time"),
        desc=f"Provides the exact date and start time of the {category_human.lower()} performance",
        parent=cat_node,
        critical=True,
    )
    # Build a robust claim with both date and time if available
    if item.datetime and item.datetime.strip():
        dt_text = item.datetime.strip()
        dt_claim = f"The performance is scheduled on '{dt_text}' as shown on the provided page(s)."
    else:
        date_text = (item.date or "").strip()
        time_text = (item.start_time or "").strip()
        if date_text and time_text:
            dt_claim = (
                f"The performance date and start time on the provided page(s) are '{date_text}' at '{time_text}'."
            )
        elif date_text:
            dt_claim = f"The performance date on the provided page(s) is '{date_text}' (start time possibly not shown)."
        elif time_text:
            dt_claim = f"The performance start time on the provided page(s) is '{time_text}' (date possibly not shown)."
        else:
            dt_claim = "The performance date and time are not present on the provided page(s)."
    await evaluator.verify(
        claim=dt_claim,
        node=dt_leaf,
        sources=sources_all if sources_all else None,
        extra_prerequisites=[ticket_url_leaf],
        additional_instruction=(
            "Confirm the exact scheduled date and start time for this performance on the ticket/event page. "
            "If time zone is not explicit, assume America/New_York (ET). Minor formatting differences are acceptable."
        ),
    )

    # 6) Ticket pricing (critical)
    price_leaf = evaluator.add_leaf(
        id=f"{category_id}_Broadway_Show_Ticket_Pricing" if category_id == "Broadway_Show"
           else (f"{category_id}_Comedy_Show_Ticket_Pricing" if category_id == "Comedy_Show"
                 else f"{category_id}_Concert_Ticket_Pricing"),
        desc=f"Provides ticket price information for the {category_human.lower()} (starting price, price range, or price points)",
        parent=cat_node,
        critical=True,
    )
    price_text = (item.ticket_price or "").strip()
    price_claim = (
        f"The ticket pricing information on the provided page(s) matches or clearly supports '{price_text}'. "
        "This may be a starting price, a price range, or representative price points."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=sources_all if sources_all else None,
        extra_prerequisites=[ticket_url_leaf],
        additional_instruction=(
            "Validate the presence of pricing info (e.g., 'from $X', ranges like '$A–$B', or specific price points). "
            "Minor currency formatting differences and fees disclosures are acceptable."
        ),
    )

    return {
        "category_node": cat_node,
        "ticket_url_leaf": ticket_url_leaf,
        "name_leaf": name_leaf,
        "venue_name_leaf": venue_name_leaf,
        "address_leaf": venue_addr_leaf,
        "datetime_leaf": dt_leaf,
        "pricing_leaf": price_leaf,
    }


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
    Evaluate an answer for the NYC live performances (Mar 20–31, 2026) task.
    """
    # Initialize evaluator with parallel root (constraints are critical children)
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_performances(),
        template_class=PerformancesExtraction,
        extraction_name="performances_extraction",
    )

    # Add ground truth info (constraints context)
    evaluator.add_ground_truth({
        "required_location": "New York City (any of the five boroughs: Manhattan, Brooklyn, Queens, The Bronx, Staten Island), NY, USA",
        "required_date_range_inclusive": "2026-03-20 to 2026-03-31",
        "required_categories": ["broadway", "comedy", "concert"]
    })

    # Normalize items (avoid None references)
    broadway = extracted.broadway or PerformanceItem(category="broadway")
    comedy = extracted.comedy or PerformanceItem(category="comedy")
    concert = extracted.concert or PerformanceItem(category="concert")

    # Build verification subtrees for each category
    b_nodes = await verify_performance(
        evaluator, root, broadway, category_id="Broadway_Show", category_human="Broadway show"
    )
    c_nodes = await verify_performance(
        evaluator, root, comedy, category_id="Comedy_Show", category_human="stand-up comedy show"
    )
    m_nodes = await verify_performance(
        evaluator, root, concert, category_id="Concert", category_human="music concert"
    )

    # --------------------- Global critical constraints --------------------- #

    # Geographic Constraint (critical)
    geo_leaf = evaluator.add_leaf(
        id="Geographic_Constraint",
        desc="Verifies that all three performances take place in New York City",
        parent=root,
        critical=True,
    )
    geo_claim = (
        "All three selected performances take place in New York City (i.e., within the five boroughs: "
        "Manhattan, Brooklyn, Queens, The Bronx, or Staten Island). "
        f"Addresses:\n- {broadway.venue_address or '(missing)'}\n"
        f"- {comedy.venue_address or '(missing)'}\n"
        f"- {concert.venue_address or '(missing)'}"
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        additional_instruction=(
            "Judge based on the textual addresses in the answer. Any address in NYC (including borough names) counts. "
            "If any address is outside NYC or missing, this should be marked incorrect."
        ),
        # Make this dependent on successful address verifications (URL-grounded)
        extra_prerequisites=[b_nodes["address_leaf"], c_nodes["address_leaf"], m_nodes["address_leaf"]],
    )

    # Temporal Constraint (critical)
    time_leaf = evaluator.add_leaf(
        id="Temporal_Constraint",
        desc="Verifies that all three performances occur between March 20-31, 2026",
        parent=root,
        critical=True,
    )
    time_claim = (
        "Each selected performance occurs between March 20 and March 31, 2026 (inclusive). "
        + format_dt_snippet(broadway, "Broadway")
        + "; "
        + format_dt_snippet(comedy, "Comedy")
        + "; "
        + format_dt_snippet(concert, "Concert")
        + "."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        additional_instruction=(
            "Evaluate using the provided date/time strings from the answer (assume local ET if unspecified). "
            "All three must fall within 2026-03-20 to 2026-03-31 inclusive for this to be correct."
        ),
        # Dependent on each item's date/time having been URL-verified
        extra_prerequisites=[b_nodes["datetime_leaf"], c_nodes["datetime_leaf"], m_nodes["datetime_leaf"]],
    )

    # Category Diversity Constraint (critical)
    # Check that we have exactly one item for each required category.
    has_broadway = bool((broadway.name or "").strip())
    has_comedy = bool((comedy.name or "").strip())
    has_concert = bool((concert.name or "").strip())
    diversity_ok = has_broadway and has_comedy and has_concert

    evaluator.add_custom_node(
        result=diversity_ok,
        id="Category_Diversity_Constraint",
        desc="Verifies that the three performances include exactly one Broadway show, one stand-up comedy show, and one music concert",
        parent=root,
        critical=True
    )

    # Return final structured evaluation summary
    return evaluator.get_summary()