import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "woodworking_events_2026"
TASK_DESCRIPTION = """A Texas-based professional woodworking artisan specializing in contemporary furniture and wood turning is planning to participate in three major juried craft shows or competitions in 2026. They have the following requirements:

1. Event Selection Criteria:
- Must participate in exactly 3 different events
- At least one event must be in California
- At least one event must be a dedicated woodworking competition offering cash prizes
- At least one event must be a general craft fair accepting wood/furniture media
- Events must be spread across different months (no two events in the same month)
- All events must be juried (have a selection/acceptance process)

2. Budget Constraints:
- Total booth/entry fees for all 3 events combined must not exceed $2,100
- Individual event booth/entry fee must be documented with exact amount

3. Timing Requirements:
- All events must occur between February 2026 and September 2026 (inclusive)
- Application deadlines for all events must be before June 1, 2026

4. Eligibility and Requirements:
- All events must accept woodworking or furniture as an eligible medium/category
- Artisan must meet age requirement (18+) for all events
- All events must have public information available online about booth fees, dates, and application requirements

5. State Compliance:
- For each event, identify whether a state sales tax permit is required for vendors from Texas
- For each event, identify if liability insurance is required or recommended

For each of the 3 events, provide:
- Event name
- Exact dates (start and end date)
- Location (city and state)
- Venue name
- Booth/entry fee amount (in dollars)
- Application deadline
- Whether the event is juried (and brief description of jury process if available)
- Eligible media/categories that include woodworking
- State sales tax permit requirement for Texas vendors
- Insurance requirement or recommendation
- Reference URL to the official event website or detailed information page
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue: Optional[str] = None
    fee_amount: Optional[str] = None
    application_deadline: Optional[str] = None
    juried: Optional[str] = None
    jury_process: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    is_dedicated_wood_competition: Optional[str] = None
    cash_prizes: Optional[str] = None
    sales_tax_permit_tx: Optional[str] = None
    insurance: Optional[str] = None
    url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    events: List[EventInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to 5 candidate 2026 events (craft shows, festivals, or woodworking competitions) mentioned in the answer.
    For each event, return a JSON object with the following fields:
    - name: The official event name as written in the answer
    - start_date: The start date string as written (e.g., "March 15, 2026")
    - end_date: The end date string as written
    - city: The city
    - state: The state (use two-letter abbreviation if present, otherwise full state name)
    - venue: The venue name if provided
    - fee_amount: The booth or entry fee amount as written (e.g., "$650", "$450-$700"); include the numeric value and currency symbol if present
    - application_deadline: The application or entry deadline as written
    - juried: "yes" if the event is juried, "no" otherwise, or null if not stated
    - jury_process: Brief description of the jury process if mentioned
    - categories: A list of eligible media/categories as provided (e.g., ["wood", "furniture", "sculpture"])
    - is_dedicated_wood_competition: "yes" if this is specifically a woodworking competition, else "no" (or null if not stated)
    - cash_prizes: "yes" if cash prizes are offered, else "no" (or null if not stated)
    - sales_tax_permit_tx: The answer's stated requirement for a state sales tax/seller's permit for vendors (e.g., "required", "not required", "TBD"); include phrasing as in the answer
    - insurance: The answer's stated requirement/recommendation for vendor liability insurance (e.g., "required", "recommended", "not required", "TBD")
    - url: A single primary official or detailed event information URL
    - extra_urls: An array of any additional URLs provided that contain vendor info, prospectus, rules, or applications for this event

    Notes:
    - Only extract what is explicitly present in the answer; do not invent missing values (use null or empty list).
    - If multiple fee tiers exist, still record the exact fee figure(s) the answer presents for the artisan’s planned participation.
    - Preserve date strings and fee strings exactly as presented in the answer.
    - Prefer the event’s official website, application portal, or prospectus page for the url field when available.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
MONTH_NAME_TO_INT = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}


def _parse_money_to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    # Find first currency-like number
    match = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)", s.replace("$", ""))
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
        return value
    except Exception:
        return None


def _month_from_date(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    s = date_str.strip()
    # Try month names
    for name, idx in MONTH_NAME_TO_INT.items():
        if re.search(rf"\b{name[:3]}\w*\b", s, flags=re.IGNORECASE):
            return idx
    # Try ISO formats YYYY-MM-DD
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        return dt.month
    except Exception:
        pass
    # Try common US formats like MM/DD/YYYY or M/D/YYYY
    try:
        dt = datetime.strptime(s.strip(), "%m/%d/%Y")
        return dt.month
    except Exception:
        pass
    # Try "Month D–D, YYYY" without year on start
    # If the year is at end, we might not need it here for month
    return None


def _gather_sources_for_event(ev: EventInfo) -> List[str]:
    srcs: List[str] = []
    if ev.url and isinstance(ev.url, str) and ev.url.strip():
        srcs.append(ev.url.strip())
    for u in ev.extra_urls:
        if u and isinstance(u, str) and u.strip():
            srcs.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in srcs:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def _safe(s: Optional[str]) -> str:
    return s if s is not None else ""


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_event(
    evaluator: Evaluator,
    parent_node,
    ev: EventInfo,
    index: int,
    all_events: List[EventInfo],
) -> None:
    """
    Build and verify the subtree for a single event (event_{index+1}).
    Structure mirrors the rubric: identification -> (basic_info, timing, financial, eligibility, compliance)
    """
    event_idx = index + 1
    ev_node = evaluator.add_sequential(
        id=f"event_{event_idx}",
        desc=f"{['First','Second','Third'][index]} event identification and verification",
        parent=parent_node,
        critical=False
    )

    identification_node = evaluator.add_parallel(
        id=f"event_{event_idx}_identification",
        desc="Event is correctly identified as a qualifying 2026 craft show or competition",
        parent=ev_node,
        critical=True
    )

    # ---------------- Basic Info ----------------
    basic_node = evaluator.add_parallel(
        id=f"event_{event_idx}_basic_info",
        desc="Basic event information is accurate",
        parent=identification_node,
        critical=True
    )

    # URL existence gate (extra safety precondition)
    url_exists_node = evaluator.add_custom_node(
        result=(ev.url is not None and ev.url.strip() != ""),
        id=f"event_{event_idx}_url_exists",
        desc="Reference URL is provided (non-empty)",
        parent=basic_node,
        critical=True
    )

    sources = _gather_sources_for_event(ev)

    # Name
    name_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_name",
        desc="Event name is correctly identified",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official event name shown on the page is '{_safe(ev.name)}'.",
        node=name_leaf,
        sources=sources,
        additional_instruction="Allow minor variations such as inclusion of year or 'Annual' qualifiers; confirm the core event name matches."
    )

    # Dates
    dates_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_dates",
        desc="Event dates (start and end) are correctly identified",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event dates on the referenced page are from '{_safe(ev.start_date)}' to '{_safe(ev.end_date)}'.",
        node=dates_leaf,
        sources=sources,
        additional_instruction="If the page presents a date range, ensure it matches or is equivalent to the cited start and end dates."
    )

    # Location
    loc_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_location",
        desc="City and state are correctly identified",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event location is '{_safe(ev.city)}', '{_safe(ev.state)}'.",
        node=loc_leaf,
        sources=sources,
        additional_instruction="Check the page for the city and state; allow short forms (e.g., 'CA' vs 'California')."
    )

    # Venue
    venue_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_venue",
        desc="Venue name is correctly identified",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue for the event is '{_safe(ev.venue)}'.",
        node=venue_leaf,
        sources=sources,
        additional_instruction="Confirm that the cited venue name appears as the event venue or host location."
    )

    # URL validity
    url_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_url",
        desc="Valid reference URL to official event information is provided",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage provides official event information (dates, fees, and/or application) for '{_safe(ev.name)}'.",
        node=url_leaf,
        sources=ev.url,
        additional_instruction="Verify that this page is an official or primary information source (e.g., event website, prospectus, application portal)."
    )

    # ---------------- Timing Constraints ----------------
    timing_node = evaluator.add_parallel(
        id=f"event_{event_idx}_timing_constraints",
        desc="Event meets all timing requirements",
        parent=identification_node,
        critical=True
    )

    timeframe_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_timeframe",
        desc="Event occurs between February 2026 and September 2026 (inclusive)",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim="Based on the dates shown, the event occurs between February and September 2026 (inclusive).",
        node=timeframe_leaf,
        sources=sources,
        additional_instruction="Confirm that both start and end dates fall within 2026-02-01 and 2026-09-30 inclusive."
    )

    deadline_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_application_deadline",
        desc="Application deadline is correctly identified and is before June 1, 2026",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The application deadline is '{_safe(ev.application_deadline)}' and occurs before June 1, 2026.",
        node=deadline_leaf,
        sources=sources,
        additional_instruction="Find the application/entry deadline on the page and confirm it is earlier than 2026-06-01."
    )

    # Month unique vs other two events (logical check across events)
    months = [_month_from_date(e.start_date) for e in all_events]
    my_month = months[index]
    other_months = [m for i, m in enumerate(months) if i != index]
    month_unique_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_month_unique",
        desc="Event occurs in a different month than the other two events",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Event #{event_idx} starts in month {my_month if my_month else 'unknown'}, which is different from the months of the other two events {other_months}.",
        node=month_unique_leaf,
        sources=None,
        additional_instruction="Treat this as a logical cross-check using the extracted start months; result is correct if all three months are present and pairwise different."
    )

    # ---------------- Financial ----------------
    financial_node = evaluator.add_parallel(
        id=f"event_{event_idx}_financial",
        desc="Financial information is accurate and meets budget constraints",
        parent=identification_node,
        critical=True
    )

    fee_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_fee_amount",
        desc="Booth or entry fee amount is correctly documented in dollars",
        parent=financial_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The booth/entry fee relevant to this artisan is cited as '{_safe(ev.fee_amount)}' on the referenced page.",
        node=fee_leaf,
        sources=sources,
        additional_instruction="Accept if the page shows a matching fee tier; minor formatting differences or multiple tier options are acceptable if one matches the cited figure."
    )

    # Cross-event budget contribution (logical summary using all 3 fees)
    # We'll compute the sum outside and phrase the claim here too.
    fees = [_parse_money_to_float(e.fee_amount) for e in all_events]
    fees_sum = sum([f for f in fees if isinstance(f, (float, int))])
    budget_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_budget_contribution",
        desc="Fee contributes to total that does not exceed $2,100 across all 3 events",
        parent=financial_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total of all three fees equals approximately ${fees_sum:.2f}, which does not exceed $2100.",
        node=budget_leaf,
        sources=None,
        additional_instruction="Perform a logical check only using the extracted fees; accept small rounding differences."
    )

    # ---------------- Eligibility ----------------
    eligibility_node = evaluator.add_parallel(
        id=f"event_{event_idx}_eligibility",
        desc="Event meets eligibility and selection criteria",
        parent=identification_node,
        critical=True
    )

    juried_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_juried",
        desc="Event has a juried selection/acceptance process",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim="This event uses a juried selection/acceptance process.",
        node=juried_leaf,
        sources=sources,
        additional_instruction="Look for 'juried', 'jury', or selection process language on the page."
    )

    woodworking_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_woodworking_accepted",
        desc="Event explicitly accepts woodworking or furniture as eligible medium/category",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim="The event's eligible media/categories include woodworking and/or furniture.",
        node=woodworking_leaf,
        sources=sources,
        additional_instruction="Check category/medium descriptions; allow 'wood', 'woodworking', 'wood art', or 'furniture' as acceptable evidence."
    )

    age_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_age_requirement",
        desc="Artisan meets age requirement (18+) for the event",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim="The event's application or vendor terms indicate applicants/vendors must be at least 18 years old.",
        node=age_leaf,
        sources=sources,
        additional_instruction="Look for explicit age requirements (e.g., '18+' or 'must be 18 years or older') on the vendor/application information."
    )

    # ---------------- Compliance ----------------
    compliance_node = evaluator.add_parallel(
        id=f"event_{event_idx}_compliance",
        desc="State and insurance requirements are correctly identified",
        parent=identification_node,
        critical=True
    )

    sales_tax_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_sales_tax",
        desc="State sales tax permit requirement is correctly identified for Texas vendors",
        parent=compliance_node,
        critical=True
    )
    await evaluator.verify(
        claim="Vendors (including those from Texas) are required to obtain and display the appropriate state sales tax/seller's permit for on-site sales at this event.",
        node=sales_tax_leaf,
        sources=sources,
        additional_instruction="Confirm that the vendor information indicates a state sales tax or seller's permit requirement (using the event's state rules)."
    )

    insurance_leaf = evaluator.add_leaf(
        id=f"event_{event_idx}_insurance",
        desc="Liability insurance requirement or recommendation is correctly identified",
        parent=compliance_node,
        critical=True
    )
    await evaluator.verify(
        claim="The vendor information states that liability insurance is required or recommended for participants.",
        node=insurance_leaf,
        sources=sources,
        additional_instruction="Look for 'insurance' requirements or recommendations in vendor rules/prospectus."
    )


async def verify_overall_requirements(
    evaluator: Evaluator,
    parent_node,
    events: List[EventInfo]
) -> None:
    """
    Build and verify the 'overall_requirements' subtree that checks cross-event constraints.
    """
    overall = evaluator.add_parallel(
        id="overall_requirements",
        desc="Overall set requirements across all three events are met",
        parent=parent_node,
        critical=True
    )

    urls_all: List[str] = []
    for ev in events:
        urls_all.extend(_gather_sources_for_event(ev))
    # Deduplicate
    seen = set()
    urls_all = [u for u in urls_all if not (u in seen or seen.add(u))]

    # California requirement
    california_leaf = evaluator.add_leaf(
        id="california_requirement",
        desc="At least one event is located in California",
        parent=overall,
        critical=True
    )
    await evaluator.verify(
        claim="This event is located in California (CA).",
        node=california_leaf,
        sources=urls_all,
        additional_instruction="Pass if at least one of the provided URLs clearly indicates the event takes place in California."
    )

    # Dedicated woodworking competition with cash prizes
    competition_leaf = evaluator.add_leaf(
        id="competition_requirement",
        desc="At least one event is a dedicated woodworking competition offering cash prizes",
        parent=overall,
        critical=True
    )
    await evaluator.verify(
        claim="This is a dedicated woodworking competition (not a general art fair) and it offers cash prizes.",
        node=competition_leaf,
        sources=urls_all,
        additional_instruction="Pass if any URL indicates a woodworking‑specific competition and mentions cash prizes."
    )

    # General craft fair accepting wood/furniture
    craft_fair_leaf = evaluator.add_leaf(
        id="craft_fair_requirement",
        desc="At least one event is a general craft fair accepting wood/furniture media",
        parent=overall,
        critical=True
    )
    await evaluator.verify(
        claim="This is a general craft fair that accepts wood and/or furniture as an eligible category.",
        node=craft_fair_leaf,
        sources=urls_all,
        additional_instruction="Pass if any URL indicates a general craft fair (not dedicated to one medium) and lists wood/woodworking/furniture among accepted media."
    )

    # Total budget <= $2,100
    total_budget_leaf = evaluator.add_leaf(
        id="total_budget",
        desc="Total combined fees for all 3 events do not exceed $2,100",
        parent=overall,
        critical=True
    )
    fees = [_parse_money_to_float(e.fee_amount) for e in events]
    fees_sum = sum([f for f in fees if isinstance(f, (float, int))])
    await evaluator.verify(
        claim=f"The sum of the three fees is approximately ${fees_sum:.2f}, which is less than or equal to $2,100.",
        node=total_budget_leaf,
        sources=None,
        additional_instruction="Perform a logical check on the extracted fee amounts; small rounding differences are acceptable."
    )

    # Month diversity: all three months distinct
    month_diversity_leaf = evaluator.add_leaf(
        id="month_diversity",
        desc="All three events occur in different months",
        parent=overall,
        critical=True
    )
    months = [_month_from_date(e.start_date) for e in events]
    await evaluator.verify(
        claim=f"The three start months are {months}; all are present and pairwise different.",
        node=month_diversity_leaf,
        sources=None,
        additional_instruction="This is a logical cross-check: pass only if all three months are non-null and all distinct."
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
    Evaluate an answer for the woodworking artisan 2026 events task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel at root; child nodes handle their own gating
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

    # IMPORTANT: Root criticality must be non-critical to avoid framework constraint of critical parent with non-critical children
    root.critical = False

    # Extract structured events
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Keep only first 3 events, pad with empty if fewer
    events: List[EventInfo] = list(extracted.events[:3])
    while len(events) < 3:
        events.append(EventInfo())

    # Add a custom info summary for quick view
    evaluator.add_custom_info(
        {
            "event_count_extracted": len(extracted.events),
            "selected_event_urls": [[e.url] + e.extra_urls for e in events]
        },
        info_type="debug",
        info_name="selection_overview"
    )

    # Build per-event verification subtrees
    # Use a parallel node for each event subtree (as required by rubric)
    # The sequential logic applies within each event subtree
    tasks = []
    for i in range(3):
        tasks.append(verify_single_event(evaluator, root, events[i], i, events))
    await asyncio.gather(*tasks)

    # Build overall requirements subtree
    await verify_overall_requirements(evaluator, root, events)

    # Return evaluation summary
    return evaluator.get_summary()