import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_events_2025"
TASK_DESCRIPTION = (
    "Identify 4 major gaming industry events that took place in 2025 and collectively meet ALL of the following "
    "requirements:\n"
    "1. At least one event must be a competitive esports tournament with a total prize pool exceeding $60 million\n"
    "2. At least one event must have been held in a European country\n"
    "3. At least one event must have occurred during the period May 1 through August 31, 2025\n"
    "4. At least one event must be a gaming convention or expo (not purely a competitive esports tournament)\n"
    "5. The 4 events must have taken place in at least 3 different countries\n"
    "6. At least one event must have achieved a documented record or milestone in attendance, participation, or scale\n\n"
    "For each of the 4 identified events, provide:\n"
    "- The official event name\n"
    "- The host city and country\n"
    "- The specific start and end dates (in YYYY-MM-DD format)\n"
    "- The event type (either \"Competitive Esports Tournament\" or \"Gaming Convention/Expo\")\n"
    "- The total prize pool amount in USD (if applicable for esports tournaments)\n"
    "- A reference URL from a reputable source supporting the provided information"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    start_date: Optional[str] = None  # expected format YYYY-MM-DD
    end_date: Optional[str] = None    # expected format YYYY-MM-DD
    event_type: Optional[str] = None  # normalize to "Competitive Esports Tournament" or "Gaming Convention/Expo"
    prize_pool_usd: Optional[str] = None  # keep as string (could be "$70,000,000", "70M", "USD 70 million", etc.)
    reference_urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return (
        "Extract up to 4 gaming industry events mentioned in the answer that took place in 2025. For each event, "
        "return an object with the following fields:\n"
        "1. name: The official event name (string)\n"
        "2. city: Host city (string)\n"
        "3. country: Host country (string)\n"
        "4. start_date: Start date in YYYY-MM-DD (string). If a precise date is not provided, infer the date from context; otherwise return null.\n"
        "5. end_date: End date in YYYY-MM-DD (string). If a precise date is not provided, infer the date from context; otherwise return null.\n"
        "6. event_type: Exactly one of the following strings:\n"
        '   - \"Competitive Esports Tournament\"\n'
        '   - \"Gaming Convention/Expo\"\n'
        "   Normalize the answer's phrasing to one of these two.\n"
        "7. prize_pool_usd: If the event is a Competitive Esports Tournament and the prize pool is mentioned, provide the amount in USD as a string (e.g., \"$70,000,000\" or \"70 million USD\"). If not available or not applicable, return null.\n"
        "8. reference_urls: An array of one or more URLs (strings) cited in the answer that support the event's details. Extract only actual URLs explicitly present in the answer.\n\n"
        "Important rules:\n"
        "- Only extract URLs explicitly provided in the answer text. If none are provided, return an empty array.\n"
        "- Dates must be in YYYY-MM-DD format. If only a month/year or date range is given, infer reasonable specific dates if the answer implies them; otherwise use null.\n"
        "- Ensure event_type is normalized to the two allowed values.\n"
        "- Return an object of the form: { \"events\": [ ... up to 4 event objects ... ] }.\n"
        "- If the answer lists more than 4 events, include only the first four in the order they appear."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if is_valid_url(uu) and uu not in seen:
            out.append(uu)
            seen.add(uu)
    return out


def first_k(items: List[Any], k: int) -> List[Any]:
    return items[:k] if items else []


def pad_to_k(items: List[Any], k: int, pad_factory) -> List[Any]:
    arr = list(items)
    while len(arr) < k:
        arr.append(pad_factory())
    return arr


def normalize_event_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    s = t.strip().lower()
    if "tournament" in s or "esport" in s or "esports" in s or "championship" in s or "cup" in s or "league" in s:
        return "Competitive Esports Tournament"
    if "expo" in s or "convention" in s or "conference" in s or "gamescom" in s or "summit" in s or "festival" in s:
        return "Gaming Convention/Expo"
    return t


def parse_usd_amount(amount_str: Optional[str]) -> Optional[float]:
    if not amount_str:
        return None
    s = amount_str.lower().replace(",", "").replace("usd", "").replace("us$", "").replace("u$s", "").strip()
    # Handle formats like "$70000000", "$70m", "70 million", "70m", "70.5 million", "~$61,000,000", "over $60 million"
    # Remove currency symbols
    s = s.replace("$", "")
    # Replace words
    s = s.replace("approx.", "").replace("approximately", "").replace("about", "").replace("around", "").replace("over", "").replace("more than", "").replace(">", "")
    s = s.replace("less than", "").replace("<", "")
    s = s.strip()

    try:
        if "billion" in s:
            num = float(s.split("billion")[0].strip())
            return num * 1_000_000_000.0
        if "million" in s:
            num = float(s.split("million")[0].strip())
            return num * 1_000_000.0
        if s.endswith("m"):
            num = float(s[:-1])
            return num * 1_000_000.0
        if s.endswith("k"):
            num = float(s[:-1])
            return num * 1_000.0
        # Plain number
        num = float(s)
        # Heuristic: if too small and looks like e.g. "70" might mean million? But avoid guessing; return as-is
        return num
    except Exception:
        return None


EUROPEAN_COUNTRIES = {
    "albania", "andorra", "armenia", "austria", "azerbaijan", "belarus", "belgium", "bosnia and herzegovina",
    "bulgaria", "croatia", "cyprus", "czechia", "czech republic", "denmark", "estonia", "finland", "france",
    "georgia", "germany", "greece", "hungary", "iceland", "ireland", "italy", "kazakhstan", "kosovo",
    "latvia", "liechtenstein", "lithuania", "luxembourg", "malta", "moldova", "monaco", "montenegro",
    "netherlands", "north macedonia", "norway", "poland", "portugal", "romania", "russia", "san marino",
    "serbia", "slovakia", "slovenia", "spain", "sweden", "switzerland", "turkey", "ukraine",
    "united kingdom", "uk", "vatican city", "holy see"
}


def is_european_country(country: Optional[str]) -> bool:
    if not country:
        return False
    return country.strip().lower() in EUROPEAN_COUNTRIES


def parse_date_str(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return datetime.strptime(d.strip(), "%Y-%m-%d").date()
    except Exception:
        return None


def ranges_overlap(a_start: Optional[date], a_end: Optional[date], b_start: date, b_end: date) -> bool:
    if not a_start and not a_end:
        return False
    if a_start and not a_end:
        # Treat as single day
        return b_start <= a_start <= b_end
    if a_end and not a_start:
        return b_start <= a_end <= b_end
    assert a_start is not None and a_end is not None
    return not (a_end < b_start or a_start > b_end)


def collect_all_urls(events: List[EventItem]) -> List[str]:
    urls: List[str] = []
    seen = set()
    for ev in events:
        for u in sanitize_urls(ev.reference_urls):
            if u not in seen:
                urls.append(u)
                seen.add(u)
    return urls


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_single_event(evaluator: Evaluator, parent_node, ev: EventItem, idx: int) -> None:
    """
    Build verification sub-tree for a single event (index 1..4).
    """
    ev_node = evaluator.add_parallel(
        id=f"Event_{idx}",
        desc=f"Event #{idx} verification",
        parent=parent_node,
        critical=False
    )

    # Reference URL presence (critical in rubric)
    urls = sanitize_urls(ev.reference_urls)
    ref_present = len(urls) > 0
    evaluator.add_custom_node(
        result=ref_present,
        id=f"Event_{idx}_Reference_URL",
        desc=f"A valid reference URL supporting the event information is provided for event #{idx}",
        parent=ev_node,
        critical=True
    )

    # Name (critical)
    name_leaf = evaluator.add_leaf(
        id=f"Event_{idx}_Name",
        desc=f"Official name of event #{idx} is provided and accurate",
        parent=ev_node,
        critical=True
    )
    name_val = ev.name or ""
    await evaluator.verify(
        claim=f"The official event name is '{name_val}'.",
        node=name_leaf,
        sources=urls,
        additional_instruction="Verify that the page explicitly names the event. Allow minor formatting differences (e.g., punctuation, year suffix)."
    )

    # Location group (critical)
    loc_node = evaluator.add_parallel(
        id=f"Event_{idx}_Location",
        desc=f"Location information for event #{idx}",
        parent=ev_node,
        critical=True
    )
    # City
    city_leaf = evaluator.add_leaf(
        id=f"Event_{idx}_City",
        desc=f"Host city is correctly identified for event #{idx}",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The host city of the event is {ev.city or ''}.",
        node=city_leaf,
        sources=urls,
        additional_instruction="Check the page for the host city. If multiple cities or a metro area are listed, this should include the provided city."
    )
    # Country
    country_leaf = evaluator.add_leaf(
        id=f"Event_{idx}_Country",
        desc=f"Host country is correctly identified for event #{idx}",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The host country of the event is {ev.country or ''}.",
        node=country_leaf,
        sources=urls,
        additional_instruction="Verify the country listed on the page. Allow reasonable variants (e.g., 'UK' vs 'United Kingdom')."
    )

    # Dates group (critical)
    dates_node = evaluator.add_parallel(
        id=f"Event_{idx}_Dates",
        desc=f"Date information for event #{idx}",
        parent=ev_node,
        critical=True
    )
    # Start date
    start_leaf = evaluator.add_leaf(
        id=f"Event_{idx}_Start_Date",
        desc=f"Event start date is accurate for event #{idx}",
        parent=dates_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event started on {ev.start_date or ''}.",
        node=start_leaf,
        sources=urls,
        additional_instruction="Verify the event start date on the page. If the page shows a range, ensure this matches the range's start."
    )
    # End date
    end_leaf = evaluator.add_leaf(
        id=f"Event_{idx}_End_Date",
        desc=f"Event end date is accurate for event #{idx}",
        parent=dates_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event ended on {ev.end_date or ''}.",
        node=end_leaf,
        sources=urls,
        additional_instruction="Verify the event end date on the page. If the page shows a range, ensure this matches the range's end."
    )

    # Event type (critical)
    ev_type_norm = normalize_event_type(ev.event_type)
    type_leaf = evaluator.add_leaf(
        id=f"Event_{idx}_Type",
        desc=f"Event type is correctly classified for event #{idx}",
        parent=ev_node,
        critical=True
    )
    if ev_type_norm == "Competitive Esports Tournament":
        type_claim = "This event is a competitive esports tournament."
    elif ev_type_norm == "Gaming Convention/Expo":
        type_claim = "This event is a gaming convention or expo (not purely a competitive esports tournament)."
    else:
        type_claim = f"This event type is correctly classified as {ev.event_type or ''}."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=urls,
        additional_instruction="Determine whether the event is a competitive esports tournament or a gaming convention/expo based on the page content. Use broad understanding; allow that a convention may include some competitions but is not primarily a tournament."
    )

    # Prize pool (non-critical; only applicable to tournaments)
    prize_applicable = (ev_type_norm == "Competitive Esports Tournament")
    if prize_applicable and (ev.prize_pool_usd and ev.prize_pool_usd.strip()):
        prize_leaf = evaluator.add_leaf(
            id=f"Event_{idx}_Prize_Pool",
            desc=f"Prize pool amount is provided and accurate for event #{idx}",
            parent=ev_node,
            critical=False
        )
        await evaluator.verify(
            claim=f"The total prize pool for the event was {ev.prize_pool_usd}.",
            node=prize_leaf,
            sources=urls,
            additional_instruction="Confirm the total prize pool amount for the tournament. Allow rounding and currency conversion to USD when explicitly indicated or implied by the page."
        )
    else:
        # Not applicable or not provided: count as pass for non-critical requirement
        evaluator.add_custom_node(
            result=True,
            id=f"Event_{idx}_Prize_Pool",
            desc=f"Prize pool not applicable or not required for event #{idx} (non-tournament or not provided)",
            parent=ev_node,
            critical=False
        )


async def verify_global_constraints(evaluator: Evaluator, parent_node, events: List[EventItem]) -> None:
    """
    Build the Global Constraints Validation node and verify each constraint.
    """
    global_node = evaluator.add_parallel(
        id="Global_Constraints_Validation",
        desc="Verification that the set of 4 identified events collectively satisfies all global requirements",
        parent=parent_node,
        critical=True
    )

    urls_all = collect_all_urls(events)

    # 1) High prize tournament present (> $60M) - verify via multi-URL if possible
    high_prize_leaf = evaluator.add_leaf(
        id="High_Prize_Tournament_Present",
        desc="At least one event is a competitive esports tournament with total prize pool exceeding $60 million",
        parent=global_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of the events listed in the provided sources had a total prize pool exceeding $60 million USD.",
        node=high_prize_leaf,
        sources=urls_all,
        additional_instruction="Look for phrases like 'total prize pool', 'prize money', 'distribution pool', etc. If a different currency is shown, estimate in USD to confirm if it exceeds 60 million."
    )

    # 2) European event present - verify via multi-URL
    european_leaf = evaluator.add_leaf(
        id="European_Event_Present",
        desc="At least one event was held in a European country",
        parent=global_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these events was held in a European country.",
        node=european_leaf,
        sources=urls_all,
        additional_instruction="European countries include (non-exhaustive): United Kingdom, Germany, France, Spain, Italy, Netherlands, Sweden, Norway, Denmark, Finland, Poland, Czech Republic, Austria, Switzerland, Portugal, Ireland, Belgium, Greece, Hungary, Romania, Bulgaria, Serbia, Croatia, Slovenia, Slovakia, Lithuania, Latvia, Estonia, Iceland, Andorra, Monaco, Liechtenstein, Luxembourg, Malta, San Marino, Vatican City, Albania, North Macedonia, Montenegro, Bosnia and Herzegovina, Moldova, Ukraine, Belarus, Georgia, Armenia, Azerbaijan, Turkey (partly in Europe), Russia (partly in Europe)."
    )

    # 3) Event during May 1 - Aug 31, 2025 - verify via multi-URL
    may_aug_leaf = evaluator.add_leaf(
        id="May_August_Event_Present",
        desc="At least one event occurred during the period May 1 - August 31, 2025",
        parent=global_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these events took place (fully or partially) between May 1 and August 31, 2025, inclusive.",
        node=may_aug_leaf,
        sources=urls_all,
        additional_instruction="If an event spans a range of dates, count it as satisfying this condition if any date overlaps with May 1 through August 31, 2025."
    )

    # 4) At least one gaming convention/expo - verify via multi-URL
    convention_leaf = evaluator.add_leaf(
        id="Gaming_Convention_Present",
        desc="At least one event is a gaming convention/expo (not purely a competitive tournament)",
        parent=global_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these events is a gaming convention or expo (not primarily a competitive esports tournament).",
        node=convention_leaf,
        sources=urls_all,
        additional_instruction="Look for terms like 'expo', 'convention', 'conference', 'trade fair', 'festival' that indicate a convention/expo rather than a tournament."
    )

    # 5) Multi-country requirement: at least 3 different countries among the 4 events - compute from extracted
    countries = [ev.country for ev in events if ev and ev.country]
    unique_countries = set([c.strip().lower() for c in countries if c and c.strip()])
    evaluator.add_custom_node(
        result=(len(unique_countries) >= 3),
        id="Multi_Country_Requirement",
        desc="The 4 events took place in at least 3 different countries",
        parent=global_node,
        critical=True
    )

    # 6) Record/Milestone present - verify via multi-URL
    record_leaf = evaluator.add_leaf(
        id="Record_Milestone_Event_Present",
        desc="At least one event achieved a documented record or milestone in attendance, participation, or scale",
        parent=global_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these events set or achieved a documented record or milestone in attendance, participation, or overall scale.",
        node=record_leaf,
        sources=urls_all,
        additional_instruction="Look for phrases like 'record attendance', 'largest-ever', 'biggest to date', 'most participants', or similar milestone claims on the page."
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
    Evaluate an answer for the 2025 gaming events task.
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

    # Extract structured events (up to 4)
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    events = first_k(extracted.events or [], 4)
    events = pad_to_k(events, 4, pad_factory=lambda: EventItem())

    # Top-level grouping node (adjusted to be non-critical to satisfy framework's critical consistency constraints)
    events_group = evaluator.add_parallel(
        id="Four_Gaming_Events_Identification",
        desc="Identification of 4 major gaming industry events in 2025 that collectively satisfy all specified requirements",
        parent=root,
        critical=False
    )

    # Build event verifications
    for i, ev in enumerate(events, start=1):
        await verify_single_event(evaluator, events_group, ev, i)

    # Build and verify global constraints (critical)
    await verify_global_constraints(evaluator, events_group, events)

    # Optional: add custom info summary
    try:
        countries = [ev.country for ev in events if ev.country]
        summary_info = {
            "extracted_event_count": len([ev for ev in events if ev.name]),
            "unique_countries_count": len(set([c.strip().lower() for c in countries])),
            "countries_list": countries
        }
        evaluator.add_custom_info(summary_info, info_type="summary", info_name="events_summary")
    except Exception:
        pass

    return evaluator.get_summary()