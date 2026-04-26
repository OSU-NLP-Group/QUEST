import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jack_white_student_rush_capacity_feb2025_ne"
TASK_DESCRIPTION = """
You are a college student planning to attend a Jack White concert in February 2025 in the northeastern United States (New York or Massachusetts) and want to take advantage of his student rush ticket policy. Among all venues hosting Jack White in February 2025 in New York state or Massachusetts that offer the $20 student rush tickets (available at the box office starting at 5pm on show day with valid student ID), which venue has the largest capacity? Provide the venue name, city, state, exact capacity, and the concert date(s).
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CandidateVenue(BaseModel):
    """Information about a single venue mentioned in the answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None
    dates: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)          # Event/venue/tour/capacity pages explicitly cited
    policy_urls: List[str] = Field(default_factory=list)   # URLs explicitly about $20 student rush policy


class AnswerExtraction(BaseModel):
    """Structured extraction from the agent's answer."""
    selected_venue: Optional[CandidateVenue] = None        # The venue the answer claims is the largest (final choice)
    qualifying_venues: List[CandidateVenue] = Field(default_factory=list)  # Other venues in-scope, if listed
    reference_urls: List[str] = Field(default_factory=list)                # Any other URLs cited as references
    general_policy_urls: List[str] = Field(default_factory=list)           # Tour-wide policy links, if provided


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer() -> str:
    return """
    Your task is to extract structured information from the answer about Jack White's February 2025 concerts in New York or Massachusetts and the student rush policy.

    Extract the following JSON fields:

    1) selected_venue: The single venue the answer claims has the largest capacity among qualifying venues.
       - name: Venue name (string)
       - city: City (string)
       - state: State (string; e.g., "New York" or "Massachusetts" or their abbreviations)
       - capacity: The exact capacity number as written in the answer (string; do not normalize or round)
       - dates: List of concert date strings (e.g., ["February 12, 2025"] or ["Feb 12, 2025"])
       - urls: List of all URLs explicitly cited in the answer that support this venue (event page, tour page, venue page, Wikipedia/capacity page, etc.)
       - policy_urls: List of URLs explicitly about the $20 student rush policy (the tour-wide policy or a venue-specific page that mentions it)

    2) qualifying_venues: A list (possibly empty) of other venues mentioned in the answer that are:
       - in New York state or Massachusetts, and
       - hosting Jack White in February 2025, and
       - covered by the $20 student rush policy.
       For each venue include: name, city, state, capacity (string as written), dates (list), urls (list), policy_urls (list).
       Only include venues the answer explicitly mentions. Do not invent data.

    3) reference_urls: A flat list of any other URLs cited anywhere in the answer (e.g., an official tour page, press release, venue/event announcements, Google/Seat map/Wikipedia links etc.). Only include URLs that appear in the answer; do not fabricate.

    4) general_policy_urls: Any tour- or artist-wide policy URLs for the $20 student rush policy mentioned in the answer (e.g., an official tour/policy announcement). Only include URLs present in the answer.

    RULES:
    - Do not invent URLs or details. Extract exactly as written in the answer.
    - If some field is missing, set it to null (for strings) or [] (for lists).
    - Dates must be returned exactly as written (strings).
    - URLs may be plain or markdown links; extract the actual URL.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls or []:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def parse_capacity_to_int(text: Optional[str]) -> Optional[int]:
    """Parse a human-written capacity string to an integer if possible.
    Handles formats like '19,000', '18k', '18.2k', 'approx 19,500', etc.
    """
    if not text:
        return None
    s = text.strip().lower()

    # Try k-suffix first (e.g., 19k or 19.5k)
    m_k = re.search(r'(\d+(?:\.\d+)?)\s*k\b', s)
    if m_k:
        try:
            val = float(m_k.group(1)) * 1000
            return int(round(val))
        except Exception:
            pass

    # Remove non-digits except commas and spaces
    digits = re.findall(r'\d+', s.replace(',', ''))
    if not digits:
        return None
    # Join found sequences into one number (common case: "19 500")
    num_str = "".join(digits)
    try:
        return int(num_str)
    except Exception:
        return None


def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().lower().replace('.', '')
    mapping = {
        'ny': 'new york',
        'new york': 'new york',
        'n y': 'new york',
        'n y ': 'new york',
        'massachusetts': 'massachusetts',
        'ma': 'massachusetts',
        'mass': 'massachusetts'
    }
    return mapping.get(s, state.strip().lower())


def is_state_ny_or_ma(state: Optional[str]) -> bool:
    ns = normalize_state(state)
    return ns in {'new york', 'massachusetts'}


def is_february_2025(date_text: str) -> bool:
    s = date_text.strip().lower()
    # Matches e.g., "February 3, 2025", "Feb 3, 2025", "Feb 2025", "2025-02-xx"
    if re.search(r'\b(feb|february)\b.*\b2025\b', s):
        return True
    if re.search(r'\b2025[-/\.]?0?2\b', s):
        return True
    return False


def any_february_2025(dates: List[str]) -> bool:
    return any(is_february_2025(d) for d in (dates or []))


def build_selected_event_sources(extracted: AnswerExtraction) -> List[str]:
    sv = extracted.selected_venue or CandidateVenue()
    urls = []
    urls.extend(sv.urls or [])
    urls.extend(extracted.reference_urls or [])
    return dedup_preserve_order(urls)


def build_policy_sources(extracted: AnswerExtraction) -> List[str]:
    sv = extracted.selected_venue or CandidateVenue()
    urls = []
    urls.extend(sv.policy_urls or [])
    urls.extend(extracted.general_policy_urls or [])
    urls.extend(extracted.reference_urls or [])
    urls.extend(sv.urls or [])
    return dedup_preserve_order(urls)


def build_capacity_sources(extracted: AnswerExtraction) -> List[str]:
    # We don't distinguish capacity URLs; pass all known selected sources and references,
    # letting the verifier locate capacity on venue/Wikipedia/official pages.
    return dedup_preserve_order(build_selected_event_sources(extracted))


def filter_qualifying_competitors(extracted: AnswerExtraction) -> List[CandidateVenue]:
    sv = extracted.selected_venue or CandidateVenue()
    selected_name = (sv.name or "").strip().lower()
    comps = []
    for v in extracted.qualifying_venues or []:
        if (v.name or "").strip().lower() == selected_name:
            continue
        if not is_state_ny_or_ma(v.state):
            continue
        if not any_february_2025(v.dates or []):
            continue
        comps.append(v)
    return comps


def largest_capacity_internal_check(extracted: AnswerExtraction) -> Tuple[bool, Dict[str, Any]]:
    """Internal logical check using extracted capacities."""
    sv = extracted.selected_venue or CandidateVenue()
    sel_cap = parse_capacity_to_int(sv.capacity)
    competitors = filter_qualifying_competitors(extracted)
    comp_caps: List[Tuple[str, Optional[int], str]] = []  # (name, capacity_int, raw_capacity_str)
    for v in competitors:
        comp_caps.append((v.name or "(unknown venue)", parse_capacity_to_int(v.capacity), v.capacity or ""))

    debug = {
        "selected_name": sv.name,
        "selected_capacity_raw": sv.capacity,
        "selected_capacity_int": sel_cap,
        "competitors": [{"name": n, "capacity_raw": raw, "capacity_int": c} for (n, c, raw) in comp_caps]
    }

    # If we can't parse selected capacity, fail
    if sel_cap is None:
        return False, debug

    # If there are no competitors extracted, we cannot confirm "largest among all" → fail conservatively
    if len(comp_caps) == 0:
        return False, debug

    # Compare against only competitors for which we have a numeric capacity
    numeric_comp_caps = [c for (_, c, _) in comp_caps if c is not None]
    if len(numeric_comp_caps) == 0:
        return False, debug

    ok = all(sel_cap >= c for c in numeric_comp_caps)
    return ok, debug


def looks_like_number(s: Optional[str]) -> bool:
    if not s:
        return False
    return bool(re.search(r'\d', s))


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_tour_and_location(evaluator: Evaluator, parent_node, extracted: AnswerExtraction) -> None:
    """Parallel verification: tour participation, date in Feb 2025, and NY/MA location."""
    sv = extracted.selected_venue or CandidateVenue()
    selected_sources = build_selected_event_sources(extracted)

    tour_loc_node = evaluator.add_parallel(
        id="tour_and_location_verification",
        desc="Verifies the venue is part of Jack White's 2025 No Name Tour, scheduled for February 2025, and located in the northeastern United States",
        parent=parent_node,
        critical=True
    )

    # tour_participation
    tour_node = evaluator.add_leaf(
        id="tour_participation",
        desc="Confirms the venue is hosting Jack White on the 2025 No Name Tour",
        parent=tour_loc_node,
        critical=True
    )
    tour_claim = (
        f"Jack White is scheduled to perform at {sv.name or 'the specified venue'} in {sv.city or ''}, "
        f"{sv.state or ''} as part of his 2025 tour (sometimes referred to as the 'No Name Tour')."
    )

    # february_2025_date
    feb_node = evaluator.add_leaf(
        id="february_2025_date",
        desc="Confirms the concert date is in February 2025",
        parent=tour_loc_node,
        critical=True
    )
    dates_str = ", ".join(sv.dates or [])
    feb_claim = (
        f"The Jack White concert date(s) for {sv.name or 'the specified venue'} occur in February 2025: {dates_str}."
    )

    # northeastern_us_location
    ne_node = evaluator.add_leaf(
        id="northeastern_us_location",
        desc="Confirms the venue is located in the northeastern United States (New York state or Massachusetts)",
        parent=tour_loc_node,
        critical=True
    )
    ne_claim = (
        f"The venue {sv.name or 'the specified venue'} is located in {sv.city or ''}, {sv.state or ''}, "
        f"which is in New York state or Massachusetts."
    )

    claims = [
        (tour_claim, selected_sources, tour_node,
         "Treat naming variants flexibly. If 'No Name Tour' wording isn't explicit, accept clear evidence that the date is part of Jack White's 2025 tour."),
        (feb_claim, selected_sources, feb_node,
         "Confirm that the listed date(s) are in February 2025."),
        (ne_claim, selected_sources, ne_node,
         "Verify the state is either New York (NY) or Massachusetts (MA). Allow standard abbreviations like NY/MA.")
    ]
    await evaluator.batch_verify(claims)


async def verify_policy_and_capacity(evaluator: Evaluator, parent_node, extracted: AnswerExtraction) -> None:
    """Sequential verification: student rush policy → selected capacity value supported → largest capacity internal check."""
    sv = extracted.selected_venue or CandidateVenue()

    pol_cap_node = evaluator.add_sequential(
        id="policy_and_capacity_verification",
        desc="Verifies the venue offers student rush tickets and has the largest capacity among qualifying venues",
        parent=parent_node,
        critical=True
    )

    # 1) Student rush policy ($20 at 5pm, valid student ID)
    policy_node = evaluator.add_leaf(
        id="student_rush_policy",
        desc="Confirms the venue participates in the tour-wide student rush ticket policy offering $20 tickets at the box office starting at 5pm on show day with valid student ID",
        parent=pol_cap_node,
        critical=True
    )
    policy_sources = build_policy_sources(extracted)
    policy_claim = (
        f"The $20 student rush ticket policy—available at the box office starting at 5pm on show day with valid student ID—"
        f"applies to Jack White's {', '.join(sv.dates or ['the specified'])} concert at {sv.name or 'the specified venue'}."
    )
    await evaluator.verify(
        claim=policy_claim,
        node=policy_node,
        sources=policy_sources,
        additional_instruction=(
            "Look for explicit language on $20 student rush tickets, availability at the box office starting at 5pm on show day, "
            "and requiring a valid student ID. If a tour-wide official policy states this applies to all dates including the cited venue/date, that is acceptable."
        )
    )

    # 2) Selected capacity value is supported by sources
    cap_support_node = evaluator.add_leaf(
        id="selected_capacity_supported",
        desc="Confirms the stated capacity for the selected venue is correct per cited sources",
        parent=pol_cap_node,
        critical=True
    )
    cap_sources = build_capacity_sources(extracted)
    cap_claim = (
        f"The capacity of {sv.name or 'the specified venue'} in {sv.city or ''}, {sv.state or ''} is {sv.capacity or 'the stated value'}."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_support_node,
        sources=cap_sources,
        additional_instruction=(
            "Check venue official pages or reliable references (e.g., venue site, Wikipedia, seating/standing capacity info). "
            "Treat small variants (e.g., sports vs concert configuration) carefully; accept the commonly cited capacity matching the claim."
        )
    )

    # 3) Largest capacity internal check (custom logic on extracted data)
    largest_ok, debug = largest_capacity_internal_check(extracted)
    evaluator.add_custom_node(
        result=largest_ok,
        id="largest_capacity",
        desc="Confirms this venue has the largest seating/standing capacity among all venues meeting the criteria (NY/MA, February 2025, $20 student rush)",
        parent=pol_cap_node,
        critical=True
    )
    # Record debug info for transparency
    evaluator.add_custom_info(debug, info_type="largest_capacity_debug", info_name="largest_capacity_comparison")


async def verify_complete_answer_info(evaluator: Evaluator, parent_node, extracted: AnswerExtraction) -> None:
    """Parallel verification of completeness and official reference presence."""
    sv = extracted.selected_venue or CandidateVenue()

    comp_node = evaluator.add_parallel(
        id="complete_answer_information",
        desc="Provides all required information about the identified venue with official verification",
        parent=parent_node,
        critical=True
    )

    # Required details presence
    required_present = (
        bool(sv.name and sv.name.strip()) and
        bool(sv.city and sv.city.strip()) and
        bool(sv.state and sv.state.strip()) and
        bool(looks_like_number(sv.capacity)) and
        bool(sv.dates and len(sv.dates) > 0)
    )
    evaluator.add_custom_node(
        result=required_present,
        id="required_details",
        desc="Includes the specific venue name, city, state, exact capacity number, and concert date(s)",
        parent=comp_node,
        critical=True
    )

    # Presence of at least one reference URL
    all_refs = dedup_preserve_order(
        (sv.urls or []) + (sv.policy_urls or []) + (extracted.reference_urls or []) + (extracted.general_policy_urls or [])
    )
    has_ref = len(all_refs) > 0
    evaluator.add_custom_node(
        result=has_ref,
        id="reference_url_present",
        desc="At least one reference URL is provided in the answer",
        parent=comp_node,
        critical=True
    )

    # Official reference check (multi-URL; passes if any provided URL is official and relevant)
    ref_leaf = evaluator.add_leaf(
        id="reference_url",
        desc="Provides a valid reference URL from official tour or venue sources",
        parent=comp_node,
        critical=True
    )
    official_claim = (
        f"At least one of these webpages is an official source (e.g., official tour/artist page, official venue event page, "
        f"or official policy announcement) relevant to Jack White's 2025 tour and/or the {sv.name or 'specified'} concert."
    )
    await evaluator.verify(
        claim=official_claim,
        node=ref_leaf,
        sources=all_refs,
        additional_instruction=(
            "Treat pages on official domains (e.g., jackwhiteiii.com, thirdmanrecords.com, the venue's own domain, or official ticketing like Ticketmaster) "
            "as official. The page should pertain to the Jack White 2025 tour and/or the specific event/policy."
        )
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
    Evaluate an answer for the Jack White student rush / largest capacity question.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Overall process follows a logical order
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

    # Extraction
    extracted: AnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_answer(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction"
    )

    # Main critical sequential node (as in rubric)
    main_node = evaluator.add_sequential(
        id="correct_venue_identification",
        desc="Correctly identifies the venue with the largest capacity among Jack White's February 2025 northeastern U.S. tour dates that offers student rush tickets",
        parent=root,
        critical=True
    )

    # Early existence gate: require at least the selected venue and some supporting URL
    sv = extracted.selected_venue or CandidateVenue()
    early_gate = bool(sv and sv.name and (sv.urls or extracted.reference_urls))
    evaluator.add_custom_node(
        result=early_gate,
        id="selected_venue_exists",
        desc="Selected venue and at least one supporting URL are provided",
        parent=main_node,
        critical=True
    )

    # Tour, date, and location verification (parallel)
    await verify_tour_and_location(evaluator, main_node, extracted)

    # Student rush policy + capacity supported + largest capacity internal check (sequential)
    await verify_policy_and_capacity(evaluator, main_node, extracted)

    # Completeness and references (parallel)
    await verify_complete_answer_info(evaluator, main_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()