import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "comedy_tour_venues_2026"
TASK_DESCRIPTION = (
    "You are planning to attend comedy shows by Matt Rife and Pete Davidson during their 2026 tours. "
    "Identify exactly 4 venues where these comedians will perform, following these specific selection criteria:\n\n"
    "1. Venue 1: The venue with the largest seating capacity where Matt Rife performs in either Kentucky or Ohio during March 2026. "
    "Provide the venue name, city, state, exact date, seating capacity, and a reference URL.\n\n"
    "2. Venue 2: Another venue in Kentucky or Ohio where Matt Rife performs on the day immediately following Venue 1. "
    "Provide the venue name, city, state, exact date, seating capacity, and a reference URL.\n\n"
    "3. Venue 3: Pete Davidson's first California show in April 2026 (chronologically earliest date). "
    "Provide the venue name, city, exact date, seating capacity, and a reference URL.\n\n"
    "4. Venue 4: A venue in Nevada where Pete Davidson performs on the day immediately following Venue 3. "
    "Provide the venue name, city, exact date, seating capacity, and a reference URL.\n\n"
    "For each venue, you must provide: venue name, city, state (where applicable), date, seating capacity, "
    "and a reference URL that verifies the information."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    comedian: Optional[str] = None  # "Matt Rife" or "Pete Davidson"
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Prefer full state name or 2-letter code if provided
    date: Optional[str] = None  # Keep as raw string from the answer
    capacity: Optional[str] = None  # Keep as string to allow ranges/approximations
    reference_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class AllVenuesExtraction(BaseModel):
    venue1: Optional[VenueExtraction] = None  # Matt Rife, KY/OH, largest capacity, March 2026
    venue2: Optional[VenueExtraction] = None  # Matt Rife, KY/OH, next day after venue1, March 2026
    venue3: Optional[VenueExtraction] = None  # Pete Davidson, CA, first show in April 2026
    venue4: Optional[VenueExtraction] = None  # Pete Davidson, NV, next day after venue3


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract exactly four venues from the provided answer, aligned to the following roles:

- venue1: Matt Rife venue in Kentucky or Ohio during March 2026 that the answer claims has the largest seating capacity among his KY/OH March 2026 shows.
- venue2: Another Matt Rife venue in Kentucky or Ohio that occurs on the calendar day immediately following venue1’s date (also in March 2026).
- venue3: Pete Davidson’s first (chronologically earliest) California show in April 2026.
- venue4: A Pete Davidson Nevada show that occurs on the calendar day immediately following venue3’s date (in April 2026).

For each venue (venue1..venue4), extract these fields exactly as written in the answer (do not infer):
- comedian: The comedian's name ("Matt Rife" or "Pete Davidson")
- venue_name: The venue name
- city: The city
- state: The state (if applicable/mentioned; use either the full name or the 2-letter code exactly as in the answer)
- date: The performance date as shown in the answer (keep its original format)
- capacity: The venue seating capacity as shown in the answer (string; can include commas or approximations if that’s how the answer gives it)
- reference_url: The primary URL the answer cites to support the venue information (if multiple are given, pick the most central one)
- additional_urls: Any other URLs explicitly mentioned in the answer that support this venue (can be empty)

If any field is missing, set it to null (or empty list for additional_urls).
Return a JSON object with keys: venue1, venue2, venue3, venue4, each following the schema.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s_clean = s.strip().lower()
    mapping = {
        "kentucky": "KY",
        "ky": "KY",
        "ohio": "OH",
        "oh": "OH",
        "california": "CA",
        "ca": "CA",
        "nevada": "NV",
        "nv": "NV",
    }
    return mapping.get(s_clean, s.strip().upper())


def _is_in_states(state_str: Optional[str], allowed: List[str]) -> bool:
    if not state_str:
        return False
    norm = _normalize_state(state_str)
    allowed_norm = {_normalize_state(x) for x in allowed}
    return norm in allowed_norm


def _is_http_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def _parse_date_str(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    s = d.strip()
    # Try a set of common date formats
    fmts = [
        "%B %d, %Y",     # March 12, 2026
        "%b %d, %Y",     # Mar 12, 2026
        "%Y-%m-%d",      # 2026-03-12
        "%m/%d/%Y",      # 03/12/2026
        "%m/%d/%y",      # 03/12/26
        "%d %B %Y",      # 12 March 2026
        "%d %b %Y",      # 12 Mar 2026
        "%B %d %Y",      # March 12 2026
        "%b %d %Y",      # Mar 12 2026
        "%Y/%m/%d",      # 2026/03/12
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    # Try handling ordinal suffixes (e.g., March 1st, 2026)
    try:
        import re
        s2 = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s)
        for fmt in fmts:
            try:
                return datetime.strptime(s2, fmt).date()
            except Exception:
                continue
    except Exception:
        pass
    return None


def _is_next_calendar_day(d1_str: Optional[str], d2_str: Optional[str]) -> bool:
    d1 = _parse_date_str(d1_str)
    d2 = _parse_date_str(d2_str)
    if not d1 or not d2:
        return False
    return (d2 - d1).days == 1


def _sources_for(v: Optional[VenueExtraction]) -> List[str]:
    if not v:
        return []
    urls: List[str] = []
    if v.reference_url and _is_http_url(v.reference_url):
        urls.append(v.reference_url.strip())
    for u in v.additional_urls:
        if _is_http_url(u):
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for u in urls:
        if u not in seen:
            ordered.append(u)
            seen.add(u)
    return ordered


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_venue_1(evaluator: Evaluator, parent_node, v1: Optional[VenueExtraction]) -> None:
    node = evaluator.add_parallel(
        id="Venue_1_Largest_Matt_Rife_KY_OH_March",
        desc="The venue with the largest seating capacity where Matt Rife performs in Kentucky or Ohio during March 2026",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence/validity (critical)
    ref_ok = _is_http_url(v1.reference_url if v1 else None)
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="V1_Reference_URL",
        desc="A valid reference URL supporting the venue information is provided",
        parent=node,
        critical=True
    )

    sources = _sources_for(v1)

    # Largest capacity criterion (critical) - evidence-based verification
    largest_leaf = evaluator.add_leaf(
        id="V1_Largest_Capacity_Criterion",
        desc="The selected venue has the largest seating capacity among all Matt Rife venues in Kentucky or Ohio in March 2026",
        parent=node,
        critical=True
    )
    largest_claim = (
        f"Among Matt Rife's March 2026 shows in Kentucky or Ohio, the venue with the largest seating capacity is "
        f"'{(v1.venue_name if v1 and v1.venue_name else 'UNKNOWN')}' with capacity "
        f"'{(v1.capacity if v1 and v1.capacity else 'UNKNOWN')}'."
    )
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        sources=sources,
        additional_instruction=(
            "Verify this 'largest capacity among KY/OH March 2026 Matt Rife shows' claim strictly using the provided sources. "
            "If the sources do not enumerate comparable KY/OH March 2026 venues with capacities (or otherwise cannot support the superlative), "
            "mark as NOT SUPPORTED."
        ),
        extra_prerequisites=[ref_node]
    )

    # Venue name (critical)
    name_leaf = evaluator.add_leaf(
        id="V1_Venue_Name",
        desc="The correct venue name is provided",
        parent=node,
        critical=True
    )
    name_claim = (
        f"The Matt Rife show described for March 2026 in {(v1.city or 'UNKNOWN')}, {(v1.state or 'UNKNOWN')} "
        f"is held at the venue '{(v1.venue_name or 'UNKNOWN')}'."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=sources,
        additional_instruction="Allow minor name variations (e.g., Theatre/Theater, punctuation). Match the underlying venue identity.",
        extra_prerequisites=[ref_node]
    )

    # City (critical)
    city_leaf = evaluator.add_leaf(
        id="V1_City",
        desc="The correct city is provided",
        parent=node,
        critical=True
    )
    city_claim = (
        f"The city for this Matt Rife show is '{(v1.city or 'UNKNOWN')}'."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=sources,
        additional_instruction="Confirm the city location of the show on the cited page(s).",
        extra_prerequisites=[ref_node]
    )

    # State (critical)
    state_leaf = evaluator.add_leaf(
        id="V1_State",
        desc="The correct state (Kentucky or Ohio) is provided",
        parent=node,
        critical=True
    )
    state_claim = (
        f"The state for this Matt Rife show is '{(v1.state or 'UNKNOWN')}', which should be either Kentucky or Ohio."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=sources,
        additional_instruction="Confirm the U.S. state on the cited page(s). Consider 'KY' equivalent to 'Kentucky' and 'OH' equivalent to 'Ohio'.",
        extra_prerequisites=[ref_node]
    )

    # Date (critical)
    date_leaf = evaluator.add_leaf(
        id="V1_Date",
        desc="The correct date in March 2026 is provided",
        parent=node,
        critical=True
    )
    date_claim = (
        f"The performance date shown is '{(v1.date or 'UNKNOWN')}', and it occurs in March 2026."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction="Confirm the exact show date and ensure it falls within March 2026.",
        extra_prerequisites=[ref_node]
    )

    # Capacity (critical)
    cap_leaf = evaluator.add_leaf(
        id="V1_Capacity",
        desc="The correct seating capacity is provided",
        parent=node,
        critical=True
    )
    cap_claim = (
        f"The seating capacity of the venue '{(v1.venue_name or 'UNKNOWN')}' is '{(v1.capacity or 'UNKNOWN')}'."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=sources,
        additional_instruction="Allow minor rounding or typical reported variations. Verify capacity as stated on the cited source(s).",
        extra_prerequisites=[ref_node]
    )


async def verify_venue_2(evaluator: Evaluator, parent_node, v1: Optional[VenueExtraction], v2: Optional[VenueExtraction]) -> None:
    node = evaluator.add_parallel(
        id="Venue_2_Consecutive_Matt_Rife_KY_OH_March",
        desc="Another Matt Rife venue in Kentucky or Ohio where he performs the day immediately following Venue 1",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence/validity (critical)
    ref_ok = _is_http_url(v2.reference_url if v2 else None)
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="V2_Reference_URL",
        desc="A valid reference URL supporting the venue information is provided",
        parent=node,
        critical=True
    )

    # Consecutive day criterion (critical) – logic check only
    consec_ok = _is_next_calendar_day(v1.date if v1 else None, v2.date if v2 else None)
    evaluator.add_custom_node(
        result=consec_ok,
        id="V2_Consecutive_Day_Criterion",
        desc="The show occurs on the day immediately following the Venue 1 show date",
        parent=node,
        critical=True
    )

    # Same region criterion (critical) – KY or OH
    region_ok = _is_in_states(v2.state if v2 else None, ["Kentucky", "KY", "Ohio", "OH"])
    evaluator.add_custom_node(
        result=region_ok,
        id="V2_Same_Region_Criterion",
        desc="The venue is located in Kentucky or Ohio",
        parent=node,
        critical=True
    )

    sources = _sources_for(v2)

    # Venue name (critical)
    name_leaf = evaluator.add_leaf(
        id="V2_Venue_Name",
        desc="The correct venue name is provided",
        parent=node,
        critical=True
    )
    name_claim = (
        f"The Matt Rife show described (day after Venue 1) in {(v2.city or 'UNKNOWN')}, {(v2.state or 'UNKNOWN')} "
        f"is held at the venue '{(v2.venue_name or 'UNKNOWN')}'."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=sources,
        additional_instruction="Allow minor name variations (Theatre/Theater, punctuation). Match venue identity.",
        extra_prerequisites=[ref_node]
    )

    # City (critical)
    city_leaf = evaluator.add_leaf(
        id="V2_City",
        desc="The correct city is provided",
        parent=node,
        critical=True
    )
    city_claim = f"The city for this Matt Rife show is '{(v2.city or 'UNKNOWN')}'."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=sources,
        additional_instruction="Confirm the city on the cited page(s).",
        extra_prerequisites=[ref_node]
    )

    # State (critical)
    state_leaf = evaluator.add_leaf(
        id="V2_State",
        desc="The correct state (Kentucky or Ohio) is provided",
        parent=node,
        critical=True
    )
    state_claim = (
        f"The state for this Matt Rife show is '{(v2.state or 'UNKNOWN')}', which should be either Kentucky or Ohio."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=sources,
        additional_instruction="Confirm the U.S. state on the cited page(s). Consider 'KY'≡'Kentucky' and 'OH'≡'Ohio'.",
        extra_prerequisites=[ref_node]
    )

    # Date (critical)
    date_leaf = evaluator.add_leaf(
        id="V2_Date",
        desc="The correct date in March 2026 is provided",
        parent=node,
        critical=True
    )
    date_claim = f"The performance date shown is '{(v2.date or 'UNKNOWN')}', and it occurs in March 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction="Confirm the exact show date and ensure it falls within March 2026.",
        extra_prerequisites=[ref_node]
    )

    # Capacity (critical)
    cap_leaf = evaluator.add_leaf(
        id="V2_Capacity",
        desc="The correct seating capacity is provided",
        parent=node,
        critical=True
    )
    cap_claim = f"The seating capacity of the venue '{(v2.venue_name or 'UNKNOWN')}' is '{(v2.capacity or 'UNKNOWN')}'."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=sources,
        additional_instruction="Allow minor rounding or typical reported variations. Verify capacity as stated.",
        extra_prerequisites=[ref_node]
    )


async def verify_venue_3(evaluator: Evaluator, parent_node, v3: Optional[VenueExtraction]) -> None:
    node = evaluator.add_parallel(
        id="Venue_3_First_Pete_Davidson_CA_April",
        desc="The first venue chronologically where Pete Davidson performs in California during April 2026",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence/validity (critical)
    ref_ok = _is_http_url(v3.reference_url if v3 else None)
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="V3_Reference_URL",
        desc="A valid reference URL supporting the venue information is provided",
        parent=node,
        critical=True
    )

    # California location (critical) – logic/state check
    ca_ok = _is_in_states(v3.state if v3 else None, ["California", "CA"])
    evaluator.add_custom_node(
        result=ca_ok,
        id="V3_California_Location",
        desc="The venue is located in California",
        parent=node,
        critical=True
    )

    # First April CA show (critical) – evidence-based verification
    sources = _sources_for(v3)
    first_leaf = evaluator.add_leaf(
        id="V3_First_April_Show",
        desc="This is Pete Davidson's first California show in April 2026 (chronologically earliest)",
        parent=node,
        critical=True
    )
    first_claim = (
        f"Pete Davidson's first California show in April 2026 is at '{(v3.venue_name or 'UNKNOWN')}' "
        f"in {(v3.city or 'UNKNOWN')}, CA on '{(v3.date or 'UNKNOWN')}'."
    )
    await evaluator.verify(
        claim=first_claim,
        node=first_leaf,
        sources=sources,
        additional_instruction=(
            "Use the provided sources (e.g., official tour schedule or reputable listings) to confirm that this is the earliest "
            "California date in April 2026. If the sources don't establish 'first/earliest' clearly, mark as NOT SUPPORTED."
        ),
        extra_prerequisites=[ref_node]
    )

    # Venue name (critical)
    name_leaf = evaluator.add_leaf(
        id="V3_Venue_Name",
        desc="The correct venue name is provided",
        parent=node,
        critical=True
    )
    name_claim = (
        f"Pete Davidson's California show described for April 2026 is at the venue '{(v3.venue_name or 'UNKNOWN')}'."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=sources,
        additional_instruction="Allow minor venue name variants. Match the underlying venue identity.",
        extra_prerequisites=[ref_node]
    )

    # City (critical)
    city_leaf = evaluator.add_leaf(
        id="V3_City",
        desc="The correct city is provided",
        parent=node,
        critical=True
    )
    city_claim = f"The city for this Pete Davidson show is '{(v3.city or 'UNKNOWN')}'."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=sources,
        additional_instruction="Confirm the city on the cited page(s).",
        extra_prerequisites=[ref_node]
    )

    # State (critical)
    state_leaf = evaluator.add_leaf(
        id="V3_State",
        desc="The state California is explicitly provided",
        parent=node,
        critical=True
    )
    state_claim = f"The state for this Pete Davidson show is '{(v3.state or 'UNKNOWN')}', i.e., California."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=sources,
        additional_instruction="Confirm the state. Consider 'CA' equivalent to 'California'.",
        extra_prerequisites=[ref_node]
    )

    # Date (critical)
    date_leaf = evaluator.add_leaf(
        id="V3_Date",
        desc="The correct date in April 2026 is provided",
        parent=node,
        critical=True
    )
    date_claim = f"The performance date shown is '{(v3.date or 'UNKNOWN')}', and it occurs in April 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction="Confirm the exact show date and ensure it falls within April 2026.",
        extra_prerequisites=[ref_node]
    )

    # Capacity (critical)
    cap_leaf = evaluator.add_leaf(
        id="V3_Capacity",
        desc="The correct seating capacity is provided",
        parent=node,
        critical=True
    )
    cap_claim = f"The seating capacity of the venue '{(v3.venue_name or 'UNKNOWN')}' is '{(v3.capacity or 'UNKNOWN')}'."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=sources,
        additional_instruction="Allow minor rounding or typical reported variations. Verify capacity as stated.",
        extra_prerequisites=[ref_node]
    )


async def verify_venue_4(evaluator: Evaluator, parent_node, v3: Optional[VenueExtraction], v4: Optional[VenueExtraction]) -> None:
    node = evaluator.add_parallel(
        id="Venue_4_Next_Day_Pete_Davidson_NV_April",
        desc="The Pete Davidson venue in Nevada where he performs the day immediately following Venue 3",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence/validity (critical)
    ref_ok = _is_http_url(v4.reference_url if v4 else None)
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id="V4_Reference_URL",
        desc="A valid reference URL supporting the venue information is provided",
        parent=node,
        critical=True
    )

    # Nevada location (critical)
    nv_ok = _is_in_states(v4.state if v4 else None, ["Nevada", "NV"])
    evaluator.add_custom_node(
        result=nv_ok,
        id="V4_Nevada_Location",
        desc="The venue is located in Nevada",
        parent=node,
        critical=True
    )

    # Consecutive day criterion relative to Venue 3 (critical)
    consec_ok = _is_next_calendar_day(v3.date if v3 else None, v4.date if v4 else None)
    evaluator.add_custom_node(
        result=consec_ok,
        id="V4_Consecutive_Day_Criterion",
        desc="The show occurs on the day immediately following the Venue 3 show date",
        parent=node,
        critical=True
    )

    sources = _sources_for(v4)

    # Venue name (critical)
    name_leaf = evaluator.add_leaf(
        id="V4_Venue_Name",
        desc="The correct venue name is provided",
        parent=node,
        critical=True
    )
    name_claim = (
        f"Pete Davidson's Nevada show (day after Venue 3) is at the venue '{(v4.venue_name or 'UNKNOWN')}'."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=sources,
        additional_instruction="Allow minor venue name variants. Match venue identity.",
        extra_prerequisites=[ref_node]
    )

    # City (critical)
    city_leaf = evaluator.add_leaf(
        id="V4_City",
        desc="The correct city is provided",
        parent=node,
        critical=True
    )
    city_claim = f"The city for this Pete Davidson show is '{(v4.city or 'UNKNOWN')}'."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=sources,
        additional_instruction="Confirm the city on the cited page(s).",
        extra_prerequisites=[ref_node]
    )

    # State (critical)
    state_leaf = evaluator.add_leaf(
        id="V4_State",
        desc="The state Nevada is explicitly provided",
        parent=node,
        critical=True
    )
    state_claim = f"The state for this Pete Davidson show is '{(v4.state or 'UNKNOWN')}', i.e., Nevada."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=sources,
        additional_instruction="Confirm the state. Consider 'NV' equivalent to 'Nevada'.",
        extra_prerequisites=[ref_node]
    )

    # Date (critical)
    date_leaf = evaluator.add_leaf(
        id="V4_Date",
        desc="The correct date in April 2026 is provided",
        parent=node,
        critical=True
    )
    date_claim = f"The performance date shown is '{(v4.date or 'UNKNOWN')}', and it occurs in April 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction="Confirm the exact show date and ensure it falls within April 2026.",
        extra_prerequisites=[ref_node]
    )

    # Capacity (critical)
    cap_leaf = evaluator.add_leaf(
        id="V4_Capacity",
        desc="The correct seating capacity is provided",
        parent=node,
        critical=True
    )
    cap_claim = f"The seating capacity of the venue '{(v4.venue_name or 'UNKNOWN')}' is '{(v4.capacity or 'UNKNOWN')}'."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=sources,
        additional_instruction="Allow minor rounding or typical reported variations. Verify capacity as stated.",
        extra_prerequisites=[ref_node]
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
    extracted: AllVenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=AllVenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Build verification tree per rubric
    # Parent root already parallel/non-critical

    # Venue 1
    await verify_venue_1(evaluator, root, extracted.venue1)

    # Venue 2 (depends logically on Venue 1 for the consecutive day check)
    await verify_venue_2(evaluator, root, extracted.venue1, extracted.venue2)

    # Venue 3
    await verify_venue_3(evaluator, root, extracted.venue3)

    # Venue 4 (depends logically on Venue 3 for the consecutive day check)
    await verify_venue_4(evaluator, root, extracted.venue3, extracted.venue4)

    return evaluator.get_summary()