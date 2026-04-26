import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "sara_evans_apr_2026_venues"
TASK_DESCRIPTION = (
    "Country music artist Sara Evans is on tour throughout 2026. Identify 4 venues from her April 2026 tour schedule "
    "where she is performing that meet ALL of the following criteria:\n\n"
    "1. The venue must be located in one of these states: Illinois, Ohio, Georgia, or Texas\n"
    "2. The venue must have a total capacity between 600 and 2,500 people\n"
    "3. VIP ticket packages must be available for purchase at the venue\n"
    "4. The venue must be a dedicated music or entertainment venue (not a festival or outdoor fair)\n"
    "5. The performance must occur during April 2026\n\n"
    "For each of the 4 venues, provide:\n"
    "- The venue name\n"
    "- The city and state\n"
    "- The venue capacity\n"
    "- The performance date\n"
    "- A reference URL (from Sara Evans' official tour page or the venue's official website)"
)

SARA_TOUR_URL = "https://www.saraevans.com/tour"
ALLOWED_STATE_NAMES = {"illinois", "ohio", "georgia", "texas"}
STATE_ABBR_TO_NAME = {"il": "illinois", "oh": "ohio", "ga": "georgia", "tx": "texas"}
MAX_URLS_PER_VENUE = 5


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None
    date: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_venues() -> str:
    return """
Extract up to the first 6 distinct venues for Sara Evans' tour as presented in the answer.

For each venue mentioned, extract the following fields exactly as stated in the answer:
- name: the venue name
- city: the city of the venue
- state: the state of the venue (as written, e.g., IL, Illinois)
- capacity: the venue's total capacity or seating capacity (keep the exact string, do not convert to number)
- date: the performance date as written (e.g., "April 5, 2026" or "Apr 5, 2026")
- reference_urls: a list of all URLs explicitly shown in the answer that support this venue (these should be actual URLs only; do not infer)

Rules:
- Do not invent missing information. If a field is missing, set it to null (or [] for reference_urls).
- Include only the URLs explicitly present in the answer text (plain or markdown links). If none are present for a venue, use an empty list.
- Keep textual fields as-is; do not normalize formatting.
- Preserve venue order as they appear in the answer.

Return a JSON object:
{
  "venues": [
    {
      "name": ...,
      "city": ...,
      "state": ...,
      "capacity": ...,
      "date": ...,
      "reference_urls": [...]
    },
    ...
  ]
}
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _norm_text(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _norm_key_for_distinct(v: VenueItem) -> str:
    return "|".join([_norm_text(v.name), _norm_text(v.city), _norm_text(v.state)])


def _normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not _nonempty(state):
        return None
    s = _norm_text(state)
    s = s.replace(".", "")
    # Map abbreviations to full names
    if s in STATE_ABBR_TO_NAME:
        return STATE_ABBR_TO_NAME[s]
    # Already a full name?
    if s in ALLOWED_STATE_NAMES:
        return s
    # Try capitalized full names that might include trailing punctuation
    s2 = re.sub(r"[^a-z]", "", s)
    if s2 in ALLOWED_STATE_NAMES:
        return s2
    return None


def _is_allowed_state(state: Optional[str]) -> bool:
    sn = _normalize_state_name(state)
    return sn in ALLOWED_STATE_NAMES if sn else False


def _extract_capacity_numbers(cap: Optional[str]) -> List[int]:
    if not _nonempty(cap):
        return []
    s = cap.lower()
    nums: List[int] = []

    # Find k-suffixed forms, e.g., "1.5k", "2k"
    for m in re.findall(r"(\d+(?:\.\d+)?)\s*k\b", s):
        try:
            nums.append(int(float(m) * 1000))
        except Exception:
            pass

    # Remove commas and pick integer groups
    s_nocomma = s.replace(",", " ")
    for m in re.findall(r"\b(\d{3,5})\b", s_nocomma):
        try:
            nums.append(int(m))
        except Exception:
            pass

    # Deduplicate while preserving order
    out: List[int] = []
    seen = set()
    for n in nums:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _capacity_in_range(cap: Optional[str], lo: int = 600, hi: int = 2500) -> bool:
    values = _extract_capacity_numbers(cap)
    return any(lo <= v <= hi for v in values)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_sara_evans_site(url: str) -> bool:
    return "saraevans.com" in _domain(url)


def _dedup_urls(urls: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls:
        if not _nonempty(u):
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _venue_only_urls(refs: List[str]) -> List[str]:
    return [u for u in refs if not _is_sara_evans_site(u)]


def _tour_plus_refs(refs: List[str]) -> List[str]:
    urls = [SARA_TOUR_URL] + refs
    return _dedup_urls(urls)


# -----------------------------------------------------------------------------
# Venue verification logic
# -----------------------------------------------------------------------------
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
) -> None:
    v_id = index + 1
    v_node = evaluator.add_parallel(
        id=f"Venue_{v_id}",
        desc=f"Qualifying venue #{v_id}.",
        parent=parent_node,
        critical=False,
    )

    # Extract fields
    name = venue.name or ""
    city = venue.city or ""
    state = venue.state or ""
    date = venue.date or ""
    capacity_str = venue.capacity or ""
    refs_all = _dedup_urls(venue.reference_urls)[:MAX_URLS_PER_VENUE]
    refs_venue_only = _venue_only_urls(refs_all)
    refs_tour_plus = _tour_plus_refs(refs_all)

    # 1) Presence checks and simple constraints (critical custom nodes)
    evaluator.add_custom_node(
        result=_nonempty(name),
        id=f"V{v_id}_Venue_Name_Provided",
        desc="Venue name is provided.",
        parent=v_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(city) and _nonempty(state) and _is_allowed_state(state),
        id=f"V{v_id}_City_State_Provided_And_State_Allowed",
        desc="City and state are provided; state is one of Illinois, Ohio, Georgia, or Texas.",
        parent=v_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(date),
        id=f"V{v_id}_Performance_Date_Provided",
        desc="Performance date is provided.",
        parent=v_node,
        critical=True,
    )

    # 2) Performance date is in April 2026 (URL-grounded)
    date_apr_leaf = evaluator.add_leaf(
        id=f"V{v_id}_Performance_Date_In_April_2026",
        desc="The performance occurs during April 2026.",
        parent=v_node,
        critical=True,
    )
    claim_date_in_apr = (
        f"Sara Evans will perform at {name} in {city}, {state} during April 2026."
        f"{' The answer-supplied date is: ' + date if _nonempty(date) else ''}"
    )
    await evaluator.verify(
        claim=claim_date_in_apr,
        node=date_apr_leaf,
        sources=refs_tour_plus,
        additional_instruction="Confirm that the event date is in April 2026. Accept reasonable date formats like 'Apr' abbreviation.",
    )

    # 3) Listed on Sara Evans' official tour page
    listed_leaf = evaluator.add_leaf(
        id=f"V{v_id}_Listed_On_SaraEvans_Tour_Page",
        desc="This specific venue/date is listed as part of Sara Evans' officially scheduled 2026 tour on saraevans.com/tour.",
        parent=v_node,
        critical=True,
    )
    claim_listed = (
        f"Sara Evans' official tour page lists a show at {name} in {city}, {state} "
        f"{'on ' + date if _nonempty(date) else 'during April 2026'}."
    )
    await evaluator.verify(
        claim=claim_listed,
        node=listed_leaf,
        sources=SARA_TOUR_URL,
        additional_instruction="Check the April 2026 section on the official tour page. Fuzzy-match venue naming if needed.",
    )

    # 4) Dedicated venue (not festival/fair)
    dedicated_leaf = evaluator.add_leaf(
        id=f"V{v_id}_Dedicated_Venue_Not_Festival",
        desc="Venue is a dedicated music/entertainment venue (not a festival or outdoor fair).",
        parent=v_node,
        critical=True,
    )
    claim_dedicated = (
        f"The event at {name} is hosted at a dedicated music or entertainment venue (not a festival or outdoor fair)."
    )
    await evaluator.verify(
        claim=claim_dedicated,
        node=dedicated_leaf,
        sources=refs_venue_only if refs_venue_only else refs_tour_plus,
        additional_instruction="Look for indications the location is a permanent venue (theater, club, hall, arena) and not a temporary festival/fairground.",
    )

    # 5) VIP packages available
    vip_leaf = evaluator.add_leaf(
        id=f"V{v_id}_VIP_Packages_Available",
        desc="VIP ticket packages are available for purchase for this performance/venue.",
        parent=v_node,
        critical=True,
    )
    claim_vip = (
        f"VIP ticket packages are available for purchase for Sara Evans' performance at {name} in {city}, {state} "
        f"{'on ' + date if _nonempty(date) else ''}."
    )
    await evaluator.verify(
        claim=claim_vip,
        node=vip_leaf,
        sources=refs_tour_plus,
        additional_instruction="Accept language such as 'VIP', 'VIP packages', 'VIP experience', 'meet & greet', or similar premium packages that can be purchased.",
    )

    # 6) Capacity check (sequential: presence -> numeric range)
    cap_seq = evaluator.add_sequential(
        id=f"V{v_id}_Capacity_Check",
        desc="Capacity is provided and meets the required range.",
        parent=v_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(capacity_str),
        id=f"V{v_id}_Capacity_Value_Provided",
        desc="Venue capacity value is provided.",
        parent=cap_seq,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_capacity_in_range(capacity_str, 600, 2500),
        id=f"V{v_id}_Capacity_In_Range_600_2500",
        desc="Venue capacity is between 600 and 2,500 inclusive (standing or total capacity).",
        parent=cap_seq,
        critical=True,
    )

    # 7) Public address info exists (URL-grounded)
    addr_leaf = evaluator.add_leaf(
        id=f"V{v_id}_Public_Address_Info_Available",
        desc="Publicly available venue address information exists.",
        parent=v_node,
        critical=True,
    )
    claim_address = (
        f"The official website for {name} publicly lists the venue's address or location details (street address/city/state)."
    )
    await evaluator.verify(
        claim=claim_address,
        node=addr_leaf,
        sources=refs_venue_only if refs_venue_only else refs_tour_plus,
        additional_instruction="Look for an address block, 'Contact/Visit Us' section, footer address, or embedded map indicating a street address.",
    )

    # 8) Reference URL checks (sequential: provided -> all from allowed sources)
    ref_seq = evaluator.add_sequential(
        id=f"V{v_id}_Reference_URL_Check",
        desc="Reference URL(s) are provided and come only from allowed sources.",
        parent=v_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(refs_all) > 0,
        id=f"V{v_id}_Reference_URL_Provided",
        desc="At least one reference URL is provided for this venue.",
        parent=ref_seq,
        critical=True,
    )

    # Allowed sources: expand into per-URL verification
    allowed_container = evaluator.add_parallel(
        id=f"V{v_id}_Reference_URLs_From_Allowed_Sources",
        desc="All provided reference URL(s) are from Sara Evans' official site or the venue's official website.",
        parent=ref_seq,
        critical=True,
    )
    # For each provided URL, verify allowedness
    for j, u in enumerate(refs_all):
        leaf = evaluator.add_leaf(
            id=f"V{v_id}_Allowed_Source_{j+1}",
            desc=f"Reference URL #{j+1} is from an allowed source",
            parent=allowed_container,
            critical=True,
        )
        dom = _domain(u)
        if _is_sara_evans_site(u):
            claim_allowed = "This webpage is part of Sara Evans' official website."
            await evaluator.verify(
                claim=claim_allowed,
                node=leaf,
                sources=u,
                additional_instruction="If the domain contains 'saraevans.com', treat it as the official artist website.",
            )
        else:
            claim_allowed = f"This webpage is the official website of the venue named '{name}'."
            await evaluator.verify(
                claim=claim_allowed,
                node=leaf,
                sources=u,
                additional_instruction="Determine if this page belongs to the venue's own official site (not a ticketing/reseller/aggregator). Look for branding, About/Contact pages, and organization identity.",
            )

    # 9) References support capacity (URL-grounded)
    cap_support_leaf = evaluator.add_leaf(
        id=f"V{v_id}_Reference_Supports_Capacity",
        desc="Provided reference URL(s) contain verifiable support for the stated venue capacity.",
        parent=v_node,
        critical=True,
    )
    claim_cap_support = (
        f"The stated venue capacity ({capacity_str}) for {name} is verifiable on one of the provided reference pages. "
        "Equivalent wording like 'seating capacity', 'max capacity', or similar is acceptable, and numeric approximations are fine."
    )
    await evaluator.verify(
        claim=claim_cap_support,
        node=cap_support_leaf,
        sources=refs_venue_only if refs_venue_only else refs_tour_plus,
        additional_instruction="Confirm that the page mentions capacity information consistent with the provided value. Accept minor rounding (e.g., 1,999 vs 2,000).",
    )

    # 10) References support VIP availability (URL-grounded)
    vip_support_leaf = evaluator.add_leaf(
        id=f"V{v_id}_Reference_Supports_VIP",
        desc="Provided reference URL(s) contain verifiable support that VIP ticket packages are available for purchase.",
        parent=v_node,
        critical=True,
    )
    claim_vip_support = (
        f"At least one provided reference explicitly indicates that VIP packages are available for Sara Evans at {name}."
    )
    await evaluator.verify(
        claim=claim_vip_support,
        node=vip_support_leaf,
        sources=refs_tour_plus,
        additional_instruction="Look for 'VIP', 'VIP experience', 'premium packages', 'meet & greet', or similar indications that can be purchased.",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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

    # 1) Extract venues from the answer
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    venues_all = extracted.venues if extracted and extracted.venues else []
    venues = list(venues_all[:4])
    while len(venues) < 4:
        venues.append(VenueItem())

    # 2) Root-level requirement: exactly 4 distinct venues (no duplicates)
    distinct_keys = [_norm_key_for_distinct(v) for v in venues if _nonempty(v.name)]
    exactly_4_distinct = (len([v for v in venues if _nonempty(v.name)]) == 4) and (len(set(distinct_keys)) == 4)
    evaluator.add_custom_node(
        result=exactly_4_distinct,
        id="Exactly_4_Distinct_Venues_Provided",
        desc="Answer provides exactly 4 distinct venues (no duplicates).",
        parent=root,
        critical=True,
    )

    # Optionally record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_venue_count": len(venues_all),
            "used_venue_count": len(venues),
            "distinct_keys": distinct_keys,
        },
        info_type="extraction_stats",
        info_name="extraction_stats",
    )

    # 3) Build verification branches for each venue
    for i in range(4):
        await verify_single_venue(evaluator, root, venues[i], i)

    # 4) Return result summary
    return evaluator.get_summary()