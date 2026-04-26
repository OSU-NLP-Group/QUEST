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
TASK_ID = "las_vegas_live_venues_v1"
TASK_DESCRIPTION = (
    "Find two live entertainment venues in Las Vegas, Nevada that offer ticketed shows or performances. "
    "For each venue, provide the following information:\n"
    "- The official venue name\n"
    "- The complete physical address (including street address, city, state, and ZIP code)\n"
    "- The seating capacity (numerical value)\n"
    "- Ticket booking information (either a phone number or an official ticketing website URL)"
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    # Capacity as it appears in the answer. Keep as string to be robust to formats like "2,000" or "~4k".
    capacity: Optional[str] = None

    # Ticketing information from the answer
    ticket_phone: Optional[str] = None
    ticket_url: Optional[str] = None

    # Helpful URLs cited in the answer (e.g., official website, ticketing pages)
    official_site_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class VenueListExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    You will extract up to TWO live entertainment venues located in Las Vegas, Nevada from the provided answer text.

    For each venue, extract the following fields exactly as they appear in the answer:
    - name: The official venue name.
    - street_address: Street line of the physical address (e.g., 123 Main St).
    - city: City name (should be Las Vegas if provided).
    - state: State (e.g., NV or Nevada).
    - zip_code: ZIP code (5-digit; if 9-digit is provided, include the full ZIP+4).
    - capacity: Seating capacity as a numeric expression if available (e.g., 1,800 or 1800). If range or approximate is given (e.g., "~2,000"), extract that text.
    - ticket_phone: A phone number specifically for ticket booking or box office, if provided.
    - ticket_url: An official ticketing website URL for the venue (e.g., Ticketmaster/AXS/venue official tickets page), if provided.
    - official_site_url: The venue’s official website URL, if provided.
    - source_urls: An array of additional relevant URLs (e.g., official venue pages, authorized ticketing platforms) explicitly cited in the answer for this venue.

    RULES:
    - Extract only information explicitly present in the answer. Do not invent or infer missing fields.
    - If a field is not provided in the answer, return null for that field (or empty array for source_urls).
    - Only include venues that are in Las Vegas, Nevada or clearly stated as such in the answer. If more than two venues are present, extract the first two mentioned.
    - For URLs, extract the full URL (include http:// or https://). If missing protocol, prepend http:// as needed.

    Return a JSON object with a single key 'venues' which is an array of at most two VenueItem objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def make_full_address(v: VenueItem) -> str:
    parts = [
        (v.street_address or "").strip(),
        (v.city or "").strip(),
        (v.state or "").strip(),
        (v.zip_code or "").strip(),
    ]
    return ", ".join([p for p in parts if p])


def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    url = url.strip()
    return url.startswith("http://") or url.startswith("https://")


def dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def candidate_ticket_url_from_sources(v: VenueItem) -> Optional[str]:
    """If ticket_url is missing, try to pull a plausible ticketing URL from sources."""
    if v.ticket_url and is_valid_url(v.ticket_url):
        return v.ticket_url

    known_ticket_domains = [
        "ticketmaster", "axs.com", "seatgeek", "livenation", "vegas.com", "ticketweb",
        "stubhub", "tickets", "show", "boxoffice", "ticket", "book", "purchase"
    ]
    for u in v.source_urls:
        if not is_valid_url(u):
            continue
        low = u.lower()
        if any(k in low for k in known_ticket_domains):
            return u
    return v.ticket_url  # May be None


def all_sources_for_venue(v: VenueItem) -> List[str]:
    urls: List[str] = []
    if is_valid_url(v.official_site_url):
        urls.append(v.official_site_url.strip())
    if is_valid_url(v.ticket_url):
        urls.append(v.ticket_url.strip())
    # Ticket URL inferred if necessary
    inferred_ticket = candidate_ticket_url_from_sources(v)
    if inferred_ticket and is_valid_url(inferred_ticket):
        urls.append(inferred_ticket.strip())
    # Additional source URLs
    for su in v.source_urls:
        if is_valid_url(su):
            urls.append(su.strip())
    return dedupe_preserve_order(urls)


def looks_like_phone(s: Optional[str]) -> bool:
    if not s:
        return False
    # Flexible US phone pattern e.g., (702) 555-1234, 702-555-1234, +1 702 555 1234
    phone_re = re.compile(r"(\+1[\s\-\.]?)?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}")
    return phone_re.search(s) is not None


def capacity_has_digits(s: Optional[str]) -> bool:
    if not s:
        return False
    return re.search(r"\d", s) is not None


def is_las_vegas_nv(city: Optional[str], state: Optional[str]) -> bool:
    if not city or not state:
        return False
    city_ok = city.strip().lower() == "las vegas"
    st = state.strip().lower()
    state_ok = st in ("nv", "nevada")
    return city_ok and state_ok


def zip_looks_valid(z: Optional[str]) -> bool:
    if not z:
        return False
    return bool(re.fullmatch(r"\d{5}(-\d{4})?", z.strip()))


# --------------------------------------------------------------------------- #
# Verification builder per-venue                                              #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    v: VenueItem,
    idx: int,
) -> None:
    """
    Build verification sub-tree for a single venue.
    """
    venue_node = evaluator.add_parallel(
        id=f"venue_{idx+1}",
        desc=f"Venue {idx+1} satisfies all constraints and required fields",
        parent=parent_node,
        critical=False
    )

    # 1) Qualifies (critical): live entertainment venue in Las Vegas, NV with ticketed shows/performances
    qualifies_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_qualifies",
        desc=f"Venue {idx+1} is a live entertainment venue in Las Vegas, NV offering ticketed shows/performances",
        parent=venue_node,
        critical=True
    )
    sources = all_sources_for_venue(v)
    name_for_claim = v.name or "the venue"
    claim_qualifies = (
        f"{name_for_claim} is located in Las Vegas, Nevada and offers ticketed shows or performances."
    )
    if sources:
        await evaluator.verify(
            claim=claim_qualifies,
            node=qualifies_leaf,
            sources=sources,
            additional_instruction=(
                "Confirm both: (1) the venue is in Las Vegas, NV and (2) the venue sells tickets or hosts ticketed "
                "shows/performances (look for terms like 'tickets', 'buy tickets', 'box office', 'purchase', or a "
                "ticketing platform page)."
            )
        )
    else:
        # No citations: cannot verify qualification via evidence
        qualifies_leaf.score = 0.0
        qualifies_leaf.status = "failed"

    # 2) Official venue name provided (critical)
    name_exists = evaluator.add_custom_node(
        result=bool(v.name and v.name.strip()),
        id=f"venue_{idx+1}_name",
        desc=f"Venue {idx+1} official venue name is provided",
        parent=venue_node,
        critical=True
    )

    # 3) Complete physical address provided (critical)
    addr_complete = bool(
        (v.street_address and v.street_address.strip())
        and (v.city and v.city.strip())
        and (v.state and v.state.strip())
        and (v.zip_code and v.zip_code.strip())
    )
    # Optionally tighten to Las Vegas, NV and ZIP format presence
    addr_complete = addr_complete and is_las_vegas_nv(v.city, v.state) and zip_looks_valid(v.zip_code)
    address_exists = evaluator.add_custom_node(
        result=addr_complete,
        id=f"venue_{idx+1}_address",
        desc=f"Venue {idx+1} complete physical address is provided (street, city, state, ZIP)",
        parent=venue_node,
        critical=True
    )

    # 4) Seating capacity provided as numerical value (critical)
    capacity_ok = capacity_has_digits(v.capacity)
    capacity_node = evaluator.add_custom_node(
        result=capacity_ok,
        id=f"venue_{idx+1}_capacity",
        desc=f"Venue {idx+1} seating capacity is provided as a numerical value",
        parent=venue_node,
        critical=True
    )

    # 5) Ticket booking information provided (critical): phone or official ticketing URL
    inferred_ticket = candidate_ticket_url_from_sources(v)
    has_ticket_url = is_valid_url(v.ticket_url) or (is_valid_url(inferred_ticket))
    has_ticket_phone = looks_like_phone(v.ticket_phone)
    has_ticketing = has_ticket_url or has_ticket_phone
    ticketing_node = evaluator.add_custom_node(
        result=has_ticketing,
        id=f"venue_{idx+1}_ticketing",
        desc=f"Venue {idx+1} ticket booking information is provided (phone number and/or official ticketing website URL)",
        parent=venue_node,
        critical=True
    )

    # 6) Verifiable sources (critical): evidence URLs exist and are relevant (official site and/or authorized ticketing)
    verif_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_verifiable_sources",
        desc=f"Venue {idx+1} details are verifiable via official venue website and/or authorized ticketing platform URLs (citations provided)",
        parent=venue_node,
        critical=True
    )

    # If we have no sources at all, fail immediately for this leaf
    if not sources:
        verif_leaf.score = 0.0
        verif_leaf.status = "failed"
    else:
        claim_sources = (
            f"At least one of these URLs is the official venue website or an authorized ticketing platform page "
            f"for {name_for_claim}, providing sufficient details (such as address, ticket info, or capacity) to "
            f"verify the venue information."
        )
        await evaluator.verify(
            claim=claim_sources,
            node=verif_leaf,
            sources=sources,
            additional_instruction=(
                "Judge whether this URL is either the venue's official site or a legitimate ticketing platform "
                "(e.g., Ticketmaster, AXS, SeatGeek, Live Nation, vegas.com, or an official resort brand site "
                "hosting the venue). The page should include relevant details like address, tickets/buy links, "
                "or stated capacity. If none of the URLs are relevant/official, mark as not supported."
            )
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
    Evaluate an answer for the Las Vegas live entertainment venues task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Allow partial credit across the two venues
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

    # Extract up to two venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenueListExtraction,
        extraction_name="venues_extraction"
    )

    # Normalize to exactly two venues (pad with empty if needed; cut if more)
    venues: List[VenueItem] = extracted.venues[:2]
    while len(venues) < 2:
        venues.append(VenueItem())

    # Build verification subtrees for each venue
    await verify_single_venue(evaluator, root, venues[0], 0)
    await verify_single_venue(evaluator, root, venues[1], 1)

    # Summarize and return
    return evaluator.get_summary()