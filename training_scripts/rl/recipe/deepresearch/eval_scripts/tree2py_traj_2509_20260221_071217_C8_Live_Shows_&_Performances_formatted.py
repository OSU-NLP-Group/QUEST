import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "midwest_venues_spring_2026"
TASK_DESCRIPTION = (
    "I am a tour manager planning a spring 2026 concert tour for a mid-sized performing artist through the Midwest region. "
    "I need to identify three different indoor performance venues that meet the following criteria:\n\n"
    "1. Each venue must be located in Ohio, Indiana, or Michigan\n"
    "2. Each venue must have a seating capacity between 1,500 and 4,000 people\n"
    "3. Each venue must be an indoor facility (not an outdoor amphitheater)\n"
    "4. Each venue must regularly host live music concerts or performances (evidence of music events on their calendar or history)\n"
    "5. Each venue must have an official website or official venue information page\n"
    "6. The three venues must be in three different cities\n\n"
    "For each venue, provide:\n"
    "- Venue name\n"
    "- Complete physical address (street, city, state, ZIP code)\n"
    "- Link to official website or official information page\n"
    "- Seating capacity\n"
    "- Evidence that the venue hosts live music (link to event calendar, past event listing, or promotional page showing music events)"
)

ALLOWED_STATES = {"OH", "OHIO", "IN", "INDIANA", "MI", "MICHIGAN"}
CAPACITY_MIN = 1500
CAPACITY_MAX = 4000


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Allow "OH" or "Ohio"
    zip_code: Optional[str] = None
    full_address: Optional[str] = None  # If provided as single string, keep full copy
    website_url: Optional[str] = None  # Official website or official info page
    capacity: Optional[str] = None  # Keep as free text (e.g., "3,200", "2,500-3,100", "approx. 2,700")
    music_evidence_urls: List[str] = Field(default_factory=list)  # URLs showing calendar/past concerts


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to three (3) venues mentioned in the answer. For each venue, extract the following fields exactly as presented in the answer:

    - name: Official venue name (string)
    - street: Street address (string or null)
    - city: City name (string or null)
    - state: State (use the value exactly as given; do not normalize; string or null)
    - zip_code: ZIP or ZIP+4 (string or null)
    - full_address: The complete address line exactly as shown in the answer (string or null). If not present as a single line, you may construct it as "street, city, state ZIP" from the extracted components where available; otherwise leave null.
    - website_url: The official website URL OR official venue information page URL for the venue (string or null). Extract only explicit URLs mentioned in the answer. If missing protocol, prepend http://
    - capacity: The seating capacity as presented (string or null). Do not convert; keep original text such as "3,200", "2,700–3,100", or "about 2,800".
    - music_evidence_urls: An array of URLs (can be empty) that demonstrate live music events (e.g., event calendar, past event listings, or promotional pages). Extract only explicit URLs mentioned in the answer. If missing protocol, prepend http://

    Return a JSON object with a top-level field "venues", which is an array of up to 3 items. If the answer lists more than 3 venues, include only the first 3. If fewer than 3 are present, include as many as provided.

    IMPORTANT:
    - Do not invent or infer information. Only extract what is explicitly in the answer.
    - If any field is missing, set it to null (or [] for music_evidence_urls).
    - Accept both full state names and two-letter abbreviations exactly as provided in the answer (e.g., "Ohio" or "OH").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(s: Optional[str]) -> str:
    return s if isinstance(s, str) else ""


def _clean_urls(urls: List[str]) -> List[str]:
    cleaned = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # If missing protocol, prepend http:// (Extractor should already do this, but be safe)
        if not re.match(r"^https?://", u, flags=re.IGNORECASE):
            u = "http://" + u
        cleaned.append(u)
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in cleaned:
        low = u.lower()
        if low not in seen:
            seen.add(low)
            deduped.append(u)
    return deduped


def _format_full_address(v: VenueItem) -> str:
    # Prefer full_address if present
    if v.full_address and v.full_address.strip():
        return v.full_address.strip()
    parts = []
    if v.street and v.street.strip():
        parts.append(v.street.strip())
    locality = ", ".join([p for p in [(_safe_str(v.city).strip() or None), (_safe_str(v.state).strip() or None)] if p])
    if locality:
        if v.zip_code and v.zip_code.strip():
            parts.append(f"{locality} {v.zip_code.strip()}")
        else:
            parts.append(locality)
    elif v.zip_code and v.zip_code.strip():
        parts.append(v.zip_code.strip())
    return ", ".join(parts).strip()


def _normalize_state_abbrev(s: Optional[str]) -> Optional[str]:
    if not s or not s.strip():
        return None
    st = s.strip().upper()
    mapping = {
        "OHIO": "OH",
        "INDIANA": "IN",
        "MICHIGAN": "MI",
        "OH": "OH",
        "IN": "IN",
        "MI": "MI",
    }
    return mapping.get(st, st)


def _extract_city_state_from_full_address(full_address: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not full_address:
        return None, None
    # Heuristic: look for "... City, ST ZIP"
    # Split by comma
    parts = [p.strip() for p in full_address.split(",") if p.strip()]
    if len(parts) >= 2:
        city = parts[-2]
        state_zip = parts[-1]
        # Get state token (first two letters or first word)
        m = re.match(r"([A-Za-z]{2,})(?:\s+\d{5}(?:-\d{4})?)?$", state_zip)
        if m:
            state = m.group(1)
        else:
            # Try two-letter token
            mm = re.match(r"([A-Za-z]{2})", state_zip)
            state = mm.group(1) if mm else None
        return city, state
    return None, None


def _parse_capacity_numbers(capacity_text: Optional[str]) -> List[int]:
    if not capacity_text:
        return []
    txt = capacity_text.lower().strip()
    # Replace commas
    txt = txt.replace(",", "")
    # Handle k-shorthand like "2.5k"
    txt = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", lambda m: str(int(float(m.group(1)) * 1000)), txt)
    # Find all integer-like numbers
    nums = re.findall(r"\d{3,6}", txt)
    ints = []
    for n in nums:
        try:
            val = int(n)
            ints.append(val)
        except Exception:
            continue
    return ints


def _capacity_in_range(capacity_text: Optional[str], low: int, high: int) -> bool:
    candidates = _parse_capacity_numbers(capacity_text)
    # Accept if any plausible number falls in range
    return any(low <= n <= high for n in candidates)


def _venue_sources_primary(v: VenueItem) -> List[str]:
    return _clean_urls([_safe_str(v.website_url)])


def _venue_sources_all(v: VenueItem) -> List[str]:
    return _clean_urls([_safe_str(v.website_url)] + list(v.music_evidence_urls or []))


def _get_city_state_for_distinct_check(v: VenueItem) -> Tuple[Optional[str], Optional[str]]:
    city = v.city.strip() if v.city else None
    state = v.state.strip() if v.state else None
    if not (city and state):
        # Try deriving from full_address
        d_city, d_state = _extract_city_state_from_full_address(v.full_address)
        city = city or d_city
        state = state or d_state
    return city, state


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
) -> None:
    """
    Build verification subtree and run verifications for a single venue.
    """
    # Container node for this venue (parallel aggregation; non-critical so each venue contributes partial credit)
    v_node = evaluator.add_parallel(
        id=f"venue_{index}",
        desc=f"{['First','Second','Third'][index-1]} venue identification and verification",
        parent=parent_node,
        critical=False
    )

    # Leaf: Official name (verify against official site if available, else other evidence)
    name_leaf = evaluator.add_leaf(
        id=f"venue_{index}_name",
        desc=f"Provide the official name of venue {index}",
        parent=v_node,
        critical=True
    )
    name_claim = f"The official name of the venue is '{_safe_str(venue.name)}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=_venue_sources_all(venue),
        additional_instruction="Verify that the page clearly identifies the venue with this official name. Allow minor punctuation, capitalization, or suffix variations (e.g., 'Theatre' vs 'Theater')."
    )

    # Leaf: Address (complete physical address)
    address_leaf = evaluator.add_leaf(
        id=f"venue_{index}_address",
        desc=f"Provide complete physical address for venue {index} (street, city, state, ZIP)",
        parent=v_node,
        critical=True
    )
    addr_text = _format_full_address(venue)
    addr_claim = f"The venue's complete physical address is '{addr_text}'."
    await evaluator.verify(
        claim=addr_claim,
        node=address_leaf,
        sources=_venue_sources_primary(venue),
        additional_instruction="Verify the venue's address on the official site or official info page. Allow minor formatting differences (comma placement, ZIP+4 vs ZIP5)."
    )

    # Leaf: Website exists (official website or info page) - treat as existence check
    website_exists = bool(_venue_sources_primary(venue))
    evaluator.add_custom_node(
        result=website_exists,
        id=f"venue_{index}_website",
        desc=f"Provide link to official website or information page for venue {index}",
        parent=v_node,
        critical=True
    )

    # Leaf: Region OH / IN / MI (verify location via official page)
    region_leaf = evaluator.add_leaf(
        id=f"venue_{index}_region",
        desc=f"Venue {index} is located in Ohio, Indiana, or Michigan",
        parent=v_node,
        critical=True
    )
    # If state available, include it; otherwise a general state verification
    norm_state = _normalize_state_abbrev(venue.state)
    if norm_state in {"OH", "IN", "MI"}:
        region_claim = f"The venue is located in the state of {norm_state}, which is one of Ohio, Indiana, or Michigan."
    else:
        region_claim = "The venue is located in one of the following states: Ohio, Indiana, or Michigan."
    await evaluator.verify(
        claim=region_claim,
        node=region_leaf,
        sources=_venue_sources_primary(venue),
        additional_instruction="Use the location/address on the page to confirm the state. If the page shows a city/state in OH/IN/MI, mark as supported."
    )

    # Leaf: Capacity value verification (verify the capacity as provided)
    capacity_value_leaf = evaluator.add_leaf(
        id=f"venue_{index}_capacity_value",
        desc=f"Provide the seating capacity for venue {index}",
        parent=v_node,
        critical=True
    )
    if venue.capacity and venue.capacity.strip():
        cap_claim = f"The venue's seating capacity is '{venue.capacity}'."
    else:
        cap_claim = "The venue's seating capacity is provided."
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_value_leaf,
        sources=_venue_sources_primary(venue),
        additional_instruction="Verify the capacity figure on the official site or official info page. If the site lists multiple configurations, accept if the provided value matches any described configuration or clearly corresponds."
    )

    # Leaf: Capacity range check (1,500–4,000) as a custom constraint check
    cap_in_range = _capacity_in_range(venue.capacity, CAPACITY_MIN, CAPACITY_MAX)
    evaluator.add_custom_node(
        result=cap_in_range,
        id=f"venue_{index}_capacity_range",
        desc=f"Venue {index} capacity is between {CAPACITY_MIN} and {CAPACITY_MAX} people",
        parent=v_node,
        critical=True
    )

    # Leaf: Indoor facility verification
    indoor_leaf = evaluator.add_leaf(
        id=f"venue_{index}_indoor",
        desc=f"Venue {index} is an indoor facility",
        parent=v_node,
        critical=True
    )
    indoor_claim = (
        "This venue is an indoor facility (e.g., theater, auditorium, indoor arena, concert hall) "
        "and not an outdoor amphitheater."
    )
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_leaf,
        sources=_venue_sources_primary(venue),
        additional_instruction="Look for cues that it is an indoor space (e.g., 'theater', 'auditorium', 'indoor arena', 'concert hall'). "
                               "If the page indicates 'amphitheater' or 'outdoor' setting, it is not indoor."
    )

    # Leaf: Music evidence link existence (provide link showing live music)
    music_link_exists = bool(_clean_urls(list(venue.music_evidence_urls or [])))
    evaluator.add_custom_node(
        result=music_link_exists,
        id=f"venue_{index}_music_evidence",
        desc=f"Provide link demonstrating venue {index} hosts live music events",
        parent=v_node,
        critical=True
    )

    # Leaf: Music verification (evidence confirms regular live music/concerts)
    music_verify_leaf = evaluator.add_leaf(
        id=f"venue_{index}_music_verification",
        desc=f"Evidence confirms venue {index} regularly hosts live music concerts or performances",
        parent=v_node,
        critical=True
    )
    music_claim = (
        f"The provided page(s) show that '{_safe_str(venue.name)}' hosts live music events or concerts "
        f"(e.g., an event calendar, past concert listings, or promotional pages for bands or music acts)."
    )
    await evaluator.verify(
        claim=music_claim,
        node=music_verify_leaf,
        sources=_clean_urls(list(venue.music_evidence_urls or [])),
        additional_instruction="Accept if the page shows multiple music events, a calendar with concert listings, or past performances by bands or music artists at this venue. "
                               "Ensure the events are for the same venue, not another location."
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
    Evaluate an answer for the Midwest indoor venues task.
    """
    # Initialize evaluator (root is non-critical by default; we will add a critical 'different_cities' child as per rubric)
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

    # Extract up to three venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Keep only first 3 venues; pad with empty placeholders if fewer
    venues: List[VenueItem] = list(extracted.venues[:3])
    while len(venues) < 3:
        venues.append(VenueItem())

    # Build and verify each venue subtree
    await verify_single_venue(evaluator, root, venues[0], 1)
    await verify_single_venue(evaluator, root, venues[1], 2)
    await verify_single_venue(evaluator, root, venues[2], 3)

    # Global constraint: The three venues are located in three different cities (critical at root level)
    # Determine city/state pairs
    city_state_pairs: List[Tuple[Optional[str], Optional[str]]] = [_get_city_state_for_distinct_check(v) for v in venues]
    # Normalize states to abbreviations for uniqueness, keep city case-insensitive
    normalized_pairs = []
    for c, s in city_state_pairs:
        city_norm = c.strip().lower() if c else None
        state_norm = _normalize_state_abbrev(s)
        normalized_pairs.append((city_norm, state_norm))

    unique_pairs = {(c, s) for (c, s) in normalized_pairs if c and s}
    different_cities_ok = (len(unique_pairs) == 3)

    evaluator.add_custom_node(
        result=different_cities_ok,
        id="different_cities",
        desc="The three venues are located in three different cities",
        parent=root,
        critical=True  # As per rubric, this is a critical global constraint
    )

    # Optional: add custom info for debugging
    evaluator.add_custom_info(
        info={
            "allowed_states": ["Ohio (OH)", "Indiana (IN)", "Michigan (MI)"],
            "extracted_city_state_pairs": city_state_pairs
        },
        info_type="debug",
        info_name="region_and_city_debug"
    )

    return evaluator.get_summary()