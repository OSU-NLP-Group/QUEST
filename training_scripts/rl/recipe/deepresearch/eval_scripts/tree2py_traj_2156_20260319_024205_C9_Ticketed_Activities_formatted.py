import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tour_west_amphitheaters_2026"
TASK_DESCRIPTION = (
    "A mid-sized touring artist is planning a summer 2026 concert tour of outdoor amphitheaters across the Western "
    "United States. Identify exactly 3 suitable venues that collectively meet all of the following requirements:\n\n"
    "Tour-Wide Requirements:\n"
    "- All 3 venues must be outdoor amphitheaters\n"
    "- The 3 venues must be located in 3 different states in the Western U.S. (defined as: Arizona, Colorado, "
    "Nevada, New Mexico, Utah, or Wyoming)\n"
    "- All venues must be operational and available for summer 2026 concerts\n\n"
    "Individual Venue Requirements (each of the 3 venues must meet ALL of these):\n"
    "- Total venue capacity must be between 8,000 and 20,000 attendees\n"
    "- Must have wheelchair-accessible seating compliant with ADA requirements\n"
    "- Must have a permanent or substantial stage structure suitable for concert productions\n"
    "- Must provide electrical power infrastructure adequate for touring productions (minimum 200 amps three-phase)\n"
    "- Must have rigging capability for hanging lighting and sound equipment\n"
    "- Must have on-site or immediately adjacent parking facilities\n"
    "- Must have backstage facilities including dressing rooms or green rooms for performers\n"
    "- Must allow minimum 4-hour load-in time before performances for touring productions\n"
    "- Must have adequate restroom facilities for the venue's capacity\n"
    "- Must have emergency vehicle access and adequate emergency exits\n"
    "- Must have ticketing/box office capabilities\n\n"
    "For each venue, provide: (1) the venue name, (2) the city and state, (3) the capacity, and (4) a reference URL "
    "confirming the venue specifications."
)

ALLOWED_WEST_STATES: Set[str] = {"AZ", "CO", "NV", "NM", "UT", "WY"}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # full name or 2-letter abbreviation accepted
    capacity: Optional[str] = None  # keep as string (can be a range or approx)
    reference_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to three distinct venue entries as presented in the answer (if more than three are present, return the first three; if fewer than three, return all given). Each entry should capture only what is explicitly stated in the answer.

    For each venue, extract:
    - name: The venue name
    - city: The city
    - state: The U.S. state (accept either full name like "Arizona" or a 2-letter abbreviation like "AZ")
    - capacity: The stated total capacity (keep the original string, even if it is a range, approx., or includes lawn/seated split)
    - reference_url: A primary reference URL for the venue’s specifications (the single best page if multiple are cited)
    - additional_urls: Any other URLs mentioned for this venue (can be empty)

    Rules:
    - Only include venues explicitly listed in the answer.
    - For URLs, extract only valid URLs explicitly present (plain or markdown links). Do not invent URLs.
    - If any field is missing, set it to null (or [] for additional_urls).

    Return a JSON object:
    {
      "venues": [
        {
          "name": ...,
          "city": ...,
          "state": ...,
          "capacity": ...,
          "reference_url": ...,
          "additional_urls": [...]
        }
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
_STATE_ALIASES = {
    "ARIZONA": "AZ", "AZ": "AZ",
    "COLORADO": "CO", "CO": "CO",
    "NEVADA": "NV", "NV": "NV", "NEV": "NV",
    "NEW MEXICO": "NM", "NM": "NM", "N.M.": "NM",
    "UTAH": "UT", "UT": "UT",
    "WYOMING": "WY", "WY": "WY",
}


def normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    clean = re.sub(r"[^A-Za-z ]", "", s).strip().upper()
    # Fast path exact match
    if clean in _STATE_ALIASES:
        return _STATE_ALIASES[clean]
    # Try collapsing spaces (e.g., "NEW  MEXICO")
    clean2 = re.sub(r"\s+", " ", clean)
    return _STATE_ALIASES.get(clean2, None)


def is_valid_url(u: Optional[str]) -> bool:
    if not u:
        return False
    return bool(re.match(r"^https?://", u.strip()))


def compile_all_urls(venue: VenueItem) -> List[str]:
    urls: List[str] = []
    if is_valid_url(venue.reference_url):
        urls.append(venue.reference_url.strip())
    for v in (venue.additional_urls or []):
        if is_valid_url(v):
            urls.append(v.strip())
    # Deduplicate preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def get_nonempty_first3(venues: List[VenueItem]) -> List[VenueItem]:
    filtered = [v for v in venues if v and (v.name or v.city or v.state or v.reference_url)]
    # keep exactly first 3
    return filtered[:3]


def count_nonempty_names(venues: List[VenueItem]) -> int:
    return sum(1 for v in venues if v.name and v.name.strip())


def venue_label(v: VenueItem, idx: int) -> str:
    name = (v.name or "").strip() or f"Venue #{idx+1}"
    city = (v.city or "").strip()
    state = (v.state or "").strip()
    loc = f"{city}, {state}" if city or state else "Unknown city/state"
    return f"{name} ({loc})"


# --------------------------------------------------------------------------- #
# Venue verification logic                                                    #
# --------------------------------------------------------------------------- #
class VenueVerifyArtifacts(BaseModel):
    idx: int
    name_leaf: VerificationNode
    url_leaf: VerificationNode
    all_urls: List[str]
    name: str = ""
    city: str = ""
    state: str = ""


async def verify_single_venue(evaluator: Evaluator, parent: VerificationNode, venue: VenueItem, idx: int) -> VenueVerifyArtifacts:
    # Container node for this venue (non-critical to allow partial across venues)
    venue_node = evaluator.add_parallel(
        id=f"Venue_{idx+1}",
        desc=f"{['First','Second','Third'][idx] if idx < 3 else 'Venue'} venue meets all individual venue requirements",
        parent=parent,
        critical=False
    )

    # ---- Basic Info (critical group) ----
    basic_info = evaluator.add_parallel(
        id=f"Venue_{idx+1}_Basic_Info",
        desc="Basic venue information provided",
        parent=venue_node,
        critical=True
    )
    name_ok = bool(venue.name and venue.name.strip())
    city_ok = bool(venue.city and venue.city.strip())
    state_ok = bool(venue.state and venue.state.strip())

    name_leaf = evaluator.add_custom_node(
        result=name_ok,
        id=f"Venue_{idx+1}_Name",
        desc="Venue name is provided",
        parent=basic_info,
        critical=True
    )
    city_state_leaf = evaluator.add_custom_node(
        result=(city_ok and state_ok),
        id=f"Venue_{idx+1}_City_State",
        desc="City and state are provided",
        parent=basic_info,
        critical=True
    )
    capacity_stated_leaf = evaluator.add_custom_node(
        result=bool(venue.capacity and venue.capacity.strip()),
        id=f"Venue_{idx+1}_Capacity_Stated",
        desc="Venue capacity number is provided",
        parent=basic_info,
        critical=True
    )
    urls = compile_all_urls(venue)
    url_leaf = evaluator.add_custom_node(
        result=bool(urls),
        id=f"Venue_{idx+1}_Reference_URL",
        desc="Reference URL is provided that confirms venue specifications",
        parent=basic_info,
        critical=True
    )

    # Common prerequisites for URL-grounded checks
    prereqs = [name_leaf, url_leaf]

    # ---- Capacity Compliance (critical, sequential) ----
    cap_node = evaluator.add_sequential(
        id=f"Venue_{idx+1}_Capacity_Compliance",
        desc="Venue capacity is within required range",
        parent=venue_node,
        critical=True
    )
    cap_min_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Minimum_Capacity",
        desc="Venue capacity is at least 8,000",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} has a total capacity of at least 8,000 attendees.",
        node=cap_min_leaf,
        sources=urls,
        additional_instruction="Use the venue or official/production specs pages to confirm total capacity. If multiple numbers (seated + lawn) are shown, consider their total."
    )
    cap_max_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Maximum_Capacity",
        desc="Venue capacity does not exceed 20,000",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} has a total capacity not exceeding 20,000 attendees.",
        node=cap_max_leaf,
        sources=urls,
        additional_instruction="Confirm that the stated total capacity is 20,000 or fewer. If multiple capacities are shown, use the largest stated total."
    )

    # ---- Technical Infrastructure (critical) ----
    tech_node = evaluator.add_parallel(
        id=f"Venue_{idx+1}_Technical_Infrastructure",
        desc="Venue has required technical infrastructure",
        parent=venue_node,
        critical=True
    )
    stage_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Stage",
        desc="Venue has permanent or substantial stage structure for concert productions",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} has a permanent or substantial stage suitable for concert productions.",
        node=stage_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Look for references to a fixed stage, amphitheater stage, or comparable substantial performance structure; images on the page can also count if clearly indicating a permanent stage."
    )
    power_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Power",
        desc="Venue provides adequate electrical power infrastructure (minimum 200 amps three-phase for touring productions)",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} provides at least 200A three-phase electrical power (or greater) for touring productions.",
        node=power_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Accept statements like '200A 3-phase', '208V 3-phase 200A', 'company switch 200A+', '400A 3-phase' etc. The evidence must clearly indicate ≥200A three-phase availability."
    )
    rigging_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Rigging",
        desc="Venue has rigging capability for hanging lighting and sound equipment",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} supports rigging (e.g., rigging points, load-bearing truss/roof) for hanging lighting and sound equipment.",
        node=rigging_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Look for terms like 'rigging points', 'fly points', 'riggable roof/truss', or production specs describing overhead load capacities."
    )

    # ---- Accessibility & Safety (critical) ----
    acc_node = evaluator.add_parallel(
        id=f"Venue_{idx+1}_Accessibility_Safety",
        desc="Venue meets accessibility and safety requirements",
        parent=venue_node,
        critical=True
    )
    ada_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_ADA_Seating",
        desc="Venue has wheelchair-accessible seating compliant with ADA requirements",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} offers wheelchair-accessible seating compliant with ADA requirements.",
        node=ada_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Evidence may include 'ADA accessible seating', 'wheelchair accessible areas', companion seating policies, or accessibility statements indicating ADA compliance."
    )
    rest_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Restrooms",
        desc="Venue has adequate restroom facilities for its capacity",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} has adequate restroom facilities for its capacity.",
        node=rest_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Accept explicit mentions of plentiful/ample restrooms, restroom counts, permanent restroom buildings, or event guides describing adequate facilities."
    )
    emerg_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Emergency_Access",
        desc="Venue has emergency vehicle access and adequate emergency exits",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} provides emergency vehicle access and adequate emergency exits.",
        node=emerg_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Look for 'emergency exits', 'egress', 'fire lanes', 'emergency access' in venue policies, event guides, or site plans."
    )

    # ---- Operational & Support (critical) ----
    ops_node = evaluator.add_parallel(
        id=f"Venue_{idx+1}_Operational_Requirements",
        desc="Venue meets operational and support requirements",
        parent=venue_node,
        critical=True
    )
    summer_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Summer_2026",
        desc="Venue is operational and available for summer 2026 concerts",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} is operational and available to host concerts during summer 2026 (June–September 2026).",
        node=summer_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Support may include: active booking information referencing 2026, upcoming events (or recurring seasonal operations) that include summer 2026, or no indications of closure combined with current operations into 2026. If the page clearly shows current seasons and bookings extending through 2026, count as supported."
    )
    parking_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Parking",
        desc="Venue has on-site or immediately adjacent parking facilities",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} has on-site or immediately adjacent parking facilities.",
        node=parking_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Accept mentions of 'on-site parking', 'adjacent lots', venue parking maps, or official parking information for attendees or production."
    )
    backstage_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Backstage",
        desc="Venue has backstage facilities including dressing rooms or green rooms",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} has backstage facilities including dressing rooms and/or green rooms for performers.",
        node=backstage_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Look in production specs or venue info for 'dressing rooms', 'green rooms', 'artist facilities', or similar."
    )
    loadin_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Load_In",
        desc="Venue allows minimum 4-hour load-in time before performances",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} allows at least a 4-hour load-in window prior to performances for touring productions.",
        node=loadin_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Accept explicit policies or production schedules showing ≥4 hours access before show time; examples include 'load-in 12:00 for 8:00pm show', 'access from morning for evening show', or production rider guidelines ≥4 hours."
    )
    boxoffice_leaf = evaluator.add_leaf(
        id=f"Venue_{idx+1}_Box_Office",
        desc="Venue has ticketing/box office capabilities",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {venue_label(venue, idx)} has ticketing or box office capabilities on site.",
        node=boxoffice_leaf,
        sources=urls,
        extra_prerequisites=prereqs,
        additional_instruction="Look for 'box office', on-site ticket windows, or official ticketing handled by/at the venue."
    )

    return VenueVerifyArtifacts(
        idx=idx,
        name_leaf=name_leaf,
        url_leaf=url_leaf,
        all_urls=urls,
        name=(venue.name or "").strip(),
        city=(venue.city or "").strip(),
        state=(venue.state or "").strip()
    )


# --------------------------------------------------------------------------- #
# Tour-wide verification logic                                                #
# --------------------------------------------------------------------------- #
def compute_tour_wide_logic(venues: List[VenueItem]) -> Tuple[bool, bool, bool]:
    """
    Returns:
      - exactly_three_identified: exactly 3 venues with non-empty names
      - three_different_states: all three states are distinct (after normalization)
      - all_in_western_states: all states are within allowed set
    """
    exactly_three_identified = (count_nonempty_names(venues) == 3)

    norm_states: List[Optional[str]] = [normalize_state(v.state) for v in venues]
    # Only consider venues that have names (identified)
    named_mask = [(v.name or "").strip() != "" for v in venues]
    named_states = [s for s, m in zip(norm_states, named_mask) if m and s is not None]

    three_different_states = (len(named_states) == 3 and len(set(named_states)) == 3)
    all_in_western_states = (len(named_states) == 3 and all(s in ALLOWED_WEST_STATES for s in named_states))

    return exactly_three_identified, three_different_states, all_in_western_states


async def add_tour_wide_outdoor_checks(
    evaluator: Evaluator,
    parent: VerificationNode,
    venues: List[VenueItem],
    venue_artifacts: List[VenueVerifyArtifacts],
) -> None:
    """
    Build 'All Outdoor Amphitheaters' as a parallel aggregator with 3 critical leaves,
    each grounded by that venue's URLs and gated by its basic-info prerequisites.
    """
    all_outdoor_node = evaluator.add_parallel(
        id="All_Outdoor_Amphitheaters",
        desc="All 3 identified venues must be outdoor amphitheaters",
        parent=parent,
        critical=True
    )

    for art, v in zip(venue_artifacts, venues):
        leaf = evaluator.add_leaf(
            id=f"Venue_{art.idx+1}_Outdoor_Amphitheater",
            desc=f"Venue #{art.idx+1} is an outdoor amphitheater",
            parent=all_outdoor_node,
            critical=True
        )
        # Prerequisites: name + url provided
        prereqs = [art.name_leaf, art.url_leaf]
        label = venue_label(v, art.idx)
        await evaluator.verify(
            claim=f"The venue {label} is an outdoor amphitheater (open-air amphitheatre).",
            node=leaf,
            sources=art.all_urls,
            extra_prerequisites=prereqs,
            additional_instruction="Look for descriptors like 'outdoor amphitheater', 'open-air', lawn seating, bowl, or official venue type indicating an outdoor amphitheater. Allow spelling 'amphitheatre'/'amphitheater'."
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
    # Initialize evaluator with a parallel root (non-critical root)
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

    # Record GT info (allowed states definition)
    evaluator.add_ground_truth(
        {
            "western_states_allowed": sorted(list(ALLOWED_WEST_STATES)),
            "must_select_exactly": 3,
            "capacity_range": "8000-20000 inclusive",
            "season": "Summer 2026"
        },
        gt_type="tour_requirements"
    )

    # Extract venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Prepare exactly 3 venues (filter/pad logic)
    venues_all = extracted.venues or []
    venues3 = get_nonempty_first3(venues_all)
    # Pad with empties if fewer than 3 to force checks to run/skips
    while len(venues3) < 3:
        venues3.append(VenueItem())

    # Build per-venue verification subtrees first
    venue_artifacts: List[VenueVerifyArtifacts] = []
    for i in range(3):
        art = await verify_single_venue(evaluator, root, venues3[i], i)
        venue_artifacts.append(art)

    # Tour-wide requirements node (critical)
    tour_wide = evaluator.add_parallel(
        id="Tour_Wide_Requirements",
        desc="Tour-wide geographic and venue type requirements",
        parent=root,
        critical=True
    )

    # Exactly 3 venues identified (with non-empty names)
    exactly_three_identified, three_diff_states, all_in_west = compute_tour_wide_logic(venues3)
    evaluator.add_custom_node(
        result=exactly_three_identified,
        id="Exactly_Three_Venues",
        desc="Exactly 3 venues are identified (no more, no fewer)",
        parent=tour_wide,
        critical=True
    )

    # Three different states
    evaluator.add_custom_node(
        result=three_diff_states,
        id="Three_Different_States",
        desc="The 3 identified venues must be located in 3 different U.S. states",
        parent=tour_wide,
        critical=True
    )

    # All in Western states (AZ, CO, NV, NM, UT, WY)
    evaluator.add_custom_node(
        result=all_in_west,
        id="Western_US_States",
        desc="All 3 venues must be located in Western U.S. states (Arizona, Colorado, Nevada, New Mexico, Utah, or Wyoming)",
        parent=tour_wide,
        critical=True
    )

    # All Outdoor Amphitheaters (as 3 URL-grounded leaves)
    await add_tour_wide_outdoor_checks(evaluator, tour_wide, venues3, venue_artifacts)

    # Return final structured summary
    return evaluator.get_summary()