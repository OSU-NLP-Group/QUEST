import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient  # for typing only


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nlt_2026_venues"
TASK_DESCRIPTION = (
    "Find two concert venues from the NE-YO & AKON \"Nights Like This Tour 2026\" where each venue has a minimum "
    "seating capacity of 18,000 attendees and the two venues are located in different U.S. states or Canadian provinces. "
    "For each venue, provide: (1) The venue's official name and complete physical address, (2) The venue's seating "
    "capacity, (3) The scheduled concert date for the NE-YO & AKON performance, and (4) A direct link to purchase "
    "tickets (either from Ticketmaster or the venue's official website)."
)
CAPACITY_THRESHOLD = 18000


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    state_province: Optional[str] = None  # e.g., "CA", "California", "ON", "Ontario"
    capacity: Optional[str] = None  # keep as string to be flexible ("~18,500", "19,000+", etc.)
    date: Optional[str] = None  # keep as free-form string (e.g., "July 10, 2026")
    ticket_url: Optional[str] = None  # direct ticket purchase URL (Ticketmaster or official venue)
    event_page_url: Optional[str] = None  # optional: the venue's event page URL (if present)
    tour_source_urls: List[str] = Field(default_factory=list)  # sources showing the event is part of this tour
    details_source_urls: List[str] = Field(default_factory=list)  # sources for name/address
    capacity_source_urls: List[str] = Field(default_factory=list)  # sources for capacity (venue site, Wikipedia, etc.)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to TWO venues listed in the answer for the NE-YO & AKON "Nights Like This Tour 2026".
    For each venue, return the following fields:

    - name: The venue’s official name as written in the answer.
    - address: The complete physical address for the venue, as presented in the answer. Include street, city, state/province (or abbreviation), and country if present.
    - state_province: The U.S. state or Canadian province for the venue. If the answer provides a postal abbreviation, extract that; otherwise, extract the full state/province name from the address.
    - capacity: The seating capacity cited in the answer (keep exactly as stated, even if it's approximate or a range).
    - date: The scheduled concert date for NE-YO & AKON at this venue as stated in the answer (keep formatting as-is).
    - ticket_url: A direct ticket purchase link (prefer Ticketmaster or the official venue website if provided).
    - event_page_url: The venue's official event page URL for this show if it appears in the answer (otherwise null).
    - tour_source_urls: All URLs cited that support that this stop is part of the NE-YO & AKON "Nights Like This Tour 2026".
    - details_source_urls: All URLs cited that support the official venue name and complete physical address.
    - capacity_source_urls: All URLs cited that support the venue’s seating capacity.

    Special rules:
    - Extract only URLs explicitly present in the answer text. Do not invent or infer any URLs.
    - If a field is not present for a venue, set it to null (for strings) or [] (for URL lists).
    - If the answer lists more than two venues, extract only the first two.
    - If the address includes multi-line formatting, keep it as a single line concatenated with commas.
    - For state_province, do your best to parse the state/province from the address text in the answer (abbreviation or full name).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def collect_sources(*candidates: Any) -> List[str]:
    """Collect and de-duplicate URL sources from a mix of strings, lists, or None."""
    urls: List[str] = []
    for c in candidates:
        if not c:
            continue
        if isinstance(c, str):
            if c.strip():
                urls.append(c.strip())
        elif isinstance(c, list):
            for u in c:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
    venue1_name_addr_leaf=None,  # For venue #2 cross-region dependency
) -> Dict[str, Any]:
    """
    Build the verification subtree for a single venue.
    Returns a dictionary of key nodes for potential cross-checks.
    """
    venue_number = index + 1
    vid = f"venue_{venue_number}"

    # Parent node for this venue (sequential: identification first, then details)
    venue_node = evaluator.add_sequential(
        id=vid,
        desc=("First venue meeting all requirements" if index == 0
              else "Second venue meeting all requirements and located in a different state/province than the first venue"),
        parent=evaluator.root,
        critical=False
    )

    # ---------------- Identification (Critical aggregator) ---------------- #
    ident_node = evaluator.add_parallel(
        id=f"{vid}_identification",
        desc=("Correctly identifies a venue from the NE-YO & AKON 'Nights Like This Tour 2026' with capacity of at least 18,000"
              if index == 0
              else "Correctly identifies a second venue from the NE-YO & AKON 'Nights Like This Tour 2026' with capacity of at least 18,000 and in a different state/province than venue 1"),
        parent=venue_node,
        critical=True  # All children under this must be critical
    )

    # Identification sources: use strongest available (ticket/event/tour sources)
    ident_sources = collect_sources(venue.ticket_url, venue.event_page_url, venue.tour_source_urls, venue.details_source_urls)

    # 1) In-tour verification leaf
    in_tour_leaf = evaluator.add_leaf(
        id=f"{vid}_in_tour",
        desc=f"Venue {venue_number} is a stop on NE-YO & AKON's 'Nights Like This Tour 2026'",
        parent=ident_node,
        critical=True
    )
    in_tour_claim = (
        f"The provided page(s) show that NE-YO & AKON's 'Nights Like This Tour 2026' includes a performance at "
        f"{venue.name or 'this venue'}."
    )
    await evaluator.verify(
        claim=in_tour_claim,
        node=in_tour_leaf,
        sources=ident_sources,
        additional_instruction="Accept minor text variations like 'Nights Like This Tour' or shorthand 'Nights Like This 2026'. "
                              "The page should clearly indicate NE-YO & AKON together and that this stop is part of the 2026 tour."
    )

    # 2) Capacity threshold verification leaf (>= 18,000)
    cap_threshold_leaf = evaluator.add_leaf(
        id=f"{vid}_capacity_threshold",
        desc=f"Venue {venue_number} capacity meets or exceeds {CAPACITY_THRESHOLD}",
        parent=ident_node,
        critical=True
    )
    cap_threshold_claim = (
        f"The standard seating capacity of {venue.name or 'the venue'} is at least {CAPACITY_THRESHOLD}."
    )
    await evaluator.verify(
        claim=cap_threshold_claim,
        node=cap_threshold_leaf,
        sources=collect_sources(venue.capacity_source_urls, venue.details_source_urls),
        additional_instruction=(
            "Use reliable venue references (official site, Wikipedia, reputable sources) that mention seating capacity. "
            "If multiple capacities are listed for different configurations, it is acceptable as long as at least one "
            "typical concert configuration is ≥ 18,000. Ignore temporary/standing-room-only boosts unless clearly stated as standard."
        )
    )

    # 3) For venue #2 only: ensure different state/province than venue #1
    different_region_leaf = None
    regions_provided_check = None
    if index == 1:
        # Existence check for states/provinces
        regions_provided_check = evaluator.add_custom_node(
            result=bool(venue.state_province) and isinstance(venue.state_province, str),
            id=f"{vid}_regions_provided",
            desc="Venue 2 has a recognizable state/province extracted",
            parent=ident_node,
            critical=True
        )

        different_region_leaf = evaluator.add_leaf(
            id=f"{vid}_different_region",
            desc="Venue 2 is located in a different U.S. state or Canadian province than Venue 1",
            parent=ident_node,
            critical=True
        )
        s1 = "unknown"  # will be set by caller through pre-known extraction
        s2 = venue.state_province or "unknown"

        # We cannot read venue 1 state here; we'll add it to the claim text but rely on LLM reasoning with additional instruction.
        # To help gating, require Venue 1's name/address verification as a prerequisite if provided by caller.
        prereqs = [regions_provided_check] if regions_provided_check else []
        if venue1_name_addr_leaf is not None:
            prereqs.append(venue1_name_addr_leaf)

        # We still craft a clear claim text; actual s1 (venue 1 state) will be injected by caller in evaluate function.
        different_region_claim = (
            f"The state/province for Venue 1 is different from Venue 2's state/province '{s2}'. "
            "They are in different U.S. states or Canadian provinces."
        )
        await evaluator.verify(
            claim=different_region_claim,
            node=different_region_leaf,
            sources=None,  # logical comparison; relies on extracted fields and context
            extra_prerequisites=prereqs,
            additional_instruction=(
                "Determine the state/province for both venues from the provided answer context. "
                "Consider standard U.S. states or Canadian provinces (names or postal abbreviations). "
                "Judge 'different' if they clearly refer to different states/provinces. "
                "If either one is missing/ambiguous, mark as Incorrect."
            )
        )

    # ---------------- Details (Non-critical aggregator) ------------------- #
    details_node = evaluator.add_parallel(
        id=f"{vid}_details",
        desc=(f"Provides complete information for the {'first' if index == 0 else 'second'} venue"),
        parent=venue_node,
        critical=False
    )

    # a) Venue name + address
    name_addr_provided = evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()) and bool(venue.address and venue.address.strip()),
        id=f"{vid}_name_address_provided",
        desc=f"Venue {venue_number} name and full address are provided",
        parent=details_node,
        critical=True
    )
    name_addr_leaf = evaluator.add_leaf(
        id=f"{vid}_name_address",
        desc="Provides the venue's official name and complete physical address",
        parent=details_node,
        critical=True
    )
    name_addr_claim = (
        f"The venue's official name is '{venue.name or ''}' and its complete physical address is '{venue.address or ''}'."
    )
    await evaluator.verify(
        claim=name_addr_claim,
        node=name_addr_leaf,
        sources=collect_sources(venue.details_source_urls, venue.event_page_url, venue.ticket_url),
        additional_instruction=(
            "Verify the official venue name and a full physical address (street, city, state/province, and country if present). "
            "Allow minor formatting variations (commas, line breaks, abbreviations like 'CA' vs 'California'). "
            "Accept if the provided source(s) clearly present both the official name and the full address."
        )
    )

    # b) Capacity value (as stated)
    capacity_provided = evaluator.add_custom_node(
        result=bool(venue.capacity and venue.capacity.strip()),
        id=f"{vid}_capacity_provided",
        desc=f"Venue {venue_number} capacity value is provided",
        parent=details_node,
        critical=True
    )
    capacity_leaf = evaluator.add_leaf(
        id=f"{vid}_capacity",
        desc="States the venue's seating capacity",
        parent=details_node,
        critical=True
    )
    capacity_claim = (
        f"The seating capacity of {venue.name or 'the venue'} is {venue.capacity or ''}."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=collect_sources(venue.capacity_source_urls, venue.details_source_urls),
        additional_instruction=(
            "Check if the cited sources support the stated capacity value (or a clearly equivalent/approximate figure). "
            "If sources show a significantly different capacity, mark Incorrect."
        )
    )

    # c) Scheduled date
    date_provided = evaluator.add_custom_node(
        result=bool(venue.date and venue.date.strip()),
        id=f"{vid}_date_provided",
        desc=f"Venue {venue_number} scheduled concert date is provided",
        parent=details_node,
        critical=True
    )
    date_leaf = evaluator.add_leaf(
        id=f"{vid}_date",
        desc="Provides the scheduled concert date at this venue",
        parent=details_node,
        critical=True
    )
    date_claim = (
        f"The scheduled concert date for NE-YO & AKON at {venue.name or 'this venue'} is {venue.date or ''}."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=collect_sources(venue.ticket_url, venue.event_page_url, venue.tour_source_urls),
        additional_instruction=(
            "Verify that the provided date matches the event date on the ticketing page or the official venue event page. "
            "Allow minor formatting differences (e.g., 'July 7, 2026' vs '07/07/2026'), but the same calendar date must match."
        )
    )

    # d) Ticketing link
    ticket_provided = evaluator.add_custom_node(
        result=bool(venue.ticket_url and venue.ticket_url.strip()),
        id=f"{vid}_ticketing_provided",
        desc=f"Venue {venue_number} includes a direct ticketing link",
        parent=details_node,
        critical=True
    )
    ticket_leaf = evaluator.add_leaf(
        id=f"{vid}_ticketing",
        desc="Includes a valid ticketing link (Ticketmaster or official venue website)",
        parent=details_node,
        critical=True
    )
    ticket_claim = (
        f"The link {venue.ticket_url or ''} is a direct page to purchase tickets for NE-YO & AKON at "
        f"{venue.name or 'this venue'}."
    )
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_leaf,
        sources=venue.ticket_url,
        additional_instruction=(
            "Confirm that this URL is a live ticket purchase page for the specific NE-YO & AKON event at the venue. "
            "Accept Ticketmaster (.com/.ca) or the venue's official website. The page should show a Buy/Find Tickets action "
            "specific to this event."
        )
    )

    return {
        "venue_node": venue_node,
        "ident_node": ident_node,
        "name_addr_leaf": name_addr_leaf,
        "date_leaf": date_leaf,
        "capacity_leaf": capacity_leaf,
        "ticket_leaf": ticket_leaf,
        "cap_threshold_leaf": cap_threshold_leaf,
        "in_tour_leaf": in_tour_leaf,
        "different_region_leaf": different_region_leaf,
    }


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the NE-YO & AKON 'Nights Like This Tour 2026' venues task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Two venues evaluated independently at the root
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

    # Record useful custom info about constraints
    evaluator.add_custom_info(
        info={"capacity_threshold": CAPACITY_THRESHOLD, "required_venues": 2},
        info_type="constraints",
        info_name="task_constraints"
    )

    # Extract up to two venues from the answer
    extraction: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Normalize to exactly two entries (pad with empty if needed)
    venues: List[VenueItem] = list(extraction.venues[:2])
    while len(venues) < 2:
        venues.append(VenueItem())

    # Verify Venue #1 subtree
    v1_nodes = await verify_single_venue(evaluator, root, venues[0], index=0)

    # For the "different state/province" check in Venue #2, we want the claim to reference Venue 1 region.
    # We'll rebuild that specific claim with actual states and re-run if necessary, but we can guide LLM via additional instruction.
    # Build Venue #2 subtree, passing Venue 1's name/address verification node as a prerequisite for region-difference.
    v2_nodes = await verify_single_venue(
        evaluator,
        root,
        venues[1],
        index=1,
        venue1_name_addr_leaf=v1_nodes.get("name_addr_leaf")
    )

    # Optionally, refine the region-difference claim with explicit states if both are provided
    try:
        v1_state = (venues[0].state_province or "").strip()
        v2_state = (venues[1].state_province or "").strip()
        if v2_nodes.get("different_region_leaf") and v1_state and v2_state:
            # Overwrite/append a refined verification (adds another check, but we keep the original result as well).
            # To avoid duplicating nodes, we can just perform a second verify on the same node with a clearer claim.
            refined_claim = (
                f"Venue 1 is in state/province '{v1_state}', and Venue 2 is in state/province '{v2_state}', "
                f"and these are different U.S. states or Canadian provinces."
            )
            await evaluator.verify(
                claim=refined_claim,
                node=v2_nodes["different_region_leaf"],
                sources=None,
                additional_instruction="Treat common U.S. state and Canadian province names/abbreviations as valid. "
                                      "If they are clearly different regions, this should be marked Correct."
            )
    except Exception:
        # If anything goes wrong, we leave the existing result
        pass

    # Return structured summary
    return evaluator.get_summary()