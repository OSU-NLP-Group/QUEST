import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ncaa_2026_b_g_o_p_hosts"
TASK_DESCRIPTION = (
    "Identify all four 2026 NCAA Division I Men's Basketball Tournament first and second round host venues that are located in cities whose names begin with the letter 'B', 'G', 'O', or 'P'. "
    "For each venue, provide: (1) the complete official venue name, (2) the host city and state, (3) the primary host organization or athletic conference that submitted the bid, and "
    "(4) a reference URL from an official source confirming the venue's selection as a 2026 tournament host site."
)

ALLOWED_INITIALS = {"B", "G", "O", "P"}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    host_org: Optional[str] = None
    confirmation_urls: List[str] = Field(default_factory=list)
    capacity_text: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt builders                                                  #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract every venue item that the answer claims is a 2026 NCAA Division I Men's Basketball Tournament first/second round host site, regardless of the host city's starting letter.
    For EACH such venue mentioned in the answer (preserve the original order), extract the following fields:

    - venue_name: The complete official venue/arena name as written in the answer. Do not invent or normalize beyond what the answer provides.
    - city: The host city name (without the state).
    - state: The host state (two-letter postal abbreviation or full state name) as presented in the answer.
    - host_org: The primary host organization or athletic conference that submitted the bid, as stated in the answer. If multiple are listed, extract the primary or the one explicitly called out.
    - confirmation_urls: ALL reference URLs cited in the answer that directly support or confirm that this venue is selected as a 2026 NCAA first/second round host site.
        Treat as valid "official" sources if they are from: NCAA (ncaa.com, NCAA press releases), a host conference, a host university/athletic department, or the venue/arena official site.
        However, you must ONLY extract URLs that explicitly appear in the answer text. If the answer gives a source name without a URL, do not create one; leave as an empty list.
    - capacity_text: Any explicit seating capacity mentioned in the answer for this venue (e.g., "capacity 12,500"). If none is mentioned, set to null.
    - capacity_source_urls: ALL URLs cited in the answer that back up the capacity figure (if any). If none are cited, return an empty list.

    IMPORTANT EXTRACTION RULES:
    1) Do NOT add or infer any information that does not explicitly appear in the answer.
    2) Only extract URLs that are literally present in the answer (plain or markdown links). Ignore non-URL citations.
    3) If any field is not present for a venue in the answer, set it to null (or an empty list for URL fields).
    4) Return a JSON object with key "venues" that is an array of objects matching the above schema.
    5) Do not filter by city initial letters here; include all items that the answer claims are 2026 NCAA first/second round host venues. Filtering will be applied later.
    """


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def city_starts_with_allowed_letter(city: Optional[str]) -> bool:
    if not city or not city.strip():
        return False
    first_char = city.strip()[0].upper()
    return first_char in ALLOWED_INITIALS


def pick_first_k_matching(venues: List[VenueItem], k: int) -> List[VenueItem]:
    matched = [v for v in venues if city_starts_with_allowed_letter(v.city)]
    return matched[:k]


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    venue_index: int
) -> None:
    """
    Build and evaluate all leaf checks for a single venue under a parallel parent.
    """
    node = evaluator.add_parallel(
        id=f"venue_{venue_index + 1}",
        desc=f"Venue #{venue_index + 1} meeting the city-initial-letter constraint",
        parent=parent_node,
        critical=False
    )

    # Prepare sources
    confirm_sources = venue.confirmation_urls or []
    capacity_sources = venue.capacity_source_urls or (venue.confirmation_urls or [])
    has_confirm_sources = len(confirm_sources) > 0
    has_capacity_sources = len(capacity_sources) > 0

    # 1) Official venue name (critical)
    name_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index + 1}_name",
        desc="Complete official venue name is provided as used in NCAA documentation",
        parent=node,
        critical=True
    )
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    if venue.venue_name and venue.venue_name.strip() and has_confirm_sources:
        claim_name = (
            f"The venue selected as a 2026 NCAA Division I Men's Basketball Tournament first/second round host site "
            f"is named '{venue.venue_name}'."
        )
        claims_and_sources.append((
            claim_name,
            confirm_sources,
            name_leaf,
            "Match the stated venue name to what is shown on the official announcement page(s). "
            "Allow minor naming variants (e.g., sponsor naming, arena vs. center) if they clearly refer to the same venue."
        ))
    else:
        # Missing name or no confirmation source to ground it -> fail
        name_leaf.score = 0.0
        name_leaf.status = "failed"

    # 2) City and state (critical)
    city_state_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index + 1}_city_state",
        desc="Host city and state are correctly identified",
        parent=node,
        critical=True
    )
    if venue.city and venue.city.strip() and venue.state and venue.state.strip() and has_confirm_sources:
        claim_city_state = f"The host site is located in {venue.city}, {venue.state}."
        claims_and_sources.append((
            claim_city_state,
            confirm_sources,
            city_state_leaf,
            "Verify the city and state as indicated (or clearly implied) on the official announcement page(s). "
            "Allow reasonable formatting variations (e.g., state abbreviation vs. full name)."
        ))
    else:
        city_state_leaf.score = 0.0
        city_state_leaf.status = "failed"

    # 3) City initial letter constraint (critical) - purely logical check
    city_letter_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index + 1}_city_letter",
        desc="Host city name begins with B, G, O, or P",
        parent=node,
        critical=True
    )
    if venue.city and venue.city.strip():
        claim_letter = f"The city name '{venue.city}' begins with one of the letters B, G, O, or P (case-insensitive)."
        claims_and_sources.append((
            claim_letter,
            None,
            city_letter_leaf,
            "This is a simple logical check on the first letter of the provided city name."
        ))
    else:
        city_letter_leaf.score = 0.0
        city_letter_leaf.status = "failed"

    # 4) Capacity requirement (critical)
    capacity_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index + 1}_capacity",
        desc="Venue meets NCAA minimum 10,000 seating capacity requirement for preliminary rounds",
        parent=node,
        critical=True
    )
    if has_capacity_sources and (venue.venue_name and venue.venue_name.strip()):
        claim_capacity = (
            f"The basketball seating capacity at '{venue.venue_name}' is at least 10,000."
        )
        claims_and_sources.append((
            claim_capacity,
            capacity_sources,
            capacity_leaf,
            "Use reliable content on the provided page(s) to verify capacity for basketball configuration. "
            "If multiple capacities are listed, use the basketball/game configuration. "
            "Accept if clearly >= 10,000."
        ))
    else:
        # No sources to ground capacity or missing venue name -> fail
        capacity_leaf.score = 0.0
        capacity_leaf.status = "failed"

    # 5) Host organization (critical)
    host_org_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index + 1}_host_org",
        desc="Officially designated primary host organization or athletic conference that submitted the bid is identified",
        parent=node,
        critical=True
    )
    if venue.host_org and venue.host_org.strip() and has_confirm_sources:
        claim_host = (
            f"The primary host organization (or athletic conference) for this 2026 site is '{venue.host_org}'."
        )
        claims_and_sources.append((
            claim_host,
            confirm_sources,
            host_org_leaf,
            "Confirm that the named organization is listed as a host or primary bid submitter on the official announcement. "
            "Allow reasonable variants (e.g., university vs. athletic department) and co-host phrasing."
        ))
    else:
        host_org_leaf.score = 0.0
        host_org_leaf.status = "failed"

    # 6) Official confirmation URL (critical)
    confirm_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index + 1}_official_confirmation_url",
        desc=("An official-source URL (NCAA, relevant conference, or venue/host organization) is provided that confirms the "
              "venue is selected as a 2026 first/second round host site"),
        parent=node,
        critical=True
    )
    if has_confirm_sources:
        # Build a robust claim tying venue + city/state to 2026 F/S round selection
        loc_fragment = ""
        if venue.city and venue.city.strip() and venue.state and venue.state.strip():
            loc_fragment = f" in {venue.city}, {venue.state}"
        name_fragment = f"the venue '{venue.venue_name}'" if (venue.venue_name and venue.venue_name.strip()) else "the specified venue"

        claim_confirm = (
            f"This webpage confirms that {name_fragment}{loc_fragment} is selected as a 2026 NCAA Division I Men's Basketball "
            f"Tournament first/second round host site."
        )
        claims_and_sources.append((
            claim_confirm,
            confirm_sources,
            confirm_leaf,
            "The page must be an official source (e.g., NCAA.com, a host conference, a host university/athletic department, or the arena's official site) "
            "and it must clearly state the venue is selected for the 2026 NCAA first/second round. "
            "If the page is non-official (e.g., general news) or does not clearly confirm selection, mark as not supported."
        ))
    else:
        # No confirmation URL provided; fail
        confirm_leaf.score = 0.0
        confirm_leaf.status = "failed"

    # Execute all verifications for this venue in parallel
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for identifying the four 2026 NCAA DI MBB first/second round host venues
    in cities starting with B, G, O, or P, and verifying required details for each.
    """
    # Initialize evaluator and root node (parallel aggregation)
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

    # Extract structured venue information from answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Filter venues by city initial letter constraint (B/G/O/P), keep first four for detailed verification
    all_venues: List[VenueItem] = extraction.venues or []
    matching_venues: List[VenueItem] = [v for v in all_venues if city_starts_with_allowed_letter(v.city)]
    selected_venues: List[VenueItem] = matching_venues[:4]

    # Global check: exactly four venues provided in total and they all satisfy the city-letter constraint
    # We consider the intent: the response should identify exactly four such venues (no more, no fewer).
    exactly_four_flag = (len(all_venues) == 4 == len(matching_venues))

    evaluator.add_custom_node(
        result=exactly_four_flag,
        id="global_exactly_four",
        desc="Response identifies exactly four venues total (no fewer and no additional venues beyond four)",
        parent=root,
        critical=True
    )

    # Add some helpful custom info for debugging
    evaluator.add_custom_info(
        {
            "total_extracted_venues": len(all_venues),
            "matching_letter_venues": len(matching_venues),
            "evaluated_venues": len(selected_venues),
            "allowed_initials": sorted(list(ALLOWED_INITIALS)),
        },
        info_type="debug_stats",
        info_name="extraction_stats"
    )

    # Ensure we evaluate exactly 4 venues (pad with empty placeholders if fewer)
    while len(selected_venues) < 4:
        selected_venues.append(VenueItem())

    # Verify each of the four venues (parallel children under root)
    tasks = []
    for idx, venue in enumerate(selected_venues[:4]):
        tasks.append(verify_single_venue(evaluator, root, venue, idx))
    await asyncio.gather(*tasks)

    # Return structured evaluation summary
    return evaluator.get_summary()