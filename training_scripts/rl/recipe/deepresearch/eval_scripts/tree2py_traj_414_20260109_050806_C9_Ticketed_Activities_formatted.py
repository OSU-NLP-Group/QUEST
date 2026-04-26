import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "la_entertainment_venues_4"
TASK_DESCRIPTION = """
Identify four distinct entertainment venues located in Louisiana that each satisfy ALL of the following requirements:

1. The venue must have a documented seating capacity for its primary event type (such as basketball, football, or concerts)
2. The venue must provide wheelchair accessible seating that complies with ADA standards
3. The venue must offer at least one type of premium seating option, such as luxury suites, club seats, or loge boxes
4. The venue must have on-site parking facilities (garages or surface lots) available to ticket holders
5. The venue must offer ticket discounts for either military personnel or students with proper verification
6. The venue must have a group ticket sales program that specifies a minimum number of attendees required to qualify for group rates
7. The venue must offer season ticket packages with documented benefits (such as playoff ticket priority or parking privileges)
8. The venue must be capable of hosting at least two different types of events (for example, both sports games and concerts)

For each venue, provide:
- The venue's name and location (city, Louisiana)
- The documented seating capacity for its primary event type
- Confirmation that wheelchair accessible seating is available
- The type(s) of premium seating offered
- Description of available parking options
- The type of discount program offered (military or student)
- The minimum number of attendees required for group ticket rates
- Description of season ticket holder benefits
- At least two types of events the venue hosts
- Reference URLs that verify each of these attributes
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    # Identification
    name: Optional[str] = None
    city: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)

    # Capacity
    capacity_primary_event: Optional[str] = None
    capacity_value: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)

    # Accessibility (ADA)
    accessibility_statement: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=list)

    # Premium seating
    premium_types: List[str] = Field(default_factory=list)
    suite_capacity_range: Optional[str] = None  # e.g., "12-18" or "12 to 18 guests"
    premium_urls: List[str] = Field(default_factory=list)

    # Parking
    parking_description: Optional[str] = None
    parking_urls: List[str] = Field(default_factory=list)

    # Discounts
    discount_type: Optional[str] = None  # "military" or "student"
    discount_verification_requirement: Optional[str] = None  # e.g., "valid ID required"
    discount_urls: List[str] = Field(default_factory=list)

    # Group tickets
    group_minimum_attendees: Optional[str] = None
    group_urls: List[str] = Field(default_factory=list)

    # Season tickets
    season_benefits_description: Optional[str] = None
    season_urls: List[str] = Field(default_factory=list)

    # Multi-event
    multi_event_types: List[str] = Field(default_factory=list)
    multi_event_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract up to four (4) distinct entertainment venues in Louisiana from the answer. For each venue, extract the following fields exactly as stated in the answer, along with attribute-specific reference URLs (repeat a URL in multiple categories if the same page supports multiple attributes):

For each venue (return as an array 'venues' of at most 4 items):
- name: Venue name (string)
- city: City in Louisiana (string; e.g., "New Orleans")
- identity_urls: List of URLs that verify the venue identity/location
- capacity_primary_event: The primary event type the stated capacity applies to (e.g., football, basketball, concerts)
- capacity_value: The seating capacity number (string; do not parse to a number)
- capacity_urls: List of URLs that explicitly support the seating capacity for that primary event type
- accessibility_statement: A phrase indicating wheelchair accessible / ADA seating availability (string)
- accessibility_urls: List of URLs that support ADA or wheelchair-accessible seating availability
- premium_types: List of premium seating types offered (e.g., "luxury suites", "club seats", "loge boxes")
- suite_capacity_range: If suites are mentioned, provide the documented guest capacity or range as text (e.g., "12-18 guests"); otherwise null
- premium_urls: List of URLs that support premium seating offerings (and suite capacity range if applicable)
- parking_description: Description of onsite parking for attendees (string)
- parking_urls: List of URLs that support parking availability (garages or surface lots for attendees)
- discount_type: Discount program type, either "military" or "student" (string); set to null if not clearly specified
- discount_verification_requirement: Stated verification requirement (e.g., "valid military ID" / "student ID") (string); set to null if not specified
- discount_urls: List of URLs supporting the discount program and verification requirement
- group_minimum_attendees: The minimum number of attendees required to qualify for group rates (string; e.g., "10" or "10+"); set to null if not specified
- group_urls: List of URLs supporting the group ticket program and minimum requirement
- season_benefits_description: Description of season ticket holder benefits (string; include any stated benefits)
- season_urls: List of URLs supporting season ticket packages/benefits
- multi_event_types: List at least two different event types the venue hosts (e.g., "football", "concerts", "basketball"); provide what the answer lists
- multi_event_urls: List of URLs supporting that the venue hosts at least two different event types

GENERAL RULES:
- Do not invent information. Return null or [] when the answer did not provide it.
- Extract URLs explicitly stated in the answer text. Return valid, complete http(s) URLs only.
- If the answer provides more than 4 venues, only include the first 4.
- If the same URL is used to justify multiple attributes, include it separately in each relevant URL list.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(vals: Optional[List[str]]) -> List[str]:
    return vals or []


def _urls_are_public(urls: List[str]) -> bool:
    if not urls:
        return False
    for u in urls:
        if not isinstance(u, str):
            return False
        u2 = u.strip().lower()
        if not (u2.startswith("http://") or u2.startswith("https://")):
            return False
        if u2.startswith("mailto:") or u2.startswith("ftp://") or u2.startswith("file://"):
            return False
    return True


def _collect_all_venue_urls(v: VenueItem) -> List[str]:
    buckets = [
        v.identity_urls,
        v.capacity_urls,
        v.accessibility_urls,
        v.premium_urls,
        v.parking_urls,
        v.discount_urls,
        v.group_urls,
        v.season_urls,
        v.multi_event_urls,
    ]
    all_urls: List[str] = []
    for b in buckets:
        if b:
            all_urls.extend(b)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _has_suite(types: List[str]) -> bool:
    for t in types or []:
        t_low = t.lower()
        if "suite" in t_low or "skybox" in t_low:
            return True
    return False


def _normalize_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def _has_digits(s: Optional[str]) -> bool:
    return bool(s and re.search(r"\d", s))


# --------------------------------------------------------------------------- #
# Verification builder for a single venue                                     #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx1_based: int,
) -> None:
    """
    Build verification sub-tree and run checks for one venue.
    """
    prefix = f"venue_{idx1_based}"

    # Venue container node (non-critical to allow partial credit per venue)
    venue_node = evaluator.add_parallel(
        id=prefix,
        desc=f"Venue {idx1_based} (qualifying venue with all required attributes and citations)",
        parent=parent_node,
        critical=False,
    )

    # ---------------- Identification ----------------
    ident_node = evaluator.add_parallel(
        id=f"{prefix}_identification",
        desc="Venue name and Louisiana location (city, Louisiana) provided",
        parent=venue_node,
        critical=True,
    )

    # name provided (existence)
    evaluator.add_custom_node(
        result=(venue.name is not None and venue.name.strip() != ""),
        id=f"{prefix}_name_provided",
        desc="Venue name is stated",
        parent=ident_node,
        critical=True,
    )

    # location louisiana (verify with identity urls)
    loc_leaf = evaluator.add_leaf(
        id=f"{prefix}_location_louisiana",
        desc="Venue location explicitly indicates it is in Louisiana",
        parent=ident_node,
        critical=True,
    )
    city_part = f"{venue.city}, Louisiana" if venue.city else "Louisiana"
    claim_loc = f"The venue named '{venue.name or ''}' is located in {city_part}."
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=_safe_list(venue.identity_urls),
        additional_instruction="Confirm from the provided URL(s) that the venue is in Louisiana. If a city is given, verify the city is in Louisiana (e.g., 'LA' abbreviation is acceptable).",
    )

    # identity/location reference URL(s) provided
    evaluator.add_custom_node(
        result=_urls_are_public(_safe_list(venue.identity_urls)),
        id=f"{prefix}_location_reference",
        desc="Reference URL verifying venue identity/location is provided",
        parent=ident_node,
        critical=True,
    )

    # ---------------- Capacity ----------------
    cap_node = evaluator.add_parallel(
        id=f"{prefix}_capacity",
        desc="Documented seating capacity for the venue’s primary event type is provided and cited",
        parent=venue_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(venue.capacity_primary_event is not None and venue.capacity_primary_event.strip() != ""),
        id=f"{prefix}_primary_event_type_stated",
        desc="Primary event type for which capacity is given is stated (e.g., basketball/football/concerts)",
        parent=cap_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_digits(venue.capacity_value),
        id=f"{prefix}_capacity_value",
        desc="Specific seating capacity number is stated",
        parent=cap_node,
        critical=True,
    )

    cap_ref_leaf = evaluator.add_leaf(
        id=f"{prefix}_capacity_reference",
        desc="Reference URL verifying the seating capacity is provided",
        parent=cap_node,
        critical=True,
    )
    claim_cap = f"The seating capacity for {venue.capacity_primary_event or 'the primary event type'} at '{venue.name or ''}' is {venue.capacity_value or ''}."
    await evaluator.verify(
        claim=claim_cap,
        node=cap_ref_leaf,
        sources=_safe_list(venue.capacity_urls),
        additional_instruction="Verify the stated capacity matches what the page says for the specified primary event type. Allow minor formatting differences (e.g., commas).",
    )

    # ---------------- Accessibility (ADA) ----------------
    acc_node = evaluator.add_parallel(
        id=f"{prefix}_accessibility",
        desc="Wheelchair-accessible seating that complies with ADA standards is confirmed and cited",
        parent=venue_node,
        critical=True,
    )

    acc_conf_leaf = evaluator.add_leaf(
        id=f"{prefix}_accessibility_confirmation",
        desc="Wheelchair-accessible seating / ADA accessibility is explicitly stated",
        parent=acc_node,
        critical=True,
    )
    claim_ada = f"'{venue.name or ''}' provides wheelchair-accessible seating that complies with ADA standards (or explicitly mentions ADA/accessible seating)."
    await evaluator.verify(
        claim=claim_ada,
        node=acc_conf_leaf,
        sources=_safe_list(venue.accessibility_urls),
        additional_instruction="Confirm the page explicitly refers to ADA accessibility or wheelchair-accessible seating at the venue.",
    )

    evaluator.add_custom_node(
        result=_urls_are_public(_safe_list(venue.accessibility_urls)),
        id=f"{prefix}_accessibility_reference",
        desc="Reference URL verifying ADA/wheelchair-accessible seating is provided",
        parent=acc_node,
        critical=True,
    )

    # ---------------- Premium Seating ----------------
    prem_node = evaluator.add_parallel(
        id=f"{prefix}_premium_seating",
        desc="At least one premium seating option is identified and cited (and suite guest-capacity range is included if suites are offered)",
        parent=venue_node,
        critical=True,
    )

    prem_types_leaf = evaluator.add_leaf(
        id=f"{prefix}_premium_type",
        desc="Type(s) of premium seating offered are identified (e.g., luxury suites/club seats/loge boxes)",
        parent=prem_node,
        critical=True,
    )
    types_str = ", ".join(venue.premium_types) if venue.premium_types else "premium seating"
    claim_prem = f"'{venue.name or ''}' offers premium seating options such as {types_str}."
    await evaluator.verify(
        claim=claim_prem,
        node=prem_types_leaf,
        sources=_safe_list(venue.premium_urls),
        additional_instruction="Verify that the page lists premium seating offerings (e.g., suites, club seats, loge boxes). Minor naming variations are acceptable.",
    )

    # Suite capacity range if applicable (conditional)
    suites_offered = _has_suite(venue.premium_types)
    if suites_offered and venue.suite_capacity_range and venue.suite_capacity_range.strip():
        suite_leaf = evaluator.add_leaf(
            id=f"{prefix}_suite_capacity_range_if_applicable",
            desc="If luxury suites are offered/mentioned, a documented guest-capacity range for suites is provided",
            parent=prem_node,
            critical=True,
        )
        claim_suite = f"The suites at '{venue.name or ''}' have a documented guest capacity or range of {venue.suite_capacity_range}."
        await evaluator.verify(
            claim=claim_suite,
            node=suite_leaf,
            sources=_safe_list(venue.premium_urls),
            additional_instruction="Verify the suite capacity or capacity range is explicitly documented on the page.",
        )
    else:
        # If no suites offered, pass; if suites offered but no range provided, fail
        evaluator.add_custom_node(
            result=(not suites_offered) or (bool(venue.suite_capacity_range and venue.suite_capacity_range.strip())),
            id=f"{prefix}_suite_capacity_range_if_applicable",
            desc="If luxury suites are offered/mentioned, a documented guest-capacity range for suites is provided",
            parent=prem_node,
            critical=True,
        )

    evaluator.add_custom_node(
        result=_urls_are_public(_safe_list(venue.premium_urls)),
        id=f"{prefix}_premium_reference",
        desc="Reference URL verifying premium seating (and suite capacity range if applicable) is provided",
        parent=prem_node,
        critical=True,
    )

    # ---------------- Parking ----------------
    park_node = evaluator.add_parallel(
        id=f"{prefix}_parking",
        desc="On-site parking facilities for ticket holders are described and cited",
        parent=venue_node,
        critical=True,
    )

    park_leaf = evaluator.add_leaf(
        id=f"{prefix}_parking_details",
        desc="On-site parking options are described (e.g., garages/surface lots for attendees/ticket holders)",
        parent=park_node,
        critical=True,
    )
    claim_park = f"'{venue.name or ''}' provides on-site parking for attendees (garages or surface lots available to ticket holders)."
    await evaluator.verify(
        claim=claim_park,
        node=park_leaf,
        sources=_safe_list(venue.parking_urls),
        additional_instruction="Confirm that on-site parking is available to attendees, such as garages or surface lots on or adjacent to the venue campus.",
    )

    evaluator.add_custom_node(
        result=_urls_are_public(_safe_list(venue.parking_urls)),
        id=f"{prefix}_parking_reference",
        desc="Reference URL verifying parking availability is provided",
        parent=park_node,
        critical=True,
    )

    # ---------------- Discounts ----------------
    disc_node = evaluator.add_parallel(
        id=f"{prefix}_discounts",
        desc="Military OR student discount program is described (including verification requirement) and cited",
        parent=venue_node,
        critical=True,
    )

    disc_type_leaf = evaluator.add_leaf(
        id=f"{prefix}_discount_type",
        desc="Discount type is identified (military or student)",
        parent=disc_node,
        critical=True,
    )
    which_disc = (venue.discount_type or "a military or student").strip()
    claim_disc_type = f"'{venue.name or ''}' offers {which_disc} ticket discounts."
    await evaluator.verify(
        claim=claim_disc_type,
        node=disc_type_leaf,
        sources=_safe_list(venue.discount_urls),
        additional_instruction="Verify that the page mentions military or student ticket discounts. Either one is acceptable.",
    )

    disc_verif_leaf = evaluator.add_leaf(
        id=f"{prefix}_discount_verification",
        desc="Verification requirement is stated (e.g., military ID/student ID or equivalent proof)",
        parent=disc_node,
        critical=True,
    )
    verif_text = venue.discount_verification_requirement or "a valid ID is required"
    claim_disc_req = f"For the {which_disc} discount at '{venue.name or ''}', {verif_text}."
    await evaluator.verify(
        claim=claim_disc_req,
        node=disc_verif_leaf,
        sources=_safe_list(venue.discount_urls),
        additional_instruction="Confirm that the discount requires verification (e.g., valid military ID or student ID). Minor wording variations are acceptable.",
    )

    evaluator.add_custom_node(
        result=_urls_are_public(_safe_list(venue.discount_urls)),
        id=f"{prefix}_discount_reference",
        desc="Reference URL verifying discount policy is provided",
        parent=disc_node,
        critical=True,
    )

    # ---------------- Group Tickets ----------------
    group_node = evaluator.add_parallel(
        id=f"{prefix}_group_tickets",
        desc="Group ticket program is described with a minimum attendee requirement and cited",
        parent=venue_node,
        critical=True,
    )

    group_min_leaf = evaluator.add_leaf(
        id=f"{prefix}_group_minimum",
        desc="Minimum number of attendees required for group rates is specified",
        parent=group_node,
        critical=True,
    )
    claim_group = f"Group ticket rates at '{venue.name or ''}' require a minimum of {venue.group_minimum_attendees or 'a specified minimum'} attendees."
    await evaluator.verify(
        claim=claim_group,
        node=group_min_leaf,
        sources=_safe_list(venue.group_urls),
        additional_instruction="Verify the page states a concrete minimum number required to qualify for group rates (e.g., 10, 15, 20+).",
    )

    evaluator.add_custom_node(
        result=_urls_are_public(_safe_list(venue.group_urls)),
        id=f"{prefix}_group_reference",
        desc="Reference URL verifying group ticket policy is provided",
        parent=group_node,
        critical=True,
    )

    # ---------------- Season Tickets ----------------
    season_node = evaluator.add_parallel(
        id=f"{prefix}_season_tickets",
        desc="Season ticket packages are described with documented benefits (including specific privileges) and cited",
        parent=venue_node,
        critical=True,
    )

    season_benef_leaf = evaluator.add_leaf(
        id=f"{prefix}_season_benefits",
        desc="Season ticket holder benefits are described",
        parent=season_node,
        critical=True,
    )
    claim_season = f"'{venue.name or ''}' offers season ticket packages with benefits (e.g., {venue.season_benefits_description or 'documented benefits'})."
    await evaluator.verify(
        claim=claim_season,
        node=season_benef_leaf,
        sources=_safe_list(venue.season_urls),
        additional_instruction="Verify the page describes benefits for season ticket holders (e.g., priority access, discounts, playoff priority, parking privileges).",
    )

    season_priv_leaf = evaluator.add_leaf(
        id=f"{prefix}_season_privilege_requirement",
        desc="Benefits include at least one specific privilege such as playoff ticket priority and/or parking advantages",
        parent=season_node,
        critical=True,
    )
    claim_season_priv = f"The season ticket benefits at '{venue.name or ''}' include at least one of: playoff ticket priority or parking privileges."
    await evaluator.verify(
        claim=claim_season_priv,
        node=season_priv_leaf,
        sources=_safe_list(venue.season_urls),
        additional_instruction="Look for explicit mention of playoff ticket priority and/or parking privileges as part of season ticket benefits. Either one suffices.",
    )

    evaluator.add_custom_node(
        result=_urls_are_public(_safe_list(venue.season_urls)),
        id=f"{prefix}_season_reference",
        desc="Reference URL verifying season ticket program/benefits is provided",
        parent=season_node,
        critical=True,
    )

    # ---------------- Multi-Event Capability ----------------
    mult_node = evaluator.add_parallel(
        id=f"{prefix}_multi_event",
        desc="Venue can host at least two different event types and is cited",
        parent=venue_node,
        critical=True,
    )

    # Presence of at least two event types listed in the answer
    evaluator.add_custom_node(
        result=(len(venue.multi_event_types) >= 2),
        id=f"{prefix}_event_types",
        desc="At least two different event types the venue hosts are listed",
        parent=mult_node,
        critical=True,
    )

    event_ref_leaf = evaluator.add_leaf(
        id=f"{prefix}_event_reference",
        desc="Reference URL verifying multi-event hosting is provided",
        parent=mult_node,
        critical=True,
    )
    # Use the first two as examples if present
    ev_a = venue.multi_event_types[0] if len(venue.multi_event_types) >= 1 else "one event type"
    ev_b = venue.multi_event_types[1] if len(venue.multi_event_types) >= 2 else "another event type"
    claim_events = f"'{venue.name or ''}' hosts at least two different event types, such as {ev_a} and {ev_b}."
    await evaluator.verify(
        claim=claim_events,
        node=event_ref_leaf,
        sources=_safe_list(venue.multi_event_urls),
        additional_instruction="Confirm from the page that the venue hosts multiple types of events (e.g., sports and concerts). It does not have to be an exhaustive list.",
    )

    # ---------------- Public references (http/https) ----------------
    all_urls = _collect_all_venue_urls(venue)
    evaluator.add_custom_node(
        result=_urls_are_public(all_urls),
        id=f"{prefix}_public_references",
        desc="Reference URLs are provided as publicly accessible URLs (http/https) for the venue’s required attributes",
        parent=venue_node,
        critical=True,
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
    """
    Evaluate an answer for the task:
    Identify four distinct entertainment venues in Louisiana that meet all specified requirements.
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
        default_model=model,
    )

    # Extract venues
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    venues: List[VenueItem] = list(extracted.venues or [])
    # Keep only first 4 items
    venues = venues[:4]
    # Pad if fewer than 4 venues to keep evaluation tree shape deterministic
    while len(venues) < 4:
        venues.append(VenueItem())

    # Critical distinctness check across 4 venues
    names = [_normalize_name(v.name) for v in venues]
    nonempty = [n for n in names if n]
    all_four_present = len(nonempty) == 4
    all_unique = len(set(nonempty)) == len(nonempty) == 4
    evaluator.add_custom_node(
        result=(all_four_present and all_unique),
        id="venues_distinct",
        desc="All four venues are distinct (no duplicate venue names/entities)",
        parent=root,
        critical=True,
    )

    # Build per-venue verification subtrees
    tasks: List[asyncio.Task] = []
    for i in range(4):
        coro = verify_single_venue(evaluator, root, venues[i], i + 1)
        tasks.append(asyncio.create_task(coro))

    # Run verifications concurrently
    await asyncio.gather(*tasks)

    # Return the evaluation summary
    return evaluator.get_summary()