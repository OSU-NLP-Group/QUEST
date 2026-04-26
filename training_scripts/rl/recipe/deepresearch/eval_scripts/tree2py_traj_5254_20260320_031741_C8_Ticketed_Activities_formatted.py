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
TASK_ID = "tx_concert_venues_requirements"
TASK_DESCRIPTION = """
Identify three concert venues located in Texas that meet all of the following requirements for hosting a live music series:

1. Capacity: Each venue must have a documented seating capacity between 1,000 and 6,500 attendees.
2. Accessibility: Each venue must provide wheelchair-accessible seating in compliance with ADA standards appropriate for its capacity range.
3. Age Policy: Each venue must have a clearly documented age admission policy.
4. Group Booking: Each venue must offer group booking options with a minimum threshold of 10 or more attendees qualifying for group rates.
5. Ticket Purchase Limits: Each venue must have a documented maximum ticket purchase limit per transaction.
6. Location: All venues must be physically located within the state of Texas.

For each venue, provide:
- The venue name
- Documentation of how it meets each of the six requirements listed above
- A valid reference URL supporting each requirement
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None

    capacity_text: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)

    accessibility_text: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=list)

    age_policy_text: Optional[str] = None
    age_policy_urls: List[str] = Field(default_factory=list)

    group_min_size_text: Optional[str] = None
    group_booking_urls: List[str] = Field(default_factory=list)

    ticket_limit_text: Optional[str] = None
    ticket_limit_urls: List[str] = Field(default_factory=list)

    location_text: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract all concert venues listed in the answer that are candidates for a Texas live music series. Return them in the order they appear.
    For each venue, extract the following fields exactly as presented in the answer:

    - name: Venue name (string)
    - capacity_text: The capacity figure or statement as written in the answer (e.g., "2,800 seats", "capacity 4,500", "approximately 1,200"), or null if not stated.
    - capacity_urls: A list of URL(s) cited in the answer that specifically support the capacity information.
    - accessibility_text: The ADA/wheelchair-accessible seating statement as written in the answer (e.g., "wheelchair accessible seating available", "ADA compliant seating"), or null if not stated.
    - accessibility_urls: A list of URL(s) cited that support the accessibility information.
    - age_policy_text: The age admission policy statement as written in the answer (e.g., "All ages", "18+ only", "Under 18 must be accompanied by an adult"), or null if not stated.
    - age_policy_urls: A list of URL(s) cited that support the age policy.
    - group_min_size_text: The minimum group size requirement as written (e.g., "groups of 10+", "minimum of 12"), or null if not stated.
    - group_booking_urls: A list of URL(s) cited that support group booking information.
    - ticket_limit_text: The maximum ticket purchase limit per transaction/order/customer as written (e.g., "limit 6 tickets per order"), or null if not stated.
    - ticket_limit_urls: A list of URL(s) cited that support the ticket limit information.
    - location_text: The location/address as written (e.g., "Austin, TX", "Houston, Texas"), or null if not stated.
    - location_urls: A list of URL(s) cited that support the location (address) in Texas.

    Rules:
    - Extract only what is explicitly present in the answer.
    - For any missing text value, return null.
    - For any set of URLs, return only the URLs explicitly present in the answer text (valid URLs only). If none, return an empty list.
    - Do not invent or infer information. Do not copy from your knowledge.
    - Return a JSON object with a top-level key 'venues' that is an array of venue objects following the schema above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    clean = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s:
                clean.append(s)
    return clean


def _first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    try:
        nums = re.findall(r"\d{1,6}", text.replace(",", ""))
        if not nums:
            return None
        return int(nums[0])
    except Exception:
        return None


def _ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third"}
    return mapping.get(n, f"#{n}")


# --------------------------------------------------------------------------- #
# Claim builders                                                              #
# --------------------------------------------------------------------------- #
def build_capacity_claim(v: VenueItem) -> str:
    name = v.name or "the venue"
    # Incorporate the stated capacity text if available to help the judge
    if v.capacity_text:
        return f"The venue '{name}' has a documented seating or attendance capacity that is between 1,000 and 6,500. The answer cites: {v.capacity_text}."
    return f"The venue '{name}' has a documented seating or attendance capacity between 1,000 and 6,500."


def build_accessibility_claim(v: VenueItem) -> str:
    name = v.name or "the venue"
    if v.accessibility_text:
        return f"The venue '{name}' provides wheelchair-accessible seating and ADA-compliant accommodations. The answer cites: {v.accessibility_text}."
    return f"The venue '{name}' provides wheelchair-accessible seating and ADA-compliant accommodations."


def build_age_policy_claim(v: VenueItem) -> str:
    name = v.name or "the venue"
    if v.age_policy_text:
        return f"The venue '{name}' has a clearly documented age admission policy: {v.age_policy_text}."
    return f"The venue '{name}' has a clearly documented age admission policy."


def build_group_booking_claim(v: VenueItem) -> str:
    name = v.name or "the venue"
    min_size = _first_int(v.group_min_size_text)
    if min_size is not None:
        return f"The venue '{name}' offers group booking options with a minimum threshold of at least 10 attendees (the policy specifies {min_size})."
    return f"The venue '{name}' offers group booking options with a minimum threshold of at least 10 attendees."


def build_ticket_limit_claim(v: VenueItem) -> str:
    name = v.name or "the venue"
    if v.ticket_limit_text:
        return f"The venue '{name}' has a documented maximum ticket purchase limit per transaction/order/customer: {v.ticket_limit_text}."
    return f"The venue '{name}' has a documented maximum ticket purchase limit per transaction/order/customer."


def build_location_claim(v: VenueItem) -> str:
    name = v.name or "the venue"
    if v.location_text:
        return f"The venue '{name}' is physically located within the state of Texas. The answer cites: {v.location_text}."
    return f"The venue '{name}' is physically located within the state of Texas."


# --------------------------------------------------------------------------- #
# Additional instructions for verifier                                        #
# --------------------------------------------------------------------------- #
ADD_INS_CAPACITY = (
    "Verify from the provided URL(s) that the page explicitly states a seating or attendance capacity that falls "
    "between 1,000 and 6,500 inclusive. Accept minor textual variants like 'approx.' or 'about'. If multiple capacities are "
    "listed, use the primary capacity for concerts or seating/attendance. If the page implies a capacity above 6,500 or below 1,000, it should fail."
)

ADD_INS_ACCESS = (
    "Verify from the provided URL(s) that the venue offers wheelchair-accessible seating or ADA-compliant seating/areas. "
    "Look for terms like 'ADA', 'wheelchair accessible seating', 'accessible seating', 'mobility accommodations'."
)

ADD_INS_AGE = (
    "Verify from the provided URL(s) that an age admission policy is stated (e.g., 'All ages', '18+ only', "
    "'under 18 must be accompanied by an adult', '21+'). The policy text should be clearly documented."
)

ADD_INS_GROUP = (
    "Verify from the provided URL(s) that group booking/sales/discounts are offered and that the minimum group size "
    "threshold is at least 10 attendees. Look for phrases like 'group tickets', 'group sales', 'minimum of 10', '10+', etc."
)

ADD_INS_TICKET_LIMIT = (
    "Verify from the provided URL(s) that there is a maximum ticket purchase limit per transaction/order/customer/account/household. "
    "Accept synonymous phrasing like 'per order', 'per transaction', 'per customer', 'per account', 'per household'."
)

ADD_INS_LOCATION = (
    "Verify from the provided URL(s) that the physical venue is located in Texas (TX). Accept addresses listing a Texas city with 'TX' or 'Texas'. "
    "Ensure it refers to the physical venue location (not corporate HQ elsewhere)."
)


# --------------------------------------------------------------------------- #
# Venue verification                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(evaluator: Evaluator, parent_node, v: VenueItem, idx: int) -> None:
    """
    Build the verification sub-tree for a single venue and run all checks.
    We use requirement-specific 'blocks' so that each requirement's source-existence gate only affects its own check.
    """
    ordinal = _ordinal(idx)
    venue_node = evaluator.add_parallel(
        id=f"venue_{idx}",
        desc=f"{ordinal} venue meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Prepare sanitized URL lists up front
    cap_urls = _sanitize_urls(v.capacity_urls)
    acc_urls = _sanitize_urls(v.accessibility_urls)
    age_urls = _sanitize_urls(v.age_policy_urls)
    grp_urls = _sanitize_urls(v.group_booking_urls)
    lim_urls = _sanitize_urls(v.ticket_limit_urls)
    loc_urls = _sanitize_urls(v.location_urls)

    # Collect batch verifications
    batch_items = []

    # 1) Capacity
    cap_block = evaluator.add_parallel(
        id=f"venue_{idx}_capacity_block",
        desc=f"Capacity requirement verification for venue #{idx}",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(cap_urls) > 0,
        id=f"venue_{idx}_capacity_source_exists",
        desc=f"At least one reference URL is provided for capacity of venue #{idx}",
        parent=cap_block,
        critical=True
    )
    cap_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_capacity",
        desc="Venue has a documented seating capacity between 1,000 and 6,500 attendees, with valid reference URL provided",
        parent=cap_block,
        critical=True
    )
    batch_items.append((build_capacity_claim(v), cap_urls, cap_leaf, ADD_INS_CAPACITY))

    # 2) Accessibility
    acc_block = evaluator.add_parallel(
        id=f"venue_{idx}_accessibility_block",
        desc=f"Accessibility (ADA) verification for venue #{idx}",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(acc_urls) > 0,
        id=f"venue_{idx}_accessibility_source_exists",
        desc=f"At least one reference URL is provided for accessibility of venue #{idx}",
        parent=acc_block,
        critical=True
    )
    acc_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_accessibility",
        desc="Venue provides wheelchair-accessible seating in compliance with ADA standards for its capacity range, with valid reference URL provided",
        parent=acc_block,
        critical=True
    )
    batch_items.append((build_accessibility_claim(v), acc_urls, acc_leaf, ADD_INS_ACCESS))

    # 3) Age policy
    age_block = evaluator.add_parallel(
        id=f"venue_{idx}_age_policy_block",
        desc=f"Age admission policy verification for venue #{idx}",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(age_urls) > 0,
        id=f"venue_{idx}_age_policy_source_exists",
        desc=f"At least one reference URL is provided for age policy of venue #{idx}",
        parent=age_block,
        critical=True
    )
    age_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_age_policy",
        desc="Venue has a clearly documented age admission policy, with valid reference URL provided",
        parent=age_block,
        critical=True
    )
    batch_items.append((build_age_policy_claim(v), age_urls, age_leaf, ADD_INS_AGE))

    # 4) Group booking (>= 10)
    grp_block = evaluator.add_parallel(
        id=f"venue_{idx}_group_booking_block",
        desc=f"Group booking verification for venue #{idx}",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(grp_urls) > 0,
        id=f"venue_{idx}_group_booking_source_exists",
        desc=f"At least one reference URL is provided for group booking of venue #{idx}",
        parent=grp_block,
        critical=True
    )
    grp_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_group_booking",
        desc="Venue offers group booking options with minimum threshold of 10 or more attendees, with valid reference URL provided",
        parent=grp_block,
        critical=True
    )
    batch_items.append((build_group_booking_claim(v), grp_urls, grp_leaf, ADD_INS_GROUP))

    # 5) Ticket purchase limit
    lim_block = evaluator.add_parallel(
        id=f"venue_{idx}_ticket_limit_block",
        desc=f"Ticket purchase limit verification for venue #{idx}",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(lim_urls) > 0,
        id=f"venue_{idx}_ticket_limit_source_exists",
        desc=f"At least one reference URL is provided for ticket purchase limit of venue #{idx}",
        parent=lim_block,
        critical=True
    )
    lim_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_ticket_limit",
        desc="Venue has a documented maximum ticket purchase limit per transaction, with valid reference URL provided",
        parent=lim_block,
        critical=True
    )
    batch_items.append((build_ticket_limit_claim(v), lim_urls, lim_leaf, ADD_INS_TICKET_LIMIT))

    # 6) Location in Texas
    loc_block = evaluator.add_parallel(
        id=f"venue_{idx}_location_block",
        desc=f"Location verification for venue #{idx}",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(loc_urls) > 0,
        id=f"venue_{idx}_location_source_exists",
        desc=f"At least one reference URL is provided for location of venue #{idx}",
        parent=loc_block,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_location",
        desc="Venue is physically located in Texas, with valid reference URL provided",
        parent=loc_block,
        critical=True
    )
    batch_items.append((build_location_claim(v), loc_urls, loc_leaf, ADD_INS_LOCATION))

    # Run all six verifications for this venue in parallel
    await evaluator.batch_verify(batch_items)


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
    Evaluate an answer for the Texas concert venues requirements task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent venues; allow partial credit
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

    # Extract structured venue data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    venues_all = extracted.venues if extracted and extracted.venues else []

    # Keep only the first 3 venues; pad with empty placeholders if fewer than 3
    venues_first_three: List[VenueItem] = list(venues_all[:3])
    while len(venues_first_three) < 3:
        venues_first_three.append(VenueItem())

    # Optional info for summary/debug
    evaluator.add_custom_info(
        info={
            "total_venues_found_in_answer": len(venues_all),
            "venues_used_for_evaluation": min(3, len(venues_all))
        },
        info_type="extraction_stats",
        info_name="extraction_stats"
    )

    # Build verification for each of the three venues
    for i, v in enumerate(venues_first_three, start=1):
        await verify_single_venue(evaluator, root, v, i)

    return evaluator.get_summary()