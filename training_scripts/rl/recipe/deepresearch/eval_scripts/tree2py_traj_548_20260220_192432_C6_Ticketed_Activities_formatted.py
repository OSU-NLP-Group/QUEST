import asyncio
import logging
import math
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "soCal_theaters_march_2026"
TASK_DESCRIPTION = (
    "Identify 4 performing arts theaters in Southern California (Los Angeles, Orange, San Diego, or Riverside counties) "
    "that are suitable for hosting a touring Broadway-style musical production during March 2026. Each venue must meet the following requirements: "
    "(1) Seating capacity between 800 and 2,500 seats; "
    "(2) ADA compliant with required wheelchair-accessible seating (minimum 1% of total capacity); "
    "(3) Professional stage facilities suitable for musical theater productions; "
    "(4) Available for booking during March 2026. "
    "For each venue, provide: venue name and specific location (city and address), total seating capacity, number of wheelchair-accessible seats, "
    "confirmation of stage facilities suitable for musical theater, and evidence of availability or booking information for March 2026."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None  # street address line(s)
    city: Optional[str] = None
    county: Optional[str] = None  # if explicitly provided
    capacity: Optional[str] = None  # keep as string to tolerate formats (e.g., "1,500")
    wheelchair_seats: Optional[str] = None  # string to tolerate formats (e.g., "16+/at least 16")
    stage_facilities: Optional[str] = None  # short description or list captured as text
    availability_desc: Optional[str] = None  # any statement about March 2026 availability provided by the answer
    basic_urls: List[str] = Field(default_factory=list)  # URLs supporting name/location/capacity
    compliance_urls: List[str] = Field(default_factory=list)  # URLs supporting ADA & stage facilities
    availability_urls: List[str] = Field(default_factory=list)  # URLs supporting March 2026 availability/booking


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to 4 performing arts venues from the answer in the order they are presented (if more than 4 are present, keep only the first 4; if fewer than 4, return whatever is present). 
    For each venue, return the following fields:

    - name: The venue name exactly as written.
    - address: The full street address line(s) (if given). If only partial, extract whatever is present.
    - city: The city name (if given).
    - county: The county if explicitly provided (e.g., "Los Angeles County"); otherwise return null.
    - capacity: The total seating capacity as it appears (e.g., "1,500", "1500 seats"). Do not convert to number; preserve the text.
    - wheelchair_seats: The number of wheelchair-accessible seats as stated (e.g., "16", "at least 16"). If not provided, return null.
    - stage_facilities: A short text snippet that summarizes whether the venue has professional stage facilities suitable for Broadway-style musical productions (e.g., mentions of proscenium stage, fly system, orchestra pit, technical specifications). Extract as plain text.
    - availability_desc: A short text snippet summarizing the availability or booking information for March 2026 from the answer text (if provided). If not provided, return null.
    - basic_urls: A list of URLs that support the basic venue information (name/location/capacity). Only include URLs explicitly present in the answer text.
    - compliance_urls: A list of URLs that support ADA accessibility and/or professional stage facilities. Only include URLs explicitly present in the answer text.
    - availability_urls: A list of URLs that support availability or booking information specifically for March 2026 (e.g., rental calendar for March 2026, booking portal indicating 2026 reservations accepted). Only include URLs explicitly present in the answer text.

    SPECIAL RULES:
    - Extract URLs only if they appear explicitly in the answer (plain links or links in markdown). Do not invent or guess URLs.
    - Keep capacity and wheelchair seat counts exactly as shown (strings). Do not normalize to numbers.
    - If any field is missing in the answer for a venue, return null for that field (or empty list for URL arrays).
    - The 'venues' array should include at most 4 venue objects.

    Output a JSON object with a single field:
    {
      "venues": [ { ... venue fields ... }, ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_int_safe(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # If it's a range or includes words, take the first integer-like token
    nums = re.findall(r"\d{1,3}(?:,\d{3})*|\d+", text)
    if not nums:
        return None
    try:
        # Remove commas from the first number token
        n = int(nums[0].replace(",", ""))
        return n
    except Exception:
        return None


def compute_ada_minimum(capacity_int: Optional[int]) -> Optional[int]:
    if capacity_int is None:
        return None
    return max(1, math.ceil(capacity_int * 0.01))


def is_nonempty_url_list(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification logic for a single venue                                       #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    root: VerificationNode,
    venue: VenueItem,
    idx: int
) -> None:
    nth = idx + 1

    # Wrapper node for this venue (non-critical, single child)
    venue_node = evaluator.add_sequential(
        id=f"venue_{nth}",
        desc=f"{['First', 'Second', 'Third', 'Fourth'][idx]} venue meeting all specified requirements",
        parent=root,
        critical=False
    )

    # Critical verification umbrella for this venue
    verify_node = evaluator.add_parallel(
        id=f"venue_{nth}_verification",
        desc=f"Verification of {['first','second','third','fourth'][idx]} venue's compliance with all requirements",
        parent=venue_node,
        critical=True
    )

    # ---------------- Basic properties ----------------
    basic_node = evaluator.add_parallel(
        id=f"venue_{nth}_basic_properties",
        desc=f"Basic venue identification and capacity information for {['first','second','third','fourth'][idx]} venue",
        parent=verify_node,
        critical=True
    )

    # Custom node to ensure basic reference URLs exist
    basic_ref_exists = evaluator.add_custom_node(
        result=is_nonempty_url_list(venue.basic_urls),
        id=f"venue_{nth}_basic_reference",
        desc="URL reference supporting basic venue information",
        parent=basic_node,
        critical=True
    )

    # Leaf: Venue name and location (county constraint)
    name_loc_leaf = evaluator.add_leaf(
        id=f"venue_{nth}_name_location",
        desc="Venue name and complete address (city and street address) in Los Angeles, Orange, San Diego, or Riverside county",
        parent=basic_node,
        critical=True
    )
    venue_name = venue.name or ""
    address_str = venue.address or ""
    city_str = venue.city or ""
    county_str = venue.county or ""
    name_loc_claim = (
        f"The venue is named '{venue_name}' and is located at '{address_str}', {city_str}. "
        f"It is in Southern California within one of these counties: Los Angeles, Orange, San Diego, or Riverside."
    )
    await evaluator.verify(
        claim=name_loc_claim,
        node=name_loc_leaf,
        sources=venue.basic_urls if is_nonempty_url_list(venue.basic_urls) else None,
        additional_instruction=(
            "Confirm both the venue name and the stated address/city from the provided source(s). "
            "Also determine if the venue is located in one of these counties: Los Angeles, Orange, San Diego, or Riverside. "
            "If the page provides the city (e.g., Anaheim, Riverside, San Diego, Los Angeles, etc.), "
            "you may infer the county accordingly. If the page lacks sufficient location detail, consider it not supported."
        )
    )

    # Leaf: Capacity within range 800–2,500
    capacity_leaf = evaluator.add_leaf(
        id=f"venue_{nth}_capacity",
        desc="Total seating capacity between 800 and 2,500 seats",
        parent=basic_node,
        critical=True
    )
    cap_text = venue.capacity or ""
    capacity_claim = (
        f"The venue has a total seating capacity of {cap_text} seats, and this capacity lies within the range 800 to 2,500 seats."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=venue.basic_urls if is_nonempty_url_list(venue.basic_urls) else None,
        additional_instruction=(
            "Locate the capacity on the provided page. If the page lists a single number (e.g., '1,500 seats'), "
            "verify it lies within [800, 2,500]. If multiple halls are shown, ensure the stated capacity corresponds to the hall "
            "implicitly referenced by the answer. If no capacity appears on the page, or if it falls outside the range, fail."
        )
    )

    # ---------------- Compliance properties ----------------
    compliance_node = evaluator.add_parallel(
        id=f"venue_{nth}_compliance",
        desc="Compliance verification for accessibility, facilities, and availability",
        parent=verify_node,
        critical=True
    )

    # Custom node: Compliance reference URLs exist (stage/ADA)
    compliance_ref_exists = evaluator.add_custom_node(
        result=is_nonempty_url_list(venue.compliance_urls),
        id=f"venue_{nth}_compliance_reference",
        desc="URL reference supporting compliance information",
        parent=compliance_node,
        critical=True
    )

    # Leaf: Stage facilities suitable for Broadway-style musical theater
    stage_leaf = evaluator.add_leaf(
        id=f"venue_{nth}_stage_facilities",
        desc="Professional stage facilities suitable for Broadway-style musical theater productions",
        parent=compliance_node,
        critical=True
    )
    stage_desc = venue.stage_facilities or ""
    stage_claim = (
        f"The venue provides professional stage and technical facilities suitable for Broadway-style musical theater productions. "
        f"Examples can include features like a proscenium stage, fly system, orchestra pit, dressing rooms, professional sound and lighting, "
        f"and accommodating touring technical riders. Claimed features: {stage_desc}."
    )
    await evaluator.verify(
        claim=stage_claim,
        node=stage_leaf,
        sources=venue.compliance_urls if is_nonempty_url_list(venue.compliance_urls) else None,
        additional_instruction=(
            "Look for evidence on the provided page(s) that indicates professional stage capabilities suitable for touring musicals—"
            "such as a proscenium stage, fly system (rigging), orchestra pit, wing space, professional lighting and sound, "
            "loading dock, and other technical specifications or a rental/tech packet. If the page only shows a general event/ticket page "
            "without any indication of stage/technical suitability, do not consider it sufficient."
        )
    )

    # Leaf: Availability or booking information for March 2026
    availability_leaf = evaluator.add_leaf(
        id=f"venue_{nth}_availability",
        desc="Evidence of availability or booking capability during March 2026",
        parent=compliance_node,
        critical=True
    )
    avail_desc = venue.availability_desc or ""
    availability_claim = (
        "The venue has evidence of availability or booking capability specifically for March 2026 (e.g., rental calendar for March 2026, "
        "a booking portal indicating reservations are accepted for March 2026, or explicit language that 2026 bookings including March are open). "
        f"Claim context from the answer: {avail_desc}"
    )
    await evaluator.verify(
        claim=availability_claim,
        node=availability_leaf,
        sources=venue.availability_urls if is_nonempty_url_list(venue.availability_urls) else None,
        additional_instruction=(
            "Acceptable support includes: (1) a rental/availability calendar showing March 2026 (with open or bookable dates), "
            "(2) an official rental/booking page that explicitly indicates bookings are accepted for dates in 2026 including March "
            "(e.g., 'Now booking 2026' or a date selector showing March 2026), or "
            "(3) an official policy/rental PDF that specifies booking windows covering March 2026. "
            "A generic contact form without any indication of 2026 availability is not sufficient."
        )
    )

    # Leaf: Accessibility (ADA 1% rule)
    accessibility_leaf = evaluator.add_leaf(
        id=f"venue_{nth}_accessibility",
        desc="Wheelchair-accessible seating meeting ADA requirement (minimum 1% of total capacity, calculated based on stated capacity)",
        parent=compliance_node,
        critical=True
    )
    capacity_int = parse_int_safe(venue.capacity)
    ada_min = compute_ada_minimum(capacity_int)
    wh_text = venue.wheelchair_seats or ""
    if ada_min is not None:
        ada_instr_extra = f"Based on the provided capacity, at least {ada_min} wheelchair-accessible seats are required (1% of capacity). "
    else:
        ada_instr_extra = "If capacity could not be determined from the answer, you should still look for explicit statements of wheelchair-accessible seating counts and ADA compliance. "

    accessibility_claim = (
        f"The venue provides ADA-compliant wheelchair-accessible seating meeting or exceeding 1% of total capacity. "
        f"The answer states wheelchair-accessible seats: '{wh_text}'. "
        f"Total capacity text: '{cap_text}'."
    )
    await evaluator.verify(
        claim=accessibility_claim,
        node=accessibility_leaf,
        sources=venue.compliance_urls if is_nonempty_url_list(venue.compliance_urls) else None,
        additional_instruction=(
            ada_instr_extra +
            "Verify from the provided page(s) that the venue has a stated number of wheelchair-accessible seats "
            "that meets or exceeds the 1% threshold. If the page only states 'ADA compliant' without a count or clear capacity-based compliance, "
            "do not consider it sufficient for the numeric requirement."
        ),
        extra_prerequisites=[capacity_leaf]  # Gate on capacity check
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
    # Initialize evaluator with a parallel root (4 venues evaluated independently)
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

    # Extract up to 4 venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    venues: List[VenueItem] = (extracted.venues or [])[:4]
    # Pad to exactly 4 venues for the evaluation tree
    while len(venues) < 4:
        venues.append(VenueItem())

    # Verify each of the 4 venues
    for i in range(4):
        await verify_single_venue(evaluator, root, venues[i], i)

    return evaluator.get_summary()