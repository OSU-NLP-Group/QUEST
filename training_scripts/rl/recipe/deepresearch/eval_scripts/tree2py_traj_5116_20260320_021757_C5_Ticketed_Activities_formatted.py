import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "comedy_venues_accessibility"
TASK_DESCRIPTION = """Find three comedy venues in different U.S. cities that regularly host live stand-up comedy shows. For each venue, provide the following information:

1. Venue name and city location
2. Complete physical address (street address, city, state, ZIP code)
3. Venue capacity (total number of seats or standing room capacity)
4. Wheelchair accessibility: Confirmation that the venue is wheelchair accessible (via elevator, ramp, or ground-level access)
5. Accessible seating: Confirmation that accessible seating is available for wheelchair users
6. Age restriction policy: The venue's age requirement for entry (e.g., 21+, All Ages, 18+)
7. Drink minimum policy: Whether the venue has a drink minimum requirement and what it is (or state if no minimum exists)
8. Online ticketing: Confirmation that tickets can be purchased online through the venue's official website or an official ticketing platform
9. Refund/cancellation policy: The venue's policy on refunds or cancellations
10. Reference URL: A link to the venue's official website or official information page where this information can be verified

All three venues must be located in different U.S. cities, and each venue must be a dedicated comedy club or theater that regularly hosts live comedy performances (not one-time events or festivals)."""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Venue(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    street_address: Optional[str] = None
    full_address: Optional[str] = None  # If the answer gives a single-line complete address
    capacity: Optional[str] = None
    wheelchair_accessible: Optional[str] = None  # e.g., "Yes - ramp at entrance", "No"
    accessible_seating: Optional[str] = None     # e.g., "Yes - ADA seating available"
    age_restriction: Optional[str] = None        # e.g., "21+", "18+", "All Ages"
    drink_minimum: Optional[str] = None          # e.g., "2 drink minimum", "No drink minimum"
    online_ticketing: Optional[str] = None       # e.g., "Yes - Ticketmaster", "Yes - via official site", "No"
    refund_policy: Optional[str] = None          # e.g., "All sales final", "Refunds up to 24 hours before show"
    reference_url: Optional[str] = None          # official URL or official ticketing platform URL


class VenuesExtraction(BaseModel):
    venues: List[Venue] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to the first three comedy venues presented in the answer that regularly host live stand-up comedy shows (dedicated comedy clubs or theaters; not one-time events or festivals).

    For each venue, extract the following fields exactly as provided in the answer:
    - name: Venue name
    - city: City where the venue is located (city only, no state here)
    - state: 2-letter state abbreviation if present (e.g., CA, NY), otherwise the full state name if that is how the answer presents it
    - zip_code: ZIP code if provided
    - street_address: Street address including number and street name (e.g., "123 Main St"), if provided
    - full_address: A single-line complete address if the answer provides it as one line (e.g., "123 Main St, Springfield, IL 62704"). If the address is only given in parts, set this to null.
    - capacity: Total capacity or seating count as stated (leave as text; do not normalize numbers)
    - wheelchair_accessible: The statement about wheelchair accessibility (e.g., "Yes - ramp at entrance", "Yes - elevator available", "No", or similar phrasing)
    - accessible_seating: The statement about accessible/ADA seating (e.g., "Yes - ADA seating available", "No", or similar)
    - age_restriction: The stated age policy (e.g., "21+", "18+", "All Ages", or specific conditions)
    - drink_minimum: The stated drink minimum policy (e.g., "2 drink minimum", "No drink minimum")
    - online_ticketing: Whether and how online ticket purchases are supported (e.g., "Yes - via official site", "Yes - Ticketmaster", "No")
    - refund_policy: The stated refund or cancellation policy text
    - reference_url: A single official URL (the venue’s own site or an official ticketing platform page) that supports most of the information above. If multiple are listed, choose the most authoritative/official one.

    Rules:
    - Do not invent or infer details that are not in the answer; missing items must be null.
    - Preserve the exact phrasing of policies as written in the answer whenever possible.
    - Only include comedy venues (not individual events or festivals).
    - Return a JSON object with a 'venues' array of up to 3 items in the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def assemble_full_address(v: Venue) -> Optional[str]:
    # Prefer the provided single-line full_address if present
    if v.full_address and v.full_address.strip():
        return v.full_address.strip()

    parts = []
    if v.street_address and v.street_address.strip():
        parts.append(v.street_address.strip())
    city_line = []
    if v.city and v.city.strip():
        city_line.append(v.city.strip())
    if v.state and v.state.strip():
        city_line.append(v.state.strip())
    # Join "city, state"
    if city_line:
        parts.append(", ".join([city_line[0]] + city_line[1:]) if len(city_line) > 1 else city_line[0])
    # Add ZIP if present
    if v.zip_code and v.zip_code.strip():
        if parts:
            parts[-1] = f"{parts[-1]} {v.zip_code.strip()}"
        else:
            parts.append(v.zip_code.strip())

    if parts:
        return ", ".join(parts) if len(parts) > 1 else parts[0]
    return None


def sources_for_venue(v: Venue) -> Optional[str]:
    return v.reference_url if is_valid_url(v.reference_url) else None


def normalize_city(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    return city.strip().lower()


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    root: Any,
    venue: Venue,
    idx: int
) -> None:
    # Venue container node (non-critical to allow partial credit across venues)
    venue_id = f"Venue_{idx + 1}"
    venue_node = evaluator.add_parallel(
        id=venue_id,
        desc=f"{['First','Second','Third'][idx] if idx < 3 else f'#{idx+1}th'} comedy venue meeting all specified requirements",
        parent=root,
        critical=False
    )

    # Existence checks to gate downstream verifications
    name_city_node = evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()) and bool(venue.city and venue.city.strip()),
        id=f"V{idx + 1}_Name_and_City",
        desc="Venue name and city location are provided",
        parent=venue_node,
        critical=True
    )

    ref_url_ok = is_valid_url(venue.reference_url)
    ref_url_node = evaluator.add_custom_node(
        result=ref_url_ok,
        id=f"V{idx + 1}_Reference_URL",
        desc="Valid reference URL to venue's official website or information page is provided",
        parent=venue_node,
        critical=True
    )

    # Dedicated comedy venue check
    dedicated_leaf = evaluator.add_leaf(
        id=f"V{idx + 1}_Dedicated_Comedy_Venue",
        desc="Venue is a dedicated comedy club or theater that regularly hosts live stand-up comedy shows (not one-time events or festivals)",
        parent=venue_node,
        critical=True
    )
    dedicated_claim = f"{venue.name or 'The venue'} is a dedicated comedy club or theater that regularly hosts live stand-up comedy shows (not just a one-time event or festival)."
    await evaluator.verify(
        claim=dedicated_claim,
        node=dedicated_leaf,
        sources=sources_for_venue(venue),
        additional_instruction="Confirm the page indicates the venue regularly hosts stand-up comedy (e.g., a recurring schedule, weekly shows, a calendar of comedy events). Ignore individual one-off event listings without indication of regular programming.",
        extra_prerequisites=[name_city_node, ref_url_node]
    )

    # Physical address (complete) verification
    address_leaf = evaluator.add_leaf(
        id=f"V{idx + 1}_Physical_Address",
        desc="Complete physical address including street, city, state, and ZIP code is provided",
        parent=venue_node,
        critical=True
    )
    claimed_address = assemble_full_address(venue) or (venue.full_address or "")
    address_claim = f"The venue's complete physical address is '{claimed_address}'. It should include street address, city, state, and ZIP code."
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=sources_for_venue(venue),
        additional_instruction="Verify the webpage explicitly lists the venue's street address and city/state/ZIP. Allow minor formatting differences (e.g., abbreviations like St. vs Street) but ensure all components are present.",
        extra_prerequisites=[name_city_node, ref_url_node]
    )

    # Capacity verification
    capacity_leaf = evaluator.add_leaf(
        id=f"V{idx + 1}_Venue_Capacity",
        desc="Stated venue capacity (number of seats or standing capacity) is provided",
        parent=venue_node,
        critical=True
    )
    capacity_text = venue.capacity or ""
    capacity_claim = f"The venue's capacity is stated as '{capacity_text}' (total seats or standing capacity)."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=sources_for_venue(venue),
        additional_instruction="Check for any mention of seating capacity, total capacity, or standing room capacity. Accept approximate numbers if they clearly correspond to capacity.",
        extra_prerequisites=[name_city_node, ref_url_node]
    )

    # Wheelchair accessibility verification
    wheelchair_leaf = evaluator.add_leaf(
        id=f"V{idx + 1}_Wheelchair_Accessible",
        desc="Venue explicitly states wheelchair accessibility via elevator or ramp",
        parent=venue_node,
        critical=True
    )
    wheelchair_text = venue.wheelchair_accessible or ""
    wheelchair_claim = f"The venue is wheelchair accessible (e.g., via ramp, elevator, or ground-level access). Details: '{wheelchair_text}'."
    await evaluator.verify(
        claim=wheelchair_claim,
        node=wheelchair_leaf,
        sources=sources_for_venue(venue),
        additional_instruction="Look for terms like 'wheelchair accessible', 'ADA accessible', 'elevator', 'ramp', or 'accessible entrance'.",
        extra_prerequisites=[name_city_node, ref_url_node]
    )

    # Accessible seating verification
    accessible_seating_leaf = evaluator.add_leaf(
        id=f"V{idx + 1}_Accessible_Seating",
        desc="Venue offers accessible seating options for wheelchair users",
        parent=venue_node,
        critical=True
    )
    accessible_seating_text = venue.accessible_seating or ""
    accessible_seating_claim = f"Accessible/ADA seating for wheelchair users is available at the venue. Details: '{accessible_seating_text}'."
    await evaluator.verify(
        claim=accessible_seating_claim,
        node=accessible_seating_leaf,
        sources=sources_for_venue(venue),
        additional_instruction="Verify explicit mention of ADA/accessible seating (e.g., wheelchair seating, companion seating).",
        extra_prerequisites=[name_city_node, ref_url_node]
    )

    # Age restriction policy verification
    age_leaf = evaluator.add_leaf(
        id=f"V{idx + 1}_Age_Restriction",
        desc="Venue's age restriction policy is clearly stated (e.g., 21+, All Ages)",
        parent=venue_node,
        critical=True
    )
    age_text = venue.age_restriction or ""
    age_claim = f"The venue's age restriction policy is '{age_text}'."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=sources_for_venue(venue),
        additional_instruction="Accept formats like '21+', '18+', 'All Ages', or conditions such as '21+ to drink'. Verify that the policy matches the page.",
        extra_prerequisites=[name_city_node, ref_url_node]
    )

    # Drink minimum policy verification
    drink_leaf = evaluator.add_leaf(
        id=f"V{idx + 1}_Drink_Minimum",
        desc="Venue's drink minimum policy is stated (including if no minimum exists)",
        parent=venue_node,
        critical=True
    )
    drink_text = venue.drink_minimum or ""
    drink_claim = f"The venue's drink minimum policy is '{drink_text}'."
    await evaluator.verify(
        claim=drink_claim,
        node=drink_leaf,
        sources=sources_for_venue(venue),
        additional_instruction="Look for mentions like 'two drink minimum', '1 drink minimum', or explicit statements of 'no drink minimum'.",
        extra_prerequisites=[name_city_node, ref_url_node]
    )

    # Online ticketing verification
    ticket_leaf = evaluator.add_leaf(
        id=f"V{idx + 1}_Online_Ticketing",
        desc="Venue offers online ticket purchasing capability",
        parent=venue_node,
        critical=True
    )
    ticket_text = venue.online_ticketing or ""
    ticket_claim = f"Tickets can be purchased online via the venue's official site or an official ticketing platform. Details: '{ticket_text}'."
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_leaf,
        sources=sources_for_venue(venue),
        additional_instruction="Confirm the page has an online 'Buy Tickets' capability or links to an official platform (e.g., Ticketmaster, Eventbrite, Etix).",
        extra_prerequisites=[name_city_node, ref_url_node]
    )

    # Refund/cancellation policy verification
    refund_leaf = evaluator.add_leaf(
        id=f"V{idx + 1}_Refund_Policy",
        desc="Venue's refund or cancellation policy is stated",
        parent=venue_node,
        critical=True
    )
    refund_text = venue.refund_policy or ""
    refund_claim = f"The venue's refund/cancellation policy is: '{refund_text}'."
    await evaluator.verify(
        claim=refund_claim,
        node=refund_leaf,
        sources=sources_for_venue(venue),
        additional_instruction="Check for phrases like 'All sales final', 'No refunds', 'Refunds available', or specific cancellation windows.",
        extra_prerequisites=[name_city_node, ref_url_node]
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
    # Initialize evaluator (root is parallel: venues are independent)
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

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Keep only the first 3 venues (pad to 3 if fewer)
    venues: List[Venue] = list(extracted.venues[:3])
    while len(venues) < 3:
        venues.append(Venue())

    # 2) Verify each venue subtree
    for i in range(3):
        await verify_single_venue(evaluator, root, venues[i], i)

    # 3) After venue verifications, add the cross-venue city uniqueness requirement (critical).
    # Placing it after ensures it does not skip other checks during verification.
    cities_norm = [normalize_city(v.city) for v in venues]
    all_three_present = all(c is not None and c != "" for c in cities_norm)
    all_different = len(set(cities_norm)) == 3 if all_three_present else False
    evaluator.add_custom_node(
        result=all_three_present and all_different,
        id="Different_Cities_Requirement",
        desc="All three venues must be located in different U.S. cities",
        parent=root,
        critical=True
    )

    # Also record a small custom info block for debugging
    evaluator.add_custom_info(
        info={
            "extracted_cities": [v.city for v in venues],
            "city_uniqueness_passed": all_three_present and all_different
        },
        info_type="diagnostics",
        info_name="city_uniqueness_check"
    )

    # 4) Return evaluation summary
    return evaluator.get_summary()