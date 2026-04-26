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
TASK_ID = "la_large_venues_accessible_parking"
TASK_DESCRIPTION = (
    "Find four large concert venues in Los Angeles (each with a seating capacity greater than 5,000) that offer "
    "comprehensive accessibility features and on-site parking options. For each venue, provide the following information:\n\n"
    "1. Venue Name and Location: The official name of the venue and its complete address.\n\n"
    "2. Seating Capacity: The stated seating capacity for concerts or events, which must exceed 5,000.\n\n"
    "3. Accessibility Features: Detailed information about accessibility accommodations, including:\n"
    "   - Availability of wheelchair accessible seating (specify the number of wheelchair/semi-ambulatory spaces or describe the accessible seating locations)\n"
    "   - Companion seating policy (how many additional seats can be purchased adjacent to accessible seats)\n"
    "   - Availability of assistive listening devices or similar accessibility technology\n\n"
    "4. Parking Information: Complete parking details, including:\n"
    "   - Confirmation that on-site parking is available\n"
    "   - Parking pricing information (provide at least one parking option with its price)\n"
    "   - Explicit confirmation that accessible/ADA parking is available\n\n"
    "For each piece of information provided, include reference URLs from the venue's official website or reliable sources to verify the details."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueAccessibility(BaseModel):
    wheelchair_seating_details: Optional[str] = None
    companion_seating_policy: Optional[str] = None
    assistive_listening_details: Optional[str] = None


class VenueParking(BaseModel):
    onsite_parking_statement: Optional[str] = None
    parking_pricing_text: Optional[str] = None
    accessible_parking_statement: Optional[str] = None


class VenueItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    capacity: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    parking_urls: List[str] = Field(default_factory=list)
    accessibility: VenueAccessibility = Field(default_factory=VenueAccessibility)
    parking: VenueParking = Field(default_factory=VenueParking)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract up to four (4) concert venues presented in the answer that satisfy the task. For each venue, extract exactly the following fields:

- name: The official venue name as written in the answer.
- address: The complete postal address as provided in the answer (include city and state if present).
- capacity: The stated seating capacity for concerts/events as written in the answer (keep text format, do NOT convert to a number; e.g., "6,000", "≈7,100", "approximately 17,500").
- identification_urls: An array of reference URLs supporting the venue's identification/location (prefer official venue pages; may include reliable pages). Must be URLs explicitly present in the answer.
- capacity_urls: An array of URLs that support the capacity figure (prefer official venue pages, seating charts, or credible sources). Only include URLs explicitly present in the answer.
- accessibility_urls: An array of URLs supporting the accessibility features. Only include URLs explicitly present in the answer.
- parking_urls: An array of URLs supporting on-site parking, pricing, and ADA parking details. Only include URLs explicitly present in the answer.

- accessibility: An object with:
  - wheelchair_seating_details: The wording in the answer that indicates wheelchair accessible seating (include numbers of spaces if stated OR a description of accessible seating locations/sections).
  - companion_seating_policy: The wording in the answer that describes companion seating (e.g., how many adjacent seats are allowed).
  - assistive_listening_details: The wording in the answer that indicates availability of assistive listening devices or similar technology.

- parking: An object with:
  - onsite_parking_statement: The wording in the answer confirming on-site parking is available.
  - parking_pricing_text: The wording in the answer that shows at least one specific on-site parking option with a price (e.g., "$25", "$30 event parking", "garage parking $20–$30").
  - accessible_parking_statement: The wording in the answer explicitly indicating accessible/ADA parking is available.

Rules:
1) Extract ONLY what is explicitly present in the answer text. Do not infer or invent any content or URLs.
2) For any field not mentioned in the answer, return null (for strings) or an empty array (for URL arrays).
3) URLs must be valid and complete; accept plain URLs or markdown links, but extract the actual URL.
4) Return a JSON with a single top-level field "venues": an array of venue objects as described. If more than 4 venues are present, include all but the evaluator will later take the first 4.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0 and any(u.strip() for u in urls))


# --------------------------------------------------------------------------- #
# Venue verification                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    venue: VenueItem,
) -> None:
    """
    Build the verification subtree for a single venue (idx is 1-based).
    """
    v_id = f"venue_{idx}"

    # Parent node for this venue
    venue_node = evaluator.add_parallel(
        id=v_id,
        desc=f"{['First','Second','Third','Fourth'][idx-1]} venue meeting all requirements",
        parent=parent_node,
        critical=False  # venue nodes are non-critical so partial credit across venues is allowed
    )

    # ---------------- Identification & Location ----------------
    ident_node = evaluator.add_parallel(
        id=f"{v_id}_identification",
        desc="Venue identification and location information",
        parent=venue_node,
        critical=True
    )

    # Name provided (critical)
    evaluator.add_custom_node(
        result=_nonempty(venue.name),
        id=f"{v_id}_name",
        desc="Venue name is provided",
        parent=ident_node,
        critical=True
    )

    # Address provided (critical)
    evaluator.add_custom_node(
        result=_nonempty(venue.address),
        id=f"{v_id}_address",
        desc="Full address is provided",
        parent=ident_node,
        critical=True
    )

    # Identification URL exists (critical)
    evaluator.add_custom_node(
        result=_has_urls(venue.identification_urls),
        id=f"{v_id}_identification_url",
        desc="Reference URL for venue identification verification",
        parent=ident_node,
        critical=True
    )

    # Location verification (critical)
    loc_leaf = evaluator.add_leaf(
        id=f"{v_id}_location_verification",
        desc="Venue is located in Los Angeles area",
        parent=ident_node,
        critical=True
    )
    name_for_claim = venue.name or "the venue"
    addr_for_claim = venue.address or ""
    await evaluator.verify(
        claim=(
            f"{name_for_claim} is located in Los Angeles, California (City of Los Angeles or a recognized LA neighborhood). "
            f"Address (if provided): '{addr_for_claim}'."
        ),
        node=loc_leaf,
        sources=venue.identification_urls,
        additional_instruction=(
            "Support the claim only if the page indicates the venue is within the City of Los Angeles (including neighborhoods "
            "like Downtown LA, Hollywood, Exposition Park, Westwood, Echo Park, etc.). "
            "If the page clearly indicates a different incorporated city (e.g., Inglewood, Pasadena, Anaheim, Glendale), "
            "mark as not supported."
        ),
    )

    # ---------------- Capacity ----------------
    cap_node = evaluator.add_parallel(
        id=f"{v_id}_capacity",
        desc="Venue capacity information",
        parent=venue_node,
        critical=True
    )

    # Capacity URL exists (critical)
    evaluator.add_custom_node(
        result=_has_urls(venue.capacity_urls),
        id=f"{v_id}_capacity_url",
        desc="Reference URL for capacity verification",
        parent=cap_node,
        critical=True
    )

    # Capacity > 5,000 (critical)
    cap_leaf = evaluator.add_leaf(
        id=f"{v_id}_capacity_value",
        desc="Stated capacity is greater than 5,000",
        parent=cap_node,
        critical=True
    )
    cap_text = venue.capacity or ""
    await evaluator.verify(
        claim=(
            f"The stated seating capacity for concerts/events at {name_for_claim} is '{cap_text}', and this capacity exceeds 5,000."
        ),
        node=cap_leaf,
        sources=venue.capacity_urls,
        additional_instruction=(
            "Check the page for the venue's seating capacity (prefer concert/event seating capacity). "
            "Allow reasonable formatting (commas, approximate). "
            "Mark as supported only if the capacity shown on the page is clearly greater than 5,000."
        ),
    )

    # ---------------- Accessibility ----------------
    acc_node = evaluator.add_parallel(
        id=f"{v_id}_accessibility",
        desc="Comprehensive accessibility features",
        parent=venue_node,
        critical=True
    )

    # Accessibility URL exists (critical)
    evaluator.add_custom_node(
        result=_has_urls(venue.accessibility_urls),
        id=f"{v_id}_accessibility_url",
        desc="Reference URL for accessibility information verification",
        parent=acc_node,
        critical=True
    )

    # Accessibility features container (critical)
    acc_features_node = evaluator.add_parallel(
        id=f"{v_id}_accessibility_features",
        desc="Detailed accessibility accommodations",
        parent=acc_node,
        critical=True
    )

    # Wheelchair accessible seating (critical)
    wheel_leaf = evaluator.add_leaf(
        id=f"{v_id}_wheelchair_seating",
        desc="Wheelchair accessible seating is available with specific number of spaces or locations described",
        parent=acc_features_node,
        critical=True
    )
    wheel_text = venue.accessibility.wheelchair_seating_details or ""
    await evaluator.verify(
        claim=(
            f"The venue offers wheelchair accessible seating. Details from the answer: '{wheel_text}'. "
            "The page should confirm availability and either specify the number of wheelchair/semi-ambulatory spaces "
            "or describe the locations/sections of accessible seating."
        ),
        node=wheel_leaf,
        sources=venue.accessibility_urls,
        additional_instruction=(
            "Support only if the source indicates wheelchair-accessible seating AND includes either a number of spaces "
            "or a meaningful description of where accessible seating is located (e.g., sections/levels)."
        ),
    )

    # Companion seating policy (critical)
    comp_leaf = evaluator.add_leaf(
        id=f"{v_id}_companion_seating",
        desc="Companion seating policy is described (allows purchase of additional seats adjacent to accessible seats)",
        parent=acc_features_node,
        critical=True
    )
    comp_text = venue.accessibility.companion_seating_policy or ""
    await evaluator.verify(
        claim=(
            f"The venue's companion seating policy permits purchase of adjacent companion seat(s) next to accessible seating. "
            f"Details from the answer: '{comp_text}'."
        ),
        node=comp_leaf,
        sources=venue.accessibility_urls,
        additional_instruction=(
            "Look for explicit mention of companion/adjacent seats for accessible ticket holders (e.g., one or more companion seats). "
            "If unclear or not stated, mark as not supported."
        ),
    )

    # Assistive listening devices (critical)
    ald_leaf = evaluator.add_leaf(
        id=f"{v_id}_assistive_devices",
        desc="Assistive listening devices or similar accessibility technology is available",
        parent=acc_features_node,
        critical=True
    )
    ald_text = venue.accessibility.assistive_listening_details or ""
    await evaluator.verify(
        claim=(
            f"Assistive listening devices or similar hearing assistance technology are available at the venue. "
            f"Details from the answer: '{ald_text}'."
        ),
        node=ald_leaf,
        sources=venue.accessibility_urls,
        additional_instruction=(
            "Accept terms like 'assistive listening devices', 'ALDs', 'hearing assistance', 'hearing loop', 'FM/IR system'. "
            "If no such technology is mentioned, mark as not supported."
        ),
    )

    # ---------------- Parking ----------------
    park_node = evaluator.add_parallel(
        id=f"{v_id}_parking",
        desc="On-site parking information",
        parent=venue_node,
        critical=True
    )

    # Parking URL exists (critical)
    evaluator.add_custom_node(
        result=_has_urls(venue.parking_urls),
        id=f"{v_id}_parking_url",
        desc="Reference URL for parking information verification",
        parent=park_node,
        critical=True
    )

    # Parking details container (critical)
    park_details_node = evaluator.add_parallel(
        id=f"{v_id}_parking_details",
        desc="Complete parking information",
        parent=park_node,
        critical=True
    )

    # On-site parking available (critical)
    onsite_leaf = evaluator.add_leaf(
        id=f"{v_id}_parking_available",
        desc="On-site parking is confirmed to be available",
        parent=park_details_node,
        critical=True
    )
    onsite_text = venue.parking.onsite_parking_statement or ""
    await evaluator.verify(
        claim=(
            f"On-site parking is available at {name_for_claim}. Details from the answer (if any): '{onsite_text}'."
        ),
        node=onsite_leaf,
        sources=venue.parking_urls,
        additional_instruction=(
            "Confirm that the venue itself provides on-site parking (lots/garages under venue control or on premises). "
            "Do not count unrelated third-party or only street parking."
        ),
    )

    # Parking pricing present (critical)
    price_leaf = evaluator.add_leaf(
        id=f"{v_id}_parking_pricing",
        desc="Parking pricing information is provided (at least one parking option with price)",
        parent=park_details_node,
        critical=True
    )
    price_text = venue.parking.parking_pricing_text or ""
    await evaluator.verify(
        claim=(
            f"The venue provides at least one on-site parking option with a price. "
            f"A price stated in the answer is: '{price_text}'."
        ),
        node=price_leaf,
        sources=venue.parking_urls,
        additional_instruction=(
            "Verify that the provided page includes a concrete on-site parking price (e.g., '$25', '$30 event parking'). "
            "If the answer's price does not match or no price is present on the page, mark as not supported."
        ),
    )

    # Accessible/ADA parking available (critical)
    ada_leaf = evaluator.add_leaf(
        id=f"{v_id}_accessible_parking",
        desc="Accessible/ADA parking is explicitly mentioned as available",
        parent=park_details_node,
        critical=True
    )
    ada_text = venue.parking.accessible_parking_statement or ""
    await evaluator.verify(
        claim=(
            f"Accessible/ADA parking is available at the venue's on-site parking facilities. "
            f"Details from the answer: '{ada_text}'."
        ),
        node=ada_leaf,
        sources=venue.parking_urls,
        additional_instruction=(
            "Look for explicit mention of ADA/accessible/disabled parking. "
            "If not explicitly stated, mark as not supported."
        ),
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
    Evaluate an answer for the Los Angeles large venues accessibility & parking task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # venues evaluated independently
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

    # IMPORTANT: Set root to non-critical to allow partial scoring across venues
    # (If root were critical=True, all children must be critical due to framework rule.)
    root.critical = False

    # 1) Extract venues information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # 2) Normalize to exactly 4 venues (pad with empty if fewer; truncate if more)
    venues: List[VenueItem] = list(extracted.venues or [])
    if len(venues) < 4:
        venues.extend([VenueItem() for _ in range(4 - len(venues))])
    if len(venues) > 4:
        venues = venues[:4]

    # 3) Build verification subtrees per venue
    for i in range(4):
        await verify_single_venue(evaluator, root, i + 1, venues[i])

    # 4) Return the summary
    return evaluator.get_summary()