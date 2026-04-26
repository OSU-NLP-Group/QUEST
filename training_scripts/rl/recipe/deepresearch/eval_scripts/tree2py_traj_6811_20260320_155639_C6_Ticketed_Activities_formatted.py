import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ri_venue_spring_2026"
TASK_DESCRIPTION = """
A touring theater company is planning their spring 2026 performance schedule and needs to identify a major performing arts venue in Rhode Island that meets all of the following requirements:

1. Capacity Requirement: The venue must have a total seating capacity between 3,000 and 3,200 seats
2. Regional Significance: The venue must be classified as the second-largest theater of its kind in New England
3. Seating Structure: The venue must have seating distributed across exactly four distinct levels
4. ADA Accessibility Compliance:
   - Must provide wheelchair accessible seating representing at least 1% of total capacity
   - Must offer accessible seating at all price levels across all seating sections
5. Group Booking Capability: Must accommodate group bookings (standard minimum of 10+ tickets)
6. Current Programming: Must have an active 2025-2026 season with confirmed Broadway or touring productions

Identify the venue that meets all these requirements and provide:
- The venue's complete official name
- The exact street address including city, state, and zip code
- The precise total seating capacity
- Verification that the venue meets the specified accessibility requirements
- Information about group booking options or contact details
- The official website URL showing the 2025-2026 season schedule

Your answer must include reference URLs from the venue's official website or authoritative sources to support each piece of information provided.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueIdentification(BaseModel):
    name: Optional[str] = None
    full_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    official_url: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CapacityExtraction(BaseModel):
    capacity_exact_number: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)
    classification_text: Optional[str] = None  # e.g., "second-largest theater of its kind in New England"
    classification_urls: List[str] = Field(default_factory=list)
    seating_levels_count: Optional[str] = None  # "4" or "four"
    seating_levels_description: Optional[str] = None  # e.g., "Orchestra, Loge, Mezzanine, Balcony"
    seating_levels_urls: List[str] = Field(default_factory=list)


class AccessibilityExtraction(BaseModel):
    wheelchair_accessible: Optional[str] = None  # yes/no or description
    wheelchair_percent_or_count: Optional[str] = None  # e.g., "1%" or "at least 32 seats"
    wheelchair_urls: List[str] = Field(default_factory=list)

    accessible_all_price_levels: Optional[str] = None  # yes/no or text
    price_levels_urls: List[str] = Field(default_factory=list)

    additional_features: List[str] = Field(default_factory=list)  # elevators, restrooms, etc.
    additional_features_urls: List[str] = Field(default_factory=list)


class BookingExtraction(BaseModel):
    min_group_size: Optional[str] = None  # e.g., "10", "12", "10+"
    group_booking_urls: List[str] = Field(default_factory=list)
    booking_contact: Optional[str] = None  # email/phone/form description
    booking_contact_urls: List[str] = Field(default_factory=list)
    advance_booking_timeline: Optional[str] = None  # e.g., "request 6-8 weeks in advance"
    advance_booking_urls: List[str] = Field(default_factory=list)
    rental_availability: Optional[str] = None  # yes/no text about rentals
    rental_urls: List[str] = Field(default_factory=list)


class SeasonExtraction(BaseModel):
    has_2025_2026_season: Optional[str] = None  # yes/no
    season_page_url: Optional[str] = None
    season_urls: List[str] = Field(default_factory=list)
    spring_2026_programming: Optional[str] = None  # yes/no or examples
    spring_2026_urls: List[str] = Field(default_factory=list)
    example_shows: List[str] = Field(default_factory=list)
    shows_urls: List[str] = Field(default_factory=list)
    tickets_on_sale: Optional[str] = None  # yes/no or text
    ticket_sales_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_identification() -> str:
    return """
Extract the venue identification details explicitly mentioned in the answer. Return:
- name: complete official venue name
- full_address: a single-line full street address including city, state, and zip code, exactly as stated (or composed from parts if presented separately)
- city: city name (if given)
- state: state as "Rhode Island" or "RI" (if given)
- zip_code: zip/postal code (if given)
- official_url: the official venue website URL (homepage or "About"/"Visit" page is acceptable)
- reference_urls: all URLs in the answer that directly support the venue identification/address/location

If any item is missing, set it to null (or [] for lists). Do not invent URLs.
"""


def prompt_extract_capacity() -> str:
    return """
Extract the capacity and seating structure details from the answer. Return:
- capacity_exact_number: the stated total seating capacity (as a string, e.g., "3100")
- capacity_urls: URL(s) supporting the capacity figure
- classification_text: any statement that the venue is the "second-largest theater of its kind in New England" (or equivalent wording)
- classification_urls: URL(s) supporting the classification statement
- seating_levels_count: the count of distinct seating levels as stated (as a string, e.g., "4" or "four")
- seating_levels_description: names of the levels if given (e.g., "Orchestra, Loge, Mezzanine, Balcony")
- seating_levels_urls: URL(s) supporting the seating levels information

If any field is not provided in the answer, set it to null (or [] for lists).
"""


def prompt_extract_accessibility() -> str:
    return """
Extract accessibility details from the answer. Return:
- wheelchair_accessible: a text indicator that wheelchair-accessible seating is provided (e.g., "yes", or a descriptive sentence)
- wheelchair_percent_or_count: the quantity or percentage if stated (e.g., "at least 1%", "32 seats")
- wheelchair_urls: URL(s) supporting wheelchair accessibility details
- accessible_all_price_levels: a statement that accessible seating is available at all price levels across all seating sections (yes/no or sentence)
- price_levels_urls: URL(s) supporting the accessibility across all price levels claim
- additional_features: list additional accessibility features mentioned (e.g., elevators, accessible restrooms, assistive listening)
- additional_features_urls: URL(s) supporting these additional features

If any field is missing, return null (or [] for lists).
"""


def prompt_extract_booking() -> str:
    return """
Extract group booking and rental information from the answer. Return:
- min_group_size: the minimum number of tickets for group bookings (e.g., "10", "10+", "12")
- group_booking_urls: URL(s) with group booking or group sales information
- booking_contact: a contact method or process for group bookings (email, phone, or "group request form" etc.)
- booking_contact_urls: URL(s) supporting the contact/process information
- advance_booking_timeline: any stated advance notice/timeline requirement (if any)
- advance_booking_urls: URL(s) supporting the timeline requirements
- rental_availability: whether venue rentals (for touring productions) are offered (text)
- rental_urls: URL(s) supporting rental information

If any field is missing, return null (or [] for lists).
"""


def prompt_extract_season() -> str:
    return """
Extract season schedule information from the answer. Return:
- has_2025_2026_season: statement indicating an active 2025–2026 season with Broadway or touring productions (yes/no or text)
- season_page_url: the official page URL that lists the 2025–2026 season or schedule
- season_urls: additional URL(s) supporting the 2025–2026 season information
- spring_2026_programming: statement indicating programming in March–May 2026 (yes/no or examples)
- spring_2026_urls: URL(s) supporting spring 2026 programming
- example_shows: list of example Broadway or touring shows in the 2025–2026 season
- shows_urls: URL(s) supporting the show examples
- tickets_on_sale: statement indicating that tickets are on sale or available to purchase for 2025–2026 season events
- ticket_sales_urls: URL(s) supporting ticket sale availability

If any field is missing, return null (or [] for lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_groups: Optional[List[str] | str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for group in url_groups:
        if group is None:
            continue
        if isinstance(group, str):
            candidates = [group]
        else:
            candidates = group
        for u in candidates:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _address_string(vi: VenueIdentification) -> str:
    if vi.full_address:
        return vi.full_address
    parts = [vi.city, vi.state, vi.zip_code]
    parts = [p for p in parts if p]
    return ", ".join(parts) if parts else ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_venue_identification_section(
    evaluator: Evaluator,
    parent,
    vi: VenueIdentification
) -> None:
    node = evaluator.add_parallel(
        id="venue_identification",
        desc="Provide complete identification information for the venue",
        parent=parent,
        critical=True
    )

    # Custom existence check for official URL (critical to ensure provenance)
    evaluator.add_custom_node(
        result=bool(vi.official_url),
        id="official_url_provided",
        desc="Official venue website URL is provided",
        parent=node,
        critical=True
    )

    # Leaf: venue_name
    name_leaf = evaluator.add_leaf(
        id="venue_name",
        desc="Provide the complete official name of the venue",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue's complete official name is '{vi.name}'.",
        node=name_leaf,
        sources=_merge_urls(vi.official_url, vi.reference_urls),
        additional_instruction="Allow minor punctuation or capitalization differences. The supporting page should clearly display this official name."
    )

    # Leaf: venue_location (Rhode Island)
    location_leaf = evaluator.add_leaf(
        id="venue_location",
        desc="Specify that the venue is located in Rhode Island",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is located in the state of Rhode Island (RI).",
        node=location_leaf,
        sources=_merge_urls(vi.official_url, vi.reference_urls),
        additional_instruction="Confirm that the venue is in Rhode Island. City and state on the page should imply Rhode Island or RI."
    )

    # Leaf: venue_address
    address_leaf = evaluator.add_leaf(
        id="venue_address",
        desc="Provide the complete street address including city, state, and zip code",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue's street address is '{_address_string(vi)}'.",
        node=address_leaf,
        sources=_merge_urls(vi.official_url, vi.reference_urls),
        additional_instruction="Allow minor formatting differences (e.g., commas or abbreviations like RI). The page should clearly show the full address."
    )

    # Leaf: reference_url (official website)
    ref_leaf = evaluator.add_leaf(
        id="reference_url",
        desc="Provide the official venue website URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is the official website of the venue '{vi.name}'.",
        node=ref_leaf,
        sources=vi.official_url,
        additional_instruction="Prefer the venue's own domain (official site). If a subpage (About/Visit/Contact) is used, that is acceptable."
    )


async def build_capacity_section(
    evaluator: Evaluator,
    parent,
    cap: CapacityExtraction
) -> None:
    node = evaluator.add_parallel(
        id="capacity_requirements",
        desc="Verify the venue's seating capacity meets the specified requirements",
        parent=parent,
        critical=True
    )

    # capacity_range
    cap_range_leaf = evaluator.add_leaf(
        id="capacity_range",
        desc="Confirm the venue has a total seating capacity between 3,000 and 3,200 seats",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue's total seating capacity is between 3,000 and 3,200 seats.",
        node=cap_range_leaf,
        sources=_merge_urls(cap.capacity_urls),
        additional_instruction="Use the source to check the total seat count. If an exact number is stated (e.g., 3,100), verify it lies in [3000, 3200]."
    )

    # capacity_classification: second-largest of its kind in New England
    class_leaf = evaluator.add_leaf(
        id="capacity_classification",
        desc="Verify the venue is classified as the second-largest theater of its kind in New England",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is classified as the second-largest theater of its kind in New England.",
        node=class_leaf,
        sources=_merge_urls(cap.classification_urls, cap.capacity_urls),
        additional_instruction="Accept equivalent wording such as 'second largest' and 'of its kind' (e.g., historic/performing arts/theatre). The page should explicitly state this fact."
    )

    # seating_levels: exactly four levels
    levels_leaf = evaluator.add_leaf(
        id="seating_levels",
        desc="Verify that seating is distributed across exactly four distinct levels",
        parent=node,
        critical=True
    )
    levels_desc = cap.seating_levels_description or "four distinct levels"
    await evaluator.verify(
        claim=f"The venue's seating is distributed across exactly four distinct levels (e.g., {levels_desc}).",
        node=levels_leaf,
        sources=_merge_urls(cap.seating_levels_urls, cap.capacity_urls),
        additional_instruction="Look for explicit statements or maps naming four levels (e.g., Orchestra, Loge, Mezzanine, Balcony). Minor naming variations are acceptable."
    )

    # capacity_reference: URLs support capacity info
    cap_ref_leaf = evaluator.add_leaf(
        id="capacity_reference",
        desc="Provide URL supporting capacity information",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URL(s) explicitly state the venue's total seating capacity.",
        node=cap_ref_leaf,
        sources=_merge_urls(cap.capacity_urls),
        additional_instruction="Verify that at least one of the URLs includes the capacity figure on the page."
    )


async def build_accessibility_section(
    evaluator: Evaluator,
    parent,
    acc: AccessibilityExtraction,
    cap: CapacityExtraction
) -> None:
    node = evaluator.add_parallel(
        id="accessibility_compliance",
        desc="Verify the venue meets ADA accessibility requirements",
        parent=parent,
        critical=False  # must be non-critical to allow non-critical children; critical items are nested
    )

    # Wheelchair seating (sequential critical)
    wc_node = evaluator.add_sequential(
        id="wheelchair_seating",
        desc="Confirm availability of wheelchair accessible seating meeting minimum 1% of total capacity requirement",
        parent=node,
        critical=True
    )
    wc_avail_leaf = evaluator.add_leaf(
        id="wheelchair_availability",
        desc="Verify wheelchair accessible seating is available",
        parent=wc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue provides wheelchair-accessible seating and the quantity meets or exceeds 1% of total capacity.",
        node=wc_avail_leaf,
        sources=_merge_urls(acc.wheelchair_urls, cap.capacity_urls),
        additional_instruction="If the page provides a specific count of wheelchair spaces, compare it to 1% of the total capacity. If the page explicitly states compliance with ADA 2010 Standards that imply ≥1%, consider the requirement satisfied."
    )

    wc_ref_leaf = evaluator.add_leaf(
        id="wheelchair_reference",
        desc="Provide URL confirming wheelchair accessibility",
        parent=wc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URL(s) confirm that the venue offers wheelchair-accessible seating.",
        node=wc_ref_leaf,
        sources=_merge_urls(acc.wheelchair_urls),
        additional_instruction="The page should explicitly mention wheelchair-accessible seating or ADA accessibility."
    )

    # Accessible seating across all price levels (sequential critical)
    price_node = evaluator.add_sequential(
        id="accessible_price_levels",
        desc="Confirm accessible seating is available at all price levels across all seating sections",
        parent=node,
        critical=True
    )
    price_avail_leaf = evaluator.add_leaf(
        id="price_level_availability",
        desc="Verify accessible seating is available at all price levels",
        parent=price_node,
        critical=True
    )
    await evaluator.verify(
        claim="Accessible seating is available at all price levels across all seating sections at the venue.",
        node=price_avail_leaf,
        sources=_merge_urls(acc.price_levels_urls, acc.wheelchair_urls),
        additional_instruction="Look for explicit language that accessible seating is available across all sections/price levels. Synonyms like 'available in all sections' are acceptable."
    )

    price_ref_leaf = evaluator.add_leaf(
        id="price_reference",
        desc="Provide URL supporting price level accessibility",
        parent=price_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URL(s) confirm accessible seating availability at all price levels across all seating sections.",
        node=price_ref_leaf,
        sources=_merge_urls(acc.price_levels_urls),
        additional_instruction="At least one URL should clearly state this policy."
    )

    # Additional accessible features (non-critical)
    features_leaf = evaluator.add_leaf(
        id="accessible_features",
        desc="Identify additional accessibility features such as elevators and accessible restrooms",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue provides additional accessibility features such as elevators and/or accessible restrooms or assistive listening.",
        node=features_leaf,
        sources=_merge_urls(acc.additional_features_urls, acc.wheelchair_urls),
        additional_instruction="Any clearly stated accessibility feature beyond seating counts (elevators, accessible restrooms, assistive listening, etc.) qualifies."
    )


async def build_booking_section(
    evaluator: Evaluator,
    parent,
    bk: BookingExtraction
) -> None:
    node = evaluator.add_parallel(
        id="booking_capabilities",
        desc="Verify the venue's group booking capabilities",
        parent=parent,
        critical=False  # must be non-critical to carry non-critical children
    )

    # Group booking minimum (sequential critical)
    gb_node = evaluator.add_sequential(
        id="group_booking_minimum",
        desc="Confirm the venue accommodates group bookings with standard minimum of 10 or more tickets",
        parent=node,
        critical=True
    )
    min_leaf = evaluator.add_leaf(
        id="minimum_group_size",
        desc="Verify the minimum number of tickets required for group booking meets or is below the 10+ standard",
        parent=gb_node,
        critical=True
    )
    await evaluator.verify(
        claim="Group bookings are available starting at 10 tickets (i.e., groups of 10+ are eligible).",
        node=min_leaf,
        sources=_merge_urls(bk.group_booking_urls),
        additional_instruction="If the page states a minimum of 10 tickets, pass. If the minimum is greater than 10 (e.g., 12+), this should fail."
    )

    gb_ref_leaf = evaluator.add_leaf(
        id="group_booking_reference",
        desc="Provide URL for group booking information",
        parent=gb_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URL(s) contain official group booking or group sales information for the venue.",
        node=gb_ref_leaf,
        sources=_merge_urls(bk.group_booking_urls),
        additional_instruction="The page should be the venue's group sales page or a clearly authoritative page describing group booking."
    )

    # Advance booking timeline (non-critical)
    timeline_leaf = evaluator.add_leaf(
        id="advance_booking_timeline",
        desc="Identify the advance booking requirements",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue specifies an advance booking requirement or recommended timeline for group sales.",
        node=timeline_leaf,
        sources=_merge_urls(bk.advance_booking_urls, bk.group_booking_urls),
        additional_instruction="Any explicit statement about lead time or advance notice qualifies."
    )

    # Booking contact/process (non-critical)
    contact_leaf = evaluator.add_leaf(
        id="booking_contact",
        desc="Provide group booking contact information or process",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue provides a group booking contact method or process (e.g., email, phone, or a request form).",
        node=contact_leaf,
        sources=_merge_urls(bk.booking_contact_urls, bk.group_booking_urls),
        additional_instruction="The URL(s) should show a form or list a contact (email/phone) specific to groups."
    )

    # Rental availability (non-critical)
    rental_leaf = evaluator.add_leaf(
        id="rental_availability",
        desc="Confirm the venue offers rental opportunities for touring productions",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue offers rental opportunities or facility rentals suitable for touring productions.",
        node=rental_leaf,
        sources=_merge_urls(bk.rental_urls),
        additional_instruction="Pages describing theatre rental, stage rentals, or presenting opportunities qualify."
    )


async def build_season_section(
    evaluator: Evaluator,
    parent,
    ssn: SeasonExtraction
) -> None:
    node = evaluator.add_parallel(
        id="season_schedule",
        desc="Verify the venue's 2025-2026 season schedule",
        parent=parent,
        critical=False  # must be non-critical to allow non-critical children
    )

    # Season confirmation (sequential critical)
    conf_node = evaluator.add_sequential(
        id="season_confirmation",
        desc="Confirm the venue has an active 2025-2026 season with Broadway or touring productions",
        parent=node,
        critical=True
    )

    season_active_leaf = evaluator.add_leaf(
        id="season_active",
        desc="Verify 2025-2026 season information is available",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has an active 2025–2026 season that includes Broadway or touring productions.",
        node=season_active_leaf,
        sources=_merge_urls(ssn.season_page_url, ssn.season_urls),
        additional_instruction="Look for explicit 2025–2026 season pages or announcements that include Broadway/touring shows."
    )

    season_ref_leaf = evaluator.add_leaf(
        id="season_reference",
        desc="Provide URL for 2025-2026 season information",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URL(s) show the 2025–2026 season schedule or lineup for the venue.",
        node=season_ref_leaf,
        sources=_merge_urls(ssn.season_page_url, ssn.season_urls),
        additional_instruction="The page should clearly reference the 2025–2026 season."
    )

    # Spring 2026 availability (non-critical)
    spring_leaf = evaluator.add_leaf(
        id="spring_2026_availability",
        desc="Identify if the venue has programming scheduled for spring 2026 (March-May)",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The venue's schedule shows programming in March, April, or May 2026.",
        node=spring_leaf,
        sources=_merge_urls(ssn.spring_2026_urls, ssn.season_page_url),
        additional_instruction="At least one event in Mar–May 2026 should be visible on the schedule."
    )

    # Show examples (non-critical)
    examples_str = ", ".join(ssn.example_shows) if ssn.example_shows else "no examples provided"
    examples_leaf = evaluator.add_leaf(
        id="show_examples",
        desc="List examples of Broadway or touring shows in the 2025-2026 season",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Examples of Broadway or touring shows in the 2025–2026 season at this venue include: {examples_str}.",
        node=examples_leaf,
        sources=_merge_urls(ssn.shows_urls, ssn.season_page_url),
        additional_instruction="Verify at least one example is indeed listed on the season page. Allow minor title variations (e.g., hyphens, subtitles)."
    )

    # Ticket sales status (non-critical)
    tickets_leaf = evaluator.add_leaf(
        id="ticket_sales_status",
        desc="Confirm tickets are available for purchase for the 2025-2026 season",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="Tickets are available to purchase for at least some events in the 2025–2026 season.",
        node=tickets_leaf,
        sources=_merge_urls(ssn.ticket_sales_urls, ssn.season_page_url),
        additional_instruction="Look for 'Buy Tickets' buttons, ticket links, or explicit on-sale notices for 2025–2026 shows."
    )


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Rhode Island venue selection task.
    """
    # Initialize evaluator with sequential root to reflect required workflow
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Create a task-level sequential node (non-critical to allow partial credit computation),
    # while each major section will enforce its own critical checks.
    task_node = evaluator.add_sequential(
        id="task",
        desc="Identify a performing arts venue in Rhode Island that meets all specified requirements",
        parent=root,
        critical=False
    )

    # Extract structured data from the answer
    venue_id = await evaluator.extract(
        prompt=prompt_extract_venue_identification(),
        template_class=VenueIdentification,
        extraction_name="venue_identification"
    )
    capacity_ex = await evaluator.extract(
        prompt=prompt_extract_capacity(),
        template_class=CapacityExtraction,
        extraction_name="capacity"
    )
    accessibility_ex = await evaluator.extract(
        prompt=prompt_extract_accessibility(),
        template_class=AccessibilityExtraction,
        extraction_name="accessibility"
    )
    booking_ex = await evaluator.extract(
        prompt=prompt_extract_booking(),
        template_class=BookingExtraction,
        extraction_name="booking"
    )
    season_ex = await evaluator.extract(
        prompt=prompt_extract_season(),
        template_class=SeasonExtraction,
        extraction_name="season"
    )

    # Build verification sections in order (sequential dependency)
    await build_venue_identification_section(evaluator, task_node, venue_id)
    await build_capacity_section(evaluator, task_node, capacity_ex)
    await build_accessibility_section(evaluator, task_node, accessibility_ex, capacity_ex)
    await build_booking_section(evaluator, task_node, booking_ex)
    await build_season_section(evaluator, task_node, season_ex)

    # Return evaluation summary
    return evaluator.get_summary()