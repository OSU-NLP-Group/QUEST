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
TASK_ID = "touring_venues_4_states"
TASK_DESCRIPTION = """
I am planning a national tour for a Broadway musical production and need to identify suitable theater venues in four different U.S. states. Find one theater venue in each of the following states: California, Texas, Florida, and Pennsylvania. For each venue, provide: (1) Venue Identification: The official name of the theater venue, the city where the venue is located, and the official website URL of the venue. (2) Technical Specifications: Seating capacity (must be between 2,000 and 3,500 seats), stage configuration (must have a proscenium stage suitable for touring Broadway productions), orchestra pit capability (must be able to accommodate live musicians), loading dock access (must have backstage loading capability for touring production trucks), and a URL reference that verifies these technical specifications. (3) Accessibility Compliance: Confirmation that the venue provides wheelchair accessible seating that meets or exceeds ADA requirements based on the venue's total capacity, confirmation that the venue provides companion seating adjacent to wheelchair spaces, and a URL reference that verifies accessibility features. (4) Contact Information: The complete street address of the venue and contact information for booking or rentals (phone number, email, or booking page URL). All information must be verified through official venue websites or reputable performing arts venue directories. Provide the specific URL references for technical specifications and accessibility information for each venue.
"""

STATE_MAP = {
    "CA": "California",
    "TX": "Texas",
    "FL": "Florida",
    "PA": "Pennsylvania",
}


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    # Venue identification
    name: Optional[str] = None
    city: Optional[str] = None
    website_url: Optional[str] = None

    # Technical specifications (content from the answer; may be free text)
    stated_capacity: Optional[str] = None
    stage_configuration: Optional[str] = None
    orchestra_pit_capability: Optional[str] = None
    loading_dock_access: Optional[str] = None
    tech_specs_urls: List[str] = Field(default_factory=list)

    # Accessibility
    wheelchair_accessible_seating: Optional[str] = None
    companion_seating: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=list)

    # Contact info
    address: Optional[str] = None
    booking_contact: Optional[str] = None  # phone/email or a booking page URL mentioned in the answer
    booking_contact_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    california: Optional[VenueItem] = None
    texas: Optional[VenueItem] = None
    florida: Optional[VenueItem] = None
    pennsylvania: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract one theater venue per state from the answer text for each of the following U.S. states: California, Texas, Florida, and Pennsylvania.
    If the answer lists multiple venues for a state, select the first one mentioned for that state.

    For each state, extract the following fields exactly as presented in the answer:
    - name: the official venue name (string)
    - city: the city where the venue is located (string)
    - website_url: the official venue website URL (string URL, if multiple URLs are given, choose the official venue site)
    - stated_capacity: any capacity value or range mentioned in the answer (string; do not parse to number)
    - stage_configuration: stage configuration terms (e.g., "proscenium") as mentioned in the answer (string)
    - orchestra_pit_capability: description or yes/no from the answer (string)
    - loading_dock_access: description from the answer (string)
    - tech_specs_urls: list of URL(s) that the answer cites specifically to support technical specifications (list of strings)
    - wheelchair_accessible_seating: description of wheelchair seating/ADA compliance from the answer (string)
    - companion_seating: description of companion seating from the answer (string)
    - accessibility_urls: list of URL(s) that the answer cites to support accessibility features (list of strings)
    - address: the complete street address as written in the answer (string)
    - booking_contact: a phone/email or a booking/rentals page URL as written in the answer (string)
    - booking_contact_urls: list of URL(s) the answer cites for booking/rentals (list of strings)

    Return a JSON object with top-level fields:
    - california: VenueItem (or null if no California venue is mentioned)
    - texas: VenueItem (or null)
    - florida: VenueItem (or null)
    - pennsylvania: VenueItem (or null)

    URL extraction requirements:
    - Extract only URLs explicitly present in the answer (including those inside markdown links).
    - Include full URLs with protocol; if missing, prepend http://
    - Do not fabricate or infer URLs.

    If any field is not present in the answer, set it to null (or empty list for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup_list(items: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if not _nonempty(x):
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)  # type: ignore
    return out


def _state_id_prefix(abbr: str) -> str:
    return abbr.upper()


# --------------------------------------------------------------------------- #
# Verification Builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identity_checks(
    evaluator: Evaluator,
    parent,
    abbr: str,
    state_name: str,
    venue: VenueItem,
):
    """
    Build and verify identity checks:
      - Official website URL (verified as official)
      - Venue name (matches official website)
      - City (matches official website)
    All are critical under Identity.
    """
    pid = _state_id_prefix(abbr)
    node = evaluator.add_parallel(
        id=f"{pid}_Venue_Identity",
        desc=f"Provide the official name and city location of a theater venue in {state_name}",
        parent=parent,
        critical=True,
    )

    # 1) Official website URL
    website_node = evaluator.add_leaf(
        id=f"{pid}_Venue_Website_URL",
        desc=f"The official website URL of the {state_name} venue",
        parent=node,
        critical=True,
    )

    # Form the claim, using best available info even if some fields are missing
    parts = []
    if _nonempty(venue.name):
        parts.append(f"named '{venue.name}'")
    loc_part = ""
    if _nonempty(venue.city):
        loc_part = f"in {venue.city}, {state_name}"
    else:
        loc_part = f"in {state_name}"
    if parts:
        name_part = " ".join(parts)
        claim_official = f"This URL is the official website for the theater venue {name_part} {loc_part}."
    else:
        claim_official = f"This URL is the official website for a theater venue {loc_part}."

    await evaluator.verify(
        claim=claim_official,
        node=website_node,
        sources=venue.website_url,
        additional_instruction=(
            "Determine whether the URL appears to be the venue's official website (not a reseller or 3rd-party directory). "
            "Check branding, ownership, and self-identification on the page."
        ),
    )

    # 2) Venue name matches website
    name_node = evaluator.add_leaf(
        id=f"{pid}_Venue_Name",
        desc=f"The official name of the {state_name} theater venue",
        parent=node,
        critical=True,
    )
    name_claim = (
        f"The venue's official website shows the venue's official name as '{venue.name}', "
        f"or an equivalent minor variant (case-insensitive, punctuation variants, 'Theatre' vs 'Theater', etc.)."
        if _nonempty(venue.name)
        else "The venue's official website clearly shows the venue's official name (but no name was provided in the answer)."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=venue.website_url,
        additional_instruction=(
            "Allow minor variants like casing, punctuation, abbreviations, or 'Theatre' vs 'Theater'. "
            "Match to the site's self-identified venue name."
        ),
    )

    # 3) City matches website
    city_node = evaluator.add_leaf(
        id=f"{pid}_Venue_City",
        desc=f"The city where the {state_name} venue is located",
        parent=node,
        critical=True,
    )
    city_claim = (
        f"The venue's official website indicates the venue is located in {venue.city}, {state_name}."
        if _nonempty(venue.city)
        else f"The venue's official website specifies the venue's city within {state_name} (but no city was provided in the answer)."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_node,
        sources=venue.website_url,
        additional_instruction="Look for the city in the contact, footer, 'Visit', or 'About' sections.",
    )


async def build_technical_specs_checks(
    evaluator: Evaluator,
    parent,
    abbr: str,
    state_name: str,
    venue: VenueItem,
):
    """
    Technical specifications (critical group):
      - Tech specs URL provided (existence check)
      - Seating capacity within [2,000, 3,500]
      - Proscenium stage
      - Orchestra pit capability
      - Loading dock/backstage loading capability
    All checked against tech_specs_urls.
    """
    pid = _state_id_prefix(abbr)
    node = evaluator.add_parallel(
        id=f"{pid}_Technical_Specifications",
        desc=f"Verify technical specifications of the {state_name} venue",
        parent=parent,
        critical=True,
    )

    tech_urls = venue.tech_specs_urls or []
    # Existence of verifying URL(s)
    evaluator.add_custom_node(
        result=bool(tech_urls),
        id=f"{pid}_Tech_Specs_URL",
        desc="Provide a URL reference that verifies the technical specifications",
        parent=node,
        critical=True,
    )

    # Seating capacity in range
    capacity_node = evaluator.add_leaf(
        id=f"{pid}_Seating_Capacity",
        desc="The venue has a seating capacity between 2,000 and 3,500 seats",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="According to the provided technical specifications source(s), the venue's seating capacity is between 2,000 and 3,500 seats (inclusive).",
        node=capacity_node,
        sources=tech_urls,
        additional_instruction=(
            "Find the capacity on the page(s). If it lists a single capacity or a range entirely within 2000–3500, consider supported. "
            "If multiple spaces are listed, the main/proscenium house must fall within that range."
        ),
    )

    # Proscenium stage configuration
    stage_node = evaluator.add_leaf(
        id=f"{pid}_Stage_Configuration",
        desc="The venue has a proscenium stage configuration suitable for touring Broadway productions",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue has a proscenium stage configuration suitable for touring Broadway productions.",
        node=stage_node,
        sources=tech_urls,
        additional_instruction="Look for 'proscenium', 'proscenium arch', or equivalent terminology indicating a proscenium stage.",
    )

    # Orchestra pit capability
    pit_node = evaluator.add_leaf(
        id=f"{pid}_Orchestra_Pit",
        desc="The venue has an orchestra pit or capability to accommodate live musicians",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue has an orchestra pit or otherwise provides capability to accommodate live musicians.",
        node=pit_node,
        sources=tech_urls,
        additional_instruction="Accept mentions of 'orchestra pit', 'pit lift', 'pit elevator', or similar features.",
    )

    # Loading dock/backstage capability
    loading_node = evaluator.add_leaf(
        id=f"{pid}_Loading_Access",
        desc="The venue has loading dock access or backstage loading capability for touring production trucks",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue provides loading dock access or backstage loading capability suitable for touring production trucks.",
        node=loading_node,
        sources=tech_urls,
        additional_instruction="Look for 'loading dock', 'truck access', 'stage door loading', dock dimensions, or similar backstage loading info.",
    )


async def build_accessibility_checks(
    evaluator: Evaluator,
    parent,
    abbr: str,
    state_name: str,
    venue: VenueItem,
):
    """
    Accessibility compliance (critical group):
      - Accessibility URL provided (existence check)
      - Wheelchair accessible seating meets/exceeds ADA
      - Companion seating adjacent to wheelchair spaces
    Verified against accessibility_urls.
    """
    pid = _state_id_prefix(abbr)
    node = evaluator.add_parallel(
        id=f"{pid}_Accessibility_Compliance",
        desc=f"Verify ADA accessibility compliance of the {state_name} venue",
        parent=parent,
        critical=True,
    )

    acc_urls = venue.accessibility_urls or []
    evaluator.add_custom_node(
        result=bool(acc_urls),
        id=f"{pid}_Accessibility_URL",
        desc="Provide a URL reference that verifies accessibility features",
        parent=node,
        critical=True,
    )

    # Wheelchair accessible seating meets/exceeds ADA
    wheelchair_node = evaluator.add_leaf(
        id=f"{pid}_Wheelchair_Seating",
        desc="The venue provides wheelchair accessible seating that meets or exceeds ADA requirements based on the venue's total capacity",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue provides wheelchair accessible seating that meets or exceeds ADA requirements appropriate for the venue's total capacity.",
        node=wheelchair_node,
        sources=acc_urls,
        additional_instruction=(
            "Accept statements like 'ADA compliant accessible seating available' or references to ADA-compliant wheelchair seating."
        ),
    )

    # Companion seating adjacent to wheelchair spaces
    companion_node = evaluator.add_leaf(
        id=f"{pid}_Companion_Seating",
        desc="The venue provides companion seating adjacent to wheelchair spaces",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue provides companion seating adjacent to wheelchair spaces.",
        node=companion_node,
        sources=acc_urls,
        additional_instruction="Look for 'companion seats' or equivalent phrasing explicitly indicating adjacency to wheelchair spaces.",
    )


async def build_contact_info_checks(
    evaluator: Evaluator,
    parent,
    abbr: str,
    state_name: str,
    venue: VenueItem,
):
    """
    Contact information (non-critical group):
      - Complete street address present on official site
      - Booking/rentals contact available (phone, email, or rentals/booking page)
    """
    pid = _state_id_prefix(abbr)
    node = evaluator.add_parallel(
        id=f"{pid}_Contact_Information",
        desc=f"Provide contact information for the {state_name} venue",
        parent=parent,
        critical=False,
    )

    contact_sources = _dedup_list([venue.website_url, *venue.booking_contact_urls])

    # Complete street address
    addr_node = evaluator.add_leaf(
        id=f"{pid}_Physical_Address",
        desc=f"The complete street address of the {state_name} venue",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim="The venue's official website provides a complete street address (including street number, street name, city, state, and ZIP).",
        node=addr_node,
        sources=contact_sources if contact_sources else venue.website_url,
        additional_instruction="Check Contact, Visit, Footer, or About pages for a full postal address.",
    )

    # Booking/rentals contact
    booking_node = evaluator.add_leaf(
        id=f"{pid}_Booking_Contact",
        desc="Contact information for booking or rentals (phone, email, or booking page URL)",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim="The venue's official website provides booking or rentals contact information (phone, email, or a rentals/booking web page/form).",
        node=booking_node,
        sources=contact_sources if contact_sources else venue.website_url,
        additional_instruction="Look for 'Rentals', 'Book the Venue', 'Venue Booking', 'Event Services', or similar pages or contact info.",
    )


async def verify_state_venue(
    evaluator: Evaluator,
    parent,
    abbr: str,
    venue: Optional[VenueItem],
):
    """
    Build the full verification sequence for a single state venue.
    Sequential flow:
      1) Identity (critical)
      2) Technical Specifications (critical)
      3) Accessibility Compliance (critical)
      4) Contact Information (non-critical)
    """
    state_name = STATE_MAP[abbr]
    pid = _state_id_prefix(abbr)

    state_node = evaluator.add_sequential(
        id=f"{pid}_Venue",
        desc=f"Identify one suitable theater venue in {state_name}",
        parent=parent,
        critical=False,  # allow partial credit across states
    )

    # Use an empty VenueItem placeholder if None to ensure nodes are created
    v = venue or VenueItem()

    # 1) Identity checks
    await build_identity_checks(evaluator, state_node, abbr, state_name, v)

    # 2) Technical specifications
    await build_technical_specs_checks(evaluator, state_node, abbr, state_name, v)

    # 3) Accessibility compliance
    await build_accessibility_checks(evaluator, state_node, abbr, state_name, v)

    # 4) Contact information (non-critical)
    await build_contact_info_checks(evaluator, state_node, abbr, state_name, v)


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an answer for identifying 4 suitable touring theater venues across CA, TX, FL, and PA.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Aggregate across states independently
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

    # Extract structured venue info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Build the tree and run verifications for each state
    await verify_state_venue(evaluator, root, "CA", extracted.california)
    await verify_state_venue(evaluator, root, "TX", extracted.texas)
    await verify_state_venue(evaluator, root, "FL", extracted.florida)
    await verify_state_venue(evaluator, root, "PA", extracted.pennsylvania)

    return evaluator.get_summary()