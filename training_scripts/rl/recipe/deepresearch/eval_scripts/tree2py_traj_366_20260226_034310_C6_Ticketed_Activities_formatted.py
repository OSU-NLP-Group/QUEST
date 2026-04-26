import asyncio
import logging
import math
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_venue_2026"
TASK_DESCRIPTION = (
    "A touring Broadway production company needs to identify a suitable theater venue in New York City for hosting a "
    "6-week run of a musical production scheduled for April-May 2026. The selected venue must meet the following "
    "comprehensive requirements: (1) Location: The theater must be a major venue in Manhattan suitable for "
    "Broadway-caliber productions. (2) Seating Capacity: The venue must have a total seating capacity between 1,000 "
    "and 1,800 seats with the exact seat count verified. (3) ADA Accessibility - Wheelchair Seating: The venue must "
    "provide wheelchair-accessible seating that meets or exceeds federal ADA requirements. Calculate the minimum number "
    "of wheelchair-accessible spaces required based on the venue's total capacity using ADA standards (for a 1,000-seat "
    "venue, 10 wheelchair spaces are required; calculate proportionally for other capacities). Verify that the venue "
    "meets this requirement with both the calculated minimum and confirmation of compliance. (4) Ticket Pricing "
    "Structure: The venue must offer at least three distinct ticket price tiers corresponding to different seating "
    "sections (such as Orchestra, Mezzanine, Balcony, or equivalent). Provide the section names and typical price ranges "
    "for each tier. (5) Premium Seating: Identify whether premium or VIP seating options are available. (6) Recent "
    "Production History: The venue must have hosted a Broadway show or major theatrical production between February 2025 "
    "and February 2026. Provide the production title and performance dates or evidence. (7) Accessible Facilities: The "
    "venue must provide accessible patron amenities including accessible seating and facilities. Provide the official "
    "venue name, complete address, exact seating capacity, ADA wheelchair accessibility verification with calculation, "
    "three-tier seating structure with price ranges, premium seating details, recent production information with dates, "
    "confirmation of accessible facilities, and supporting URL references for each category."
)

PRODUCTION_WINDOW_START = "2025-02-01"
PRODUCTION_WINDOW_END = "2026-02-28"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    manhattan_indicator: Optional[str] = None  # e.g., "Manhattan", "NYC-Manhattan", "Yes (Manhattan)"
    capacity_text: Optional[str] = None        # raw text as stated in the answer
    capacity_exact: Optional[str] = None       # exact seat count as stated in the answer (string preferred)
    venue_sources: List[str] = Field(default_factory=list)


class ADAExtraction(BaseModel):
    ada_calc_min_spaces_text: Optional[str] = None  # stated minimum spaces (string, e.g., "15")
    ada_calc_method: Optional[str] = None           # description of formula/reference shown (string)
    venue_wheelchair_spaces_text: Optional[str] = None  # any claim of availability/compliance
    accessibility_sources: List[str] = Field(default_factory=list)


class PricingTier(BaseModel):
    section_name: Optional[str] = None
    price_range: Optional[str] = None  # e.g., "$59-$149", "$100–$200", "around $80–$120"


class PricingExtraction(BaseModel):
    tier1: Optional[PricingTier] = None
    tier2: Optional[PricingTier] = None
    tier3: Optional[PricingTier] = None
    pricing_sources: List[str] = Field(default_factory=list)
    premium_seating_text: Optional[str] = None  # e.g., "VIP/premium seating available"
    premium_sources: List[str] = Field(default_factory=list)


class ProductionExtraction(BaseModel):
    production_title: Optional[str] = None
    performance_dates_text: Optional[str] = None  # Raw dates text from answer
    production_sources: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    accessible_restrooms_text: Optional[str] = None
    accessible_entrances_text: Optional[str] = None
    facilities_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the venue identification and capacity details exactly as stated in the answer.

    Required fields:
    - venue_name: Official venue name (string)
    - venue_address: Complete street address including city and ZIP if available (string)
    - manhattan_indicator: The borough or explicit statement indicating the venue is in Manhattan (string as stated)
    - capacity_text: The seating capacity description exactly as stated (string; can be "approx. 1,500 seats", etc.)
    - capacity_exact: If an exact capacity is stated, extract the exact seat count as a string (e.g., "1600"); otherwise null
    - venue_sources: URLs explicitly cited that document the venue identity/capacity/location (list of strings)

    Notes:
    - Sources must be explicit URLs in the answer. Include official venues, reputable databases (e.g., Playbill, IBDB, Wikipedia).
    - Do not invent any information; only extract what the answer provides.
    """


def prompt_extract_ada_info() -> str:
    return """
    Extract ADA wheelchair seating calculation and availability details from the answer.

    Required fields:
    - ada_calc_min_spaces_text: The stated minimum number of wheelchair-accessible spaces calculated using ADA standards (string; e.g., "16")
    - ada_calc_method: The formula or ADA standards reference text shown (string; e.g., "10 per 1000, proportional (1%)")
    - venue_wheelchair_spaces_text: Any statement confirming the venue meets ADA wheelchair seating requirements (string as stated)
    - accessibility_sources: URLs explicitly cited that document wheelchair seating/ADA accessibility and accessible facilities (list of strings)

    Notes:
    - If the answer cites ADA documentation or venue accessibility pages, include those URLs.
    - Do not infer; only extract what is explicitly stated.
    """


def prompt_extract_pricing_info() -> str:
    return """
    Extract the three-tier pricing structure and premium seating details from the answer.

    Required fields:
    - tier1: { "section_name": string, "price_range": string } for the first tier (e.g., Orchestra/premium ground)
    - tier2: { "section_name": string, "price_range": string } for the second tier (e.g., Mezzanine/mid-level)
    - tier3: { "section_name": string, "price_range": string } for the third tier (e.g., Balcony/upper level)
    - pricing_sources: URLs explicitly cited that document ticket pricing by section (list of strings)
    - premium_seating_text: Any statement about premium/VIP seating availability (string as stated)
    - premium_sources: URLs explicitly cited that document premium/VIP seating (list of strings)

    Notes:
    - Price ranges should be dollar amounts or textual ranges as stated in the answer (e.g., "$59–$149").
    - If a tier is missing, set the corresponding fields to null.
    """


def prompt_extract_production_info() -> str:
    return """
    Extract recent production history details from the answer.

    Required fields:
    - production_title: Specific Broadway/major production title (string)
    - performance_dates_text: Performance dates or run period text exactly as stated (string)
    - production_sources: URLs explicitly cited that document this production at the venue, including dates (list of strings)

    Notes:
    - The required window is February 2025 to February 2026.
    - Sources may include official venue pages, Playbill, IBDB, Ticketmaster/Telecharge, or reputable press.
    """


def prompt_extract_facilities_info() -> str:
    return """
    Extract accessible patron facilities beyond wheelchair seating.

    Required fields:
    - accessible_restrooms_text: Statement confirming accessible restrooms (string as stated)
    - accessible_entrances_text: Statement confirming accessible entrance/lobby access (string as stated)
    - facilities_sources: URLs explicitly cited that document accessible facilities (list of strings)

    Notes:
    - Sources can overlap with accessibility_sources; extract all URLs mentioned for facilities.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", text)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return int(num)
    except Exception:
        return None


def _parse_capacity(venue_ex: VenueExtraction) -> Optional[int]:
    # Prefer capacity_exact if present; otherwise parse from capacity_text
    if venue_ex.capacity_exact:
        val = _parse_first_int(venue_ex.capacity_exact)
        if val:
            return val
    return _parse_first_int(venue_ex.capacity_text)


def _compute_ada_min_spaces(capacity: Optional[int]) -> Optional[int]:
    if capacity is None or capacity <= 0:
        return None
    # Proportional rule: 10 per 1000 seats => 1%; round up
    return math.ceil(capacity / 100)


def _any_urls(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        if lst:
            merged.extend(lst)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in merged:
        if u and u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_venue_and_capacity_specifications(
    evaluator: Evaluator,
    parent_node,
    venue_ex: VenueExtraction
) -> None:
    node = evaluator.add_parallel(
        id="venue_and_capacity_specifications",
        desc="Venue identification, location, and capacity requirements verified",
        parent=parent_node,
        critical=True
    )

    # Sub-node: venue_identity (parallel, critical)
    identity_node = evaluator.add_parallel(
        id="venue_identity",
        desc="Official venue name and complete address provided",
        parent=node,
        critical=True
    )

    # Leaf: venue_name provided (existence)
    evaluator.add_custom_node(
        result=bool(venue_ex.venue_name and venue_ex.venue_name.strip()),
        id="venue_name",
        desc="Official venue name provided",
        parent=identity_node,
        critical=True
    )

    # Leaf: venue_address provided (existence)
    evaluator.add_custom_node(
        result=bool(venue_ex.venue_address and venue_ex.venue_address.strip()),
        id="venue_address",
        desc="Complete street address provided",
        parent=identity_node,
        critical=True
    )

    # Leaf: manhattan_location (verify by URLs)
    manhattan_leaf = evaluator.add_leaf(
        id="manhattan_location",
        desc="Venue confirmed as major Manhattan theater suitable for Broadway productions",
        parent=node,
        critical=True
    )
    manhattan_claim = (
        f"The venue '{venue_ex.venue_name or 'the venue'}' is located in Manhattan, New York City, "
        f"and is a major theater suitable for Broadway-caliber productions."
    )
    await evaluator.verify(
        claim=manhattan_claim,
        node=manhattan_leaf,
        sources=venue_ex.venue_sources,
        additional_instruction=(
            "Confirm the venue is in the Manhattan borough (not other boroughs) and is recognized as a major theater "
            "hosting Broadway-level productions. Use the provided official venue page or reputable sources."
        )
    )

    # Leaf: capacity_compliance (verify exact count and range)
    capacity_leaf = evaluator.add_leaf(
        id="capacity_compliance",
        desc="Exact seating capacity between 1,000-1,800 seats verified",
        parent=node,
        critical=True
    )
    capacity_val = _parse_capacity(venue_ex)
    if capacity_val is not None:
        capacity_claim = (
            f"The theater's total seating capacity is exactly {capacity_val} seats, which lies between 1,000 and 1,800 seats."
        )
    else:
        # fallback textual claim if exact numeric not parsed
        capacity_claim = (
            f"The theater's total seating capacity is stated as '{venue_ex.capacity_text or 'unknown'}', "
            f"and should be between 1,000 and 1,800 seats."
        )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=venue_ex.venue_sources,
        additional_instruction=(
            "Verify the official seat count on the venue site or reputable sources. Also confirm the count falls within "
            "the 1,000–1,800 requirement window."
        )
    )

    # Leaf: venue_documentation_url (existence of official source URL)
    evaluator.add_custom_node(
        result=bool(venue_ex.venue_sources and len(venue_ex.venue_sources) > 0),
        id="venue_documentation_url",
        desc="Official venue source URL provided for verification",
        parent=node,
        critical=True
    )


async def build_accessibility_standards_compliance(
    evaluator: Evaluator,
    parent_node,
    venue_ex: VenueExtraction,
    ada_ex: ADAExtraction
) -> None:
    node = evaluator.add_parallel(
        id="accessibility_standards_compliance",
        desc="ADA wheelchair seating and accessible facilities requirements fully met",
        parent=parent_node,
        critical=True
    )

    # Sub-tree: wheelchair_seating_requirements (sequential, critical)
    ws_node = evaluator.add_sequential(
        id="wheelchair_seating_requirements",
        desc="ADA wheelchair-accessible seating calculated and verified",
        parent=node,
        critical=True
    )

    # Sub-node: ada_requirement_calculation (parallel, critical)
    calc_node = evaluator.add_parallel(
        id="ada_requirement_calculation",
        desc="Minimum required wheelchair spaces correctly calculated using ADA standards",
        parent=ws_node,
        critical=True
    )

    capacity_val = _parse_capacity(venue_ex)
    expected_min = _compute_ada_min_spaces(capacity_val)
    stated_min = _parse_first_int(ada_ex.ada_calc_min_spaces_text)

    # Leaf: capacity_range_determination (simple verify - formula applicability)
    crd_leaf = evaluator.add_leaf(
        id="capacity_range_determination",
        desc="Venue capacity matched to correct ADA standards table range",
        parent=calc_node,
        critical=True
    )
    crd_claim = (
        f"Using ADA proportional guidance (10 wheelchair spaces per 1,000 seats, i.e., 1%), "
        f"a venue capacity of {capacity_val if capacity_val is not None else 'unknown'} seats falls under the same 1% rule applied proportionally."
    )
    await evaluator.verify(
        claim=crd_claim,
        node=crd_leaf,
        sources=None,
        additional_instruction=(
            "Treat 10 per 1000 seats as the base ADA proportional rule for wheelchair spaces. This applies proportionally "
            "to capacities other than 1,000. Verify that the proportional rule is applicable."
        )
    )

    # Leaf: minimum_spaces_derived (simple verify correctness)
    msd_leaf = evaluator.add_leaf(
        id="minimum_spaces_derived",
        desc="Correct minimum wheelchair spaces number stated based on ADA standards",
        parent=calc_node,
        critical=True
    )
    msd_claim = (
        f"The minimum required wheelchair-accessible seating for a capacity of {capacity_val if capacity_val is not None else 'unknown'} "
        f"seats is {expected_min if expected_min is not None else 'unknown'} spaces based on the 10-per-1000 (1%) proportional rule. "
        f"The answer's stated minimum is {stated_min if stated_min is not None else 'not stated'}."
    )
    await evaluator.verify(
        claim=msd_claim,
        node=msd_leaf,
        sources=None,
        additional_instruction=(
            "Judge whether the stated minimum equals ceil(capacity/100). If the stated minimum is missing or does not match, this should fail."
        )
    )

    # Leaf: calculation_shown (simple verify that formula/reference shown)
    calc_shown_leaf = evaluator.add_leaf(
        id="calculation_shown",
        desc="Formula or ADA standards reference shown for calculation",
        parent=calc_node,
        critical=True
    )
    calc_shown_claim = (
        f"The answer shows a formula or ADA standards reference for wheelchair space calculation "
        f"(e.g., '10 per 1000', '1%', 'ADA'). Extracted method: '{ada_ex.ada_calc_method or 'none'}'."
    )
    await evaluator.verify(
        claim=calc_shown_claim,
        node=calc_shown_leaf,
        sources=ada_ex.accessibility_sources if ada_ex.accessibility_sources else None,
        additional_instruction=(
            "Check the answer text for explicit formula or ADA reference; if present, pass. If absent, fail."
        )
    )

    # Sub-node: venue_wheelchair_availability (sequential, critical)
    avail_node = evaluator.add_sequential(
        id="venue_wheelchair_availability",
        desc="Venue meets or exceeds ADA wheelchair seating requirement",
        parent=ws_node,
        critical=True
    )

    # Leaf: ada_compliance_confirmed (verify by URLs)
    ada_conf_leaf = evaluator.add_leaf(
        id="ada_compliance_confirmed",
        desc="Venue confirmed to meet ADA wheelchair seating standards",
        parent=avail_node,
        critical=True
    )
    ada_conf_claim = (
        "The venue meets ADA wheelchair seating standards (i.e., provides designated wheelchair-accessible seating that "
        "meets or exceeds ADA requirements)."
    )
    await evaluator.verify(
        claim=ada_conf_claim,
        node=ada_conf_leaf,
        sources=_any_urls(ada_ex.accessibility_sources, venue_ex.venue_sources),
        additional_instruction=(
            "Look for official or authoritative statements indicating ADA-compliant wheelchair seating availability at the venue."
        )
    )

    # Leaf: accessibility_verification (verify by URLs)
    acc_ver_leaf = evaluator.add_leaf(
        id="accessibility_verification",
        desc="Wheelchair seating availability verified through authoritative source",
        parent=avail_node,
        critical=True
    )
    acc_ver_claim = (
        "Wheelchair-accessible seating is explicitly described or confirmed by an authoritative/official source for this venue."
    )
    await evaluator.verify(
        claim=acc_ver_claim,
        node=acc_ver_leaf,
        sources=ada_ex.accessibility_sources if ada_ex.accessibility_sources else venue_ex.venue_sources,
        additional_instruction=(
            "Verify that at least one provided URL explicitly mentions wheelchair seating availability or accessible seating."
        )
    )

    # Sub-node: accessible_facilities (parallel, critical)
    fac_node = evaluator.add_parallel(
        id="accessible_facilities",
        desc="Accessible patron facilities beyond wheelchair seating verified",
        parent=node,
        critical=True
    )

    # Leaf: accessible_restrooms (verify by URLs)
    rest_leaf = evaluator.add_leaf(
        id="accessible_restrooms",
        desc="Accessible restroom facilities confirmed",
        parent=fac_node,
        critical=True
    )
    rest_claim = "The venue provides accessible restroom facilities for patrons with disabilities."
    await evaluator.verify(
        claim=rest_claim,
        node=rest_leaf,
        sources=_any_urls(ada_ex.accessibility_sources, ada_ex.accessibility_sources, []),
        additional_instruction=(
            "Confirm presence of accessible restrooms on official venue or reputable accessibility pages. "
            "If unclear, check venue accessibility/FAQ sections."
        )
    )

    # Leaf: accessible_entrances (verify by URLs)
    ent_leaf = evaluator.add_leaf(
        id="accessible_entrances",
        desc="Accessible entrance and lobby access confirmed",
        parent=fac_node,
        critical=True
    )
    ent_claim = "The venue provides accessible entrance and lobby access for wheelchair users."
    await evaluator.verify(
        claim=ent_claim,
        node=ent_leaf,
        sources=_any_urls(ada_ex.accessibility_sources, []),
        additional_instruction=(
            "Look for statements indicating accessible entrance/lobby, ramps, elevators, or similar access accommodations."
        )
    )

    # Leaf: accessibility_documentation_url (existence)
    evaluator.add_custom_node(
        result=bool(ada_ex.accessibility_sources and len(ada_ex.accessibility_sources) > 0),
        id="accessibility_documentation_url",
        desc="Supporting URL for accessibility information including wheelchair seating and facilities",
        parent=node,
        critical=True
    )


async def build_commercial_pricing_structure(
    evaluator: Evaluator,
    parent_node,
    pricing_ex: PricingExtraction,
    venue_ex: VenueExtraction
) -> None:
    # Important: To allow a non-critical child (premium seating), this parent must be non-critical (framework rule).
    node = evaluator.add_parallel(
        id="commercial_pricing_structure",
        desc="Three-tier pricing and premium seating documented",
        parent=parent_node,
        critical=False
    )

    # Sub-tree: three_tier_system (sequential, critical)
    tts_node = evaluator.add_sequential(
        id="three_tier_system",
        desc="Three distinct price tiers with seating sections and price ranges verified",
        parent=node,
        critical=True
    )

    # Sub-node: tier_sections_identified (parallel, critical)
    tsi_node = evaluator.add_parallel(
        id="tier_sections_identified",
        desc="Three seating section names identified (Orchestra/Mezzanine/Balcony or equivalent)",
        parent=tts_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(pricing_ex.tier1 and pricing_ex.tier1.section_name and pricing_ex.tier1.section_name.strip()),
        id="tier_one_name",
        desc="First tier section name stated (typically Orchestra or premium ground level)",
        parent=tsi_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pricing_ex.tier2 and pricing_ex.tier2.section_name and pricing_ex.tier2.section_name.strip()),
        id="tier_two_name",
        desc="Second tier section name stated (typically Mezzanine or mid-level)",
        parent=tsi_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pricing_ex.tier3 and pricing_ex.tier3.section_name and pricing_ex.tier3.section_name.strip()),
        id="tier_three_name",
        desc="Third tier section name stated (typically Balcony or upper level)",
        parent=tsi_node,
        critical=True
    )

    # Sub-node: tier_pricing_ranges (parallel, critical)
    tpr_node = evaluator.add_parallel(
        id="tier_pricing_ranges",
        desc="Dollar price ranges provided for all three tiers",
        parent=tts_node,
        critical=True
    )

    # Prepare leaves for each tier prices
    tier1_price_leaf = evaluator.add_leaf(
        id="tier_one_prices",
        desc="First tier price range stated with dollar amounts",
        parent=tpr_node,
        critical=True
    )
    tier2_price_leaf = evaluator.add_leaf(
        id="tier_two_prices",
        desc="Second tier price range stated with dollar amounts",
        parent=tpr_node,
        critical=True
    )
    tier3_price_leaf = evaluator.add_leaf(
        id="tier_three_prices",
        desc="Third tier price range stated with dollar amounts",
        parent=tpr_node,
        critical=True
    )

    # Build claims for tier pricing (fall back to venue sources if pricing sources empty)
    pricing_urls = pricing_ex.pricing_sources if pricing_ex.pricing_sources else venue_ex.venue_sources

    t1_section = pricing_ex.tier1.section_name if pricing_ex.tier1 else None
    t1_range = pricing_ex.tier1.price_range if pricing_ex.tier1 else None
    claim_t1 = (
        f"Tickets in the {t1_section or 'first tier'} section are typically priced in the range {t1_range or 'unknown'}."
    )
    t2_section = pricing_ex.tier2.section_name if pricing_ex.tier2 else None
    t2_range = pricing_ex.tier2.price_range if pricing_ex.tier2 else None
    claim_t2 = (
        f"Tickets in the {t2_section or 'second tier'} section are typically priced in the range {t2_range or 'unknown'}."
    )
    t3_section = pricing_ex.tier3.section_name if pricing_ex.tier3 else None
    t3_range = pricing_ex.tier3.price_range if pricing_ex.tier3 else None
    claim_t3 = (
        f"Tickets in the {t3_section or 'third tier'} section are typically priced in the range {t3_range or 'unknown'}."
    )

    await evaluator.batch_verify([
        (claim_t1, pricing_urls, tier1_price_leaf, "Verify the price range for the specified section using the provided pricing URLs."),
        (claim_t2, pricing_urls, tier2_price_leaf, "Verify the price range for the specified section using the provided pricing URLs."),
        (claim_t3, pricing_urls, tier3_price_leaf, "Verify the price range for the specified section using the provided pricing URLs."),
    ])

    # Leaf: pricing_documentation_url (existence)
    evaluator.add_custom_node(
        result=bool(pricing_ex.pricing_sources and len(pricing_ex.pricing_sources) > 0),
        id="pricing_documentation_url",
        desc="Supporting URL for pricing information",
        parent=tts_node,
        critical=True
    )

    # Leaf: premium_seating_options (non-critical)
    premium_leaf = evaluator.add_leaf(
        id="premium_seating_options",
        desc="Premium or VIP seating availability confirmed",
        parent=node,
        critical=False
    )
    premium_claim = (
        f"Premium or VIP seating options are available at this venue. Stated: '{pricing_ex.premium_seating_text or 'unknown'}'."
    )
    premium_urls = _any_urls(pricing_ex.premium_sources, pricing_ex.pricing_sources, venue_ex.venue_sources)
    await evaluator.verify(
        claim=premium_claim,
        node=premium_leaf,
        sources=premium_urls,
        additional_instruction="Confirm that the venue offers premium/VIP seating options using the provided URLs."
    )


async def build_operational_production_history(
    evaluator: Evaluator,
    parent_node,
    prod_ex: ProductionExtraction
) -> None:
    node = evaluator.add_sequential(
        id="operational_production_history",
        desc="Recent theatrical production within February 2025 - February 2026 verified",
        parent=parent_node,
        critical=True
    )

    # Sub-node: production_details (parallel, critical)
    details_node = evaluator.add_parallel(
        id="production_details",
        desc="Broadway show or major theatrical production name and dates verified",
        parent=node,
        critical=True
    )

    # Leaf: production_name (verify by URLs)
    prod_name_leaf = evaluator.add_leaf(
        id="production_name",
        desc="Specific production title stated",
        parent=details_node,
        critical=True
    )
    prod_name_claim = (
        f"The venue hosted a Broadway or major theatrical production titled '{prod_ex.production_title or 'unknown'}'."
    )
    await evaluator.verify(
        claim=prod_name_claim,
        node=prod_name_leaf,
        sources=prod_ex.production_sources,
        additional_instruction="Verify that the provided sources show the production title at this venue."
    )

    # Leaf: production_dates_verified (verify by URLs)
    prod_dates_leaf = evaluator.add_leaf(
        id="production_dates_verified",
        desc="Performance dates within February 2025 - February 2026 window confirmed",
        parent=details_node,
        critical=True
    )
    prod_dates_claim = (
        f"The production '{prod_ex.production_title or 'unknown'}' has performance dates within the window "
        f"from {PRODUCTION_WINDOW_START} to {PRODUCTION_WINDOW_END}. Stated dates: '{prod_ex.performance_dates_text or 'unknown'}'."
    )
    await evaluator.verify(
        claim=prod_dates_claim,
        node=prod_dates_leaf,
        sources=prod_ex.production_sources,
        additional_instruction=(
            f"Confirm from the source(s) that performance dates fall within the inclusive window {PRODUCTION_WINDOW_START} to {PRODUCTION_WINDOW_END}."
        )
    )

    # Leaf: production_history_documentation_url (existence)
    evaluator.add_custom_node(
        result=bool(prod_ex.production_sources and len(prod_ex.production_sources) > 0),
        id="production_history_documentation_url",
        desc="Supporting URL for production history",
        parent=node,
        critical=True
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
    """
    Evaluate the answer for the Broadway venue identification and verification task.
    Returns a structured evaluation summary containing the verification tree and scores.
    """
    # Initialize evaluator (root is non-critical to allow partial credit where appropriate)
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

    # Concurrent extraction of all sections
    venue_task = evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_info"
    )
    ada_task = evaluator.extract(
        prompt=prompt_extract_ada_info(),
        template_class=ADAExtraction,
        extraction_name="ada_info"
    )
    pricing_task = evaluator.extract(
        prompt=prompt_extract_pricing_info(),
        template_class=PricingExtraction,
        extraction_name="pricing_info"
    )
    prod_task = evaluator.extract(
        prompt=prompt_extract_production_info(),
        template_class=ProductionExtraction,
        extraction_name="production_info"
    )
    fac_task = evaluator.extract(
        prompt=prompt_extract_facilities_info(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_info"
    )

    venue_ex, ada_ex, pricing_ex, prod_ex, fac_ex = await asyncio.gather(
        venue_task, ada_task, pricing_task, prod_task, fac_task
    )

    # Add ground truth / constraints info (for context in summary)
    evaluator.add_ground_truth({
        "capacity_range_requirement": "1,000–1,800 seats",
        "ada_proportional_rule": "10 wheelchair spaces per 1,000 seats (1%) - use ceil(capacity/100)",
        "production_window": {"start": PRODUCTION_WINDOW_START, "end": PRODUCTION_WINDOW_END},
        "pricing_structure_requirement": "Three distinct price tiers by section with price ranges"
    }, gt_type="requirements")

    # Build and verify the tree
    await build_venue_and_capacity_specifications(evaluator, root, venue_ex)
    await build_accessibility_standards_compliance(evaluator, root, venue_ex, ada_ex)
    await build_commercial_pricing_structure(evaluator, root, pricing_ex, venue_ex)
    await build_operational_production_history(evaluator, root, prod_ex)

    # Note: Facilities extraction provides additional context/sources; used within accessibility builder via accessibility_sources.
    # Record facilities extraction info explicitly into summary for transparency.
    evaluator.add_custom_info({
        "accessible_restrooms_text": fac_ex.accessible_restrooms_text,
        "accessible_entrances_text": fac_ex.accessible_entrances_text,
        "facilities_sources": fac_ex.facilities_sources
    }, info_type="facilities_context", info_name="facilities_extraction_summary")

    # Return structured result
    return evaluator.get_summary()