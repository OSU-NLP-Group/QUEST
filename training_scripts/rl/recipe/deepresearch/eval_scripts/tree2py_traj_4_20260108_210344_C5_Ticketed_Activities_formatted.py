import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nw_arkansas_2026_venue_and_razorbacks_policies"
TASK_DESCRIPTION = (
    "A family from out of state is planning to visit Northwest Arkansas in 2026 and wants to attend multiple cultural "
    "and sporting events. They need to research venue policies before purchasing tickets. Identify the performing arts "
    "center in Fayetteville, Arkansas that offers a Broadway subscription series for the 2025-26 season, and provide "
    "the following information with URL references from official sources: (1) The venue's complete name and street "
    "address, (2) The number of shows included in their standard Broadway subscription package, (3) The subscriber "
    "discount percentage offered on ticket prices, (4) The child ticket policy at Arkansas Razorbacks athletic venues "
    "(specifically whether children of any age require tickets), and (5) The maximum dimensions for clear bags allowed "
    "at Arkansas Razorbacks stadiums."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    """Performing arts center info and subscription details extracted from the answer."""
    venue_name: Optional[str] = None
    venue_full_address: Optional[str] = None
    # URLs to official venue pages (home, contact, address, etc.)
    venue_address_urls: List[str] = Field(default_factory=list)
    # One or more official URLs that specifically discuss the Broadway subscription and/or season
    broadway_subscription_urls: List[str] = Field(default_factory=list)
    # Number of shows in the standard Broadway subscription package (string to allow variants)
    standard_show_count: Optional[str] = None
    # Subscriber discount percentage (e.g., "10%"); if answer explicitly states none, set to "none" or "no"
    subscriber_discount_percent: Optional[str] = None


class RazorbacksInfo(BaseModel):
    """Arkansas Razorbacks policies extracted from the answer."""
    # Boolean: True if children of any age require tickets; False if they do not; Null if unknown
    child_ticket_any_age_require: Optional[bool] = None
    # The exact text/policy statement as provided in the answer (optional)
    child_ticket_policy_text: Optional[str] = None
    # Official Razorbacks/UofA URLs supporting the child ticket policy
    child_ticket_policy_urls: List[str] = Field(default_factory=list)

    # Clear bag maximum dimensions string, in the format: length x width x height (+ optional units)
    clear_bag_max_dimensions: Optional[str] = None
    # Acceptable materials list (e.g., ["plastic", "vinyl", "PVC"])
    clear_bag_materials: List[str] = Field(default_factory=list)
    # Official Razorbacks/UofA URLs supporting clear bag policy
    clear_bag_policy_urls: List[str] = Field(default_factory=list)


class OverallExtraction(BaseModel):
    """Top-level extraction result."""
    venue: Optional[VenueInfo] = None
    razorbacks: Optional[RazorbacksInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the required information from the answer text. Return a JSON object with two sub-objects: 'venue' and 'razorbacks'.

    For 'venue', extract:
    - venue_name: The complete official name of the performing arts center identified in Fayetteville, Arkansas.
    - venue_full_address: The full street address including street number, street name, city, state, and ZIP code.
    - venue_address_urls: All official URLs (from the venue’s own website) that support the venue’s location/address or contact info.
    - broadway_subscription_urls: All official venue URLs that describe the Broadway subscription series for the 2025–26 season.
    - standard_show_count: The number of shows included in the standard Broadway subscription package (do not count optional add-ons).
    - subscriber_discount_percent: The subscriber discount percentage offered on ticket prices if mentioned. If the answer states there is no discount, set this field to "none" or "no". If not mentioned at all, set it to null.

    For 'razorbacks', extract:
    - child_ticket_any_age_require: A boolean. Set true if the answer states children of any age require tickets at Razorbacks venues; false if the answer states they do not; null if unclear.
    - child_ticket_policy_text: The short text snippet explaining the child ticket policy (optional, can be null).
    - child_ticket_policy_urls: All official Razorbacks (arkansasrazorbacks.com) or University of Arkansas (uark.edu) URLs that support the child ticket policy.
    - clear_bag_max_dimensions: The maximum dimensions for clear bags allowed at Razorbacks stadiums, formatted as "length x width x height" (units can be included).
    - clear_bag_materials: A list of acceptable clear-bag materials mentioned (e.g., ["plastic", "vinyl", "PVC"]). If the answer lists a combined phrase, split into items.
    - clear_bag_policy_urls: All official Razorbacks (arkansasrazorbacks.com) or University of Arkansas (uark.edu) URLs that support the clear bag policy.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer text.
    - Prefer official sources: the venue’s own website for venue and subscription details, and Razorbacks/UofA sites for athletic policies.
    - If any field is missing in the answer, set it to null (or empty list for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _slugify(text: Optional[str]) -> str:
    if not text:
        return ""
    import re
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_official_razorbacks(url: str) -> bool:
    d = _domain(url)
    return (d.endswith("arkansasrazorbacks.com") or d.endswith(".arkansasrazorbacks.com")
            or d.endswith("uark.edu") or d.endswith(".uark.edu"))


def _is_official_venue(url: str, venue_name: Optional[str]) -> bool:
    d = _domain(url)
    slug = _slugify(venue_name)
    # Known Fayetteville performing arts center official domains (helpful heuristics)
    known_officials = [
        "waltonartscenter.org",  # Walton Arts Center (Fayetteville, AR)
        "tickets.waltonartscenter.org",
        "faulkner.uark.edu",     # Faulkner Performing Arts Center (U of A)
        "uark.edu"
    ]
    # Direct match against known domains
    if any(d == kd or d.endswith("." + kd) for kd in known_officials):
        return True
    # Heuristic: venue slug appears in domain
    if slug and slug in d.replace("-", "").replace(".", ""):
        return True
    return False


def _any_official_venue(urls: List[str], venue_name: Optional[str]) -> bool:
    return any(_is_official_venue(u, venue_name) for u in urls)


def _any_official_razorbacks(urls: List[str]) -> bool:
    return any(_is_official_razorbacks(u) for u in urls)


def _materials_str(materials: List[str]) -> str:
    # Join materials into a readable phrase
    cleaned = [m.strip() for m in materials if m and m.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identify_correct_venue(
    evaluator: Evaluator,
    root_node,
    info: OverallExtraction
) -> None:
    node = evaluator.add_parallel(
        id="identify_correct_venue",
        desc="Identify a performing arts center located in Fayetteville, Arkansas that offers a Broadway subscription series for the 2025–26 season.",
        parent=root_node,
        critical=True
    )

    venue_name = info.venue.venue_name if info.venue else None
    venue_address = info.venue.venue_full_address if info.venue else None
    venue_address_urls = info.venue.venue_address_urls if info.venue else []
    subscription_urls = info.venue.broadway_subscription_urls if info.venue else []

    # Leaf: venue is in Fayetteville, AR
    leaf_location = evaluator.add_leaf(
        id="venue_is_in_fayetteville_ar",
        desc="Answer identifies a performing arts center located in Fayetteville, Arkansas.",
        parent=node,
        critical=True
    )
    claim_loc = f"The venue '{venue_name}' is located in Fayetteville, Arkansas, at address: {venue_address}."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_location,
        sources=venue_address_urls,
        additional_instruction="Verify the venue's location is Fayetteville, Arkansas using official venue address/contact pages."
    )

    # Leaf: venue offers 2025–26 Broadway subscription
    leaf_broadway = evaluator.add_leaf(
        id="venue_offers_2025_26_broadway_subscription",
        desc="Answer shows the venue offers a Broadway subscription series for the 2025–26 season.",
        parent=node,
        critical=True
    )
    claim_series = "This venue offers a Broadway subscription series for the 2025–26 season."
    await evaluator.verify(
        claim=claim_series,
        node=leaf_broadway,
        sources=subscription_urls,
        additional_instruction="Confirm that the official venue pages explicitly show a Broadway subscription series for the 2025–26 season (e.g., 'Broadway Series', 'Season Subscription')."
    )


async def build_provide_required_information(
    evaluator: Evaluator,
    root_node,
    info: OverallExtraction
) -> None:
    node = evaluator.add_parallel(
        id="provide_required_information",
        desc="Provide each requested piece of information.",
        parent=root_node,
        critical=True
    )

    venue_name = info.venue.venue_name if info.venue else None
    venue_address = info.venue.venue_full_address if info.venue else None
    venue_address_urls = info.venue.venue_address_urls if info.venue else []

    show_count = info.venue.standard_show_count if info.venue else None
    subscription_urls = info.venue.broadway_subscription_urls if info.venue else []

    discount = info.venue.subscriber_discount_percent if info.venue else None

    child_any_age_require = info.razorbacks.child_ticket_any_age_require if info.razorbacks else None
    child_policy_urls = info.razorbacks.child_ticket_policy_urls if info.razorbacks else []

    bag_dims = info.razorbacks.clear_bag_max_dimensions if info.razorbacks else None
    bag_materials = info.razorbacks.clear_bag_materials if info.razorbacks else []
    bag_policy_urls = info.razorbacks.clear_bag_policy_urls if info.razorbacks else []

    # (1) Venue name and full address
    leaf_name_addr = evaluator.add_leaf(
        id="venue_name_and_full_address",
        desc="Provide the venue’s complete name and full street address (street number, street name, city, state, ZIP).",
        parent=node,
        critical=True
    )
    claim_name_addr = f"The venue's complete name is '{venue_name}' and its full street address is '{venue_address}'."
    await evaluator.verify(
        claim=claim_name_addr,
        node=leaf_name_addr,
        sources=venue_address_urls,
        additional_instruction="Verify that the official venue page shows both the full legal venue name and full street address (including ZIP)."
    )

    # (2) Standard subscription show count
    leaf_show_count = evaluator.add_leaf(
        id="standard_subscription_show_count",
        desc="State the number of shows included in the standard Broadway subscription package.",
        parent=node,
        critical=True
    )
    claim_show_count = f"The standard Broadway subscription package includes {show_count} shows."
    await evaluator.verify(
        claim=claim_show_count,
        node=leaf_show_count,
        sources=subscription_urls,
        additional_instruction="Confirm the number of shows in the core Broadway subscription (exclude optional add-ons)."
    )

    # (3) Subscriber discount information
    leaf_discount = evaluator.add_leaf(
        id="subscriber_discount_information",
        desc="Provide the subscriber discount percentage on ticket prices if one is offered; if no discount is offered, state that no subscriber discount is offered.",
        parent=node,
        critical=True
    )
    # Build claim depending on extraction
    if (discount is None) or (str(discount).strip().lower() in {"", "none", "no", "n/a"}):
        claim_discount = "No subscriber discount is offered on ticket prices for the Broadway subscription."
        add_ins_discount = ("Check the official venue's subscription pages. If no discount percentage is explicitly "
                            "mentioned, conclude that no subscriber discount is offered.")
    else:
        claim_discount = f"The subscriber discount percentage on ticket prices is {discount}."
        add_ins_discount = ("Check the official venue subscription pages for a stated subscriber discount percentage. "
                            "Allow equivalent phrasing like 'save X%' or 'X% discount for subscribers'.")
    await evaluator.verify(
        claim=claim_discount,
        node=leaf_discount,
        sources=subscription_urls,
        additional_instruction=add_ins_discount
    )

    # (4) Razorbacks child ticket policy
    leaf_child = evaluator.add_leaf(
        id="razorbacks_child_ticket_policy",
        desc="State whether children of any age require tickets at Arkansas Razorbacks athletic venues.",
        parent=node,
        critical=True
    )
    if child_any_age_require is True:
        claim_child = "At Arkansas Razorbacks athletic venues, children of any age are required to have tickets."
    elif child_any_age_require is False:
        claim_child = "At Arkansas Razorbacks athletic venues, children of any age are not required to have tickets."
    else:
        # If unknown, construct a generic claim that will likely fail without support
        claim_child = "The Arkansas Razorbacks child ticket policy regarding whether children of any age require tickets is as stated in the answer."
    await evaluator.verify(
        claim=claim_child,
        node=leaf_child,
        sources=child_policy_urls,
        additional_instruction=("Verify the official Razorbacks/UofA policy (ticketing or fan guide) about whether all ages "
                                "require tickets. If policy exempts certain ages (e.g., under 2), then 'children of any age require tickets' is false.")
    )

    # (5) Razorbacks clear bag maximum dimensions (format: L x W x H)
    leaf_bag_dims = evaluator.add_leaf(
        id="razorbacks_clear_bag_max_dimensions_format",
        desc="Provide the maximum dimensions for clear bags allowed at Arkansas Razorbacks stadiums in the format: length x width x height.",
        parent=node,
        critical=True
    )
    claim_bag_dims = f"The maximum clear-bag dimensions allowed at Arkansas Razorbacks stadiums are {bag_dims}."
    await evaluator.verify(
        claim=claim_bag_dims,
        node=leaf_bag_dims,
        sources=bag_policy_urls,
        additional_instruction=("Confirm the maximum dimensions for clear bags and ensure the stated value is in 'length x width x height' format "
                                "(e.g., '12 x 6 x 12 inches'). The claim must align with official Razorbacks/UofA policy pages.")
    )

    # (6) Razorbacks clear bag materials
    leaf_bag_materials = evaluator.add_leaf(
        id="razorbacks_clear_bag_materials",
        desc="Specify acceptable clear-bag materials (plastic, vinyl, or PVC) at Arkansas Razorbacks stadiums.",
        parent=node,
        critical=True
    )
    materials_text = _materials_str(bag_materials)
    claim_bag_materials = f"Acceptable clear-bag materials at Arkansas Razorbacks stadiums include {materials_text}."
    await evaluator.verify(
        claim=claim_bag_materials,
        node=leaf_bag_materials,
        sources=bag_policy_urls,
        additional_instruction=("Verify that the official policy specifies acceptable clear bag materials. Commonly accepted materials are plastic, vinyl, or PVC.")
    )


def build_official_citations_check(
    evaluator: Evaluator,
    root_node,
    info: OverallExtraction
) -> None:
    """
    Add custom node verifying that all major claims have at least one supporting official URL:
    - Venue identification/location + subscription info: must have at least one official venue URL.
    - Razorbacks policies: must have at least one official Razorbacks/UofA URL.
    """
    venue_name = info.venue.venue_name if info.venue else None
    venue_address_urls = info.venue.venue_address_urls if info.venue else []
    subscription_urls = info.venue.broadway_subscription_urls if info.venue else []

    child_policy_urls = info.razorbacks.child_ticket_policy_urls if info.razorbacks else []
    bag_policy_urls = info.razorbacks.clear_bag_policy_urls if info.razorbacks else []

    venue_ok = _any_official_venue(venue_address_urls + subscription_urls, venue_name)
    subscription_ok = _any_official_venue(subscription_urls, venue_name)
    # For venue name/address specifically also require official address source
    address_ok = _any_official_venue(venue_address_urls, venue_name)

    child_ok = _any_official_razorbacks(child_policy_urls)
    bag_dims_ok = _any_official_razorbacks(bag_policy_urls)
    bag_materials_ok = _any_official_razorbacks(bag_policy_urls)

    all_ok = (venue_ok and subscription_ok and address_ok and child_ok and bag_dims_ok and bag_materials_ok)

    # Record diagnostic info
    evaluator.add_custom_info(
        info={
            "venue_name": venue_name,
            "official_check": {
                "venue_address_urls": venue_address_urls,
                "subscription_urls": subscription_urls,
                "child_policy_urls": child_policy_urls,
                "bag_policy_urls": bag_policy_urls,
                "venue_ok": venue_ok,
                "subscription_ok": subscription_ok,
                "address_ok": address_ok,
                "child_ok": child_ok,
                "bag_dims_ok": bag_dims_ok,
                "bag_materials_ok": bag_materials_ok
            }
        },
        info_type="official_url_citations_diagnostics",
        info_name="official_url_citations_diagnostics"
    )

    evaluator.add_custom_node(
        result=all_ok,
        id="official_url_citations_for_all_major_claims",
        desc=("All major claims (venue identification + each requested info item) have at least one supporting valid URL "
              "reference from an official venue or official organization (e.g., Razorbacks/UofA) website."),
        parent=root_node,
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
    Evaluate the agent's answer for the Northwest Arkansas venue and Razorbacks policy task.
    """
    # Initialize evaluator with a critical root (all children must be critical)
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

    # Mark root as critical per rubric (Evaluator.initialize creates non-critical root by default)
    # We implement critical root by creating a critical child that acts as the root aggregator content.
    # However, to strictly follow the rubric, we wrap actual checks under a critical top-level parallel node.
    # Create a critical top-level wrapper node and use it as parent for all subsequent checks.
    critical_root = evaluator.add_parallel(
        id="root",
        desc="Identify the Fayetteville, AR performing arts center offering a 2025–26 Broadway subscription series and provide the requested subscription and Arkansas Razorbacks policy details.",
        parent=root,
        critical=True
    )

    # Extract all required structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=OverallExtraction,
        extraction_name="structured_extraction"
    )

    # Build verification subtrees under the critical root
    await build_identify_correct_venue(evaluator, critical_root, extracted_info)
    await build_provide_required_information(evaluator, critical_root, extracted_info)
    build_official_citations_check(evaluator, critical_root, extracted_info)

    # Return evaluation summary
    return evaluator.get_summary()