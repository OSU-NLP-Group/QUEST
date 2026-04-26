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
TASK_ID = "lv_office_leed_4props"
TASK_DESCRIPTION = (
    "I am a corporate real estate consultant helping a technology company relocate their headquarters to the Las Vegas metropolitan area. "
    "The company is committed to sustainability and environmental responsibility. I need to identify four commercial office properties in the Las Vegas metro area "
    "(including Henderson and Summerlin) that meet the following requirements:\n\n"
    "1. Each property must have at least 5,000 square feet of office space currently available for lease.\n"
    "2. Each property must have LEED certification at any level (Certified, Silver, Gold, or Platinum).\n"
    "3. Each property must be currently available for lease with an active listing.\n"
    "4. Each property listing must include basic building specifications such as total building square footage, number of stories, or year built.\n"
    "5. Each property listing must provide verifiable lease rates (price per square foot per month or per year).\n"
    "6. Each property must be Class A or Class B commercial office space.\n\n"
    "For each of the four properties, please provide:\n"
    "- The property name and full address\n"
    "- The available office space (in square feet) for the specific suite or floor\n"
    "- The LEED certification level (Certified, Silver, Gold, or Platinum)\n"
    "- The lease rate (per square foot per month or per year, and specify the lease type: NNN, Full Service/Gross, or Modified Gross)\n"
    "- Basic building specifications (total building size, number of stories, and/or year built)\n"
    "- A direct link to the property listing"
)

ORDINALS = ["First", "Second", "Third", "Fourth"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LeaseInfo(BaseModel):
    lease_rate_value: Optional[str] = None  # e.g., "$2.75", "$33.00"
    lease_rate_period: Optional[str] = None  # e.g., "per SF per month", "per SF per year"
    lease_type: Optional[str] = None  # e.g., "NNN", "Full Service", "Modified Gross"
    notes: Optional[str] = None


class BuildingSpecs(BaseModel):
    total_building_size_sqft: Optional[str] = None  # e.g., "150,000 SF"
    number_of_stories: Optional[str] = None  # e.g., "10"
    year_built: Optional[str] = None  # e.g., "2008"
    building_class: Optional[str] = None  # e.g., "Class A" or "Class B"


class PropertyItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    available_space_sqft: Optional[str] = None  # for the specific suite/floor
    leed_level: Optional[str] = None  # Certified/Silver/Gold/Platinum

    lease: LeaseInfo = Field(default_factory=LeaseInfo)
    specs: BuildingSpecs = Field(default_factory=BuildingSpecs)

    listing_url: Optional[str] = None
    platform_name: Optional[str] = None
    parking_info: Optional[str] = None

    additional_source_urls: List[str] = Field(default_factory=list)


class PropertiesExtraction(BaseModel):
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
    Extract up to four commercial office properties described in the answer. Focus on the Las Vegas metropolitan area including Las Vegas, Henderson, Summerlin, or North Las Vegas.

    For each property, return an object with the following fields:
    - name: The property/building name, if provided (string)
    - address: The full street address as presented (string)
    - city: City name (e.g., "Las Vegas", "Henderson", "North Las Vegas", "Summerlin" if specified as part of Las Vegas) (string or null)
    - state: The state, typically "NV" (string or null)
    - zip_code: The ZIP code (string or null)

    - available_space_sqft: The available office space for the specific suite or floor (string, keep formatting like "5,500 SF" or ranges)

    - leed_level: The LEED certification level (one of: "Certified", "Silver", "Gold", "Platinum"), if mentioned; otherwise null.
      If only "LEED" is mentioned without level, set to "Certified" if implied; otherwise null.

    - lease: 
      - lease_rate_value: The numeric rate as presented (e.g., "$2.75", "$33.00", "2.75") (string or null)
      - lease_rate_period: The period/denominator (e.g., "per SF per month", "per SF per year") (string or null)
      - lease_type: The lease type (e.g., "NNN", "Triple Net", "Full Service", "Full Service Gross", "Modified Gross", "MG") (string or null)
      - notes: Any extra notes such as "Plus NNN", "Negotiable", etc. (string or null)

    - specs:
      - total_building_size_sqft: Total building size (string or null)
      - number_of_stories: Number of floors/stories (string or null)
      - year_built: Year built (string or null)
      - building_class: Building class (e.g., "Class A", "Class B") (string or null)

    - listing_url: A direct URL to the property listing (string or null)
    - platform_name: The platform/brokerage name if provided (e.g., LoopNet, CBRE, JLL, Colliers, CREXi) (string or null)
    - parking_info: Any parking/transportation info included (e.g., "parking ratio 4/1,000", "garage", "near transit") (string or null)

    - additional_source_urls: Any additional URLs mentioned for this property (list of strings)

    GENERAL RULES:
    - Extract exactly what appears in the answer; do not invent information.
    - If an item is missing, set it to null; do not guess.
    - Keep numbers/dates/sizes as strings to preserve formatting (e.g., "5,000 SF", "2008").
    - For lease_rate_period, use phrases like "per SF per month" or "per SF per year".
    - Only include up to four properties. If more are presented, take the first four with listing URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        su = u.strip()
        if not su:
            continue
        if su not in seen:
            seen.add(su)
            result.append(su)
    return result


def _prop_sources(prop: PropertyItem) -> List[str]:
    # Prefer the direct listing URL, but include any additional sources for robustness
    urls = []
    if prop.listing_url:
        urls.append(prop.listing_url)
    urls.extend(prop.additional_source_urls or [])
    return _unique_urls(urls)


def _domain_from_url(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        # crude parse without external libs
        clean = url.split("://", 1)[-1]
        domain = clean.split("/", 1)[0]
        return domain.lower()
    except Exception:
        return ""


def _ordinal_desc(index: int) -> str:
    if 0 <= index < len(ORDINALS):
        return ORDINALS[index]
    return f"Property #{index + 1}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_property(
    evaluator: Evaluator,
    root_node,
    prop: PropertyItem,
    index: int,
) -> None:
    """
    Build verification tree for one property and perform all checks.
    """
    ordinal = _ordinal_desc(index)

    # Property container node (parallel aggregation)
    prop_node = evaluator.add_parallel(
        id=f"property_{index + 1}",
        desc=f"{ordinal} qualifying commercial office property identified and documented",
        parent=root_node,
        critical=False,
    )

    # 1) Location check (critical)
    loc_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_location",
        desc="Property is located within the Las Vegas metropolitan area (including Henderson and Summerlin)",
        parent=prop_node,
        critical=True,
    )
    location_claim = (
        f"The property is in the Las Vegas metropolitan area (Las Vegas, Henderson, North Las Vegas, or Summerlin in Nevada). "
        f"Address from the answer: {prop.address or 'unknown'}; City: {prop.city or 'unknown'}; State: {prop.state or 'unknown'}."
    )
    await evaluator.verify(
        claim=location_claim,
        node=loc_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Verify by reading the listing page's address/city. Consider Las Vegas, Henderson, North Las Vegas, and Summerlin "
            "as part of the Las Vegas metro in Nevada. If the property is clearly outside these areas, mark as not supported."
        ),
    )

    # 2) Specifications bundle (critical, parallel)
    specs_node = evaluator.add_parallel(
        id=f"property_{index + 1}_specifications",
        desc="Property specifications meet requirements",
        parent=prop_node,
        critical=True,
    )

    # 2.a) Reference URL is provided (existence check, critical)
    ref_exists = bool(prop.listing_url and prop.listing_url.strip())
    evaluator.add_custom_node(
        result=ref_exists,
        id=f"property_{index + 1}_reference",
        desc="Property listing reference URL is provided and accessible",
        parent=specs_node,
        critical=True,
    )

    # 2.b) Available office space >= 5,000 SF (critical)
    office_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_office_space",
        desc="Property provides at least 5,000 square feet of office space currently available",
        parent=specs_node,
        critical=True,
    )
    office_claim = (
        f"The listing shows at least 5,000 square feet of office space currently available for lease "
        f"for the specific suite/floor. Extracted available space: {prop.available_space_sqft or 'unknown'}."
    )
    await evaluator.verify(
        claim=office_claim,
        node=office_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Find the available office space on the listing page (suite/floor level). If multiple suites are listed, "
            "it's acceptable if any single suite/floor has >= 5,000 SF available now. If only ranges are shown, "
            "reasonably judge whether minimum meets 5,000 SF. If the page says 'Call for availability' without any size, fail."
        ),
    )

    # 2.c) Building class is A or B (critical)
    class_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_building_class",
        desc="Property is designated as Class A or Class B commercial office space",
        parent=specs_node,
        critical=True,
    )
    class_claim = (
        f"The property is designated as Class A or Class B office space. Extracted class from the answer: "
        f"{prop.specs.building_class or 'unknown'}."
    )
    await evaluator.verify(
        claim=class_claim,
        node=class_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Look for a 'Building Class' field or description on the listing page indicating 'Class A' or 'Class B'. "
            "Accept reasonable variants (e.g., 'A Class Office'). If class is 'C' or absent, fail."
        ),
    )

    # 2.d) Listing includes building specs: total size / stories / year built (critical)
    details_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_listing_details",
        desc="Property listing includes basic building specifications such as total square footage, number of floors, or year built",
        parent=specs_node,
        critical=True,
    )
    details_claim = (
        f"The listing includes basic building specifications (at least one of: total building size, number of stories, year built). "
        f"Extracted size: {prop.specs.total_building_size_sqft or 'unknown'}; stories: {prop.specs.number_of_stories or 'unknown'}; "
        f"year built: {prop.specs.year_built or 'unknown'}."
    )
    await evaluator.verify(
        claim=details_claim,
        node=details_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Scan the listing page for any of the specified specs. At least one must be explicitly present in the page content. "
            "Do not infer from unrelated text; if none are present, fail."
        ),
    )

    # 3) Sustainability: LEED certification (critical)
    sustain_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_sustainability",
        desc="Property has LEED certification at any level",
        parent=prop_node,
        critical=True,
    )
    sustain_claim = (
        f"The property has LEED certification at any level. Extracted level: {prop.leed_level or 'unknown'}."
    )
    await evaluator.verify(
        claim=sustain_claim,
        node=sustain_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Confirm the listing (or provided source) mentions LEED certification: Certified, Silver, Gold, or Platinum. "
            "If 'LEED' is referenced without level, ensure certification (not just 'LEED-ready'). If no LEED mention, fail."
        ),
    )

    # 4) Financial bundle (critical, parallel)
    fin_node = evaluator.add_parallel(
        id=f"property_{index + 1}_financial",
        desc="Property financial information meets requirements",
        parent=prop_node,
        critical=True,
    )

    # 4.a) Availability active (critical)
    avail_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_availability",
        desc="Property is currently available for lease with active listing",
        parent=fin_node,
        critical=True,
    )
    avail_claim = "The listing indicates the space is currently available for lease (active listing)."
    await evaluator.verify(
        claim=avail_claim,
        node=avail_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Look for words like 'Available', 'Vacant', 'Now Leasing', or active availability tables. "
            "If the listing shows 'Leased' or 'Not available', fail."
        ),
    )

    # 4.b) Pricing shown (critical)
    price_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_pricing",
        desc="Property listing provides verifiable lease rates",
        parent=fin_node,
        critical=True,
    )
    price_claim = (
        f"The listing provides a numeric, verifiable lease rate (e.g., $/SF/month or $/SF/year). "
        f"Extracted in answer: {prop.lease.lease_rate_value or 'unknown'} {prop.lease.lease_rate_period or ''}."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Verify the page shows an explicit rate with units (e.g., $2.75/SF/Mo, $33/SF/Yr). "
            "If it only says 'Call for pricing' or rate is absent, fail."
        ),
    )

    # 4.c) Lease type specified (critical)
    lt_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_lease_type",
        desc="Property listing specifies the lease type (NNN, Full Service/Gross, or Modified Gross)",
        parent=fin_node,
        critical=True,
    )
    lt_claim = (
        f"The listing specifies the lease type (NNN, Full Service/Gross, or Modified Gross). "
        f"Extracted in answer: {prop.lease.lease_type or 'unknown'}."
    )
    await evaluator.verify(
        claim=lt_claim,
        node=lt_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Look for 'NNN' (Triple Net), 'Full Service'/'Full Service Gross' (FS/FG), or 'Modified Gross' (MG). "
            "Accept common abbreviations (NNN, FS, FG, MG). If unspecified, fail."
        ),
    )

    # 5) Platform recognition (non-critical)
    platform_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_platform",
        desc="Property is listed on a recognized commercial real estate platform or brokerage website",
        parent=prop_node,
        critical=False,
    )
    domain = _domain_from_url(prop.listing_url)
    platform_claim = (
        f"The listing domain '{domain or 'unknown'}' represents a recognized commercial real estate platform or brokerage "
        f"(e.g., loopnet.com, crexi.com, costar.com public listing pages, cbre.com, jll.com, colliers.com, "
        f"cushmanwakefield.com, newmark.com, svn.com)."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Judge by the site identity on the page (logo/footer/domain). Recognized examples include LoopNet, CREXi, CoStar public pages, "
            "CBRE, JLL, Colliers, Cushman & Wakefield, Newmark, SVN, Marcus & Millichap. If the site seems to be an obscure blog or unrelated, fail."
        ),
    )

    # 6) Parking/transport access info (non-critical)
    parking_leaf = evaluator.add_leaf(
        id=f"property_{index + 1}_parking",
        desc="Property listing includes parking information or transportation access details",
        parent=prop_node,
        critical=False,
    )
    parking_claim = (
        "The listing includes parking or transportation access information (e.g., parking ratio, surface/garage parking, "
        "EV charging, proximity to transit, shuttle, freeway access)."
    )
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=_prop_sources(prop),
        additional_instruction=(
            "Scan amenities/specs for parking ratio, type of parking, EV charging, transit access, freeway proximity, or similar. "
            "If none of these appear, fail."
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
    Evaluate an answer for the Las Vegas LEED office properties task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # properties assessed independently
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

    # Extract properties from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="properties_extraction",
    )

    # Keep first four properties; pad with empty if fewer
    props = list(extracted.properties[:4])
    while len(props) < 4:
        props.append(PropertyItem())

    # Ground truth constraints for context
    evaluator.add_ground_truth({
        "required_properties_count": 4,
        "constraints": [
            "Las Vegas metro (Las Vegas/Henderson/North Las Vegas/Summerlin)",
            ">= 5,000 SF available office space (suite/floor)",
            "LEED certification (Certified/Silver/Gold/Platinum)",
            "Active listing (currently available)",
            "Listing includes building specs (size/stories/year built)",
            "Verifiable lease rates ($/SF/mo or $/SF/yr)",
            "Building Class A or Class B",
        ]
    })

    # Verify each property
    for i, p in enumerate(props):
        await verify_property(evaluator, root, p, i)

    # Return summary
    return evaluator.get_summary()