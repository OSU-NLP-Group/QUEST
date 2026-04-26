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
TASK_ID = "san_jose_homes"
TASK_DESCRIPTION = """I am relocating to San Jose, California for work and need help finding suitable single-family homes to purchase. Please identify four single-family residential properties currently for sale in San Jose, CA that meet ALL of the following criteria:

Property Specifications:
- At least 3 bedrooms
- At least 2 bathrooms
- Minimum 1,800 square feet of living space
- Listed price between $350,000 and $550,000
- Property type must be single-family residential (not condos, townhomes, or multi-family)

Location and Neighborhood Features:
- Located within San Jose, California city limits
- At least one assigned elementary school must have a GreatSchools rating of 7 or higher
- Property must have a Walk Score of at least 50 (classified as "Somewhat Walkable")

Property Features and Amenities:
- Must include at least a 2-car garage
- Property must have been built in 1990 or later
- Either no HOA, or if HOA exists, monthly fees must be less than $200

Listing Requirements:
- Property must be actively listed for sale (not pending, contingent, or sold)
- Property must have been on the market for 90 days or less
- Listing must be available on at least one of these platforms: Zillow, Realtor.com, or Redfin

For each of the four properties, please provide:
1. Complete property address
2. Number of bedrooms and bathrooms
3. Total square footage
4. Listed price
5. Year built
6. Garage information (number of car spaces)
7. HOA status and monthly fees (if applicable)
8. Days on market
9. Walk Score
10. Name and GreatSchools rating of at least one assigned elementary school
11. Direct URL to the listing on Zillow, Realtor.com, or Redfin
"""

ALLOWED_PLATFORMS = ["zillow.com", "realtor.com", "redfin.com"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SchoolInfo(BaseModel):
    name: Optional[str] = None
    rating: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PropertyItem(BaseModel):
    address: Optional[str] = None
    property_type: Optional[str] = None
    bedrooms: Optional[str] = None
    bathrooms: Optional[str] = None
    square_feet: Optional[str] = None
    price: Optional[str] = None
    year_built: Optional[str] = None
    garage_spaces: Optional[str] = None
    hoa: Optional[str] = None
    hoa_fee_monthly: Optional[str] = None
    days_on_market: Optional[str] = None
    walk_score: Optional[str] = None
    walkscore_url: Optional[str] = None
    listing_url: Optional[str] = None
    listing_platform: Optional[str] = None
    listing_status: Optional[str] = None
    school: Optional[SchoolInfo] = None
    extra_urls: List[str] = Field(default_factory=list)


class PropertiesExtraction(BaseModel):
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return (
        "Extract up to four properties from the answer that match the requested details. "
        "For each property, return a JSON object with the following fields:\n"
        "- address: complete street address, city, and state\n"
        "- property_type: as stated (e.g., Single Family Residence, Condo, Townhome)\n"
        "- bedrooms: number of bedrooms as mentioned\n"
        "- bathrooms: number of bathrooms as mentioned\n"
        "- square_feet: total living area as mentioned (include units if present)\n"
        "- price: listed price as mentioned (include currency symbol if present)\n"
        "- year_built: year built\n"
        "- garage_spaces: number of car spaces for the garage as described\n"
        "- hoa: text about HOA presence (e.g., 'None', 'Yes')\n"
        "- hoa_fee_monthly: monthly HOA fees as stated (if any)\n"
        "- days_on_market: value mentioned for days on market or time on market\n"
        "- walk_score: the Walk Score if provided\n"
        "- walkscore_url: URL to a WalkScore page for this property if provided\n"
        "- listing_url: direct URL to the listing on Zillow, Realtor.com, or Redfin\n"
        "- listing_platform: one of Zillow, Realtor.com, or Redfin if stated\n"
        "- listing_status: listing status as stated (e.g., Active, Pending, Contingent, Sold)\n"
        "- school: an object with:\n"
        "    • name: name of at least one assigned elementary school\n"
        "    • rating: the GreatSchools rating if provided\n"
        "    • urls: a list of any URLs given that support the school info (e.g., GreatSchools page)\n"
        "- extra_urls: any additional URLs mentioned that support facts for this property\n\n"
        "Rules:\n"
        "1. Only extract information explicitly present in the answer; do not invent anything.\n"
        "2. If a field is missing, set it to null. For URL fields, set to null or empty list if not given.\n"
        "3. Include URLs in full form; accept plain URLs or markdown links.\n"
        "4. Preserve original formatting for numbers (e.g., '1,850 sq ft', '$499,000').\n"
        "5. If more than four properties are in the answer, only extract the first four in order of appearance."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_allowed_listing_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    lower = url.lower()
    return any(domain in lower for domain in ALLOWED_PLATFORMS)


def safe_list(value: Optional[List[str]]) -> List[str]:
    return value if isinstance(value, list) else []


def combined_sources(prop: PropertyItem, extra: Optional[List[str]] = None) -> List[str]:
    urls: List[str] = []
    if prop.listing_url:
        urls.append(prop.listing_url)
    if prop.school and prop.school.urls:
        urls.extend([u for u in prop.school.urls if isinstance(u, str) and u])
    if prop.walkscore_url:
        urls.append(prop.walkscore_url)
    if extra:
        urls.extend([u for u in extra if isinstance(u, str) and u])
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


# --------------------------------------------------------------------------- #
# Verification per property                                                   #
# --------------------------------------------------------------------------- #
async def verify_property(
    evaluator: Evaluator,
    root_node,
    prop: PropertyItem,
    prop_index: int,
) -> None:
    """
    Construct verification tree for one property and run checks.
    """
    pnum = prop_index + 1
    property_node = evaluator.add_parallel(
        id=f"Property_{pnum}",
        desc=("First qualifying property identified and verified" if pnum == 1 else
              "Second qualifying property identified and verified" if pnum == 2 else
              "Third qualifying property identified and verified" if pnum == 3 else
              "Fourth qualifying property identified and verified"),
        parent=root_node,
        critical=False
    )

    # Listing Information group (create early to capture URL node reference)
    listing_info_node = evaluator.add_parallel(
        id=f"Listing_Information_P{pnum}",
        desc="Listing meets freshness and accessibility requirements",
        parent=property_node,
        critical=True
    )

    # Listing URL provided (critical)
    listing_url_ok = is_allowed_listing_url(prop.listing_url)
    listing_url_leaf = evaluator.add_custom_node(
        result=listing_url_ok,
        id=f"Listing_URL_P{pnum}",
        desc="Valid listing URL provided from Zillow, Realtor.com, or Redfin",
        parent=listing_info_node,
        critical=True
    )

    # Active status (critical)
    active_leaf = evaluator.add_leaf(
        id=f"Active_Status_P{pnum}",
        desc="Property is actively listed for sale (not pending or sold)",
        parent=listing_info_node,
        critical=True
    )
    await evaluator.verify(
        claim="This listing is currently active (for sale) and not marked pending, contingent, or sold.",
        node=active_leaf,
        sources=prop.listing_url,
        additional_instruction="Check the listing status label or text. Accept synonyms like 'Active', 'For Sale'. Reject 'Pending', 'Contingent', 'Under Contract', 'Sold'.",
        extra_prerequisites=[listing_url_leaf]
    )

    # Days on market <= 90 (critical)
    dom_leaf = evaluator.add_leaf(
        id=f"Days_On_Market_P{pnum}",
        desc="Property has been on market for 90 days or less",
        parent=listing_info_node,
        critical=True
    )
    await evaluator.verify(
        claim="The listing's Days on Market (or equivalent metric like 'Time on site') is 90 days or less.",
        node=dom_leaf,
        sources=prop.listing_url,
        additional_instruction="Look for 'Days on Market', 'Time on Redfin', 'Days listed', or similar. If multiple values appear, use the current listing's value.",
        extra_prerequisites=[listing_url_leaf]
    )

    # Basic Property Specs group
    basic_specs_node = evaluator.add_parallel(
        id=f"Basic_Property_Specs_P{pnum}",
        desc="Property meets all basic specification requirements",
        parent=property_node,
        critical=True
    )

    # Address provided (critical existence)
    address_leaf = evaluator.add_custom_node(
        result=bool(prop.address and prop.address.strip()),
        id=f"Property_Address_P{pnum}",
        desc="Complete property address is provided",
        parent=basic_specs_node,
        critical=True
    )

    # Single-family type (critical)
    type_leaf = evaluator.add_leaf(
        id=f"Property_Type_P{pnum}",
        desc="Property is a single-family residential home",
        parent=basic_specs_node,
        critical=True
    )
    await evaluator.verify(
        claim="This listing is a single-family residential home (house), not a condo, townhome, or multi-family.",
        node=type_leaf,
        sources=prop.listing_url,
        additional_instruction="Look for labels like 'Single Family Residence' or 'SFH'. Reject types like 'Condo', 'Townhouse', 'Multi-family'.",
        extra_prerequisites=[listing_url_leaf]
    )

    # Bedrooms >= 3 (critical)
    beds_leaf = evaluator.add_leaf(
        id=f"Bedroom_Count_P{pnum}",
        desc="Property has at least 3 bedrooms",
        parent=basic_specs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property has at least 3 bedrooms.",
        node=beds_leaf,
        sources=prop.listing_url,
        additional_instruction="Accept '3' or higher. Minor formatting variations (e.g., '3+ bedrooms') are acceptable.",
        extra_prerequisites=[listing_url_leaf]
    )

    # Bathrooms >= 2 (critical)
    baths_leaf = evaluator.add_leaf(
        id=f"Bathroom_Count_P{pnum}",
        desc="Property has at least 2 bathrooms",
        parent=basic_specs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property has at least 2 bathrooms.",
        node=baths_leaf,
        sources=prop.listing_url,
        additional_instruction="Accept '2' or higher. Include half baths if the platform counts them but ensure total full+half meets the threshold if the platform aggregates.",
        extra_prerequisites=[listing_url_leaf]
    )

    # Square footage >= 1,800 (critical)
    sqft_leaf = evaluator.add_leaf(
        id=f"Square_Footage_P{pnum}",
        desc="Property has at least 1,800 square feet of living space",
        parent=basic_specs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property's living area (square footage) is at least 1,800 sq ft.",
        node=sqft_leaf,
        sources=prop.listing_url,
        additional_instruction="Look for 'sq ft' or 'square feet'. Accept reasonable rounding. If multiple values appear, use the primary living area.",
        extra_prerequisites=[listing_url_leaf]
    )

    # Price within range $350k-$550k (critical)
    price_leaf = evaluator.add_leaf(
        id=f"Price_Range_P{pnum}",
        desc="Property is listed between $350,000 and $550,000",
        parent=basic_specs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The list price is between $350,000 and $550,000 (inclusive).",
        node=price_leaf,
        sources=prop.listing_url,
        additional_instruction="Use the current asking price. Ignore estimated values (e.g., Zestimate).",
        extra_prerequisites=[listing_url_leaf]
    )

    # Location Features group
    location_node = evaluator.add_parallel(
        id=f"Location_Features_P{pnum}",
        desc="Property meets location-based requirements",
        parent=property_node,
        critical=True
    )

    # Geographic location in San Jose (critical)
    geo_leaf = evaluator.add_leaf(
        id=f"Geographic_Location_P{pnum}",
        desc="Property is located in San Jose, California",
        parent=location_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property is located within the city limits of San Jose, California.",
        node=geo_leaf,
        sources=prop.listing_url,
        additional_instruction="Check the address or location field. Accept 'San Jose, CA' or equivalent. Minor spelling variants like 'San José' are acceptable.",
        extra_prerequisites=[listing_url_leaf, address_leaf]
    )

    # School Information subgroup (critical)
    school_node = evaluator.add_parallel(
        id=f"School_Information_P{pnum}",
        desc="Elementary school information is complete and meets rating requirement",
        parent=location_node,
        critical=True
    )

    # School name provided (critical existence)
    school_name_leaf = evaluator.add_custom_node(
        result=bool(prop.school and prop.school.name and prop.school.name.strip()),
        id=f"School_Name_Provided_P{pnum}",
        desc="Name of at least one assigned elementary school is provided",
        parent=school_node,
        critical=True
    )

    # School rating >= 7 (critical)
    school_rating_leaf = evaluator.add_leaf(
        id=f"School_Rating_Threshold_P{pnum}",
        desc="The identified elementary school has a GreatSchools rating of 7 or higher",
        parent=school_node,
        critical=True
    )
    school_sources = combined_sources(prop, extra=None)
    school_name = prop.school.name if (prop.school and prop.school.name) else "the assigned elementary school"
    await evaluator.verify(
        claim=f"{school_name} has a GreatSchools rating of 7 or higher.",
        node=school_rating_leaf,
        sources=school_sources,
        additional_instruction="Prefer GreatSchools pages or embedded ratings on listing pages. Ensure the school is 'Assigned' (not merely nearby).",
        extra_prerequisites=[school_name_leaf]
    )

    # Walk Score >= 50 (critical)
    walk_leaf = evaluator.add_leaf(
        id=f"Walk_Score_P{pnum}",
        desc="Property has a Walk Score of at least 50 (Somewhat Walkable)",
        parent=location_node,
        critical=True
    )
    walk_sources = combined_sources(prop, extra=None)
    await evaluator.verify(
        claim="The property's Walk Score is at least 50 (classified as 'Somewhat Walkable' or better).",
        node=walk_leaf,
        sources=walk_sources,
        additional_instruction="Use WalkScore.com pages, widgets, or listing page indicators that show Walk Score.",
        extra_prerequisites=[listing_url_leaf]
    )

    # Property Features group
    features_node = evaluator.add_parallel(
        id=f"Property_Features_P{pnum}",
        desc="Property includes required amenities and features",
        parent=property_node,
        critical=True
    )

    # Garage >= 2-car (critical)
    garage_leaf = evaluator.add_leaf(
        id=f"Garage_P{pnum}",
        desc="Property includes at least a 2-car garage",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property has a garage with at least 2 car spaces.",
        node=garage_leaf,
        sources=prop.listing_url,
        additional_instruction="Look for 'Garage', 'Attached garage', 'Detached garage' with '2-car' or '2 spaces'. Do not count open parking without a garage.",
        extra_prerequisites=[listing_url_leaf]
    )

    # Year built >= 1990 (critical)
    year_leaf = evaluator.add_leaf(
        id=f"Year_Built_P{pnum}",
        desc="Property was built in 1990 or later",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property was built in 1990 or later.",
        node=year_leaf,
        sources=prop.listing_url,
        additional_instruction="Check 'Year built' or 'Built in' field.",
        extra_prerequisites=[listing_url_leaf]
    )

    # HOA none or monthly < $200 (critical)
    hoa_leaf = evaluator.add_leaf(
        id=f"HOA_Status_P{pnum}",
        desc="Property either has no HOA or HOA fees are less than $200 per month",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property either has no HOA, or the monthly HOA fee is less than $200.",
        node=hoa_leaf,
        sources=prop.listing_url,
        additional_instruction="Check HOA section. Accept 'None', 'HOA: $0', or fees under $200/month.",
        extra_prerequisites=[listing_url_leaf]
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
    Evaluate the agent's answer for the San Jose homes task and return an evaluation summary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # As per rubric root parallel
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find four single-family homes for sale that meet all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract properties
    extracted = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="properties_extraction"
    )

    # Ensure we evaluate exactly 4 properties: take first 4, pad with empty if fewer
    props = list(extracted.properties[:4])
    while len(props) < 4:
        props.append(PropertyItem())

    # Build property subtrees
    for idx, prop in enumerate(props):
        await verify_property(evaluator, root, prop, idx)

    # Return summary
    return evaluator.get_summary()