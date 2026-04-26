import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "phoenix_sf_homes"
TASK_DESCRIPTION = """I am relocating to Phoenix, Arizona, and am searching for single-family homes that meet specific criteria. Please find four different single-family homes currently for sale in Phoenix, Arizona that satisfy ALL of the following requirements:

Property Specifications:
- Property type must be a single-family home (not a condo, townhouse, or multi-family property)
- At least 3 bedrooms
- At least 2 bathrooms
- At least 1,800 square feet of living space
- Built in 2010 or later
- Lot size of at least 5,000 square feet

Financial Requirements:
- Listing price between $350,000 and $550,000 (inclusive)
- If the property has a Homeowners Association (HOA), the monthly HOA fees must be less than $150. Properties with no HOA are also acceptable.

Location:
- The property must be located within Phoenix, Arizona city limits

Listing Requirements:
For each property, you must provide:
1. A valid, accessible listing URL from one of these major real estate websites: Zillow, Realtor.com, Redfin, or Trulia
2. The complete property address
3. Confirmation that listing agent or contact information is displayed in the listing
4. Confirmation that the listing status is currently "For Sale" or "Active" (not sold, pending, or contingent)

Please provide the information for all four properties, ensuring each property meets every requirement listed above.
"""

ALLOWED_DOMAINS = ["zillow.com", "realtor.com", "redfin.com", "trulia.com"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PropertyItem(BaseModel):
    listing_url: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    property_type: Optional[str] = None
    bedrooms: Optional[str] = None
    bathrooms: Optional[str] = None
    square_feet: Optional[str] = None
    year_built: Optional[str] = None
    lot_size: Optional[str] = None
    price: Optional[str] = None
    hoa_monthly: Optional[str] = None
    listing_status: Optional[str] = None
    agent_info: Optional[str] = None
    site_name: Optional[str] = None


class PropertiesExtraction(BaseModel):
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
Extract up to the first 6 properties mentioned in the answer. For each property, return the following fields exactly as stated in the answer (do not infer anything not explicitly written):
- listing_url: The direct URL to the listing page. Only include it if it is explicitly present in the answer text and is from one of these sites: Zillow, Realtor.com, Redfin, or Trulia. If multiple URLs are present for a property, return the first one from the allowed websites. If none provided, return null.
- address: The complete street address string as written in the answer (e.g., "1234 W Example St, Phoenix, AZ 85001"). If not provided, return null.
- city: The city name if provided for the property (e.g., "Phoenix"), else null.
- state: The state (prefer two-letter code like "AZ") if provided, else null.
- property_type: The property type string as written (e.g., "Single Family Residence", "Townhouse", "Condo", etc.), else null.
- bedrooms: The bedrooms info string as written (e.g., "3 bd", "4 bedrooms"), else null.
- bathrooms: The bathrooms info string as written (e.g., "2 ba", "2.5 baths"), else null.
- square_feet: The interior living area string (e.g., "1,850 sqft"), else null.
- year_built: The year built string if stated (e.g., "Built in 2015"), else null.
- lot_size: The lot size string (e.g., "5,500 sqft", "0.14 acres"), else null.
- price: The listing price string (e.g., "$489,900"), else null.
- hoa_monthly: The HOA fee string as written. If the answer states "no HOA", use "no HOA"; if it states an annual fee, still return the original text (e.g., "$600 annually"); else null.
- listing_status: The listing status string as written (e.g., "Active", "For Sale"), else null.
- agent_info: If the answer explicitly mentions that the listing shows a listing agent or contact info for that property, set "yes"; if it says it does not, set "no"; otherwise null.
- site_name: If listing_url is provided and belongs to an allowed site, set to "Zillow", "Realtor.com", "Redfin", or "Trulia" respectively; else null.

Important:
- Only extract what is explicitly present in the answer text. Do not guess or look up elsewhere during extraction.
- Keep strings as they appear (including units and symbols).
- If the URL lacks protocol, prepend "http://".
- Return a JSON object with a top-level "properties" array of objects with these fields.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def get_allowed_site(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = (url or "").lower()
    if "zillow.com" in host:
        return "Zillow"
    if "realtor.com" in host:
        return "Realtor.com"
    if "redfin.com" in host:
        return "Redfin"
    if "trulia.com" in host:
        return "Trulia"
    return None


def url_is_allowed(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(dom in u for dom in ALLOWED_DOMAINS)


# --------------------------------------------------------------------------- #
# Verification for one property                                               #
# --------------------------------------------------------------------------- #
async def verify_property(
    evaluator: Evaluator,
    parent_node,
    prop: PropertyItem,
    index: int,
) -> None:
    """
    Build the verification tree and run checks for a single property.
    """
    pid = f"Property_{index + 1}"

    # Property-level container (non-critical to allow partial credit across properties)
    prop_node = evaluator.add_parallel(
        id=pid,
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth'][index] if index < 6 else f'#{index+1}'} single-family home meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # Sub-containers (critical as per rubric)
    basic_node = evaluator.add_parallel(
        id=f"{pid}_Basic_Info",
        desc=f"Basic property specifications for {pid}",
        parent=prop_node,
        critical=True
    )

    features_node = evaluator.add_parallel(
        id=f"{pid}_Features",
        desc=f"Property features and characteristics for {pid}",
        parent=prop_node,
        critical=True
    )

    financial_node = evaluator.add_parallel(
        id=f"{pid}_Financial",
        desc=f"Financial information for {pid}",
        parent=prop_node,
        critical=True
    )

    listing_node = evaluator.add_parallel(
        id=f"{pid}_Listing_Verification",
        desc=f"Listing verification information for {pid}",
        parent=prop_node,
        critical=True
    )

    # 1) Listing URL checks (first, to gate the rest)
    url_exists_allowed = evaluator.add_custom_node(
        result=url_is_allowed(prop.listing_url),
        id=f"{pid}_URL_Provided_Allowed",
        desc=f"{pid}: Listing URL provided from allowed websites (Zillow, Realtor.com, Redfin, or Trulia)",
        parent=listing_node,
        critical=True
    )

    url_leaf = evaluator.add_leaf(
        id=f"{pid}_URL",
        desc="Valid listing URL from Zillow, Realtor.com, Redfin, or Trulia is provided",
        parent=listing_node,
        critical=True
    )

    site = get_allowed_site(prop.listing_url)
    site_hint = site or "an allowed site (Zillow, Realtor.com, Redfin, Trulia)"
    await evaluator.verify(
        claim=f"This URL is a legitimate, accessible residential property listing page on {site_hint}.",
        node=url_leaf,
        sources=prop.listing_url,
        additional_instruction="Do not verify price or status here; only confirm this is a real listing page (not a search result or article) on the specified real estate site."
    )

    prereq = [url_exists_allowed, url_leaf]

    # 2) Basic Info (all verified via the listing URL)
    # Property Type: single-family home
    type_leaf = evaluator.add_leaf(
        id=f"{pid}_Type",
        desc="Property type is single-family home",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property type shown on this listing is a single-family home (also acceptable: 'single family residence', 'single family'). It is not a condo, townhouse, apartment, or multi-family.",
        node=type_leaf,
        sources=prop.listing_url,
        additional_instruction="Allow minor wording variations. Reject if condo/townhouse/multi-family/duplex/triplex/fourplex/apartment is indicated as the property type.",
        extra_prerequisites=prereq
    )

    # Bedrooms >= 3
    beds_leaf = evaluator.add_leaf(
        id=f"{pid}_Bedrooms",
        desc="Property has at least 3 bedrooms",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim="The home has at least 3 bedrooms (3 or more).",
        node=beds_leaf,
        sources=prop.listing_url,
        additional_instruction="If the listing shows a bedroom count (e.g., '3 bd', '4 beds'), pass only if it is 3 or higher.",
        extra_prerequisites=prereq
    )

    # Bathrooms >= 2
    baths_leaf = evaluator.add_leaf(
        id=f"{pid}_Bathrooms",
        desc="Property has at least 2 bathrooms",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim="The home has at least 2 bathrooms in total (2.0 or more).",
        node=baths_leaf,
        sources=prop.listing_url,
        additional_instruction="Count full baths; accept 2.0, 2.5, 3, etc. Do not count 1.5 as meeting the requirement.",
        extra_prerequisites=prereq
    )

    # Square footage >= 1,800
    sqft_leaf = evaluator.add_leaf(
        id=f"{pid}_Square_Footage",
        desc="Property has at least 1,800 square feet of living space",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim="The interior living area is at least 1,800 square feet.",
        node=sqft_leaf,
        sources=prop.listing_url,
        additional_instruction="If the listing shows square footage (e.g., '1,850 sqft'), pass only if it is 1,800 or greater.",
        extra_prerequisites=prereq
    )

    # 3) Features
    # Year built >= 2010
    year_leaf = evaluator.add_leaf(
        id=f"{pid}_Year_Built",
        desc="Property was built in 2010 or later",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property was built in 2010 or later.",
        node=year_leaf,
        sources=prop.listing_url,
        additional_instruction="If a range is shown, use the stated 'Year built'. Pass only if year >= 2010.",
        extra_prerequisites=prereq
    )

    # Lot size >= 5,000 sqft (allow acre conversion)
    lot_leaf = evaluator.add_leaf(
        id=f"{pid}_Lot_Size",
        desc="Property has a lot size of at least 5,000 square feet",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="The lot size is at least 5,000 square feet.",
        node=lot_leaf,
        sources=prop.listing_url,
        additional_instruction="If lot size is in acres, convert: 1 acre = 43,560 sqft. Pass only if lot >= 5,000 sqft.",
        extra_prerequisites=prereq
    )

    # Location in Phoenix city limits
    loc_leaf = evaluator.add_leaf(
        id=f"{pid}_Location",
        desc="Property is located in Phoenix, Arizona",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property's address indicates the city is Phoenix, AZ (within Phoenix city limits).",
        node=loc_leaf,
        sources=prop.listing_url,
        additional_instruction="Pass only if the city on the listing is 'Phoenix' (or clearly within Phoenix city limits). Do not pass if it is a different city such as Glendale, Scottsdale, Tempe, Mesa, etc.",
        extra_prerequisites=prereq
    )

    # 4) Financials
    # Price within range
    price_leaf = evaluator.add_leaf(
        id=f"{pid}_Price",
        desc="Listing price is between $350,000 and $550,000",
        parent=financial_node,
        critical=True
    )
    await evaluator.verify(
        claim="The current listing price is between $350,000 and $550,000 inclusive.",
        node=price_leaf,
        sources=prop.listing_url,
        additional_instruction="Use the price shown on the page (ignore historical prices). Pass only if 350,000 <= price <= 550,000.",
        extra_prerequisites=prereq
    )

    # HOA monthly < $150 or no HOA
    hoa_leaf = evaluator.add_leaf(
        id=f"{pid}_HOA",
        desc="Monthly HOA fees are less than $150 OR property has no HOA",
        parent=financial_node,
        critical=True
    )
    await evaluator.verify(
        claim="Either the property has no HOA or the monthly HOA fee is less than $150.",
        node=hoa_leaf,
        sources=prop.listing_url,
        additional_instruction="If HOA is listed annually, convert to monthly by dividing by 12. Pass if monthly < $150 or listing explicitly indicates no HOA. If HOA info is absent or unclear, fail.",
        extra_prerequisites=prereq
    )

    # 5) Listing verification additional requirements
    # Address provided in the answer (existence check)
    address_exists = bool(prop.address and prop.address.strip())
    evaluator.add_custom_node(
        result=address_exists,
        id=f"{pid}_Address",
        desc="Complete property address is provided",
        parent=listing_node,
        critical=True
    )

    # Agent/contact info displayed on listing (verify by URL)
    agent_leaf = evaluator.add_leaf(
        id=f"{pid}_Agent_Info",
        desc="Listing agent or contact information is displayed in the listing",
        parent=listing_node,
        critical=True
    )
    await evaluator.verify(
        claim="The listing page displays a listing agent name or brokerage contact information (e.g., agent name, phone, email, or a clear 'Contact agent' section).",
        node=agent_leaf,
        sources=prop.listing_url,
        additional_instruction="Pass if there is explicit agent or brokerage contact detail or a dedicated contact agent widget. Fail if no contact/agent info is present.",
        extra_prerequisites=prereq
    )

    # Listing status is active / for sale
    status_leaf = evaluator.add_leaf(
        id=f"{pid}_Status",
        desc="Listing status is currently active (for sale, not sold or pending)",
        parent=listing_node,
        critical=True
    )
    await evaluator.verify(
        claim="The listing status is currently Active or For Sale (not sold, pending, under contract, or contingent).",
        node=status_leaf,
        sources=prop.listing_url,
        additional_instruction="Accept: 'Active', 'For Sale'. Reject: 'Pending', 'Under Contract', 'Contingent', 'Sold', 'Off Market'.",
        extra_prerequisites=prereq
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
    Evaluate an answer for the Phoenix single-family homes task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Properties are evaluated independently
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

    # Optional task wrapper node (non-critical to allow partial scoring across properties)
    task_node = evaluator.add_parallel(
        id="Find_Four_Single_Family_Homes_in_Phoenix",
        desc="Find four single-family homes for sale in Phoenix, Arizona that meet all specified criteria",
        parent=root,
        critical=False
    )

    # Extract properties from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="extracted_properties"
    )

    # Keep only the first four properties; pad if fewer
    props = list(extracted.properties[:4])
    while len(props) < 4:
        props.append(PropertyItem())

    # Record constraints as ground truth/context
    evaluator.add_ground_truth({
        "constraints": {
            "type": "single-family home only",
            "bedrooms": ">= 3",
            "bathrooms": ">= 2",
            "living_area_sqft": ">= 1,800",
            "year_built": ">= 2010",
            "lot_size_sqft": ">= 5,000",
            "price_range_usd": "[350,000, 550,000]",
            "hoa_monthly": "< 150 or no HOA",
            "location": "Phoenix, AZ (within city limits)",
            "listing_status": "Active/For Sale",
            "allowed_sites": ALLOWED_DOMAINS
        }
    })

    evaluator.add_custom_info(
        info={"allowed_domains": ALLOWED_DOMAINS},
        info_type="config",
        info_name="allowed_listing_websites"
    )

    # Verify each property sequentially to ensure URL gating is evaluated first per property
    for i in range(4):
        await verify_property(evaluator, task_node, props[i], i)

    return evaluator.get_summary()