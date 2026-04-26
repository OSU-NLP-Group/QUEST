import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "phoenix_rental_house_2026"
TASK_DESCRIPTION = (
    "I am relocating to Phoenix, Arizona in April 2026 and am searching for a rental house for my family. "
    "Find a single-family house for rent in the Phoenix area meeting REQUIRED criteria: "
    "single-family house; monthly rent $2,500–$3,500; at least 3 bedrooms; explicitly allows large dogs (>50 lbs). "
    "Preferred (non-critical): ≥1,800 sqft; ≥2 full bathrooms; ≥2 dedicated parking spaces; in-unit washer/dryer; 12-month lease; "
    "available by April 15, 2026 or earlier; private fenced backyard; central heating and air; within 5 miles of downtown Phoenix; "
    "attached garage; hardwood or LVP flooring in main living areas. "
    "Provide address, rent, square footage, beds/baths, key matching amenities, direct listing link, and confirmation of which criteria are met."
)

MOVE_IN_DEADLINE_ISO = "2026-04-15"
MOVE_IN_DEADLINE_READABLE = "April 15, 2026"
DOWNTOWN_PHOENIX_ANCHOR = "200 W Washington St, Phoenix, AZ 85003"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RentalPropertyExtraction(BaseModel):
    # Core identification
    address: Optional[str] = None
    property_type: Optional[str] = None  # e.g., "single-family", "house", "single family residence"
    listing_urls: List[str] = Field(default_factory=list)  # direct links to the current rental listing(s)

    # Key required fields
    monthly_rent: Optional[str] = None  # rent text as written
    bedrooms: Optional[str] = None      # beds as text (e.g., "3", "3+", "4", "3 beds")
    pet_policy: Optional[str] = None    # text describing pets policy, explicitly mention large dogs if present

    # Preferred fields (non-critical)
    square_footage: Optional[str] = None        # e.g., "1,850 sqft"
    bathrooms: Optional[str] = None             # e.g., "2", "2.5", "2 baths"
    parking: Optional[str] = None               # e.g., "2-car garage", "driveway for 2", etc.
    in_unit_laundry: Optional[str] = None       # e.g., "washer and dryer in unit" (not just hookups)
    lease_term: Optional[str] = None            # e.g., "12-month lease"
    availability_date: Optional[str] = None     # textual move-in availability date, e.g., "available April 1, 2026" or "available now"
    backyard: Optional[str] = None              # e.g., "private fenced backyard"
    central_hvac: Optional[str] = None          # e.g., "central air and heat", "central A/C and heating"
    proximity_note: Optional[str] = None        # any note about distance to downtown if provided
    garage: Optional[str] = None                # e.g., "attached 2-car garage"
    flooring: Optional[str] = None              # e.g., "hardwood", "LVP", "vinyl plank", "carpet", "tile"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property() -> str:
    return """
    Extract details for the single specific rental property recommended in the answer (the primary/first one if multiple are mentioned).
    Return the fields below exactly as they appear in the answer; do not invent values.
    - address: Full street address string as given (ideally including street number, street name, city, state, and ZIP).
    - property_type: The property type as stated (e.g., "single-family house", "single family residence", "house", etc.).
    - listing_urls: Array of direct URLs to the current rental listing(s) for this exact property. Include all URLs shown in the answer that lead directly to the listing page (Zillow, Realtor, Apartments.com house listing, etc.).
    - monthly_rent: The monthly rent exactly as written in the answer (e.g., "$2,950 per month").
    - bedrooms: The bedroom count as written (e.g., "3", "4", "3 beds", "3+").
    - pet_policy: The pet policy text as written; include any weight limits or phrases like "large dogs allowed", etc.
    - square_footage: The total living space size as written (e.g., "1,850 sqft").
    - bathrooms: The bathroom count as written (e.g., "2", "2 baths", "2.5").
    - parking: Any description of parking that indicates capacity (e.g., "2-car garage", "2 parking spaces", "driveway fits 2 cars").
    - in_unit_laundry: Text indicating in-unit washer and dryer if present (avoid "hookups only" or "on-site" if not in-unit).
    - lease_term: The lease term text if present (e.g., "12-month lease").
    - availability_date: The stated or implied availability date (e.g., "available now", "available April 1, 2026").
    - backyard: Text indicating a private fenced backyard if present.
    - central_hvac: Text indicating central air conditioning and heating (HVAC) if present.
    - proximity_note: Any text indicating the property’s distance to or proximity to downtown Phoenix if mentioned.
    - garage: Text indicating an attached garage if present (e.g., "attached 2-car garage").
    - flooring: Flooring information for main living areas if present (e.g., "hardwood", "luxury vinyl plank", "LVP", "tile", "carpet").

    Return null for any field not present in the answer. Do not infer or compute values; only extract from the answer text.
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def is_complete_address(addr: Optional[str]) -> bool:
    if not addr:
        return False
    # Heuristic: requires street number + street name, city, AZ, and 5-digit ZIP
    # Example pattern: "1234 W Example St, Phoenix, AZ 85003"
    pattern = r"\d+\s+[^,]+,\s*[^,]+,\s*AZ\s*\d{5}"
    return re.search(pattern, addr, flags=re.IGNORECASE) is not None


def _urls_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    # Filter obviously malformed URLs
    valid = [u for u in urls if isinstance(u, str) and len(u.strip()) > 0]
    return valid if valid else None


# --------------------------------------------------------------------------- #
# Main verification builder                                                   #
# --------------------------------------------------------------------------- #
async def _build_and_verify(evaluator: Evaluator, root, data: RentalPropertyExtraction) -> None:
    # Normalize sources (listing URLs)
    listing_urls = _urls_or_none(data.listing_urls)

    # Critical: listing URL provided (existence)
    url_provided_node = evaluator.add_custom_node(
        result=bool(listing_urls),
        id="listing_url_provided",
        desc="A direct link to the current rental listing is provided",
        parent=root,
        critical=True,
    )

    # Critical: listing URL accessible and is a rental listing for this property
    url_accessible_node = evaluator.add_leaf(
        id="listing_url_accessible",
        desc="The provided URL is accessible and is a current rental listing for the identified property",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim="This webpage is an accessible rental listing page that corresponds to the property described in the answer.",
        node=url_accessible_node,
        sources=listing_urls if listing_urls else None,
        additional_instruction="Verify that the URL loads and clearly represents an active rental listing for the described property (house for rent) in the Phoenix area.",
    )

    # Critical: property address provided (completeness in the answer text)
    address_complete = is_complete_address(data.address)
    evaluator.add_custom_node(
        result=address_complete,
        id="property_address_provided",
        desc="A complete property address (street address, city, state, ZIP) is provided",
        parent=root,
        critical=True,
    )

    # Prepare verification leaves for required and preferred criteria
    # Required (critical) leaves
    property_type_node = evaluator.add_leaf(
        id="property_type",
        desc="The property is a single-family house (not an apartment, condo, or townhome)",
        parent=root,
        critical=True,
    )
    price_range_node = evaluator.add_leaf(
        id="price_range",
        desc="Monthly rent is between $2,500 and $3,500",
        parent=root,
        critical=True,
    )
    bedroom_count_node = evaluator.add_leaf(
        id="bedroom_count",
        desc="The property has at least 3 bedrooms",
        parent=root,
        critical=True,
    )
    pet_policy_node = evaluator.add_leaf(
        id="pet_policy",
        desc="The property explicitly allows large dogs (over 50 lbs)",
        parent=root,
        critical=True,
    )

    # Preferred (non-critical) leaves
    square_footage_node = evaluator.add_leaf(
        id="square_footage",
        desc="The property has at least 1,800 square feet of living space",
        parent=root,
        critical=False,
    )
    bathroom_count_node = evaluator.add_leaf(
        id="bathroom_count",
        desc="The property has at least 2 full bathrooms",
        parent=root,
        critical=False,
    )
    parking_node = evaluator.add_leaf(
        id="parking",
        desc="The property includes at least 2 dedicated parking spaces (garage or driveway)",
        parent=root,
        critical=False,
    )
    in_unit_laundry_node = evaluator.add_leaf(
        id="in_unit_laundry",
        desc="The property has a washer and dryer in the unit",
        parent=root,
        critical=False,
    )
    lease_term_node = evaluator.add_leaf(
        id="lease_term",
        desc="The property offers a 12-month lease option",
        parent=root,
        critical=False,
    )
    availability_node = evaluator.add_leaf(
        id="availability",
        desc=f"The property is available for move-in by {MOVE_IN_DEADLINE_READABLE} or earlier",
        parent=root,
        critical=False,
    )
    outdoor_space_node = evaluator.add_leaf(
        id="outdoor_space",
        desc="The property includes a private fenced backyard",
        parent=root,
        critical=False,
    )
    central_hvac_node = evaluator.add_leaf(
        id="central_hvac",
        desc="The property has central air conditioning and heating",
        parent=root,
        critical=False,
    )
    proximity_node = evaluator.add_leaf(
        id="proximity",
        desc="The property is located within 5 miles of downtown Phoenix, AZ",
        parent=root,
        critical=False,
    )
    garage_node = evaluator.add_leaf(
        id="garage",
        desc="The property includes an attached garage",
        parent=root,
        critical=False,
    )
    hardwood_floors_node = evaluator.add_leaf(
        id="hardwood_floors",
        desc="The property has hardwood or luxury vinyl plank flooring (not carpet) in main living areas",
        parent=root,
        critical=False,
    )

    # Build claims and verify. We set listing_url_accessible as a prerequisite for web-grounded checks.
    prereq = [url_accessible_node]

    claims_and_sources: List[tuple] = []

    # Required criteria
    claims_and_sources.append((
        "The listing identifies the property type as a single-family house (detached home / single family residence) and not an apartment, condo, or townhome.",
        listing_urls,
        property_type_node,
        "Accept synonyms such as 'single-family', 'single family residence', 'house', 'detached'. "
        "Reject if the listing indicates apartment, condo, townhome, duplex/triplex, or multi-family."
    ))

    claims_and_sources.append((
        "The monthly rent shown on the listing is between $2,500 and $3,500 (inclusive).",
        listing_urls,
        price_range_node,
        "Use the advertised base monthly rent (exclude fees). Accept numbers like $2500, $2,500, or $3,500. Boundary values are acceptable."
    ))

    claims_and_sources.append((
        "The listing shows at least 3 bedrooms.",
        listing_urls,
        bedroom_count_node,
        "Treat '3', '3+', '4', etc. as meeting the requirement. "
        "Do not count dens/lofts as bedrooms unless explicitly called bedrooms."
    ))

    claims_and_sources.append((
        "The pet policy explicitly allows large dogs over 50 lbs, such as stating 'large dogs allowed', 'no weight limit', or a weight limit of at least 50 lb.",
        listing_urls,
        pet_policy_node,
        "Pass if the listing allows large dogs or has a dog weight limit ≥ 50 lb. "
        "If it says small dogs only, weight limit < 50 lb, or only 'case-by-case' without clarity, it should fail."
    ))

    # Preferred criteria
    claims_and_sources.append((
        "The listing shows at least 1,800 square feet of living space.",
        listing_urls,
        square_footage_node,
        "Accept '1,800 sqft' or more (including 'approx 1800'). If the listing shows less than 1,800 sqft or omits square footage, fail."
    ))

    claims_and_sources.append((
        "The listing shows at least 2 full bathrooms.",
        listing_urls,
        bathroom_count_node,
        "Accept '2 baths', '2.0 baths', or '2.5 baths' (which implies 2 full plus one half). "
        "If it clearly indicates fewer than 2 full baths (e.g., 1.5), fail."
    ))

    claims_and_sources.append((
        "The property includes at least 2 dedicated parking spaces, such as a 2-car garage or a driveway that fits two cars.",
        listing_urls,
        parking_node,
        "Pass if the listing mentions '2-car garage', 'two-car garage', or clear indication of two dedicated parking spaces. "
        "Street parking alone is not sufficient."
    ))

    claims_and_sources.append((
        "The property has in-unit washer and dryer.",
        listing_urls,
        in_unit_laundry_node,
        "Pass if the listing states 'washer and dryer in unit' or similar. "
        "Do not accept 'hookups only' or 'on-site' shared laundry."
    ))

    claims_and_sources.append((
        "The property offers a 12-month lease option.",
        listing_urls,
        lease_term_node,
        "Pass if the listing mentions a 12-month lease or 'one year lease' option."
    ))

    claims_and_sources.append((
        f"The property is available for move-in by {MOVE_IN_DEADLINE_READABLE} or earlier.",
        listing_urls,
        availability_node,
        f"Pass if the listing states 'available now' or a date on/before {MOVE_IN_DEADLINE_READABLE}."
    ))

    claims_and_sources.append((
        "The property includes a private fenced backyard.",
        listing_urls,
        outdoor_space_node,
        "Look for phrases like 'fenced yard', 'block wall', 'private backyard', 'fenced backyard'. "
        "Shared or unfenced yards should fail."
    ))

    claims_and_sources.append((
        "The property has central air conditioning and central heating (HVAC).",
        listing_urls,
        central_hvac_node,
        "Accept 'central A/C', 'central air', and central heating indications. Window units or portable ACs do not count."
    ))

    # Proximity: non-critical, allow LLM estimation using address on listing/map
    proximity_additional = (
        "Determine whether the property is within 5 miles (approx. 8 km) of downtown Phoenix, "
        f"using 'Downtown Phoenix' near {DOWNTOWN_PHOENIX_ANCHOR} as the reference point. "
        "If the listing provides a map, neighborhood, or address that clearly lies within ~5 miles, pass; "
        "if uncertain or clearly farther, fail. Do not rely on external tools—use only the listing content."
    )
    claims_and_sources.append((
        "The property is within 5 miles of downtown Phoenix.",
        listing_urls,
        proximity_node,
        proximity_additional,
    ))

    claims_and_sources.append((
        "The property has an attached garage.",
        listing_urls,
        garage_node,
        "Pass only if the listing indicates the garage is attached (e.g., 'attached 2-car garage')."
    ))

    claims_and_sources.append((
        "The main living areas feature hardwood or luxury vinyl plank (LVP) flooring (not carpet).",
        listing_urls,
        hardwood_floors_node,
        "Pass if the listing explicitly mentions hardwood, engineered wood, or LVP/vinyl plank in main living areas. "
        "If the main living areas are carpet or only tile, fail."
    ))

    # Kick off verifications; ensure they are gated on URL accessibility
    batch_payload = []
    for claim, sources, node, add_ins in claims_and_sources:
        batch_payload.append((
            claim,
            sources,
            node,
            add_ins
        ))

    # Run all non-address checks in parallel (after URL accessibility check already executed above)
    await evaluator.batch_verify(
        [
            (c, s, n, ai) for (c, s, n, ai) in batch_payload
        ],
        # Add listing URL accessibility as an extra prerequisite to each verification
        # so if URL is invalid/failed, these will be skipped automatically.
        extra_prerequisites=[url_accessible_node],
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_property(),
        template_class=RentalPropertyExtraction,
        extraction_name="property_extraction",
    )

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={
            "address": extracted.address,
            "property_type": extracted.property_type,
            "monthly_rent": extracted.monthly_rent,
            "bedrooms": extracted.bedrooms,
            "bathrooms": extracted.bathrooms,
            "square_footage": extracted.square_footage,
            "pet_policy": extracted.pet_policy,
            "lease_term": extracted.lease_term,
            "availability_date": extracted.availability_date,
            "parking": extracted.parking,
            "in_unit_laundry": extracted.in_unit_laundry,
            "backyard": extracted.backyard,
            "central_hvac": extracted.central_hvac,
            "proximity_note": extracted.proximity_note,
            "garage": extracted.garage,
            "flooring": extracted.flooring,
            "listing_urls": extracted.listing_urls,
        },
        info_type="extracted_summary",
    )

    # Build verification tree and run checks
    await _build_and_verify(evaluator, root, extracted)

    return evaluator.get_summary()