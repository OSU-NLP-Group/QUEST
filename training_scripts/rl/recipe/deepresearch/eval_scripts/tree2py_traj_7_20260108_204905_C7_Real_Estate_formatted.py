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
TASK_ID = "austin_tx_listing_eval"
TASK_DESCRIPTION = (
    "Find a residential property listing currently for sale in Austin, Texas that meets ALL of the following requirements:\n\n"
    "1. The property must be a single-family detached home\n"
    "2. The property must have at least 3 bedrooms\n"
    "3. Each bedroom must meet the minimum size requirement of 70 square feet\n"
    "4. The property must have a minimum ceiling height of 7 feet in all habitable spaces\n"
    "5. All bedrooms must have emergency egress windows meeting IRC requirements (minimum 5.7 square feet opening area)\n"
    "6. The property must have smoke detectors installed in all bedrooms and on every level\n"
    "7. The property must have carbon monoxide detectors installed outside sleeping areas (if the property has fuel-burning appliances or attached garage)\n"
    "8. The property must include at least 2 off-street parking spaces\n"
    "9. The listing price must be between $400,000 and $500,000\n"
    "10. The property must have at least 1,800 square feet of total living space\n"
    "11. The property must have been built in 2015 or later\n"
    "12. If an HOA exists, the monthly fees must not exceed $300\n"
    "13. The property listing must provide annual property tax information or an estimate\n"
    "14. The property must be currently listed as active/for sale\n"
    "15. The property listing must be from a legitimate real estate platform (Zillow, Realtor.com, Redfin, or local MLS)\n\n"
    "Provide the property address, listing URL, and verification that all requirements are met based on the listing information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PropertyExtraction(BaseModel):
    address: Optional[str] = None
    listing_url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)
    platform: Optional[str] = None
    verification_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property_info() -> str:
    return """
    Extract the key deliverables for the identified Austin, TX property listing from the answer.

    Return a JSON object with the following fields:
    - address: The full street address of the property as stated in the answer (include city/state if present). If not provided, return null.
    - listing_url: The primary property listing URL mentioned in the answer. Prefer a direct listing page from Zillow, Realtor.com, Redfin, or a local MLS. If multiple are present, choose the most authoritative or the first direct listing link. If none provided, return null.
    - extra_urls: An array of any additional listing URLs (duplicates excluded) that refer to the same property mentioned in the answer. These must be explicit URLs present in the answer text. If none, return [].
    - platform: The platform name that the listing_url belongs to. Choose from: "Zillow", "Realtor.com", "Redfin", "MLS", or "Other". Infer from the domain of listing_url. If listing_url is null, return null.
    - verification_text: Copy a concise snippet from the answer where the author verifies that requirements are met using the listing information (e.g., by citing specific details from the listing). If the answer does not provide verification text, return null.

    Special rules for URL extraction:
    - Only extract URLs explicitly present in the answer.
    - Extract full URLs, including protocol. If a URL is missing protocol, prepend http://
    - Ignore malformed or clearly invalid URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc) and "." in parsed.netloc
    except Exception:
        return False


def gather_sources(extracted: PropertyExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.listing_url and is_valid_url(extracted.listing_url):
        urls.append(extracted.listing_url.strip())
    for u in extracted.extra_urls or []:
        if is_valid_url(u):
            urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Build verification tree and run checks                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    root_node,
    extracted: PropertyExtraction,
) -> None:
    # Top-level evaluation node (critical; all children must also be critical)
    ple_node = evaluator.add_parallel(
        id="property_listing_evaluation",
        desc="Evaluate whether the response identifies a qualifying Austin, TX residential listing and verifies all required constraints using listing information",
        parent=root_node,
        critical=True,
    )

    # 1) Required response fields (critical)
    required_node = evaluator.add_parallel(
        id="required_response_fields",
        desc="Response includes the required deliverables (address, listing URL, and verification)",
        parent=ple_node,
        critical=True,
    )

    # 1.a) Provides property address (existence check)
    evaluator.add_custom_node(
        result=bool(extracted.address and extracted.address.strip()),
        id="provides_property_address",
        desc="Response provides the property address",
        parent=required_node,
        critical=True,
    )

    # 1.b) Provides listing URL (valid URL check)
    evaluator.add_custom_node(
        result=is_valid_url(extracted.listing_url),
        id="provides_listing_url",
        desc="Response provides a listing URL",
        parent=required_node,
        critical=True,
    )

    # 1.c) Provides verification based on listing (LLM check on the answer text)
    provides_verif_leaf = evaluator.add_leaf(
        id="provides_verification_based_on_listing",
        desc="Response provides verification that each requirement is met based on the listing information (not merely assumptions)",
        parent=required_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The response includes explicit verification that each requirement is met based on the listing information (e.g., citing details shown on the listing, not just assumptions).",
        node=provides_verif_leaf,
        additional_instruction=(
            "Judge using only the provided answer text. Look for explicit references to listing-stated facts "
            "(e.g., quoting bedrooms, square footage, year built, HOA fees) rather than vague assumptions."
        ),
    )

    # 2) Property constraints (critical)
    constraints_node = evaluator.add_parallel(
        id="property_constraints",
        desc="Listing meets all specified property constraints",
        parent=ple_node,
        critical=True,
    )

    # Prepare sources (listing URLs)
    sources = gather_sources(extracted)

    # Create all constraint leaves
    # Note: All these leaves are critical as per rubric.
    # We rely on auto-preconditions so that if 'provides_listing_url' fails, these will be skipped.
    checks: List[Dict[str, Any]] = []

    # 2.1 Location
    location_leaf = evaluator.add_leaf(
        id="location",
        desc="Property is located in Austin, Texas",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The property is located in Austin, Texas.",
        node=location_leaf,
        sources=sources,
        add_ins=(
            "Use the address/location as shown on the listing page. Accept 'Austin, TX' or 'Austin, Texas'. "
            "If the city is not Austin (e.g., Round Rock, Pflugerville, Cedar Park), mark as not supported."
        ),
    ))

    # 2.2 Listing status
    status_leaf = evaluator.add_leaf(
        id="listing_status",
        desc="Property is currently listed as active/for sale",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The property is currently listed as active/for sale.",
        node=status_leaf,
        sources=sources,
        add_ins=(
            "Confirm the listing status indicates 'Active', 'For Sale', or a clearly equivalent status. "
            "Do not accept 'Pending', 'Contingent', 'Under Contract', 'Coming Soon', or 'Off Market'."
        ),
    ))

    # 2.3 Legitimate platform
    platform_leaf = evaluator.add_leaf(
        id="legitimate_platform",
        desc="Listing is from a legitimate real estate platform (Zillow, Realtor.com, Redfin, or local MLS)",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The listing URL belongs to one of these legitimate platforms: Zillow, Realtor.com, Redfin, or a local MLS.",
        node=platform_leaf,
        sources=sources[:1] if sources else None,  # primary URL is sufficient here
        add_ins=(
            "Determine legitimacy primarily by the URL domain. Accept: zillow.com, realtor.com, redfin.com, or recognized local MLS domains "
            "(e.g., abor.com, actris.com, matrix/broker-hosted MLS sites). If domain is not clearly one of these, mark not supported."
        ),
    ))

    # 2.4 Property type
    property_type_leaf = evaluator.add_leaf(
        id="property_type",
        desc="Property is a single-family detached home",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The property is a single-family detached home.",
        node=property_type_leaf,
        sources=sources,
        add_ins=(
            "Verify that the listing specifies 'single family' or 'single-family detached'. "
            "Do not accept condos, townhomes, duplexes, multi-family, or attached units."
        ),
    ))

    # 2.5 Bedroom count
    br_count_leaf = evaluator.add_leaf(
        id="bedroom_count",
        desc="Property has at least 3 bedrooms",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The property has at least 3 bedrooms.",
        node=br_count_leaf,
        sources=sources,
        add_ins="Confirm the bedroom count is 3 or more as shown on the listing page.",
    ))

    # 2.6 Minimum bedroom size
    br_size_leaf = evaluator.add_leaf(
        id="minimum_bedroom_size",
        desc="Each bedroom meets the minimum size requirement of 70 square feet",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="Each bedroom is at least 70 square feet.",
        node=br_size_leaf,
        sources=sources,
        add_ins=(
            "Only pass if the listing explicitly provides bedroom dimensions or areas demonstrating each bedroom is ≥ 70 sq ft. "
            "If dimensions are available (e.g., 10x8), you may compute area ≈ 80 sq ft. If bedroom sizes are missing or ambiguous, mark not supported."
        ),
    ))

    # 2.7 Ceiling height
    ceiling_leaf = evaluator.add_leaf(
        id="ceiling_height",
        desc="Property has a minimum 7-foot ceiling height in all habitable spaces",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="All habitable spaces have a minimum ceiling height of 7 feet.",
        node=ceiling_leaf,
        sources=sources,
        add_ins=(
            "Only pass if the listing states ceiling height(s) that are ≥ 7 feet for habitable areas. "
            "If the listing does not state ceiling height, mark not supported."
        ),
    ))

    # 2.8 Egress windows
    egress_leaf = evaluator.add_leaf(
        id="egress_windows",
        desc="All bedrooms have emergency egress windows meeting IRC requirements (minimum 5.7 sq ft opening area)",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="All bedrooms have emergency egress windows meeting IRC minimum opening area of 5.7 square feet.",
        node=egress_leaf,
        sources=sources,
        add_ins=(
            "Only pass if the listing explicitly mentions compliant egress windows or provides window opening sizes that meet the requirement. "
            "If not explicitly stated, mark not supported."
        ),
    ))

    # 2.9 Smoke detectors
    smoke_leaf = evaluator.add_leaf(
        id="smoke_detectors",
        desc="Smoke detectors are installed in all bedrooms and on every level",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="Smoke detectors are installed in all bedrooms and on every level of the home.",
        node=smoke_leaf,
        sources=sources,
        add_ins=(
            "Only pass if the listing explicitly states installation of smoke detectors in all bedrooms and on every level. "
            "If not explicitly stated, mark not supported."
        ),
    ))

    # 2.10 Carbon monoxide detectors (conditional)
    co_leaf = evaluator.add_leaf(
        id="carbon_monoxide_detectors",
        desc="Carbon monoxide detectors are installed outside sleeping areas if applicable (e.g., fuel-burning appliances or attached garage)",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim=(
            "If the property has fuel-burning appliances or an attached garage, then carbon monoxide detectors are installed outside sleeping areas. "
            "If the listing clearly indicates no fuel-burning appliances and no attached garage, the requirement is not applicable and thus satisfied."
        ),
        node=co_leaf,
        sources=sources,
        add_ins=(
            "Check listing details for gas/fuel-burning appliances or attached garage. "
            "If either is present, require explicit mention of CO detectors outside sleeping areas. "
            "If neither is present and this is clear, consider the requirement satisfied. If unclear, mark not supported."
        ),
    ))

    # 2.11 Parking spaces
    parking_leaf = evaluator.add_leaf(
        id="parking_spaces",
        desc="Property includes at least 2 off-street parking spaces",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The property includes at least 2 off-street parking spaces.",
        node=parking_leaf,
        sources=sources,
        add_ins=(
            "Use listing fields for parking/garage/driveway capacity. Accept 2+ garage spaces or equivalent off-street parking. "
            "Street parking does not count."
        ),
    ))

    # 2.12 Price range
    price_leaf = evaluator.add_leaf(
        id="price_range",
        desc="Listing price is between $400,000 and $500,000",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The listing price is between $400,000 and $500,000 (inclusive).",
        node=price_leaf,
        sources=sources,
        add_ins=(
            "Check the list price on the page. Normalize currency and commas. Accept values from 400,000 up to 500,000 inclusive."
        ),
    ))

    # 2.13 Total square footage
    sqft_leaf = evaluator.add_leaf(
        id="total_square_footage",
        desc="Property has at least 1,800 square feet of total living space",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The property has at least 1,800 square feet of total living space.",
        node=sqft_leaf,
        sources=sources,
        add_ins=(
            "Use the listing's living area field (finished/heated living space). "
            "Accept 1,800 sq ft or greater. Do not count lot size."
        ),
    ))

    # 2.14 Year built
    year_leaf = evaluator.add_leaf(
        id="year_built",
        desc="Property was built in 2015 or later",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The property was built in 2015 or later.",
        node=year_leaf,
        sources=sources,
        add_ins="Use the listing's 'Year Built' field. Pass only if year >= 2015.",
    ))

    # 2.15 HOA fees (conditional)
    hoa_leaf = evaluator.add_leaf(
        id="hoa_fees",
        desc="If an HOA exists, monthly fees do not exceed $300",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="If an HOA exists for this property, the monthly HOA fee does not exceed $300. If there is no HOA, the requirement is satisfied.",
        node=hoa_leaf,
        sources=sources,
        add_ins=(
            "Check listing fields for HOA presence and fee. "
            "If HOA exists, ensure the monthly fee (or monthly equivalent) is <= $300. "
            "If HOA exists but the fee amount is not provided, mark not supported."
        ),
    ))

    # 2.16 Property tax info
    tax_leaf = evaluator.add_leaf(
        id="property_tax_info",
        desc="Listing provides annual property tax information or an estimate",
        parent=constraints_node,
        critical=True,
    )
    checks.append(dict(
        claim="The listing provides annual property tax information or an estimate.",
        node=tax_leaf,
        sources=sources,
        add_ins=(
            "Look for fields like 'Property tax', 'Annual tax', 'Estimated taxes', or similar. "
            "If the listing does not include any annual tax information or estimate, mark not supported."
        ),
    ))

    # Execute verifications (in parallel where possible)
    claims_and_sources = []
    for item in checks:
        claims_and_sources.append((
            item["claim"],
            item["sources"] if item["sources"] else None,
            item["node"],
            item["add_ins"],
        ))
    await evaluator.batch_verify(claims_and_sources)


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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root aggregator
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

    # Extract property info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_property_info(),
        template_class=PropertyExtraction,
        extraction_name="property_info",
    )

    # Build tree and run verifications
    await build_and_verify(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()