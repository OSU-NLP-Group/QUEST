import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "kroger_tx_center"
TASK_DESCRIPTION = """
Identify a Kroger-anchored community shopping center property currently available for investment or lease in Texas. The property must meet the following specifications: (1) the total center size must be between 125,000 and 400,000 square feet, (2) the Kroger anchor store must be a Marketplace format store (99,000 to 130,000 square feet), (3) the anchor tenant (Kroger) must occupy between 45% and 70% of the total center square footage, (4) the property must be listed on an official Kroger real estate platform (such as kroger.cbre-properties.com), and (5) a verifiable reference URL must be available for the property listing. Provide the property name or address, total center square footage, Kroger store square footage, the percentage of space occupied by Kroger, and the reference URL.
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class PropertyExtraction(BaseModel):
    """
    Structured extraction of the single property the answer presents.
    Prefer strings for robustness; numeric parsing will be handled in code.
    """
    property_name: Optional[str] = None
    property_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    total_center_sqft: Optional[str] = None  # e.g., "250,000 SF"
    kroger_store_sqft: Optional[str] = None  # e.g., "120,000 SF"
    kroger_occupancy_percent: Optional[str] = None  # e.g., "48%", "0.48", "48 percent"

    kroger_store_format: Optional[str] = None  # e.g., "Marketplace"
    anchor_tenant_name: Optional[str] = None   # e.g., "Kroger"

    availability_status: Optional[str] = None  # e.g., "Available", "For Lease", "For Sale"

    listing_url: Optional[str] = None          # Preferred official Kroger real estate link
    additional_urls: List[str] = Field(default_factory=list)  # Any other URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property() -> str:
    return """
    Extract details for a single Kroger-anchored community shopping center property mentioned in the answer. If multiple properties are presented, extract only the first one.

    Return the following fields:
    - property_name: The name of the shopping center (if given), else null.
    - property_address: The street address (if given), else null.
    - city: The city of the property (if given), else null.
    - state: The U.S. state of the property (e.g., "TX", "Texas") if given, else null.

    - total_center_sqft: The total center size / GLA as presented (string, e.g., "250,000 SF"), else null.
    - kroger_store_sqft: The Kroger anchor store square footage (string), else null.
    - kroger_occupancy_percent: The percentage of total center square footage occupied by Kroger as explicitly presented in the answer (string, e.g., "48%" or "0.48"), else null.

    - kroger_store_format: The Kroger store format (e.g., "Marketplace") if mentioned, else null.
    - anchor_tenant_name: The anchor tenant name if mentioned (string), else null.

    - availability_status: Availability wording in the answer such as "Available", "For Lease", "For Sale", "Sublease", else null.

    - listing_url: The main reference URL to the property listing (prefer official Kroger platform URLs such as "https://kroger.cbre-properties.com/..."). If none, null.
    - additional_urls: Any other URLs cited for this property (array). Do not invent URLs; include only those explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    """Extract the first integer number from a text like '250,000 SF' -> 250000."""
    if not text:
        return None
    # Find first group of digits possibly with commas
    m = re.search(r"(\d[\d,\.]*)", text)
    if not m:
        return None
    raw = m.group(1)
    # If contains decimal point with no commas and few digits, treat as float; else remove commas and decimals
    try:
        # Prefer interpreting as integer number of square feet
        num_str = raw.replace(",", "")
        # If it's a pure float like '250000.0', cast to int
        if "." in num_str:
            # If decimals present in SF, drop decimals
            num_float = float(num_str)
            return int(round(num_float))
        return int(num_str)
    except Exception:
        return None


def parse_percent_to_float(percent_text: Optional[str]) -> Optional[float]:
    """
    Parse a percent-like string into a 0-100 float (percentage points).
    Examples:
      "48%" -> 48.0
      "0.48" -> 48.0
      "48 percent" -> 48.0
    """
    if not percent_text:
        return None
    s = percent_text.strip().lower()
    # Extract first numeric token
    m = re.search(r"(-?\d+(\.\d+)?)", s)
    if not m:
        return None
    val = float(m.group(1))
    # If input includes '%' or 'percent', interpret value as percentage points
    if "%" in s or "percent" in s:
        return val
    # Otherwise, if it's a decimal between 0 and 1, interpret as proportion
    if 0.0 <= val <= 1.0:
        return val * 100.0
    # Else assume it's already percentage points
    return val


def is_valid_url(url: Optional[str]) -> bool:
    """Basic URL validity check."""
    if not url or not isinstance(url, str):
        return False
    s = url.strip()
    return s.startswith("http://") or s.startswith("https://")


def name_or_address(extracted: PropertyExtraction) -> str:
    """Return a best identifier string: name if present, else address, else empty string."""
    if extracted.property_name and extracted.property_name.strip():
        return extracted.property_name.strip()
    if extracted.property_address and extracted.property_address.strip():
        return extracted.property_address.strip()
    return ""


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_property_requirements_tree(evaluator: Evaluator, root, extracted: PropertyExtraction) -> None:
    """
    Build the verification tree for the property requirements and run verifications.
    """
    # Parent node: critical, parallel aggregation across requirements
    property_node = evaluator.add_parallel(
        id="Property_Meets_Requirements",
        desc="The response identifies one Kroger-anchored community shopping center property in Texas that is currently available and meets all stated size/source requirements, and provides all requested fields.",
        parent=root,
        critical=True
    )

    # Sources handling: prefer listing_url; if not present, fall back to additional_urls
    primary_url = extracted.listing_url if extracted.listing_url else None
    fallback_sources = extracted.additional_urls if extracted.additional_urls else []
    sources_for_general_checks: List[str] = []
    if primary_url:
        sources_for_general_checks = [primary_url]
    elif fallback_sources:
        sources_for_general_checks = fallback_sources

    # 1) Property identifier provided (name or address) – logic-only existence check
    prop_id_exists = bool(name_or_address(extracted))
    evaluator.add_custom_node(
        result=prop_id_exists,
        id="Property_Identified_Name_Or_Address",
        desc="Provides a property identifier: property name OR property address (at least one).",
        parent=property_node,
        critical=True
    )

    # 2) Verifiable reference URL provided – logic-only URL existence/format check
    evaluator.add_custom_node(
        result=is_valid_url(primary_url),
        id="Verifiable_Reference_URL_Provided",
        desc="Provides a valid, verifiable reference URL that links to the property listing.",
        parent=property_node,
        critical=True
    )

    # Prepare leaf nodes that require webpage verification
    # 3) Listed on official Kroger real estate platform
    listed_official_node = evaluator.add_leaf(
        id="Listed_On_Official_Kroger_Real_Estate_Platform",
        desc="The property listing is on an official Kroger real estate platform (e.g., kroger.cbre-properties.com).",
        parent=property_node,
        critical=True
    )

    # 4) Located in Texas
    located_tx_node = evaluator.add_leaf(
        id="Located_In_Texas",
        desc="Property is located in Texas.",
        parent=property_node,
        critical=True
    )

    # 5) Currently available (sale/lease/sublease)
    available_node = evaluator.add_leaf(
        id="Currently_Available",
        desc="Property is currently available for investment or lease (sale/lease/sublease).",
        parent=property_node,
        critical=True
    )

    # 6) Kroger is anchor tenant
    kroger_anchor_node = evaluator.add_leaf(
        id="Kroger_Is_Anchor_Tenant",
        desc="Kroger is the anchor tenant for the shopping center.",
        parent=property_node,
        critical=True
    )

    # 7) Total center size in range [125,000, 400,000] – verify against listing
    total_center_size_node = evaluator.add_leaf(
        id="Total_Center_Size_In_Range",
        desc="Total center size (GLA) is provided and is between 125,000 and 400,000 square feet (inclusive).",
        parent=property_node,
        critical=True
    )

    # 8) Kroger Marketplace store and size in range [99,000, 130,000] – verify against listing
    marketplace_store_node = evaluator.add_leaf(
        id="Kroger_Marketplace_Store_Size_In_Range",
        desc="Kroger anchor store is Marketplace format and its square footage is provided and is between 99,000 and 130,000 square feet (inclusive).",
        parent=property_node,
        critical=True
    )

    # 9) Kroger occupancy percent provided and in range [45%, 70%] – logic-only check on answer content
    #    The rubric requires the percentage to be explicitly stated in the response and within range.
    occ_val = parse_percent_to_float(extracted.kroger_occupancy_percent)
    occ_provided_and_in_range = occ_val is not None and 45.0 <= occ_val <= 70.0
    evaluator.add_custom_node(
        result=occ_provided_and_in_range,
        id="Kroger_Occupancy_Percent_Provided_And_In_Range",
        desc="The percentage of total center square footage occupied by Kroger is explicitly stated in the response and is between 45% and 70% (inclusive).",
        parent=property_node,
        critical=True
    )

    # Build claims for URL-backed verifications (batch verification)
    # Official platform
    claim_official = (
        "This webpage is part of an official Kroger real estate platform (e.g., a Kroger-branded CBRE properties site such as kroger.cbre-properties.com). "
        "Use the page domain, branding, and content to judge whether it is an official Kroger real estate listing."
    )
    add_ins_official = (
        "If the URL is missing or invalid, judge Incorrect. Prefer domain patterns like 'kroger.cbre-properties.com'. "
        "Look for Kroger branding or explicit statements that this is Kroger's real estate listing portal."
    )

    # Located in Texas
    claimed_city = extracted.city or ""
    claimed_state = extracted.state or ""
    claim_tx = (
        "The property shown on this listing is located in Texas (TX). "
        f"If the answer mentions a city/state, they are: city='{claimed_city}' state='{claimed_state}'."
    )
    add_ins_tx = (
        "Check the address block, property overview, or map on the page. Accept 'TX' or 'Texas'. "
        "Minor formatting differences are acceptable."
    )

    # Availability
    availability_text = extracted.availability_status or ""
    claim_available = (
        "This property listing indicates that the property (or spaces in the center) is currently available for sale, lease, or sublease."
    )
    add_ins_available = (
        f"The answer mentions availability status text: '{availability_text}'. "
        "On the webpage, look for words such as 'Available', 'Availabilities', 'For Lease', 'For Sale', or active space listings."
    )

    # Kroger anchor tenant
    anchor_name = extracted.anchor_tenant_name or "Kroger"
    claim_anchor = (
        f"Kroger is the anchor tenant of the shopping center on this listing (e.g., labeled as 'Anchor Tenant' or clearly the principal anchor)."
    )
    add_ins_anchor = (
        "Check tenant rosters, site plan legends, or overview sections for Kroger being explicitly identified as the anchor tenant."
    )

    # Total center size range
    total_center_sf_text = extracted.total_center_sqft or ""
    claim_total_size = (
        "The total center size (GLA) reported on this listing lies between 125,000 and 400,000 square feet, inclusive."
        f" If the answer provided a specific GLA ('{total_center_sf_text}'), confirm the page supports that value and it lies within the required range."
    )
    add_ins_total_size = (
        "Look for 'GLA', 'Total center size', or similar. If a specific value is shown on the page, verify that it falls within [125,000, 400,000] SF."
    )

    # Kroger Marketplace and store size range
    kroger_sf_text = extracted.kroger_store_sqft or ""
    store_format = extracted.kroger_store_format or ""
    claim_marketplace_size = (
        "The Kroger anchor store on this listing is a 'Marketplace' format store and its square footage lies between 99,000 and 130,000 square feet, inclusive."
        f" If the answer provided a specific Kroger store size ('{kroger_sf_text}') or format ('{store_format}'), verify that the page supports these details and they satisfy the range and format criteria."
    )
    add_ins_marketplace_size = (
        "Specifically check for the 'Marketplace' label (e.g., 'Kroger Marketplace') and the store's size in SF. "
        "Accept minor formatting variations, but the size must be within [99,000, 130,000] SF."
    )

    # Collect claims and nodes for batch verification
    claims_and_sources: List[tuple[str, Optional[List[str] | str | None], Any, Optional[str]]] = []

    # Listed on official platform – use only the primary listing URL; if missing, simple verify will be used automatically
    claims_and_sources.append((claim_official, primary_url, listed_official_node, add_ins_official))

    # Located in Texas
    claims_and_sources.append((claim_tx, sources_for_general_checks if sources_for_general_checks else primary_url, located_tx_node, add_ins_tx))

    # Availability
    claims_and_sources.append((claim_available, sources_for_general_checks if sources_for_general_checks else primary_url, available_node, add_ins_available))

    # Kroger anchor tenant
    claims_and_sources.append((claim_anchor, sources_for_general_checks if sources_for_general_checks else primary_url, kroger_anchor_node, add_ins_anchor))

    # Total center size
    claims_and_sources.append((claim_total_size, sources_for_general_checks if sources_for_general_checks else primary_url, total_center_size_node, add_ins_total_size))

    # Marketplace store size range
    claims_and_sources.append((claim_marketplace_size, sources_for_general_checks if sources_for_general_checks else primary_url, marketplace_store_node, add_ins_marketplace_size))

    # Execute batch verifications (parallel). If URL is None for some items, the evaluator routes to simple_verify.
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
    """
    Evaluate an answer for the Kroger-anchored community shopping center in Texas task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Requirements evaluated independently
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

    # Extract structured property info
    extracted: PropertyExtraction = await evaluator.extract(
        prompt=prompt_extract_property(),
        template_class=PropertyExtraction,
        extraction_name="property_extraction",
    )

    # Add custom info: normalized numeric interpretations to aid debugging
    norm_total = parse_int_from_text(extracted.total_center_sqft)
    norm_kroger = parse_int_from_text(extracted.kroger_store_sqft)
    norm_occ = parse_percent_to_float(extracted.kroger_occupancy_percent)
    evaluator.add_custom_info(
        {
            "identifier": name_or_address(extracted),
            "listing_url": extracted.listing_url,
            "additional_urls": extracted.additional_urls,
            "parsed_total_center_sqft": norm_total,
            "parsed_kroger_store_sqft": norm_kroger,
            "parsed_kroger_occupancy_percent": norm_occ,
            "claimed_city": extracted.city,
            "claimed_state": extracted.state,
            "claimed_format": extracted.kroger_store_format,
            "claimed_availability": extracted.availability_status,
        },
        info_type="debug",
        info_name="parsed_fields_summary",
    )

    # Build verification tree and run checks
    await build_property_requirements_tree(evaluator, root, extracted)

    # Return final structured summary
    return evaluator.get_summary()