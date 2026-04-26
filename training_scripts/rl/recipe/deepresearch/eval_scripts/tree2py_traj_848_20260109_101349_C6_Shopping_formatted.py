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
TASK_ID = "refrigerator_houston_solution"
TASK_DESCRIPTION = (
    "I'm shopping for a new refrigerator for my family of four in Houston, Texas, and need help finding the right option. "
    "I'm specifically looking for a counter-depth refrigerator (with depth between 24-30 inches excluding handles) that has at least 20 cubic feet of capacity and is ENERGY STAR certified. "
    "The height cannot exceed 72 inches due to my kitchen ceiling clearance. "
    "I want to purchase from one of the three major national retailers: Best Buy, Home Depot, or Lowe's, and the retailer must have a physical store location in Houston, Texas. "
    "Additionally, I require the retailer to offer delivery service to the Houston area, installation service for the refrigerator (either included or as a paid add-on), and old appliance haul-away service (either included or as a paid add-on). "
    "Please identify a specific refrigerator model from one of these retailers that meets all these requirements, and provide documentation showing: "
    "(1) the retailer's Houston store location, "
    "(2) the product listing with availability status, "
    "(3) the product's dimensions (depth and height), "
    "(4) the product's capacity in cubic feet, "
    "(5) the product's ENERGY STAR certification status, "
    "(6) the retailer's delivery service policy, "
    "(7) the retailer's installation service information, and "
    "(8) the retailer's haul-away service policy."
)

ALLOWED_RETAILERS = ["Best Buy", "Home Depot", "Lowe's"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RetailerInfo(BaseModel):
    name: Optional[str] = None
    store_location_url: Optional[str] = None  # URL that shows a Houston, TX store location
    store_location_address_text: Optional[str] = None  # Address text if present in the answer


class ServiceInfo(BaseModel):
    delivery_policy_url: Optional[str] = None
    installation_policy_url: Optional[str] = None
    haul_away_policy_url: Optional[str] = None


class ProductSpecs(BaseModel):
    depth_excluding_handles: Optional[str] = None  # e.g., "28.5 in"
    height: Optional[str] = None  # e.g., "70 in"
    capacity_cuft: Optional[str] = None  # e.g., "22.1 cu. ft."
    energy_star_status_text: Optional[str] = None  # e.g., "ENERGY STAR certified", "Yes"
    depth_doc_url: Optional[str] = None
    height_doc_url: Optional[str] = None
    capacity_doc_url: Optional[str] = None
    energy_star_doc_url: Optional[str] = None


class ProductInfo(BaseModel):
    name_or_model: Optional[str] = None
    listing_url: Optional[str] = None
    availability_text: Optional[str] = None
    counter_depth_label: Optional[str] = None  # Any phrase like "Counter-Depth" if provided in answer
    specs: Optional[ProductSpecs] = None


class RefrigeratorSolutionExtraction(BaseModel):
    retailer: Optional[RetailerInfo] = None
    product: Optional[ProductInfo] = None
    services: Optional[ServiceInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_solution() -> str:
    return """
    Extract the specific retailer, product, and service information exactly as presented in the answer. Do not infer or invent anything not explicitly stated.

    Required JSON structure and fields:
    {
      "retailer": {
        "name": string | null,                      // The chosen retailer's name exactly as written (e.g., "Best Buy", "Home Depot", or "Lowe's")
        "store_location_url": string | null,        // A URL that shows a physical store location in Houston, Texas for the chosen retailer (store page or store locator page that clearly shows "Houston, TX")
        "store_location_address_text": string | null // The address text if provided in the answer (optional)
      },
      "product": {
        "name_or_model": string | null,             // The refrigerator's name or model identifier as written in the answer
        "listing_url": string | null,               // The URL to the product listing page on the retailer's website
        "availability_text": string | null,         // The availability or status text as written (e.g., "In stock", "Available for delivery", "Out of stock", etc.)
        "counter_depth_label": string | null,       // Any text in the answer that indicates the product is counter-depth (e.g., "Counter-Depth", "counter depth design")
        "specs": {
          "depth_excluding_handles": string | null,   // Depth excluding handles exactly as written (e.g., "28.5 in")
          "height": string | null,                    // Height exactly as written (e.g., "69.75 in")
          "capacity_cuft": string | null,             // Capacity in cubic feet as written (e.g., "22.1 cu. ft.")
          "energy_star_status_text": string | null,   // ENERGY STAR status text as written (e.g., "ENERGY STAR certified")
          "depth_doc_url": string | null,             // URL that shows depth (excluding handles). If not provided, return null (do NOT invent).
          "height_doc_url": string | null,            // URL that shows height. If not provided, return null.
          "capacity_doc_url": string | null,          // URL that shows capacity. If not provided, return null.
          "energy_star_doc_url": string | null        // URL that shows ENERGY STAR certification. If not provided, return null.
        }
      },
      "services": {
        "delivery_policy_url": string | null,       // URL to retailer delivery policy/info page
        "installation_policy_url": string | null,   // URL to retailer installation service info page
        "haul_away_policy_url": string | null       // URL to retailer haul-away/removal service info page
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer.
    - For any URL field, return the exact URL string as shown (support plain URLs or markdown links).
    - If some info is missing from the answer, set that field to null.
    - Do not parse numbers; keep values as strings exactly as shown (e.g., "28.5 in", "22 cu ft").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _present_url_list(*urls: Optional[str]) -> List[str]:
    """Return a list of non-empty URLs."""
    result: List[str] = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            result.append(u.strip())
    return result


def _safe_str(v: Optional[str]) -> str:
    return v.strip() if isinstance(v, str) else ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_retailer_solution(evaluator: Evaluator, parent_node, data: RefrigeratorSolutionExtraction) -> None:
    """
    Build and verify the 'retailer_solution' subtree:
    - Retailer must be Best Buy, Home Depot, or Lowe's.
    - Must have a Houston, Texas physical store location (URL reference).
    """
    retailer_node = evaluator.add_parallel(
        id="retailer_solution",
        desc="Identify and verify a qualifying retailer with Houston location",
        parent=parent_node,
        critical=True
    )

    retailer_name = _safe_str(data.retailer.name if data.retailer else None)
    store_location_url = _safe_str(data.retailer.store_location_url if data.retailer else None)

    # Leaf: retailer_type
    retailer_type_leaf = evaluator.add_leaf(
        id="retailer_type",
        desc="Retailer must be Best Buy, Home Depot, or Lowe's",
        parent=retailer_node,
        critical=True
    )
    retailer_type_claim = (
        f"The chosen retailer '{retailer_name}' is one of the following: Best Buy, Home Depot, or Lowe's. "
        "Minor name variants should be considered equivalent (e.g., 'The Home Depot' equals 'Home Depot')."
    )
    await evaluator.verify(
        claim=retailer_type_claim,
        node=retailer_type_leaf,
        additional_instruction="Accept minor variations like 'The Home Depot' for 'Home Depot', casing differences, or apostrophe variations for Lowe's."
    )

    # Leaf: houston_location (URL-based verification)
    houston_location_leaf = evaluator.add_leaf(
        id="houston_location",
        desc="Retailer must have a physical store in Houston, Texas",
        parent=retailer_node,
        critical=True
    )
    houston_location_claim = (
        f"This URL is a {retailer_name if retailer_name else 'retailer'} store page or locator entry that clearly indicates a physical store located in Houston, Texas."
    )
    await evaluator.verify(
        claim=houston_location_claim,
        node=houston_location_leaf,
        sources=store_location_url if store_location_url else None,
        additional_instruction="Look for 'Houston, TX' or a Houston address on the page. Store pages or locator entries that explicitly say Houston are acceptable."
    )

    # Leaf: retailer_location_reference (existence of a URL reference)
    retailer_location_ref_leaf = evaluator.add_custom_node(
        result=bool(store_location_url),
        id="retailer_location_reference",
        desc="Provide URL reference verifying Houston store location",
        parent=retailer_node,
        critical=True
    )


async def verify_product_solution(evaluator: Evaluator, parent_node, data: RefrigeratorSolutionExtraction) -> None:
    """
    Build and verify the 'product_solution' subtree:
    Sequential:
      1) product_identification (parallel, critical)
      2) specifications_verification (parallel, critical)
    """
    product_solution_node = evaluator.add_sequential(
        id="product_solution",
        desc="Identify and verify a qualifying counter-depth refrigerator from the chosen retailer",
        parent=parent_node,
        critical=True
    )

    product = data.product or ProductInfo()
    specs = product.specs or ProductSpecs()
    listing_url = _safe_str(product.listing_url)
    product_name = _safe_str(product.name_or_model)

    # 1) Product identification
    product_ident_node = evaluator.add_parallel(
        id="product_identification",
        desc="Identify a specific counter-depth refrigerator model and show listing/availability",
        parent=product_solution_node,
        critical=True
    )

    # 1.a Product Type: counter-depth
    product_type_leaf = evaluator.add_leaf(
        id="product_type",
        desc="Product must be classified as a counter-depth refrigerator",
        parent=product_ident_node,
        critical=True
    )
    product_type_claim = (
        "This product page describes the refrigerator as 'counter-depth' or equivalent phrasing indicating a counter-depth design."
    )
    await evaluator.verify(
        claim=product_type_claim,
        node=product_type_leaf,
        sources=listing_url if listing_url else None,
        additional_instruction="Look for terms like 'Counter-Depth', 'counter depth', or 'counter-depth design' in the product specs or features."
    )

    # 1.b Product Availability
    product_availability_leaf = evaluator.add_leaf(
        id="product_availability",
        desc="Product must be currently available for purchase (in stock or available to order)",
        parent=product_ident_node,
        critical=True
    )
    product_availability_claim = (
        "This product listing indicates availability for purchase (e.g., in stock, delivery available, or available to order)."
    )
    await evaluator.verify(
        claim=product_availability_claim,
        node=product_availability_leaf,
        sources=listing_url if listing_url else None,
        additional_instruction="Look for status indicators like 'In Stock', 'Available for delivery', 'Get it by', or 'Available to order'."
    )

    # 1.c Product Listing Reference
    product_listing_ref_leaf = evaluator.add_leaf(
        id="product_listing_reference",
        desc="Provide URL reference to product listing showing availability status",
        parent=product_ident_node,
        critical=True
    )
    if product_name:
        listing_ref_claim = f"This URL is a product listing page for refrigerator model '{product_name}' and shows an availability/status indicator."
    else:
        listing_ref_claim = "This URL is a product listing page for a refrigerator and shows an availability/status indicator."
    await evaluator.verify(
        claim=listing_ref_claim,
        node=product_listing_ref_leaf,
        sources=listing_url if listing_url else None,
        additional_instruction="Accept typical retailer product pages that include title, specs, price, and availability/fulfillment info."
    )

    # 2) Specifications verification (parallel, critical)
    specs_node = evaluator.add_parallel(
        id="specifications_verification",
        desc="Verify product meets all specification requirements (with documentation)",
        parent=product_solution_node,
        critical=True
    )

    # Helper URL sets for spec checks (prefer dedicated doc URL; fall back to listing)
    depth_urls = _present_url_list(specs.depth_doc_url, listing_url)
    height_urls = _present_url_list(specs.height_doc_url, listing_url)
    capacity_urls = _present_url_list(specs.capacity_doc_url, listing_url)
    energy_urls = _present_url_list(specs.energy_star_doc_url, listing_url)

    # 2.a Depth
    depth_group = evaluator.add_parallel(
        id="depth_specification",
        desc="Verify and document depth requirement",
        parent=specs_node,
        critical=True
    )

    depth_measure_leaf = evaluator.add_leaf(
        id="depth_measurement",
        desc="Refrigerator depth (excluding handles) must be between 24 and 30 inches",
        parent=depth_group,
        critical=True
    )
    depth_measure_claim = (
        "The refrigerator's depth without handles (i.e., excluding handles) is between 24 and 30 inches."
    )
    await evaluator.verify(
        claim=depth_measure_claim,
        node=depth_measure_leaf,
        sources=depth_urls if depth_urls else None,
        additional_instruction="Look for spec labels like 'Depth (Without Handles)', 'Depth without handles', or similar. Confirm the number falls within 24–30 inches."
    )

    depth_doc_ref_leaf = evaluator.add_leaf(
        id="depth_documentation_reference",
        desc="Provide URL showing depth specification (excluding handles)",
        parent=depth_group,
        critical=True
    )
    depth_doc_claim = "This page explicitly provides the refrigerator's 'Depth without handles' (or equivalent phrasing)."
    await evaluator.verify(
        claim=depth_doc_claim,
        node=depth_doc_ref_leaf,
        sources=(specs.depth_doc_url if specs.depth_doc_url else (listing_url if listing_url else None)),
        additional_instruction="The page should show the depth excluding handles figure. Accept synonymous labels."
    )

    # 2.b Height
    height_group = evaluator.add_parallel(
        id="height_specification",
        desc="Verify and document height requirement",
        parent=specs_node,
        critical=True
    )

    height_measure_leaf = evaluator.add_leaf(
        id="height_measurement",
        desc="Refrigerator height must not exceed 72 inches",
        parent=height_group,
        critical=True
    )
    height_measure_claim = "The refrigerator's overall height is 72 inches or less."
    await evaluator.verify(
        claim=height_measure_claim,
        node=height_measure_leaf,
        sources=height_urls if height_urls else None,
        additional_instruction="Look for spec labels like 'Height', 'Height to Top of Hinge', or 'Height to Top of Case'. If multiple heights appear, consider the maximum overall height."
    )

    height_doc_ref_leaf = evaluator.add_leaf(
        id="height_documentation_reference",
        desc="Provide URL showing height specification",
        parent=height_group,
        critical=True
    )
    height_doc_claim = "This page explicitly provides the refrigerator's height specification."
    await evaluator.verify(
        claim=height_doc_claim,
        node=height_doc_ref_leaf,
        sources=(specs.height_doc_url if specs.height_doc_url else (listing_url if listing_url else None)),
        additional_instruction="The page should display a numeric height measurement for the product."
    )

    # 2.c Capacity
    capacity_group = evaluator.add_parallel(
        id="capacity_specification",
        desc="Verify and document capacity requirement",
        parent=specs_node,
        critical=True
    )

    capacity_measure_leaf = evaluator.add_leaf(
        id="capacity_measurement",
        desc="Refrigerator must have a minimum capacity of 20 cubic feet",
        parent=capacity_group,
        critical=True
    )
    capacity_measure_claim = "The total capacity of the refrigerator is at least 20 cubic feet."
    await evaluator.verify(
        claim=capacity_measure_claim,
        node=capacity_measure_leaf,
        sources=capacity_urls if capacity_urls else None,
        additional_instruction="Look for total capacity measured in cubic feet (e.g., 'Total Capacity', 'Capacity (cu. ft.)'). If multiple capacities are listed (fresh food + freezer), ensure the total is ≥ 20 cu. ft."
    )

    capacity_doc_ref_leaf = evaluator.add_leaf(
        id="capacity_documentation_reference",
        desc="Provide URL showing capacity specification",
        parent=capacity_group,
        critical=True
    )
    capacity_doc_claim = "This page explicitly provides the refrigerator's capacity in cubic feet."
    await evaluator.verify(
        claim=capacity_doc_claim,
        node=capacity_doc_ref_leaf,
        sources=(specs.capacity_doc_url if specs.capacity_doc_url else (listing_url if listing_url else None)),
        additional_instruction="The page should display capacity in cubic feet (cu. ft.)."
    )

    # 2.d ENERGY STAR
    energy_group = evaluator.add_parallel(
        id="energy_efficiency_specification",
        desc="Verify and document ENERGY STAR certification requirement",
        parent=specs_node,
        critical=True
    )

    energy_star_leaf = evaluator.add_leaf(
        id="energy_star_certification",
        desc="Refrigerator must be ENERGY STAR certified",
        parent=energy_group,
        critical=True
    )
    energy_star_claim = "The refrigerator is ENERGY STAR certified."
    await evaluator.verify(
        claim=energy_star_claim,
        node=energy_star_leaf,
        sources=energy_urls if energy_urls else None,
        additional_instruction="Look for 'ENERGY STAR certified' badge or text in specifications/overview."
    )

    energy_doc_ref_leaf = evaluator.add_leaf(
        id="energy_star_documentation_reference",
        desc="Provide URL or documentation showing ENERGY STAR certification status",
        parent=energy_group,
        critical=True
    )
    energy_doc_claim = "This page explicitly indicates ENERGY STAR certification for the product."
    await evaluator.verify(
        claim=energy_doc_claim,
        node=energy_doc_ref_leaf,
        sources=(specs.energy_star_doc_url if specs.energy_star_doc_url else (listing_url if listing_url else None)),
        additional_instruction="The page should include 'ENERGY STAR' or an ENERGY STAR certification indicator."
    )


async def verify_service_solution(evaluator: Evaluator, parent_node, data: RefrigeratorSolutionExtraction) -> None:
    """
    Build and verify the 'service_solution' subtree:
    - Delivery service availability and policy reference
    - Installation service availability and policy reference
    - Haul-away service availability and policy reference
    """
    service_node = evaluator.add_parallel(
        id="service_solution",
        desc="Verify all required appliance services are available from the retailer (with policies documented)",
        parent=parent_node,
        critical=True
    )

    services = data.services or ServiceInfo()
    store_location_url = _safe_str(data.retailer.store_location_url if data.retailer else None)  # may help demonstrate local presence

    # Delivery
    delivery_group = evaluator.add_parallel(
        id="delivery_service",
        desc="Verify refrigerator delivery service to Houston area",
        parent=service_node,
        critical=True
    )
    delivery_avail_leaf = evaluator.add_leaf(
        id="delivery_service_availability",
        desc="Retailer must offer delivery services for refrigerators to the Houston, Texas area",
        parent=delivery_group,
        critical=True
    )
    delivery_sources = _present_url_list(services.delivery_policy_url)
    delivery_claim = (
        "The retailer offers delivery service for refrigerators. "
        "It is acceptable if the page describes major appliance delivery generally; explicit 'Houston' mention is not required."
    )
    await evaluator.verify(
        claim=delivery_claim,
        node=delivery_avail_leaf,
        sources=delivery_sources if delivery_sources else None,
        additional_instruction="A general retailer delivery policy page for appliances suffices. The separate store location evidence demonstrates presence in Houston."
    )

    delivery_ref_leaf = evaluator.add_custom_node(
        result=bool(_safe_str(services.delivery_policy_url)),
        id="delivery_policy_reference",
        desc="Provide URL reference to retailer's delivery service policy",
        parent=delivery_group,
        critical=True
    )

    # Installation
    installation_group = evaluator.add_parallel(
        id="installation_service",
        desc="Verify refrigerator installation service availability",
        parent=service_node,
        critical=True
    )
    installation_avail_leaf = evaluator.add_leaf(
        id="installation_service_availability",
        desc="Retailer must offer installation services for refrigerators (either included or as a paid add-on option)",
        parent=installation_group,
        critical=True
    )
    installation_sources = _present_url_list(services.installation_policy_url)
    installation_claim = (
        "The retailer offers installation services for refrigerators (either included or available as a paid add-on)."
    )
    await evaluator.verify(
        claim=installation_claim,
        node=installation_avail_leaf,
        sources=installation_sources if installation_sources else None,
        additional_instruction="Look for 'installation', 'install services', or 'professional installation' for refrigerators or major appliances."
    )

    installation_ref_leaf = evaluator.add_custom_node(
        result=bool(_safe_str(services.installation_policy_url)),
        id="installation_policy_reference",
        desc="Provide URL reference to retailer's installation service information",
        parent=installation_group,
        critical=True
    )

    # Haul-away
    haul_group = evaluator.add_parallel(
        id="haul_away_service",
        desc="Verify old appliance haul-away service availability",
        parent=service_node,
        critical=True
    )
    haul_avail_leaf = evaluator.add_leaf(
        id="haul_away_service_availability",
        desc="Retailer must offer old appliance haul-away and removal services (either included or as a paid add-on option)",
        parent=haul_group,
        critical=True
    )
    haul_sources = _present_url_list(services.haul_away_policy_url)
    haul_claim = (
        "The retailer offers old appliance haul-away or removal service for refrigerators (either included or available as a paid add-on)."
    )
    await evaluator.verify(
        claim=haul_claim,
        node=haul_avail_leaf,
        sources=haul_sources if haul_sources else None,
        additional_instruction="Look for 'haul-away', 'haul away', 'old appliance removal', or similar phrasing in the service/policy page."
    )

    haul_ref_leaf = evaluator.add_custom_node(
        result=bool(_safe_str(services.haul_away_policy_url)),
        id="haul_away_policy_reference",
        desc="Provide URL reference to retailer's haul-away service policy",
        parent=haul_group,
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
    Evaluate an answer for the Houston refrigerator purchase task.
    """
    # Initialize evaluator with a non-critical root; we will add a critical sequential aggregator under it
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root can be parallel; we'll add the actual sequential critical node beneath
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

    # Extract structured information from the answer
    extraction: RefrigeratorSolutionExtraction = await evaluator.extract(
        prompt=prompt_extract_solution(),
        template_class=RefrigeratorSolutionExtraction,
        extraction_name="refrigerator_solution_extraction"
    )

    # Build the top-level sequential, critical node representing the entire solution
    overall = evaluator.add_sequential(
        id="complete_solution",
        desc="Complete refrigerator purchase solution meeting all requirements for a Houston, Texas customer",
        parent=root,
        critical=True
    )

    # Subtree 1: Retailer solution (parallel, critical)
    await verify_retailer_solution(evaluator, overall, extraction)

    # Subtree 2: Product solution (sequential, critical)
    await verify_product_solution(evaluator, overall, extraction)

    # Subtree 3: Service solution (parallel, critical)
    await verify_service_solution(evaluator, overall, extraction)

    # Return the structured evaluation summary
    return evaluator.get_summary()