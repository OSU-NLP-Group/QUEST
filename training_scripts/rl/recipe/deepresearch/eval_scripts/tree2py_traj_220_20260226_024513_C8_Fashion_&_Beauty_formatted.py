import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_beauty_brands_2024_2026"
TASK_DESCRIPTION = (
    "Identify four currently active celebrity-founded beauty brands that offer skincare products as one of their "
    "product categories and have launched at least one new product between 2024 and 2026. These brands must be "
    "available at major US retailers (such as Sephora, Ulta Beauty, Amazon, Nordstrom, or the brand's own website). "
    "For each of the four brands, provide: (1) The name of the celebrity founder, (2) At least two distinct product "
    "category names that the brand offers (one must be skincare or skin-focused), (3) The specific name and retail "
    "price of one new product that was launched or designated as 'new' between 2024-2026, and (4) At least one major "
    "US retailer where the brand's products are available."
)

MAJOR_US_RETAILER_DOMAINS = [
    "sephora.com",
    "ulta.com",
    "amazon.com",
    "nordstrom.com",
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NewProductInfo(BaseModel):
    name: Optional[str] = None
    price: Optional[str] = None
    launch_info: Optional[str] = None  # e.g., "launched May 2025", "new 2026", "release date 2024-09"
    sources: List[str] = Field(default_factory=list)  # product page, PR/news, retailer listing


class BrandEntry(BaseModel):
    brand_name: Optional[str] = None
    founder_name: Optional[str] = None
    category_names: List[str] = Field(default_factory=list)

    product: Optional[NewProductInfo] = None

    retailer_names: List[str] = Field(default_factory=list)
    retailer_urls: List[str] = Field(default_factory=list)

    brand_sources: List[str] = Field(default_factory=list)     # brand site pages, about pages, shop pages
    founder_sources: List[str] = Field(default_factory=list)   # interviews/news/brand about page mentioning founder
    category_sources: List[str] = Field(default_factory=list)  # pages showing categories (brand or retailer)
    product_sources: List[str] = Field(default_factory=list)   # product page / PR release / retailer listing


class BeautyBrandsExtraction(BaseModel):
    brands: List[BrandEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return """
    Extract information for up to four celebrity-founded beauty brands mentioned in the answer. If more than four brands
    are present, extract only the first four in the order they appear. If fewer than four are present, return fewer entries.

    For each brand, extract the following fields exactly as presented in the answer:

    - brand_name: The brand's name.
    - founder_name: The celebrity founder's name.
    - category_names: A list of at least two distinct product category names the brand offers (e.g., "skincare", "makeup",
      "hair care", "fragrance", "body care", etc.). If fewer are provided, extract what is present.
    - product: An object describing one specific "new" product the brand launched or designated as 'new' between 2024-2026:
        • name: The exact product name.
        • price: The claimed retail price string (e.g., "$28", "USD 34", "£25" if provided; use exactly what the answer says).
        • launch_info: Any textual info indicating launch timeframe or "new" designation between 2024-2026 (e.g., "launched in 2025", 
          "New 2026", "released June 2024").
        • sources: A list of URLs that specifically reference or list this product, such as brand product pages, PR/news posts, or retailer pages.
    - retailer_names: A list of retailer names where the brand is sold (e.g., "Sephora", "Ulta Beauty", "Amazon", "Nordstrom", or
      "Official brand website").
    - retailer_urls: A list of URLs for retailer product listing or brand pages from major US retailers (Sephora, Ulta Beauty, Amazon, Nordstrom)
      or the brand's own official website shop page.
    - brand_sources: Additional URLs about the brand (e.g., official site, shop pages, "About" page, press releases) that can help verify the brand is active.
    - founder_sources: URLs (brand page, credible news/interviews) that support the founder identity.
    - category_sources: URLs (brand site or retailer category pages) that show the brand's categories or a page that clearly indicates the brand
      offers skincare and other categories.
    - product_sources: URLs that support the specific product's details (name, price, and "new" status/timeframe).

    IMPORTANT:
    - Extract only URLs explicitly present in the answer text. Do not invent or infer URLs.
    - Include complete valid URLs. If a URL lacks a protocol, prepend 'http://'.
    - If a field is missing in the answer, set it to null (for single values) or an empty list (for lists).
    - Do not normalize or rewrite names or prices—extract them exactly as shown in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_n_or_pad(brands: List[BrandEntry], n: int = 4) -> List[BrandEntry]:
    result = list(brands[:n])
    while len(result) < n:
        result.append(BrandEntry())
    return result


def _merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _ordinal(idx: int) -> str:
    return ["First", "Second", "Third", "Fourth"][idx] if idx < 4 else f"Brand #{idx + 1}"


# --------------------------------------------------------------------------- #
# Verification for one brand                                                  #
# --------------------------------------------------------------------------- #
async def verify_brand(
        evaluator: Evaluator,
        parent_node,
        brand: BrandEntry,
        brand_index: int,
) -> None:
    brand_label = _ordinal(brand_index)
    brand_name = (brand.brand_name or "").strip()
    founder_name = (brand.founder_name or "").strip()

    # Create brand node (non-critical to allow partial scoring across brands)
    brand_node = evaluator.add_parallel(
        id=f"brand_{brand_index + 1}",
        desc=f"{brand_label} celebrity beauty brand meets all requirements",
        parent=parent_node,
        critical=False
    )

    # Collect source groups
    brand_srcs = brand.brand_sources or []
    founder_srcs = (brand.founder_sources or []) + brand_srcs
    category_srcs = (brand.category_sources or []) + brand_srcs + (brand.retailer_urls or [])
    product_srcs = (brand.product_sources or []) + (brand.product.sources if brand.product else []) + (brand.retailer_urls or [])
    retailer_urls = brand.retailer_urls or []

    # Precondition nodes (non-critical gatekeepers so missing sources skip verifications that require URLs)
    any_brand_level_sources = evaluator.add_custom_node(
        result=bool(brand_srcs or retailer_urls or product_srcs),
        id=f"brand_{brand_index + 1}_brand_level_sources_present",
        desc="At least one brand-level source URL is present (brand site, retailer listing, or product page)",
        parent=brand_node,
        critical=False
    )
    founder_sources_present = evaluator.add_custom_node(
        result=bool(founder_srcs),
        id=f"brand_{brand_index + 1}_founder_sources_present",
        desc="Founder-related source URL(s) present",
        parent=brand_node,
        critical=False
    )
    category_sources_present = evaluator.add_custom_node(
        result=bool(category_srcs),
        id=f"brand_{brand_index + 1}_category_sources_present",
        desc="Category-related source URL(s) present",
        parent=brand_node,
        critical=False
    )
    product_sources_present = evaluator.add_custom_node(
        result=bool(product_srcs),
        id=f"brand_{brand_index + 1}_product_sources_present",
        desc="Product-related source URL(s) present",
        parent=brand_node,
        critical=False
    )
    retailer_urls_present = evaluator.add_custom_node(
        result=bool(retailer_urls),
        id=f"brand_{brand_index + 1}_retailer_urls_present",
        desc="Retailer URL(s) present",
        parent=brand_node,
        critical=False
    )

    # 1) Operational status
    operational_node = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_operational_status",
        desc="The brand is currently active and operational as of February 2026",
        parent=brand_node,
        critical=True
    )
    op_claim = (
        f"As of February 2026, the brand '{brand_name or 'this brand'}' is active and operational, "
        f"with products currently available for purchase or listed on official/retailer pages."
    )
    await evaluator.verify(
        claim=op_claim,
        node=operational_node,
        sources=_merge_sources(brand_srcs, retailer_urls, product_srcs),
        additional_instruction=(
            "Determine whether the brand appears active in 2026 via evidence such as live shop pages, "
            "recent product listings, or 'new' product launches between 2024-2026. If all URLs are invalid or "
            "irrelevant, mark as not supported."
        ),
        extra_prerequisites=[any_brand_level_sources]
    )

    # 2) Founder verification
    founder_node = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_founder",
        desc="The answer correctly identifies the celebrity founder of the brand",
        parent=brand_node,
        critical=True
    )
    founder_claim = (
        f"The brand '{brand_name or 'this brand'}' was founded or created by {founder_name or 'the claimed founder'}."
    )
    await evaluator.verify(
        claim=founder_claim,
        node=founder_node,
        sources=_merge_sources(founder_srcs),
        additional_instruction=(
            "Check that the provided sources explicitly link the celebrity founder to the brand "
            "(phrases like 'founded by', 'created by', 'owned by'). Allow reasonable name variations."
        ),
        extra_prerequisites=[founder_sources_present]
    )

    # 3) Product categories (aggregated)
    categories_node = evaluator.add_parallel(
        id=f"brand_{brand_index + 1}_product_categories",
        desc="The brand's product category information is complete and accurate",
        parent=brand_node,
        critical=True
    )

    # 3.1) Skincare category present (verify with URLs)
    skincare_node = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_skincare_category",
        desc="The brand offers skincare products as one of its product categories",
        parent=categories_node,
        critical=True
    )
    skincare_claim = (
        f"The brand '{brand_name or 'this brand'}' offers skincare products (skin care/skin-focused) as one category."
    )
    await evaluator.verify(
        claim=skincare_claim,
        node=skincare_node,
        sources=_merge_sources(category_srcs),
        additional_instruction=(
            "Confirm that the brand offers skincare (skin care/skin-focused) products. "
            "Evidence can be category pages, product listings, or retailer pages showing 'Skincare' or similar."
        ),
        extra_prerequisites=[category_sources_present]
    )

    # 3.2) Multiple categories (verify with URLs)
    multiple_cats_node = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_multiple_categories",
        desc="The brand offers at least two distinct product categories",
        parent=categories_node,
        critical=True
    )
    cats_display = ", ".join(brand.category_names[:5]) if brand.category_names else "the listed categories"
    multiple_cats_claim = (
        f"The brand '{brand_name or 'this brand'}' offers at least two distinct product categories "
        f"(e.g., {cats_display})."
    )
    await evaluator.verify(
        claim=multiple_cats_claim,
        node=multiple_cats_node,
        sources=_merge_sources(category_srcs),
        additional_instruction=(
            "Verify that there are at least two distinct categories offered by the brand (e.g., skincare and makeup). "
            "Categories may be named slightly differently across pages; allow reasonable variants."
        ),
        extra_prerequisites=[category_sources_present]
    )

    # 3.3) Category names provided (existence check only)
    category_names_node = evaluator.add_custom_node(
        result=(brand.category_names is not None and len([c for c in brand.category_names if c and c.strip()]) >= 2),
        id=f"brand_{brand_index + 1}_category_names",
        desc="At least two specific product category names are provided",
        parent=categories_node,
        critical=True
    )

    # 4) New product (aggregated)
    new_product_node = evaluator.add_parallel(
        id=f"brand_{brand_index + 1}_new_product",
        desc="Information about a new product launched between 2024-2026 is complete and accurate",
        parent=brand_node,
        critical=True
    )

    # 4.1) Specific product name provided (existence check)
    product_name_str = (brand.product.name if brand.product and brand.product.name else "").strip()
    product_name_node = evaluator.add_custom_node(
        result=bool(product_name_str),
        id=f"brand_{brand_index + 1}_new_product_name",
        desc="The specific name of the new product is provided",
        parent=new_product_node,
        critical=True
    )

    # 4.2) Launch timeframe between 2024-2026 (verify with URLs)
    launch_node = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_new_product_launch_timeframe",
        desc="The product was launched or designated as new between 2024-2026",
        parent=new_product_node,
        critical=True
    )
    launch_claim = (
        f"The product '{product_name_str or 'the product'}' was launched or designated 'new' between 2024 and 2026."
    )
    await evaluator.verify(
        claim=launch_claim,
        node=launch_node,
        sources=_merge_sources(product_srcs),
        additional_instruction=(
            "Look for explicit launch dates or 'new' designation indicating the product is new from 2024–2026 "
            "(inclusive). Accept retailer 'NEW' badges or PR/brand posts within this window."
        ),
        extra_prerequisites=[product_sources_present]
    )

    # 4.3) Retail price provided and verifiable (verify with URLs)
    price_str = (brand.product.price if brand.product and brand.product.price else "").strip()
    price_node = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_new_product_price",
        desc="The retail price of the new product is provided and verifiable",
        parent=new_product_node,
        critical=True
    )
    price_claim = (
        f"The retail price of '{product_name_str or 'the product'}' is {price_str or 'the claimed price'}."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_node,
        sources=_merge_sources(product_srcs, retailer_urls),
        additional_instruction=(
            "Verify the price on the product or retailer page. Allow reasonable variants due to sizes/sets "
            "(judge correctness if the claimed price matches any listed configuration)."
        ),
        extra_prerequisites=[product_sources_present]
    )

    # 5) Retailer availability at major US retailer or official brand site (verify with URLs)
    retailer_node = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_retailer_availability",
        desc="At least one major US retailer where the brand is available is correctly identified",
        parent=brand_node,
        critical=True
    )
    retailer_names_str = ", ".join(brand.retailer_names[:5]) if brand.retailer_names else "the listed retailer(s)"
    retailer_claim = (
        f"The brand '{brand_name or 'this brand'}' has products available at {retailer_names_str}, "
        "which qualifies as major US retailers (Sephora, Ulta Beauty, Amazon, Nordstrom) or the official brand website."
    )
    await evaluator.verify(
        claim=retailer_claim,
        node=retailer_node,
        sources=_merge_sources(retailer_urls),
        additional_instruction=(
            "Confirm availability via retailer URLs or the official brand shop URL. Major US retailers include "
            "Sephora (sephora.com), Ulta Beauty (ulta.com), Amazon (amazon.com), Nordstrom (nordstrom.com). "
            "The official brand website also qualifies."
        ),
        extra_prerequisites=[retailer_urls_present]
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
    Evaluate an answer for the celebrity beauty brands 2024–2026 task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Brands evaluated independently
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

    # NOTE: We intentionally set the root node as non-critical during initialize(). If the rubric requires all four
    # brands strictly, aggregation will still reflect failures of critical sub-criteria; partial credit is allowed.

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BeautyBrandsExtraction,
        extraction_name="beauty_brands_extraction"
    )

    # Normalize to exactly four brand entries
    brands = _first_n_or_pad(extracted.brands, n=4)

    # Build and verify each brand
    for i, brand in enumerate(brands):
        await verify_brand(evaluator, root, brand, i)

    # Return structured summary
    return evaluator.get_summary()