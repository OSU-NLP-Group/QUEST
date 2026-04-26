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
TASK_ID = "beauty_allure_sephora_2025"
TASK_DESCRIPTION = """
Identify four distinct beauty products that simultaneously meet all of the following criteria:

1. Each product must be a winner of Allure's 2025 Best of Beauty Awards (verifiable at https://www.allure.com/best-of-beauty-2025-winners)
2. Each product must be currently available for purchase at Sephora (online or in-store)
3. Each product must hold cruelty-free certification from either Leaping Bunny (Cruelty Free International) or PETA's Beauty Without Bunnies program
4. Each product must meet Clean at Sephora standards (formulated without over 50 restricted ingredients including parabens, phthalates, and certain sulfates)
5. Each product must use sustainable packaging that includes at least one of the following: post-consumer recycled (PCR) materials, refillable design, or FSC-certified paper/cardboard
6. The brand manufacturing each product must hold Certified B Corporation status (verifiable through the official B Corp directory or B Corp Beauty Coalition)
7. Each product must be in the skincare or makeup category (not fragrance, hair care, or body care)
8. Each product must provide transparent ingredient disclosure, with a full ingredient list available on the product page or packaging

For each product, provide: (a) the product name and brand, (b) the specific Allure Best of Beauty award category it won, (c) the cruelty-free certification it holds, (d) the type of sustainable packaging it uses, and (e) reference URLs for verification.
"""

ALLURE_WINNERS_URL = "https://www.allure.com/best-of-beauty-2025-winners"
ALLOWED_PACKAGING_KEYWORDS = [
    "pcr", "post-consumer recycled", "recycled content", "recycled",
    "refillable", "refill", "fsc", "fsc-certified", "forest stewardship council"
]
ALLOWED_CRUELTY_DOMAINS = ["leapingbunny.org", "crueltyfree.peta.org", "peta.org"]
ALLOWED_BCORP_DOMAINS = ["bcorporation.net", "directory.bcorporation.net", "bcorpbeauty.org", "bcorpbeautycoalition.org", "bcorpbeautycoalition.com"]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AwardInfo(BaseModel):
    category: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CertInfo(BaseModel):
    program: Optional[str] = None  # e.g., "Leaping Bunny" or "PETA Beauty Without Bunnies"
    urls: List[str] = Field(default_factory=list)


class PackagingInfo(BaseModel):
    packaging_type: Optional[str] = None  # e.g., "PCR", "refillable", "FSC-certified"
    urls: List[str] = Field(default_factory=list)


class ProductItem(BaseModel):
    product_name: Optional[str] = None
    brand: Optional[str] = None
    allure_award: AwardInfo = Field(default_factory=AwardInfo)
    sephora_url: Optional[str] = None
    cruelty_free_cert: CertInfo = Field(default_factory=CertInfo)
    packaging: PackagingInfo = Field(default_factory=PackagingInfo)
    bcorp_urls: List[str] = Field(default_factory=list)
    category: Optional[str] = None  # "skincare" or "makeup"
    ingredient_list_urls: List[str] = Field(default_factory=list)


class ProductsExtraction(BaseModel):
    products: List[ProductItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_products() -> str:
    return """
    Extract all distinct product entries mentioned in the answer. For each product, return a JSON object with:
    - product_name: The product's name as stated
    - brand: The brand/manufacturer
    - allure_award: 
        - category: The specific Allure 2025 Best of Beauty award category stated for this product
        - urls: All Allure URLs provided for verification (e.g., winners page or product award page). Extract only actual URLs from the answer.
    - sephora_url: The Sephora product page URL provided (must be a valid Sephora URL if present)
    - cruelty_free_cert:
        - program: The certification program name ("Leaping Bunny" or "PETA Beauty Without Bunnies")
        - urls: Verification URL(s) provided that substantiate this certification (e.g., brand listing pages)
    - packaging:
        - packaging_type: The type of sustainable packaging (e.g., "PCR materials", "refillable", "FSC-certified paper")
        - urls: Verification URL(s) supporting the packaging claim (brand site, product page, sustainability page)
    - bcorp_urls: URL(s) provided that verify the brand is a Certified B Corporation (prefer official B Corp directory or B Corp Beauty Coalition site)
    - category: The product category stated (prefer "skincare" or "makeup")
    - ingredient_list_urls: URL(s) to a full ingredient list (could be the Sephora page or brand site). If no dedicated URL is provided but the Sephora page is claimed to have a full list, include the Sephora URL here too.

    GENERAL RULES:
    - Extract only URLs explicitly present in the answer (plain or markdown).
    - Do not invent URLs; if none are provided for a field, return an empty list or null as appropriate.
    - Maintain the ordering of products as in the answer.
    - If any field is missing, set it to null (strings) or [] (arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _has_domain(url: str, domain: str) -> bool:
    return domain in (url or "")


def _urls_have_any_domain(urls: List[str], domains: List[str]) -> bool:
    for u in urls:
        for d in domains:
            if _has_domain(u, d):
                return True
    return False


def _contains_any_keyword(text: Optional[str], keywords: List[str]) -> bool:
    if not text:
        return False
    t = _normalize_text(text)
    return any(k in t for k in keywords)


def _allowed_category(text: Optional[str]) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    allowed = ["skincare", "skin care", "makeup", "make-up"]
    disallowed = ["fragrance", "hair", "body", "body care", "hair care"]
    return any(a in t for a in allowed) and not any(x in t for x in disallowed)


def _safe_first4(products: List[ProductItem]) -> List[ProductItem]:
    # Filter only the first 4 items; pad if fewer
    selected = products[:4]
    while len(selected) < 4:
        selected.append(ProductItem())
    return selected


# --------------------------------------------------------------------------- #
# Verification per product                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_product(evaluator: Evaluator, parent_node, product: ProductItem, index: int) -> None:
    """
    Build verification subtree for a single product based on the rubric.
    """
    prod_node = evaluator.add_parallel(
        id=f"product_{index + 1}",
        desc=f"Product {index + 1} validity against all constraints and required fields.",
        parent=parent_node,
        critical=False  # Non-critical at product level (partial credit allowed across products)
    )

    name_brand_ok = bool(product.product_name and product.product_name.strip()) and bool(product.brand and product.brand.strip())
    evaluator.add_custom_node(
        result=name_brand_ok,
        id=f"p{index + 1}_name_brand",
        desc="Provides product name and brand.",
        parent=prod_node,
        critical=True
    )

    # Allure award URL provided and category stated (existence check)
    allure_url_provided = (product.allure_award and product.allure_award.category and
                           len(product.allure_award.urls) > 0 and
                           any("allure.com" in (u or "") for u in product.allure_award.urls))
    evaluator.add_custom_node(
        result=allure_url_provided,
        id=f"p{index + 1}_allure_award_url_provided",
        desc="Allure award: category stated and an Allure URL for verification is provided.",
        parent=prod_node,
        critical=True
    )

    # Verify Allure award claim with provided Allure URLs (plus official winners page as supportive evidence)
    allure_sources = product.allure_award.urls[:] if product.allure_award.urls else []
    if ALLURE_WINNERS_URL not in allure_sources:
        allure_sources.append(ALLURE_WINNERS_URL)

    allure_leaf = evaluator.add_leaf(
        id=f"p{index + 1}_allure_award",
        desc="Product is verifiably a winner on Allure's 2025 Best of Beauty winners page; states the specific award category won and provides an Allure URL for verification.",
        parent=prod_node,
        critical=True
    )
    allure_claim = f"The product '{product.product_name or ''}' by {product.brand or ''} is listed as a winner of Allure's 2025 Best of Beauty Awards in the category '{product.allure_award.category or ''}'."
    await evaluator.verify(
        claim=allure_claim,
        node=allure_leaf,
        sources=allure_sources,
        additional_instruction="Confirm the exact product and category on Allure's official 2025 Best of Beauty winners pages. Allow reasonable name variations, but the award category must match the 2025 listing."
    )

    # Sephora availability
    sephora_url_ok = bool(product.sephora_url and "sephora.com" in product.sephora_url)
    evaluator.add_custom_node(
        result=sephora_url_ok,
        id=f"p{index + 1}_sephora_url_provided",
        desc="Sephora product URL is provided.",
        parent=prod_node,
        critical=True
    )
    seph_avail_leaf = evaluator.add_leaf(
        id=f"p{index + 1}_sephora_availability",
        desc="Product is currently available for purchase at Sephora (online or in-store) and provides a Sephora URL as evidence.",
        parent=prod_node,
        critical=True
    )
    seph_claim = "This Sephora product page indicates the item is available for purchase now (online or in-store)."
    await evaluator.verify(
        claim=seph_claim,
        node=seph_avail_leaf,
        sources=product.sephora_url,
        additional_instruction="Look for signals like 'Add to Basket', 'In Stock', 'Find in Store', or inventory indicators. If marked 'in-store only', availability counts."
    )

    # Cruelty-free certification
    cruelty_url_ok = _urls_have_any_domain(product.cruelty_free_cert.urls, ALLOWED_CRUELTY_DOMAINS)
    cruelty_program_ok = bool(product.cruelty_free_cert.program and product.cruelty_free_cert.program.strip())
    evaluator.add_custom_node(
        result=cruelty_url_ok and cruelty_program_ok,
        id=f"p{index + 1}_cruelty_free_cert_url_provided",
        desc="Cruelty-free certification URL(s) from Leaping Bunny or PETA provided and program named.",
        parent=prod_node,
        critical=True
    )
    cruelty_leaf = evaluator.add_leaf(
        id=f"p{index + 1}_cruelty_free_cert",
        desc="Product/brand holds cruelty-free certification from either Leaping Bunny or PETA Beauty Without Bunnies; specifies which program and provides a verification URL.",
        parent=prod_node,
        critical=True
    )
    cert_claim = f"The brand '{product.brand or ''}' or the product '{product.product_name or ''}' is certified cruelty-free by {product.cruelty_free_cert.program or ''}."
    await evaluator.verify(
        claim=cert_claim,
        node=cruelty_leaf,
        sources=product.cruelty_free_cert.urls,
        additional_instruction="Verify the brand (or product) listing on the official Leaping Bunny or PETA cruelty-free directories."
    )

    # Clean at Sephora
    clean_leaf = evaluator.add_leaf(
        id=f"p{index + 1}_clean_at_sephora",
        desc="Product meets Clean at Sephora standards (e.g., identified as Clean at Sephora on Sephora) and provides a verification URL.",
        parent=prod_node,
        critical=True
    )
    clean_claim = "This Sephora product page shows the 'Clean at Sephora' badge or explicitly states the product meets Clean at Sephora standards."
    await evaluator.verify(
        claim=clean_claim,
        node=clean_leaf,
        sources=product.sephora_url,
        additional_instruction="Check for the 'Clean at Sephora' badge, icon, or text indicating compliance with Clean standards."
    )

    # Sustainable packaging
    packaging_exists = bool(product.packaging.packaging_type and _contains_any_keyword(product.packaging.packaging_type, ALLOWED_PACKAGING_KEYWORDS))
    packaging_url_ok = len(product.packaging.urls) > 0
    evaluator.add_custom_node(
        result=packaging_exists and packaging_url_ok,
        id=f"p{index + 1}_sustainable_packaging_provided",
        desc="Sustainable packaging type specified (PCR/refillable/FSC) with a verification URL.",
        parent=prod_node,
        critical=True
    )
    packaging_leaf = evaluator.add_leaf(
        id=f"p{index + 1}_sustainable_packaging",
        desc="Packaging includes at least one of: PCR materials, refillable design, or FSC-certified paper/cardboard; specifies which applies and provides a verification URL.",
        parent=prod_node,
        critical=True
    )
    packaging_claim = f"The product '{product.product_name or ''}' uses sustainable packaging: {product.packaging.packaging_type or ''}."
    await evaluator.verify(
        claim=packaging_claim,
        node=packaging_leaf,
        sources=product.packaging.urls,
        additional_instruction="Accept brand sustainability pages or product detail pages as evidence. Confirm that the specified type (PCR/refillable/FSC) is explicitly stated."
    )

    # Brand B Corp
    bcorp_url_ok = _urls_have_any_domain(product.bcorp_urls, ALLOWED_BCORP_DOMAINS)
    evaluator.add_custom_node(
        result=bcorp_url_ok,
        id=f"p{index + 1}_brand_bcorp_url_provided",
        desc="B Corp verification URL provided (official directory or Beauty Coalition).",
        parent=prod_node,
        critical=True
    )
    bcorp_leaf = evaluator.add_leaf(
        id=f"p{index + 1}_brand_bcorp",
        desc="Brand is a Certified B Corporation and provides a B Corp directory (or B Corp Beauty Coalition) URL verifying status.",
        parent=prod_node,
        critical=True
    )
    bcorp_claim = f"The brand '{product.brand or ''}' is a Certified B Corporation."
    await evaluator.verify(
        claim=bcorp_claim,
        node=bcorp_leaf,
        sources=product.bcorp_urls,
        additional_instruction="Verify the brand listing on the official B Corp directory or the B Corp Beauty Coalition site."
    )

    # Allowed category (existence check and verification)
    category_ok = _allowed_category(product.category)
    evaluator.add_custom_node(
        result=category_ok,
        id=f"p{index + 1}_allowed_category_presence",
        desc="Product category is stated as skincare or makeup (not fragrance, hair care, or body care).",
        parent=prod_node,
        critical=True
    )
    category_leaf = evaluator.add_leaf(
        id=f"p{index + 1}_allowed_category",
        desc="Product is in the skincare or makeup category (not fragrance, hair care, or body care).",
        parent=prod_node,
        critical=True
    )
    category_claim = "This product belongs to either the skincare or makeup category (and is not fragrance, hair care, or body care)."
    await evaluator.verify(
        claim=category_claim,
        node=category_leaf,
        sources=product.sephora_url,
        additional_instruction="Check the category breadcrumbs or taxonomy on the Sephora page to confirm it's skincare or makeup."
    )

    # Ingredient transparency
    ingredients_sources = product.ingredient_list_urls[:]
    if product.sephora_url and product.sephora_url not in ingredients_sources:
        ingredients_sources.append(product.sephora_url)
    ingredients_present = len(ingredients_sources) > 0
    evaluator.add_custom_node(
        result=ingredients_present,
        id=f"p{index + 1}_ingredient_transparency_source_provided",
        desc="Ingredient list source (URL) is provided.",
        parent=prod_node,
        critical=True
    )
    ingredients_leaf = evaluator.add_leaf(
        id=f"p{index + 1}_ingredient_transparency",
        desc="A full ingredient list is available on the product page or packaging (transparent ingredient disclosure) and provides a supporting URL or clearly cited source location.",
        parent=prod_node,
        critical=True
    )
    ingredients_claim = "The product page or cited source provides a full ingredient list (transparent disclosure)."
    await evaluator.verify(
        claim=ingredients_claim,
        node=ingredients_leaf,
        sources=ingredients_sources,
        additional_instruction="On the product page, look for a clearly enumerated ingredient list. Summaries are insufficient; a full list must be accessible."
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
    Evaluate the provided answer against the beauty product rubric using Mind2Web2.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Evaluate products independently
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

    # Extract products
    extracted = await evaluator.extract(
        prompt=prompt_extract_products(),
        template_class=ProductsExtraction,
        extraction_name="products_extraction"
    )

    # Record official reference for Allure winners
    evaluator.add_ground_truth({
        "official_allure_winners_url": ALLURE_WINNERS_URL,
        "allowed_packaging_keywords": ALLOWED_PACKAGING_KEYWORDS,
        "allowed_cruelty_domains": ALLOWED_CRUELTY_DOMAINS,
        "allowed_bcorp_domains": ALLOWED_BCORP_DOMAINS
    })

    # Count requirement (critical)
    original_count = len(extracted.products)
    evaluator.add_custom_node(
        result=(original_count == 4),
        id="count_requirement",
        desc="Response lists exactly four products.",
        parent=root,
        critical=True
    )

    # Work with first 4 products, pad if fewer
    first_four = _safe_first4(extracted.products)

    # Distinct products requirement (critical) among the first four
    seen: set = set()
    distinct_ok = True
    for p in first_four:
        key = f"{_normalize_text(p.brand)}::{_normalize_text(p.product_name)}"
        if key in seen and key != "::":
            distinct_ok = False
            break
        seen.add(key)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinct_products",
        desc="All four listed products are distinct (no duplicates).",
        parent=root,
        critical=True
    )

    # Build product verification subtrees
    for i, p in enumerate(first_four):
        await verify_single_product(evaluator, root, p, i)

    return evaluator.get_summary()