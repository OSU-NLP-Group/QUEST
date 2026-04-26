import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "food_multi_cert_nationwide"
TASK_DESCRIPTION = (
    "Find three food products available for purchase nationwide in the United States. Each product must meet the following requirements:\n\n"
    "1. Multiple Certifications: Each product must hold at least three of the following four certifications: USDA Organic; Non-GMO Project Verified; GFCO Gluten-Free Certified; Certified Humane.\n"
    "2. Different Categories: The three products must each be from a different food product category (e.g., eggs, cereal, dairy, meat, etc.).\n"
    "3. Nationwide Availability: Each product must be available for purchase nationwide, not limited to a single state or specific region.\n\n"
    "For each of the three products, provide brand name, specific product name, product category, all certifications the product holds (from the four listed above), and at least one reference URL that verifies the product's certifications and nationwide availability."
)

ALLOWED_CERTIFICATIONS = [
    "USDA Organic",
    "Non-GMO Project Verified",
    "GFCO Gluten-Free Certified",
    "Certified Humane",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Product(BaseModel):
    brand: Optional[str] = None
    product_name: Optional[str] = None
    category: Optional[str] = None
    certifications: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class ProductsExtraction(BaseModel):
    products: List[Product] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_products() -> str:
    return """
    Extract up to the first three distinct food products described in the answer that meet the task requirements.
    For each product, extract these fields:
    - brand: Brand name (string)
    - product_name: Specific product name (string)
    - category: Product category (string), e.g., "eggs", "cereal", "dairy", "meat", "snacks", etc.
    - certifications: List all certifications the product is claimed to have, but ONLY choose from this canonical list:
        • "USDA Organic"
        • "Non-GMO Project Verified"
        • "GFCO Gluten-Free Certified"
        • "Certified Humane"
      Normalize common aliases to these exact canonical names. If unsure, omit.
    - reference_urls: A list of URL(s) that the answer provides to support this product’s certifications and nationwide availability.
      Extract only valid URLs explicitly present in the answer. Include both brand/product pages and official certification directory links if they are provided.

    Rules:
    - Do not invent information; extract only what appears in the answer.
    - If a field is missing, set it to null for strings and [] for lists.
    - Return an object with a single field 'products' which is an array of up to three product objects in the order they appear.
    """


# --------------------------------------------------------------------------- #
# Helper normalization and validation                                         #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def normalize_certification(cert: Optional[str]) -> Optional[str]:
    if not cert:
        return None
    s = cert.strip().lower()
    s_no_dash = s.replace("-", " ").replace("–", " ").replace("—", " ")
    s_no_punct = "".join(ch for ch in s_no_dash if ch.isalnum() or ch.isspace())
    toks = s_no_punct.split()

    # USDA Organic
    if ("usda" in toks and "organic" in toks) or "usda organic" in s or "usda certified organic" in s:
        return "USDA Organic"

    # Non-GMO Project Verified
    if ("non" in toks and "gmo" in toks and "project" in toks) or "non gmo project verified" in s_no_punct:
        return "Non-GMO Project Verified"

    # GFCO Gluten-Free Certified
    if (
        "gfco" in toks
        or ("gluten" in toks and "free" in toks and "certification" in toks and "organization" in toks)
        or "certified gluten free" in s_no_punct
    ):
        return "GFCO Gluten-Free Certified"

    # Certified Humane
    if "certified humane" in s_no_punct:
        return "Certified Humane"

    return None


def canonicalize_certs(certs: List[str]) -> List[str]:
    result = []
    seen = set()
    for c in certs:
        canon = normalize_certification(c)
        if canon and canon in ALLOWED_CERTIFICATIONS and canon not in seen:
            result.append(canon)
            seen.add(canon)
    return result


def canonicalize_category(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    return cat.strip().lower()


# --------------------------------------------------------------------------- #
# Per-product verification                                                    #
# --------------------------------------------------------------------------- #
async def verify_product(
    evaluator: Evaluator,
    parent_node,
    product: Product,
    index: int,
    prior_categories: List[Optional[str]],
) -> None:
    """
    Build verification subtree for one product.
    """
    pid = index + 1
    prod_node = evaluator.add_parallel(
        id=f"product_{pid}",
        desc=f"Product #{pid} verification (must satisfy identification, category uniqueness, >=3 certifications, nationwide availability, and references)",
        parent=parent_node,
        critical=False,  # Each product contributes partial credit independently
    )

    brand = (product.brand or "").strip()
    pname = (product.product_name or "").strip()
    category_raw = product.category or ""
    category_canon = canonicalize_category(category_raw)

    # Identification (critical): brand and product_name both present
    id_ok = bool(brand) and bool(pname)
    evaluator.add_custom_node(
        result=id_ok,
        id=f"product_{pid}_identification",
        desc=f"Product is clearly identified with both brand name and specific product name",
        parent=prod_node,
        critical=True,
    )

    # Category (critical): present; and must be different from earlier products as specified
    different_from_prior = True
    if index == 1:  # product #2 must differ from product #1
        different_from_prior = (
            category_canon is not None and category_canon != canonicalize_category(prior_categories[0])
        )
    elif index == 2:  # product #3 must differ from product #1 and #2
        different_from_prior = (
            category_canon is not None
            and category_canon != canonicalize_category(prior_categories[0])
            and category_canon != canonicalize_category(prior_categories[1])
        )

    cat_ok = bool(category_canon) and different_from_prior
    evaluator.add_custom_node(
        result=cat_ok,
        id=f"product_{pid}_category",
        desc=(
            "Product category is specified and is a food product"
            if index == 0
            else (
                "Product category is specified and is a food product, different from Product 1's category"
                if index == 1
                else "Product category is specified and is a food product, different from Product 1's and Product 2's categories"
            )
        ),
        parent=prod_node,
        critical=True,
    )

    # References (critical): at least one valid URL provided
    valid_refs = [u for u in (product.reference_urls or []) if is_valid_url(u)]
    has_valid_ref = len(valid_refs) > 0
    ref_node = evaluator.add_custom_node(
        result=has_valid_ref,
        id=f"product_{pid}_reference",
        desc="At least one valid reference URL is provided that supports the product's certifications and availability",
        parent=prod_node,
        critical=True,
    )

    # Certifications (critical): need sequential checks: count >=3, then verification
    certs_node = evaluator.add_sequential(
        id=f"product_{pid}_certifications",
        desc="Product holds at least three of the specified certifications (USDA Organic, Non-GMO Project Verified, GFCO Gluten-Free, Certified Humane)",
        parent=prod_node,
        critical=True,
    )

    normalized_certs = canonicalize_certs(product.certifications or [])
    cert_count_ok = len(normalized_certs) >= 3
    evaluator.add_custom_node(
        result=cert_count_ok,
        id=f"product_{pid}_cert_count",
        desc="At least three distinct certifications are claimed",
        parent=certs_node,
        critical=True,
    )

    # Verify certifications individually with standalone verifications, then aggregate into one custom node
    cert_verified_count = 0
    per_cert_results: List[Tuple[str, bool]] = []

    if cert_count_ok and has_valid_ref:
        # Prepare per-cert verification tasks
        tasks = []
        for cert in normalized_certs:
            if cert == "USDA Organic":
                claim = f"The product '{brand} {pname}' is USDA Organic certified."
                add_ins = (
                    "Confirm the product is USDA Organic certified. Accept exact wording like 'USDA Organic', "
                    "'USDA Certified Organic', or clear display of the USDA Organic seal on the page. "
                    "Reject vague mentions without product-specific evidence."
                )
            elif cert == "Non-GMO Project Verified":
                claim = f"The product '{brand} {pname}' is Non-GMO Project Verified."
                add_ins = (
                    "Confirm Non-GMO Project Verified status for this specific product. "
                    "Look for the Non-GMO Project butterfly seal or explicit 'Non-GMO Project Verified' text."
                )
            elif cert == "GFCO Gluten-Free Certified":
                claim = f"The product '{brand} {pname}' is Certified Gluten-Free by GFCO (Gluten-Free Certification Organization)."
                add_ins = (
                    "Confirm GFCO certification for this product. Accept phrases like 'Certified Gluten-Free (GFCO)' "
                    "or GFCO's certification mark. Distinguish from generic 'gluten free' claims."
                )
            else:  # Certified Humane
                claim = f"The product '{brand} {pname}' is Certified Humane."
                add_ins = (
                    "Confirm Certified Humane status for this product (Certified Humane Raised and Handled). "
                    "Look for the Certified Humane logo or explicit statement."
                )

            # Standalone verification (not tied to a leaf node) for each certification
            tasks.append(
                evaluator.verify(
                    claim=claim,
                    node=None,
                    sources=valid_refs,
                    additional_instruction=add_ins,
                )
            )

        cert_results = await asyncio.gather(*tasks, return_exceptions=True)
        for cert, res in zip(normalized_certs, cert_results):
            ok = bool(res) if isinstance(res, bool) else False
            per_cert_results.append((cert, ok))
            if ok:
                cert_verified_count += 1

    # Aggregate certification verification into a single custom leaf under the sequential node
    # Pass if at least 3 certifications are successfully verified from the provided URLs
    certs_verified_ok = cert_verified_count >= 3
    evaluator.add_custom_node(
        result=certs_verified_ok,
        id=f"product_{pid}_cert_verification",
        desc="Each claimed certification is verifiable through official certification databases or the provided reference URL",
        parent=certs_node,
        critical=True,
    )

    # Nationwide availability (critical): verify using provided URLs; depend on reference presence
    nationwide_leaf = evaluator.add_leaf(
        id=f"product_{pid}_nationwide",
        desc="Product is available for purchase nationwide, not limited to a single state or region",
        parent=prod_node,
        critical=True,
    )
    nationwide_claim = (
        f"The product '{brand} {pname}' is sold or can be purchased nationwide across the United States "
        f"(not restricted to a single state or region)."
    )
    await evaluator.verify(
        claim=nationwide_claim,
        node=nationwide_leaf,
        sources=valid_refs if has_valid_ref else None,
        additional_instruction=(
            "Look for clear evidence of nationwide availability: shipping across the U.S.; phrases like 'available nationwide'; "
            "broad store locator coverage; or presence at major national retailers (e.g., Walmart, Target, Whole Foods, Kroger, Costco, Amazon). "
            "If the information is explicitly restricted to a specific state/region or ambiguous, consider it not nationwide."
        ),
        extra_prerequisites=[ref_node],  # Require that references exist
    )

    # Optional: record useful debugging info for this product
    evaluator.add_custom_info(
        info={
            "product_index": pid,
            "extracted_brand": brand,
            "extracted_product_name": pname,
            "extracted_category": category_raw,
            "normalized_certifications": normalized_certs,
            "valid_reference_urls": valid_refs,
            "per_cert_results": [{"cert": c, "verified": v} for c, v in per_cert_results],
        },
        info_type="product_debug",
        info_name=f"product_{pid}_debug",
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
    Evaluate an answer for the multi‑certified nationwide food products task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Products are independent; allow partial credit
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

    # Record allowed certifications for transparency
    evaluator.add_custom_info(
        info={"allowed_certifications": ALLOWED_CERTIFICATIONS},
        info_type="config",
        info_name="allowed_certifications",
    )

    # Extract up to three products
    extracted = await evaluator.extract(
        prompt=prompt_extract_products(),
        template_class=ProductsExtraction,
        extraction_name="extracted_products",
    )

    products: List[Product] = list(extracted.products[:3]) if extracted and extracted.products else []

    # Pad to exactly 3 slots for consistent tree structure
    while len(products) < 3:
        products.append(Product())

    # Keep track of prior categories (raw) for uniqueness checks
    prior_categories: List[Optional[str]] = []

    # Build verification subtrees for each product
    for idx in range(3):
        await verify_product(
            evaluator=evaluator,
            parent_node=root,
            product=products[idx],
            index=idx,
            prior_categories=prior_categories,
        )
        prior_categories.append(products[idx].category)

    return evaluator.get_summary()