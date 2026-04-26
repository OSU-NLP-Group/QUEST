import asyncio
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_beauty_brands_2010_2020"
TASK_DESCRIPTION = (
    "Identify four celebrity-founded beauty brands that were launched between 2010 and 2020 (inclusive). "
    "Each brand must meet the following criteria:\n"
    "1. The celebrity must be the founder or co-founder of the brand, with ownership stake (not merely an ambassador)\n"
    "2. The brand must have been launched between 2010 and 2020 (inclusive)\n"
    "3. The brand must currently offer products in at least two distinct product categories "
    "(such as skincare, makeup/cosmetics, fragrance/perfume, or haircare)\n"
    "4. The brand must be currently available for purchase through at least one major U.S. retail partner "
    "(examples: Ulta Beauty, Sephora, Target, Amazon, QVC)\n\n"
    "For each of the four brands, provide: brand name, celebrity founder's name, launch year, at least two product "
    "categories, at least one major U.S. retail partner, and a reference URL that supports the brand information."
)

# Retailer normalization and recognition
MAJOR_RETAILERS = {
    "ulta beauty",
    "ulta",
    "sephora",
    "target",
    "amazon",
    "qvc",
}

DOMAIN_TO_MAJOR_RETAILER = {
    "ulta.com": "ulta beauty",
    "www.ulta.com": "ulta beauty",
    "sephora.com": "sephora",
    "www.sephora.com": "sephora",
    "target.com": "target",
    "www.target.com": "target",
    "amazon.com": "amazon",
    "www.amazon.com": "amazon",
    "qvc.com": "qvc",
    "www.qvc.com": "qvc",
}

CATEGORY_SYNONYMS = {
    "makeup": {"makeup", "cosmetics", "color cosmetics"},
    "skincare": {"skincare", "skin care", "skin-care", "skin"},
    "fragrance": {"fragrance", "perfume", "eau de parfum", "eau de toilette", "cologne"},
    "haircare": {"haircare", "hair care", "hair"},
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BrandItem(BaseModel):
    brand_name: Optional[str] = None
    celebrity_founder: Optional[str] = None
    launch_year: Optional[str] = None
    product_categories: List[str] = Field(default_factory=list)
    major_retail_partners: List[str] = Field(default_factory=list)
    retailer_urls: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class BrandsExtraction(BaseModel):
    brands: List[BrandItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return """
    Extract up to 6 celebrity-founded beauty brands from the answer. For each brand, extract the following fields:

    - brand_name: The name of the beauty brand.
    - celebrity_founder: The celebrity founder or co-founder (a person; not simply a brand ambassador).
    - launch_year: The brand's launch year as written (e.g., "2016"). If a range or month is given, return the year (e.g., "2016"). If unknown, set to null.
    - product_categories: A list of at least two distinct product categories the brand offers (e.g., "skincare", "makeup", "fragrance", "haircare"). Normalize obvious synonyms where possible (e.g., "cosmetics" -> "makeup", "perfume" -> "fragrance", "hair" -> "haircare", "skin care" -> "skincare").
    - major_retail_partners: A list of U.S. retailer names (e.g., "Ulta Beauty", "Sephora", "Target", "Amazon", "QVC") where the brand is currently available, based on the answer text.
    - retailer_urls: A list of retailer product or brand listing URLs (if present in the answer). Include only actual URLs from the answer.
    - reference_urls: A list of reference URLs that support the brand information. These can be the brand’s official website, credible industry publications, or retailer listings. Include only URLs explicitly present in the answer.

    Important rules:
    - Do not invent information. Only extract what is explicitly present in the answer.
    - If any field is missing, set it to null (for single values) or an empty list (for lists).
    - Keep the order of brands as presented in the answer.

    Return JSON with a single field:
    {
      "brands": [
         {
           "brand_name": ...,
           "celebrity_founder": ...,
           "launch_year": ...,
           "product_categories": [...],
           "major_retail_partners": [...],
           "retailer_urls": [...],
           "reference_urls": [...]
         },
         ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_retailer_name(name: Optional[str]) -> str:
    name = normalize_text(name)
    # Simple canonicalization for common variants
    if name in {"ulta", "ulta beauty", "ulta.com"}:
        return "ulta beauty"
    if name in {"sephora", "sephora.com"}:
        return "sephora"
    if name in {"target", "target.com"}:
        return "target"
    if name in {"amazon", "amazon.com"}:
        return "amazon"
    if name in {"qvc", "qvc.com"}:
        return "qvc"
    return name


def domain_to_retailer(url: str) -> Optional[str]:
    try:
        dom = urlparse(url).netloc.lower()
        return DOMAIN_TO_MAJOR_RETAILER.get(dom)
    except Exception:
        return None


def is_major_retailer(name: Optional[str]) -> bool:
    return normalize_retailer_name(name) in MAJOR_RETAILERS


def pick_major_retailer(brand: BrandItem) -> Tuple[Optional[str], List[str]]:
    """
    Returns (retailer_name, supporting_urls_for_that_retailer)
    """
    # Prefer named major retailers provided explicitly
    for r in brand.major_retail_partners:
        rn = normalize_retailer_name(r)
        if rn in MAJOR_RETAILERS:
            urls = [u for u in brand.retailer_urls if domain_to_retailer(u) == rn]
            # If none matched by URL domain, keep all retailer_urls as fallback
            return rn, urls if urls else brand.retailer_urls

    # Next, infer from retailer_urls' domains
    inferred_counts: Dict[str, int] = {}
    for u in brand.retailer_urls:
        rn = domain_to_retailer(u)
        if rn and rn in MAJOR_RETAILERS:
            inferred_counts[rn] = inferred_counts.get(rn, 0) + 1
    if inferred_counts:
        chosen = max(inferred_counts.items(), key=lambda x: x[1])[0]
        urls = [u for u in brand.retailer_urls if domain_to_retailer(u) == chosen]
        return chosen, urls if urls else brand.retailer_urls

    return None, brand.retailer_urls


def canonical_category(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    c = normalize_text(cat)
    for canon, syns in CATEGORY_SYNONYMS.items():
        if c in syns:
            return canon
    # Keep as-is if not matched, but normalized
    return c if c else None


def distinct_top_two_categories(cats: List[str]) -> Tuple[Optional[str], Optional[str]]:
    normalized = []
    for c in cats:
        cc = canonical_category(c)
        if cc:
            normalized.append(cc)
    # Preserve order but ensure distinct for the first two
    seen = set()
    ordered = []
    for c in normalized:
        key = c
        if key not in seen:
            ordered.append(c)
            seen.add(key)
        if len(ordered) >= 2:
            break
    c1 = ordered[0] if len(ordered) >= 1 else None
    c2 = ordered[1] if len(ordered) >= 2 else None
    return c1, c2


def merge_sources(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in lists:
        for u in lst:
            if not u:
                continue
            if u not in seen:
                merged.append(u)
                seen.add(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification logic per brand                                                #
# --------------------------------------------------------------------------- #
async def verify_brand(
    evaluator: Evaluator,
    parent_node,
    brand: BrandItem,
    brand_index: int
) -> None:
    """
    Build verification subtree for a single brand.
    Structure (adjusted to satisfy framework's critical-child constraint):
      - Brand_i (parallel, non-critical)
        - Brand_i_Founder (leaf, critical)
        - Brand_i_Launch_Year (leaf, critical)
        - Brand_i_Product_Categories (parallel, critical)
            - Brand_i_Category_1 (leaf, critical)
            - Brand_i_Category_2 (leaf, critical)
        - Brand_i_Additional_Categories (custom, non-critical)  <-- sibling at brand level
        - Brand_i_Retail_Availability (parallel, critical)
            - Brand_i_Retailer_Name (custom, critical)
            - Brand_i_Retailer_Evidence (leaf, critical)
        - Brand_i_Reference_URL (leaf, critical)
    """

    # Create the brand node
    brand_node = evaluator.add_parallel(
        id=f"Brand_{brand_index+1}",
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth'][brand_index] if brand_index < 6 else f'Brand #{brand_index+1}'} celebrity beauty brand meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # Extract convenient variables
    brand_name = brand.brand_name or ""
    founder = brand.celebrity_founder or ""
    launch_year = (brand.launch_year or "").strip()
    cat1, cat2 = distinct_top_two_categories(brand.product_categories or [])
    primary_retailer, retailer_specific_urls = pick_major_retailer(brand)

    # Sources for different checks
    founder_sources = brand.reference_urls
    launch_sources = brand.reference_urls
    categories_sources = merge_sources(brand.reference_urls, brand.retailer_urls)
    retailer_sources = retailer_specific_urls if retailer_specific_urls else brand.retailer_urls
    reference_sources = brand.reference_urls

    # 1) Founder check (critical)
    founder_node = evaluator.add_leaf(
        id=f"Brand_{brand_index+1}_Founder",
        desc="The celebrity founder/co-founder of the brand is identified and the brand is confirmed to be owned by that celebrity (not merely an ambassadorship)",
        parent=brand_node,
        critical=True
    )
    founder_claim = (
        f"'{founder}' is a founder or co-founder with an ownership stake (not merely a brand ambassador) "
        f"of the beauty brand '{brand_name}'."
    )
    await evaluator.verify(
        claim=founder_claim,
        node=founder_node,
        sources=founder_sources,
        additional_instruction=(
            "Treat 'founder' and 'co-founder' as positive indicators. "
            "Pages that only indicate 'ambassador', 'collaboration', or 'spokesperson' should NOT count. "
            "Look for explicit wording such as 'founded by', 'co-founded by', 'launched by', or similar. "
            "Prefer official brand pages, credible industry publications, or major news outlets."
        )
    )

    # 2) Launch year in 2010–2020 (critical)
    launch_node = evaluator.add_leaf(
        id=f"Brand_{brand_index+1}_Launch_Year",
        desc="The brand's launch year is verified to be between 2010 and 2020 (inclusive)",
        parent=brand_node,
        critical=True
    )
    launch_claim = (
        f"The beauty brand '{brand_name}' launched in {launch_year}, and this year lies between 2010 and 2020 inclusive."
    )
    await evaluator.verify(
        claim=launch_claim,
        node=launch_node,
        sources=launch_sources,
        additional_instruction=(
            "Accept wording like 'founded in', 'launched in', or 'debuted in'. "
            "If multiple years are reported, prefer the year explicitly labeled 'launched' or 'founded'. "
            "If the cited year is not within 2010–2020 inclusive, this should be marked incorrect."
        )
    )

    # 3) Product categories (critical aggregate for 2 categories)
    categories_main = evaluator.add_parallel(
        id=f"Brand_{brand_index+1}_Product_Categories",
        desc="The brand offers products in at least two distinct product categories",
        parent=brand_node,
        critical=True
    )

    # Category 1 (critical)
    cat1_node = evaluator.add_leaf(
        id=f"Brand_{brand_index+1}_Category_1",
        desc="First product category is identified and verified (e.g., skincare, makeup, fragrance, haircare)",
        parent=categories_main,
        critical=True
    )
    cat1_claim = (
        f"The beauty brand '{brand_name}' offers products in the '{cat1 or ''}' category."
    )
    await evaluator.verify(
        claim=cat1_claim,
        node=cat1_node,
        sources=categories_sources,
        additional_instruction=(
            "Consider category synonyms: makeup ≈ cosmetics; skincare ≈ skin care; fragrance ≈ perfume; haircare ≈ hair. "
            "Accept reasonable evidence such as category pages on retailer sites or official brand site collections."
        )
    )

    # Category 2 (critical, must be distinct from Category 1)
    cat2_node = evaluator.add_leaf(
        id=f"Brand_{brand_index+1}_Category_2",
        desc="Second product category is identified and verified (must be distinct from the first)",
        parent=categories_main,
        critical=True
    )
    cat2_claim = (
        f"The beauty brand '{brand_name}' offers products in the '{cat2 or ''}' category, "
        f"and this category is distinct from '{cat1 or ''}'."
    )
    await evaluator.verify(
        claim=cat2_claim,
        node=cat2_node,
        sources=categories_sources,
        additional_instruction=(
            "Ensure the second category is genuinely different from the first (do not treat 'cosmetics' vs 'makeup' as distinct). "
            "Use the same synonym rules as Category 1."
        )
    )

    # Additional categories (non-critical; implemented as a simple custom check on extraction presence)
    addl_categories_present = len(brand.product_categories) >= 3
    evaluator.add_custom_node(
        result=addl_categories_present,
        id=f"Brand_{brand_index+1}_Additional_Categories",
        desc="Additional product categories beyond the required two are documented",
        parent=brand_node,
        critical=False
    )

    # 4) Retail availability (critical)
    retail_main = evaluator.add_parallel(
        id=f"Brand_{brand_index+1}_Retail_Availability",
        desc="The brand is currently available at a major U.S. retail partner",
        parent=brand_node,
        critical=True
    )

    # Retailer name identified (critical, custom boolean check)
    retailer_name_ok = False
    # Check explicit retailer names
    if brand.major_retail_partners:
        retailer_name_ok = any(is_major_retailer(r) for r in brand.major_retail_partners)
    # Or infer from retailer URLs
    if not retailer_name_ok and brand.retailer_urls:
        for u in brand.retailer_urls:
            rn = domain_to_retailer(u)
            if rn in MAJOR_RETAILERS:
                retailer_name_ok = True
                break

    evaluator.add_custom_node(
        result=retailer_name_ok,
        id=f"Brand_{brand_index+1}_Retailer_Name",
        desc="At least one major U.S. retail partner is identified (e.g., Ulta Beauty, Sephora, Target, Amazon, QVC)",
        parent=retail_main,
        critical=True
    )

    # Retailer evidence (critical)
    retailer_evidence_node = evaluator.add_leaf(
        id=f"Brand_{brand_index+1}_Retailer_Evidence",
        desc="Evidence of retail availability is provided (e.g., link to product listing on retailer's website)",
        parent=retail_main,
        critical=True
    )

    retailer_display = primary_retailer or (brand.major_retail_partners[0] if brand.major_retail_partners else "")
    retailer_evidence_claim = (
        f"There is a product or brand listing for '{brand_name}' available on the {retailer_display} website."
    )
    await evaluator.verify(
        claim=retailer_evidence_claim,
        node=retailer_evidence_node,
        sources=retailer_sources if retailer_sources else brand.reference_urls,
        additional_instruction=(
            "Verify that the URL(s) point to a product or brand listing on a major U.S. retailer's site "
            "(Ulta Beauty, Sephora, Target, Amazon, or QVC) for the stated brand."
        )
    )

    # 5) Reference URL (critical)
    reference_node = evaluator.add_leaf(
        id=f"Brand_{brand_index+1}_Reference_URL",
        desc="A reference URL is provided that supports the brand information (official brand website, major beauty publication, or retail listing)",
        parent=brand_node,
        critical=True
    )
    reference_claim = (
        f"At least one of these pages is a credible reference relevant to the beauty brand '{brand_name}' "
        f"(e.g., official brand site, credible industry publication, or retailer listing)."
    )
    await evaluator.verify(
        claim=reference_claim,
        node=reference_node,
        sources=reference_sources if reference_sources else brand.retailer_urls,
        additional_instruction=(
            "Accept the claim if any provided URL clearly pertains to the stated brand (official brand site, credible industry "
            "publication article, or a major retailer listing). Focus on relevance and credibility."
        )
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
    Evaluate an answer for the celebrity-founded beauty brands (2010–2020) task.
    """
    # Initialize evaluator (root is a non-critical node with parallel aggregation)
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
        default_model=model
    )

    # Extract brands from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="extracted_brands"
    )

    # Add a top-level aggregator corresponding to "Complete_Task"
    complete_task_node = evaluator.add_parallel(
        id="Complete_Task",
        desc="Identify four celebrity-founded beauty brands launched between 2010-2020 that offer products in at least two categories and are available at major U.S. retailers",
        parent=root,
        critical=False  # Adjusted to satisfy framework constraints (critical parent can't have non-critical children)
    )

    # Prepare up to four brands: take first 4; if fewer, pad with empty BrandItem
    brands = list(extracted.brands) if extracted and extracted.brands else []
    if len(brands) > 4:
        brands = brands[:4]
    while len(brands) < 4:
        brands.append(BrandItem())

    # Build verification subtrees for the four brands
    for i in range(4):
        await verify_brand(evaluator, complete_task_node, brands[i], i)

    # Return evaluation summary
    return evaluator.get_summary()