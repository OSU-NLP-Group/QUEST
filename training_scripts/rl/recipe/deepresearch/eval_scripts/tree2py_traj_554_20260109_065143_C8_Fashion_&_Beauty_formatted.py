import asyncio
import logging
import re
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sustainable_brands_na_certifications"
TASK_DESCRIPTION = (
    "Identify four distinct sustainable fashion or beauty brands headquartered in North America (United States or Canada) "
    "that each hold at least one of the following third-party certifications: B Corp Certification, Leaping Bunny Certification, or GOTS (Global Organic Textile Standard) Certification.\n\n"
    "For each brand, provide the following information:\n"
    "1. Brand name\n"
    "2. Headquarters location (city and state/province, country)\n"
    "3. Year founded\n"
    "4. Primary certification(s) held (B Corp, Leaping Bunny, and/or GOTS)\n"
    "5. Main product category/categories from this list: women's apparel, men's apparel, activewear/athletic wear, skincare products, color cosmetics/makeup, haircare products\n"
    "6. At least one specific sustainability feature the brand demonstrates, chosen from: uses organic or recycled materials in products, offers refillable/reusable packaging systems, has plastic-free or zero-waste product options, or operates a take-back or recycling program\n"
    "7. Official website URL\n\n"
    "Additional requirements:\n"
    "- Each brand must have been founded in or after 1995\n"
    "- All four brands must be distinct companies (not different brands owned by the same parent company at the time of their founding)\n"
    "- Across the four brands, at least three different product categories from the provided list must be represented\n"
    "- Each website URL must be accessible and contain verifiable information about the brand's certifications and sustainability practices"
)

ALLOWED_CERTIFICATIONS = {"B Corp", "Leaping Bunny", "GOTS"}
ALLOWED_CATEGORIES_CANONICAL = {
    "women's apparel",
    "men's apparel",
    "activewear/athletic wear",
    "skincare products",
    "color cosmetics/makeup",
    "haircare products",
}
ALLOWED_FEATURES_CANONICAL = {
    "uses organic or recycled materials",
    "refillable/reusable packaging",
    "plastic-free or zero-waste options",
    "take-back or recycling program",
}
NA_COUNTRIES = {"united states", "usa", "us", "united states of america", "canada"}
MIN_YEAR = 1995

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BrandInfo(BaseModel):
    name: Optional[str] = None
    headquarters_city: Optional[str] = None
    headquarters_state_province: Optional[str] = None
    headquarters_country: Optional[str] = None
    year_founded: Optional[str] = None
    certifications: List[str] = Field(default_factory=list)
    product_categories: List[str] = Field(default_factory=list)
    sustainability_features: List[str] = Field(default_factory=list)
    website_url: Optional[str] = None


class BrandsExtraction(BaseModel):
    brands: List[BrandInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return (
        "Extract all sustainable fashion or beauty brands mentioned in the answer. For each brand, return the following fields:\n"
        "1. name: The brand name as shown in the answer\n"
        "2. headquarters_city: The headquarters city as stated in the answer (return null if not provided)\n"
        "3. headquarters_state_province: The headquarters state or province as stated in the answer (return null if not provided)\n"
        "4. headquarters_country: The headquarters country as stated in the answer (return null if not provided)\n"
        "5. year_founded: The year founded exactly as noted in the answer (string; return null if not provided)\n"
        "6. certifications: List of certifications explicitly mentioned for the brand in the answer (e.g., 'Certified B Corporation', 'Leaping Bunny', 'GOTS'). Do not infer or add.\n"
        "7. product_categories: List of the brand's main product categories as stated in the answer (e.g., 'women's apparel', 'makeup', 'skincare'). Do not infer or add.\n"
        "8. sustainability_features: List of sustainability features explicitly mentioned in the answer (e.g., 'recycled materials', 'refillable packaging', 'zero-waste'). Do not infer or add.\n"
        "9. website_url: The official website URL for the brand as provided in the answer. If missing protocol, prepend http://\n\n"
        "Return a JSON object with a single key 'brands' that is an array of brand objects in the same order as in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper normalization & parsing functions                                    #
# --------------------------------------------------------------------------- #
def _safe_str(x: Optional[str]) -> str:
    return x.strip() if isinstance(x, str) else ""

def _parse_year(year_text: Optional[str]) -> Optional[int]:
    if not year_text:
        return None
    m = re.search(r"\b(19\d{2}|20\d{2})\b", year_text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _canonical_country(country: Optional[str]) -> Optional[str]:
    s = _safe_str(country).lower()
    if not s:
        return None
    if s in {"united states", "usa", "us", "united states of america"}:
        return "United States"
    if s == "canada":
        return "Canada"
    return None

def _canonical_cert(cert_text: str) -> Optional[str]:
    s = cert_text.lower().strip()
    if not s:
        return None
    if "b corp" in s or "b-corp" in s or "certified b corporation" in s or "b corporation" in s:
        return "B Corp"
    if "leaping bunny" in s or "cruelty free international" in s or "ccic" in s:
        return "Leaping Bunny"
    if "gots" in s or "global organic textile standard" in s:
        return "GOTS"
    return None

CATEGORY_SYNONYMS = {
    "women's apparel": {"women's apparel", "womens apparel", "womenswear", "women's clothing", "women’s apparel", "women fashion", "women apparel"},
    "men's apparel": {"men's apparel", "menswear", "men's clothing", "mens apparel", "men apparel"},
    "activewear/athletic wear": {"activewear", "athletic wear", "sportswear", "performance wear", "fitness apparel", "athleisure"},
    "skincare products": {"skincare", "skin care", "skincare products", "skin-care"},
    "color cosmetics/makeup": {"makeup", "color cosmetics", "cosmetics", "beauty makeup", "color-cosmetics"},
    "haircare products": {"haircare", "hair care", "hair products", "hair-care"},
}

def _canonical_category(cat_text: str) -> Optional[str]:
    s = cat_text.lower().strip()
    for canonical, variants in CATEGORY_SYNONYMS.items():
        for v in variants:
            if s == v or v in s:
                return canonical
    return None

FEATURE_SYNONYMS = {
    "uses organic or recycled materials": {"organic materials", "recycled materials", "organic cotton", "recycled polyester", "recycled fabric", "organic/recycled", "recycled"},
    "refillable/reusable packaging": {"refillable packaging", "refill system", "reusable packaging", "refills", "refillable"},
    "plastic-free or zero-waste options": {"plastic-free", "plastic free", "zero waste", "zerowaste"},
    "take-back or recycling program": {"take-back program", "recycling program", "take back", "product take-back", "return program"},
}

def _canonical_feature(feat_text: str) -> Optional[str]:
    s = feat_text.lower().strip()
    for canonical, variants in FEATURE_SYNONYMS.items():
        for v in variants:
            if s == v or v in s:
                return canonical
    return None

def _canonicalize_certifications(certs: List[str]) -> List[str]:
    result = []
    for c in certs:
        canon = _canonical_cert(c or "")
        if canon:
            result.append(canon)
    return sorted(list(set(result)))

def _canonicalize_categories(cats: List[str]) -> List[str]:
    result = []
    for c in cats:
        canon = _canonical_category(c or "")
        if canon:
            result.append(canon)
    return sorted(list(set(result)))

def _canonicalize_features(feats: List[str]) -> List[str]:
    result = []
    for f in feats:
        canon = _canonical_feature(f or "")
        if canon:
            result.append(canon)
    return sorted(list(set(result)))

def _is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    s = url.strip().lower()
    return s.startswith("http://") or s.startswith("https://")

# --------------------------------------------------------------------------- #
# Verification for a single brand                                             #
# --------------------------------------------------------------------------- #
async def verify_brand(
    evaluator: Evaluator,
    parent_node,
    brand: BrandInfo,
    brand_index: int,
) -> Tuple[List[str], List[str]]:
    """
    Verify per-brand constraints and fields (Brand i node).
    Returns (canonicalized_categories, canonicalized_features) for coverage aggregation.
    """
    # Create brand parallel node
    brand_node = evaluator.add_parallel(
        id=f"brand_{brand_index + 1}",
        desc=f"Brand {brand_index + 1} satisfies all per-brand constraints and required fields",
        parent=parent_node,
        critical=False
    )

    # Canonicalize lists for checks and later aggregation
    canonical_certs = _canonicalize_certifications(brand.certifications or [])
    canonical_cats = _canonicalize_categories(brand.product_categories or [])
    canonical_feats = _canonicalize_features(brand.sustainability_features or [])

    # 1. Brand name provided (critical)
    evaluator.add_custom_node(
        result=bool(_safe_str(brand.name)),
        id=f"brand_{brand_index + 1}_name_provided",
        desc=f"Brand {brand_index + 1} name is provided",
        parent=brand_node,
        critical=True
    )

    # 2. Headquarters provided (city, state/province, country)
    hq_provided = all([
        bool(_safe_str(brand.headquarters_city)),
        bool(_safe_str(brand.headquarters_state_province)),
        bool(_safe_str(brand.headquarters_country)),
    ])
    evaluator.add_custom_node(
        result=hq_provided,
        id=f"brand_{brand_index + 1}_headquarters_provided",
        desc=f"Brand {brand_index + 1} headquarters location is provided with city and state/province and country",
        parent=brand_node,
        critical=True
    )

    # 3. Headquarters in North America (US or Canada)
    country_canon = _canonical_country(brand.headquarters_country)
    evaluator.add_custom_node(
        result=(country_canon in {"United States", "Canada"}),
        id=f"brand_{brand_index + 1}_headquarters_in_na",
        desc=f"Brand {brand_index + 1} headquarters is in North America (United States or Canada)",
        parent=brand_node,
        critical=True
    )

    # 4. Year founded provided
    evaluator.add_custom_node(
        result=bool(_safe_str(brand.year_founded)),
        id=f"brand_{brand_index + 1}_year_founded_provided",
        desc=f"Brand {brand_index + 1} year founded is provided",
        parent=brand_node,
        critical=True
    )

    # 5. Year founded >= 1995
    yf = _parse_year(brand.year_founded)
    evaluator.add_custom_node(
        result=(yf is not None and yf >= MIN_YEAR),
        id=f"brand_{brand_index + 1}_year_founded_1995_or_later",
        desc=f"Brand {brand_index + 1} was founded in or after 1995",
        parent=brand_node,
        critical=True
    )

    # 6. At least one allowed certification
    evaluator.add_custom_node(
        result=(len(canonical_certs) >= 1),
        id=f"brand_{brand_index + 1}_certification",
        desc=f"Brand {brand_index + 1} has at least one listed certification from: B Corp, Leaping Bunny, and/or GOTS",
        parent=brand_node,
        critical=True
    )

    # 7. Product categories from allowed list
    evaluator.add_custom_node(
        result=(len(canonical_cats) >= 1 and all(cat in ALLOWED_CATEGORIES_CANONICAL for cat in canonical_cats)),
        id=f"brand_{brand_index + 1}_product_categories_from_list",
        desc=f"Brand {brand_index + 1} lists main product category/categories and they are from the allowed list (women's apparel, men's apparel, activewear/athletic wear, skincare products, color cosmetics/makeup, haircare products)",
        parent=brand_node,
        critical=True
    )

    # 8. Sustainability feature from allowed list
    evaluator.add_custom_node(
        result=(len(canonical_feats) >= 1 and any(f in ALLOWED_FEATURES_CANONICAL for f in canonical_feats)),
        id=f"brand_{brand_index + 1}_sustainability_feature_from_list",
        desc=f"Brand {brand_index + 1} provides at least one specific sustainability feature from: uses organic or recycled materials; refillable/reusable packaging; plastic-free/zero-waste options; take-back/recycling program",
        parent=brand_node,
        critical=True
    )

    # 9. Website URL provided
    evaluator.add_custom_node(
        result=_is_valid_url(brand.website_url),
        id=f"brand_{brand_index + 1}_website_url_provided",
        desc=f"Brand {brand_index + 1} official website URL is provided",
        parent=brand_node,
        critical=True
    )

    # 10. Website accessible (verify by URL)
    node_access = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_website_accessible",
        desc=f"Brand {brand_index + 1} website URL is accessible",
        parent=brand_node,
        critical=True
    )
    await evaluator.verify(
        claim="The webpage is accessible and loads content.",
        node=node_access,
        sources=brand.website_url,
        additional_instruction="Pass if the page can be retrieved and contains any visible content (text or screenshot). Fail if retrieval fails or page is obviously inaccessible."
    )

    # 11. Website verifies certifications and sustainability practices (verify by URL)
    # Build a claim referencing allowed certs and features; allow synonyms.
    certs_for_claim = canonical_certs if canonical_certs else list(ALLOWED_CERTIFICATIONS)
    feats_for_claim = canonical_feats if canonical_feats else list(ALLOWED_FEATURES_CANONICAL)
    claim_verify = (
        f"The brand's official website page contains verifiable information about at least one of its third-party certifications "
        f"(one of: {', '.join(certs_for_claim)}) and at least one sustainability practice "
        f"(one of: {', '.join(feats_for_claim)})."
    )
    node_verify_claims = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_website_verifies_claims",
        desc=f"Brand {brand_index + 1} website contains verifiable information about the brand's certifications and sustainability practices",
        parent=brand_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_verify,
        node=node_verify_claims,
        sources=brand.website_url,
        additional_instruction=(
            "Search the provided webpage content and screenshot for keywords and statements indicating third-party certifications "
            "(e.g., 'Certified B Corporation', 'B Corp', 'Leaping Bunny', 'Global Organic Textile Standard', 'GOTS') "
            "and sustainability practices (e.g., 'organic materials', 'recycled materials', 'refillable', 'reusable packaging', "
            "'plastic-free', 'zero waste', 'take-back', 'recycling program'). "
            "Minor variants or synonyms should count as a match if clearly indicating the listed items."
        )
    )

    return canonical_cats, canonical_feats


# --------------------------------------------------------------------------- #
# Collective requirements verification                                         #
# --------------------------------------------------------------------------- #
def compute_collective_stats(brands_for_eval: List[BrandInfo]) -> Dict[str, Any]:
    names = [(_safe_str(b.name).lower()) for b in brands_for_eval if _safe_str(b.name)]
    unique_names = sorted(list(set(names)))

    coverage_categories = set()
    for b in brands_for_eval:
        coverage_categories.update(_canonicalize_categories(b.product_categories or []))

    stats = {
        "count": len(brands_for_eval),
        "unique_brand_count": len(unique_names),
        "unique_brand_names": unique_names,
        "distinct_categories": sorted(list(coverage_categories)),
        "distinct_category_count": len(coverage_categories),
    }
    return stats


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
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

    # Extract brand information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="brands_extraction",
    )

    # Add ground-truth-like allowed lists and constraints for transparency
    evaluator.add_ground_truth({
        "allowed_certifications": sorted(list(ALLOWED_CERTIFICATIONS)),
        "allowed_categories": sorted(list(ALLOWED_CATEGORIES_CANONICAL)),
        "allowed_features": sorted(list(ALLOWED_FEATURES_CANONICAL)),
        "min_year_founded": MIN_YEAR,
        "na_countries": ["United States", "Canada"],
    }, gt_type="constraints")

    # Prepare the list of brands to evaluate (up to 4; do not invent)
    brands_list = extracted.brands or []
    # We will evaluate only first 4 for per-brand checks, but collective requirement checks the exact count reported
    if len(brands_list) >= 4:
        brands_for_eval = brands_list[:4]
    else:
        # Pad with empty BrandInfo to maintain tree shape
        brands_for_eval = brands_list[:] + [BrandInfo() for _ in range(max(0, 4 - len(brands_list)))]

    # Collective requirements node (critical)
    collective_node = evaluator.add_parallel(
        id="collective_requirements",
        desc="Requirements that must be satisfied across the full set of brands",
        parent=root,
        critical=True
    )

    # exactly_four_brands (critical custom leaf)
    evaluator.add_custom_node(
        result=(len(brands_list) == 4),
        id="exactly_four_brands",
        desc="The response identifies exactly four distinct brands (no more, no fewer)",
        parent=collective_node,
        critical=True
    )

    # distinct_companies (critical leaf) — check distinct brand names; stronger corporate parent validation is not feasible here
    # We treat "distinct companies" as "distinct brand names" for a conservative check.
    names_lower = [(_safe_str(b.name).lower()) for b in brands_for_eval if _safe_str(b.name)]
    evaluator.add_custom_node(
        result=(len(names_lower) == 4 and len(set(names_lower)) == 4),
        id="distinct_companies",
        desc="All four brands are distinct companies (not different brands owned by the same parent company at the time of founding)",
        parent=collective_node,
        critical=True
    )

    # category_coverage (critical custom leaf) — at least 3 distinct categories across the four brands
    collective_categories = set()
    for b in brands_for_eval:
        collective_categories.update(_canonicalize_categories(b.product_categories or []))
    evaluator.add_custom_node(
        result=(len(collective_categories) >= 3),
        id="category_coverage",
        desc="Across the four brands, at least three distinct product categories are represented from: women's apparel, men's apparel, activewear/athletic wear, skincare products, color cosmetics/makeup, haircare products",
        parent=collective_node,
        critical=True
    )

    # Verify each brand (Brand 1..4) under root
    # Collect coverage info to include in summary
    coverage_cats_all: List[str] = []
    coverage_feats_all: List[str] = []

    for idx in range(4):
        cats, feats = await verify_brand(evaluator, root, brands_for_eval[idx], idx)
        coverage_cats_all.extend(cats)
        coverage_feats_all.extend(feats)

    # Add custom info for transparency
    collective_stats = compute_collective_stats(brands_for_eval)
    evaluator.add_custom_info(
        info={
            "evaluated_brand_count": collective_stats["count"],
            "unique_brand_names": collective_stats["unique_brand_names"],
            "distinct_categories": collective_stats["distinct_categories"],
            "distinct_category_count": collective_stats["distinct_category_count"],
            "coverage_categories_all": sorted(list(set(coverage_cats_all))),
            "coverage_features_all": sorted(list(set(coverage_feats_all))),
        },
        info_type="aggregation",
        info_name="collective_coverage_stats"
    )

    # Return standard evaluation summary
    return evaluator.get_summary()