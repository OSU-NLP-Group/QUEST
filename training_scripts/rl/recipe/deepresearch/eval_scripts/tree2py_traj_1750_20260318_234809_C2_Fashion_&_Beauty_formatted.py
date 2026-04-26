import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celeb_beauty_multi_retail_2026"
TASK_DESCRIPTION = """
Identify two celebrity beauty brands that are each currently available for purchase at three or more major U.S. retail chains as of March 2026. For each brand, provide: (1) The brand name, (2) The name of the celebrity founder or co-founder, (3) At least one product category the brand offers (choose from: makeup, skincare, or fragrance), (4) The names of three major U.S. retail chains where the brand is currently available (major retail chains include Sephora, Ulta, Target, Walmart, Kohl's, Nordstrom, CVS, or Walgreens - online-only retailers do not count unless they have physical store presence), and (5) A reference URL from an official source (such as the brand's store locator page, a major retailer's product page, or an official press release) that verifies the brand's availability at multiple retailers. The two brands you identify must be different from each other, and each must meet all the requirements above.
"""

# Allowed retailers and canonical domains for URL filtering
ALLOWED_RETAILERS = [
    "Sephora", "Ulta", "Target", "Walmart", "Kohl's", "Nordstrom", "CVS", "Walgreens"
]
RETAILER_DOMAINS = {
    "Sephora": "sephora.com",
    "Ulta": "ulta.com",
    "Target": "target.com",
    "Walmart": "walmart.com",
    "Kohl's": "kohls.com",
    "Nordstrom": "nordstrom.com",
    "CVS": "cvs.com",
    "Walgreens": "walgreens.com",
}

ALLOWED_CATEGORIES = ["makeup", "skincare", "fragrance"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BrandEntry(BaseModel):
    brand_name: Optional[str] = None
    celebrity_founder: Optional[str] = None
    product_categories: List[str] = Field(default_factory=list)
    retailers: List[str] = Field(default_factory=list)  # Names of (ideally) 3 major chains from the allowed list
    retailer_urls: List[str] = Field(default_factory=list)  # Product/category pages on retailer domains (if provided)
    founder_sources: List[str] = Field(default_factory=list)  # URLs supporting the founder/co-founder claim
    category_sources: List[str] = Field(default_factory=list)  # URLs showing the brand offers the category
    reference_urls: List[str] = Field(default_factory=list)  # Official URLs verifying multi-retailer availability


class BrandsExtraction(BaseModel):
    brands: List[BrandEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return f"""
You must extract exactly TWO celebrity beauty brand entries from the answer (the first two in the order they appear). For each brand, extract the following fields:

1) brand_name: The brand's name (string).
2) celebrity_founder: The celebrity founder or co-founder name (string).
3) product_categories: A list of at least one product category the brand offers. Only use these normalized labels when possible: {ALLOWED_CATEGORIES}. If the answer uses near-synonyms (e.g., "cosmetics" for makeup, "skin care" for skincare, "perfume"/"cologne" for fragrance), normalize them to the closest allowed label.
4) retailers: A list of up to three MAJOR U.S. retail chains from this exact allowed set: {ALLOWED_RETAILERS}. If the answer mentions more than three, keep the first three that match the allowed set. If fewer than three are present, include as many as are explicitly stated.
5) retailer_urls: A list of any retailer product/category pages or brand pages hosted on retailer domains (e.g., sephora.com, ulta.com, target.com, walmart.com, kohls.com, nordstrom.com, cvs.com, walgreens.com) that are explicitly provided in the answer for this brand. Include all that appear for this brand. If none, return an empty list.
6) founder_sources: A list of official URLs in the answer that support the founder/co-founder claim (e.g., brand About page, press release, retailer/brand page, or reputable profile). If none, return an empty list.
7) category_sources: A list of official URLs in the answer that show the brand offers the extracted category (brand product pages or major retailer pages featuring the brand in that category). If none, return an empty list.
8) reference_urls: A list (1–3) of official URLs in the answer that help verify the brand's availability at multiple retailers (e.g., brand store locator page, official press release announcing multiple retail partners, or multiple retailer product pages). If none, return an empty list.

Important extraction rules:
- Return exactly two BrandEntry objects under "brands". If the answer lists more than two brands, extract the first two only. If only one brand is present, return the second as null/empty fields.
- For URLs, extract only valid URLs explicitly present in the answer (including those in markdown links). Do not invent URLs.
- For "retailers", include only names from {ALLOWED_RETAILERS}. Ignore others (like online-only shops without physical stores).
- Keep all text exactly as in the answer for brand_name and celebrity_founder (except category normalization).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower())


def retailer_specific_urls(all_urls: List[str], retailer_name: str) -> List[str]:
    """Filter URLs that appear to be hosted on the expected retailer's domain."""
    domain = RETAILER_DOMAINS.get(retailer_name)
    if not domain:
        return []
    filtered = []
    for u in all_urls:
        try:
            netloc = urlparse(u).netloc.lower()
            if netloc.endswith(domain):
                filtered.append(u)
        except Exception:
            continue
    return filtered


def pick_first_allowed_category(categories: List[str]) -> Optional[str]:
    """Pick the first category (normalized) that matches allowed categories."""
    # Normalize common synonyms
    norm_map = {
        "cosmetics": "makeup",
        "color cosmetics": "makeup",
        "skin care": "skincare",
        "perfume": "fragrance",
        "cologne": "fragrance",
        "eau de parfum": "fragrance",
        "edp": "fragrance",
        "edt": "fragrance",
        "fragrances": "fragrance",
        "make up": "makeup"
    }
    for c in categories or []:
        c_norm = c.strip().lower()
        c_norm = norm_map.get(c_norm, c_norm)
        if c_norm in ALLOWED_CATEGORIES:
            return c_norm
    return None


def first_k_unique_major_retailers(retailer_names: List[str], k: int = 3) -> List[str]:
    seen = set()
    result = []
    for r in retailer_names or []:
        r_clean = r.strip()
        # Match allowed set case-insensitively; keep original case if it matched
        match = next((a for a in ALLOWED_RETAILERS if a.lower() == r_clean.lower()), None)
        if match and match not in seen:
            seen.add(match)
            result.append(match)
        if len(result) >= k:
            break
    return result


# --------------------------------------------------------------------------- #
# Verification logic per brand                                                #
# --------------------------------------------------------------------------- #
async def verify_brand(evaluator: Evaluator, parent_node, brand: BrandEntry, brand_index: int) -> None:
    """Build the verification sub-tree for a single brand and run checks."""

    # Create the brand main node (non-critical to allow partial credit per brand)
    brand_node = evaluator.add_parallel(
        id=f"brand_{brand_index+1}",
        desc=f"{'First' if brand_index == 0 else 'Second'} celebrity beauty brand meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # 0) Basic presence: brand name exists
    evaluator.add_custom_node(
        result=bool(brand.brand_name and brand.brand_name.strip()),
        id=f"brand_{brand_index+1}_name_present",
        desc="Brand name is provided",
        parent=brand_node,
        critical=True
    )

    # 1) Celebrity founder (critical)
    founder_node = evaluator.add_parallel(
        id=f"brand_{brand_index+1}_celebrity_founder",
        desc="The brand is founded or co-founded by a verified celebrity (actor, singer, musician, influencer, or public figure)",
        parent=brand_node,
        critical=True
    )

    # 1.a) Source existence for founder claim (critical)
    evaluator.add_custom_node(
        result=bool(brand.founder_sources),
        id=f"brand_{brand_index+1}_founder_sources_provided",
        desc="Founder/co-founder claim has at least one official source URL provided in the answer",
        parent=founder_node,
        critical=True
    )

    # 1.b) Verify founder/co-founder claim against sources (critical)
    founder_leaf = evaluator.add_leaf(
        id=f"brand_{brand_index+1}_founder_supported",
        desc=f"Founder/co-founder claim for {brand.brand_name or 'the brand'} is supported by cited sources",
        parent=founder_node,
        critical=True
    )
    founder_claim = f"{brand.celebrity_founder or 'The named celebrity'} is the founder or co-founder of the beauty brand '{brand.brand_name or 'UNKNOWN BRAND'}'."
    await evaluator.verify(
        claim=founder_claim,
        node=founder_leaf,
        sources=brand.founder_sources or None,
        additional_instruction="Accept official confirmation from the brand website, an official press release, a retailer's official brand page, or a reputable profile that explicitly states the founder/co-founder relationship."
    )

    # 2) Multi-retailer availability (critical)
    multi_node = evaluator.add_parallel(
        id=f"brand_{brand_index+1}_multi_retailer_availability",
        desc="The brand is available at three or more major U.S. retail chains with verifiable official documentation",
        parent=brand_node,
        critical=True
    )

    # Determine the three primary retailers from allowed set
    primary_retailers = first_k_unique_major_retailers(brand.retailers, k=3)

    # 2.a) Ensure at least three allowed major retailers are listed (critical)
    evaluator.add_custom_node(
        result=(len(primary_retailers) >= 3),
        id=f"brand_{brand_index+1}_has_three_major_retailers",
        desc="At least three major U.S. retail chains from the allowed list are listed for this brand",
        parent=multi_node,
        critical=True
    )

    # Consolidate all available URLs we can use for retailer checks
    all_brand_urls = (brand.retailer_urls or []) + (brand.reference_urls or [])

    # 2.b) Verify availability at each of the three retailers (each critical)
    for idx, retailer_name in enumerate(primary_retailers[:3], start=1):
        ret_leaf = evaluator.add_leaf(
            id=f"brand_{brand_index+1}_retailer_{idx}_{slugify(retailer_name)}",
            desc=f"{retailer_name} carries the brand '{brand.brand_name or 'UNKNOWN BRAND'}'",
            parent=multi_node,
            critical=True
        )
        # Prefer URLs from the specific retailer's domain; fallback to all provided if none
        candidate_urls = retailer_specific_urls(all_brand_urls, retailer_name)
        if not candidate_urls:
            candidate_urls = all_brand_urls

        retailer_claim = f"The brand '{brand.brand_name or 'UNKNOWN BRAND'}' is currently available for purchase at {retailer_name} in the United States (online or in-store)."
        await evaluator.verify(
            claim=retailer_claim,
            node=ret_leaf,
            sources=candidate_urls or None,
            additional_instruction=(
                "Look for product/category/brand pages on the retailer's official domain that display items from the brand, "
                "or an official retailer announcement. The evidence must be on the retailer's own site or an official announcement."
            )
        )

    # 2.c) Reference URL(s) verifying multi-retailer availability (critical)
    # 2.c.i) Reference URL provided
    evaluator.add_custom_node(
        result=bool(brand.reference_urls),
        id=f"brand_{brand_index+1}_url_reference_provided",
        desc="At least one official reference URL is provided to verify multi-retailer availability",
        parent=multi_node,
        critical=True
    )
    # 2.c.ii) Reference supports multi-retailer availability
    ref_leaf = evaluator.add_leaf(
        id=f"brand_{brand_index+1}_url_reference_support",
        desc="Reference URL(s) explicitly support that the brand is available at multiple major U.S. retail chains",
        parent=multi_node,
        critical=True
    )
    retailers_str = ", ".join(primary_retailers) if primary_retailers else "multiple major U.S. retail chains"
    ref_claim = (
        f"This official page confirms that the brand '{brand.brand_name or 'UNKNOWN BRAND'}' is available at more than one major U.S. retail chain "
        f"(e.g., mentions or links to multiple retailers such as {retailers_str})."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=brand.reference_urls or None,
        additional_instruction=(
            "Accept a brand store locator or official announcement that names multiple retail partners, "
            "or multiple retailer product pages that together show multi-retailer availability. "
            "The page(s) must clearly indicate availability at more than one of the allowed major chains."
        )
    )

    # 3) Product category (critical)
    cat_node = evaluator.add_parallel(
        id=f"brand_{brand_index+1}_product_category",
        desc="The brand offers products in at least one of the following categories: makeup, skincare, or fragrance",
        parent=brand_node,
        critical=True
    )

    chosen_category = pick_first_allowed_category(brand.product_categories or [])

    # 3.a) Category provided and allowed (critical)
    evaluator.add_custom_node(
        result=bool(chosen_category in ALLOWED_CATEGORIES),
        id=f"brand_{brand_index+1}_category_allowed",
        desc="At least one provided product category is among the allowed set (makeup, skincare, fragrance)",
        parent=cat_node,
        critical=True
    )

    # 3.b) Category supported by sources (critical)
    cat_leaf = evaluator.add_leaf(
        id=f"brand_{brand_index+1}_category_supported",
        desc=f"The brand '{brand.brand_name or 'UNKNOWN BRAND'}' offers {chosen_category or 'the claimed'} products",
        parent=cat_node,
        critical=True
    )
    category_claim = f"The brand '{brand.brand_name or 'UNKNOWN BRAND'}' offers {chosen_category or 'makeup/skincare/fragrance'} products."
    category_sources = (brand.category_sources or []) + all_brand_urls
    await evaluator.verify(
        claim=category_claim,
        node=cat_leaf,
        sources=category_sources or None,
        additional_instruction=(
            "Accept evidence on the brand website or major retailer product/category pages that clearly shows products in the stated category. "
            "Allow reasonable synonyms: makeup ~ cosmetics; skincare ~ skin care; fragrance ~ perfume/cologne/eau de parfum."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the celebrity beauty brand multi-retailer availability task.
    """
    # Initialize evaluator (root: parallel aggregation as specified)
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

    # Record allowed retailer list for transparency
    evaluator.add_custom_info(
        info={"allowed_retailers": ALLOWED_RETAILERS, "allowed_categories": ALLOWED_CATEGORIES},
        info_type="constraints",
        info_name="allowed_values"
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="brands_extraction",
    )

    # Normalize to exactly two BrandEntry objects
    brands: List[BrandEntry] = list(extracted.brands or [])
    if len(brands) < 2:
        # Pad with empty entries
        for _ in range(2 - len(brands)):
            brands.append(BrandEntry())
    else:
        brands = brands[:2]

    # Root-level critical check: the two brands must be different
    name1 = (brands[0].brand_name or "").strip().lower()
    name2 = (brands[1].brand_name or "").strip().lower()
    evaluator.add_custom_node(
        result=bool(name1 and name2 and name1 != name2),
        id="brands_are_distinct",
        desc="The two identified brands are different from each other",
        parent=root,
        critical=True
    )

    # Build brand subtrees
    await verify_brand(evaluator, root, brands[0], 0)
    await verify_brand(evaluator, root, brands[1], 1)

    # Return evaluation summary
    return evaluator.get_summary()