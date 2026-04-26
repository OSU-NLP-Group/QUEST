import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sustainable_fashion_brands_eu_gots_bcorp_retail"
TASK_DESCRIPTION = (
    "Identify four European fashion brands that meet the following sustainability and operational requirements:\n\n"
    "1. Each brand must hold current GOTS (Global Organic Textile Standard) certification, with certified products containing at least 70% organic fibers for 'made with organic' designation or at least 95% for 'organic' designation.\n\n"
    "2. Each brand must be a certified B Corporation with a verified B Impact Assessment score of at least 80 out of 200 points.\n\n"
    "3. Each brand must be headquartered in or have significant operations within Europe.\n\n"
    "4. Each brand must operate at least one physical retail store location. Preference is given to stores that meet or exceed 1,000 square feet (approximately 93 square meters) in size, typical of small boutique retail spaces.\n\n"
    "For each brand, provide:\n"
    "- The brand name\n"
    "- Documentation or reference URLs confirming GOTS certification and organic fiber percentage\n"
    "- Documentation or reference URLs confirming B Corp certification and score\n"
    "- Documentation confirming European location\n"
    "- Information about physical retail store locations and sizes where available"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreLocation(BaseModel):
    store_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    # Raw size text as stated in the answer (e.g., "1,200 sq ft", "95 m²")
    size_text: Optional[str] = None
    # Optional parsed numeric strings (if provided by the answer); keep as strings for robustness
    size_sqft: Optional[str] = None
    size_sqm: Optional[str] = None
    store_urls: List[str] = Field(default_factory=list)


class BrandItem(BaseModel):
    brand_name: Optional[str] = None

    # Evidence that this is a fashion/apparel brand (brand site, product pages, etc.)
    fashion_brand_urls: List[str] = Field(default_factory=list)

    # GOTS evidence and details
    gots_urls: List[str] = Field(default_factory=list)
    gots_designation: Optional[str] = None  # "organic" or "made with organic" (if provided)
    gots_percent: Optional[str] = None      # e.g., "95%", "70 percent" (if provided)

    # B Corp evidence and score
    bcorp_urls: List[str] = Field(default_factory=list)
    bcorp_score: Optional[str] = None       # e.g., "84.6" (if provided)

    # European HQ/operations evidence
    europe_urls: List[str] = Field(default_factory=list)
    headquarters_text: Optional[str] = None

    # Stores
    store_locations: List[StoreLocation] = Field(default_factory=list)

    # Catch-all additional URLs cited by the answer for this brand
    additional_urls: List[str] = Field(default_factory=list)


class BrandsExtraction(BaseModel):
    brands: List[BrandItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return """
Extract up to FOUR (4) distinct fashion/apparel brands exactly as presented in the answer. For each brand, extract the following fields strictly from the answer text:

- brand_name: The brand name.
- fashion_brand_urls: A list of URLs that demonstrate the brand sells fashion/apparel (e.g., official site, product categories for clothing/footwear/accessories, reputable profiles).
- gots_urls: A list of URLs indicating current GOTS certification or GOTS-certified products (e.g., GOTS database entries, official brand pages, credible certifier pages, product listings explicitly mentioning GOTS).
- gots_designation: One of "organic" or "made with organic" if the answer explicitly provides such a GOTS label; otherwise null.
- gots_percent: The organic fiber percentage mentioned for the GOTS-certified product(s) in the answer (e.g., "95%", "at least 70%"). If not stated, set to null.
- bcorp_urls: A list of URLs demonstrating B Corp certification (e.g., bcorporation.net listing, official pages).
- bcorp_score: The B Impact Assessment score as stated in the answer text (numeric string if available, e.g., "84.6"). If not provided, set to null.
- europe_urls: A list of URLs that demonstrate the brand is headquartered in Europe or has significant operations in Europe (e.g., company page with HQ address in a European country, store locator pages listing European locations, Wikipedia/company pages).
- headquarters_text: If the answer text itself mentions the HQ city/country, copy that text here; otherwise null.
- store_locations: A list of up to three store objects from the answer. Each store includes:
    - store_name (if stated),
    - address (if provided),
    - city,
    - country,
    - size_text (e.g., "1,200 sq ft" or "95 m²", if stated),
    - size_sqft (numeric string like "1200" if the answer gives a sq ft number; otherwise null),
    - size_sqm (numeric string like "93" if the answer gives a m² number; otherwise null),
    - store_urls (a list of URLs that show the store exists: store locator pages, press articles, Google Maps business pages, etc.)
- additional_urls: Any other URLs cited for this brand that do not fit the above categories.

Important rules:
1) Extract ONLY what is explicitly in the answer; do not invent or infer anything.
2) If the answer lists more than 4 brands, keep only the first 4 in order.
3) If the answer provides fewer than 4 brands, include all provided; for missing brands, return an empty placeholder.
4) For URL lists, include every URL cited for the corresponding purpose. If none are present, return an empty list.
5) Keep numbers as strings when uncertain (e.g., "about 95").
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_brand_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9&+\- ]", "", s)  # keep alnum and common punctuation in brand names
    s = re.sub(r"\s+", " ", s)
    return s or None


def _parse_number_from_text(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    # Extract first numeric (with optional decimal) from string
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _size_text_to_sqft(size_text: Optional[str]) -> Optional[float]:
    if not size_text:
        return None
    txt = size_text.lower().replace(",", "").strip()

    # Try to detect explicit sqft first
    if "sq ft" in txt or "sqft" in txt or "ft²" in txt or "square feet" in txt:
        val = _parse_number_from_text(txt)
        return val if val and val > 0 else None

    # Detect m2/m² and convert to sq ft
    if "m2" in txt or "m²" in txt or "square meters" in txt or "square metres" in txt or "sqm" in txt:
        val = _parse_number_from_text(txt)
        if val and val > 0:
            return val * 10.7639

    return None


def _store_size_to_sqft(store: StoreLocation) -> Optional[float]:
    # Prefer explicit numeric fields if provided
    if store.size_sqft:
        v = _parse_number_from_text(store.size_sqft)
        if v and v > 0:
            return v
    if store.size_sqm:
        v = _parse_number_from_text(store.size_sqm)
        if v and v > 0:
            return v * 10.7639
    # Fallback to parsing from size_text
    return _size_text_to_sqft(store.size_text)


def _combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for url in lst or []:
            u = (url or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _designation_threshold(designation: Optional[str]) -> Optional[int]:
    if not designation:
        return None
    d = designation.strip().lower()
    if d == "organic":
        return 95
    if d == "made with organic":
        return 70
    return None


def _score_at_least_80(score_text: Optional[str]) -> Tuple[bool, Optional[float]]:
    val = _parse_number_from_text(score_text)
    if val is None:
        return False, None
    return (val >= 80.0), val


# --------------------------------------------------------------------------- #
# Verification per brand                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_brand(
    evaluator: Evaluator,
    parent_node,
    brand: BrandItem,
    index: int,
) -> None:
    idx1 = index + 1
    idx_label = f"{idx1}{'st' if idx1==1 else 'nd' if idx1==2 else 'rd' if idx1==3 else 'th'}"
    brand_desc = f"{idx_label} identified brand (evaluated independently for partial credit)."

    # Brand node (parallel, non-critical to allow partial credit per brand)
    brand_node = evaluator.add_parallel(
        id=f"brand_{idx1}",
        desc=brand_desc,
        parent=parent_node,
        critical=False,
    )

    # 1) Brand name provided (Critical)
    name_present = bool(brand.brand_name and brand.brand_name.strip())
    evaluator.add_custom_node(
        result=name_present,
        id=f"brand_{idx1}_Brand_Name_Provided",
        desc="Provides the brand name.",
        parent=brand_node,
        critical=True,
    )

    # 2) Is fashion/apparel brand (Critical) — verify using any relevant URLs
    fashion_sources = _combine_sources(
        brand.fashion_brand_urls,
        brand.additional_urls,
        brand.europe_urls,
        # store pages often demonstrate physical retail for a fashion brand
        *[loc.store_urls for loc in brand.store_locations],
    )
    leaf_is_fashion = evaluator.add_leaf(
        id=f"brand_{idx1}_Is_Fashion_Brand",
        desc="Brand is a fashion/apparel brand (e.g., sells clothing/footwear/accessories), evidenced by the response description or a cited source.",
        parent=brand_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The brand '{brand.brand_name or 'UNKNOWN'}' is a fashion/apparel brand that sells clothing, footwear, or accessories.",
        node=leaf_is_fashion,
        sources=fashion_sources if fashion_sources else None,
        additional_instruction=(
            "Accept if any provided source shows the brand sells fashion/apparel products (e.g., product "
            "pages, category pages, or reputable profiles). If no sources are provided, judge using the answer text only."
        ),
    )

    # 3) GOTS checks (grouped to gate with presence inside group only)
    gots_group = evaluator.add_parallel(
        id=f"brand_{idx1}_gots_main",
        desc="GOTS certification checks",
        parent=brand_node,
        critical=False,
    )
    # 3a) GOTS URLs present (Critical within GOTS group)
    gots_urls_present = evaluator.add_custom_node(
        result=bool(brand.gots_urls),
        id=f"brand_{idx1}_gots_urls_present",
        desc="GOTS evidence URLs are provided.",
        parent=gots_group,
        critical=True,
    )
    # 3b) Current GOTS certification (Critical)
    leaf_gots_current = evaluator.add_leaf(
        id=f"brand_{idx1}_GOTS_Certification_Current_With_URL",
        desc="Provides URL(s) evidencing the brand/company holds current GOTS certification (or that relevant products are currently GOTS certified).",
        parent=gots_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The brand '{brand.brand_name or 'UNKNOWN'}' holds current GOTS certification, or its products are currently GOTS-certified.",
        node=leaf_gots_current,
        sources=brand.gots_urls if brand.gots_urls else None,
        additional_instruction=(
            "Look for explicit evidence on the cited pages (e.g., GOTS public database entries, credible certifier "
            "pages, brand pages identifying products as GOTS-certified). Reject if the sources do not confirm current GOTS status."
        ),
    )
    # 3c) GOTS organic fiber threshold met (Critical)
    desig = (brand.gots_designation or "").strip().lower() if brand.gots_designation else None
    threshold = _designation_threshold(desig)
    # Formulate a robust claim even if the answer did not explicitly provide percentage
    if threshold is not None:
        threshold_clause = f"for the label '{desig}' (≥{threshold}% organic fiber)"
    else:
        threshold_clause = "for the appropriate GOTS label (≥70% for 'made with organic' or ≥95% for 'organic')"
    leaf_gots_threshold = evaluator.add_leaf(
        id=f"brand_{idx1}_GOTS_Organic_Fiber_Threshold_Met",
        desc="Confirms (with citation/URL) that the GOTS-certified product(s) referenced meet ≥70% organic fiber for 'made with organic' OR ≥95% for 'organic'.",
        parent=gots_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The cited sources show that the referenced GOTS-certified product(s) from '{brand.brand_name or 'UNKNOWN'}' "
            f"meet the minimum organic fiber threshold {threshold_clause}."
        ),
        node=leaf_gots_threshold,
        sources=brand.gots_urls if brand.gots_urls else None,
        additional_instruction=(
            "Accept if the page explicitly states the organic fiber percentage meeting the required GOTS threshold, "
            "or if it explicitly uses the GOTS label 'organic' (95%+) or 'made with organic' (70%+). Prefer explicit statements."
        ),
    )

    # 4) B Corp checks (grouped)
    bcorp_group = evaluator.add_parallel(
        id=f"brand_{idx1}_bcorp_main",
        desc="B Corp certification checks",
        parent=brand_node,
        critical=False,
    )
    # 4a) B Corp URLs present (Critical within B Corp group)
    evaluator.add_custom_node(
        result=bool(brand.bcorp_urls),
        id=f"brand_{idx1}_bcorp_urls_present",
        desc="B Corp evidence URLs are provided.",
        parent=bcorp_group,
        critical=True,
    )
    # 4b) Certified B Corporation (Critical)
    leaf_bcorp_cert = evaluator.add_leaf(
        id=f"brand_{idx1}_B_Corp_Certified_With_URL",
        desc="Provides URL(s) evidencing the brand/company is a certified B Corporation.",
        parent=bcorp_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The brand '{brand.brand_name or 'UNKNOWN'}' is a certified B Corporation.",
        node=leaf_bcorp_cert,
        sources=brand.bcorp_urls if brand.bcorp_urls else None,
        additional_instruction=(
            "Prefer official B Lab directory pages (bcorporation.net) showing certification. Reject if sources "
            "do not confirm certification."
        ),
    )
    # 4c) B Impact score provided (Critical within B Corp group; rubric requires providing the score)
    score_provided = bool(brand.bcorp_score and brand.bcorp_score.strip())
    evaluator.add_custom_node(
        result=score_provided,
        id=f"brand_{idx1}_bimpact_score_provided",
        desc="B Impact Assessment score is provided in the answer.",
        parent=bcorp_group,
        critical=True,
    )
    # 4d) B Impact score ≥ 80 (Critical)
    at_least_80, parsed_score = _score_at_least_80(brand.bcorp_score)
    leaf_bscore = evaluator.add_leaf(
        id=f"brand_{idx1}_B_Impact_Score_At_Least_80_With_URL",
        desc="Provides the verified B Impact Assessment score and shows it is ≥80/200 (with citation/URL).",
        parent=bcorp_group,
        critical=True,
    )
    if parsed_score is not None:
        score_claim = (
            f"The brand '{brand.brand_name or 'UNKNOWN'}' has a verified B Impact Assessment score of "
            f"{parsed_score:.2f}, which is at least 80 out of 200."
        )
    else:
        score_claim = (
            f"At least one cited source shows the brand '{brand.brand_name or 'UNKNOWN'}' has a verified "
            f"B Impact Assessment score of at least 80 out of 200."
        )
    await evaluator.verify(
        claim=score_claim,
        node=leaf_bscore,
        sources=brand.bcorp_urls if brand.bcorp_urls else None,
        additional_instruction=(
            "Verify the numeric score (≥80) from the cited sources. Prefer official B Lab directory or other "
            "credible listings. Reject if score < 80 or not evidenced."
        ),
    )

    # 5) European HQ/operations (grouped)
    europe_group = evaluator.add_parallel(
        id=f"brand_{idx1}_europe_main",
        desc="European HQ/operations checks",
        parent=brand_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=bool(brand.europe_urls),
        id=f"brand_{idx1}_europe_urls_present",
        desc="European HQ/operations evidence URLs are provided.",
        parent=europe_group,
        critical=True,
    )
    leaf_europe = evaluator.add_leaf(
        id=f"brand_{idx1}_European_Location_With_Documentation",
        desc="Provides documentation/URL showing the brand is headquartered in Europe OR has significant operations within Europe.",
        parent=europe_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The brand '{brand.brand_name or 'UNKNOWN'}' is headquartered in a European country or has significant "
            f"operations within Europe."
        ),
        node=leaf_europe,
        sources=brand.europe_urls if brand.europe_urls else None,
        additional_instruction=(
            "Accept if the cited pages show a European HQ address, corporate registration in Europe, or substantial "
            "European operations (multiple stores/offices)."
        ),
    )

    # 6) Physical retail store location (Critical): verify at least one store
    # Presence (name/address/city/country OR store URLs)
    has_any_store = any(
        (s.address or s.city or s.country or s.store_name or s.store_urls) for s in brand.store_locations
    )
    # We add a small sequential sub-structure to gate verification on presence
    store_group = evaluator.add_sequential(
        id=f"brand_{idx1}_store_main",
        desc="Physical retail store verification",
        parent=brand_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=has_any_store,
        id=f"brand_{idx1}_store_location_present",
        desc="At least one store location is provided (name/address/city/country or store URL).",
        parent=store_group,
        critical=True,
    )

    # Build a representative example for the claim
    example_city = None
    example_country = None
    example_addr = None
    example_urls: List[str] = []
    for s in brand.store_locations:
        if not example_city and s.city:
            example_city = s.city
        if not example_country and s.country:
            example_country = s.country
        if not example_addr and s.address:
            example_addr = s.address
        if s.store_urls:
            example_urls.extend(s.store_urls)
    # Fallback sources if specific store URLs are missing: use general brand/store locator URLs
    store_verify_sources = _combine_sources(
        example_urls,
        brand.fashion_brand_urls,
        brand.additional_urls,
        brand.europe_urls,
    )

    leaf_store = evaluator.add_leaf(
        id=f"brand_{idx1}_Physical_Retail_Store_Location_Provided",
        desc="Provides at least one physical retail store location (e.g., address/city/store name), satisfying the requirement that the brand operates at least one physical retail store.",
        parent=store_group,
        critical=True,
    )
    example_bits = []
    if example_addr:
        example_bits.append(example_addr)
    if example_city:
        example_bits.append(example_city)
    if example_country:
        example_bits.append(example_country)
    example_loc_txt = ", ".join(example_bits) if example_bits else "a listed location"

    await evaluator.verify(
        claim=(
            f"The brand '{brand.brand_name or 'UNKNOWN'}' operates at least one physical retail store, e.g., at {example_loc_txt}."
        ),
        node=leaf_store,
        sources=store_verify_sources if store_verify_sources else None,
        additional_instruction=(
            "Accept if any cited source confirms a physical store (address, city, or storefront details). "
            "Store locator or city-specific store pages are acceptable."
        ),
    )

    # 7) Store size info provided (Non-Critical): succeeds if any store has size info in the answer
    has_size_info = any(
        (s.size_text or s.size_sqft or s.size_sqm) for s in brand.store_locations
    )
    evaluator.add_custom_node(
        result=has_size_info,
        id=f"brand_{idx1}_Store_Size_Information_Provided",
        desc="Includes store size information for at least one store (if provided in the response).",
        parent=brand_node,
        critical=False,
    )

    # 8) Notes 1000 sqft preference (Non-Critical): if any size is ≥ 1000 sq ft (≈ 93 m²)
    meets_1000_sqft = False
    for s in brand.store_locations:
        sqft = _store_size_to_sqft(s)
        if sqft and sqft >= 1000.0:
            meets_1000_sqft = True
            break
    # If no size provided, we consider preference not met (False). It's optional (non-critical).
    evaluator.add_custom_node(
        result=meets_1000_sqft,
        id=f"brand_{idx1}_Notes_1000_sqft_Preference_When_Size_Given",
        desc="If store size is provided, indicates whether any listed store meets/exceeds 1,000 sq ft (~93 m²) (preference only).",
        parent=brand_node,
        critical=False,
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    # Initialize evaluator with root as PARALLEL aggregation (allow per-brand partial credit)
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

    # Create top-level evaluation node (parallel)
    top_node = evaluator.add_parallel(
        id="Sustainable_Fashion_Brands_Evaluation",
        desc="Evaluate four identified European fashion brands against the stated sustainability/operational requirements and required reporting fields, allowing partial credit per brand.",
        parent=root,
        critical=False,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="brands_extraction",
    )

    # Normalize: keep exactly 4 items (pad with empty)
    brands: List[BrandItem] = list(extracted.brands or [])
    if len(brands) > 4:
        brands = brands[:4]
    while len(brands) < 4:
        brands.append(BrandItem())

    # Critical check: four distinct brands (by normalized names)
    normalized_names = [
        _normalize_brand_name(b.brand_name) for b in brands if b.brand_name
    ]
    # Valid names are those that are non-empty after normalization
    valid_names = [n for n in normalized_names if n]
    unique_names = set(valid_names)
    four_distinct = len(valid_names) == 4 and len(unique_names) == 4

    evaluator.add_custom_node(
        result=four_distinct,
        id="Four_Distinct_Brands",
        desc="Provides four distinct brands (no duplicates).",
        parent=top_node,
        critical=True,
    )

    # Verify each brand (parallel children under top_node)
    for i in range(4):
        await verify_single_brand(evaluator, top_node, brands[i], i)

    # Return final summary (score + verification tree)
    return evaluator.get_summary()