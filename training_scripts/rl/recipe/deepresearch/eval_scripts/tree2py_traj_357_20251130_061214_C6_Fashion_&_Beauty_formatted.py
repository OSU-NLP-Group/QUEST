import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "beauty_certifications_retail_availability"
TASK_DESCRIPTION = (
    "Identify three beauty or skincare brands that each hold at least three distinct third-party certifications from "
    "recognized certification bodies (such as B Corp, Leaping Bunny/Cruelty Free International, COSMOS Organic/ECOCERT, "
    "Vegan Society, PETA's Beauty Without Bunnies, or EWG Verified) and are available for purchase at physical Sephora "
    "or Ulta Beauty store locations in California, Texas, and Florida. For each brand, provide: (1) The brand name and "
    "official brand website URL; (2) The names of at least three distinct third-party certifications the brand currently "
    "holds, with reference URLs from official certification body websites or the brand's official certification "
    "documentation pages; (3) The retailer name (Sephora or Ulta Beauty) where the brand is sold; (4) At least one "
    "specific product from the brand that carries these certifications; (5) One specific store location with complete "
    "address in each of the three states (California, Texas, and Florida) where the brand's products are available for "
    "in-store purchase, with reference URLs to the official retailer's store locator or store listing pages. All "
    "certifications must be currently active and verifiable. Each brand must be distinct from the others, and the three "
    "certifications for each brand must come from different certification programs."
)
CURRENT_DATE_STR = "2026-01-11"  # For "currently active" phrasing

# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class CertificationEntry(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class BrandEntry(BaseModel):
    brand_name: Optional[str] = None
    brand_website: Optional[str] = None
    certifications: List[CertificationEntry] = Field(default_factory=list)

    retailer: Optional[str] = None  # "Sephora" or "Ulta Beauty"
    product_name: Optional[str] = None
    product_urls: List[str] = Field(default_factory=list)

    store_ca_address: Optional[str] = None
    store_ca_urls: List[str] = Field(default_factory=list)

    store_tx_address: Optional[str] = None
    store_tx_urls: List[str] = Field(default_factory=list)

    store_fl_address: Optional[str] = None
    store_fl_urls: List[str] = Field(default_factory=list)


class BrandsExtraction(BaseModel):
    brands: List[BrandEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_brands() -> str:
    return """
Extract up to 5 candidate beauty/skincare brand entries from the answer text. For each entry, return:

- brand_name: The brand's name (string).
- brand_website: The official brand website URL (e.g., https://brand.com). Use the exact URL provided in the answer. If multiple, pick the most official.
- certifications: An array of objects. Each object must have:
  - name: The certification program name (e.g., "B Corp", "Leaping Bunny", "COSMOS Organic", "ECOCERT", "Vegan Society", "PETA Beauty Without Bunnies", "EWG Verified", etc.).
  - reference_urls: An array of one or more URLs that directly support that certification (official certification body directory pages and/or the brand’s official certification documentation pages). Extract only URLs present in the answer text.
- retailer: One of ["Sephora", "Ulta Beauty"]. Use exactly this naming if the answer claims one retailer.
- product_name: At least one specific product from the brand that (as claimed) carries these certifications.
- product_urls: All product-related URLs present in the answer (brand product page, retailer product page, or certification program page listing the product, if provided).
- store_ca_address: One specific physical store address in California where the brand’s products are available for in-store purchase (as claimed).
- store_ca_urls: One or more official retailer store-locator or store-page URL(s) for that California store (from the answer).
- store_tx_address: One specific physical store address in Texas where the brand’s products are available for in-store purchase (as claimed).
- store_tx_urls: One or more official retailer store-locator or store-page URL(s) for that Texas store (from the answer).
- store_fl_address: One specific physical store address in Florida where the brand’s products are available for in-store purchase (as claimed).
- store_fl_urls: One or more official retailer store-locator or store-page URL(s) for that Florida store (from the answer).

Rules:
- Only extract information explicitly present in the answer text. Do not invent or infer missing items.
- Return null for missing scalars and [] for missing arrays.
- Preserve URLs exactly as shown; normalize by prepending http:// if protocol is missing.
- If more than 3 brands are provided, still extract them; the evaluator may only use the first three.
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def is_non_empty_string(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""

def is_valid_url(u: Optional[str]) -> bool:
    if not is_non_empty_string(u):
        return False
    try:
        if not _URL_RE.search(u.strip()):
            return False
        parsed = urlparse(u.strip())
        return bool(parsed.scheme) and bool(parsed.netloc)
    except Exception:
        return False

def get_domain(u: str) -> str:
    try:
        netloc = urlparse(u).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def collect_cert_urls(brand: BrandEntry) -> List[str]:
    urls: List[str] = []
    for c in brand.certifications:
        urls.extend([x for x in c.reference_urls if is_valid_url(x)])
    # de-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for x in urls:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def union_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for x in lst:
            if is_valid_url(x) and x not in seen:
                out.append(x)
                seen.add(x)
    return out

def normalize_brand_name(name: Optional[str]) -> str:
    if not is_non_empty_string(name):
        return ""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(inc|llc|ltd|beauty|skincare|cosmetics|co|company|corp|corporation)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Canonical mapping for certification program distinctness normalization
def normalize_cert_program(name: Optional[str]) -> str:
    if not is_non_empty_string(name):
        return ""
    s = name.lower().strip()

    # Common normalizations
    mapping = [
        (["b corp", "bcorporation", "b corporation", "certified b corp", "certified b corporation"], "b corp"),
        (["leaping bunny", "cruelty free international", "cfi leaping bunny"], "leaping bunny"),
        (["cosmos", "cosmos organic", "cosmos natural", "ecocert", "cosmos/ecocert", "ecocert cosmos"], "cosmos/ecocert"),
        (["vegan society", "the vegan society", "vegan trademark"], "vegan society"),
        (["peta", "beauty without bunnies", "peta beauty without bunnies"], "peta beauty without bunnies"),
        (["ewg verified", "ewg"], "ewg verified"),
        (["made safe", "madesafe"], "made safe"),
        (["natrue"], "natrue"),
        (["nsf", "nsf/ansi"], "nsf"),
        (["usda organic", "usda"], "usda organic"),
    ]
    for keys, canon in mapping:
        for k in keys:
            if k in s:
                return canon
    # Generic cleanup
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def count_distinct_programs(cert_names: List[str]) -> int:
    normalized = [normalize_cert_program(n) for n in cert_names if is_non_empty_string(n)]
    unique = set([n for n in normalized if n])
    return len(unique)

RECOGNIZED_CERT_DOMAINS = {
    "bcorporation.net",
    "crueltyfreeinternational.org",
    "leapingbunny.org",
    "ecocert.com",
    "cosmos-standard.org",
    "soilassociation.org",
    "vegansociety.com",
    "peta.org",
    "ewg.org",
    "madesafe.org",
    "natrue.org",
    "nsf.org",
    # Retailers for store pages:
    "sephora.com",
    "ulta.com",
    "ulta-beauty.com",  # rarely appears
}

def has_official_cert_reference_for_each(brand: BrandEntry) -> bool:
    brand_dom = get_domain(brand.brand_website) if is_valid_url(brand.brand_website or "") else ""
    for cert in brand.certifications:
        refs = [x for x in cert.reference_urls if is_valid_url(x)]
        if len(refs) == 0:
            return False
        # At least one official-looking URL per certification: certification body or brand domain
        found_official = False
        for r in refs:
            d = get_domain(r)
            if d == brand_dom or any(d.endswith(k) or k in d for k in RECOGNIZED_CERT_DOMAINS):
                found_official = True
                break
        if not found_official:
            return False
    return True

def retailer_is_valid(retailer: Optional[str]) -> bool:
    if not is_non_empty_string(retailer):
        return False
    s = retailer.strip().lower()
    return s in {"sephora", "ulta", "ulta beauty"}

def addresses_present(brand: BrandEntry) -> bool:
    return all([
        is_non_empty_string(brand.store_ca_address),
        is_non_empty_string(brand.store_tx_address),
        is_non_empty_string(brand.store_fl_address),
    ])

def build_cert_names_string(brand: BrandEntry) -> str:
    names = [c.name for c in brand.certifications if is_non_empty_string(c.name)]
    return ", ".join(names[:10])

def first_k_brands(brands: List[BrandEntry], k: int = 3) -> List[BrandEntry]:
    if len(brands) >= k:
        return brands[:k]
    # pad with empty entries to length k
    out = brands[:]
    for _ in range(k - len(out)):
        out.append(BrandEntry())
    return out


# -----------------------------------------------------------------------------
# Verification per brand
# -----------------------------------------------------------------------------
async def verify_brand(evaluator: Evaluator, parent_node, brand: BrandEntry, idx: int) -> None:
    brand_idx = idx + 1
    brand_node = evaluator.add_parallel(
        id=f"brand_{idx+1}",
        desc=f"Brand {brand_idx} entry meets all requirements",
        parent=parent_node,
        critical=False,
    )

    # 1) Identity: brand name + official site present (existence check)
    identity_ok = is_non_empty_string(brand.brand_name) and is_valid_url(brand.brand_website)
    evaluator.add_custom_node(
        result=identity_ok,
        id=f"brand_{idx+1}_identity",
        desc="Provides brand name AND official brand website URL",
        parent=brand_node,
        critical=True
    )

    # 2) Certifications group (critical)
    certs_group = evaluator.add_parallel(
        id=f"brand_{idx+1}_certifications",
        desc="Provides certification information meeting all certification constraints",
        parent=brand_node,
        critical=True
    )

    # 2.1) Count distinct programs (custom, critical)
    cert_names = [c.name for c in brand.certifications if is_non_empty_string(c.name)]
    distinct_count = count_distinct_programs(cert_names)
    evaluator.add_custom_node(
        result=(distinct_count >= 3),
        id=f"brand_{idx+1}_cert_count_distinct",
        desc="Lists at least three DISTINCT third-party certifications from different certification programs",
        parent=certs_group,
        critical=True
    )

    # 2.2) Recognized bodies (verify via URLs; critical)
    recognized_node = evaluator.add_leaf(
        id=f"brand_{idx+1}_cert_recognized_bodies",
        desc="Certifications are from recognized third-party certification bodies/programs (e.g., B Corp, Leaping Bunny/CFI, COSMOS/ECOCERT, Vegan Society, PETA Beauty Without Bunnies, EWG Verified, or other legitimate third-party programs)",
        parent=certs_group,
        critical=True
    )
    cert_urls_union = collect_cert_urls(brand)
    cert_names_str = build_cert_names_string(brand)
    claim_recognized = (
        f"The certifications listed for brand '{brand.brand_name}' ({cert_names_str}) are recognized third-party "
        "certification programs (e.g., B Corp, Leaping Bunny/CFI, COSMOS/ECOCERT, Vegan Society, PETA Beauty Without "
        "Bunnies, EWG Verified, or another legitimate third-party program)."
    )
    await evaluator.verify(
        claim=claim_recognized,
        node=recognized_node,
        sources=cert_urls_union,
        additional_instruction="Judge based on whether the provided URLs are official certification bodies or official program pages that clearly indicate a recognized third-party certification program."
    )

    # 2.3) Active and verifiable (verify via URLs; critical)
    active_node = evaluator.add_leaf(
        id=f"brand_{idx+1}_cert_active_verifiable",
        desc="All listed certifications are currently active and verifiable",
        parent=certs_group,
        critical=True
    )
    claim_active = (
        f"As of {CURRENT_DATE_STR}, the listed certifications for brand '{brand.brand_name}' are currently active and "
        "verifiable (either as brand-level certifications or via certified products under the brand)."
    )
    await evaluator.verify(
        claim=claim_active,
        node=active_node,
        sources=cert_urls_union,
        additional_instruction="Check certificate directories or official program pages for evidence that the certification is current/active (e.g., brand or product is listed as certified)."
    )

    # 2.4) Official references for each certification (custom heuristic + presence; critical)
    refs_ok = has_official_cert_reference_for_each(brand)
    evaluator.add_custom_node(
        result=refs_ok,
        id=f"brand_{idx+1}_cert_references",
        desc="Provides official reference URL(s) for each listed certification (certification body site and/or brand official certification documentation)",
        parent=certs_group,
        critical=True
    )

    # 3) Retail + in-store locations group (critical)
    retail_group = evaluator.add_parallel(
        id=f"brand_{idx+1}_retail_in_store",
        desc="Provides retailer + in-store availability evidence for CA/TX/FL",
        parent=brand_node,
        critical=True
    )

    # 3.1) Retailer specified (custom, critical)
    evaluator.add_custom_node(
        result=retailer_is_valid(brand.retailer),
        id=f"brand_{idx+1}_retailer",
        desc="Specifies the retailer as Sephora OR Ulta Beauty",
        parent=retail_group,
        critical=True
    )

    # 3.2) Product carries certifications (verify; critical)
    product_node = evaluator.add_leaf(
        id=f"brand_{idx+1}_product",
        desc="Identifies at least one specific product from the brand that carries the cited certifications (as claimed in the response)",
        parent=retail_group,
        critical=True
    )
    product_sources = union_sources(brand.product_urls, cert_urls_union)
    # Use a robust claim that allows brand-level certifications to apply to products when appropriate.
    claim_product = (
        f"The product '{brand.product_name}' from brand '{brand.brand_name}' is presented on the provided sources as "
        "certified by one or more of the listed certification programs (brand-level or product-level evidence is acceptable "
        "if clearly indicated)."
    )
    await evaluator.verify(
        claim=claim_product,
        node=product_node,
        sources=product_sources,
        additional_instruction="Accept if the product page itself or official certification directory clearly indicates that this product or the brand covering this product is certified by at least one of the listed programs."
    )

    # 3.3) Store locations addresses provided (custom, critical)
    evaluator.add_custom_node(
        result=addresses_present(brand),
        id=f"brand_{idx+1}_store_locations_addresses",
        desc="Provides ONE specific physical store location with COMPLETE address in EACH state: California, Texas, and Florida, where the brand is available for in-store purchase",
        parent=retail_group,
        critical=True
    )

    # 3.4) Store locations references (verify; critical)
    store_refs_node = evaluator.add_leaf(
        id=f"brand_{idx+1}_store_locations_references",
        desc="Provides official retailer store-locator/store-listing reference URL(s) supporting the CA/TX/FL store location claims",
        parent=retail_group,
        critical=True
    )
    store_sources = union_sources(brand.store_ca_urls, brand.store_tx_urls, brand.store_fl_urls)
    claim_store = (
        f"For retailer '{brand.retailer}', the provided store-locator/store-page URLs show valid physical store locations "
        f"with addresses in California ('{brand.store_ca_address}'), Texas ('{brand.store_tx_address}'), and Florida "
        f"('{brand.store_fl_address}')."
    )
    await evaluator.verify(
        claim=claim_store,
        node=store_refs_node,
        sources=store_sources,
        additional_instruction="Confirm each URL is an official Sephora or Ulta store-locator or store page that lists the corresponding store's address. Minor formatting differences are acceptable."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------
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

    # Extract candidate brands
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="brands_extraction",
    )

    # Only verify first 3 (pad if fewer)
    brands = first_k_brands(extracted.brands, k=3)

    # Build brand subtrees
    for i, brand in enumerate(brands):
        await verify_brand(evaluator, root, brand, i)

    # Cross-brand distinctness (critical)
    brand_names_norm = [normalize_brand_name(b.brand_name) for b in brands]
    all_distinct = len([bn for bn in brand_names_norm if bn]) == len(set([bn for bn in brand_names_norm if bn])) and \
                   all(is_non_empty_string(b.brand_name) for b in brands)
    evaluator.add_custom_node(
        result=all_distinct,
        id="cross_brand_distinctness",
        desc="All three brands are distinct from one another (no duplicate brand names)",
        parent=root,
        critical=True
    )

    return evaluator.get_summary()