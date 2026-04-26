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
TASK_ID = "beauty_sustainability_brands_v1"
TASK_DESCRIPTION = """
Identify four distinct beauty or personal care brands available in the United States, where each brand meets a specific combination of sustainability and ethical certifications. The four brands must satisfy the following requirements:

Brand 1: Must be B Corp certified, Leaping Bunny certified for cruelty-free status, and use 100% post-consumer recycled (PCR) plastic or other fully recyclable packaging materials (glass or aluminum) for their primary product containers.

Brand 2: Must be B Corp certified, hold vegan certification from a recognized certification body (such as PETA, The Vegan Society, or BeVeg), and be Leaping Bunny certified for cruelty-free status.

Brand 3: Must be B Corp certified, hold Fair Trade certification for products or ingredients, and be Leaping Bunny certified for cruelty-free status.

Brand 4: Must be B Corp certified and hold Carbon Neutral or Climate Neutral certification, or have documented and verified climate-positive commitments.

For each brand, provide:
- The complete brand name
- Official brand website URL
- Verification URLs for each required certification from official certification bodies or directories
- A URL confirming US market availability

All certifications must be current and valid as of 2025-2026. Each brand must be distinct and cannot be counted toward multiple requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BrandData(BaseModel):
    name: Optional[str] = None
    website_url: Optional[str] = None
    category: Optional[str] = None  # e.g., 'beauty', 'personal care', 'skincare', 'cosmetics', etc.

    # Certification/verification URLs
    bcorp_urls: List[str] = Field(default_factory=list)
    leaping_bunny_urls: List[str] = Field(default_factory=list)
    vegan_urls: List[str] = Field(default_factory=list)
    fair_trade_urls: List[str] = Field(default_factory=list)
    climate_urls: List[str] = Field(default_factory=list)

    # Brand 1 packaging requirement
    packaging_statement: Optional[str] = None
    packaging_urls: List[str] = Field(default_factory=list)

    # US availability evidence
    us_urls: List[str] = Field(default_factory=list)


class AllBrandsExtraction(BaseModel):
    brand1: Optional[BrandData] = None
    brand2: Optional[BrandData] = None
    brand3: Optional[BrandData] = None
    brand4: Optional[BrandData] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return """
    Extract the four brands and their required fields from the answer text.

    GENERAL RULES:
    - Extract ONLY what is explicitly present in the answer text.
    - For any missing item, return null (for strings) or [] (for url arrays).
    - For URLs:
      • Extract full URLs only, no placeholders.
      • Include protocol; if missing, prepend http://
      • Prefer official certification body/directory URLs for verification fields.

    DATA TO EXTRACT (JSON with four top-level objects: brand1, brand2, brand3, brand4):

    Each brand object must contain:
    - name: Complete brand name as stated.
    - website_url: Official brand website URL.
    - category: Short phrase indicating sector (e.g., "beauty", "personal care", "cosmetics", "skincare", etc.).

    - bcorp_urls: [array of URLs] Official B Corp directory/evidence URLs.
    - leaping_bunny_urls: [array of URLs] Official Leaping Bunny or Cruelty Free International directory/evidence URLs.
    - vegan_urls: [array of URLs] URLs evidencing vegan certification (PETA, The Vegan Society, BeVeg, etc.). (Primarily needed for brand 2.)
    - fair_trade_urls: [array of URLs] URLs evidencing Fair Trade certification for products or ingredients (Fair Trade USA, Fairtrade International, etc.). (Primarily needed for brand 3.)
    - climate_urls: [array of URLs] URLs evidencing Climate Neutral/Carbon Neutral certification or verified climate-positive commitments. (Primarily needed for brand 4.)

    - packaging_statement: (brand 1 only if provided) A short quote/summary from the answer about 100% PCR plastic or fully recyclable (glass/aluminum) primary packaging.
    - packaging_urls: [array of URLs] URLs confirming the 100% PCR or fully recyclable primary packaging commitment (brand 1 requirement).

    - us_urls: [array of URLs] URLs confirming US availability (e.g., US website/store, shipping policy showing ships to US, US retailer pages).

    OUTPUT FORMAT:
    {
      "brand1": {...},
      "brand2": {...},
      "brand3": {...},
      "brand4": {...}
    }

    IMPORTANT:
    - If the answer provides more than one URL for a given field, include all of them in the corresponding array.
    - If a certification is claimed without a URL, leave the array empty (do not invent).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _collect_cert_urls(brands: AllBrandsExtraction) -> List[str]:
    urls: List[str] = []
    for b in [brands.brand1, brands.brand2, brands.brand3, brands.brand4]:
        if not b:
            continue
        urls.extend(b.bcorp_urls or [])
        urls.extend(b.leaping_bunny_urls or [])
        urls.extend(b.vegan_urls or [])
        urls.extend(b.fair_trade_urls or [])
        urls.extend(b.climate_urls or [])
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen and _nonempty(u):
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification building blocks                                                #
# --------------------------------------------------------------------------- #
async def verify_identity(evaluator: Evaluator, parent, brand: BrandData, idx: int) -> None:
    node = evaluator.add_parallel(
        id=f"Brand_{idx}_Identity",
        desc="Provide the complete brand identification information",
        parent=parent,
        critical=True
    )

    # Name provided (existence check)
    evaluator.add_custom_node(
        result=_nonempty(brand.name),
        id=f"Brand_{idx}_Identity_Name",
        desc="The brand name is clearly stated",
        parent=node,
        critical=True
    )

    # Category verified via website if possible
    cat_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_Identity_Category",
        desc="The brand operates in the beauty/personal care sector",
        parent=node,
        critical=True
    )
    cat_claim = f"The brand '{brand.name or ''}' operates in the beauty or personal care sector (e.g., cosmetics, skincare, haircare, body care)."
    await evaluator.verify(
        claim=cat_claim,
        node=cat_leaf,
        sources=brand.website_url if _nonempty(brand.website_url) else None,
        additional_instruction="Use the official website if provided. Look for cues like product categories (makeup, skincare, haircare, cosmetics), or clear positioning as a beauty/personal care brand."
    )

    # Official website URL provided (existence check)
    evaluator.add_custom_node(
        result=_nonempty(brand.website_url),
        id=f"Brand_{idx}_Identity_URL",
        desc="Provide the official brand website URL",
        parent=node,
        critical=True
    )


async def verify_bcorp(evaluator: Evaluator, parent, brand: BrandData, idx: int) -> None:
    node = evaluator.add_sequential(
        id=f"Brand_{idx}_BCorp",
        desc="Verify the brand holds valid B Corp certification",
        parent=parent,
        critical=True
    )
    status_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_BCorp_Status",
        desc="The brand is listed as a certified B Corporation",
        parent=node,
        critical=True
    )
    bcorp_claim = f"The brand '{brand.name or ''}' is listed as a certified B Corporation on an official B Corp directory/certification page, current for 2025-2026."
    await evaluator.verify(
        claim=bcorp_claim,
        node=status_leaf,
        sources=brand.bcorp_urls,
        additional_instruction="Evidence should come from bcorporation.net (B Lab/official B Corp directory) or an official B Corp-controlled regional site. Directory entries generally indicate current certification. Allow legal entity vs consumer brand naming variations."
    )

    verif_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_BCorp_Verification_URL",
        desc="Provide URL from bcorporation.net or other official B Corp source confirming the certification",
        parent=node,
        critical=True
    )
    bcorp_url_claim = f"This page is an official B Corp directory/evidence page (e.g., hosted on bcorporation.net) that confirms B Corp certification for '{brand.name or ''}' (or its legal entity)."
    await evaluator.verify(
        claim=bcorp_url_claim,
        node=verif_leaf,
        sources=brand.bcorp_urls,
        additional_instruction="Check the domain and page content to confirm it is an official B Corp source and that it confirms certification."
    )


async def verify_leaping_bunny(evaluator: Evaluator, parent, brand: BrandData, idx: int) -> None:
    node = evaluator.add_sequential(
        id=f"Brand_{idx}_LeapingBunny",
        desc="Verify the brand holds valid Leaping Bunny certification",
        parent=parent,
        critical=True
    )
    status_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_LeapingBunny_Status",
        desc="The brand is listed as Leaping Bunny certified",
        parent=node,
        critical=True
    )
    lb_claim = f"The brand '{brand.name or ''}' is listed as Leaping Bunny certified (cruelty-free) on an official directory/certification page, current for 2025-2026."
    await evaluator.verify(
        claim=lb_claim,
        node=status_leaf,
        sources=brand.leaping_bunny_urls,
        additional_instruction="Accept evidence from leapingbunny.org or crueltyfreeinternational.org directories or certification pages."
    )

    verif_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_LeapingBunny_Verification_URL",
        desc="Provide URL from leapingbunny.org or crueltyfreeinternational.org confirming the certification",
        parent=node,
        critical=True
    )
    lb_url_claim = f"This URL is hosted on leapingbunny.org or crueltyfreeinternational.org and confirms Leaping Bunny certification for '{brand.name or ''}'."
    await evaluator.verify(
        claim=lb_url_claim,
        node=verif_leaf,
        sources=brand.leaping_bunny_urls,
        additional_instruction="Verify the page domain and that it confirms Leaping Bunny certification for the brand."
    )


async def verify_vegan(evaluator: Evaluator, parent, brand: BrandData, idx: int) -> None:
    node = evaluator.add_sequential(
        id=f"Brand_{idx}_Vegan",
        desc="Verify the brand holds valid vegan certification",
        parent=parent,
        critical=True
    )
    status_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_Vegan_Status",
        desc="The brand or its products are certified vegan by PETA, The Vegan Society, or other recognized body",
        parent=node,
        critical=True
    )
    vegan_claim = f"The brand '{brand.name or ''}' (or its products) holds vegan certification from a recognized body (e.g., PETA, The Vegan Society, BeVeg), current for 2025-2026."
    await evaluator.verify(
        claim=vegan_claim,
        node=status_leaf,
        sources=brand.vegan_urls,
        additional_instruction="Evidence should be from recognized certification bodies/directories (e.g., peta.org Beauty Without Bunnies vegan listings, vegansociety.com product registration, or beveg.com)."
    )

    verif_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_Vegan_Verification_URL",
        desc="Provide URL confirming vegan certification status",
        parent=node,
        critical=True
    )
    vegan_url_claim = f"This URL is from a recognized vegan certification body (peta.org, vegansociety.com, or beveg.com) and confirms vegan certification for '{brand.name or ''}'."
    await evaluator.verify(
        claim=vegan_url_claim,
        node=verif_leaf,
        sources=brand.vegan_urls,
        additional_instruction="Check the domain and content to confirm recognized vegan certification for the brand or its products."
    )


async def verify_fair_trade(evaluator: Evaluator, parent, brand: BrandData, idx: int) -> None:
    node = evaluator.add_sequential(
        id=f"Brand_{idx}_FairTrade",
        desc="Verify the brand holds Fair Trade certification",
        parent=parent,
        critical=True
    )
    status_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_FairTrade_Status",
        desc="The brand's products or ingredients are Fair Trade certified",
        parent=node,
        critical=True
    )
    ft_claim = f"The brand '{brand.name or ''}' has products or ingredients that are Fair Trade certified, current for 2025-2026."
    await evaluator.verify(
        claim=ft_claim,
        node=status_leaf,
        sources=brand.fair_trade_urls,
        additional_instruction="Prefer evidence from fairtradecertified.org (Fair Trade USA), fairtrade.net/fairtradeamerica.org (Fairtrade International/America), or FLO-CERT (flo-cert.com)."
    )

    verif_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_FairTrade_Verification_URL",
        desc="Provide URL confirming Fair Trade certification",
        parent=node,
        critical=True
    )
    ft_url_claim = f"This URL is from an official Fair Trade certification body/directory and confirms Fair Trade certification for '{brand.name or ''}'."
    await evaluator.verify(
        claim=ft_url_claim,
        node=verif_leaf,
        sources=brand.fair_trade_urls,
        additional_instruction="Accept fairtradecertified.org, fairtrade.net, fairtradeamerica.org, or flo-cert.com. Brand sustainability pages with independent verification are acceptable if clearly confirmed."
    )


async def verify_climate(evaluator: Evaluator, parent, brand: BrandData, idx: int) -> None:
    node = evaluator.add_sequential(
        id=f"Brand_{idx}_Climate",
        desc="Verify the brand holds Carbon Neutral or Climate Neutral certification or has documented climate commitments",
        parent=parent,
        critical=True
    )
    status_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_Climate_Status",
        desc="The brand is certified Carbon Neutral, Climate Neutral, or has verified climate-positive commitments",
        parent=node,
        critical=True
    )
    climate_claim = f"The brand '{brand.name or ''}' is certified Climate Neutral/Carbon Neutral or has documented and independently verified climate-positive commitments, current for 2025-2026."
    await evaluator.verify(
        claim=climate_claim,
        node=status_leaf,
        sources=brand.climate_urls,
        additional_instruction="Prefer evidence from climateneutral.org, carbonneutral.com / climateimpact.com, or official sustainability reports with third-party verification on the brand's site."
    )

    verif_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_Climate_Verification_URL",
        desc="Provide URL from official certification body or brand's verified sustainability report confirming climate status",
        parent=node,
        critical=True
    )
    climate_url_claim = f"This URL is an official certification/directory page (e.g., climateneutral.org, carbonneutral.com/climateimpact.com) or a verified sustainability report page confirming the climate status for '{brand.name or ''}'."
    await evaluator.verify(
        claim=climate_url_claim,
        node=verif_leaf,
        sources=brand.climate_urls,
        additional_instruction="Verify domain and content to ensure it is an official certification source or a brand page with third-party verification."
    )


async def verify_packaging(evaluator: Evaluator, parent, brand: BrandData, idx: int) -> None:
    node = evaluator.add_sequential(
        id=f"Brand_{idx}_Packaging",
        desc="Verify the brand uses 100% recyclable or post-consumer recycled packaging",
        parent=parent,
        critical=True
    )
    materials_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_Packaging_Materials",
        desc="The brand's packaging is made from 100% post-consumer recycled plastic, glass, or aluminum",
        parent=node,
        critical=True
    )
    pack_stmt = brand.packaging_statement or "uses 100% PCR plastic or fully recyclable glass/aluminum for primary containers"
    pack_claim = f"The brand '{brand.name or ''}' {pack_stmt}, specifically for primary product containers."
    await evaluator.verify(
        claim=pack_claim,
        node=materials_leaf,
        sources=brand.packaging_urls,
        additional_instruction="Confirm that primary containers use 100% post-consumer recycled plastic, or are fully recyclable glass or aluminum. Look for explicit '100%' or equivalent statements on official sources."
    )

    verif_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_Packaging_Verification_URL",
        desc="Provide URL confirming the 100% recyclable or PCR packaging commitment",
        parent=node,
        critical=True
    )
    pack_url_claim = f"These URLs confirm that '{brand.name or ''}' uses 100% PCR plastic or fully recyclable glass/aluminum for primary product containers."
    await evaluator.verify(
        claim=pack_url_claim,
        node=verif_leaf,
        sources=brand.packaging_urls,
        additional_instruction="Page(s) should clearly state 100% post-consumer recycled plastic or fully recyclable glass/aluminum for primary containers."
    )


async def verify_us_availability(evaluator: Evaluator, parent, brand: BrandData, idx: int) -> None:
    node = evaluator.add_parallel(
        id=f"Brand_{idx}_US_Availability",
        desc="Verify the brand's products are available in the US market",
        parent=parent,
        critical=True
    )

    available_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_US_Available",
        desc="The brand sells products in the United States",
        parent=node,
        critical=True
    )
    us_sources = brand.us_urls if brand.us_urls else (brand.website_url if _nonempty(brand.website_url) else None)
    us_claim = f"The brand '{brand.name or ''}' sells products in the United States (e.g., US site/store, ships to the US, or available via US retailers)."
    await evaluator.verify(
        claim=us_claim,
        node=available_leaf,
        sources=us_sources,
        additional_instruction="Evidence can include a US-specific site, a shipping policy indicating shipping to the USA, a US store locator, or listings on US retailers."
    )

    verif_leaf = evaluator.add_leaf(
        id=f"Brand_{idx}_US_Verification_URL",
        desc="Provide URL showing US availability",
        parent=node,
        critical=True
    )
    us_url_claim = f"This URL shows US availability for '{brand.name or ''}' (e.g., 'United States' in country selector, US store, shipping to US, or US retailer presence)."
    await evaluator.verify(
        claim=us_url_claim,
        node=verif_leaf,
        sources=us_sources,
        additional_instruction="Look for 'United States' in selectors, 'ships to US', US-specific pages (.com/us) or prominent US retailer pages."
    )


# --------------------------------------------------------------------------- #
# Brand-specific verification orchestration                                   #
# --------------------------------------------------------------------------- #
async def verify_brand_1(evaluator: Evaluator, root, b: BrandData) -> None:
    brand_node = evaluator.add_parallel(
        id="Brand_1",
        desc="Identify a beauty brand that is B Corp certified, Leaping Bunny certified, and uses 100% post-consumer recycled plastic or fully recyclable packaging materials",
        parent=root,
        critical=False
    )
    await verify_identity(evaluator, brand_node, b, 1)
    await verify_bcorp(evaluator, brand_node, b, 1)
    await verify_leaping_bunny(evaluator, brand_node, b, 1)
    await verify_packaging(evaluator, brand_node, b, 1)
    await verify_us_availability(evaluator, brand_node, b, 1)


async def verify_brand_2(evaluator: Evaluator, root, b: BrandData) -> None:
    brand_node = evaluator.add_parallel(
        id="Brand_2",
        desc="Identify a beauty brand that is B Corp certified, holds vegan certification, and is Leaping Bunny certified",
        parent=root,
        critical=False
    )
    await verify_identity(evaluator, brand_node, b, 2)
    await verify_bcorp(evaluator, brand_node, b, 2)
    await verify_vegan(evaluator, brand_node, b, 2)
    await verify_leaping_bunny(evaluator, brand_node, b, 2)
    await verify_us_availability(evaluator, brand_node, b, 2)


async def verify_brand_3(evaluator: Evaluator, root, b: BrandData) -> None:
    brand_node = evaluator.add_parallel(
        id="Brand_3",
        desc="Identify a beauty brand that is B Corp certified, Fair Trade certified, and Leaping Bunny certified",
        parent=root,
        critical=False
    )
    await verify_identity(evaluator, brand_node, b, 3)
    await verify_bcorp(evaluator, brand_node, b, 3)
    await verify_fair_trade(evaluator, brand_node, b, 3)
    await verify_leaping_bunny(evaluator, brand_node, b, 3)
    await verify_us_availability(evaluator, brand_node, b, 3)


async def verify_brand_4(evaluator: Evaluator, root, b: BrandData) -> None:
    brand_node = evaluator.add_parallel(
        id="Brand_4",
        desc="Identify a beauty brand that is B Corp certified and holds Carbon Neutral or Climate Neutral certification",
        parent=root,
        critical=False
    )
    await verify_identity(evaluator, brand_node, b, 4)
    await verify_bcorp(evaluator, brand_node, b, 4)
    await verify_climate(evaluator, brand_node, b, 4)
    await verify_us_availability(evaluator, brand_node, b, 4)


# --------------------------------------------------------------------------- #
# Global checks                                                               #
# --------------------------------------------------------------------------- #
async def verify_brand_distinctness(evaluator: Evaluator, root, brands: AllBrandsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Brand_Distinctness",
        desc="Verify that all four brands are distinct entities and no brand is used to satisfy multiple brand requirements",
        parent=root,
        critical=False
    )
    leaf = evaluator.add_leaf(
        id="Brand_Distinctness_Check",
        desc="All four brands are distinct entities",
        parent=node,
        critical=False
    )
    names = [b.name for b in [brands.brand1, brands.brand2, brands.brand3, brands.brand4] if _nonempty(b.name if b else None)]
    sites = [b.website_url for b in [brands.brand1, brands.brand2, brands.brand3, brands.brand4] if (b and _nonempty(b.website_url))]
    distinct_claim = f"The following brand names refer to four distinct companies/brands with no duplication: {names}."
    await evaluator.verify(
        claim=distinct_claim,
        node=leaf,
        sources=[u for u in sites if _nonempty(u)],
        additional_instruction="Confirm they are different brands. Consider official sites; different domains and brand identities should indicate distinctness. Ignore shared parent conglomerates."
    )


async def verify_certification_currency(evaluator: Evaluator, root, brands: AllBrandsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Certification_Currency",
        desc="Verify that all certifications provided are current and valid as of 2025-2026",
        parent=root,
        critical=False
    )
    leaf = evaluator.add_leaf(
        id="Certification_Currency_2025_2026",
        desc="At least one provided certification page clearly indicates currency/validity in 2025-2026",
        parent=node,
        critical=False
    )
    urls = _collect_cert_urls(brands)
    claim = "The certification evidence pages indicate that the associated certifications are current and valid as of 2025 or 2026 (e.g., active directory entry, 'Certified YYYY', or recently updated status)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Check each page for indicators of current validity (active listing, validity dates covering 2025-2026, or language implying present/active certification). Passing if any page clearly shows currency."
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
    # Initialize evaluator
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=AllBrandsExtraction,
        extraction_name="brands_extraction"
    )

    # Add some helpful custom info to summary
    evaluator.add_custom_info(
        info={
            "brand_names": [
                (extracted.brand1.name if extracted.brand1 else None),
                (extracted.brand2.name if extracted.brand2 else None),
                (extracted.brand3.name if extracted.brand3 else None),
                (extracted.brand4.name if extracted.brand4 else None),
            ],
            "brand_websites": [
                (extracted.brand1.website_url if extracted.brand1 else None),
                (extracted.brand2.website_url if extracted.brand2 else None),
                (extracted.brand3.website_url if extracted.brand3 else None),
                (extracted.brand4.website_url if extracted.brand4 else None),
            ]
        },
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    # Global checks
    await verify_brand_distinctness(evaluator, root, extracted)
    await verify_certification_currency(evaluator, root, extracted)

    # Brand-specific verifications
    b1 = extracted.brand1 or BrandData()
    b2 = extracted.brand2 or BrandData()
    b3 = extracted.brand3 or BrandData()
    b4 = extracted.brand4 or BrandData()

    await verify_brand_1(evaluator, root, b1)
    await verify_brand_2(evaluator, root, b2)
    await verify_brand_3(evaluator, root, b3)
    await verify_brand_4(evaluator, root, b4)

    return evaluator.get_summary()