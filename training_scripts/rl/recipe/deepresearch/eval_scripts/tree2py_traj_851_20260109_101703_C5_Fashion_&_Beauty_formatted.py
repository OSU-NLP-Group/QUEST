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
TASK_ID = "ca_bcorp_activewear"
TASK_DESCRIPTION = """
Identify two activewear brands headquartered in California that are Certified B Corporations and use GOTS certified organic cotton or equivalent organic certification in their products. For each brand, provide the brand name, headquarters location, B Corp status with a reference, evidence of organic textile certification, and confirmation that they produce activewear.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BrandEntry(BaseModel):
    """Structured info for one brand as cited in the answer."""
    brand_name: Optional[str] = None
    headquarters: Optional[str] = None

    b_corp_claim: Optional[str] = None
    b_corp_sources: List[str] = Field(default_factory=list)

    organic_cert_claim: Optional[str] = None
    organic_cert_sources: List[str] = Field(default_factory=list)

    activewear_claim: Optional[str] = None
    activewear_sources: List[str] = Field(default_factory=list)


class BrandsExtraction(BaseModel):
    """All brands extracted from the answer."""
    brands: List[BrandEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return """
    Extract up to the first 3 brand entries mentioned in the answer that appear intended to meet the criteria. For each brand, return a JSON object with:
    - brand_name: The brand's name as written.
    - headquarters: The stated headquarters location string (e.g., "Oakland, CA, USA" or "Los Angeles, California").
    - b_corp_claim: The exact statement in the answer that claims B Corp certification (if any).
    - b_corp_sources: All URLs provided in the answer that directly support the B Corp certification (e.g., B Lab directory page or official brand page).
    - organic_cert_claim: The exact statement in the answer about using GOTS certified organic textiles or equivalent organic certification (GOTS, OCS, USDA Organic, etc.).
    - organic_cert_sources: All URLs provided that support the organic certification claim (brand pages, certification pages).
    - activewear_claim: The exact statement in the answer confirming the brand produces activewear or athletic apparel.
    - activewear_sources: All URLs provided that support the activewear confirmation (product category pages, official brand pages).

    Rules:
    - Extract only what is explicitly present in the answer text; do not invent information.
    - For any missing field, set it to null; for missing URL lists, return an empty array.
    - Accept URLs in plain or markdown link formats; ensure they are complete (prepend http:// if protocol is missing).
    - Preserve the order of appearance from the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_brand(
    evaluator: Evaluator,
    parent_node,
    brand: BrandEntry,
    brand_index: int,
) -> None:
    """
    Build and execute verification nodes for one brand.
    """
    brand_node = evaluator.add_parallel(
        id=f"brand_{brand_index + 1}",
        desc=f"Brand {brand_index + 1} evaluation",
        parent=parent_node,
        critical=False,
    )

    # 1) Brand name provided (critical)
    name_provided = evaluator.add_custom_node(
        result=bool(brand.brand_name and brand.brand_name.strip()),
        id=f"brand_{brand_index + 1}_brand_name_provided",
        desc="Brand name is provided",
        parent=brand_node,
        critical=True
    )

    # 2) Headquarters provided (critical) + Headquarters is in California (critical leaf)
    hq_provided = evaluator.add_custom_node(
        result=bool(brand.headquarters and brand.headquarters.strip()),
        id=f"brand_{brand_index + 1}_hq_provided",
        desc="Headquarters location string is provided",
        parent=brand_node,
        critical=True
    )

    hq_is_ca = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_headquarters_in_california",
        desc="Headquarters location is provided and is in California, United States",
        parent=brand_node,
        critical=True
    )
    hq_claim = f"The headquarters location '{brand.headquarters or ''}' is in the U.S. state of California."
    await evaluator.verify(
        claim=hq_claim,
        node=hq_is_ca,
        additional_instruction="Judge based on the location string itself. Consider variants like 'CA', 'California', and city names in California (e.g., Los Angeles, San Francisco, Oakland). Country context may be implicit."
    )

    # 3) B Corp with reference: sources existence (critical) + verification by URLs (critical)
    bcorp_sources_exist = evaluator.add_custom_node(
        result=bool(brand.b_corp_sources and len(brand.b_corp_sources) > 0),
        id=f"brand_{brand_index + 1}_b_corp_sources_provided",
        desc="Brand is stated to be a Certified B Corporation and supporting reference URL(s) are provided",
        parent=brand_node,
        critical=True
    )

    bcorp_verified = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_b_corp_verified",
        desc="Brand is a Certified B Corporation (verified by cited reference)",
        parent=brand_node,
        critical=True
    )
    bcorp_claim = f"The brand '{brand.brand_name or ''}' is certified as a B Corporation by B Lab."
    await evaluator.verify(
        claim=bcorp_claim,
        node=bcorp_verified,
        sources=brand.b_corp_sources,
        additional_instruction="Verify the page explicitly shows the brand is a Certified B Corporation (e.g., B Lab directory entry or official brand statement). Accept synonyms like 'Certified B Corp' or 'B Corp Certified'."
    )

    # 4) Organic certification with evidence: sources existence (critical) + verification by URLs (critical)
    organic_sources_exist = evaluator.add_custom_node(
        result=bool(brand.organic_cert_sources and len(brand.organic_cert_sources) > 0),
        id=f"brand_{brand_index + 1}_organic_cert_sources_provided",
        desc="Evidence is provided that the brand uses GOTS certified organic textiles or an equivalent organic certification (e.g., USDA Organic, OCS)",
        parent=brand_node,
        critical=True
    )

    organic_verified = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_organic_cert_verified",
        desc="Organic textile certification is supported (GOTS or equivalent)",
        parent=brand_node,
        critical=True
    )
    organic_claim = (
        f"The brand '{brand.brand_name or ''}' uses GOTS-certified organic textiles or an equivalent organic certification "
        f"(such as USDA Organic or Organic Content Standard) in its products."
    )
    await evaluator.verify(
        claim=organic_claim,
        node=organic_verified,
        sources=brand.organic_cert_sources,
        additional_instruction="Look for explicit mentions of GOTS, Organic Content Standard (OCS), or USDA Organic for textiles/products. A credible certification page or brand page must clearly indicate certified organic materials."
    )

    # 5) Activewear confirmation: sources existence (critical) + verification by URLs (critical)
    activewear_sources_exist = evaluator.add_custom_node(
        result=bool(brand.activewear_sources and len(brand.activewear_sources) > 0),
        id=f"brand_{brand_index + 1}_activewear_sources_provided",
        desc="Confirmation is provided that the brand produces activewear or athletic apparel (with URL evidence)",
        parent=brand_node,
        critical=True
    )

    activewear_verified = evaluator.add_leaf(
        id=f"brand_{brand_index + 1}_activewear_confirmation",
        desc="Brand produces activewear or athletic apparel (verified by cited reference)",
        parent=brand_node,
        critical=True
    )
    activewear_claim = (
        f"The brand '{brand.brand_name or ''}' produces activewear or athletic apparel (e.g., leggings, sports bras, performance shirts, workout shorts, or similar)."
    )
    await evaluator.verify(
        claim=activewear_claim,
        node=activewear_verified,
        sources=brand.activewear_sources,
        additional_instruction="Check product/category pages or brand documentation showing activewear/athletic/performance apparel. Accept reasonable synonyms like 'sportswear', 'fitness apparel', 'workout gear'."
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
    Evaluate an answer for the California B Corp activewear brands task.
    """
    # Initialize evaluator (root kept non-critical to allow partial credit across brands)
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

    # Extract brands cited in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="brands_extraction"
    )

    brands = extracted.brands or []

    # Ensure at least two slots; pad if necessary
    selected: List[BrandEntry] = brands[:2]
    while len(selected) < 2:
        selected.append(BrandEntry())

    # Critical gate: at least two distinct brand names provided
    names = [b.brand_name.strip() for b in selected if b.brand_name]
    two_distinct = (len(names) >= 2) and (names[0].lower() != names[1].lower())
    evaluator.add_custom_node(
        result=two_distinct,
        id="two_distinct_brands_provided",
        desc="Response identifies at least two different (distinct) brands that are intended to meet the criteria",
        parent=root,
        critical=True
    )

    # Build brand verification nodes
    for i, brand in enumerate(selected):
        await verify_brand(evaluator, root, brand, i)

    # Summary
    return evaluator.get_summary()