import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "models_founded_beauty_brands_2015_2022"
TASK_DESCRIPTION = (
    "I'm researching the intersection of modeling and beauty entrepreneurship for a market analysis report. "
    "I need to identify beauty brands that were founded by professional models between 2015 and 2022 (inclusive) "
    "and that meet specific operational and distribution criteria as of March 2026.\n\n"
    "Please identify three (3) beauty brands that satisfy ALL of the following requirements:\n"
    "1. The brand was founded by someone with an established professional modeling career (someone who has worked in "
    "fashion shows, campaigns, or editorials for major brands)\n"
    "2. The brand was founded between January 1, 2015, and December 31, 2022 (inclusive)\n"
    "3. The brand is still operational and actively selling products as of March 2026\n"
    "4. The original model-founder is still actively involved with the brand as of March 2026 (has not stepped down, "
    "completely sold their stake, or publicly severed ties with the brand)\n"
    "5. The brand is available for purchase at either Sephora or Ulta Beauty stores in the United States as of March 2026\n"
    "6. The brand offers at least one of the following product categories: skincare, haircare, or makeup\n"
    "7. The brand positions itself as clean, sustainable, organic, or natural beauty (as evidenced by official brand "
    "descriptions or retailer categorization)\n\n"
    "For each brand, provide:\n"
    "- The brand name\n"
    "- The founder's name\n"
    "- The year the brand was founded\n"
    "- The primary product category (skincare, haircare, or makeup)\n"
    "- The US retailer where it's available (Sephora or Ulta Beauty)\n"
    "- A reference URL that verifies the key information"
)

AS_OF_TIMEFRAME = "March 2026"
YEAR_MIN = 2015
YEAR_MAX = 2022

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BrandItem(BaseModel):
    brand_name: Optional[str] = None
    founder_name: Optional[str] = None  # If multiple founders, choose the founder with the modeling career
    founding_year: Optional[str] = None  # Keep as string to be robust to formats like "2017" or "2017 (launched)"
    primary_category: Optional[str] = None  # One of: "skincare", "haircare", or "makeup"
    retailer: Optional[str] = None  # "Sephora" or "Ulta Beauty"
    retailer_urls: List[str] = Field(default_factory=list)  # US pages on sephora.com or ulta.com
    source_urls: List[str] = Field(default_factory=list)  # Other references cited in the answer (brand site, press, Wikipedia, etc.)


class BrandsExtraction(BaseModel):
    brands: List[BrandItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return """
Extract up to three (3) beauty brands from the answer that claim to meet ALL of the task requirements. If more than three are mentioned, keep only the first three in the same order. If fewer are mentioned, include as many as present.

For each brand, extract the following fields:
- brand_name: The brand's name, exactly as written in the answer.
- founder_name: The founder's name who is/was a professional model. If multiple founders, choose the one with an established modeling career. If unclear, choose the model most associated with the brand.
- founding_year: A 4-digit year indicating when the brand was founded. If a full date is provided, return just the year. If a range or multiple dates are given, return the year corresponding to the founding/launch of the brand entity referenced in the answer.
- primary_category: One of "skincare", "haircare", or "makeup". If the brand spans multiple, choose the one presented as primary in the answer; otherwise, pick the most prominent category based on the answer text.
- retailer: Either "Sephora" or "Ulta Beauty" if the answer states the brand is available at that US retailer as of March 2026. Use exactly "Sephora" or "Ulta Beauty". If not specified or ambiguous, set to null.
- retailer_urls: A list of US retailer URLs (sephora.com or ulta.com) that the answer cites for this brand. If the answer lists international or non-US sites (e.g., sephora.ca, sephora.sg), DO NOT include them. If none provided, return an empty array.
- source_urls: A list of other reference URLs mentioned in the answer that support key facts (e.g., brand official site/about page, press releases, interviews, Wikipedia, Business of Fashion, Vogue Business, model profile pages). Include only URLs explicitly present in the answer. If none, return an empty array.

Rules:
- Only extract URLs that are explicitly present in the answer text (including in markdown links). Do not invent URLs.
- Do not merge or infer information not stated. If some field is missing, set it to null (or empty array for URL lists).
- Preserve the exact spelling/casing of brand and founder names as written in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(brand: BrandItem) -> List[str]:
    """Merge and deduplicate all available URLs for a brand."""
    urls: List[str] = []
    urls.extend(brand.retailer_urls or [])
    urls.extend(brand.source_urls or [])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification logic per brand                                                #
# --------------------------------------------------------------------------- #
async def verify_single_brand(
    evaluator: Evaluator,
    parent_node,
    brand: BrandItem,
    brand_index: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single brand.

    The sub-tree follows the rubric's structure for:
    - Founder modeling career
    - Founding year within 2015-2022 (inclusive)
    - Operational status as of March 2026
    - Founder still involved as of March 2026
    - Retail availability at Sephora or Ulta Beauty (US) as of March 2026
    - Product category in skincare/haircare/makeup
    - Clean/sustainable/organic/natural positioning
    - Reference URL existence (non-critical)
    """
    brand_num = brand_index + 1
    brand_node = evaluator.add_parallel(
        id=f"brand_{brand_num}",
        desc=f"{['First','Second','Third'][brand_index]} qualifying beauty brand",
        parent=parent_node,
        critical=False  # Allow partial credit across brands
    )

    all_urls = _merge_sources(brand)

    # --- Leaf nodes (critical unless noted otherwise) ---
    founder_node = evaluator.add_leaf(
        id=f"brand_{brand_num}_founder_verification",
        desc="Verify the founder has an established professional modeling career",
        parent=brand_node,
        critical=True
    )

    founding_year_node = evaluator.add_leaf(
        id=f"brand_{brand_num}_founding_year",
        desc="Verify the brand was founded between 2015 and 2022 (inclusive)",
        parent=brand_node,
        critical=True
    )

    operational_node = evaluator.add_leaf(
        id=f"brand_{brand_num}_operational_status",
        desc=f"Verify the brand is still operational and selling products as of {AS_OF_TIMEFRAME}",
        parent=brand_node,
        critical=True
    )

    involvement_node = evaluator.add_leaf(
        id=f"brand_{brand_num}_founder_involvement",
        desc=f"Verify the original founder is still actively involved with the brand as of {AS_OF_TIMEFRAME}",
        parent=brand_node,
        critical=True
    )

    retail_node = evaluator.add_leaf(
        id=f"brand_{brand_num}_retail_distribution",
        desc=f"Verify the brand is available at Sephora or Ulta Beauty in the US as of {AS_OF_TIMEFRAME}",
        parent=brand_node,
        critical=True
    )

    category_node = evaluator.add_leaf(
        id=f"brand_{brand_num}_product_category",
        desc="Verify the brand offers skincare, haircare, or makeup products",
        parent=brand_node,
        critical=True
    )

    clean_node = evaluator.add_leaf(
        id=f"brand_{brand_num}_clean_positioning",
        desc="Verify the brand positions itself as clean, sustainable, organic, or natural beauty",
        parent=brand_node,
        critical=True
    )

    # Non-critical: reference URL existence (at least one supporting URL provided)
    reference_node = evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id=f"brand_{brand_num}_reference_url",
        desc="Provide reference URL supporting the brand identification and key facts",
        parent=brand_node,
        critical=False
    )

    # --- Build claims and run batch verification ---
    # Founder modeling career
    founder_name = brand.founder_name or ""
    founder_claim = (
        f"{founder_name} has an established professional modeling career (e.g., runway shows, "
        f"campaigns, or editorials for major fashion brands)."
    )
    founder_ins = (
        "Confirm that the named founder is/was a professional model with recognizable, professional work "
        "such as runway shows, campaigns, magazine editorials, or agency/modeling profiles. "
        "Do NOT rely on generic 'influencer' status; the evidence must clearly indicate a modeling career."
    )

    # Founding year
    founding_year_text = brand.founding_year or ""
    brand_name = brand.brand_name or ""
    founding_claim = (
        f"The beauty brand '{brand_name}' was founded in {founding_year_text}, and that year is between "
        f"{YEAR_MIN} and {YEAR_MAX} inclusive."
    )
    founding_ins = (
        f"Verify that the founding year for the brand is explicitly {founding_year_text}. "
        f"If multiple dates are mentioned, focus on the founding/launch year of the brand entity itself. "
        f"Additionally, confirm that this year lies within {YEAR_MIN}–{YEAR_MAX} inclusive."
    )

    # Operational status as of March 2026
    operational_claim = (
        f"As of {AS_OF_TIMEFRAME}, the brand '{brand_name}' is still operational and actively selling products."
    )
    operational_ins = (
        f"Look for recent, active product listings (e.g., add-to-cart, in-stock indicators) on the official brand site "
        f"or US retailer pages, or recent official communications clearly indicating ongoing operations. "
        f"Treat a live US Sephora/Ulta listing as sufficient evidence of ongoing operations as of {AS_OF_TIMEFRAME}."
    )

    # Founder involvement as of March 2026
    involvement_claim = (
        f"As of {AS_OF_TIMEFRAME}, the original founder {founder_name} is still actively involved with the brand "
        f"(e.g., as founder, creative director, executive, ambassador), and has not stepped down, fully sold their stake, "
        f"or publicly severed ties."
    )
    involvement_ins = (
        f"Check official bios, press releases, recent interviews, LinkedIn, or About pages for titles/roles. "
        f"Accept language such as 'founder', 'co-founder', 'creative director', 'still leads', 'continues to helm', "
        f"or similar. If there is evidence of total exit, step-down, or severed ties before {AS_OF_TIMEFRAME}, "
        f"then the claim should be rejected."
    )

    # Retail distribution at Sephora or Ulta Beauty US
    retailer_text = brand.retailer or "Sephora or Ulta Beauty"
    retail_claim = (
        f"As of {AS_OF_TIMEFRAME}, the brand '{brand_name}' is available for purchase at {retailer_text} in the United States."
    )
    retail_ins = (
        f"Verify using US retailer pages only: sephora.com (US) or ulta.com. "
        f"Ignore non-US domains like sephora.ca, sephora.sg, etc. "
        f"A valid brand/product listing on the US site counts as 'available'."
    )

    # Product category
    category_text = brand.primary_category or ""
    category_claim = (
        f"The brand '{brand_name}' offers {category_text} products (one of skincare, haircare, or makeup)."
    )
    category_ins = (
        "Confirm at least one of the categories 'skincare', 'haircare', or 'makeup' is truly offered by the brand. "
        "Accept synonyms like 'color cosmetics' for makeup, 'skin' or 'skin care' for skincare. "
        "A US retailer listing or the brand's official product pages are acceptable evidence."
    )

    # Clean/sustainable/organic/natural positioning
    clean_claim = (
        f"The brand '{brand_name}' positions itself as clean, sustainable, organic, or natural beauty."
    )
    clean_ins = (
        "Accept official brand claims (e.g., 'clean', 'natural', 'organic', 'non-toxic', 'sustainable') "
        "or retailer categorizations such as 'Clean at Sephora' or Ulta's 'Conscious Beauty' badges. "
        "Evidence should be explicit."
    )

    claims_and_sources = [
        (founder_claim, all_urls, founder_node, founder_ins),
        (founding_claim, all_urls, founding_year_node, founding_ins),
        (operational_claim, all_urls, operational_node, operational_ins),
        (involvement_claim, all_urls, involvement_node, involvement_ins),
        (retail_claim, brand.retailer_urls or [], retail_node, retail_ins),
        (category_claim, all_urls, category_node, category_ins),
        (clean_claim, all_urls, clean_node, clean_ins),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the 'models_founded_beauty_brands_2015_2022' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Evaluate each brand independently
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

    # Extract up to 3 brands as structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="extracted_brands"
    )

    # Normalize to exactly 3 entries (pad with empty BrandItem if fewer)
    brands: List[BrandItem] = list(extracted.brands or [])
    if len(brands) > 3:
        brands = brands[:3]
    while len(brands) < 3:
        brands.append(BrandItem())

    # Build verification subtrees for each brand
    for idx in range(3):
        await verify_single_brand(evaluator, root, brands[idx], idx)

    # Return standardized summary with the verification tree
    return evaluator.get_summary()