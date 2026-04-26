import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "demi_moore_bioeffect_serum_eval"
TASK_DESCRIPTION = (
    "Demi Moore, the 63-year-old actress known for her sensitive skin, uses a specific serum in her nighttime "
    "skincare routine that contains epidermal growth factor (EGF) as its primary active ingredient. This serum is "
    "known for having a notably pure formulation and is manufactured by a luxury Icelandic skincare brand. Identify "
    "the brand name of this serum and list its three key active ingredients: the primary growth factor that supports "
    "collagen production, the barley-derived growth factor that fortifies the skin barrier, and the hydrating compound "
    "that provides plumping effects. Provide reference URLs that verify both the serum's use in Moore's routine and its "
    "ingredient composition."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SerumExtraction(BaseModel):
    # Identity
    brand: Optional[str] = None
    product_name: Optional[str] = None
    brand_official_url: Optional[str] = None  # Official product page if provided

    # Routine-use references (specific required sources)
    new_beauty_url: Optional[str] = None
    harpers_bazaar_url: Optional[str] = None
    us_weekly_url: Optional[str] = None

    # Any other routine-use URLs mentioned in the answer
    routine_use_urls: List[str] = Field(default_factory=list)

    # Ingredient composition references (pages that list/claim ingredients/functions)
    ingredient_reference_urls: List[str] = Field(default_factory=list)

    # References that explicitly state the formula has seven ingredients (or list exactly seven)
    minimal_formula_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_serum_info() -> str:
    return """
Extract the following fields exactly as they appear in the answer:

1) brand: The brand name of the serum mentioned for Demi Moore’s nighttime routine.
2) product_name: The product/serum name if provided (e.g., "EGF Serum", "EGF Power Serum").
3) brand_official_url: The official brand product page URL for this serum, if included.

Routine-use references:
4) new_beauty_url: The first URL in the answer from the NewBeauty domain (newbeauty.com) that supports or states that Demi Moore uses this serum (ideally mentioning Mary Wiles).
5) harpers_bazaar_url: The first URL in the answer from the Harper’s Bazaar domain (harpersbazaar.com) or Harper’s Bazaar official YouTube/video page that supports Moore’s routine-use claim.
6) us_weekly_url: The first URL in the answer from the US Weekly domain (usmagazine.com) that supports Moore’s routine-use claim.
7) routine_use_urls: Any other URLs in the answer that support Moore’s use of this serum in her nighttime routine.

Ingredient composition references:
8) ingredient_reference_urls: All URLs in the answer that explicitly list or confirm the serum includes EGF (epidermal growth factor), barley-derived KGF (keratinocyte growth factor), and hyaluronic acid, and/or describe their functions.
9) minimal_formula_urls: All URLs in the answer that explicitly state the serum has a notably pure/minimal formula with seven ingredients total or list exactly seven ingredients.

Special rules:
- Extract only URLs explicitly present in the answer (plain, markdown, etc.). Do not invent URLs.
- Ensure full valid URLs (prepend http:// if missing).
- Avoid duplicates across fields; if a URL appears in multiple categories, still include it in all applicable arrays.
- If a required URL is not present in the answer, set the field to null (for single URL fields) or an empty array (for lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_distinct_urls(*url_groups: List[Optional[str] | List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for group in url_groups:
        if group is None:
            continue
        if isinstance(group, list):
            for u in group:
                if isinstance(u, str) and u.strip() and u not in seen:
                    seen.add(u)
                    merged.append(u)
        else:
            if isinstance(group, str) and group.strip() and group not in seen:
                seen.add(group)
                merged.append(group)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identity_and_constraints(evaluator: Evaluator, parent_node, ex: SerumExtraction) -> None:
    """
    Build and verify the 'serum_identity_and_constraint_fit' subtree.
    All children are critical under this critical parent.
    """
    node = evaluator.add_parallel(
        id="serum_identity_and_constraint_fit",
        desc="Confirm the identified product matches the constrained serum identity and positioning.",
        parent=parent_node,
        critical=True,
    )

    # brand_is_bioeffect
    brand_leaf = evaluator.add_leaf(
        id="brand_is_bioeffect",
        desc="Answer identifies the serum brand as BioEffect.",
        parent=node,
        critical=True,
    )
    brand_claim = (
        f"The answer identifies the serum brand as BIOEFFECT (BioEffect). "
        f"Extracted brand from the answer: '{ex.brand}'. Consider case-insensitive and minor variants."
    )
    await evaluator.verify(
        claim=brand_claim,
        node=brand_leaf,
        additional_instruction="Judge based solely on whether the answer itself names the brand as BIOEFFECT/BioEffect. Case/format variations are acceptable."
    )

    # nighttime_serum_use
    routine_sources = _merge_distinct_urls(
        [ex.new_beauty_url] if ex.new_beauty_url else [],
        [ex.harpers_bazaar_url] if ex.harpers_bazaar_url else [],
        [ex.us_weekly_url] if ex.us_weekly_url else [],
        ex.routine_use_urls
    )
    night_leaf = evaluator.add_leaf(
        id="nighttime_serum_use",
        desc="Answer states the product is a serum used in Demi Moore's nighttime skincare routine.",
        parent=node,
        critical=True,
    )
    night_claim = (
        "The provided references confirm that Demi Moore uses this serum (a BIOEFFECT serum) as part of her nighttime skincare routine "
        "(phrasing like 'night routine', 'before bed', or equivalent is acceptable)."
    )
    await evaluator.verify(
        claim=night_claim,
        node=night_leaf,
        sources=routine_sources if routine_sources else None,
        additional_instruction=(
            "Evaluate the URLs. If no valid routine-use URL is provided in the answer, mark this as Incorrect. "
            "Focus on confirming the product is used specifically at night (not just daytime)."
        )
    )

    # sensitive_skin_and_avoids_harsh_or_fragrance
    sensitive_leaf = evaluator.add_leaf(
        id="sensitive_skin_and_avoids_harsh_or_fragrance",
        desc="Answer states Moore has sensitive skin and avoids harsh ingredients and/or fragrances.",
        parent=node,
        critical=True,
    )
    sensitive_claim = (
        "The references support that Demi Moore has sensitive skin and avoids harsh ingredients and/or fragrance (or uses fragrance-free products)."
    )
    await evaluator.verify(
        claim=sensitive_claim,
        node=sensitive_leaf,
        sources=routine_sources if routine_sources else None,
        additional_instruction=(
            "Accept equivalent phrasings that clearly indicate sensitive skin and avoidance of harsh ingredients and/or fragrance-free preference. "
            "If no relevant supporting URL is provided, return Incorrect."
        )
    )

    # pure_minimal_seven_ingredients
    minimal_sources = _merge_distinct_urls(ex.minimal_formula_urls, ex.ingredient_reference_urls, [ex.brand_official_url] if ex.brand_official_url else [])
    minimal_leaf = evaluator.add_leaf(
        id="pure_minimal_seven_ingredients",
        desc="Answer states the serum has a notably pure/minimal formula with seven ingredients total.",
        parent=node,
        critical=True,
    )
    minimal_claim = (
        "This serum is known for a notably pure/minimal formulation with a total of seven (7) ingredients."
    )
    await evaluator.verify(
        claim=minimal_claim,
        node=minimal_leaf,
        sources=minimal_sources if minimal_sources else None,
        additional_instruction=(
            "The page(s) must either explicitly say 'seven ingredients' or clearly list exactly seven distinct ingredients. "
            "Prefer pages about the same BIOEFFECT serum mentioned in the answer. If no valid supporting URL is provided, mark Incorrect."
        )
    )

    # luxury_anti_aging_positioning
    luxury_sources = _merge_distinct_urls([ex.brand_official_url] if ex.brand_official_url else [], ex.ingredient_reference_urls)
    luxury_leaf = evaluator.add_leaf(
        id="luxury_anti_aging_positioning",
        desc="Answer states the product is positioned/marketed as a luxury anti-aging skincare item.",
        parent=node,
        critical=True,
    )
    luxury_claim = (
        "This serum is positioned/marketed as a luxury anti-aging skincare item (e.g., targeting wrinkles, firmness, rejuvenation)."
    )
    await evaluator.verify(
        claim=luxury_claim,
        node=luxury_leaf,
        sources=luxury_sources if luxury_sources else None,
        additional_instruction=(
            "Confirm from the product/brand/retailer page(s). Accept synonymous phrasing indicating premium/luxury status and anti-aging focus. "
            "If no valid supporting URL is provided, mark Incorrect."
        )
    )


async def build_ingredients_and_functions(evaluator: Evaluator, parent_node, ex: SerumExtraction) -> None:
    """
    Build and verify the 'ingredients_and_functions' subtree.
    All children are critical under this critical parent.
    """
    node = evaluator.add_parallel(
        id="ingredients_and_functions",
        desc="List the three key active ingredients and match each to the function specified in the question/constraints.",
        parent=parent_node,
        critical=True,
    )

    comp_sources = _merge_distinct_urls(ex.ingredient_reference_urls, [ex.brand_official_url] if ex.brand_official_url else [])

    # egf_primary_growth_factor_collagen_support
    egf_leaf = evaluator.add_leaf(
        id="egf_primary_growth_factor_collagen_support",
        desc="Answer lists Epidermal Growth Factor (EGF) as the primary growth factor and states it supports natural collagen production.",
        parent=node,
        critical=True,
    )
    egf_claim = (
        "The serum's primary active growth factor is EGF (Epidermal Growth Factor), and it supports natural collagen production."
    )
    await evaluator.verify(
        claim=egf_claim,
        node=egf_leaf,
        sources=comp_sources if comp_sources else None,
        additional_instruction=(
            "Confirm both inclusion of EGF and the stated function of supporting collagen production from the provided URLs. "
            "If either piece is missing or URLs are absent, mark Incorrect."
        )
    )

    # kgf_barley_derived_barrier_support
    kgf_leaf = evaluator.add_leaf(
        id="kgf_barley_derived_barrier_support",
        desc="Answer lists the barley-derived Keratinocyte Growth Factor (KGF) and states it fortifies/supports the skin barrier.",
        parent=node,
        critical=True,
    )
    kgf_claim = (
        "The serum includes barley-derived KGF (Keratinocyte Growth Factor), and it fortifies/supports the skin barrier."
    )
    await evaluator.verify(
        claim=kgf_claim,
        node=kgf_leaf,
        sources=comp_sources if comp_sources else None,
        additional_instruction=(
            "The evidence should explicitly mention KGF (barley-derived) in this serum and associate it with skin barrier support. "
            "If not explicitly supported by the URLs (or URLs missing), mark Incorrect."
        )
    )

    # hyaluronic_acid_hydration_plumping
    ha_leaf = evaluator.add_leaf(
        id="hyaluronic_acid_hydration_plumping",
        desc="Answer lists hyaluronic acid and states it provides hydration and plumping effects.",
        parent=node,
        critical=True,
    )
    ha_claim = (
        "The serum contains hyaluronic acid, which provides hydration and plumping effects."
    )
    await evaluator.verify(
        claim=ha_claim,
        node=ha_leaf,
        sources=comp_sources if comp_sources else None,
        additional_instruction=(
            "Confirm hyaluronic acid is included and associated with hydration/plumping on the cited page(s). "
            "If absent or not supported by the URLs (or URLs missing), mark Incorrect."
        )
    )

    # rejuvenation_support_claim
    rej_leaf = evaluator.add_leaf(
        id="rejuvenation_support_claim",
        desc="Answer states the serum supports skin rejuvenation.",
        parent=node,
        critical=True,
    )
    rej_claim = "This serum supports skin rejuvenation (anti-aging benefits)."
    await evaluator.verify(
        claim=rej_claim,
        node=rej_leaf,
        sources=comp_sources if comp_sources else None,
        additional_instruction=(
            "Confirm that the product page(s) or authoritative retailer page(s) describe rejuvenation/anti-aging benefits. "
            "If not supported or URLs are missing, mark Incorrect."
        )
    )


async def build_required_references(evaluator: Evaluator, parent_node, ex: SerumExtraction) -> None:
    """
    Build and verify the 'required_references' subtree, including specific source checks and ingredient composition proof.
    All nodes are critical.
    """
    node = evaluator.add_parallel(
        id="required_references",
        desc="Provide reference URLs that verify (a) use in Moore’s routine and (b) ingredient composition, matching the source constraints.",
        parent=parent_node,
        critical=True,
    )

    # Routine-use specific required sources
    routine_required = evaluator.add_parallel(
        id="routine_use_references_include_required_sources",
        desc="References for routine use include the required sources per constraints.",
        parent=node,
        critical=True,
    )

    # New Beauty reference (Mary Wiles)
    nb_leaf = evaluator.add_leaf(
        id="new_beauty_mary_wiles_reference",
        desc="Provide a reference URL to New Beauty showing Mary Wiles endorsing/confirming the serum in Moore's routine.",
        parent=routine_required,
        critical=True,
    )
    nb_claim = (
        "This URL is a NewBeauty (newbeauty.com) page that confirms Demi Moore uses the BIOEFFECT serum in her routine, "
        "ideally citing or quoting her makeup artist Mary Wiles."
    )
    await evaluator.verify(
        claim=nb_claim,
        node=nb_leaf,
        sources=ex.new_beauty_url if ex.new_beauty_url else None,
        additional_instruction=(
            "Only accept if the page is from newbeauty.com and clearly supports Moore's routine use of the serum. "
            "If the URL is missing, not NewBeauty, or does not support the claim, mark Incorrect."
        )
    )

    # Harper's Bazaar reference
    hb_leaf = evaluator.add_leaf(
        id="harpers_bazaar_video_reference",
        desc="Provide a reference URL to a Harper’s Bazaar video/source that supports the routine-use claim.",
        parent=routine_required,
        critical=True,
    )
    hb_claim = (
        "This URL is a Harper’s Bazaar source (harpersbazaar.com or an official Harper’s Bazaar video post) that "
        "supports Demi Moore's use of the BIOEFFECT serum in her routine."
    )
    await evaluator.verify(
        claim=hb_claim,
        node=hb_leaf,
        sources=ex.harpers_bazaar_url if ex.harpers_bazaar_url else None,
        additional_instruction=(
            "Only accept if the page is from harpersbazaar.com (or Harper’s Bazaar official video channel/page) and supports the routine-use claim. "
            "If the URL is missing or not supportive, mark Incorrect."
        )
    )

    # US Weekly reference
    usw_leaf = evaluator.add_leaf(
        id="us_weekly_reference",
        desc="Provide a reference URL to US Weekly that supports the routine-use claim.",
        parent=routine_required,
        critical=True,
    )
    usw_claim = (
        "This URL is a US Weekly (usmagazine.com) page that supports Demi Moore's use of the BIOEFFECT serum in her routine."
    )
    await evaluator.verify(
        claim=usw_claim,
        node=usw_leaf,
        sources=ex.us_weekly_url if ex.us_weekly_url else None,
        additional_instruction=(
            "Only accept if the page is from usmagazine.com and clearly supports the routine-use claim. "
            "If the URL is missing or not supportive, mark Incorrect."
        )
    )

    # Ingredient composition reference (EGF + barley KGF + HA)
    comp_leaf = evaluator.add_leaf(
        id="ingredient_composition_reference_url",
        desc="Provide at least one reference URL that verifies the serum’s ingredient composition includes EGF, barley-derived KGF, and hyaluronic acid.",
        parent=node,
        critical=True,
    )
    comp_claim = (
        "The provided reference(s) verify that this same serum includes EGF (epidermal growth factor), barley-derived KGF "
        "(keratinocyte growth factor), and hyaluronic acid."
    )
    await evaluator.verify(
        claim=comp_claim,
        node=comp_leaf,
        sources=ex.ingredient_reference_urls if ex.ingredient_reference_urls else None,
        additional_instruction=(
            "The evidence must be about the same BIOEFFECT serum cited in the answer, and must explicitly include all three: "
            "EGF, barley-derived KGF, and hyaluronic acid. If any is missing or the URLs are absent, mark Incorrect."
        )
    )

    # Seven-ingredient minimal formula reference
    seven_leaf = evaluator.add_leaf(
        id="seven_ingredient_minimal_formula_reference",
        desc="Provide at least one reference URL that verifies the serum has a notably pure/minimal formula with seven ingredients total.",
        parent=node,
        critical=True,
    )
    seven_claim = (
        "The provided reference(s) verify that this serum has a notably pure/minimal formula with exactly seven total ingredients."
    )
    seven_sources = _merge_distinct_urls(ex.minimal_formula_urls, ex.ingredient_reference_urls, [ex.brand_official_url] if ex.brand_official_url else [])
    await evaluator.verify(
        claim=seven_claim,
        node=seven_leaf,
        sources=seven_sources if seven_sources else None,
        additional_instruction=(
            "The page must either explicitly say 'seven ingredients' or list exactly seven distinct ingredients. "
            "If not clearly supported or URLs missing, mark Incorrect."
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
    Evaluate an answer for the Demi Moore BIOEFFECT serum task using the Mind2Web2 framework.
    """
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

    # Extraction
    ex: SerumExtraction = await evaluator.extract(
        prompt=prompt_extract_serum_info(),
        template_class=SerumExtraction,
        extraction_name="serum_extraction",
    )

    # Record some custom info to aid debugging
    evaluator.add_custom_info(
        info={
            "brand": ex.brand,
            "product_name": ex.product_name,
            "brand_official_url": ex.brand_official_url,
            "new_beauty_url": ex.new_beauty_url,
            "harpers_bazaar_url": ex.harpers_bazaar_url,
            "us_weekly_url": ex.us_weekly_url,
            "routine_use_urls_count": len(ex.routine_use_urls),
            "ingredient_reference_urls_count": len(ex.ingredient_reference_urls),
            "minimal_formula_urls_count": len(ex.minimal_formula_urls),
        },
        info_type="extraction_overview",
        info_name="extraction_overview",
    )

    # Build a critical top-level node (since initialize() root is non-critical by design)
    top_critical = evaluator.add_parallel(
        id="root_critical",
        desc="Identify the serum brand used in Demi Moore's nighttime routine, list the three specified key active ingredients with their specified functions, and provide URLs verifying routine use and ingredient composition per constraints.",
        parent=root,
        critical=True,
    )

    # Subtrees as per rubric
    await build_identity_and_constraints(evaluator, top_critical, ex)
    await build_ingredients_and_functions(evaluator, top_critical, ex)
    await build_required_references(evaluator, top_critical, ex)

    # Return structured evaluation summary
    return evaluator.get_summary()