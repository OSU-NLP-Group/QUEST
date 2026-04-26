import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ------------------------------------------------------------------------------
# Task constants
# ------------------------------------------------------------------------------
TASK_ID = "luxury_beauty_collabs_2024"
TASK_DESCRIPTION = (
    "Identify two distinct limited-edition luxury beauty collaborations that were officially launched in 2024 "
    "(between January and December). Each collaboration must meet ALL of the following criteria:\n\n"
    "1. The collaboration must be a partnership between a recognized fashion house or designer and an established beauty brand\n"
    "2. Products must be available through at least one luxury department store or high-end beauty retailer (such as "
    "Saks Fifth Avenue, Neiman Marcus, Nordstrom, Harrods, Selfridges, John Lewis, or Sephora)\n"
    "3. The collection must include at least three distinct product categories (for example, combinations of eyeshadow "
    "palettes, lipsticks, bronzers, finishing powders, brushes, or other makeup items)\n"
    "4. At least one product in the collection must be priced at $80 USD or above (or equivalent pricing if initially "
    "marketed in another currency)\n"
    "5. The collection must feature the fashion partner's signature design motif, pattern, or aesthetic integrated into "
    "the product packaging or design\n"
    "6. The collection must be explicitly marketed as limited-edition, seasonal, or exclusive\n"
    "7. For each collaboration, provide the collection name, the beauty brand and fashion partner involved, the official "
    "launch date or month, and at least one reference URL documenting the collaboration details\n\n"
    "Both collaborations must be distinct partnerships (you cannot list the same collaboration twice with different products)."
)

ALLOWED_RETAILER_KEYWORDS = [
    "saksfifthavenue.com", "saks.com",
    "neimanmarcus.com",
    "nordstrom.com",
    "harrods.com",
    "selfridges.com",
    "johnlewis.com",
    "sephora",  # allow all sephora TLDs
]


# ------------------------------------------------------------------------------
# Data Models for Extraction
# ------------------------------------------------------------------------------
class Collaboration(BaseModel):
    collection_name: Optional[str] = None
    beauty_brand: Optional[str] = None
    fashion_partner: Optional[str] = None
    launch_timing_text: Optional[str] = None  # e.g., "October 2024", "Launched September 2024"
    product_categories: List[str] = Field(default_factory=list)  # categories like "eyeshadow palette", "lipstick", etc.
    reference_urls: List[str] = Field(default_factory=list)  # product/press/brand/announcement pages
    retailer_urls: List[str] = Field(default_factory=list)   # retailer listings (Saks/NM/Nordstrom/Harrods/Selfridges/JohnLewis/Sephora)
    price_example_text: Optional[str] = None  # e.g., "Face Palette $95", "£75 lipstick" (optional helper)
    design_motif_text: Optional[str] = None   # description of motif/pattern/house aesthetic on packaging
    limited_edition_text: Optional[str] = None  # text explicitly stating limited/exclusive/seasonal


class CollaborationsExtraction(BaseModel):
    collaborations: List[Collaboration] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Extraction Prompt
# ------------------------------------------------------------------------------
def prompt_extract_collaborations() -> str:
    return """
    Extract up to two distinct limited-edition luxury beauty collaborations mentioned in the answer that launched in 2024.
    For each collaboration, extract the following fields (return null or empty if not explicitly provided in the answer):
    - collection_name: The collaboration or collection name (string)
    - beauty_brand: The established beauty brand involved (string)
    - fashion_partner: The recognized fashion house or designer partner (string)
    - launch_timing_text: An official launch date or month text as stated (e.g., "September 2024", "Launched in October 2024")
    - product_categories: A list of distinct product categories included (e.g., "eyeshadow palette", "lipstick", "bronzer", "finishing powder", "brushes")
    - reference_urls: A list of URLs (brand pages, press releases, announcement articles, product pages) documenting collaboration details
    - retailer_urls: A list of URLs from luxury/high-end retailers (e.g., Saks Fifth Avenue, Neiman Marcus, Nordstrom, Harrods, Selfridges, John Lewis, Sephora) that sell/stock products from this collaboration
    - price_example_text: A text snippet with a price at or above $80 USD (or clearly equivalent in another currency), if provided (e.g., "Face palette $95 USD")
    - design_motif_text: A short description of the fashion partner's signature design motif/pattern/aesthetic integrated into packaging or product design, if mentioned
    - limited_edition_text: A text snippet explicitly indicating "limited edition", "seasonal", or "exclusive", if mentioned

    Important:
    - Only extract collaborations that the answer explicitly mentions.
    - Preserve URLs exactly as they appear; include full protocols if present.
    - If more than two collaborations are present, include only the first two.
    """


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def norm_name(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def filter_allowed_retailer_urls(urls: List[str]) -> List[str]:
    allowed = []
    for u in urls:
        try:
            netloc = urlparse(u).netloc.lower()
        except Exception:
            netloc = ""
        if any(k in netloc for k in ALLOWED_RETAILER_KEYWORDS):
            allowed.append(u)
    return allowed


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def safe_join(lst: List[str]) -> str:
    return ", ".join(lst) if lst else ""


# ------------------------------------------------------------------------------
# Verification per Collaboration
# ------------------------------------------------------------------------------
async def verify_collaboration(
    evaluator: Evaluator,
    parent_node,
    collab: Collaboration,
    collab_idx: int
) -> None:
    """
    Build the verification subtree for a single collaboration according to the rubric.
    collab_idx is 1-based (1 or 2).
    """
    collab_node = evaluator.add_parallel(
        id=f"collaboration_{collab_idx}",
        desc=f"{'First' if collab_idx == 1 else 'Second'} qualifying luxury beauty collaboration (with required details)",
        parent=parent_node,
        critical=False
    )

    # 1) Provides collection/collaboration name (critical, existence)
    has_collection_name = bool(collab.collection_name and collab.collection_name.strip())
    evaluator.add_custom_node(
        result=has_collection_name,
        id=f"provides_collection_name_{collab_idx}",
        desc="Provides the collection/collaboration name",
        parent=collab_node,
        critical=True
    )

    # 2) Explicitly names both the fashion house/designer and the beauty brand (critical, existence)
    has_both_partners = bool(collab.beauty_brand and collab.fashion_partner and collab.beauty_brand.strip() and collab.fashion_partner.strip())
    evaluator.add_custom_node(
        result=has_both_partners,
        id=f"provides_partner_names_{collab_idx}",
        desc="Explicitly names both the fashion house/designer and the beauty brand",
        parent=collab_node,
        critical=True
    )

    # 11) Provides at least one valid reference URL documenting the collaboration details (critical, existence)
    has_reference_url = bool(collab.reference_urls)
    evaluator.add_custom_node(
        result=has_reference_url,
        id=f"reference_url_{collab_idx}",
        desc="Provides at least one valid reference URL documenting the collaboration details",
        parent=collab_node,
        critical=True
    )

    # Prepare frequently used data
    name = collab.collection_name or "the collection"
    brand = collab.beauty_brand or "the beauty brand"
    partner = collab.fashion_partner or "the fashion partner"
    refs = collab.reference_urls or []
    retailers_all = collab.retailer_urls or []
    retailers_allowed = filter_allowed_retailer_urls(retailers_all)
    all_urls = unique_preserve_order((refs or []) + (retailers_allowed or []))

    # Helper: instruction to enforce failure when no sources available
    no_sources_fail_hint = "If no URL is provided for verification, return Incorrect."

    # 3) Launch timing in 2024 (critical, verify against sources)
    launch_node = evaluator.add_leaf(
        id=f"launch_timing_{collab_idx}",
        desc="Provides an official launch date or month, and it is in 2024 (January–December)",
        parent=collab_node,
        critical=True
    )
    launch_text = collab.launch_timing_text or "a launch in 2024"
    claim_launch = (
        f"The collaboration '{name}' between {brand} and {partner} officially launched in 2024 "
        f"(January through December). Reported timing: {launch_text}."
    )
    await evaluator.verify(
        claim=claim_launch,
        node=launch_node,
        sources=refs,
        additional_instruction=(
            "Verify that the launch or release (not just announcement) occurred in 2024. "
            "Accept pages such as brand posts/press, credible media, or retailer pages stating availability dates. "
            f"{no_sources_fail_hint}"
        )
    )

    # 4) Fashion partnership between a recognized fashion house/designer and an established beauty brand (critical)
    partnership_node = evaluator.add_leaf(
        id=f"fashion_partnership_{collab_idx}",
        desc="Collaboration is a partnership between a recognized fashion house/designer and an established beauty brand",
        parent=collab_node,
        critical=True
    )
    claim_partner = (
        f"There is a collaboration between the fashion house/designer '{partner}' and the beauty brand '{brand}' "
        f"on '{name}'."
    )
    await evaluator.verify(
        claim=claim_partner,
        node=partnership_node,
        sources=refs,
        additional_instruction=(
            "The page(s) should explicitly mention a collaboration between the named fashion house/designer and the "
            "beauty brand on this collection. Minor name variants are acceptable. "
            f"{no_sources_fail_hint}"
        )
    )

    # 5) Luxury retailer availability (critical)
    retailer_node = evaluator.add_leaf(
        id=f"luxury_retailer_availability_{collab_idx}",
        desc="Products are available through at least one luxury department store or high-end beauty retailer",
        parent=collab_node,
        critical=True
    )
    claim_retail = (
        "Products from this collaboration were sold by at least one luxury/high-end retailer such as "
        "Saks Fifth Avenue, Neiman Marcus, Nordstrom, Harrods, Selfridges, John Lewis, or Sephora."
    )
    await evaluator.verify(
        claim=claim_retail,
        node=retailer_node,
        sources=retailers_allowed,
        additional_instruction=(
            "Only pass if the provided page(s) is clearly a product/listing page from one of the allowed retailers. "
            "Allowed retailer domains include saksfifthavenue.com/saks.com, neimanmarcus.com, nordstrom.com, "
            "harrods.com, selfridges.com, johnlewis.com, or any sephora.* domain. "
            f"{no_sources_fail_hint}"
        )
    )

    # 6) Product diversity: at least three distinct product categories (critical)
    diversity_node = evaluator.add_leaf(
        id=f"product_diversity_{collab_idx}",
        desc="Collection includes at least three distinct product categories",
        parent=collab_node,
        critical=True
    )
    cats = list({c.strip().lower() for c in (collab.product_categories or []) if c and c.strip()})
    cats_text = safe_join(cats)
    claim_diverse = (
        f"The collection includes at least three distinct product categories (examples from the answer: {cats_text}). "
        "Count categories (e.g., palette, lipstick, bronzer, brush) rather than shade variants or bundle duplicates."
    )
    await evaluator.verify(
        claim=claim_diverse,
        node=diversity_node,
        sources=all_urls,
        additional_instruction=(
            "Confirm there are three or more distinct categories (not just multiple shades of the same product). "
            "If the example categories listed are fewer than three but the pages clearly show >=3 categories, still pass. "
            f"{no_sources_fail_hint}"
        )
    )

    # 7) Luxury price point: >= $80 USD (critical)
    price_node = evaluator.add_leaf(
        id=f"luxury_price_point_{collab_idx}",
        desc="At least one product in the collection is priced at $80 USD or above (or clearly equivalent in another currency)",
        parent=collab_node,
        critical=True
    )
    price_hint = collab.price_example_text or "a price example at or above $80"
    claim_price = (
        f"At least one product in the collaboration '{name}' is priced at $80 USD or above (or clearly above that "
        f"in another currency). Example from the answer (if any): {price_hint}."
    )
    await evaluator.verify(
        claim=claim_price,
        node=price_node,
        sources=all_urls,
        additional_instruction=(
            "Check the product price(s) shown on brand/retailer pages. Equivalent currencies are acceptable if the listed "
            "price is clearly above $80 (e.g., €80+, £70+, or higher). Borderline cases should be conservative. "
            f"{no_sources_fail_hint}"
        )
    )

    # 8) Design integration: signature motif/pattern/aesthetic on packaging (critical)
    design_node = evaluator.add_leaf(
        id=f"design_integration_{collab_idx}",
        desc="Collection features the fashion partner's signature design motif/pattern/aesthetic integrated into packaging or product design",
        parent=collab_node,
        critical=True
    )
    motif_hint = collab.design_motif_text or f"{partner}'s signature motif/pattern"
    claim_design = (
        f"The packaging or product design for '{name}' integrates {motif_hint} associated with {partner}."
    )
    await evaluator.verify(
        claim=claim_design,
        node=design_node,
        sources=refs,
        additional_instruction=(
            "Look for explicit mentions or images of packaging featuring the fashion house's signature motif, "
            "logo pattern, print, or aesthetic. "
            f"{no_sources_fail_hint}"
        )
    )

    # 9) Limited-edition/seasonal/exclusive (critical)
    limited_node = evaluator.add_leaf(
        id=f"limited_edition_status_{collab_idx}",
        desc="Collection is explicitly marketed as limited-edition, seasonal, or exclusive",
        parent=collab_node,
        critical=True
    )
    limited_hint = collab.limited_edition_text or "limited/exclusive wording"
    claim_limited = (
        f"The collection '{name}' is explicitly described as limited-edition, seasonal, or exclusive "
        f"(e.g., wording like '{limited_hint}')."
    )
    await evaluator.verify(
        claim=claim_limited,
        node=limited_node,
        sources=refs,
        additional_instruction=(
            "Look for phrases such as 'limited edition', 'seasonal collection', or 'exclusive'. "
            f"{no_sources_fail_hint}"
        )
    )


# ------------------------------------------------------------------------------
# Main Evaluation
# ------------------------------------------------------------------------------
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
    Evaluate an answer for the luxury beauty collaborations (2024) task.
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

    # Extract up to two collaborations from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_collaborations(),
        template_class=CollaborationsExtraction,
        extraction_name="collaborations_extraction"
    )

    # Normalize to exactly two entries (pad with empty if fewer)
    collabs = extracted.collaborations[:2]
    while len(collabs) < 2:
        collabs.append(Collaboration())

    # Distinct partnerships check (critical)
    bb1, fp1 = norm_name(collabs[0].beauty_brand), norm_name(collabs[0].fashion_partner)
    bb2, fp2 = norm_name(collabs[1].beauty_brand), norm_name(collabs[1].fashion_partner)
    distinct = bool(bb1 and fp1 and bb2 and fp2 and ((bb1 != bb2) or (fp1 != fp2)))
    evaluator.add_custom_node(
        result=distinct,
        id="distinct_partnerships",
        desc="The two collaborations are distinct partnerships (different fashion house/designer–beauty brand combinations)",
        parent=root,
        critical=True
    )

    # Build verification trees for both collaborations
    await verify_collaboration(evaluator, root, collabs[0], 1)
    await verify_collaboration(evaluator, root, collabs[1], 2)

    # Add custom info for debugging/traceability
    evaluator.add_custom_info(
        {
            "allowed_retailer_domains_hint": ALLOWED_RETAILER_KEYWORDS,
            "extracted_overview": [
                {
                    "collection_name": c.collection_name,
                    "beauty_brand": c.beauty_brand,
                    "fashion_partner": c.fashion_partner,
                    "launch_timing_text": c.launch_timing_text,
                    "product_categories": c.product_categories,
                    "reference_urls_count": len(c.reference_urls),
                    "retailer_urls_count": len(c.retailer_urls),
                } for c in collabs
            ]
        },
        info_type="debug_info",
        info_name="evaluation_context"
    )

    return evaluator.get_summary()