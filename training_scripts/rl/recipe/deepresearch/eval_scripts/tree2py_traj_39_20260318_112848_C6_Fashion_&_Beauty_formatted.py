import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_collab_2025"
TASK_DESCRIPTION = """
Identify three distinct celebrity fashion or beauty collaborations that were launched or officially announced in 2025. For each collaboration, you must provide:
(1) Celebrity Name: The full name of the celebrity who collaborated or partnered with the brand;
(2) Brand Name: The name of the fashion or beauty brand partner;
(3) Collaboration Type: Whether it is a celebrity-designed fashion capsule collection OR a celebrity beauty brand partnership with a product launch;
(4) Product Details: The number of items/products in the collection (for fashion capsules) OR the specific product type (for beauty partnerships), and the product categories included (e.g., outerwear and accessories, or lip care products);
(5) Pricing: The price or price range for products in the collaboration (in USD);
(6) Launch Information: The specific launch date or launch month in 2025;
(7) URL References: Provide URLs from official brand sources or major fashion/beauty publications that confirm the celebrity's involvement in the collaboration, the brand partnership, and the pricing information.

Each collaboration must meet these requirements:
- Must have been launched or officially announced between January 1, 2025 and December 31, 2025;
- Must be either a celebrity-designed fashion capsule collection with specific pieces OR a celebrity beauty brand partnership that includes a product launch;
- All information must be verifiable through the provided URLs.
"""

YEAR_REQUIRED = 2025

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CollaborationItem(BaseModel):
    celebrity_name: Optional[str] = None
    celebrity_urls: List[str] = Field(default_factory=list)

    brand_name: Optional[str] = None
    brand_urls: List[str] = Field(default_factory=list)

    collab_type: Optional[str] = None  # e.g., "celebrity-designed fashion capsule collection" or "beauty partnership product launch"

    product_scope: Optional[str] = None  # e.g., "12-piece capsule", "lip oil", "fragrance"
    product_categories: List[str] = Field(default_factory=list)  # e.g., ["outerwear", "accessories"]

    pricing: Optional[str] = None  # e.g., "$39–$129"
    pricing_urls: List[str] = Field(default_factory=list)

    launch_date: Optional[str] = None  # Specific date or month in 2025, e.g., "June 2025" or "2025-06-15"
    launch_urls: List[str] = Field(default_factory=list)

    reference_urls: List[str] = Field(default_factory=list)  # Other official brand sources or major pubs


class CollaborationExtraction(BaseModel):
    collaborations: List[CollaborationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_collaborations() -> str:
    return """
    From the provided answer, extract up to three distinct celebrity fashion or beauty collaborations that were launched or officially announced in 2025.

    For each collaboration found in the answer, extract the following fields:

    - celebrity_name: Full name of the celebrity collaborator.
    - celebrity_urls: Array of URLs explicitly cited in the answer that confirm the celebrity's involvement (official brand pages, press releases, or major fashion/beauty publications).
    - brand_name: Name of the fashion or beauty brand partner.
    - brand_urls: Array of URLs explicitly cited in the answer that confirm the brand partnership (official brand pages, press releases, or major fashion/beauty publications).
    - collab_type: A short phrase describing the collaboration type exactly as stated in the answer (e.g., "celebrity-designed fashion capsule", "beauty partnership with product launch", "makeup collection", "fragrance launch").
    - product_scope: Either the number of items (for fashion capsule) or the specific product type (for beauty partnership) as stated in the answer (e.g., "12-piece capsule", "lip oil", "fragrance", "skin-care trio").
    - product_categories: Array of product categories explicitly listed in the answer (e.g., ["outerwear", "accessories", "lip products", "fragrance"]).
    - pricing: The price or price range string exactly as given in the answer in USD (e.g., "$39", "$49–$129", "from $59").
    - pricing_urls: Array of URLs explicitly cited in the answer that confirm the pricing information (brand pages, US retailer pages, or major publications).
    - launch_date: The specific launch date or launch month in 2025 exactly as written in the answer (e.g., "June 2025", "2025-06-15", "June 12, 2025").
    - launch_urls: Array of URLs explicitly cited in the answer that confirm the launch date/month (brand press releases, product pages, news articles).
    - reference_urls: Any additional URLs cited in the answer that discuss the collaboration and can help verify collaboration type and details (avoid duplicates with the above).

    Rules:
    - Only extract information explicitly present in the answer text.
    - Only include valid URLs that appear in the answer (plain URLs or markdown links).
    - If a field is not present for a collaboration, set the field to null (for strings) or an empty array (for lists).
    - Extract the collaborations in the same order they appear in the answer; return at most three.

    Return a JSON object with a 'collaborations' array where each element has exactly the fields listed above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_sources(
    item: CollaborationItem,
    kinds: Optional[List[str]] = None
) -> List[str]:
    # kinds can include: "celebrity", "brand", "pricing", "launch", "reference", "all"
    if kinds is None or "all" in kinds:
        all_urls = (
            item.celebrity_urls
            + item.brand_urls
            + item.pricing_urls
            + item.launch_urls
            + item.reference_urls
        )
        return dedup_preserve_order(all_urls)
    mapping = {
        "celebrity": item.celebrity_urls,
        "brand": item.brand_urls,
        "pricing": item.pricing_urls,
        "launch": item.launch_urls,
        "reference": item.reference_urls,
    }
    urls: List[str] = []
    for k in kinds:
        urls.extend(mapping.get(k, []))
    return dedup_preserve_order(urls)


def nonempty_or_placeholder(s: Optional[str], placeholder: str = "") -> str:
    return s.strip() if isinstance(s, str) else placeholder


# --------------------------------------------------------------------------- #
# Verification logic per collaboration                                        #
# --------------------------------------------------------------------------- #
async def verify_single_collaboration(
    evaluator: Evaluator,
    parent_node,
    item: CollaborationItem,
    index: int,
) -> None:
    """
    Build verification subtree and run checks for one collaboration.
    Follows the provided rubric tree structure.
    """
    human_idx = index + 1

    # Collaboration root (sequential)
    collab_node = evaluator.add_sequential(
        id=f"collaboration_{human_idx}",
        desc=f"{['First','Second','Third'][index] if index < 3 else f'#{human_idx}'} qualifying celebrity fashion or beauty collaboration",
        parent=parent_node,
        critical=False
    )

    # 1) Category & Timing check (critical, parallel)
    cat_node = evaluator.add_parallel(
        id=f"collab{human_idx}_category_check",
        desc="Verify collaboration category and timing requirements",
        parent=collab_node,
        critical=True
    )

    # 1.a) Type verification (critical leaf)
    type_leaf = evaluator.add_leaf(
        id=f"collab{human_idx}_type",
        desc="Collaboration must be either a celebrity-designed fashion capsule collection OR a celebrity beauty brand partnership with product launch",
        parent=cat_node,
        critical=True
    )
    celeb = nonempty_or_placeholder(item.celebrity_name, "")
    brand = nonempty_or_placeholder(item.brand_name, "")
    ctype = nonempty_or_placeholder(item.collab_type, "")
    type_claim = (
        f"The collaboration between '{celeb}' and '{brand}' is described as '{ctype}', "
        f"and this qualifies as one of the allowed categories: "
        f"(a) a celebrity-designed fashion capsule collection, or "
        f"(b) a celebrity beauty brand partnership that includes a product launch."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=collect_sources(item, ["brand", "celebrity", "reference", "launch"]),
        additional_instruction=(
            "Verify from the provided URLs whether the described collaboration fits one of the two allowed categories. "
            "Accept reasonable synonyms such as 'capsule', 'limited-edition collection', 'designed by', "
            "'beauty collaboration with product launch', 'makeup collection launch', or 'fragrance launch'. "
            "If sources are irrelevant or absent, mark as not supported."
        ),
    )

    # 1.b) Year in 2025 (critical leaf)
    year_leaf = evaluator.add_leaf(
        id=f"collab{human_idx}_year",
        desc="Collaboration must have been launched or officially announced between January 1, 2025 and December 31, 2025",
        parent=cat_node,
        critical=True
    )
    year_claim = (
        f"The collaboration between '{celeb}' and '{brand}' was launched or officially announced in {YEAR_REQUIRED}."
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=collect_sources(item, ["launch", "brand", "reference"]),
        additional_instruction=(
            f"Confirm that the page(s) show a launch or official announcement date occurring in calendar year {YEAR_REQUIRED}. "
            "Month-level precision is acceptable (e.g., 'June 2025'). "
            "If the date clearly falls outside 2025, mark as not supported."
        ),
    )

    # 2) Details (non-critical, parallel)
    details_node = evaluator.add_parallel(
        id=f"collab{human_idx}_details",
        desc="Detailed information about the collaboration",
        parent=collab_node,
        critical=False
    )

    # 2.a) Parties (critical, parallel)
    parties_node = evaluator.add_parallel(
        id=f"collab{human_idx}_parties",
        desc="Identify the collaborating parties",
        parent=details_node,
        critical=True
    )

    # 2.a.i) Celebrity (critical, parallel) -> has child 'celebrity_url'
    celeb_node = evaluator.add_parallel(
        id=f"collab{human_idx}_celebrity",
        desc="Provide the full name of the celebrity collaborator",
        parent=parties_node,
        critical=True
    )
    celeb_url_leaf = evaluator.add_leaf(
        id=f"collab{human_idx}_celebrity_url",
        desc="URL reference confirming the celebrity's involvement",
        parent=celeb_node,
        critical=True
    )
    celeb_url_claim = (
        f"The provided page(s) confirm that '{celeb}' is involved as the celebrity collaborator/partner "
        f"for a {YEAR_REQUIRED} collaboration with '{brand}'."
    )
    await evaluator.verify(
        claim=celeb_url_claim,
        node=celeb_url_leaf,
        sources=collect_sources(item, ["celebrity", "brand", "reference"]),
        additional_instruction=(
            "Look for explicit mentions that the named celebrity is collaborating or partnering with the brand on this project. "
            "Accept official brand/retailer pages or major fashion/beauty publications. "
            "If no URL confirms the involvement, mark as not supported."
        ),
    )

    # 2.a.ii) Brand (critical, parallel) -> has child 'brand_url'
    brand_node = evaluator.add_parallel(
        id=f"collab{human_idx}_brand",
        desc="Provide the name of the fashion or beauty brand partner",
        parent=parties_node,
        critical=True
    )
    brand_url_leaf = evaluator.add_leaf(
        id=f"collab{human_idx}_brand_url",
        desc="URL reference confirming the brand partnership",
        parent=brand_node,
        critical=True
    )
    brand_url_claim = (
        f"The provided page(s) confirm that the collaborating brand is '{brand}' for the {YEAR_REQUIRED} collaboration with '{celeb}'."
    )
    await evaluator.verify(
        claim=brand_url_claim,
        node=brand_url_leaf,
        sources=collect_sources(item, ["brand", "reference"]),
        additional_instruction=(
            "Verify that the brand named is explicitly the collaboration partner for this 2025 project. "
            "Accept official brand pages, press releases, or major fashion/beauty publications."
        ),
    )

    # 2.b) Product info (critical, parallel)
    product_node = evaluator.add_parallel(
        id=f"collab{human_idx}_product_info",
        desc="Product specifications for the collaboration",
        parent=details_node,
        critical=True
    )

    # 2.b.i) Product scope (critical leaf)
    scope_leaf = evaluator.add_leaf(
        id=f"collab{human_idx}_product_scope",
        desc="Specify the number of items/products in the collaboration or the specific product type",
        parent=product_node,
        critical=True
    )
    scope_text = nonempty_or_placeholder(item.product_scope, "")
    scope_claim = (
        f"The collaboration's product scope is correctly stated as: '{scope_text}'. "
        "For a fashion capsule, this should indicate the number of pieces; "
        "for a beauty partnership, this should indicate the specific product type that launched."
    )
    await evaluator.verify(
        claim=scope_claim,
        node=scope_leaf,
        sources=collect_sources(item, ["brand", "launch", "reference"]),
        additional_instruction=(
            "Confirm the number of pieces when it is a fashion capsule, or confirm the specific product type for a beauty launch. "
            "Accept synonyms and near-equivalents (e.g., '12-piece capsule' vs. '12 items')."
        ),
    )

    # 2.b.ii) Product categories (critical leaf)
    categories_leaf = evaluator.add_leaf(
        id=f"collab{human_idx}_product_categories",
        desc="Describe the product categories included (e.g., outerwear, accessories, lip products, fragrances)",
        parent=product_node,
        critical=True
    )
    categories_text = ", ".join(item.product_categories) if item.product_categories else ""
    categories_claim = (
        f"The collaboration includes the following product categories: {categories_text}."
    )
    await evaluator.verify(
        claim=categories_claim,
        node=categories_leaf,
        sources=collect_sources(item, ["brand", "reference", "launch"]),
        additional_instruction=(
            "Check that the listed categories are explicitly mentioned or clearly implied by the provided pages. "
            "Minor wording differences are acceptable."
        ),
    )

    # 2.c) Pricing (critical, parallel)
    pricing_node = evaluator.add_parallel(
        id=f"collab{human_idx}_pricing",
        desc="Pricing information for the collaboration",
        parent=details_node,
        critical=True
    )

    # 2.c.i) Price range (critical, parallel) -> has child 'pricing_url'
    price_range_node = evaluator.add_parallel(
        id=f"collab{human_idx}_price_range",
        desc="Provide the price or price range for products in the collaboration",
        parent=pricing_node,
        critical=True
    )
    pricing_url_leaf = evaluator.add_leaf(
        id=f"collab{human_idx}_pricing_url",
        desc="URL reference confirming the pricing information",
        parent=price_range_node,
        critical=True
    )
    pricing_text = nonempty_or_placeholder(item.pricing, "")
    pricing_claim = (
        f"The pricing for this collaboration is correctly given as '{pricing_text}' (USD)."
    )
    await evaluator.verify(
        claim=pricing_claim,
        node=pricing_url_leaf,
        sources=collect_sources(item, ["pricing", "brand", "reference"]),
        additional_instruction=(
            "Confirm the stated price or price range using the provided URLs. "
            "Prefer official brand or US retailer pricing; major publications reporting USD are also acceptable. "
            "If no source shows matching pricing in USD or convertible equivalence, mark as not supported."
        ),
    )

    # 2.d) Launch date/month (critical leaf)
    launch_leaf = evaluator.add_leaf(
        id=f"collab{human_idx}_launch_date",
        desc="Provide the specific launch date or launch month for the collaboration",
        parent=details_node,
        critical=True
    )
    launch_text = nonempty_or_placeholder(item.launch_date, "")
    launch_claim = (
        f"The collaboration's launch information is correctly stated as '{launch_text}', "
        f"and it occurs in {YEAR_REQUIRED}."
    )
    await evaluator.verify(
        claim=launch_claim,
        node=launch_leaf,
        sources=collect_sources(item, ["launch", "brand", "reference"]),
        additional_instruction=(
            f"Verify that the specified launch date/month is supported by the provided URLs and that it falls in {YEAR_REQUIRED}. "
            "Month-level statements are acceptable."
        ),
    )


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
    """
    Evaluate an answer for the 2025 celebrity collaboration task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Three collaborations evaluated independently
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

    # Extract collaborations from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_collaborations(),
        template_class=CollaborationExtraction,
        extraction_name="collaborations_extraction"
    )

    # Build verification for first three collaborations (pad if fewer)
    items: List[CollaborationItem] = list(extracted.collaborations[:3])
    while len(items) < 3:
        items.append(CollaborationItem())

    # Create three collaboration subtrees (sequential per item as per rubric)
    for idx in range(3):
        await verify_single_collaboration(
            evaluator=evaluator,
            parent_node=root,
            item=items[idx],
            index=idx
        )

    # Return standardized summary
    return evaluator.get_summary()