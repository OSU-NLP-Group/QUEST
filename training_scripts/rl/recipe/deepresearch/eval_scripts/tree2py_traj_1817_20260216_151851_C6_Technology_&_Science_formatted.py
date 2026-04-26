import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ces2026_products"
TASK_DESCRIPTION = (
    "At CES 2026, which took place in Las Vegas in January 2026, numerous innovative consumer "
    "technology products were announced and showcased. Identify four distinct innovative consumer "
    "technology products that were announced or demonstrated at CES 2026, ensuring that each product "
    "comes from a different product category (such as robotics, smart home, wearables, displays, "
    "gaming, mobility, etc.). For each of the four products, provide: (1) The specific product name "
    "and manufacturer/developer; (2) The product category; (3) Evidence that the product was announced "
    "or showcased at CES 2026; (4) At least three major technical specifications or key features; "
    "(5) Evidence of notable innovation (such as awards, press recognition, or industry-first "
    "achievements) OR demonstration of AI integration as a core functionality; (6) Release timeline "
    "or availability information (if available); (7) Reference URLs that verify the product's CES 2026 "
    "presence, technical specifications, and innovation claims. At least one of the four products must "
    "be from the robotics or 'physical AI' category, which was a major theme at CES 2026. Your answer "
    "must include proper source attribution with URLs for all factual claims about each product."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProductSources(BaseModel):
    ces_presence_urls: List[str] = Field(default_factory=list)
    specs_urls: List[str] = Field(default_factory=list)
    innovation_urls: List[str] = Field(default_factory=list)
    release_urls: List[str] = Field(default_factory=list)
    ai_urls: List[str] = Field(default_factory=list)
    stair_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class ProductExtract(BaseModel):
    name: Optional[str] = None
    manufacturer: Optional[str] = None
    category: Optional[str] = None

    # Specs & features (free-form text list)
    specs_or_features: List[str] = Field(default_factory=list)

    # Innovation evidence text (awards/press/official recognition/industry-first)
    innovation_description: Optional[str] = None

    # Release / availability text (may be approximate or “in development”)
    release_timeline: Optional[str] = None

    # AI core (string to allow flexible answers: "yes"/"no"/"core AI"/"integrated AI", etc.)
    ai_core: Optional[str] = None
    ai_core_description: Optional[str] = None

    # Robotics special claim: stair climbing
    claims_stair_climbing: Optional[str] = None
    stair_mechanism_description: Optional[str] = None

    # Grouped sources
    sources: ProductSources = Field(default_factory=ProductSources)


class ProductsExtraction(BaseModel):
    products: List[ProductExtract] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_products() -> str:
    return """
Extract up to the first six distinct consumer technology products described in the answer that are said to be showcased or announced at CES 2026. For each product, extract the following fields exactly as stated in the answer text (do not invent information; if missing, use null or empty arrays as specified):

For each product:
- name: The specific product name (string; null if missing)
- manufacturer: The manufacturer or developer (string; null if missing)
- category: The product category (e.g., robotics, smart home, wearables, displays, gaming, mobility, etc.)
- specs_or_features: An array of at least three key technical specifications or notable features mentioned for this product (empty array if not provided)
- innovation_description: A concise text describing innovation evidence (awards, press recognition, official CES recognition, industry-first, or clearly stated core AI as a key differentiator), if present; otherwise null
- release_timeline: The release or availability information if stated (e.g., "shipping Q3 2026", "preorders open January 2026", "prototype with no confirmed date"); null if not provided
- ai_core: Whether the answer explicitly claims AI is a core functionality (string like "yes", "no", "core AI", etc.; null if not mentioned)
- ai_core_description: A brief phrase from the answer describing how AI is central (e.g., "on-device LLM for autonomy", "AI-powered perception"), if present; otherwise null
- claims_stair_climbing: If the product (especially robotics) claims stair-climbing capability, set to "yes"; if not claimed or not applicable, set "no" or null
- stair_mechanism_description: If stair-climbing is claimed, a brief textual description of the mechanism (e.g., "wheel-leg hybrid", "two-legged design"); null otherwise

- sources: Group the URLs the answer cites for this product into:
  - ces_presence_urls: URLs that explicitly support CES 2026 presence/announcement
  - specs_urls: URLs (preferably official manufacturer/developer pages or press releases) that support the specs/features
  - innovation_urls: URLs that support innovation evidence (awards, press, official CES recognition, industry-first, or AI-core)
  - release_urls: URLs that support release/availability info if stated
  - ai_urls: URLs specifically supporting AI as a core functionality, if claimed
  - stair_urls: URLs that support the stair-climbing mechanism, if claimed
  - other_urls: Any other cited URLs for this product
All URLs must be explicitly present in the answer text; if none, return empty arrays.

Important:
- Only extract what appears in the answer; do not infer or create new data.
- Preserve product names and manufacturer exactly as written.
- If the answer lists more than four products, we will evaluate the first four only. Still extract all mentioned (up to six) so we can choose the first four later.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _truthy_str(s: Optional[str]) -> bool:
    """Interpret a loosely labeled yes/true string."""
    if not s:
        return False
    v = s.strip().lower()
    return v in {"yes", "true", "y", "core", "core ai", "ai core", "ai", "enabled", "integrated ai", "built-in ai"}


def _is_robotics_category(cat: Optional[str]) -> bool:
    c = (cat or "").strip().lower()
    return ("robot" in c) or ("physical ai" in c) or ("humanoid" in c) or ("home robot" in c)


def _normalize_category(cat: Optional[str]) -> str:
    c = (cat or "").strip().lower()
    # simple normalization for pairwise distinctness
    synonyms = {
        "smart home": "smart home",
        "home": "smart home",
        "home automation": "smart home",
        "wearable": "wearables",
        "wearables": "wearables",
        "robotics": "robotics",
        "humanoid robot": "robotics",
        "physical ai": "robotics",  # treat 'physical ai' theme as robotics family for distinctness
        "tv": "displays",
        "display": "displays",
        "displays": "displays",
        "mobility": "mobility",
        "gaming": "gaming",
        "ar": "xr",
        "vr": "xr",
        "xr": "xr",
    }
    # try exact synonyms
    if c in synonyms:
        return synonyms[c]
    # broader containment rules
    if "robot" in c:
        return "robotics"
    if "smart" in c and "home" in c:
        return "smart home"
    if "display" in c or "tv" in c or "screen" in c or "monitor" in c:
        return "displays"
    if "wear" in c:
        return "wearables"
    if "game" in c:
        return "gaming"
    if "mobility" in c or "vehicle" in c or "ev" in c or "scooter" in c or "bike" in c:
        return "mobility"
    if "ar" in c or "vr" in c or "mixed reality" in c or "xr" in c:
        return "xr"
    if "audio" in c or "earbud" in c or "headphone" in c or "speaker" in c:
        return "audio"
    if "camera" in c or "imaging" in c:
        return "imaging"
    if "pc" in c or "laptop" in c or "computer" in c:
        return "computing"
    return c


def _has_text(x: Optional[str]) -> bool:
    return bool((x or "").strip())


def _merge_urls(*lists: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for lst in lists:
        for u in lst:
            u2 = u.strip()
            if not u2:
                continue
            if u2 not in seen:
                out.append(u2)
                seen.add(u2)
    return out


def _product_all_urls(p: ProductExtract) -> List[str]:
    s = p.sources
    return _merge_urls(
        s.ces_presence_urls,
        s.specs_urls,
        s.innovation_urls,
        s.release_urls,
        s.ai_urls,
        s.stair_urls,
        s.other_urls,
    )


# --------------------------------------------------------------------------- #
# Verification for a single product                                           #
# --------------------------------------------------------------------------- #
async def verify_product(evaluator: Evaluator, parent_node, product: ProductExtract, idx: int) -> None:
    """
    Build the verification subtree for a single product.
    """
    pnode = evaluator.add_parallel(
        id=f"product_{idx + 1}",
        desc=f"Product {idx + 1} (one of four CES 2026 products)",
        parent=parent_node,
        critical=False,  # per rubric: product blocks are non-critical under root
    )

    name = _normalize_str(product.name)
    manufacturer = _normalize_str(product.manufacturer)
    category = _normalize_str(product.category)
    specs = product.specs_or_features or []
    innovation_desc = _normalize_str(product.innovation_description)
    release_text = _normalize_str(product.release_timeline)
    ai_core = _truthy_str(product.ai_core)
    ai_desc = _normalize_str(product.ai_core_description)
    claims_stair = _truthy_str(product.claims_stair_climbing)
    stair_desc = _normalize_str(product.stair_mechanism_description)

    urls = product.sources
    all_urls = _product_all_urls(product)

    # Identity (parallel, critical)
    identity = evaluator.add_parallel(
        id=f"p{idx + 1}_identity",
        desc="Provides product name and manufacturer/developer",
        parent=pnode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(name),
        id=f"p{idx + 1}_name_present",
        desc="Specific product name is provided",
        parent=identity,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(manufacturer),
        id=f"p{idx + 1}_manufacturer_present",
        desc="Manufacturer/developer is provided",
        parent=identity,
        critical=True
    )

    # Category present (critical)
    evaluator.add_custom_node(
        result=bool(category),
        id=f"p{idx + 1}_category_present",
        desc="Product category is specified",
        parent=pnode,
        critical=True
    )

    # Consumer market check (critical, URL-supported)
    consumer_leaf = evaluator.add_leaf(
        id=f"p{idx + 1}_consumer_market",
        desc="Product is intended for consumer markets (home/personal), not purely industrial/enterprise",
        parent=pnode,
        critical=True
    )
    consumer_claim = (
        f"The product '{name}' by '{manufacturer}' is intended for consumer/home/personal markets "
        f"(not purely industrial/enterprise)."
    )
    await evaluator.verify(
        claim=consumer_claim,
        node=consumer_leaf,
        sources=all_urls,
        additional_instruction=(
            "Look for signals like 'consumer', 'home', 'personal', 'household', 'for families', "
            "or similar positioning. If the page clearly targets enterprise/industrial only, it should not pass. "
            "If mixed, accept if consumer/home is clearly included."
        )
    )

    # CES 2026 presence (critical, URL-supported)
    ces_leaf = evaluator.add_leaf(
        id=f"p{idx + 1}_ces_2026_presence",
        desc="Provides evidence the product was announced or showcased at CES 2026 (Las Vegas, Jan 2026)",
        parent=pnode,
        critical=True
    )
    ces_claim = (
        f"The product '{name}' by '{manufacturer}' was announced or showcased at CES 2026 in Las Vegas, January 2026."
    )
    await evaluator.verify(
        claim=ces_claim,
        node=ces_leaf,
        sources=urls.ces_presence_urls,
        additional_instruction=(
            "The page must explicitly indicate CES 2026 presence or announcement. Accept official exhibitor pages, "
            "press releases, or credible media coverage naming 'CES 2026'."
        )
    )

    # Specs count (critical; make sure at least 3 listed)
    evaluator.add_custom_node(
        result=(len([s for s in specs if _normalize_str(s)]) >= 3),
        id=f"p{idx + 1}_specs_count",
        desc="At least three major technical specifications or key features are provided",
        parent=pnode,
        critical=True
    )

    # Innovation evidence required (critical: ensure some innovation text present)
    evaluator.add_custom_node(
        result=bool(innovation_desc),
        id=f"p{idx + 1}_innovation_evidence_required",
        desc=(
            "Provides evidence of notable innovation for the product (e.g., awards, press recognition, "
            "official CES recognition, industry-first achievement)"
        ),
        parent=pnode,
        critical=True
    )

    # Release timeline present (critical; allow approximate or 'in development')
    evaluator.add_custom_node(
        result=bool(release_text),
        id=f"p{idx + 1}_release_timeline_present",
        desc=(
            "Release timeline/availability is provided (can be approximate; 'in development' or "
            "'unconfirmed launch date' acceptable if announced prototype)"
        ),
        parent=pnode,
        critical=True
    )

    # Stair-climbing mechanism if claimed (conditional critical)
    if _is_robotics_category(category) and claims_stair:
        stair_leaf = evaluator.add_leaf(
            id=f"p{idx + 1}_stair_climbing_mechanism_if_claimed",
            desc=(
                "If the product is a robotics product claiming stair-climbing capability, evidence shows a "
                "mechanism specifically designed to navigate stairs (e.g., wheel-leg, two-legged design)"
            ),
            parent=pnode,
            critical=True
        )
        stair_claim = (
            f"The product '{name}' by '{manufacturer}' claims stair-climbing; the cited page describes a "
            f"specific mechanism to navigate stairs (e.g., wheel-leg hybrid, articulated legs, tracked system)."
        )
        await evaluator.verify(
            claim=stair_claim,
            node=stair_leaf,
            sources=_merge_urls(urls.stair_urls, urls.specs_urls, urls.ces_presence_urls, urls.other_urls),
            additional_instruction=(
                "Look for explicit mention of stair navigation mechanism—e.g., legged locomotion, wheel-leg hybrids, "
                "tracked climbers, or similar. Generic 'can climb stairs' without mechanism detail should not pass."
            )
        )
    else:
        # Not applicable -> pass
        evaluator.add_custom_node(
            result=True,
            id=f"p{idx + 1}_stair_climbing_mechanism_if_claimed",
            desc=(
                "If robotics product claims stair-climbing, evidence of mechanism is required; not applicable here."
            ),
            parent=pnode,
            critical=True
        )

    # Sources group (critical, parallel)
    sources_group = evaluator.add_parallel(
        id=f"p{idx + 1}_sources",
        desc=(
            "Reference URLs are provided to support required aspects (CES presence, specs/features, innovation, "
            "and release/availability where available) and claims are attributed"
        ),
        parent=pnode,
        critical=True
    )

    # CES presence supported by URL (critical)
    ces_support_leaf = evaluator.add_leaf(
        id=f"p{idx + 1}_url_supports_ces_presence",
        desc="At least one reference URL supports the claim that the product was announced/showcased at CES 2026",
        parent=sources_group,
        critical=True
    )
    await evaluator.verify(
        claim=ces_claim,
        node=ces_support_leaf,
        sources=urls.ces_presence_urls,
        additional_instruction=(
            "Confirm the page explicitly mentions CES 2026 and links it to this product."
        )
    )

    # Specs supported by allowed source types (critical)
    specs_support_leaf = evaluator.add_leaf(
        id=f"p{idx + 1}_url_supports_specs_features_from_allowed_source_types",
        desc=(
            "At least one reference URL from an official announcement, press release, or manufacturer/developer "
            "source supports the listed technical specifications/key features"
        ),
        parent=sources_group,
        critical=True
    )
    listed_specs_preview = "; ".join([_normalize_str(s) for s in specs[:6]])
    specs_claim = (
        f"This page is an official manufacturer/developer page or press release for '{name}' by '{manufacturer}', "
        f"and it explicitly lists at least two of these specs/features: {listed_specs_preview}."
    )
    await evaluator.verify(
        claim=specs_claim,
        node=specs_support_leaf,
        sources=urls.specs_urls,
        additional_instruction=(
            "Check if the domain is the manufacturer/developer or an official press release/newsroom. "
            "Verify the page lists at least two of the provided specs/features."
        )
    )

    # Innovation evidence supported by URL (critical)
    innovation_support_leaf = evaluator.add_leaf(
        id=f"p{idx + 1}_url_supports_innovation_evidence",
        desc=(
            "At least one reference URL supports the innovation evidence "
            "(awards/press/official CES recognition/industry-first achievement)"
        ),
        parent=sources_group,
        critical=True
    )
    innovation_claim = (
        f"The page provides evidence of notable innovation for '{name}' by '{manufacturer}'—such as awards, "
        f"press recognition, official CES recognition, an industry-first, or explicitly positioning AI as a core functionality."
    )
    await evaluator.verify(
        claim=innovation_claim,
        node=innovation_support_leaf,
        sources=_merge_urls(urls.innovation_urls, urls.ai_urls),
        additional_instruction=(
            "Look for phrases like 'CES Innovation Award', 'Best of CES', 'industry-first', or explicit statements "
            "that AI is a key/core functionality backed by credible sources."
        )
    )

    # Release timeline supported by URL if stated (critical)
    if release_text:
        release_support_leaf = evaluator.add_leaf(
            id=f"p{idx + 1}_url_supports_release_timeline_if_stated",
            desc=(
                "If release/availability info is stated, at least one reference URL supports it "
                "(or the answer indicates it is unconfirmed/in development for a prototype)"
            ),
            parent=sources_group,
            critical=True
        )
        release_claim = (
            f"The page states the release/availability information for '{name}' by '{manufacturer}' as: {release_text}"
        )
        await evaluator.verify(
            claim=release_claim,
            node=release_support_leaf,
            sources=urls.release_urls,
            additional_instruction=(
                "Match the stated release timing/window (e.g., a month/quarter/year or 'in development') on the page."
            )
        )
    else:
        # Not stated, treat as pass (requirement applies only if stated)
        evaluator.add_custom_node(
            result=True,
            id=f"p{idx + 1}_url_supports_release_timeline_if_stated",
            desc="Release timeline not stated in answer; URL support not applicable",
            parent=sources_group,
            critical=True
        )

    # Proper source attribution for factual claims (critical)
    has_presence = len(urls.ces_presence_urls) > 0
    has_specs = len(urls.specs_urls) > 0
    has_innovation = len(urls.innovation_urls) > 0 or len(urls.ai_urls) > 0
    release_requires_url = bool(release_text)
    has_release = (len(urls.release_urls) > 0) if release_requires_url else True

    evaluator.add_custom_node(
        result=(has_presence and has_specs and has_innovation and has_release),
        id=f"p{idx + 1}_source_attribution_for_factual_claims",
        desc=(
            "Provides proper source attribution with URLs for factual claims about this product "
            "(CES presence, specs/features, innovation; and release if provided)"
        ),
        parent=sources_group,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Set-level requirements verification                                         #
# --------------------------------------------------------------------------- #
async def verify_set_level_requirements(evaluator: Evaluator, root, extracted: ProductsExtraction) -> None:
    """
    Build the cross-product (set-level) verification subtree.
    """
    set_node = evaluator.add_parallel(
        id="set_level_requirements",
        desc="Cross-product (set-level) requirements",
        parent=root,
        critical=True  # As per rubric, these are mandatory gating checks
    )

    # For robust evaluation, we will consider the first four products as the evaluation set.
    first_four = extracted.products[:4]
    names = [(_normalize_str(p.name), _normalize_str(p.manufacturer)) for p in first_four]
    categories = [_normalize_category(p.category) for p in first_four]

    # Exactly four distinct products are provided (len >= 4 and the first four are distinct by name+manufacturer)
    # Note: Following evaluation policy, we allow answers that list >= 4 products and evaluate the first four.
    distinct_pairs = set([f"{n}|{m}" for n, m in names if n or m])
    result_four_distinct = len(first_four) >= 4 and len(distinct_pairs) >= 4 \
        and all(n or m for n, m in names)  # ensure non-empty identities

    evaluator.add_custom_node(
        result=result_four_distinct,
        id="exactly_four_products_provided",
        desc="Exactly four distinct products are provided",
        parent=set_node,
        critical=True
    )

    # Categories pairwise distinct among first four
    norm_cats_nonempty = [c for c in categories if c]
    result_distinct_cats = (len(norm_cats_nonempty) == 4) and (len(set(norm_cats_nonempty)) == 4)
    evaluator.add_custom_node(
        result=result_distinct_cats,
        id="all_four_categories_pairwise_distinct",
        desc="Each of the four products is from a different product category (pairwise distinct categories)",
        parent=set_node,
        critical=True
    )

    # At least one robotics or 'physical AI'
    result_has_robotics = any(_is_robotics_category(p.category) for p in first_four)
    evaluator.add_custom_node(
        result=result_has_robotics,
        id="at_least_one_robotics_or_physical_ai",
        desc="At least one of the four products is in the robotics or 'physical AI' category",
        parent=set_node,
        critical=True
    )

    # At least one product has AI as core with URL evidence
    ai_urls_all: List[str] = []
    product_names_for_ai: List[str] = []
    for p in first_four:
        if _truthy_str(p.ai_core) or _has_text(p.ai_core_description):
            ai_urls_all.extend(p.sources.ai_urls or [])
            if p.name and p.manufacturer:
                product_names_for_ai.append(f"{p.name} by {p.manufacturer}")
            elif p.name:
                product_names_for_ai.append(p.name)

    ai_leaf = evaluator.add_leaf(
        id="at_least_one_product_has_core_ai_with_evidence",
        desc="At least one product explicitly incorporates AI as a core functional component AND the answer provides evidence/URL support for that AI-as-core claim",
        parent=set_node,
        critical=True
    )
    ai_claim = (
        "At least one of these products is described on this page as using artificial intelligence as a core functionality: "
        + "; ".join(product_names_for_ai) if product_names_for_ai else
        "At least one of the four products is described on this page as using AI as a core functionality."
    )
    await evaluator.verify(
        claim=ai_claim,
        node=ai_leaf,
        sources=ai_urls_all,
        additional_instruction=(
            "You only need to find evidence for at least one of the four products. "
            "Look for statements indicating AI is a core or central capability (e.g., AI-powered autonomy, on-device LLM, "
            "AI-driven perception/planning/personalization)."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the CES 2026 products task using the Mind2Web2 evaluation framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root-level: evaluate set-level and products independently
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

    # Extract structured product information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_products(),
        template_class=ProductsExtraction,
        extraction_name="products_extraction"
    )

    # Build set-level requirement nodes
    await verify_set_level_requirements(evaluator, root, extracted)

    # Verify the first four products only (pad with empties if fewer)
    products_to_check: List[ProductExtract] = list(extracted.products[:4])
    while len(products_to_check) < 4:
        products_to_check.append(ProductExtract())

    # Build each product subtree
    for i, prod in enumerate(products_to_check):
        await verify_product(evaluator, root, prod, i)

    # Return full evaluation summary
    return evaluator.get_summary()