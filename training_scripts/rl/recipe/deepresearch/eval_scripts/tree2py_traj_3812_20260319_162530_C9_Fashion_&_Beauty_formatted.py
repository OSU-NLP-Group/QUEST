import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fashion_beauty_brands_celebrity_2025_2026_pa_kop"
TASK_DESCRIPTION = """
In 2025-2026, several fashion and beauty brands announced new celebrity partnerships or ambassador roles. Identify four (4) distinct brands or retail stores that meet ALL of the following criteria:

1. The brand announced a celebrity ambassador partnership OR a celebrity founder/co-founder launched the brand, with the announcement OR brand launch occurring between January 1, 2025, and March 19, 2026.

2. The brand or its products must be available for purchase at physical retail locations in Pennsylvania (not online-only).

3. Each brand must fall into one of these categories: fashion accessories (eyewear, watches, jewelry), athletic/fitness apparel, skincare/beauty products, or sustainable fashion/apparel.

4. For brands in the fashion accessories or athletic apparel categories: The brand must have participated in OR been officially associated with a major fashion week event OR major athletic competition in 2025-2026.

5. For brands in the skincare/beauty or sustainable fashion categories: The brand must feature at least one of the following: (a) a proprietary or signature ingredient/technology with a specific trademarked or branded name, OR (b) a documented sustainability initiative (such as regenerative agriculture, recycled materials, or circular economy programs) that was active or announced in 2025-2026.

6. At least ONE of the four brands must have a retail presence specifically at King of Prussia Mall in Pennsylvania, and you must provide the store opening date OR the specific store address within the mall.

7. For each brand, provide: (a) the celebrity's name and their role (ambassador, founder, etc.), (b) the specific date or month when the partnership/launch was announced, (c) at least one specific Pennsylvania retail location or retailer that carries the brand, (d) the relevant category-specific requirement (fashion week participation, proprietary ingredient name, or sustainability initiative description), and (e) reference URLs supporting each claim.

Your answer must identify exactly four (4) brands that satisfy all applicable criteria, with complete verification information for each.
"""

EARLIEST_DATE_STR = "2025-01-01"
LATEST_DATE_STR = "2026-03-19"

ALLOWED_CATEGORIES_NOTE = (
    "Allowed categories: "
    "1) fashion accessories (eyewear, watches, jewelry), "
    "2) athletic/fitness apparel, "
    "3) skincare/beauty products, "
    "4) sustainable fashion/apparel."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BrandItem(BaseModel):
    # Basic brand identity
    brand_name: Optional[str] = None

    # Celebrity partnership/founder details
    celebrity_name: Optional[str] = None
    celebrity_role: Optional[str] = None  # e.g., ambassador, founder, co-founder, creative director
    announcement_date: Optional[str] = None  # e.g., "2025-05-12" or "May 2025"
    celebrity_urls: List[str] = Field(default_factory=list)       # URLs evidencing the celebrity + role
    announcement_urls: List[str] = Field(default_factory=list)    # URLs evidencing the date

    # Pennsylvania physical retail presence
    pa_retailer: Optional[str] = None          # store name or retailer chain carrying products
    pa_store_address: Optional[str] = None     # street address if given
    pa_store_city: Optional[str] = None        # e.g., "Philadelphia", "King of Prussia"
    pa_store_urls: List[str] = Field(default_factory=list)  # URLs proving physical store in PA

    # Category and category-specific requirement
    category: Optional[str] = None             # normalized to: accessories, athletic_apparel, beauty, sustainable_fashion
    category_urls: List[str] = Field(default_factory=list)        # URLs supporting the category classification
    category_requirement_desc: Optional[str] = None               # text of the requirement evidence (event, proprietary tech, initiative, etc.)
    category_requirement_date_or_year: Optional[str] = None       # ensure 2025 or 2026 for time-bound ones
    category_requirement_urls: List[str] = Field(default_factory=list)

    # King of Prussia requirement (optional; at least one brand should satisfy)
    kop_presence: Optional[bool] = None
    kop_detail: Optional[str] = None           # either opening date or in-mall specific address
    kop_urls: List[str] = Field(default_factory=list)


class BrandsExtraction(BaseModel):
    brands: List[BrandItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brands() -> str:
    return f"""
Extract up to four brands (if more are present in the answer, keep only the first four) and return them as a JSON list under "brands". For each brand, extract ONLY what is explicitly present in the answer text. Do not invent any URLs or facts.

For each brand, extract the following fields (use null for missing scalar fields, and [] for missing URL lists):
- brand_name
- celebrity_name
- celebrity_role
- announcement_date  (prefer ISO 'YYYY-MM-DD'; if not available, 'Month YYYY' is acceptable)
- celebrity_urls     (URLs proving the celebrity + role)
- announcement_urls  (URLs proving the announcement date)
- pa_retailer        (a Pennsylvania physical retailer or store name where the brand is available)
- pa_store_address   (street address if given)
- pa_store_city      (city if given; e.g., Philadelphia, Pittsburgh, King of Prussia)
- pa_store_urls      (URLs proving PA physical retail presence; store locator, mall/store page, etc.)
- category           (normalize to one of: accessories, athletic_apparel, beauty, sustainable_fashion)
- category_urls      (URLs supporting the category classification)
- category_requirement_desc (description text of the category-specific requirement)
- category_requirement_date_or_year (e.g., '2025-02-10' or '2025' if only year is given)
- category_requirement_urls (URLs proving the category-specific evidence)
- kop_presence       (true/false if the brand has a retail presence at King of Prussia Mall)
- kop_detail         (either the specific in-mall address or the opening date, if provided)
- kop_urls           (URLs proving the King of Prussia presence and the provided detail)

Important constraints to respect when extracting:
- Only include URLs that are explicitly present in the answer. If a claim lacks a URL in the answer, leave the corresponding URL list empty.
- For category normalization:
  • accessories → fashion accessories (eyewear, watches, jewelry)
  • athletic_apparel → athletic/fitness apparel
  • beauty → skincare/beauty products
  • sustainable_fashion → sustainable fashion/apparel
- If multiple retailers/URLs are listed, include at least one that clearly corresponds to a PHYSICAL location in Pennsylvania.
- If none of the brands explicitly mentions King of Prussia Mall, set kop_presence=false (or null) for all brands.

Return only the JSON object matching the schema. Do not include explanations.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists, keep order, drop falsy and duplicates."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                merged.append(url)
                seen.add(url)
    return merged


async def verify_with_sources_or_fail(
    evaluator: Evaluator,
    node_id: str,
    node_desc: str,
    parent,
    claim: str,
    sources: Optional[List[str]],
    critical: bool = True,
    add_ins: str = "None",
) -> None:
    """
    Create a leaf node and verify the claim using the provided sources if present.
    If no sources are provided, mark the node as failed (source-grounding required).
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=critical,
    )
    srcs = sources or []
    if len(srcs) == 0:
        # Enforce source-grounding: no sources → fail
        leaf.score = 0.0
        leaf.status = "failed"
        return
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction=add_ins,
    )


def normalize_category(cat: Optional[str]) -> str:
    if not cat:
        return ""
    c = cat.strip().lower()
    if c in {"accessory", "accessories", "fashion accessories", "eyewear", "watches", "jewelry", "jewellery"}:
        return "accessories"
    if c in {"athletic", "fitness", "athletic apparel", "fitness apparel", "athleisure", "sportswear", "athletic_apparel"}:
        return "athletic_apparel"
    if c in {"beauty", "skincare", "skin care", "cosmetics", "skincare/beauty", "skincare_beauty"}:
        return "beauty"
    if c in {"sustainable", "sustainable fashion", "sustainability", "eco", "eco fashion", "sustainable_fashion"}:
        return "sustainable_fashion"
    return c  # fallback


# --------------------------------------------------------------------------- #
# Brand verification sub-tree                                                 #
# --------------------------------------------------------------------------- #
async def verify_single_brand(
    evaluator: Evaluator,
    parent_node,
    brand: BrandItem,
    idx: int,
) -> None:
    """
    Build the verification sub-tree for a single brand (brand_{idx}).
    """
    brand_id = f"brand_{idx}"
    brand_desc = f"{['First','Second','Third','Fourth'][idx-1]} brand identification and verification"

    brand_node = evaluator.add_parallel(
        id=brand_id,
        desc=brand_desc,
        parent=parent_node,
        critical=False,  # allow partial credit per brand
    )

    # ---------- Celebrity partnership / founder role group ----------
    celeb_group = evaluator.add_parallel(
        id=f"{brand_id}_celebrity_partnership",
        desc="Celebrity partnership or founder role verification",
        parent=brand_node,
        critical=True,
    )

    brand_name = brand.brand_name or f"Brand #{idx}"
    celeb_name = brand.celebrity_name or ""
    celeb_role = brand.celebrity_role or ""
    ann_date = brand.announcement_date or ""

    # Leaf: celebrity name and role (with URL)
    celeb_claim = (
        f"The brand '{brand_name}' publicly identified {celeb_name} as {celeb_role} "
        f"(e.g., ambassador/founder/co-founder/creative director) during 2025–2026."
    )
    await verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{brand_id}_celebrity_name_and_role",
        node_desc="Celebrity name and specific role (ambassador, founder, co-founder, etc.) are correctly identified with supporting reference URL",
        parent=celeb_group,
        claim=celeb_claim,
        sources=merge_sources(brand.celebrity_urls, brand.announcement_urls),
        critical=True,
        add_ins="Verify both the celebrity's exact identity and the precise role title stated by the source. Minor wording variations are fine if meaning is the same.",
    )

    # Leaf: announcement date within window (with URL)
    date_claim = (
        f"The partnership/launch announcement for '{brand_name}' involving {celeb_name} "
        f"occurred on {ann_date}, and that date falls between {EARLIEST_DATE_STR} and {LATEST_DATE_STR}, inclusive."
    )
    await verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{brand_id}_announcement_date",
        node_desc=f"Partnership/launch announcement date falls between January 1, 2025, and March 19, 2026, with supporting reference URL",
        parent=celeb_group,
        claim=date_claim,
        sources=merge_sources(brand.announcement_urls, brand.celebrity_urls),
        critical=True,
        add_ins=(
            f"Strictly check that the page evidences the announcement date as {ann_date} "
            f"and that this date is within the inclusive window [{EARLIEST_DATE_STR}, {LATEST_DATE_STR}]. "
            "If the page only shows a month/year, judge whether that still lies within the window."
        ),
    )

    # ---------- Pennsylvania retail presence group ----------
    pa_group = evaluator.add_parallel(
        id=f"{brand_id}_pennsylvania_retail",
        desc="Pennsylvania retail presence verification",
        parent=brand_node,
        critical=True,
    )

    pa_retailer = brand.pa_retailer or ""
    pa_addr = brand.pa_store_address or ""
    pa_city = brand.pa_store_city or ""

    pa_claim_detail = f"{pa_retailer}".strip()
    if pa_addr:
        pa_claim_detail += f", address: {pa_addr}"
    if pa_city:
        pa_claim_detail += f", {pa_city}, PA"

    pa_claim = (
        f"Products from the brand '{brand_name}' are available for purchase at a PHYSICAL retail location in Pennsylvania: "
        f"{pa_claim_detail}. This is not an online-only listing."
    )
    await verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{brand_id}_physical_location",
        node_desc="At least one specific Pennsylvania retail location or retailer name is provided with supporting reference URL",
        parent=pa_group,
        claim=pa_claim,
        sources=brand.pa_store_urls,
        critical=True,
        add_ins="The source must clearly indicate an in-person store in Pennsylvania carrying the brand/products. Store locators, mall/store pages, or official retailer pages are acceptable.",
    )

    # ---------- Category + category-specific requirement group ----------
    cat_group = evaluator.add_parallel(
        id=f"{brand_id}_category_and_requirements",
        desc="Brand category and category-specific requirements verification",
        parent=brand_node,
        critical=True,
    )

    cat_norm = normalize_category(brand.category)
    # Leaf: category classification
    cat_claim = (
        f"The brand '{brand_name}' is correctly classified under '{cat_norm}', which must correspond to one of: "
        "fashion accessories (eyewear, watches, jewelry), athletic/fitness apparel, skincare/beauty products, or sustainable fashion/apparel."
    )
    cat_sources = merge_sources(brand.category_urls, brand.category_requirement_urls, brand.celebrity_urls, brand.announcement_urls)
    await verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{brand_id}_category_classification",
        node_desc="Brand correctly classified into one of four categories: fashion accessories (eyewear, watches, jewelry), athletic/fitness apparel, skincare/beauty products, or sustainable fashion/apparel",
        parent=cat_group,
        claim=cat_claim,
        sources=cat_sources,
        critical=True,
        add_ins="Accept reasonable synonyms. Focus on the brand's primary product category as evidenced by the source.",
    )

    # Leaf: category-specific requirement
    req_desc = brand.category_requirement_desc or ""
    req_when = brand.category_requirement_date_or_year or ""
    req_sources = merge_sources(brand.category_requirement_urls)

    if cat_norm == "accessories":
        req_claim = (
            f"In {req_when or '2025–2026'}, the accessories brand '{brand_name}' participated in or was officially "
            f"associated with a major fashion week event (e.g., NYFW/Paris/Milan/London). Evidence: {req_desc}"
        )
        add_ins = "Ensure the event is a recognized fashion week and falls in 2025 or 2026; verify the brand's official involvement/association."
    elif cat_norm == "athletic_apparel":
        req_claim = (
            f"In {req_when or '2025–2026'}, the athletic/fitness apparel brand '{brand_name}' participated in or was "
            f"officially associated with a major athletic competition or event. Evidence: {req_desc}"
        )
        add_ins = "Confirm an official association (e.g., team/athlete sponsorship, event partnership) during 2025–2026."
    elif cat_norm == "beauty":
        req_claim = (
            f"The skincare/beauty brand '{brand_name}' features a proprietary or signature ingredient/technology with a "
            f"specific trademarked/branded name: {req_desc}."
        )
        add_ins = "Verify the presence of a distinct proprietary/branded name (e.g., with ™/® or a unique trademark-style name)."
    elif cat_norm == "sustainable_fashion":
        req_claim = (
            f"In {req_when or '2025–2026'}, the sustainable fashion/apparel brand '{brand_name}' had a documented "
            f"sustainability initiative active or announced, described as: {req_desc}."
        )
        add_ins = "Confirm the initiative is documented and was active or announced in 2025–2026 (e.g., recycled materials, regenerative agriculture, circular programs)."
    else:
        # Unknown or missing category → create a claim that will almost certainly fail without proper sources
        req_claim = (
            f"The brand '{brand_name}' satisfies the category-specific requirement for category '{cat_norm}': {req_desc} (timing {req_when})."
        )
        add_ins = (
            "Category is unclear; verify only if the page clearly supports the stated requirement for a valid category."
        )

    await verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id=f"{brand_id}_category_specific_requirement",
        node_desc="Category-specific requirement is satisfied with supporting reference URL: fashion week participation for accessories, athletic competition association for fitness apparel, proprietary ingredient/technology name for skincare/beauty, or sustainability initiative for sustainable fashion",
        parent=cat_group,
        claim=req_claim,
        sources=req_sources,
        critical=True,
        add_ins=add_ins,
    )


# --------------------------------------------------------------------------- #
# King of Prussia requirement sub-tree                                        #
# --------------------------------------------------------------------------- #
async def verify_kop_requirement(
    evaluator: Evaluator,
    parent_node,
    brands: List[BrandItem],
) -> None:
    """
    Build and verify the King of Prussia (KOP) requirement:
    - One of the four brands has a store at King of Prussia Mall (with URLs)
    - Provide either the store opening date or the specific store address within the mall (with URLs)
    """
    kop_node = evaluator.add_parallel(
        id="king_of_prussia_requirement",
        desc="At least one brand has specific retail presence at King of Prussia Mall with required details",
        parent=parent_node,
        critical=True,
    )

    # Pick the first brand with kop_presence == True and has sources
    chosen_idx = None
    for i, b in enumerate(brands):
        if b.kop_presence and len(b.kop_urls) > 0:
            chosen_idx = i
            break

    if chosen_idx is None:
        # Create leaves that will fail due to missing sources
        node1 = evaluator.add_leaf(
            id="kop_brand_identified",
            desc="One of the four brands is confirmed to have a store at King of Prussia Mall, Pennsylvania, with supporting reference URL",
            parent=kop_node,
            critical=True,
        )
        node1.score = 0.0
        node1.status = "failed"

        node2 = evaluator.add_leaf(
            id="kop_specific_detail",
            desc="Either the store opening date OR the specific store address within King of Prussia Mall is provided",
            parent=kop_node,
            critical=True,
        )
        node2.score = 0.0
        node2.status = "failed"
        return

    b = brands[chosen_idx]
    bname = b.brand_name or f"Brand #{chosen_idx+1}"

    # Leaf: Confirm KOP presence
    kop_presence_claim = f"The brand '{bname}' has a physical retail presence at King of Prussia Mall in Pennsylvania."
    await verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="kop_brand_identified",
        node_desc="One of the four brands is confirmed to have a store at King of Prussia Mall, Pennsylvania, with supporting reference URL",
        parent=kop_node,
        claim=kop_presence_claim,
        sources=b.kop_urls,
        critical=True,
        add_ins="The source should clearly indicate a store at King of Prussia Mall (KOP), not just a generic presence in King of Prussia city.",
    )

    # Leaf: Opening date OR in-mall address
    kop_detail_txt = b.kop_detail or ""
    kop_detail_claim = (
        f"For the brand '{bname}' at King of Prussia Mall, the following specific detail is provided: {kop_detail_txt}. "
        "This detail is either the store opening date or the specific in-mall address/location."
    )
    await verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="kop_specific_detail",
        node_desc="Either the store opening date OR the specific store address within King of Prussia Mall is provided",
        parent=kop_node,
        claim=kop_detail_claim,
        sources=b.kop_urls,
        critical=True,
        add_ins="Accept common in-mall address formats (e.g., suite/unit numbers, 'Plaza Level', 'The Court'). Alternatively, accept a clearly stated opening date.",
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
    Evaluate the answer for the fashion/beauty brands task using obj_task_eval.
    """
    # Initialize evaluator (root as non-critical to allow partial credit aggregation)
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

    # Record task constraints for transparency
    evaluator.add_custom_info(
        info={
            "date_window_inclusive": [EARLIEST_DATE_STR, LATEST_DATE_STR],
            "allowed_categories_note": ALLOWED_CATEGORIES_NOTE,
        },
        info_type="constraints",
        info_name="task_constraints",
    )

    # Extract structured brand info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_brands(),
        template_class=BrandsExtraction,
        extraction_name="brands_extraction",
    )

    # Keep exactly four entries (pad with empties if fewer)
    brands: List[BrandItem] = list(extracted.brands[:4])
    while len(brands) < 4:
        brands.append(BrandItem())

    evaluator.add_custom_info(
        info={
            "extracted_count": len(extracted.brands) if extracted and extracted.brands is not None else 0,
            "used_for_verification": 4,
        },
        info_type="extraction_meta",
        info_name="extraction_summary",
    )

    # Build brand sub-trees
    for i in range(4):
        await verify_single_brand(evaluator, root, brands[i], i + 1)

    # King of Prussia requirement (global)
    await verify_kop_requirement(evaluator, root, brands)

    # Return final structured summary
    return evaluator.get_summary()