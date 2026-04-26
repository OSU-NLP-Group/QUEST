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
TASK_ID = "beauty_brand_partnerships_2024"
TASK_DESCRIPTION = (
    "Identify 4 beauty brand partnerships that were officially announced in 2024, where each partnership involves a "
    "celebrity partner from the entertainment or sports industry serving as a brand ambassador, global ambassador, or "
    "similar official role. For each partnership, provide: (1) The celebrity's full name and their primary professional "
    "category (actress, singer, athlete, or multi-hyphenate), (2) The beauty brand's name and its specific industry "
    "sector (cosmetics, makeup, haircare, skincare, or fragrance), (3) The exact announcement date in 2024 (including "
    "month and day, or at minimum the month and year), (4) The specific partnership role title as stated in official "
    "announcements (e.g., 'Global Brand Ambassador,' 'Brand Ambassador,' 'International Ambassador'). Each of the 4 "
    "partnerships must involve a different celebrity and a different brand. The partnerships must be documented by "
    "verifiable industry sources such as fashion/beauty publications, official press releases, or brand announcements."
)

ALLOWED_CELEB_CATEGORIES_HINT = (
    "entertainment or sports categories such as actress, actor, singer, musician, rapper, performer, athlete, or multi-hyphenate"
)
ALLOWED_BRAND_SECTORS_HINT = "cosmetics, makeup, haircare, skincare, or fragrance"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PartnershipItem(BaseModel):
    celebrity_name: Optional[str] = None
    celebrity_category: Optional[str] = None
    brand_name: Optional[str] = None
    brand_sector: Optional[str] = None
    announcement_date: Optional[str] = None  # Free-form date text as stated in answer (e.g., "Jan 15, 2024" or "March 2024")
    role_title: Optional[str] = None
    parent_company: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)  # Sources documenting the partnership
    parent_company_sources: List[str] = Field(default_factory=list)  # Optional explicit sources for ownership info


class PartnershipsExtraction(BaseModel):
    partnerships: List[PartnershipItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_partnerships() -> str:
    return """
    Extract up to all partnership entries that the answer claims were officially announced in 2024 and that involve a celebrity (from entertainment or sports) serving as an official ambassador-type role for a beauty brand.
    
    For each partnership mentioned in the answer, return an object with:
    - celebrity_name: the celebrity's full name as written
    - celebrity_category: the celebrity's primary professional category as provided (e.g., actress, singer, musician, rapper, athlete, or multi-hyphenate)
    - brand_name: the beauty brand's name as written
    - brand_sector: the brand's industry sector as written (ideally one of: cosmetics, makeup, haircare, skincare, fragrance)
    - announcement_date: the announcement date text exactly as stated in the answer (month+day+year or at least month+year)
    - role_title: the specific partnership role title as stated (e.g., Global Brand Ambassador, Brand Ambassador, International Ambassador, Creative Director)
    - parent_company: the brand parent company or ownership as stated in the answer (if present), otherwise null
    - source_urls: an array of all URLs cited in the answer that document the partnership (industry publication article, official press release, or brand announcement site)
    - parent_company_sources: an array of URLs cited (if any) that support the parent_company ownership claim; if none provided, return an empty array

    Return JSON with a top-level field:
    {
      "partnerships": [ ... ]
    }

    Rules:
    - Extract exactly what appears in the answer text; do not invent new info or URLs.
    - If a field is missing for a partnership, set it to null (for strings) or [] for url lists.
    - Include all partnership entries mentioned in the answer (we will select the first 4 for evaluation if there are more).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _nonempty(s: Optional[str]) -> bool:
    return bool((s or "").strip())


def _first_k(items: List[Any], k: int) -> List[Any]:
    return items[:k] if len(items) >= k else items + [PartnershipItem() for _ in range(k - len(items))]


# --------------------------------------------------------------------------- #
# Verification for one partnership                                            #
# --------------------------------------------------------------------------- #
async def verify_partnership(
    evaluator: Evaluator,
    parent_node,
    item: PartnershipItem,
    idx: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single partnership item.
    """
    partnership_node = evaluator.add_parallel(
        id=f"partnership_{idx+1}",
        desc=f"Partnership #{idx+1} (one of the four required 2024 announcements)",
        parent=parent_node,
        critical=False  # Each partnership allows partial credit
    )

    # Sources presence (Critical prerequisite for all URL-based checks)
    all_sources = item.source_urls or []
    sources_present_node = evaluator.add_custom_node(
        result=len(all_sources) > 0,
        id=f"partnership_{idx+1}_sources_provided",
        desc=f"At least one source URL is provided for partnership #{idx+1}",
        parent=partnership_node,
        critical=True
    )

    # ---------------- Celebrity info ----------------
    celeb_info_node = evaluator.add_parallel(
        id=f"partnership_{idx+1}_celebrity_info",
        desc="Celebrity full name and primary category provided and fit allowed types",
        parent=partnership_node,
        critical=True
    )
    celeb_name_exists = evaluator.add_custom_node(
        result=_nonempty(item.celebrity_name),
        id=f"partnership_{idx+1}_celebrity_name_exists",
        desc="Celebrity name is provided",
        parent=celeb_info_node,
        critical=True
    )
    celeb_category_allowed = evaluator.add_leaf(
        id=f"partnership_{idx+1}_celebrity_category_allowed",
        desc="Celebrity primary category is an entertainment/sports category (e.g., actress/singer/musician/athlete/multi-hyphenate)",
        parent=celeb_info_node,
        critical=True
    )
    celeb_cat = item.celebrity_category or ""
    celeb_cat_claim = (
        f"The primary professional category '{celeb_cat}' belongs to entertainment or sports "
        f"(e.g., {ALLOWED_CELEB_CATEGORIES_HINT})."
    )
    await evaluator.verify(
        claim=celeb_cat_claim,
        node=celeb_category_allowed,
        additional_instruction="Judge this as a simple logical/membership check. Allow reasonable synonyms like actor/actress, singer/musician/rapper, performer, athlete. "
                               "Accept 'multi-hyphenate' if they are known across multiple entertainment/sports roles."
    )

    # ---------------- Brand & sector ----------------
    brand_sector_node = evaluator.add_parallel(
        id=f"partnership_{idx+1}_brand_and_sector",
        desc="Beauty brand name provided and sector is allowed (cosmetics/makeup/haircare/skincare/fragrance)",
        parent=partnership_node,
        critical=True
    )
    brand_name_exists = evaluator.add_custom_node(
        result=_nonempty(item.brand_name),
        id=f"partnership_{idx+1}_brand_name_exists",
        desc="Brand name is provided",
        parent=brand_sector_node,
        critical=True
    )
    sector_allowed_leaf = evaluator.add_leaf(
        id=f"partnership_{idx+1}_brand_sector_allowed",
        desc="Brand sector is one of cosmetics, makeup, haircare, skincare, or fragrance",
        parent=brand_sector_node,
        critical=True
    )
    sector_txt = item.brand_sector or ""
    sector_claim = f"The brand sector '{sector_txt}' is one of: {ALLOWED_BRAND_SECTORS_HINT} (allow close synonyms like 'skin care' for skincare, 'make-up' for makeup, 'perfume' for fragrance)."
    await evaluator.verify(
        claim=sector_claim,
        node=sector_allowed_leaf,
        additional_instruction="This is a logical category membership check; allow minor synonyms and casing variants."
    )

    # ---------------- Announcement date in 2024 ----------------
    date_node = evaluator.add_parallel(
        id=f"partnership_{idx+1}_announcement_date_2024",
        desc="Announcement date is provided and is in 2024 (month+day, or at least month+year)",
        parent=partnership_node,
        critical=True
    )
    date_provided = evaluator.add_custom_node(
        result=_nonempty(item.announcement_date),
        id=f"partnership_{idx+1}_announcement_date_provided",
        desc="Announcement date text is provided",
        parent=date_node,
        critical=True
    )
    date_supported_leaf = evaluator.add_leaf(
        id=f"partnership_{idx+1}_announcement_date_supported",
        desc="Provided source(s) support that the announcement occurred in 2024 on the stated date (or at least month+year)",
        parent=date_node,
        critical=True
    )
    date_claim = (
        f"The partnership between {item.celebrity_name or '[celebrity]'} and {item.brand_name or '[brand]'} "
        f"was officially announced on {item.announcement_date or '[date]'} in 2024."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_supported_leaf,
        sources=all_sources,
        additional_instruction="Verify on the page that the announcement date is in 2024 and matches the stated date text. "
                               "If only month+year is clearly indicated in 2024, that is acceptable."
    )

    # ---------------- Role title ----------------
    role_node = evaluator.add_parallel(
        id=f"partnership_{idx+1}_role_title",
        desc="Specific partnership role title is provided and supported by the source(s)",
        parent=partnership_node,
        critical=True
    )
    role_provided = evaluator.add_custom_node(
        result=_nonempty(item.role_title),
        id=f"partnership_{idx+1}_role_title_provided",
        desc="Role title is provided",
        parent=role_node,
        critical=True
    )
    role_supported_leaf = evaluator.add_leaf(
        id=f"partnership_{idx+1}_role_title_supported",
        desc="Provided source(s) support the stated role title for the celebrity at the brand",
        parent=role_node,
        critical=True
    )
    role_claim = (
        f"The source states that {item.celebrity_name or '[celebrity]'} serves as '{item.role_title or '[role]'}' for {item.brand_name or '[brand]'}."
    )
    await evaluator.verify(
        claim=role_claim,
        node=role_supported_leaf,
        sources=all_sources,
        additional_instruction="Check that the specific role title (e.g., Global Brand Ambassador, Brand Ambassador, International Ambassador, Creative Director) matches the wording on the source."
    )

    # ---------------- Official role vs one-time campaign ----------------
    official_role_leaf = evaluator.add_leaf(
        id=f"partnership_{idx+1}_not_one_time_campaign",
        desc="Partnership is an official role (ambassador/global ambassador/creative director or similar); not merely a one-time campaign appearance",
        parent=partnership_node,
        critical=True
    )
    official_role_claim = (
        f"The page indicates that {item.celebrity_name or '[celebrity]'} holds an official appointment (e.g., ambassador/global ambassador/creative director) with {item.brand_name or '[brand]'}, not just appearing in a single ad campaign."
    )
    await evaluator.verify(
        claim=official_role_claim,
        node=official_role_leaf,
        sources=all_sources,
        additional_instruction="Pass only if the page explicitly frames it as an official appointment/ambassadorship or similar ongoing role, rather than just appearing in a one-off campaign."
    )

    # ---------------- Parent company / ownership ----------------
    parent_node_pc = evaluator.add_parallel(
        id=f"partnership_{idx+1}_parent_company_identifiable",
        desc="Brand parent company/ownership provided (or clearly identifiable via cited sources)",
        parent=partnership_node,
        critical=True
    )
    parent_company_provided = evaluator.add_custom_node(
        result=_nonempty(item.parent_company),
        id=f"partnership_{idx+1}_parent_company_provided",
        desc="Parent company/ownership name is provided",
        parent=parent_node_pc,
        critical=True
    )
    parent_company_supported_leaf = evaluator.add_leaf(
        id=f"partnership_{idx+1}_parent_company_supported",
        desc="Provided source(s) support the stated parent company/ownership of the brand",
        parent=parent_node_pc,
        critical=True
    )
    pc_sources = item.parent_company_sources or all_sources
    parent_company_claim = (
        f"The brand {item.brand_name or '[brand]'} is owned by or part of '{item.parent_company or '[parent company]'}'."
    )
    await evaluator.verify(
        claim=parent_company_claim,
        node=parent_company_supported_leaf,
        sources=pc_sources,
        additional_instruction="Confirm that the cited page indicates ownership/parent company (e.g., 'a brand of L'Oréal', 'part of Estée Lauder Companies', 'owned by Coty'). Allow synonymous phrasing."
    )

    # ---------------- Source quality and explicit naming ----------------
    verifiable_source_leaf = evaluator.add_leaf(
        id=f"partnership_{idx+1}_verifiable_source",
        desc="At least one source is a verifiable industry/official source (brand site/press release or recognized fashion/beauty publication)",
        parent=partnership_node,
        critical=True
    )
    verifiable_source_claim = (
        "This webpage is an official brand/press-release page or a recognized fashion/beauty industry publication (e.g., WWD, Vogue, Harper's Bazaar, ELLE, Allure, official brand newsroom)."
    )
    await evaluator.verify(
        claim=verifiable_source_claim,
        node=verifiable_source_leaf,
        sources=all_sources,
        additional_instruction="Judge by the page branding/domain and page content (press release markers, masthead, brand site). Pass if any provided URL qualifies."
    )

    source_explicit_leaf = evaluator.add_leaf(
        id=f"partnership_{idx+1}_source_explicitly_names_celebrity_and_brand",
        desc="Source explicitly states both the celebrity and brand names in connection with the partnership",
        parent=partnership_node,
        critical=True
    )
    explicit_claim = (
        f"The page explicitly mentions both {item.celebrity_name or '[celebrity]'} and {item.brand_name or '[brand]'} in connection with the ambassador/official partnership."
    )
    await evaluator.verify(
        claim=explicit_claim,
        node=source_explicit_leaf,
        sources=all_sources,
        additional_instruction="Look for explicit mention of both the celebrity and the brand linked to the partnership/role on the page."
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
    Evaluate an answer for the 2024 beauty brand partnerships task.
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
        default_model=model
    )

    # Extract partnerships from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_partnerships(),
        template_class=PartnershipsExtraction,
        extraction_name="partnerships_extraction"
    )

    total_found = len(extracted.partnerships)
    evaluator.add_custom_info(
        {"reported_partnership_count_in_answer": total_found},
        info_type="stats",
        info_name="extraction_stats"
    )

    # Select exactly 4 items for evaluation (pad with blanks if fewer)
    selected: List[PartnershipItem] = _first_k(extracted.partnerships, 4)

    # ---------------- Global constraints ----------------
    global_node = evaluator.add_parallel(
        id="global_requirements",
        desc="Global constraints across the full set of partnerships",
        parent=root,
        critical=True
    )

    # Following the evaluation policy, allow answers that list >=4; we evaluate the first 4.
    provides_four = evaluator.add_custom_node(
        result=(total_found >= 4),
        id="provides_four_partnerships",
        desc="Response provides at least 4 partnership entries (first 4 are evaluated)",
        parent=global_node,
        critical=True
    )

    # Uniqueness across the selected 4 (require all non-empty and all distinct)
    celeb_names = [_norm(p.celebrity_name) for p in selected]
    brand_names = [_norm(p.brand_name) for p in selected]

    unique_celebrities_result = all(n for n in celeb_names) and len(set(celeb_names)) == 4
    unique_brands_result = all(n for n in brand_names) and len(set(brand_names)) == 4

    evaluator.add_custom_node(
        result=unique_celebrities_result,
        id="unique_celebrities",
        desc="All 4 partnerships involve different celebrities (no repeats among the evaluated four)",
        parent=global_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=unique_brands_result,
        id="unique_brands",
        desc="All 4 partnerships involve different beauty brands (no repeats among the evaluated four)",
        parent=global_node,
        critical=True
    )

    # ---------------- Per-partnership verification ----------------
    for i in range(4):
        await verify_partnership(
            evaluator=evaluator,
            parent_node=root,
            item=selected[i],
            idx=i
        )

    return evaluator.get_summary()