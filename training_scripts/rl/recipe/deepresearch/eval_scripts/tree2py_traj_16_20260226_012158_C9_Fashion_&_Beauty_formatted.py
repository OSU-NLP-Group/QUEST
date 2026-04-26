import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_beauty_partnerships_2025"
TASK_DESCRIPTION = (
    "Identify at least three celebrities from the entertainment industry who entered into new official partnerships "
    "with established beauty or cosmetics brands in 2025. For each celebrity-brand partnership you identify, provide "
    "the following information: (1) The celebrity's full name and their primary field in entertainment (music, film, "
    "television, or modeling); (2) The beauty or cosmetics brand's name; (3) The official title or role designation "
    "given to the celebrity in this partnership (e.g., Global Brand Partner, Global Ambassador, International "
    "Ambassador, etc.); (4) The specific date or month in 2025 when the partnership was publicly announced; (5) The "
    "year the beauty brand was originally founded or established; (6) Any publicly documented ethical certifications "
    "or standards held by the brand (such as cruelty-free certification, vegan certification, or similar), if "
    "applicable; (7) Reference URLs supporting each piece of information. The partnerships must meet ALL of the "
    "following criteria: The partnership was publicly announced or officially established during the year 2025 (not "
    "ongoing relationships from previous years); The partnership is specifically with a beauty or cosmetics brand "
    "(including skincare, makeup, or fragrance - not wine, wellness supplements, fitness products, fashion clothing, "
    "or other non-beauty categories); The celebrity holds an official, named title or role with the brand; The brand "
    "was established before 2025; All information must be verifiable through official brand communications, press "
    "releases, or credible beauty industry sources."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class PartnershipEntry(BaseModel):
    celebrity_full_name: Optional[str] = None
    celebrity_primary_field: Optional[str] = None  # as written in the answer (e.g., music, film, television, modeling, actor, singer, etc.)
    brand_name: Optional[str] = None
    role_title: Optional[str] = None
    announcement_date: Optional[str] = None  # e.g., "Jan 2025", "2025-03-12", "March 2025"
    brand_founding_year: Optional[str] = None  # keep as string for flexibility
    ethics_info: List[str] = Field(default_factory=list)  # e.g., ["cruelty-free", "vegan", "Leaping Bunny"]
    ethics_note: Optional[str] = None  # e.g., "none found", "N/A"
    sources: List[str] = Field(default_factory=list)  # all URLs cited in the answer relevant to this partnership


class PartnershipsExtraction(BaseModel):
    partnerships: List[PartnershipEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_partnerships() -> str:
    return (
        "Extract up to five (5) celebrity–beauty/cosmetics brand partnerships described in the answer. "
        "Return them in a JSON object with field 'partnerships' containing an array of objects, each with:\n"
        "- celebrity_full_name: The celebrity’s full name exactly as written in the answer.\n"
        "- celebrity_primary_field: The primary entertainment field named in the answer (e.g., music, film, television, modeling). "
        "If the answer uses a near-synonym (e.g., actor/actress), extract exactly that text.\n"
        "- brand_name: The beauty or cosmetics brand’s name.\n"
        "- role_title: The official partnership title/role as written (e.g., Global Ambassador, Global Brand Partner).\n"
        "- announcement_date: The specific date or month in 2025 stated for when the partnership was publicly announced. "
        "If only a month is given, extract the month with year (e.g., 'March 2025'). If no date/month is provided, set to null.\n"
        "- brand_founding_year: The year the beauty brand was originally founded/established, as provided in the answer (string; do not convert to integer). "
        "If missing, set to null.\n"
        "- ethics_info: A list of any ethical certifications/standards claimed in the answer for the brand (e.g., cruelty-free, vegan, Leaping Bunny). "
        "If none claimed, return an empty list.\n"
        "- ethics_note: If the answer explicitly states that no ethical certifications were found or that it is not applicable, capture that phrasing here; else null.\n"
        "- sources: All URLs explicitly provided in the answer that support any of the above details for this partnership. "
        "Extract actual URLs only (plain or markdown links). Include brand press releases, brand newsroom pages, industry outlets, etc. "
        "Do not invent URLs. If none are provided for this partnership, return an empty list.\n\n"
        "Only extract what is explicitly present in the answer; do not infer or add information not stated. "
        "If the answer contains more than 5 partnerships, extract the first 5 mentioned."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_primary_field(field_text: Optional[str]) -> Optional[str]:
    if not field_text:
        return None
    ft = field_text.strip().lower()
    # Accept common synonyms and map to target categories when possible
    mapping = {
        "music": "music",
        "musician": "music",
        "singer": "music",
        "rapper": "music",
        "film": "film",
        "movie": "film",
        "movies": "film",
        "cinema": "film",
        "actor": "film",     # could be TV as well, but accept as film/television category coverage
        "actress": "film",
        "television": "television",
        "tv": "television",
        "t.v.": "television",
        "model": "modeling",
        "modeling": "modeling",
        "modelling": "modeling",
    }
    # Exact match first
    if ft in mapping:
        return mapping[ft]
    # Multi-word includes
    if "music" in ft or "singer" in ft or "musician" in ft or "rapper" in ft:
        return "music"
    if "television" in ft or ft.startswith("tv") or "tv" in ft:
        return "television"
    if "film" in ft or "movie" in ft or "cinema" in ft:
        return "film"
    if "model" in ft:
        return "modeling"
    return None


def _has_2025(text: Optional[str]) -> bool:
    if not text:
        return False
    return "2025" in text


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic heuristic: keep strings that look like URLs
    keep = []
    for u in urls:
        if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")):
            keep.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in keep:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification for one partnership                                            #
# --------------------------------------------------------------------------- #
async def verify_partnership(
    evaluator: Evaluator,
    parent_node,
    entry: PartnershipEntry,
    index_zero_based: int,
) -> Any:
    idx = index_zero_based + 1
    partnership_node = evaluator.add_parallel(
        id=f"partnership_{idx}",
        desc=f"Partnership entry {idx} (counts toward the ≥3 requirement if it passes all per-partnership critical checks)",
        parent=parent_node,
        critical=False
    )

    # Normalize sources early
    sources_list = _valid_urls(entry.sources)

    # 1) Celebrity full name provided (critical, existence check)
    evaluator.add_custom_node(
        result=_nonempty(entry.celebrity_full_name),
        id=f"p{idx}_celebrity_full_name",
        desc="Provides the celebrity's full name",
        parent=partnership_node,
        critical=True
    )

    # 2) Celebrity primary field provided and is one of the allowed categories (or acceptable synonym)
    normalized_field = _normalize_primary_field(entry.celebrity_primary_field)
    evaluator.add_custom_node(
        result=normalized_field in {"music", "film", "television", "modeling"},
        id=f"p{idx}_celebrity_primary_field",
        desc="Specifies the celebrity's primary entertainment field (music, film, television, or modeling)",
        parent=partnership_node,
        critical=True
    )

    # 3) Brand name provided (critical, existence check)
    evaluator.add_custom_node(
        result=_nonempty(entry.brand_name),
        id=f"p{idx}_brand_name",
        desc="Provides the beauty/cosmetics brand name",
        parent=partnership_node,
        critical=True
    )

    # 4) Brand category is beauty/cosmetics (critical, verify via sources)
    brand_cat_leaf = evaluator.add_leaf(
        id=f"p{idx}_brand_category_is_beauty",
        desc="Brand/category is beauty or cosmetics (e.g., skincare/makeup/fragrance) and not excluded categories (e.g., wine, supplements, fashion clothing)",
        parent=partnership_node,
        critical=True
    )
    brand_for_claim = entry.brand_name or "the brand"
    claim_brand_category = f"The brand '{brand_for_claim}' is a beauty or cosmetics brand (e.g., skincare, makeup, cosmetics, or fragrance)."
    await evaluator.verify(
        claim=claim_brand_category,
        node=brand_cat_leaf,
        sources=sources_list,
        additional_instruction=(
            "Confirm that the brand clearly operates in beauty/cosmetics categories. "
            "Look for cues like 'skincare', 'makeup', 'cosmetics', 'fragrance', 'beauty brand'. "
            "Do not accept unrelated categories like wine, supplements, fitness equipment, or fashion clothing."
        )
    )

    # 5) Role title provided and supported (critical, verify via sources)
    role_leaf = evaluator.add_leaf(
        id=f"p{idx}_role_title",
        desc="Provides the official title/role designation given to the celebrity (e.g., Global Ambassador/Brand Partner)",
        parent=partnership_node,
        critical=True
    )
    celeb_for_claim = entry.celebrity_full_name or "the celebrity"
    role_for_claim = entry.role_title or "an official ambassador/partner title"
    claim_role = f"In 2025, {celeb_for_claim} was appointed by {brand_for_claim} as its official '{role_for_claim}' (or a clearly equivalent role title)."
    await evaluator.verify(
        claim=claim_role,
        node=role_leaf,
        sources=sources_list,
        additional_instruction=(
            "Verify that the page(s) explicitly mention the celebrity's official title (e.g., Global Ambassador, Global Brand Partner, International Ambassador, Muse, Face of). "
            "Allow reasonable synonyms (e.g., 'Face of' ≈ ambassador/partner)."
        )
    )

    # 6) Announcement date/month is in 2025 (critical, verify via sources)
    date_leaf = evaluator.add_leaf(
        id=f"p{idx}_announcement_date_or_month_in_2025",
        desc="Provides a specific announcement date or month in 2025",
        parent=partnership_node,
        critical=True
    )
    date_fragment = entry.announcement_date.strip() if _nonempty(entry.announcement_date) else "sometime"
    claim_date = (
        f"The partnership between {celeb_for_claim} and {brand_for_claim} was publicly announced in 2025, "
        f"specifically around {date_fragment} in 2025."
    )
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=sources_list,
        additional_instruction=(
            "Check the press release date or article publication date. "
            "Month-year (e.g., 'March 2025') is acceptable. "
            "If multiple dates appear, ensure the announcement/appointment reference is in 2025."
        )
    )

    # 7) New in 2025 (not ongoing from prior years) (critical, verify via sources)
    new_leaf = evaluator.add_leaf(
        id=f"p{idx}_new_not_ongoing_from_prior_years",
        desc="Establishes that this is a new 2025 partnership (not merely an ongoing relationship from previous years)",
        parent=partnership_node,
        critical=True
    )
    claim_new = (
        f"This 2025 appointment of {celeb_for_claim} with {brand_for_claim} represents a new official partnership announced in 2025, "
        f"and is not merely a continuation of a partnership announced in an earlier year."
    )
    await evaluator.verify(
        claim=claim_new,
        node=new_leaf,
        sources=sources_list,
        additional_instruction=(
            "Look for language indicating 'new', 'joins', 'appointed', 'announced', 'debut', etc., in 2025. "
            "If the pages show a clearly earlier (pre-2025) appointment of the same role, then this fails."
        )
    )

    # 8) Brand founding year provided and pre-2025 (critical, verify via sources)
    founding_leaf = evaluator.add_leaf(
        id=f"p{idx}_brand_founding_year_pre_2025",
        desc="Provides the brand founding/establishment year and it is before 2025",
        parent=partnership_node,
        critical=True
    )
    founding_year_txt = entry.brand_founding_year or "an earlier year"
    claim_founding = f"The brand {brand_for_claim} was founded in {founding_year_txt}, which is before 2025."
    await evaluator.verify(
        claim=claim_founding,
        node=founding_leaf,
        sources=sources_list,
        additional_instruction=(
            "Confirm the brand's founding/established year from official brand pages or credible sources. "
            "The year must be strictly before 2025."
        )
    )

    # 9) Ethics info if applicable (critical). If claims exist, verify; if explicitly none, pass.
    ethics_leaf = evaluator.add_leaf(
        id=f"p{idx}_ethics_info_if_applicable",
        desc="Provides any publicly documented ethical certifications/standards held by the brand (or explicitly states none found/not applicable), with support if claims are made",
        parent=partnership_node,
        critical=True
    )
    if entry.ethics_info:
        ethics_list_txt = ", ".join(entry.ethics_info)
        claim_ethics = (
            f"The brand {brand_for_claim} holds the following publicly documented ethical certifications or standards: {ethics_list_txt}."
        )
        await evaluator.verify(
            claim=claim_ethics,
            node=ethics_leaf,
            sources=sources_list,
            additional_instruction=(
                "Look for explicit mentions such as 'cruelty-free', 'Leaping Bunny certified', 'PETA-certified', 'vegan', etc. "
                "Accept reasonable synonyms. If the pages do not substantiate these claims, mark as not supported."
            )
        )
    else:
        # If explicitly stated 'none' or 'not applicable' in the answer, we allow pass without external verification.
        note = (entry.ethics_note or "").strip().lower()
        explicitly_none = any(tok in note for tok in ["none", "not applicable", "n/a", "no certification", "no certifications"])
        if explicitly_none or not note:
            ethics_leaf.score = 1.0
            ethics_leaf.status = "passed"
        else:
            # If a note is present but ambiguous, attempt to verify that no explicit certifications are claimed on sources.
            claim_no_ethics = (
                f"There are no explicit publicly documented ethical certifications for the brand {brand_for_claim} mentioned in these sources."
            )
            await evaluator.verify(
                claim=claim_no_ethics,
                node=ethics_leaf,
                sources=sources_list,
                additional_instruction=(
                    "If the provided sources do not include explicit certification claims (e.g., Leaping Bunny, PETA, vegan), "
                    "then this statement is considered supported. If sources claim certifications, then this statement is not supported."
                )
            )

    # 10) Sources quality and coverage (critical, verify that at least one URL is credible/official and covers the partnership core)
    sources_quality_leaf = evaluator.add_leaf(
        id=f"p{idx}_sources_quality_and_coverage",
        desc="Provides reference URL(s) from official brand communications/press releases or credible beauty-industry sources supporting the key claims for this partnership",
        parent=partnership_node,
        critical=True
    )
    claim_sources_quality = (
        f"At least one of these URLs is an official brand communication (brand website newsroom/press release or brand social account) "
        f"or a credible beauty-industry source (e.g., WWD, Vogue, Elle, Allure, Business Wire, PR Newswire) that explicitly reports "
        f"that {celeb_for_claim} was appointed as {role_for_claim} by {brand_for_claim} in 2025."
    )
    await evaluator.verify(
        claim=claim_sources_quality,
        node=sources_quality_leaf,
        sources=sources_list,
        additional_instruction=(
            "Accept brand domains, official press releases (Business Wire, PR Newswire), and top-tier beauty/fashion trades. "
            "The page should clearly state the appointment and the brand-celebrity linkage in 2025."
        )
    )

    return partnership_node


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
    Evaluate an answer for celebrity-beauty partnerships announced in 2025.
    """
    evaluator = Evaluator()
    # Important: Use PARALLEL at root to avoid sequential short-circuiting that would skip 'at least 3' check.
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_partnerships(),
        template_class=PartnershipsExtraction,
        extraction_name="extracted_partnerships"
    )

    # Select up to 5 partnerships from the answer
    selected: List[PartnershipEntry] = list(extracted.partnerships[:5]) if extracted and extracted.partnerships else []

    # Build subtree for evaluating partnerships
    eval_partnerships_parent = evaluator.add_parallel(
        id="evaluate_partnership_entries_up_to_5",
        desc="Evaluate each provided partnership entry (up to 5) against the per-partnership requirements",
        parent=root,
        critical=False
    )

    partnership_nodes = []
    for i, p in enumerate(selected):
        node = await verify_partnership(evaluator, eval_partnerships_parent, p, i)
        partnership_nodes.append(node)

    # Compute how many partnership nodes fully passed (all critical checks satisfied)
    valid_count = 0
    for node in partnership_nodes:
        # Ensure scores are computed
        _ = node.aggregated_score
        if node.aggregated_score == 1.0:
            valid_count += 1

    # Add the critical check: at least three valid partnerships
    at_least_3_node = evaluator.add_custom_node(
        result=(valid_count >= 3),
        id="at_least_3_valid_partnerships",
        desc="At least three partnership entries pass all per-partnership critical checks (i.e., are qualifying 2025 new celebrity–beauty/cosmetics partnerships with required fields and credible sources)",
        parent=root,
        critical=True
    )

    # Record additional info for transparency
    evaluator.add_custom_info(
        info={
            "partnerships_extracted_count": len(selected),
            "valid_partnerships_count": valid_count,
            "requirement_met_at_least_3": valid_count >= 3
        },
        info_type="counts",
        info_name="partnership_validation_counts"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()