import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "intl_film_festivals_2026_shorts"
TASK_DESCRIPTION = """
Identify 4 major international film festivals that took place or will take place in 2026. For each festival, provide the following information about their short film submission process: (1) The official name of the festival, (2) The exact start and end dates of the festival in 2026, (3) The submission deadline for short films, (4) The submission fee for short films, including the specific currency (EUR, USD, etc.), (5) The maximum duration limit for short films, and whether this limit includes or excludes credits, (6) The festival's policy regarding premiere status or prior screenings (e.g., world premiere required, regional premieres accepted, etc.), and (7) The URL to the official submission guidelines page on the festival's website. The festivals must be recognized as major international film festivals (not regional or local festivals), must have taken place or be scheduled to take place in 2026, and must have an official short film submission category.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FestivalItem(BaseModel):
    official_name: Optional[str] = None
    start_date_2026: Optional[str] = None
    end_date_2026: Optional[str] = None
    # Short film category (optional descriptive label like "Short Film Competition")
    short_category_name: Optional[str] = None

    # Submission specifics for short films
    short_submission_deadline: Optional[str] = None
    short_submission_fee: Optional[str] = None
    short_submission_currency: Optional[str] = None

    max_duration: Optional[str] = None  # keep as string for flexibility (e.g., "20 minutes", "00:20:00")
    duration_credits_policy: Optional[str] = None  # e.g., "includes credits", "excludes credits", or descriptive text

    premiere_policy: Optional[str] = None

    # URLs
    official_guidelines_url: Optional[str] = None  # must be an official festival domain (not aggregator)
    additional_official_urls: List[str] = Field(default_factory=list)  # other official pages like regulations, FAQ, calendar


class FestivalsExtraction(BaseModel):
    festivals: List[FestivalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals() -> str:
    return """
Extract exactly the first 4 distinct major international film festivals listed in the answer (ignore any extras after the first four). For each of these four, extract the following fields strictly as presented in the answer:

- official_name: The official festival name as written.
- start_date_2026: The start date (must refer to the 2026 edition).
- end_date_2026: The end date (must refer to the 2026 edition).
- short_category_name: The official short film category name (e.g., "Short Film Competition", "Shorts", etc.), if presented.
- short_submission_deadline: The submission deadline for short films (prefer the final/last cutoff if multiple deadlines appear; keep the descriptive label if needed, e.g., "Final deadline: 2026-02-01").
- short_submission_fee: The submission fee amount/value specifically for short films (include descriptors like Early/Regular/Late if that is what the answer gives).
- short_submission_currency: The explicit currency string/code/symbol tied to the short film submission fee (e.g., "EUR", "USD", "£", "€"). Do NOT infer a currency not mentioned.
- max_duration: The maximum duration limit for short films (e.g., "20 minutes", "00:20:00", or a textual form like "20 min").
- duration_credits_policy: Whether the duration limit includes or excludes credits; use a short phrase like "includes credits" or "excludes credits". If the answer doesn’t say, return null.
- premiere_policy: The festival’s premiere/prior-screening eligibility policy for short films as text (e.g., "world premiere required", "international or national premiere required", "no premiere required but must not be publicly available online", etc.).
- official_guidelines_url: The URL to the official submission guidelines/rules/regulations page on the festival’s own website/domain. This must be an official festival domain, not a third-party aggregator or submission platform (e.g., NOT filmfreeway.com, festhome.com, eventival.com, shortfilmdepot.com). If the answer only lists a third-party platform or no URL at all, set this to null.
- additional_official_urls: A list of other official festival domain URLs (if any were cited in the answer) that help verify dates, categories, deadlines, fees, durations, and premiere policies (e.g., "Regulations", "Rules", "FAQ", "Calendar/Schedule", "Industry/Professionals" sections). Exclude third-party platforms and aggregators.

General rules:
1) Extract ONLY what is explicitly stated in the answer. Do not invent or copy from your own knowledge.
2) If a required field is missing in the answer, return null for that field.
3) For URLs, extract ONLY full URLs that appear in the answer, in any format (plain, markdown, etc.). If the URL is missing a protocol, prepend "http://".
4) Do NOT include third-party submission platforms or aggregator sites in the official_guidelines_url or additional_official_urls.
5) Return a JSON with a "festivals" array of exactly 4 objects in the same order as they first appear in the answer. If the answer contains fewer than 4, return as many as available (and the evaluator will pad later).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s is not None and str(s).strip() != "")


def _normalize_name(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _gather_official_urls(item: FestivalItem) -> List[str]:
    urls: List[str] = []
    if _non_empty_str(item.official_guidelines_url):
        urls.append(item.official_guidelines_url.strip())  # type: ignore
    for u in item.additional_official_urls or []:
        if _non_empty_str(u):
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification for one festival                                               #
# --------------------------------------------------------------------------- #
async def verify_one_festival(
    evaluator: Evaluator,
    parent_node,
    item: FestivalItem,
    idx: int
) -> None:
    """
    Build verification sub-tree and run checks for a single festival.
    All leaves under this festival node are critical (as per rubric).
    """
    fest_label = item.official_name or f"Festival #{idx + 1}"
    fest_node = evaluator.add_parallel(
        id=f"festival_{idx + 1}",
        desc=f"Festival {idx + 1} (evaluate one of the four festivals provided): {fest_label}",
        parent=parent_node,
        critical=True  # The four festivals are required; each festival block is critical
    )

    urls_for_verification = _gather_official_urls(item)

    # 1) Official festival name is provided (existence)
    evaluator.add_custom_node(
        result=_non_empty_str(item.official_name),
        id=f"festival_{idx + 1}_official_name",
        desc="Official festival name is provided",
        parent=fest_node,
        critical=True
    )

    # 2) Major international status (knowledge-based; simple verify using answer context)
    node_major = evaluator.add_leaf(
        id=f"festival_{idx + 1}_major_international_status",
        desc="Festival qualifies as a major, well-established international film festival (not regional/local)",
        parent=fest_node,
        critical=True
    )
    claim_major = f"The festival named '{item.official_name or ''}' is a major, well-established international film festival (not a regional or local event)."
    await evaluator.verify(
        claim=claim_major,
        node=node_major,
        additional_instruction="Base your judgment on the common recognition and context within the provided answer. Focus on widely known top-tier international festivals (e.g., Cannes, Berlin, Venice, Toronto, Sundance, Locarno, Karlovy Vary, Busan, Rotterdam, etc.) or similar stature. If unclear or likely regional/local, mark as incorrect."
    )

    # 3) Exact start and end dates in 2026
    node_dates = evaluator.add_leaf(
        id=f"festival_{idx + 1}_dates_in_2026_start_end",
        desc="Exact start and end dates of the festival are provided and are in 2026",
        parent=fest_node,
        critical=True
    )
    claim_dates = (
        f"The 2026 edition of '{item.official_name or ''}' runs from '{item.start_date_2026 or ''}' to '{item.end_date_2026 or ''}', "
        f"and both dates are in the year 2026."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=node_dates,
        sources=urls_for_verification if urls_for_verification else None,
        additional_instruction="Verify the 2026 festival dates (start and end) from the provided official URL(s). Allow for reasonable formatting variations (e.g., 'May 14–25, 2026' vs '14–25 May 2026'). If the page only shows the 2026 dates without exact day precision and the claim states exact days, ensure they match when available."
    )

    # 4) Official short film category exists
    node_short_cat = evaluator.add_leaf(
        id=f"festival_{idx + 1}_official_short_film_category",
        desc="Festival has an official short film submission category",
        parent=fest_node,
        critical=True
    )
    if item.short_category_name:
        claim_short_cat = (
            f"The festival '{item.official_name or ''}' has an official short film category, e.g., '{item.short_category_name}'."
        )
    else:
        claim_short_cat = (
            f"The festival '{item.official_name or ''}' has an official short film category for submissions (short films accepted)."
        )
    await evaluator.verify(
        claim=claim_short_cat,
        node=node_short_cat,
        sources=urls_for_verification if urls_for_verification else None,
        additional_instruction="Confirm an official short film category/section is recognized by the festival (e.g., 'Short Film Competition', 'Shorts', 'Short Films')."
    )

    # 5) Submission deadline for short films
    node_deadline = evaluator.add_leaf(
        id=f"festival_{idx + 1}_short_film_submission_deadline",
        desc="Submission deadline date for short films is provided",
        parent=fest_node,
        critical=True
    )
    claim_deadline = (
        f"The short film submission deadline for the 2026 edition of '{item.official_name or ''}' is '{item.short_submission_deadline or ''}'."
    )
    await evaluator.verify(
        claim=claim_deadline,
        node=node_deadline,
        sources=urls_for_verification if urls_for_verification else None,
        additional_instruction="Check the official rules/guidelines for short film submission deadlines. If multiple deadlines (Early/Regular/Late) exist, it is acceptable if the provided deadline matches one of the official ones, ideally the final cutoff for 2026 short films."
    )

    # 6) Submission fee for short films including currency
    node_fee = evaluator.add_leaf(
        id=f"festival_{idx + 1}_short_film_submission_fee_currency",
        desc="Submission fee for short films is provided including the specific currency",
        parent=fest_node,
        critical=True
    )
    fee_desc = f"{item.short_submission_fee or ''}"
    currency_desc = f"{item.short_submission_currency or ''}"
    claim_fee = (
        f"The submission fee for short films is '{fee_desc}' and the currency stated is '{currency_desc}'."
    )
    await evaluator.verify(
        claim=claim_fee,
        node=node_fee,
        sources=urls_for_verification if urls_for_verification else None,
        additional_instruction="Verify that the page explicitly mentions the fee for short films and the currency (e.g., EUR, USD, £, €). Accept reasonable expressions (tiers, early/regular/late) as long as the claimed fee and currency appear on the official page(s)."
    )

    # 7) Maximum duration limit for short films
    node_duration = evaluator.add_leaf(
        id=f"festival_{idx + 1}_short_film_max_duration",
        desc="Maximum duration limit for short films is provided",
        parent=fest_node,
        critical=True
    )
    claim_duration = (
        f"The maximum duration for short films is '{item.max_duration or ''}'."
    )
    await evaluator.verify(
        claim=claim_duration,
        node=node_duration,
        sources=urls_for_verification if urls_for_verification else None,
        additional_instruction="Verify the maximum running time permitted for short films (e.g., 15 or 20 minutes). Allow equivalent formatting (e.g., '20 minutes' vs '00:20:00')."
    )

    # 8) Whether duration includes or excludes credits
    node_credits = evaluator.add_leaf(
        id=f"festival_{idx + 1}_duration_includes_or_excludes_credits",
        desc="States whether the duration limit includes or excludes credits",
        parent=fest_node,
        critical=True
    )
    claim_credits = (
        f"The duration limit policy for short films explicitly states: '{item.duration_credits_policy or ''}' (i.e., whether it includes or excludes credits)."
    )
    await evaluator.verify(
        claim=claim_credits,
        node=node_credits,
        sources=urls_for_verification if urls_for_verification else None,
        additional_instruction="Confirm if the duration limit includes credits, excludes credits, or otherwise specifies how credits are treated."
    )

    # 9) Premiere/prior-screening policy
    node_premiere = evaluator.add_leaf(
        id=f"festival_{idx + 1}_premiere_prior_screening_policy",
        desc="Premiere/prior-screening eligibility policy is described",
        parent=fest_node,
        critical=True
    )
    claim_premiere = (
        f"The short film premiere/prior-screening policy is: '{item.premiere_policy or ''}'."
    )
    await evaluator.verify(
        claim=claim_premiere,
        node=node_premiere,
        sources=urls_for_verification if urls_for_verification else None,
        additional_instruction="Verify the specific premiere/prior screening requirements for short films (e.g., world premiere required, regional premiere accepted, must not be publicly available online, etc.)."
    )

    # 10) Official guidelines URL on an official festival domain (not an aggregator)
    node_guidelines = evaluator.add_leaf(
        id=f"festival_{idx + 1}_official_guidelines_url",
        desc="Provides a publicly accessible URL to the official submission guidelines page on an official festival domain (not a third-party aggregator/submission platform)",
        parent=fest_node,
        critical=True
    )
    if _non_empty_str(item.official_guidelines_url):
        claim_guidelines = (
            f"This URL is an official festival site page (not a third-party aggregator) that contains submission rules/guidelines for short films: {item.official_guidelines_url}"
        )
        await evaluator.verify(
            claim=claim_guidelines,
            node=node_guidelines,
            sources=item.official_guidelines_url,
            additional_instruction="Confirm the page belongs to the official festival domain and includes submission rules/guidelines/regulations for short films. Treat platforms like filmfreeway.com, festhome.com, eventival.com, shortfilmdepot.com, filmchief.com as third-party (NOT official)."
        )
    else:
        # No URL provided -> immediate failure for this critical requirement
        node_guidelines.score = 0.0
        node_guidelines.status = "failed"

    # 11) Verifiable via official sources
    node_verifiable = evaluator.add_leaf(
        id=f"festival_{idx + 1}_verifiable_via_official_sources",
        desc="The provided attribute values are verifiable via official festival sources (e.g., the festival’s official site pages)",
        parent=fest_node,
        critical=True
    )
    claim_verifiable = (
        "The provided official URL(s) contain sufficient information to verify the short film submission details, including relevant items like 2026 dates, category existence, deadlines, fees with currency, maximum duration, and premiere policy."
    )
    await evaluator.verify(
        claim=claim_verifiable,
        node=node_verifiable,
        sources=urls_for_verification if urls_for_verification else None,
        additional_instruction="Check whether, across the cited official URL(s), the necessary short-film submission facts can be verified. It's acceptable if the details are distributed across multiple official pages (e.g., 'Regulations', 'Rules', 'FAQ', 'Submissions', 'Calendar')."
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
    Evaluate an answer for the 2026 international film festivals short film submission task.
    """
    # Initialize evaluator (root kept non-critical to avoid structural constraints; critical checks are added as children)
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_festivals(),
        template_class=FestivalsExtraction,
        extraction_name="festivals_extraction"
    )

    # Normalize to exactly 4 items (pad with empty if fewer; trim if more)
    items: List[FestivalItem] = list(extracted.festivals or [])
    if len(items) > 4:
        items = items[:4]
    while len(items) < 4:
        items.append(FestivalItem())

    # Global critical checks under root
    # 1) Exactly 4 distinct festivals with names provided
    names = [_normalize_name(f.official_name) for f in items if _non_empty_str(f.official_name)]
    exactly_4 = (len([n for n in names if n]) == 4)
    evaluator.add_custom_node(
        result=exactly_4,
        id="global_exactly_4_festivals",
        desc="Exactly 4 distinct festivals are evaluated (first four in the answer; all with names).",
        parent=root,
        critical=True
    )

    # 2) No duplicates among the four names
    unique_names_count = len(set(names))
    no_duplicates = (len(names) == 4 and unique_names_count == 4)
    evaluator.add_custom_node(
        result=no_duplicates,
        id="global_no_duplicates",
        desc="No duplicate festivals among the four (case-insensitive).",
        parent=root,
        critical=True
    )

    # Build verification subtrees for each festival (all critical as per rubric)
    for i in range(4):
        await verify_one_festival(evaluator, root, items[i], i)

    # Return final structured summary
    return evaluator.get_summary()