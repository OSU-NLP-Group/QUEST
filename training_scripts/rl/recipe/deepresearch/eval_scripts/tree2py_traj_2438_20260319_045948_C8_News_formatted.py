import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "trump_eo_2025_policy_buckets"
TASK_DESCRIPTION = """
In 2025, President Donald Trump signed 225 executive orders addressing various policy priorities. As part of a comprehensive policy analysis, identify four specific executive orders that meet the following criteria:

1. One executive order signed on January 20, 2025 (Inauguration Day) that addresses energy policy, specifically promoting American energy production or resource development.

2. One executive order signed in May 2025 that relates to trade policy or implements an international trade agreement.

3. One executive order signed in January 2025 that addresses government efficiency, regulatory reform, or deregulation initiatives.

4. One executive order signed in June 2025 that implements, references, or relates to an international agreement or treaty.

For each of the four executive orders, provide:
- The official executive order number
- The official title
- The exact date it was signed
- A reference URL from an official U.S. government source (whitehouse.gov or federalregister.gov) that documents the executive order
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EOItem(BaseModel):
    eo_number: Optional[str] = None
    title: Optional[str] = None
    signed_date: Optional[str] = None  # Keep string; allow natural formats
    reference_urls: List[str] = Field(default_factory=list)

    # Optional subject clues from the answer to help mapping (non-binding)
    subject_summary: Optional[str] = None
    subject_tags: List[str] = Field(default_factory=list)

    # Helper boolean flags (LLM-derived from the answer text; used for mapping only)
    is_on_jan_20_2025: Optional[bool] = None
    is_in_may_2025: Optional[bool] = None
    is_in_jan_2025: Optional[bool] = None
    is_in_jun_2025: Optional[bool] = None
    # Subject buckets
    is_energy_promote: Optional[bool] = None
    is_trade_policy_or_trade_agreement: Optional[bool] = None
    is_gov_efficiency_or_regreform_or_dereg: Optional[bool] = None
    is_international_agreement_or_treaty: Optional[bool] = None


class EOExtraction(BaseModel):
    executive_orders: List[EOItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_executive_orders() -> str:
    return """
    Extract all executive orders referenced in the answer (up to 8 items). For each, extract:

    - eo_number: The official Executive Order number exactly as written in the answer (e.g., "14212" or "Executive Order 14212"). Do not fabricate.
    - title: The official title exactly as written in the answer. Do not fabricate.
    - signed_date: The exact signing date as reported in the answer (keep as a string; e.g., "January 20, 2025").
    - reference_urls: A list of all URLs provided in the answer that directly document this EO (if any). Extract only actual URLs present in the answer text (including Markdown links). Do not invent URLs.
    - subject_summary: A short phrase from the answer that describes the policy area/purpose of the EO (verbatim or close paraphrase).
    - subject_tags: Zero or more coarse tags based on the answer text, from among: 
        ["energy", "trade", "international_agreement", "treaty", "government_efficiency", "regulatory_reform", "deregulation", "economy", "foreign_policy", "resources", "production"].

    Also compute boolean helper flags based ONLY on the answer text (not external knowledge):
    - is_on_jan_20_2025: true if the signed_date explicitly indicates January 20, 2025.
    - is_in_may_2025: true if the signed_date indicates any day in May 2025.
    - is_in_jan_2025: true if the signed_date indicates any day in January 2025.
    - is_in_jun_2025: true if the signed_date indicates any day in June 2025.

    Subject bucket flags (true only if the answer text clearly supports it):
    - is_energy_promote: true if the EO addresses energy policy that promotes American energy production or resource/resource development.
    - is_trade_policy_or_trade_agreement: true if the EO relates to trade policy or implements an international trade agreement.
    - is_gov_efficiency_or_regreform_or_dereg: true if the EO addresses government efficiency, regulatory reform, or deregulation initiatives.
    - is_international_agreement_or_treaty: true if the EO implements, references, or relates to an international agreement or treaty.

    Return a JSON object with:
    {
      "executive_orders": [ ... up to 8 EO objects as defined above ... ]
    }

    IMPORTANT:
    - If any field is missing in the answer, set it to null (or empty list for reference_urls/subject_tags).
    - Do NOT infer or fabricate any URLs or numbers that are not explicitly mentioned in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_truthy(v: Optional[bool]) -> bool:
    return True if v is True else False


def _safe_lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _maybe_contains(text: Optional[str], keywords: List[str]) -> bool:
    t = _safe_lower(text)
    return any(kw in t for kw in keywords)


def _looks_like_month(text: Optional[str], month_keyword: str) -> bool:
    # Fuzzy check for month name in a free-form date string
    t = _safe_lower(text)
    return month_keyword.lower() in t


def _looks_like_specific_date(text: Optional[str], month_keyword: str, day: str, year: str) -> bool:
    t = _safe_lower(text)
    return (month_keyword.lower() in t) and (day in t) and (year in t)


def is_official_url(url: str) -> bool:
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = (parsed.netloc or "").lower()
        # Allow whitehouse.gov (including subdomains) and federalregister.gov (including subdomains)
        return (
            host == "whitehouse.gov" or host.endswith(".whitehouse.gov") or
            host == "federalregister.gov" or host.endswith(".federalregister.gov")
        )
    except Exception:
        return False


def filter_official_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if is_official_url(u)]


def select_item_for_bucket(items: List[EOItem], used: set, cond) -> Optional[int]:
    for idx, it in enumerate(items):
        if idx in used:
            continue
        if cond(it):
            return idx
    return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_eo_bucket(
    evaluator: Evaluator,
    parent_node,
    bucket_index: int,
    bucket_desc: str,
    eo_item: EOItem,
    date_kind: str,  # one of: "exact_jan20_2025", "month_may_2025", "month_jan_2025", "month_jun_2025"
) -> None:
    """
    Build verification nodes for one EO bucket and perform verifications.
    This function adheres to the rubric tree leaf IDs and descriptions.
    """
    # Parent node (parallel, non-critical to allow partial credit across buckets)
    bucket_node = evaluator.add_parallel(
        id=f"executive_order_{bucket_index}",
        desc=bucket_desc,
        parent=parent_node,
        critical=False
    )

    # Prepare official URLs, use as evidence for all factual checks
    official_urls = filter_official_urls(eo_item.reference_urls or [])

    # 1) Reference URL requirement (critical, added FIRST to gate others automatically)
    ref_exists_and_official = len(official_urls) > 0
    evaluator.add_custom_node(
        result=ref_exists_and_official,
        id=f"eo{bucket_index}_reference",
        desc="Provide a reference URL from an official government source (whitehouse.gov or federalregister.gov) documenting this executive order",
        parent=bucket_node,
        critical=True
    )

    # 2) Number (critical)
    number_node = evaluator.add_leaf(
        id=f"eo{bucket_index}_number",
        desc="Provide the official executive order number",
        parent=bucket_node,
        critical=True
    )
    num_val = eo_item.eo_number or ""
    await evaluator.verify(
        claim=f'The official number of this executive order is "{num_val}". Accept formats like "Executive Order {num_val}" or "EO {num_val}".',
        node=number_node,
        sources=official_urls,
        additional_instruction="Verify on the provided official page that the EO number matches exactly. Allow minor formatting variations (e.g., 'Executive Order 14212' vs 'EO 14212')."
    )

    # 3) Title (critical)
    title_node = evaluator.add_leaf(
        id=f"eo{bucket_index}_title",
        desc="Provide the official title of the executive order",
        parent=bucket_node,
        critical=True
    )
    title_val = eo_item.title or ""
    await evaluator.verify(
        claim=f'The official title of this executive order is "{title_val}".',
        node=title_node,
        sources=official_urls,
        additional_instruction="Match the official EO title as shown on the page. Allow minor punctuation/casing differences."
    )

    # 4) Date/Month (critical; rubric-specific IDs per bucket)
    # Map bucket to node ID and claim
    if date_kind == "exact_jan20_2025":
        date_leaf_id = f"eo{bucket_index}_date"
        date_claim = "This executive order was signed on January 20, 2025."
        date_desc = "The executive order was signed on January 20, 2025"
    elif date_kind == "month_may_2025":
        date_leaf_id = f"eo{bucket_index}_month"
        date_claim = "This executive order was signed in May 2025."
        date_desc = "The executive order was signed in May 2025"
    elif date_kind == "month_jan_2025":
        date_leaf_id = f"eo{bucket_index}_month"
        date_claim = "This executive order was signed in January 2025."
        date_desc = "The executive order was signed in January 2025"
    elif date_kind == "month_jun_2025":
        date_leaf_id = f"eo{bucket_index}_month"
        date_claim = "This executive order was signed in June 2025."
        date_desc = "The executive order was signed in June 2025"
    else:
        # Fallback (should not occur)
        date_leaf_id = f"eo{bucket_index}_month"
        date_claim = "This executive order was signed in 2025."
        date_desc = "The executive order was signed in 2025"

    date_node = evaluator.add_leaf(
        id=date_leaf_id,
        desc=date_desc,
        parent=bucket_node,
        critical=True
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=official_urls,
        additional_instruction="Check the 'Signed' (or equivalent) date on the official page. For month-only requirements, ensure month and year match even if the day differs."
    )

    # 5) Subject/policy criterion (critical; rubric-specific wording per bucket)
    if date_kind == "exact_jan20_2025":
        subj_desc = "The executive order addresses energy policy, specifically promoting American energy production or resource development"
        subj_claim = "This executive order addresses energy policy, promoting American energy production or resource/resource development."
    elif date_kind == "month_may_2025":
        subj_desc = "The executive order relates to trade policy or implements an international trade agreement"
        subj_claim = "This executive order relates to trade policy or implements an international trade agreement."
    elif date_kind == "month_jan_2025":
        subj_desc = "The executive order addresses government efficiency, regulatory reform, or deregulation initiatives"
        subj_claim = "This executive order addresses government efficiency, regulatory reform, or deregulation initiatives."
    elif date_kind == "month_jun_2025":
        subj_desc = "The executive order implements, references, or relates to an international agreement or treaty"
        subj_claim = "This executive order implements, references, or relates to an international agreement or treaty."
    else:
        subj_desc = "Subject requirement"
        subj_claim = "This executive order meets the required subject criterion."

    subject_node = evaluator.add_leaf(
        id=f"eo{bucket_index}_subject",
        desc=subj_desc,
        parent=bucket_node,
        critical=True
    )
    await evaluator.verify(
        claim=subj_claim,
        node=subject_node,
        sources=official_urls,
        additional_instruction="Use the content of the official page (title, summary, body) to confirm the described policy area. Allow close paraphrases that clearly indicate the same subject."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 2025 Trump executive orders policy-bucket task.
    """
    # Initialize evaluator (make root non-critical to avoid forcing all children critical per framework constraints)
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

    # Extract EO candidates from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_executive_orders(),
        template_class=EOExtraction,
        extraction_name="executive_orders_extraction"
    )
    items: List[EOItem] = extracted.executive_orders or []

    # Simple selection for each bucket using extracted booleans and light heuristics
    used_indices: set = set()

    # Bucket 1: Jan 20, 2025 AND energy/resource development
    def cond_bucket1(e: EOItem) -> bool:
        date_match = _is_truthy(e.is_on_jan_20_2025) or _looks_like_specific_date(e.signed_date, "january", "20", "2025")
        subj_match = (
            _is_truthy(e.is_energy_promote) or
            ("energy" in [t.lower() for t in (e.subject_tags or [])]) or
            _maybe_contains(e.subject_summary, ["energy", "oil", "gas", "drilling", "lease", "resource", "production"])
        )
        return date_match and subj_match

    idx1 = select_item_for_bucket(items, used_indices, cond_bucket1)
    if idx1 is not None:
        used_indices.add(idx1)
        eo1 = items[idx1]
    else:
        eo1 = EOItem()

    # Bucket 2: May 2025 AND trade policy/trade agreement
    def cond_bucket2(e: EOItem) -> bool:
        date_match = _is_truthy(e.is_in_may_2025) or _looks_like_month(e.signed_date, "may") and "2025" in _safe_lower(e.signed_date)
        subj_match = (
            _is_truthy(e.is_trade_policy_or_trade_agreement) or
            ("trade" in [t.lower() for t in (e.subject_tags or [])]) or
            _maybe_contains(e.subject_summary, ["trade", "tariff", "agreement", "fta", "free trade"])
        )
        return date_match and subj_match

    idx2 = select_item_for_bucket(items, used_indices, cond_bucket2)
    if idx2 is not None:
        used_indices.add(idx2)
        eo2 = items[idx2]
    else:
        eo2 = EOItem()

    # Bucket 3: January 2025 AND government efficiency/regulatory reform/deregulation
    def cond_bucket3(e: EOItem) -> bool:
        date_match = _is_truthy(e.is_in_jan_2025) or _looks_like_month(e.signed_date, "january") and "2025" in _safe_lower(e.signed_date)
        subj_match = (
            _is_truthy(e.is_gov_efficiency_or_regreform_or_dereg) or
            any(t in [x.lower() for x in (e.subject_tags or [])] for t in ["government_efficiency", "regulatory_reform", "deregulation"]) or
            _maybe_contains(e.subject_summary, ["efficiency", "regulatory", "regulation", "deregulation", "red tape", "streamline"])
        )
        return date_match and subj_match

    idx3 = select_item_for_bucket(items, used_indices, cond_bucket3)
    if idx3 is not None:
        used_indices.add(idx3)
        eo3 = items[idx3]
    else:
        eo3 = EOItem()

    # Bucket 4: June 2025 AND international agreement/treaty
    def cond_bucket4(e: EOItem) -> bool:
        date_match = _is_truthy(e.is_in_jun_2025) or _looks_like_month(e.signed_date, "june") and "2025" in _safe_lower(e.signed_date)
        subj_match = (
            _is_truthy(e.is_international_agreement_or_treaty) or
            any(t in [x.lower() for x in (e.subject_tags or [])] for t in ["international_agreement", "treaty"]) or
            _maybe_contains(e.subject_summary, ["treaty", "international agreement", "agreement", "accord", "convention", "protocol"])
        )
        return date_match and subj_match

    idx4 = select_item_for_bucket(items, used_indices, cond_bucket4)
    if idx4 is not None:
        used_indices.add(idx4)
        eo4 = items[idx4]
    else:
        eo4 = EOItem()

    # Record mapping info for transparency/debugging
    evaluator.add_custom_info(
        info={
            "selected_indices": {
                "bucket1_energy_jan_20_2025": idx1,
                "bucket2_trade_may_2025": idx2,
                "bucket3_gov_eff_jan_2025": idx3,
                "bucket4_international_jun_2025": idx4
            },
            "total_items_extracted": len(items)
        },
        info_type="selection_mapping",
        info_name="eo_bucket_selection"
    )

    # Build verification tree per rubric
    # Root: "Find four executive orders ..." — we'll keep root as already initialized

    # Executive Order 1 (Energy on Jan 20, 2025)
    await verify_eo_bucket(
        evaluator=evaluator,
        parent_node=root,
        bucket_index=1,
        bucket_desc="Identify an executive order signed on January 20, 2025 (Inauguration Day) related to energy policy",
        eo_item=eo1,
        date_kind="exact_jan20_2025"
    )

    # Executive Order 2 (Trade in May 2025)
    await verify_eo_bucket(
        evaluator=evaluator,
        parent_node=root,
        bucket_index=2,
        bucket_desc="Identify an executive order signed in May 2025 related to trade policy or international economic agreements",
        eo_item=eo2,
        date_kind="month_may_2025"
    )

    # Executive Order 3 (Gov efficiency / reg reform in Jan 2025)
    await verify_eo_bucket(
        evaluator=evaluator,
        parent_node=root,
        bucket_index=3,
        bucket_desc="Identify an executive order signed in January 2025 related to government efficiency or regulatory reform",
        eo_item=eo3,
        date_kind="month_jan_2025"
    )

    # Executive Order 4 (International agreement/treaty in June 2025)
    await verify_eo_bucket(
        evaluator=evaluator,
        parent_node=root,
        bucket_index=4,
        bucket_desc="Identify an executive order signed in June 2025 that implements or references an international agreement",
        eo_item=eo4,
        date_kind="month_jun_2025"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()