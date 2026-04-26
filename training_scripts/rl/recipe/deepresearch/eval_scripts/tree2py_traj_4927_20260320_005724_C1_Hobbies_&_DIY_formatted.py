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
TASK_ID = "hobby_lobby_christmas_eve_2025_hours"
TASK_DESCRIPTION = """
What time does Hobby Lobby close on Christmas Eve 2025?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HoursAndSources(BaseModel):
    """
    Structured information extracted from the agent's answer.
    """
    open_status: Optional[str] = None  # expected values: "open", "closed", or free text; leave None if not specified
    closing_time: Optional[str] = None  # e.g., "5:30 PM", "6 p.m.", "5 pm local time"
    hours_urls: List[str] = Field(default_factory=list)   # URLs cited to support holiday hours claims
    craft_urls: List[str] = Field(default_factory=list)   # URLs cited that show Hobby Lobby sells Christmas craft supplies / DIY ornament materials
    all_urls: List[str] = Field(default_factory=list)     # Every URL mentioned in the answer (deduplicated if possible)


# --------------------------------------------------------------------------- #
# Extraction prompt builders                                                  #
# --------------------------------------------------------------------------- #
def prompt_extract_hours_and_sources() -> str:
    return """
    Extract the following fields from the provided answer text. Only extract what is explicitly present in the answer; do not invent or infer.

    Fields to extract:
    1) open_status:
       - If the answer explicitly states whether Hobby Lobby is open on Christmas Eve 2025 (Dec 24, 2025), extract a concise value:
         • "open" if the answer indicates the stores are open (even if closing early).
         • "closed" if the answer indicates the stores are closed.
         • If the status is ambiguous or not directly stated, set to null.

    2) closing_time:
       - The specific closing time stated for Christmas Eve 2025. Extract the raw string as written (e.g., "5:30 p.m.", "6 PM", "5:30 pm local time").
       - If not provided, set to null.

    3) hours_urls:
       - All URLs in the answer that are intended to support the holiday hours information (e.g., Hobby Lobby newsroom/press releases, official website pages with hours, etc.).
       - If the answer does not distinguish which URLs support hours, leave this empty.

    4) craft_urls:
       - All URLs in the answer that indicate Hobby Lobby sells Christmas craft supplies or DIY ornament-making materials (e.g., category pages, product pages).
       - If none are provided, leave this empty.

    5) all_urls:
       - List every URL mentioned anywhere in the answer (including those already in hours_urls and craft_urls).
       - Include only URLs that appear in the answer, in any format (plain URL or markdown link). Provide full absolute URLs.

    Output a JSON object with keys: open_status, closing_time, hours_urls, craft_urls, all_urls.
    If any required field is missing from the answer, use null for single-value fields or an empty array for lists as described above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    res = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            res.append(x)
    return res


def _official_hobby_lobby_urls(urls: List[str]) -> List[str]:
    """
    Filter to likely-official Hobby Lobby sources (company website or newsroom).
    """
    allowed_domains = ["hobbylobby.com", "newsroom.hobbylobby.com"]
    out = []
    for u in urls or []:
        lu = u.lower().strip()
        if any(dom in lu for dom in allowed_domains):
            out.append(u)
    return _dedupe_keep_order(out)


def _pick_hours_candidate_urls(extracted: HoursAndSources) -> List[str]:
    """
    Prefer hours_urls if provided; otherwise fall back to all_urls.
    """
    primary = extracted.hours_urls or []
    fallback = extracted.all_urls or []
    candidates = primary if len(primary) > 0 else fallback
    return _dedupe_keep_order(candidates)


def _pick_craft_candidate_urls(extracted: HoursAndSources) -> List[str]:
    """
    Prefer craft_urls if provided; otherwise fall back to all_urls.
    """
    primary = extracted.craft_urls or []
    fallback = extracted.all_urls or []
    candidates = primary if len(primary) > 0 else fallback
    return _dedupe_keep_order(candidates)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_answer_completeness(
    evaluator: Evaluator,
    parent_node,
    extracted: HoursAndSources,
) -> None:
    """
    Build the verification tree for the rubric's "Answer_Completeness" node and run verifications.
    """

    # Create the main parallel node as per rubric
    completeness_node = evaluator.add_parallel(
        id="Answer_Completeness",
        desc="The answer provides complete and verifiable information about Hobby Lobby's Christmas Eve 2025 hours and confirms it sells craft supplies.",
        parent=parent_node,
        critical=False,
    )

    # 1) Store_Open_Status (critical)
    store_open_leaf = evaluator.add_leaf(
        id="Store_Open_Status",
        desc="The answer confirms that Hobby Lobby is open on Christmas Eve 2025 (not closed for the holiday).",
        parent=completeness_node,
        critical=True,
    )

    # Construct claim – prefer using the answer-stated status if present; otherwise use a neutral claim
    open_claim_text = None
    if extracted.open_status:
        normalized = extracted.open_status.strip().lower()
        if "open" in normalized and "close" not in normalized:
            open_claim_text = "Hobby Lobby stores are open on Christmas Eve 2025 (December 24, 2025)."
        elif "closed" in normalized:
            # If the answer explicitly claims closed, we still verify the claim (likely to fail if sources say otherwise).
            open_claim_text = "Hobby Lobby stores are closed on Christmas Eve 2025 (December 24, 2025)."
    if not open_claim_text:
        # Fallback neutral pro-open claim aligned with the task
        open_claim_text = "Hobby Lobby stores are open on Christmas Eve 2025 (December 24, 2025)."

    open_status_sources = _pick_hours_candidate_urls(extracted)
    await evaluator.verify(
        claim=open_claim_text,
        node=store_open_leaf,
        sources=open_status_sources,
        additional_instruction=(
            "Judge based on the provided webpages whether Hobby Lobby stores are open on Dec 24, 2025. "
            "If a page states 'open' or provides opening/closing times for Christmas Eve, that supports 'open'. "
            "If no supporting URL is provided, or the pages are unrelated, mark as not supported."
        ),
    )

    # 2) Closing_Time (critical)
    closing_time_leaf = evaluator.add_leaf(
        id="Closing_Time",
        desc="The answer provides the specific closing time for Hobby Lobby on Christmas Eve 2025.",
        parent=completeness_node,
        critical=True,
    )

    closing_time_str = extracted.closing_time or ""
    close_claim = (
        f"Hobby Lobby stores close at {closing_time_str} on Christmas Eve 2025 (December 24, 2025)."
        if closing_time_str
        else "The answer provides a specific closing time for Hobby Lobby on Christmas Eve 2025, and that time matches what official sources state."
    )

    close_time_sources = _pick_hours_candidate_urls(extracted)
    await evaluator.verify(
        claim=close_claim,
        node=closing_time_leaf,
        sources=close_time_sources,
        additional_instruction=(
            "Confirm the closing time for Dec 24, 2025 from the provided pages. "
            "Allow minor formatting variants (e.g., 5:30 PM vs 5:30 p.m.). "
            "If the page states 'close early at 5:30 p.m. local time', treat it as matching '5:30 p.m.'. "
            "If the answer did not provide a specific time or there are no supporting URLs, mark as not supported."
        ),
    )

    # 3) Craft_Supplies_Verification (critical)
    craft_leaf = evaluator.add_leaf(
        id="Craft_Supplies_Verification",
        desc="The answer confirms or the reference source indicates that Hobby Lobby sells Christmas craft supplies suitable for DIY ornament making.",
        parent=completeness_node,
        critical=True,
    )

    craft_sources = _pick_craft_candidate_urls(extracted)
    await evaluator.verify(
        claim="Hobby Lobby sells Christmas craft supplies suitable for DIY ornament making.",
        node=craft_leaf,
        sources=craft_sources,
        additional_instruction=(
            "Accept support from Hobby Lobby category pages or product pages that mention 'Christmas crafts', "
            "'ornament kits', 'DIY ornaments', or similar. "
            "If only hours/press pages are provided with no product/category evidence, this is not supported."
        ),
    )

    # 4) Reference_URL (critical)
    reference_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="A valid URL reference from an official Hobby Lobby source (company website or newsroom) is provided to verify the holiday hours information.",
        parent=completeness_node,
        critical=True,
    )

    official_urls = _official_hobby_lobby_urls(extracted.all_urls or [])
    # We verify by trying these official URLs; If none provided, the instruction forces a fail.
    await evaluator.verify(
        claim=(
            "This webpage is an official Hobby Lobby source (company website or newsroom) and it provides "
            "or confirms store holiday hours information (e.g., Christmas Eve hours)."
        ),
        node=reference_leaf,
        sources=official_urls,  # may be empty; the additional instruction below handles that case
        additional_instruction=(
            "Pass if at least one provided URL is from hobbylobby.com or newsroom.hobbylobby.com AND includes "
            "holiday hours info or Christmas Eve hours details. "
            "If no such official URL is provided in the answer, mark as not supported."
        ),
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
    Evaluate an answer for Hobby Lobby Christmas Eve 2025 closing time.
    """
    # Initialize evaluator with a parallel root (one task with parallel checks beneath)
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
    extracted: HoursAndSources = await evaluator.extract(
        prompt=prompt_extract_hours_and_sources(),
        template_class=HoursAndSources,
        extraction_name="hours_and_sources",
    )

    # Ensure URL arrays are deduped for cleaner downstream behavior
    extracted.hours_urls = _dedupe_keep_order(extracted.hours_urls or [])
    extracted.craft_urls = _dedupe_keep_order(extracted.craft_urls or [])
    extracted.all_urls = _dedupe_keep_order(extracted.all_urls or [])

    # Build tree and run verifications
    await verify_answer_completeness(evaluator, root, extracted)

    # Return evaluator summary
    return evaluator.get_summary()